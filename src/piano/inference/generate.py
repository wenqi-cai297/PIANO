"""Full inference pipeline: text + object → interaction latent → motion.

End-to-end generation pipeline that:
    1. Encodes text via CLIP
    2. Encodes object point cloud via ObjectEncoder
    3. Predicts interaction latent via InteractionPredictor
    4. Generates motion tokens via InteractionMaskTransformer
    5. Decodes tokens to motion features via RVQVAE
    6. Optionally visualizes the result

Usage:
    piano-generate --text "pick up the box" --object data/objects/box.ply \
                   --output runs/generated/sample.npz
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from piano.data.humanml3d_repr import denormalize_motion
from piano.models.interaction_tokenizer import InteractionTokenizer
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.backbones.momask_adapter import load_momask_vqvae
from piano.models.motion_generator import InteractionMaskTransformer
from piano.models.object_encoder import ObjectEncoder
from piano.utils.clip_utils import encode_text_per_token
from piano.utils.io_utils import save_npz


@dataclass(slots=True)
class GenerationConfig:
    """Configuration for motion generation."""

    # Generation parameters
    num_frames: int = 196
    fps: float = 20.0                     # InterAct preprocessed data rate
    num_unmasking_steps: int = 10
    temperature: float = 1.0
    topk_filter_thres: float = 0.9
    cond_scale: float = 4.5
    interaction_scale: float = 2.0

    # Object
    num_object_points: int = 1024

    # Initial pose dimension (66 = 22 SMPL joints × 3)
    init_pose_dim: int = 66


class PIANOPipeline:
    """End-to-end PIANO inference pipeline.

    Loads all models and provides a simple ``generate`` method.
    """

    def __init__(
        self,
        predictor: InteractionPredictor,
        object_encoder: ObjectEncoder,
        transformer: InteractionMaskTransformer,
        vq_vae: nn.Module,
        interaction_tokenizer: InteractionTokenizer,
        clip_model: torch.nn.Module,
        device: str = "cuda",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.predictor = predictor.to(device).eval()
        self.object_encoder = object_encoder.to(device).eval()
        self.transformer = transformer.to(device).eval()
        self.vq_vae = vq_vae.to(device).eval()
        self.interaction_tokenizer = interaction_tokenizer.to(device).eval()
        self.clip_model = clip_model
        self.device = device
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def generate(
        self,
        text: str,
        object_pc: np.ndarray,
        init_pose: np.ndarray | None = None,
        obj_com_canonical: np.ndarray | None = None,
        obj_rot6d_canonical: np.ndarray | None = None,
        config: GenerationConfig | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate motion from text and object.

        Parameters
        ----------
        text : text description of the action
        object_pc : (N, 3) object point cloud
        init_pose : (66,) initial pose — SMPL-22 joint positions
            flattened (xyz per joint). Pass None for zeros.
        config : generation configuration

        Returns
        -------
        Dictionary with:
            ``motion_263`` : (T, 263) HumanML3D features (denormalized if stats provided)
            ``contact_state`` : (T, 5) predicted contact (sigmoid probabilities)
            ``contact_target_xyz`` : (T, 5, 3) predicted contact xyz in
                                     object-local frame (5 body parts × 3 coords)
            ``phase`` : (T, P) predicted phase (softmax probabilities)
            ``support`` : (T, S) predicted support (softmax probabilities)
            ``token_ids`` : (S,) generated VQ token indices
        """
        if config is None:
            config = GenerationConfig()

        # --- Encode text ---
        # Per-token features for the predictor's text cross-attn, pooled
        # features for the MaskTransformer.
        text_features, text_mask = encode_text_per_token(
            self.clip_model, [text], torch.device(self.device),
        )
        text_pooled = self._encode_text_pooled(text)   # (1, clip_dim)

        # --- Encode object ---
        # The structured_head predictor (v8+) needs both token features
        # AND token centroid xyz for affordance attention / hierarchical
        # mask decoder. Use return_xyz=True to get both.
        obj_pc_tensor = torch.from_numpy(object_pc).float().unsqueeze(0).to(self.device)
        if obj_pc_tensor.shape[1] != config.num_object_points:
            indices = np.random.choice(len(object_pc), config.num_object_points, replace=True)
            obj_pc_tensor = torch.from_numpy(object_pc[indices]).float().unsqueeze(0).to(self.device)
        obj_xyz_tensor, obj_tokens = self.object_encoder(
            obj_pc_tensor, return_xyz=True,
        )  # (1, M, 3), (1, M, d)

        # --- Initial pose (66-d: SMPL-22 joint xyz) ---
        if init_pose is not None:
            pose_tensor = torch.from_numpy(init_pose).float().unsqueeze(0).to(self.device)
        else:
            pose_tensor = torch.zeros(1, config.init_pose_dim, device=self.device)

        # --- Predict interaction latent ---
        # The predictor's output dict keys are:
        #   contact_state         (B, T, num_body_parts) — sigmoid probs
        #   contact_target_xyz    (B, T, num_body_parts, 3) — closest-mesh-
        #                          point xyz in object-local frame
        #   phase                 (B, T, num_phases) — softmax probs
        #   support               (B, T, num_support) — softmax probs
        # When ``structured_head.target_attn_kind="hierarchical_mask_decoder"``
        # (v9.5) the dict ALSO contains contact_target_attn /
        # contact_target_patch_logits, but those are not consumed by the
        # tokenizer here — the v8.1b refactor that would consume them
        # directly is deferred (see analyses/2026-05-04_v95_local_results.md).
        pred = self.predictor(
            text_features, obj_tokens, pose_tensor,
            seq_length=config.num_frames,
            text_key_padding_mask=text_mask,
            object_xyz=obj_xyz_tensor,
        )

        # --- Build interaction tokens ---
        # InteractionTokenizer.forward signature requires per-frame object
        # pose channels (obj_com_canonical / obj_rot6d_canonical) when
        # num_obj_pose_channels > 0 — caller must canonicalize world-frame
        # object trajectories first via
        # ``piano.utils.canonical_frame.world_to_canonical_object_pose``.
        # See interaction_tokenizer.py:265-281 for the contract.
        if obj_com_canonical is None or obj_rot6d_canonical is None:
            raise ValueError(
                "PIANOPipeline.generate requires obj_com_canonical + "
                "obj_rot6d_canonical for the v0.2 InteractionTokenizer "
                "(num_obj_pose_channels=9). Pass per-frame canonicalized "
                "object COM (T, 3) and 6D rotation (T, 6) — convert via "
                "piano.utils.canonical_frame.world_to_canonical_object_pose "
                "before calling."
            )
        obj_com_t = torch.from_numpy(obj_com_canonical).float().unsqueeze(0).to(self.device)
        obj_rot6d_t = torch.from_numpy(obj_rot6d_canonical).float().unsqueeze(0).to(self.device)
        interaction_tokens, _ = self.interaction_tokenizer(
            contact_state=pred["contact_state"],
            contact_target_xyz=pred["contact_target_xyz"],
            phase=pred["phase"],
            support=pred["support"],
            obj_com_canonical=obj_com_t,
            obj_rot6d_canonical=obj_rot6d_t,
        )  # (1, S_int, d_model)

        # --- Generate motion tokens ---
        token_len = config.num_frames // 4  # VQ temporal downsampling
        m_lens = torch.tensor([token_len], device=self.device)

        token_ids = self.transformer.generate(
            cond=text_pooled,
            m_lens=m_lens,
            interaction_tokens=interaction_tokens,
            timesteps=config.num_unmasking_steps,
            cond_scale=config.cond_scale,
            interaction_scale=config.interaction_scale,
            temperature=config.temperature,
            topk_filter_thres=config.topk_filter_thres,
        )  # (1, S)

        # --- Decode to motion ---
        # Build full indices (base level only, pad residual with 0)
        full_indices = torch.zeros(1, token_len, self.vq_vae.num_quantizers,
                                   dtype=torch.long, device=self.device)
        valid_ids = token_ids[0]
        valid_ids = valid_ids[valid_ids >= 0]  # remove padding (-1)
        full_indices[0, :len(valid_ids), 0] = valid_ids

        motion_263 = self.vq_vae.decode(full_indices)  # (1, T, 263)
        motion_263 = motion_263[0].cpu().numpy()  # (T, 263)

        # Denormalize if statistics available
        if self.mean is not None and self.std is not None:
            motion_263 = denormalize_motion(motion_263, self.mean, self.std)

        return {
            "motion_263": motion_263,
            "contact_state": pred["contact_state"][0].cpu().numpy(),
            "contact_target_xyz": pred["contact_target_xyz"][0].cpu().numpy(),
            "phase": pred["phase"][0].cpu().numpy(),
            "support": pred["support"][0].cpu().numpy(),
            "token_ids": token_ids[0].cpu().numpy(),
        }

    def _encode_text_pooled(self, text: str) -> Tensor:
        """Encode text to the pooled CLIP vector (used by the MaskTransformer)."""
        import clip

        tokens = clip.tokenize([text], truncate=True).to(self.device)
        text_emb = self.clip_model.encode_text(tokens).float()
        return text_emb  # (1, clip_dim)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for ``piano-generate``."""
    parser = argparse.ArgumentParser(description="Generate motion from text + object")
    parser.add_argument("--text", type=str, required=True, help="Action description")
    parser.add_argument("--object", type=Path, required=True, help="Object point cloud (.npy or .ply)")
    parser.add_argument("--output", type=Path, required=True, help="Output npz path")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                        help="Directory containing model checkpoints")
    parser.add_argument("--num-frames", type=int, default=196, help="Number of frames to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cond-scale", type=float, default=4.5)
    parser.add_argument("--interaction-scale", type=float, default=2.0)
    args = parser.parse_args()

    # TODO: Load models from checkpoint directory and run pipeline
    # This will be fully functional once checkpoints are available on server
    print(f"Would generate motion for: '{args.text}'")
    print(f"Object: {args.object}")
    print(f"Output: {args.output}")
    raise NotImplementedError(
        "Full inference requires trained checkpoints. "
        "Run training stages first, then update checkpoint paths."
    )


if __name__ == "__main__":
    main()
