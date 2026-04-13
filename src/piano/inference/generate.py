"""Full inference pipeline: text + object → interaction latent → motion.

End-to-end generation pipeline that:
    1. Encodes text via CLIP
    2. Encodes object point cloud via ObjectEncoder
    3. Predicts interaction latent via InteractionPredictor
    4. Generates motion tokens via MaskedTransformerWithInteraction
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
from torch import Tensor

from piano.data.humanml3d_repr import denormalize_motion
from piano.models.interaction_cross_attn import InteractionTokenizer
from piano.models.interaction_predictor import InteractionPredictor
from piano.models.motion_generator import MaskedTransformerWithInteraction, RVQVAE
from piano.models.object_encoder import ObjectEncoder
from piano.utils.io_utils import save_npz


@dataclass(slots=True)
class GenerationConfig:
    """Configuration for motion generation."""

    # Generation parameters
    num_frames: int = 196
    fps: float = 30.0
    num_unmasking_steps: int = 10
    temperature: float = 1.0
    topk_filter_thres: float = 0.9
    cond_scale: float = 4.5
    interaction_scale: float = 2.0

    # Object
    num_object_points: int = 1024


class PIANOPipeline:
    """End-to-end PIANO inference pipeline.

    Loads all models and provides a simple ``generate`` method.
    """

    def __init__(
        self,
        predictor: InteractionPredictor,
        object_encoder: ObjectEncoder,
        transformer: MaskedTransformerWithInteraction,
        vq_vae: RVQVAE,
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
        config: GenerationConfig | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate motion from text and object.

        Parameters
        ----------
        text : text description of the action
        object_pc : (N, 3) object point cloud
        init_pose : (263,) initial pose features, or None for zeros
        config : generation configuration

        Returns
        -------
        Dictionary with:
            ``motion_263`` : (T, 263) HumanML3D features (denormalized if stats provided)
            ``contact_state`` : (T, 5) predicted contact
            ``contact_target`` : (T, 5, K) predicted contact target
            ``phase`` : (T, P) predicted phase
            ``support`` : (T, S) predicted support
            ``token_ids`` : (S,) generated VQ token indices
        """
        if config is None:
            config = GenerationConfig()

        # --- Encode text ---
        text_emb = self._encode_text(text)  # (1, clip_dim)

        # --- Encode object ---
        obj_pc_tensor = torch.from_numpy(object_pc).float().unsqueeze(0).to(self.device)
        if obj_pc_tensor.shape[1] != config.num_object_points:
            indices = np.random.choice(len(object_pc), config.num_object_points, replace=True)
            obj_pc_tensor = torch.from_numpy(object_pc[indices]).float().unsqueeze(0).to(self.device)
        obj_tokens = self.object_encoder(obj_pc_tensor)  # (1, M, d)

        # --- Initial pose ---
        if init_pose is not None:
            pose_tensor = torch.from_numpy(init_pose).float().unsqueeze(0).to(self.device)
        else:
            pose_tensor = torch.zeros(1, 263, device=self.device)

        # --- Predict interaction latent ---
        pred = self.predictor(text_emb, obj_tokens, pose_tensor, seq_length=config.num_frames)

        # --- Build interaction tokens ---
        interaction_tokens = self.interaction_tokenizer(
            pred["contact_state"],
            pred["contact_target"],
            pred["phase"],
            pred["support"],
        )  # (1, S_int, d_model)

        # --- Generate motion tokens ---
        token_len = config.num_frames // 4  # VQ temporal downsampling
        m_lens = torch.tensor([token_len], device=self.device)

        token_ids = self.transformer.generate(
            cond=text_emb,
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
            "contact_target": pred["contact_target"][0].cpu().numpy(),
            "phase": pred["phase"][0].cpu().numpy(),
            "support": pred["support"][0].cpu().numpy(),
            "token_ids": token_ids[0].cpu().numpy(),
        }

    def _encode_text(self, text: str) -> Tensor:
        """Encode text using CLIP."""
        # This will use the CLIP model's tokenizer and encoder
        # Implementation depends on how CLIP is loaded (OpenAI CLIP vs HuggingFace)
        import clip

        tokens = clip.tokenize([text]).to(self.device)
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
