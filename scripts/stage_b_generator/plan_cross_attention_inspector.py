"""Round-23 P0 — Inspect plan cross-attention weights inside the v12 DiT
Stage-2 denoiser.

Hypothesis being tested: motion tokens at far-from-anchor frames do NOT
attend to nearby plan tokens. If true, the §7.4 plan-not-routed-to-unobs
failure mode is caused at the attention layer itself (Scenario A) rather
than downstream gating (Scenario B, already ruled out — plan_xattn is
unmodulated) or feature/positional collapse (Scenario C/D).

Methodology
-----------
1. Load a config + ckpt (v18 production OR an R22 v25 ckpt).
2. Build the model and load weights.
3. Wrap each ``ConditionedEncoderLayer.plan_xattn`` with a capturing
   wrapper that forces ``need_weights=True, average_attn_weights=False``
   on the inner ``MultiheadAttention`` and stores the per-head weights.
4. Build cond from a real clip (same path as plan_condition_diagnostics).
5. Run ONE forward at a fixed diffusion timestep with the clean cond
   (GT plan + GT route if branch enabled).
6. Extract attention weights per layer: ``(n_heads, T_motion, K_plan)``.
7. Compute diagnostics:
   - ``mean_entropy_per_layer``: avg over (heads, motion frames) of
     ``H(p_attn)``. Low entropy = sharp attention.
   - ``frac_attention_on_nearest_anchor``: per motion frame, is the
     top-1 plan token the temporally-nearest valid anchor?
   - ``effective_K_per_layer``: ``exp(H)`` averaged over frames; gives a
     "how many plan tokens is each motion frame effectively reading from"
     measure. ~1 = sharp focus. ~K_valid = uniform.
8. Save heatmaps per layer (PNG) + a JSON summary.

The inspector does NOT modify the model weights — wrappers are pure
forward interception, removed at script exit.

Usage
-----
    conda run -n piano python scripts/stage_b_generator/plan_cross_attention_inspector.py \
        --config configs/training/anchordiff_v18_a1_FULL_DATA.yaml \
        --ckpt   runs/training/stageB_anchordiff_v18_a1_FULL_DATA/final.pt \
        --output analyses/round23_attention_inspection/v18.{json,png} \
        --bucket train --clip-idx 0 --t-step 200

Output:
    <output>.json     # summary metrics per layer + head
    <output>__layer<L>.png  # heatmap per layer
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from torch import Tensor, nn

# Import shared helpers from plan_condition_diagnostics (same path the
# trainer + diagnostic use, so cond construction matches the production
# pipeline exactly).
import sys
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import (  # type: ignore  # noqa: E402
    _build_cond,
    _build_dataset,
    _build_model,
    _stage1_norm_for_cfg,
)

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.utils.clip_utils import load_clip_text_encoder  # noqa: E402

from torch.utils.data import DataLoader, Subset  # noqa: E402


# ---------------------------------------------------------------------------
# Attention capture wrapper
# ---------------------------------------------------------------------------


class CapturingMHA(nn.Module):
    """Wraps an ``nn.MultiheadAttention`` and captures the last
    per-head attention matrix on every forward call.

    Forward signature mirrors ``nn.MultiheadAttention.__call__`` enough
    for ConditionedEncoderLayer's existing call site to keep working.
    """

    def __init__(self, inner: nn.MultiheadAttention) -> None:
        super().__init__()
        self.inner = inner
        # Captured on every forward: (B, n_heads, T_q, T_k)
        self.last_weights: Tensor | None = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        # Always request per-head weights so we can capture them. The
        # caller may pass need_weights=False, but they discard the
        # weights anyway, so this is safe.
        out, weights = self.inner(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        # weights: (B, n_heads, T_q, T_k)
        self.last_weights = weights.detach().cpu()
        return out, None  # caller discards second return; preserve API


def wrap_plan_xattn_modules(model) -> list[CapturingMHA]:
    """Find every ConditionedEncoderLayer.plan_xattn in the model and
    replace it with a CapturingMHA wrapper. Returns the list of
    wrappers (one per layer)."""
    wrappers: list[CapturingMHA] = []
    if not hasattr(model, "denoiser"):
        raise ValueError("expected MotionAnchorDiff with .denoiser")
    denoiser = model.denoiser
    if not hasattr(denoiser, "v12_blocks"):
        raise ValueError(
            "denoiser has no v12_blocks — v11 path is not supported here. "
            "Use use_dit_block=True in the config."
        )
    for block in denoiser.v12_blocks:
        inner = block.plan_xattn
        wrapper = CapturingMHA(inner)
        block.plan_xattn = wrapper
        wrappers.append(wrapper)
    return wrappers


# ---------------------------------------------------------------------------
# Diagnostic metrics
# ---------------------------------------------------------------------------


def compute_attention_metrics(
    attn: Tensor,                      # (B, n_heads, T_q, T_k) — single layer
    plan_anchor_times: Tensor,         # (B, K_max) long — valid anchor frame indices
    plan_anchor_mask: Tensor,          # (B, K_max) bool — True at valid anchors
    plan_token_anchor_idx: Tensor,     # (B, T_k) long — anchor index each plan token belongs to (or -1 if pad)
    motion_token_start: int,           # index where motion tokens start (after init_pose prefix)
    T_motion: int,
) -> dict[str, Any]:
    """Per-layer attention diagnostics. Returns a dict of scalars + per-frame arrays."""
    B, n_heads, T_q, T_k = attn.shape
    # Sub-select motion tokens (skip init_pose prefix at index 0).
    motion_attn = attn[:, :, motion_token_start:motion_token_start + T_motion, :]
    # (B, n_heads, T_motion, T_k)

    # Mask out padded plan-token positions before entropy. Padding rows
    # have attention exactly zero (PyTorch's MHA softmax over masked keys).
    valid_k_mask = plan_token_anchor_idx[0] >= 0       # (T_k,) — assume B==1
    valid_K = int(valid_k_mask.sum().item())
    if valid_K == 0:
        return {"valid_K": 0, "warning": "no valid plan tokens"}

    # Renormalize attention over valid plan tokens only (defensive — MHA
    # should already do this when key_padding_mask is set).
    eps = 1e-12
    motion_attn_v = motion_attn[..., valid_k_mask]      # (B, n_heads, T_motion, valid_K)
    motion_attn_v = motion_attn_v / motion_attn_v.sum(dim=-1, keepdim=True).clamp_min(eps)

    # Entropy per (head, frame): H = -Σ p log p
    H = -(motion_attn_v.clamp_min(eps) * motion_attn_v.clamp_min(eps).log()).sum(-1)
    # H shape: (B, n_heads, T_motion)
    log_K = float(np.log(valid_K))
    H_mean = float(H.mean().item())
    H_per_head = H.mean(dim=(0, 2)).tolist()           # (n_heads,)
    effective_K_mean = float(math.exp(H_mean))

    # Top-1 plan token per (head, frame).
    top1 = motion_attn_v.argmax(dim=-1)                # (B, n_heads, T_motion)
    # Anchor times of valid plan tokens (in same order as valid_k_mask).
    valid_token_anchor_idx = plan_token_anchor_idx[0, valid_k_mask]   # (valid_K,)
    # Map: token_pos_in_valid -> anchor_time (long)
    anchor_times = plan_anchor_times[0]                # (K_max,)
    token_anchor_times = anchor_times[valid_token_anchor_idx]         # (valid_K,)

    # For each motion frame t, find the temporally-nearest valid anchor.
    motion_t = torch.arange(T_motion).view(1, 1, -1).expand_as(top1)   # (B, n_heads, T_motion)
    # Distance from frame t to each anchor.
    # anchor_times_b: (valid_K,) — same for B==1.
    dist = (motion_t.unsqueeze(-1).float() - token_anchor_times.float().view(1, 1, 1, -1)).abs()
    nearest_anchor_per_frame = dist.argmin(dim=-1)     # (B, n_heads, T_motion)

    # Fraction of (head, frame) cells where top-1 == nearest-anchor token.
    frac_top1_is_nearest = float(
        (top1 == nearest_anchor_per_frame).float().mean().item()
    )

    # Per-frame top-1 mode and top-1 attention value.
    top1_value = motion_attn_v.gather(-1, top1.unsqueeze(-1)).squeeze(-1)
    top1_value_mean = float(top1_value.mean().item())

    # Attention weight ON the nearest anchor (regardless of where the
    # top-1 actually is). If attention is well-routed, this should be
    # high; if attention is uniform or focused on wrong tokens, low.
    attn_on_nearest = motion_attn_v.gather(
        -1, nearest_anchor_per_frame.unsqueeze(-1),
    ).squeeze(-1)
    attn_on_nearest_mean = float(attn_on_nearest.mean().item())

    return {
        "valid_K": valid_K,
        "log_K": log_K,
        "entropy_mean": H_mean,
        "entropy_per_head": H_per_head,
        "effective_K_mean": effective_K_mean,
        "frac_top1_is_nearest_anchor": frac_top1_is_nearest,
        "top1_attention_value_mean": top1_value_mean,
        "attention_on_nearest_anchor_mean": attn_on_nearest_mean,
    }


# ---------------------------------------------------------------------------
# Plot heatmaps
# ---------------------------------------------------------------------------


def plot_attention_heatmaps(
    per_layer_attn: list[Tensor],          # list of (n_heads, T_motion, K_valid)
    plan_token_anchor_idx: Tensor,         # (T_k,) — anchor index per plan token
    plan_anchor_times: Tensor,             # (K_max,) long
    plan_anchor_mask: Tensor,              # (K_max,) bool
    out_prefix: Path,
    title_suffix: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # For each layer, plot mean-over-heads attention as heatmap.
    n_layers = len(per_layer_attn)
    valid_anchor_idx = plan_token_anchor_idx[plan_token_anchor_idx >= 0]
    K_valid = len(valid_anchor_idx)
    anchor_times_valid = plan_anchor_times[valid_anchor_idx].cpu().numpy()

    n_rows = (n_layers + 3) // 4
    n_cols = min(4, n_layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]

    for L, attn in enumerate(per_layer_attn):
        # attn: (n_heads, T_motion, K_valid)
        ax = axes[L // n_cols][L % n_cols]
        mean_attn = attn.mean(0).numpy()                # (T_motion, K_valid)
        im = ax.imshow(mean_attn, aspect="auto", cmap="viridis", vmin=0, vmax=mean_attn.max())
        ax.set_xlabel("plan token (sorted by anchor_time)")
        ax.set_ylabel("motion frame t")
        ax.set_title(f"Layer {L}  mean over heads")
        # Mark anchor times as horizontal lines (the "expected best attend frame").
        for tok_idx, a_t in enumerate(anchor_times_valid):
            ax.axhline(a_t, color="white", alpha=0.15, linestyle=":", linewidth=0.5)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(f"Plan cross-attention weights — {title_suffix}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = Path(str(out_prefix) + "__all_layers.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_plan_token_anchor_idx_per_part(
    plan: dict[str, Tensor],
    n_tokens: int,
    cfg_num_parts: int,
) -> Tensor:
    """When per_part_tokens=True, plan_encoder emits one token per
    (anchor_idx, part_idx) cell where the anchor is valid AND part is
    active. The order is row-major: anchor 0 part 0, anchor 0 part 1, ...
    (same as the encoder's flatten). Padded slots get anchor_idx = -1.

    For per_part_tokens=False this returns just anchor_idx (which may
    include segments, but we only inspect anchors for simplicity).
    """
    B = plan["anchor_time"].shape[0]
    K = plan["anchor_time"].shape[1]
    anchor_mask = plan["anchor_mask"].bool()                            # (B, K)
    anchor_part = plan["anchor_part"].float()                           # (B, K, P)
    P = cfg_num_parts
    # Per-part validity: anchor valid AND part active.
    part_valid = (anchor_part > 0.5) & anchor_mask.unsqueeze(-1)         # (B, K, P)
    # Flatten in row-major (K outer, P inner).
    flat_idx = torch.full((B, n_tokens), -1, dtype=torch.long)
    for b in range(B):
        token_pos = 0
        for k in range(K):
            for p in range(P):
                if not part_valid[b, k, p]:
                    continue
                if token_pos < n_tokens:
                    flat_idx[b, token_pos] = k
                    token_pos += 1
    return flat_idx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Prefix path; .json + __all_layers.png suffixes are appended.")
    parser.add_argument("--bucket", default="train", choices=["train", "val"])
    parser.add_argument("--clip-idx", type=int, default=0)
    parser.add_argument("--t-step", type=int, default=200,
                        help="Diffusion timestep at which to compute attention (range 0..num_steps-1).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build dataset and pick the target clip ----
    dataset = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        dataset = Subset(dataset, list(range(min(overfit_n, len(dataset)))))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_hoi, num_workers=0,
    )
    target_batch = None
    for i, batch in enumerate(loader):
        if i == args.clip_idx:
            target_batch = batch
            break
    if target_batch is None:
        raise RuntimeError(f"clip_idx={args.clip_idx} out of range")

    # ---- Build model and load ckpt ----
    model, object_encoder, z_dims = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])

    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get("download_root", "cache/clip")),
    )

    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    cond, T = _build_cond(
        target_batch, model, object_encoder, clip_model, z_dims, cfg, device,
        stage1_norm=stage1_norm,
    )

    # Plan dict from cond (for anchor_time / anchor_mask metadata).
    plan_keys = [
        "anchor_time", "anchor_part", "anchor_target_local",
        "anchor_target_world", "anchor_type", "anchor_phase",
        "anchor_support", "anchor_conf", "anchor_mask",
        "segment_start", "segment_end", "segment_part",
        "segment_target_summary_local", "segment_phase",
        "segment_support", "segment_conf", "segment_mask",
    ]
    plan = {k: target_batch[f"plan_{k}"].to(device) for k in plan_keys}
    cond["interaction_plan"] = plan

    # ---- Wrap plan_xattn modules ----
    wrappers = wrap_plan_xattn_modules(model)
    print(f"[inspector] wrapped {len(wrappers)} plan_xattn modules")

    # ---- Forward pass at fixed diffusion step ----
    motion_gt = target_batch["motion"].to(device).float()                   # (1, T, motion_dim)
    B = motion_gt.shape[0]
    t = torch.tensor([args.t_step], device=device, dtype=torch.long)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(motion_gt.shape, generator=g).to(device)
    x_t = model.diffusion.q_sample(motion_gt, t, noise)
    cond_drop_mask = torch.zeros(B, dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        _x0_pred = model.denoiser(x_t, t, cond, cond_drop_mask=cond_drop_mask)

    # ---- Collect attention weights ----
    per_layer_attn: list[Tensor] = []
    for L, w in enumerate(wrappers):
        if w.last_weights is None:
            raise RuntimeError(f"layer {L}: no weights captured")
        per_layer_attn.append(w.last_weights[0])    # (n_heads, T_q, T_k)

    n_heads, T_q, T_k = per_layer_attn[0].shape
    motion_token_start = 1                          # init_pose token prepended
    T_motion = T_q - motion_token_start
    assert T_motion == T, f"T_motion={T_motion} != T={T}"
    print(f"[inspector] T_q={T_q} T_k={T_k} T_motion={T_motion} n_heads={n_heads}")

    # ---- Build plan_token_anchor_idx ----
    per_part = bool(cfg.model.denoiser.get("plan_per_part_tokens", False))
    P = int(cfg.model.denoiser.get("plan_num_parts", 5))
    if per_part:
        plan_token_anchor_idx = _build_plan_token_anchor_idx_per_part(
            plan, n_tokens=T_k, cfg_num_parts=P,
        )
    else:
        # One token per anchor.
        plan_token_anchor_idx = torch.full((1, T_k), -1, dtype=torch.long)
        anchor_mask = plan["anchor_mask"].bool()
        for k in range(plan_token_anchor_idx.shape[1]):
            plan_token_anchor_idx[0, k] = k if (k < anchor_mask.shape[1] and anchor_mask[0, k]) else -1

    # ---- Compute metrics ----
    plan_anchor_times = plan["anchor_time"].cpu()
    plan_anchor_mask = plan["anchor_mask"].cpu().bool()

    layer_metrics = []
    valid_attns_for_plot = []
    valid_mask_t = plan_token_anchor_idx[0] >= 0
    for L, attn in enumerate(per_layer_attn):
        m = compute_attention_metrics(
            attn.unsqueeze(0),                              # add B=1 dim
            plan_anchor_times=plan_anchor_times,
            plan_anchor_mask=plan_anchor_mask,
            plan_token_anchor_idx=plan_token_anchor_idx,
            motion_token_start=motion_token_start,
            T_motion=T_motion,
        )
        m["layer"] = L
        layer_metrics.append(m)
        valid_attns_for_plot.append(
            attn[:, motion_token_start:motion_token_start + T_motion, :][..., valid_mask_t]
        )

    # ---- Save JSON ----
    out_json = Path(str(args.output) + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "bucket": args.bucket,
        "clip_idx": args.clip_idx,
        "t_step": args.t_step,
        "seed": args.seed,
        "T_motion": T_motion,
        "T_k_total": T_k,
        "valid_K": int(valid_mask_t.sum().item()),
        "n_heads": n_heads,
        "n_layers": len(per_layer_attn),
        "plan_per_part_tokens": per_part,
        "valid_anchor_times": plan_anchor_times[0][
            plan_anchor_mask[0]
        ].tolist(),
        "layer_metrics": layer_metrics,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    # ---- Plot heatmaps ----
    plot_attention_heatmaps(
        per_layer_attn=valid_attns_for_plot,
        plan_token_anchor_idx=plan_token_anchor_idx[0],
        plan_anchor_times=plan_anchor_times[0],
        plan_anchor_mask=plan_anchor_mask[0],
        out_prefix=args.output,
        title_suffix=f"{Path(args.config).stem} clip={args.clip_idx} t={args.t_step}",
    )

    # ---- Stdout summary ----
    print("\nPer-layer summary:")
    print(f"  {'L':>2}  {'H_mean':>7}  {'eff_K':>7}  {'top1=nearest':>13}  {'attn_on_nearest':>16}")
    for m in layer_metrics:
        if m.get("valid_K", 0) == 0:
            continue
        print(
            f"  {m['layer']:>2}  {m['entropy_mean']:>7.3f}  "
            f"{m['effective_K_mean']:>7.2f}  "
            f"{m['frac_top1_is_nearest_anchor']:>13.3f}  "
            f"{m['attention_on_nearest_anchor_mean']:>16.4f}"
        )
    print(
        f"\nlog(valid_K) = {layer_metrics[0].get('log_K', float('nan')):.3f}  "
        f"(uniform attention would give H_mean ≈ log(valid_K) and "
        f"eff_K ≈ valid_K).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
