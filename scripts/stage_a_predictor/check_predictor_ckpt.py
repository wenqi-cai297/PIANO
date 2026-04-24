"""Sanity-check a Stage A predictor checkpoint.

Verifies that the checkpoint contains the expected predictor +
ObjectEncoder state_dicts, and that critical weight tensors have
finite, non-degenerate values (no NaN / Inf, non-zero std). Exits
nonzero under ``--strict`` when anything is missing or broken.

Usage:
    python scripts/stage_a_predictor/check_predictor_ckpt.py runs/training/predictor/final.pt
    python scripts/stage_a_predictor/check_predictor_ckpt.py runs/training/predictor/epoch_0050.pt --strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Critical tensors to probe — if any of these is NaN / Inf / missing, the
# checkpoint is not usable. Covers input projections, all three attention
# paths in the first + last block, final norm, and every output head.
# ---------------------------------------------------------------------------

CRITICAL_KEYS_PREDICTOR: tuple[str, ...] = (
    "pose_proj.weight",
    "text_proj.weight",
    "time_tokens",
    "layers.0.self_attn.in_proj_weight",
    "layers.0.text_attn.in_proj_weight",
    "layers.0.object_attn.in_proj_weight",
    "layers.0.ffn.0.weight",
    "layers.9.self_attn.in_proj_weight",
    "layers.9.object_attn.in_proj_weight",
    "final_norm.weight",
    "contact_head.weight",
    "target_head.weight",
    "phase_head.weight",
    "support_head.weight",
)

CRITICAL_KEYS_OBJECT_ENCODER: tuple[str, ...] = (
    "sa1.mlp.0.weight",
    "sa2.mlp.0.weight",
    "refine.fc1.weight",
    "refine.fc2.weight",
    "refine.norm.weight",
)


# ---------------------------------------------------------------------------
# Tensor health check
# ---------------------------------------------------------------------------

def _check_tensor(name: str, t: torch.Tensor) -> tuple[str, bool]:
    """Return a printable summary line + bool flagging bad weights."""
    t = t.float()
    mean = t.mean().item()
    std = t.std().item()
    mn = t.min().item()
    mx = t.max().item()
    has_nan = bool(torch.isnan(t).any().item())
    has_inf = bool(torch.isinf(t).any().item())
    is_degenerate = std == 0.0
    bad = has_nan or has_inf or is_degenerate

    shape = tuple(t.shape)
    tag = "  [BAD]" if bad else ""
    line = (
        f"  {name:<46s} shape={str(shape):<22s} "
        f"mean={mean:+.3e}  std={std:.3e}  min={mn:+.3e}  max={mx:+.3e}"
        f"  nan={int(has_nan)}  inf={int(has_inf)}{tag}"
    )
    return line, bad


def _inspect_state_dict(
    state_dict: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    label: str,
) -> int:
    """Report stats on the requested keys in one state_dict.

    Returns the number of keys that were missing, NaN, Inf, or degenerate.
    """
    print(f"\n[{label}]  ({len(state_dict)} tensors in state_dict)")
    issues = 0
    for k in keys:
        if k in state_dict:
            line, bad = _check_tensor(k, state_dict[k])
            print(line)
            if bad:
                issues += 1
        else:
            print(f"  {k:<46s}  <MISSING>")
            issues += 1
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument(
        "checkpoint", type=Path,
        help="path to the .pt checkpoint (typically runs/training/predictor/final.pt)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 if any expected module is missing or any critical "
             "tensor is NaN / Inf / degenerate; useful for CI gating",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Top-level keys: {sorted(ck.keys())}")
    print(f"Epoch: {ck.get('epoch')}    global_step: {ck.get('global_step')}")

    n_issues = 0

    if "model" in ck:
        n_issues += _inspect_state_dict(
            ck["model"], CRITICAL_KEYS_PREDICTOR, "predictor  (key='model')",
        )
    else:
        print("\n[predictor]  <NOT IN CHECKPOINT>")
        n_issues += 1

    if "object_encoder" in ck:
        n_issues += _inspect_state_dict(
            ck["object_encoder"], CRITICAL_KEYS_OBJECT_ENCODER, "object_encoder",
        )
    else:
        print("\n[object_encoder]  <NOT IN CHECKPOINT>")
        print("  WARNING: predictor alone cannot do inference without the")
        print("  object encoder weights. Re-train with the updated trainer")
        print("  that saves both modules.")
        n_issues += 1

    print()
    if n_issues:
        print(f"[FAIL] {n_issues} issue(s) found. "
              f"{'Exiting nonzero.' if args.strict else 'Run with --strict to fail the CI gate.'}")
    else:
        print("[OK] checkpoint looks healthy.")

    return 1 if (n_issues and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
