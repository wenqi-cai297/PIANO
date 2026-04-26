"""Diagnostic dump for a Stage B generator checkpoint.

Prints + saves to JSON:
    - Per-layer ``γ_int`` values (8 numbers; useful to see if any layer
      stayed at 0 / saturated / sign of polarity).
    - ``null_int_kv`` per-token norm distribution (mean / std / max);
      should be small but non-zero. If huge, the null branch absorbed
      noise; if exactly zero, learning didn't happen.
    - Param counts: trainable backbone vs new (IntXAttn + tokenizer +
      γ + null_int_kv) — sanity-check vs the design doc's
      "~5M new + ~13M MoMask finetune" budget.
    - For comparison, optionally compares ``best_val.pt`` against
      ``final.pt`` so we can tell if the best-val ckpt over-fits.

Usage::

    python scripts/stage_b_generator/inspect_generator_ckpt.py \\
        --ckpt-dir runs/training/generator \\
        --model-config configs/model/motion_generator.yaml \\
        --output runs/training/generator/inspect_summary.json

Pure inspection — does not run any training step or load the
heavyweight CLIP / VQ-VAE / MaskTransformer; we only need the saved
state_dict to read γ_int + null_int_kv tensors and tally params.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_state(ckpt_path: Path) -> dict[str, Any]:
    """Load a Stage B checkpoint payload (top-level keys: model,
    optimizer, epoch, global_step). Returns the model state_dict."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"checkpoint {ckpt_path} has no 'model' key")
    return payload


def _tally_gamma_int(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Pull every ``...gamma_int`` parameter out of the state_dict and
    return per-layer values plus aggregate stats.

    Layer index is parsed from the state-dict key by looking for
    ``layers.<i>.``. Falls back to insertion order if not present.
    """
    gamma_keys = [k for k in state.keys() if k.endswith("gamma_int") or "gamma_int" in k.split(".")[-1]]
    # The wrapper stores them at
    # ``mask_transformer.seqTransEncoder.layers.<i>.gamma_int`` —
    # accept either bare or any prefix.
    gammas: dict[int, float] = {}
    for k in gamma_keys:
        # Find the last ".layers.<int>." segment.
        layer_idx = None
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        v = state[k].detach().float().flatten().tolist()
        # γ_int is shape (1,) — just take the first element if so.
        scalar = v[0] if len(v) == 1 else float(state[k].detach().float().abs().mean().item())
        if layer_idx is None:
            # Fallback: use whatever ordering we found
            gammas[len(gammas)] = scalar
        else:
            gammas[layer_idx] = scalar

    if not gammas:
        return {"present": False}

    values = [gammas[i] for i in sorted(gammas.keys())]
    abs_values = [abs(v) for v in values]
    return {
        "present": True,
        "n_layers": len(values),
        "per_layer": values,
        "mean_abs": sum(abs_values) / len(abs_values),
        "max_abs": max(abs_values),
        "min_abs": min(abs_values),
    }


def _tally_null_int_kv(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Stats on the learnable null_int_kv bank: norm per token, mean
    abs, max abs. The bank is a (S, 1, d) tensor; per-token norm
    has length S."""
    keys = [k for k in state.keys() if k.endswith("null_int_kv") or k.endswith(".null_int_kv")]
    if not keys:
        return {"present": False}
    if len(keys) > 1:
        return {"present": True, "warning": f"multiple null_int_kv keys: {keys}"}
    t = state[keys[0]].detach().float()
    # (S, 1, d) → squeeze batch dim if there
    t = t.reshape(t.shape[0], -1)            # (S, d)
    per_token_norm = t.norm(dim=-1)
    return {
        "present": True,
        "shape": list(state[keys[0]].shape),
        "per_token_norm": {
            "mean": float(per_token_norm.mean().item()),
            "std": float(per_token_norm.std().item()),
            "max": float(per_token_norm.max().item()),
            "min": float(per_token_norm.min().item()),
        },
        "abs_value": {
            "mean": float(t.abs().mean().item()),
            "max": float(t.abs().max().item()),
        },
    }


def _tally_params(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Split params into 4 buckets and report total counts.

    Buckets (matched against state_dict key prefixes):
      - tokenizer:  ``interaction_tokenizer.*``
      - new_xattn:  any ``layers.<i>.{int_attn, norm_int, gamma_int, dropout_int}``
      - null:       ``null_int_kv``
      - backbone:   everything else under ``mask_transformer.*``
                    (CLIP NOT included — clip_model.* would normally be
                    excluded from the saved state_dict by our trainer,
                    but if present we list them separately)
    """
    counts: dict[str, int] = {
        "tokenizer": 0,
        "new_xattn_per_block": 0,
        "null_int_kv": 0,
        "backbone": 0,
        "clip": 0,
        "other": 0,
    }
    for k, v in state.items():
        n = int(v.numel())
        if k.startswith("interaction_tokenizer."):
            counts["tokenizer"] += n
        elif k.endswith("null_int_kv") or k.endswith(".null_int_kv"):
            counts["null_int_kv"] += n
        elif "clip_model." in k:
            counts["clip"] += n
        elif k.startswith("mask_transformer."):
            # Per-block IntXAttn / γ / norm_int / dropout_int — these
            # live at ...seqTransEncoder.layers.<i>.<sublayer>.* where
            # <sublayer> is ANYTHING except "layer" (the wrapper holds
            # the original MoMask layer at ".layer.*"). Use that to
            # split.
            parts = k.split(".")
            if "seqTransEncoder" in parts and "layers" in parts:
                idx = parts.index("layers")
                if idx + 2 < len(parts):
                    sub = parts[idx + 2]
                    if sub == "layer":
                        counts["backbone"] += n
                    else:
                        counts["new_xattn_per_block"] += n
                else:
                    counts["backbone"] += n
            else:
                counts["backbone"] += n
        else:
            counts["other"] += n

    new_total = (
        counts["tokenizer"]
        + counts["new_xattn_per_block"]
        + counts["null_int_kv"]
    )
    counts["new_total"] = new_total
    counts["all_total"] = sum(
        counts[k] for k in ("tokenizer", "new_xattn_per_block",
                            "null_int_kv", "backbone", "clip", "other")
    )
    return counts


def _inspect_one(ckpt_path: Path) -> dict[str, Any]:
    """Build the full diagnostic dict for a single checkpoint."""
    payload = _load_state(ckpt_path)
    state = payload["model"]
    return {
        "path": str(ckpt_path),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "gamma_int": _tally_gamma_int(state),
        "null_int_kv": _tally_null_int_kv(state),
        "params": _tally_params(state),
    }


def _print_summary(label: str, info: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"  path:          {info['path']}")
    print(f"  epoch:         {info['epoch']}")
    print(f"  global_step:   {info['global_step']}")

    g = info["gamma_int"]
    if g.get("present"):
        per = " ".join(f"{v:+.4f}" for v in g["per_layer"])
        print(f"  γ_int per-layer ({g['n_layers']}): [{per}]")
        print(f"    abs mean: {g['mean_abs']:.5f}   max: {g['max_abs']:.5f}   min: {g['min_abs']:.5f}")
    else:
        print("  γ_int: NOT FOUND — wrapper not applied?")

    n = info["null_int_kv"]
    if n.get("present") and "per_token_norm" in n:
        nt = n["per_token_norm"]
        print(f"  null_int_kv shape:           {n['shape']}")
        print(f"  null_int_kv per-token norm:  mean={nt['mean']:.4f}  std={nt['std']:.4f}  max={nt['max']:.4f}")
        print(f"  null_int_kv abs:             mean={n['abs_value']['mean']:.4f}  max={n['abs_value']['max']:.4f}")
    elif n.get("present"):
        print(f"  null_int_kv: present (warning: {n.get('warning')})")
    else:
        print("  null_int_kv: NOT FOUND")

    p = info["params"]
    print(f"  params (M):")
    print(f"    tokenizer:           {p['tokenizer'] / 1e6:>7.3f}")
    print(f"    new_xattn_per_block: {p['new_xattn_per_block'] / 1e6:>7.3f}")
    print(f"    null_int_kv:         {p['null_int_kv'] / 1e6:>7.3f}")
    print(f"    -> new total:        {p['new_total'] / 1e6:>7.3f}")
    print(f"    backbone (MoMask):   {p['backbone'] / 1e6:>7.3f}")
    print(f"    clip:                {p['clip'] / 1e6:>7.3f}")
    print(f"    other:               {p['other'] / 1e6:>7.3f}")
    print(f"    ---------------")
    print(f"    all:                 {p['all_total'] / 1e6:>7.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt-dir", type=Path, required=True,
        help="directory containing best_val.pt / final.pt / epoch_*.pt",
    )
    parser.add_argument(
        "--names", nargs="+", default=["best_val", "final"],
        help="checkpoint names to inspect (without .pt extension)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="optional JSON output path",
    )
    args = parser.parse_args()

    if not args.ckpt_dir.exists():
        print(f"ERROR: ckpt-dir not found: {args.ckpt_dir}")
        return 1

    summary: dict[str, Any] = {}
    for name in args.names:
        ckpt_path = args.ckpt_dir / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"  [skip] {ckpt_path} not found")
            continue
        info = _inspect_one(ckpt_path)
        _print_summary(name, info)
        summary[name] = info

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary → {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
