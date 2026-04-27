"""Diagnostic dump for a Stage B generator checkpoint.

Prints + saves to JSON, for every requested ckpt:

  - Per-layer ``γ_int`` values. **For per-head γ (v0.6+) the full
    8×n_heads matrix is reported**, plus per-layer abs-mean / abs-max /
    abs-min, per-head abs-mean across layers (does some particular
    head do all the work?), sign distribution, "dead head" count
    below a threshold, and top/bottom-K heads by |γ| with their
    (layer, head) coordinates. For scalar γ (v0.1-v0.5) the report
    collapses to per-layer scalars (backward-compatible).
  - Per-block IntXAttn weight norms (``int_attn.in_proj_weight``
    split into Q/K/V + ``int_attn.out_proj.weight``). With per-head
    γ_int, also reports W_O column-Frobenius per head (the magnitude
    each head's attention output gets amplified by **before** the γ
    gate). Combined: ``effective_per_head_gain = |γ| · ||W_O[:, h]||``
    is a more honest "is this head contributing?" indicator than γ
    alone.
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
        --ckpt-dir runs/training/generator_v06_per_head_gamma \\
        --output runs/training/generator_v06_per_head_gamma/inspect_summary.json

Pure inspection — does not run any training step or load the
heavyweight CLIP / VQ-VAE / MaskTransformer; we only need the saved
state_dict to read γ_int + W_O + null_int_kv tensors and tally params.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


# Below this absolute γ value, a head is treated as "dead" (didn't
# learn to use this layer's IntXAttn). 5e-3 is ~1/5 of the typical
# converged abs-mean (~0.025) so this is a generous threshold —
# below it the head's contribution is well into noise.
DEAD_HEAD_THRESHOLD = 5e-3


def _load_state(ckpt_path: Path) -> dict[str, Any]:
    """Load a Stage B checkpoint payload (top-level keys: model,
    optimizer, epoch, global_step). Returns the model state_dict."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"checkpoint {ckpt_path} has no 'model' key")
    return payload


def _layer_index_from_key(key: str) -> int | None:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


def _is_int_attn_block_key(key: str) -> bool:
    """True for keys under ``...seqTransEncoder.layers.<i>.<sublayer>.*``
    where ``<sublayer>`` is part of the IntXAttn pieces (anything other
    than the wrapped MoMask ``.layer.*``)."""
    parts = key.split(".")
    if "seqTransEncoder" not in parts or "layers" not in parts:
        return False
    idx = parts.index("layers")
    if idx + 2 >= len(parts):
        return False
    return parts[idx + 2] != "layer"


# ============================================================================
# γ_int — handles BOTH scalar (v0.1-v0.5) and per-head (v0.6+) shapes.
# ============================================================================

def _tally_gamma_int(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Pull every ``...gamma_int`` parameter out of the state_dict.

    For each layer, ``γ_int`` is one of:
      - shape ``(1,)`` → scalar gate (v0.1-v0.5).
      - shape ``(n_heads,)`` → per-head gate (v0.6+ LLaMA-Adapter style).

    Returns a dict whose ``kind`` field reflects the detected shape and
    whose ``per_layer_per_head`` is the full 8×n_heads matrix when
    per_head, else ``None``. ``per_layer`` always carries an 8-vector
    of layer scalars (= the value itself for scalar, = abs-mean across
    heads for per-head) for backward compat.
    """
    gamma_keys = sorted(
        [k for k in state if k.endswith("gamma_int") or "gamma_int" == k.split(".")[-1]],
        key=lambda k: (_layer_index_from_key(k) or 0, k),
    )
    if not gamma_keys:
        return {"present": False}

    # Discover shape from the first one. All layers should agree.
    first = state[gamma_keys[0]].detach().float()
    if first.ndim != 1:
        return {"present": True, "kind": "unknown", "warning": f"unexpected γ shape {tuple(first.shape)}"}
    if first.numel() == 1:
        kind = "scalar"
        n_heads = 1
    else:
        kind = "per_head"
        n_heads = int(first.numel())

    # Collect per-layer values.
    per_layer_full: dict[int, list[float]] = {}
    for k in gamma_keys:
        idx = _layer_index_from_key(k)
        if idx is None:
            idx = len(per_layer_full)
        v = state[k].detach().float().flatten().tolist()
        if len(v) != n_heads:
            return {
                "present": True, "kind": kind,
                "warning": f"layer {idx} γ has {len(v)} elems but expected {n_heads}",
            }
        per_layer_full[idx] = v

    layer_indices = sorted(per_layer_full.keys())
    matrix = [per_layer_full[i] for i in layer_indices]                # list[list[float]]
    n_layers = len(matrix)

    # Per-layer scalars: scalar mode = the value, per_head mode = abs-mean.
    if kind == "scalar":
        per_layer = [matrix[i][0] for i in range(n_layers)]
    else:
        per_layer = [
            sum(abs(x) for x in matrix[i]) / n_heads
            for i in range(n_layers)
        ]
    abs_per_layer = [abs(x) for x in per_layer]

    out: dict[str, Any] = {
        "present": True,
        "kind": kind,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "per_layer_per_head": matrix if kind == "per_head" else None,
        "per_layer": per_layer,
        "per_layer_abs_mean": abs_per_layer,
        "abs_mean": sum(abs_per_layer) / n_layers,
        "abs_max": max(abs_per_layer),
        "abs_min": min(abs_per_layer),
    }

    if kind == "per_head":
        # Aggregate across layers per head → does any specific head do
        # all the work or are heads roughly balanced?
        per_head_abs_mean = [
            sum(abs(matrix[L][h]) for L in range(n_layers)) / n_layers
            for h in range(n_heads)
        ]
        per_head_abs_max = [
            max(abs(matrix[L][h]) for L in range(n_layers))
            for h in range(n_heads)
        ]
        # Per-layer per-head abs stats.
        per_layer_abs_max = [max(abs(x) for x in matrix[L]) for L in range(n_layers)]
        per_layer_abs_min = [min(abs(x) for x in matrix[L]) for L in range(n_layers)]
        per_layer_abs_std = [
            (sum((abs(x) - abs_per_layer[L]) ** 2 for x in matrix[L]) / n_heads) ** 0.5
            for L in range(n_layers)
        ]

        # Sign + dead-head accounting over all layer×head dofs.
        flat = [(L, h, matrix[L][h]) for L in range(n_layers) for h in range(n_heads)]
        n_pos = sum(1 for _, _, v in flat if v >  DEAD_HEAD_THRESHOLD)
        n_neg = sum(1 for _, _, v in flat if v < -DEAD_HEAD_THRESHOLD)
        n_dead = sum(1 for _, _, v in flat if abs(v) <= DEAD_HEAD_THRESHOLD)

        # Top-K / bottom-K heads by |γ|.
        flat_sorted = sorted(flat, key=lambda t: abs(t[2]))
        K = min(8, len(flat))
        bottom_k = [
            {"layer": L, "head": h, "value": round(v, 6)}
            for L, h, v in flat_sorted[:K]
        ]
        top_k = [
            {"layer": L, "head": h, "value": round(v, 6)}
            for L, h, v in flat_sorted[-K:][::-1]
        ]

        out.update({
            "per_layer_abs_max": per_layer_abs_max,
            "per_layer_abs_min": per_layer_abs_min,
            "per_layer_abs_std": per_layer_abs_std,
            "per_head_abs_mean_across_layers": per_head_abs_mean,
            "per_head_abs_max_across_layers": per_head_abs_max,
            "sign_distribution": {
                "n_dofs": len(flat),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_near_zero":  n_dead,
                "near_zero_threshold": DEAD_HEAD_THRESHOLD,
            },
            "top_k_by_abs": top_k,
            "bottom_k_by_abs": bottom_k,
        })

    return out


# ============================================================================
# IntXAttn weight norms (Q / K / V / O Frobenius) — diagnoses whether
# the cross-attn projections themselves carry signal, independent of γ.
# ============================================================================

def _tally_int_attn_weights(
    state: dict[str, torch.Tensor],
    gamma_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-layer Frobenius norms of the IntXAttn projection weights.

    PyTorch's ``nn.MultiheadAttention`` packs Q/K/V into a single
    ``in_proj_weight`` of shape ``(3*d, d)`` (or ``in_proj_weight=None``
    when separate q/k/v_proj_weight are used; we detect both).
    ``out_proj.weight`` is ``(d, d)``, mapping concatenated head
    outputs back to the residual stream.

    For per-head γ checkpoints, we additionally split
    ``out_proj.weight`` into ``n_heads`` column blocks of size
    ``(d, head_dim)`` and report per-head Frobenius norms — this is
    "how strongly each head's attention output is amplified before
    being γ-gated and added to the residual". Combined with γ, the
    product ``|γ_h| · ||W_O[:, h*hd:(h+1)*hd]||_F`` is a sharper
    "effective per-head gain" diagnostic.
    """
    # Discover (layer_idx → bag of weight tensors) for the int_attn sublayer.
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for k, v in state.items():
        if not _is_int_attn_block_key(k):
            continue
        # Only keep keys under the int_attn sub-module.
        if ".int_attn." not in k:
            continue
        idx = _layer_index_from_key(k)
        if idx is None:
            continue
        by_layer.setdefault(idx, {})[k] = v.detach().float()

    if not by_layer:
        return {"present": False}

    # Detect head dim from gamma_info if available, else assume 64
    # (MoMask's d_model=384, n_heads=6 → head_dim=64). We can also
    # recover from out_proj shape: out_proj.weight is (d, d).
    n_heads = None
    if gamma_info and gamma_info.get("kind") == "per_head":
        n_heads = gamma_info.get("n_heads")

    per_layer: list[dict[str, Any]] = []
    for idx in sorted(by_layer.keys()):
        bag = by_layer[idx]
        info: dict[str, Any] = {"layer_idx": idx}

        # In-proj: prefer packed in_proj_weight, fall back to separated.
        in_proj_keys = [k for k in bag if k.endswith("in_proj_weight")]
        if in_proj_keys:
            W = bag[in_proj_keys[0]]                       # (3d, d)
            if W.shape[0] % 3 == 0:
                d = W.shape[0] // 3
                Wq, Wk, Wv = W[:d], W[d:2 * d], W[2 * d:]
                info["Wq_fro"] = float(Wq.norm().item())
                info["Wk_fro"] = float(Wk.norm().item())
                info["Wv_fro"] = float(Wv.norm().item())

        # Out-proj.
        out_keys = [k for k in bag if k.endswith("out_proj.weight")]
        if out_keys:
            Wo = bag[out_keys[0]]                          # (d_out, d_in) where d_in = n_heads * head_dim
            d_out, d_in = Wo.shape
            info["Wo_fro"] = float(Wo.norm().item())
            info["Wo_shape"] = [d_out, d_in]

            # Per-head input-side split (columns = which head's output
            # this row reads from after the head concat).
            if n_heads is not None and d_in % n_heads == 0:
                head_dim = d_in // n_heads
                per_head_norms = []
                for h in range(n_heads):
                    block = Wo[:, h * head_dim:(h + 1) * head_dim]
                    per_head_norms.append(float(block.norm().item()))
                info["Wo_per_head_fro"] = per_head_norms
                info["head_dim"] = head_dim

        per_layer.append(info)

    out: dict[str, Any] = {
        "present": True,
        "n_layers": len(per_layer),
        "per_layer": per_layer,
    }

    # Aggregate Wo_per_head across layers (for per-head γ ckpts).
    if all("Wo_per_head_fro" in p for p in per_layer):
        n = len(per_layer[0]["Wo_per_head_fro"])
        agg_mean = [
            sum(p["Wo_per_head_fro"][h] for p in per_layer) / len(per_layer)
            for h in range(n)
        ]
        agg_max = [
            max(p["Wo_per_head_fro"][h] for p in per_layer)
            for h in range(n)
        ]
        out["Wo_per_head_aggregate"] = {
            "mean_across_layers": agg_mean,
            "max_across_layers":  agg_max,
        }

    return out


# ============================================================================
# Effective per-head gain — γ · ||W_O[:, head_block]||_F per (layer, head).
# Only meaningful when γ is per-head and W_O has been split per head.
# ============================================================================

def _compute_effective_per_head_gain(
    gamma_info: dict[str, Any],
    int_attn_info: dict[str, Any],
) -> dict[str, Any]:
    """Combine |γ_layer_head| with ||W_O[:, head]||_F to get a sharper
    "is this head contributing?" indicator than γ alone."""
    if gamma_info.get("kind") != "per_head":
        return {"present": False, "reason": "γ is scalar — no per-head split"}
    if not int_attn_info.get("present"):
        return {"present": False, "reason": "int_attn weights not found"}

    matrix = gamma_info.get("per_layer_per_head")
    if matrix is None:
        return {"present": False, "reason": "per_layer_per_head missing"}

    n_layers = len(matrix)
    per_layer_norms: list[list[float] | None] = []
    for L in range(n_layers):
        # Match by layer_idx in int_attn_info.
        match = next(
            (p for p in int_attn_info["per_layer"] if p["layer_idx"] == L),
            None,
        )
        if match is None or "Wo_per_head_fro" not in match:
            per_layer_norms.append(None)
            continue
        per_layer_norms.append(match["Wo_per_head_fro"])

    if any(x is None for x in per_layer_norms):
        return {"present": False, "reason": "missing W_O per-head split for some layer"}

    n_heads = len(matrix[0])
    eff = [
        [abs(matrix[L][h]) * per_layer_norms[L][h] for h in range(n_heads)]
        for L in range(n_layers)
    ]
    flat = [(L, h, eff[L][h]) for L in range(n_layers) for h in range(n_heads)]
    K = min(8, len(flat))
    flat_sorted = sorted(flat, key=lambda t: t[2])
    return {
        "present": True,
        "per_layer_per_head": eff,
        "abs_mean": sum(v for _, _, v in flat) / len(flat),
        "abs_max":  max(v for _, _, v in flat),
        "abs_min":  min(v for _, _, v in flat),
        "top_k": [
            {"layer": L, "head": h, "value": round(v, 6)}
            for L, h, v in flat_sorted[-K:][::-1]
        ],
        "bottom_k": [
            {"layer": L, "head": h, "value": round(v, 6)}
            for L, h, v in flat_sorted[:K]
        ],
    }


# ============================================================================
# null_int_kv stats (unchanged from prior version)
# ============================================================================

def _tally_null_int_kv(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Stats on the learnable null_int_kv bank."""
    keys = [k for k in state if k.endswith("null_int_kv") or k.endswith(".null_int_kv")]
    if not keys:
        return {"present": False}
    if len(keys) > 1:
        return {"present": True, "warning": f"multiple null_int_kv keys: {keys}"}
    t = state[keys[0]].detach().float()
    t = t.reshape(t.shape[0], -1)            # (S, d)
    per_token_norm = t.norm(dim=-1)
    return {
        "present": True,
        "shape": list(state[keys[0]].shape),
        "per_token_norm": {
            "mean": float(per_token_norm.mean().item()),
            "std":  float(per_token_norm.std().item()),
            "max":  float(per_token_norm.max().item()),
            "min":  float(per_token_norm.min().item()),
        },
        "abs_value": {
            "mean": float(t.abs().mean().item()),
            "max":  float(t.abs().max().item()),
        },
    }


# ============================================================================
# Tokenizer weight stats (final Linear projection that produces K/V).
# ============================================================================

def _tally_tokenizer_weights(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Last interaction tokenizer Linear weight: stats useful for
    sanity-checking that the upstream of IntXAttn isn't degenerate."""
    keys = [
        k for k in state
        if k.startswith("interaction_tokenizer.")
        and (k.endswith(".weight") or k.endswith(".bias"))
    ]
    if not keys:
        return {"present": False}
    by_param: dict[str, dict[str, float]] = {}
    for k in keys:
        t = state[k].detach().float()
        by_param[k] = {
            "shape": list(t.shape),
            "fro":   float(t.norm().item()),
            "abs_mean": float(t.abs().mean().item()),
            "abs_max":  float(t.abs().max().item()),
        }
    return {"present": True, "by_param": by_param}


# ============================================================================
# Param tally (unchanged)
# ============================================================================

def _tally_params(state: dict[str, torch.Tensor]) -> dict[str, Any]:
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

    counts["new_total"] = (
        counts["tokenizer"]
        + counts["new_xattn_per_block"]
        + counts["null_int_kv"]
    )
    counts["all_total"] = sum(
        counts[k] for k in ("tokenizer", "new_xattn_per_block",
                            "null_int_kv", "backbone", "clip", "other")
    )
    return counts


# ============================================================================
# Top-level: one ckpt
# ============================================================================

def _inspect_one(ckpt_path: Path) -> dict[str, Any]:
    payload = _load_state(ckpt_path)
    state = payload["model"]
    gamma = _tally_gamma_int(state)
    int_attn = _tally_int_attn_weights(state, gamma_info=gamma)
    eff = _compute_effective_per_head_gain(gamma, int_attn)
    return {
        "path": str(ckpt_path),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "gamma_int": gamma,
        "int_attn_weights": int_attn,
        "effective_per_head_gain": eff,
        "null_int_kv": _tally_null_int_kv(state),
        "tokenizer_weights": _tally_tokenizer_weights(state),
        "params": _tally_params(state),
    }


# ============================================================================
# Stdout pretty-print
# ============================================================================

def _print_gamma(g: dict[str, Any]) -> None:
    if not g.get("present"):
        print("  γ_int: NOT FOUND — wrapper not applied?")
        return
    if g.get("warning"):
        print(f"  γ_int: warning — {g['warning']}")
        return

    kind = g["kind"]
    n_layers = g["n_layers"]
    n_heads = g["n_heads"]
    print(f"  γ_int: kind={kind}, n_layers={n_layers}, n_heads={n_heads}")
    print(
        f"    abs_mean={g['abs_mean']:.5f}  "
        f"abs_max={g['abs_max']:.5f}  abs_min={g['abs_min']:.5f}"
    )

    if kind == "scalar":
        per = " ".join(f"{v:+.4f}" for v in g["per_layer"])
        print(f"    per-layer (8): [{per}]")
        return

    # Per-head matrix.
    matrix = g["per_layer_per_head"]
    head_cols = "  ".join(f"h{h:>1}" + " " * 5 for h in range(n_heads))
    print(f"    per-layer × per-head matrix:")
    print(f"          {head_cols} | abs_mean")
    for L in range(n_layers):
        row = "  ".join(f"{matrix[L][h]:+.4f}" for h in range(n_heads))
        print(f"      L{L}  {row}  | {g['per_layer_abs_mean'][L]:.4f}")
    print(f"    per-head abs-mean across layers:")
    for h, v in enumerate(g["per_head_abs_mean_across_layers"]):
        print(f"      h{h}: {v:.5f}    "
              f"(max across layers: {g['per_head_abs_max_across_layers'][h]:.5f})")
    sd = g["sign_distribution"]
    print(
        f"    sign distribution ({sd['n_dofs']} dofs total, |γ|≤{sd['near_zero_threshold']:.0e} = near-zero):"
    )
    print(
        f"      positive: {sd['n_positive']}  "
        f"negative: {sd['n_negative']}  "
        f"near-zero: {sd['n_near_zero']}"
    )
    print("    top-K |γ|:    " + ", ".join(
        f"L{e['layer']}h{e['head']}={e['value']:+.4f}" for e in g["top_k_by_abs"]
    ))
    print("    bottom-K |γ|: " + ", ".join(
        f"L{e['layer']}h{e['head']}={e['value']:+.4f}" for e in g["bottom_k_by_abs"]
    ))


def _print_int_attn(w: dict[str, Any]) -> None:
    if not w.get("present"):
        print("  int_attn weights: NOT FOUND")
        return
    print(f"  int_attn weight norms ({w['n_layers']} layers):")
    has_q = "Wq_fro" in w["per_layer"][0]
    has_per_head = "Wo_per_head_fro" in w["per_layer"][0]
    if has_q:
        header = "        Wq_fro    Wk_fro    Wv_fro    Wo_fro"
    else:
        header = "        Wo_fro"
    print(header)
    for p in w["per_layer"]:
        if has_q:
            print(
                f"      L{p['layer_idx']}  "
                f"{p.get('Wq_fro', float('nan')):>7.3f}   "
                f"{p.get('Wk_fro', float('nan')):>7.3f}   "
                f"{p.get('Wv_fro', float('nan')):>7.3f}   "
                f"{p.get('Wo_fro', float('nan')):>7.3f}"
            )
        else:
            print(f"      L{p['layer_idx']}  {p.get('Wo_fro', float('nan')):>7.3f}")

    if has_per_head:
        print(f"    W_O per-head Frobenius norms:")
        n_heads = len(w["per_layer"][0]["Wo_per_head_fro"])
        head_cols = "  ".join(f"h{h:>1}    " for h in range(n_heads))
        print(f"          {head_cols}")
        for p in w["per_layer"]:
            row = "  ".join(f"{x:>5.3f}" for x in p["Wo_per_head_fro"])
            print(f"      L{p['layer_idx']}  {row}")
        agg = w.get("Wo_per_head_aggregate", {})
        if agg:
            print(f"    W_O per-head mean across layers: " + ", ".join(
                f"h{h}={x:.3f}" for h, x in enumerate(agg["mean_across_layers"])
            ))


def _print_effective_gain(eff: dict[str, Any]) -> None:
    if not eff.get("present"):
        return
    print(f"  effective per-head gain |γ| · ||W_O[:, head_block]||_F:")
    print(
        f"    abs_mean={eff['abs_mean']:.5f}  "
        f"abs_max={eff['abs_max']:.5f}  abs_min={eff['abs_min']:.5f}"
    )
    print("    top-K:    " + ", ".join(
        f"L{e['layer']}h{e['head']}={e['value']:.4f}" for e in eff["top_k"]
    ))
    print("    bottom-K: " + ", ".join(
        f"L{e['layer']}h{e['head']}={e['value']:.4f}" for e in eff["bottom_k"]
    ))


def _print_summary(label: str, info: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"  path:          {info['path']}")
    print(f"  epoch:         {info['epoch']}")
    print(f"  global_step:   {info['global_step']}")

    _print_gamma(info["gamma_int"])
    _print_int_attn(info["int_attn_weights"])
    _print_effective_gain(info["effective_per_head_gain"])

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

    tw = info["tokenizer_weights"]
    if tw.get("present"):
        # Just summarize: total Frobenius across all tokenizer params + count.
        total_fro = sum(p["fro"] for p in tw["by_param"].values())
        weight_count = sum(1 for k in tw["by_param"] if k.endswith(".weight"))
        bias_count = len(tw["by_param"]) - weight_count
        print(f"  tokenizer weights:           "
              f"{weight_count} weights + {bias_count} biases, "
              f"sum-Frobenius={total_fro:.2f}")

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
        "--ckpt-dir", type=Path, required=False, default=None,
        help="directory containing best_val.pt / final.pt / epoch_*.pt; "
             "use --ckpt for a single explicit ckpt path",
    )
    parser.add_argument(
        "--ckpt", type=Path, default=None,
        help="explicit checkpoint path (overrides --ckpt-dir / --names)",
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

    # Two modes: explicit single ckpt OR ckpt-dir + names.
    if args.ckpt is not None:
        if not args.ckpt.exists():
            print(f"ERROR: ckpt not found: {args.ckpt}")
            return 1
        info = _inspect_one(args.ckpt)
        _print_summary(args.ckpt.stem, info)
        summary = {args.ckpt.stem: info}
    else:
        if args.ckpt_dir is None or not args.ckpt_dir.exists():
            print(f"ERROR: --ckpt-dir not found: {args.ckpt_dir}")
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
        if not summary:
            return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary → {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
