"""Round-29 condition-usage probe — Phase 0 (mandatory pre-PB).

Per Codex review of `analyses/2026-05-29_round29_cond_injection_prior_review_for_codex.md`
§3, before any architectural ablation (PB1/PB2) we must directly measure
whether each R29 condition family (coarse_extra / interaction / support /
body_refine) is actually consumed by a trained Stage-2 ckpt — not just
inferred from paired-bootstrap comparisons across runs.

Mechanism:
  1. For each (ckpt, family, perturbation) combination, run the model's
     sampler on the same val 48-clip selection.
  2. Compare generated joints against an unperturbed baseline produced
     with the SAME random seed, so the resulting delta is the model's
     condition response, not denoising noise.
  3. Aggregate per-family deltas and emit Codex's heuristic triage labels:
       ignored, weakly_used, actively_used, temporally_used.

Perturbations (Codex §3.1):
  - baseline       : no perturbation; reference sample with seed
  - zero           : family tensor multiplied by 0
  - time_shuffle   : within each clip, permute the valid_T frames of the
                     family (padded frames untouched); see if the model
                     uses temporal structure
  - batch_shuffle  : swap the family tensor across clips inside a mini-
                     batch of 2 (rejects batch_size=1 if requested)
  - scale_0.5      : family tensor scaled by 0.5
  - scale_2.0      : family tensor scaled by 2.0

We only ever mutate the 4 R29 cond keys:
    stage2_coarse_extra, stage2_interaction, stage2_support, stage2_body_refine
Everything else — Stage-1 Coarse-v1, object tokens, init_pose, text — is
left exactly as the dataset / oracle produced it.

Outputs:
    <output-dir>/cond_usage_stats.json   # full aggregate with all rows
    <output-dir>/cond_usage_summary.md   # human-readable triage labels

Usage:
    python scripts/stage_b_generator/round29_cond_usage_probe.py \\
        --config configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml \\
        --ckpt   runs/training/stageB_anchordiff_r29_ns_a1_c41_s4_g1/final.pt \\
        --selection-json analyses/round29_val_diag_indices_48_balanced.json \\
        --bucket val \\
        --output-dir analyses/round29_cond_usage_a1_val \\
        --variant-id r29_ns_a1_c41_s4_g1

Each run probes ONE ckpt. The shell launcher loops over A1/R0/G1.

Pure helpers (perturbation primitives + label thresholds) are importable
without torch/omegaconf so unit tests can validate the math on synthetic
arrays.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# R29 condition family keys in cond dict. Mirrors
# src/piano/models/round29_cond_injection.py FAMILY_NAMES.
R29_COND_KEYS: tuple[str, ...] = (
    "stage2_coarse_extra",
    "stage2_interaction",
    "stage2_support",
    "stage2_body_refine",
)
FAMILY_OF_KEY: dict[str, str] = {
    "stage2_coarse_extra": "coarse_extra",
    "stage2_interaction": "interaction",
    "stage2_support": "support",
    "stage2_body_refine": "body_refine",
}
KEY_OF_FAMILY: dict[str, str] = {v: k for k, v in FAMILY_OF_KEY.items()}

# SMPL-22 indices for the 6 key joints in the summary.
KEY_JOINT_INDICES: dict[str, int] = {
    "left_wrist": 20, "right_wrist": 21,
    "left_ankle": 7, "right_ankle": 8,
    "neck": 12, "pelvis": 0,
}

# Codex §3.3 thresholds (heuristic triage labels, not paper claims).
THRESH_IGNORED_KEY_CM: float = 1.0       # zeroing changes key joints by <1 cm
THRESH_IGNORED_RELATIVE: float = 0.05    # <5% relative target metric change
THRESH_WEAK_KEY_CM: float = 3.0
THRESH_WEAK_RELATIVE: float = 0.15
THRESH_TEMPORALLY_USED_FRACTION: float = 1.20  # time_shuffle hurts ≥20% more than zero


# --------------------------------------------------------------------------- #
# Perturbation primitives (pure, importable without torch).
# --------------------------------------------------------------------------- #


def perturbation_zero(family_tensor: np.ndarray) -> np.ndarray:
    """Zero perturbation: replace the family tensor with zeros of same shape."""
    return np.zeros_like(family_tensor)


def perturbation_scale(family_tensor: np.ndarray, k: float) -> np.ndarray:
    """Scale family tensor by k."""
    return family_tensor * float(k)


def perturbation_time_shuffle(
    family_tensor: np.ndarray,
    valid_T: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Within each clip, permute only the valid_T frames; leave padded
    frames (indices >= valid_T) unchanged.

    family_tensor shape: (B, T, D)
    """
    if family_tensor.ndim != 3:
        raise ValueError(
            f"time_shuffle expects (B, T, D) tensor, got shape "
            f"{family_tensor.shape}"
        )
    if valid_T < 2:
        # Nothing to permute meaningfully; return as-is.
        return family_tensor.copy()
    out = family_tensor.copy()
    for b in range(family_tensor.shape[0]):
        perm = rng.permutation(min(valid_T, family_tensor.shape[1]))
        out[b, :len(perm)] = family_tensor[b, perm]
    return out


def perturbation_batch_shuffle(
    family_tensor: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Permute the batch dimension. Requires B >= 2 (caller responsibility).

    family_tensor shape: (B, T, D); returns shape (B, T, D) with rows shuffled.
    """
    if family_tensor.ndim != 3:
        raise ValueError(
            f"batch_shuffle expects (B, T, D) tensor, got shape "
            f"{family_tensor.shape}"
        )
    B = family_tensor.shape[0]
    if B < 2:
        raise ValueError(
            "batch_shuffle requires batch size >= 2; got 1. "
            "Caller must build a mini-batch of >= 2 clips for this "
            "perturbation."
        )
    # Derangement-style permutation: each row gets a different row's data.
    # Try up to 8 times; otherwise fall back to a roll-by-1 which is always
    # a valid derangement for B >= 2.
    for _ in range(8):
        perm = rng.permutation(B)
        if not (perm == np.arange(B)).any():
            return family_tensor[perm].copy()
    perm = np.roll(np.arange(B), 1)
    return family_tensor[perm].copy()


# --------------------------------------------------------------------------- #
# Aggregation + labelling (pure, importable).
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PerClipPerturbationResult:
    """One row: (variant_id, family, perturbation, clip) → delta metrics."""
    variant_id: str
    bucket: str
    family: str
    perturbation: str
    subset: str
    seq_id: str
    pred_delta_joints_cm_mean: float
    pred_delta_joints_cm_p95: float
    key_joint_delta_cm: dict[str, float]


def _per_joint_delta_cm(
    joints_a: np.ndarray, joints_b: np.ndarray,
) -> np.ndarray:
    """Per-frame, per-joint Euclidean error in cm. Shape (T, 22)."""
    T = min(joints_a.shape[0], joints_b.shape[0])
    diff = joints_a[:T] - joints_b[:T]
    err_m = np.linalg.norm(diff, axis=-1)
    return err_m * 100.0


def compute_clip_delta(
    base_joints: np.ndarray, pert_joints: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    """Mean / p95 of per-frame-joint delta + per-key-joint mean delta."""
    err = _per_joint_delta_cm(base_joints, pert_joints)   # (T, 22)
    if err.size == 0:
        return float("nan"), float("nan"), {
            n: float("nan") for n in KEY_JOINT_INDICES
        }
    mean_cm = float(err.mean())
    p95_cm = float(np.percentile(err, 95))
    key_joint = {}
    for name, idx in KEY_JOINT_INDICES.items():
        key_joint[name] = float(err[:, idx].mean())
    return mean_cm, p95_cm, key_joint


def aggregate_per_family(
    rows: list[PerClipPerturbationResult],
) -> dict[str, Any]:
    """Aggregate (family, perturbation) over clips."""
    out: dict[str, dict[str, Any]] = {}
    by_fam: dict[str, list[PerClipPerturbationResult]] = {}
    for r in rows:
        by_fam.setdefault(r.family, []).append(r)
    for family, frows in by_fam.items():
        by_pert: dict[str, list[PerClipPerturbationResult]] = {}
        for r in frows:
            by_pert.setdefault(r.perturbation, []).append(r)
        out[family] = {}
        for pert, prows in by_pert.items():
            means = [r.pred_delta_joints_cm_mean for r in prows]
            p95s = [r.pred_delta_joints_cm_p95 for r in prows]
            key_means: dict[str, list[float]] = {n: [] for n in KEY_JOINT_INDICES}
            for r in prows:
                for n, v in r.key_joint_delta_cm.items():
                    if math.isfinite(v):
                        key_means[n].append(v)
            out[family][pert] = {
                "n_clips": len(prows),
                "pred_delta_joints_cm_mean": float(np.mean(means)) if means else None,
                "pred_delta_joints_cm_p95": float(np.mean(p95s)) if p95s else None,
                "key_joint_delta_cm_mean": {
                    n: (float(np.mean(vs)) if vs else None)
                    for n, vs in key_means.items()
                },
            }
    return out


def label_family_usage(
    family_agg: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Apply Codex §3.3 thresholds. ``family_agg`` is one family's
    perturbation dict from ``aggregate_per_family``.

    Returns a dict containing:
      label: ignored / weakly_used / actively_used / unknown
      temporally_used: True if time_shuffle delta > THRESH_TEMPORALLY_USED_FRACTION × zero delta
      reasons: list of human-readable reasons
    """
    reasons: list[str] = []
    zero = family_agg.get("zero", {})
    zero_mean = zero.get("pred_delta_joints_cm_mean")
    zero_key_joint_max = None
    kj = zero.get("key_joint_delta_cm_mean") or {}
    finite_kj = [v for v in kj.values() if v is not None and math.isfinite(v)]
    if finite_kj:
        zero_key_joint_max = max(finite_kj)

    if zero_mean is None or zero_key_joint_max is None:
        return {
            "label": "unknown",
            "temporally_used": False,
            "reasons": ["zero perturbation result missing"],
        }

    # Primary label by zero perturbation magnitude (Codex §3.3).
    if zero_key_joint_max < THRESH_IGNORED_KEY_CM:
        label = "ignored"
        reasons.append(
            f"zero key-joint max = {zero_key_joint_max:.2f} cm < "
            f"{THRESH_IGNORED_KEY_CM:.2f} cm"
        )
    elif zero_key_joint_max < THRESH_WEAK_KEY_CM:
        label = "weakly_used"
        reasons.append(
            f"zero key-joint max = {zero_key_joint_max:.2f} cm in "
            f"[{THRESH_IGNORED_KEY_CM:.2f}, {THRESH_WEAK_KEY_CM:.2f}] cm"
        )
    else:
        label = "actively_used"
        reasons.append(
            f"zero key-joint max = {zero_key_joint_max:.2f} cm ≥ "
            f"{THRESH_WEAK_KEY_CM:.2f} cm"
        )

    # Independent "temporally_used" flag from time_shuffle vs zero.
    time_shuffle = family_agg.get("time_shuffle", {})
    time_shuffle_mean = time_shuffle.get("pred_delta_joints_cm_mean")
    temporally_used = False
    if (time_shuffle_mean is not None and zero_mean is not None
            and zero_mean > 1e-6):
        ratio = float(time_shuffle_mean) / float(zero_mean)
        if ratio > THRESH_TEMPORALLY_USED_FRACTION:
            temporally_used = True
            reasons.append(
                f"time_shuffle / zero = {ratio:.2f} > "
                f"{THRESH_TEMPORALLY_USED_FRACTION:.2f} → temporally used"
            )
        else:
            reasons.append(
                f"time_shuffle / zero = {ratio:.2f}; not temporally used"
            )
    return {
        "label": label,
        "temporally_used": temporally_used,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Diagnostic main (uses torch + dataset).
# --------------------------------------------------------------------------- #


PERTURBATIONS_DEFAULT: tuple[str, ...] = (
    "baseline", "zero", "time_shuffle", "batch_shuffle", "scale_0.5", "scale_2.0",
)


def _parse_perturbations(s: str) -> list[str]:
    out = [p.strip() for p in s.split(",") if p.strip()]
    for p in out:
        if p not in PERTURBATIONS_DEFAULT:
            raise ValueError(
                f"unknown perturbation {p!r}; must be one of "
                f"{PERTURBATIONS_DEFAULT}"
            )
    return out


def _parse_families(s: str | None, active: list[str]) -> list[str]:
    """Returns the families to probe. If ``s`` is None, probe every
    active family. Otherwise comma-separated list, validated against
    active list."""
    if s is None:
        return active
    out = [f.strip() for f in s.split(",") if f.strip()]
    for f in out:
        if f not in FAMILY_OF_KEY.values():
            raise ValueError(f"unknown family {f!r}")
        if f not in active:
            raise ValueError(
                f"family {f!r} not active in this config; active families "
                f"are {active}"
            )
    return out


def _apply_perturbation(
    cond_b1: dict, key: str, pert: str, valid_T: int,
    rng: np.random.RandomState, cond_b2: dict | None = None,
) -> dict:
    """Return a copy of cond_b1 with the given family key perturbed.

    For batch_shuffle, also requires cond_b2 (a different clip's cond
    bundle); the family tensor on cond_b1 is replaced with cond_b2's.
    Other keys are unchanged.
    """
    import torch
    out = dict(cond_b1)
    if key not in cond_b1:
        return out   # family not present — nothing to perturb
    t = cond_b1[key]                                  # (1, T, D)
    if pert == "baseline":
        return out
    if pert == "zero":
        out[key] = torch.zeros_like(t)
        return out
    if pert == "scale_0.5":
        out[key] = t * 0.5
        return out
    if pert == "scale_2.0":
        out[key] = t * 2.0
        return out
    if pert == "time_shuffle":
        np_t = t.detach().cpu().numpy()
        perturbed_np = perturbation_time_shuffle(np_t, valid_T, rng)
        out[key] = torch.from_numpy(perturbed_np).to(t.device).to(t.dtype)
        return out
    if pert == "batch_shuffle":
        if cond_b2 is None or key not in cond_b2:
            raise ValueError(
                "batch_shuffle requires a second clip's cond bundle "
                "(cond_b2) with the same family key"
            )
        out[key] = cond_b2[key].clone()
        return out
    raise AssertionError(f"unreachable perturbation {pert!r}")


def _active_families_from_cfg(cfg) -> list[str]:
    """Return the family names with dim > 0 according to the model config."""
    den = cfg.model.denoiser
    active: list[str] = []
    if int(den.get("r29_coarse_extra_dim", 0)) > 0:
        active.append("coarse_extra")
    if int(den.get("r29_interaction_dim", 0)) > 0:
        active.append("interaction")
    if int(den.get("r29_support_dim", 0)) > 0:
        active.append("support")
    if int(den.get("r29_body_refine_dim", 0)) > 0:
        active.append("body_refine")
    return active


def _write_summary_md(
    out_path: Path,
    variant_id: str,
    bucket: str,
    ckpt: str,
    perturbations: list[str],
    aggregate: dict[str, dict[str, Any]],
    family_labels: dict[str, dict[str, Any]],
    n_clips: int,
) -> None:
    L: list[str] = []
    L.append(f"# Round-29 cond-usage probe — `{variant_id}` ({bucket})")
    L.append("")
    L.append(f"**Ckpt:** `{ckpt}`")
    L.append(f"**Clips:** {n_clips}")
    L.append(f"**Perturbations:** {', '.join(perturbations)}")
    L.append("")
    L.append("## Triage labels (per family)")
    L.append("")
    L.append("| family | label | temporally_used? | reasons |")
    L.append("| --- | --- | :---: | --- |")
    for family, lab in family_labels.items():
        reasons_str = "; ".join(lab.get("reasons", [])) or "—"
        tu = "✓" if lab.get("temporally_used") else "—"
        L.append(
            f"| `{family}` | **{lab.get('label', 'unknown')}** | {tu} | {reasons_str} |"
        )
    L.append("")
    L.append("## Per-family per-perturbation aggregate")
    L.append("")
    L.append(
        "Each row: mean over clips of per-clip mean / p95 of per-frame, "
        "per-joint Euclidean delta vs the baseline sample (with same seed). "
        "Key-joint columns are mean across clips of per-clip mean."
    )
    L.append("")
    for family, perts in aggregate.items():
        L.append(f"### `{family}`")
        L.append("")
        L.append(
            "| perturbation | n_clips | mean (cm) | p95 (cm) | LW | RW | LA | RA | Neck | Pelvis |"
        )
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for pert in perturbations:
            row = perts.get(pert)
            if row is None:
                L.append(
                    f"| {pert} | — | — | — | — | — | — | — | — | — |"
                )
                continue
            kj = row.get("key_joint_delta_cm_mean") or {}

            def _fmt(x, prec=2):
                if x is None or not math.isfinite(float(x)):
                    return "—"
                return f"{float(x):.{prec}f}"

            L.append(
                f"| {pert} | {row.get('n_clips', 0)} | "
                f"{_fmt(row.get('pred_delta_joints_cm_mean'))} | "
                f"{_fmt(row.get('pred_delta_joints_cm_p95'))} | "
                f"{_fmt(kj.get('left_wrist'))} | "
                f"{_fmt(kj.get('right_wrist'))} | "
                f"{_fmt(kj.get('left_ankle'))} | "
                f"{_fmt(kj.get('right_ankle'))} | "
                f"{_fmt(kj.get('neck'))} | "
                f"{_fmt(kj.get('pelvis'))} |"
            )
        L.append("")
    L.append("## How to read")
    L.append("")
    L.append(
        f"- `ignored` (zero key-joint max < {THRESH_IGNORED_KEY_CM:.2f} cm): "
        "model output barely changes when this family is zeroed. The "
        "family is effectively unused."
    )
    L.append(
        f"- `weakly_used` (in [{THRESH_IGNORED_KEY_CM:.2f}, "
        f"{THRESH_WEAK_KEY_CM:.2f}] cm): some impact but small."
    )
    L.append(
        f"- `actively_used` (≥ {THRESH_WEAK_KEY_CM:.2f} cm): clear impact on "
        "key joints."
    )
    L.append(
        f"- `temporally_used`: time_shuffle / zero > "
        f"{THRESH_TEMPORALLY_USED_FRACTION:.2f} ⇒ model uses the temporal "
        "structure of this family, not just its average magnitude."
    )
    L.append("")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Round-29 cond-usage probe — measures whether each active R29 "
            "condition family is consumed by a trained ckpt via zero / "
            "time_shuffle / batch_shuffle / scale perturbations."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--variant-id", required=True,
                        help="Label written into the JSON / MD output.")
    parser.add_argument("--families", default=None,
                        help="Comma-separated family names to probe; default "
                             "= all active families in the config.")
    parser.add_argument(
        "--perturbations", default=",".join(PERTURBATIONS_DEFAULT),
        help=f"Comma-separated; valid: {','.join(PERTURBATIONS_DEFAULT)}",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Heavy imports deferred so pure helpers above remain importable in tests.
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from piano.data.dataset import collate_hoi
    from piano.inference.diagnostic_helpers import (
        _build_cond, _build_dataset, _build_model, _fk_22joints,
        _stage1_norm_for_cfg, extract_train_time_meta,
    )
    from piano.utils.clip_utils import load_clip_text_encoder

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perturbations = _parse_perturbations(args.perturbations)
    active_families = _active_families_from_cfg(cfg)
    if not active_families:
        raise SystemExit(
            f"no active R29 families in config {args.config} — nothing to "
            f"probe. Check r29_coarse_extra_dim / r29_interaction_dim / "
            f"r29_support_dim / r29_body_refine_dim."
        )
    families = _parse_families(args.families, active_families)
    print(f"[cond_probe] variant={args.variant_id} bucket={args.bucket}")
    print(f"[cond_probe] active families: {active_families}")
    print(f"[cond_probe] probing families: {families}")
    print(f"[cond_probe] perturbations: {perturbations}")

    # Selection JSON.
    sel_obj = json.loads(args.selection_json.read_text("utf-8"))
    selection = (
        sel_obj.get("selected") or sel_obj.get("candidates")
        or sel_obj.get("clips") or []
    )
    if not selection:
        raise SystemExit(f"empty selection: {args.selection_json}")
    sel_pairs = {(e["subset"], e["seq_id"]) for e in selection}
    print(f"[cond_probe] selection: {len(sel_pairs)} clips")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    model, object_encoder = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_meta = extract_train_time_meta(state)
    model_state = state.get("model", state)
    model.load_state_dict(model_state)
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif "extra_modules" in state and "object_encoder" in state["extra_modules"]:
        object_encoder.load_state_dict(state["extra_modules"]["object_encoder"])
    if int(cfg.model.denoiser.get("text_dim", 0)) > 0:
        clip_model = load_clip_text_encoder(
            device=device,
            model_name=str(cfg.model.text_encoder.clip_version),
            download_root=str(cfg.model.text_encoder.get(
                "download_root", "cache/clip")),
        )
    else:
        clip_model = None
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    # Pre-collect ALL selected batches (single-clip) and their cond dicts
    # so batch_shuffle can pair them later.
    selected_batches: list[tuple[str, str, dict, dict, int]] = []
    for batch in loader:
        subset = str(batch["subset"][0]); seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in sel_pairs:
            continue
        with torch.no_grad():
            cond, T = _build_cond(
                batch, model, object_encoder, clip_model, cfg, device,
                stage1_norm=stage1_norm,
            )
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)
        selected_batches.append((subset, seq_id, batch, cond, valid_T))
        if len(selected_batches) % 8 == 0:
            print(f"  [cond_probe] precomputed {len(selected_batches)} clips")
    if not selected_batches:
        raise SystemExit("no clips matched selection in dataset")
    print(f"[cond_probe] precomputed {len(selected_batches)} cond bundles")

    # Helper to sample with seed reset.
    def _sample(cond_in: dict, T: int) -> torch.Tensor:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            return model.sample(
                cond=cond_in, seq_length=T, cfg_scale=args.cfg_scale,
            )

    rng = np.random.RandomState(args.seed)
    rows: list[PerClipPerturbationResult] = []

    # For batch_shuffle we pair each clip with the NEXT clip (with wrap-around).
    n_clips = len(selected_batches)
    pair_idx = [(i + 1) % n_clips for i in range(n_clips)]

    for i, (subset, seq_id, batch, cond, valid_T) in enumerate(selected_batches):
        T = cond["object_world_traj"].shape[1]
        rest_offsets = batch["rest_offsets"].to(device).float()

        # Baseline sample for this clip — anchored by seed.
        pred_motion_base = _sample(cond, T)
        base_joints = _fk_22joints(
            pred_motion_base, rest_offsets,
        )[0, :valid_T].cpu().numpy()

        for family in families:
            cond_key = KEY_OF_FAMILY[family]
            if cond_key not in cond:
                # Family active in config but the dataset didn't emit it
                # (shouldn't happen on R29 configs — sanity-skip).
                continue
            for pert in perturbations:
                if pert == "baseline":
                    # zero delta against itself
                    rows.append(PerClipPerturbationResult(
                        variant_id=args.variant_id, bucket=args.bucket,
                        family=family, perturbation=pert,
                        subset=subset, seq_id=seq_id,
                        pred_delta_joints_cm_mean=0.0,
                        pred_delta_joints_cm_p95=0.0,
                        key_joint_delta_cm={n: 0.0 for n in KEY_JOINT_INDICES},
                    ))
                    continue
                cond_b2 = (
                    selected_batches[pair_idx[i]][3]
                    if pert == "batch_shuffle" else None
                )
                cond_pert = _apply_perturbation(
                    cond, cond_key, pert, valid_T, rng, cond_b2=cond_b2,
                )
                pred_motion_pert = _sample(cond_pert, T)
                pert_joints = _fk_22joints(
                    pred_motion_pert, rest_offsets,
                )[0, :valid_T].cpu().numpy()
                mean_cm, p95_cm, kj = compute_clip_delta(
                    base_joints, pert_joints,
                )
                rows.append(PerClipPerturbationResult(
                    variant_id=args.variant_id, bucket=args.bucket,
                    family=family, perturbation=pert,
                    subset=subset, seq_id=seq_id,
                    pred_delta_joints_cm_mean=mean_cm,
                    pred_delta_joints_cm_p95=p95_cm,
                    key_joint_delta_cm=kj,
                ))

        if (i + 1) % 4 == 0:
            print(
                f"  [cond_probe] {i + 1}/{n_clips} clips, "
                f"{len(rows)} rows so far"
            )

    print(f"[cond_probe] collected {len(rows)} rows total")

    aggregate = aggregate_per_family(rows)
    family_labels = {f: label_family_usage(aggregate[f]) for f in aggregate}

    out = {
        "variant_id": args.variant_id,
        "bucket": args.bucket,
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "selection_json": str(args.selection_json),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "n_clips": n_clips,
        "active_families": active_families,
        "probed_families": families,
        "perturbations": perturbations,
        "train_time": train_meta,
        "thresholds": {
            "ignored_key_cm": THRESH_IGNORED_KEY_CM,
            "weak_key_cm": THRESH_WEAK_KEY_CM,
            "temporally_used_fraction": THRESH_TEMPORALLY_USED_FRACTION,
        },
        "aggregate": aggregate,
        "family_labels": family_labels,
        "rows": [
            {
                "variant_id": r.variant_id, "bucket": r.bucket,
                "family": r.family, "perturbation": r.perturbation,
                "subset": r.subset, "seq_id": r.seq_id,
                "pred_delta_joints_cm_mean": r.pred_delta_joints_cm_mean,
                "pred_delta_joints_cm_p95": r.pred_delta_joints_cm_p95,
                "key_joint_delta_cm": r.key_joint_delta_cm,
            }
            for r in rows
        ],
    }
    out_json = args.output_dir / "cond_usage_stats.json"
    out_md = args.output_dir / "cond_usage_summary.md"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_summary_md(
        out_md, args.variant_id, args.bucket, str(args.ckpt),
        perturbations, aggregate, family_labels, n_clips,
    )
    print(f"[cond_probe] wrote {out_json}")
    print(f"[cond_probe] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
