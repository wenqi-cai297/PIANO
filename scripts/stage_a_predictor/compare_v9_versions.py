"""Compare Stage A predictor eval JSONs across v9.x versions.

Reads eval JSONs from runs/eval/stageA_predictor_*_val/ and prints a
side-by-side metric table for the headline numbers. Intended for the
v9.5 retrospective: does encoder finer-grained + hierarchical decoder
move the architectural ceiling we measured at topk_iou ~0.13 across
v9.1 / v9.2 / v9.4?

Usage:
    python scripts/stage_a_predictor/compare_v9_versions.py

Auto-detects which versions are available; missing JSONs are skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "runs" / "eval"

# (label, json path relative to runs/eval) — extend as new versions land.
SOURCES = [
    ("v9.1 baseline (best)",
     "stageA_predictor_v9_1_3way_support_val/predictor_v9_1_3way_support_val_best.json"),
    ("v9.2 with_joints (best, GT motion upper bound)",
     "stageA_predictor_v9_2_asl_motion_val/predictor_v9_2_with_joints_best.json"),
    ("v9.2 all_mask (best, Stage-B equivalent)",
     "stageA_predictor_v9_2_asl_motion_val/predictor_v9_2_all_mask_best.json"),
    ("v9.4 best",
     "stageA_predictor_v9_4_aux_xyz_pos_enc_val/predictor_v9_4_val_best.json"),
    ("v9.4 final",
     "stageA_predictor_v9_4_aux_xyz_pos_enc_val/predictor_v9_4_val_final.json"),
    ("v9.5 best (LOCAL)",
     "stageA_predictor_v9_5_finer_encoder_local_val/predictor_v9_5_val_best.json"),
    ("v9.5 final (LOCAL)",
     "stageA_predictor_v9_5_finer_encoder_local_val/predictor_v9_5_val_final.json"),
    # Server v9.5 (when it lands)
    ("v9.5 best (SERVER)",
     "stageA_predictor_v9_5_finer_encoder_val/predictor_v9_5_val_best.json"),
    ("v9.5 final (SERVER)",
     "stageA_predictor_v9_5_finer_encoder_val/predictor_v9_5_val_final.json"),
]


def _fmt(x, fmt: str = "{:.4f}", missing: str = "    n/a") -> str:
    return missing if x is None else fmt.format(x)


def main() -> int:
    rows = []
    for label, rel in SOURCES:
        p = EVAL_ROOT / rel
        if not p.exists():
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"[skip] {label}: failed to read {p}: {e}")
            continue
        t = d.get("target", {})
        c = d.get("contact", {})
        rows.append({
            "label": label,
            "ep": d.get("epoch", "?"),
            "topk3_iou": t.get("topk3_mean_iou"),
            "multihot_iou": t.get("multihot_mean_iou"),
            "tok_top1": t.get("token_top1_recall"),
            "tok_top3": t.get("token_top3_recall"),
            "patch_top1": t.get("patch_top1_acc"),
            "patch_top3": t.get("patch_top3_acc"),
            "L2_overall_cm": (t["mean_l2_m_overall_gated"] * 100)
                             if t.get("mean_l2_m_overall_gated") else None,
            "pct_5cm": t.get("pct_within_5cm_overall_gated"),
            "pct_10cm": t.get("pct_within_10cm_overall_gated"),
            "L2_LH_cm": _per_part_L2(t, "left_hand"),
            "L2_RH_cm": _per_part_L2(t, "right_hand"),
            "L2_LF_cm": _per_part_L2(t, "left_foot"),
            "L2_RF_cm": _per_part_L2(t, "right_foot"),
            "L2_pelvis_cm": _per_part_L2(t, "pelvis"),
            "contact_macroF1": c.get("macro_f1_over_body_parts"),
            "contact_anyF1": (c.get("any_part_f1") or {}).get("f1"),
            "contact_anyP": (c.get("any_part_f1") or {}).get("precision"),
            "contact_anyR": (c.get("any_part_f1") or {}).get("recall"),
            "phase_macroF1": d.get("phase", {}).get("macro_f1"),
            "support_macroF1": d.get("support", {}).get("macro_f1"),
        })

    if not rows:
        print("No eval JSONs found in", EVAL_ROOT)
        return 1

    # Print table — group rows by section (target / contact / aux) to
    # keep horizontal width manageable.
    print()
    print("=" * 88)
    print("Target — token-level (1-of-M ranking)")
    print(f"  {'version':<48s} ep   topk3_iou multi_iou  tok_top1  tok_top3")
    for r in rows:
        print(
            f"  {r['label']:<48s} {str(r['ep']):>3s} "
            f"{_fmt(r['topk3_iou'])} {_fmt(r['multihot_iou'])} "
            f"{_fmt(r['tok_top1'])} {_fmt(r['tok_top3'])}"
        )

    print()
    print("Target — patch-level (1-of-K=16, v9.5 hierarchical only)")
    print(f"  {'version':<48s} ep   patch_top1 patch_top3   (random: 0.062 / 0.188)")
    for r in rows:
        print(
            f"  {r['label']:<48s} {str(r['ep']):>3s} "
            f"{_fmt(r['patch_top1'])}   {_fmt(r['patch_top3'])}"
        )

    print()
    print("Target — xyz L2 (cm; pelvis target = chair seat ≈ 14cm baseline)")
    print(f"  {'version':<48s} overall  LH    RH    LF    RF   pelvis")
    for r in rows:
        print(
            f"  {r['label']:<48s} "
            f"{_fmt(r['L2_overall_cm'], '{:5.1f}')}  "
            f"{_fmt(r['L2_LH_cm'], '{:4.1f}')}  "
            f"{_fmt(r['L2_RH_cm'], '{:4.1f}')}  "
            f"{_fmt(r['L2_LF_cm'], '{:4.1f}')}  "
            f"{_fmt(r['L2_RF_cm'], '{:4.1f}')}  "
            f"{_fmt(r['L2_pelvis_cm'], '{:5.1f}')}"
        )

    print()
    print("Contact / phase / support (preserve gates)")
    print(f"  {'version':<48s} c_macroF1  c_anyP  c_anyR   ph_F1  sup_F1")
    for r in rows:
        print(
            f"  {r['label']:<48s} "
            f"{_fmt(r['contact_macroF1'])}    "
            f"{_fmt(r['contact_anyP'])} "
            f"{_fmt(r['contact_anyR'])}  "
            f"{_fmt(r['phase_macroF1'])}  "
            f"{_fmt(r['support_macroF1'])}"
        )
    return 0


def _per_part_L2(target_dict: dict, part: str) -> float | None:
    """Return per-part mean L2 in cm, or None if missing."""
    pp = target_dict.get("per_body_part", {})
    m = pp.get(part)
    if not m or m.get("mean_l2_m") is None:
        return None
    return float(m["mean_l2_m"] * 100)


if __name__ == "__main__":
    raise SystemExit(main())
