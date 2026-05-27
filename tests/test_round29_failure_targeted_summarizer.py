"""Round-29 failure-targeted ablation summarizer tests.

Validates the summarizer at
``scripts/stage_b_generator/round29_summarize_failure_targeted_ablation.py``
on synthesised per-variant stats JSONs. Specifically checks:

- All 6 variants are rendered.
- Per-subset GT row reads from gt_aggregate (not a hardcoded value).
- Per-part contact drift table is emitted (Codex Q7 — hands vs feet vs pelvis).
- Decision table compares R1-R5 against R0 on val drift_max + body delta.
- Empty results dir produces a clean placeholder report (no crash).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = (
    ROOT / "scripts" / "stage_b_generator"
    / "round29_summarize_failure_targeted_ablation.py"
)

VARIANTS = (
    "r29_ft_r0_clean_a3_baseline",
    "r29_ft_r1_no_coarse_extra",
    "r29_ft_r2_behavior_gait_loss",
    "r29_ft_r3_oracle_s4_gait_loss",
    "r29_ft_r4_i3_contact_lock",
    "r29_ft_r5_allpart_interaction_lock",
)
SUBLABELS = ("train", "val")
PER_PART_KEYS = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")


def _make_sustained_stats(
    *,
    drift_max_mean: float,
    drift_max_p95: float = 30.0,
    track_below: float = 0.06,
    per_part_drift: dict[str, float] | None = None,
) -> dict:
    pp = per_part_drift or {k: drift_max_mean * 1.5 for k in PER_PART_KEYS}
    return {
        "config": "fake.yaml", "ckpt": "fake.pt", "use_gt_as_pred": False,
        "overall": {
            "n_segments": 289,
            "drift_max_cm": {
                "mean": drift_max_mean, "median": drift_max_mean * 0.8,
                "p75": drift_max_mean * 1.3, "p95": drift_max_p95,
                "max": drift_max_p95 * 2.0,
            },
            "drift_end_cm": {"mean": 1.0, "median": 0.5},
            "drift_mean_cm": {"mean": drift_max_mean * 0.3, "median": drift_max_mean * 0.2},
            "tracking_fraction": {
                "mean": 1.0, "median": 1.0, "n": 141,
                "n_below_0.5": int(track_below * 141),
                "rate_below_0.5": track_below, "n_with_obj_motion": 141,
            },
            "rel_var_ratio": {"mean": 3.0, "median": 1.5, "n": 289},
            "n_drift_max_above_5cm": 200,
            "n_drift_max_above_10cm": 130,
        },
        "per_part": {
            k: {
                "drift_max_cm": {
                    "mean": v, "median": v * 0.8, "p75": v * 1.3,
                    "p95": v * 1.8, "max": v * 2.5,
                },
                "tracking_fraction": {"mean": 1.0},
            }
            for k, v in pp.items()
        },
        "per_subset": {},
        "rows": [],
    }


def _make_gait_stats(gt_swing: float, gt_stance: float, pred_swing: float = 0.5) -> dict:
    return {
        "config": "fake.yaml", "ckpt": "fake.pt", "use_gt_as_pred": False,
        "fps": 20.0, "n_walking_segments": 59,
        "pred_aggregate": {
            "n_segments": 59,
            "frac_both_stance": {"mean": 0.10},
            "frac_both_swing": {"mean": pred_swing},
            "frac_L_only_stance": {"mean": 0.20},
            "frac_R_only_stance": {"mean": 0.20},
            "transitions_per_second": {"mean": 0.80},
            "L_R_height_corr": {"mean": -0.10},
            "step_period_frames": {"rate_with_period": 0.20, "mean": 28.0},
        },
        "gt_aggregate": {
            "n_segments": 59,
            "frac_both_stance": {"mean": gt_stance},
            "frac_both_swing": {"mean": gt_swing},
            "frac_L_only_stance": {"mean": 0.25},
            "frac_R_only_stance": {"mean": 0.25},
            "transitions_per_second": {"mean": 0.92},
            "L_R_height_corr": {"mean": -0.32},
            "step_period_frames": {"rate_with_period": 0.40, "mean": 31.0},
        },
        "per_segment": [],
    }


def _make_body_action_stats(delta_err_mean: float = 12.0) -> dict:
    joint_stats = {
        "delta_error_cm_mean": delta_err_mean, "delta_error_cm_median": delta_err_mean * 0.9,
        "delta_error_p95_cm_mean": delta_err_mean * 1.5,
        "amp_gt_cm_mean": 25.0, "amp_gt_cm_median": 23.0,
        "amp_pred_cm_mean": 20.0, "amp_pred_cm_median": 18.0,
        "amp_ratio_mean": 0.80, "amp_ratio_median": 0.78,
        "direction_cosine_mean": 0.50, "direction_cosine_median": 0.48,
        "active_frame_frac_mean": 0.95, "active_frame_frac_median": 1.0,
        "energy_mask_gt_mean": 1.0, "energy_mask_gt_median": 1.0,
        "energy_mask_pred_mean": 1.0, "energy_mask_pred_median": 1.0,
    }
    return {
        "config": "fake.yaml", "ckpt": "fake.pt", "use_gt_as_pred": False,
        "energy_threshold_m": 0.05, "n_clips_processed": 48,
        "aggregate": {
            "left_wrist": joint_stats,
            "right_wrist": joint_stats,
            "left_knee": joint_stats,
            "right_knee": joint_stats,
            "neck": joint_stats,
            "pelvis": joint_stats,
        },
        "per_clip": [],
    }


def _seed_results_dir(results_root: Path, *, drift_per_variant: dict[str, float]) -> None:
    """Create 36 diag output dirs + minimal stats JSONs for 6 variants × 2 subsets × 3 kinds."""
    for v in VARIANTS:
        for sub in SUBLABELS:
            for kind in ("sustained_contact", "gait", "body_action"):
                out_dir = results_root / f"round29_{v}_diag_{kind}_{sub}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if kind == "sustained_contact":
                    drift = drift_per_variant.get(v, 13.0)
                    data = _make_sustained_stats(
                        drift_max_mean=drift,
                        per_part_drift={
                            "left_hand": drift * 1.8,
                            "right_hand": drift * 1.8,
                            "left_foot": drift * 0.8,
                            "right_foot": drift * 0.9,
                            "pelvis": drift * 0.3,
                        },
                    )
                elif kind == "gait":
                    # Train/val subsets get slightly different GT (mimics
                    # the real R29 LSF dataset).
                    gt_swing = 0.29 if sub == "train" else 0.26
                    gt_stance = 0.15 if sub == "train" else 0.19
                    data = _make_gait_stats(
                        gt_swing=gt_swing, gt_stance=gt_stance,
                        pred_swing=0.5 + 0.05 * (drift_per_variant.get(v, 13.0) - 13.0) / 5.0,
                    )
                else:
                    data = _make_body_action_stats(
                        delta_err_mean=drift_per_variant.get(v, 13.0) * 0.85,
                    )
                (out_dir / f"{kind}_stats.json").write_text(
                    json.dumps(data), encoding="utf-8",
                )


def test_summarizer_renders_six_variants(tmp_path: Path) -> None:
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    _seed_results_dir(
        results_root,
        drift_per_variant={
            "r29_ft_r0_clean_a3_baseline": 13.5,        # baseline
            "r29_ft_r1_no_coarse_extra": 14.0,           # slightly worse
            "r29_ft_r2_behavior_gait_loss": 13.4,        # ~equal contact
            "r29_ft_r3_oracle_s4_gait_loss": 13.6,
            "r29_ft_r4_i3_contact_lock": 9.5,            # big contact win
            "r29_ft_r5_allpart_interaction_lock": 8.5,   # even bigger
        },
    )

    res = subprocess.run(
        [
            sys.executable, str(SUMMARIZER),
            "--results-root", str(results_root),
            "--out", str(out_md),
        ],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")

    for v in VARIANTS:
        assert f"`{v}`" in report, f"missing variant {v} in report"


def test_summarizer_includes_per_subset_gt_row(tmp_path: Path) -> None:
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    _seed_results_dir(
        results_root,
        drift_per_variant={v: 13.0 for v in VARIANTS},
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0
    report = out_md.read_text(encoding="utf-8")
    assert "GT (train subset)" in report
    assert "GT (val subset)" in report
    # Train + val have different GT swing (0.29 / 0.26).
    train_section = report.split("## Subset: `train`")[1].split("## Subset: `val`")[0]
    val_section = report.split("## Subset: `val`")[1]
    assert "0.290" in train_section
    assert "0.260" in val_section


def test_summarizer_emits_per_part_drift_table(tmp_path: Path) -> None:
    """Codex Q7: per-part hand vs foot vs pelvis drift table must exist."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    _seed_results_dir(
        results_root,
        drift_per_variant={v: 13.0 for v in VARIANTS},
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0
    report = out_md.read_text(encoding="utf-8")
    # Per-part header must include all 5 parts.
    for part in PER_PART_KEYS:
        assert part in report, f"missing per-part column {part}"
    assert "Sustained contact (per part" in report


def test_summarizer_decision_table_compares_to_r0(tmp_path: Path) -> None:
    """The auto decision table must surface R4's contact gain vs R0."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    _seed_results_dir(
        results_root,
        drift_per_variant={
            "r29_ft_r0_clean_a3_baseline": 13.5,
            "r29_ft_r1_no_coarse_extra": 14.0,
            "r29_ft_r2_behavior_gait_loss": 13.4,
            "r29_ft_r3_oracle_s4_gait_loss": 13.6,
            "r29_ft_r4_i3_contact_lock": 9.5,   # -4.0 vs R0
            "r29_ft_r5_allpart_interaction_lock": 8.5,
        },
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0
    report = out_md.read_text(encoding="utf-8")
    assert "Automatic decision table" in report
    # R4's negative delta (improvement) and R5's bigger improvement must
    # both appear with sign.
    assert "-4.00" in report or "-4.0" in report  # R4 drift Δ
    assert "-5.00" in report or "-5.0" in report  # R5 drift Δ
    # Per-variant decision questions must be rendered.
    assert "is C41 extra" in report          # R1
    assert "contact-lock" in report          # R4
    assert "all-part interaction" in report  # R5


def test_summarizer_smoke_no_data(tmp_path: Path) -> None:
    """No diag dirs exist → report still renders with placeholder dashes."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"
    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    assert "GT (train subset)" in report
    assert "GT (val subset)" in report
    # All 6 variants still listed even with no data.
    for v in VARIANTS:
        assert f"`{v}`" in report
