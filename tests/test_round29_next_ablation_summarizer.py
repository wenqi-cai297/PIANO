"""Round-29 next-baseline ablation summarizer tests.

Validates ``scripts/stage_b_generator/round29_summarize_next_ablation.py``
on synthesised per-variant stats JSONs.

Specifically checks:

  - Report renders all 6 rows (R0 reference + B0/B1/G1/G2/H1).
  - Decision table uses correct references:
        B0  -> R0
        B1  -> R0 (primary) + B0 (secondary)
        G1  -> R0
        G2  -> R0 (primary) + G1 (secondary)
        H1  -> R0
  - R2-style degeneration flags fire when synthetic G1/G2 gait stats
    contain high both_swing / low both_stance / low trans/sec.
  - Report fails clearly (FileNotFoundError nonzero exit) when R0
    reference stats are missing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = (
    ROOT / "scripts" / "stage_b_generator"
    / "round29_summarize_next_ablation.py"
)

NEW_VARIANTS = (
    "r29_nb_b0_no_r29_cond",
    "r29_nb_b1_c41_only",
    "r29_nb_g1_phasefree_gait_fixed",
    "r29_nb_g2_strong_s4_oracle",
    "r29_nb_h1_r0_plus_oracle_full_hint",
)
R0_REF_VARIANT = "r29_ft_r0_clean_a3_baseline"
ALL_VARIANTS = (R0_REF_VARIANT, *NEW_VARIANTS)
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


def _make_gait_stats(
    *,
    gt_swing: float = 0.26,
    gt_stance: float = 0.19,
    pred_swing: float = 0.52,
    pred_stance: float = 0.09,
    pred_trans: float = 0.82,
) -> dict:
    return {
        "config": "fake.yaml", "ckpt": "fake.pt", "use_gt_as_pred": False,
        "fps": 20.0, "n_walking_segments": 59,
        "pred_aggregate": {
            "n_segments": 59,
            "frac_both_stance": {"mean": pred_stance},
            "frac_both_swing": {"mean": pred_swing},
            "frac_L_only_stance": {"mean": 0.20},
            "frac_R_only_stance": {"mean": 0.20},
            "transitions_per_second": {"mean": pred_trans},
            "L_R_height_corr": {"mean": -0.18},
            "step_period_frames": {"rate_with_period": 0.32, "mean": 28.0},
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


def _make_body_action_stats(delta_err_mean: float = 8.0) -> dict:
    joint_stats = {
        "delta_error_cm_mean": delta_err_mean,
        "delta_error_cm_median": delta_err_mean * 0.9,
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
            "left_wrist": joint_stats, "right_wrist": joint_stats,
            "left_knee": joint_stats, "right_knee": joint_stats,
            "neck": joint_stats, "pelvis": joint_stats,
        },
        "per_clip": [],
    }


def _seed_results_dir(
    results_root: Path, *,
    drift_per_variant: dict[str, float],
    gait_overrides: dict[str, dict] | None = None,
    variants: tuple[str, ...] = ALL_VARIANTS,
) -> None:
    """Create diag output dirs + minimal stats JSONs for the given variants."""
    gait_overrides = gait_overrides or {}
    for v in variants:
        for sub in SUBLABELS:
            for kind in ("sustained_contact", "gait", "body_action"):
                out_dir = results_root / f"round29_{v}_diag_{kind}_{sub}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if kind == "sustained_contact":
                    drift = drift_per_variant.get(v, 8.0)
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
                    gt_swing = 0.29 if sub == "train" else 0.26
                    gt_stance = 0.15 if sub == "train" else 0.19
                    override = gait_overrides.get(v, {})
                    if sub == "val":
                        data = _make_gait_stats(
                            gt_swing=gt_swing, gt_stance=gt_stance,
                            pred_swing=override.get("pred_swing", 0.52),
                            pred_stance=override.get("pred_stance", 0.09),
                            pred_trans=override.get("pred_trans", 0.82),
                        )
                    else:
                        data = _make_gait_stats(
                            gt_swing=gt_swing, gt_stance=gt_stance,
                        )
                else:
                    data = _make_body_action_stats(
                        delta_err_mean=drift_per_variant.get(v, 8.0) * 0.85,
                    )
                (out_dir / f"{kind}_stats.json").write_text(
                    json.dumps(data), encoding="utf-8",
                )


def test_summarizer_renders_all_rows(tmp_path: Path) -> None:
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    _seed_results_dir(
        results_root,
        drift_per_variant={
            R0_REF_VARIANT: 8.19,
            "r29_nb_b0_no_r29_cond": 14.0,
            "r29_nb_b1_c41_only": 8.5,
            "r29_nb_g1_phasefree_gait_fixed": 8.4,
            "r29_nb_g2_strong_s4_oracle": 8.6,
            "r29_nb_h1_r0_plus_oracle_full_hint": 6.8,
        },
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    for v in ALL_VARIANTS:
        assert f"`{v}`" in report, f"missing variant {v} in report"


def test_summarizer_decision_table_uses_correct_references(tmp_path: Path) -> None:
    """Decision table must compare:
        B0  -> R0
        B1  -> R0 + B0
        G1  -> R0
        G2  -> R0 + G1
        H1  -> R0
    """
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    _seed_results_dir(
        results_root,
        drift_per_variant={
            R0_REF_VARIANT: 8.19,
            "r29_nb_b0_no_r29_cond": 14.0,
            "r29_nb_b1_c41_only": 9.0,
            "r29_nb_g1_phasefree_gait_fixed": 8.4,
            "r29_nb_g2_strong_s4_oracle": 8.0,
            "r29_nb_h1_r0_plus_oracle_full_hint": 6.5,
        },
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    assert "Automatic decision table" in report
    # B0, B1, G1, G2, H1 each get a row with R0 as primary reference.
    assert "| `r29_nb_b0_no_r29_cond` | `r29_ft_r0_clean_a3_baseline` |" in report
    assert "| `r29_nb_b1_c41_only` | `r29_ft_r0_clean_a3_baseline` |" in report
    assert "| `r29_nb_g1_phasefree_gait_fixed` | `r29_ft_r0_clean_a3_baseline` |" in report
    assert "| `r29_nb_g2_strong_s4_oracle` | `r29_ft_r0_clean_a3_baseline` |" in report
    assert "| `r29_nb_h1_r0_plus_oracle_full_hint` | `r29_ft_r0_clean_a3_baseline` |" in report
    # Secondary references: B1 vs B0; G2 vs G1.
    assert "| `r29_nb_b1_c41_only` | `r29_nb_b0_no_r29_cond` |" in report
    assert "| `r29_nb_g2_strong_s4_oracle` | `r29_nb_g1_phasefree_gait_fixed` |" in report


def test_summarizer_flags_r2_style_degeneration(tmp_path: Path) -> None:
    """G1 or G2 with high both_swing + low both_stance + low trans/sec on val
    must be flagged as degenerate."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    # Make G1 look like R2's degenerate case on val gait stats.
    _seed_results_dir(
        results_root,
        drift_per_variant={v: 8.0 for v in ALL_VARIANTS},
        gait_overrides={
            "r29_nb_g1_phasefree_gait_fixed": dict(
                pred_swing=0.872,   # > 0.70 threshold
                pred_stance=0.004,  # < 0.02 threshold
                pred_trans=0.202,   # < 0.40 threshold
            ),
        },
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    assert "Gait degeneration flags" in report
    # G1 must be flagged with all three triggers.
    assert "`r29_nb_g1_phasefree_gait_fixed`" in report.split(
        "Gait degeneration flags"
    )[1]
    flagged_section = report.split("Gait degeneration flags")[1].split(
        "Per-variant decision questions"
    )[0]
    assert "frac_both_swing=0.872" in flagged_section
    assert "frac_both_stance=0.004" in flagged_section
    assert "transitions_per_sec=0.202" in flagged_section


def test_summarizer_no_degeneration_when_healthy(tmp_path: Path) -> None:
    """When all gait stats are healthy, the flags section says (none)."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    _seed_results_dir(
        results_root,
        drift_per_variant={v: 8.0 for v in ALL_VARIANTS},
        # Defaults: pred_swing=0.52, pred_stance=0.09, pred_trans=0.82 — all healthy.
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    assert "Gait degeneration flags" in report
    flagged_section = report.split("Gait degeneration flags")[1].split(
        "Per-variant decision questions"
    )[0]
    assert "(none — all new variants pass" in flagged_section


def test_summarizer_fails_when_r0_reference_missing(tmp_path: Path) -> None:
    """No R0 reference diag stats → summarizer must fail loudly (nonzero
    exit). Partial reports without R0 would mislead the verdict."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"
    # Seed only the new variants, no R0 reference.
    _seed_results_dir(
        results_root,
        drift_per_variant={v: 8.0 for v in NEW_VARIANTS},
        variants=NEW_VARIANTS,
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode != 0, "summarizer should fail when R0 ref missing"
    err = (res.stderr or "") + (res.stdout or "")
    assert "R0 reference" in err or "r29_ft_r0_clean_a3_baseline" in err


def test_summarizer_allow_missing_r0_escape_hatch(tmp_path: Path) -> None:
    """``--allow-missing-r0`` lets the test suite render without R0."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"
    _seed_results_dir(
        results_root,
        drift_per_variant={v: 8.0 for v in NEW_VARIANTS},
        variants=NEW_VARIANTS,
    )

    res = subprocess.run(
        [sys.executable, str(SUMMARIZER),
         "--results-root", str(results_root), "--out", str(out_md),
         "--allow-missing-r0"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    # All 5 new variants still listed; R0 row shows dashes.
    for v in NEW_VARIANTS:
        assert f"`{v}`" in report
    assert f"`{R0_REF_VARIANT}`" in report
