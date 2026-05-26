"""Round-29 loss-strategy FULL-DATA summarizer tests.

P1 regression test per Codex (2026-05-27): the v1 summarizer
hardcoded a single GT reference paragraph (`both_swing=0.291`,
`both_stance=0.153`, ...) for ALL subsets, but the val subset is
a DIFFERENT 48 clips with a different walking composition. The fix
reads ``gt_aggregate`` from each gait_stats.json and renders a
per-subset GT row.

This test synthesizes a fake results directory with two subsets'
gt_aggregate set to clearly different values, runs the summarizer,
and asserts the report contains BOTH GT references at the right
magnitudes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = (
    ROOT / "scripts" / "stage_b_generator"
    / "round29_summarize_loss_strategy_full_data.py"
)

VARIANTS = (
    "r29_lsf_a2_baseline_from_scratch",
    "r29_lsf_a2_anchor2_mixed",
    "r29_lsf_a3_baseline_from_scratch",
    "r29_lsf_a3_anchor2_mixed",
)
SUBLABELS = ("train", "val")


def _make_gait_stats(gt_swing: float, gt_stance: float, gt_trans: float,
                    gt_lr_corr: float, gt_step_rate: float,
                    pred_swing: float = 0.50) -> dict:
    """Build a minimal gait_stats.json structure with both pred + gt."""
    return {
        "config": "fake.yaml",
        "ckpt": "fake.pt",
        "use_gt_as_pred": False,
        "fps": 20.0,
        "n_walking_segments": 64,
        "pred_aggregate": {
            "n_segments": 64,
            "frac_both_stance": {"mean": 0.20},
            "frac_both_swing": {"mean": pred_swing},
            "frac_L_only_stance": {"mean": 0.15},
            "frac_R_only_stance": {"mean": 0.15},
            "transitions_per_second": {"mean": 0.40},
            "L_R_height_corr": {"mean": 0.88},
            "step_period_frames": {"rate_with_period": 0.05, "mean": 28.0},
        },
        "gt_aggregate": {
            "n_segments": 64,
            "frac_both_stance": {"mean": gt_stance},
            "frac_both_swing": {"mean": gt_swing},
            "frac_L_only_stance": {"mean": 0.25},
            "frac_R_only_stance": {"mean": 0.25},
            "transitions_per_second": {"mean": gt_trans},
            "L_R_height_corr": {"mean": gt_lr_corr},
            "step_period_frames": {"rate_with_period": gt_step_rate, "mean": 31.0},
        },
        "per_segment": [],
    }


def _make_sustained_stats() -> dict:
    return {
        "config": "fake.yaml", "ckpt": "fake.pt", "use_gt_as_pred": False,
        "n_clips_processed": 48, "n_segments": 251, "min_segment_len": 5,
        "obj_motion_min_cm": 10.0,
        "overall": {
            "n_segments": 251,
            "drift_max_cm": {"mean": 15.0, "median": 12.0, "p75": 20.0, "p95": 35.0, "max": 60.0},
            "drift_end_cm": {"mean": 1.0, "median": 0.5},
            "drift_mean_cm": {"mean": 3.0, "median": 2.0},
            "tracking_fraction": {"mean": 0.90, "median": 0.95, "n": 141, "n_below_0.5": 25, "rate_below_0.5": 0.25, "n_with_obj_motion": 141},
            "rel_var_ratio": {"mean": 3.0, "median": 1.5, "n": 251},
            "n_drift_max_above_5cm": 180,
            "n_drift_max_above_10cm": 130,
        },
        "per_part": {}, "per_subset": {}, "rows": [],
    }


def _make_body_action_stats() -> dict:
    joint_stats = {
        "delta_error_cm_mean": 20.0, "delta_error_cm_median": 18.0,
        "delta_error_p95_cm_mean": 35.0, "delta_error_p95_cm_median": 32.0,
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


def _seed_results_dir(
    results_root: Path,
    gt_train: dict[str, float],
    gt_val: dict[str, float],
) -> None:
    """Create the 24 expected diag output dirs + minimal stats JSONs.

    ``gt_train`` and ``gt_val`` each provide the five gait GT values
    (frac_both_swing, frac_both_stance, transitions_per_sec,
    L_R_height_corr, step_period_rate) for the corresponding subset.
    """
    for v in VARIANTS:
        for sub in SUBLABELS:
            gt = gt_train if sub == "train" else gt_val
            for kind in ("sustained_contact", "gait", "body_action"):
                out_dir = results_root / f"round29_{v}_diag_{kind}_{sub}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if kind == "gait":
                    data = _make_gait_stats(
                        gt_swing=gt["frac_both_swing"],
                        gt_stance=gt["frac_both_stance"],
                        gt_trans=gt["transitions_per_sec"],
                        gt_lr_corr=gt["L_R_height_corr"],
                        gt_step_rate=gt["step_period_rate"],
                    )
                elif kind == "sustained_contact":
                    data = _make_sustained_stats()
                else:
                    data = _make_body_action_stats()
                (out_dir / f"{kind}_stats.json").write_text(
                    json.dumps(data), encoding="utf-8",
                )


def test_summarizer_renders_per_subset_gt_row(tmp_path: Path) -> None:
    """Train-subset and val-subset GT values differ → report must show
    both, NOT a single hardcoded reference.
    """
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"

    gt_train = {
        "frac_both_swing": 0.291,    # matches the old hardcoded value
        "frac_both_stance": 0.153,
        "transitions_per_sec": 0.790,
        "L_R_height_corr": -0.309,
        "step_period_rate": 0.266,
    }
    gt_val = {
        # Deliberately different — val is a different 48 clips.
        "frac_both_swing": 0.412,
        "frac_both_stance": 0.099,
        "transitions_per_sec": 0.555,
        "L_R_height_corr": -0.450,
        "step_period_rate": 0.180,
    }
    _seed_results_dir(results_root, gt_train, gt_val)

    res = subprocess.run(
        [
            sys.executable, str(SUMMARIZER),
            "--results-root", str(results_root),
            "--out", str(out_md),
        ],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr
    assert out_md.exists()
    report = out_md.read_text(encoding="utf-8")

    # ----- Train-subset GT row must show the train GT numbers ------
    assert "GT (train subset)" in report
    # Both ``0.291`` (frac_both_swing) and ``0.153`` (frac_both_stance) must
    # appear inside the train-subset gait section.
    train_section_start = report.index("## Subset: `train`")
    val_section_start = report.index("## Subset: `val`")
    train_section = report[train_section_start:val_section_start]
    assert "0.291" in train_section, (
        f"train-subset section must contain GT frac_both_swing=0.291 "
        f"(read from gt_aggregate). Section:\n{train_section[:1000]}"
    )
    assert "0.153" in train_section, "train-subset section must contain GT frac_both_stance=0.153"
    assert "-0.309" in train_section, "train-subset section must contain GT L_R_corr=-0.309"

    # ----- Val-subset GT row must show DIFFERENT (val) GT numbers ------
    val_section = report[val_section_start:]
    assert "GT (val subset)" in val_section
    assert "0.412" in val_section, "val-subset section must contain GT frac_both_swing=0.412 (val ≠ train)"
    assert "0.099" in val_section, "val-subset section must contain GT frac_both_stance=0.099"
    assert "-0.450" in val_section, "val-subset section must contain GT L_R_corr=-0.450"

    # ----- Crucial cross-check: the v1 bug would show train values
    #       on val side. Make sure val section does NOT show train
    #       GT values that are not also val GT values.
    # frac_both_stance: train=0.153 vs val=0.099 — val section must
    # not contain "0.153" (which would mean the bug recurred).
    assert "0.153" not in val_section, (
        "val-subset section must not contain the train GT value 0.153 — "
        "that would indicate the v1 hardcoded-reference bug has recurred."
    )

    # ----- The v1 hardcoded reference paragraph should be gone -----
    assert (
        "GT physical reference for gait: `both_swing=0.291`" not in report
    ), (
        "v1 had a hardcoded single GT reference paragraph at the top of the "
        "report. The fix should have removed it."
    )


def test_summarizer_smoke_no_data(tmp_path: Path) -> None:
    """No diag dirs exist → report must still render cleanly with
    placeholder dashes everywhere, GT rows showing '-'."""
    results_root = tmp_path / "analyses"
    results_root.mkdir(parents=True)
    out_md = tmp_path / "report.md"
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
    assert "GT (train subset)" in report
    assert "GT (val subset)" in report
    # Both GT rows should be all dashes since no gt_aggregate was provided.
    train_section = report[
        report.index("## Subset: `train`"):report.index("## Subset: `val`")
    ]
    assert "| **GT (train subset)** | - | **-** | **-** | **-** | **-** | **-** |" in train_section
