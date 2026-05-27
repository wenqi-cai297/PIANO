"""Round-29 next-step ablation summarizer tests.

Per analyses/2026-05-28_round29_next_step_ablation_execution_prompt_for_claude_code.md §8.2.

Synthetic stats only — no real training outputs. Checks:

  - Missing required REFERENCE stats fails by default.
  - Missing required NEW-variant stats fails by default.
  - `--allow-partial` renders incomplete report.
  - Decision table includes A0 vs B1/G1, A1 vs A0/G1, H1 vs R0.
  - Paired bootstrap section appears with matched rows.
  - Invalid old H1, if its diag dirs exist on disk, is marked invalid in
    the report and NOT used as a decision reference.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = (
    ROOT / "scripts" / "stage_b_generator"
    / "round29_summarize_next_step_ablation.py"
)

NEW_VARIANTS = (
    "r29_ns_a0_c41_g1_loss_s4",
    "r29_ns_a1_c41_s4_g1",
    "r29_ns_h1_i5_upper_bound",
    "r29_ns_a2_c41_i5_g1",
)
REFERENCES = (
    "r29_ft_r0_clean_a3_baseline",
    "r29_nb_b1_c41_only",
    "r29_nb_g1_phasefree_gait_fixed",
)
INVALID_OLD = "r29_nb_h1_r0_plus_oracle_full_hint"
ALL = (*REFERENCES, *NEW_VARIANTS)
SUBLABELS = ("train", "val")
PER_PART_KEYS = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")


def _make_sustained_stats(
    *, drift_max_mean: float, drift_max_p95: float = 30.0,
    track_below: float = 0.06,
    seq_offset: int = 0,
) -> dict:
    pp = {k: drift_max_mean * 1.5 for k in PER_PART_KEYS}
    # Synthetic per-segment rows that match across variants (same
    # (subset, seq_id, part_name, t0, t1) keys) so paired bootstrap can
    # find paired comparisons.
    rows = []
    for clip in range(8):
        seq_id = f"clip_{clip + seq_offset:03d}"
        for part in PER_PART_KEYS:
            rows.append({
                "subset": "chairs", "seq_id": seq_id,
                "part_idx": 0, "part_name": part,
                "t0": 5, "t1": 80,
                "drift_max_cm": drift_max_mean + (hash(part) % 5),
                "drift_mean_cm": 0.5, "drift_end_cm": 0.3,
            })
    return {
        "config": "fake.yaml", "ckpt": "fake.pt", "use_gt_as_pred": False,
        "overall": {
            "n_segments": 289,
            "drift_max_cm": {
                "mean": drift_max_mean, "median": drift_max_mean * 0.8,
                "p75": drift_max_mean * 1.3, "p95": drift_max_p95,
                "max": drift_max_p95 * 2.0,
            },
            "drift_end_cm": {"mean": 1.0},
            "drift_mean_cm": {"mean": drift_max_mean * 0.3},
            "tracking_fraction": {
                "mean": 1.0, "n": 141, "n_below_0.5": int(track_below * 141),
                "rate_below_0.5": track_below,
            },
            "n_drift_max_above_5cm": 200,
            "n_drift_max_above_10cm": 130,
        },
        "per_part": {
            k: {"drift_max_cm": {
                "mean": v, "median": v * 0.8, "p75": v * 1.3,
                "p95": v * 1.8, "max": v * 2.5,
            }} for k, v in pp.items()
        },
        "rows": rows,
    }


def _make_gait_stats(
    *, gt_swing: float = 0.26, gt_stance: float = 0.19,
    pred_swing: float = 0.52, pred_stance: float = 0.09,
    pred_trans: float = 0.82, pred_lr_corr: float = -0.18,
    seq_offset: int = 0,
) -> dict:
    per_seg = []
    for clip in range(6):
        seq_id = f"clip_{clip + seq_offset:03d}"
        per_seg.append({
            "subset": "chairs", "seq_id": seq_id,
            "t0": 0, "t1": 40,
            "gt": {
                "frac_both_swing": gt_swing, "frac_both_stance": gt_stance,
                "transitions_per_second": 0.92, "L_R_height_corr": -0.32,
            },
            "pred": {
                "frac_both_swing": pred_swing, "frac_both_stance": pred_stance,
                "transitions_per_second": pred_trans, "L_R_height_corr": pred_lr_corr,
            },
        })
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
            "L_R_height_corr": {"mean": pred_lr_corr},
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
        "per_segment": per_seg,
    }


def _make_body_action_stats(delta_err_mean: float = 8.0) -> dict:
    joint_stats = {
        "delta_error_cm_mean": delta_err_mean,
        "amp_gt_cm_mean": 25.0, "amp_pred_cm_mean": 20.0,
        "amp_ratio_mean": 0.80, "direction_cosine_mean": 0.50,
    }
    return {
        "aggregate": {
            "left_wrist": joint_stats, "right_wrist": joint_stats,
            "left_knee": joint_stats, "right_knee": joint_stats,
            "neck": joint_stats, "pelvis": joint_stats,
        },
        "per_clip": [],
    }


def _make_g1_soft_stats(
    *, constant_mid_rate: float = 0.10, soft_alt_std: float = 0.30,
    soft_trans: float = 0.10, gt_trans: float = 0.15,
) -> dict:
    return {
        "soft_aggregate": {
            "n_segments": 50,
            "pL_mean": 0.50, "pR_mean": 0.50,
            "pL_std": 0.20, "pR_std": 0.20,
            "soft_alt_mean": 0.0, "soft_alt_std": soft_alt_std,
            "soft_transition_density": soft_trans,
            "soft_both_stance": 0.15, "soft_both_swing": 0.30,
            "gt_transition_density": gt_trans,
            "constant_mid_rate": constant_mid_rate,
            "low_alt_amplitude_rate": 0.10,
            "low_transition_rate": 0.20,
        },
        "hard_pred_aggregate": {},
        "hard_gt_aggregate": {},
        "per_segment": [],
    }


def _make_repr_floor_stats(*, hand_mean: float = 1.5, hand_p95: float = 4.0) -> dict:
    per_joint = {}
    for j in ("pelvis", "left_wrist", "right_wrist", "left_ankle",
              "right_ankle", "left_foot", "right_foot", "neck"):
        per_joint[j] = {"mean_cm": hand_mean if "wrist" in j else 0.5,
                        "p95_cm": hand_p95 if "wrist" in j else 1.5,
                        "max_cm": hand_p95 * 1.5 if "wrist" in j else 3.0}
    return {
        "aggregate": {
            "n_clips": 48,
            "per_joint": per_joint,
            "per_part_contact_floor": {p: {"n_frames": 100, "mean_cm": 1.0,
                                            "p95_cm": 3.0, "max_cm": 5.0}
                                       for p in PER_PART_KEYS},
        },
        "interpretation": {
            "verdict": (
                "representation_floor_low" if hand_mean < 2.0 and hand_p95 < 5.0
                else "representation_floor_critical"
            ),
            "reason": "synthetic",
            "hand_mean_cm": hand_mean,
            "hand_p95_cm": hand_p95,
        },
    }


def _seed(
    results_root: Path,
    variants: tuple[str, ...],
    *,
    drift_by_variant: dict[str, float] | None = None,
    include_g1_soft: bool = True,
    include_repr_floor: bool = True,
    invalid_old_h1_diag: bool = False,
) -> None:
    drift_by_variant = drift_by_variant or {}
    for v in variants:
        for sub in SUBLABELS:
            for kind in ("sustained_contact", "gait", "body_action"):
                out_dir = results_root / f"round29_{v}_diag_{kind}_{sub}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if kind == "sustained_contact":
                    data = _make_sustained_stats(
                        drift_max_mean=drift_by_variant.get(v, 8.0),
                    )
                elif kind == "gait":
                    data = _make_gait_stats(
                        pred_lr_corr=-0.20 + (hash(v) % 100) * 0.001,
                    )
                else:
                    data = _make_body_action_stats(
                        delta_err_mean=drift_by_variant.get(v, 8.0) * 0.85,
                    )
                (out_dir / f"{kind}_stats.json").write_text(
                    json.dumps(data), encoding="utf-8",
                )
            # G1 soft-stance only for variants that have G1 losses.
            if include_g1_soft and (
                v == "r29_nb_g1_phasefree_gait_fixed"
                or v.startswith("r29_ns_a")
            ):
                g1_dir = results_root / f"round29_{v}_diag_g1_soft_stance_{sub}"
                g1_dir.mkdir(parents=True, exist_ok=True)
                (g1_dir / "g1_soft_stance_stats.json").write_text(
                    json.dumps(_make_g1_soft_stats()), encoding="utf-8",
                )
    if include_repr_floor:
        for sub in SUBLABELS:
            d = results_root / f"round29_repr_floor_{sub}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "repr_floor_stats.json").write_text(
                json.dumps(_make_repr_floor_stats()), encoding="utf-8",
            )
    if invalid_old_h1_diag:
        # Seed just one diag dir so the warning triggers.
        d = results_root / f"round29_{INVALID_OLD}_diag_sustained_contact_val"
        d.mkdir(parents=True, exist_ok=True)
        (d / "sustained_contact_stats.json").write_text(
            json.dumps(_make_sustained_stats(drift_max_mean=10.0)),
            encoding="utf-8",
        )


def _run_summarizer(results_root: Path, out_md: Path, *, allow_partial: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SUMMARIZER),
           "--results-root", str(results_root), "--out", str(out_md)]
    if allow_partial:
        cmd.append("--allow-partial")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))


# ----------------------------- tests -----------------------------

def test_full_seed_renders_all_sections(tmp_path: Path) -> None:
    out_md = tmp_path / "report.md"
    _seed(tmp_path, ALL,
          drift_by_variant={
              "r29_ft_r0_clean_a3_baseline": 8.19,
              "r29_nb_b1_c41_only": 7.38,
              "r29_nb_g1_phasefree_gait_fixed": 8.22,
              "r29_ns_a0_c41_g1_loss_s4": 7.5,
              "r29_ns_a1_c41_s4_g1": 8.0,
              "r29_ns_h1_i5_upper_bound": 7.0,
              "r29_ns_a2_c41_i5_g1": 6.9,
          })
    res = _run_summarizer(tmp_path, out_md)
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    for v in ALL:
        assert f"`{v}`" in report, f"missing variant {v}"
    assert "Paired bootstrap CI" in report
    assert "G1 soft-stance diagnostic" in report
    assert "Motion representation round-trip floor" in report
    assert "Decision verdict" in report


def test_paired_bootstrap_appears_with_matched_rows(tmp_path: Path) -> None:
    out_md = tmp_path / "report.md"
    _seed(tmp_path, ALL, drift_by_variant={v: 8.0 for v in ALL})
    res = _run_summarizer(tmp_path, out_md)
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    # Paired comparisons must appear with n_paired > 0 for sustained.
    assert "A0 vs B1" in report
    assert "A0 vs G1" in report
    assert "A1 vs A0" in report
    assert "H1 vs R0" in report
    # The synthetic data has 8 clips × 5 parts = 40 matchable rows.
    # At least one paired count > 0 must appear.
    assert " 40 " in report or " 39 " in report  # some matched count


def test_missing_reference_fails_by_default(tmp_path: Path) -> None:
    out_md = tmp_path / "report.md"
    # Seed only the new variants; no references.
    _seed(tmp_path, NEW_VARIANTS, drift_by_variant={v: 8.0 for v in NEW_VARIANTS})
    res = _run_summarizer(tmp_path, out_md, allow_partial=False)
    assert res.returncode != 0, "summarizer must fail when references missing"
    err = (res.stderr or "") + (res.stdout or "")
    assert "missing" in err.lower() or "fatal" in err.lower()


def test_missing_new_variant_stats_fails_by_default(tmp_path: Path) -> None:
    """If references are present but a new variant's stats are missing, fail."""
    out_md = tmp_path / "report.md"
    _seed(tmp_path, REFERENCES,
          drift_by_variant={v: 8.0 for v in REFERENCES})
    # Only seed A0 + H1 (skip A1, A2) → A1/A2 stats missing.
    _seed(tmp_path,
          ("r29_ns_a0_c41_g1_loss_s4", "r29_ns_h1_i5_upper_bound"),
          drift_by_variant={"r29_ns_a0_c41_g1_loss_s4": 7.5,
                            "r29_ns_h1_i5_upper_bound": 7.0})
    res = _run_summarizer(tmp_path, out_md, allow_partial=False)
    assert res.returncode != 0


def test_allow_partial_renders_incomplete_report(tmp_path: Path) -> None:
    out_md = tmp_path / "report.md"
    _seed(tmp_path, NEW_VARIANTS, drift_by_variant={v: 8.0 for v in NEW_VARIANTS})
    res = _run_summarizer(tmp_path, out_md, allow_partial=True)
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    assert "Partial report" in report
    # All ALL variants still appear (even references — with placeholders).
    for v in ALL:
        assert f"`{v}`" in report


def test_invalid_old_h1_marked_when_diag_present(tmp_path: Path) -> None:
    out_md = tmp_path / "report.md"
    _seed(tmp_path, ALL,
          drift_by_variant={v: 8.0 for v in ALL},
          invalid_old_h1_diag=True)
    res = _run_summarizer(tmp_path, out_md)
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    assert "Invalid historical reference" in report
    assert INVALID_OLD in report
    # Critically: the invalid old H1 must NOT appear in the paired
    # bootstrap as a valid reference.
    # Paired comparisons table only contains the live H1 (r29_ns_h1_i5_upper_bound).
    paired_section = report.split("Paired bootstrap CI")[1].split(
        "G1 soft-stance diagnostic"
    )[0]
    assert "r29_nb_h1_r0_plus_oracle_full_hint" not in paired_section


def test_decision_text_references_correct_comparisons(tmp_path: Path) -> None:
    out_md = tmp_path / "report.md"
    _seed(tmp_path, ALL,
          drift_by_variant={
              "r29_ft_r0_clean_a3_baseline": 8.19,
              "r29_nb_b1_c41_only": 7.38,
              "r29_nb_g1_phasefree_gait_fixed": 8.22,
              "r29_ns_a0_c41_g1_loss_s4": 7.5,
              "r29_ns_a1_c41_s4_g1": 8.0,
              "r29_ns_h1_i5_upper_bound": 7.0,
              "r29_ns_a2_c41_i5_g1": 6.9,
          })
    res = _run_summarizer(tmp_path, out_md)
    assert res.returncode == 0, res.stderr
    report = out_md.read_text(encoding="utf-8")
    decision = report.split("## Decision verdict")[1]
    # Section headers should exist.
    assert "A0 (`r29_ns_a0_c41_g1_loss_s4`) mainline verdict" in decision
    assert "A1 (`r29_ns_a1_c41_s4_g1`)" in decision
    assert "H1 (`r29_ns_h1_i5_upper_bound`)" in decision
    # A0 section compares to B1 and G1.
    a0_section = decision.split("mainline verdict")[1].split(
        "A1 (`r29_ns_a1_c41_s4_g1`)"
    )[0]
    assert "B1" in a0_section
    assert "G1" in a0_section
    # A1 section compares to A0.
    a1_section = decision.split("A1 (`r29_ns_a1_c41_s4_g1`)")[1].split(
        "H1 (`r29_ns_h1_i5_upper_bound`)"
    )[0]
    assert "A0" in a1_section
    # H1 section compares to R0.
    h1_section = decision.split("H1 (`r29_ns_h1_i5_upper_bound`)")[1]
    assert "R0" in h1_section
