"""Summarizer tests with fake diagnostic JSONs (Codex post-review §P1).

These tests fabricate minimal but valid sustained_contact_stats.json /
gait_stats.json / body_action_stats.json files in ``tmp_path``, run the
summarizer with ``--diagnostics-root tmp_path``, and verify:

  * The summarizer loads the REAL diagnostic filenames (not the
    placeholder {summary,diag,metrics}.json the original draft tried).
  * Nested metrics are flattened correctly.
  * `amp_ratio_error` is computed as |amp_ratio - 1|.
  * A candidate with no diagnostics is NOT ranked as best.
  * Missing baseline -> "not_rankable" + non-zero exit unless
    ``--allow-missing-results`` is set.

Per Codex P2, these tests must NOT mutate the real repo's
``analyses/round29_*_diag_*`` directories or canonical manifest. All
fake diag files live under ``tmp_path``; the manifest is written to
``tmp_path`` and pointed at via ``--manifest <tmp_path/...>``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "scripts" / "stage_b_generator" / "round29_summarize_stage2_cond_ablation.py"

from piano.data.interaction_hint import BODY_ACTION_KEY_JOINT_NAMES


# ---------------------------------------------------------------------------
# Fixture builders (writing into tmp_path only)
# ---------------------------------------------------------------------------

def _write_sustained(out_dir: Path, *,
                     drift_mean: float, drift_p95: float,
                     n_segments: int, n_gt5: int, n_gt10: int,
                     tracking_mean: float, tracking_lt05: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": "fake", "ckpt": "fake",
        "overall": {
            "n_segments": n_segments,
            "drift_max_cm": {"mean": drift_mean, "median": drift_mean,
                             "p75": drift_p95 * 0.9, "p95": drift_p95, "max": drift_p95 * 1.1},
            "drift_end_cm": {"mean": 0.0, "median": 0.0},
            "tracking_fraction": {
                "n_with_obj_motion": n_segments,
                "mean": tracking_mean, "median": tracking_mean,
                "n_below_0.5": int(tracking_lt05 * n_segments),
                "rate_below_0.5": tracking_lt05,
            },
            "rel_var_ratio": {"n": n_segments, "mean": 1.0, "median": 1.0},
            "n_drift_max_above_5cm": n_gt5,
            "n_drift_max_above_10cm": n_gt10,
        },
        "per_part": {}, "per_subset": {}, "rows": [],
    }
    (out_dir / "sustained_contact_stats.json").write_text(json.dumps(payload), "utf-8")


def _write_gait(out_dir: Path, *,
                both_swing: float, both_stance: float,
                trans_per_sec: float, lr_corr: float,
                step_period_rate: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": "fake", "ckpt": "fake",
        "n_walking_segments": 50,
        "pred_aggregate": {
            "n_segments": 50,
            "frac_both_swing": {"mean": both_swing},
            "frac_both_stance": {"mean": both_stance},
            "transitions_per_second": {"mean": trans_per_sec},
            "L_R_height_corr": {"mean": lr_corr, "n": 50},
            "step_period_frames": {"rate_with_period": step_period_rate,
                                    "mean": 20.0 if step_period_rate > 0 else None},
        },
        "gt_aggregate": {},
        "per_segment": [],
    }
    (out_dir / "gait_stats.json").write_text(json.dumps(payload), "utf-8")


def _write_body_action(out_dir: Path, *,
                       delta_err_cm: float, amp_ratio: float,
                       direction_cos: float, active_frac: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = {}
    for joint in BODY_ACTION_KEY_JOINT_NAMES:
        agg[joint] = {
            "delta_error_cm_mean": delta_err_cm,
            "delta_error_p95_cm_mean": delta_err_cm * 1.5,
            "amp_pred_cm_mean": 5.0 * amp_ratio,
            "amp_gt_cm_mean": 5.0,
            "amp_ratio_mean": amp_ratio,
            "direction_cosine_mean": direction_cos,
            "active_frame_frac_mean": active_frac,
            "energy_mask_pred_mean": 1.0,
            "energy_mask_gt_mean": 1.0,
        }
    payload = {
        "config": "fake", "ckpt": "fake",
        "aggregate": agg, "per_clip": [],
    }
    (out_dir / "body_action_stats.json").write_text(json.dumps(payload), "utf-8")


def _variant(vid: str, **over) -> dict:
    base = {
        "variant_id": vid,
        "group": over.get("group", "A_injection"),
        "purpose": "test",
        "coarse_variant": "C41-current",
        "interaction_variant": "I3-contact-offset-masked",
        "support_variant": "S4-S1-phase-footstep",
        "body_variant": "B4-lowpass-residual-mask",
        "injection_mode": "input_add",
        "gate_bias_init": -1.0,
        "per_family_modes": None,
        "expected_dense_dims": {"coarse_extra": 18, "interaction": 8, "support": 13, "body_refine": 20},
        "subset_kind": "balanced",
        "subset_file": "analyses/subset.json",
        "num_epochs": 300,
        "seed": 42,
        "val_on_train_subset": True,
        "config_path": f"configs/training/anchordiff_{vid}.yaml",
        "output_dir": f"runs/training/stageB_anchordiff_{vid}",
        "diagnostics": ["sustained_contact", "gait", "body_action"],
    }
    base.update(over)
    return base


def _write_manifest(path: Path, variants: list[dict]) -> None:
    payload = {"variants": variants, "best_resolved": {}, "defaults": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), "utf-8")


def _run_summarizer(
    tmp_path: Path, manifest_path: Path,
    *, allow_missing: bool = True,
    diagnostics_root: Path | None = None,
) -> tuple[int, dict, str]:
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    cmd = [
        sys.executable, str(SUMMARIZER),
        "--manifest", str(manifest_path),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
        "--diagnostics-root", str(diagnostics_root or tmp_path),
    ]
    if allow_missing:
        cmd.append("--allow-missing-results")
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    payload = json.loads(out_json.read_text("utf-8")) if out_json.exists() else {}
    return res.returncode, payload, res.stderr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_summarizer_loads_real_diagnostic_filenames(tmp_path: Path) -> None:
    """With real sustained/gait/body_action stats files under tmp_path,
    the summarizer must read them, score, and pick the better variant."""
    for vid, drift, swing, derr, amp in (
        ("r29_f0_baseline", 10.0, 0.5, 10.0, 0.5),
        ("r29_a0_input_add", 7.0, 0.2, 6.0, 0.95),
    ):
        _write_sustained(
            tmp_path / f"round29_{vid}_diag_sustained_contact",
            drift_mean=drift, drift_p95=drift * 2, n_segments=200,
            n_gt5=int(drift * 8), n_gt10=int(drift * 3),
            tracking_mean=0.6 if vid.endswith("baseline") else 0.85,
            tracking_lt05=0.3 if vid.endswith("baseline") else 0.1,
        )
        _write_gait(
            tmp_path / f"round29_{vid}_diag_gait",
            both_swing=swing, both_stance=0.3 if vid.endswith("baseline") else 0.2,
            trans_per_sec=0.5 if vid.endswith("baseline") else 0.8,
            lr_corr=-0.1 if vid.endswith("baseline") else -0.3,
            step_period_rate=0.5 if vid.endswith("baseline") else 0.9,
        )
        _write_body_action(
            tmp_path / f"round29_{vid}_diag_body_action",
            delta_err_cm=derr, amp_ratio=amp,
            direction_cos=0.4 if vid.endswith("baseline") else 0.85,
            active_frac=0.6 if vid.endswith("baseline") else 0.7,
        )

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        _variant("r29_f0_baseline", group="F_final",
                 coarse_variant="C23", interaction_variant="I0",
                 support_variant="S0", body_variant="B0",
                 expected_dense_dims={"coarse_extra": 0, "interaction": 0,
                                      "support": 0, "body_refine": 0}),
        _variant("r29_a0_input_add", group="A_injection"),
    ])

    rc, payload, stderr = _run_summarizer(tmp_path, manifest_path)
    assert rc == 0, f"summarizer failed: {stderr}"
    by_vid = {r["variant_id"]: r for r in payload["rows"]}
    assert by_vid["r29_f0_baseline"]["status"]["rankable"]
    assert by_vid["r29_a0_input_add"]["status"]["rankable"]
    assert by_vid["r29_a0_input_add"]["scores"]["composite"] > 0
    assert payload["recommendation"]["best_variant"] == "r29_a0_input_add"


def test_nested_metrics_flatten_correctly(tmp_path: Path) -> None:
    """Drill into one row's flat_metrics and verify the nested-key mapping."""
    _write_sustained(
        tmp_path / "round29_r29_f0_baseline_diag_sustained_contact",
        drift_mean=12.5, drift_p95=25.0, n_segments=200,
        n_gt5=80, n_gt10=30, tracking_mean=0.6, tracking_lt05=0.3,
    )
    _write_gait(
        tmp_path / "round29_r29_f0_baseline_diag_gait",
        both_swing=0.55, both_stance=0.35,
        trans_per_sec=0.55, lr_corr=-0.15, step_period_rate=0.55,
    )
    _write_body_action(
        tmp_path / "round29_r29_f0_baseline_diag_body_action",
        delta_err_cm=11.0, amp_ratio=0.6,
        direction_cos=0.4, active_frac=0.7,
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_variant("r29_f0_baseline", group="F_final")])
    rc, payload, stderr = _run_summarizer(tmp_path, manifest_path)
    assert rc == 0, stderr
    row = payload["rows"][0]
    fm = row["flat_metrics"]
    assert fm["drift_max_mean_cm"] == pytest.approx(12.5)
    assert fm["drift_max_p95_cm"] == pytest.approx(25.0)
    assert fm["pct_drift_gt_5cm"] == pytest.approx(80 / 200)
    assert fm["pct_drift_gt_10cm"] == pytest.approx(30 / 200)
    assert fm["tracking_fraction_mean"] == pytest.approx(0.6)
    assert fm["tracking_fraction_lt_05"] == pytest.approx(0.3)
    assert fm["frac_both_swing"] == pytest.approx(0.55)
    assert fm["frac_both_stance"] == pytest.approx(0.35)
    assert fm["transitions_per_second"] == pytest.approx(0.55)
    assert fm["step_period_rate"] == pytest.approx(0.55)
    assert fm["key_joint_delta_error_cm"] == pytest.approx(11.0)
    # |0.6 - 1| = 0.4
    assert fm["amp_ratio_error"] == pytest.approx(0.4)


def test_amp_ratio_is_scored_by_closeness_to_1(tmp_path: Path) -> None:
    """If two variants beat baseline equally on all OTHER axes but one
    over-articulates (amp_ratio=1.5) and the other matches (1.0), the
    closer-to-1 variant should rank higher on body_action."""
    # Baseline: under-articulating (0.5).
    _write_sustained(tmp_path / "round29_r29_f0_baseline_diag_sustained_contact",
                     drift_mean=5.0, drift_p95=10.0, n_segments=100,
                     n_gt5=40, n_gt10=10, tracking_mean=0.8, tracking_lt05=0.1)
    _write_gait(tmp_path / "round29_r29_f0_baseline_diag_gait",
                both_swing=0.2, both_stance=0.2, trans_per_sec=0.8,
                lr_corr=-0.3, step_period_rate=0.9)
    _write_body_action(tmp_path / "round29_r29_f0_baseline_diag_body_action",
                       delta_err_cm=8.0, amp_ratio=0.5,
                       direction_cos=0.7, active_frac=0.7)
    for vid, ar in (("r29_x_over", 1.5), ("r29_x_perfect", 1.0)):
        _write_sustained(tmp_path / f"round29_{vid}_diag_sustained_contact",
                         drift_mean=5.0, drift_p95=10.0, n_segments=100,
                         n_gt5=40, n_gt10=10, tracking_mean=0.8, tracking_lt05=0.1)
        _write_gait(tmp_path / f"round29_{vid}_diag_gait",
                    both_swing=0.2, both_stance=0.2, trans_per_sec=0.8,
                    lr_corr=-0.3, step_period_rate=0.9)
        _write_body_action(tmp_path / f"round29_{vid}_diag_body_action",
                           delta_err_cm=8.0, amp_ratio=ar,
                           direction_cos=0.7, active_frac=0.7)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        _variant("r29_f0_baseline", group="F_final"),
        _variant("r29_x_over"),
        _variant("r29_x_perfect"),
    ])
    rc, payload, stderr = _run_summarizer(tmp_path, manifest_path)
    assert rc == 0, stderr
    by_vid = {r["variant_id"]: r for r in payload["rows"]}
    assert by_vid["r29_x_perfect"]["flat_metrics"]["amp_ratio_error"] == pytest.approx(0.0)
    assert by_vid["r29_x_over"]["flat_metrics"]["amp_ratio_error"] == pytest.approx(0.5)
    # Perfect should beat over on body_action axis (higher = better here).
    perf_axis = by_vid["r29_x_perfect"]["scores"]["per_axis"]["body_action"]
    over_axis = by_vid["r29_x_over"]["scores"]["per_axis"]["body_action"]
    assert perf_axis > over_axis


def test_missing_diagnostics_variant_not_best(tmp_path: Path) -> None:
    """A variant with NO diagnostic files cannot be the recommendation."""
    for vid, drift in (("r29_f0_baseline", 10.0), ("r29_a0_input_add", 6.0)):
        _write_sustained(tmp_path / f"round29_{vid}_diag_sustained_contact",
                         drift_mean=drift, drift_p95=drift * 2,
                         n_segments=100, n_gt5=int(drift * 5),
                         n_gt10=int(drift), tracking_mean=0.8,
                         tracking_lt05=0.1)
        _write_gait(tmp_path / f"round29_{vid}_diag_gait",
                    both_swing=0.3, both_stance=0.2, trans_per_sec=0.8,
                    lr_corr=-0.3, step_period_rate=0.9)
        _write_body_action(tmp_path / f"round29_{vid}_diag_body_action",
                           delta_err_cm=drift, amp_ratio=1.0,
                           direction_cos=0.7, active_frac=0.7)
    # r29_missing has NO diag dirs.
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        _variant("r29_f0_baseline", group="F_final"),
        _variant("r29_a0_input_add"),
        _variant("r29_missing"),
    ])
    rc, payload, stderr = _run_summarizer(tmp_path, manifest_path)
    assert rc == 0, stderr
    by_vid = {r["variant_id"]: r for r in payload["rows"]}
    assert not by_vid["r29_missing"]["status"]["rankable"]
    rec = payload["recommendation"]
    assert rec["best_variant"] != "r29_missing"
    assert rec["minimal_near_best_variant"] != "r29_missing"


def test_missing_baseline_returns_nonzero_without_allow_flag(tmp_path: Path) -> None:
    """No baseline -> not_rankable; without --allow-missing-results, exit nonzero."""
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_variant("r29_a0_input_add")])
    rc, payload, stderr = _run_summarizer(tmp_path, manifest_path, allow_missing=False)
    assert rc != 0, "expected nonzero when baseline missing and no --allow-missing-results"
    assert payload["recommendation"] == {}
    assert payload["not_computable_reason"]
