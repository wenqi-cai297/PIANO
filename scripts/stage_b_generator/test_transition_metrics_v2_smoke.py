"""Smoke test for transition_metrics(metric_version='v2').

Confirms:
1. v1 default behavior unchanged (signature backward compatible).
2. v2 produces validity-flagged event rows.
3. v2 does not divide by tiny denominators for M2/M3.
4. Surface fallback to COM when object_pc / object_rotations absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Add the scripts dir so we can import diagnostic_common as a sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnostic_common import transition_metrics  # type: ignore


def _make_toy(B: int = 1, T: int = 32) -> dict[str, torch.Tensor]:
    rng = np.random.RandomState(0)
    # Hand starts far from object, approaches over 10 frames, contacts, releases.
    joints = np.zeros((B, T, 22, 3), dtype=np.float32)
    obj = np.zeros((B, T, 3), dtype=np.float32)
    # Hand sweeps in along x from 0.5 m to 0.05 m around onset @ t=10
    for t in range(T):
        if t < 10:
            x = 0.5 - 0.045 * t
        elif t < 22:
            x = 0.05
        else:
            x = 0.05 + 0.04 * (t - 22)
        joints[0, t, 20, 0] = x
    contact = np.zeros((B, T, 5), dtype=np.float32)
    contact[0, 10:22, 0] = 1.0
    obj[0, :] = 0.0
    seq_mask = np.ones((B, T), dtype=bool)
    # Tiny object_pc + identity rotations for surface variant
    pc = np.array([[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]]], dtype=np.float32)
    rot = np.zeros((B, T, 3), dtype=np.float32)
    return {
        "joints": torch.from_numpy(joints),
        "object_positions": torch.from_numpy(obj),
        "contact_state": torch.from_numpy(contact),
        "seq_mask": torch.from_numpy(seq_mask),
        "object_pc": torch.from_numpy(pc),
        "object_rotations": torch.from_numpy(rot),
    }


def main() -> None:
    inputs = _make_toy()

    # 1. v1 default — signature must not error
    v1_out = transition_metrics(
        inputs["joints"], inputs["object_positions"], inputs["contact_state"],
        inputs["seq_mask"], gt_joints=inputs["joints"],
    )
    assert v1_out.get("metric_version") == "v1", f"v1 missing tag: {v1_out.get('metric_version')!r}"
    assert "onset_positive_closing_cm" in v1_out, "v1 onset_positive missing"
    assert "ratios_over_gt" in v1_out, "v1 ratios_over_gt missing (gt was provided)"
    print(f"  v1 onset_positive_closing_cm mean: {v1_out['onset_positive_closing_cm']['mean']:.3f}")
    print(f"  v1 ratios_over_gt onset: {v1_out['ratios_over_gt']['onset_positive_closing']:.3f}")

    # 2. v2 with surface available
    v2_surf = transition_metrics(
        inputs["joints"], inputs["object_positions"], inputs["contact_state"],
        inputs["seq_mask"], gt_joints=inputs["joints"],
        metric_version="v2",
        object_pc=inputs["object_pc"], object_rotations=inputs["object_rotations"],
    )
    assert v2_surf["metric_version"] == "v2"
    assert v2_surf["use_surface"] is True
    assert "events" in v2_surf and len(v2_surf["events"]) > 0
    for ev in v2_surf["events"]:
        assert "valid_v2_slope" in ev
        assert "valid_v2_signed" in ev
        assert "valid_v2_ratio_2cm" in ev
        assert "distance_source" in ev
    print(f"  v2 use_surface={v2_surf['use_surface']} n_events={v2_surf['n_events_total']} n_valid_slope={v2_surf['n_valid_slope']}")
    print(f"  v2 onset_slope_cm_per_frame mean: {v2_surf['onset_slope_cm_per_frame']['mean']:.3f}")
    print(f"  v2 release_slope_cm_per_frame mean: {v2_surf['release_slope_cm_per_frame']['mean']:.3f}")

    # 3. v2 COM fallback (no object_pc / rotations)
    v2_com = transition_metrics(
        inputs["joints"], inputs["object_positions"], inputs["contact_state"],
        inputs["seq_mask"], gt_joints=inputs["joints"],
        metric_version="v2",
    )
    assert v2_com["use_surface"] is False
    for ev in v2_com["events"]:
        assert ev["distance_source"] == "com_fallback"
    print(f"  v2 COM-fallback n_events={v2_com['n_events_total']}")

    # 4. v2 does not divide by tiny denominators for M2/M3 (no /0)
    # Construct a near-zero motion case
    inputs2 = _make_toy()
    inputs2["joints"][0, :, 20, 0] = 0.05  # hand stays still
    v2_static = transition_metrics(
        inputs2["joints"], inputs2["object_positions"], inputs2["contact_state"],
        inputs2["seq_mask"], gt_joints=inputs2["joints"],
        metric_version="v2",
        object_pc=inputs2["object_pc"], object_rotations=inputs2["object_rotations"],
    )
    # No inf / nan should appear
    for ev in v2_static["events"]:
        assert np.isfinite(ev["com"]["m2_slope_cm_per_frame"]), "M2 produced non-finite value"
        assert np.isfinite(ev["com"]["m3_signed_diff_cm"]), "M3 produced non-finite value"
    print(f"  v2 static case: {v2_static['n_denom_unstable_2cm']}/{v2_static['n_events_total']} denom-unstable; M2/M3 finite OK")

    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
