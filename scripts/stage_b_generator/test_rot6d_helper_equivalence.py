"""Preflight (Round 12, Step 1): rot6d -> matrix convention test.

The Codex review (analyses/2026-05-21_codex_stage1_coarse_prior_design_review.md
§3, §10.1) flagged that
``scripts/stage_b_generator/extract_coarse_motion_representation.py::_rot6d_to_R``
stacks the orthonormal basis along ``dim=-1`` (columns) while the project-local
``piano.training.smpl_kinematics.rotation_6d_to_matrix`` stacks along
``dim=-2`` (rows). The 6D rep stored by ``matrix_to_rotation_6d`` is
``R[..., :2, :].reshape(..., 6)`` (Zhou-style ROWS), so the project's
inverse is the correct one.

This script demonstrates:

1. The custom helper returns the TRANSPOSE of the project helper's output
   for the same 6D input.
2. For pure yaw rotations around Y, the custom helper's column-2
   extraction yields the wrong "world forward": it picks up
   ``R^T[:, 2]`` = ``R[2, :]`` instead of ``R[:, 2]``. For pure yaw,
   that flips the sign of the X component, so the custom-derived yaw
   equals ``-yaw_true``.

Run with:

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/test_rot6d_helper_equivalence.py

Exit code 0 = expected divergence pattern confirmed. Non-zero = surprise.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

from piano.training.smpl_kinematics import (
    rotation_6d_to_matrix as project_rot6d_to_matrix,
    matrix_to_rotation_6d as project_matrix_to_rotation_6d,
)


def custom_rot6d_to_R(rot6d: torch.Tensor) -> torch.Tensor:
    """Bit-exact copy of the bugged helper from
    ``scripts/stage_b_generator/extract_coarse_motion_representation.py``
    so we can compare against the project-local upstream."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def yaw_axis_y_matrix(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=torch.float32)


def main() -> int:
    print("[round-12] rot6d helper convention test")
    rng = np.random.default_rng(42)

    # === Test 1: random 6D in both helpers — are they transposes? ===
    d6_rand = torch.from_numpy(rng.standard_normal((32, 6)).astype(np.float32))
    R_proj = project_rot6d_to_matrix(d6_rand)         # (32, 3, 3)
    R_cust = custom_rot6d_to_R(d6_rand)               # (32, 3, 3)
    # Expectation: R_cust = R_proj.transpose(-1, -2)
    diff_t = (R_cust - R_proj.transpose(-1, -2)).abs().max().item()
    print(f"  max |R_custom - R_project^T| over 32 random samples: {diff_t:.3e}")
    if diff_t > 1e-5:
        print("  [FAIL] Custom != project^T. Unexpected.")
        return 1

    # === Test 2: round-trip — does project helper invert matrix_to_rotation_6d? ===
    R_in = yaw_axis_y_matrix(0.5)
    d6_proj = project_matrix_to_rotation_6d(R_in)
    R_back = project_rot6d_to_matrix(d6_proj)
    rt_err = (R_in - R_back).abs().max().item()
    print(f"  project round-trip max |R - R'|: {rt_err:.3e} (expect ~0)")
    if rt_err > 1e-5:
        print("  [FAIL] Project helper is not a clean round-trip.")
        return 1

    # === Test 3: same round-trip with custom — should FAIL because custom returns R^T ===
    R_back_cust = custom_rot6d_to_R(d6_proj)
    rt_cust = (R_in - R_back_cust).abs().max().item()
    print(f"  custom  round-trip max |R - R'|: {rt_cust:.3e} (expect ~|2*sin(theta)|≈0.96)")
    expected_diff = abs(2.0 * math.sin(0.5))
    if abs(rt_cust - expected_diff) > 0.02:
        print(f"  [WARN] custom round-trip error {rt_cust:.4f} unexpected; want ≈{expected_diff:.4f}")
        # not a hard fail — could differ in edge cases — but flag

    # === Test 4: yaw extraction equivalence with the canonical pelvis_z_forward rule ===
    # If forward = R[:, :, 2] (column 2 — body Z in world coords),
    # then yaw = atan2(forward_x, forward_z) = atan2(sin(theta), cos(theta)) = theta.
    thetas = torch.linspace(-math.pi + 0.01, math.pi - 0.01, 9, dtype=torch.float32)
    proj_yaws = []
    cust_yaws = []
    for t in thetas:
        R_t = yaw_axis_y_matrix(float(t))
        d6_t = project_matrix_to_rotation_6d(R_t)

        R_proj_t = project_rot6d_to_matrix(d6_t)
        R_cust_t = custom_rot6d_to_R(d6_t)

        # Project (correct) yaw: forward = column 2
        fp = R_proj_t[..., :, 2]
        proj_yaw = math.atan2(float(fp[0]), float(fp[2]))
        proj_yaws.append(proj_yaw)

        # Custom (buggy) yaw: same code as in extract_coarse_motion_representation.py
        fc = R_cust_t[..., :, 2]
        cust_yaw = math.atan2(float(fc[0]), float(fc[2]))
        cust_yaws.append(cust_yaw)

    proj_yaws_t = torch.tensor(proj_yaws)
    cust_yaws_t = torch.tensor(cust_yaws)
    diff_proj = (proj_yaws_t - thetas).abs().max().item()
    diff_cust = (cust_yaws_t - thetas).abs().max().item()
    diff_neg = (cust_yaws_t - (-thetas)).abs().max().item()
    print(f"  proj yaw vs theta: max err = {diff_proj:.3e}  (expect ~0)")
    print(f"  cust yaw vs theta: max err = {diff_cust:.3e}  (expect ~2|theta| at large theta)")
    print(f"  cust yaw vs -theta: max err = {diff_neg:.3e}  (expect ~0  if sign flip)")

    if diff_proj > 1e-5:
        print("  [FAIL] Project yaw extraction does not match theta — convention assumption wrong.")
        return 1
    if diff_neg > 1e-5:
        print("  [FAIL] Custom yaw extraction is NOT a pure sign flip of theta.")
        return 1

    print("[round-12] Confirmed: custom helper produces R^T, and "
          "the yaw it extracts is -theta where the correct yaw is +theta.")
    print("[round-12] All probes consistent with Codex review §3.1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
