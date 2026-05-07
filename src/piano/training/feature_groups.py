"""Single source of truth for motion_263 feature groups.

Used by:
- compute_initial_feature_weights.py (offline std stats)
- compute_geometry_prior.py (offline task-space sensitivity)
- anchordiff_feature_diag.py (per-feature RMSE diagnostic)
- train_anchordiff.py FeatureWeightState (training-time MSE weights)
- v2 / v2.1 configs (group names referenced in motion_feature_weights)

Layout (HumanML3D motion_263, 22 joints):

    [0]      : root rotation velocity (Y-axis arcsin, rad/frame)
    [1:3]    : root linear velocity (XZ in body frame, m/frame)
    [3]      : root height Y (absolute, m)
    [4:67]   : 21 body-relative joint positions (3 each, J0=root excluded)
    [67:193] : 21 joint rotations in 6D rep (Zhou et al. 2019)
    [193:259]: 22 joint velocities (3 each, includes root)
    [259:263]: 4 foot contact labels (left/right ankle/toe)

Per the M1.5 spec, do not duplicate slice constants across scripts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    name: str
    lo: int
    hi: int

    @property
    def n_dims(self) -> int:
        return self.hi - self.lo


FEATURE_GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup("root_rot_vel", 0, 1),
    FeatureGroup("root_lin_vel", 1, 3),
    FeatureGroup("root_height_y", 3, 4),
    FeatureGroup("joint_pos_local", 4, 67),
    FeatureGroup("joint_rot_6d", 67, 193),
    FeatureGroup("joint_velocity", 193, 259),
    FeatureGroup("foot_contact", 259, 263),
)

# Names that should NEVER collapse to the floor — root motion needs a
# nonzero weight even after early convergence so the model doesn't
# silently lose root signal mid-training.
ROOT_MOTION_GROUPS: frozenset[str] = frozenset({
    "root_rot_vel", "root_lin_vel", "root_height_y",
})

MOTION_DIM: int = 263


def group_by_name(name: str) -> FeatureGroup:
    for g in FEATURE_GROUPS:
        if g.name == name:
            return g
    raise KeyError(f"Unknown feature group: {name!r}")


def total_dims() -> int:
    return sum(g.n_dims for g in FEATURE_GROUPS)


# Sanity check: dims sum to motion_dim.
assert total_dims() == MOTION_DIM, (
    f"feature_groups sum to {total_dims()}, expected {MOTION_DIM}"
)
