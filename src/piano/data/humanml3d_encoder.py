"""HumanML3D 263-dim encoder — wraps MoMask's official ``process_file``.

MoMask's preprocessing pipeline (``utils/motion_process.process_file``) is
the canonical encoder used to prepare data for its pretrained VQ-VAE.
We call it directly instead of reimplementing, so that our features are
byte-compatible with what MoMask's VQ-VAE was trained on.

Two caveats about ``process_file``:

1. It uses module-level globals (``tgt_offsets``, ``n_raw_offsets``,
   ``kinematic_chain``, ``face_joint_indx``, ``fid_l``, ``fid_r``) rather
   than function parameters. This adapter sets them once at init time on
   the ``utils.motion_process`` module.

2. It applies ``uniform_skeleton`` which RESCALES every sequence to a
   single reference skeleton. This is essential for VQ-VAE compatibility
   but warps absolute world positions. Callers needing geometrically
   accurate joint positions (e.g. for HOI contact detection) must keep
   a separate copy of the raw joints, not derive them from the 263-dim
   output.

3. The returned features have length T-1 (one frame is lost to the
   velocity computation). Any parallel data (joints, object positions)
   that needs to be aligned with the features should be truncated to T-1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Side-effect import: adds MoMask repo to sys.path and monkeypatches numpy
# aliases needed by its legacy code.
import piano.models.backbones.momask_adapter  # noqa: F401


_FACE_JOINT_INDX: list[int] = [2, 1, 17, 16]   # r_hip, l_hip, sdr_r, sdr_l
_FOOT_L: list[int] = [7, 10]                   # left ankle, left foot
_FOOT_R: list[int] = [8, 11]                   # right ankle, right foot
_JOINTS_NUM: int = 22


@dataclass(slots=True)
class HumanML3DEncoder:
    """Adapter that prepares MoMask globals and calls ``process_file``.

    Parameters
    ----------
    reference_joints : (22, 3) array — reference T-pose (or first frame of a
        representative sequence) used to derive ``tgt_offsets``. MoMask's
        upstream example uses HumanML3D's 000021.npy first frame; for HOI
        work we pass in the first frame of a representative OMOMO sequence.
    feet_thre : foot contact velocity threshold (MoMask default: 0.002)
    """

    reference_joints: np.ndarray
    feet_thre: float = 0.002

    def __post_init__(self) -> None:
        # Lazy import: these only exist after momask_adapter sets sys.path.
        import utils.motion_process as mp
        from common.skeleton import Skeleton
        from utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets

        n_raw_offsets_t = torch.from_numpy(t2m_raw_offsets).float()
        ref = torch.from_numpy(self.reference_joints.astype(np.float32))

        skel = Skeleton(n_raw_offsets_t, t2m_kinematic_chain, "cpu")
        tgt_offsets = skel.get_offsets_joints(ref)

        # Set process_file's module-level globals.
        mp.tgt_offsets = tgt_offsets
        mp.n_raw_offsets = n_raw_offsets_t
        mp.kinematic_chain = t2m_kinematic_chain
        mp.face_joint_indx = _FACE_JOINT_INDX
        mp.fid_l = _FOOT_L
        mp.fid_r = _FOOT_R
        mp.l_idx1 = 5   # lower-leg indices used by uniform_skeleton
        mp.l_idx2 = 8

    def encode(
        self, joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode world-frame joint positions to HumanML3D 263-dim features.

        Parameters
        ----------
        joints : (T, 22, 3) — world-frame, Y-up, meters, 20 fps

        Returns
        -------
        features : (T-1, 263) — HumanML3D features; compatible with MoMask's
            pretrained VQ-VAE and with ``utils.motion_process.recover_from_ric``
        aligned_joints : (T-1, 22, 3) — joints after HumanML3D's canonical
            alignment (ground-put + xz-origin + heading-aligned); useful for
            visualizing what the VQ-VAE actually saw
        """
        from utils.motion_process import process_file

        data, global_positions, _local_positions, _l_velocity = process_file(
            joints.astype(np.float32).copy(), self.feet_thre,
        )
        # process_file returns:
        #   data              : (T-1, 263) encoded features
        #   global_positions  : (T-1, 22, 3) aligned global joint positions
        return data.astype(np.float32), global_positions.astype(np.float32)
