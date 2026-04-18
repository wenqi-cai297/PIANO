"""SMPL-X forward kinematics helper.

Thin wrapper around the ``smplx`` library to compute joint positions
from SMPL-X parameters (betas, root_orient, pose_body, trans).

Used during OMOMO / HOI data preprocessing, where raw datasets ship
SMPL-X parameters rather than joint coordinates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import smplx
import torch
from torch import Tensor


def load_smplx_model(
    model_root: str | Path,
    gender: str = "neutral",
    num_betas: int = 16,
    device: str = "cpu",
    batch_size: int = 1,
) -> smplx.SMPLX:
    """Load a SMPL-X model.

    Parameters
    ----------
    model_root : path to the directory containing SMPLX_*.npz files
        (e.g. checkpoints/smpl_x_v1.1/models/smplx/)
    gender : "neutral" | "male" | "female"
    num_betas : number of shape coefficients (16 for CHOIS/OMOMO data)
    device : "cpu" or "cuda"
    batch_size : batch size the model is configured for

    Returns
    -------
    smplx.SMPLX model in eval mode on the requested device
    """
    model_root = Path(model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"SMPL-X model root not found: {model_root}")

    model = smplx.create(
        str(model_root.parent),  # smplx.create expects parent of 'smplx/'
        model_type="smplx",
        gender=gender,
        num_betas=num_betas,
        use_pca=False,            # no hand PCA; OMOMO doesn't store hand pose anyway
        flat_hand_mean=True,
        batch_size=batch_size,
    )
    model.eval()
    model.to(device)
    return model


def run_smplx_fk(
    model: smplx.SMPLX,
    betas: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    device: str = "cpu",
    chunk_size: int = 512,
) -> np.ndarray:
    """Run SMPL-X forward kinematics to produce joint positions.

    Parameters
    ----------
    model : SMPL-X model from ``load_smplx_model``
    betas : (1, num_betas) or (T, num_betas) body shape coefficients
    root_orient : (T, 3) root axis-angle rotation
    pose_body : (T, 63) body pose (21 joints × 3) axis-angle
    trans : (T, 3) root translation in world frame
    device : computation device
    chunk_size : process this many frames at a time (memory trade-off)

    Returns
    -------
    joints : (T, 55, 3) — SMPL-X joint positions (first 22 are body joints)
    """
    T = len(trans)

    # Expand betas to (T, num_betas) if needed
    if betas.ndim == 2 and betas.shape[0] == 1:
        betas_t = np.broadcast_to(betas, (T, betas.shape[1]))
    else:
        betas_t = betas

    all_joints: list[np.ndarray] = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = end - start

        betas_chunk = torch.from_numpy(np.ascontiguousarray(betas_t[start:end])).float().to(device)
        root_orient_chunk = torch.from_numpy(np.ascontiguousarray(root_orient[start:end])).float().to(device)
        pose_body_chunk = torch.from_numpy(np.ascontiguousarray(pose_body[start:end])).float().to(device)
        trans_chunk = torch.from_numpy(np.ascontiguousarray(trans[start:end])).float().to(device)

        # The model was created with a fixed batch_size, but smplx supports
        # passing any batch size at call time via the forward kwargs.
        with torch.no_grad():
            output = model(
                betas=betas_chunk,
                global_orient=root_orient_chunk,
                body_pose=pose_body_chunk,
                transl=trans_chunk,
                return_verts=False,
            )

        joints_chunk = output.joints.detach().cpu().numpy()  # (chunk, 55, 3)
        all_joints.append(joints_chunk)

    return np.concatenate(all_joints, axis=0)  # (T, 55, 3)
