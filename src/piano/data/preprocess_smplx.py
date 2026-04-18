"""SMPL-X to SMPL 22-joint preprocessing pipeline (generic, DEPRECATED).

This generic preprocessor was written before we integrated MoMask's official
``process_file``. It still uses the naive ``joints_to_humanml3d`` and
therefore produces features NOT compatible with MoMask's VQ-VAE.

For OMOMO, use ``piano.data.preprocess_omomo`` instead (which calls
``HumanML3DEncoder`` wrapping MoMask's process_file). For new datasets,
write a similar adapter following preprocess_omomo as the template.

Usage (standalone):
    python -m piano.data.preprocess_smplx \
        --input-dir data/interact/raw \
        --output-dir data/interact/processed \
        --fps 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from piano.data.humanml3d_repr import joints_to_humanml3d
from piano.utils.io_utils import ensure_dir, save_npz
from piano.utils.smpl_utils import smplx_joints_to_smpl22


def preprocess_sequence(
    joints_smplx: np.ndarray,
    object_verts: np.ndarray | None = None,
    fps: float = 30.0,
) -> dict[str, np.ndarray]:
    """Preprocess a single SMPL-X motion sequence.

    Parameters
    ----------
    joints_smplx : (T, J, 3) with J >= 22
        SMPL-X joint positions.
    object_verts : (T, V, 3) or None
        Per-frame object vertex positions (if available).
    fps : frame rate

    Returns
    -------
    Dictionary with keys:
        ``joints_22`` : (T, 22, 3)
        ``motion_263`` : (T, 263)
        ``object_verts`` : (T, V, 3) if provided
    """
    joints_22 = smplx_joints_to_smpl22(joints_smplx)
    motion_263 = joints_to_humanml3d(joints_22, fps=fps)

    result = {
        "joints_22": joints_22,
        "motion_263": motion_263,
    }
    if object_verts is not None:
        result["object_verts"] = object_verts

    return result


def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    fps: float = 30.0,
) -> None:
    """Batch preprocess all ``.npz`` sequences in *input_dir*.

    Expects each npz to contain at least a ``joints`` key with shape
    ``(T, J, 3)`` where ``J >= 22``.  Optionally, ``object_verts``
    with shape ``(T, V, 3)``.
    """
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    npz_files = sorted(input_dir.glob("**/*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} sequences in {input_dir}")
    for npz_path in tqdm(npz_files, desc="Preprocessing"):
        data = np.load(npz_path, allow_pickle=True)

        if "joints" not in data:
            print(f"  Skipping {npz_path.name}: no 'joints' key")
            continue

        joints_smplx = data["joints"]  # (T, J, 3)
        object_verts = data.get("object_verts", None)

        result = preprocess_sequence(joints_smplx, object_verts, fps=fps)

        # Mirror directory structure under output_dir
        rel_path = npz_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        save_npz(out_path, **result)

    print(f"Saved processed sequences to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess SMPL-X HOI data to SMPL 22-joint / HumanML3D format",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Raw data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate (default: 30)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preprocess_directory(args.input_dir, args.output_dir, fps=args.fps)


if __name__ == "__main__":
    main()
