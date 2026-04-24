"""PyTorch Dataset classes for PIANO training.

Provides ``HOIDataset`` that loads preprocessed motion sequences, object
point clouds, pseudo interaction labels, and text descriptions for training
the interaction predictor and motion generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from piano.utils.io_utils import load_json


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HOISample:
    """A single human-object interaction sample."""

    motion: np.ndarray              # (T, 263) HumanML3D features
    joints: np.ndarray              # (T, 22, 3) joint positions
    object_pc: np.ndarray           # (N, 3) object point cloud
    text: str                       # text description

    # Pseudo interaction labels (None until extracted)
    contact_state: np.ndarray | None = None    # (T, B)
    contact_target: np.ndarray | None = None   # (T, B, K)
    phase: np.ndarray | None = None            # (T,) int
    support: np.ndarray | None = None          # (T,) int

    seq_len: int = 0                # actual sequence length before padding


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HOIDataset(Dataset):
    """Dataset for human-object interaction motion generation.

    Directory layout expected::

        root/
            metadata.json           # list of {seq_id, text, object_id, ...}
            motions/
                seq_001.npz         # contains motion_263, joints_22
                ...
            objects/
                obj_001.npy         # (N, 3) object point cloud
                ...
            pseudo_labels/          # optional, from pseudo-label extraction
                seq_001.npz         # contains contact_state, contact_target, phase, support
                ...

    Parameters
    ----------
    root : path to dataset root
    pseudo_label_dir : path to pseudo-label directory (overrides root/pseudo_labels)
    max_seq_length : pad/truncate to this length
    num_object_points : subsample object point cloud to this size
    use_clean_metadata : if True and ``root/metadata_clean.json`` exists,
        load that instead of ``root/metadata.json``. The cleaned version
        is produced by ``scripts/stage1_pseudo_labels/clean_pseudo_labels.py`` and drops
        sequences whose pseudo-labels are unusable or contradict the
        text description. Default True so training naturally uses the
        filtered set once it exists; set False to train on the raw set.
    """

    def __init__(
        self,
        root: str | Path,
        pseudo_label_dir: str | Path | None = None,
        max_seq_length: int = 196,
        num_object_points: int = 1024,
        use_clean_metadata: bool = True,
    ) -> None:
        self.root = Path(root)
        self.max_seq_length = max_seq_length
        self.num_object_points = num_object_points

        # Load metadata — prefer metadata_clean.json when it exists so
        # training skips sequences the cleaning tool flagged as bad
        # (zero contact, text-label contradictions, garbled text, etc).
        # Fall back to metadata.json for datasets that haven't been
        # cleaned yet.
        meta_path: Path | None = None
        if use_clean_metadata:
            candidate = self.root / "metadata_clean.json"
            if candidate.exists():
                meta_path = candidate
        if meta_path is None:
            meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata not found in {self.root}")
        self.metadata: list[dict] = load_json(meta_path)
        self.metadata_source = meta_path.name

        # Pseudo-label directory
        if pseudo_label_dir is not None:
            self.pseudo_label_dir = Path(pseudo_label_dir)
        else:
            self.pseudo_label_dir = self.root / "pseudo_labels"

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        meta = self.metadata[idx]
        seq_id = meta["seq_id"]

        # --- Load motion ---
        motion_path = self.root / "motions" / f"{seq_id}.npz"
        motion_data = np.load(motion_path, allow_pickle=False)
        motion = motion_data["motion_263"].astype(np.float32)    # (T, 263)
        joints = motion_data["joints_22"].astype(np.float32)     # (T, 22, 3)
        seq_len = min(len(motion), self.max_seq_length)

        # Object trajectory in world frame (per-frame object center position).
        # Used for pseudo-label extraction and for overlay during visualization.
        object_positions = None
        if "object_positions" in motion_data.files:
            object_positions = motion_data["object_positions"].astype(np.float32)

        # --- Load object point cloud ---
        obj_id = meta["object_id"]
        obj_path = self.root / "objects" / f"{obj_id}.npy"
        object_pc = np.load(obj_path).astype(np.float32)  # (N, 3)
        object_pc = self._subsample_points(object_pc, self.num_object_points)

        # --- Load pseudo-labels (if available) ---
        label_path = self.pseudo_label_dir / f"{seq_id}.npz"
        labels = self._load_pseudo_labels(label_path, seq_len)

        # --- Pad or truncate to max_seq_length ---
        motion = self._pad_or_truncate(motion, self.max_seq_length)
        joints = self._pad_or_truncate(joints, self.max_seq_length)
        if object_positions is not None:
            object_positions = self._pad_or_truncate(object_positions, self.max_seq_length)

        # --- Build output dict ---
        result: dict[str, torch.Tensor] = {
            "motion": torch.from_numpy(motion),
            "joints": torch.from_numpy(joints),
            "object_pc": torch.from_numpy(object_pc),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "text": meta.get("text", ""),
        }
        if object_positions is not None:
            result["object_positions"] = torch.from_numpy(object_positions)

        # Add pseudo-labels if present
        for key in ("contact_state", "contact_target", "phase", "support"):
            if labels.get(key) is not None:
                padded = self._pad_or_truncate(labels[key], self.max_seq_length)
                result[key] = torch.from_numpy(padded)

        return result

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _subsample_points(points: np.ndarray, n: int) -> np.ndarray:
        """Randomly subsample *n* points (with replacement if needed)."""
        if len(points) >= n:
            indices = np.random.choice(len(points), n, replace=False)
        else:
            indices = np.random.choice(len(points), n, replace=True)
        return points[indices]

    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, length: int) -> np.ndarray:
        """Pad with zeros or truncate the first axis to *length*."""
        if len(arr) >= length:
            return arr[:length]
        pad_width = [(0, length - len(arr))] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, pad_width, mode="constant", constant_values=0)

    def _load_pseudo_labels(
        self, path: Path, seq_len: int,
    ) -> dict[str, np.ndarray | None]:
        """Load pseudo-labels if they exist, otherwise return Nones."""
        result: dict[str, np.ndarray | None] = {
            "contact_state": None,
            "contact_target": None,
            "phase": None,
            "support": None,
        }
        if not path.exists():
            return result

        data = np.load(path, allow_pickle=False)
        for key in result:
            if key in data:
                arr = data[key].astype(np.float32) if key != "phase" and key != "support" else data[key]
                result[key] = arr[:seq_len]

        return result


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_hoi(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Custom collate that handles variable-presence pseudo-labels and text strings."""
    result: dict[str, torch.Tensor | list[str]] = {}

    # Stack all tensor fields
    tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    for key in tensor_keys:
        result[key] = torch.stack([sample[key] for sample in batch])

    # Collect text as list of strings
    if "text" in batch[0]:
        result["text"] = [sample["text"] for sample in batch]

    return result
