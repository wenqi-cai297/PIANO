"""PyTorch Dataset classes for PIANO training.

Provides ``HOIDataset`` that loads preprocessed motion sequences, object
point clouds, pseudo interaction labels, and text descriptions for training
the interaction predictor and motion generator.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
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
# SMPL-22 left/right joint pairs for mirror augmentation
# ---------------------------------------------------------------------------

# (left_idx, right_idx) pairs. Pelvis (0), spines (3/6/9), neck (12), head
# (15) are midline and don't swap.
_SMPL22_LR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),   # hip
    (4, 5),   # knee
    (7, 8),   # ankle
    (10, 11), # foot (mid-foot)
    (13, 14), # collar
    (16, 17), # shoulder
    (18, 19), # elbow
    (20, 21), # wrist
)

# Body-part indices in contact_state / contact_target (see
# piano.utils.smpl_utils.INTERACTION_BODY_PARTS): 0=L_hand, 1=R_hand,
# 2=L_foot, 3=R_foot, 4=pelvis. Mirror swaps 0↔1 and 2↔3.
_BODY_PART_LR_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (2, 3))


# ---------------------------------------------------------------------------
# Object-id split (H5)
# ---------------------------------------------------------------------------

def build_object_split(
    object_ids: list[str],
    train_pct: int = 85,
    val_pct: int = 8,
    test_pct: int = 7,
    seed: int = 42,
) -> dict[str, set[str]]:
    """Deterministically assign each object_id to train / val / test.

    Uses md5(seed || obj_id) % 100 so the split is reproducible across
    processes (no global state), stable under object-id additions, and
    does not require a pre-shuffled list. All processes in DDP end up
    with the same split without having to broadcast it.

    Parameters
    ----------
    object_ids : unique object ids across all subsets.
    train_pct / val_pct / test_pct : must sum to 100.
    seed : changes the hash salt, shuffles the assignment.

    Returns
    -------
    dict with keys "train" / "val" / "test" → set of object_ids.
    """
    if train_pct + val_pct + test_pct != 100:
        raise ValueError(
            f"train_pct + val_pct + test_pct must sum to 100, "
            f"got {train_pct + val_pct + test_pct}",
        )

    train: set[str] = set()
    val: set[str] = set()
    test: set[str] = set()
    for obj_id in object_ids:
        # Salted hash — seed=0 gives a different split than seed=42 etc.
        h = hashlib.md5(f"{seed}::{obj_id}".encode("utf-8")).hexdigest()[:8]
        bucket = int(h, 16) % 100
        if bucket < train_pct:
            train.add(obj_id)
        elif bucket < train_pct + val_pct:
            val.add(obj_id)
        else:
            test.add(obj_id)
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Augmentation config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AugmentConfig:
    """Stage A data augmentation. All augmentations preserve the
    pseudo-label validity — see design notes in predictor.yaml."""

    enabled: bool = False
    mirror_prob: float = 0.0
    rotate_around_y_prob: float = 0.0
    pc_jitter_std: float = 0.0


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
    object_id_filter : optional set of object_ids. When provided, only
        metadata entries whose object_id is in the set are retained —
        used by the H5 train/val/test split.
    augment : optional AugmentConfig. When ``augment.enabled`` is True,
        each __getitem__ randomly applies (mirror, rotate around Y, pc
        jitter) per the config probabilities. Disabled by default so
        the dataset class is reusable for eval / inference.
    """

    def __init__(
        self,
        root: str | Path,
        pseudo_label_dir: str | Path | None = None,
        max_seq_length: int = 196,
        num_object_points: int = 1024,
        use_clean_metadata: bool = True,
        object_id_filter: set[str] | None = None,
        augment: AugmentConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.max_seq_length = max_seq_length
        self.num_object_points = num_object_points
        self.augment = augment or AugmentConfig()

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
        metadata: list[dict] = load_json(meta_path)

        # Object-id split filter (H5). Drops metadata entries whose
        # object_id is not in the allowed set.
        if object_id_filter is not None:
            metadata = [m for m in metadata if m.get("object_id") in object_id_filter]

        self.metadata = metadata
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

        padded_labels: dict[str, np.ndarray] = {}
        for key in ("contact_state", "contact_target", "phase", "support"):
            if labels.get(key) is not None:
                padded_labels[key] = self._pad_or_truncate(labels[key], self.max_seq_length)

        text = meta.get("text", "")

        # --- Augment (training only; disabled by default) ---
        if self.augment.enabled:
            joints, object_pc, text, padded_labels = self._apply_augmentation(
                joints, object_pc, text, padded_labels,
            )

        # --- Build output dict ---
        result: dict[str, torch.Tensor] = {
            "motion": torch.from_numpy(motion),
            "joints": torch.from_numpy(joints),
            "object_pc": torch.from_numpy(object_pc),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "text": text,
        }
        if object_positions is not None:
            result["object_positions"] = torch.from_numpy(object_positions)
        for key, arr in padded_labels.items():
            result[key] = torch.from_numpy(arr)

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

    # -------------------------------------------------------------------
    # Augmentation
    # -------------------------------------------------------------------

    def _apply_augmentation(
        self,
        joints: np.ndarray,             # (T, 22, 3)
        object_pc: np.ndarray,          # (N, 3)
        text: str,
        labels: dict[str, np.ndarray],  # may contain contact_state, contact_target, phase, support
    ) -> tuple[np.ndarray, np.ndarray, str, dict[str, np.ndarray]]:
        """Apply mirror / Y-rotation / pc-jitter augmentations.

        All three are pseudo-label-safe because labels were extracted in
        object-local frame (rotation-invariant w.r.t. world heading) and
        mirror only swaps L/R channels on human-side arrays.
        """
        # 1. Left-right mirror. Flips world-x, swaps SMPL L/R joint
        #    pairs, swaps L/R body-part channels in contact_state /
        #    contact_target, swaps 'left' ↔ 'right' in text. Object
        #    point cloud is in object-local frame (no L/R structure
        #    the label space knows about), so we leave it alone.
        #    Phase / support are L/R-agnostic.
        if self.augment.mirror_prob > 0 and random.random() < self.augment.mirror_prob:
            joints = joints.copy()
            joints[:, :, 0] *= -1.0
            # Swap L/R joint pairs
            for li, ri in _SMPL22_LR_PAIRS:
                tmp = joints[:, li, :].copy()
                joints[:, li, :] = joints[:, ri, :]
                joints[:, ri, :] = tmp

            if "contact_state" in labels:
                cs = labels["contact_state"].copy()
                for li, ri in _BODY_PART_LR_PAIRS:
                    tmp = cs[:, li].copy()
                    cs[:, li] = cs[:, ri]
                    cs[:, ri] = tmp
                labels["contact_state"] = cs

            if "contact_target" in labels:
                ct = labels["contact_target"].copy()
                for li, ri in _BODY_PART_LR_PAIRS:
                    tmp = ct[:, li, :].copy()
                    ct[:, li, :] = ct[:, ri, :]
                    ct[:, ri, :] = tmp
                labels["contact_target"] = ct

            text = _swap_left_right_in_text(text)

        # 2. Random rotation around world Y (up axis). Rotates every
        #    joint frame by the same θ. Labels are rotation-invariant
        #    (object-local frame). Object PC in object-local frame is
        #    untouched.
        if (
            self.augment.rotate_around_y_prob > 0
            and random.random() < self.augment.rotate_around_y_prob
        ):
            theta = random.uniform(-math.pi, math.pi)
            c, s = math.cos(theta), math.sin(theta)
            # R @ v for v = (x, y, z):  (c x + s z, y, -s x + c z)
            x = joints[..., 0]
            z = joints[..., 2]
            new_x = c * x + s * z
            new_z = -s * x + c * z
            joints = joints.copy()
            joints[..., 0] = new_x
            joints[..., 2] = new_z

        # 3. Small Gaussian jitter on the object point cloud. PointNeXt
        #    training recipe. Label-independent.
        if self.augment.pc_jitter_std > 0:
            noise = np.random.randn(*object_pc.shape).astype(np.float32)
            object_pc = object_pc + self.augment.pc_jitter_std * noise

        return joints, object_pc, text, labels


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


# ---------------------------------------------------------------------------
# Text L/R swap (helper for mirror augmentation)
# ---------------------------------------------------------------------------

_LEFT_RE = re.compile(r"\bleft\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\bright\b", re.IGNORECASE)
_LEFT_PLACEHOLDER = "\x00LEFT\x00"


def _swap_left_right_in_text(text: str) -> str:
    """Case-insensitive word-level swap of 'left' ↔ 'right'.

    Uses a NUL-byte placeholder to avoid the classic double-substitution
    bug (``left → right → left`` in a single pass). Word boundaries stop
    false positives on substrings like 'bright' or 'cleft'.
    """
    if not text:
        return text
    tmp = _LEFT_RE.sub(_LEFT_PLACEHOLDER, text)
    tmp = _RIGHT_RE.sub("left", tmp)
    return tmp.replace(_LEFT_PLACEHOLDER, "right")
