"""PyTorch Dataset classes for PIANO training.

Provides ``HOIDataset`` that loads preprocessed motion sequences, object
point clouds, pseudo interaction labels, and text descriptions for training
the interaction predictor and motion generator.
"""
from __future__ import annotations

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
# Train/val splits live in piano.data.split — pure-Python, no torch dep.
# Re-exported here so existing call sites (training scripts) keep their
# imports stable: `from piano.data.dataset import build_subject_split`
# still works; the test suite imports from the underlying module
# directly to avoid pulling in torch.
# ---------------------------------------------------------------------------

from piano.data.split import (
    build_object_split,
    build_subject_split,
    extract_subject_id,
)


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
        used by the secondary (object-id) split, kept for novel-object
        ablation eval.
    subject_id_filter : optional set of namespaced subject keys of the
        form ``"{subset}/{raw_subject_id}"``. When provided, only
        metadata entries whose seq_id parses to a subject in the set
        are retained — used by the primary (subject-id) split as of
        2026-04-26. The subset name is taken from ``self.root.name``;
        subject extraction uses ``extract_subject_id``. Entries whose
        seq_id doesn't parse (subset has no pattern, or seq_id format
        unexpected) are dropped — preferable to silently keeping them
        in train and leaking into val.
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
        subject_id_filter: set[str] | None = None,
        augment: AugmentConfig | None = None,
        surface_obj_pose: bool = False,
        force_world_frame: bool = False,
    ) -> None:
        self.root = Path(root)
        self.max_seq_length = max_seq_length
        self.num_object_points = num_object_points
        self.augment = augment or AugmentConfig()
        # v0.2: when True, compute body-canonical-frame object pose
        # (obj_com_canonical, obj_rot6d_canonical) per clip and surface
        # them in __getitem__. Stage B's tokenizer uses these as the
        # per-frame object pose channels (per
        # analyses/2026-04-27_object_conditioning_review.md §5.2).
        # Off by default so Stage A predictor training (which doesn't
        # need them) doesn't pay the MoMask-recovery import cost.
        self.surface_obj_pose = surface_obj_pose
        # v0.3-α (2026-04-27 evening): when True AND surface_obj_pose is
        # True, the obj-pose channels are returned in WORLD frame instead
        # of body-canonical frame. Tests Hypothesis E (frame-choice
        # confused the model) per
        # analyses/2026-04-27_v0_3_root_cause_research.md §"v0.3-α".
        # The 7-method lit consensus
        # (analyses/2026-04-27_object_conditioning_review.md §3.2)
        # is world-frame; v0.2 deviated to body-canonical and got
        # body-in-place visual failure despite a measurably alive
        # adapter (effect-size 9.3% mean / 21.2% peak), so this flag
        # tests the canonicalization choice in isolation.
        self.force_world_frame = force_world_frame

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

        # Object-id split filter (legacy primary, now secondary —
        # used only by the novel-object ablation eval). Drops metadata
        # entries whose object_id is not in the allowed set.
        if object_id_filter is not None:
            metadata = [m for m in metadata if m.get("object_id") in object_id_filter]

        # Subject-id split filter (primary as of 2026-04-26). Drops
        # metadata entries whose extracted subject (namespaced as
        # "{subset}/{raw_subject_id}") is not in the allowed set.
        # Entries whose seq_id doesn't parse fall through to "drop";
        # cheaper than silently keeping them in the wrong bucket.
        if subject_id_filter is not None:
            subset_name = self.root.name
            kept = []
            for m in metadata:
                raw_id = extract_subject_id(subset_name, m.get("seq_id", ""))
                if raw_id is None:
                    continue
                if f"{subset_name}/{raw_id}" in subject_id_filter:
                    kept.append(m)
            metadata = kept

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

        # Object trajectory in world frame (per-frame object center position
        # + per-frame axis-angle rotation). Used for pseudo-label extraction
        # and — when present — for overlay during visualization.
        object_positions = None
        if "object_positions" in motion_data.files:
            object_positions = motion_data["object_positions"].astype(np.float32)
        object_rotations = None
        if "object_rotations" in motion_data.files:
            object_rotations = motion_data["object_rotations"].astype(np.float32)

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
        # v0.2: compute body-canonical-frame object pose BEFORE padding,
        # so the canonicalization sees the genuine frame-0 pelvis (not
        # a zero-pad row). The output is then padded along the time axis
        # to ``max_seq_length`` like the other per-frame fields.
        obj_com_canonical: np.ndarray | None = None
        obj_rot6d_canonical: np.ndarray | None = None
        if (
            self.surface_obj_pose
            and object_positions is not None
            and object_rotations is not None
        ):
            valid = min(seq_len, len(joints), len(motion))
            obj_com_canonical, obj_rot6d_canonical = (
                self._compute_canonical_object_pose(
                    motion[:valid],
                    joints[:valid],
                    object_positions[:valid],
                    object_rotations[:valid],
                    force_world_frame=self.force_world_frame,
                )
            )

        if object_positions is not None:
            object_positions = self._pad_or_truncate(object_positions, self.max_seq_length)
        if object_rotations is not None:
            object_rotations = self._pad_or_truncate(object_rotations, self.max_seq_length)
        if obj_com_canonical is not None:
            obj_com_canonical = self._pad_or_truncate(obj_com_canonical, self.max_seq_length)
        if obj_rot6d_canonical is not None:
            obj_rot6d_canonical = self._pad_or_truncate(obj_rot6d_canonical, self.max_seq_length)

        padded_labels: dict[str, np.ndarray] = {}
        for key in ("contact_state", "contact_target", "contact_target_xyz", "phase", "support"):
            if labels.get(key) is not None:
                padded_labels[key] = self._pad_or_truncate(labels[key], self.max_seq_length)

        text = meta.get("text", "")

        # --- Augment (training only; disabled by default) ---
        if self.augment.enabled:
            joints, object_pc, text, padded_labels = self._apply_augmentation(
                joints, object_pc, text, padded_labels,
            )

        # --- Build output dict ---
        # `object_id` and `subset` are passed through as plain strings
        # so downstream code (eval_predictor per-object aggregation) can
        # group clips by object identity. The collate fn accumulates them
        # as parallel string lists alongside the tensor batches.
        result: dict[str, torch.Tensor | str] = {
            "motion": torch.from_numpy(motion),
            "joints": torch.from_numpy(joints),
            "object_pc": torch.from_numpy(object_pc),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "text": text,
            "object_id": str(obj_id),
            "subset": self.root.name,
            "seq_id": str(seq_id),
        }
        if object_positions is not None:
            result["object_positions"] = torch.from_numpy(object_positions)
        if object_rotations is not None:
            result["object_rotations"] = torch.from_numpy(object_rotations)
        if obj_com_canonical is not None:
            result["obj_com_canonical"] = torch.from_numpy(obj_com_canonical)
        if obj_rot6d_canonical is not None:
            result["obj_rot6d_canonical"] = torch.from_numpy(obj_rot6d_canonical)
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
    def _compute_canonical_object_pose(
        motion_263: np.ndarray,         # (T, 263) HumanML3D canonical
        joints_world: np.ndarray,       # (T, 22, 3) world frame
        object_positions: np.ndarray,   # (T, 3) world frame
        object_rotations: np.ndarray,   # (T, 3) axis-angle, world frame
        force_world_frame: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lift the per-frame world-frame object pose into body-canonical
        frame (default), matching the body's representation.

        The transform ``(R_y, T_xz)`` is recovered from frame 0 of
        ``joints_world`` vs ``recover_from_ric(motion_263)`` (canonical),
        then inverted and applied to every frame. See
        :mod:`piano.utils.canonical_frame` for the maths and
        :doc:`analyses/2026-04-27_object_conditioning_review.md` for the
        v0.2 design rationale.

        When ``force_world_frame=True``, short-circuits the recovery and
        passes ``(R_y=0, T_xz=[0, 0])`` to
        :func:`world_to_canonical_object_pose` — the result is then the
        world-frame obj pose in 6D rep (since the inverse transform is
        identity). Used by v0.3-α to test Hypothesis E (frame-choice
        deviation from the 7/7 lit consensus). Skips the costly
        ``recover_from_ric`` call.
        """
        # Lazy import: MoMask path setup is paid only when callers pass
        # ``surface_obj_pose=True`` (Stage B), not by Stage A predictor
        # training.
        from piano.utils.canonical_frame import (
            world_to_canonical_object_pose,
        )

        if force_world_frame:
            R_y = 0.0
            T_xz = np.zeros(2, dtype=np.float32)
            return world_to_canonical_object_pose(
                object_positions, object_rotations, R_y, T_xz,
            )

        import torch as _torch
        import piano.models.backbones.momask_adapter  # noqa: F401 — path side-effect
        from utils.motion_process import recover_from_ric

        from piano.utils.canonical_frame import (
            get_canonicalize_transform_from_clip,
        )

        canonical_joints = (
            recover_from_ric(
                _torch.from_numpy(motion_263).float().unsqueeze(0),
                joints_num=22,
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        R_y, T_xz = get_canonicalize_transform_from_clip(
            joints_world, canonical_joints,
        )
        return world_to_canonical_object_pose(
            object_positions, object_rotations, R_y, T_xz,
        )

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
        """Load pseudo-labels and surface the xyz target the predictor uses.

        ``contact_target_xyz`` source priority:

        1. ``contact_target_xyz_gt`` field if present in the npz —
           **closest-surface-point on the mesh** in object-local frame
           (v10 / v3 extraction). Use this when available; it has zero
           GT bias and is the right ground-truth for the predictor's
           xyz regression head.
        2. Fallback: derive at load time from the legacy
           ``contact_target (T, 5, K)`` soft distribution and the per-
           object ``patch_centers (K, 3)`` via softmax-weighted
           centroid Σ_k soft[t,b,k] × patch_centers[k]. This carries
           an estimated 5-10 cm bias against the true closest-surface
           point — only used when the npz pre-dates the v10 extractor.

        ``contact_target`` (original soft K-way distribution) is kept
        for backward compat / visualisation; the predictor doesn't
        consume it any more.
        """
        result: dict[str, np.ndarray | None] = {
            "contact_state": None,
            "contact_target": None,
            "contact_target_xyz": None,
            "phase": None,
            "support": None,
        }
        if not path.exists():
            return result

        data = np.load(path, allow_pickle=False)
        for key in ("contact_state", "contact_target", "phase", "support"):
            if key in data:
                arr = data[key].astype(np.float32) if key not in ("phase", "support") else data[key]
                result[key] = arr[:seq_len]

        # 1. Preferred path: exact closest-surface-point GT from the
        # extractor. Always use this when the npz has it.
        if "contact_target_xyz_gt" in data.files:
            result["contact_target_xyz"] = (
                data["contact_target_xyz_gt"].astype(np.float32)[:seq_len]
            )
        # 2. Fallback: softmax-weighted centroid (legacy, biased).
        elif (
            result["contact_target"] is not None
            and "patch_centers" in data.files
        ):
            patch_centers = data["patch_centers"].astype(np.float32)   # (K, 3)
            soft = result["contact_target"]                             # (T, 5, K)
            result["contact_target_xyz"] = np.einsum(
                "tbk,kd->tbd", soft, patch_centers,
            ).astype(np.float32)

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

            if "contact_target_xyz" in labels:
                # Swap L/R body-part channels. The xyz values themselves
                # stay in the object-local frame (which doesn't flip
                # under human-side mirror). For an object roughly
                # symmetric about the human's sagittal plane this is
                # exact; for asymmetric objects it's an approximation
                # that the community treats as a regularisation benefit.
                ctx = labels["contact_target_xyz"].copy()
                for li, ri in _BODY_PART_LR_PAIRS:
                    tmp = ctx[:, li, :].copy()
                    ctx[:, li, :] = ctx[:, ri, :]
                    ctx[:, ri, :] = tmp
                labels["contact_target_xyz"] = ctx

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

def compute_class_priors(
    dataset,
    num_phases: int,
    num_support: int,
    sample_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tally per-class frequencies for phase + support over a dataset.

    Used by Logit Adjustment (Menon ICLR'21) — needs ``log π_y`` for
    each class, computed from the training-set marginal frequency.
    Iterates the dataset's underlying npzs directly (cheap — phase
    and support are int arrays with no augmentation needed).

    Parameters
    ----------
    dataset : ConcatDataset of HOIDataset, or a single HOIDataset
    num_phases : expected number of phase classes (defines vector length)
    num_support : expected number of support classes
    sample_limit : if set, stop after this many sequences (for fast smoke).
        Default None = scan the whole dataset.

    Returns
    -------
    phase_freq : (num_phases,) float32 — sums to 1
    support_freq : (num_support,) float32 — sums to 1
    """
    from torch.utils.data import ConcatDataset as _ConcatDataset

    if isinstance(dataset, _ConcatDataset):
        sub_datasets = list(dataset.datasets)
    else:
        sub_datasets = [dataset]

    phase_counts = np.zeros(num_phases, dtype=np.int64)
    support_counts = np.zeros(num_support, dtype=np.int64)
    n_seq = 0
    for sub in sub_datasets:
        if not isinstance(sub, HOIDataset):
            continue
        for entry in sub.metadata:
            if sample_limit is not None and n_seq >= sample_limit:
                break
            seq_id = entry["seq_id"]
            label_path = sub.pseudo_label_dir / f"{seq_id}.npz"
            if not label_path.exists():
                continue
            data = np.load(label_path, allow_pickle=False)
            if "phase" in data.files:
                ph = data["phase"]
                phase_counts += np.bincount(ph, minlength=num_phases)[:num_phases]
            if "support" in data.files:
                su = data["support"]
                support_counts += np.bincount(su, minlength=num_support)[:num_support]
            n_seq += 1
        if sample_limit is not None and n_seq >= sample_limit:
            break

    phase_freq = phase_counts.astype(np.float32) / max(phase_counts.sum(), 1)
    support_freq = support_counts.astype(np.float32) / max(support_counts.sum(), 1)
    return phase_freq, support_freq


def collate_hoi(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Custom collate that handles variable-presence pseudo-labels and string fields.

    Tensor fields are stacked. String fields (text, object_id, subset,
    seq_id) are accumulated as parallel ``list[str]`` so downstream
    code can pair each batch row with its provenance (per-object
    eval analysis, debugging, etc.).
    """
    result: dict[str, torch.Tensor | list[str]] = {}

    # Stack all tensor fields
    tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    for key in tensor_keys:
        result[key] = torch.stack([sample[key] for sample in batch])

    # Collect string fields as parallel lists (text + provenance fields)
    for key in ("text", "object_id", "subset", "seq_id"):
        if key in batch[0]:
            result[key] = [sample[key] for sample in batch]

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
