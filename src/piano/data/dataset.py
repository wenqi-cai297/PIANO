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
    mirror_duplicate: bool = False
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
    motion_representation : {"motion_263", "joints22_world"}
        Which per-frame body representation to return under the ``motion``
        key. ``motion_263`` preserves the MoMask/HumanML3D path. The
        ``joints22_world`` branch returns flattened world-frame SMPL-22
        joint positions and is used by AnchorDiff v4 to train in HOI task
        space rather than through HumanML3D's cumulative root-yaw decoder.
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
        subsample_n_per_object: int | None = None,
        subsample_seed: int = 42,
        augment: AugmentConfig | None = None,
        surface_obj_pose: bool = False,
        force_world_frame: bool = False,
        motion_representation: str = "motion_263",
        keyframe_subdir: str = "keyframes/v8_default",
        keyframe_max_k: int = 12,
        # v9.1 (2026-05-03): collapse hand_support (id=3) into both_feet
        # (id=0) at load time. The compound class hand_support
        # = (hand_contact ∧ pelvis_static ∧ phase_stable) is essentially
        # "both_feet on floor + extra hand bracing" in InterAct (no
        # gymnastic poses with feet airborne); collapsing it discards
        # zero useful information for Stage B (which can derive the
        # hand-bracing condition from contact_state[hand] + phase).
        # When True, num_support_states should be 3 in the model
        # config and SUPPORT_NAMES is implicitly truncated to the
        # first 3 names.
        support_collapse_hand_support: bool = False,
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
        self.support_collapse_hand_support = bool(support_collapse_hand_support)
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
        if motion_representation not in {
            "motion_263",
            "joints22_world",
            "joints22_world_with_rot6d",
            "smpl_pose_135",
            "smpl_pose_135_keyframed",
            "smpl_pose_135_condmdi",
            "smpl_pose_135_plan",
        }:
            raise ValueError(
                "motion_representation must be one of "
                "{motion_263, joints22_world, joints22_world_with_rot6d, "
                "smpl_pose_135, smpl_pose_135_keyframed, smpl_pose_135_condmdi, "
                "smpl_pose_135_plan}, "
                f"got {motion_representation!r}"
            )
        self.motion_representation = motion_representation
        self.keyframe_subdir = str(keyframe_subdir)
        self.keyframe_max_k = int(keyframe_max_k)
        # InteractionPlan compiler config (lazily built in __getitem__ for
        # the smpl_pose_135_plan branch). Attribute is None when not in
        # plan mode so the lazy import + construction is paid only by the
        # v10 plan-tokens path. See
        # analyses/piano_interaction_plan_pipeline_reframe_for_claude_code.md.
        self._interaction_plan_compiler_cfg = None
        # The 198-D (jpos+rot_6d) and 135-D (rot_6d+root) reps both contain
        # SMPL-22 global 6D rotations derived from raw smplx_poses. The
        # current ``_apply_augmentation`` rotates joints / motion_263 /
        # object pose but does NOT rotate smplx-derived global rotations,
        # which would desync the rep. v5 / v6 configs must therefore disable
        # mirror + Y-rotation augmentation.
        if motion_representation in {
            "joints22_world_with_rot6d",
            "smpl_pose_135",
            "smpl_pose_135_keyframed",
            "smpl_pose_135_condmdi",
            "smpl_pose_135_plan",
        }:
            aug = augment or AugmentConfig()
            if aug.enabled and (aug.mirror_prob > 0 or aug.rotate_around_y_prob > 0):
                raise ValueError(
                    f"motion_representation={motion_representation!r} is not "
                    "compatible with mirror_prob>0 or rotate_around_y_prob>0 — "
                    "global rot 6D would desync. Set both probs to 0 in the config."
                )

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

        # Method-validation subsampling (2026-05-09): randomly keep at
        # most ``subsample_n_per_object`` clips per object_id from the
        # full metadata. Applied BEFORE any train/val split so the same
        # global subset is used by all dataset instances sharing the
        # same ``subsample_seed``. Trims a 8.5K-clip pipeline to ~3K
        # for fast iteration; switch off (set to None) for full-data
        # production training.
        if subsample_n_per_object is not None:
            import numpy as _np
            rng = _np.random.default_rng(int(subsample_seed))
            by_obj: dict[str, list[dict]] = {}
            for m in metadata:
                by_obj.setdefault(str(m.get("object_id", "_unknown")), []).append(m)
            sampled: list[dict] = []
            for obj_id in sorted(by_obj.keys()):
                entries = by_obj[obj_id]
                if len(entries) <= subsample_n_per_object:
                    sampled.extend(entries)
                else:
                    picked_idx = rng.choice(
                        len(entries),
                        size=int(subsample_n_per_object),
                        replace=False,
                    )
                    sampled.extend(entries[i] for i in picked_idx)
            print(
                f"[HOIDataset] Subsampled {self.root.name}: "
                f"{len(metadata)} -> {len(sampled)} clips "
                f"({subsample_n_per_object} per object, seed={subsample_seed})"
            )
            metadata = sampled

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
        if self.augment.enabled and self.augment.mirror_duplicate:
            return len(self.metadata) * 2
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        force_mirror = False
        if self.augment.enabled and self.augment.mirror_duplicate:
            base_idx = idx // 2
            force_mirror = bool(idx % 2)
        else:
            base_idx = idx
        meta = self.metadata[base_idx]
        seq_id = meta["seq_id"]

        # --- Load motion ---
        motion_path = self.root / "motions" / f"{seq_id}.npz"
        motion_data = np.load(motion_path, allow_pickle=False)
        motion_263 = motion_data["motion_263"].astype(np.float32)    # (T, 263)
        joints = motion_data["joints_22"].astype(np.float32)     # (T, 22, 3)
        seq_len = min(len(motion_263), self.max_seq_length)

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

        # v8 Rule C: zero hand contact when support==sitting + pelvis
        # has stable contact (sofa-sit incidental armrest contact that
        # would otherwise flicker as hands raise/lower). Applied to
        # contact_state before any downstream use (z_int packing,
        # anchor loss). Identical logic to keyframe_extraction so
        # offline keyframes and online conditioning stay consistent.
        if (
            labels.get("contact_state") is not None
            and labels.get("support") is not None
        ):
            from piano.data.contact_postprocess import suppress_sitting_hand_contact
            labels["contact_state"] = suppress_sitting_hand_contact(
                labels["contact_state"], labels["support"],
            )

        # --- Pad or truncate to max_seq_length ---
        motion_263 = self._pad_or_truncate(motion_263, self.max_seq_length)
        joints = self._pad_or_truncate(joints, self.max_seq_length)
        if object_positions is not None:
            object_positions = self._pad_or_truncate(object_positions, self.max_seq_length)
        if object_rotations is not None:
            object_rotations = self._pad_or_truncate(object_rotations, self.max_seq_length)

        padded_labels: dict[str, np.ndarray] = {}
        for key in ("contact_state", "contact_target", "contact_target_xyz", "phase", "support"):
            if labels.get(key) is not None:
                padded_labels[key] = self._pad_or_truncate(labels[key], self.max_seq_length)

        text = meta.get("text", "")

        # --- Augment (training only; disabled by default) ---
        # Runs BEFORE canonical-frame computation so that the canonical
        # transform derives from frame-0 of the (possibly mirrored) body
        # pose. For Stage B (surface_obj_pose=True) we pass motion +
        # object_positions + object_rotations through so they get
        # mirrored alongside joints / labels / text — required for the
        # encoder to see consistent-handedness inputs (v0.7+).
        if self.augment.enabled:
            joints, object_pc, text, padded_labels, motion, object_positions, object_rotations = (
                self._apply_augmentation(
                    joints, object_pc, text, padded_labels,
                    motion=motion_263,
                    object_positions=object_positions,
                    object_rotations=object_rotations,
                    force_mirror=force_mirror,
                )
            )
            motion_263 = motion

        # --- Body-canonical object pose (Stage B; v0.2+) ---
        # Computed AFTER augmentation so mirror flips through. The
        # canonical-frame transform is derived from frame-0 of joints
        # + motion_263; both have been mirrored together at this point
        # (see _apply_augmentation Stage-B branch). Uses the unpadded
        # prefix [:valid] so frame-0 references real data, not a
        # zero-pad row.
        obj_com_canonical: np.ndarray | None = None
        obj_rot6d_canonical: np.ndarray | None = None
        if (
            self.surface_obj_pose
            and object_positions is not None
            and object_rotations is not None
        ):
            valid = min(seq_len, len(joints), len(motion_263))
            obj_com_canonical, obj_rot6d_canonical = (
                self._compute_canonical_object_pose(
                    motion_263[:valid],
                    joints[:valid],
                    object_positions[:valid],
                    object_rotations[:valid],
                    force_world_frame=self.force_world_frame,
                )
            )
            obj_com_canonical = self._pad_or_truncate(obj_com_canonical, self.max_seq_length)
            obj_rot6d_canonical = self._pad_or_truncate(obj_rot6d_canonical, self.max_seq_length)

        # --- Build output dict ---
        # `object_id` and `subset` are passed through as plain strings
        # so downstream code (eval_predictor per-object aggregation) can
        # group clips by object identity. The collate fn accumulates them
        # as parallel string lists alongside the tensor batches.
        # Optionally extend motion with global per-joint 6D rotations for
        # OMOMO/CHOIS-style 198-D representation (v5). The 6D rotations
        # are derived from the GT SMPL-X axis-angle local pose by chaining
        # through the SMPL-22 kinematic tree. ``rest_offsets`` is computed
        # by reverse-FK so FK(rot, offsets, root_pos) reproduces the GT
        # joints exactly (verified < 0.1 mm in scratch testing).
        rest_offsets_np: np.ndarray | None = None
        if self.motion_representation == "motion_263":
            motion = motion_263
        elif self.motion_representation == "joints22_world":
            motion = joints.reshape(joints.shape[0], 22 * 3).astype(
                np.float32,
                copy=False,
            )
        elif self.motion_representation in {
            "joints22_world_with_rot6d",
            "smpl_pose_135",
            "smpl_pose_135_keyframed",
            "smpl_pose_135_condmdi",
            "smpl_pose_135_plan",
        }:
            # v5 (joints22_world_with_rot6d) / v6 (smpl_pose_135) both need
            # un-augmented smplx_poses to compute global 6D rotations. Augment
            # rotation propagation is not implemented, so configs must disable
            # mirror + Y-rotation augmentation (enforced in __init__).
            if "smplx_poses" not in motion_data.files:
                raise KeyError(
                    f"smplx_poses missing in {motion_path}; "
                    f"motion_representation={self.motion_representation!r} requires "
                    "smplx_poses (T, 156) and joints_22 (T, 22, 3) in the npz."
                )
            smplx_poses_full = motion_data["smplx_poses"].astype(np.float32)
            # First 66 dims of smplx_poses_full = SMPL-22 root_orient(3)
            # + body_pose(63=21*3) axis-angle local. The remaining 90 dims
            # are SMPL-X jaw/eyes/hands which we ignore here.
            pose_22_aa = smplx_poses_full[:, :66].reshape(-1, 22, 3)
            pose_22_aa = self._pad_or_truncate(
                pose_22_aa, self.max_seq_length,
            ).astype(np.float32)                                # (max_T, 22, 3)

            # Compute global per-joint rotation matrices then 6D rep.
            # Done in torch CPU because our smpl_kinematics utilities are
            # torch-native (Rodrigues + chain through SMPL-22 parents).
            from piano.training.smpl_kinematics import (
                axis_angle_to_matrix as _aa2mat,
                local2global_pose as _l2g,
                matrix_to_rotation_6d as _mat2_6d,
                SMPL22_PARENTS as _PARENTS,
            )
            aa_t = torch.from_numpy(pose_22_aa)                 # (max_T, 22, 3)
            local_R = _aa2mat(aa_t)                             # (max_T, 22, 3, 3)
            global_R = _l2g(local_R)                            # (max_T, 22, 3, 3)
            global_rot_6d = _mat2_6d(global_R)                  # (max_T, 22, 6)

            # Reverse FK on valid frames: rest_offset[j] is the bone vector
            # from parent(j) to j in T-pose coordinates of the parent.
            # Solve: jpos[t,j] - jpos[t,parent(j)] = R_global[t,parent(j)] @ rest_offset[j]
            #     => rest_offset[j] = R_global[t,parent(j)].T @ bone[t,j]
            # Average over valid frames for stability (verified < 0.03 mm
            # standard deviation in scratch test on chairs Sub0001).
            joints_v = torch.from_numpy(joints[:seq_len])       # (seq_len, 22, 3)
            global_R_v = global_R[:seq_len]                     # (seq_len, 22, 3, 3)
            rest_offsets_t = torch.zeros(22, 3, dtype=torch.float32)
            for j in range(1, 22):
                p = int(_PARENTS[j])
                bone = joints_v[:, j, :] - joints_v[:, p, :]    # (seq_len, 3)
                R_inv = global_R_v[:, p, :, :].transpose(-1, -2)
                offset_t = torch.einsum("tij,tj->ti", R_inv, bone)
                rest_offsets_t[j] = offset_t.mean(dim=0)
            rest_offsets_np = rest_offsets_t.numpy()            # (22, 3)

            if self.motion_representation == "joints22_world_with_rot6d":
                # v5: cat(jpos: 66, global_rot_6d: 132) = 198-D
                motion = np.concatenate(
                    [
                        joints.reshape(joints.shape[0], 22 * 3),
                        global_rot_6d.numpy().reshape(global_rot_6d.shape[0], 132),
                    ],
                    axis=-1,
                ).astype(np.float32, copy=False)                # (max_T, 198)
            elif self.motion_representation == "smpl_pose_135_plan":
                # v10: same 135-D base as smpl_pose_135. The
                # InteractionPlan is added below as separate batch fields.
                root_world_pos = joints[:, 0, :].astype(np.float32)
                motion = np.concatenate(
                    [
                        global_rot_6d.numpy().reshape(global_rot_6d.shape[0], 132),
                        root_world_pos,
                    ],
                    axis=-1,
                ).astype(np.float32, copy=False)                # (max_T, 135)
            else:
                # v6 (smpl_pose_135): cat(global_rot_6d: 132, root_world_pos: 3) = 135-D
                # joints[:, 0, :] is root world pos already in our preprocessing
                # (it equals smplx_trans + trans2joint). We use joints[:, 0, :]
                # directly for consistency with the FK chain target.
                root_world_pos = joints[:, 0, :].astype(np.float32)   # (max_T, 3)
                motion = np.concatenate(
                    [
                        global_rot_6d.numpy().reshape(global_rot_6d.shape[0], 132),
                        root_world_pos,
                    ],
                    axis=-1,
                ).astype(np.float32, copy=False)                # (max_T, 135)
        else:                                                   # pragma: no cover
            raise AssertionError(
                f"motion_representation={self.motion_representation!r} not handled"
            )

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
        if rest_offsets_np is not None:
            result["rest_offsets"] = torch.from_numpy(rest_offsets_np)

        # ---------------------------------------------------------------
        # InteractionPlan (v10 plan-tokens path)
        # ---------------------------------------------------------------
        # Compile a sparse interaction program from the dense
        # pseudo-labels and surface as ``plan_*`` tensors. The compiler
        # is deterministic given inputs + config (test_determinism in
        # tests/test_interaction_plan_compiler.py), so caching on disk is
        # not required for now — it adds < 1 ms per __getitem__ call on
        # 196-frame clips.
        if (
            self.motion_representation == "smpl_pose_135_plan"
            and padded_labels.get("contact_state") is not None
            and padded_labels.get("contact_target_xyz") is not None
            and padded_labels.get("phase") is not None
            and padded_labels.get("support") is not None
            and object_positions is not None
            and object_rotations is not None
        ):
            from piano.data.interaction_plan_compiler import (
                InteractionPlanCompilerConfig,
                compile_interaction_plan,
            )
            if self._interaction_plan_compiler_cfg is None:
                # Tied to default compiler hyperparameters (audited in
                # analyses/2026-05-10_interaction_plan_compiler_audit.md).
                # Number of phase / support classes derives from the
                # pseudo-label set the dataset was constructed against.
                num_phase = int(padded_labels["phase"].max()) + 1
                num_phase = max(num_phase, 3)
                # Support is collapsed to 3 in our v18+ ship label set.
                num_support = 3
                self._interaction_plan_compiler_cfg = InteractionPlanCompilerConfig(
                    num_parts=int(padded_labels["contact_state"].shape[1]),
                    num_phase_classes=num_phase,
                    num_support_classes=num_support,
                )

            cfg_compile = self._interaction_plan_compiler_cfg

            # Convert phase / support GT integer arrays to one-hot softmax
            # of the same shape the compiler expects.
            phase_int = padded_labels["phase"].astype(np.int64)
            support_int = padded_labels["support"].astype(np.int64)
            phase_softmax = np.zeros(
                (len(phase_int), cfg_compile.num_phase_classes), dtype=np.float32,
            )
            support_softmax = np.zeros(
                (len(support_int), cfg_compile.num_support_classes), dtype=np.float32,
            )
            phase_safe = np.clip(phase_int, 0, cfg_compile.num_phase_classes - 1)
            support_safe = np.clip(support_int, 0, cfg_compile.num_support_classes - 1)
            phase_softmax[np.arange(len(phase_int)), phase_safe] = 1.0
            support_softmax[np.arange(len(support_int)), support_safe] = 1.0

            plan = compile_interaction_plan(
                contact_prob=padded_labels["contact_state"][:seq_len].astype(np.float32),
                target_local=padded_labels["contact_target_xyz"][:seq_len].astype(np.float32),
                phase_softmax=phase_softmax[:seq_len],
                support_softmax=support_softmax[:seq_len],
                object_pos_world=object_positions[:seq_len].astype(np.float32),
                object_rot_world_aa=object_rotations[:seq_len].astype(np.float32),
                seq_len=seq_len,
                cfg=cfg_compile,
            )
            for k, v in plan.items():
                result[f"plan_{k}"] = torch.from_numpy(v)
        # v8 hierarchical: load offline-precomputed keyframes for the
        # keyframed motion representation. Stage 1 uses these as
        # supervision targets; Stage 2 uses them as sparse condition.
        # Output schema: padded to keyframe_max_k with mask. Variable-K
        # per clip handled via collate_fn default behavior.
        if self.motion_representation == "smpl_pose_135_keyframed":
            kf_path = self.root / self.keyframe_subdir / f"{seq_id}.npz"
            if not kf_path.exists():
                raise FileNotFoundError(
                    f"v8 keyframe file missing: {kf_path}. Run "
                    "scripts/stage_b_generator/extract_v8_keyframes.sh first."
                )
            kf_data = np.load(kf_path, allow_pickle=False)
            indices = kf_data["indices"].astype(np.int64)         # (K,)
            targets = kf_data["targets"].astype(np.float32)       # (K, 6, 3)
            K = int(kf_data["num_keyframes"])
            K_max = self.keyframe_max_k

            # Drop keyframes that fall outside the truncated [0, seq_len)
            # window (offline pass used original T which may exceed
            # self.max_seq_length). Maintain alignment between indices
            # and targets.
            in_window = indices < seq_len
            indices = indices[in_window]
            targets = targets[in_window]
            K = len(indices)

            # Pad to K_max
            kf_indices_padded = np.zeros(K_max, dtype=np.int64)
            kf_targets_padded = np.zeros((K_max, 6, 3), dtype=np.float32)
            kf_mask = np.zeros(K_max, dtype=np.float32)
            n = min(K, K_max)
            kf_indices_padded[:n] = indices[:n]
            kf_targets_padded[:n] = targets[:n]
            kf_mask[:n] = 1.0

            result["keyframe_indices"] = torch.from_numpy(kf_indices_padded)
            result["keyframe_targets"] = torch.from_numpy(kf_targets_padded)
            result["keyframe_mask"] = torch.from_numpy(kf_mask)
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
        R_y, T_xz, T_y = get_canonicalize_transform_from_clip(
            joints_world, canonical_joints,
        )
        return world_to_canonical_object_pose(
            object_positions, object_rotations, R_y, T_xz, T_y=T_y,
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
        # v9.1: collapse hand_support (id=3) → both_feet (id=0) at load
        # time. Saves re-extracting 8475 npzs while letting the model
        # train as 3-way support. See HOIDataset docstring for rationale.
        if (
            getattr(self, "support_collapse_hand_support", False)
            and result["support"] is not None
        ):
            sup = result["support"].copy()
            sup[sup == 3] = 0  # SUPPORT_HAND → SUPPORT_BOTH_FEET
            result["support"] = sup

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
        *,
        motion: np.ndarray | None = None,            # (T, 263) HumanML3D rep (Stage B)
        object_positions: np.ndarray | None = None,  # (T, 3) world frame (Stage B)
        object_rotations: np.ndarray | None = None,  # (T, 3) axis-angle (Stage B)
        force_mirror: bool = False,
    ) -> tuple[
        np.ndarray, np.ndarray, str, dict[str, np.ndarray],
        np.ndarray | None, np.ndarray | None, np.ndarray | None,
    ]:
        """Apply mirror / Y-rotation / pc-jitter augmentations.

        All three are pseudo-label-safe because labels were extracted in
        object-local frame (rotation-invariant w.r.t. world heading) and
        mirror only swaps L/R channels on human-side arrays.

        Stage B path (when ``motion`` / ``object_positions`` /
        ``object_rotations`` are passed): mirror also applies to
        ``motion_263`` (via :func:`piano.utils.humanml3d_mirror.mirror_motion_263`)
        and the world-frame object pose. This is required for Stage B
        because the trainer feeds ``motion_263`` to the VQ encoder; if
        only joints/labels were mirrored the encoder would receive the
        original-handedness tokens contradicting the mirrored z_int.
        Caller must invoke this BEFORE
        :meth:`_compute_canonical_object_pose` so the canonical-frame
        pose derives from frame-0 of the (possibly mirrored) body.

        Stage A path (Stage B fields = None): keeps original behaviour;
        only ``joints`` / labels / text mirror.
        """
        # 1. Left-right mirror. Flips world-x, swaps SMPL L/R joint
        #    pairs, swaps L/R body-part channels in contact_state /
        #    contact_target, swaps 'left' ↔ 'right' in text. Object
        #    point cloud is in object-local frame (no L/R structure
        #    the label space knows about), so we leave it alone.
        #    Phase / support are L/R-agnostic.
        do_mirror = force_mirror or (
            self.augment.mirror_prob > 0
            and random.random() < self.augment.mirror_prob
        )
        if do_mirror:
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

            # Stage B: also mirror motion_263 + world-frame object pose
            # so the encoder + canonical-frame derivation see consistent
            # handedness. Lazy-import to keep Stage A test path light.
            if motion is not None:
                from piano.utils.humanml3d_mirror import mirror_motion_263
                motion = mirror_motion_263(motion)
            if object_positions is not None or object_rotations is not None:
                from piano.utils.humanml3d_mirror import mirror_object_world_pose
                # Both must be present together for a full mirror; if
                # only one is, mirror it in isolation.
                if object_positions is not None and object_rotations is not None:
                    object_positions, object_rotations = mirror_object_world_pose(
                        object_positions, object_rotations,
                    )
                elif object_positions is not None:
                    object_positions = object_positions.copy()
                    object_positions[..., 0] *= -1.0
                else:
                    object_rotations = object_rotations.copy()
                    object_rotations[..., 1] *= -1.0
                    object_rotations[..., 2] *= -1.0

        # 2. Random rotation around world Y (up axis). Rotates the body's
        #    world-frame joints AND (when surface_obj_pose=True) the
        #    object's world pose by the same θ, so the human-object
        #    spatial relationship is preserved. ``motion_263`` is
        #    body-canonical (rotation-invariant by construction) so it
        #    needs no update. ``contact_target_xyz`` is object-local so
        #    it also needs no update.
        if (
            self.augment.rotate_around_y_prob > 0
            and random.random() < self.augment.rotate_around_y_prob
        ):
            theta = random.uniform(-math.pi, math.pi)
            c, s = math.cos(theta), math.sin(theta)
            # R_y @ v for v = (x, y, z):  (c x + s z, y, -s x + c z)
            x = joints[..., 0]
            z = joints[..., 2]
            new_x = c * x + s * z
            new_z = -s * x + c * z
            joints = joints.copy()
            joints[..., 0] = new_x
            joints[..., 2] = new_z

            # Stage B: rotate the object's world pose by the same θ.
            # object_positions: rotate XZ.
            # object_rotations: axis-angle. Composition R_obj_aug =
            # R_y(θ) @ R_obj. Done via quaternion multiplication; the
            # naïve "rotate the axis by R_y, keep angle" is WRONG —
            # rotation composition is non-commutative and the resulting
            # axis-angle's angle changes too. Do proper quat * quat.
            if object_positions is not None:
                # Compute both rotated coordinates BEFORE assigning back —
                # `op_x = object_positions[..., 0]` is a view, and writing
                # to `object_positions[..., 0]` would mutate it before
                # `object_positions[..., 2]`'s formula uses it. Numpy
                # in-place-on-view bug.
                op_x = object_positions[..., 0].copy()
                op_z = object_positions[..., 2].copy()
                object_positions = object_positions.copy()
                object_positions[..., 0] = c * op_x + s * op_z
                object_positions[..., 2] = -s * op_x + c * op_z
            if object_rotations is not None:
                aa = object_rotations.astype(np.float32, copy=True)
                angle = np.linalg.norm(aa, axis=-1, keepdims=True)
                safe_angle = np.where(angle < 1e-12, 1.0, angle)
                axis = aa / safe_angle
                # quaternion of in_rot: (cos(angle/2), axis*sin(angle/2))
                ha = (angle * 0.5).squeeze(-1)
                cha = np.cos(ha)
                sha = np.sin(ha)
                qw = cha
                qx = axis[..., 0] * sha
                qy = axis[..., 1] * sha
                qz = axis[..., 2] * sha
                # Y-rotation quat: (cos(θ/2), 0, sin(θ/2), 0)
                hy = theta * 0.5
                yw = math.cos(hy)
                ys = math.sin(hy)
                # q_y * q_obj  (apply object rotation FIRST in world,
                # then Y-rotation)
                nw = yw * qw - 0   - ys * qy - 0
                nx = yw * qx + 0   + ys * qz - 0   # = yw*qx + ys*qz
                ny = yw * qy - 0   + ys * qw + 0   # = yw*qy + ys*qw
                nz = yw * qz + 0   - ys * qx + 0   # = yw*qz - ys*qx
                # Quaternion → axis-angle.
                # angle_new = 2 * atan2(|imag|, real)
                imag_norm = np.sqrt(nx * nx + ny * ny + nz * nz)
                new_angle = 2.0 * np.arctan2(imag_norm, nw)
                safe_in = np.where(imag_norm < 1e-12, 1.0, imag_norm)
                new_axis_x = nx / safe_in
                new_axis_y = ny / safe_in
                new_axis_z = nz / safe_in
                object_rotations = np.stack(
                    [new_axis_x * new_angle,
                     new_axis_y * new_angle,
                     new_axis_z * new_angle],
                    axis=-1,
                ).astype(np.float32)

        # 3. Small Gaussian jitter on the object point cloud. PointNeXt
        #    training recipe. Label-independent.
        if self.augment.pc_jitter_std > 0:
            noise = np.random.randn(*object_pc.shape).astype(np.float32)
            object_pc = object_pc + self.augment.pc_jitter_std * noise

        return joints, object_pc, text, labels, motion, object_positions, object_rotations


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def compute_class_priors(
    dataset,
    num_phases: int,
    num_support: int,
    sample_limit: int | None = None,
    num_body_parts: int = 5,
    contact_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tally per-class frequencies for phase + support + per-part contact.

    Used by:
    - Logit Adjustment (Menon ICLR'21) for phase + support — needs
      ``log π_y`` for each class, computed from training-set marginals.
    - v9 contact pos_weight for class-balanced BCE — needs π_part
      (positive rate per body part) to set
      ``pos_weight = (1 - π_part) / π_part``.

    Iterates the dataset's underlying npzs directly (cheap — labels
    are int / float arrays with no augmentation needed).

    Parameters
    ----------
    dataset : ConcatDataset of HOIDataset, or a single HOIDataset
    num_phases : expected number of phase classes (defines vector length)
    num_support : expected number of support classes
    sample_limit : if set, stop after this many sequences (for fast smoke).
        Default None = scan the whole dataset.
    num_body_parts : number of body parts in contact_state (default 5)
    contact_threshold : threshold to binarise soft contact_state when
        counting positive frames per body part (default 0.5)

    Returns
    -------
    phase_freq : (num_phases,) float32 — sums to 1
    support_freq : (num_support,) float32 — sums to 1
    contact_part_freq : (num_body_parts,) float32 — per-part positive
        rate (NOT normalised to sum 1; each entry ∈ [0, 1])
    """
    from torch.utils.data import ConcatDataset as _ConcatDataset

    if isinstance(dataset, _ConcatDataset):
        sub_datasets = list(dataset.datasets)
    else:
        sub_datasets = [dataset]

    phase_counts = np.zeros(num_phases, dtype=np.int64)
    support_counts = np.zeros(num_support, dtype=np.int64)
    contact_pos_counts = np.zeros(num_body_parts, dtype=np.int64)
    contact_total_frames = 0
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
            if "contact_state" in data.files:
                cs = data["contact_state"]                              # (T, num_body_parts)
                if cs.ndim == 2 and cs.shape[1] == num_body_parts:
                    pos = (cs > contact_threshold).astype(np.int64)
                    contact_pos_counts += pos.sum(axis=0)
                    contact_total_frames += cs.shape[0]
            n_seq += 1
        if sample_limit is not None and n_seq >= sample_limit:
            break

    phase_freq = phase_counts.astype(np.float32) / max(phase_counts.sum(), 1)
    support_freq = support_counts.astype(np.float32) / max(support_counts.sum(), 1)
    if contact_total_frames > 0:
        contact_part_freq = contact_pos_counts.astype(np.float32) / contact_total_frames
    else:
        contact_part_freq = np.zeros(num_body_parts, dtype=np.float32)
    return phase_freq, support_freq, contact_part_freq


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
