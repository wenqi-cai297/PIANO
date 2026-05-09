"""Full pseudo-label extraction pipeline.

Runs all four extraction steps (contact → target → phase → support)
sequentially for each motion sequence, and saves results as compressed
npz files.

Usage:
    piano-pseudo-labels --data-dir data/interact/processed --output-dir runs/pseudo_labels
    python -m piano.data.pseudo_labels.run_all --data-dir ... --output-dir ...
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import hashlib

from scipy.ndimage import median_filter

from piano.data.preprocess_interact import downsample_temporal
from piano.data.pseudo_labels.extract_contact import ContactConfig, extract_contact_state
from piano.data.pseudo_labels.extract_strict_contact import (
    StrictContactConfig,
    extract_strict_contact_state,
    _filter_short_contacts,
)
from piano.data.pseudo_labels.extract_phase import PhaseConfig, extract_interaction_phase
from piano.data.pseudo_labels.extract_support import SupportConfig, extract_support_state
from piano.data.pseudo_labels.extract_target import TargetConfig, extract_contact_target
from piano.data.pseudo_labels.refine_phase_hmm import (
    HMMConfig,
    build_phase_features,
    refine_phases_hmm,
)
from piano.data.pseudo_labels.stats import (
    aggregate_stats,
    compute_seq_stats,
    make_quality_flags,
)
from piano.utils.geometry import cluster_surface_patches, load_mesh
from piano.utils.io_utils import ensure_dir, load_json, save_json, save_npz
from piano.utils.smpl_utils import BODY_PART_INDICES, BODY_PART_NAMES, NUM_BODY_PARTS


DEFAULT_FPS: float = 20.0  # PIANO preprocessed data rate
OFFICIAL_MARKER_MOTION_CONTACT_OFFSET: int = 231 + 231 + 7 + 7 + 9 + 9
OFFICIAL_MARKER_COUNT: int = 77
DEFAULT_OFFICIAL_MARKER_OBJECTS: tuple[str, ...] = ("*",)
DEFAULT_OFFICIAL_MARKER_PARTS: tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
    "pelvis",
)
DEFAULT_OFFICIAL_MARKER_THRESHOLDS_M: dict[str, float] = {
    "left_hand": 0.05,
    "right_hand": 0.05,
    "left_foot": 0.05,
    "right_foot": 0.05,
    "pelvis": 0.20,
}
# Official InterAct 77-marker semantic groups, from
# src/external/InterAct_official/visualization/visualize_marker.py.
# Hands include both hand and finger markers; feet include foot and toe
# markers. Pelvis keeps the existing joint-nearest fallback because the
# upstream marker groups do not expose a dedicated pelvis set.
OFFICIAL_MARKER_SEMANTIC_INDICES: dict[str, tuple[int, ...]] = {
    "left_hand": (10, 11, 14, 31, 13, 17, 23, 28, 27, 72, 73, 74, 75, 76),
    "right_hand": (60, 43, 44, 47, 62, 46, 51, 57, 67, 68, 69, 70, 71),
    "left_foot": (29, 30, 18, 19, 7, 2, 15, 32, 25, 20, 21, 16),
    "right_foot": (61, 52, 53, 40, 34, 49, 40, 54, 55, 59, 64, 50, 55),
}


def _object_patch_seed(obj_id: str) -> int:
    """Deterministic 32-bit seed derived from object id.

    Guarantees that the same object yields the same patch atlas across
    re-runs and machines. Different objects get independent seeds.
    """
    h = hashlib.md5(obj_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _resolve_fps(data_dir: Path, fps_override: float | None) -> float:
    """Find the fps of preprocessed motions.

    Preference order:
        1. Explicit ``fps_override`` from CLI.
        2. ``target_fps`` in ``<data_dir>/summary.json`` (per-subset summary
           written by newer preprocess_interact).
        3. ``config.target_fps`` in ``<data_dir>/../summary.json`` (top-level
           preprocess summary — covers datasets preprocessed before the
           per-subset fps field was added).
        4. ``DEFAULT_FPS`` with a warning.
    """
    if fps_override is not None:
        return float(fps_override)

    per_subset = data_dir / "summary.json"
    if per_subset.exists():
        try:
            s = load_json(per_subset)
            if "target_fps" in s:
                return float(s["target_fps"])
        except Exception:
            pass

    top_level = data_dir.parent / "summary.json"
    if top_level.exists():
        try:
            s = load_json(top_level)
            fps = s.get("config", {}).get("target_fps")
            if fps is not None:
                return float(fps)
        except Exception:
            pass

    print(f"  [warn] fps not found in {per_subset} or {top_level}; "
          f"defaulting to {DEFAULT_FPS}")
    return DEFAULT_FPS


def _official_sequence_dir(
    official_interact_root: Path,
    subset: str,
    seq_id: str,
) -> Path:
    """Resolve an official InterAct sequence directory for one clip."""
    candidates = (
        official_interact_root / subset / "sequences_canonical" / seq_id,
        official_interact_root / "sequences_canonical" / seq_id,
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_official_marker_contact_data(
    seq_dir: Path,
    T: int,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load official marker-to-object distances and targets from ``motion.npy``.

    Official InterAct canonical motion has 962 channels:
    77 marker positions, marker velocities, two 7-marker foot-height blocks,
    object pose/velocity, then 78 * 6 contact channels. For each marker,
    the first three contact channels are the vector from the marker to the
    closest transformed object point, and the last three are that object's
    canonical/local coordinates. Marker index 77 is an extra ground marker,
    so PIANO only uses the first 77 body markers.
    """
    motion_path = seq_dir / "motion.npy"
    if not motion_path.exists():
        return None
    motion = np.load(motion_path, allow_pickle=False)
    contact_flat = motion[:, OFFICIAL_MARKER_MOTION_CONTACT_OFFSET:]
    expected = (OFFICIAL_MARKER_COUNT + 1) * 6
    if contact_flat.shape[1] != expected:
        raise ValueError(
            f"Unexpected official motion contact width {contact_flat.shape[1]} "
            f"at {motion_path}; expected {expected}"
        )
    contact = contact_flat.reshape(len(motion), OFFICIAL_MARKER_COUNT + 1, 6)
    distances = np.linalg.norm(contact[:, :OFFICIAL_MARKER_COUNT, :3], axis=-1)
    targets_local = contact[:, :OFFICIAL_MARKER_COUNT, 3:6]
    distances_ds = downsample_temporal(distances, 30.0, target_fps)[:T].astype(np.float32)
    targets_ds = downsample_temporal(targets_local, 30.0, target_fps)[:T].astype(np.float32)
    return distances_ds, targets_ds


def _load_official_markers(seq_dir: Path, T: int, target_fps: float) -> np.ndarray | None:
    markers_path = seq_dir / "markers.npy"
    if not markers_path.exists():
        return None
    markers = np.load(markers_path, allow_pickle=False)
    if markers.ndim != 3 or markers.shape[1:] != (OFFICIAL_MARKER_COUNT, 3):
        raise ValueError(f"Unexpected markers.npy shape at {markers_path}: {markers.shape}")
    return downsample_temporal(markers, 30.0, target_fps)[:T].astype(np.float32)


def _nearest_marker_indices(
    markers: np.ndarray,
    joints: np.ndarray,
    *,
    body_part: str,
    k: int,
) -> np.ndarray:
    """Pick surface markers closest to one PIANO body-part joint."""
    joint_idx = BODY_PART_INDICES[BODY_PART_NAMES.index(body_part)]
    distances = np.linalg.norm(markers - joints[:, None, joint_idx, :], axis=-1)
    return np.argsort(distances.mean(axis=0))[:k]


def _official_marker_indices(
    markers: np.ndarray,
    joints: np.ndarray,
    *,
    body_part: str,
    k: int,
) -> np.ndarray:
    """Pick official surface markers for one PIANO body part."""
    semantic = OFFICIAL_MARKER_SEMANTIC_INDICES.get(body_part)
    if semantic is not None:
        return np.asarray(semantic, dtype=np.int64)
    return _nearest_marker_indices(markers, joints, body_part=body_part, k=k)


def _postprocess_marker_binary(
    contact: np.ndarray,
    *,
    median_size: int,
    min_duration: int,
) -> np.ndarray:
    """Match strict-contact temporal smoothing for marker-derived signals."""
    out = contact.astype(np.float32, copy=True)
    for idx in range(out.shape[1]):
        out[:, idx] = median_filter(out[:, idx], size=median_size)
    return (_filter_short_contacts(out, min_duration) > 0.5).astype(np.float32)


def _official_marker_contact_prior(
    *,
    official_interact_root: Path | None,
    subset: str,
    seq_id: str,
    object_id: str,
    joints: np.ndarray,
    target_fps: float,
    enabled_objects: set[str],
    enabled_parts: tuple[str, ...],
    distance_thresholds_m: dict[str, float],
    num_markers_per_part: int,
    median_filter_size: int,
    min_contact_duration: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return optional contact extra labels from official surface markers.

    The official InterAct representation stores nearest-object vectors for
    77 body surface markers. These are closer to physical contact than the
    coarse 22-joint proxies, especially for hands (wrist offset), feet (ankle
    / mid-foot offset), and sitting/lying poses (pelvis joint offset).
    """
    if official_interact_root is None:
        return None, None, None
    object_key = str(object_id).lower()
    if "*" not in enabled_objects and object_key not in enabled_objects:
        return None, None, None

    T = len(joints)
    seq_dir = _official_sequence_dir(official_interact_root, subset, seq_id)
    marker_contact = _load_official_marker_contact_data(seq_dir, T, target_fps)
    markers = _load_official_markers(seq_dir, T, target_fps)
    if marker_contact is None or markers is None:
        return None, None, None
    marker_distances, marker_targets_local = marker_contact

    extra = np.zeros((T, NUM_BODY_PARTS), dtype=np.float32)
    anchors = joints[:, BODY_PART_INDICES, :].astype(np.float32, copy=True)
    targets_local = np.full((T, NUM_BODY_PARTS, 3), np.nan, dtype=np.float32)
    valid_part_indices: list[int] = []

    for part_name in enabled_parts:
        if part_name not in BODY_PART_NAMES:
            raise ValueError(f"Unknown official marker body part: {part_name}")
        bp_idx = BODY_PART_NAMES.index(part_name)
        marker_idx = _official_marker_indices(
            markers, joints, body_part=part_name, k=num_markers_per_part,
        )
        part_dist = marker_distances[:, marker_idx]
        threshold = distance_thresholds_m[part_name]
        binary = part_dist.min(axis=1) < threshold
        extra[:, bp_idx] = binary.astype(np.float32)
        valid_part_indices.append(bp_idx)

        nearest_local = part_dist.argmin(axis=1)
        nearest_marker_idx = marker_idx[nearest_local]
        anchors[:, bp_idx, :] = markers[np.arange(T), nearest_marker_idx, :]
        targets_local[:, bp_idx, :] = marker_targets_local[np.arange(T), nearest_marker_idx, :]

    if valid_part_indices:
        part_block = extra[:, valid_part_indices]
        extra[:, valid_part_indices] = _postprocess_marker_binary(
            part_block,
            median_size=median_filter_size,
            min_duration=min_contact_duration,
        )
    return extra, anchors, targets_local


def _marker_thresholds(
    *,
    hand_m: float,
    foot_m: float,
    pelvis_m: float,
) -> dict[str, float]:
    thresholds = dict(DEFAULT_OFFICIAL_MARKER_THRESHOLDS_M)
    thresholds["left_hand"] = hand_m
    thresholds["right_hand"] = hand_m
    thresholds["left_foot"] = foot_m
    thresholds["right_foot"] = foot_m
    thresholds["pelvis"] = pelvis_m
    return thresholds


def process_sequence(
    joints: np.ndarray,
    object_mesh: "trimesh.Trimesh | str | Path",
    object_positions: np.ndarray | None = None,
    object_rotations: np.ndarray | None = None,
    patch_centers: np.ndarray | None = None,
    object_id: str | None = None,
    contact_config: ContactConfig | None = None,
    target_config: TargetConfig | None = None,
    phase_config: PhaseConfig | None = None,
    support_config: SupportConfig | None = None,
    hmm_config: HMMConfig | None = None,
    use_hmm_refinement: bool = True,
    contact_version: str = "v11",
    strict_contact_config: StrictContactConfig | None = None,
    contact_state_extra: np.ndarray | None = None,
    contact_anchor_points_world: np.ndarray | None = None,
    contact_state_override: np.ndarray | None = None,
    contact_target_points_local: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the full pseudo-label extraction for one sequence.

    Parameters
    ----------
    joints : (T, 22, 3) — world-frame SMPL 22-joint positions
    object_mesh : a pre-loaded ``trimesh.Trimesh`` (preferred, enables caching
        across sequences with the same object) or a path to load. Mesh
        is in object-local frame.
    object_positions : (T, 3) or None — per-frame object translation in world
    object_rotations : (T, 3) or None — per-frame object axis-angle rotation.
        Needed for geometrically correct contact / target extraction. If
        None, rotation is treated as identity (still uses translation).
    *_config : per-stage configuration (uses defaults if None)
    use_hmm_refinement : whether to refine phase labels with HMM

    Returns
    -------
    Dictionary containing:
        ``contact_state`` : (T, 5)
        ``contact_target`` : (T, 5, K)
        ``patch_centers`` : (K, 3)
        ``phase`` : (T,)
        ``support`` : (T,)
    """
    if isinstance(object_mesh, (str, Path)):
        mesh = load_mesh(str(object_mesh))
    else:
        mesh = object_mesh

    # Step 1: Contact state — inverse-transforms joints to object-local.
    # contact_version selects v11 (legacy "approach within ~12 cm") vs
    # v12_strict (real-contact: tight distance OR engagement, with drift
    # filter). See analyses/2026-05-03_pseudo_label_v12_strict_design.md.
    if contact_state_override is not None:
        if contact_state_override.shape != (len(joints), NUM_BODY_PARTS):
            raise ValueError(
                f"contact_state_override shape {contact_state_override.shape} "
                f"does not match expected {(len(joints), NUM_BODY_PARTS)}"
            )
        contact_state = contact_state_override.astype(np.float32, copy=True)
    elif contact_version == "v11":
        contact_state = extract_contact_state(
            joints, mesh,
            object_positions=object_positions,
            object_rotations=object_rotations,
            config=contact_config,
            object_id=object_id,
        )
    elif contact_version == "v12_strict":
        contact_state = extract_strict_contact_state(
            joints, mesh,
            object_positions=object_positions,
            object_rotations=object_rotations,
            strict_config=strict_contact_config,
            base_kin_config=contact_config,
        )
    else:
        raise ValueError(
            f"contact_version must be 'v11' or 'v12_strict', got {contact_version!r}"
        )

    if contact_state_extra is not None:
        if contact_state_extra.shape != contact_state.shape:
            raise ValueError(
                f"contact_state_extra shape {contact_state_extra.shape} does "
                f"not match contact_state shape {contact_state.shape}"
            )
        contact_state = np.maximum(contact_state, contact_state_extra.astype(np.float32))

    # v19 (2026-05-09): directional gate for pelvis contact, applied after
    # max-combining mesh-distance and official-marker signals. Filters the
    # frequent false positives during bat-swing / lift / carry motions
    # where an object passes within 20 cm of pelvis at the side or above.
    # Reuses the cylinder + upward-facing-normal helper from sitting
    # detection — same per-mesh seat-points cache, no extra surface
    # sampling. Only fires when contact_config.use_directional_pelvis_gate.
    if (
        contact_config is not None
        and contact_config.use_directional_pelvis_gate
        and object_positions is not None
    ):
        from piano.data.pseudo_labels.extract_contact import _PELVIS_BP_IDX
        from piano.data.pseudo_labels.extract_support import (
            SupportConfig as _SupportConfig,
            _pelvis_object_below_mask,
        )
        _gate_support_cfg = _SupportConfig(
            sitting_below_horz_radius=contact_config.pelvis_below_horz_radius,
            sitting_below_vert_gate=contact_config.pelvis_below_vert_gate,
            sitting_below_upward_normal_threshold=
                contact_config.pelvis_below_upward_normal_threshold,
            fps=contact_config.fps,
        )
        T_gate = len(joints)
        below_mask = _pelvis_object_below_mask(
            joints=joints,
            object_mesh=mesh,
            object_positions=object_positions,
            object_rotations=object_rotations,
            config=_gate_support_cfg,
            T=T_gate,
            object_id=object_id,
        ).astype(np.float32)
        contact_state[:, _PELVIS_BP_IDX] = (
            contact_state[:, _PELVIS_BP_IDX] * below_mask
        )

    # Step 2: Contact target. Now returns three arrays:
    # - contact_target_xyz_gt (T, 5, 3): closest-surface-point per body
    #   part in object-local frame — the v3 GT for the predictor's xyz
    #   regression head (replaces the previous softmax-weighted patch
    #   centroid approximation, which had ~5-10 cm bias).
    # - contact_target (T, 5, K): legacy soft K-way distribution, kept
    #   for visualisation + backward compat.
    # - patch_centers (K, 3): per-object FPS atlas (stable across reruns).
    contact_target_xyz_gt, contact_target, patch_centers = extract_contact_target(
        joints, mesh, contact_state,
        object_positions=object_positions,
        object_rotations=object_rotations,
        config=target_config,
        patch_centers=patch_centers,
        anchor_points_world=contact_anchor_points_world,
        target_points_local_override=contact_target_points_local,
    )

    # Step 3: Interaction phase — rotation-aware so that rotation-only
    # manipulations (bat swing, chair rotate) reach manipulation instead
    # of collapsing to stable-contact.
    phase = extract_interaction_phase(
        joints, contact_state, object_positions, object_rotations, phase_config,
    )

    # Optional: HMM refinement
    if use_hmm_refinement:
        features = build_phase_features(
            joints, contact_state, object_positions, object_rotations,
            fps=(phase_config or PhaseConfig()).fps,
        )
        phase = refine_phases_hmm(features, phase, hmm_config)

    # Step 4: Support state. `sitting` has two gates beyond pelvis contact:
    #   (a) joints → pelvis XZ-speed < 0.15 m/s (rejects push/drag)
    #   (b) object_mesh + positions + rotations → geometric "object below
    #       pelvis" test (rejects standing-beside-object where the pelvis
    #       joint is within 20 cm of a backrest/leg but not *above* a seat)
    # `hand_support` has two gates beyond hand contact:
    #   (c) pelvis stationary (rejects carry-while-walking)
    #   (d) phase == stable-contact (rejects carrying / manipulating /
    #       approach / release — hand applying force to object, not the
    #       other way round)
    support = extract_support_state(
        contact_state,
        joints=joints,
        object_mesh=mesh,
        object_positions=object_positions,
        object_rotations=object_rotations,
        object_id=object_id,
        phase=phase,
        config=support_config,
    )

    return {
        "contact_state": contact_state,
        "contact_target": contact_target,
        "contact_target_xyz_gt": contact_target_xyz_gt,
        "patch_centers": patch_centers,
        "phase": phase,
        "support": support,
    }


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    mesh_dir: Path,
    metadata_path: Path | None = None,
    use_hmm: bool = True,
    mesh_suffixes: tuple[str, ...] = ("_cleaned_simplified", ""),
    fps: float | None = None,
    contact_version: str = "v11",
    official_interact_root: Path | None = None,
    official_marker_objects: tuple[str, ...] = DEFAULT_OFFICIAL_MARKER_OBJECTS,
    official_marker_parts: tuple[str, ...] = DEFAULT_OFFICIAL_MARKER_PARTS,
    official_marker_hand_distance_m: float = 0.05,
    official_marker_foot_distance_m: float = 0.05,
    official_marker_pelvis_distance_m: float = 0.20,
    official_marker_k: int = 8,
    target_query_contact_only: bool = False,
    official_marker_contact_only: bool = False,
    use_directional_pelvis_gate: bool = False,
) -> None:
    """Batch pseudo-label extraction for all sequences.

    Expects preprocessed data at *data_dir*::

        data_dir/
            metadata.json            # list of {seq_id, object_id, ...}
            motions/<seq_id>.npz     # contains joints_22, object_positions

    Object meshes live at *mesh_dir* (typically the source dataset's
    captured_objects folder), as ``<obj_id><suffix>.{obj,ply,...}`` files.

    Parameters
    ----------
    data_dir : root of preprocessed (PIANO-format) dataset
    output_dir : where to write pseudo-label npz files
    mesh_dir : directory containing source object meshes
    metadata_path : override metadata.json location
    use_hmm : whether to refine phases with HMM
    mesh_suffixes : suffixes to try appending to object_id when searching
        for the mesh file (OMOMO uses ``_cleaned_simplified``).
    """
    t_start = time.time()
    data_dir = Path(data_dir)
    mesh_dir = Path(mesh_dir)
    output_dir = ensure_dir(output_dir)
    atlas_dir = ensure_dir(output_dir / "patch_atlas")

    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path)

    resolved_fps = _resolve_fps(data_dir, fps)
    contact_cfg = ContactConfig(
        fps=resolved_fps,
        use_directional_pelvis_gate=use_directional_pelvis_gate,
    )
    phase_cfg = PhaseConfig(fps=resolved_fps)
    target_cfg = TargetConfig(query_all_frames=not target_query_contact_only)
    support_cfg = SupportConfig(fps=resolved_fps)
    # v12: use the strict definition (analyses/2026-05-03_pseudo_label_v12_strict_design.md)
    strict_cfg: StrictContactConfig | None = None
    if contact_version == "v12_strict":
        strict_cfg = StrictContactConfig(fps=resolved_fps)
        # v12 uses a wider kin_local_sigma than v11 default — pass via
        # contact_config so _kinematic_contact_score sees the loosened σ.
        contact_cfg = ContactConfig(
            kin_local_sigma=0.06,
            kin_local_transition=0.025,
            kin_world_eps=0.15,
            kin_world_sigma=0.04,
            kin_radius_proxy=0.3,
            kin_window_sec=0.5,
            fps=resolved_fps,
            use_directional_pelvis_gate=use_directional_pelvis_gate,
        )
    official_marker_object_set = {obj.lower() for obj in official_marker_objects}
    official_marker_thresholds = _marker_thresholds(
        hand_m=official_marker_hand_distance_m,
        foot_m=official_marker_foot_distance_m,
        pelvis_m=official_marker_pelvis_distance_m,
    )

    print(f"Extracting pseudo-labels for {len(metadata)} sequences")
    print(f"  Data:   {data_dir}")
    print(f"  Meshes: {mesh_dir}")
    print(f"  Output: {output_dir}")
    print(f"  FPS:    {resolved_fps}  (used for velocity thresholds)")
    print(f"  Contact version: {contact_version}")
    if official_interact_root is not None:
        print(
            "  Official marker contact prior: "
            f"{official_interact_root} objects={sorted(official_marker_object_set)} "
            f"parts={list(official_marker_parts)} thresholds={official_marker_thresholds}"
        )
    if official_marker_contact_only:
        print("  Contact source: official surface markers only")

    # Cache LOADED meshes (not just paths) so each object is loaded once
    # and its trimesh spatial index is reused across all sequences using
    # that object — critical for speed/memory on datasets with large meshes.
    import trimesh
    mesh_cache: dict[str, trimesh.Trimesh | None] = {}
    # Per-object deterministic patch atlas, cached on disk so re-runs and
    # separate machines produce identical patch ids.
    atlas_cache: dict[str, np.ndarray | None] = {}

    n_ok = 0
    n_skip = 0
    n_resume = 0
    n_marker_prior = 0
    skip_reasons: list[dict[str, str]] = []
    per_seq_stats = []
    for entry in tqdm(metadata, desc="Pseudo-labels"):
        seq_id = entry["seq_id"]
        obj_id = entry["object_id"]

        # Resume support: skip sequences we've already written
        out_path = output_dir / f"{seq_id}.npz"
        if out_path.exists():
            n_resume += 1
            continue

        # Load preprocessed motion
        motion_path = data_dir / "motions" / f"{seq_id}.npz"
        if not motion_path.exists():
            n_skip += 1
            skip_reasons.append({"seq_id": seq_id, "reason": "motion_file_missing"})
            continue
        motion_data = np.load(motion_path, allow_pickle=False)
        joints = motion_data["joints_22"]  # (T, 22, 3)

        # Lazily load and cache the mesh for this object_id
        if obj_id not in mesh_cache:
            mesh_path = _find_mesh(mesh_dir, obj_id, mesh_suffixes)
            if mesh_path is None:
                mesh_cache[obj_id] = None
            else:
                try:
                    mesh_cache[obj_id] = load_mesh(str(mesh_path))
                except Exception as e:
                    print(f"  [warn] failed to load mesh {mesh_path}: {e}")
                    mesh_cache[obj_id] = None
        mesh = mesh_cache[obj_id]
        if mesh is None:
            n_skip += 1
            skip_reasons.append({
                "seq_id": seq_id,
                "reason": f"mesh_not_found_or_failed_to_load (object_id={obj_id})",
            })
            continue

        # Per-object deterministic patch atlas (shared across all sequences
        # of this object). Disk-cached so re-runs stay consistent.
        if obj_id not in atlas_cache:
            atlas_path = atlas_dir / f"{obj_id}.npy"
            if atlas_path.exists():
                atlas_cache[obj_id] = np.load(atlas_path)
            else:
                atlas = cluster_surface_patches(
                    mesh,
                    num_patches=target_cfg.num_patches,
                    num_surface_samples=target_cfg.num_surface_samples,
                    seed=_object_patch_seed(obj_id),
                )
                np.save(atlas_path, atlas)
                atlas_cache[obj_id] = atlas
        patch_centers = atlas_cache[obj_id]

        # Object pose from preprocessing. object_rotations is only present
        # for data preprocessed with the updated preprocess_interact that
        # saves rotation. Older data will fall back to translation-only.
        files = set(motion_data.files)
        object_positions = motion_data["object_positions"] if "object_positions" in files else None
        object_rotations = motion_data["object_rotations"] if "object_rotations" in files else None
        contact_extra = None
        contact_anchors = None
        contact_targets_local = None
        try:
            contact_extra, contact_anchors, contact_targets_local = _official_marker_contact_prior(
                official_interact_root=official_interact_root,
                subset=data_dir.name,
                seq_id=seq_id,
                object_id=obj_id,
                joints=joints,
                target_fps=resolved_fps,
                enabled_objects=official_marker_object_set,
                enabled_parts=official_marker_parts,
                distance_thresholds_m=official_marker_thresholds,
                num_markers_per_part=official_marker_k,
                median_filter_size=(strict_cfg or StrictContactConfig(fps=resolved_fps)).median_filter_size,
                min_contact_duration=(strict_cfg or StrictContactConfig(fps=resolved_fps)).min_contact_duration,
            )
            if contact_extra is not None:
                n_marker_prior += 1
        except Exception as e:
            print(f"  [warn] official marker contact prior failed for {seq_id}: {e}")
        if official_marker_contact_only and contact_extra is None:
            n_skip += 1
            skip_reasons.append({
                "seq_id": seq_id,
                "reason": "official_marker_contact_unavailable",
            })
            continue

        try:
            labels = process_sequence(
                joints=joints,
                object_mesh=mesh,
                object_positions=object_positions,
                object_rotations=object_rotations,
                patch_centers=patch_centers,
                object_id=obj_id,
                contact_config=contact_cfg,
                target_config=target_cfg,
                phase_config=phase_cfg,
                support_config=support_cfg,
                use_hmm_refinement=use_hmm,
                contact_version=contact_version,
                strict_contact_config=strict_cfg,
                contact_state_extra=None if official_marker_contact_only else contact_extra,
                contact_anchor_points_world=contact_anchors,
                contact_state_override=contact_extra if official_marker_contact_only else None,
                contact_target_points_local=contact_targets_local,
            )
        except Exception as e:
            print(f"  [warn] {seq_id}: {e}")
            n_skip += 1
            skip_reasons.append({"seq_id": seq_id, "reason": f"exception: {e}"})
            continue

        save_npz(out_path, **labels)
        n_ok += 1

        # Accumulate quality stats for the summary. Uses the just-computed
        # labels in memory — no extra disk I/O and one set of traversals
        # per sequence (cheap compared to mesh distance queries).
        try:
            per_seq_stats.append(
                compute_seq_stats(
                    seq_id=seq_id,
                    labels=labels,
                    joints_22=joints,
                    object_positions=object_positions,
                )
            )
        except Exception as e:
            print(f"  [warn] stats failed for {seq_id}: {e}")

    elapsed = time.time() - t_start
    print(f"Done. {n_ok} labels written, {n_resume} resumed (already existed), "
          f"{n_skip} skipped. Output: {output_dir}")
    print(f"Elapsed: {elapsed:.1f}s  ({n_ok / max(elapsed, 1e-6):.1f} seq/s)")

    # --- Aggregate quality stats + derive readable flags ---
    subset_hint = data_dir.name or None
    stats_agg = aggregate_stats(per_seq_stats, num_patches=target_cfg.num_patches)
    quality_flags = make_quality_flags(stats_agg, subset_hint=subset_hint)

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "mesh_dir": str(mesh_dir),
        "output_dir": str(output_dir),
        "subset": subset_hint,
        "fps": resolved_fps,
        "use_hmm": use_hmm,
        "contact_version": contact_version,
        "mesh_suffixes": list(mesh_suffixes),
        "official_marker_contact_prior": {
            "official_interact_root": str(official_interact_root) if official_interact_root else None,
            "enabled_objects": sorted(official_marker_object_set),
            "enabled_parts": list(official_marker_parts),
            "distance_thresholds_m": official_marker_thresholds,
            "num_markers_per_part": official_marker_k,
            "num_sequences_augmented": n_marker_prior,
        },
        "target_query_contact_only": target_query_contact_only,
        "official_marker_contact_only": official_marker_contact_only,
        "num_objects_with_atlas": len([k for k, v in atlas_cache.items() if v is not None]),
        "counts": {
            "num_in_metadata": len(metadata),
            "num_labels_written": n_ok,
            "num_resumed": n_resume,
            "num_skipped": n_skip,
            "num_official_marker_contact_prior": n_marker_prior,
        },
        "elapsed_sec": round(elapsed, 2),
        "throughput_seq_per_sec": round(n_ok / max(elapsed, 1e-6), 2),
        "skip_reasons": skip_reasons,
        "quality_flags": quality_flags,
        "stats": stats_agg,
    }
    save_json(output_dir / "summary.json", summary)

    if quality_flags:
        print(f"\nQuality flags ({len(quality_flags)}):")
        for f in quality_flags:
            print(f"  - {f}")
    else:
        print("\nQuality flags: none fired")


def _find_mesh(
    mesh_dir: Path,
    obj_id: str,
    suffixes: tuple[str, ...],
) -> Path | None:
    """Look up an object mesh file by id + suffix.

    Tries two layouts to support different upstream conventions:
        - Flat:    ``mesh_dir/<obj_id><suffix>.<ext>``         (OMOMO/CHOIS)
        - Nested:  ``mesh_dir/<obj_id>/<obj_id><suffix>.<ext>`` (InterAct)
    """
    extensions = (".obj", ".ply", ".stl", ".off")
    candidate_dirs = (mesh_dir, mesh_dir / obj_id)
    suffixes = tuple(dict.fromkeys((*suffixes, "")))
    for d in candidate_dirs:
        for suffix in suffixes:
            for ext in extensions:
                path = d / f"{obj_id}{suffix}{ext}"
                if path.exists():
                    return path
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract pseudo interaction labels from preprocessed HOI data",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of preprocessed PIANO dataset (contains motions/, metadata.json)",
    )
    parser.add_argument(
        "--mesh-dir", type=Path, required=True,
        help="Directory containing source object meshes (e.g. OMOMO captured_objects/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write pseudo-label npz files",
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="Override metadata file path (default: <data-dir>/metadata.json)",
    )
    parser.add_argument(
        "--mesh-suffixes", nargs="+", default=["_cleaned_simplified", ""],
        help="Filename suffixes to try in order when searching for the mesh. "
             "Empty string = bare filename. Default favors simplified variants: "
             "('_cleaned_simplified', '') for OMOMO. For InterAct subsets that "
             "ship simplified variants, pass '_face1000 _simplified \"\"'.",
    )
    parser.add_argument(
        "--no-hmm", action="store_true",
        help="Skip HMM refinement for phase labels",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override fps used for velocity thresholds. Default: read "
             "target_fps from <data-dir>/summary.json, else 20.",
    )
    parser.add_argument(
        "--contact-version",
        choices=["v11", "v12_strict"],
        default="v11",
        help="Contact-state extraction definition. v11 (default) is the "
             "legacy 'distance-OR-kinematic' formulation with anatomy-"
             "calibrated thresholds (hand 12 cm, foot 6 cm, pelvis 20 cm). "
             "v12_strict is the 2026-05-03 'real-contact' definition: "
             "loose-distance × engagement (kinematic OR static), with "
             "drift filter. See "
             "analyses/2026-05-03_pseudo_label_v12_strict_design.md for "
             "the design and rationale.",
    )
    parser.add_argument(
        "--official-interact-root",
        type=Path,
        default=None,
        help="Optional official InterAct root. When provided, hand contact "
             "for configured handheld objects can be augmented from official "
             "sequences_canonical/<seq_id>/motion.npy marker-object distances.",
    )
    parser.add_argument(
        "--official-marker-hand-objects",
        nargs="+",
        dest="official_marker_objects",
        default=list(DEFAULT_OFFICIAL_MARKER_OBJECTS),
        help="Deprecated alias for --official-marker-objects.",
    )
    parser.add_argument(
        "--official-marker-objects",
        nargs="+",
        dest="official_marker_objects",
        default=None,
        help="Object ids allowed to use the official marker contact prior. "
             "Use '*' for all objects. Default: '*'.",
    )
    parser.add_argument(
        "--official-marker-parts",
        nargs="+",
        default=list(DEFAULT_OFFICIAL_MARKER_PARTS),
        choices=list(DEFAULT_OFFICIAL_MARKER_PARTS),
        help="Body parts to augment from official surface markers.",
    )
    parser.add_argument(
        "--official-marker-hand-distance-m",
        type=float,
        default=0.05,
        help="Surface-marker to object distance threshold for official hand "
             "contact. Default: 5 cm.",
    )
    parser.add_argument(
        "--official-marker-foot-distance-m",
        type=float,
        default=0.05,
        help="Surface-marker to object distance threshold for official foot "
             "contact. Default: 5 cm.",
    )
    parser.add_argument(
        "--official-marker-pelvis-distance-m",
        type=float,
        default=0.20,
        help="Surface-marker to object distance threshold for official pelvis "
             "contact. Default: 20 cm.",
    )
    parser.add_argument(
        "--official-marker-hand-k",
        dest="official_marker_k",
        type=int,
        default=8,
        help="Deprecated alias for --official-marker-k.",
    )
    parser.add_argument(
        "--official-marker-k",
        dest="official_marker_k",
        type=int,
        default=None,
        help="Number of nearest official surface markers to use per body part. "
             "Default: 8.",
    )
    parser.add_argument(
        "--target-query-contact-only",
        action="store_true",
        help="Only compute contact_target_xyz_gt closest mesh points on "
             "frame/body-part cells with contact_state >= threshold. Current "
             "Stage A/B losses gate target supervision by contact; leaving "
             "this off preserves dense legacy labels.",
    )
    parser.add_argument(
        "--official-marker-contact-only",
        action="store_true",
        help="Use official InterAct surface-marker distances as the contact "
             "state instead of running the joint-to-mesh contact extractor. "
             "Requires --official-interact-root and marker data for every "
             "sequence. The official nearest object point is also used as "
             "contact_target_xyz_gt where available.",
    )
    parser.add_argument(
        "--use-directional-pelvis-gate",
        action="store_true",
        help="v19: gate pelvis contact by 'object below pelvis in cylinder + "
             "upward-facing normal' check (mirroring sitting detector). "
             "Eliminates false-positive pelvis contacts during dynamic "
             "motions like bat-swing where an object passes 15-20cm to the "
             "side of the pelvis. Off by default (v18 behavior).",
    )
    return parser


def main() -> None:
    """CLI entrypoint for ``piano-pseudo-labels``."""
    args = build_parser().parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mesh_dir=args.mesh_dir,
        metadata_path=args.metadata,
        use_hmm=not args.no_hmm,
        mesh_suffixes=tuple(args.mesh_suffixes),
        fps=args.fps,
        contact_version=args.contact_version,
        official_interact_root=args.official_interact_root,
        official_marker_objects=tuple(args.official_marker_objects or DEFAULT_OFFICIAL_MARKER_OBJECTS),
        official_marker_parts=tuple(args.official_marker_parts),
        official_marker_hand_distance_m=args.official_marker_hand_distance_m,
        official_marker_foot_distance_m=args.official_marker_foot_distance_m,
        official_marker_pelvis_distance_m=args.official_marker_pelvis_distance_m,
        official_marker_k=args.official_marker_k or 8,
        target_query_contact_only=args.target_query_contact_only,
        official_marker_contact_only=args.official_marker_contact_only,
        use_directional_pelvis_gate=args.use_directional_pelvis_gate,
    )


if __name__ == "__main__":
    main()
