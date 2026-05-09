"""Offline keyframe extraction for AnchorDiff v8 hierarchical generation.

For each clip, computes:
  - keyframe_indices: K frames where the body should be pinned (start,
    end, contact_state changes, pose-change-rate peaks). 5 <= K <= 12.
  - keyframe_targets: (K, 6, 3) world XYZ for 6 key joints
    (root, L_hand, R_hand, L_foot, R_foot, head).

Output: ``<data_dir>/keyframes/<v18_subdir>/<seq_id>.npz`` with fields
``indices``, ``targets``, ``num_keyframes`` (effective K, ≤ K_MAX).

Selection logic (v8 design §3):
  1. Always include frame 0 and last valid frame.
  2. Add frames where any of the 5 body-part contact_state crosses
     0.5 threshold.
  3. Add pose-change-rate peaks: ``find_peaks(||diff(joints)||,
     distance=15)`` after a small smoothing window.
  4. If total < K_MIN: pad with uniform filler frames.
  5. If total > K_MAX: keep top by pose-diff magnitude (forced
     start/end have score = +inf).

Inference-time selection differs slightly: pose-peak frames are
unavailable (no GT joints), replaced by uniform filler. See dataset
loader for the inference-side path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from piano.data.contact_postprocess import suppress_sitting_hand_contact


# Body parts: SMPL-22 indices for the 6 keyjoints we track in v8.
# Order: root, L_hand, R_hand, L_foot, R_foot, head.
KEYJOINT_INDICES: tuple[int, ...] = (0, 20, 21, 10, 11, 15)
NUM_KEYJOINTS: int = len(KEYJOINT_INDICES)

K_MIN: int = 5
K_MAX: int = 12

CONTACT_THRESHOLD: float = 0.5
POSE_PEAK_MIN_DISTANCE: int = 15
POSE_DIFF_SMOOTH_WIN: int = 5

# Rule A: contact-state stability gate window (frames). Transitions
# are detected on the rolling-mean smoothed contact_state instead of
# the raw value, so single-frame flickers (and short repeated
# raise/lower cycles) don't generate keyframes.
CONTACT_STABILITY_WINDOW: int = 15
CONTACT_STABILITY_THRESHOLD: float = 0.7


def select_keyframes(
    joints_22: np.ndarray,                  # (T, 22, 3) world XYZ
    contact_state: np.ndarray,              # (T, 5) soft contact prob
    seq_len: int,                           # actual valid frames
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices: (K,) int, targets: (K, 6, 3) float32).

    K is between K_MIN and K_MAX, varying per clip.
    """
    T_eff = int(seq_len)
    if T_eff < 2:
        # Degenerate: just frame 0
        idx = np.array([0], dtype=np.int64)
        targets = joints_22[idx][:, list(KEYJOINT_INDICES), :].astype(np.float32)
        return idx, targets

    last_frame = T_eff - 1

    # Step 1: contact-change frames (rule A — stability gate).
    # Use rolling-mean smoothed contact_state (window=15 frames) and
    # require >70% in-window to count as "stable contact". This filters
    # out single-frame flickers and short repeated raise/lower cycles
    # that would otherwise produce many spurious keyframe boundaries.
    boundary_set: set[int] = set()
    if T_eff >= 2:
        for bp_idx in range(contact_state.shape[1]):
            cs_raw = (contact_state[:T_eff, bp_idx] >= CONTACT_THRESHOLD).astype(np.float32)
            cs_smooth = uniform_filter1d(
                cs_raw, size=CONTACT_STABILITY_WINDOW, mode="nearest",
            )
            cs_stable = (cs_smooth >= CONTACT_STABILITY_THRESHOLD).astype(np.int8)
            diff = np.diff(cs_stable)
            for t in np.nonzero(diff != 0)[0]:
                boundary_set.add(int(t + 1))   # change happens at frame t+1

    # Step 2: pose-change-rate peaks
    diffs = np.linalg.norm(
        joints_22[1:T_eff] - joints_22[: T_eff - 1],
        axis=-1,
    ).sum(axis=-1)                                  # (T_eff-1,)
    smoothed = uniform_filter1d(diffs, size=POSE_DIFF_SMOOTH_WIN, mode="nearest")
    peaks, _ = find_peaks(smoothed, distance=POSE_PEAK_MIN_DISTANCE)
    peak_set: set[int] = {int(p + 1) for p in peaks}

    # Step 3: union + force start/end
    chosen: set[int] = {0, last_frame} | boundary_set | peak_set

    # Step 4: pad with uniform fillers if too few
    if len(chosen) < K_MIN:
        needed = K_MIN - len(chosen)
        fillers = np.linspace(0, last_frame, needed + 2, dtype=int)[1:-1]
        for f in fillers:
            chosen.add(int(f))

    # Step 5: drop down to K_MAX if too many
    chosen_sorted = sorted(chosen)
    if len(chosen_sorted) > K_MAX:
        # Score: forced start/end get +inf, peaks get smoothed[i-1],
        # contact-changes get bigger of (smoothed[i-1], median).
        forced = {0, last_frame}
        scores: list[float] = []
        for i in chosen_sorted:
            if i in forced:
                scores.append(float("inf"))
            elif 0 < i <= len(smoothed):
                scores.append(float(smoothed[i - 1]))
            else:
                scores.append(0.0)
        # Keep top K_MAX
        order = np.argsort(scores)[::-1][:K_MAX]
        chosen_sorted = sorted(chosen_sorted[i] for i in order)

    indices = np.array(chosen_sorted, dtype=np.int64)
    targets = joints_22[indices][:, list(KEYJOINT_INDICES), :].astype(np.float32)
    return indices, targets


def extract_for_subset(
    data_dir: Path,
    pseudo_label_subdir: str,
    output_subdir: str = "keyframes/v8_default",
    overwrite: bool = False,
) -> dict[str, int]:
    """Run keyframe selection across all clips in a subset.

    Saves per-clip ``keyframes.npz`` next to motion files.

    Returns counts: ``{ok, skip, error}``.
    """
    metadata_path = data_dir / "metadata_clean.json"
    if not metadata_path.exists():
        metadata_path = data_dir / "metadata.json"
    import json
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    out_dir = data_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = data_dir / pseudo_label_subdir

    counts = {"ok": 0, "skip_no_motion": 0, "skip_no_label": 0, "error": 0}
    for entry in metadata:
        seq_id = entry["seq_id"]
        out_path = out_dir / f"{seq_id}.npz"
        if out_path.exists() and not overwrite:
            counts["ok"] += 1
            continue

        motion_path = data_dir / "motions" / f"{seq_id}.npz"
        if not motion_path.exists():
            counts["skip_no_motion"] += 1
            continue

        label_path = label_dir / f"{seq_id}.npz"
        if not label_path.exists():
            counts["skip_no_label"] += 1
            continue

        try:
            md = np.load(motion_path, allow_pickle=False)
            joints = md["joints_22"]                     # (T, 22, 3)
            ld = np.load(label_path, allow_pickle=False)
            contact_state = ld["contact_state"]          # (T, 5)
            support = ld["support"] if "support" in ld.files else None
            seq_len = min(len(joints), len(contact_state))

            # Rule C: zero hand contact when support==sitting + pelvis
            # has stable contact. Applied BEFORE keyframe selection so
            # rule A's stability gate sees the corrected contact_state.
            contact_state = suppress_sitting_hand_contact(
                contact_state[:seq_len], support[:seq_len] if support is not None else None,
            )

            indices, targets = select_keyframes(
                joints, contact_state, seq_len=seq_len,
            )
            np.savez_compressed(
                out_path,
                indices=indices.astype(np.int32),
                targets=targets.astype(np.float32),
                num_keyframes=np.int32(len(indices)),
            )
            counts["ok"] += 1
        except Exception as exc:                         # noqa: BLE001
            print(f"  [error] {seq_id}: {exc!s}")
            counts["error"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v8 offline keyframe extraction"
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="PIANO subset root (chairs/imhd/...)")
    parser.add_argument(
        "--pseudo-label-subdir",
        type=str,
        default="pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker",
        help="subdir relative to data_dir containing contact_state per clip",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="keyframes/v8_default",
        help="subdir relative to data_dir to write keyframes/<seq>.npz",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    counts = extract_for_subset(
        data_dir=args.data_dir,
        pseudo_label_subdir=args.pseudo_label_subdir,
        output_subdir=args.output_subdir,
        overwrite=args.overwrite,
    )
    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
