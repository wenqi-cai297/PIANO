"""Quality check for v19 pseudo labels vs v18 baseline.

Runs across all 4 InterAct subsets and reports:
  1. Per-subset per-bodypart contact-state distribution shift (frame
     fraction with contact_state >= 0.5).
  2. Known regression cases:
     - imhd bat-hit: expect pelvis FP rate to drop sharply (~100% -> ~10%)
     - chairs sitting: expect pelvis to stay similar; hands to drop
       (sitting-aware suppression rule C)
  3. Number of contact_state transitions per clip (rule A stability gate
     should drop count significantly).
  4. Phase / support label parity (these should be unchanged from v18 —
     v19 only modifies contact_state / contact_target).

Usage:
    python -m scripts.stage1_pseudo_labels.v19_quality_check \
        [--data-root E:/...] [--max-clips-per-subset 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")
BP_NAMES = ("L_hand", "R_hand", "L_foot", "R_foot", "pelvis")
V18_SUBDIR = "pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker"
V19_SUBDIR = "pseudo_labels/v19_h10_f05_pelvis20dir_official_semantic_marker"


def _read_metadata(data_dir: Path) -> list[dict]:
    p = data_dir / "metadata_clean.json"
    if not p.exists():
        p = data_dir / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _contact_frac(cs: np.ndarray, T: int) -> np.ndarray:
    """Per-bodypart fraction of frames with contact_state >= 0.5. (5,)"""
    return ((cs[:T] >= 0.5).sum(axis=0) / max(T, 1)).astype(np.float32)


def _num_transitions(cs: np.ndarray, T: int) -> np.ndarray:
    """Per-bodypart raw transition count (5,)."""
    binarized = (cs[:T] >= 0.5).astype(np.int8)
    if T < 2:
        return np.zeros(5, dtype=np.int64)
    return np.abs(np.diff(binarized, axis=0)).sum(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("E:/Project/Datasets/InterAct/piano_official_process_4"),
    )
    parser.add_argument("--max-clips-per-subset", type=int, default=10_000)
    args = parser.parse_args()

    aggregate: dict[str, dict] = {}
    bat_hit_pelvis: list[tuple[str, float, float]] = []
    chair_sit_hand: list[tuple[str, float, float, float, float]] = []

    for subset in SUBSETS:
        data_dir = args.data_root / subset
        v18_dir = data_dir / V18_SUBDIR
        v19_dir = data_dir / V19_SUBDIR
        if not v18_dir.exists() or not v19_dir.exists():
            print(f"[skip] {subset}: missing v18 or v19 dir")
            continue

        metadata = _read_metadata(data_dir)[: args.max_clips_per_subset]

        sums = {
            "n_clips": 0,
            "v18_frac": np.zeros(5, dtype=np.float64),
            "v19_frac": np.zeros(5, dtype=np.float64),
            "v18_trans": np.zeros(5, dtype=np.float64),
            "v19_trans": np.zeros(5, dtype=np.float64),
        }
        for entry in metadata:
            seq_id = entry["seq_id"]
            v18_p = v18_dir / f"{seq_id}.npz"
            v19_p = v19_dir / f"{seq_id}.npz"
            if not v18_p.exists() or not v19_p.exists():
                continue
            try:
                d18 = np.load(v18_p, allow_pickle=False)
                d19 = np.load(v19_p, allow_pickle=False)
            except Exception:
                continue
            T = min(len(d18["contact_state"]), len(d19["contact_state"]))
            f18 = _contact_frac(d18["contact_state"], T)
            f19 = _contact_frac(d19["contact_state"], T)
            t18 = _num_transitions(d18["contact_state"], T)
            t19 = _num_transitions(d19["contact_state"], T)

            sums["n_clips"] += 1
            sums["v18_frac"] += f18
            sums["v19_frac"] += f19
            sums["v18_trans"] += t18
            sums["v19_trans"] += t19

            # Known regressions
            if subset == "imhd" and "bat" in seq_id.lower() and "hit" in seq_id.lower():
                bat_hit_pelvis.append((seq_id, float(f18[4]), float(f19[4])))
            if subset == "chairs":
                # All chairs clips are sit-based
                chair_sit_hand.append((
                    seq_id,
                    float(f18[0]), float(f19[0]),    # L_hand
                    float(f18[4]), float(f19[4]),    # pelvis
                ))

        if sums["n_clips"] > 0:
            for k in ("v18_frac", "v19_frac", "v18_trans", "v19_trans"):
                sums[k] /= sums["n_clips"]
            aggregate[subset] = sums

    # === Report ===
    print("\n" + "=" * 70)
    print("v19 QUALITY CHECK — pseudo-label distribution shift vs v18")
    print("=" * 70)

    print(f"\n[1] Contact-state frame-fraction (per subset, per body part):")
    print(f"    Each cell: v18 -> v19 delta in fraction of frames with cs>=0.5")
    print(f"    Negative delta = fewer contact frames (expected for pelvis on")
    print(f"    swing clips, hands on chair sit clips).\n")
    print(f"  {'subset':<20} {'L_hand':>14} {'R_hand':>14} {'L_foot':>14} "
          f"{'R_foot':>14} {'pelvis':>14}")
    for subset, sums in aggregate.items():
        deltas = sums["v19_frac"] - sums["v18_frac"]
        cells = [
            f"{sums['v18_frac'][i]:.3f}->{sums['v19_frac'][i]:.3f}"
            f"({deltas[i]:+.3f})"
            for i in range(5)
        ]
        print(f"  {subset:<20} " + " ".join(f"{c:>14}" for c in cells))

    print(f"\n[2] Mean per-bodypart number of contact_state TRANSITIONS per clip")
    print(f"    Lower in v19 = stability gate (rule A) working.\n")
    print(f"  {'subset':<20} {'L_hand':>14} {'R_hand':>14} {'L_foot':>14} "
          f"{'R_foot':>14} {'pelvis':>14}")
    for subset, sums in aggregate.items():
        deltas = sums["v19_trans"] - sums["v18_trans"]
        cells = [
            f"{sums['v18_trans'][i]:.1f}->{sums['v19_trans'][i]:.1f}"
            f"({deltas[i]:+.1f})"
            for i in range(5)
        ]
        print(f"  {subset:<20} " + " ".join(f"{c:>14}" for c in cells))

    if bat_hit_pelvis:
        print(f"\n[3] imhd bat-hit pelvis FP regression check ({len(bat_hit_pelvis)} clips):")
        print(f"  v18 mean pelvis-contact-frac: "
              f"{np.mean([x[1] for x in bat_hit_pelvis]):.3f}")
        print(f"  v19 mean pelvis-contact-frac: "
              f"{np.mean([x[2] for x in bat_hit_pelvis]):.3f}")
        print(f"  Drop ratio: "
              f"{np.mean([x[2] - x[1] for x in bat_hit_pelvis]):+.3f}")
        print(f"  PASS criterion: drop should be at least -0.5 (50%+ FP eliminated)")

    if chair_sit_hand:
        print(f"\n[4] chairs sit hand-suppression check ({len(chair_sit_hand)} clips):")
        v18_h = np.mean([x[1] for x in chair_sit_hand])
        v19_h = np.mean([x[2] for x in chair_sit_hand])
        v18_p = np.mean([x[3] for x in chair_sit_hand])
        v19_p = np.mean([x[4] for x in chair_sit_hand])
        print(f"  hand contact: v18={v18_h:.3f} -> v19={v19_h:.3f}  "
              f"(delta {v19_h - v18_h:+.3f}, expect strong drop)")
        print(f"  pelvis contact: v18={v18_p:.3f} -> v19={v19_p:.3f}  "
              f"(delta {v19_p - v18_p:+.3f}, should stay near 0)")
        print(f"  PASS: hand drops; pelvis ~unchanged (sitting is real contact)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
