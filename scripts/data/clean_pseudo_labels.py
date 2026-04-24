"""Filter low-quality sequences out of a pseudo-label extraction.

After a full extraction run, a small fraction of sequences carry labels
that are either unusable (zero contact everywhere, no interaction phase)
or clearly contradict the text description (text says "sits" but the
pseudo-label shows 0% sitting; text is garbled). Training on those is
worse than training on fewer sequences: the predictor learns either
noise or systematic contradictions. This tool emits a filtered
``metadata_clean.json`` per subset that training code can point at
instead of the raw ``metadata.json``.

Each gate is documented inline with its rationale. Gates are
conservative — they only drop sequences where the pseudo-label is
clearly unusable or contradicts the text. Borderline noisy sequences
(e.g. brief flicker, partial sit) stay; weak supervision tolerates
that class of noise.

Usage:
    python scripts/data/clean_pseudo_labels.py \\
        --data-dir /media/.../InterAct/piano \\
        --subsets chairs imhd neuraldome omomo_correct_v2

Output (per subset):
    <subset>/metadata_clean.json       — filtered list of seq entries
    <subset>/cleaning_report.json      — per-reason counts + sample of
                                         dropped seqs with text + reasons
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from piano.data.pseudo_labels.extract_phase import (
    PHASE_MANIPULATION,
    PHASE_STABLE_CONTACT,
)
from piano.data.pseudo_labels.extract_support import SUPPORT_SITTING
from piano.utils.io_utils import load_json, save_json


# ----- Text keyword patterns -------------------------------------------------
# Word-boundary regex so "move" doesn't match "movement" inside another word,
# and "sit" doesn't match "situated".

SIT_PATTERN = re.compile(
    r"\b(sit|sits|sitting|sat|seated|"
    r"lay\sdown|laid\sdown|lying|lies|"
    r"reclin(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)

ACTION_PATTERN = re.compile(
    r"\b(lift|lifts|lifted|lifting|"
    r"push|pushes|pushed|pushing|"
    r"pull|pulls|pulled|pulling|"
    r"swing|swings|swung|swinging|"
    r"move|moves|moved|moving|"
    r"carry|carries|carried|carrying|"
    r"throw|throws|threw|thrown|throwing|"
    r"pick|picks|picked|picking|"
    r"hit|hits|hitting|"
    r"grip|grips|gripped|gripping|"
    r"hold|holds|held|holding|"
    r"tap|taps|tapped|tapping|"
    r"drag|drags|dragged|dragging|"
    r"rotate|rotates|rotated|rotating|"
    r"turn|turns|turned|turning|"
    r"wave|waves|waved|waving|"
    r"place|places|placed|placing|"
    r"grab|grabs|grabbed|grabbing|"
    r"put\sdown|set\sdown|set\sback\sdown)\b",
    re.IGNORECASE,
)

# Fragments known to appear in obviously-corrupted text fields. Kept
# narrow so we only flag clear cases; do not try to detect prose quality
# in general.
GARBLED_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"\bchat\s*bot\b",
        r"\bprogram\s+that\s+can\s+talk\b",
        r"\bcustomer\s+service\b",
        r"\becommerce\b",
        r"\be-commerce\b",
        r"\blorem\s+ipsum\b",
    )
]


# ----- Per-sequence gate -----------------------------------------------------


def classify_sequence(
    seq_id: str,
    meta_entry: dict,
    labels: dict,
    seq_motion: dict | None = None,
    geometric_outlier_m: float = 2.0,
    sit_min_frac: float = 0.03,
    manip_min_frac: float = 0.02,
    stable_min_frac: float = 0.01,
) -> tuple[bool, list[str]]:
    """Decide whether a sequence should be kept.

    Returns ``(keep, reasons_if_dropped)``. ``reasons_if_dropped`` may
    list multiple gates that the sequence failed; all are reported so
    the cleaning report can show why each seq was removed.
    """
    reasons: list[str] = []
    text = (meta_entry.get("text") or "").strip()

    contact = labels["contact_state"]          # (T, 5) soft
    phase = labels["phase"].astype(np.int64)   # (T,)
    support = labels["support"].astype(np.int64)

    T = len(contact)
    if T == 0:
        return False, ["empty_sequence"]

    contact_binary = contact > 0.5              # (T, 5)
    any_contact_ever = bool(contact_binary.any())
    sitting_frac = float((support == SUPPORT_SITTING).mean())
    manip_frac = float((phase == PHASE_MANIPULATION).mean())
    stable_frac = float((phase == PHASE_STABLE_CONTACT).mean())
    any_hand_contact = bool(contact_binary[:, :2].any())

    # Gate 1: no tracked body part ever crossed the contact threshold.
    # For a HOI sequence this is definitionally useless — nothing to
    # learn. Catches the v7 "dead clips" class (bat_holdhead_hit_0_1501,
    # subject06_baseball_907 etc) if they're still dead after v8.
    if not any_contact_ever:
        reasons.append("zero_contact_all_parts")

    # Gate 2: phase never reached a meaningful interaction state. Pure
    # approach / release / pre-contact across the whole seq means either
    # contact was too fleeting to register or the clip has no real
    # interaction content. Under v8 smoothing this should be rare.
    if manip_frac < stable_min_frac and stable_frac < stable_min_frac:
        reasons.append("no_interaction_phase")

    # Gate 3: text is clearly corrupt (chatbot prompt pasted in etc).
    # Narrow patterns only — we do not try to detect "bad prose".
    if any(p.search(text) for p in GARBLED_PATTERNS):
        reasons.append("garbled_text")

    # Gate 4: text unambiguously says the person sits (or equivalent
    # pelvis-supported pose) but the sitting support label barely fires.
    # The pseudo-label is wrong for training purposes — either the mesh
    # geometry is off (known Obj141 case, some bigsofa clips) or the
    # below-gate filters out the real seat surface. Either way, we'd
    # rather drop the seq than train "sits" → both_feet.
    if SIT_PATTERN.search(text) and sitting_frac < sit_min_frac:
        reasons.append("text_says_sit_but_no_sitting")

    # Gate 5: text describes active object manipulation but the labels
    # show neither manipulation phase nor any hand contact. Dead
    # hand-held-object clips land here (bat clips that never fire
    # contact, box_1565 etc). Not firing just because the phase is
    # "stable-contact" (holding the object steady) is fine — we only
    # drop when contact is also completely absent.
    if (
        ACTION_PATTERN.search(text)
        and manip_frac < manip_min_frac
        and not any_hand_contact
    ):
        reasons.append("text_says_action_but_static")

    # Gate 6: min hand-to-object-center distance > threshold across the
    # entire sequence. Hand never got within 2 m of the object — usually
    # a preprocessing alignment bug or a seq where the "object" was
    # unrelated to what the human was doing. Only fires when both motion
    # and object_positions are available.
    if seq_motion is not None:
        obj_pos = seq_motion.get("object_positions")
        joints_22 = seq_motion.get("joints_22")
        if obj_pos is not None and joints_22 is not None:
            T_min = min(len(obj_pos), len(joints_22))
            if T_min > 0:
                hands = joints_22[:T_min, 20:22, :]         # wrist joints
                obj = obj_pos[:T_min, None, :]
                d = np.linalg.norm(hands - obj, axis=-1)
                min_d = float(d.min()) if d.size else float("inf")
                if min_d > geometric_outlier_m:
                    reasons.append("geometric_outlier")

    return len(reasons) == 0, reasons


# ----- Per-subset processing -------------------------------------------------


def clean_subset(subset_dir: Path, pseudo_label_subdir: str) -> dict[str, Any]:
    metadata_path = subset_dir / "metadata.json"
    if not metadata_path.exists():
        return {"subset": subset_dir.name, "error": f"missing {metadata_path}"}

    pseudo_dir = subset_dir / pseudo_label_subdir
    motions_dir = subset_dir / "motions"

    metadata = load_json(metadata_path)
    kept: list[dict] = []
    dropped: list[dict] = []
    reason_counts: dict[str, int] = {}

    for entry in metadata:
        seq_id = entry["seq_id"]
        label_path = pseudo_dir / f"{seq_id}.npz"

        if not label_path.exists():
            dropped.append({
                "seq_id": seq_id,
                "text": (entry.get("text") or "")[:160],
                "reasons": ["missing_labels"],
            })
            reason_counts["missing_labels"] = reason_counts.get("missing_labels", 0) + 1
            continue

        labels = np.load(label_path, allow_pickle=False)
        label_dict = {k: labels[k] for k in labels.files}

        motion_path = motions_dir / f"{seq_id}.npz"
        motion_dict: dict | None = None
        if motion_path.exists():
            m = np.load(motion_path, allow_pickle=False)
            motion_dict = {
                k: m[k] for k in ("object_positions", "joints_22")
                if k in m.files
            }

        keep, reasons = classify_sequence(seq_id, entry, label_dict, motion_dict)
        if keep:
            kept.append(entry)
        else:
            dropped.append({
                "seq_id": seq_id,
                "text": (entry.get("text") or "")[:160],
                "reasons": reasons,
            })
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    clean_metadata_path = subset_dir / "metadata_clean.json"
    save_json(clean_metadata_path, kept)

    report = {
        "timestamp": datetime.now().isoformat(),
        "subset": subset_dir.name,
        "source_metadata": str(metadata_path),
        "pseudo_label_subdir": pseudo_label_subdir,
        "num_in_metadata": len(metadata),
        "num_kept": len(kept),
        "num_dropped": len(dropped),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
        "dropped_samples": dropped[:200],
    }
    save_json(subset_dir / "cleaning_report.json", report)
    return {
        "subset": subset_dir.name,
        "num_in_metadata": len(metadata),
        "num_kept": len(kept),
        "num_dropped": len(dropped),
        "reason_counts": reason_counts,
    }


# ----- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="InterAct piano root (contains chairs/, imhd/, ...)",
    )
    parser.add_argument(
        "--subsets", nargs="+",
        default=("chairs", "imhd", "neuraldome", "omomo_correct_v2"),
    )
    parser.add_argument(
        "--pseudo-label-subdir", default="pseudo_labels",
        help="Subdirectory under each subset that holds the per-seq npz labels",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    totals = {"num_in_metadata": 0, "num_kept": 0, "num_dropped": 0}
    all_reasons: dict[str, int] = {}

    for subset in args.subsets:
        subset_dir = args.data_dir / subset
        if not subset_dir.exists():
            print(f"[skip] {subset}: {subset_dir} not found")
            continue

        result = clean_subset(subset_dir, args.pseudo_label_subdir)
        if "error" in result:
            print(f"[{subset}] error: {result['error']}")
            continue

        kept = result["num_kept"]
        total = result["num_in_metadata"]
        dropped = result["num_dropped"]
        pct = 100.0 * kept / max(total, 1)
        print(f"[{subset}] kept {kept} / {total} ({pct:.1f}%) — dropped {dropped}")
        for reason, count in sorted(result["reason_counts"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

        totals["num_in_metadata"] += total
        totals["num_kept"] += kept
        totals["num_dropped"] += dropped
        for r, c in result["reason_counts"].items():
            all_reasons[r] = all_reasons.get(r, 0) + c

    print("\n=== TOTALS ===")
    kept_pct = 100.0 * totals["num_kept"] / max(totals["num_in_metadata"], 1)
    print(f"kept {totals['num_kept']} / {totals['num_in_metadata']} ({kept_pct:.1f}%) — "
          f"dropped {totals['num_dropped']}")
    for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
