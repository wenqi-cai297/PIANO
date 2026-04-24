"""Sample seq_ids for the v9 pseudo-label visualisation groups.

Each vis group tests ONE aspect of v9. Rather than hard-coding a handful
of seq_ids, this sampler picks N clips per group that actually match the
failure mode the group is meant to illustrate, read from the live
``metadata_clean.json`` + ``cleaning_report.json``. The shell wrapper
(``vis_v9_pseudo_labels.sh``) pipes the stdout output into
``piano-visualize-pseudo-labels --seq-ids``.

Groups (keep the same order as vis_v9_pseudo_labels.sh):

    1. neuraldome_wrapgrip_recovery      — v9-KEPT, hold/carry/lift text
                                           on wrap-grip-prone objects
    2. omomo_kick_recovery               — v9-KEPT, "kick ..." text
    3. omomo_scoot_recovery              — v9-KEPT, "scoot / use ... foot" text
    4. chairs_sit_preservation           — v9-KEPT, clean sit text (regression guard)
    5. chairs_regression_check           — v9-DROPPED, zero_contact on Obj98/110/33
    6. neuraldome_bigsofa_sit_failing    — v9-DROPPED, sofa sit clips

Usage:
    python scripts/stage1_pseudo_labels/sample_v9_vis_seqs.py \\
        --piano-root /media/.../InterAct/piano \\
        --group 1 --n 12 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


# ---- Text-pattern regexes (shared across groups) ----

HOLD_CARRY = re.compile(
    r"\b(hold|holds|held|holding|"
    r"carry|carries|carried|carrying|"
    r"lift|lifts|lifted|lifting|"
    r"grab|grabs|grabbed|grabbing|"
    r"grip|grips|gripped|gripping|"
    r"wave|waves|waved|waving|"
    r"push|pushes|pushed|pushing|"
    r"pull|pulls|pulled|pulling|"
    r"shake|shakes|shook|shaking|"
    r"throw|throws|threw|thrown|throwing|"
    r"swing|swings|swung|swinging)\b",
    re.IGNORECASE,
)
KICK = re.compile(r"\b(kick|kicks|kicked|kicking)\b", re.IGNORECASE)
SCOOT = re.compile(r"\b(scoot|scoots|scooting|scooted|use.*foot|with.*foot)\b", re.IGNORECASE)
PURE_SIT = re.compile(r"\bsits?\s+(down\s+)?on\b", re.IGNORECASE)

# Objects that dominate the wrap-grip drop class on neuraldome
WRAPGRIP_OBJS = {"flower", "trolleycase", "case", "tennis", "box", "pingpong", "book"}
# Chair objects where v9 hand-threshold rollback lost borderline hand contacts
REGRESSION_OBJS = {"Obj98", "Obj110", "Obj33"}
# Sofa objects still failing the sitting gate after v9
SOFA_OBJS = {"bigsofa", "smallsofa"}


# ---- IO helpers ----

def _load_json(p: Path) -> list | dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _kept_ids(piano_root: Path, subset: str) -> set[str]:
    """seq_ids that passed the cleaning pass (v9 metadata_clean.json)."""
    meta = _load_json(piano_root / subset / "metadata_clean.json")
    return {m["seq_id"] for m in meta}


def _all_meta(piano_root: Path, subset: str) -> list[dict]:
    return _load_json(piano_root / subset / "metadata.json")


def _dropped_samples(piano_root: Path, subset: str) -> list[dict]:
    """dropped_samples[:200] from cleaning_report.json — enough for our groups."""
    return _load_json(piano_root / subset / "cleaning_report.json")["dropped_samples"]


def _object_of(seq_id: str) -> str:
    """Pull the object name out of a seq_id (e.g. ``sub10_largebox_041`` → largebox)."""
    parts = seq_id.split("_")
    return parts[1] if len(parts) >= 2 else ""


# ---- Per-group samplers ----

def group_1_neuraldome_wrapgrip(piano_root: Path, rng: random.Random) -> list[str]:
    kept = _kept_ids(piano_root, "neuraldome")
    meta = _all_meta(piano_root, "neuraldome")
    pool = [
        m["seq_id"] for m in meta
        if m["seq_id"] in kept
        and HOLD_CARRY.search(m.get("text") or "")
        and _object_of(m["seq_id"]) in WRAPGRIP_OBJS
    ]
    rng.shuffle(pool)
    return pool


def group_2_omomo_kick(piano_root: Path, rng: random.Random) -> list[str]:
    kept = _kept_ids(piano_root, "omomo_correct_v2")
    meta = _all_meta(piano_root, "omomo_correct_v2")
    pool = [
        m["seq_id"] for m in meta
        if m["seq_id"] in kept
        and KICK.search(m.get("text") or "")
    ]
    rng.shuffle(pool)
    return pool


def group_3_omomo_scoot(piano_root: Path, rng: random.Random) -> list[str]:
    kept = _kept_ids(piano_root, "omomo_correct_v2")
    meta = _all_meta(piano_root, "omomo_correct_v2")
    pool = [
        m["seq_id"] for m in meta
        if m["seq_id"] in kept
        and SCOOT.search(m.get("text") or "")
        # exclude kicks (kick can co-fire with use-foot pattern below)
        and not KICK.search(m.get("text") or "")
    ]
    rng.shuffle(pool)
    return pool


def group_4_chairs_sit_preservation(piano_root: Path, rng: random.Random) -> list[str]:
    kept = _kept_ids(piano_root, "chairs")
    meta = _all_meta(piano_root, "chairs")
    # Prefer clean sits (no "stands up" / "walks around" compound actions)
    # that are likely to have clean sitting labels across the whole clip.
    pool = [
        m["seq_id"] for m in meta
        if m["seq_id"] in kept
        and PURE_SIT.search(m.get("text") or "")
        and "stand" not in (m.get("text") or "").lower()
        and "walk" not in (m.get("text") or "").lower()
    ]
    rng.shuffle(pool)
    return pool


def group_5_chairs_regression(piano_root: Path, rng: random.Random) -> list[str]:
    drops = _dropped_samples(piano_root, "chairs")
    pool = [
        d["seq_id"] for d in drops
        if "zero_contact_all_parts" in d["reasons"]
        and _object_of(d["seq_id"]) in REGRESSION_OBJS
    ]
    rng.shuffle(pool)
    return pool


def group_6_neuraldome_bigsofa_sit(piano_root: Path, rng: random.Random) -> list[str]:
    drops = _dropped_samples(piano_root, "neuraldome")
    pool = [
        d["seq_id"] for d in drops
        if "text_says_sit_but_no_sitting" in d["reasons"]
        and _object_of(d["seq_id"]) in SOFA_OBJS
    ]
    rng.shuffle(pool)
    return pool


GROUPS = {
    1: ("neuraldome_wrapgrip_recovery",     group_1_neuraldome_wrapgrip),
    2: ("omomo_kick_recovery",              group_2_omomo_kick),
    3: ("omomo_scoot_recovery",             group_3_omomo_scoot),
    4: ("chairs_sit_preservation",          group_4_chairs_sit_preservation),
    5: ("chairs_regression_check",          group_5_chairs_regression),
    6: ("neuraldome_bigsofa_sit_failing",   group_6_neuraldome_bigsofa_sit),
}


SUBSET_OF = {
    1: "neuraldome",
    2: "omomo_correct_v2",
    3: "omomo_correct_v2",
    4: "chairs",
    5: "chairs",
    6: "neuraldome",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--piano-root", type=Path, required=True)
    parser.add_argument("--group", type=int, required=True, choices=list(GROUPS.keys()))
    parser.add_argument("--n", type=int, default=12, help="Max clips to output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--emit", choices=("seq_ids", "label", "subset"), default="seq_ids",
        help="What to print. seq_ids: space-separated ids (default). "
             "label: group dir name. subset: subset name (e.g. neuraldome).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    label, sampler = GROUPS[args.group]
    subset = SUBSET_OF[args.group]

    if args.emit == "label":
        print(label)
        return
    if args.emit == "subset":
        print(subset)
        return

    pool = sampler(args.piano_root, rng)
    picks = pool[: args.n]
    print(" ".join(picks))


if __name__ == "__main__":
    main()
