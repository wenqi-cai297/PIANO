"""Train/val (and legacy train/val/test) split logic for HOI datasets.

Two split strategies live here:

* ``build_subject_split`` — primary as of 2026-04-26 / v6+. Per-subset
  stratified by-subject split. Aligns with the HOI generation field
  standard (BEHAVE/GRAB/OMOMO/InterAct all split by subject in CG-HOI
  CVPR'24, HOI-Diff, OMOMO SIGGRAPH Asia'23, etc).
* ``build_object_split`` — legacy primary (v1-v5), now secondary. Kept
  for the optional "novel object" ablation table.

Subject-id extraction patterns differ per InterAct subset; see
``_SUBJECT_PATTERNS``. The functions in this module are pure-Python +
hashlib; no numpy / torch dep so they can be imported and tested
independently of the dataset/torch stack.
"""
from __future__ import annotations

import hashlib
import re


# ---------------------------------------------------------------------------
# Subject-id extraction
# ---------------------------------------------------------------------------

# Per-subset regex for extracting subject id from seq_id. Each InterAct
# subset has its own naming convention — verified 100% coverage on the
# 8475-clip release of v11 metadata (chairs 1502, imhd 592, neuraldome
# 1491, omomo_correct_v2 4890). Add a new entry when introducing a new
# subset; assign None for subsets where subject is not extractable.
_SUBJECT_PATTERNS: dict[str, re.Pattern] = {
    # chairs:           Sub<digits>_Obj<digits>_Seg<X>_<frame>
    "chairs":           re.compile(r"^(Sub\d+)_"),
    # imhd:             <YYYYMMDD>_<subjectname>_<rest>  (e.g. 20230825_songzn_bat_...)
    "imhd":             re.compile(r"^\d{8}_([a-zA-Z]+)_"),
    # neuraldome:       subject<digits>_<rest>
    "neuraldome":       re.compile(r"^(subject\d+)_"),
    # omomo_correct_v2: sub<digits>_<rest>
    "omomo_correct_v2": re.compile(r"^(sub\d+)_"),
}


def extract_subject_id(subset: str, seq_id: str) -> str | None:
    """Return the raw subject id embedded in ``seq_id`` for ``subset``.

    The returned id is the **raw** subject string (e.g. ``"Sub0001"`` /
    ``"songzn"`` / ``"subject06"`` / ``"sub10"``). Namespaced split keys
    are formed downstream as ``f"{subset}/{raw_id}"`` to guarantee no
    cross-subset collision (omomo's ``sub10`` vs a hypothetical chairs
    ``Sub10`` — different patterns, currently no overlap, but namespacing
    is cheap insurance).

    Returns None if the subset is unknown or the seq_id doesn't match
    the expected pattern; callers should treat unknown subjects as
    unfilterable (typically by dropping the entry from the split set).
    """
    pat = _SUBJECT_PATTERNS.get(subset)
    if pat is None:
        return None
    m = pat.match(seq_id)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Subject-id split (PRIMARY as of 2026-04-26 / v6+)
# ---------------------------------------------------------------------------

def build_subject_split(
    subject_keys: list[tuple[str, str]],
    train_pct: int = 85,
    val_pct: int = 15,
    seed: int = 42,
) -> dict[str, set[str]]:
    """Per-subset stratified subject-level train/val split.

    Why subject-level: the HOI generation literature (BEHAVE/GRAB/
    OMOMO/InterAct) all use this convention. CG-HOI CVPR'24, HOI-Diff,
    OMOMO SIGGRAPH Asia'23 split BEHAVE / OMOMO by subject. The Stage A
    v5 per-object analysis (see analyses/2026-04-26_*) demonstrated
    that the previous by-object split put 2 geometrically-novel objects
    (omomo `tripod`, `largetable`) into val with no train neighbour;
    they alone added +11 cm to the global val mean L2.

    Why per-subset stratified: subset sizes are uneven (chairs 403
    subjects, imhd 9, neuraldome 10, omomo 17). Without stratification
    an unlucky seed could put 0 imhd subjects in val.

    Algorithm: for each subset, score each subject by
    ``int(md5(seed || subset || subj_id)[:16], 16) / 2^64`` — a
    deterministic uniform-[0,1) hash. Sort within-subset, take the
    lowest ``train_pct%`` for train, the rest for val. Clamp so that
    when ``n_subjects >= 2`` each bucket has ≥1 subject (otherwise a
    9-subject subset at 85% rounds to 8 and val gets all rounding
    error on top of small N).

    Parameters
    ----------
    subject_keys : list of ``(subset_name, raw_subject_id)`` tuples.
        Duplicates are deduped within each subset before splitting.
    train_pct / val_pct : must sum to 100. Default 85/15.
    seed : changes the hash salt → different deterministic split.

    Returns
    -------
    dict with keys ``"train"`` / ``"val"`` → set of namespaced ids
    of the form ``"{subset}/{raw_subject_id}"``.
    """
    if train_pct + val_pct != 100:
        raise ValueError(
            f"train_pct + val_pct must sum to 100, got {train_pct + val_pct}",
        )

    # Group subjects by subset (dedup within each)
    by_subset: dict[str, set[str]] = {}
    for subset, raw_id in subject_keys:
        by_subset.setdefault(subset, set()).add(raw_id)

    splits: dict[str, set[str]] = {"train": set(), "val": set()}
    for subset, raw_ids in by_subset.items():
        # Deterministic uniform-[0,1) score per subject
        scored: list[tuple[float, str]] = []
        for raw_id in sorted(raw_ids):
            namespaced = f"{subset}/{raw_id}"
            h_hex = hashlib.md5(
                f"{seed}::{namespaced}".encode("utf-8")
            ).hexdigest()[:16]
            score = int(h_hex, 16) / float(1 << 64)
            scored.append((score, namespaced))
        scored.sort()
        n = len(scored)
        # Train cutoff: at least 1, at most n-1 (when n >= 2) so every
        # subset contributes to both buckets even at small N.
        n_train = round(n * train_pct / 100)
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = n  # single subject → can only go in train
        for i, (_, key) in enumerate(scored):
            if i < n_train:
                splits["train"].add(key)
            else:
                splits["val"].add(key)
    return splits


# ---------------------------------------------------------------------------
# Object-id split (LEGACY primary; demoted 2026-04-26 to secondary use
# for the optional "novel object" ablation table only).
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

    **Status: secondary as of 2026-04-26.** Replaced by
    ``build_subject_split`` for primary training. Kept for the
    "novel object" ablation table — toggle ``data.object_split.enabled``
    in the config to use this path instead of subject_split.

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
