"""Round-30 E0 — build the Idle-Local-Detail (ILD) subset.

Per ``analyses/2026-05-29_round30_idle_local_detail_diagnosis_plan.md``
§E0. Produces three JSON selection files (train / val / control) and a
Markdown stats report. The first decision gate of the entire round
fires off the train ILD fraction.

A clip ∈ ILD iff ALL of:
  * is_stationary       : root XZ p95 < 0.05 m AND walking-frame frac < 5%
  * not has_contact_event : no 0→1 hand-contact transition (allows clips
                             that have stable contact, e.g. seat contact,
                             but excludes clips where contact is being
                             acquired or released — the model already
                             gets enough signal there)
  * has_upper_body_motion : keyword regex match OR upper-body velocity
                             RMS > 0.03 m/s on non-walking frames

A clip ∈ control iff:
  * (not is_stationary) OR has_contact_event
The control set is then size-matched per subset to the ILD set so paired
metric comparisons remain stratified.

Outputs:
  analyses/round30_ild/selection_train.json
  analyses/round30_ild/selection_val.json
  analyses/round30_ild/selection_control.json   (combined train+val
                                                  control matched to
                                                  total ILD size)
  analyses/round30_ild/subset_stats.md

Usage on the server:
    python scripts/stage_b_generator/round30_build_ild_subset.py \\
        --config configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml \\
        --output-dir analyses/round30_ild

The script reads the train-time ``cfg.data.datasets`` list (so it
picks up the server-side dataset root automatically) plus
``cfg.data.subject_split`` to mirror the trainer's val cut. Re-runs are
deterministic for a fixed selection seed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# SMPL-22 joint indices (mirror src/piano/utils/smpl_utils.py:16-39).
PELVIS_IDX = 0
SPINE1_IDX = 3
SPINE2_IDX = 6
SPINE3_IDX = 9
NECK_IDX = 12
L_SHOULDER_IDX = 16
R_SHOULDER_IDX = 17
L_ELBOW_IDX = 18
R_ELBOW_IDX = 19

UPPER_BODY_JOINT_INDICES: tuple[int, ...] = (
    NECK_IDX, L_SHOULDER_IDX, R_SHOULDER_IDX,
    L_ELBOW_IDX, R_ELBOW_IDX,
    SPINE1_IDX, SPINE2_IDX, SPINE3_IDX,
)

# Keywords that imply text-only upper-body local detail (no contact
# event, no walking implied). Case-insensitive regex. Kept conservative
# to limit ILD subset noise.
#
# Pattern note: each verb phrase tolerates an optional possessive /
# determiner between verb and noun (e.g. "stretches their arms"). The
# ``(?:\s+(?:his|her|their|the))?`` group catches the common cases
# without flooding false positives.
_DET = r"(?:\s+(?:his|her|their|the|its|my|your))?"
IDLE_LOCAL_DETAIL_KEYWORDS: tuple[str, ...] = (
    rf"\bstretch(?:es|ing)?{_DET}\s+(?:arm|back|body|neck|leg)",
    rf"\bcross(?:es|ed|ing)?{_DET}\s+arm",
    rf"\bfold(?:s|ed|ing)?{_DET}\s+arm",
    rf"\brest(?:s|ed|ing)?{_DET}\s+(?:hand|chin|head|elbow)",
    r"\bhand(?:s)?\s+on\s+(?:head|face|chin|forehead|hip|knee|lap|chest)",
    rf"\btouch(?:es|ed|ing)?{_DET}\s+(?:head|face|chin|forehead|hair|cheek)",
    r"\bscratch(?:es|ed|ing)?",
    rf"\bcover(?:s|ed|ing)?{_DET}\s+(?:face|eye|mouth)",
    rf"\brub(?:s|bed|bing)?{_DET}\s+(?:eye|face|head|hand)",
    rf"\bput(?:s)?{_DET}\s+hand(?:s)?\s+(?:on|behind|over)",
    rf"\braise(?:s|d)?{_DET}\s+(?:hand|arm)",
    r"\bwave(?:s|d)?",
    r"\bclap(?:s|ped)?",
    r"\bnod(?:s|ded)?",
    rf"\blean(?:s|ed)?{_DET}\s+back",
    r"\bsit(?:s|ting)?\s+(?:still|quietly|calmly)",
)


@dataclass(slots=True)
class ClipFeatures:
    """All scalars derivable from a single clip needed for ILD/control
    classification."""
    subset: str
    seq_id: str
    split: str
    text: str
    num_frames: int
    root_xz_p95_m: float
    walking_frac: float
    contact_event_count: int   # number of 0→1 hand-contact transitions
    contact_any_frac: float    # fraction of frames with any hand-contact
    upper_body_vel_rms_mps: float  # excluding walking frames
    keyword_hit: bool

    def is_stationary(self, max_root_xz_p95_m: float, max_walking_frac: float) -> bool:
        return (
            self.root_xz_p95_m < max_root_xz_p95_m
            and self.walking_frac < max_walking_frac
        )

    def has_contact_event(self) -> bool:
        return self.contact_event_count > 0

    def has_upper_body_motion(self, vel_threshold_mps: float) -> bool:
        return self.keyword_hit or self.upper_body_vel_rms_mps > vel_threshold_mps


def _root_xz_p95_m(joints_22: np.ndarray, valid_T: int) -> float:
    """p95 of pelvis XZ displacement vs frame 0, in metres."""
    if valid_T < 2:
        return 0.0
    pelvis = joints_22[:valid_T, PELVIS_IDX, :]   # (valid_T, 3)
    delta_xz = pelvis[:, [0, 2]] - pelvis[0:1, [0, 2]]
    return float(np.percentile(np.linalg.norm(delta_xz, axis=-1), 95))


def _walking_frac(joints_22: np.ndarray, valid_T: int, fps: float = 20.0) -> float:
    """Fraction of frames in the walking_mask, via
    piano.data.interaction_hint.derive_walking_mask_from_gt.
    """
    if valid_T < 2:
        return 0.0
    from piano.data.interaction_hint import derive_walking_mask_from_gt

    mask = derive_walking_mask_from_gt(joints_22[:valid_T], fps=fps)
    # mask shape may be (valid_T, 1) or (valid_T,).
    mask = np.asarray(mask).reshape(-1)
    return float(mask.mean())


def _contact_event_count_and_frac(
    contact_state: np.ndarray | None, valid_T: int,
) -> tuple[int, float]:
    """Count 0→1 transitions on either hand column (idx 0 / 1) and the
    fraction of frames with any hand in contact.

    contact_state shape: (T, ≥2) — column 0 = left hand, 1 = right hand.
    """
    if contact_state is None or valid_T < 2 or contact_state.shape[-1] < 2:
        return 0, 0.0
    cs = contact_state[:valid_T, :2]                       # (valid_T, 2)
    binary = (cs > 0.5).astype(np.int8)
    # 0→1 transition on each hand independently.
    transitions = np.maximum(0, binary[1:] - binary[:-1])   # (valid_T-1, 2)
    event_count = int(transitions.sum())
    any_contact = (binary.sum(axis=-1) > 0).astype(np.float32)
    return event_count, float(any_contact.mean())


def _upper_body_vel_rms_mps(
    joints_22: np.ndarray, valid_T: int, fps: float = 20.0,
    walking_mask: np.ndarray | None = None,
) -> float:
    """RMS speed of the upper-body 8 joints over non-walking frames, m/s."""
    if valid_T < 2:
        return 0.0
    upper = joints_22[:valid_T, list(UPPER_BODY_JOINT_INDICES), :]   # (T, 8, 3)
    vel = np.diff(upper, axis=0) * fps                                # (T-1, 8, 3)
    speed = np.linalg.norm(vel, axis=-1)                              # (T-1, 8)
    if walking_mask is not None:
        wm = np.asarray(walking_mask).reshape(-1)[: valid_T - 1].astype(bool)
        # If there are any non-walking frames, restrict to those; otherwise
        # the clip is walking through-and-through and there is nothing
        # meaningful to say about its "idle upper-body" velocity. Return 0
        # in that case rather than falling back to all frames, since the
        # caller's intent is "how much does the upper body move when NOT
        # walking" — answer is undefined for a fully-walking clip.
        if (~wm).any():
            speed = speed[~wm]
        else:
            return 0.0
    if speed.size == 0:
        return 0.0
    # Joint-mean then RMS over (valid_T-1) frames.
    per_frame = speed.mean(axis=-1)
    return float(np.sqrt((per_frame ** 2).mean()))


def _keyword_hit(text: str, compiled_patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in compiled_patterns)


def _load_clip_features(
    npz_path: Path, meta_entry: dict, compiled_patterns: list[re.Pattern],
    pseudo_label_root: Path | None,
    fps: float = 20.0,
) -> ClipFeatures | None:
    """Load one clip's npz + (optional) pseudo-label npz and compute all
    scalars needed for ILD classification. Returns None if the clip
    cannot be opened or is too short.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as e:
        print(f"  [WARN] skip {npz_path}: {e}", file=sys.stderr)
        return None
    # Prefer the FK-derived ``joints_22`` field (see HOIDataset L417);
    # fall back to legacy ``joints`` for older npzs.
    if "joints_22" in data.files:
        joints_22 = data["joints_22"].astype(np.float32)
    elif "joints" in data.files:
        joints_22 = data["joints"].astype(np.float32)
    else:
        return None
    T = int(meta_entry.get("num_frames", joints_22.shape[0]))
    valid_T = min(T, joints_22.shape[0])
    if valid_T < 5:
        return None
    # Pseudo-label sidecar for contact_state.
    contact_state: np.ndarray | None = None
    if pseudo_label_root is not None:
        side = pseudo_label_root / npz_path.name
        if side.exists():
            try:
                side_data = np.load(side, allow_pickle=False)
                if "contact_state" in side_data.files:
                    contact_state = side_data["contact_state"].astype(np.float32)
            except (OSError, ValueError):
                contact_state = None

    root_xz = _root_xz_p95_m(joints_22, valid_T)
    walking_frac = _walking_frac(joints_22, valid_T, fps=fps)
    from piano.data.interaction_hint import derive_walking_mask_from_gt
    walking_mask = np.asarray(
        derive_walking_mask_from_gt(joints_22[:valid_T], fps=fps)
    ).reshape(-1)
    ub_vel = _upper_body_vel_rms_mps(
        joints_22, valid_T, fps=fps, walking_mask=walking_mask,
    )
    ev_cnt, any_contact_frac = _contact_event_count_and_frac(contact_state, valid_T)
    text = str(meta_entry.get("text", ""))
    return ClipFeatures(
        subset=str(meta_entry.get("subset", npz_path.parent.parent.name)),
        seq_id=str(meta_entry["seq_id"]),
        split=str(meta_entry.get("split", "train")),
        text=text,
        num_frames=valid_T,
        root_xz_p95_m=root_xz,
        walking_frac=walking_frac,
        contact_event_count=ev_cnt,
        contact_any_frac=any_contact_frac,
        upper_body_vel_rms_mps=ub_vel,
        keyword_hit=_keyword_hit(text, compiled_patterns),
    )


def _resolve_npz_path(root: Path, seq_id: str) -> Path | None:
    """Find the motion npz file for a seq_id. The dataset layout is
    ``root / "motions" / "{seq_id}.npz"`` (see HOIDataset.__getitem__
    in src/piano/data/dataset.py:414). We probe a few legacy fallbacks
    in case an older layout is in play locally.
    """
    for sub in ("motions", ""):
        cand = root / sub / f"{seq_id}.npz" if sub else root / f"{seq_id}.npz"
        if cand.exists():
            return cand
    return None


@dataclass(slots=True)
class _BucketStats:
    """Aggregate counters per (subset, split) bucket."""
    total: int = 0
    ild: int = 0
    control: int = 0
    keyword_hits: int = 0
    upper_body_only: int = 0   # ILD by velocity, not by keyword
    keyword_only: int = 0      # keyword but velocity below threshold
    text_counter: Counter = field(default_factory=Counter)


def _process_subset(
    subset_name: str,
    root: Path,
    pseudo_label_root: Path | None,
    compiled_patterns: list[re.Pattern],
    *,
    bucket_filter: set[tuple[str, str]] | None,
    max_root_xz_p95_m: float,
    max_walking_frac: float,
    ub_vel_threshold_mps: float,
    fps: float,
) -> tuple[list[ClipFeatures], dict[str, _BucketStats]]:
    """Scan one subset's metadata_clean.json and classify every clip.

    ``bucket_filter`` (subset, seq_id) set: when given, only clips in the
    filter are processed (used to mirror subject_split val cut).
    """
    meta_path = root / "metadata_clean.json"
    if not meta_path.exists():
        meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no metadata under {root}")
    metadata: list[dict] = json.loads(meta_path.read_text("utf-8"))

    features: list[ClipFeatures] = []
    buckets: dict[str, _BucketStats] = {"train": _BucketStats(), "val": _BucketStats()}

    for entry in metadata:
        seq_id = str(entry["seq_id"])
        if bucket_filter is not None and (subset_name, seq_id) not in bucket_filter:
            continue
        npz = _resolve_npz_path(root, seq_id)
        if npz is None:
            continue
        feat = _load_clip_features(
            npz, entry, compiled_patterns, pseudo_label_root, fps=fps,
        )
        if feat is None:
            continue
        feat.subset = subset_name
        features.append(feat)
        b = buckets.setdefault(feat.split, _BucketStats())
        b.total += 1
        is_ild = (
            feat.is_stationary(max_root_xz_p95_m, max_walking_frac)
            and not feat.has_contact_event()
            and feat.has_upper_body_motion(ub_vel_threshold_mps)
        )
        if is_ild:
            b.ild += 1
            if feat.keyword_hit:
                b.keyword_hits += 1
            if feat.keyword_hit and feat.upper_body_vel_rms_mps <= ub_vel_threshold_mps:
                b.keyword_only += 1
            if (
                feat.upper_body_vel_rms_mps > ub_vel_threshold_mps
                and not feat.keyword_hit
            ):
                b.upper_body_only += 1
            b.text_counter[feat.text[:80]] += 1
        else:
            b.control += 1
    return features, buckets


def _stratified_size_match_control(
    features: list[ClipFeatures],
    ild_keys: set[tuple[str, str]],
    target_per_subset: dict[str, int],
    rng: np.random.Generator,
) -> list[ClipFeatures]:
    """For each subset, sample (target_per_subset[subset]) clips from
    features that are NOT in ild_keys. Returns the combined list."""
    out: list[ClipFeatures] = []
    by_subset: dict[str, list[ClipFeatures]] = {}
    for f in features:
        if (f.subset, f.seq_id) in ild_keys:
            continue
        by_subset.setdefault(f.subset, []).append(f)
    for subset, n in target_per_subset.items():
        pool = by_subset.get(subset, [])
        if not pool or n <= 0:
            continue
        n_take = min(n, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        out.extend(pool[int(i)] for i in idx)
    return out


def _selection_dict(features: list[ClipFeatures]) -> dict:
    return {
        "selected": [
            {
                "subset": f.subset,
                "seq_id": f.seq_id,
                "split": f.split,
                "text": f.text,
                "num_frames": f.num_frames,
                "root_xz_p95_m": f.root_xz_p95_m,
                "walking_frac": f.walking_frac,
                "contact_event_count": f.contact_event_count,
                "contact_any_frac": f.contact_any_frac,
                "upper_body_vel_rms_mps": f.upper_body_vel_rms_mps,
                "keyword_hit": f.keyword_hit,
            }
            for f in features
        ],
        "n_clips": len(features),
    }


def _write_stats_md(
    out_path: Path,
    buckets_per_subset: dict[str, dict[str, _BucketStats]],
    thresholds: dict[str, float],
    train_ild_frac: float,
    val_ild_frac: float,
) -> None:
    L: list[str] = []
    a = L.append
    a("# Round-30 ILD subset stats")
    a("")
    a(f"- `is_stationary` thresholds: root XZ p95 < "
      f"{thresholds['max_root_xz_p95_m']*100:.1f} cm AND walking-frame frac "
      f"< {thresholds['max_walking_frac']*100:.1f} %")
    a(f"- `has_contact_event` = ≥ 1 hand 0→1 transition")
    a(f"- `has_upper_body_motion` = keyword match OR "
      f"non-walking upper-body velocity RMS > "
      f"{thresholds['ub_vel_threshold_mps']*100:.1f} cm/s")
    a("")
    a("## Decision gate")
    a("")
    a(f"- Train ILD fraction: **{train_ild_frac*100:.2f}%**")
    a(f"- Val ILD fraction:   {val_ild_frac*100:.2f}%")
    if train_ild_frac < 0.01:
        a("- ⚠️  **Train ILD < 1% → H1 (data sparsity) supported. "
          "Recommend STOP and write to limitations or move to D4.**")
    elif train_ild_frac < 0.05:
        a("- ⚠️  Train ILD in [1%, 5%]. Continue but record as caveat.")
    else:
        a("- ✓  Train ILD ≥ 5%. Continue normally.")
    a("")
    a("## Per-subset breakdown")
    a("")
    a("| subset | split | total | ILD | control | keyword_only | ub_vel_only | both |")
    a("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for subset, by_split in sorted(buckets_per_subset.items()):
        for split, b in by_split.items():
            both = b.ild - b.keyword_only - b.upper_body_only
            a(
                f"| {subset} | {split} | {b.total} | {b.ild} | {b.control} | "
                f"{b.keyword_only} | {b.upper_body_only} | {both} |"
            )
    a("")
    a("## Top-10 ILD text patterns (per subset, train split)")
    a("")
    for subset, by_split in sorted(buckets_per_subset.items()):
        train = by_split.get("train", _BucketStats())
        if not train.text_counter:
            continue
        a(f"### {subset}")
        for text, n in train.text_counter.most_common(10):
            a(f"- ({n}×) {text}")
        a("")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Trainer config (mirrors dataset roots + subject_split)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analyses/round30_ild"))
    parser.add_argument("--max-root-xz-p95-m", type=float, default=0.05)
    parser.add_argument("--max-walking-frac", type=float, default=0.05)
    parser.add_argument("--ub-vel-threshold-mps", type=float, default=0.03)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--max-train-ild", type=int, default=50,
        help="Cap on ILD train clips written to selection_train.json (E2/E3 "
             "use overfit subsets). 0 = no cap.",
    )
    parser.add_argument(
        "--max-val-ild", type=int, default=48,
        help="Cap on ILD val clips. 0 = no cap.",
    )
    args = parser.parse_args()

    # Deferred imports — keep main() the only torch/omegaconf entry point.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compiled = [re.compile(p, re.IGNORECASE) for p in IDLE_LOCAL_DETAIL_KEYWORDS]

    # Subject_split: mirror the trainer's val cut so val ILD is a true
    # held-out subset.
    subj_cfg = cfg.data.get("subject_split", None)
    bucket_filter: set[tuple[str, str]] | None = None
    if subj_cfg is not None and bool(subj_cfg.get("enabled", False)):
        from piano.data.split import build_subject_split, extract_subject_id

        keys: set[tuple[str, str]] = set()
        for entry in cfg.data.datasets:
            root = Path(entry.root)
            mp = root / "metadata_clean.json"
            if not mp.exists():
                mp = root / "metadata.json"
            for m in json.loads(mp.read_text("utf-8")):
                sid = extract_subject_id(root.name, m.get("seq_id", ""))
                if sid is not None:
                    keys.add((root.name, sid))
        splits = build_subject_split(
            sorted(keys),
            train_pct=int(subj_cfg.train_pct),
            val_pct=int(subj_cfg.val_pct),
            seed=int(subj_cfg.seed),
        )
        # We don't filter here — we annotate `split` per clip later using
        # the subject's bucket. Easier path: build a (subset, subject_id)
        # → bucket lookup and rewrite feat.split on the fly.
        subj_to_bucket: dict[tuple[str, str], str] = {}
        for bucket in ("train", "val"):
            for k in splits[bucket]:
                subj_to_bucket[k] = bucket
    else:
        subj_to_bucket = {}

    all_features: list[ClipFeatures] = []
    per_subset_buckets: dict[str, dict[str, _BucketStats]] = {}
    for entry in cfg.data.datasets:
        root = Path(entry.root)
        subset_name = root.name
        # Pseudo-label root: resolve same as HOIDataset.
        pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)
        pseudo_label_dir_cfg = cfg.data.get("pseudo_label_dir", None)
        if pseudo_label_dir_cfg is not None:
            pseudo_root = Path(pseudo_label_dir_cfg)
        elif pseudo_label_subdir:
            pseudo_root = root / pseudo_label_subdir
        else:
            pseudo_root = root / "pseudo_labels"
        if not pseudo_root.exists():
            pseudo_root = None

        print(f"[round30_ild] scanning {subset_name} root={root}")
        feats, buckets = _process_subset(
            subset_name=subset_name,
            root=root,
            pseudo_label_root=pseudo_root,
            compiled_patterns=compiled,
            bucket_filter=None,
            max_root_xz_p95_m=args.max_root_xz_p95_m,
            max_walking_frac=args.max_walking_frac,
            ub_vel_threshold_mps=args.ub_vel_threshold_mps,
            fps=args.fps,
        )
        # Re-bucket by subject if subject_split is enabled.
        if subj_to_bucket:
            from piano.data.split import extract_subject_id
            for f in feats:
                sid = extract_subject_id(subset_name, f.seq_id)
                if sid is not None and (subset_name, sid) in subj_to_bucket:
                    f.split = subj_to_bucket[(subset_name, sid)]
            # Recompute buckets with new splits.
            buckets = {"train": _BucketStats(), "val": _BucketStats()}
            for f in feats:
                b = buckets.setdefault(f.split, _BucketStats())
                b.total += 1
                is_ild = (
                    f.is_stationary(args.max_root_xz_p95_m, args.max_walking_frac)
                    and not f.has_contact_event()
                    and f.has_upper_body_motion(args.ub_vel_threshold_mps)
                )
                if is_ild:
                    b.ild += 1
                    if f.keyword_hit and f.upper_body_vel_rms_mps <= args.ub_vel_threshold_mps:
                        b.keyword_only += 1
                    elif f.upper_body_vel_rms_mps > args.ub_vel_threshold_mps and not f.keyword_hit:
                        b.upper_body_only += 1
                    b.text_counter[f.text[:80]] += 1
                else:
                    b.control += 1
        per_subset_buckets[subset_name] = buckets
        all_features.extend(feats)
        print(
            f"  → kept {len(feats)} clips, "
            f"train ILD = {buckets.get('train', _BucketStats()).ild}, "
            f"val ILD = {buckets.get('val', _BucketStats()).ild}"
        )

    # Aggregate gate.
    total_train = sum(
        b.get("train", _BucketStats()).total for b in per_subset_buckets.values()
    )
    total_val = sum(
        b.get("val", _BucketStats()).total for b in per_subset_buckets.values()
    )
    ild_train = sum(
        b.get("train", _BucketStats()).ild for b in per_subset_buckets.values()
    )
    ild_val = sum(
        b.get("val", _BucketStats()).ild for b in per_subset_buckets.values()
    )
    train_frac = ild_train / max(total_train, 1)
    val_frac = ild_val / max(total_val, 1)
    print(
        f"[round30_ild] train ILD = {ild_train}/{total_train} "
        f"({train_frac*100:.2f}%)"
    )
    print(
        f"[round30_ild] val   ILD = {ild_val}/{total_val} "
        f"({val_frac*100:.2f}%)"
    )

    # Build selections.
    train_ild = [
        f for f in all_features
        if f.split == "train"
        and f.is_stationary(args.max_root_xz_p95_m, args.max_walking_frac)
        and not f.has_contact_event()
        and f.has_upper_body_motion(args.ub_vel_threshold_mps)
    ]
    val_ild = [
        f for f in all_features
        if f.split == "val"
        and f.is_stationary(args.max_root_xz_p95_m, args.max_walking_frac)
        and not f.has_contact_event()
        and f.has_upper_body_motion(args.ub_vel_threshold_mps)
    ]

    # Deterministic sub-sample if a cap is set.
    rng = np.random.default_rng(args.selection_seed)
    if args.max_train_ild > 0 and len(train_ild) > args.max_train_ild:
        idx = rng.choice(len(train_ild), size=args.max_train_ild, replace=False)
        train_ild = [train_ild[int(i)] for i in idx]
    if args.max_val_ild > 0 and len(val_ild) > args.max_val_ild:
        idx = rng.choice(len(val_ild), size=args.max_val_ild, replace=False)
        val_ild = [val_ild[int(i)] for i in idx]

    # Size-matched control per subset.
    ild_keys = {(f.subset, f.seq_id) for f in (train_ild + val_ild)}
    target_per_subset: dict[str, int] = Counter(f.subset for f in (train_ild + val_ild))
    control = _stratified_size_match_control(
        features=all_features, ild_keys=ild_keys,
        target_per_subset=target_per_subset, rng=rng,
    )

    # Write outputs.
    (args.output_dir / "selection_train.json").write_text(
        json.dumps(_selection_dict(train_ild), indent=2), encoding="utf-8",
    )
    (args.output_dir / "selection_val.json").write_text(
        json.dumps(_selection_dict(val_ild), indent=2), encoding="utf-8",
    )
    (args.output_dir / "selection_control.json").write_text(
        json.dumps(_selection_dict(control), indent=2), encoding="utf-8",
    )
    _write_stats_md(
        args.output_dir / "subset_stats.md",
        buckets_per_subset=per_subset_buckets,
        thresholds={
            "max_root_xz_p95_m": args.max_root_xz_p95_m,
            "max_walking_frac": args.max_walking_frac,
            "ub_vel_threshold_mps": args.ub_vel_threshold_mps,
        },
        train_ild_frac=train_frac,
        val_ild_frac=val_frac,
    )
    print(
        f"[round30_ild] wrote selection_train.json (n={len(train_ild)}), "
        f"selection_val.json (n={len(val_ild)}), "
        f"selection_control.json (n={len(control)})"
    )
    print(f"[round30_ild] wrote subset_stats.md")
    # Exit 2 if the gate fails so the launcher can fail-closed.
    if train_frac < 0.01:
        print("[round30_ild] GATE FAILED: train ILD < 1% — H1 supported.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
