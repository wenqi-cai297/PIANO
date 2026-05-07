"""Evaluate pseudo-label / caption alignment for InterAct.

This is a conservative semantic scan. It does not try to understand every
caption perfectly; it only flags clips where the caption contains a strong
interaction cue but the pseudo labels say the opposite, or where labels show
a strong support/contact state that the caption does not mention.

Outputs:
  - per_clip.csv: all clips with label rates and parsed cues
  - issues.csv: clips with rule violations
  - by_rule.csv: issue counts by rule
  - by_class.csv: aggregate contact/support rates by subset/object/action
  - exclude_candidates.json: major issues suitable for exclusion if manual
    review confirms they are sample noise rather than rule failures
  - report.md: compact human-readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_ROOT = Path("E:/Project/Datasets/InterAct/piano_official_process_4")
DEFAULT_LABEL = "v18_h10_f05_pelvis20_official_semantic_marker"
DEFAULT_SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")

HANDHELD_OBJECTS = {
    "baseball",
    "bat",
    "broom",
    "bucket",
    "kettlebell",
    "pingpong",
    "racket",
    "racquet",
    "tennis",
    "umbrella",
}

HAND_ACTION_RE = re.compile(
    r"\b("
    r"hold|holds|holding|held|grab|grabs|grabbing|touch|touches|touching|"
    r"carry|carries|carrying|pick|picks|picking|pickup|put|puts|putting|"
    r"push|pushes|pushing|pull|pulls|pulling|move|moves|moving|"
    r"lift|lifts|lifting|drag|drags|dragging|use|uses|using|"
    r"swing|swings|swinging|hit|hits|hitting|throw|throws|throwing|"
    r"rotate|rotates|rotating|place|places|placing"
    r")\b"
)
STRONG_TOOL_RE = re.compile(
    r"\b(swing|swings|swinging|hit|hits|hitting|use|uses|using)\b"
)
MANIP_RE = re.compile(
    r"\b("
    r"swing|swings|swinging|hit|hits|hitting|push|pushes|pushing|"
    r"pull|pulls|pulling|move|moves|moving|drag|drags|dragging|"
    r"carry|carries|carrying|lift|lifts|lifting|throw|throws|throwing|"
    r"rotate|rotates|rotating|pick|picks|picking|pickup|put|puts|putting"
    r")\b"
)
SIT_RE = re.compile(r"\b(sit|sits|sitting|sat|seat|seated)\b")
LEAN_RE = re.compile(r"\b(lean|leans|leaning|rest|rests|resting|lie|lies|lying|lay|lays)\b")
OBJECT_NOUN_PATTERN = (
    r"chair|sofa|table|desk|box|suitcase|case|trashcan|monitor|keyboard|"
    r"bat|baseball|racket|racquet|tennis|ping\s*pong|pingpong|broom|"
    r"bucket|umbrella|pillow|pan|book|flower|lamp|tripod|skateboard"
)
FOOT_RE = re.compile(
    r"\b(kick|kicks|kicking|scoot|scoots|scooting)\b|"
    rf"\bstep(?:s|ping)?\s+(?:on|onto|off)\s+(?:the\s+)?(?:{OBJECT_NOUN_PATTERN})\b|"
    rf"\b(?:foot|feet|leg|legs)\s+(?:on|onto|against)\s+(?:the\s+)?(?:{OBJECT_NOUN_PATTERN})\b"
)
NO_CONTACT_RE = re.compile(r"\b(walks? around|walks? past|stands? near|stands? beside)\b")
SELF_HAND_RE = re.compile(
    r"\b("
    r"hold(?:s|ing)?\s+(?:their\s+|his\s+|her\s+)?hands?|"
    r"hands?\s+(?:together|in front|behind)|"
    r"clasp(?:s|ing)?\s+(?:their\s+|his\s+|her\s+)?hands?|"
    r"hand\s+on\s+(?:neck|head|leg|knee|chest)|"
    r"touch(?:es|ing)?\s+(?:the\s+)?ground|"
    r"collect(?:s|ing)?\s+from\s+(?:the\s+)?ground|"
    r"grab(?:s|bing)?\s+something"
    r")\b"
)
OBJECT_NOUN_RE = re.compile(rf"\b({OBJECT_NOUN_PATTERN})\b")
OBJECT_DIRECTED_HAND_RE = re.compile(
    rf"\b(push|pushes|pushing|pull|pulls|pulling|move|moves|moving|"
    rf"rotate|rotates|rotating|hold|holds|holding|grab|grabs|grabbing|"
    rf"touch|touches|touching|place|places|placing|put|puts|putting)\b"
    rf"(?:\W+\w+){{0,5}}\W+(?:the\s+)?(?:{OBJECT_NOUN_PATTERN})\b|"
    rf"\b(?:{OBJECT_NOUN_PATTERN})\b(?:\W+\w+){{0,6}}\W+"
    rf"(?:with|using|by)\s+(?:their\s+|his\s+|her\s+)?(?:left\s+|right\s+|both\s+)?hands?\b|"
    rf"\b(?:left\s+|right\s+|both\s+)?hands?\s+"
    rf"(?:on|onto|against|under|grabbing|holding)\s+(?:the\s+)?(?:{OBJECT_NOUN_PATTERN})\b"
)


@dataclass
class ClipStats:
    subset: str
    seq_id: str
    object_id: str
    text: str
    num_frames: int
    l_hand: float
    r_hand: float
    any_hand: float
    l_foot: float
    r_foot: float
    any_foot: float
    pelvis: float
    any_contact: float
    phase_non: float
    phase_stable: float
    phase_manip: float
    support_both: float
    support_single_foot: float
    support_sitting: float
    support_hand: float
    has_hand_action: bool
    has_strong_tool_action: bool
    has_manip_action: bool
    has_sit: bool
    has_lean: bool
    has_foot_action: bool
    has_no_contact_hint: bool
    action: str


def _contains(pattern: re.Pattern[str], text: str) -> bool:
    return bool(pattern.search(text))


def _infer_action(seq_id: str, text: str) -> str:
    s = f"{seq_id} {text}".lower()
    if "pickup" in s or "pick up" in s or "putdown" in s or "put down" in s:
        return "pickup_putdown"
    if re.search(r"\b(hit|hits|hitting)\b", s):
        return "hit"
    if re.search(r"\b(swing|swings|swinging)\b", s):
        return "swing"
    if re.search(r"\b(push|pushes|pushing|pull|pulls|pulling)\b", s):
        return "push_pull"
    if re.search(r"\b(carry|carries|carrying)\b", s):
        return "carry"
    if re.search(r"\b(hold|holds|holding|held)\b", s):
        return "hold"
    if re.search(r"\b(sit|sits|sitting|sat|seat|seated)\b", s):
        return "sit"
    if re.search(r"\b(lean|leans|leaning)\b", s):
        return "lean"
    if re.search(r"\b(kick|kicks|kicking|step|steps|stepping)\b", s):
        return "foot"
    return "other"


def _load_clip_stats(
    root: Path,
    subset: str,
    label: str,
    entry: dict,
) -> ClipStats | None:
    seq_id = entry["seq_id"]
    label_path = root / subset / "pseudo_labels" / label / f"{seq_id}.npz"
    if not label_path.exists():
        return None
    data = np.load(label_path, allow_pickle=False)
    contact = data["contact_state"].astype(np.float32)
    phase = data["phase"].astype(np.int64)
    support = data["support"].astype(np.int64)
    T = int(contact.shape[0])
    text = str(entry.get("text", "") or "")
    object_id = str(entry.get("object_id", ""))
    joined = f"{seq_id} {object_id} {text}".lower()

    return ClipStats(
        subset=subset,
        seq_id=seq_id,
        object_id=object_id,
        text=text,
        num_frames=T,
        l_hand=float((contact[:, 0] > 0.5).mean()),
        r_hand=float((contact[:, 1] > 0.5).mean()),
        any_hand=float(np.maximum(contact[:, 0] > 0.5, contact[:, 1] > 0.5).mean()),
        l_foot=float((contact[:, 2] > 0.5).mean()),
        r_foot=float((contact[:, 3] > 0.5).mean()),
        any_foot=float(np.maximum(contact[:, 2] > 0.5, contact[:, 3] > 0.5).mean()),
        pelvis=float((contact[:, 4] > 0.5).mean()),
        any_contact=float((contact > 0.5).any(axis=1).mean()),
        phase_non=float((phase == 0).mean()),
        phase_stable=float((phase == 1).mean()),
        phase_manip=float((phase == 2).mean()),
        support_both=float((support == 0).mean()),
        support_single_foot=float((support == 1).mean()),
        support_sitting=float((support == 2).mean()),
        support_hand=float((support == 3).mean()),
        has_hand_action=_contains(HAND_ACTION_RE, joined),
        has_strong_tool_action=_contains(STRONG_TOOL_RE, joined),
        has_manip_action=_contains(MANIP_RE, joined),
        has_sit=_contains(SIT_RE, joined),
        has_lean=_contains(LEAN_RE, joined),
        has_foot_action=_contains(FOOT_RE, joined),
        has_no_contact_hint=_contains(NO_CONTACT_RE, joined),
        action=_infer_action(seq_id, text),
    )


def _issue(
    stats: ClipStats,
    rule: str,
    severity: str,
    message: str,
    expected: str,
    observed: str,
) -> dict:
    return {
        "severity": severity,
        "rule": rule,
        "subset": stats.subset,
        "seq_id": stats.seq_id,
        "object_id": stats.object_id,
        "action": stats.action,
        "num_frames": stats.num_frames,
        "expected": expected,
        "observed": observed,
        "message": message,
        "text": stats.text,
    }


def _has_object_directed_hand_cue(stats: ClipStats) -> bool:
    joined = f"{stats.seq_id} {stats.object_id} {stats.text}".lower()
    if _contains(SELF_HAND_RE, joined):
        return False
    if stats.object_id.lower() in HANDHELD_OBJECTS and (
        stats.has_hand_action or stats.has_strong_tool_action
    ):
        return True
    return stats.has_hand_action and _contains(OBJECT_DIRECTED_HAND_RE, joined)


def _evaluate_clip(stats: ClipStats) -> list[dict]:
    issues: list[dict] = []
    object_key = stats.object_id.lower()

    if (
        object_key in HANDHELD_OBJECTS
        and stats.has_strong_tool_action
        and stats.action not in {"pickup_putdown"}
        and stats.any_hand < 0.85
    ):
        issues.append(_issue(
            stats,
            "handheld_strong_tool_low_hand",
            "major",
            "Caption/object imply continuous handheld tool use, but hand contact is low.",
            "any_hand >= 0.85",
            f"any_hand={stats.any_hand:.3f}",
        ))

    if (
        object_key in HANDHELD_OBJECTS
        and stats.action in {"hold", "carry"}
        and stats.any_hand < 0.45
    ):
        issues.append(_issue(
            stats,
            "handheld_hold_carry_low_hand",
            "major",
            "Caption/object imply held handheld object, but hand contact is very low.",
            "any_hand >= 0.45",
            f"any_hand={stats.any_hand:.3f}",
        ))

    if (
        _has_object_directed_hand_cue(stats)
        and not stats.has_no_contact_hint
        and stats.any_hand < 0.08
    ):
        issues.append(_issue(
            stats,
            "caption_hand_action_near_zero_hand",
            "major",
            "Caption has a hand/object interaction verb, but hand contact is near zero.",
            "any_hand >= 0.08",
            f"any_hand={stats.any_hand:.3f}",
        ))

    if stats.has_sit and stats.support_sitting < 0.08 and stats.pelvis < 0.08:
        issues.append(_issue(
            stats,
            "caption_sit_low_sitting_or_pelvis",
            "major",
            "Caption mentions sitting/seated, but both sitting support and pelvis contact are near zero.",
            "support_sitting >= 0.08 or pelvis >= 0.08",
            f"support_sitting={stats.support_sitting:.3f}; pelvis={stats.pelvis:.3f}",
        ))

    if stats.has_foot_action and stats.any_foot < 0.04 and stats.any_hand < 0.20:
        issues.append(_issue(
            stats,
            "caption_foot_action_low_foot_and_hand",
            "major",
            "Caption mentions foot/leg interaction, but foot contact is near zero and no hand/object contact explains the interaction.",
            "any_foot >= 0.04 or any_hand >= 0.20",
            f"any_foot={stats.any_foot:.3f}; any_hand={stats.any_hand:.3f}",
        ))

    if stats.has_manip_action and not stats.has_sit and stats.any_contact > 0.20 and stats.phase_manip < 0.08:
        issues.append(_issue(
            stats,
            "caption_manip_action_low_manip_phase",
            "warn",
            "Caption has manipulation verbs and contact, but manipulation phase is very low.",
            "phase_manip >= 0.08",
            f"phase_manip={stats.phase_manip:.3f}; any_contact={stats.any_contact:.3f}",
        ))

    if not (stats.has_sit or stats.has_lean) and stats.support_sitting > 0.60:
        issues.append(_issue(
            stats,
            "label_high_sitting_without_caption_sit",
            "warn",
            "Labels are mostly sitting support, but caption does not mention sit/lean.",
            "caption contains sit/lean or support_sitting <= 0.60",
            f"support_sitting={stats.support_sitting:.3f}",
        ))

    if not stats.has_foot_action and stats.any_foot > 0.70:
        issues.append(_issue(
            stats,
            "label_high_foot_without_caption_foot",
            "warn",
            "Labels are mostly foot-object contact, but caption does not mention foot/leg/step/kick.",
            "caption contains foot cue or any_foot <= 0.70",
            f"any_foot={stats.any_foot:.3f}",
        ))

    return issues


def _write_csv(path: Path, rows: list[dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clip_row(s: ClipStats) -> dict:
    return {
        "subset": s.subset,
        "seq_id": s.seq_id,
        "object_id": s.object_id,
        "action": s.action,
        "num_frames": s.num_frames,
        "any_hand": round(s.any_hand, 4),
        "l_hand": round(s.l_hand, 4),
        "r_hand": round(s.r_hand, 4),
        "any_foot": round(s.any_foot, 4),
        "pelvis": round(s.pelvis, 4),
        "any_contact": round(s.any_contact, 4),
        "phase_manip": round(s.phase_manip, 4),
        "phase_stable": round(s.phase_stable, 4),
        "support_sitting": round(s.support_sitting, 4),
        "support_hand": round(s.support_hand, 4),
        "has_hand_action": int(s.has_hand_action),
        "has_strong_tool_action": int(s.has_strong_tool_action),
        "has_sit": int(s.has_sit),
        "has_foot_action": int(s.has_foot_action),
        "text": s.text,
    }


def _aggregate_by_class(clips: list[ClipStats]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[ClipStats]] = defaultdict(list)
    for clip in clips:
        grouped[(clip.subset, clip.object_id, clip.action)].append(clip)

    rows: list[dict] = []
    for (subset, obj, action), items in sorted(grouped.items()):
        total_frames = sum(s.num_frames for s in items) or 1

        def wmean(attr: str) -> float:
            return sum(getattr(s, attr) * s.num_frames for s in items) / total_frames

        rows.append({
            "subset": subset,
            "object_id": obj,
            "action": action,
            "n_clips": len(items),
            "total_frames": total_frames,
            "any_hand": round(wmean("any_hand"), 4),
            "any_foot": round(wmean("any_foot"), 4),
            "pelvis": round(wmean("pelvis"), 4),
            "any_contact": round(wmean("any_contact"), 4),
            "phase_manip": round(wmean("phase_manip"), 4),
            "support_sitting": round(wmean("support_sitting"), 4),
        })
    return rows


def _write_report(
    out_dir: Path,
    clips: list[ClipStats],
    issues: list[dict],
    by_rule: list[dict],
    root: Path,
    label: str,
) -> None:
    major = [i for i in issues if i["severity"] == "major"]
    warn = [i for i in issues if i["severity"] == "warn"]
    subsets = Counter(s.subset for s in clips)
    major_by_subset = Counter(i["subset"] for i in major)
    warn_by_subset = Counter(i["subset"] for i in warn)

    lines = [
        f"# {label} Caption Alignment Evaluation",
        "",
        f"Root: `{root}`",
        f"Label: `{label}`",
        "",
        f"Scanned clips: {len(clips)}",
        f"Major issues: {len(major)} ({len(major) / max(len(clips), 1):.2%})",
        f"Warnings: {len(warn)} ({len(warn) / max(len(clips), 1):.2%})",
        "",
        "## Subsets",
        "",
        "| subset | clips | major | warn |",
        "|---|---:|---:|---:|",
    ]
    for subset, n in sorted(subsets.items()):
        lines.append(
            f"| {subset} | {n} | {major_by_subset[subset]} | {warn_by_subset[subset]} |"
        )

    lines += [
        "",
        "## Rules",
        "",
        "| severity | rule | count |",
        "|---|---|---:|",
    ]
    for row in by_rule:
        lines.append(f"| {row['severity']} | {row['rule']} | {row['count']} |")

    lines += [
        "",
        "## Major Examples",
        "",
    ]
    for issue in major[:30]:
        text = issue["text"].replace("\n", " ")[:180]
        lines.append(
            f"- `{issue['subset']}/{issue['seq_id']}` "
            f"({issue['rule']}): {issue['observed']} | {text}"
        )

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    clips: list[ClipStats] = []
    issues: list[dict] = []
    for subset in args.subsets:
        meta_path = args.root / subset / "metadata.json"
        if not meta_path.exists():
            print(f"[skip] {subset}: missing metadata.json")
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for entry in metadata:
            stats = _load_clip_stats(args.root, subset, args.label, entry)
            if stats is None:
                continue
            clips.append(stats)
            issues.extend(_evaluate_clip(stats))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_rows = [_clip_row(s) for s in clips]
    _write_csv(
        args.output_dir / "per_clip.csv",
        clip_rows,
        clip_rows[0].keys() if clip_rows else [],
    )

    issue_fields = [
        "severity", "rule", "subset", "seq_id", "object_id", "action",
        "num_frames", "expected", "observed", "message", "text",
    ]
    _write_csv(args.output_dir / "issues.csv", issues, issue_fields)

    rule_counts = Counter((i["severity"], i["rule"]) for i in issues)
    by_rule = [
        {"severity": severity, "rule": rule, "count": count}
        for (severity, rule), count in sorted(rule_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    _write_csv(args.output_dir / "by_rule.csv", by_rule, ["severity", "rule", "count"])

    by_class = _aggregate_by_class(clips)
    _write_csv(args.output_dir / "by_class.csv", by_class, by_class[0].keys() if by_class else [])

    exclude = [
        {
            "subset": i["subset"],
            "seq_id": i["seq_id"],
            "rule": i["rule"],
            "observed": i["observed"],
            "text": i["text"],
        }
        for i in issues
        if i["severity"] == "major"
    ]
    (args.output_dir / "exclude_candidates.json").write_text(
        json.dumps(exclude, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _write_report(args.output_dir, clips, issues, by_rule, args.root, args.label)

    n_major = sum(1 for i in issues if i["severity"] == "major")
    n_warn = sum(1 for i in issues if i["severity"] == "warn")
    print(f"Scanned {len(clips)} clips. major={n_major}, warn={n_warn}")
    print(f"Saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
