"""Subset-balanced failure taxonomy (Round 9, Task 4).

Consumes the Task-2 baseline audit JSON and assigns per-clip failure
tags by metric thresholds. Tag definitions follow the round-9 spec.

Tags (multi-label per clip):
1. under_motion           — body/hand vel xGT well below 1.0
2. over_motion            — acc p95 / jerk p95 xGT well above 1.0
3. wrong_contact_direction— M2 onset OR M2 release direction <= 0 on average
4. root_drift             — root_drift_cm >> 0
5. anchor_realization_fail— anchor realization cm >> GT-side anatomy floor
6. metric_artifact        — low n_valid events OR many flicker/boundary
7. pseudo_label_event     — high data-side per-type onset/release anchor target err
8. object_geometry        — heuristic: imhd bat clip + previously-known
9. subset_specific        — per-subset dominant mode tag (computed at aggregate)

Outputs:
  analyses/2026-05-19_subset_balanced_failure_taxonomy.{json,md}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# Thresholds — calibrated to v18 norms (see round-7/8 baseline).
TAG_RULES = {
    "under_motion_body_vel_below": 0.50,
    "under_motion_hand_vel_below": 0.40,
    "over_motion_acc_p95_above": 1.20,
    "over_motion_jerk_p95_above": 1.20,
    "wrong_direction_m2_at_or_below": 0.0,
    "root_drift_cm_above": 30.0,
    "anchor_realization_cm_above": 30.0,
    "metric_artifact_n_valid_min": 3,
    "metric_artifact_flicker_pct_above": 0.30,
    "metric_artifact_boundary_pct_above": 0.30,
    "pseudo_label_anchor_target_err_above_cm": 25.0,
}


def _tags_for_row(row: dict[str, Any], rules: dict[str, float]) -> list[str]:
    tags: list[str] = []
    dyn = row.get("dyn", {})
    v2 = row.get("v2", {})
    geom = row.get("geom", {})
    if float(dyn.get("body_vel_xGT", 0.0)) < rules["under_motion_body_vel_below"] or \
       float(dyn.get("hand_vel_xGT", 0.0)) < rules["under_motion_hand_vel_below"]:
        tags.append("under_motion")
    if float(dyn.get("acc_p95_xGT", 0.0)) > rules["over_motion_acc_p95_above"] or \
       float(dyn.get("jerk_p95_xGT", 0.0)) > rules["over_motion_jerk_p95_above"]:
        tags.append("over_motion")
    m2_on = float(v2.get("M2_onset_direction_cm_per_frame_mean", 0.0))
    m2_re = float(v2.get("M2_release_direction_cm_per_frame_mean", 0.0))
    if m2_on <= rules["wrong_direction_m2_at_or_below"] and m2_re <= rules["wrong_direction_m2_at_or_below"]:
        tags.append("wrong_contact_direction")
    if float(geom.get("root_drift_cm", 0.0)) > rules["root_drift_cm_above"]:
        tags.append("root_drift")
    if float(geom.get("plan_anchor_contact_realization_cm", 0.0)) > rules["anchor_realization_cm_above"]:
        tags.append("anchor_realization_fail")
    n_events = int(v2.get("n_events_total", 0))
    n_valid = int(v2.get("n_valid_slope", 0))
    n_flicker = int(v2.get("n_flicker", 0))
    n_boundary = int(v2.get("n_boundary", 0))
    if n_valid < rules["metric_artifact_n_valid_min"] \
       or (n_events > 0 and n_flicker / max(n_events, 1) > rules["metric_artifact_flicker_pct_above"]) \
       or (n_events > 0 and n_boundary / max(n_events, 1) > rules["metric_artifact_boundary_pct_above"]):
        tags.append("metric_artifact")
    # Pseudo-label / event suspicion — high data-side anchor target err on any anchor type
    per_type = row.get("per_type_anchor_err", {})
    pseudo_flag = False
    for t_name, st in per_type.items():
        if not isinstance(st, dict):
            continue
        if float(st.get("mean", 0.0)) > rules["pseudo_label_anchor_target_err_above_cm"] and int(st.get("n", 0)) >= 2:
            pseudo_flag = True
            break
    if pseudo_flag:
        tags.append("pseudo_label_event")
    # Object geometry heuristic — imhd bat clips flagged from round 5
    seq_id = str(row.get("seq_id", ""))
    if "bat" in seq_id.lower() and row.get("subset") == "imhd":
        tags.append("object_geometry_imhd_bat")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_metric_v2_baseline_audit.json"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_failure_taxonomy.json"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("analyses/2026-05-19_subset_balanced_failure_taxonomy.md"))
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    rules = dict(TAG_RULES)

    # Average per-clip across seeds (use mean v2/dyn/geom for tagging stability)
    per_clip_agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = f"{r['subset']}|{r['seq_id']}"
        if key not in per_clip_agg:
            per_clip_agg[key] = {
                "subset": r["subset"],
                "seq_id": r["seq_id"],
                "text": r["text"],
                "seq_len": r["seq_len"],
                "v2_list": [],
                "dyn_list": [],
                "geom_list": [],
                "per_type_anchor_err": r["per_type_anchor_err"],
            }
        per_clip_agg[key]["v2_list"].append(r["v2"])
        per_clip_agg[key]["dyn_list"].append(r["dyn"])
        per_clip_agg[key]["geom_list"].append(r["geom"])

    def _avg_dict(dlist: list[dict[str, Any]]) -> dict[str, float]:
        keys = set(k for d in dlist for k in d.keys())
        return {k: float(np.mean([float(d.get(k, 0.0)) for d in dlist])) for k in keys}

    per_clip_tagged: list[dict[str, Any]] = []
    for key, info in per_clip_agg.items():
        avg_row = {
            "subset": info["subset"],
            "seq_id": info["seq_id"],
            "text": info["text"],
            "v2": _avg_dict(info["v2_list"]),
            "dyn": _avg_dict(info["dyn_list"]),
            "geom": _avg_dict(info["geom_list"]),
            "per_type_anchor_err": info["per_type_anchor_err"],
        }
        tags = _tags_for_row(avg_row, rules)
        per_clip_tagged.append({**avg_row, "tags": tags})

    # Aggregate
    counts_by_tag: dict[str, int] = defaultdict(int)
    counts_by_subset_tag: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in per_clip_tagged:
        for t in r["tags"]:
            counts_by_tag[t] += 1
            counts_by_subset_tag[r["subset"]][t] += 1

    # Per-subset dominant tag
    dominant_by_subset: dict[str, str] = {}
    for sname, sub_counts in counts_by_subset_tag.items():
        if not sub_counts:
            dominant_by_subset[sname] = "none"
            continue
        dominant_by_subset[sname] = max(sub_counts.items(), key=lambda kv: kv[1])[0]

    # Examples per category
    examples_per_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_clip_tagged:
        for t in r["tags"]:
            if len(examples_per_tag[t]) < 3:
                examples_per_tag[t].append({
                    "subset": r["subset"], "seq_id": r["seq_id"],
                    "text": r["text"][:100],
                })

    out_payload = {
        "source_audit": str(args.input),
        "rules": rules,
        "tag_counts_global": dict(counts_by_tag),
        "tag_counts_by_subset": {s: dict(c) for s, c in counts_by_subset_tag.items()},
        "dominant_tag_by_subset": dominant_by_subset,
        "examples_per_tag": dict(examples_per_tag),
        "per_clip_tagged": per_clip_tagged,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_payload, indent=2, default=float), encoding="utf-8")

    lines = [
        "# Subset-Balanced Failure Taxonomy (Round 9, Task 4)",
        "",
        f"- Source audit: `{args.input}`",
        f"- Number of clips: {len(per_clip_tagged)}",
        "",
        "## Tag thresholds",
        "",
    ]
    for k, v in rules.items():
        lines.append(f"- `{k}` = {v}")
    lines += [
        "",
        "## Tag counts (global)",
        "",
        "| tag | n_clips |",
        "|-----|---------|",
    ]
    for t, c in sorted(counts_by_tag.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {t} | {c} |")
    lines += [
        "",
        "## Tag counts by subset",
        "",
        "| subset | dominant tag | tags (count) |",
        "|--------|--------------|--------------|",
    ]
    for sname in counts_by_subset_tag:
        dom = dominant_by_subset[sname]
        tag_str = ", ".join(
            f"{t}={c}" for t, c in sorted(counts_by_subset_tag[sname].items(), key=lambda kv: -kv[1])
        )
        lines.append(f"| {sname} | {dom} | {tag_str} |")
    lines += [
        "",
        "## Examples per tag",
        "",
    ]
    for t, exs in sorted(examples_per_tag.items()):
        lines.append(f"### {t}")
        for ex in exs:
            lines.append(f"- `{ex['subset']}/{ex['seq_id']}` — {ex['text']}")
        lines.append("")
    lines += [
        "## Per-clip tags",
        "",
        "| subset | seq_id | M2 onset | body vel xGT | acc p95 xGT | root drift cm | anchor realiz cm | tags |",
        "|--------|--------|----------|---------------|--------------|----------------|------------------|------|",
    ]
    for r in per_clip_tagged:
        lines.append(
            f"| {r['subset']} | {r['seq_id']} | "
            f"{r['v2'].get('M2_onset_direction_cm_per_frame_mean', 0.0):+.3f} | "
            f"{r['dyn'].get('body_vel_xGT', 0.0):.3f} | "
            f"{r['dyn'].get('acc_p95_xGT', 0.0):.3f} | "
            f"{r['geom'].get('root_drift_cm', 0.0):.1f} | "
            f"{r['geom'].get('plan_anchor_contact_realization_cm', 0.0):.1f} | "
            f"{', '.join(r['tags']) if r['tags'] else 'ok'} |"
        )
    lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
