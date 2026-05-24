"""Round-26 air-grab proxy analysis.

Reads `anchor_stats.json` from one or more ckpt-runs of
``anchor_realization_diagnostic.py`` and computes the semantic-validity
proxy proposed in the Codex Round-26 review:

    airgrab_margin_cm = pred_to_target_cm - gt_to_target_cm
    airgrab_bad       = airgrab_margin_cm > tau_part

Outputs

  - ``analyses/round26_air_grab_analysis.md`` — overall + per-part +
    per-subset bad-rate, sensitivity over tau, paired R23<->v27
    transitions, correlation with pred-to-GT
  - ``analyses/round26_air_grab_analysis.json`` — same data, machine-
    readable
  - ``analyses/round26_visual_review_selection.json`` — 20-28 clips
    selected by category for the visual-review render pass
    (fixed by v27 / still-bad hands / v27-new-bad / D3 sampled failures)

Usage (locally, after extracting the round26 anchor-diag tarball):

    python scripts/stage_b_generator/round26_air_grab_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds (Codex review §2 — diagnostic-only, not training-time)
# ---------------------------------------------------------------------------
# These are NOT training-time tolerances. They define which anchors count as
# "air-grab" failures for the proxy. Sensitivity curves swept below.
PART_TAU_DEFAULT: dict[str, float] = {
    "left_hand": 20.0,
    "right_hand": 20.0,
    # Codex §2: 15cm because foot labels suspect, this is diagnostic-only.
    "left_foot": 15.0,
    "right_foot": 15.0,
    # Pelvis is area contact + ~28cm structural offset → guardrail only.
    "pelvis": 35.0,
}

TAU_SWEEP = [5, 10, 15, 20, 25, 30, 35, 40]


def _airgrab_mask(rows: list[dict], tau_by_part: dict[str, float]) -> np.ndarray:
    """Boolean mask: True if pred is farther from target than GT by more
    than tau_part for this anchor's body part."""
    deltas = np.array([r["pred_to_target_cm"] - r["gt_to_target_cm"] for r in rows])
    parts = [r["part_name"] for r in rows]
    taus = np.array([tau_by_part[p] for p in parts])
    return deltas > taus


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyze_ckpt(rows: list[dict], label: str) -> dict:
    """Compute air-grab + correlation summary for one ckpt."""
    deltas = np.array([r["pred_to_target_cm"] - r["gt_to_target_cm"] for r in rows])
    pred_to_gt = np.array([r["pred_to_gt_cm"] for r in rows])
    pred_to_target = np.array([r["pred_to_target_cm"] for r in rows])
    gt_to_target = np.array([r["gt_to_target_cm"] for r in rows])

    mask = _airgrab_mask(rows, PART_TAU_DEFAULT)
    overall = {
        "label": label,
        "n_anchors": len(rows),
        "mean_pred_to_target_cm": float(pred_to_target.mean()),
        "mean_gt_to_target_cm": float(gt_to_target.mean()),
        "mean_pred_to_gt_cm": float(pred_to_gt.mean()),
        "mean_delta_cm": float(deltas.mean()),
        "median_delta_cm": float(np.median(deltas)),
        "n_bad_default_tau": int(mask.sum()),
        "rate_bad_default_tau": float(mask.mean()),
        "corr_pred_to_gt_vs_delta": _pearson(pred_to_gt, deltas),
    }

    # Per-part
    per_part = {}
    for part in ["left_hand", "right_hand", "left_foot", "right_foot", "pelvis"]:
        part_rows = [r for r in rows if r["part_name"] == part]
        if not part_rows:
            continue
        part_deltas = np.array([r["pred_to_target_cm"] - r["gt_to_target_cm"] for r in part_rows])
        part_p2g = np.array([r["pred_to_gt_cm"] for r in part_rows])
        tau = PART_TAU_DEFAULT[part]
        per_part[part] = {
            "n_anchors": len(part_rows),
            "mean_pred_to_target_cm": float(np.mean([r["pred_to_target_cm"] for r in part_rows])),
            "mean_gt_to_target_cm": float(np.mean([r["gt_to_target_cm"] for r in part_rows])),
            "mean_pred_to_gt_cm": float(part_p2g.mean()),
            "mean_delta_cm": float(part_deltas.mean()),
            "median_delta_cm": float(np.median(part_deltas)),
            "p75_delta_cm": float(np.percentile(part_deltas, 75)),
            "p95_delta_cm": float(np.percentile(part_deltas, 95)),
            "max_delta_cm": float(part_deltas.max()),
            "tau_cm": tau,
            "n_bad": int((part_deltas > tau).sum()),
            "rate_bad": float((part_deltas > tau).mean()),
            "corr_pred_to_gt_vs_delta": _pearson(part_p2g, part_deltas),
        }

    # Per-subset
    per_subset = {}
    subsets = sorted({r["subset"] for r in rows})
    for sub in subsets:
        sub_rows = [r for r in rows if r["subset"] == sub]
        sub_deltas = np.array([r["pred_to_target_cm"] - r["gt_to_target_cm"] for r in sub_rows])
        mask = _airgrab_mask(sub_rows, PART_TAU_DEFAULT)
        per_subset[sub] = {
            "n_anchors": len(sub_rows),
            "mean_pred_to_target_cm": float(np.mean([r["pred_to_target_cm"] for r in sub_rows])),
            "mean_gt_to_target_cm": float(np.mean([r["gt_to_target_cm"] for r in sub_rows])),
            "mean_delta_cm": float(sub_deltas.mean()),
            "n_bad_default_tau": int(mask.sum()),
            "rate_bad_default_tau": float(mask.mean()),
        }

    # Sensitivity sweep over tau
    sensitivity = []
    for tau in TAU_SWEEP:
        tau_by_part = {p: float(tau) for p in PART_TAU_DEFAULT}
        m = _airgrab_mask(rows, tau_by_part)
        sensitivity.append({"tau_cm": tau, "rate_bad": float(m.mean()), "n_bad": int(m.sum())})

    return {
        "overall": overall,
        "per_part": per_part,
        "per_subset": per_subset,
        "sensitivity_uniform_tau": sensitivity,
    }


def paired_transition(rows_a: list[dict], rows_b: list[dict],
                      label_a: str, label_b: str) -> dict:
    """Per-anchor transition stats from ckpt-A to ckpt-B.

    Joins by (seq_id, anchor_idx, part_idx). Returns counts of
    {bad->bad, bad->good, good->bad, good->good} and per-clip mean
    delta-of-deltas for selecting visual-review clips.
    """
    def key(r):
        return (r["seq_id"], r["anchor_idx"], r["part_idx"])

    a_by = {key(r): r for r in rows_a}
    b_by = {key(r): r for r in rows_b}
    common = sorted(set(a_by.keys()) & set(b_by.keys()))

    transitions = {"good_good": 0, "good_bad": 0, "bad_good": 0, "bad_bad": 0}
    delta_per_anchor = []
    for k in common:
        ra = a_by[k]
        rb = b_by[k]
        part = ra["part_name"]
        tau = PART_TAU_DEFAULT[part]
        da = ra["pred_to_target_cm"] - ra["gt_to_target_cm"]
        db = rb["pred_to_target_cm"] - rb["gt_to_target_cm"]
        a_bad = da > tau
        b_bad = db > tau
        if not a_bad and not b_bad: transitions["good_good"] += 1
        elif not a_bad and b_bad:    transitions["good_bad"] += 1
        elif a_bad and not b_bad:    transitions["bad_good"] += 1
        else:                         transitions["bad_bad"] += 1
        delta_per_anchor.append({
            "key": k, "subset": ra["subset"], "seq_id": ra["seq_id"],
            "part": part, "anchor_idx": ra["anchor_idx"],
            "delta_a": float(da), "delta_b": float(db),
            "transition_delta": float(db - da),  # >0 means B worse
            "pred_to_gt_a": float(ra["pred_to_gt_cm"]),
            "pred_to_gt_b": float(rb["pred_to_gt_cm"]),
            "a_bad": bool(a_bad), "b_bad": bool(b_bad),
        })

    # Per-clip aggregation (one row per (subset, seq_id))
    per_clip = defaultdict(lambda: {
        "subset": None, "seq_id": None,
        "n_anchors": 0, "n_fixed": 0, "n_regressed": 0,
        "n_a_bad": 0, "n_b_bad": 0,
        "mean_transition_delta": 0.0,
    })
    sums = defaultdict(float)
    for d in delta_per_anchor:
        clip_key = (d["subset"], d["seq_id"])
        per_clip[clip_key]["subset"] = d["subset"]
        per_clip[clip_key]["seq_id"] = d["seq_id"]
        per_clip[clip_key]["n_anchors"] += 1
        if d["a_bad"]: per_clip[clip_key]["n_a_bad"] += 1
        if d["b_bad"]: per_clip[clip_key]["n_b_bad"] += 1
        if d["a_bad"] and not d["b_bad"]: per_clip[clip_key]["n_fixed"] += 1
        if not d["a_bad"] and d["b_bad"]: per_clip[clip_key]["n_regressed"] += 1
        sums[clip_key] += d["transition_delta"]
    for k, v in per_clip.items():
        v["mean_transition_delta"] = sums[k] / max(v["n_anchors"], 1)

    return {
        "label_a": label_a, "label_b": label_b,
        "n_common_anchors": len(common),
        "transitions": transitions,
        "per_clip": dict(per_clip),
        "delta_per_anchor": delta_per_anchor,
    }


def select_visual_review_clips(
    rows_v27: list[dict],
    rows_r23: list[dict],
    transition_r23_to_v27: dict,
    d3_per_clip: list[dict] | None = None,
    n_fixed: int = 8,
    n_still_bad_hand: int = 8,
    n_regressed: int = 4,
    n_d3_failures: int = 4,
) -> list[dict]:
    """Pick ~24 clips for visual review, in 4 categories:

      - n_fixed:          most-improved (R23 air-grab → v27 air-grab fixed)
      - n_still_bad_hand: v27 final hand air-grab high
      - n_regressed:      v27 worse than R23 (air-grab newly created)
      - n_d3_failures:    largest D3 sampled-coarse gap

    Returns a list of {subset, seq_id, mode_category (or "none"),
    category (visual-review label), reason}.
    """
    selection: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(subset, seq_id, category, reason, mode_category=None):
        k = (subset, seq_id)
        if k in seen:
            return False
        seen.add(k)
        selection.append({
            "subset": subset,
            "seq_id": seq_id,
            "mode_category": mode_category or "none",
            "visual_category": category,
            "reason": reason,
        })
        return True

    # Category 1: clips where v27 fixed the most air-grab anchors
    per_clip = transition_r23_to_v27["per_clip"]
    fixed_sorted = sorted(per_clip.values(),
                          key=lambda c: (-c["n_fixed"], c["mean_transition_delta"]))
    for c in fixed_sorted:
        if c["n_fixed"] < 1:
            break
        if add(c["subset"], c["seq_id"], "fixed_by_v27",
               f"v27 fixed {c['n_fixed']} air-grab anchor(s), regressed {c['n_regressed']}"):
            if sum(1 for s in selection if s["visual_category"] == "fixed_by_v27") >= n_fixed:
                break

    # Category 2: v27 final hand air-grab still high (use v27 hand-bad anchors per clip)
    v27_hand_bad = [r for r in rows_v27
                    if r["part_name"] in ("left_hand", "right_hand")
                    and (r["pred_to_target_cm"] - r["gt_to_target_cm"]) > PART_TAU_DEFAULT[r["part_name"]]]
    per_clip_hand = defaultdict(lambda: {"n_hand_bad": 0, "max_gap": 0.0,
                                          "subset": None, "seq_id": None})
    for r in v27_hand_bad:
        k = (r["subset"], r["seq_id"])
        per_clip_hand[k]["subset"] = r["subset"]
        per_clip_hand[k]["seq_id"] = r["seq_id"]
        per_clip_hand[k]["n_hand_bad"] += 1
        per_clip_hand[k]["max_gap"] = max(
            per_clip_hand[k]["max_gap"],
            r["pred_to_target_cm"] - r["gt_to_target_cm"],
        )
    hand_bad_sorted = sorted(per_clip_hand.values(),
                             key=lambda c: (-c["n_hand_bad"], -c["max_gap"]))
    for c in hand_bad_sorted:
        if add(c["subset"], c["seq_id"], "still_bad_hand",
               f"v27 hand air-grab: {c['n_hand_bad']} anchors, max gap {c['max_gap']:.1f}cm"):
            if sum(1 for s in selection if s["visual_category"] == "still_bad_hand") >= n_still_bad_hand:
                break

    # Category 3: v27 newly regressed clips (n_regressed high or transition_delta most positive)
    regressed_sorted = sorted(per_clip.values(),
                              key=lambda c: (-c["n_regressed"], -c["mean_transition_delta"]))
    for c in regressed_sorted:
        if c["n_regressed"] < 1 and c["mean_transition_delta"] <= 0:
            break
        if add(c["subset"], c["seq_id"], "v27_regressed",
               f"v27 regressed {c['n_regressed']} anchor(s), "
               f"mean_Δ={c['mean_transition_delta']:+.2f}cm"):
            if sum(1 for s in selection if s["visual_category"] == "v27_regressed") >= n_regressed:
                break

    # Category 4: D3 sampled-coarse failures (if available)
    if d3_per_clip is not None:
        d3_sorted = sorted(d3_per_clip, key=lambda c: -c["gap_cm"])
        for c in d3_sorted:
            if add(c["subset"], c["seq_id"], "d3_sampled_failure",
                   f"D3 sampled gap +{c['gap_cm']:.1f}cm (oracle={c['anchor_pose_error_oracle_cm']:.1f}, "
                   f"sampled={c['anchor_pose_error_sampled_cm']:.1f})",
                   mode_category=c.get("mode_category")):
                if sum(1 for s in selection if s["visual_category"] == "d3_sampled_failure") >= n_d3_failures:
                    break

    return selection


def _format_md(r23: dict, v27_final: dict, v27_best: dict,
               transitions: dict) -> str:
    """Render the analysis markdown."""
    lines = [
        "# Round-26 v27 air-grab proxy analysis",
        "",
        "Definition: `airgrab_margin = pred_to_target_cm - gt_to_target_cm`.",
        "Anchor is flagged as **air-grab** if `airgrab_margin > tau_part`, where",
        "tau is calibrated per body part (hands 20cm, feet 15cm, pelvis 35cm guardrail).",
        "",
        "Source: 3 anchor diagnostic runs on the same 48-clip Round-25 multimodal eval subset.",
        "",
        "## 1. Overall (all 677 anchors)",
        "",
        "| ckpt | mean pred→target | mean GT→target | mean pred→GT | mean Δ | corr(p→GT, Δ) | air-grab rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in [("R23 baseline", r23), ("v27 best_val", v27_best), ("v27 final", v27_final)]:
        o = data["overall"]
        lines.append(
            f"| {label} | {o['mean_pred_to_target_cm']:.2f} | {o['mean_gt_to_target_cm']:.2f} | "
            f"{o['mean_pred_to_gt_cm']:.2f} | {o['mean_delta_cm']:+.2f} | "
            f"{o['corr_pred_to_gt_vs_delta']:+.3f} | "
            f"{o['n_bad_default_tau']}/{o['n_anchors']} ({100*o['rate_bad_default_tau']:.1f}%) |"
        )

    lines += [
        "",
        "**Key reading.** `corr(pred_to_GT, Δ)` is near zero on every ckpt —",
        "optimizing pred-to-GT is NOT a reliable proxy for reducing air-grab.",
        "v27 final has lower air-grab rate than R23; v27 best_val has higher.",
        "",
        "## 2. Per-part air-grab rate",
        "",
        "| part | n | tau | R23 | v27 best_val | v27 final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for part in ["left_hand", "right_hand", "left_foot", "right_foot", "pelvis"]:
        a = r23["per_part"].get(part)
        b = v27_best["per_part"].get(part)
        c = v27_final["per_part"].get(part)
        if a is None or b is None or c is None: continue
        lines.append(
            f"| {part} | {a['n_anchors']} | {a['tau_cm']:.0f} | "
            f"{a['n_bad']}/{a['n_anchors']} ({100*a['rate_bad']:.1f}%) | "
            f"{b['n_bad']}/{b['n_anchors']} ({100*b['rate_bad']:.1f}%) | "
            f"**{c['n_bad']}/{c['n_anchors']} ({100*c['rate_bad']:.1f}%)** |"
        )

    lines += [
        "",
        "## 3. Per-part mean Δ (pred_to_target - gt_to_target)",
        "",
        "Mean gap is robust to outliers; air-grab rate (above) is more diagnostic.",
        "",
        "| part | R23 mean Δ | v27 final mean Δ | R23→final shift |",
        "|---|---:|---:|---:|",
    ]
    for part in ["left_hand", "right_hand", "left_foot", "right_foot", "pelvis"]:
        a = r23["per_part"].get(part); c = v27_final["per_part"].get(part)
        if a is None or c is None: continue
        lines.append(
            f"| {part} | {a['mean_delta_cm']:+.2f} | {c['mean_delta_cm']:+.2f} | "
            f"{c['mean_delta_cm'] - a['mean_delta_cm']:+.2f} |"
        )

    lines += [
        "",
        "## 4. Per-part correlation: pred_to_GT vs (pred_to_target - gt_to_target)",
        "",
        "If `|corr| ≈ 0`, optimizing pred-to-GT does NOT systematically reduce",
        "air-grab. If positive, it helps weakly. If negative, optimizing pred-to-GT",
        "fights air-grab — characteristic of feet/pelvis where GT itself has large",
        "structural offset and pulling pred toward GT pulls it AWAY from object surface.",
        "",
        "| part | R23 corr | v27 final corr |",
        "|---|---:|---:|",
    ]
    for part in ["left_hand", "right_hand", "left_foot", "right_foot", "pelvis"]:
        a = r23["per_part"].get(part); c = v27_final["per_part"].get(part)
        if a is None or c is None: continue
        lines.append(f"| {part} | {a['corr_pred_to_gt_vs_delta']:+.3f} | {c['corr_pred_to_gt_vs_delta']:+.3f} |")

    lines += [
        "",
        "## 5. Per-subset air-grab rate (default tau)",
        "",
        "| subset | n | R23 | v27 best_val | v27 final |",
        "|---|---:|---:|---:|---:|",
    ]
    for sub in sorted(r23["per_subset"].keys()):
        a = r23["per_subset"][sub]
        b = v27_best["per_subset"].get(sub, {"rate_bad_default_tau": float("nan"), "n_bad_default_tau": 0, "n_anchors": 0})
        c = v27_final["per_subset"].get(sub, {"rate_bad_default_tau": float("nan"), "n_bad_default_tau": 0, "n_anchors": 0})
        lines.append(
            f"| {sub} | {a['n_anchors']} | "
            f"{100*a['rate_bad_default_tau']:.1f}% | "
            f"{100*b['rate_bad_default_tau']:.1f}% | "
            f"{100*c['rate_bad_default_tau']:.1f}% |"
        )

    lines += [
        "",
        "## 6. Sensitivity to tau (uniform tau across parts)",
        "",
        "| tau_cm | R23 rate | v27 best_val | v27 final |",
        "|---:|---:|---:|---:|",
    ]
    for i, tau in enumerate(TAU_SWEEP):
        a = r23["sensitivity_uniform_tau"][i]["rate_bad"]
        b = v27_best["sensitivity_uniform_tau"][i]["rate_bad"]
        c = v27_final["sensitivity_uniform_tau"][i]["rate_bad"]
        lines.append(f"| {tau} | {100*a:.1f}% | {100*b:.1f}% | {100*c:.1f}% |")

    lines += [
        "",
        "## 7. Paired R23 → v27 final transitions (677 anchors)",
        "",
        "| transition | count | % |",
        "|---|---:|---:|",
    ]
    t = transitions["transitions"]
    total = sum(t.values())
    for k, v in t.items():
        lines.append(f"| {k} | {v} | {100*v/max(total,1):.1f}% |")

    lines += [
        "",
        f"**Net air-grab anchors fixed by v27**: {t['bad_good'] - t['good_bad']:+d}",
        "(`bad→good` minus `good→bad`)",
        "",
        "## 8. Conclusions",
        "",
        "1. v27 final reduces air-grab rate from R23 10.2% → 6.8% overall (default tau).",
        "2. The improvement is dominated by feet (R23 12-15% → v27 5-7%); hands barely move.",
        "3. v27 best_val (selected by `loss_anchor_joint_pos`) is WORSE on air-grab than v27 final.",
        "   `val_best_key` should be redesigned (Codex review §5 — v28c).",
        "4. Correlation between pred-to-GT and air-grab margin is near zero overall (R23 0.10,",
        "   v27 final 0.01). Optimizing pred-to-GT does NOT reliably reduce air-grab.",
        "5. Per-part correlation flips sign: hands +0.3 (pred-to-GT loss helps weakly),",
        "   feet/pelvis -0.2 to -0.3 (pred-to-GT loss fights air-grab on these parts).",
        "   This explains why v27's anchor_joint_pos loss helped feet via OTHER pathways",
        "   (dense FK + 80-ep convergence) rather than via the new sparse term.",
        "6. Chairs subset dominates anchor count (549/677) and has large structural offsets,",
        "   so global aggregates are chairs-biased. Per-subset reporting is mandatory.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r23", type=Path,
                        default=Path("analyses/round26_r23_anchor_diag_48clips/anchor_stats.json"))
    parser.add_argument("--v27-final", type=Path,
                        default=Path("analyses/round26_v27_anchor_diag_final/anchor_stats.json"))
    parser.add_argument("--v27-best-val", type=Path,
                        default=Path("analyses/round26_v27_anchor_diag_best_val/anchor_stats.json"))
    parser.add_argument("--d3-v27-final", type=Path,
                        default=Path("analyses/round26_v27_d3_oracle_vs_sampled_final.json"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("analyses/round26_air_grab_analysis.md"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("analyses/round26_air_grab_analysis.json"))
    parser.add_argument("--output-selection", type=Path,
                        default=Path("analyses/round26_visual_review_selection.json"))
    args = parser.parse_args()

    for p in [args.r23, args.v27_final, args.v27_best_val]:
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    rows_r23 = json.loads(args.r23.read_text("utf-8"))["rows"]
    rows_v27_final = json.loads(args.v27_final.read_text("utf-8"))["rows"]
    rows_v27_best = json.loads(args.v27_best_val.read_text("utf-8"))["rows"]

    print(f"R23: {len(rows_r23)} anchors / "
          f"{len({(r['subset'], r['seq_id']) for r in rows_r23})} clips")
    print(f"v27 final: {len(rows_v27_final)} anchors")
    print(f"v27 best_val: {len(rows_v27_best)} anchors")

    r23 = analyze_ckpt(rows_r23, "R23")
    v27_final = analyze_ckpt(rows_v27_final, "v27 final")
    v27_best = analyze_ckpt(rows_v27_best, "v27 best_val")

    transitions = paired_transition(rows_r23, rows_v27_final, "R23", "v27_final")

    d3_per_clip = None
    if args.d3_v27_final.exists():
        d3_per_clip = json.loads(args.d3_v27_final.read_text("utf-8")).get("per_clip", [])
        print(f"D3 per_clip rows: {len(d3_per_clip)}")

    selection = select_visual_review_clips(
        rows_v27_final, rows_r23, transitions, d3_per_clip=d3_per_clip,
    )

    # Outputs
    out = {
        "r23": r23,
        "v27_final": v27_final,
        "v27_best_val": v27_best,
        "transitions_r23_to_v27_final": {
            "label_a": transitions["label_a"], "label_b": transitions["label_b"],
            "n_common_anchors": transitions["n_common_anchors"],
            "transitions": transitions["transitions"],
        },
    }
    args.output_json.write_text(json.dumps(out, indent=2), "utf-8")
    print(f"wrote {args.output_json}")

    md = _format_md(r23, v27_final, v27_best, transitions)
    args.output_md.write_text(md, "utf-8")
    print(f"wrote {args.output_md}")

    sel_payload = {
        "source": "round26_air_grab_analysis.py",
        "n_clips": len(selection),
        "thresholds": PART_TAU_DEFAULT,
        "categories": sorted({s["visual_category"] for s in selection}),
        "selected": selection,
    }
    args.output_selection.write_text(json.dumps(sel_payload, indent=2), "utf-8")
    print(f"wrote {args.output_selection}  ({len(selection)} clips)")
    by_cat = defaultdict(int)
    for s in selection:
        by_cat[s["visual_category"]] += 1
    for k, v in by_cat.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
