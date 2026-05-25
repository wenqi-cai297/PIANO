"""Summarize the Round-28 A-group ablation matrix.

This is intentionally a report helper, not a training/eval runner. It reads
the diagnostic JSON files produced by ``run_round28_train.sh`` and writes a
compact table that compares same-train overfit diagnostics against true
held-out val diagnostics. The winner should be selected from held-out rows,
with train rows used only as "can the model consume this hint?" evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = [
    "r28_a0_input_add",
    "r28_a1_gated_input",
    "r28_a1b_gated_input_open",
    "r28_a2_per_layer_adapter",
    "r28_a2b_adapter_only",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(n: float | None, den: float | None) -> float | None:
    if n is None or den is None or den == 0:
        return None
    return 100.0 * float(n) / float(den)


def _mean_body_delta(body: dict[str, Any] | None) -> float | None:
    if not body:
        return None
    vals = []
    for joint_stats in body.get("aggregate", {}).values():
        v = joint_stats.get("delta_error_cm_mean")
        if v is not None:
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def _row(variant: str, split: str, tag: str, root: Path) -> dict[str, Any] | None:
    if split == "train":
        d = root / f"round28_{variant}_diag_{tag}"
    else:
        d = root / f"round28_{variant}_heldout_val_diag_{tag}"
    sustained = _load_json(d / "sustained_contact_stats.json")
    gait = _load_json(d / "gait_stats.json")
    body = _load_json(d / "body_action_stats.json")
    if not sustained and not gait and not body:
        return None

    overall = sustained.get("overall", {}) if sustained else {}
    n_segments = overall.get("n_segments")
    tracking = overall.get("tracking_fraction", {})
    pred_gait = gait.get("pred_aggregate", {}) if gait else {}
    gt_gait = gait.get("gt_aggregate", {}) if gait else {}

    pred_both_swing = pred_gait.get("frac_both_swing", {}).get("mean")
    gt_both_swing = gt_gait.get("frac_both_swing", {}).get("mean")
    pred_trans = pred_gait.get("transitions_per_second", {}).get("mean")
    gt_trans = gt_gait.get("transitions_per_second", {}).get("mean")
    pred_corr = pred_gait.get("L_R_height_corr", {}).get("mean")
    gt_corr = gt_gait.get("L_R_height_corr", {}).get("mean")

    return {
        "variant": variant,
        "split": split,
        "tag": tag,
        "n_clips": sustained.get("n_clips_processed") if sustained else None,
        "drift_max_cm_mean": overall.get("drift_max_cm", {}).get("mean"),
        "drift_gt10_rate_pct": _pct(overall.get("n_drift_max_above_10cm"), n_segments),
        "tracking_lt05_pct": (
            100.0 * float(tracking["rate_below_0.5"])
            if tracking.get("rate_below_0.5") is not None else None
        ),
        "gait_both_swing_gap": (
            abs(float(pred_both_swing) - float(gt_both_swing))
            if pred_both_swing is not None and gt_both_swing is not None else None
        ),
        "gait_transitions_gap": (
            abs(float(pred_trans) - float(gt_trans))
            if pred_trans is not None and gt_trans is not None else None
        ),
        "gait_corr_gap": (
            abs(float(pred_corr) - float(gt_corr))
            if pred_corr is not None and gt_corr is not None else None
        ),
        "body_delta_cm_mean": _mean_body_delta(body),
    }


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _sort_value(v: Any) -> float:
    return 1e9 if v is None else float(v)


def _write_md(rows: list[dict[str, Any]], out: Path) -> None:
    lines = [
        "# Round-28 A-Group Ablation Matrix Summary",
        "",
        "Use held-out val rows to choose the best injection mechanism. "
        "Train rows only show whether the branch can consume the oracle hint "
        "on the memorized 48-clip diagnostic subset.",
        "",
        "| split | ckpt | variant | drift mean cm | drift>10% | track<0.5% | gait both-swing gap | gait trans gap | gait corr gap | body delta cm |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {split} | {tag} | {variant} | {drift} | {drift10} | "
            "{track} | {swing} | {trans} | {corr} | {body} |".format(
                split=r["split"],
                tag=r["tag"],
                variant=r["variant"],
                drift=_fmt(r["drift_max_cm_mean"]),
                drift10=_fmt(r["drift_gt10_rate_pct"], 1),
                track=_fmt(r["tracking_lt05_pct"], 1),
                swing=_fmt(r["gait_both_swing_gap"], 3),
                trans=_fmt(r["gait_transitions_gap"], 3),
                corr=_fmt(r["gait_corr_gap"], 3),
                body=_fmt(r["body_delta_cm_mean"]),
            )
        )
    heldout_rows = [
        r for r in rows
        if r["split"] == "heldout_val" and r["drift_max_cm_mean"] is not None
    ]
    if heldout_rows:
        ranked = sorted(
            heldout_rows,
            key=lambda r: (
                _sort_value(r["drift_max_cm_mean"]),
                _sort_value(r["drift_gt10_rate_pct"]),
                _sort_value(r["tracking_lt05_pct"]),
                _sort_value(r["gait_both_swing_gap"]),
                _sort_value(r["body_delta_cm_mean"]),
            ),
        )
        lines += [
            "",
            "## Held-Out Contact Ranking",
            "",
            "Ranking is lexicographic: lower drift mean, drift>10%, "
            "tracking<0.5%, gait both-swing gap, then body delta. "
            "Treat this as a shortlist, not a substitute for visual review.",
            "",
        ]
        for i, r in enumerate(ranked, start=1):
            lines.append(
                f"{i}. `{r['variant']}` `{r['tag']}`: "
                f"drift={_fmt(r['drift_max_cm_mean'])} cm, "
                f"drift>10={_fmt(r['drift_gt10_rate_pct'], 1)}%, "
                f"track<0.5={_fmt(r['tracking_lt05_pct'], 1)}%, "
                f"both-swing gap={_fmt(r['gait_both_swing_gap'], 3)}, "
                f"body={_fmt(r['body_delta_cm_mean'])} cm"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("analyses"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("analyses/round28_a_group_ablation_matrix_summary.md"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("analyses/round28_a_group_ablation_matrix_summary.json"))
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--tags", nargs="*", default=["best_val", "final"])
    parser.add_argument("--splits", nargs="*", default=["train", "heldout_val"])
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for split in args.splits:
        for tag in args.tags:
            for variant in args.variants:
                row = _row(str(variant), str(split), str(tag), args.analysis_root)
                if row is not None:
                    rows.append(row)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_md(rows, args.output_md)
    print(f"wrote {len(rows)} rows to {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
