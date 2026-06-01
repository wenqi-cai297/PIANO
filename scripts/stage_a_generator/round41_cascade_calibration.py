"""R41 cascade calibration — actual-gradient-ratio probe for each cell.

For every R41 cell (``configs/training/stage1_r41_*.yaml`` by default)
this script invokes ``round41_stage1_cascade_p0_diag.py
--calibration-only`` to measure the **actual cascade gradient ratio**:

    grad_norm(actual cascade stack at this cfg's weights -> Stage-1)
    /
    grad_norm(Stage-1 self loss -> Stage-1)

That ratio drives a linear ``w_total`` recommendation:

    new_w_total = current_w_total * target_center / measured_ratio

target_center is 1.0 by default; in-band is [0.5, 1.5]; >3.0 trips
the R36 disaster guard. The recommendation respects the actual stack
the trainer would compute (Codex review §2: loss ratio is a misleading
proxy because PB1's Jacobian is in the chain).

A loss-ratio is also recorded for cross-reference but is **not** used
for the recommendation.

Inputs
------
Default: scans configs/training/stage1_r41_*.yaml. Use ``--cfgs`` to
restrict.

Output
------
analyses/round41_cascade_calibration/<stamp>.md  +  .json

Run
---
On the server (PB1 ckpt + GPU + omegaconf required):

  python -u scripts/stage_a_generator/round41_cascade_calibration.py \\
      --out-dir analyses/round41_cascade_calibration
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# Target band: cascade weighted contribution as a fraction of mse_x0.
# Below this band cascade signal is too weak to move the optimization
# direction; above it cascade dominates and Stage-1 self loss can't
# anchor the model (= R36 disaster shape).
DEFAULT_TARGET_MIN = 0.5
DEFAULT_TARGET_MAX = 1.5
DEFAULT_TARGET_CENTER = 1.0

# R36 disaster threshold: if measured ratio exceeds this, recommend a
# much harder rescale + flag for user review.
DEFAULT_ABORT_RATIO = 3.0

# (Loss-based stdout parsing removed; calibration now reads grad ratios
# from the P0 diagnostic's JSON output, see _extract_calibration_metrics.)


def _run_p0_calibration(
    cfg_path: Path,
    *,
    stage1_ckpt: Path,
    pb1_ckpt: Path,
    out_dir: Path,
    bucket: str = "val",
    batch_size: int = 16,
) -> tuple[int, dict[str, Any], str]:
    """Invoke ``round41_stage1_cascade_p0_diag.py --calibration-only`` for
    one cfg, then read the resulting p0_stats.json.

    The PB1 cfg is read from the target cfg's cascade.pb1_config field
    (the cfg generator already wrote it there). Single source of
    truth — Codex review §4.

    Returns: (rc, p0_stats_dict, stdout_tail)
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(cfg_path))
    casc = cfg.get("cascade", None)
    pb1_cfg_path = (
        str(casc.get("pb1_config", "")) if casc else ""
    )
    if not pb1_cfg_path:
        # Control cell — no cascade. Run P0 anyway so we still get
        # batch_contract + warm_start + grad_path verified.
        pb1_cfg_path = "configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml"
    cell_dir = out_dir / f"p0_{cfg_path.stem}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u",
        "scripts/stage_a_generator/round41_stage1_cascade_p0_diag.py",
        "--stage1-config", str(cfg_path),
        "--stage1-ckpt", str(stage1_ckpt),
        "--pb1-config", pb1_cfg_path,
        "--pb1-ckpt", str(pb1_ckpt),
        "--out-dir", str(cell_dir),
        "--bucket", bucket,
        "--batch-size", str(int(batch_size)),
        "--calibration-only",
    ]
    print(f"[calib] running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    p0_json = cell_dir / "p0_stats.json"
    stats: dict[str, Any] = {}
    if p0_json.exists():
        try:
            stats = json.loads(p0_json.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[calib] WARN couldn't parse {p0_json}: {exc}")
    return proc.returncode, stats, proc.stdout[-3000:]


def _extract_calibration_metrics(p0_stats: dict[str, Any]) -> dict[str, Any]:
    """Pull the grad-ratio metrics from P0 stats.

    Returns dict with:
      - ratio_present: bool
      - ratio_actual_cascade_over_self: float | None
      - grad_norm_stage1_self
      - grad_norm_actual_cascade_weighted
      - cascade_weighted_value
      - component_loss_values (informational)
      - recommended_w_total_for_ratio_1 (from check 10)
    """
    out: dict[str, Any] = {"ratio_present": False}
    checks = p0_stats.get("checks", {})
    c10 = checks.get("grad_scale_actual_stack", {})
    if not c10:
        return out
    ratio = c10.get("ratio_actual_cascade_over_self")
    if ratio is None:
        return out
    out["ratio_present"] = True
    out["ratio_actual_cascade_over_self"] = float(ratio)
    out["grad_norm_stage1_self"] = float(
        c10.get("grad_norm_stage1_self", 0.0)
    )
    out["grad_norm_actual_cascade_weighted"] = float(
        c10.get("grad_norm_actual_cascade_weighted", 0.0)
    )
    out["cascade_weighted_value"] = float(
        c10.get("cascade_weighted_value", 0.0)
    )
    out["component_loss_values"] = c10.get("component_loss_values", {})
    out["recommended_w_total_p0"] = c10.get(
        "recommended_w_total_for_ratio_1"
    )
    out["cascade_weights_at_probe"] = c10.get("cascade_weights", {})
    return out


def _recommend_w_total(
    measured_ratio: float,
    *,
    target_center: float = DEFAULT_TARGET_CENTER,
    target_min: float = DEFAULT_TARGET_MIN,
    target_max: float = DEFAULT_TARGET_MAX,
    abort_ratio: float = DEFAULT_ABORT_RATIO,
    current_w_total: float = 1.0,
) -> dict[str, Any]:
    """Suggest a new ``cascade.w_total`` that brings the ratio into band.

    The relationship is linear: ratio scales linearly with w_total
    (cascade_weighted = w_total * sum(w_k * loss_k)).
    So new_w_total = current * (target_center / measured_ratio).
    """
    if measured_ratio <= 0:
        return {
            "in_band": False,
            "exceeds_abort": False,
            "recommended_w_total": current_w_total,
            "rec_reason": (
                "measured ratio is 0 — cascade likely off, no rescale needed"
            ),
        }
    in_band = target_min <= measured_ratio <= target_max
    exceeds_abort = measured_ratio > abort_ratio
    rec_w = current_w_total * (target_center / measured_ratio)
    # Round to 2 significant figures for cfg readability.
    if rec_w >= 1.0:
        rec_w_round = round(rec_w, 2)
    elif rec_w >= 0.01:
        rec_w_round = round(rec_w, 3)
    else:
        rec_w_round = float(f"{rec_w:.2e}")
    if in_band:
        reason = f"in band [{target_min}, {target_max}] — keep current w_total"
        rec_w_round = current_w_total
    elif exceeds_abort:
        reason = (
            f"ratio {measured_ratio:.2f} > abort threshold {abort_ratio} — "
            f"R36-style scale-dominate risk. Rescale w_total to {rec_w_round:.4g}"
        )
    else:
        reason = (
            f"ratio {measured_ratio:.3f} outside band — "
            f"scale w_total to {rec_w_round:.4g}"
        )
    return {
        "in_band": in_band,
        "exceeds_abort": exceeds_abort,
        "recommended_w_total": rec_w_round,
        "rec_reason": reason,
    }


def _write_summary_md(
    out_md: Path, stamp: str, rows: list[dict[str, Any]],
    target_min: float, target_max: float, target_center: float,
    abort_ratio: float,
) -> None:
    lines = [
        "# R41 cascade calibration",
        "",
        f"- stamp: {stamp}",
        f"- target band: actual grad ratio ∈ [{target_min}, {target_max}] "
        f"(center {target_center})",
        f"- abort threshold: actual grad ratio > {abort_ratio}",
        "- Recommendation is based on the **actual cascade gradient ratio** "
        "(grad_norm(weighted cascade)/grad_norm(self loss)), measured by "
        "round41_stage1_cascade_p0_diag.py --calibration-only.",
        "",
        "## Headline",
        "",
        "| cell | p0 rc | actual grad ratio | current w_total | "
        "recommended w_total | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("p0_rc") != 0:
            status = "✗ P0 crashed"
        elif not r.get("ratio_present"):
            status = "✗ no grad ratio (cascade disabled?)"
        elif r["recommendation"].get("exceeds_abort"):
            status = "⚠ exceeds abort"
        elif r["recommendation"].get("in_band"):
            status = "✓ in band"
        else:
            status = "↻ rescale"
        ratio = (
            f"{r.get('ratio_actual_cascade_over_self', 0.0):.3f}"
            if r.get("ratio_present") else "?"
        )
        cur_w = r.get("current_w_total")
        rec_w = r["recommendation"].get("recommended_w_total", cur_w)
        lines.append(
            f"| {r['vid']} | {r.get('p0_rc', '?')} | {ratio} | "
            f"{cur_w} | {rec_w} | {status} |"
        )

    lines.extend([
        "",
        "## Per-cell detail",
        "",
    ])
    for r in rows:
        lines.append(f"### {r['vid']}")
        lines.append("")
        lines.append(f"- cfg: `{r['cfg']}`")
        lines.append(f"- P0 rc: {r.get('p0_rc')}")
        if r.get("p0_rc") == 0:
            if r.get("ratio_present"):
                lines.append(
                    f"- grad_norm(stage1 self): "
                    f"{r.get('grad_norm_stage1_self', '?'):.4e}"
                )
                lines.append(
                    f"- grad_norm(actual cascade weighted): "
                    f"{r.get('grad_norm_actual_cascade_weighted', '?'):.4e}"
                )
                lines.append(
                    f"- **actual grad ratio**: "
                    f"{r.get('ratio_actual_cascade_over_self', '?'):.4f}"
                )
                lines.append(
                    f"- cascade_weighted_value (loss, informational): "
                    f"{r.get('cascade_weighted_value', '?')}"
                )
                lines.append(
                    f"- component loss values (informational): "
                    f"{r.get('component_loss_values', {})}"
                )
                lines.append(
                    f"- cascade weights at probe: "
                    f"{r.get('cascade_weights_at_probe', {})}"
                )
                lines.append("")
                lines.append(f"  **{r['recommendation']['rec_reason']}**")
            else:
                lines.append(
                    "- (no grad_scale_actual_stack — likely a control cell "
                    "or P0 didn't complete check 10)"
                )
        else:
            lines.append(f"- p0 stdout tail:")
            lines.append("  ```")
            for ln in (r.get("p0_stdout_tail") or "").splitlines()[-15:]:
                lines.append(f"  {ln}")
            lines.append("  ```")
        lines.append("")

    lines.extend([
        "## Next step",
        "",
        "1. Review the table. Cells marked ✓ in-band can be trained as-is.",
        "2. Cells marked ↻ or ⚠ need their cfg's `cascade.w_total` updated.",
        "   Run:",
        "",
        f"        python scripts/stage_a_generator/round41_apply_calibration.py \\",
        f"            --calibration analyses/round41_cascade_calibration/{stamp}.json",
        "",
        "3. Re-run this calibration script to verify the new w_total brings the",
        "   ratio into band.",
        "4. Once all cells are ✓ in-band, launch the matrix with",
        "   `run_round41_stage1_cascade_matrix.sh` (it will skip calibration",
        "   because that's been done by this script).",
        "",
    ])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _load_current_w_total(cfg_path: Path) -> float:
    """Read cascade.w_total from a cfg file. Returns 1.0 if absent or
    cascade is disabled."""
    try:
        from omegaconf import OmegaConf  # imported lazily so script can be -h'd locally
    except ImportError:
        return 1.0
    try:
        cfg = OmegaConf.load(str(cfg_path))
        casc = cfg.get("cascade", None)
        if casc is None:
            return 1.0
        return float(casc.get("w_total", 1.0))
    except Exception:
        return 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfgs", type=str, default=None,
        help="Comma-separated list of cfgs to calibrate. Default: glob "
             "configs/training/stage1_r41_*.yaml",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("analyses/round41_cascade_calibration"),
    )
    ap.add_argument(
        "--target-min", type=float, default=DEFAULT_TARGET_MIN,
    )
    ap.add_argument(
        "--target-max", type=float, default=DEFAULT_TARGET_MAX,
    )
    ap.add_argument(
        "--target-center", type=float, default=DEFAULT_TARGET_CENTER,
    )
    ap.add_argument(
        "--abort-ratio", type=float, default=DEFAULT_ABORT_RATIO,
    )
    ap.add_argument(
        "--stage1-ckpt", type=Path,
        default=Path("runs/training/stage1_v8_v6_full_f1/final.pt"),
        help="Stage-1 warm-start ckpt (V8 V6).",
    )
    ap.add_argument(
        "--pb1-ckpt", type=Path,
        default=Path(
            "runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt"
        ),
        help="PB1 frozen ckpt. Must match each cfg's cascade.pb1_checkpoint "
             "(launcher verifies this).",
    )
    ap.add_argument(
        "--bucket", choices=["train", "val"], default="val",
        help="Dataset bucket for the calibration smoke (val is cheaper).",
    )
    ap.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for the calibration smoke (default 16 fits single "
             "GPU; gradient ratio is bs-invariant).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve cfgs.
    if args.cfgs:
        cfgs = [Path(p.strip()) for p in args.cfgs.split(",") if p.strip()]
    else:
        cfgs = sorted(Path("configs/training").glob("stage1_r41_*.yaml"))
    if not cfgs:
        print("[calib] no cfgs to calibrate")
        return 1
    print(f"[calib] {len(cfgs)} cells: {[str(p.stem) for p in cfgs]}")

    rows: list[dict[str, Any]] = []
    for cfg_path in cfgs:
        vid = cfg_path.stem
        current_w = _load_current_w_total(cfg_path)
        rc, p0_stats, stdout_tail = _run_p0_calibration(
            cfg_path,
            stage1_ckpt=Path(args.stage1_ckpt),
            pb1_ckpt=Path(args.pb1_ckpt),
            out_dir=args.out_dir,
            bucket=args.bucket,
            batch_size=int(args.batch_size),
        )
        metrics = _extract_calibration_metrics(p0_stats)
        row: dict[str, Any] = {
            "vid": vid,
            "cfg": str(cfg_path),
            "current_w_total": current_w,
            "p0_rc": rc,
            "p0_stdout_tail": stdout_tail if rc != 0 else None,
            **metrics,
        }
        if rc == 0 and metrics.get("ratio_present"):
            row["recommendation"] = _recommend_w_total(
                metrics["ratio_actual_cascade_over_self"],
                target_center=args.target_center,
                target_min=args.target_min,
                target_max=args.target_max,
                abort_ratio=args.abort_ratio,
                current_w_total=current_w,
            )
        else:
            row["recommendation"] = {
                "in_band": False,
                "exceeds_abort": False,
                "recommended_w_total": current_w,
                "rec_reason": (
                    "P0 failed or grad_scale_actual_stack missing — "
                    "review p0_stdout_tail"
                ),
            }
        rows.append(row)

    out_json = args.out_dir / f"{stamp}.json"
    out_md = args.out_dir / f"{stamp}.md"
    out_json.write_text(
        json.dumps({
            "stamp": stamp,
            "target_min": args.target_min,
            "target_max": args.target_max,
            "target_center": args.target_center,
            "abort_ratio": args.abort_ratio,
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(
        out_md, stamp, rows,
        target_min=args.target_min, target_max=args.target_max,
        target_center=args.target_center, abort_ratio=args.abort_ratio,
    )
    print(f"[calib] wrote {out_md}")
    print(f"[calib] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
