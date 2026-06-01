"""R41 cascade calibration — pre-training grad-scale probe for each cell.

Runs ``train_stage1.py --smoke-test`` once per R41 cell, parses the
``R41 cascade weighted/mse_x0=X.XXX`` line from stdout, and reports a
suggested ``cascade.w_total`` per cell that brings the ratio into the
target band [0.5, 1.5].

This separates the **measurement** of cascade vs stage1_self grad
scale from the **decision** of which w_total to ship. After this script
writes its report, the user reviews and then runs
``round41_apply_calibration.py`` to mutate the cfgs (and re-runs
calibration to verify before launching the training matrix).

Why separate from the launcher
------------------------------
R36 / R37 / R40 all failed in part because per-cell calibration was
either skipped or done in-band (parsed during training launch). Doing
it as a standalone read-only phase means:

  - the user sees all 5 cells' ratios in one table before any decision;
  - the recommended w_total comes with the actual measured ratio so
    the user can sanity-check the rec;
  - the launcher stays simple (no log-parsing inside a tee'd shell).

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
import re
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

# Regex on the smoke test's stdout. Matches the line emitted by
# train_stage1.py main() smoke-test path:
#   "  R41 cascade weighted=2.3456e-01  weighted/mse_x0=0.123"
RATIO_PATTERN = re.compile(
    r"R41\s+cascade\s+weighted\s*=\s*([\d.eE+-]+)\s+weighted/mse_x0\s*=\s*([\d.eE+-]+)"
)
MSE_X0_PATTERN = re.compile(r"^\s*mse_x0\s*=\s*([\d.eE+-]+)", re.MULTILINE)
COMPONENT_PATTERN = re.compile(
    r"motion_mse\s*=\s*([\d.eE+-]+)\s+world_vel\s*=\s*([\d.eE+-]+)\s+"
    r"l_pos\s*=\s*([\d.eE+-]+)\s+anchor\s*=\s*([\d.eE+-]+)"
)


def _run_smoke_test(cfg_path: Path) -> tuple[int, str]:
    """Run train_stage1.py --smoke-test once. Returns (rc, stdout+stderr)."""
    cmd = [
        sys.executable, "-u", "src/piano/training/train_stage1.py",
        "--config", str(cfg_path), "--smoke-test",
    ]
    print(f"[calib] running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def _parse_smoke_output(text: str) -> dict[str, Any]:
    """Extract cascade ratio + per-component numbers from smoke stdout."""
    out: dict[str, Any] = {"ratio_present": False}
    m = RATIO_PATTERN.search(text)
    if m:
        out["ratio_present"] = True
        out["cascade_weighted"] = float(m.group(1))
        out["ratio_weighted_over_mse_x0"] = float(m.group(2))
    mc = COMPONENT_PATTERN.search(text)
    if mc:
        out["component_motion_mse"] = float(mc.group(1))
        out["component_world_joint_vel"] = float(mc.group(2))
        out["component_l_pos_full"] = float(mc.group(3))
        out["component_anchor_joint_pos"] = float(mc.group(4))
    mx = MSE_X0_PATTERN.search(text)
    if mx:
        out["mse_x0"] = float(mx.group(1))
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
        f"- target band: ratio ∈ [{target_min}, {target_max}] (center {target_center})",
        f"- abort threshold: ratio > {abort_ratio}",
        "",
        "## Headline",
        "",
        "| cell | smoke rc | ratio | current w_total | recommended w_total | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("smoke_rc") != 0:
            status = "✗ smoke crashed"
        elif not r.get("ratio_present"):
            status = "✗ no R41 line (cascade off?)"
        elif r["recommendation"].get("exceeds_abort"):
            status = "⚠ exceeds abort"
        elif r["recommendation"].get("in_band"):
            status = "✓ in band"
        else:
            status = "↻ rescale"
        ratio = (
            f"{r.get('ratio_weighted_over_mse_x0', 0.0):.3f}"
            if r.get("ratio_present") else "?"
        )
        cur_w = r.get("current_w_total")
        rec_w = r["recommendation"].get("recommended_w_total", cur_w)
        lines.append(
            f"| {r['vid']} | {r.get('smoke_rc', '?')} | {ratio} | "
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
        lines.append(f"- smoke rc: {r.get('smoke_rc')}")
        if r.get("smoke_rc") == 0:
            lines.append(f"- mse_x0: {r.get('mse_x0', '?')}")
            if r.get("ratio_present"):
                lines.append(
                    f"- cascade_weighted: {r.get('cascade_weighted', '?')}"
                )
                lines.append(
                    f"- ratio (weighted / mse_x0): "
                    f"{r.get('ratio_weighted_over_mse_x0', '?')}"
                )
                lines.append(
                    f"- motion_mse: {r.get('component_motion_mse', '?')}"
                )
                lines.append(
                    f"- world_joint_vel: {r.get('component_world_joint_vel', '?')}"
                )
                lines.append(f"- l_pos_full: {r.get('component_l_pos_full', '?')}")
                lines.append(
                    f"- anchor_joint_pos: {r.get('component_anchor_joint_pos', '?')}"
                )
                lines.append("")
                lines.append(f"  **{r['recommendation']['rec_reason']}**")
            else:
                lines.append(
                    "- (no R41 cascade line — likely cascade disabled in this cfg)"
                )
        else:
            lines.append(f"- smoke stdout tail:")
            lines.append("  ```")
            for ln in (r.get("smoke_tail") or "").splitlines()[-15:]:
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
        rc, stdout = _run_smoke_test(cfg_path)
        parsed = _parse_smoke_output(stdout)
        row: dict[str, Any] = {
            "vid": vid,
            "cfg": str(cfg_path),
            "current_w_total": current_w,
            "smoke_rc": rc,
            "smoke_tail": stdout[-2000:] if rc != 0 else None,
            **parsed,
        }
        if rc == 0 and parsed.get("ratio_present"):
            row["recommendation"] = _recommend_w_total(
                parsed["ratio_weighted_over_mse_x0"],
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
                    "smoke failed or no R41 ratio line — review smoke stdout"
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
