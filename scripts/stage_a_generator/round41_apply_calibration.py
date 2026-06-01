"""R41 cascade — apply calibration result by patching ``cascade.w_total``.

Reads the JSON produced by ``round41_cascade_calibration.py`` and
updates the ``cascade.w_total`` field in each cell's yaml to the
recommended value. Cells already in-band (✓) are not modified. Cells
that failed smoke are not modified — user must fix the cfg before
re-calibrating.

This is the only script that mutates committed config files. By
default it shows a dry-run preview; pass ``--apply`` to actually write.

After applying, re-run calibration to confirm the new w_total brings
ratio into band. Once all cells are ✓ in-band, launch training.

Run
---

  python scripts/stage_a_generator/round41_apply_calibration.py \\
      --calibration analyses/round41_cascade_calibration/<stamp>.json \\
      --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibration", type=Path, required=True,
        help="Path to <stamp>.json written by round41_cascade_calibration.py",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually mutate the yaml files. Without this flag, "
             "the script only prints what it would do.",
    )
    ap.add_argument(
        "--force-in-band", action="store_true",
        help="Also overwrite cells already in-band (default: skip ✓ cells).",
    )
    args = ap.parse_args()

    payload = json.loads(args.calibration.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        print(f"[apply] no rows in {args.calibration}")
        return 1

    n_apply = 0
    n_skip = 0
    n_skip_failed = 0
    for row in rows:
        vid = row["vid"]
        cfg_path = Path(row["cfg"])
        rec = row.get("recommendation", {})
        current = float(row.get("current_w_total", 1.0))
        new_w = float(rec.get("recommended_w_total", current))
        in_band = bool(rec.get("in_band", False))

        if row.get("smoke_rc") != 0 or not row.get("ratio_present"):
            print(f"[apply] {vid}: SKIP (smoke failed or no cascade line)")
            n_skip_failed += 1
            continue

        if in_band and not args.force_in_band:
            print(f"[apply] {vid}: skip (in band, w_total stays {current})")
            n_skip += 1
            continue

        if abs(new_w - current) < 1e-9:
            print(f"[apply] {vid}: skip (recommended == current = {current})")
            n_skip += 1
            continue

        action = "WOULD WRITE" if not args.apply else "WRITING"
        print(
            f"[apply] {vid}: {action} cascade.w_total {current} → {new_w} "
            f"({cfg_path})"
        )

        if args.apply:
            cfg = OmegaConf.load(str(cfg_path))
            if cfg.get("cascade", None) is None:
                print(f"[apply] {vid}: WARN cfg has no cascade block, skipping")
                continue
            cfg.cascade.w_total = float(new_w)
            OmegaConf.save(cfg, cfg_path)
            n_apply += 1

    print()
    if args.apply:
        print(f"[apply] DONE — patched {n_apply} cells "
              f"(skipped {n_skip} in-band, {n_skip_failed} failed)")
    else:
        print(f"[apply] DRY-RUN — would patch {n_apply} cells "
              f"(would skip {n_skip} in-band, {n_skip_failed} failed)")
        print("[apply] re-run with --apply to actually mutate the yamls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
