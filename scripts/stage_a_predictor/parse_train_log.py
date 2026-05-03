"""Parse a Stage A predictor launch.log into a wandb-history-compatible CSV.

When training is run offline (no `wandb login`), `dump_wandb_history.py`
can't reach the cloud API. But the launch log (tee'd via PowerShell or
bash) already contains every per-epoch train + val line in a format
that's just `key=value | key=value | ...` — easy to parse.

Output CSV columns mirror the wandb history schema we use elsewhere
(epoch, train loss_*, val_loss_*, _runtime, _step), so the same
analysis tooling (`compare_v9_versions.py`, ad-hoc awk scripts) works
unchanged.

Usage:
    python scripts/stage_a_predictor/parse_train_log.py \\
        --log runs/training/predictor_v9_5_finer_encoder_local_launch.log \\
        --output runs/wandb_logs/wandb_history_predictor_v9_5_finer_encoder_local.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Train line pattern: "Epoch 5/100 (148s) | loss=9.0247 | loss_unweighted=...|"
# Val line pattern  : "Val @ epoch 5 | val_loss=8.1519 | val_loss_unweighted=...|"
TRAIN_RE = re.compile(r"^\s*Epoch\s+(\d+)/(\d+)\s+\((\d+(?:\.\d+)?)s\)\s*\|\s*(.*)$")
VAL_RE = re.compile(r"^\s*Val\s+@\s+epoch\s+(\d+)\s*\|\s*(.*)$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _parse_kv(rest: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in KV_RE.findall(rest):
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def parse(log_path: Path) -> list[dict]:
    rows: dict[int, dict] = {}
    cumulative_runtime = 0.0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN_RE.match(line)
            if m:
                epoch = int(m.group(1))
                wall = float(m.group(3))
                kvs = _parse_kv(m.group(4))
                row = rows.setdefault(epoch, {"epoch": epoch})
                cumulative_runtime += wall
                row["_runtime"] = cumulative_runtime
                row["epoch_time_sec"] = wall
                for k, v in kvs.items():
                    row[k] = v
                continue
            m = VAL_RE.match(line)
            if m:
                epoch = int(m.group(1))
                kvs = _parse_kv(m.group(2))
                row = rows.setdefault(epoch, {"epoch": epoch})
                for k, v in kvs.items():
                    row[k] = v
    return [rows[k] for k in sorted(rows)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True,
                   help="path to the tee'd launch log")
    p.add_argument("--output", type=Path, required=True,
                   help="output CSV path")
    args = p.parse_args()
    if not args.log.exists():
        print(f"ERROR: log not found: {args.log}", file=sys.stderr)
        return 1
    rows = parse(args.log)
    if not rows:
        print(f"ERROR: no Epoch / Val lines found in {args.log}", file=sys.stderr)
        return 1
    # Union of all keys across rows (some columns are train-only, some
    # val-only) so the CSV header is complete.
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    # Column order: epoch first, then train cols, then val cols, then misc.
    train_keys = sorted(k for k in all_keys
                        if k.startswith("loss") or k in ("n_contact_frames", "prior_scale"))
    val_keys = sorted(k for k in all_keys if k.startswith("val_"))
    misc_keys = sorted(all_keys - {"epoch"} - set(train_keys) - set(val_keys))
    fieldnames = ["epoch"] + train_keys + val_keys + misc_keys
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    n_train = sum(1 for r in rows if "loss_unweighted" in r)
    n_val = sum(1 for r in rows if "val_loss_unweighted" in r)
    print(f"Wrote {args.output} ({len(rows)} epochs; {n_train} train, "
          f"{n_val} val)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
