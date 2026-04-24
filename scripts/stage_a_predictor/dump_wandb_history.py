"""Dump a wandb run's full history to CSV.

The wandb web UI is a SPA — opening the run URL in a browser shows
loss curves, but ``WebFetch`` (and other static scrapers) only get
the page shell. The wandb public API can return the full history
DataFrame given the run path; this script wraps that call so the
user can pull a CSV with one command.

Accepts either a wandb run URL or the canonical path:
    https://wandb.ai/<entity>/<project>/runs/<run_id>?nw=...
    <entity>/<project>/<run_id>

Output: a CSV with one row per logged step (or per epoch, depending
on what the run logged) with all logged columns.

Usage:
    python scripts/stage_a_predictor/dump_wandb_history.py \\
        https://wandb.ai/wenqicai297-university-of-toyama/piano/runs/xgy5q1az \\
        --output runs/wandb_logs/wandb_history_v2.csv

Requires ``wandb`` installed and ``wandb login`` already done (the
user's own run is always accessible).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_URL_RE = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)",
)


def _parse_run_path(s: str) -> str:
    """Return the canonical ``<entity>/<project>/<run_id>`` form.

    Accepts the full URL (with or without query string) or the bare
    ``entity/project/run_id`` triple.
    """
    s = s.strip()
    if (m := _URL_RE.match(s)) is not None:
        return f"{m['entity']}/{m['project']}/{m['run_id']}"
    parts = s.split("/")
    if len(parts) == 3 and all(parts):
        return s
    raise ValueError(
        f"could not parse wandb run path: {s!r}. Expected either a "
        f"wandb.ai run URL or '<entity>/<project>/<run_id>'."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run", type=str,
        help="wandb run URL or '<entity>/<project>/<run_id>' path",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="output CSV path",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="cap on number of history rows fetched (default: all)",
    )
    parser.add_argument(
        "--print-summary", action="store_true",
        help="also print run summary (final values + state) to stdout",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. `pip install wandb`.", file=sys.stderr)
        return 2

    run_path = _parse_run_path(args.run)
    print(f"Fetching wandb run: {run_path}")

    api = wandb.Api()
    try:
        run = api.run(run_path)
    except Exception as exc:  # pragma: no cover — network / auth
        print(f"ERROR: could not load run ({type(exc).__name__}): {exc}", file=sys.stderr)
        print("Hints: run `wandb login` once, and make sure the run id is correct.",
              file=sys.stderr)
        return 1

    history_kwargs: dict = {}
    if args.samples is not None:
        history_kwargs["samples"] = args.samples
    df = run.history(**history_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {args.output}")

    if args.print_summary:
        print()
        print("=== Run summary ===")
        print(f"  state:    {run.state}")
        print(f"  runtime:  {run.summary.get('_runtime', 'n/a')} s")
        print(f"  step:     {run.summary.get('_step', 'n/a')}")
        for k in sorted(run.summary.keys()):
            if k.startswith("_"):
                continue
            v = run.summary[k]
            print(f"  {k:<28s} {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
