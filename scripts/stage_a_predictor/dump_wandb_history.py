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


def _parse_columns(raw: list[str] | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for chunk in raw:
        for col in chunk.split(","):
            col = col.strip()
            if col and col not in seen:
                out.append(col)
                seen.add(col)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run", type=str, nargs="?", default=None,
        help="wandb run URL or '<entity>/<project>/<run_id>' path. "
             "Optional if --name is provided.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="wandb run NAME (the human-readable display name set by "
             "logging.run_name in the training config, e.g. "
             "'predictor_stageB_v04_normalize'). When given, the script "
             "queries wandb for the most recent run with this name in the "
             "given --project and uses that. Mutually exclusive with the "
             "positional run path.",
    )
    parser.add_argument(
        "--project", type=str, default="piano",
        help="wandb project to search when --name is used. Default: 'piano'.",
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help="wandb entity (username or team) to search when --name is "
             "used. Default: the wandb default for the logged-in user.",
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
        "--columns",
        action="append",
        default=None,
        help=(
            "comma-separated allow-list of columns to write. Can be repeated. "
            "When omitted, the full wandb history is exported."
        ),
    )
    parser.add_argument(
        "--print-summary", action="store_true",
        help="also print run summary (final values + state) to stdout",
    )
    args = parser.parse_args()

    if args.run is None and args.name is None:
        parser.error("must provide either positional `run` (URL/path) or --name")
    if args.run is not None and args.name is not None:
        parser.error("--name is mutually exclusive with positional `run`")

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. `pip install wandb`.", file=sys.stderr)
        return 2

    api = wandb.Api()
    if args.name is not None:
        # Look up by display name in the given project. wandb's `runs()`
        # filter is server-side; the order_by gives latest first so we
        # take runs[0]. There can be multiple runs with the same name
        # (re-runs); we pick the most recent.
        entity = args.entity or api.default_entity
        if entity is None:
            print("ERROR: no entity given and wandb has no default. Pass --entity.",
                  file=sys.stderr)
            return 1
        scope = f"{entity}/{args.project}"
        print(f"Searching wandb {scope} for runs with display_name = {args.name!r}...")
        try:
            matches = api.runs(
                scope, filters={"display_name": args.name}, order="-created_at",
            )
            matches_list = list(matches)
        except Exception as exc:  # pragma: no cover — network / auth
            print(f"ERROR: wandb query failed ({type(exc).__name__}): {exc}",
                  file=sys.stderr)
            return 1
        if not matches_list:
            print(f"ERROR: no runs in {scope} matched display_name {args.name!r}.",
                  file=sys.stderr)
            return 1
        run = matches_list[0]
        run_path = f"{run.entity}/{run.project}/{run.id}"
        print(f"Matched: {run_path} (state={run.state}, created={run.created_at})")
        if len(matches_list) > 1:
            print(f"  note: {len(matches_list)} runs share this name; using most recent.")
    else:
        run_path = _parse_run_path(args.run)
        print(f"Fetching wandb run: {run_path}")
        try:
            run = api.run(run_path)
        except Exception as exc:  # pragma: no cover — network / auth
            print(f"ERROR: could not load run ({type(exc).__name__}): {exc}",
                  file=sys.stderr)
            print("Hints: run `wandb login` once, and make sure the run id is correct.",
                  file=sys.stderr)
            return 1

    history_kwargs: dict = {}
    if args.samples is not None:
        history_kwargs["samples"] = args.samples
    df = run.history(**history_kwargs)
    columns = _parse_columns(args.columns)
    if columns:
        keep = [c for c in columns if c in df.columns]
        if "_step" in df.columns and "_step" not in keep:
            keep.insert(0, "_step")
        missing = [c for c in columns if c not in df.columns]
        df = df.loc[:, keep]
        if missing:
            print(
                "Warning: requested columns absent from run history: "
                + ", ".join(missing),
                file=sys.stderr,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {args.output}")

    if args.print_summary:
        print()
        print("=== Run summary ===")
        print(f"  state:    {run.state}")
        print(f"  runtime:  {run.summary.get('_runtime', 'n/a')} s")
        print(f"  step:     {run.summary.get('_step', 'n/a')}")
        summary_filter = set(columns)
        for k in sorted(run.summary.keys()):
            if k.startswith("_"):
                continue
            if summary_filter and k not in summary_filter:
                continue
            v = run.summary[k]
            print(f"  {k:<28s} {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
