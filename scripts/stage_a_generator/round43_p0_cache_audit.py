"""R43 P0 cache preflight + per-channel distribution audit.

Run AFTER ``sample_substitute_conds_cli.py`` populates the flat
generated cache ``<cache_root>/<subset>/<seq_id>.npz`` and BEFORE
Stage-1.5 training begins. Two responsibilities:

1. **Preflight integrity check.** For every (subset, seq_id) in the
   union of the train and val selection JSONs, assert:

   - the npz file exists at ``cache_root/<subset>/<seq_id>.npz``;
   - the key ``stage1_coarse`` is present;
   - the shape is ``(T, 23)`` for some T;
   - all values are finite.

   This is the same contract enforced at training time by
   ``stage1p5_cond_sources.load_generated_coarse_z_for_batch``; the
   audit just front-loads the failure to a 5-minute step instead of
   3 hours into Stage-1.5 training (Codex r43_stage_a_code_review §1
   adopt-verbatim).

2. **Subject-split disjointness defense.** Concat the train + val
   selection lists. If any ``(subset, seq_id)`` appears in both,
   fail loudly. Cause is almost always either: the train + val
   selection JSONs were dumped at different ``subject_split.seed``,
   or someone manually edited a selection. Either way, training
   would silently see the same clip in two modes.

3. **Per-channel mean/std report vs the oracle Stage-1 normalizer.**
   The cache values are *generated z-scored* coarse, so each channel
   should sit near mean 0 / std 1 if the Stage-1 sampler produces
   distribution-faithful output. Large gaps indicate the Stage-1 ckpt
   has its own bias; we report them so the operator can decide
   whether to proceed.

Outputs
-------

``<out_dir>/round43_p0_cache_audit.md`` — human-readable summary.
``<out_dir>/round43_p0_cache_audit.json`` — machine-readable detail.

Exit code 0 on green, 1 on any hard fail (missing/dup/non-finite).

Usage
-----

    python scripts/stage_a_generator/round43_p0_cache_audit.py \\
        --cache-root analyses/round43_stage1_substitute_conds_a2_<stamp> \\
        --sel-train  analyses/round43_full_selection_train.json \\
        --sel-val    analyses/round43_full_selection_val.json \\
        --oracle-norm cache/stage1_coarse_v1_full \\
        --out-dir analyses/round43_p0_cache_audit_<stamp>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from piano.data.stage1_coarse_oracle import load_stage1_coarse_norm


_EXPECTED_C = 23


def _read_selection(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text("utf-8"))
    sel = (
        payload.get("selected")
        or payload.get("candidates")
        or payload.get("clips")
        or []
    )
    return [(str(e["subset"]), str(e["seq_id"])) for e in sel]


def _check_dup_across_buckets(
    train_rows: list[tuple[str, str]],
    val_rows: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return any (subset, seq_id) appearing in both bucket selections.

    Empty result = subject-split is properly disjoint.
    """
    train_set = set(train_rows)
    val_set = set(val_rows)
    return sorted(train_set & val_set)


def _check_within_bucket_dup(
    rows: list[tuple[str, str]],
) -> list[tuple[tuple[str, str], int]]:
    """Return duplicate entries within a single bucket selection."""
    c = Counter(rows)
    return sorted(
        ((key, count) for key, count in c.items() if count > 1),
        key=lambda kc: (-kc[1], kc[0]),
    )


def _scan_cache_entry(
    cache_root: Path, subset: str, seq_id: str,
) -> dict[str, Any]:
    """Return per-entry diagnostic + (T,23) array or error.

    Output keys:
      - exists, has_key, shape_ok, finite_ok, T
      - error: str | None
      - per_channel_mean, per_channel_std : list[float] | None
    """
    path = cache_root / subset / f"{seq_id}.npz"
    out: dict[str, Any] = {
        "subset": subset, "seq_id": seq_id, "path": str(path),
        "exists": False, "has_key": False, "shape_ok": False,
        "finite_ok": False, "T": None, "error": None,
        "per_channel_mean": None, "per_channel_std": None,
    }
    if not path.is_file():
        out["error"] = "missing_file"
        return out
    out["exists"] = True
    try:
        with np.load(path) as data:
            if "stage1_coarse" not in data.files:
                out["error"] = f"missing_key keys={list(data.files)}"
                return out
            out["has_key"] = True
            arr = np.asarray(data["stage1_coarse"], dtype=np.float32)
    except Exception as exc:
        out["error"] = f"read_error: {exc!r}"
        return out
    if arr.ndim != 2 or arr.shape[1] != _EXPECTED_C:
        out["error"] = f"bad_shape {arr.shape}; expected (T, {_EXPECTED_C})"
        return out
    out["shape_ok"] = True
    out["T"] = int(arr.shape[0])
    if not np.isfinite(arr).all():
        out["error"] = "non_finite"
        return out
    out["finite_ok"] = True
    out["per_channel_mean"] = arr.mean(axis=0).tolist()
    out["per_channel_std"] = arr.std(axis=0).tolist()
    return out


def _aggregate(
    entry_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-channel mean/std aggregated across all clips (length-weighted)."""
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    weights: list[float] = []
    for e in entry_stats:
        if e.get("per_channel_mean") is None or e.get("T") is None:
            continue
        means.append(np.asarray(e["per_channel_mean"], dtype=np.float32))
        stds.append(np.asarray(e["per_channel_std"], dtype=np.float32))
        weights.append(float(e["T"]))
    if not means:
        return {
            "n_entries_used": 0,
            "agg_mean": None, "agg_std": None,
        }
    w = np.asarray(weights, dtype=np.float64)
    M = np.stack(means, axis=0).astype(np.float64)        # (N, 23)
    S = np.stack(stds, axis=0).astype(np.float64)         # (N, 23)
    agg_mean = (M * w[:, None]).sum(0) / w.sum()
    # Combine per-clip variances assuming clips are independent
    # samples of the same distribution; this is approximate but
    # good enough for an audit.
    agg_var = (((S ** 2 + M ** 2) * w[:, None]).sum(0) / w.sum()
               - agg_mean ** 2)
    agg_std = np.sqrt(np.clip(agg_var, 0.0, None))
    return {
        "n_entries_used": int(M.shape[0]),
        "agg_mean": agg_mean.astype(np.float32).tolist(),
        "agg_std": agg_std.astype(np.float32).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--sel-train", type=Path, required=True)
    ap.add_argument("--sel-val", type=Path, required=True)
    ap.add_argument(
        "--oracle-norm", type=Path, required=True,
        help=(
            "Path to cache/stage1_coarse_v1_full or equivalent; used to "
            "load the oracle z-score mean/std (load_stage1_coarse_norm). "
            "The generated cache should sit near oracle mean 0 / std 1; "
            "deviations are reported."
        ),
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--fail-on-warnings", action="store_true",
        help=(
            "Treat per-channel distribution gaps over the warning "
            "thresholds (|mean| > 0.3 OR |std/1 − 1| > 0.4) as hard "
            "failures. Default: warn only."
        ),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_md = args.out_dir / "round43_p0_cache_audit.md"
    out_json = args.out_dir / "round43_p0_cache_audit.json"

    print(f"[audit] cache_root={args.cache_root}", flush=True)
    print(f"[audit] sel_train={args.sel_train}", flush=True)
    print(f"[audit] sel_val  ={args.sel_val}", flush=True)
    print(f"[audit] oracle_norm={args.oracle_norm}", flush=True)

    if not args.cache_root.is_dir():
        print(f"[audit] FATAL: --cache-root {args.cache_root} missing.",
              file=sys.stderr)
        return 1
    if not args.sel_train.is_file() or not args.sel_val.is_file():
        print("[audit] FATAL: selection JSONs missing.", file=sys.stderr)
        return 1

    train_rows = _read_selection(args.sel_train)
    val_rows = _read_selection(args.sel_val)
    print(
        f"[audit] selections: train={len(train_rows)}  val={len(val_rows)}",
        flush=True,
    )

    # ─── Disjointness check (Codex §1 adopt) ───────────────────────
    cross_dup = _check_dup_across_buckets(train_rows, val_rows)
    train_internal_dup = _check_within_bucket_dup(train_rows)
    val_internal_dup = _check_within_bucket_dup(val_rows)

    hard_fail = False
    if cross_dup:
        hard_fail = True
        print(
            f"[audit] FATAL: {len(cross_dup)} (subset, seq_id) keys "
            "appear in BOTH train and val selections. Subject-split "
            "is broken — check that both selections were dumped at "
            "the same subject_split.seed.",
            file=sys.stderr,
        )
        for s, q in cross_dup[:10]:
            print(f"[audit]   dup: subset={s!r} seq_id={q!r}", file=sys.stderr)
    if train_internal_dup:
        print(
            f"[audit] WARN: {len(train_internal_dup)} duplicates within "
            "train selection (will dedup for audit; sampler may have "
            "already deduped on its end).",
            file=sys.stderr,
        )
    if val_internal_dup:
        print(
            f"[audit] WARN: {len(val_internal_dup)} duplicates within "
            "val selection.",
            file=sys.stderr,
        )

    # ─── Per-entry scan ───────────────────────────────────────────
    all_rows = sorted(set(train_rows) | set(val_rows))
    print(f"[audit] scanning {len(all_rows)} unique entries…", flush=True)
    entry_stats: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    bad: list[dict[str, Any]] = []
    for i, (subset, seq_id) in enumerate(all_rows):
        s = _scan_cache_entry(args.cache_root, subset, seq_id)
        entry_stats.append(s)
        if not s["exists"]:
            missing.append((subset, seq_id))
        elif s.get("error"):
            bad.append({
                "subset": subset, "seq_id": seq_id, "error": s["error"],
            })
        if (i + 1) % 500 == 0:
            print(
                f"[audit]   visited {i + 1}/{len(all_rows)}  "
                f"missing={len(missing)}  bad={len(bad)}",
                flush=True,
            )

    if missing:
        hard_fail = True
        print(
            f"[audit] FATAL: {len(missing)} cache entries are missing.",
            file=sys.stderr,
        )
        for s, q in missing[:10]:
            print(f"[audit]   missing: {s!r}/{q!r}", file=sys.stderr)
    if bad:
        hard_fail = True
        print(f"[audit] FATAL: {len(bad)} cache entries are malformed.",
              file=sys.stderr)
        for b in bad[:10]:
            print(f"[audit]   bad: {b['subset']!r}/{b['seq_id']!r} "
                  f"error={b['error']}", file=sys.stderr)

    # ─── Aggregate distribution vs oracle normalizer ──────────────
    ok_stats = [e for e in entry_stats if e.get("finite_ok")]
    agg = _aggregate(ok_stats)

    oracle_mean = oracle_std = None
    distribution_warnings: list[dict[str, Any]] = []
    if agg["agg_mean"] is not None:
        try:
            oracle_mean_np, oracle_std_np = load_stage1_coarse_norm(
                str(args.oracle_norm)
            )
            oracle_mean = oracle_mean_np.tolist()
            oracle_std = oracle_std_np.tolist()
        except Exception as exc:
            print(f"[audit] WARN: oracle norm load failed: {exc!r}; "
                  "skipping distribution-vs-oracle gap analysis.")
        if oracle_mean is not None:
            am = np.asarray(agg["agg_mean"], dtype=np.float32)
            asd = np.asarray(agg["agg_std"], dtype=np.float32)
            # Generated cache is *z-scored*: agg should be ~(0, 1).
            # Oracle norm is the reference frame for that z-scoring;
            # we report agg ~ (0, 1) and also gap vs (0, 1) directly.
            mean_gap = np.abs(am - 0.0)
            std_gap = np.abs(asd - 1.0)
            for c in range(_EXPECTED_C):
                if float(mean_gap[c]) > 0.3 or float(std_gap[c]) > 0.4:
                    distribution_warnings.append({
                        "channel": c,
                        "agg_mean": float(am[c]),
                        "agg_std": float(asd[c]),
                        "mean_gap_vs_0": float(mean_gap[c]),
                        "std_gap_vs_1": float(std_gap[c]),
                    })

    if distribution_warnings:
        print(
            f"[audit] {len(distribution_warnings)} channel(s) deviate from "
            "z-score expectation (|mean| > 0.3 OR |std − 1| > 0.4). This "
            "is informational; Stage-1 sampling can have a bias and the "
            "Stage-1.5 R43 P0 training is meant to absorb it.",
        )
        if args.fail_on_warnings:
            hard_fail = True
            print("[audit] FATAL: --fail-on-warnings set + warnings present.",
                  file=sys.stderr)

    # ─── Write report ─────────────────────────────────────────────
    Ts = [e["T"] for e in entry_stats if e.get("T") is not None]
    payload: dict[str, Any] = {
        "cache_root": str(args.cache_root),
        "sel_train": str(args.sel_train),
        "sel_val": str(args.sel_val),
        "oracle_norm": str(args.oracle_norm),
        "selections": {
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_unique_total": len(all_rows),
            "cross_bucket_dup_count": len(cross_dup),
            "cross_bucket_dup_first_10": cross_dup[:10],
            "train_within_dup_count": len(train_internal_dup),
            "val_within_dup_count": len(val_internal_dup),
        },
        "cache": {
            "n_entries_scanned": len(entry_stats),
            "n_missing": len(missing),
            "n_malformed": len(bad),
            "T_min": int(min(Ts)) if Ts else None,
            "T_max": int(max(Ts)) if Ts else None,
            "T_mean": float(sum(Ts) / len(Ts)) if Ts else None,
        },
        "distribution": {
            "n_entries_used": agg["n_entries_used"],
            "agg_mean_per_channel": agg["agg_mean"],
            "agg_std_per_channel": agg["agg_std"],
            "oracle_norm_mean_per_channel": oracle_mean,
            "oracle_norm_std_per_channel": oracle_std,
            "warnings": distribution_warnings,
        },
        "hard_fail": bool(hard_fail),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown
    lines = [
        "# R43 P0 generated-cache audit",
        "",
        f"- cache_root: `{args.cache_root}`",
        f"- sel_train:  `{args.sel_train}`  ({len(train_rows)} rows)",
        f"- sel_val:    `{args.sel_val}`    ({len(val_rows)} rows)",
        f"- oracle_norm: `{args.oracle_norm}`",
        "",
        "## Disjointness (Codex r43_stage_a_code_review §1)",
        "",
        f"- cross-bucket duplicates: **{len(cross_dup)}**",
        f"- train within-bucket duplicates: {len(train_internal_dup)}",
        f"- val within-bucket duplicates:   {len(val_internal_dup)}",
        "",
        "## Cache integrity",
        "",
        f"- entries scanned: {len(entry_stats)}",
        f"- missing: **{len(missing)}**",
        f"- malformed: **{len(bad)}**",
        f"- T (seq length): min={payload['cache']['T_min']}  "
        f"max={payload['cache']['T_max']}  mean={payload['cache']['T_mean']}",
        "",
        "## Distribution (z-score expectation: per-channel mean ≈ 0, std ≈ 1)",
        "",
        f"- entries used for aggregation: {agg['n_entries_used']}",
        f"- channels with mean|0 gap > 0.3 OR std|1 gap > 0.4: "
        f"**{len(distribution_warnings)}**",
        "",
    ]
    if distribution_warnings:
        lines.append("| channel | agg_mean | agg_std | mean_gap_vs_0 | std_gap_vs_1 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for w in distribution_warnings:
            lines.append(
                f"| {w['channel']} | {w['agg_mean']:+.3f} | {w['agg_std']:.3f}"
                f" | {w['mean_gap_vs_0']:.3f} | {w['std_gap_vs_1']:.3f} |"
            )
        lines.append("")
    lines.extend([
        "## Verdict",
        "",
        f"- hard_fail: **{hard_fail}**",
        "",
        ("PASS — cache is ready for Stage-1.5 R43 P0 training."
         if not hard_fail else
         "FAIL — fix the issues above before training. Re-sample any "
         "missing clip(s), regenerate the selection JSONs at a "
         "consistent subject_split seed if cross-bucket dups exist, "
         "and re-run this audit."),
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[audit] wrote {out_md}")
    print(f"[audit] wrote {out_json}")
    # R44 verdict clarity (Codex r43_pipeline_bottleneck §4):
    # distinguish "no warnings" (PASS) from "warnings but caller chose
    # not to fail" (PASS-WITH-WARNINGS) from "warnings + fail-on-warnings
    # set" (FAIL). R43 P0 hit the silent middle case and trained on a
    # collapsed cache because the prior log said "verdict: PASS".
    if hard_fail:
        print(
            "[audit] verdict: FAIL "
            "(per-channel deviations exceed warning thresholds; "
            "--fail-on-warnings set, or missing/malformed cache entries)"
        )
    elif distribution_warnings:
        print(
            f"[audit] verdict: PASS-WITH-WARNINGS "
            f"({len(distribution_warnings)} channel(s) deviate; "
            "downstream training MUST treat this cache as OOD source. "
            "Use --fail-on-warnings to make this a hard failure.)"
        )
    else:
        print(
            "[audit] verdict: PASS "
            "(per-channel z-score within ±0.3 mean / ±0.4 std)"
        )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
