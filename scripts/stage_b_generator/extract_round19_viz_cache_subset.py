"""Extract minimal cache subset for the Round-19 visual review (32 clips).

Server-side helper: produces TWO self-contained mini-caches (one for
Plan A and one for S1-O) containing ONLY the cache files needed by
``render_round19_visual_review.py`` for the 32 clips in the Round-19
eval selection JSON. Total size ~10-15 MB (vs the full caches at
~300-500 MB each).

Outputs:
    runs/training/round19_viz_cache_subset/
      stage1_coarse_v1_full/                          (Plan A mini-cache)
        text_embeddings_clip_vit_b32.npz
        text_embeddings_index.json
        normalization_train.json
        manifest_val.jsonl       (filtered to selection clips only)
        clips/<subset>/<safe_id>.npz  (32 files)
      stage1_coarse_v1_objtraj_root0_world_round18_fix/   (S1-O mini-cache, same layout)

Both mini-caches mirror the live cache structure so the local viz can
point at them with the same paths.

Usage
-----

    python scripts/stage_b_generator/extract_round19_viz_cache_subset.py \\
        --selection-json analyses/2026-05-20_round19_eval_selection.json \\
        --cache-plan-a cache/stage1_coarse_v1_full \\
        --cache-s1o cache/stage1_coarse_v1_objtraj_root0_world_round18_fix \\
        --output-root runs/training/round19_viz_cache_subset
    tar czf runs/training/round19_viz_cache_subset.tar.gz \\
        runs/training/round19_viz_cache_subset/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _extract_one_cache(
    cache_root: Path, out_root: Path, selection: list[dict],
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy global metadata files (small, ~6 MB total).
    for fname in (
        "text_embeddings_clip_vit_b32.npz",
        "text_embeddings_index.json",
        "normalization_train.json",
    ):
        src = cache_root / fname
        if not src.exists():
            raise FileNotFoundError(f"missing {src}")
        shutil.copy2(src, out_root / fname)

    # Filter manifest to selection clips and locate their npz files.
    src_manifest_path = cache_root / "manifest_val.jsonl"
    manifest = [
        json.loads(line)
        for line in src_manifest_path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    lookup = {(r["subset"], r["seq_id"]): r for r in manifest}

    kept_rows: list[dict] = []
    copied = 0
    missing: list[dict] = []
    for e in selection:
        key = (e["subset"], e["seq_id"])
        r = lookup.get(key)
        if r is None:
            missing.append({"subset": e["subset"], "seq_id": e["seq_id"]})
            continue
        src_npz = cache_root / r["npz_path"]
        if not src_npz.exists():
            missing.append({"subset": e["subset"], "seq_id": e["seq_id"]})
            continue
        dst_npz = out_root / r["npz_path"]
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_npz, dst_npz)
        kept_rows.append(r)
        copied += 1

    # Write filtered manifest.
    (out_root / "manifest_val.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in kept_rows),
        encoding="utf-8",
    )
    return {
        "cache_root": str(cache_root),
        "out_root": str(out_root),
        "n_selection": len(selection),
        "n_copied": copied,
        "n_missing": len(missing),
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-json", type=Path,
        default=Path("analyses/2026-05-20_round19_eval_selection.json"),
    )
    parser.add_argument(
        "--cache-plan-a", type=Path,
        default=Path("cache/stage1_coarse_v1_full"),
    )
    parser.add_argument(
        "--cache-s1o", type=Path,
        default=Path("cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("runs/training/round19_viz_cache_subset"),
    )
    args = parser.parse_args()

    sel = json.loads(args.selection_json.read_text(encoding="utf-8"))
    selection = sel.get("selected", sel)
    if not isinstance(selection, list):
        raise SystemExit(f"bad selection JSON: {args.selection_json}")

    print(f"[extract] selection: {len(selection)} clips")
    out_a = args.output_root / args.cache_plan_a.name
    out_o = args.output_root / args.cache_s1o.name

    print(f"[extract] Plan A: {args.cache_plan_a} -> {out_a}")
    stats_a = _extract_one_cache(args.cache_plan_a, out_a, selection)
    print(f"[extract]   copied {stats_a['n_copied']}/{stats_a['n_selection']} clips; missing {stats_a['n_missing']}")

    print(f"[extract] S1-O:   {args.cache_s1o} -> {out_o}")
    stats_o = _extract_one_cache(args.cache_s1o, out_o, selection)
    print(f"[extract]   copied {stats_o['n_copied']}/{stats_o['n_selection']} clips; missing {stats_o['n_missing']}")

    # Manifest of what we extracted.
    (args.output_root / "_extract_manifest.json").write_text(json.dumps({
        "selection_json": str(args.selection_json),
        "plan_a": stats_a,
        "s1o": stats_o,
    }, indent=2), encoding="utf-8")
    print(f"[extract] wrote {args.output_root / '_extract_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
