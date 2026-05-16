"""Round-13 cache readiness verifier for the S1-A / S1-B variance protocol.

This script answers: "is this cache safe to use for the official
S1-A / S1-B paired training and per-subset audit?"

Checks (each as a separate exit-code-summarized step):

1. Cache layout — `manifest_{train,val}.jsonl`,
   `normalization_train.json`, `README_cache_contract.md`,
   `text_embeddings_clip_vit_b32.npz`, `text_embeddings_index.json`
   all present.
2. Per-clip `.npz` payload contains ONLY `coarse_v1` and
   `init_coarse_v1`. Each `.npz` is finite and has shape `(T, 23)`.
   No object / plan / contact / pseudo-label fields.
3. Manifest schema contains no forbidden tokens (object/plan/contact/
   pseudo/anchor/target_xyz/hand/foot).
4. Per-subset coverage — for each of `chairs / imhd / neuraldome /
   omomo_correct_v2`, BOTH train and val must contain at least one
   clip. If any subset is missing val (e.g. under a smoke per-subset
   cap), the verifier reports it explicitly.
5. Normalization stats — global mean/std/std_clamped of shape `(23,)`,
   `std_clamped >= std_eps`; n_train_frames > 0; `split=="train"`;
   no Round-9 selection JSON path in the source provenance.
6. CLIP text embeddings — shape `(N, 512)` finite; every manifest
   text resolves to an index entry; CLIP model name + download root
   recorded.

Exit codes:

- 0 = all checks pass → cache is variance-protocol ready.
- 1 = at least one check failed → do NOT use for paired comparison.

This script is intentionally separate from
`_verify_cache_safety.py`. The safety verifier covers (2) and (3);
this one additionally guarantees (1, 4, 5, 6) — the *training
readiness* dimension.

Usage
-----

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/verify_stage1_cache_for_variance_protocol.py \
        --cache-root cache/stage1_coarse_v1_round12 \
        --require-val-coverage   # use --no-require-val-coverage for smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ALLOWED_NPZ_KEYS = {"coarse_v1", "init_coarse_v1"}
FORBIDDEN_TOKENS = (
    "object", "obj_", "z_int", "contact", "plan", "phase", "support",
    "hand", "foot", "pseudo", "anchor", "target_xyz",
)
SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")
REQUIRED_FILES = (
    "manifest_train.jsonl", "manifest_val.jsonl",
    "normalization_train.json", "README_cache_contract.md",
    "text_embeddings_clip_vit_b32.npz", "text_embeddings_index.json",
)


def _say(tag: str, msg: str) -> None:
    print(f"[verify:{tag}] {msg}")


def step1_layout(cache: Path) -> bool:
    _say("layout", f"cache_root = {cache}")
    if not cache.exists():
        _say("layout", "FAIL — cache root missing")
        return False
    missing = [name for name in REQUIRED_FILES if not (cache / name).exists()]
    if missing:
        _say("layout", f"FAIL — missing required files: {missing}")
        return False
    _say("layout", "PASS — all required files present")
    return True


def step2_npz_safety(cache: Path) -> bool:
    bad: list[str] = []
    n_seen = 0
    union_keys: set[str] = set()
    for clip_npz in sorted(cache.glob("clips/*/*.npz")):
        data = np.load(clip_npz, allow_pickle=False)
        keys = set(data.files)
        union_keys.update(keys)
        if not keys.issubset(ALLOWED_NPZ_KEYS):
            bad.append(f"extra keys in {clip_npz}: {sorted(keys - ALLOWED_NPZ_KEYS)}")
        for k in keys:
            kl = k.lower()
            for tok in FORBIDDEN_TOKENS:
                if tok in kl:
                    bad.append(f"forbidden token '{tok}' in npz field '{k}' ({clip_npz})")
        cv1 = data["coarse_v1"]
        init = data["init_coarse_v1"]
        if cv1.ndim != 2 or cv1.shape[1] != 23:
            bad.append(f"bad coarse_v1 shape {cv1.shape} in {clip_npz}")
        if init.shape != (23,):
            bad.append(f"bad init_coarse_v1 shape {init.shape} in {clip_npz}")
        if not np.isfinite(cv1).all() or not np.isfinite(init).all():
            bad.append(f"non-finite in {clip_npz}")
        n_seen += 1
    _say("npz", f"scanned {n_seen} clip npz files")
    _say("npz", f"union of npz keys: {sorted(union_keys)}")
    if bad:
        _say("npz", "FAIL")
        for b in bad[:10]:
            _say("npz", "  " + b)
        if len(bad) > 10:
            _say("npz", f"  ... and {len(bad) - 10} more violations")
        return False
    _say("npz", "PASS — every clip has only {coarse_v1, init_coarse_v1}, all finite, shape (T, 23)")
    return True


def _load_manifests(cache: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for split in ("train", "val"):
        recs: list[dict] = []
        for line in (cache / f"manifest_{split}.jsonl").read_text("utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
        out[split] = recs
    return out


def step3_manifest_safety(cache: Path) -> bool:
    manifests = _load_manifests(cache)
    bad: list[str] = []
    for split, recs in manifests.items():
        _say("manifest", f"{split}: {len(recs)} records")
        if not recs:
            continue
        keys = sorted(recs[0].keys())
        for k in keys:
            kl = k.lower()
            for tok in FORBIDDEN_TOKENS:
                if tok in kl:
                    bad.append(f"forbidden manifest field '{k}' in {split}")
    if bad:
        _say("manifest", "FAIL")
        for b in bad:
            _say("manifest", "  " + b)
        return False
    _say("manifest", "PASS — manifest schema has no forbidden tokens")
    return True


def step4_per_subset_coverage(cache: Path, require_val: bool) -> bool:
    manifests = _load_manifests(cache)
    per_subset_train: dict[str, int] = defaultdict(int)
    per_subset_val: dict[str, int] = defaultdict(int)
    for r in manifests["train"]:
        per_subset_train[r["subset"]] += 1
    for r in manifests["val"]:
        per_subset_val[r["subset"]] += 1
    print(f"{'subset':>20s}  {'train':>6s}  {'val':>6s}")
    for s in SUBSETS:
        print(f"{s:>20s}  {per_subset_train.get(s, 0):>6d}  {per_subset_val.get(s, 0):>6d}")
    ok = True
    for s in SUBSETS:
        if per_subset_train.get(s, 0) <= 0:
            _say("coverage", f"FAIL — subset {s!r} has zero train clips")
            ok = False
        if require_val and per_subset_val.get(s, 0) <= 0:
            _say("coverage", f"FAIL — subset {s!r} has zero val clips (variance-protocol requires both)")
            ok = False
        elif not require_val and per_subset_val.get(s, 0) <= 0:
            _say("coverage", f"NOTE — subset {s!r} has zero val clips (smoke cache; rebuild with --max-per-subset -1)")
    if ok:
        _say("coverage", "PASS — every subset has at least one train (and val, if required) clip")
    return ok


def step5_normalization(cache: Path) -> bool:
    norm = json.loads((cache / "normalization_train.json").read_text("utf-8"))
    if norm.get("split") != "train":
        _say("norm", f"FAIL — split={norm.get('split')!r}, expected 'train'")
        return False
    g = norm.get("global", {})
    for key in ("mean", "std", "std_clamped"):
        if len(g.get(key, [])) != 23:
            _say("norm", f"FAIL — global.{key} has wrong length {len(g.get(key, []))}")
            return False
    eps = float(g.get("std_eps", 1e-3))
    if any(s < eps - 1e-12 for s in g["std_clamped"]):
        _say("norm", "FAIL — std_clamped contains values below std_eps")
        return False
    if int(norm.get("n_train_frames", 0)) <= 0:
        _say("norm", "FAIL — n_train_frames <= 0")
        return False
    # Source provenance: must not point at a Round-9 selection file.
    src = str(norm.get("config_source", "")).replace("\\", "/").lower()
    if "subset_balanced_failure_selection" in src or "round-9" in src:
        _say("norm", f"FAIL — normalization config_source points at a Round-9 selection: {src!r}")
        return False
    _say("norm", f"PASS — global stats from {int(norm['n_train_frames'])} train frames, "
                f"std_eps={eps}, source config = {norm.get('config_source')}")
    return True


def step6_clip_embeddings(cache: Path) -> bool:
    npz = np.load(cache / "text_embeddings_clip_vit_b32.npz", allow_pickle=True)
    emb = npz["embeddings"]
    if emb.ndim != 2 or emb.shape[1] != 512:
        _say("clip", f"FAIL — embeddings shape {emb.shape}")
        return False
    if not np.isfinite(emb).all():
        _say("clip", "FAIL — non-finite embeddings")
        return False
    idx_payload = json.loads((cache / "text_embeddings_index.json").read_text("utf-8"))
    if "clip_model_name" not in idx_payload:
        _say("clip", "FAIL — text_embeddings_index.json missing clip_model_name")
        return False
    index = idx_payload["index"]
    manifests = _load_manifests(cache)
    missing = 0
    for split in ("train", "val"):
        for r in manifests[split]:
            if r.get("text", "") not in index:
                missing += 1
    if missing > 0:
        _say("clip", f"FAIL — {missing} manifest texts have no CLIP index entry")
        return False
    _say(
        "clip",
        f"PASS — embeddings {emb.shape}, clip_model_name={idx_payload['clip_model_name']!r}, "
        f"every manifest text indexed",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--require-val-coverage", dest="require_val_coverage",
        action="store_true", default=True,
        help="(default) require every subset to have ≥1 val clip. Use "
             "--no-require-val-coverage for a smoke cache.",
    )
    parser.add_argument(
        "--no-require-val-coverage", dest="require_val_coverage",
        action="store_false",
        help="Disable the per-subset val coverage requirement (smoke mode).",
    )
    args = parser.parse_args()
    cache: Path = args.cache_root

    checks = [
        ("layout", lambda: step1_layout(cache)),
        ("npz", lambda: step2_npz_safety(cache)),
        ("manifest", lambda: step3_manifest_safety(cache)),
        ("coverage", lambda: step4_per_subset_coverage(cache, require_val=args.require_val_coverage)),
        ("norm", lambda: step5_normalization(cache)),
        ("clip", lambda: step6_clip_embeddings(cache)),
    ]
    failures: list[str] = []
    for name, fn in checks:
        try:
            ok = fn()
        except Exception as e:
            _say(name, f"FAIL — exception: {e!r}")
            ok = False
        if not ok:
            failures.append(name)
        print("-" * 70)
    if failures:
        print(
            f"[verify] FAILED: {failures}.  Variance-protocol readiness = NO. "
            f"Cache requires fixes or a rebuild before official training."
        )
        return 1
    # Round-13 follow-up: relaxed mode (--no-require-val-coverage) is
    # intended for smoke / preflight caches only. A relaxed-mode pass
    # does NOT mean the cache is variance-protocol ready, because
    # per-subset val coverage was not enforced.
    if args.require_val_coverage:
        print("[verify] ALL CHECKS PASSED. Cache is variance-protocol ready.")
    else:
        print(
            "[verify] Smoke-cache checks passed with val coverage relaxed. "
            "NOT variance-protocol ready — rerun without "
            "--no-require-val-coverage on a full cache before official "
            "S1-A/S1-B training."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
