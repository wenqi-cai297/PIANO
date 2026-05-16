"""Round-12 preflight: verify Stage-1 cache contains no object/plan/contact data.

Iterates the cache and asserts each .npz contains only the
Stage-1-safe field set. Also spot-checks manifest record consistency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE_ROOT = Path("cache/stage1_coarse_v1_round12")
ALLOWED_NPZ_KEYS = {"coarse_v1", "init_coarse_v1"}
FORBIDDEN_TOKENS = (
    "object", "obj_", "z_int", "contact", "plan", "phase", "support",
    "hand", "foot", "pseudo", "anchor", "target_xyz",
)


def main() -> int:
    if not CACHE_ROOT.exists():
        print(f"[verify] {CACHE_ROOT} does not exist — run build_stage1_coarse_v1_cache.py first.")
        return 1

    # Manifests
    manifests = {}
    for name in ("manifest_train.jsonl", "manifest_val.jsonl"):
        recs = []
        for line in (CACHE_ROOT / name).read_text("utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
        manifests[name] = recs
        print(f"[verify] {name}: {len(recs)} records")

    # Inspect every clip .npz file.
    bad: list[str] = []
    sample_keys_seen: set[str] = set()
    for clip_npz in sorted(CACHE_ROOT.glob("clips/*/*.npz")):
        data = np.load(clip_npz, allow_pickle=False)
        keys = set(data.files)
        sample_keys_seen.update(keys)
        if not keys.issubset(ALLOWED_NPZ_KEYS):
            bad.append(f"  extra keys in {clip_npz}: {keys - ALLOWED_NPZ_KEYS}")
        for k in keys:
            kl = k.lower()
            for tok in FORBIDDEN_TOKENS:
                if tok in kl:
                    bad.append(f"  forbidden token '{tok}' in key '{k}' of {clip_npz}")
        # Shape sanity
        cv1 = data["coarse_v1"]
        init = data["init_coarse_v1"]
        if cv1.ndim != 2 or cv1.shape[1] != 23:
            bad.append(f"  bad coarse_v1 shape {cv1.shape} in {clip_npz}")
        if init.shape != (23,):
            bad.append(f"  bad init_coarse_v1 shape {init.shape} in {clip_npz}")
        if not np.isfinite(cv1).all():
            bad.append(f"  non-finite coarse_v1 in {clip_npz}")

    print(f"[verify] union of .npz keys: {sorted(sample_keys_seen)}")
    print(f"[verify] allowed set:        {sorted(ALLOWED_NPZ_KEYS)}")

    # Manifest fields — should have no forbidden tokens in the schema.
    for name, recs in manifests.items():
        if not recs:
            continue
        keys = sorted(recs[0].keys())
        print(f"[verify] {name} record keys: {keys}")
        for k in keys:
            kl = k.lower()
            for tok in FORBIDDEN_TOKENS:
                if tok in kl:
                    bad.append(f"  forbidden token '{tok}' in manifest field '{k}' ({name})")

    if bad:
        print("[verify] FAILED — Stage-1 safety violations:")
        for b in bad:
            print(b)
        return 1
    print("[verify] OK — cache is object/plan/contact-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
