#!/usr/bin/env bash
# Round-20 text-cache refresh only.
#
# Use this when the Stage-1 manifest/npz caches are already valid but the
# mirrored-text helper changed. It rebuilds the Plan A CLIP text cache with
# direction-mirrored captions, copies the exact same text cache into the S1-O
# cache, and performs a quick parity check. It does not rebuild motion caches
# and does not launch training.

set -euo pipefail
cd "$(dirname "$0")/../.."

PLAN_A_CACHE="cache/stage1_coarse_v1_full"
S1O_CACHE="cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"
CLIP_VERSION="${CLIP_VERSION:-ViT-B/32}"
CLIP_DOWNLOAD_ROOT="${CLIP_DOWNLOAD_ROOT:-cache/clip}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CONDA_ENV="${CONDA_ENV:-piano}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_CMD=("${PYTHON_BIN}")
elif command -v python.exe >/dev/null 2>&1; then
    PYTHON_CMD=(python.exe)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
    echo "[round20-recache-text] ERROR: no python executable found; activate the ${CONDA_ENV} env or set PYTHON_BIN" >&2
    exit 1
fi

echo "[round20-recache-text] repo root: $(pwd)"
echo "[round20-recache-text] Plan A cache: ${PLAN_A_CACHE}"
echo "[round20-recache-text] S1-O cache:   ${S1O_CACHE}"
echo "[round20-recache-text] CLIP:         ${CLIP_VERSION}"
echo "[round20-recache-text] python cmd:   ${PYTHON_CMD[*]}"
echo

"${PYTHON_CMD[@]}" scripts/stage_b_generator/cache_stage1_clip_text_embeddings.py \
    --cache-root "${PLAN_A_CACHE}" \
    --clip-version "${CLIP_VERSION}" \
    --clip-download-root "${CLIP_DOWNLOAD_ROOT}" \
    --batch-size "${BATCH_SIZE}" \
    --include-mirrored-texts

cp "${PLAN_A_CACHE}/text_embeddings_clip_vit_b32.npz" \
   "${S1O_CACHE}/text_embeddings_clip_vit_b32.npz"
cp "${PLAN_A_CACHE}/text_embeddings_index.json" \
   "${S1O_CACHE}/text_embeddings_index.json"

"${PYTHON_CMD[@]}" - <<'PY'
import json
import numpy as np
from pathlib import Path

roots = [
    Path("cache/stage1_coarse_v1_full"),
    Path("cache/stage1_coarse_v1_objtraj_root0_world_round18_fix"),
]
payloads = [
    json.loads((root / "text_embeddings_index.json").read_text(encoding="utf-8"))
    for root in roots
]
for root, payload in zip(roots, payloads):
    print(
        f"[round20-recache-text] {root}: "
        f"include_mirrored_texts={payload.get('include_mirrored_texts')} "
        f"n_unique_texts={payload.get('n_unique_texts')} "
        f"n_total_candidates={payload.get('n_total_encoded_text_candidates')}"
    )
if payloads[0].get("index") != payloads[1].get("index"):
    raise SystemExit("[round20-recache-text] ERROR: Plan A and S1-O text indices differ after copy")
npzs = [
    np.load(root / "text_embeddings_clip_vit_b32.npz", allow_pickle=True)
    for root in roots
]
if [str(x) for x in npzs[0]["texts"].tolist()] != [str(x) for x in npzs[1]["texts"].tolist()]:
    raise SystemExit("[round20-recache-text] ERROR: Plan A and S1-O npz texts arrays differ after copy")
if not np.array_equal(npzs[0]["embeddings"], npzs[1]["embeddings"]):
    raise SystemExit("[round20-recache-text] ERROR: Plan A and S1-O embedding arrays differ after copy")
print("[round20-recache-text] text index parity OK")
print("[round20-recache-text] embedding npz parity OK")
PY

echo
echo "[round20-recache-text] done. Run:"
echo "  python scripts/stage_b_generator/test_stage1_round18_preflight.py"
