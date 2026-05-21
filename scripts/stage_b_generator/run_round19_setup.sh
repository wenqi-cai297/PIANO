#!/usr/bin/env bash
# Round-19 server-side setup: build caches + run preflights for paired
# Plan A vs S1-O ablation. All stdout/stderr captured into
# runs/training/round19_setup/ and bundled into a tar for upload.
#
# Steps:
#   3a) Plan A motion cache (cache/stage1_coarse_v1_full)
#   3b) CLIP ViT-B/32 text embeddings (in 3a's cache root), including
#       mirrored left/right text variants needed by Round-20 mirror aug.
#   3c) S1-O obj_traj cache (cache/stage1_coarse_v1_objtraj_root0_world_round18_fix)
#   4)  Frame convention preflight (4 subsets)
#   5)  Round-18 19-test preflight suite
#
# Usage (from anywhere; script cd's to repo root):
#   bash scripts/stage_b_generator/run_round19_setup.sh
#
# Prereq: conda env `piano` active. PIANO_V18_CFG env var optional.

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/training/round19_setup"
mkdir -p "${LOG_DIR}"

echo "[round19-setup] repo root: $(pwd)"
echo "[round19-setup] log dir:   ${LOG_DIR}"
echo "[round19-setup] python:    $(which python)"
echo

echo "===== Step 3a: Plan A motion cache ====="
python scripts/stage_b_generator/build_stage1_coarse_v1_cache.py \
    --config configs/training/anchordiff_v18_a1_FULL_DATA_local.yaml \
    --output-root cache/stage1_coarse_v1_full \
    --max-per-subset -1 \
    --cache-version full_2026-05-17_v1 \
    2>&1 | tee "${LOG_DIR}/step3a_cache_planA.log"

echo
echo "===== Step 3b: CLIP text embeddings ====="
python scripts/stage_b_generator/cache_stage1_clip_text_embeddings.py \
    --cache-root cache/stage1_coarse_v1_full \
    --clip-version ViT-B/32 \
    --clip-download-root cache/clip \
    --include-mirrored-texts \
    2>&1 | tee "${LOG_DIR}/step3b_clip_text.log"

echo
echo "===== Step 3c: S1-O obj_traj cache ====="
python scripts/stage_b_generator/build_stage1_coarse_v1_objtraj_root0_world_cache.py \
    --config configs/training/anchordiff_v18_a1_FULL_DATA_local.yaml \
    --output-root cache/stage1_coarse_v1_objtraj_root0_world_round18_fix \
    --max-per-subset -1 \
    --cache-version r18fix_full_2026-05-17_v1 \
    --copy-text-cache-from cache/stage1_coarse_v1_full \
    2>&1 | tee "${LOG_DIR}/step3c_cache_s1o.log"

echo
echo "===== Step 4: Frame convention preflight ====="
python scripts/stage_b_generator/preflight_round18_frame_convention.py \
    2>&1 | tee "${LOG_DIR}/step4_frame_preflight.log"

echo
echo "===== Step 5: Round-18 preflight test suite ====="
python scripts/stage_b_generator/test_stage1_round18_preflight.py \
    2>&1 | tee "${LOG_DIR}/step5_r18_preflight.log"

echo
echo "===== Packing tarball ====="
tar czf "runs/training/round19_setup.tar.gz" \
    "${LOG_DIR}/" \
    analyses/round18_preflight/frame_convention_preflight.json

echo
echo "[round19-setup] ALL STEPS COMPLETE"
echo "[round19-setup] upload: runs/training/round19_setup.tar.gz"
