#!/usr/bin/env bash
# Pack Round-41 cascade fine-tune results for sync-back.
#
# EXCLUDES by default:
#   - sampled .npz substitute caches (large)
#   - model ckpts
#
# Sync-back contents:
#   - R41 configs (configs/training/stage1_r41_*.yaml)
#   - per-variant train logs + metrics.jsonl + smoke logs
#   - per-variant direct diag dirs (sustained_contact / gait / body_action / g1)
#   - matrix summary log
#   - R41 P0 diag + helper docs
#
# Opt-ins:
#   ROUND41_PACK_CKPTS=1
#   ROUND41_PACK_NPZ=1

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="${ROUND41_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TARBALL="${ROUND41_TARBALL:-analyses/round41_cascade_results_${STAMP}.tar.gz}"
SUMMARY_LOG="${ROUND41_SUMMARY_LOG:-runs/round41_cascade_matrix/summary_${STAMP}.log}"
VARIANT_LOG_DIR="${ROUND41_VARIANT_LOG_DIR:-runs/round41_cascade_matrix}"
INCLUDE_CKPTS="${ROUND41_PACK_CKPTS:-0}"
INCLUDE_NPZ="${ROUND41_PACK_NPZ:-0}"

TRAINED_VIDS_STR="${ROUND41_TRAINED_VIDS:-}"
DIAGED_VIDS_STR="${ROUND41_DIAGED_VIDS:-}"
if [[ -z "${TRAINED_VIDS_STR}" && -z "${DIAGED_VIDS_STR}" ]]; then
    DEFAULT_VIDS="stage1_r41_a0_cascade_off stage1_r41_a1_motion_mse stage1_r41_a2_world_vel stage1_r41_a3_l_pos_full stage1_r41_a4_anchor_pos"
    TRAINED_VIDS_STR="${DEFAULT_VIDS}"
    DIAGED_VIDS_STR="${DEFAULT_VIDS}"
fi

# shellcheck disable=SC2206
TRAINED_VIDS=(${TRAINED_VIDS_STR})
# shellcheck disable=SC2206
DIAGED_VIDS=(${DIAGED_VIDS_STR})

PACK_TARGETS=()
add_if_exists_file() { [[ -f "$1" ]] && PACK_TARGETS+=("$1"); }
add_if_exists_dir()  { [[ -d "$1" ]] && PACK_TARGETS+=("$1"); }

for VID in "${TRAINED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_file "configs/training/${VID}.yaml"
    add_if_exists_file "${VARIANT_LOG_DIR}/${VID}.log"
    add_if_exists_file "${VARIANT_LOG_DIR}/${VID}.smoke.log"
    add_if_exists_file "runs/training/${VID}/metrics.jsonl"
    if [[ "${INCLUDE_CKPTS}" == "1" ]]; then
        add_if_exists_file "runs/training/${VID}/final.pt"
    fi
done

for VID in "${DIAGED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_dir "analyses/round41_stage1_direct_diag_${VID}"
    add_if_exists_dir "analyses/round41_stage1_ood_${VID}"
    add_if_exists_dir "analyses/round41_stage1_kdiv_${VID}"
    add_if_exists_dir "analyses/round41_full_cascade_${VID}"
    if [[ "${INCLUDE_NPZ}" == "1" ]]; then
        add_if_exists_dir "analyses/round41_stage1_substitute_conds_${VID}"
    fi
done

# Calibration reports (always small).
add_if_exists_dir "analyses/round41_cascade_calibration"

add_if_exists_file "${SUMMARY_LOG}"
add_if_exists_file "analyses/2026-06-01_r41_stage1_cascade_handoff_for_codex.md"
add_if_exists_file "analyses/2026-06-02_r41_stage1_cascade_experiment_plan_for_claude.md"
add_if_exists_file "analyses/2026-06-02_r41_code_review_fix_instructions_for_claude.md"
add_if_exists_file "analyses/2026-06-02_r41_return_for_codex.md"
add_if_exists_dir "analyses/round41_p0_cascade_diag"

if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
    echo "[R41 PACK] nothing to pack" >&2
    exit 0
fi

mkdir -p "$(dirname "${TARBALL}")"
echo "[R41 PACK] writing ${TARBALL} (${#PACK_TARGETS[@]} targets)"
tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "[R41 PACK] wrote ${TARBALL}  (${SIZE})"

MANIFEST="${TARBALL%.tar.gz}_manifest.txt"
{
    echo "# R41 sync-back manifest"
    echo "stamp: ${STAMP}"
    echo "tarball: ${TARBALL}"
    echo "size: ${SIZE}"
    echo "include_ckpts: ${INCLUDE_CKPTS}"
    echo "include_npz: ${INCLUDE_NPZ}"
    echo "trained_vids: ${TRAINED_VIDS[*]:-}"
    echo "diaged_vids: ${DIAGED_VIDS[*]:-}"
    echo
    echo "## Contents:"
    printf '  %s\n' "${PACK_TARGETS[@]}"
} > "${MANIFEST}"
echo "[R41 PACK] manifest at ${MANIFEST}"
