#!/usr/bin/env bash
# Pack Round-40 Stage-1 plan-sampler results for sync-back to laptop.
#
# Intentionally EXCLUDES large transient files:
#   - sampled .npz substitute caches (typically 100+ MB per variant)
#   - model ckpts in runs/training/<vid>/ (~80-200 MB each)
#
# Sync-back contents (always small):
#   - R40 configs (configs/training/stage1_r40_*.yaml)
#   - per-variant train logs (runs/round40_stage1_plan_matrix/<vid>.log)
#   - per-variant metrics.jsonl
#   - direct-diag summaries (analyses/round40_stage1_direct_diag_*)
#   - plan-diag outputs (analyses/round40_stage1_plan_diag_*)
#   - full-cascade summaries (analyses/round40_fullcascade_diag_*)
#   - K-sample audit md+json (analyses/round40_stage1_kdiv_*, excluding samples/)
#   - matrix summary md + log
#   - handoff doc + return doc
#
# Input — env vars (set by run_round40_stage1_plan_matrix.sh):
#   ROUND40_STAMP, ROUND40_TARBALL, ROUND40_SUMMARY_MD, ROUND40_SUMMARY_LOG
#   ROUND40_VARIANT_LOG_DIR
#   ROUND40_TRAINED_VIDS, ROUND40_DIAGED_VIDS, ROUND40_PLAN_VIDS,
#   ROUND40_CASCADED_VIDS, ROUND40_KDIV_VIDS_LIST   (space-separated lists)
#
# Optional opt-ins:
#   ROUND40_PACK_CKPTS=1            include runs/training/<vid>/final.pt
#   ROUND40_PACK_NPZ=1              include sampled stage1_coarse npz caches
#   ROUND40_PACK_KDIV_SAMPLES=1     include kdiv per-seed samples (huge)

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY_DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) ONLY_DRY_RUN=1 ;;
    esac
done

STAMP="${ROUND40_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TARBALL="${ROUND40_TARBALL:-analyses/round40_stage1_plan_results_${STAMP}.tar.gz}"
SUMMARY_MD="${ROUND40_SUMMARY_MD:-analyses/round40_stage1_plan_matrix_summary_${STAMP}.md}"
SUMMARY_LOG="${ROUND40_SUMMARY_LOG:-runs/round40_stage1_plan_matrix/summary_${STAMP}.log}"
VARIANT_LOG_DIR="${ROUND40_VARIANT_LOG_DIR:-runs/round40_stage1_plan_matrix}"
INCLUDE_CKPTS="${ROUND40_PACK_CKPTS:-0}"
INCLUDE_NPZ="${ROUND40_PACK_NPZ:-0}"
INCLUDE_KDIV_SAMPLES="${ROUND40_PACK_KDIV_SAMPLES:-0}"

# Allow either env-list mode or fall back to "all 4" for standalone re-packing.
TRAINED_VIDS_STR="${ROUND40_TRAINED_VIDS:-}"
DIAGED_VIDS_STR="${ROUND40_DIAGED_VIDS:-}"
PLAN_VIDS_STR="${ROUND40_PLAN_VIDS:-}"
CASCADED_VIDS_STR="${ROUND40_CASCADED_VIDS:-}"
KDIV_VIDS_STR="${ROUND40_KDIV_VIDS_LIST:-}"

if [[ -z "${TRAINED_VIDS_STR}" && -z "${DIAGED_VIDS_STR}" ]]; then
    DEFAULT_VIDS="stage1_r40_c0_v8v6_baseline stage1_r40_c1_weak_gt stage1_r40_c2_plan_energy stage1_r40_c3_plan_energy_strong"
    TRAINED_VIDS_STR="${DEFAULT_VIDS}"
    DIAGED_VIDS_STR="${DEFAULT_VIDS}"
    PLAN_VIDS_STR="${DEFAULT_VIDS}"
    CASCADED_VIDS_STR="${DEFAULT_VIDS}"
    KDIV_VIDS_STR="${DEFAULT_VIDS}"
fi

# shellcheck disable=SC2206
TRAINED_VIDS=(${TRAINED_VIDS_STR})
# shellcheck disable=SC2206
DIAGED_VIDS=(${DIAGED_VIDS_STR})
# shellcheck disable=SC2206
PLAN_VIDS=(${PLAN_VIDS_STR})
# shellcheck disable=SC2206
CASCADED_VIDS=(${CASCADED_VIDS_STR})
# shellcheck disable=SC2206
KDIV_VIDS=(${KDIV_VIDS_STR})

PACK_TARGETS=()
PACK_EXCLUDES=()

add_if_exists_file() { [[ -f "$1" ]] && PACK_TARGETS+=("$1"); }
add_if_exists_dir() { [[ -d "$1" ]] && PACK_TARGETS+=("$1"); }

# Configs.
for VID in "${TRAINED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_file "configs/training/${VID}.yaml"
done

# Per-variant training logs + metrics.
for VID in "${TRAINED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_file "${VARIANT_LOG_DIR}/${VID}.log"
    add_if_exists_file "runs/training/${VID}/metrics.jsonl"
    if [[ "${INCLUDE_CKPTS}" == "1" ]]; then
        add_if_exists_file "runs/training/${VID}/final.pt"
    fi
done

# Direct diag summaries.
for VID in "${DIAGED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_dir "analyses/round40_stage1_direct_diag_${VID}"
done

# Plan diag outputs.
for VID in "${PLAN_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_dir "analyses/round40_stage1_plan_diag_${VID}"
done

# Full cascade summaries.
for VID in "${CASCADED_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    add_if_exists_dir "analyses/round40_fullcascade_diag_${VID}"
done

# K-sample audit: include the md+json but exclude per-seed sample npz
# unless explicitly opted in. tar --exclude is per-pattern; we add the dir
# but rely on PACK_EXCLUDES to drop the samples/ subdir.
for VID in "${KDIV_VIDS[@]:-}"; do
    [[ -z "${VID}" ]] && continue
    if [[ -d "analyses/round40_stage1_kdiv_${VID}" ]]; then
        PACK_TARGETS+=("analyses/round40_stage1_kdiv_${VID}")
        if [[ "${INCLUDE_KDIV_SAMPLES}" != "1" ]]; then
            PACK_EXCLUDES+=("--exclude=analyses/round40_stage1_kdiv_${VID}/samples")
        fi
    fi
done

# Sampled substitute-conds caches (opt-in; usually too big to sync).
if [[ "${INCLUDE_NPZ}" == "1" ]]; then
    for VID in "${DIAGED_VIDS[@]:-}"; do
        [[ -z "${VID}" ]] && continue
        add_if_exists_dir "analyses/round40_stage1_substitute_conds_${VID}"
    done
fi

# Matrix-level outputs.
add_if_exists_file "${SUMMARY_MD}"
add_if_exists_file "${SUMMARY_LOG}"

# Handoff + return docs (if present).
add_if_exists_file "analyses/2026-06-01_round40_stage1_plan_sampler_handoff_for_claude.md"
add_if_exists_file "analyses/2026-06-01_round40_return_for_codex.md"

if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
    echo "[R40 PACK] nothing to pack" >&2
    exit 0
fi

if [[ ${ONLY_DRY_RUN} -eq 1 ]]; then
    echo "[R40 PACK DRY-RUN] tarball: ${TARBALL}"
    echo "[R40 PACK DRY-RUN] include_ckpts=${INCLUDE_CKPTS} include_npz=${INCLUDE_NPZ} include_kdiv_samples=${INCLUDE_KDIV_SAMPLES}"
    echo "[R40 PACK DRY-RUN] targets:"
    printf "  %s\n" "${PACK_TARGETS[@]}"
    if [[ ${#PACK_EXCLUDES[@]} -gt 0 ]]; then
        echo "[R40 PACK DRY-RUN] excludes:"
        printf "  %s\n" "${PACK_EXCLUDES[@]}"
    fi
    exit 0
fi

mkdir -p "$(dirname "${TARBALL}")"
echo "[R40 PACK] writing ${TARBALL} (${#PACK_TARGETS[@]} targets)"
tar -czf "${TARBALL}" "${PACK_EXCLUDES[@]}" "${PACK_TARGETS[@]}"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "[R40 PACK] wrote ${TARBALL}  (${SIZE})"

MANIFEST="${TARBALL%.tar.gz}_manifest.txt"
{
    echo "# R40 sync-back manifest"
    echo "stamp: ${STAMP}"
    echo "tarball: ${TARBALL}"
    echo "size: ${SIZE}"
    echo "include_ckpts: ${INCLUDE_CKPTS}"
    echo "include_npz: ${INCLUDE_NPZ}"
    echo "include_kdiv_samples: ${INCLUDE_KDIV_SAMPLES}"
    echo "trained_vids: ${TRAINED_VIDS[*]:-}"
    echo "diaged_vids: ${DIAGED_VIDS[*]:-}"
    echo "plan_vids: ${PLAN_VIDS[*]:-}"
    echo "cascaded_vids: ${CASCADED_VIDS[*]:-}"
    echo "kdiv_vids: ${KDIV_VIDS[*]:-}"
    echo
    echo "## Contents:"
    printf '  %s\n' "${PACK_TARGETS[@]}"
    if [[ ${#PACK_EXCLUDES[@]} -gt 0 ]]; then
        echo
        echo "## Excludes:"
        printf '  %s\n' "${PACK_EXCLUDES[@]}"
    fi
} > "${MANIFEST}"
echo "[R40 PACK] manifest at ${MANIFEST}"
