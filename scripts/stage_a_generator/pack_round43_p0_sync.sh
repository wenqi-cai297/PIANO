#!/usr/bin/env bash
# Pack Round-43 P0 outputs for sync-back.
#
# Default: NO ckpt, NO npz cache (they are bulky and the operator can
# resync per-clip from server if needed). Opt-in:
#   ROUND43_PACK_CKPT=1        include runs/training/<run>/final.pt
#   ROUND43_PACK_CACHE_NPZ=1   include the per-clip generated cache
#
# Uses explicit `if [[ -f X ]]; then ...; fi` (NOT `[[ -f X ]] && ...`)
# so `set -e` does not exit on the first absent file — that was the
# silent-exit bug fixed in R41 commit 27f3005.

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="${ROUND43_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TARBALL="${ROUND43_TARBALL:-analyses/round43_p0_results_${STAMP}.tar.gz}"
MANIFEST="${ROUND43_MANIFEST:-${TARBALL%.tar.gz}_manifest.txt}"

LOG_DIR="${ROUND43_LOG_DIR:-runs/round43_p0_${STAMP}}"
CACHE_DIR="${ROUND43_CACHE_DIR:-analyses/round43_stage1_substitute_conds_a2_${STAMP}}"
AUDIT_DIR="${ROUND43_AUDIT_DIR:-analyses/round43_p0_cache_audit_${STAMP}}"
R42_OUT_ROOT="${ROUND43_R42_OUT_ROOT:-analyses/round43_p0_r42_rerun_${STAMP}}"
RESOLVED_CFG="${ROUND43_RESOLVED_CFG:-configs/training/stage1p5_r43_p0_mixed_a2.yaml}"
OUT_DIR_NAME="${ROUND43_OUT_DIR_NAME:-stage1p5_r43_p0_mixed_a2}"
SEL_TRAIN="${ROUND43_SEL_TRAIN:-}"
SEL_VAL="${ROUND43_SEL_VAL:-}"

INCLUDE_CKPT="${ROUND43_PACK_CKPT:-0}"
INCLUDE_CACHE_NPZ="${ROUND43_PACK_CACHE_NPZ:-0}"

PACK_TARGETS=()

# Explicit form avoids `set -e + && short-circuit` exit-on-false.
add_file() {
    if [[ -f "$1" ]]; then PACK_TARGETS+=("$1"); fi
}
add_dir() {
    if [[ -d "$1" ]]; then PACK_TARGETS+=("$1"); fi
}

# ─── Config + resolved provenance ─────────────────────────────────────
add_file "${RESOLVED_CFG}"
add_file "configs/training/stage1p5_r43_p0_mixed_a2.yaml.template"
add_file "runs/training/${OUT_DIR_NAME}/${OUT_DIR_NAME}_resolved.yaml"

# ─── Selection JSONs ──────────────────────────────────────────────────
[[ -n "${SEL_TRAIN}" ]] && add_file "${SEL_TRAIN}"
[[ -n "${SEL_VAL}" ]] && add_file "${SEL_VAL}"

# ─── Pipeline logs ────────────────────────────────────────────────────
add_dir "${LOG_DIR}"

# ─── Cache audit (md + json) ──────────────────────────────────────────
add_dir "${AUDIT_DIR}"

# ─── Stage-1.5 training metrics + (optional) ckpt ────────────────────
add_file "runs/training/${OUT_DIR_NAME}/metrics.jsonl"
if [[ "${INCLUDE_CKPT}" == "1" ]]; then
    add_file "runs/training/${OUT_DIR_NAME}/final.pt"
    add_file "runs/training/${OUT_DIR_NAME}/best_val.pt"
fi

# ─── R42 2x2 rerun — full tree (configs/diag md/json for all 4 cells) ─
add_dir "${R42_OUT_ROOT}"
# Companion runs/round42_cond_2x2_<STAMP> log dir written by R42 launcher.
add_dir "runs/round42_cond_2x2_${STAMP}"

# ─── Generated cache NPZ (opt-in; huge) ───────────────────────────────
if [[ "${INCLUDE_CACHE_NPZ}" == "1" ]]; then
    add_dir "${CACHE_DIR}"
fi

# ─── R43 chain docs (for future Claude / Codex restart) ───────────────
for DOC in \
    analyses/2026-06-11_r42_cond_2x2_verdict.md \
    analyses/2026-06-11_r42_verdict_and_r43_plan_for_codex.md \
    analyses/2026-06-12_r42_r43_plan_review_for_claude_code.md \
    analyses/2026-06-12_r43_plan_ack_for_codex.md \
    analyses/2026-06-12_r43_p0_finalized_for_codex.md \
    analyses/2026-06-12_r43_p0_finalized_review_for_claude_code.md \
    analyses/2026-06-12_r43_p0_stage_a_ready_for_codex.md \
    analyses/2026-06-12_r43_stage_a_code_review_for_claude_code.md \
    analyses/2026-06-12_r43_stage_a_layout_fix_for_codex.md
do
    add_file "${DOC}"
done

if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
    echo "[R43 PACK] nothing to pack" >&2
    exit 0
fi

mkdir -p "$(dirname "${TARBALL}")"
echo "[R43 PACK] writing ${TARBALL} (${#PACK_TARGETS[@]} targets)"
tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "[R43 PACK] wrote ${TARBALL}  (${SIZE})"

{
    echo "# R43 P0 sync-back manifest"
    echo "stamp: ${STAMP}"
    echo "tarball: ${TARBALL}"
    echo "size: ${SIZE}"
    echo "include_ckpt: ${INCLUDE_CKPT}"
    echo "include_cache_npz: ${INCLUDE_CACHE_NPZ}"
    echo
    echo "## Contents"
    printf '  %s\n' "${PACK_TARGETS[@]}"
} > "${MANIFEST}"
echo "[R43 PACK] manifest at ${MANIFEST}"
