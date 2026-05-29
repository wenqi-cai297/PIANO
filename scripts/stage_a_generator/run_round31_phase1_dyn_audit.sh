#!/usr/bin/env bash
# Round-31 Phase 1 dynamics audit — Stage-1 generated vs GT oracle
# stage1_coarse, on the same 48 val clips that fed all R31 V2 + Phase 0.5
# downstream-diag runs.
#
# Reads the Phase 0.5 cfg=1.0 ddim_eta0 generated cache (which IS the cond
# that produced the 18.5 cm drift_max baseline) and compares it against
# on-the-fly extract_coarse_v1_batched(GT motion_135, GT rest_offsets).
#
# Reports:
#   1. per-channel-group rollup (mean/std/vel_rms/accel_rms ratios)
#   2. per-channel detail table (all 23 channels)
#   3. PSD band-energy ratio (low/mid/high Hz)
#   4. rot6d orthogonality violation (pred vs GT)
#   5. per-clip root_local drift summary + frame-0 invariant check
#   6. top-5 std mismatch channels
#   7. top-5 velocity mismatch channels
#
# Output:
#   analyses/round31_phase1_dyn_audit_<STAMP>/audit_report.md
#   analyses/round31_phase1_dyn_audit_<STAMP>/audit_stats.json
#   analyses/round31_phase1_dyn_audit_<STAMP>/per_clip_dump.npz
#
# CPU only. ~3-5 min on the 48-clip selection.
#
# Usage:
#   bash scripts/stage_a_generator/run_round31_phase1_dyn_audit.sh
#
# Env overrides:
#   ROUND31_P1_UPSTREAM_DIR=...
#       default analyses/round31_stage1_substitute_conds_phase0p5_cfg1p0_ddim_eta0/val
#   ROUND31_P1_STAGE1_CFG=...
#       default configs/training/stage1_v2_v0_baseline.yaml
#   ROUND31_P1_SELECTION=...
#       default analyses/round29_val_diag_indices_48_balanced.json
#   ROUND31_P1_OUT_DIR=...
#       default analyses/round31_phase1_dyn_audit_<STAMP>

set -euo pipefail
cd "$(dirname "$0")/../.."

UPSTREAM_DIR="${ROUND31_P1_UPSTREAM_DIR:-analyses/round31_stage1_substitute_conds_phase0p5_cfg1p0_ddim_eta0/val}"
STAGE1_CFG="${ROUND31_P1_STAGE1_CFG:-configs/training/stage1_v2_v0_baseline.yaml}"
SELECTION="${ROUND31_P1_SELECTION:-analyses/round29_val_diag_indices_48_balanced.json}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROUND31_P1_OUT_DIR:-analyses/round31_phase1_dyn_audit_${STAMP}}"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[P1] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi

# Preflight.
preflight_fail=0
for p in "${UPSTREAM_DIR}" "${STAGE1_CFG}" "${SELECTION}"; do
    [[ ! -e "${p}" ]] && { echo "[P1 PREFLIGHT FAIL] missing: ${p}"; preflight_fail=1; }
done
if [[ ${preflight_fail} -ne 0 ]]; then
    echo "[P1] FATAL preflight failures."
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[P1] upstream_dir = ${UPSTREAM_DIR}"
echo "[P1] stage1_cfg   = ${STAGE1_CFG}"
echo "[P1] selection    = ${SELECTION}"
echo "[P1] out_dir      = ${OUT_DIR}"
echo

"${PY}" -u scripts/stage_a_generator/round31_phase1_dyn_audit.py \
    --upstream-dir "${UPSTREAM_DIR}" \
    --stage1-cfg "${STAGE1_CFG}" \
    --selection-json "${SELECTION}" \
    --out-dir "${OUT_DIR}"

# Pack everything into a single tarball — handy for scp back.
TARBALL="analyses/round31_phase1_dyn_audit_${STAMP}.tar.gz"
tar -czf "${TARBALL}" "${OUT_DIR}"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo
echo "[P1] wrote ${TARBALL}  (${SIZE})"
echo "[P1] DONE."
