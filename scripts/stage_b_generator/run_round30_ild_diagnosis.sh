#!/usr/bin/env bash
# Round-30 idle-local-detail (ILD) diagnosis launcher.
#
# Per analyses/2026-05-29_round30_idle_local_detail_diagnosis_plan.md.
# Runs Phase 0 + Phase 1 (E0 + E1) of the round on a single GPU. E0
# (ILD subset filter) is CPU-only; E1 (text probe) needs one 5080.
# Designed to run in parallel with the PB1 training (which occupies
# GPU 0 + 1 via accelerate), so this launcher defaults to GPU 2.
#
# E2 / E3 / E4 (B content / injection / loss probes) are intentionally
# NOT launched here — they need new B variants in
# src/piano/data/stage2_oracle_conditions.py and a data re-extraction
# pass. We decide whether to commit to that work AFTER E0+E1 land.
#
# Gates:
#   E0: round30_build_ild_subset.py exits 2 if train ILD < 1 % → H1
#       supported, the launcher STOPs.
#   E1: round30_text_condition_probe.py exits 2 on label='text_dead'
#       → H5 supported, the launcher STOPs.
#
# Env overrides:
#   ROUND30_GPU            cuda device id (default 2)
#   ROUND30_CKPT           A1 ckpt path
#                          (default runs/training/stageB_anchordiff_r29_ns_a1_c41_s4_g1/final.pt)
#   ROUND30_CFG            A1 yaml config
#                          (default configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml)
#   ROUND30_OUTDIR         output root (default analyses/round30_ild)
#   ROUND30_SKIP_E0=1      skip E0 (assume selection files already exist)
#   ROUND30_SKIP_E1=1      skip E1 (debug only)
#   ROUND30_VARIANT_ID     label for output (default r29_ns_a1_c41_s4_g1)

set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${ROUND30_GPU:-2}"
CKPT="${ROUND30_CKPT:-runs/training/stageB_anchordiff_r29_ns_a1_c41_s4_g1/final.pt}"
CFG="${ROUND30_CFG:-configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml}"
OUTDIR="${ROUND30_OUTDIR:-analyses/round30_ild}"
VARIANT_ID="${ROUND30_VARIANT_ID:-r29_ns_a1_c41_s4_g1}"
SKIP_E0="${ROUND30_SKIP_E0:-0}"
SKIP_E1="${ROUND30_SKIP_E1:-0}"

LOG_DIR="runs/round30_ild_diagnosis"
mkdir -p "${LOG_DIR}" "${OUTDIR}"

echo "================================================================"
echo "[$(date '+%F %T')] Round-30 ILD diagnosis"
echo "    gpu:        cuda:${GPU}"
echo "    config:     ${CFG}"
echo "    ckpt:       ${CKPT}"
echo "    output dir: ${OUTDIR}"
echo "    variant id: ${VARIANT_ID}"
echo "================================================================"

# ----------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------
PF_FAIL=0
if [[ ! -f "${CFG}" ]]; then
    echo "[R30 PREFLIGHT FAIL] missing config: ${CFG}"
    echo "    Regenerate the R29-NS configs on the server via:"
    echo "      export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4"
    echo "      python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py"
    PF_FAIL=1
fi
if [[ "${SKIP_E1}" != "1" && ! -f "${CKPT}" ]]; then
    echo "[R30 PREFLIGHT FAIL] missing ckpt: ${CKPT}"
    echo "    A1 must have been trained before E1 can run."
    PF_FAIL=1
fi
if [[ ${PF_FAIL} -ne 0 ]]; then
    echo "[R30] FATAL preflight failures."
    exit 1
fi

# ----------------------------------------------------------------------
# E0: ILD subset filter (CPU-only)
# ----------------------------------------------------------------------
if [[ "${SKIP_E0}" != "1" ]]; then
    LOG="${LOG_DIR}/e0_build_ild_subset.log"
    T0=$(date +%s)
    echo
    echo "[$(date '+%F %T')] E0 build_ild_subset → ${LOG}"
    set +e
    conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round30_build_ild_subset.py \
        --config "${CFG}" \
        --output-dir "${OUTDIR}" \
        2>&1 | tee "${LOG}"
    E0_RC=${PIPESTATUS[0]}
    set -e
    T1=$(date +%s)
    echo "[$(date '+%F %T')] E0 done in $((T1-T0))s, rc=${E0_RC}"
    if [[ ${E0_RC} -eq 2 ]]; then
        echo "[R30] GATE FAILED at E0: train ILD < 1 % → H1 supported."
        echo "    Recommend STOP and either write to limitations (D0) or"
        echo "    pivot to text-architecture rework (D4)."
        echo "    Inspect ${OUTDIR}/subset_stats.md for per-subset detail."
        exit 2
    elif [[ ${E0_RC} -ne 0 ]]; then
        echo "[R30] FATAL: E0 failed with rc=${E0_RC}"
        exit 1
    fi
else
    echo "[R30] --skip-e0: assuming ${OUTDIR}/selection_{train,val,control}.json"
    for f in selection_train.json selection_val.json selection_control.json; do
        if [[ ! -f "${OUTDIR}/${f}" ]]; then
            echo "[R30 PREFLIGHT FAIL] --skip-e0 but missing ${OUTDIR}/${f}"
            exit 1
        fi
    done
fi

# ----------------------------------------------------------------------
# E1: text-condition probe (GPU)
# ----------------------------------------------------------------------
if [[ "${SKIP_E1}" != "1" ]]; then
    LOG="${LOG_DIR}/e1_text_probe.log"
    T0=$(date +%s)
    echo
    echo "[$(date '+%F %T')] E1 text_condition_probe on cuda:${GPU} → ${LOG}"
    set +e
    env CUDA_VISIBLE_DEVICES="${GPU}" \
        PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
        conda run --no-capture-output -n piano python -u \
        scripts/stage_b_generator/round30_text_condition_probe.py \
        --config "${CFG}" \
        --ckpt "${CKPT}" \
        --ild-selection "${OUTDIR}/selection_val.json" \
        --control-selection "${OUTDIR}/selection_control.json" \
        --output-dir "${OUTDIR}" \
        --variant-id "${VARIANT_ID}" \
        --bucket val \
        --cfg-scale 1.0 --seed 42 \
        2>&1 | tee "${LOG}"
    E1_RC=${PIPESTATUS[0]}
    set -e
    T1=$(date +%s)
    echo "[$(date '+%F %T')] E1 done in $((T1-T0))s, rc=${E1_RC}"
    if [[ ${E1_RC} -eq 2 ]]; then
        echo "[R30] GATE FAILED at E1: text condition dead (label='text_dead')"
        echo "    Recommend STOP and move to D4 (text architecture rework)."
        echo "    Inspect ${OUTDIR}/text_probe_summary.md for details."
        exit 2
    elif [[ ${E1_RC} -ne 0 ]]; then
        echo "[R30] FATAL: E1 failed with rc=${E1_RC}"
        exit 1
    fi
else
    echo "[R30] --skip-e1: skipping text probe"
fi

# ----------------------------------------------------------------------
# Pack outputs
# ----------------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
TARBALL="analyses/round30_ild_diagnosis_${STAMP}.tar.gz"
echo
echo "[$(date '+%F %T')] PACK → ${TARBALL}"
tar -czf "${TARBALL}" -C analyses "$(basename "${OUTDIR}")"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "wrote ${TARBALL}  (${SIZE})"
echo
echo "================================================================"
echo "Round-30 E0+E1 complete."
echo "Outputs:"
echo "  ${OUTDIR}/subset_stats.md"
echo "  ${OUTDIR}/text_probe_summary.md  (if E1 ran)"
echo "  ${OUTDIR}/text_probe_stats.json  (if E1 ran)"
echo "  ${OUTDIR}/selection_{train,val,control}.json"
echo "Logs:    ${LOG_DIR}/{e0,e1}_*.log"
echo "Tarball: ${TARBALL}"
echo
echo "scp back: scp <server>:$(pwd)/${TARBALL} ."
echo "Then on local: tar -xzf $(basename "${TARBALL}") -C analyses/"
echo
echo "Next decision: read text_probe_summary.md verdict, then"
echo "  - if text_dead → STOP, plan D4 (text architecture rework)"
echo "  - if text alive → proceed to E2 + E3 (B content + injection probes)"
echo "    which require new B5/B6/B7 oracle variants — that lands in a"
echo "    follow-up commit once we know it is worth the cost."
echo "================================================================"
