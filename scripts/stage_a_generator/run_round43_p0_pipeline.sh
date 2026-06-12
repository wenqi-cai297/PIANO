#!/usr/bin/env bash
# Round-43 P0 — full pipeline driver.
#
# Pipeline (sequential):
#   0. Preflight: verify A2 cfg + A2 ckpt + V8/V6 ckpt + PB1 ckpt + R38-B1
#      Stage-1.5 ckpt (for the R42 2x2 OO/GO cells) + Stage-1 normalizer.
#   1. Dump full train + val (subset, seq_id) selection JSONs from A2 cfg.
#   2. Sample A2's Stage-1 z-scored coarse for train + val into one flat
#      cache root (Codex r43_stage_a_code_review §1 — flat layout).
#   3. Cache audit — disjointness + integrity + distribution check.
#   4. Render R43 P0 Stage-1.5 cfg from the template (substitute cache root).
#   5. Train Stage-1.5 P0 (R38-B1 recipe + cond_source=mixed + sigma=0.02).
#   6. Re-run R42 2x2 with the new Stage-1.5 ckpt — all 4 cells.
#   7. Pack everything for sync-back.
#
# Codex r43_p0_finalized §4.3 — pipeline runs on CUDA_VISIBLE_DEVICES=0,2.
#
# Usage:
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   tmux new -s r43
#   bash scripts/stage_a_generator/run_round43_p0_pipeline.sh
#
# Common overrides:
#   ROUND43_GPUS="0,2"
#   ROUND43_DRY_RUN=1
#   ROUND43_SKIP_SAMPLE=1   (assume cache already populated)
#   ROUND43_SKIP_AUDIT=1    (skip audit; not recommended)
#   ROUND43_SKIP_TRAIN=1    (assume final.pt exists)
#   ROUND43_SKIP_2X2=1      (skip post-training R42 2x2 re-run)
#   ROUND43_SKIP_PACK=1
#   ROUND43_A2_CFG=...
#   ROUND43_A2_CKPT=...
#   ROUND43_TEMPLATE_CFG=configs/training/stage1p5_r43_p0_mixed_a2.yaml.template
#   ROUND43_OUT_DIR_NAME=stage1p5_r43_p0_mixed_a2     # for runs/training/
#
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[R43] FATAL: export DATASETS_ROOT before launching." >&2
    exit 1
fi

# ─── Defaults + overrides ─────────────────────────────────────────────
DRY_RUN="${ROUND43_DRY_RUN:-0}"
SKIP_SAMPLE="${ROUND43_SKIP_SAMPLE:-0}"
SKIP_AUDIT="${ROUND43_SKIP_AUDIT:-0}"
SKIP_TRAIN="${ROUND43_SKIP_TRAIN:-0}"
SKIP_2X2="${ROUND43_SKIP_2X2:-0}"
SKIP_PACK="${ROUND43_SKIP_PACK:-0}"

GPUS="${ROUND43_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
NUM_GPUS_IN_MASK="$(echo "${GPUS}" | tr ',' '\n' | grep -c '^[0-9]\+$' || true)"
if [[ "${NUM_GPUS_IN_MASK}" -lt 1 ]]; then NUM_GPUS_IN_MASK=1; fi
: "${ROUND43_NUM_PROCESSES:=${NUM_GPUS_IN_MASK}}"

A2_CFG="${ROUND43_A2_CFG:-configs/training/stage1_r41_a2_world_vel.yaml}"
A2_CKPT="${ROUND43_A2_CKPT:-runs/training/stage1_r41_a2_world_vel/final.pt}"
V8V6_CFG="${ROUND43_V8V6_CFG:-configs/training/stage1_v8_v6_full_f1.yaml}"
V8V6_CKPT="${ROUND43_V8V6_CKPT:-runs/training/stage1_v8_v6_full_f1/final.pt}"
PB1_CFG="${ROUND43_PB1_CFG:-configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml}"
PB1_CKPT="${ROUND43_PB1_CKPT:-runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt}"
R38B1_STAGE1P5_CFG="${ROUND43_R38B1_STAGE1P5_CFG:-configs/training/stage1p5_r38_b1_init_pose.yaml}"
R38B1_STAGE1P5_CKPT="${ROUND43_R38B1_STAGE1P5_CKPT:-runs/training/stage1p5_r38_b1_init_pose/final.pt}"
TEMPLATE_CFG="${ROUND43_TEMPLATE_CFG:-configs/training/stage1p5_r43_p0_mixed_a2.yaml.template}"
STAGE1_NORMALIZER="${ROUND43_STAGE1_NORMALIZER:-cache/stage1_coarse_v1_full}"

OUT_DIR_NAME="${ROUND43_OUT_DIR_NAME:-stage1p5_r43_p0_mixed_a2}"
RESOLVED_CFG="${ROUND43_RESOLVED_CFG:-configs/training/${OUT_DIR_NAME}.yaml}"

STAMP="${ROUND43_STAMP:-$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${ROUND43_CACHE_DIR:-analyses/round43_stage1_substitute_conds_a2_${STAMP}}"
SEL_TRAIN="${ROUND43_SEL_TRAIN:-analyses/round43_full_selection_train_${STAMP}.json}"
SEL_VAL="${ROUND43_SEL_VAL:-analyses/round43_full_selection_val_${STAMP}.json}"
AUDIT_DIR="${ROUND43_AUDIT_DIR:-analyses/round43_p0_cache_audit_${STAMP}}"
R42_OUT_ROOT="${ROUND43_R42_OUT_ROOT:-analyses/round43_p0_r42_rerun_${STAMP}}"
LOG_DIR="${ROUND43_LOG_DIR:-runs/round43_p0_${STAMP}}"
SUMMARY_LOG="${LOG_DIR}/summary_${STAMP}.log"
mkdir -p "${LOG_DIR}"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R43] FATAL: no python found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ─── Preflight ────────────────────────────────────────────────────────
log
log "===== R43 P0 pipeline launch ${STAMP} ====="
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS=${GPUS}  NUM_PROCESSES=${ROUND43_NUM_PROCESSES}"
log "A2_CFG=${A2_CFG}"
log "A2_CKPT=${A2_CKPT}"
log "V8V6_CKPT=${V8V6_CKPT}  (R42 OO/GO reference Stage-1)"
log "PB1_CKPT=${PB1_CKPT}"
log "R38B1_STAGE1P5_CKPT=${R38B1_STAGE1P5_CKPT}  (R42 OO/GO reference Stage-1.5)"
log "TEMPLATE_CFG=${TEMPLATE_CFG}"
log "STAGE1_NORMALIZER=${STAGE1_NORMALIZER}"
log "CACHE_DIR=${CACHE_DIR}"
log "SEL_TRAIN=${SEL_TRAIN}"
log "SEL_VAL=${SEL_VAL}"
log "AUDIT_DIR=${AUDIT_DIR}"
log "RESOLVED_CFG=${RESOLVED_CFG}"
log "R42_OUT_ROOT=${R42_OUT_ROOT}"
log "STAMP=${STAMP}"
log

preflight_fail=0
check_file() {
    if [[ ! -f "$1" ]]; then
        log "[R43 PREFLIGHT FAIL] missing file: $1"
        if [[ -n "${2:-}" ]]; then
            log "                     hint: $2"
        fi
        preflight_fail=1
    fi
}
check_dir() {
    if [[ ! -d "$1" ]]; then
        log "[R43 PREFLIGHT FAIL] missing dir: $1"
        if [[ -n "${2:-}" ]]; then
            log "                     hint: $2"
        fi
        preflight_fail=1
    fi
}
check_file "${A2_CFG}" \
    "Codex §4.1 — if it lives under analyses/configs/, copy it or set ROUND43_A2_CFG."
check_file "${A2_CKPT}" "Server should have it from R41 run."
check_file "${V8V6_CFG}"
check_file "${V8V6_CKPT}"
check_file "${PB1_CFG}"
check_file "${PB1_CKPT}"
check_file "${R38B1_STAGE1P5_CFG}"
check_file "${R38B1_STAGE1P5_CKPT}"
check_file "${TEMPLATE_CFG}"
check_dir "${STAGE1_NORMALIZER}" \
    "Expected cache/stage1_coarse_v1_full/normalization_train.json under it."
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R43] FATAL preflight failures." >&2
    exit 1
fi
log "[R43] Preflight OK."
log

# ─── Step 1: Dump selection JSONs ────────────────────────────────────
log "================================================================"
log "[$(date '+%F %T')] STEP 1: dump full selection JSONs"
log "================================================================"
if [[ ! -f "${SEL_TRAIN}" ]]; then
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R43 DRY] would dump train selection to ${SEL_TRAIN}"
    else
        "${PY}" -u scripts/stage_a_generator/dump_full_selection_json.py \
            --config "${A2_CFG}" \
            --bucket train \
            --out-json "${SEL_TRAIN}" 2>&1 | tee -a "${SUMMARY_LOG}"
    fi
else
    log "[R43] SEL_TRAIN exists at ${SEL_TRAIN}; skipping dump."
fi
if [[ ! -f "${SEL_VAL}" ]]; then
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R43 DRY] would dump val selection to ${SEL_VAL}"
    else
        "${PY}" -u scripts/stage_a_generator/dump_full_selection_json.py \
            --config "${A2_CFG}" \
            --bucket val \
            --out-json "${SEL_VAL}" 2>&1 | tee -a "${SUMMARY_LOG}"
    fi
else
    log "[R43] SEL_VAL exists at ${SEL_VAL}; skipping dump."
fi
log

# ─── Step 2: Sample A2 → flat cache ──────────────────────────────────
if [[ ${SKIP_SAMPLE} -eq 0 ]]; then
    log "================================================================"
    log "[$(date '+%F %T')] STEP 2: sample A2 generated cache"
    log "       cache dir = ${CACHE_DIR}  (FLAT layout)"
    log "================================================================"
    mkdir -p "${CACHE_DIR}"
    for BUCKET in train val; do
        SEL_PATH="${SEL_TRAIN}"
        if [[ "${BUCKET}" == "val" ]]; then SEL_PATH="${SEL_VAL}"; fi
        log "[R43] sampling bucket=${BUCKET}  selection=${SEL_PATH}"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[R43 DRY] would sample A2 bucket=${BUCKET}"
            continue
        fi
        # Single-GPU sampling — uses first device of CUDA_VISIBLE_DEVICES.
        "${PY}" -u scripts/stage_a_generator/sample_substitute_conds_cli.py \
            --stage stage1 \
            --config "${A2_CFG}" \
            --ckpt "${A2_CKPT}" \
            --selection-json "${SEL_PATH}" \
            --bucket "${BUCKET}" \
            --out-dir "${CACHE_DIR}" \
            --seed 42 --cfg-scale 1.0 --sampler ddim_eta0 \
            2>&1 | tee -a "${LOG_DIR}/sample_${BUCKET}.log" \
                  | tail -5 | tee -a "${SUMMARY_LOG}" >/dev/null
        log "[R43] bucket=${BUCKET} done (log: ${LOG_DIR}/sample_${BUCKET}.log)"
    done
else
    log "[R43] --skip-sample: skipping cache sampling step."
fi
log

# ─── Step 3: Audit ────────────────────────────────────────────────────
if [[ ${SKIP_AUDIT} -eq 0 ]]; then
    log "================================================================"
    log "[$(date '+%F %T')] STEP 3: cache audit"
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R43 DRY] would audit ${CACHE_DIR}"
    else
        # R44 guardrail: pass --fail-on-warnings by default so a
        # distribution-collapsed cache aborts the pipeline BEFORE
        # Stage-1.5 training begins. Opt out for diagnostic-only runs
        # with ROUND43_AUDIT_PERMISSIVE=1. Codex r43_pipeline_bottleneck §4.
        AUDIT_FLAGS=()
        if [[ "${ROUND43_AUDIT_PERMISSIVE:-0}" != "1" ]]; then
            AUDIT_FLAGS+=( --fail-on-warnings )
            log "[R43] audit guardrail ON (--fail-on-warnings)."
        else
            log "[R43] audit guardrail OFF (ROUND43_AUDIT_PERMISSIVE=1)."
        fi
        "${PY}" -u scripts/stage_a_generator/round43_p0_cache_audit.py \
            --cache-root "${CACHE_DIR}" \
            --sel-train "${SEL_TRAIN}" \
            --sel-val "${SEL_VAL}" \
            --oracle-norm "${STAGE1_NORMALIZER}" \
            --out-dir "${AUDIT_DIR}" \
            "${AUDIT_FLAGS[@]}" \
            2>&1 | tee -a "${SUMMARY_LOG}"
        # Hard fail if the audit's exit code is non-zero.
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log "[R43] FATAL: cache audit failed; see ${AUDIT_DIR}." >&2
            log "[R43]   Pass ROUND43_AUDIT_PERMISSIVE=1 to override (NOT recommended for training)."
            exit 1
        fi
    fi
else
    log "[R43] --skip-audit: NOT RECOMMENDED. Continuing anyway."
fi
log

# ─── Step 4: Render concrete cfg from template ───────────────────────
log "================================================================"
log "[$(date '+%F %T')] STEP 4: render resolved cfg"
log "       template -> ${RESOLVED_CFG}"
log "================================================================"
if [[ ${DRY_RUN} -eq 1 ]]; then
    log "[R43 DRY] would substitute __STAGE1_GENERATED_CACHE_ROOT__ -> ${CACHE_DIR}"
else
    # Use a python one-liner that does a true literal string replace, to
    # avoid sed's path-escaping issues.
    "${PY}" - <<EOF
import pathlib, sys
src = pathlib.Path("${TEMPLATE_CFG}").read_text(encoding="utf-8")
out = src.replace("__STAGE1_GENERATED_CACHE_ROOT__", "${CACHE_DIR}")
if "__STAGE1_GENERATED_CACHE_ROOT__" in out:
    sys.exit("[R43] template still has placeholder after substitution — abort.")
pathlib.Path("${RESOLVED_CFG}").write_text(out, encoding="utf-8")
print("[R43] wrote", "${RESOLVED_CFG}")
EOF
    # Stash a copy alongside training output for provenance.
    mkdir -p "runs/training/${OUT_DIR_NAME}"
    cp "${RESOLVED_CFG}" "runs/training/${OUT_DIR_NAME}/${OUT_DIR_NAME}_resolved.yaml"
fi
log

# ─── Step 5: Train Stage-1.5 P0 ──────────────────────────────────────
TRAIN_LOG="${LOG_DIR}/train_${OUT_DIR_NAME}.log"
if [[ ${SKIP_TRAIN} -eq 0 ]]; then
    log "================================================================"
    log "[$(date '+%F %T')] STEP 5: train Stage-1.5 P0"
    log "       cfg=${RESOLVED_CFG}"
    log "       cuda=${GPUS}  procs=${ROUND43_NUM_PROCESSES}"
    log "================================================================"
    NEW_STAGE1P5_CKPT="runs/training/${OUT_DIR_NAME}/final.pt"
    if [[ -f "${NEW_STAGE1P5_CKPT}" && "${ROUND43_FORCE_RETRAIN:-0}" != "1" ]]; then
        log "[R43] ${NEW_STAGE1P5_CKPT} exists; skip train (ROUND43_FORCE_RETRAIN=1 to override)"
    elif [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R43 DRY] would launch accelerate Stage-1.5 train"
    else
        if [[ "${ROUND43_NUM_PROCESSES}" -le 1 ]]; then
            TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1p5.py
                --config "${RESOLVED_CFG}")
        else
            TRAIN_CMD=(accelerate launch
                --num_processes "${ROUND43_NUM_PROCESSES}"
                --multi_gpu --mixed_precision bf16
                src/piano/training/train_stage1p5.py --config "${RESOLVED_CFG}")
        fi
        set +e
        "${TRAIN_CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"
        rc=${PIPESTATUS[0]}
        set -e
        if [[ ${rc} -ne 0 ]]; then
            log "[R43] FATAL: Stage-1.5 train failed (rc=${rc})." >&2
            exit 1
        fi
        if [[ ! -f "${NEW_STAGE1P5_CKPT}" ]]; then
            log "[R43] FATAL: final.pt missing after train." >&2
            exit 1
        fi
    fi
else
    log "[R43] --skip-train: assuming runs/training/${OUT_DIR_NAME}/final.pt exists"
fi
log

# ─── Step 6: Re-run R42 2x2 with NEW Stage-1.5 ───────────────────────
NEW_STAGE1P5_CKPT="runs/training/${OUT_DIR_NAME}/final.pt"
if [[ ${SKIP_2X2} -eq 0 ]]; then
    log "================================================================"
    log "[$(date '+%F %T')] STEP 6: R42 2x2 rerun with new Stage-1.5"
    log "       out-root = ${R42_OUT_ROOT}"
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R43 DRY] would call run_round42_cond_2x2_diag.sh"
    else
        ROUND42_2X2_STAGE1_CFG="${A2_CFG}" \
        ROUND42_2X2_STAGE1_CKPT="${A2_CKPT}" \
        ROUND42_2X2_STAGE1P5_CFG="${RESOLVED_CFG}" \
        ROUND42_2X2_STAGE1P5_CKPT="${NEW_STAGE1P5_CKPT}" \
        ROUND42_2X2_PB1_CFG="${PB1_CFG}" \
        ROUND42_2X2_PB1_CKPT="${PB1_CKPT}" \
        ROUND42_2X2_OUT_ROOT="${R42_OUT_ROOT}" \
        ROUND42_2X2_STAMP="${STAMP}" \
            bash scripts/stage_a_generator/run_round42_cond_2x2_diag.sh \
            2>&1 | tee -a "${SUMMARY_LOG}"
    fi
else
    log "[R43] --skip-2x2: skipping R42 rerun."
fi
log

# ─── Step 7: Pack ────────────────────────────────────────────────────
if [[ ${SKIP_PACK} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
    log "================================================================"
    log "[$(date '+%F %T')] STEP 7: pack sync-back tarball"
    log "================================================================"
    ROUND43_STAMP="${STAMP}" \
    ROUND43_LOG_DIR="${LOG_DIR}" \
    ROUND43_CACHE_DIR="${CACHE_DIR}" \
    ROUND43_AUDIT_DIR="${AUDIT_DIR}" \
    ROUND43_R42_OUT_ROOT="${R42_OUT_ROOT}" \
    ROUND43_RESOLVED_CFG="${RESOLVED_CFG}" \
    ROUND43_OUT_DIR_NAME="${OUT_DIR_NAME}" \
    ROUND43_SEL_TRAIN="${SEL_TRAIN}" \
    ROUND43_SEL_VAL="${SEL_VAL}" \
        bash scripts/stage_a_generator/pack_round43_p0_sync.sh \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R43 PACK] packer reported non-zero (continuing)"
fi

log
log "================================================================"
log "[$(date '+%F %T')] R43 P0 PIPELINE DONE  stamp=${STAMP}"
log "================================================================"
log "Resolved cfg : ${RESOLVED_CFG}"
log "Stage-1.5 ckpt: runs/training/${OUT_DIR_NAME}/final.pt"
log "Cache         : ${CACHE_DIR}"
log "Audit         : ${AUDIT_DIR}"
log "R42 rerun     : ${R42_OUT_ROOT}"
log "Summary log   : ${SUMMARY_LOG}"
