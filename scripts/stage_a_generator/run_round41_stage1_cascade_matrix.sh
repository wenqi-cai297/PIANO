#!/usr/bin/env bash
# Round-41 Stage-1 cascade fine-tune matrix.
#
# 5 cells = A0 control + A1-A4 cascade ablation. All cells fine-tune
# from V8 V6 ckpt (drift_max 17.43 cm direct, 16.81 cm via cascade with
# GT C41/S4) under frozen R29 PB1 ckpt.
#
# Pre-launch calibration is its own standalone script (run it first):
#   # First-round NUDGE PROBE (Codex r41_calibration_next_steps 2026-06-02):
#   # Defaults: target_center=0.3, band [0.2, 0.5], cap 5.0, N=5 batches.
#   python scripts/stage_a_generator/round41_cascade_calibration.py
#   python scripts/stage_a_generator/round41_apply_calibration.py \\
#       --calibration analyses/round41_cascade_calibration/<stamp>.json --apply
#   # Re-run calibration to confirm all cells are in-band (or capped).
# Then this launcher trains + diags without re-running smoke checks.
#
# Do NOT use target_center=1.0 for the first formal R41 run: step-0
# parity can drift large mid-training (R36/R37/R40 history). Start at
# 0.3, raise on evidence.
#
# Why N=5 batches by default: single-batch ratio estimates were observed
# to swing 4-8× between consecutive runs on 2026-06-02 (A4 jumped
# 0.122 → 1.258), making the calibration recommendation unreliable
# under that level of noise. Geometric mean over 5 batches keeps
# log-ratio stdev typically < 0.3 (factor 1.35× spread).
#
# Pass --with-inline-calibration to re-enable a per-cell smoke check
# inside the launcher (legacy mode; not recommended — its log parsing
# is less robust than the standalone script).
#
# Usage:
#   tmux new -s r41
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
#
#   bash ... --dry-run
#   bash ... --only stage1_r41_a0_cascade_off,stage1_r41_a1_motion_mse
#
# Env overrides:
#   ROUND41_GPUS="0,2"
#   ROUND41_NUM_PROCESSES=N            accelerate --num_processes
#   ROUND41_BUCKETS="val"              diag bucket
#   ROUND41_BASE_CFG=...               base V8 V6 cfg
#   ROUND41_PB1_CKPT=...               PB1 ship ckpt (must match each
#                                        yaml's cascade.pb1_checkpoint;
#                                        --regen-configs to realign, or
#                                        ROUND41_ALLOW_PB1_CKPT_MISMATCH=1
#                                        to override).
#   ROUND41_PB1_CFG=...                PB1 ship yaml (used by config
#                                        generator when --regen-configs
#                                        is requested).
#   ROUND41_REGEN_CONFIGS=1            force-regenerate all R41 cfgs
#                                        (resets calibration!). Default
#                                        is generate-if-missing.
#   ROUND41_ALLOW_PB1_CKPT_MISMATCH=1  let cfg pb1_ckpt differ from
#                                        launcher PB1_CKPT.
#   ROUND41_INLINE_CALIBRATION=1       legacy in-shell smoke check.
#   ROUND41_ABORT_IF_CASCADE_RATIO_OVER=3.0
#   ROUND41_ALLOW_PARTIAL=1            keep going on per-variant failure

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
# Calibration is its own standalone phase (round41_cascade_calibration.py).
# Default is to skip in-band guard inside the launcher; user runs the
# standalone calibration script first, applies it via
# round41_apply_calibration.py, re-checks, then launches training.
# Pass --with-inline-calibration to opt into the legacy in-band guard.
INLINE_CALIBRATION="${ROUND41_INLINE_CALIBRATION:-0}"
# Config regeneration is OPT-IN. By default the launcher generates
# missing configs only and leaves existing ones alone — this prevents
# round41_apply_calibration.py's w_total writes from being clobbered
# on the next launcher invocation. Pass --regen-configs to force a
# full regeneration (you will need to re-apply calibration afterwards).
REGEN_CONFIGS="${ROUND41_REGEN_CONFIGS:-0}"
FORCE_RETRAIN=0
FORCE_REDIAG=0
ALLOW_PARTIAL="${ROUND41_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND41_BUCKETS:-val}"
ABORT_RATIO="${ROUND41_ABORT_IF_CASCADE_RATIO_OVER:-3.0}"
# Post-train diag suite. By default run all 3 cheap diags: direct,
# R35 stage1_coarse OOD, K-sample diversity. Full cascade is opt-in.
RUN_R35_AUDIT="${ROUND41_RUN_R35_AUDIT:-1}"
RUN_KDIV="${ROUND41_RUN_KDIV:-1}"
WITH_FULL_CASCADE="${ROUND41_WITH_FULL_CASCADE:-0}"
KDIV_NUM_SAMPLES="${ROUND41_KDIV_NUM_SAMPLES:-8}"
KDIV_CFG_SCALE="${ROUND41_KDIV_CFG_SCALE:-1.0}"
# Full-cascade Stage-1.5 ckpt (R38-B1 ship reference).
STAGE1P5_CFG="${ROUND41_STAGE1P5_CFG:-configs/training/stage1p5_r38_b1_init_pose.yaml}"
STAGE1P5_CKPT="${ROUND41_STAGE1P5_CKPT:-runs/training/stage1p5_r38_b1_init_pose/final.pt}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)                     ONLY="$2"; shift 2 ;;
        --dry-run)                  DRY_RUN=1; shift ;;
        --skip-train)               SKIP_TRAIN=1; shift ;;
        --skip-diag)                SKIP_DIAG=1; shift ;;
        --with-inline-calibration)  INLINE_CALIBRATION=1; shift ;;
        --regen-configs)            REGEN_CONFIGS=1; shift ;;
        --with-full-cascade)        WITH_FULL_CASCADE=1; shift ;;
        --no-r35-audit)             RUN_R35_AUDIT=0; shift ;;
        --no-kdiv)                  RUN_KDIV=0; shift ;;
        --force-retrain)            FORCE_RETRAIN=1; shift ;;
        --force-rediag)             FORCE_REDIAG=1; shift ;;
        --buckets)                  BUCKETS_STR="$2"; shift 2 ;;
        -h|--help)                  sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[R41] FATAL: export DATASETS_ROOT before launch." >&2
    exit 1
fi

GPUS="${ROUND41_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
NUM_GPUS_IN_MASK="$(echo "${GPUS}" | tr ',' '\n' | grep -c '^[0-9]\+$' || true)"
if [[ "${NUM_GPUS_IN_MASK}" -lt 1 ]]; then NUM_GPUS_IN_MASK=1; fi
: "${ROUND41_NUM_PROCESSES:=${NUM_GPUS_IN_MASK}}"
export ROUND41_NUM_PROCESSES

ROUND41_BASE_CFG="${ROUND41_BASE_CFG:-configs/training/stage1_v8_v6_full_f1.yaml}"
PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CKPT="${ROUND41_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"
PB1_CFG="${ROUND41_PB1_CFG:-configs/training/anchordiff_${PB1_VARIANT}.yaml}"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"

OVERALL_LOG_DIR="runs/round41_cascade_matrix"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R41] FATAL: no python found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# Hard-coded variant table — matches round41_make_stage1_cascade_configs.py.
VARIANT_TABLE=(
    "stage1_r41_a0_cascade_off   configs/training/stage1_r41_a0_cascade_off.yaml   runs/training/stage1_r41_a0_cascade_off"
    "stage1_r41_a1_motion_mse    configs/training/stage1_r41_a1_motion_mse.yaml    runs/training/stage1_r41_a1_motion_mse"
    "stage1_r41_a2_world_vel     configs/training/stage1_r41_a2_world_vel.yaml     runs/training/stage1_r41_a2_world_vel"
    "stage1_r41_a3_l_pos_full    configs/training/stage1_r41_a3_l_pos_full.yaml    runs/training/stage1_r41_a3_l_pos_full"
    "stage1_r41_a4_anchor_pos    configs/training/stage1_r41_a4_anchor_pos.yaml    runs/training/stage1_r41_a4_anchor_pos"
)

VARIANTS=""
for row in "${VARIANT_TABLE[@]}"; do
    VID="$(echo "${row}" | awk '{print $1}')"
    if [[ -n "${ONLY}" ]]; then
        case ",${ONLY}," in *",${VID},"*) ;; *) continue ;; esac
    fi
    VARIANTS+="${row}"$'\n'
done

# ─── Conditional config (re)generation ───────────────────────────────
# Default: only generate when a selected variant's cfg is missing.
# --regen-configs forces a full regeneration (overwrites all configs,
# including any cascade.w_total values applied by the calibration
# pipeline; log loudly so the operator can re-calibrate).
NEED_GEN=0
if [[ ${REGEN_CONFIGS} -eq 1 ]]; then
    NEED_GEN=1
    log "[R41] --regen-configs set: WILL REGENERATE all R41 configs."
    log "[R41] Any calibrated cascade.w_total values will be reset to 1.0."
    log "[R41] You must re-run round41_cascade_calibration.py + apply afterwards."
else
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        if [[ ! -f "${CFG}" ]]; then
            NEED_GEN=1
            log "[R41] cfg missing: ${CFG} → will (re)generate."
        fi
    done <<< "${VARIANTS}"
fi
if [[ ${NEED_GEN} -eq 1 && ${DRY_RUN} -eq 0 ]]; then
    log "[R41] Generating R41 configs from ${ROUND41_BASE_CFG}."
    "${PY}" scripts/stage_a_generator/round41_make_stage1_cascade_configs.py \
        --base-cfg "${ROUND41_BASE_CFG}" \
        --pb1-config "${PB1_CFG:-configs/training/anchordiff_r29_pb_a1_adaln_s4.yaml}" \
        --pb1-ckpt "${PB1_CKPT}" \
        --out-dir configs/training/ 2>&1 | tee -a "${SUMMARY_LOG}"
elif [[ ${NEED_GEN} -eq 0 ]]; then
    log "[R41] All selected R41 cfgs exist; skipping regeneration."
    log "[R41] (pass --regen-configs to force regen — will reset w_total to 1.0)"
fi

log
log "===== R41 cascade matrix launch ${STAMP} ====="
log "*** R36/R37 GUARDRAIL: standalone calibration phase ***"
log "If you haven't run the standalone calibration yet (nudge probe defaults: center=0.3, N=5 batches):"
log "   python scripts/stage_a_generator/round41_cascade_calibration.py"
log "   python scripts/stage_a_generator/round41_apply_calibration.py \\"
log "       --calibration \$(ls -t analyses/round41_cascade_calibration/*.json | head -1) --apply"
log "Then re-run this launcher. Inline calibration is off by default."
log
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS=${GPUS}  NUM_PROCESSES=${ROUND41_NUM_PROCESSES}"
log "PB1_CKPT=${PB1_CKPT}"
log "Variants:"
echo "${VARIANTS}" | sed 's/^/  /' | tee -a "${SUMMARY_LOG}"
log

# ─── Preflight ───────────────────────────────────────────────────────
preflight_fail=0
if [[ ${DRY_RUN} -eq 0 ]]; then
    if [[ ! -f "${PB1_CKPT}" ]]; then
        log "[R41 PREFLIGHT FAIL] PB1 ckpt missing: ${PB1_CKPT}"
        preflight_fail=1
    fi
    if [[ ! -f "runs/training/stage1_v8_v6_full_f1/final.pt" ]]; then
        log "[R41 PREFLIGHT FAIL] V8 V6 ckpt missing"
        preflight_fail=1
    fi
fi
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R41] FATAL preflight failures."
    exit 1
fi

# ─── Pre-train config audit ──────────────────────────────────────────
# Read each variant's yaml, log cascade fields, verify PB1 ckpt path
# matches the launcher's PB1_CKPT (single source of truth — Codex
# blocker §4), warn if cascade.w_total == 1.0 on a non-control cell
# (likely missed calibration).
ALLOW_PB1_MISMATCH="${ROUND41_ALLOW_PB1_CKPT_MISMATCH:-0}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    log
    log "─── Pre-train config audit ──────────────────────────────────────"
    audit_fail=0
    while IFS=' ' read -r VID CFG OUTDIR; do
        [[ -z "${VID}" ]] && continue
        [[ ! -f "${CFG}" ]] && continue
        AUDIT_JSON="$("${PY}" -c "
import json, sys
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CFG}')
casc = cfg.get('cascade', None)
out = {
    'enabled': bool(casc.enabled) if casc else False,
    'w_total': float(casc.get('w_total', 1.0)) if casc else 1.0,
    'w_motion_mse': float(casc.get('w_motion_mse', 0.0)) if casc else 0.0,
    'w_world_joint_vel': float(casc.get('w_world_joint_vel', 0.0)) if casc else 0.0,
    'w_l_pos_full': float(casc.get('w_l_pos_full', 0.0)) if casc else 0.0,
    'w_anchor_joint_pos': float(casc.get('w_anchor_joint_pos', 0.0)) if casc else 0.0,
    'pb1_checkpoint': str(casc.get('pb1_checkpoint', '')) if casc else '',
    'init_checkpoint': str(cfg.training.get('init_checkpoint', '')),
}
print(json.dumps(out))
" 2>&1)"
        log "[audit] ${VID}: ${AUDIT_JSON}"
        # Parse a couple of fields back out for checks (jq may not be
        # available; fall back to python).
        CFG_PB1="$("${PY}" -c "import json; print(json.loads('''${AUDIT_JSON}''')['pb1_checkpoint'])" 2>/dev/null || echo "")"
        CFG_ENABLED="$("${PY}" -c "import json; print(json.loads('''${AUDIT_JSON}''')['enabled'])" 2>/dev/null || echo "False")"
        CFG_W_TOTAL="$("${PY}" -c "import json; print(json.loads('''${AUDIT_JSON}''')['w_total'])" 2>/dev/null || echo "1.0")"
        if [[ "${CFG_ENABLED}" == "True" ]]; then
            # Verify PB1 ckpt matches launcher's.
            if [[ "${CFG_PB1}" != "${PB1_CKPT}" ]]; then
                if [[ "${ALLOW_PB1_MISMATCH}" == "1" ]]; then
                    log "[audit] WARN ${VID} pb1_checkpoint='${CFG_PB1}' != launcher PB1_CKPT='${PB1_CKPT}'"
                    log "[audit]      (ROUND41_ALLOW_PB1_CKPT_MISMATCH=1, continuing)"
                else
                    log "[audit] FATAL ${VID} pb1_checkpoint='${CFG_PB1}' != launcher PB1_CKPT='${PB1_CKPT}'"
                    log "[audit]       Training would load a different PB1 than diag uses, invalidating the experiment."
                    log "[audit]       Either: (a) --regen-configs to align, or"
                    log "[audit]               (b) export ROUND41_ALLOW_PB1_CKPT_MISMATCH=1."
                    audit_fail=1
                fi
            fi
            # Warn on default w_total (likely missed calibration). A
            # control cell legitimately stays at 1.0 because cascade is
            # disabled; only warn on enabled cascade cells.
            if awk -v w="${CFG_W_TOTAL}" 'BEGIN {exit !(w == 1.0)}'; then
                log "[audit] WARN ${VID} cascade.w_total == 1.0 (default). Did you run"
                log "[audit]      round41_cascade_calibration.py + round41_apply_calibration.py --apply ?"
                log "[audit]      Training will proceed but step-0 grad ratio is uncalibrated."
            fi
        else
            log "[audit] ${VID}: control cell (cascade disabled) — calibration N/A"
        fi
    done <<< "${VARIANTS}"
    if [[ ${audit_fail} -ne 0 ]]; then
        log "[R41] FATAL config audit failures."
        exit 1
    fi
    log
fi

# ─── Per-variant loop ──────────────────────────────────────────────
TRAINED_OK_VIDS=()
DIAGED_OK_VIDS=()

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    VARIANT_LOG="${OVERALL_LOG_DIR}/${VID}.log"

    # ─── Inline calibration smoke (legacy; off by default) ────────
    # Standalone calibration is preferred:
    #   bash scripts/stage_a_generator/run_round41_stage1_cascade_matrix.sh
    #   → pre-train phase = round41_cascade_calibration.py (off-launcher)
    if [[ ${INLINE_CALIBRATION} -eq 1 && ${DRY_RUN} -eq 0 ]]; then
        log
        log "================================================================"
        log "[$(date '+%F %T')] CALIBRATION SMOKE ${VID}"
        log "================================================================"
        SMOKE_LOG="${OVERALL_LOG_DIR}/${VID}.smoke.log"
        set +e
        # Smoke test runs single-GPU single-process for clean log; the
        # trainer's --smoke-test path is non-DDP.
        CUDA_VISIBLE_DEVICES="$(echo "${GPUS}" | cut -d',' -f1)" \
            "${PY}" -u src/piano/training/train_stage1.py \
                --config "${CFG}" --smoke-test 2>&1 | tee "${SMOKE_LOG}"
        smoke_rc=${PIPESTATUS[0]}
        set -e
        if [[ ${smoke_rc} -ne 0 ]]; then
            log "[R41] [${VID}] smoke test FAILED (rc=${smoke_rc})"
            if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
            continue
        fi
        # Parse cascade ratio from smoke log if cascade is enabled.
        RATIO_LINE=$(grep "R41 cascade weighted" "${SMOKE_LOG}" 2>/dev/null | tail -1 || true)
        if [[ -n "${RATIO_LINE}" ]]; then
            RATIO_VAL=$(echo "${RATIO_LINE}" | awk -F'weighted/mse_x0=' '{print $2}' | awk '{print $1}')
            log "[R41] [${VID}] cascade weighted/mse_x0 = ${RATIO_VAL}"
            # Use awk for float comparison.
            if [[ -n "${RATIO_VAL}" ]] && awk -v r="${RATIO_VAL}" -v t="${ABORT_RATIO}" \
                'BEGIN {exit !(r > t)}'; then
                log "[R41] [${VID}] FATAL: cascade ratio ${RATIO_VAL} > ${ABORT_RATIO}."
                log "[R41]   Edit ${CFG} and reduce cascade.w_total (try /5)."
                log "[R41]   Then re-run with --only ${VID}."
                if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                continue
            fi
        fi
    fi

    # ─── Phase 1: TRAIN ─────────────────────────────────────────────
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ -f "${FINAL}" && ${FORCE_RETRAIN} -eq 0 ]]; then
            log "[R41] [${VID}] ckpt already exists; skipping (use --force-retrain)"
            TRAINED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] TRAIN ${VID}"
            log "    cfg: ${CFG}    out: ${OUTDIR}"
            log "    GPUs: ${GPUS}  procs=${ROUND41_NUM_PROCESSES}"
            log "================================================================"
            if [[ "${ROUND41_NUM_PROCESSES}" -le 1 ]]; then
                TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1.py
                    --config "${CFG}")
            else
                TRAIN_CMD=(accelerate launch
                    --num_processes "${ROUND41_NUM_PROCESSES}"
                    --multi_gpu --mixed_precision bf16
                    src/piano/training/train_stage1.py --config "${CFG}")
            fi
            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R41 DRY-RUN] would train ${VID}"
                log "    \$ CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD[*]}"
                TRAINED_OK_VIDS+=("${VID}")
            else
                set +e
                "${TRAIN_CMD[@]}" 2>&1 | tee "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -f "${FINAL}" ]]; then
                    TRAINED_OK_VIDS+=("${VID}")
                    log "[R41] [${VID}] TRAIN OK"
                else
                    log "[R41] [${VID}] TRAIN FAILED (rc=${rc})"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                    continue
                fi
            fi
        fi
    else
        log "[R41] --skip-train: skipping for ${VID}"
        TRAINED_OK_VIDS+=("${VID}")
    fi

    # ─── Phase 2: Direct diag (Stage-1 → PB1, oracle C41/S4) ───────
    if [[ ${SKIP_DIAG} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ ! -f "${FINAL}" && ${DRY_RUN} -eq 0 ]]; then
            log "[R41] [${VID}] no ckpt to diag; skip"
            continue
        fi
        DIRECT_ARCHIVE="analyses/round41_stage1_direct_diag_${VID}"
        DIAG_DONE_MARKER="${DIRECT_ARCHIVE}/sustained_contact_val/sustained_contact_summary.md"
        if [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
            log "[R41] [${VID}] direct diag already archived; skip"
            DIAGED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] DIRECT DIAG ${VID}  (buckets: ${BUCKETS_STR})"
            log "================================================================"
            DS_OUT_TAG="_r41_${VID}"
            DIAG_DIR_ROOT="analyses/round31_stage1_downstream_diag${DS_OUT_TAG}"
            DS_SUB_DIR_ROOT="analyses/round31_stage1_substitute_conds${DS_OUT_TAG}"
            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R41 DRY-RUN] direct diag ${VID}"
                DIAGED_OK_VIDS+=("${VID}")
            else
                rm -rf "${DS_SUB_DIR_ROOT}" "${DIAG_DIR_ROOT}"
                set +e
                ROUND31_DS_STAGE1_CFG="${CFG}" \
                ROUND31_DS_STAGE1_CKPT="${FINAL}" \
                ROUND31_DS_PB1_CKPT="${PB1_CKPT}" \
                ROUND31_DS_STAGE1_CFG_SCALE=1.0 \
                ROUND31_DS_STAGE1_SAMPLER=ddim_eta0 \
                ROUND31_DS_BUCKETS="${BUCKETS_STR}" \
                ROUND31_DS_OUT_TAG="${DS_OUT_TAG}" \
                    bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh \
                    2>&1 | tee -a "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -d "${DIAG_DIR_ROOT}" ]]; then
                    rm -rf "${DIRECT_ARCHIVE}"
                    mv "${DIAG_DIR_ROOT}" "${DIRECT_ARCHIVE}"
                    # Preserve the substitute conds cache so R35 and
                    # K-diversity audits can read it. Archive to R41
                    # canonical path.
                    R41_SUB_DIR="analyses/round41_stage1_substitute_conds_${VID}"
                    if [[ -d "${DS_SUB_DIR_ROOT}" ]]; then
                        rm -rf "${R41_SUB_DIR}"
                        mv "${DS_SUB_DIR_ROOT}" "${R41_SUB_DIR}"
                    fi
                    log "[R41] [${VID}] DIRECT DIAG OK -> ${DIRECT_ARCHIVE}"
                    DIAGED_OK_VIDS+=("${VID}")
                else
                    log "[R41] [${VID}] DIRECT DIAG FAILED (rc=${rc})"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                fi
            fi
        fi

        # ─── Phase 3: R35 stage1_coarse OOD audit ──────────────────
        R41_SUB_DIR="analyses/round41_stage1_substitute_conds_${VID}"
        R35_OUT="analyses/round41_stage1_ood_${VID}"
        if [[ ${RUN_R35_AUDIT} -eq 1 && ${DRY_RUN} -eq 0 \
              && -d "${R41_SUB_DIR}" ]]; then
            log
            log "[R41] [${VID}] R35 stage1_coarse OOD audit"
            mkdir -p "${R35_OUT}"
            FIRST_BUCKET="${BUCKETS_STR%% *}"
            SEL_JSON="${SELECTION_VAL}"
            case "${FIRST_BUCKET}" in
                train) SEL_JSON="${SELECTION_TRAIN:-analyses/round27_tier0_train_indices_48_balanced.json}" ;;
            esac
            set +e
            "${PY}" -u scripts/stage_a_generator/round35_stage1_coarse_ood_audit.py \
                --config "${CFG}" \
                --generated-dir "${R41_SUB_DIR}" \
                --selection-json "${SEL_JSON}" \
                --bucket "${FIRST_BUCKET}" \
                --out-md "${R35_OUT}/stage1_coarse_ood_audit_${FIRST_BUCKET}.md" \
                --out-json "${R35_OUT}/stage1_coarse_ood_audit_${FIRST_BUCKET}.json" \
                2>&1 | tee -a "${VARIANT_LOG}"
            rc=${PIPESTATUS[0]}
            set -e
            if [[ ${rc} -eq 0 ]]; then
                log "[R41] [${VID}] R35 AUDIT OK -> ${R35_OUT}"
            else
                log "[R41] [${VID}] R35 AUDIT FAILED (rc=${rc}); continuing"
            fi
        elif [[ ${RUN_R35_AUDIT} -eq 1 && ${DRY_RUN} -eq 1 ]]; then
            log "[R41 DRY-RUN] [${VID}] R35 OOD audit"
        fi

        # ─── Phase 4: K-sample diversity audit ─────────────────────
        KDIV_OUT="analyses/round41_stage1_kdiv_${VID}"
        if [[ ${RUN_KDIV} -eq 1 && ${DRY_RUN} -eq 0 \
              && -f "${OUTDIR}/final.pt" ]]; then
            log
            log "[R41] [${VID}] K-sample diversity (K=${KDIV_NUM_SAMPLES})"
            FIRST_BUCKET="${BUCKETS_STR%% *}"
            SEL_JSON="${SELECTION_VAL}"
            case "${FIRST_BUCKET}" in
                train) SEL_JSON="${SELECTION_TRAIN:-analyses/round27_tier0_train_indices_48_balanced.json}" ;;
            esac
            set +e
            "${PY}" -u scripts/stage_a_generator/round40_stage1_k_sample_audit.py \
                --config "${CFG}" \
                --ckpt "${OUTDIR}/final.pt" \
                --selection-json "${SEL_JSON}" \
                --bucket "${FIRST_BUCKET}" \
                --out-dir "${KDIV_OUT}" \
                --num-samples "${KDIV_NUM_SAMPLES}" \
                --cfg-scale "${KDIV_CFG_SCALE}" \
                2>&1 | tee -a "${VARIANT_LOG}"
            rc=${PIPESTATUS[0]}
            set -e
            if [[ ${rc} -eq 0 ]]; then
                log "[R41] [${VID}] KDIV OK -> ${KDIV_OUT}"
            else
                log "[R41] [${VID}] KDIV FAILED (rc=${rc}); continuing"
            fi
        elif [[ ${RUN_KDIV} -eq 1 && ${DRY_RUN} -eq 1 ]]; then
            log "[R41 DRY-RUN] [${VID}] KDIV"
        fi

        # ─── Phase 5: Full cascade diag (opt-in) ───────────────────
        if [[ ${WITH_FULL_CASCADE} -eq 1 && ${DRY_RUN} -eq 0 \
              && -d "${R41_SUB_DIR}" && -f "${STAGE1P5_CKPT}" ]]; then
            log
            log "[R41] [${VID}] FULL CASCADE (Stage-1 → R38-B1 → PB1)"
            FC_OUT_TAG="_r41_${VID}"
            FC_SUB_DIR="analyses/round32_stage1p5_substitute_conds${FC_OUT_TAG}"
            FC_DIAG_DIR="analyses/round32_stage1p5_downstream_diag${FC_OUT_TAG}"
            FC_ARCHIVE="analyses/round41_full_cascade_${VID}"
            rm -rf "${FC_SUB_DIR}" "${FC_DIAG_DIR}"
            set +e
            ROUND32_DS_STAGE1P5_CFG="${STAGE1P5_CFG}" \
            ROUND32_DS_STAGE1P5_CKPT="${STAGE1P5_CKPT}" \
            ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
            ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
            ROUND32_DS_OUT_TAG="${FC_OUT_TAG}" \
            ROUND32_DS_UPSTREAM_DIR="${R41_SUB_DIR}" \
                bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
                2>&1 | tee -a "${VARIANT_LOG}"
            rc=${PIPESTATUS[0]}
            set -e
            if [[ ${rc} -eq 0 && -d "${FC_DIAG_DIR}" ]]; then
                rm -rf "${FC_ARCHIVE}"
                mv "${FC_DIAG_DIR}" "${FC_ARCHIVE}"
                log "[R41] [${VID}] FULL CASCADE OK -> ${FC_ARCHIVE}"
            else
                log "[R41] [${VID}] FULL CASCADE FAILED (rc=${rc}); continuing"
            fi
        elif [[ ${WITH_FULL_CASCADE} -eq 1 && ${DRY_RUN} -eq 1 ]]; then
            log "[R41 DRY-RUN] [${VID}] FULL CASCADE"
        fi
    fi
done <<< "${VARIANTS}"

# ─── Summary ────────────────────────────────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] R41 COMPLETE"
log "================================================================"
log "Trained: ${TRAINED_OK_VIDS[*]:-none}"
log "Diaged:  ${DIAGED_OK_VIDS[*]:-none}"

# ─── Pack ───────────────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round41_cascade_results_${STAMP}.tar.gz"
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    ROUND41_STAMP="${STAMP}" \
    ROUND41_TARBALL="${TARBALL}" \
    ROUND41_SUMMARY_LOG="${SUMMARY_LOG}" \
    ROUND41_VARIANT_LOG_DIR="${OVERALL_LOG_DIR}" \
    ROUND41_TRAINED_VIDS="${TRAINED_OK_VIDS[*]:-}" \
    ROUND41_DIAGED_VIDS="${DIAGED_OK_VIDS[*]:-}" \
        bash scripts/stage_a_generator/pack_round41_cascade_sync.sh \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R41 PACK] packer failed (non-fatal)"
fi
