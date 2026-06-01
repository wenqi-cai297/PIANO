#!/usr/bin/env bash
# Round-40 Stage-1 plan-sampler ablation matrix.
#
# R40 turns Stage-1 from a "23-D GT regression target" into a
# "coarse-plan sampler" by (i) downweighting exact-GT MSE on under-
# determined channels and (ii) adding a plan-invariant loss that
# supervises plan-level invariants. See:
#   analyses/2026-06-01_round40_stage1_plan_sampler_handoff_for_claude.md
#   analyses/2026-06-01_stage1_underdetermination_for_codex.md
#
# 4 cells — all built on V8 V6 substrate:
#
#   C0 baseline           — exact V8 V6 (sanity baseline).
#   C1 weak GT            — channel-weighted MSE, no plan loss.
#   C2 plan energy        — C1 + plan-invariant loss at 0.20  (ship candidate).
#   C3 strong plan energy — stronger weights + plan loss at 0.50 (probe).
#
# Per-variant phases:
#   1) TRAIN — accelerate launch on ROUND40_GPUS (default 0,2).
#   2) DIAG  — Stage-1 → frozen PB1 (oracle C41/S4) via R31 downstream
#              diag launcher. Isolates Stage-1 cond quality.
#   3) PLAN  — round40_stage1_plan_diag.py on the substitute-conds dir.
#   4) CASC  — (optional) full cascade: Stage-1 → R38-B1 → PB1.
#   5) KDIV  — (optional) K-sample diversity audit.
# Then a comparison summary md + packed tarball.
#
# Usage:
#   tmux new -s r40
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh
#
#   bash scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh --dry-run
#   bash scripts/stage_a_generator/run_round40_stage1_plan_matrix.sh \
#       --only stage1_r40_c0_v8v6_baseline,stage1_r40_c2_plan_energy
#
# Env overrides (all prefixed ROUND40_*):
#   ROUND40_GPUS="0,2"                CUDA_VISIBLE_DEVICES mask
#   ROUND40_NUM_PROCESSES=N           accelerate --num_processes
#   ROUND40_BUCKETS="val"             diag bucket
#   ROUND40_BASE_CFG=...              base cfg (default V8 V6)
#   ROUND40_PB1_CKPT=...              PB1 ckpt (default r29_pb_a1_adaln_s4)
#   ROUND40_STAGE1P5_B1_CFG=...       Stage-1.5 R38-B1 cfg (for full cascade)
#   ROUND40_STAGE1P5_B1_CKPT=...      Stage-1.5 R38-B1 ckpt
#   ROUND40_SKIP_FULL_CASCADE=1       skip phase 4
#   ROUND40_K_SAMPLES=8               K for phase 5
#   ROUND40_KDIV_VIDS="..."           comma-separated VIDs for kdiv (default c0,c2,c3)
#   ROUND40_ALLOW_PARTIAL=1           keep going on per-variant failure

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
SKIP_KDIV=0
FORCE_RETRAIN=0
FORCE_REDIAG=0
ALLOW_PARTIAL="${ROUND40_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND40_BUCKETS:-val}"
SKIP_FULL_CASCADE="${ROUND40_SKIP_FULL_CASCADE:-0}"
K_SAMPLES="${ROUND40_K_SAMPLES:-8}"
KDIV_VIDS_STR="${ROUND40_KDIV_VIDS:-stage1_r40_c0_v8v6_baseline,stage1_r40_c2_plan_energy,stage1_r40_c3_plan_energy_strong}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)            ONLY="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --skip-train)      SKIP_TRAIN=1; shift ;;
        --skip-diag)       SKIP_DIAG=1; shift ;;
        --skip-kdiv)       SKIP_KDIV=1; shift ;;
        --force-retrain)   FORCE_RETRAIN=1; shift ;;
        --force-rediag)    FORCE_REDIAG=1; shift ;;
        --buckets)         BUCKETS_STR="$2"; shift 2 ;;
        -h|--help)         sed -n '1,55p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[R40] FATAL: export DATASETS_ROOT before launch." >&2
    echo "    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

GPUS="${ROUND40_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

NUM_GPUS_IN_MASK="$(echo "${GPUS}" | tr ',' '\n' | grep -c '^[0-9]\+$' || true)"
if [[ "${NUM_GPUS_IN_MASK}" -lt 1 ]]; then NUM_GPUS_IN_MASK=1; fi

: "${ROUND40_NUM_PROCESSES:=${NUM_GPUS_IN_MASK}}"
export ROUND40_NUM_PROCESSES

ROUND40_BASE_CFG="${ROUND40_BASE_CFG:-configs/training/stage1_v8_v6_full_f1.yaml}"

PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CKPT="${ROUND40_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"

# Stage-1.5 R38-B1 ckpt for full-cascade phase 4.
STAGE1P5_B1_CFG="${ROUND40_STAGE1P5_B1_CFG:-configs/training/stage1p5_r38_b1_init_pose.yaml}"
STAGE1P5_B1_CKPT="${ROUND40_STAGE1P5_B1_CKPT:-runs/training/stage1p5_r38_b1_init_pose/final.pt}"

OVERALL_LOG_DIR="runs/round40_stage1_plan_matrix"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R40] FATAL: no python found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ─── Ensure base config exists ───────────────────────────────────────
if [[ ! -f "${ROUND40_BASE_CFG}" && ${DRY_RUN} -eq 0 ]]; then
    log "[R40] FATAL: base cfg ${ROUND40_BASE_CFG} missing on server."
    log "    Regenerate Round-31 V8 configs first:"
    log "    python scripts/stage_a_generator/round31_make_stage1_v8_configs.py"
    exit 1
fi

# ─── Regenerate R40 configs (idempotent) ─────────────────────────────
log "[R40] Regenerating Stage-1 R40 ablation configs from ${ROUND40_BASE_CFG}."
if [[ ${DRY_RUN} -eq 0 ]]; then
    "${PY}" scripts/stage_a_generator/round40_make_stage1_plan_configs.py \
        --base-cfg "${ROUND40_BASE_CFG}" \
        --out-dir configs/training/ 2>&1 | tee -a "${SUMMARY_LOG}"
fi

# ─── Hard-coded variant list (matches round40_make_stage1_plan_configs.py) ──
VARIANT_TABLE=(
    "stage1_r40_c0_v8v6_baseline       configs/training/stage1_r40_c0_v8v6_baseline.yaml       runs/training/stage1_r40_c0_v8v6_baseline"
    "stage1_r40_c1_weak_gt             configs/training/stage1_r40_c1_weak_gt.yaml             runs/training/stage1_r40_c1_weak_gt"
    "stage1_r40_c2_plan_energy         configs/training/stage1_r40_c2_plan_energy.yaml         runs/training/stage1_r40_c2_plan_energy"
    "stage1_r40_c3_plan_energy_strong  configs/training/stage1_r40_c3_plan_energy_strong.yaml  runs/training/stage1_r40_c3_plan_energy_strong"
)

VARIANTS=""
for row in "${VARIANT_TABLE[@]}"; do
    VID="$(echo "${row}" | awk '{print $1}')"
    if [[ -n "${ONLY}" ]]; then
        case ",${ONLY}," in *",${VID},"*) ;; *) continue ;; esac
    fi
    VARIANTS+="${row}"$'\n'
done
if [[ -z "${VARIANTS}" ]]; then
    log "[R40] no variants matched only='${ONLY}'"
    exit 0
fi

log
log "===== R40 matrix launch ${STAMP} ====="
log "*** CALIBRATION REMINDER ***"
log "After the first few train epochs, audit each VARIANT's metrics.jsonl"
log "for r40_plan_invariant_weighted vs mse_x0. Target: weighted ≤ ~1×mse_x0."
log "If weighted ≥ 3× mse_x0 in epoch 1, kill, lower w_r40_plan_invariant 5×, restart."
log
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS=${GPUS}  NUM_PROCESSES=${ROUND40_NUM_PROCESSES}"
log "SKIP_TRAIN=${SKIP_TRAIN}  SKIP_DIAG=${SKIP_DIAG}  SKIP_KDIV=${SKIP_KDIV}"
log "BUCKETS=${BUCKETS_STR}  K_SAMPLES=${K_SAMPLES}"
log "PB1_CKPT=${PB1_CKPT}"
log "STAGE1P5_B1=${STAGE1P5_B1_CFG} | ${STAGE1P5_B1_CKPT} (SKIP_FULL_CASCADE=${SKIP_FULL_CASCADE})"
log "Variants to process:"
echo "${VARIANTS}" | sed 's/^/  /' | tee -a "${SUMMARY_LOG}"
log

# ─── Preflight ───────────────────────────────────────────────────────
preflight_fail=0
if [[ ${SKIP_DIAG} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
    if [[ ! -f "${PB1_CKPT}" ]]; then
        log "[R40 PREFLIGHT FAIL] PB1 ckpt missing: ${PB1_CKPT}"
        preflight_fail=1
    fi
    case "${BUCKETS_STR}" in
        *val*)
            [[ ! -f "${SELECTION_VAL}" ]] && {
                log "[R40 PREFLIGHT FAIL] val selection JSON missing: ${SELECTION_VAL}"
                preflight_fail=1
            } ;;
    esac
    case "${BUCKETS_STR}" in
        *train*)
            [[ ! -f "${SELECTION_TRAIN}" ]] && {
                log "[R40 PREFLIGHT FAIL] train selection JSON missing: ${SELECTION_TRAIN}"
                preflight_fail=1
            } ;;
    esac
fi
if [[ ${SKIP_FULL_CASCADE} -eq 0 && ${DRY_RUN} -eq 0 && ${SKIP_DIAG} -eq 0 ]]; then
    if [[ ! -f "${STAGE1P5_B1_CKPT}" ]]; then
        log "[R40 PREFLIGHT FAIL] Stage-1.5 B1 ckpt missing: ${STAGE1P5_B1_CKPT}"
        preflight_fail=1
    fi
fi
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -f "${CFG}" && ${DRY_RUN} -eq 0 ]]; then
        log "[R40 PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R40] FATAL preflight failures."
    exit 1
fi

# ─── Per-variant loop ──────────────────────────────────────────────
TRAINED_OK_VIDS=()
DIAGED_OK_VIDS=()
PLAN_OK_VIDS=()
CASCADED_OK_VIDS=()
KDIV_OK_VIDS=()

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    VARIANT_LOG="${OVERALL_LOG_DIR}/${VID}.log"

    # ─── Phase 1: TRAIN ─────────────────────────────────────────────
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ -f "${FINAL}" && ${FORCE_RETRAIN} -eq 0 ]]; then
            log "[R40] [${VID}] ckpt already exists; skipping train (use --force-retrain)"
            TRAINED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] TRAIN ${VID}"
            log "    config: ${CFG}"
            log "    output: ${OUTDIR}"
            log "    log:    ${VARIANT_LOG}"
            log "    GPUs:   ${GPUS}  procs=${ROUND40_NUM_PROCESSES}"
            log "================================================================"

            if [[ "${ROUND40_NUM_PROCESSES}" -le 1 ]]; then
                TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1.py --config "${CFG}")
            else
                TRAIN_CMD=(accelerate launch
                    --num_processes "${ROUND40_NUM_PROCESSES}"
                    --multi_gpu --mixed_precision bf16
                    src/piano/training/train_stage1.py --config "${CFG}")
            fi

            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R40 DRY-RUN] would train ${VID}"
                log "    \$ CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD[*]}"
                TRAINED_OK_VIDS+=("${VID}")
            else
                set +e
                "${TRAIN_CMD[@]}" 2>&1 | tee "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -f "${FINAL}" ]]; then
                    TRAINED_OK_VIDS+=("${VID}")
                    log "[R40] [${VID}] TRAIN OK -> ${FINAL}"
                else
                    log "[R40] [${VID}] TRAIN FAILED (rc=${rc}, final.pt=$([[ -f ${FINAL} ]] && echo present || echo missing))"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                    continue
                fi
            fi
        fi
    else
        log "[R40] --skip-train: skipping train for ${VID}"
        TRAINED_OK_VIDS+=("${VID}")
    fi

    # ─── Phase 2: DIRECT DIAG (Stage-1 → frozen PB1, oracle C41/S4) ─
    DIRECT_ARCHIVE="analyses/round40_stage1_direct_diag_${VID}"
    SUB_DIR_ROOT="analyses/round40_stage1_substitute_conds_${VID}"
    if [[ ${SKIP_DIAG} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ ! -f "${FINAL}" && ${DRY_RUN} -eq 0 ]]; then
            log "[R40] [${VID}] no ckpt to diag; skipping diag"
            continue
        fi

        DIAG_DONE_MARKER="${DIRECT_ARCHIVE}/sustained_contact_val/sustained_contact_summary.md"
        if [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
            log "[R40] [${VID}] direct diag already archived; skipping (use --force-rediag)"
            DIAGED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] DIRECT DIAG ${VID}  (buckets: ${BUCKETS_STR})"
            log "================================================================"

            DS_OUT_TAG="_r40_${VID}"
            DIAG_DIR_ROOT="analyses/round31_stage1_downstream_diag${DS_OUT_TAG}"
            DS_SUB_DIR_ROOT="analyses/round31_stage1_substitute_conds${DS_OUT_TAG}"

            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R40 DRY-RUN] direct diag ${VID}: ROUND31_DS_* env, run_round31_stage1_downstream_diag.sh"
                DIAGED_OK_VIDS+=("${VID}")
            else
                rm -rf "${DS_SUB_DIR_ROOT}"
                rm -rf "${DIAG_DIR_ROOT}"
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
                    # Move substitute conds to canonical R40 dir.
                    if [[ -d "${DS_SUB_DIR_ROOT}" ]]; then
                        rm -rf "${SUB_DIR_ROOT}"
                        mv "${DS_SUB_DIR_ROOT}" "${SUB_DIR_ROOT}"
                    fi
                    log "[R40] [${VID}] DIRECT DIAG OK -> ${DIRECT_ARCHIVE}"
                    DIAGED_OK_VIDS+=("${VID}")
                else
                    log "[R40] [${VID}] DIRECT DIAG FAILED (rc=${rc})"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                fi
            fi
        fi
    else
        log "[R40] --skip-diag: skipping direct diag for ${VID}"
    fi

    # ─── Phase 3: PLAN DIAG ────────────────────────────────────────
    PLAN_OUT="analyses/round40_stage1_plan_diag_${VID}"
    if [[ ${SKIP_DIAG} -eq 0 && -d "${SUB_DIR_ROOT}" ]]; then
        FIRST_BUCKET="${BUCKETS_STR%% *}"
        SEL_JSON="${SELECTION_VAL}"
        case "${FIRST_BUCKET}" in
            train) SEL_JSON="${SELECTION_TRAIN}" ;;
            val)   SEL_JSON="${SELECTION_VAL}"   ;;
        esac
        log
        log "================================================================"
        log "[$(date '+%F %T')] PLAN DIAG ${VID}  (bucket: ${FIRST_BUCKET})"
        log "================================================================"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[R40 DRY-RUN] plan diag ${VID}"
            log "    \$ ${PY} -u scripts/stage_a_generator/round40_stage1_plan_diag.py \\"
            log "          --config ${CFG} \\"
            log "          --pred-dir ${SUB_DIR_ROOT} \\"
            log "          --selection-json ${SEL_JSON} \\"
            log "          --bucket ${FIRST_BUCKET} \\"
            log "          --out-dir ${PLAN_OUT}"
            PLAN_OK_VIDS+=("${VID}")
        else
            set +e
            "${PY}" -u scripts/stage_a_generator/round40_stage1_plan_diag.py \
                --config "${CFG}" \
                --pred-dir "${SUB_DIR_ROOT}" \
                --selection-json "${SEL_JSON}" \
                --bucket "${FIRST_BUCKET}" \
                --out-dir "${PLAN_OUT}" \
                2>&1 | tee -a "${VARIANT_LOG}"
            rc=${PIPESTATUS[0]}
            set -e
            if [[ ${rc} -eq 0 ]]; then
                log "[R40] [${VID}] PLAN DIAG OK -> ${PLAN_OUT}"
                PLAN_OK_VIDS+=("${VID}")
            else
                log "[R40] [${VID}] PLAN DIAG FAILED (rc=${rc})"
            fi
        fi
    fi

    # ─── Phase 4: FULL CASCADE (optional) ──────────────────────────
    CASCADE_ARCHIVE="analyses/round40_fullcascade_diag_${VID}"
    if [[ ${SKIP_DIAG} -eq 0 && ${SKIP_FULL_CASCADE} -eq 0 && -d "${SUB_DIR_ROOT}" ]]; then
        FIRST_BUCKET="${BUCKETS_STR%% *}"
        log
        log "================================================================"
        log "[$(date '+%F %T')] FULL CASCADE ${VID}  (Stage-1 → R38-B1 → PB1)"
        log "================================================================"
        CASC_DIAG_DIR_ROOT="analyses/round32_stage1p5_downstream_diag_r40fc_${VID}"
        CASC_SUB_DIR_ROOT="analyses/round32_stage1p5_substitute_conds_r40fc_${VID}"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[R40 DRY-RUN] full cascade ${VID}: ROUND32_DS_UPSTREAM_DIR=${SUB_DIR_ROOT}"
            CASCADED_OK_VIDS+=("${VID}")
        else
            rm -rf "${CASC_SUB_DIR_ROOT}"
            rm -rf "${CASC_DIAG_DIR_ROOT}"
            set +e
            ROUND32_DS_STAGE1P5_CFG="${STAGE1P5_B1_CFG}" \
            ROUND32_DS_STAGE1P5_CKPT="${STAGE1P5_B1_CKPT}" \
            ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
            ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
            ROUND32_DS_OUT_TAG="_r40fc_${VID}" \
            ROUND32_DS_UPSTREAM_DIR="${SUB_DIR_ROOT}" \
                bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
                    2>&1 | tee -a "${VARIANT_LOG}"
            rc=${PIPESTATUS[0]}
            set -e
            if [[ ${rc} -eq 0 && -d "${CASC_DIAG_DIR_ROOT}" ]]; then
                rm -rf "${CASCADE_ARCHIVE}"
                mv "${CASC_DIAG_DIR_ROOT}" "${CASCADE_ARCHIVE}"
                log "[R40] [${VID}] FULL CASCADE OK -> ${CASCADE_ARCHIVE}"
                CASCADED_OK_VIDS+=("${VID}")
            else
                log "[R40] [${VID}] FULL CASCADE FAILED (rc=${rc})"
            fi
        fi
    fi

    # ─── Phase 5: K-SAMPLE DIVERSITY (subset of VIDs) ──────────────
    if [[ ${SKIP_KDIV} -eq 0 ]]; then
        case ",${KDIV_VIDS_STR}," in
            *",${VID},"*)
                FINAL="${OUTDIR}/final.pt"
                KDIV_OUT="analyses/round40_stage1_kdiv_${VID}"
                FIRST_BUCKET="${BUCKETS_STR%% *}"
                SEL_JSON="${SELECTION_VAL}"
                case "${FIRST_BUCKET}" in
                    train) SEL_JSON="${SELECTION_TRAIN}" ;;
                    val)   SEL_JSON="${SELECTION_VAL}"   ;;
                esac
                log
                log "================================================================"
                log "[$(date '+%F %T')] KDIV ${VID}  (K=${K_SAMPLES})"
                log "================================================================"
                if [[ ${DRY_RUN} -eq 1 ]]; then
                    log "[R40 DRY-RUN] kdiv ${VID}"
                    log "    \$ ${PY} -u scripts/stage_a_generator/round40_stage1_k_sample_audit.py \\"
                    log "          --config ${CFG} --ckpt ${FINAL} \\"
                    log "          --selection-json ${SEL_JSON} --bucket ${FIRST_BUCKET} \\"
                    log "          --out-dir ${KDIV_OUT} --num-samples ${K_SAMPLES}"
                    KDIV_OK_VIDS+=("${VID}")
                else
                    if [[ ! -f "${FINAL}" ]]; then
                        log "[R40] [${VID}] KDIV skipped (no ckpt)"
                    else
                        set +e
                        "${PY}" -u scripts/stage_a_generator/round40_stage1_k_sample_audit.py \
                            --config "${CFG}" \
                            --ckpt "${FINAL}" \
                            --selection-json "${SEL_JSON}" \
                            --bucket "${FIRST_BUCKET}" \
                            --out-dir "${KDIV_OUT}" \
                            --num-samples "${K_SAMPLES}" \
                            2>&1 | tee -a "${VARIANT_LOG}"
                        rc=${PIPESTATUS[0]}
                        set -e
                        if [[ ${rc} -eq 0 ]]; then
                            log "[R40] [${VID}] KDIV OK -> ${KDIV_OUT}"
                            KDIV_OK_VIDS+=("${VID}")
                        else
                            log "[R40] [${VID}] KDIV FAILED (rc=${rc})"
                        fi
                    fi
                fi
                ;;
            *)
                log "[R40] [${VID}] not in KDIV_VIDS; skipping K-sample audit"
                ;;
        esac
    fi
done <<< "${VARIANTS}"

# ─── Phase 6: Build comparison summary ─────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] BUILDING COMPARISON SUMMARY"
log "================================================================"

SUMMARY_MD="analyses/round40_stage1_plan_matrix_summary_${STAMP}.md"
if [[ ${DRY_RUN} -eq 0 && ${#DIAGED_OK_VIDS[@]} -gt 0 ]]; then
    SUMMARY_PY="${OVERALL_LOG_DIR}/build_summary_${STAMP}.py"
    cat > "${SUMMARY_PY}" <<'PYEOF'
import json
import re
import sys
from pathlib import Path

stamp = sys.argv[1]
out_md = Path(sys.argv[2])
variants = sys.argv[3:]


def read_metric(md_path, regex, group=1):
    if not Path(md_path).exists():
        return None
    m = re.search(regex, Path(md_path).read_text(encoding="utf-8"))
    return m.group(group) if m else None


def load_plan(vid):
    p = Path(f"analyses/round40_stage1_plan_diag_{vid}/plan_stats.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_kdiv(vid):
    p = Path(f"analyses/round40_stage1_kdiv_{vid}/k_sample_stats.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt(v, prec="0.3f"):
    if v is None or v == "":
        return "?"
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return v
    try:
        return f"{v:{prec}}"
    except Exception:
        return str(v)


rows = []
for v in variants:
    direct = Path(f"analyses/round40_stage1_direct_diag_{v}")
    sc = direct / "sustained_contact_val" / "sustained_contact_summary.md"
    cascade = Path(f"analyses/round40_fullcascade_diag_{v}")
    sc_casc = cascade / "sustained_contact_val" / "sustained_contact_summary.md"

    direct_drift = read_metric(sc, r"drift_max_cm:\s+mean=([\d.]+)")
    cascade_drift = read_metric(sc_casc, r"drift_max_cm:\s+mean=([\d.]+)")

    plan = load_plan(v)
    groups = plan.get("groups", {})
    pl = plan.get("plan_summary", {})

    vel_ratio_velxz = groups.get("velocity_xzy", {}).get("vel_rms_ratio")
    std_ratio_pel = groups.get("pelvis_rot6d", {}).get("std_ratio")
    vel_ratio_pel = groups.get("pelvis_rot6d", {}).get("vel_rms_ratio")
    root_arc_ratio = pl.get("root_arc_ratio")
    yaw_range_ratio = pl.get("yaw_range_ratio")

    kdiv = load_kdiv(v)
    kpair = kdiv.get("pairwise_summary", {})
    kpath = kpair.get("pair_root_path_rms_mean_mean")

    rows.append({
        "vid": v,
        "direct": direct_drift,
        "cascade": cascade_drift,
        "vel_ratio_velxz": vel_ratio_velxz,
        "std_ratio_pelvis": std_ratio_pel,
        "vel_ratio_pelvis": vel_ratio_pel,
        "root_arc_ratio": root_arc_ratio,
        "yaw_range_ratio": yaw_range_ratio,
        "k_root_div": kpath,
    })

out = [
    "# R40 Stage-1 plan-sampler ablation — summary (val)",
    "",
    f"Stamp: {stamp}",
    "",
    "Reference baselines:",
    "",
    "| metric | PB1 oracle | R38-B1 oracle | R38-B1 generated |",
    "|---|---:|---:|---:|",
    "| drift_max mean (cm) | 7.55 | 11.89 | 36.30 |",
    "",
    "Design notes:",
    "- C0 = V8 V6 baseline (sanity; expect ≈ generated-mode current behavior)",
    "- C1 = channel-weighted MSE only (no plan loss)",
    "- C2 = C1 + plan-invariant loss at 0.20 (ship candidate)",
    "- C3 = stronger channel weights + plan loss at 0.50 (probe)",
    "",
    "## Per-variant numbers",
    "",
    "| variant | direct drift (cm) | full cascade drift (cm) | velxz vel_ratio | pelvis std_ratio | pelvis vel_ratio | root_arc_ratio | yaw_range_ratio | k root path RMS |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    out.append(
        "| {vid} | {direct} | {cascade} | {vxz} | {pst} | {pvr} | {ra} | {yr} | {kp} |".format(
            vid=r["vid"],
            direct=fmt(r["direct"], "0.2f"),
            cascade=fmt(r["cascade"], "0.2f"),
            vxz=fmt(r["vel_ratio_velxz"]),
            pst=fmt(r["std_ratio_pelvis"]),
            pvr=fmt(r["vel_ratio_pelvis"]),
            ra=fmt(r["root_arc_ratio"]),
            yr=fmt(r["yaw_range_ratio"]),
            kp=fmt(r["k_root_div"], "0.4f"),
        )
    )

out.extend([
    "",
    "## Decision tree (R40 4-cell)",
    "",
    "Sanity:",
    "- C0 must reproduce current V8/V6 generated behavior (direct drift +",
    "  audit metrics within noise of latest R35 audit). Else launcher/config bug.",
    "",
    "Ship signal (vs C0 baseline):",
    "- C2 direct drift improves AND pelvis vel_ratio rises ≥ 0.60 → ship C2.",
    "- C2 improves audit metrics but direct drift not better → PB1 sensitive",
    "  to the new cond distribution; check full-cascade drift instead.",
    "- All cells flat → loss design insufficient. Pivot per Stage-1",
    "  underdetermination doc §9 to: CFG scale sweep, object_class cond,",
    "  or representation extension.",
    "",
    "K-diversity:",
    "- If pair_root_path_rms_mean < ~0.05 across all variants, diffusion noise",
    "  is not used as a mode-selector. Add a mode token / best-of-K next.",
])

out_md.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {out_md}")
PYEOF

    "${PY}" -u "${SUMMARY_PY}" "${STAMP}" "${SUMMARY_MD}" "${DIAGED_OK_VIDS[@]}" \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R40] WARN: summary build failed (non-fatal)."
fi

# ─── Phase 7: Pack ─────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round40_stage1_plan_results_${STAMP}.tar.gz"
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    ROUND40_STAMP="${STAMP}" \
    ROUND40_TARBALL="${TARBALL}" \
    ROUND40_SUMMARY_MD="${SUMMARY_MD}" \
    ROUND40_SUMMARY_LOG="${SUMMARY_LOG}" \
    ROUND40_VARIANT_LOG_DIR="${OVERALL_LOG_DIR}" \
    ROUND40_TRAINED_VIDS="${TRAINED_OK_VIDS[*]:-}" \
    ROUND40_DIAGED_VIDS="${DIAGED_OK_VIDS[*]:-}" \
    ROUND40_PLAN_VIDS="${PLAN_OK_VIDS[*]:-}" \
    ROUND40_CASCADED_VIDS="${CASCADED_OK_VIDS[*]:-}" \
    ROUND40_KDIV_VIDS_LIST="${KDIV_OK_VIDS[*]:-}" \
        bash scripts/stage_a_generator/pack_round40_stage1_plan_sync.sh \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R40 PACK] packer failed (non-fatal)"
fi

log
log "================================================================"
log "[$(date '+%F %T')] R40 matrix COMPLETE"
log "================================================================"
log "Trained:   ${TRAINED_OK_VIDS[*]:-none}"
log "Diaged:    ${DIAGED_OK_VIDS[*]:-none}"
log "Plan diag: ${PLAN_OK_VIDS[*]:-none}"
log "Cascaded:  ${CASCADED_OK_VIDS[*]:-none}"
log "KDIV:      ${KDIV_OK_VIDS[*]:-none}"
log
log "Summary log: ${SUMMARY_LOG}"
log "Summary MD:  ${SUMMARY_MD}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    log "Tarball:     ${TARBALL:-<none>}"
fi
