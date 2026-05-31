#!/usr/bin/env bash
# Round-37 Stage-1.5 C41 dynamics-loss ablation matrix (Stage-2 philosophy).
#
# R36 was a single-cell failure: statistical normalize_by_gt_std blew up
# acceleration loss by ~180× train-vs-val (raw 555 vs 3.06) and the
# weighted r36_c41_acceleration term dominated total loss at 27× mse_c41.
# R37 re-derives the loss in Stage-2's cm-physical-unit + SmoothL1 +
# per-bodypart-mask space, and uses 4 cells to isolate the 3 design
# choices that are different from Stage-2 / paper consensus.
#
# Four variants — all share R34 V2-A substrate (per-block obj_xattn,
# r34_wrist_lowband λ=0.005, σ_cond_aug=0):
#
#   A0 full        — wrist/knee/pelvis/neck vel + pelvis acc + jerk +
#                    speed-moment, all on per-bodypart contact/stance
#                    masks. R34 FFT loss kept at λ=0.005.
#   A1 no_fft      — A0 minus R34 wrist_lowband (test FFT redundancy)
#   A2 no_jerk     — A0 minus jerk smoothness (test paper-inform value)
#   A3 no_mask     — A0 with per-group mask disabled (use full seq_mask)
#
# Each variant: TRAIN (from scratch, 80 ep, bs=48, seed=42) → R32
# downstream diag against frozen PB1 (oracle Stage-1 cond) → DIAG mode
# generated_stage1_cond against R31 V8 V6 cache → C41/S4 quality metrics.
#
# Estimated time: 4 × ~30 min train + 4 × ~10 min diag × 2 modes +
# 4 × ~30 s quality = ~3.5 h on 2× 5080.
#
# Usage:
#   tmux new -s r37
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round37_matrix.sh
#
#   # Subset:
#   bash scripts/stage_a_generator/run_round37_matrix.sh \
#       --only stage1p5_r37_a0_full,stage1p5_r37_a1_no_fft
#
#   # Dry-run:
#   bash scripts/stage_a_generator/run_round37_matrix.sh --dry-run
#
# Environment overrides (all prefixed ROUND37_*):
#   ROUND37_GPUS="0,2"                CUDA_VISIBLE_DEVICES mask
#   ROUND37_NUM_PROCESSES=N            accelerate --num_processes
#   ROUND37_BUCKETS="val"              diag buckets
#   ROUND37_ALLOW_PARTIAL=1            keep going on a per-variant failure
#   ROUND37_BASE_CFG=…                 base cfg for R37 cfg generator
#                                        (default = R34 V2-A cfg)
#   ROUND37_GENERATED_UPSTREAM_DIR=…   R31 V8 V6 generated cond cache
#   ROUND37_SKIP_GENERATED_EVAL=1      skip the 2nd eval pass
#   ROUND37_GT_DUMP_DIR=…              GT C41/S4 dump for quality metrics
#   ROUND37_SKIP_QUALITY_METRICS=1     skip quality metrics

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
FORCE_RETRAIN=0
FORCE_REDIAG=0
ALLOW_PARTIAL="${ROUND37_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND37_BUCKETS:-val}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)            ONLY="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --skip-train)      SKIP_TRAIN=1; shift ;;
        --skip-diag)       SKIP_DIAG=1; shift ;;
        --force-retrain)   FORCE_RETRAIN=1; shift ;;
        --force-rediag)    FORCE_REDIAG=1; shift ;;
        --buckets)         BUCKETS_STR="$2"; shift 2 ;;
        -h|--help)         sed -n '1,55p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[R37] FATAL: export DATASETS_ROOT before launch." >&2
    echo "    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

GPUS="${ROUND37_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

NUM_GPUS_IN_MASK="$(echo "${GPUS}" | tr ',' '\n' | grep -c '^[0-9]\+$' || true)"
if [[ "${NUM_GPUS_IN_MASK}" -lt 1 ]]; then NUM_GPUS_IN_MASK=1; fi

: "${ROUND37_NUM_PROCESSES:=${NUM_GPUS_IN_MASK}}"
export ROUND37_NUM_PROCESSES

ROUND37_BASE_CFG="${ROUND37_BASE_CFG:-configs/training/stage1p5_r34v2_a_lambda0p005.yaml}"

PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CKPT="runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"

# Generated Stage-1 cond cache (R31 V8 V6).
GENERATED_UPSTREAM_DIR="${ROUND37_GENERATED_UPSTREAM_DIR:-analyses/round31_stage1_substitute_conds_v8_stage1_v8_v6_full_f1}"
SKIP_GENERATED_EVAL="${ROUND37_SKIP_GENERATED_EVAL:-0}"

# C41/S4 quality metrics.
GT_DUMP_DIR="${ROUND37_GT_DUMP_DIR:-analyses/2026-05-31_stage1p5_wrist_external_review_work/oracle_dump}"
SKIP_QUALITY_METRICS="${ROUND37_SKIP_QUALITY_METRICS:-0}"

OVERALL_LOG_DIR="runs/round37_matrix"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R37] FATAL: no python found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ─── Ensure R34 V2-A base config exists (R37 substrate) ──────────────
if [[ ! -f "${ROUND37_BASE_CFG}" && ${DRY_RUN} -eq 0 ]]; then
    log "[R37] Base config ${ROUND37_BASE_CFG} missing; running round34v2 cfg generator."
    "${PY}" scripts/stage_a_generator/round34v2_make_stage1p5_configs.py \
        --base-cfg configs/training/stage1p5_r33_v1_xattn.yaml \
        --out-dir configs/training/ 2>&1 | tee -a "${SUMMARY_LOG}"
fi

# ─── Regenerate R37 configs (idempotent) ─────────────────────────────
log "[R37] Regenerating Stage-1.5 R37 ablation configs from ${ROUND37_BASE_CFG}."
if [[ ${DRY_RUN} -eq 0 ]]; then
    "${PY}" scripts/stage_a_generator/round37_make_stage1p5_configs.py \
        --base-cfg "${ROUND37_BASE_CFG}" \
        --out-dir configs/training/ 2>&1 | tee -a "${SUMMARY_LOG}"
fi

# ─── Hard-coded variant list (cfg generator emits exactly these 4) ───
# Matches scripts/stage_a_generator/round37_make_stage1p5_configs.py:VARIANTS.
VARIANT_TABLE=(
    "stage1p5_r37_a0_full     configs/training/stage1p5_r37_a0_full.yaml     runs/training/stage1p5_r37_a0_full"
    "stage1p5_r37_a1_no_fft   configs/training/stage1p5_r37_a1_no_fft.yaml   runs/training/stage1p5_r37_a1_no_fft"
    "stage1p5_r37_a2_no_jerk  configs/training/stage1p5_r37_a2_no_jerk.yaml  runs/training/stage1p5_r37_a2_no_jerk"
    "stage1p5_r37_a3_no_mask  configs/training/stage1p5_r37_a3_no_mask.yaml  runs/training/stage1p5_r37_a3_no_mask"
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
    log "[R37] no train variants matched only='${ONLY}'"
    exit 0
fi

log
log "===== R37 matrix launch ${STAMP} ====="
log "*** CALIBRATION REMINDER ***"
log "After the first few train epochs, audit each VARIANT's metrics.jsonl"
log "for r37_<term>_weighted / mse_c41 ratios. Target: each weighted term"
log "≤ 1.0 × mse_c41. If any single R37 term reaches ≥ 5× mse_c41 (R36"
log "disaster mode), kill the run, lower that term's weight 10×, restart."
log
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS=${GPUS}  NUM_PROCESSES=${ROUND37_NUM_PROCESSES}"
log "SKIP_TRAIN=${SKIP_TRAIN}  SKIP_DIAG=${SKIP_DIAG}  BUCKETS=${BUCKETS_STR}"
log "PB1_CKPT=${PB1_CKPT}"
log "GENERATED_UPSTREAM_DIR=${GENERATED_UPSTREAM_DIR} (SKIP=${SKIP_GENERATED_EVAL})"
log "GT_DUMP_DIR=${GT_DUMP_DIR} (SKIP_QM=${SKIP_QUALITY_METRICS})"
log "Variants to process:"
echo "${VARIANTS}" | sed 's/^/  /' | tee -a "${SUMMARY_LOG}"
log

# ─── Preflight ───────────────────────────────────────────────────────
preflight_fail=0
if [[ ${SKIP_DIAG} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
    if [[ ! -f "${PB1_CKPT}" ]]; then
        log "[R37 PREFLIGHT FAIL] PB1 ckpt missing: ${PB1_CKPT}"
        preflight_fail=1
    fi
    case "${BUCKETS_STR}" in
        *val*)
            [[ ! -f "${SELECTION_VAL}" ]] && {
                log "[R37 PREFLIGHT FAIL] val selection JSON missing: ${SELECTION_VAL}"
                preflight_fail=1
            } ;;
    esac
    case "${BUCKETS_STR}" in
        *train*)
            [[ ! -f "${SELECTION_TRAIN}" ]] && {
                log "[R37 PREFLIGHT FAIL] train selection JSON missing: ${SELECTION_TRAIN}"
                preflight_fail=1
            } ;;
    esac
fi
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -f "${CFG}" && ${DRY_RUN} -eq 0 ]]; then
        log "[R37 PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R37] FATAL preflight failures."
    exit 1
fi

# ─── Per-variant train -> diag -> archive loop ──────────────────────
TRAINED_OK_VIDS=()
DIAGED_OK_VIDS=()

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    VARIANT_LOG="${OVERALL_LOG_DIR}/${VID}.log"
    DIAG_ARCHIVE="analyses/round37_diag_${VID}"

    # ─── Phase 1: TRAIN ─────────────────────────────────────────────
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ -f "${FINAL}" && ${FORCE_RETRAIN} -eq 0 ]]; then
            log "[R37] [${VID}] ckpt already exists; skipping train (use --force-retrain)"
            TRAINED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] TRAIN ${VID}"
            log "    config: ${CFG}"
            log "    output: ${OUTDIR}"
            log "    log:    ${VARIANT_LOG}"
            log "    GPUs:   ${GPUS}  procs=${ROUND37_NUM_PROCESSES}"
            log "================================================================"

            if [[ "${ROUND37_NUM_PROCESSES}" -le 1 ]]; then
                TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1p5.py --config "${CFG}")
            else
                TRAIN_CMD=(accelerate launch
                    --num_processes "${ROUND37_NUM_PROCESSES}"
                    --multi_gpu --mixed_precision bf16
                    src/piano/training/train_stage1p5.py --config "${CFG}")
            fi

            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R37 DRY-RUN] would train ${VID}"
                log "    \$ CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD[*]}"
                TRAINED_OK_VIDS+=("${VID}")
            else
                set +e
                "${TRAIN_CMD[@]}" 2>&1 | tee "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -f "${FINAL}" ]]; then
                    TRAINED_OK_VIDS+=("${VID}")
                    log "[R37] [${VID}] TRAIN OK -> ${FINAL}"
                else
                    log "[R37] [${VID}] TRAIN FAILED (rc=${rc}, final.pt=$([[ -f ${FINAL} ]] && echo present || echo missing))"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                    continue
                fi
            fi
        fi
    else
        log "[R37] --skip-train: skipping train for ${VID}"
        TRAINED_OK_VIDS+=("${VID}")
    fi

    # ─── Phase 2: DIAG ──────────────────────────────────────────────
    if [[ ${SKIP_DIAG} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ ! -f "${FINAL}" && ${DRY_RUN} -eq 0 ]]; then
            log "[R37] [${VID}] no ckpt to diag; skipping diag"
            continue
        fi

        DIAG_DONE_MARKER="${DIAG_ARCHIVE}/sustained_contact_val/sustained_contact_summary.md"
        if [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
            log "[R37] [${VID}] diag already archived at ${DIAG_ARCHIVE}; skipping (use --force-rediag)"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        log
        log "================================================================"
        log "[$(date '+%F %T')] DIAG ${VID}  (buckets: ${BUCKETS_STR})"
        log "================================================================"

        if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[R37 DRY-RUN] would diag ${VID} against PB1 at ${PB1_CKPT}"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        DS_OUT_TAG="_r37_${VID}"
        SUB_DIR_ROOT="analyses/round32_stage1p5_substitute_conds${DS_OUT_TAG}"
        DIAG_DIR_ROOT="analyses/round32_stage1p5_downstream_diag${DS_OUT_TAG}"

        rm -rf "${SUB_DIR_ROOT}"
        rm -rf "${DIAG_DIR_ROOT}"

        set +e
        ROUND32_DS_STAGE1P5_CFG="${CFG}" \
        ROUND32_DS_STAGE1P5_CKPT="${FINAL}" \
        ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
        ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
        ROUND32_DS_OUT_TAG="${DS_OUT_TAG}" \
            bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
                2>&1 | tee -a "${VARIANT_LOG}"
        rc=${PIPESTATUS[0]}
        set -e

        if [[ ${rc} -eq 0 && -d "${DIAG_DIR_ROOT}" ]]; then
            mkdir -p "$(dirname "${DIAG_ARCHIVE}")"
            rm -rf "${DIAG_ARCHIVE}"
            mv "${DIAG_DIR_ROOT}" "${DIAG_ARCHIVE}"
            log "[R37] [${VID}] DIAG OK (oracle Stage-1) -> ${DIAG_ARCHIVE}"
            DIAGED_OK_VIDS+=("${VID}")

            # ── Second eval mode: generated Stage-1 cond ──
            if [[ ${SKIP_GENERATED_EVAL} -eq 0 && -d "${GENERATED_UPSTREAM_DIR}" ]]; then
                DS_OUT_TAG_GEN="_r37_${VID}_genstage1"
                SUB_DIR_ROOT_GEN="analyses/round32_stage1p5_substitute_conds${DS_OUT_TAG_GEN}"
                DIAG_DIR_ROOT_GEN="analyses/round32_stage1p5_downstream_diag${DS_OUT_TAG_GEN}"
                DIAG_ARCHIVE_GEN="analyses/round37_diag_${VID}_genstage1"
                rm -rf "${SUB_DIR_ROOT_GEN}"
                rm -rf "${DIAG_DIR_ROOT_GEN}"

                log
                log "[R37] [${VID}] DIAG mode=generated_stage1_cond (UPSTREAM_DIR=${GENERATED_UPSTREAM_DIR})"
                set +e
                ROUND32_DS_STAGE1P5_CFG="${CFG}" \
                ROUND32_DS_STAGE1P5_CKPT="${FINAL}" \
                ROUND32_DS_PB1_CKPT="${PB1_CKPT}" \
                ROUND32_DS_BUCKETS="${BUCKETS_STR}" \
                ROUND32_DS_OUT_TAG="${DS_OUT_TAG_GEN}" \
                ROUND32_DS_UPSTREAM_DIR="${GENERATED_UPSTREAM_DIR}" \
                    bash scripts/stage_a_generator/run_round32_stage1p5_downstream_diag.sh \
                        2>&1 | tee -a "${VARIANT_LOG}"
                rc_gen=${PIPESTATUS[0]}
                set -e
                if [[ ${rc_gen} -eq 0 && -d "${DIAG_DIR_ROOT_GEN}" ]]; then
                    rm -rf "${DIAG_ARCHIVE_GEN}"
                    mv "${DIAG_DIR_ROOT_GEN}" "${DIAG_ARCHIVE_GEN}"
                    log "[R37] [${VID}] DIAG OK (generated Stage-1) -> ${DIAG_ARCHIVE_GEN}"
                else
                    log "[R37] [${VID}] DIAG FAILED (generated Stage-1, rc=${rc_gen}); continuing"
                fi
            elif [[ ${SKIP_GENERATED_EVAL} -ne 0 ]]; then
                log "[R37] [${VID}] generated-Stage-1 eval skipped (ROUND37_SKIP_GENERATED_EVAL=1)"
            else
                log "[R37] [${VID}] generated-Stage-1 eval skipped (UPSTREAM_DIR ${GENERATED_UPSTREAM_DIR} missing)"
            fi

            # ── C41/S4 quality metrics ──
            if [[ ${SKIP_QUALITY_METRICS} -eq 0 && -d "${GT_DUMP_DIR}/${BUCKETS_STR%% *}" ]]; then
                QUALITY_OUT="analyses/round37_quality/c41_s4_quality_${VID}.md"
                mkdir -p "$(dirname "${QUALITY_OUT}")"
                log "[R37] [${VID}] C41/S4 quality metrics → ${QUALITY_OUT}"
                set +e
                "${PY}" -u scripts/stage_a_generator/round34_c41_s4_quality_metrics.py \
                    --gt-dir "${GT_DUMP_DIR}" \
                    --pred-dir "analyses/round32_stage1p5_substitute_conds${DS_OUT_TAG}" \
                    --bucket "${BUCKETS_STR%% *}" \
                    --variant-label "${VID} (oracle Stage-1)" \
                    --out "${QUALITY_OUT}" \
                    2>&1 | tee -a "${VARIANT_LOG}"
                set -e
            fi
        else
            log "[R37] [${VID}] DIAG FAILED (rc=${rc}, dir=$([[ -d ${DIAG_DIR_ROOT} ]] && echo present || echo missing))"
            if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
        fi
    else
        log "[R37] --skip-diag: skipping diag for ${VID}"
    fi
done <<< "${VARIANTS}"

# ─── Phase 3: Comparison summary ────────────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] BUILDING COMPARISON SUMMARY"
log "================================================================"

SUMMARY_MD="analyses/round37_matrix_summary_${STAMP}.md"
if [[ ${DRY_RUN} -eq 0 && ${#DIAGED_OK_VIDS[@]} -gt 0 ]]; then
    SUMMARY_PY="${OVERALL_LOG_DIR}/build_summary_${STAMP}.py"
    cat > "${SUMMARY_PY}" <<'PYEOF'
import re
import sys
from pathlib import Path

stamp = sys.argv[1]
out_md = Path(sys.argv[2])
variants = sys.argv[3:]


def read_metric(md_path, regex, group=1):
    if not Path(md_path).exists():
        return None
    txt = Path(md_path).read_text(encoding="utf-8")
    m = re.search(regex, txt)
    return m.group(group) if m else None


def read_per_part_drift(sc_md_path, part_name):
    if not Path(sc_md_path).exists():
        return None
    txt = Path(sc_md_path).read_text(encoding="utf-8")
    m = re.search(
        rf"\|\s*{re.escape(part_name)}\s*\|\s*\d+\s*\|\s*([\d.]+)",
        txt,
    )
    return m.group(1) if m else None


rows = []
for v in variants:
    base = Path(f"analyses/round37_diag_{v}")
    sc = base / "sustained_contact_val" / "sustained_contact_summary.md"
    gait = base / "gait_val" / "gait_summary.md"
    body = base / "body_action_val" / "body_action_summary.md"
    g1 = base / "g1_soft_stance_val" / "g1_soft_stance_summary.md"

    drift = read_metric(sc, r"drift_max_cm:\s+mean=([\d.]+)")
    lr = read_metric(gait, r"mean L_R_height_corr\s*\|\s*[-\d.]+\s*\|\s*([-\d.]+)")
    sp = read_metric(gait, r"segments with detected period\s*\|\s*[\d.]+%\s*\|\s*([\d.]+)%")
    lw = read_metric(body, r"left_wrist\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*([\d.]+)")
    rw = read_metric(body, r"right_wrist\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*([\d.]+)")
    lk = read_metric(body, r"left_knee\s*\|\s*([\d.]+)")
    alt = read_metric(g1, r"low_alt_amplitude_rate \(seg\)\s*\|\s*([\d.]+)")
    lh_drift = read_per_part_drift(str(sc), "left_hand")
    rh_drift = read_per_part_drift(str(sc), "right_hand")

    base_gen = Path(f"analyses/round37_diag_{v}_genstage1")
    sc_gen = base_gen / "sustained_contact_val" / "sustained_contact_summary.md"
    drift_gen = read_metric(sc_gen, r"drift_max_cm:\s+mean=([\d.]+)")
    lh_drift_gen = read_per_part_drift(str(sc_gen), "left_hand")
    rh_drift_gen = read_per_part_drift(str(sc_gen), "right_hand")

    rows.append((v, drift, lh_drift, rh_drift, lr, sp, lw, rw, lk, alt, drift_gen, lh_drift_gen, rh_drift_gen))

out = [
    "# R37 Stage-1.5 C41 dynamics-loss ablation — summary (val)",
    "",
    f"Stamp: {stamp}",
    "",
    "Reference baselines:",
    "| metric | PB1 oracle (full GT cond) | R34 V2-A oracle | R34 V2-A generated |",
    "|---|---:|---:|---:|",
    "| drift_max mean (cm) | 7.55 | 13.63 | 38.33 |",
    "| left_hand drift_max (cm) | (~10 cm) | 20.54 | 50.59 |",
    "| right_hand drift_max (cm) | (~10 cm) | 25.05 | 53.18 |",
    "",
    "Design notes:",
    "- A0 = full R37 (wrist/knee/pelvis/neck vel + pelvis acc + jerk + speed-moment, per-bodypart mask, R34 FFT kept)",
    "- A1 = A0 minus R34 wrist_lowband (test FFT redundancy)",
    "- A2 = A0 minus c41 jerk (test paper-inform value)",
    "- A3 = A0 minus per-group mask (test mask philosophy)",
    "",
    "## Per-variant numbers — oracle Stage-1 cond",
    "",
    "| variant | drift_max | lh_drift | rh_drift | L_R_corr | step_period | lw dir_cos | rw dir_cos | lk delta_err | low_alt_amp |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    v, drift, lhd, rhd, lr, sp, lw, rw, lk, alt, *_ = row
    out.append(
        f"| {v} | {drift or '?'} | {lhd or '?'} | {rhd or '?'} | "
        f"{lr or '?'} | {sp or '?'} | "
        f"{lw or '?'} | {rw or '?'} | {lk or '?'} | {alt or '?'} |"
    )

out.append("")
out.append("## Per-variant numbers — generated Stage-1 cond")
out.append("")
out.append("| variant | drift_max | lh_drift | rh_drift |")
out.append("|---|---:|---:|---:|")
for row in rows:
    v, *_oracle, drift_gen, lhd_gen, rhd_gen = row
    out.append(
        f"| {v} | {drift_gen or '?'} | {lhd_gen or '?'} | {rhd_gen or '?'} |"
    )

out.append("")
out.append("## Decision tree (R37 4-cell)")
out.append("")
out.append("Sanity:")
out.append("- All 4 variants should produce a non-degenerate ckpt that DIAGs OK.")
out.append("- If any variant gives drift_max > 30 cm in oracle mode, scale-dominate bug suspected;")
out.append("  inspect train-log r37_*_weighted / mse_c41 ratios (target each term ≤ 1× mse_c41).")
out.append("")
out.append("Ship signal (vs R34 V2-A oracle 13.63 cm):")
out.append("- Any cell ≤ 10 cm → R37 closes PB1 oracle gap; ship that cell; pivot to Stage-1 R37 design.")
out.append("- Best cell in 10-13 cm → R37 direction supported but bounded; weight sweep / more mask groups next.")
out.append("- All cells > 13 cm → C41 dynamics not the bottleneck. Pivot: contact-window weighted MSE /")
out.append("  PB1 inference-time spatial guidance / PB1 input-pathway adapter (ChatGPT §10).")
out.append("")
out.append("Per-cell ablation conclusions:")
out.append("- A0 best → all R37 components needed; ship A0.")
out.append("- A1 ≥ A0 → R34 FFT redundant; drop r34_wrist_lowband in future configs.")
out.append("- A2 ≥ A0 → jerk smoothness no help; drop paper-inform jerk term.")
out.append("- A3 ≥ A0 → per-group mask not helpful; align with paper dense supervision.")

out_md.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {out_md}")
PYEOF

    "${PY}" -u "${SUMMARY_PY}" "${STAMP}" "${SUMMARY_MD}" "${DIAGED_OK_VIDS[@]}" \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R37] WARN: summary build failed (non-fatal)."
fi

# ─── Phase 4: Pack ─────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round37_matrix_results_${STAMP}.tar.gz"
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    # Delegate to the dedicated R37 sync packer so the sync-back archive
    # is small and reproducible. It excludes large .npz substitute caches
    # and model ckpts unless ROUND37_PACK_INCLUDE_CKPTS=1.
    ROUND37_STAMP="${STAMP}" \
    ROUND37_TARBALL="${TARBALL}" \
    ROUND37_SUMMARY_MD="${SUMMARY_MD}" \
    ROUND37_SUMMARY_LOG="${SUMMARY_LOG}" \
    ROUND37_VARIANT_LOG_DIR="${OVERALL_LOG_DIR}" \
    ROUND37_TRAINED_VIDS="${TRAINED_OK_VIDS[*]:-}" \
    ROUND37_DIAGED_VIDS="${DIAGED_OK_VIDS[*]:-}" \
        bash scripts/stage_a_generator/pack_round37_sync.sh \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R37 PACK] packer failed (non-fatal)"
fi

log
log "================================================================"
log "[$(date '+%F %T')] R37 matrix COMPLETE"
log "================================================================"
log "Trained: ${TRAINED_OK_VIDS[*]:-none}"
log "Diaged:  ${DIAGED_OK_VIDS[*]:-none}"
log
log "Summary log: ${SUMMARY_LOG}"
log "Summary MD:  ${SUMMARY_MD}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    log "Tarball:     ${TARBALL:-<none>}"
fi
