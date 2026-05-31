#!/usr/bin/env bash
# Round-34 Stage-1.5 wrist low-band + conditioning-augmentation 4-variant
# matrix — integrated train + downstream-coupling diag launcher.
#
# **DO NOT RUN until D1 + D2 diagnostics justify it** (brief §7.1):
#   - D1-V2 wrist LOW-band oracle closes drift_max substantially, OR
#   - D2-G3 wrist LOW-band residual causes steep PB1 degradation, OR
#   - D1-V1 full wrist oracle improves drift_max even if V2 only partially helps.
#
# Variants (brief §7.4) — all share R33 V1 substrate (per-block obj_xattn):
#   C0 control      σ=0,    λ_lowband=0
#   C1 cond_aug     σ=0.15, λ_lowband=0
#   C2 lowband      σ=0,    λ_lowband=1.0    cutoff=1.0 Hz
#   C3 combined     σ=0.15, λ_lowband=1.0    cutoff=1.0 Hz
#
# Per variant: TRAIN (from scratch, 80 ep, bs=48, seed=42) → R32 downstream
# diag against frozen PB1 with oracle Stage-1 cond (val bucket only by
# default) → archive diag under analyses/round34_diag_<variant>/.
#
# Estimated time: 4 × ~30 min train + 4 × ~10 min diag = ~3 h on 2× 5080.
#
# Usage:
#   tmux new -s r34
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round34_matrix.sh
#
#   # Subset (e.g., only C0 + C2):
#   bash scripts/stage_a_generator/run_round34_matrix.sh \
#       --only stage1p5_r34_c0_control,stage1p5_r34_c2_lowband
#
#   # Dry-run:
#   bash scripts/stage_a_generator/run_round34_matrix.sh --dry-run
#
# Environment overrides:
#   ROUND34_GPUS="0,2"             CUDA_VISIBLE_DEVICES mask
#   ROUND34_NUM_PROCESSES=N         accelerate --num_processes
#   ROUND34_BUCKETS="val"           diag buckets
#   ROUND34_ALLOW_PARTIAL=1         keep going on a per-variant failure
#   ROUND34_BASE_CFG=…              base cfg passed to round34_make_stage1p5_configs.py

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
FORCE_RETRAIN=0
FORCE_REDIAG=0
ALLOW_PARTIAL="${ROUND34_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND34_BUCKETS:-val}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)            ONLY="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --skip-train)      SKIP_TRAIN=1; shift ;;
        --skip-diag)       SKIP_DIAG=1; shift ;;
        --force-retrain)   FORCE_RETRAIN=1; shift ;;
        --force-rediag)    FORCE_REDIAG=1; shift ;;
        --buckets)         BUCKETS_STR="$2"; shift 2 ;;
        -h|--help)         sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[R34] FATAL: export DATASETS_ROOT before launch." >&2
    echo "    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

GPUS="${ROUND34_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

NUM_GPUS_IN_MASK="$(echo "${GPUS}" | tr ',' '\n' | grep -c '^[0-9]\+$' || true)"
if [[ "${NUM_GPUS_IN_MASK}" -lt 1 ]]; then NUM_GPUS_IN_MASK=1; fi

: "${ROUND34_NUM_PROCESSES:=${NUM_GPUS_IN_MASK}}"
export ROUND34_NUM_PROCESSES

ROUND34_BASE_CFG="${ROUND34_BASE_CFG:-configs/training/stage1p5_r33_v1_xattn.yaml}"

PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CKPT="runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"

# R34 — second eval mode: generated Stage-1 cond.
# Per ChatGPT followup §7.2, R34 variants must be evaluated under BOTH
# (a) oracle Stage-1 cond and (b) Stage-1 generated cond, to measure
# both Stage-1.5 fix and cascaded-robustness benefit.
# UPSTREAM_DIR points to the Stage-1 V8 V6 generated stage1_coarse cache,
# layout <subset>/<seq>.npz with key 'stage1_coarse' z-scored.
GENERATED_UPSTREAM_DIR="${ROUND34_GENERATED_UPSTREAM_DIR:-analyses/round31_stage1_substitute_conds_v8_stage1_v8_v6_full_f1}"
SKIP_GENERATED_EVAL="${ROUND34_SKIP_GENERATED_EVAL:-0}"

# C41/S4 quality metrics — optional post-diag analysis per ChatGPT §7.4.
GT_DUMP_DIR="${ROUND34_GT_DUMP_DIR:-analyses/2026-05-31_stage1p5_wrist_external_review_work/oracle_dump}"
SKIP_QUALITY_METRICS="${ROUND34_SKIP_QUALITY_METRICS:-0}"

OVERALL_LOG_DIR="runs/round34_matrix"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R34] FATAL: no python found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ─── Regenerate R34 configs (idempotent) ───────────────────────────
log "[R34] Regenerating Stage-1.5 R34 ablation configs from ${ROUND34_BASE_CFG}."
if [[ ${DRY_RUN} -eq 0 ]]; then
    "${PY}" scripts/stage_a_generator/round34_make_stage1p5_configs.py \
        --base-cfg "${ROUND34_BASE_CFG}" \
        --out-dir configs/training/ 2>&1 | tee -a "${SUMMARY_LOG}"
fi

# ─── Hard-coded variant list (the cfg generator emits exactly these 4) ──
VARIANT_TABLE=(
    "stage1p5_r34_c0_control configs/training/stage1p5_r34_c0_control.yaml runs/training/stage1p5_r34_c0_control"
    "stage1p5_r34_c1_cond_aug configs/training/stage1p5_r34_c1_cond_aug.yaml runs/training/stage1p5_r34_c1_cond_aug"
    "stage1p5_r34_c2_lowband configs/training/stage1p5_r34_c2_lowband.yaml runs/training/stage1p5_r34_c2_lowband"
    "stage1p5_r34_c3_combined configs/training/stage1p5_r34_c3_combined.yaml runs/training/stage1p5_r34_c3_combined"
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
    log "[R34] no train variants matched only='${ONLY}'"
    exit 0
fi

log
log "===== R34 matrix launch ${STAMP} ====="
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS=${GPUS}  NUM_PROCESSES=${ROUND34_NUM_PROCESSES}"
log "SKIP_TRAIN=${SKIP_TRAIN}  SKIP_DIAG=${SKIP_DIAG}  BUCKETS=${BUCKETS_STR}"
log "PB1_CKPT=${PB1_CKPT}"
log "Variants to process:"
echo "${VARIANTS}" | sed 's/^/  /' | tee -a "${SUMMARY_LOG}"
log

# ─── Preflight ──────────────────────────────────────────────────────
preflight_fail=0
if [[ ${SKIP_DIAG} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
    if [[ ! -f "${PB1_CKPT}" ]]; then
        log "[R34 PREFLIGHT FAIL] PB1 ckpt missing: ${PB1_CKPT}"
        preflight_fail=1
    fi
    case "${BUCKETS_STR}" in
        *val*)
            [[ ! -f "${SELECTION_VAL}" ]] && {
                log "[R34 PREFLIGHT FAIL] val selection JSON missing: ${SELECTION_VAL}"
                preflight_fail=1
            } ;;
    esac
    case "${BUCKETS_STR}" in
        *train*)
            [[ ! -f "${SELECTION_TRAIN}" ]] && {
                log "[R34 PREFLIGHT FAIL] train selection JSON missing: ${SELECTION_TRAIN}"
                preflight_fail=1
            } ;;
    esac
fi
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -f "${CFG}" && ${DRY_RUN} -eq 0 ]]; then
        log "[R34 PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R34] FATAL preflight failures."
    exit 1
fi

# ─── Per-variant train -> diag -> archive loop ─────────────────────
TRAINED_OK_VIDS=()
DIAGED_OK_VIDS=()

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    VARIANT_LOG="${OVERALL_LOG_DIR}/${VID}.log"
    DIAG_ARCHIVE="analyses/round34_diag_${VID}"

    # ─── Phase 1: TRAIN ─────────────────────────────────────────────
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ -f "${FINAL}" && ${FORCE_RETRAIN} -eq 0 ]]; then
            log "[R34] [${VID}] ckpt already exists; skipping train (use --force-retrain)"
            TRAINED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] TRAIN ${VID}"
            log "    config: ${CFG}"
            log "    output: ${OUTDIR}"
            log "    log:    ${VARIANT_LOG}"
            log "    GPUs:   ${GPUS}  procs=${ROUND34_NUM_PROCESSES}"
            log "================================================================"

            if [[ "${ROUND34_NUM_PROCESSES}" -le 1 ]]; then
                TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1p5.py --config "${CFG}")
            else
                TRAIN_CMD=(accelerate launch
                    --num_processes "${ROUND34_NUM_PROCESSES}"
                    --multi_gpu --mixed_precision bf16
                    src/piano/training/train_stage1p5.py --config "${CFG}")
            fi

            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R34 DRY-RUN] would train ${VID}"
                log "    \$ CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD[*]}"
                TRAINED_OK_VIDS+=("${VID}")
            else
                set +e
                "${TRAIN_CMD[@]}" 2>&1 | tee "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -f "${FINAL}" ]]; then
                    TRAINED_OK_VIDS+=("${VID}")
                    log "[R34] [${VID}] TRAIN OK -> ${FINAL}"
                else
                    log "[R34] [${VID}] TRAIN FAILED (rc=${rc}, final.pt=$([[ -f ${FINAL} ]] && echo present || echo missing))"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                    continue
                fi
            fi
        fi
    else
        log "[R34] --skip-train: skipping train for ${VID}"
        TRAINED_OK_VIDS+=("${VID}")
    fi

    # ─── Phase 2: DIAG ──────────────────────────────────────────────
    if [[ ${SKIP_DIAG} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ ! -f "${FINAL}" && ${DRY_RUN} -eq 0 ]]; then
            log "[R34] [${VID}] no ckpt to diag; skipping diag"
            continue
        fi

        DIAG_DONE_MARKER="${DIAG_ARCHIVE}/sustained_contact_val/sustained_contact_summary.md"
        if [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
            log "[R34] [${VID}] diag already archived at ${DIAG_ARCHIVE}; skipping (use --force-rediag)"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        log
        log "================================================================"
        log "[$(date '+%F %T')] DIAG ${VID}  (buckets: ${BUCKETS_STR})"
        log "================================================================"

        if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[R34 DRY-RUN] would diag ${VID} against PB1 at ${PB1_CKPT}"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        DS_OUT_TAG="_r34_${VID}"
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
            log "[R34] [${VID}] DIAG OK (oracle Stage-1) -> ${DIAG_ARCHIVE}"
            DIAGED_OK_VIDS+=("${VID}")

            # ── Second eval mode: generated Stage-1 cond (per §7.2) ──
            if [[ ${SKIP_GENERATED_EVAL} -eq 0 && -d "${GENERATED_UPSTREAM_DIR}" ]]; then
                DS_OUT_TAG_GEN="_r34_${VID}_genstage1"
                SUB_DIR_ROOT_GEN="analyses/round32_stage1p5_substitute_conds${DS_OUT_TAG_GEN}"
                DIAG_DIR_ROOT_GEN="analyses/round32_stage1p5_downstream_diag${DS_OUT_TAG_GEN}"
                DIAG_ARCHIVE_GEN="analyses/round34_diag_${VID}_genstage1"
                rm -rf "${SUB_DIR_ROOT_GEN}"
                rm -rf "${DIAG_DIR_ROOT_GEN}"

                log
                log "[R34] [${VID}] DIAG mode=generated_stage1_cond (UPSTREAM_DIR=${GENERATED_UPSTREAM_DIR})"
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
                    log "[R34] [${VID}] DIAG OK (generated Stage-1) -> ${DIAG_ARCHIVE_GEN}"
                else
                    log "[R34] [${VID}] DIAG FAILED (generated Stage-1, rc=${rc_gen}); continuing"
                fi
            elif [[ ${SKIP_GENERATED_EVAL} -ne 0 ]]; then
                log "[R34] [${VID}] generated-Stage-1 eval skipped (ROUND34_SKIP_GENERATED_EVAL=1)"
            else
                log "[R34] [${VID}] generated-Stage-1 eval skipped (UPSTREAM_DIR ${GENERATED_UPSTREAM_DIR} missing)"
            fi

            # ── C41/S4 quality metrics (per §7.4) ──
            if [[ ${SKIP_QUALITY_METRICS} -eq 0 && -d "${GT_DUMP_DIR}/${BUCKETS_STR%% *}" ]]; then
                QUALITY_OUT="analyses/round34_rfft_followup_20260531/c41_s4_quality_${VID}.md"
                mkdir -p "$(dirname "${QUALITY_OUT}")"
                log "[R34] [${VID}] C41/S4 quality metrics → ${QUALITY_OUT}"
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
            log "[R34] [${VID}] DIAG FAILED (rc=${rc}, dir=$([[ -d ${DIAG_DIR_ROOT} ]] && echo present || echo missing))"
            if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
        fi
    else
        log "[R34] --skip-diag: skipping diag for ${VID}"
    fi
done <<< "${VARIANTS}"

# ─── Phase 3: Comparison summary ────────────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] BUILDING COMPARISON SUMMARY"
log "================================================================"

SUMMARY_MD="analyses/round34_matrix_summary_${STAMP}.md"
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
    base = Path(f"analyses/round34_diag_{v}")
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

    rows.append((v, drift, lh_drift, rh_drift, lr, sp, lw, rw, lk, alt))

out = [
    "# R34 Stage-1.5 wrist low-band + cond-aug ablation matrix — summary (val)",
    "",
    f"Stamp: {stamp}",
    "",
    "Reference baselines:",
    "| metric | A oracle (full GT cond) | R33 V1 (no R34 knobs) |",
    "|---|---:|---:|",
    "| drift_max mean (cm) | 7.55 | 15.21 |",
    "| left_hand drift_max (cm) | (PB1 oracle ~10 cm) | 23.82 |",
    "| right_hand drift_max (cm) | (PB1 oracle ~10 cm) | 27.67 |",
    "",
    "## Per-variant numbers",
    "",
    "| variant | drift_max | lh_drift | rh_drift | L_R_corr | step_period | lw dir_cos | rw dir_cos | lk delta_err | low_alt_amp |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    v, drift, lhd, rhd, lr, sp, lw, rw, lk, alt = row
    out.append(
        f"| {v} | {drift or '?'} | {lhd or '?'} | {rhd or '?'} | "
        f"{lr or '?'} | {sp or '?'} | "
        f"{lw or '?'} | {rw or '?'} | {lk or '?'} | {alt or '?'} |"
    )
out.append("")
out.append("## Decision tree (brief §7.6)")
out.append("")
out.append("- C0 control should ~= R33 V1 (drift_max ~15.21) — substrate sanity")
out.append("- C2 - C0 >= 3 cm wrist drift -> low-band wrist supervision is the")
out.append("  main repair axis. Next: sweep weight/cutoff (brief §8.1) or add")
out.append("  contact-window loss (brief §8.2).")
out.append("- C1 - C0 >= 3 cm -> cond augmentation is effective even under")
out.append("  oracle Stage-1 eval. Next: sigma sweep, consider sigma-conditioning")
out.append("  (brief §8.3).")
out.append("- C3 improves but C1/C2 do not -> the two axes are complementary;")
out.append("  tune both.")
out.append("- All variants close < 1 cm -> abandon these axes. Consider explicit")
out.append("  contact/affordance target, PB1 condition adapter, or C41 interface")
out.append("  redesign (brief §8.4).")

out_md.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {out_md}")
PYEOF

    "${PY}" -u "${SUMMARY_PY}" "${STAMP}" "${SUMMARY_MD}" "${DIAGED_OK_VIDS[@]}" \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R34] WARN: summary build failed (non-fatal)."
fi

# ─── Phase 4: Pack ─────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round34_matrix_results_${STAMP}.tar.gz"
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    PACK_TARGETS=()
    for VID in "${TRAINED_OK_VIDS[@]}"; do
        D="runs/training/${VID}"
        if [[ -d "${D}" ]]; then
            F="${D}/final.pt"
            M="${D}/metrics.jsonl"
            [[ -f "${F}" ]] && PACK_TARGETS+=("${F}")
            [[ -f "${M}" ]] && PACK_TARGETS+=("${M}")
        fi
        L="${OVERALL_LOG_DIR}/${VID}.log"
        [[ -f "${L}" ]] && PACK_TARGETS+=("${L}")
    done
    for VID in "${DIAGED_OK_VIDS[@]}"; do
        A="analyses/round34_diag_${VID}"
        [[ -d "${A}" ]] && PACK_TARGETS+=("${A}")
    done
    [[ -f "${SUMMARY_MD}" ]] && PACK_TARGETS+=("${SUMMARY_MD}")
    [[ -f "${SUMMARY_LOG}" ]] && PACK_TARGETS+=("${SUMMARY_LOG}")

    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        log "wrote ${TARBALL}  (${SIZE})"
    else
        log "[R34 PACK] nothing to pack"
    fi
fi

log
log "================================================================"
log "[$(date '+%F %T')] R34 matrix COMPLETE"
log "================================================================"
log "Trained: ${TRAINED_OK_VIDS[*]:-none}"
log "Diaged:  ${DIAGED_OK_VIDS[*]:-none}"
log
log "Summary log: ${SUMMARY_LOG}"
log "Summary MD:  ${SUMMARY_MD}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    log "Tarball:     ${TARBALL:-<none>}"
fi
