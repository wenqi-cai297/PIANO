#!/usr/bin/env bash
# Round-33 V0 Stage-1.5 per-block obj_xattn ablation matrix —
# integrated train + downstream-coupling diag launcher.
#
# Tests the DiT-XL pattern (cross-attn over object_tokens in EVERY DiT
# layer) against R32 V0/V7's end-of-encoder-only design.
#
# Four variants — see analyses/round33_stage1p5_v0_manifest.md.
#
# For each variant: TRAIN (from scratch, 80 ep, bs=48) → R32 downstream
# diag against frozen PB1 with oracle Stage-1 cond (val bucket only by
# default) → archive diag under analyses/round33_v0_diag_<variant>/.
#
# **GPU restriction**: CUDA_VISIBLE_DEVICES=0,2 by default. Override
# with ROUND33_V0_GPUS.
#
# Total time for 4 variants: ~2 h (R33 V1 etc. add a per-block xattn
# sub-layer; each Stage-1.5 train is ~25-30 min on 2x 5080).
#
# Usage:
#   tmux new -s r33v0
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round33_v0_matrix.sh
#
#   # Subset:
#   bash scripts/stage_a_generator/run_round33_v0_matrix.sh \
#       --only stage1p5_r33_v0_control,stage1p5_r33_v1_xattn
#
#   # Dry-run:
#   bash scripts/stage_a_generator/run_round33_v0_matrix.sh --dry-run
#
# Environment overrides:
#   ROUND33_V0_GPUS="0,2"          CUDA_VISIBLE_DEVICES mask
#   ROUND33_V0_NUM_PROCESSES=N      accelerate --num_processes
#   ROUND33_V0_BUCKETS="val"        diag buckets
#   ROUND33_V0_ALLOW_PARTIAL=1      keep going on a per-variant failure

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
FORCE_RETRAIN=0
FORCE_REDIAG=0
ALLOW_PARTIAL="${ROUND33_V0_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND33_V0_BUCKETS:-val}"

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
    echo "[R33] FATAL: export DATASETS_ROOT before launch." >&2
    echo "    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

GPUS="${ROUND33_V0_GPUS:-0,2}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

NUM_GPUS_IN_MASK="$(echo "${GPUS}" | tr ',' '\n' | grep -c '^[0-9]\+$' || true)"
if [[ "${NUM_GPUS_IN_MASK}" -lt 1 ]]; then NUM_GPUS_IN_MASK=1; fi

: "${ROUND33_V0_NUM_PROCESSES:=${NUM_GPUS_IN_MASK}}"
export ROUND33_V0_NUM_PROCESSES

PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CKPT="runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"

OVERALL_LOG_DIR="runs/round33_v0_matrix"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R33] FATAL: no python found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ─── Regenerate R33 configs (idempotent) ───────────────────────────
log "[R33] Regenerating Stage-1.5 R33 ablation configs."
if [[ ${DRY_RUN} -eq 0 ]]; then
    "${PY}" scripts/stage_a_generator/round33_make_stage1p5_v0_configs.py \
        --data-root "${DATASETS_ROOT}" 2>&1 | tee -a "${SUMMARY_LOG}"
fi

MANIFEST="analyses/round33_stage1p5_v0_manifest.json"
if [[ ! -f "${MANIFEST}" && ${DRY_RUN} -eq 0 ]]; then
    log "[R33 FATAL] Manifest missing after generator run: ${MANIFEST}"
    exit 1
fi

PICK_SCRIPT='
import json, sys
m = json.load(open(sys.argv[1]))
only = sys.argv[2]
want_only = set(only.split(",")) if only else None
for v in m["variants"]:
    if not v.get("train", True): continue
    if want_only is not None and v["variant_id"] not in want_only: continue
    print(v["variant_id"], v["config_path"], v["output_dir"])
'
if [[ ${DRY_RUN} -eq 1 && ! -f "${MANIFEST}" ]]; then
    log "[R33 DRY-RUN] (manifest missing) using static V0..V3 list"
    VARIANTS="stage1p5_r33_v0_control configs/training/stage1p5_r33_v0_control.yaml runs/training/stage1p5_r33_v0_control
stage1p5_r33_v1_xattn configs/training/stage1p5_r33_v1_xattn.yaml runs/training/stage1p5_r33_v1_xattn
stage1p5_r33_v2_xattn_moment configs/training/stage1p5_r33_v2_xattn_moment.yaml runs/training/stage1p5_r33_v2_xattn_moment
stage1p5_r33_v3_xattn_v7full configs/training/stage1p5_r33_v3_xattn_v7full.yaml runs/training/stage1p5_r33_v3_xattn_v7full"
else
    VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${ONLY}")"
fi
if [[ -z "${VARIANTS}" ]]; then
    log "[R33] no train variants matched only='${ONLY}'"
    exit 0
fi

log
log "===== R33 V0 ablation matrix launch ${STAMP} ====="
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "GPUS=${GPUS}  NUM_PROCESSES=${ROUND33_V0_NUM_PROCESSES}"
log "SKIP_TRAIN=${SKIP_TRAIN}  SKIP_DIAG=${SKIP_DIAG}  BUCKETS=${BUCKETS_STR}"
log "PB1_CKPT=${PB1_CKPT}"
log "Variants to process:"
echo "${VARIANTS}" | sed 's/^/  /' | tee -a "${SUMMARY_LOG}"
log

# ─── Preflight ──────────────────────────────────────────────────────
preflight_fail=0
if [[ ${SKIP_DIAG} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
    if [[ ! -f "${PB1_CKPT}" ]]; then
        log "[R33 PREFLIGHT FAIL] PB1 ckpt missing: ${PB1_CKPT}"
        preflight_fail=1
    fi
    for sel in "${SELECTION_VAL}" "${SELECTION_TRAIN}"; do
        if [[ ! -f "${sel}" ]]; then
            case "${BUCKETS_STR}" in
                *train*)
                    if [[ "${sel}" == "${SELECTION_TRAIN}" ]]; then
                        log "[R33 PREFLIGHT FAIL] train selection JSON missing: ${sel}"
                        preflight_fail=1
                    fi ;;
            esac
            case "${BUCKETS_STR}" in
                *val*)
                    if [[ "${sel}" == "${SELECTION_VAL}" ]]; then
                        log "[R33 PREFLIGHT FAIL] val selection JSON missing: ${sel}"
                        preflight_fail=1
                    fi ;;
            esac
        fi
    done
fi
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -f "${CFG}" && ${DRY_RUN} -eq 0 ]]; then
        log "[R33 PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R33] FATAL preflight failures."
    exit 1
fi

# ─── Per-variant train -> diag -> archive loop ─────────────────────
TRAINED_OK_VIDS=()
DIAGED_OK_VIDS=()

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    VARIANT_LOG="${OVERALL_LOG_DIR}/${VID}.log"
    DIAG_ARCHIVE="analyses/round33_v0_diag_${VID}"

    # ─── Phase 1: TRAIN ─────────────────────────────────────────────
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ -f "${FINAL}" && ${FORCE_RETRAIN} -eq 0 ]]; then
            log "[R33] [${VID}] ckpt already exists; skipping train (use --force-retrain)"
            TRAINED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] TRAIN ${VID}"
            log "    config: ${CFG}"
            log "    output: ${OUTDIR}"
            log "    log:    ${VARIANT_LOG}"
            log "    GPUs:   ${GPUS}  procs=${ROUND33_V0_NUM_PROCESSES}"
            log "================================================================"

            if [[ "${ROUND33_V0_NUM_PROCESSES}" -le 1 ]]; then
                TRAIN_CMD=("${PY}" -u src/piano/training/train_stage1p5.py --config "${CFG}")
            else
                TRAIN_CMD=(accelerate launch
                    --num_processes "${ROUND33_V0_NUM_PROCESSES}"
                    --multi_gpu --mixed_precision bf16
                    src/piano/training/train_stage1p5.py --config "${CFG}")
            fi

            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[R33 DRY-RUN] would train ${VID}"
                log "    \$ CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD[*]}"
                TRAINED_OK_VIDS+=("${VID}")
            else
                set +e
                "${TRAIN_CMD[@]}" 2>&1 | tee "${VARIANT_LOG}"
                rc=${PIPESTATUS[0]}
                set -e
                if [[ ${rc} -eq 0 && -f "${FINAL}" ]]; then
                    TRAINED_OK_VIDS+=("${VID}")
                    log "[R33] [${VID}] TRAIN OK -> ${FINAL}"
                else
                    log "[R33] [${VID}] TRAIN FAILED (rc=${rc}, final.pt=$([[ -f ${FINAL} ]] && echo present || echo missing))"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                    continue
                fi
            fi
        fi
    else
        log "[R33] --skip-train: skipping train for ${VID}"
        TRAINED_OK_VIDS+=("${VID}")
    fi

    # ─── Phase 2: DIAG ──────────────────────────────────────────────
    if [[ ${SKIP_DIAG} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ ! -f "${FINAL}" && ${DRY_RUN} -eq 0 ]]; then
            log "[R33] [${VID}] no ckpt to diag; skipping diag"
            continue
        fi

        DIAG_DONE_MARKER="${DIAG_ARCHIVE}/sustained_contact_val/sustained_contact_summary.md"
        if [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
            log "[R33] [${VID}] diag already archived at ${DIAG_ARCHIVE}; skipping (use --force-rediag)"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        log
        log "================================================================"
        log "[$(date '+%F %T')] DIAG ${VID}  (buckets: ${BUCKETS_STR})"
        log "================================================================"

        if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[R33 DRY-RUN] would diag ${VID} against PB1 at ${PB1_CKPT}"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        DS_OUT_TAG="_r33_${VID}"
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
            log "[R33] [${VID}] DIAG OK -> ${DIAG_ARCHIVE}"
            DIAGED_OK_VIDS+=("${VID}")
        else
            log "[R33] [${VID}] DIAG FAILED (rc=${rc}, dir=$([[ -d ${DIAG_DIR_ROOT} ]] && echo present || echo missing))"
            if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
        fi
    else
        log "[R33] --skip-diag: skipping diag for ${VID}"
    fi
done <<< "${VARIANTS}"

# ─── Phase 3: Comparison summary ────────────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] BUILDING COMPARISON SUMMARY"
log "================================================================"

SUMMARY_MD="analyses/round33_v0_matrix_summary_${STAMP}.md"
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
    base = Path(f"analyses/round33_v0_diag_{v}")
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
    "# R33 V0 Stage-1.5 per-block obj_xattn ablation matrix — summary (val)",
    "",
    f"Stamp: {stamp}",
    "",
    "Reference baselines (Stage-1 oracle + V0 Stage-1.5 -> PB1):",
    "| metric | A oracle (full GT cond) | V0 (R32 V7 V0 baseline) |",
    "|---|---:|---:|",
    "| drift_max mean (cm) | 7.55 | 15.74 |",
    "| left_hand drift_max (cm) | (PB1 oracle ~10 cm) | 22.92 |",
    "| right_hand drift_max (cm) | (PB1 oracle ~10 cm) | 26.73 |",
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
out.append("## Decision tree")
out.append("")
out.append("- V0 (this run) should ~= R32 V7 V0 control (lh ~22, rh ~27)")
out.append("  — noise floor sanity check.")
out.append("- V1 (per-block obj_xattn ONLY) closes >= 5 cm wrist drift over V0")
out.append("  -> architecture is dominant; ship V1, queue cascade training.")
out.append("- V2 (xattn + V7-A moment) - V1 >= 2 cm -> V7-A loss has ROI only")
out.append("  inside the new architecture; ship V2.")
out.append("- V3 (xattn + V7 V5 full) is the upper bound; if V3 - V1 < 2 cm,")
out.append("  V7 losses contribute nothing on top of the structure change.")
out.append("- All R33 variants close < 2 cm -> architecture isn't dominant either;")
out.append("  pivot to cascade training (Stage-1.5 sees Stage-1 generated cond).")

out_md.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {out_md}")
PYEOF

    "${PY}" -u "${SUMMARY_PY}" "${STAMP}" "${SUMMARY_MD}" "${DIAGED_OK_VIDS[@]}" \
        2>&1 | tee -a "${SUMMARY_LOG}" || \
        log "[R33] WARN: summary build failed (non-fatal)."
fi

# ─── Phase 4: Pack ─────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round33_v0_matrix_results_${STAMP}.tar.gz"
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
        A="analyses/round33_v0_diag_${VID}"
        [[ -d "${A}" ]] && PACK_TARGETS+=("${A}")
    done
    [[ -f "${SUMMARY_MD}" ]] && PACK_TARGETS+=("${SUMMARY_MD}")
    [[ -f "${SUMMARY_LOG}" ]] && PACK_TARGETS+=("${SUMMARY_LOG}")
    [[ -f "${MANIFEST}" ]] && PACK_TARGETS+=("${MANIFEST}")
    [[ -f analyses/round33_stage1p5_v0_manifest.md ]] && PACK_TARGETS+=("analyses/round33_stage1p5_v0_manifest.md")

    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        log "wrote ${TARBALL}  (${SIZE})"
    else
        log "[R33 PACK] nothing to pack"
    fi
fi

log
log "================================================================"
log "[$(date '+%F %T')] R33 V0 matrix COMPLETE"
log "================================================================"
log "Trained: ${TRAINED_OK_VIDS[*]:-none}"
log "Diaged:  ${DIAGED_OK_VIDS[*]:-none}"
log
log "Summary log: ${SUMMARY_LOG}"
log "Summary MD:  ${SUMMARY_MD}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    log "Tarball:     ${TARBALL:-<none>}"
fi
