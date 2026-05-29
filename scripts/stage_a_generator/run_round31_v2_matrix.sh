#!/usr/bin/env bash
# Round-31 V2 Stage-1 loss ablation matrix — integrated train + diag launcher.
#
# Per analyses/2026-05-30_round31_v2_stage1_loss_ablation.md.
#
# For each variant in the matrix (default: V0..V5, override with --only):
#   1. Train (from scratch, 80 ep, bs=64, ~1.5 h on 2× 5080)
#   2. Immediately run R31 B downstream diag against frozen PB1 (val, ~50 min)
#   3. Archive that variant's diag output dir to
#      analyses/round31_v2_diag_<variant>/  so the next iteration's
#      diag run does not overwrite it.
#
# At the end, build a comparison summary table across all variants on
# the key 4 metrics (drift_max, wrist dir_cos, knee delta_err, step_period_rate)
# and pack everything into one tarball.
#
# Total time for all 6 variants: ~12 h (6 × 1.5 h train + 6 × ~50 min diag).
# A subset (V0/V3/V5) takes ~6.5 h.
#
# Usage:
#   bash scripts/stage_a_generator/run_round31_v2_matrix.sh
#   bash scripts/stage_a_generator/run_round31_v2_matrix.sh --only stage1_v2_v0_baseline,stage1_v2_v3_rot6d_full,stage1_v2_v5_capacity_kinematic
#   bash scripts/stage_a_generator/run_round31_v2_matrix.sh --dry-run
#   bash scripts/stage_a_generator/run_round31_v2_matrix.sh --skip-train  # only diag (requires ckpts on disk)
#   bash scripts/stage_a_generator/run_round31_v2_matrix.sh --skip-diag   # only train
#
# Environment overrides (also pass through to the per-phase launchers):
#   ROUND31_S1_NUM_PROCESSES=N        accelerate --num_processes for train
#   ROUND31_DS_SEED=42                seed for diag sampling
#   ROUND31_DS_CFG_SCALE=1.0          CFG scale at sample time
#   ROUND31_V2_ALLOW_PARTIAL=1        don't FATAL on a single-variant failure
#   ROUND31_V2_BUCKETS="val"          which buckets to diag (default "val"
#                                      to save time; "train val" for full)
#
# Resuming after a partial:
#   The script checks ``runs/training/<variant>/final.pt`` before each
#   train phase; if the ckpt already exists the variant is skipped to the
#   diag phase. Likewise the diag phase is skipped if
#   ``analyses/round31_v2_diag_<variant>/sustained_contact_val/sustained_contact_summary.md``
#   already exists.

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_DIAG=0
FORCE_RETRAIN=0
FORCE_REDIAG=0
ALLOW_PARTIAL="${ROUND31_V2_ALLOW_PARTIAL:-0}"
BUCKETS_STR="${ROUND31_V2_BUCKETS:-val}"

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
    echo "[V2] FATAL: export DATASETS_ROOT before launch." >&2
    echo "    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

# Default to 2 procs to match A1/PB1 schedule (bs=32 accum=1 implicit).
: "${ROUND31_S1_NUM_PROCESSES:=2}"
export ROUND31_S1_NUM_PROCESSES

PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CKPT="runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt"
SELECTION_VAL="analyses/round29_val_diag_indices_48_balanced.json"
SELECTION_TRAIN="analyses/round27_tier0_train_indices_48_balanced.json"

OVERALL_LOG_DIR="runs/round31_v2_matrix"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[V2] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ─── Regenerate configs (idempotent, picks up server DATASETS_ROOT) ─
log "[V2] Regenerating Stage-1 V2 ablation configs."
if [[ ${DRY_RUN} -eq 0 ]]; then
    "${PY}" scripts/stage_a_generator/round31_make_stage1_configs.py \
        --data-root "${DATASETS_ROOT}" 2>&1 | tee -a "${SUMMARY_LOG}"
fi

# ─── Select variant list from manifest ─────────────────────────────
MANIFEST="analyses/round31_stage1_manifest.json"
if [[ ! -f "${MANIFEST}" && ${DRY_RUN} -eq 0 ]]; then
    log "[V2 FATAL] Manifest missing after generator run: ${MANIFEST}"
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
    log "[V2 DRY-RUN] (manifest missing) using static V0..V5 list"
    VARIANTS="stage1_v2_v0_baseline configs/training/stage1_v2_v0_baseline.yaml runs/training/stage1_v2_v0_baseline
stage1_v2_v1_ortho configs/training/stage1_v2_v1_ortho.yaml runs/training/stage1_v2_v1_ortho
stage1_v2_v2_fk_pos configs/training/stage1_v2_v2_fk_pos.yaml runs/training/stage1_v2_v2_fk_pos
stage1_v2_v3_rot6d_full configs/training/stage1_v2_v3_rot6d_full.yaml runs/training/stage1_v2_v3_rot6d_full
stage1_v2_v4_kinematic_full configs/training/stage1_v2_v4_kinematic_full.yaml runs/training/stage1_v2_v4_kinematic_full
stage1_v2_v5_capacity_kinematic configs/training/stage1_v2_v5_capacity_kinematic.yaml runs/training/stage1_v2_v5_capacity_kinematic"
else
    VARIANTS="$("${PY}" -c "${PICK_SCRIPT}" "${MANIFEST}" "${ONLY}")"
fi
if [[ -z "${VARIANTS}" ]]; then
    log "[V2] no train variants matched only='${ONLY}'"
    exit 0
fi

log
log "===== R31 V2 ablation matrix launch ${STAMP} ====="
log "DATASETS_ROOT=${DATASETS_ROOT}"
log "SKIP_TRAIN=${SKIP_TRAIN}  SKIP_DIAG=${SKIP_DIAG}  BUCKETS=${BUCKETS_STR}"
log "ROUND31_S1_NUM_PROCESSES=${ROUND31_S1_NUM_PROCESSES}"
log "Variants to process:"
echo "${VARIANTS}" | sed 's/^/  /' | tee -a "${SUMMARY_LOG}"
log

# ─── Preflight (catch obvious mistakes before any 1.5h train) ─────
preflight_fail=0
if [[ ${SKIP_DIAG} -eq 0 ]]; then
    if [[ ! -f "${PB1_CKPT}" ]]; then
        log "[V2 PREFLIGHT FAIL] PB1 ckpt missing: ${PB1_CKPT}"
        preflight_fail=1
    fi
    for sel in "${SELECTION_VAL}" "${SELECTION_TRAIN}"; do
        if [[ ! -f "${sel}" ]]; then
            log "[V2 PREFLIGHT FAIL] selection JSON missing: ${sel}"
            preflight_fail=1
        fi
    done
fi
while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    if [[ ! -f "${CFG}" && ${DRY_RUN} -eq 0 ]]; then
        log "[V2 PREFLIGHT FAIL] [${VID}] missing config: ${CFG}"
        preflight_fail=1
    fi
done <<< "${VARIANTS}"
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[V2] FATAL preflight failures."
    exit 1
fi

# ─── Per-variant train -> diag -> archive loop ─────────────────────
TRAINED_OK_VIDS=()
DIAGED_OK_VIDS=()

while IFS=' ' read -r VID CFG OUTDIR; do
    [[ -z "${VID}" ]] && continue
    VARIANT_LOG="${OVERALL_LOG_DIR}/${VID}.log"
    DIAG_ARCHIVE="analyses/round31_v2_diag_${VID}"

    # ─── Phase 1: TRAIN ─────────────────────────────────────────────
    if [[ ${SKIP_TRAIN} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ -f "${FINAL}" && ${FORCE_RETRAIN} -eq 0 ]]; then
            log "[V2] [${VID}] ckpt already exists; skipping train (use --force-retrain to override)"
            TRAINED_OK_VIDS+=("${VID}")
        else
            log
            log "================================================================"
            log "[$(date '+%F %T')] TRAIN ${VID}"
            log "    config: ${CFG}"
            log "    output: ${OUTDIR}"
            log "    log:    ${VARIANT_LOG}"
            log "================================================================"
            if [[ ${DRY_RUN} -eq 1 ]]; then
                log "[V2 DRY-RUN] would train ${VID}"
                TRAINED_OK_VIDS+=("${VID}")
            else
                # Use the existing single-variant launcher with --only,
                # so configs are regenerated + DDP wiring is consistent.
                if ROUND31_S1_NUM_PROCESSES="${ROUND31_S1_NUM_PROCESSES}" \
                   bash scripts/stage_a_generator/run_round31_stage1_training.sh \
                        --only "${VID}" 2>&1 | tee "${VARIANT_LOG}"; then
                    if [[ -f "${FINAL}" ]]; then
                        TRAINED_OK_VIDS+=("${VID}")
                        log "[V2] [${VID}] TRAIN OK -> ${FINAL}"
                    else
                        log "[V2] [${VID}] TRAIN finished but final.pt missing"
                        if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                    fi
                else
                    log "[V2] [${VID}] TRAIN FAILED"
                    if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
                fi
            fi
        fi
    else
        log "[V2] --skip-train: skipping train for ${VID}"
        TRAINED_OK_VIDS+=("${VID}")
    fi

    # ─── Phase 2: DIAG (per variant, immediately after train) ───────
    if [[ ${SKIP_DIAG} -eq 0 ]]; then
        FINAL="${OUTDIR}/final.pt"
        if [[ ! -f "${FINAL}" && ${DRY_RUN} -eq 0 ]]; then
            log "[V2] [${VID}] no ckpt to diag (train failed or skipped without ckpt); skipping diag"
            continue
        fi

        DIAG_DONE_MARKER="${DIAG_ARCHIVE}/sustained_contact_val/sustained_contact_summary.md"
        if [[ -f "${DIAG_DONE_MARKER}" && ${FORCE_REDIAG} -eq 0 ]]; then
            log "[V2] [${VID}] diag already archived at ${DIAG_ARCHIVE}; skipping (use --force-rediag)"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        log
        log "================================================================"
        log "[$(date '+%F %T')] DIAG ${VID}  (buckets: ${BUCKETS_STR})"
        log "================================================================"

        if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[V2 DRY-RUN] would diag ${VID} against PB1 at ${PB1_CKPT}"
            DIAGED_OK_VIDS+=("${VID}")
            continue
        fi

        # The downstream-diag launcher writes to a fixed
        # analyses/round31_stage1_downstream_diag/ dir. Run, then move.
        # First clear any stale output from a previous variant's run.
        if [[ -d "analyses/round31_stage1_downstream_diag" ]]; then
            rm -rf "analyses/round31_stage1_downstream_diag"
        fi
        if [[ -d "analyses/round31_stage1_substitute_conds" ]]; then
            rm -rf "analyses/round31_stage1_substitute_conds"
        fi

        if ROUND31_DS_STAGE1_CFG="${CFG}" \
           ROUND31_DS_STAGE1_CKPT="${FINAL}" \
           ROUND31_DS_BUCKETS="${BUCKETS_STR}" \
           bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh \
                2>&1 | tee -a "${VARIANT_LOG}"; then
            mkdir -p "$(dirname "${DIAG_ARCHIVE}")"
            if [[ -d "analyses/round31_stage1_downstream_diag" ]]; then
                rm -rf "${DIAG_ARCHIVE}"
                mv "analyses/round31_stage1_downstream_diag" "${DIAG_ARCHIVE}"
                log "[V2] [${VID}] DIAG OK -> ${DIAG_ARCHIVE}"
                DIAGED_OK_VIDS+=("${VID}")
            else
                log "[V2] [${VID}] DIAG ran but produced no analyses dir"
                if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
            fi
        else
            log "[V2] [${VID}] DIAG FAILED"
            if [[ "${ALLOW_PARTIAL}" != "1" ]]; then exit 1; fi
        fi
    else
        log "[V2] --skip-diag: skipping diag for ${VID}"
    fi
done <<< "${VARIANTS}"

# ─── Phase 3: Comparison summary table ─────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] BUILDING COMPARISON SUMMARY"
log "================================================================"

SUMMARY_MD="analyses/round31_v2_matrix_summary_${STAMP}.md"
if [[ ${DRY_RUN} -eq 0 ]]; then
    "${PY}" -u -c "
import json, re
from pathlib import Path

variants = [v for v in '''${DIAGED_OK_VIDS[@]}'''.split() if v]
if not variants:
    print('no successful diag runs to summarise')
    raise SystemExit(0)

def read_metric(md_path, regex, group=1):
    if not Path(md_path).exists():
        return None
    txt = Path(md_path).read_text(encoding='utf-8')
    m = re.search(regex, txt)
    return m.group(group) if m else None

rows = []
for v in variants:
    base = Path(f'analyses/round31_v2_diag_{v}')
    sc_path = base / 'sustained_contact_val' / 'sustained_contact_summary.md'
    gait_path = base / 'gait_val' / 'gait_summary.md'
    body_path = base / 'body_action_val' / 'body_action_summary.md'
    g1_path = base / 'g1_soft_stance_val' / 'g1_soft_stance_summary.md'

    # drift_max mean from sustained_contact
    drift = read_metric(str(sc_path), r'drift_max_cm:\s+mean=([\d.]+)')
    # gait L_R_corr from gait
    lr = read_metric(str(gait_path), r'mean L_R_height_corr\s*\|\s*[-\d.]+\s*\|\s*([-\d.]+)')
    # step_period_rate from gait
    sp = read_metric(str(gait_path), r'segments with detected period\s*\|\s*[\d.]+%\s*\|\s*([\d.]+)%')
    # body_action left_wrist direction_cos
    lw_dir = read_metric(str(body_path), r'left_wrist\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*([\d.]+)')
    # body_action right_wrist direction_cos
    rw_dir = read_metric(str(body_path), r'right_wrist\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*([\d.]+)')
    # body_action left_knee delta_err
    lk_err = read_metric(str(body_path), r'left_knee\s*\|\s*([\d.]+)')
    # g1 low_alt_amplitude_rate
    alt = read_metric(str(g1_path), r'low_alt_amplitude_rate \(seg\)\s*\|\s*([\d.]+)')

    rows.append({
        'variant': v, 'drift_max': drift, 'L_R_corr': lr,
        'step_period_rate': sp, 'lw_dir_cos': lw_dir,
        'rw_dir_cos': rw_dir, 'lk_delta_err': lk_err,
        'low_alt_amp_rate': alt,
    })

# Render markdown.
out = ['# R31 V2 ablation matrix — comparison summary (val)', '']
out.append('Stamp: ${STAMP}')
out.append('')
out.append('PB1 oracle baseline (A) from the original R29-PB1 ckpt for reference:')
out.append('| metric | A (oracle) | ship gate |')
out.append('|---|---:|---|')
out.append('| drift_max mean (cm) | 7.55 | regression ≤ +1 cm |')
out.append('| L_R_corr | -0.219 | regression ≤ +0.05 |')
out.append('| step_period_rate | 39.0% | regression ≤ -3 pp |')
out.append('| lw direction_cos | 0.903 | — |')
out.append('| rw direction_cos | 0.912 | — |')
out.append('| lk delta_err (cm) | 7.64 | — |')
out.append('| low_alt_amplitude_rate | 0.627 | — |')
out.append('')
out.append('## Per-variant numbers')
out.append('')
out.append('| variant | drift_max | L_R_corr | step_period | lw dir_cos | rw dir_cos | lk delta_err | low_alt_amp |')
out.append('|---|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    out.append(
        f'| {r[\"variant\"]} | {r[\"drift_max\"] or \"?\"} | {r[\"L_R_corr\"] or \"?\"} | '
        f'{r[\"step_period_rate\"] or \"?\"} | {r[\"lw_dir_cos\"] or \"?\"} | '
        f'{r[\"rw_dir_cos\"] or \"?\"} | {r[\"lk_delta_err\"] or \"?\"} | '
        f'{r[\"low_alt_amp_rate\"] or \"?\"} |'
    )
out.append('')

Path('${SUMMARY_MD}').write_text('\n'.join(out), encoding='utf-8')
print(f'wrote ${SUMMARY_MD}')
"
fi

# ─── Phase 4: Pack everything ─────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 ]]; then
    TARBALL="analyses/round31_v2_matrix_results_${STAMP}.tar.gz"
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
        A="analyses/round31_v2_diag_${VID}"
        [[ -d "${A}" ]] && PACK_TARGETS+=("${A}")
    done
    [[ -f "${SUMMARY_MD}" ]] && PACK_TARGETS+=("${SUMMARY_MD}")
    [[ -f "${SUMMARY_LOG}" ]] && PACK_TARGETS+=("${SUMMARY_LOG}")
    PACK_TARGETS+=("${MANIFEST}")
    [[ -f analyses/round31_stage1_manifest.md ]] && PACK_TARGETS+=("analyses/round31_stage1_manifest.md")

    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        log "wrote ${TARBALL}  (${SIZE})"
    else
        log "[V2 PACK] nothing to pack"
    fi
fi

log
log "================================================================"
log "[$(date '+%F %T')] R31 V2 matrix COMPLETE"
log "================================================================"
log "Trained: ${TRAINED_OK_VIDS[*]:-none}"
log "Diaged:  ${DIAGED_OK_VIDS[*]:-none}"
log
log "Summary log: ${SUMMARY_LOG}"
log "Summary MD:  ${SUMMARY_MD}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    log "Tarball:     ${TARBALL}"
fi
