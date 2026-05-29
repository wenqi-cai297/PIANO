#!/usr/bin/env bash
# Round-31 Phase 0.5 — CFG / sampler sweep on Stage-1 V0 ckpt.
#
# Per analyses/2026-05-30_round31_v2_chatgpt_review_response.md §4
# ("Phase 0.5 — CFG + sampler sweep on V0 ckpt").
#
# Motivation:
#   run_round31_stage1_downstream_diag.sh defaults to CFG=1.0 + sampler=ddim_eta0
#   for Stage-1 inference; the Stage-1 design doc
#   (analyses/2026-05-29_stage1_and_stage1_5_design.md §"CFG dropout")
#   says inference CFG should start at 1.5. All 6 R31 V2 variants were
#   diag'd at cfg=1.0 + ddim_eta0. Before declaring the V2 negative result
#   final, we sweep:
#
#     Stage-1 CFG  ∈ {1.0, 1.5, 2.0, 2.5}     (4)
#     Stage-1 sampler ∈ {ddim_eta0, ddpm}     (2)
#                                            ───
#                                            8 sample+diag combos
#
#   Note on samplers: both ddim_eta0 and ddpm walk the full self.num_steps
#   (1000 for Stage-1) — see motion_anchordiff.py:p_sample_loop. They differ
#   only in noise-injection at intermediate t (deterministic vs stochastic),
#   NOT in step count. So this sweep tests "does CFG matter" + "does sampler
#   stochasticity matter", not "does step-count mismatch matter."
#
#   PB1's CFG stays at 1.0 (apples-to-apples vs PB1 oracle's 7.55 cm baseline,
#   which was measured at PB1 cfg=1.0). PB1's sampler is the model.sample()
#   default (no per-diag knob).
#
#   Each combo = 1 Stage-1 sample pass (val 48 clips) + 4 PB1 diag kinds.
#   Results land under per-combo directories so they do not clobber the V2
#   tarball's analyses/round31_v2_diag_*/ dirs.
#
# Cost estimate (single GPU per combo, val bucket only):
#   sample DDIM-1000 η=0: ~6-8 min  (deterministic, 1000 forward passes)
#   sample DDPM-1000    : ~6-8 min  (stochastic,    1000 forward passes)
#   4 PB1 diag kinds    : ~40 min
#   per combo: ~50 min × 8 combos = ~6.5 h on dual-5080
#
# Usage:
#   tmux new -s r31p05
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round31_phase0p5_cfg_sampler_sweep.sh
#
#   # Subset (cheapest first — try cfg=1.5 ddim only, before the full 8):
#   bash scripts/stage_a_generator/run_round31_phase0p5_cfg_sampler_sweep.sh \
#       --only cfg1p5_ddim_eta0
#
#   # Dry-run to confirm combos:
#   bash scripts/stage_a_generator/run_round31_phase0p5_cfg_sampler_sweep.sh --dry-run
#
# Env overrides:
#   ROUND31_P05_STAGE1_CFG=...      path to Stage-1 train config
#                                   (default configs/training/stage1_v2_v0_baseline.yaml)
#   ROUND31_P05_STAGE1_CKPT=...     path to Stage-1 ckpt
#                                   (default runs/training/stage1_v2_v0_baseline/final.pt)
#   ROUND31_P05_PB1_CKPT=...        path to PB1 ckpt (default runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt)
#   ROUND31_P05_CFGS="1.0 1.5 2.0 2.5"   space-separated CFG scales
#   ROUND31_P05_SAMPLERS="ddim_eta0 ddpm"  space-separated samplers
#   ROUND31_P05_BUCKETS="val"       diag buckets
#   ROUND31_P05_ALLOW_PARTIAL=1     don't FATAL on a single combo failure
#
# Resuming:
#   This script checks for the per-combo summary marker (val
#   sustained_contact_summary.md). If it exists, the combo is skipped.
#   To force re-run, delete the marker or use --force-rerun.

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY=""
DRY_RUN=0
FORCE_RERUN=0
ALLOW_PARTIAL="${ROUND31_P05_ALLOW_PARTIAL:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)        ONLY="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --force-rerun) FORCE_RERUN=1; shift ;;
        -h|--help)     sed -n '1,55p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    echo "[P05] FATAL: export DATASETS_ROOT before launch." >&2
    echo "    export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4" >&2
    exit 1
fi

STAGE1_CFG_FILE="${ROUND31_P05_STAGE1_CFG:-configs/training/stage1_v2_v0_baseline.yaml}"
STAGE1_CKPT="${ROUND31_P05_STAGE1_CKPT:-runs/training/stage1_v2_v0_baseline/final.pt}"
PB1_CKPT="${ROUND31_P05_PB1_CKPT:-runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt}"
CFGS_STR="${ROUND31_P05_CFGS:-1.0 1.5 2.0 2.5}"
SAMPLERS_STR="${ROUND31_P05_SAMPLERS:-ddim_eta0 ddpm}"
BUCKETS_STR="${ROUND31_P05_BUCKETS:-val}"

# shellcheck disable=SC2206
CFGS=(${CFGS_STR})
# shellcheck disable=SC2206
SAMPLERS=(${SAMPLERS_STR})

OVERALL_LOG_DIR="runs/round31_phase0p5_sweep"
mkdir -p "${OVERALL_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${OVERALL_LOG_DIR}/sweep_${STAMP}.log"

log() { echo "[P05 $(date '+%H:%M:%S')] $*" | tee -a "${SUMMARY_LOG}"; }

# Preflight: must have Stage-1 cfg + ckpt + PB1 ckpt before doing anything.
# Skipped under --dry-run (allows config-syntax sanity check on the laptop).
if [[ ${DRY_RUN} -eq 0 ]]; then
    preflight_fail=0
    for p in "${STAGE1_CFG_FILE}" "${STAGE1_CKPT}" "${PB1_CKPT}"; do
        [[ ! -e "${p}" ]] && { echo "[P05 PREFLIGHT FAIL] missing: ${p}"; preflight_fail=1; }
    done
    if [[ ${preflight_fail} -ne 0 ]]; then
        echo "[P05] FATAL preflight failures."
        exit 1
    fi
fi

# Build the list of (cfg, sampler) combos.
COMBOS=()
for cfg in "${CFGS[@]}"; do
    for sampler in "${SAMPLERS[@]}"; do
        # Combo id (used in directory tags) — replace "." with "p".
        cfg_tag="${cfg//./p}"
        combo_id="cfg${cfg_tag}_${sampler}"
        COMBOS+=("${combo_id}|${cfg}|${sampler}")
    done
done

# Apply --only filter.
if [[ -n "${ONLY}" ]]; then
    filtered=()
    IFS=',' read -ra wanted <<< "${ONLY}"
    for c in "${COMBOS[@]}"; do
        cid="${c%%|*}"
        for w in "${wanted[@]}"; do
            if [[ "${cid}" == "${w}" ]]; then
                filtered+=("${c}")
                break
            fi
        done
    done
    COMBOS=("${filtered[@]}")
fi

log "================================================================"
log "R31 Phase 0.5 sweep starting  ($(date '+%F %T'))"
log "stage1_cfg=${STAGE1_CFG_FILE}"
log "stage1_ckpt=${STAGE1_CKPT}"
log "pb1_ckpt=${PB1_CKPT}"
log "cfgs=(${CFGS[*]})  samplers=(${SAMPLERS[*]})  buckets=(${BUCKETS_STR})"
log "combos=${#COMBOS[@]}  dry_run=${DRY_RUN}  allow_partial=${ALLOW_PARTIAL}"
log "================================================================"

DONE_COMBO_IDS=()

for entry in "${COMBOS[@]}"; do
    IFS='|' read -r combo_id cfg_scale sampler <<< "${entry}"

    OUT_TAG="_phase0p5_${combo_id}"
    SUB_DIR_ROOT="analyses/round31_stage1_substitute_conds${OUT_TAG}"
    DIAG_DIR_ROOT="analyses/round31_stage1_downstream_diag${OUT_TAG}"
    LOG_DIR="runs/round31_stage1_downstream${OUT_TAG}"

    # Marker that says "this combo's val diag finished successfully."
    DONE_MARKER="${DIAG_DIR_ROOT}/sustained_contact_val/sustained_contact_summary.md"

    log
    log "================================================================"
    log "[$(date '+%F %T')] COMBO ${combo_id}  (cfg=${cfg_scale}  sampler=${sampler})"
    log "    sub_dir=${SUB_DIR_ROOT}"
    log "    diag_dir=${DIAG_DIR_ROOT}"
    log "================================================================"

    if [[ ${FORCE_RERUN} -eq 0 && -f "${DONE_MARKER}" ]]; then
        log "[${combo_id}] DONE marker exists → skipping. (Use --force-rerun to redo.)"
        DONE_COMBO_IDS+=("${combo_id}")
        continue
    fi

    LAUNCHER_LOG="${OVERALL_LOG_DIR}/${combo_id}.log"

    SUB_CMD=(bash scripts/stage_a_generator/run_round31_stage1_downstream_diag.sh)

    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[${combo_id}] DRY-RUN — would invoke:"
        log "    ROUND31_DS_STAGE1_CFG=${STAGE1_CFG_FILE} \\"
        log "    ROUND31_DS_STAGE1_CKPT=${STAGE1_CKPT} \\"
        log "    ROUND31_DS_PB1_CKPT=${PB1_CKPT} \\"
        log "    ROUND31_DS_STAGE1_CFG_SCALE=${cfg_scale} \\"
        log "    ROUND31_DS_PB1_CFG_SCALE=1.0 \\"
        log "    ROUND31_DS_STAGE1_SAMPLER=${sampler} \\"
        log "    ROUND31_DS_BUCKETS='${BUCKETS_STR}' \\"
        log "    ROUND31_DS_OUT_TAG=${OUT_TAG} \\"
        log "    ROUND31_DS_ALLOW_PARTIAL=${ALLOW_PARTIAL} \\"
        log "    ${SUB_CMD[*]}"
        DONE_COMBO_IDS+=("${combo_id}")
        continue
    fi

    # Clear any partial state from a previous half-finished run for this combo.
    rm -rf "${SUB_DIR_ROOT}"
    rm -rf "${DIAG_DIR_ROOT}"

    set +e
    ROUND31_DS_STAGE1_CFG="${STAGE1_CFG_FILE}" \
    ROUND31_DS_STAGE1_CKPT="${STAGE1_CKPT}" \
    ROUND31_DS_PB1_CKPT="${PB1_CKPT}" \
    ROUND31_DS_STAGE1_CFG_SCALE="${cfg_scale}" \
    ROUND31_DS_PB1_CFG_SCALE="1.0" \
    ROUND31_DS_STAGE1_SAMPLER="${sampler}" \
    ROUND31_DS_BUCKETS="${BUCKETS_STR}" \
    ROUND31_DS_OUT_TAG="${OUT_TAG}" \
    ROUND31_DS_ALLOW_PARTIAL="${ALLOW_PARTIAL}" \
        "${SUB_CMD[@]}" 2>&1 | tee "${LAUNCHER_LOG}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ ${rc} -ne 0 ]]; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            log "[${combo_id}] WARN: downstream-diag rc=${rc}; continuing (ALLOW_PARTIAL=1)"
            continue
        else
            log "[${combo_id}] FATAL: downstream-diag rc=${rc}"
            exit "${rc}"
        fi
    fi

    if [[ ! -f "${DONE_MARKER}" ]]; then
        log "[${combo_id}] WARN: downstream-diag returned 0 but DONE_MARKER missing (${DONE_MARKER})"
        if [[ "${ALLOW_PARTIAL}" != "1" ]]; then
            exit 1
        fi
        continue
    fi

    DONE_COMBO_IDS+=("${combo_id}")
    log "[${combo_id}] DONE → ${DIAG_DIR_ROOT}"
done

# ─── Build comparison summary ────────────────────────────────────────
log
log "================================================================"
log "[$(date '+%F %T')] BUILDING COMPARISON SUMMARY"
log "================================================================"

SUMMARY_MD="analyses/round31_phase0p5_sweep_summary_${STAMP}.md"
if [[ ${DRY_RUN} -eq 0 && ${#DONE_COMBO_IDS[@]} -gt 0 ]]; then
    SUMMARY_PY="${OVERALL_LOG_DIR}/build_summary_${STAMP}.py"
    cat > "${SUMMARY_PY}" <<'PYEOF'
import re
import sys
from pathlib import Path

stamp = sys.argv[1]
out_md = Path(sys.argv[2])
combos = sys.argv[3:]


def read_metric(md_path, regex, group=1):
    if not Path(md_path).exists():
        return None
    txt = Path(md_path).read_text(encoding="utf-8")
    m = re.search(regex, txt)
    return m.group(group) if m else None


rows = []
for combo in combos:
    base = Path(f"analyses/round31_stage1_downstream_diag_phase0p5_{combo}")
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

    rows.append((combo, drift, lr, sp, lw, rw, lk, alt))

out = [
    "# R31 Phase 0.5 — CFG / sampler sweep summary",
    "",
    f"Stamp: {stamp}",
    "",
    "Stage-1 ckpt: `runs/training/stage1_v2_v0_baseline/final.pt`",
    "PB1 ckpt   : `runs/training/stageB_anchordiff_r29_pb_a1_adaln_s4/final.pt`",
    "PB1 cfg    : fixed at 1.0 across all combos.",
    "",
    "Reference (PB1 fed GT oracle stage1_coarse at PB1 cfg=1.0, sampler=ddim_eta0):",
    "| metric | A (oracle) |",
    "|---|---:|",
    "| drift_max mean (cm) | 7.55 |",
    "| L_R_corr | -0.219 |",
    "| step_period_rate | 39.0 % |",
    "| lw dir_cos | 0.903 |",
    "| rw dir_cos | 0.912 |",
    "| lk delta_err (cm) | 7.64 |",
    "| low_alt_amplitude_rate | 0.627 |",
    "",
    "Reference (PB1 fed V0 generated stage1_coarse at Stage-1 cfg=1.0 sampler=ddim_eta0,",
    "i.e. the original R31 V2 V0 numbers — this is the baseline this sweep is testing against):",
    "| metric | V0 (cfg=1.0 ddim_eta0) |",
    "|---|---:|",
    "| drift_max | 18.47 |",
    "| L_R_corr | -0.418 |",
    "| step_period_rate | 49.2 % |",
    "| lw dir_cos | 0.500 |",
    "| rw dir_cos | 0.416 |",
    "| lk delta_err (cm) | 18.17 |",
    "| low_alt_amplitude_rate | 0.695 |",
    "",
    "## Per-combo numbers (val bucket)",
    "",
    "| combo | drift_max | L_R_corr | step_period | lw dir_cos | rw dir_cos | lk delta_err | low_alt_amp |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for combo, drift, lr, sp, lw, rw, lk, alt in rows:
    out.append(
        f"| {combo} | {drift or '?'} | {lr or '?'} | {sp or '?'} | "
        f"{lw or '?'} | {rw or '?'} | {lk or '?'} | {alt or '?'} |"
    )
out.append("")
out.append("## Decision rules")
out.append("")
out.append("- If ANY combo's drift_max <= 14.5 cm (closes >= 3 cm of the 10 cm")
out.append("  oracle gap), CFG/sampler IS load-bearing → re-launch the V2 ablation")
out.append("  matrix at that best combo before declaring V2 a real negative result.")
out.append("- If every combo lands in 17-19 cm, V2 negative result stands → proceed")
out.append("  to Phase 1 (frame-0 + cond-distribution audit).")

out_md.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {out_md}")
PYEOF

    if [[ -z "${PY:-}" ]]; then
        if command -v python >/dev/null 2>&1; then PY="python";
        elif command -v python3 >/dev/null 2>&1; then PY="python3";
        else log "[P05] WARN: no python found; skipping summary build."; PY=""; fi
    fi
    if [[ -n "${PY}" ]]; then
        "${PY}" -u "${SUMMARY_PY}" "${STAMP}" "${SUMMARY_MD}" "${DONE_COMBO_IDS[@]}" 2>&1 | tee -a "${SUMMARY_LOG}" || \
            log "[P05] WARN: summary build failed (non-fatal)."
    fi
fi

# ─── Pack everything ─────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 0 && ${#DONE_COMBO_IDS[@]} -gt 0 ]]; then
    TARBALL="analyses/round31_phase0p5_sweep_results_${STAMP}.tar.gz"
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    PACK_TARGETS=()
    for cid in "${DONE_COMBO_IDS[@]}"; do
        OUT_TAG="_phase0p5_${cid}"
        DIAG_DIR_ROOT="analyses/round31_stage1_downstream_diag${OUT_TAG}"
        SUB_DIR_ROOT="analyses/round31_stage1_substitute_conds${OUT_TAG}"
        LOG_DIR="runs/round31_stage1_downstream${OUT_TAG}"
        [[ -d "${DIAG_DIR_ROOT}" ]] && PACK_TARGETS+=("${DIAG_DIR_ROOT}")
        [[ -d "${SUB_DIR_ROOT}" ]] && PACK_TARGETS+=("${SUB_DIR_ROOT}")
        [[ -d "${LOG_DIR}" ]] && PACK_TARGETS+=("${LOG_DIR}")
        [[ -f "${OVERALL_LOG_DIR}/${cid}.log" ]] && PACK_TARGETS+=("${OVERALL_LOG_DIR}/${cid}.log")
    done
    [[ -f "${SUMMARY_MD}" ]] && PACK_TARGETS+=("${SUMMARY_MD}")
    [[ -f "${SUMMARY_LOG}" ]] && PACK_TARGETS+=("${SUMMARY_LOG}")

    if [[ ${#PACK_TARGETS[@]} -gt 0 ]]; then
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE=$(du -h "${TARBALL}" | cut -f1)
        log "wrote ${TARBALL}  (${SIZE})"
    else
        log "[P05 PACK] nothing to pack."
    fi
fi

log
log "================================================================"
log "[$(date '+%F %T')] R31 Phase 0.5 sweep COMPLETE"
log "================================================================"
log "Combos done: ${DONE_COMBO_IDS[*]:-none}"
log "Summary log : ${SUMMARY_LOG}"
log "Summary MD  : ${SUMMARY_MD}"
if [[ ${DRY_RUN} -eq 0 && ${#DONE_COMBO_IDS[@]} -gt 0 ]]; then
    log "Tarball     : ${TARBALL:-<none>}"
fi
