#!/usr/bin/env bash
# Round-29 next-step ablation — A1 visual review launcher.
#
# Per the user request 2026-05-28: A1 (r29_ns_a1_c41_s4_g1) is the best
# variant from the next-step matrix on numeric metrics but its G1 soft-
# stance diag is "degenerate" (low_alt_amplitude_rate=0.627,
# low_transition_rate=0.559). Hard gates pass; soft gates fail. Visual
# review is the only way to see what hard-pass / soft-fail actually
# looks like before designing the next-round losses.
#
# Default render set (A1 vs the two anchors):
#   A1  r29_ns_a1_c41_s4_g1                    THIS round's winner on contact + gait
#   R0  r29_ft_r0_clean_a3_baseline            original baseline (no G1 losses)
#   G1  r29_nb_g1_phasefree_gait_fixed         G1 reference (G1 losses, no S4-consume)
#
# Why this triplet:
#   - A1 vs R0  → does adding G1 losses + S4-consume visibly improve walking?
#   - A1 vs G1  → does S4-consume produce visibly different gait from
#                 phase-free G1 without S4-consume?
#   - 3 ckpts × N_CLIPS clips lets the viewer see the same scene three
#     ways side by side.
#
# DEFAULT N_CLIPS=16. Override via env. Wall-clock estimates (~30s/clip
# per ckpt on 1 GPU):
#   N_CLIPS=16, dual-GPU pairwise:        ~10-15 min total
#   N_CLIPS=16, single-GPU sequential:    ~25 min total
#
# Prerequisites on the Linux server:
#   1. git pull (must include this script + the R29-NS configs)
#   2. export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   3. ROUND29_NS_REGEN_CONFIGS=1 was passed when r29_ns_* configs were
#      first generated, OR re-run the generator now:
#        python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py
#      (uses DATASETS_ROOT)
#   4. Ckpts present at:
#        runs/training/stageB_anchordiff_r29_ns_a1_c41_s4_g1/final.pt
#        runs/training/stageB_anchordiff_r29_ft_r0_clean_a3_baseline/final.pt
#        runs/training/stageB_anchordiff_r29_nb_g1_phasefree_gait_fixed/final.pt
#   5. Val 48-clip balanced selection JSON:
#        analyses/round29_val_diag_indices_48_balanced.json
#
# Usage:
#   bash scripts/stage_b_generator/run_round29_next_step_a1_visual_review.sh
#
#   # Render only 8 clips (faster):
#   ROUND29_NS_VIS_N_CLIPS=8 bash scripts/stage_b_generator/run_round29_next_step_a1_visual_review.sh
#
#   # Single-GPU fallback (cuda:0, sequential):
#   ROUND29_NS_VIS_SINGLE_GPU=1 bash scripts/stage_b_generator/run_round29_next_step_a1_visual_review.sh
#
#   # Train subset instead of val:
#   ROUND29_NS_VIS_BUCKET=train \
#       SEL_JSON=analyses/round27_tier0_train_indices_48_balanced.json \
#       bash scripts/stage_b_generator/run_round29_next_step_a1_visual_review.sh
#
#   # Render only A1 (skip R0/G1 anchors — faster, but no side-by-side):
#   ROUND29_NS_VIS_VARIANTS="r29_ns_a1_c41_s4_g1" \
#       bash scripts/stage_b_generator/run_round29_next_step_a1_visual_review.sh
#
#   # Pick a different default ckpt filename (best_val.pt instead of final.pt):
#   ROUND29_NS_VIS_CKPT_NAME=best_val.pt \
#       bash scripts/stage_b_generator/run_round29_next_step_a1_visual_review.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round29_next_step_a1_visual_review"
mkdir -p "${LOG_DIR}"

# ---------- env overrides ----------
SEL_JSON="${SEL_JSON:-analyses/round29_val_diag_indices_48_balanced.json}"
N_CLIPS="${ROUND29_NS_VIS_N_CLIPS:-16}"
SINGLE_GPU="${ROUND29_NS_VIS_SINGLE_GPU:-0}"
BUCKET="${ROUND29_NS_VIS_BUCKET:-val}"
CKPT_NAME="${ROUND29_NS_VIS_CKPT_NAME:-final.pt}"

# Default render set: A1 + R0 anchor + G1 anchor.
DEFAULT_VARIANTS=(
    r29_ns_a1_c41_s4_g1                  # this round's best
    r29_ft_r0_clean_a3_baseline          # original baseline anchor (no G1 losses)
    r29_nb_g1_phasefree_gait_fixed       # G1 anchor (G1 losses, no S4 consume)
)
if [[ -n "${ROUND29_NS_VIS_VARIANTS:-}" ]]; then
    # shellcheck disable=SC2206
    VARIANTS=(${ROUND29_NS_VIS_VARIANTS})
else
    VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

# ---------- selection JSON ----------
if [[ ! -e "${SEL_JSON}" ]]; then
    if [[ "${BUCKET}" == "val" && "${SEL_JSON}" == *round29_val_diag_indices_48_balanced.json ]]; then
        echo "Selection JSON missing; rebuilding via round29_build_val_diag_subset.py"
        # Use A1's config — it has the same val split as everything else.
        python scripts/stage_b_generator/round29_build_val_diag_subset.py \
            --config "configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml"
    else
        echo "ERROR: missing selection JSON: ${SEL_JSON}"
        echo "  For val:   python scripts/stage_b_generator/round29_build_val_diag_subset.py \\"
        echo "                 --config configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml"
        echo "  For train: use analyses/round27_tier0_train_indices_48_balanced.json"
        exit 1
    fi
fi

# ---------- preflight ----------
PREFLIGHT_FAIL=0

# ffmpeg must be in the piano conda env. Without it the renderer falls
# back to a PIL GIF writer that silently writes 196 identical frames
# (the final pose) for matplotlib 3D scatter, producing a "video" that
# does not show motion. Catch this here so the launch does not waste
# 10-25 minutes producing useless output.
if ! conda run --no-capture-output -n piano python -c "import shutil,sys; sys.exit(0 if shutil.which('ffmpeg') else 1)" 2>/dev/null; then
    echo "[A1-VIS PREFLIGHT FAIL] ffmpeg not on PATH inside the piano conda env."
    echo "    The PIL GIF fallback silently produces a static 'video' for 3D"
    echo "    scatter animations. Install ffmpeg first:"
    echo "        conda install -y -n piano -c conda-forge ffmpeg"
    echo "    (or set PIANO_ALLOW_BROKEN_GIF_FALLBACK=1 to opt back into the"
    echo "     historical broken behaviour for a code-path smoke test.)"
    PREFLIGHT_FAIL=1
fi

for V in "${VARIANTS[@]}"; do
    CFG="configs/training/anchordiff_${V}.yaml"
    CKPT="runs/training/stageB_anchordiff_${V}/${CKPT_NAME}"
    if [[ ! -e "${CFG}" ]]; then
        echo "[A1-VIS PREFLIGHT FAIL] missing config: ${CFG}"
        if [[ "${V}" == r29_ns_* ]]; then
            echo "    -> regenerate via:"
            echo "       export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4"
            echo "       python scripts/stage_b_generator/round29_make_next_step_ablation_configs.py"
        elif [[ "${V}" == r29_nb_* ]]; then
            echo "    -> regenerate via:"
            echo "       python scripts/stage_b_generator/round29_make_next_ablation_configs.py"
        elif [[ "${V}" == r29_ft_* ]]; then
            echo "    -> regenerate via:"
            echo "       python scripts/stage_b_generator/round29_make_failure_targeted_ablation_configs.py"
        fi
        PREFLIGHT_FAIL=1
    fi
    if [[ ! -e "${CKPT}" ]]; then
        echo "[A1-VIS PREFLIGHT FAIL] missing ckpt: ${CKPT}"
        PREFLIGHT_FAIL=1
    fi
done
if [[ ${PREFLIGHT_FAIL} -ne 0 ]]; then
    echo "[A1-VIS] FATAL preflight failures."
    exit 1
fi

SEL_N=$(python -c "import json; d=json.load(open('${SEL_JSON}')); print(d.get('n_clips') or d.get('n_found') or len(d.get('selected') or d.get('clips') or []))")
echo "[A1-VIS] Selection: ${SEL_N} clips available; rendering first ${N_CLIPS} from ${BUCKET} bucket"
echo "[A1-VIS] Variants:  ${VARIANTS[*]}"
echo "[A1-VIS] Ckpt name: ${CKPT_NAME}"

# ---------- helpers ----------
run_step_bg() {
    local NAME="$1"; shift
    local GPU="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    {
        echo "================================================================"
        echo "[$(date '+%F %T')] BG-START ${NAME} on cuda:${GPU}"
        echo "    log: ${LOG}"
        echo "================================================================"
        env CUDA_VISIBLE_DEVICES="${GPU}" \
            PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1
        local T1
        T1=$(date +%s)
        echo "[$(date '+%F %T')] BG-DONE ${NAME} in $((T1-T0))s"
    } > "${LOG}" 2>&1
}

run_step_fg() {
    local NAME="$1"; shift
    local GPU="$1"; shift
    local LOG="${LOG_DIR}/${NAME}.log"
    local T0
    T0=$(date +%s)
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${NAME} on cuda:${GPU}"
    echo "    log: ${LOG}"
    echo "================================================================"
    env CUDA_VISIBLE_DEVICES="${GPU}" \
        PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$@" 2>&1 | tee "${LOG}"
    local T1
    T1=$(date +%s)
    echo "[$(date '+%F %T')] DONE ${NAME} in $((T1-T0))s"
}

render_one() {
    local NAME="$1"; local GPU="$2"; local CFG="$3"; local CKPT="$4"; local OUT="$5"; local MODE="$6"
    local RUN_FN="run_step_fg"
    if [[ "${MODE}" == "bg" ]]; then RUN_FN="run_step_bg"; fi
    mkdir -p "${OUT}"
    "${RUN_FN}" "render_${NAME}" "${GPU}" \
        conda run --no-capture-output -n piano python -u \
            scripts/stage_b_generator/render_round24_visual_review.py \
            --config "${CFG}" --ckpt "${CKPT}" --output-dir "${OUT}" \
            --bucket "${BUCKET}" --n-clips "${N_CLIPS}" \
            --selection-json "${SEL_JSON}" \
            --cfg-scale 1.0 --seed 42
}

OUT_ROOT="analyses/round29_next_step_a1_visual_review"
mkdir -p "${OUT_ROOT}"

# ---------- render ----------
if [[ "${SINGLE_GPU}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] SINGLE-GPU RENDER (cuda:0, sequential ${#VARIANTS[@]} ckpts)"
    echo "================================================================"
    for V in "${VARIANTS[@]}"; do
        CFG="configs/training/anchordiff_${V}.yaml"
        CKPT="runs/training/stageB_anchordiff_${V}/${CKPT_NAME}"
        OUT="${OUT_ROOT}/${V}"
        render_one "${V}" 0 "${CFG}" "${CKPT}" "${OUT}" fg
    done
else
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] DUAL-GPU RENDER: pairwise (cuda:0 || cuda:1)"
    echo "================================================================"
    i=0
    PIDS=()
    NAMES=()
    for V in "${VARIANTS[@]}"; do
        CFG="configs/training/anchordiff_${V}.yaml"
        CKPT="runs/training/stageB_anchordiff_${V}/${CKPT_NAME}"
        OUT="${OUT_ROOT}/${V}"
        GPU=$((i % 2))
        render_one "${V}" "${GPU}" "${CFG}" "${CKPT}" "${OUT}" bg &
        PIDS+=($!)
        NAMES+=("${V}")
        i=$((i + 1))
        # Drain every pair so we don't run more than 2 concurrent renders.
        if (( i % 2 == 0 )); then
            for j in 0 1; do
                wait "${PIDS[j]}" || echo "  ${NAMES[j]} exited non-zero"
            done
            PIDS=()
            NAMES=()
        fi
    done
    # Drain any leftover (odd number of variants).
    for k in "${!PIDS[@]}"; do
        wait "${PIDS[k]}" || echo "  ${NAMES[k]} exited non-zero"
    done
fi

# ---------- pack ----------
STAMP=$(date +%Y%m%d_%H%M%S)
TARBALL="round29_next_step_a1_visual_review_${STAMP}.tar.gz"
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
echo "================================================================"
tar -czf "${TARBALL}" -C analyses "$(basename "${OUT_ROOT}")"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "wrote ${TARBALL}  (${SIZE})"

echo
echo "================================================================"
echo "Round-29 next-step A1 visual review complete."
echo "Outputs:"
for V in "${VARIANTS[@]}"; do
    echo "  ${OUT_ROOT}/${V}/clip*_{gt,pred}.{mp4,gif} + summary.md"
done
echo "Logs:    ${LOG_DIR}/render_*.log"
echo "Tarball: ${TARBALL}"
echo
echo "scp back:  scp <server>:$(pwd)/${TARBALL} ."
echo "Then on local:  tar -xzf ${TARBALL} -C analyses/"
echo "================================================================"
