#!/usr/bin/env bash
# Round-29 next-step ablation — A1 contact-heavy non-chairs visual review.
#
# Per the user feedback 2026-05-28 evening: the standard A1 vis tarball
# was dominated by chairs sit-down clips where contact is trivial (hand on
# armrest). To judge A1's hand-object contact quality we need clips from
# imhd / neuraldome / omomo_correct_v2 with hand_contact_frac >= 0.8 and
# mode_category=manipulation.
#
# This launcher uses the same renderer + ckpts as the main A1 vis launcher
# but points at a different selection JSON and writes to a separate output
# directory so the two tarballs can be unpacked side-by-side without
# clobbering each other.
#
# Default render set (A1 vs the two anchors):
#   A1  r29_ns_a1_c41_s4_g1                    this round's winner
#   R0  r29_ft_r0_clean_a3_baseline            original baseline
#   G1  r29_nb_g1_phasefree_gait_fixed         G1 reference
#
# Default N_CLIPS=18 (all 18 selected clips: 6 imhd + 6 neuraldome + 6 omomo).
#
# Wall-clock: 3 ckpts × 18 clips × ~30 s/clip = ~9 min/ckpt; dual-GPU pairwise
# total ~15-20 min. Single-GPU sequential ~30-35 min.
#
# Prerequisites on the Linux server:
#   1. git pull (must include this script)
#   2. ffmpeg in the piano conda env (preflight will check):
#        conda install -y -n piano -c conda-forge ffmpeg
#   3. Ckpts present (same as the chairs vis launcher):
#        runs/training/stageB_anchordiff_r29_ns_a1_c41_s4_g1/final.pt
#        runs/training/stageB_anchordiff_r29_ft_r0_clean_a3_baseline/final.pt
#        runs/training/stageB_anchordiff_r29_nb_g1_phasefree_gait_fixed/final.pt
#   4. The launcher will auto-build the contact-heavy selection JSON from
#      the existing train 48-clip JSON. To use the val 48-clip JSON as the
#      source instead, pass:
#        SOURCE_SEL=analyses/round29_val_diag_indices_48_balanced.json \
#          bash scripts/stage_b_generator/run_round29_next_step_a1_contact_heavy_vis.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="runs/round29_next_step_a1_contact_heavy_vis"
mkdir -p "${LOG_DIR}"

# ---------- env overrides ----------
SOURCE_SEL="${SOURCE_SEL:-analyses/round27_tier0_train_indices_48_balanced.json}"
SEL_JSON="${SEL_JSON:-analyses/round29_contact_heavy_nonchairs_selection.json}"
N_CLIPS="${ROUND29_NS_VIS_N_CLIPS:-18}"
SINGLE_GPU="${ROUND29_NS_VIS_SINGLE_GPU:-0}"
# Source-JSON bucket — defaults to train since the rich-metadata
# selection is the train one. Pass SOURCE_SEL=... + ROUND29_NS_VIS_BUCKET=val
# to switch to val (val builder must be on disk too).
BUCKET="${ROUND29_NS_VIS_BUCKET:-train}"
CKPT_NAME="${ROUND29_NS_VIS_CKPT_NAME:-final.pt}"
MIN_HAND_CONTACT="${ROUND29_NS_VIS_MIN_HAND_CONTACT:-0.8}"

DEFAULT_VARIANTS=(
    r29_ns_a1_c41_s4_g1
    r29_ft_r0_clean_a3_baseline
    r29_nb_g1_phasefree_gait_fixed
)
if [[ -n "${ROUND29_NS_VIS_VARIANTS:-}" ]]; then
    # shellcheck disable=SC2206
    VARIANTS=(${ROUND29_NS_VIS_VARIANTS})
else
    VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

# ---------- preflight ----------
PREFLIGHT_FAIL=0

# ffmpeg.
if ! conda run --no-capture-output -n piano python -c "import shutil,sys; sys.exit(0 if shutil.which('ffmpeg') else 1)" 2>/dev/null; then
    echo "[A1-CH-VIS PREFLIGHT FAIL] ffmpeg not on PATH inside the piano conda env."
    echo "    Install ffmpeg first:"
    echo "        conda install -y -n piano -c conda-forge ffmpeg"
    PREFLIGHT_FAIL=1
fi

# Configs + ckpts.
for V in "${VARIANTS[@]}"; do
    CFG="configs/training/anchordiff_${V}.yaml"
    CKPT="runs/training/stageB_anchordiff_${V}/${CKPT_NAME}"
    if [[ ! -e "${CFG}" ]]; then
        echo "[A1-CH-VIS PREFLIGHT FAIL] missing config: ${CFG}"
        PREFLIGHT_FAIL=1
    fi
    if [[ ! -e "${CKPT}" ]]; then
        echo "[A1-CH-VIS PREFLIGHT FAIL] missing ckpt: ${CKPT}"
        PREFLIGHT_FAIL=1
    fi
done

# Source selection JSON must exist (has the mode_category + hand_contact_frac
# metadata that the builder filters on).
if [[ ! -e "${SOURCE_SEL}" ]]; then
    echo "[A1-CH-VIS PREFLIGHT FAIL] source selection JSON missing: ${SOURCE_SEL}"
    if [[ "${SOURCE_SEL}" == *round29_val_diag_indices_48_balanced.json ]]; then
        echo "    -> rebuild via:"
        echo "       python scripts/stage_b_generator/round29_build_val_diag_subset.py \\"
        echo "           --config configs/training/anchordiff_r29_ns_a1_c41_s4_g1.yaml"
    fi
    PREFLIGHT_FAIL=1
fi

if [[ ${PREFLIGHT_FAIL} -ne 0 ]]; then
    echo "[A1-CH-VIS] FATAL preflight failures."
    exit 1
fi

# ---------- build contact-heavy selection ----------
echo "[A1-CH-VIS] Building contact-heavy non-chairs selection from ${SOURCE_SEL}..."
python scripts/stage_b_generator/round29_build_contact_heavy_selection.py \
    --source "${SOURCE_SEL}" \
    --output "${SEL_JSON}" \
    --min-hand-contact "${MIN_HAND_CONTACT}"

SEL_N=$(python -c "import json; d=json.load(open('${SEL_JSON}')); print(d.get('n_clips') or d.get('n_found') or len(d.get('selected') or d.get('clips') or []))")
echo "[A1-CH-VIS] Selection: ${SEL_N} clips; rendering first ${N_CLIPS} from ${BUCKET} bucket"
echo "[A1-CH-VIS] Variants:  ${VARIANTS[*]}"
echo "[A1-CH-VIS] Ckpt name: ${CKPT_NAME}"

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

OUT_ROOT="analyses/round29_next_step_a1_contact_heavy_vis"
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
        if (( i % 2 == 0 )); then
            for j in 0 1; do
                wait "${PIDS[j]}" || echo "  ${NAMES[j]} exited non-zero"
            done
            PIDS=()
            NAMES=()
        fi
    done
    for k in "${!PIDS[@]}"; do
        wait "${PIDS[k]}" || echo "  ${NAMES[k]} exited non-zero"
    done
fi

# ---------- pack ----------
STAMP=$(date +%Y%m%d_%H%M%S)
TARBALL="round29_next_step_a1_contact_heavy_vis_${STAMP}.tar.gz"
echo
echo "================================================================"
echo "[$(date '+%F %T')] PACK -> ${TARBALL}"
echo "================================================================"
tar -czf "${TARBALL}" -C analyses "$(basename "${OUT_ROOT}")"
SIZE=$(du -h "${TARBALL}" | cut -f1)
echo "wrote ${TARBALL}  (${SIZE})"

echo
echo "================================================================"
echo "Round-29 next-step A1 contact-heavy visual review complete."
echo "Outputs:"
for V in "${VARIANTS[@]}"; do
    echo "  ${OUT_ROOT}/${V}/clip*_{gt,pred}.mp4 + summary.md"
done
echo "Logs:    ${LOG_DIR}/render_*.log"
echo "Tarball: ${TARBALL}"
echo
echo "scp back:  scp <server>:$(pwd)/${TARBALL} ."
echo "Then on local:  tar -xzf ${TARBALL} -C analyses/"
echo "================================================================"
