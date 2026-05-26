#!/usr/bin/env bash
# Round-29 loss-strategy full-data status check (safe to run anytime).
#
# Reports, for each of the 4 variants:
#   - latest epoch checkpointed (epoch_N.pt)
#   - whether final.pt exists
#   - last 3 metrics.jsonl entries (epoch + total loss + val_loss if present)
#   - whether training process is still running (pgrep accelerate)
#
# Read-only. Does NOT touch ckpts or write any files. Safe to run
# while training is in progress (no locks, no I/O contention).
#
# Usage:
#   bash scripts/stage_b_generator/round29_loss_strategy_full_data_status.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

VARIANTS=(
    r29_lsf_a2_baseline_from_scratch
    r29_lsf_a3_baseline_from_scratch
    r29_lsf_a2_anchor2_mixed
    r29_lsf_a3_anchor2_mixed
)

# Try to find python so the metrics.jsonl parser works even if PY is unset.
if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python"
    elif command -v python3 >/dev/null 2>&1; then PY="python3"
    else PY=""
    fi
fi

echo "================================================================"
echo "[$(date '+%F %T')] R29 loss-strategy full-data status"
echo "================================================================"

# 1) Running processes (accelerate or python train_anchordiff).
if command -v pgrep >/dev/null 2>&1; then
    n_train=$(pgrep -f "train_anchordiff.py" 2>/dev/null | wc -l)
    n_accel=$(pgrep -f "accelerate launch" 2>/dev/null | wc -l)
    echo "Running:  ${n_train} train_anchordiff process(es), ${n_accel} accelerate launcher(s)"
fi

# 2) GPU status (one-shot, no watch).
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{printf "GPU %s: util %s%% mem %s/%s MB\n", $1, $2, $3, $4}'
fi

echo

# 3) Per-variant ckpt + metrics.
for v in "${VARIANTS[@]}"; do
    outdir="runs/training/stageB_anchordiff_${v}"
    echo "----- ${v}"
    if [[ ! -d "${outdir}" ]]; then
        echo "    (no output dir yet — not started)"
        continue
    fi

    # Latest periodic ckpt.
    latest=$(ls "${outdir}"/epoch_*.pt 2>/dev/null | sort -V | tail -n 1)
    if [[ -n "${latest}" ]]; then
        echo "    latest ckpt:  $(basename "${latest}") ($(stat -c '%y' "${latest}" 2>/dev/null | cut -d. -f1))"
    else
        echo "    latest ckpt:  (none yet — < epoch 10)"
    fi

    # final.pt + best_val.pt.
    for f in final.pt best_val.pt; do
        if [[ -e "${outdir}/${f}" ]]; then
            echo "    ${f}: ✓ ($(stat -c '%y' "${outdir}/${f}" 2>/dev/null | cut -d. -f1))"
        fi
    done

    # Latest metrics.jsonl entries (epoch_summary if present, else last 3 lines).
    mjsonl="${outdir}/metrics.jsonl"
    if [[ -e "${mjsonl}" && -n "${PY}" ]]; then
        size=$(stat -c '%s' "${mjsonl}" 2>/dev/null || echo 0)
        nlines=$(wc -l < "${mjsonl}" 2>/dev/null || echo 0)
        echo "    metrics:      ${nlines} lines, ${size} bytes"
        # Pull the most recent epoch summary if any.
        "${PY}" -c "
import json, sys
try:
    lines = open('${mjsonl}').readlines()
except Exception:
    sys.exit(0)
# Walk backwards looking for the most recent 'epoch_summary' or 'val' or 'complete' event.
for ln in reversed(lines):
    try:
        d = json.loads(ln)
    except Exception:
        continue
    ev = d.get('event', '')
    if ev in ('epoch_summary', 'val', 'complete', 'contact'):
        keys = ('event', 'epoch', 'global_step', 'loss', 'mse_x0',
                'loss_anchor_joint_pos', 'val_loss', 'val_loss_anchor_joint_pos',
                'train_wallclock_seconds', 'train_started_at')
        kept = {k: d.get(k) for k in keys if k in d}
        print('    last ' + ev + ':', kept)
        break
" 2>/dev/null || true
    fi

    # Recent wallclock from start time if training is still going.
    log="runs/round29_loss_strategy_full_data/${v}.log"
    if [[ -e "${log}" ]]; then
        first_epoch=$(grep -m 1 "epoch 1 |" "${log}" 2>/dev/null | head -n 1)
        if [[ -n "${first_epoch}" ]]; then
            # Best-effort: report when epoch 1 finished (helps estimate wallclock per epoch).
            echo "    first epoch line: ${first_epoch:0:120}..."
        fi
    fi
done

echo
echo "----- Diag outputs (post-training)"
n_done=$(find analyses -maxdepth 1 -type d -name "round29_r29_lsf_*_diag_*" 2>/dev/null \
    | xargs -I{} sh -c 'ls "{}"/*_stats.json 2>/dev/null | head -n 1' 2>/dev/null \
    | wc -l)
echo "    diag stats present: ${n_done}/24"

if [[ -e analyses/2026-05-27_round29_loss_strategy_full_data_report.md ]]; then
    echo "    summarizer report: ✓ (last modified $(stat -c '%y' analyses/2026-05-27_round29_loss_strategy_full_data_report.md | cut -d. -f1))"
fi

# Most recent tarball.
tarball=$(ls -t analyses/round29_loss_strategy_full_data_results_*.tar.gz 2>/dev/null | head -n 1)
if [[ -n "${tarball}" ]]; then
    echo "    latest tarball: ${tarball} ($(stat -c '%y' "${tarball}" | cut -d. -f1))"
fi
