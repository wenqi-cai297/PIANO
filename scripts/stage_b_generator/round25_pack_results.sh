#!/usr/bin/env bash
# Round-25 P0 diagnostic — pack all analysis-relevant files into a
# single tarball for transfer to the local machine.
#
# Includes (small, ~MB total):
#   - analyses/round25_*.{json,md}        — D1/D2/D3 outputs + curated subsets
#   - runs/round25_diagnostic/*.log       — D2/D3/D4/D5 per-stage logs
#   - runs/training/stageB_anchordiff_v26_d4_*/{metrics.jsonl,*.log}
#   - runs/training/stageB_anchordiff_v26_d5_*/{metrics.jsonl,*.log}
#
# Excludes (large, not needed on local):
#   - *.pt checkpoints (~600 MB each × ~15 files = 9 GB)
#   - wandb/ subdirs (synced to cloud already)
#
# Usage:
#   bash scripts/stage_b_generator/round25_pack_results.sh
#
# Output:
#   round25_results_YYYY-MM-DD_HHMM.tar.gz at project root.
#   Final line prints the absolute path + a sample scp/rsync command.

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round25_results_${STAMP}.tar.gz"

# Build the list of files. Use shell glob expansion; tar
# --ignore-failed-read makes missing globs non-fatal.
ITEMS=()

# Analyses outputs (D1 candidates, curate, D4 indices, D2/D3 stats).
for f in \
    analyses/round25_multimodal_candidates_val.json \
    analyses/round25_multimodal_candidates_train.json \
    analyses/round25_multimodal_eval_subset.json \
    analyses/round25_d4_train_selection.json \
    analyses/round25_d4_indices_8.json \
    analyses/round25_d4_indices_16.json \
    analyses/round25_d2_diversity_stats.json \
    analyses/round25_d2_diversity_stats.md \
    analyses/round25_d3_oracle_vs_sampled.json \
    analyses/round25_d3_oracle_vs_sampled.md \
    analyses/round25_curation_summary.md ; do
    [[ -e "$f" ]] && ITEMS+=("$f") || echo "  [skip] missing: $f"
done

# Stage logs (D2/D3/D4/D5 progress + errors).
for f in runs/round25_diagnostic/*.log ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

# D4 / D5 training metrics + log files (NOT .pt checkpoints).
for run_dir in \
    runs/training/stageB_anchordiff_v26_d4_overfit8 \
    runs/training/stageB_anchordiff_v26_d4_overfit16 \
    runs/training/stageB_anchordiff_v26_d5_v0_baseline \
    runs/training/stageB_anchordiff_v26_d5_v1_hand2x_foot2x \
    runs/training/stageB_anchordiff_v26_d5_v2_hand5x_foot5x ; do
    if [[ -d "$run_dir" ]]; then
        # Add metrics.jsonl + any non-pt files (config snapshot, train log
        # if separately written). Exclude wandb/ and *.pt.
        for f in "$run_dir"/metrics.jsonl "$run_dir"/*.log "$run_dir"/*.txt "$run_dir"/*.yaml ; do
            [[ -e "$f" ]] && ITEMS+=("$f")
        done
    else
        echo "  [skip] missing run dir: $run_dir"
    fi
done

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no Round-25 result files found. Did the diagnostic actually run?"
    exit 1
fi

echo
echo "Packing ${#ITEMS[@]} files into ${OUT} ..."
tar czf "${OUT}" "${ITEMS[@]}"

SIZE_HUMAN=$(du -h "${OUT}" | cut -f1)
ABS_PATH="$(readlink -f "${OUT}")"

echo
echo "================================================================"
echo "Tarball ready: ${OUT}  (${SIZE_HUMAN})"
echo "Absolute path: ${ABS_PATH}"
echo
echo "File inventory:"
tar tzf "${OUT}" | sed 's/^/  /'
echo
echo "================================================================"
echo "To pull to local Windows (one of these, depending on your tooling):"
echo
echo "  # via scp + Git Bash / WSL on Windows:"
echo "  scp gpu-server-1@<host>:${ABS_PATH} 'E:\\Project\\2026-04-13\\'"
echo
echo "  # via rsync:"
echo "  rsync -avz gpu-server-1@<host>:${ABS_PATH} 'E:/Project/2026-04-13/'"
echo
echo "  # then on local Windows (PowerShell or Git Bash from project root):"
echo "  tar xzf ${OUT}"
echo "================================================================"
