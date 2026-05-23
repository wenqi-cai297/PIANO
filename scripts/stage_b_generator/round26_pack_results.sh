#!/usr/bin/env bash
# Round-26 v27 motion-faithful Stage-2 train+eval — pack all analysis-
# relevant files into a single tarball for transfer to the local machine.
#
# Includes (small, ~10 MB total):
#   - analyses/round26_v27_d2_diversity_{best_val,final}.{json,md}
#   - analyses/round26_v27_d3_oracle_vs_sampled_{best_val,final}.{json,md}
#   - analyses/2026-05-24_round26_v27_motion_faithful_patch.md
#   - analyses/2026-05-24_claude_round26_stage2_strategy_review.md
#   - runs/round26_v27_train/*.log                                — stage logs
#   - runs/training/stageB_anchordiff_v27_*/metrics.jsonl + *.log + *.txt + *.yaml
#
# Excludes (large, not needed on local):
#   - *.pt checkpoints (~600 MB × ~10 ckpts = 6 GB)
#   - wandb/ subdirs (synced to cloud already if wandb was on)
#
# Usage:
#   bash scripts/stage_b_generator/round26_pack_results.sh
#
# Output:
#   round26_results_YYYY-MM-DD_HHMM.tar.gz at project root.
#   Final line prints the absolute path + a sample scp/rsync command.

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round26_results_${STAMP}.tar.gz"

ITEMS=()

# Eval JSONs/MDs (D2, D3 on best_val + final ckpts).
for f in \
    analyses/round26_v27_d2_diversity_best_val.json \
    analyses/round26_v27_d2_diversity_best_val.md \
    analyses/round26_v27_d2_diversity_final.json \
    analyses/round26_v27_d2_diversity_final.md \
    analyses/round26_v27_d3_oracle_vs_sampled_best_val.json \
    analyses/round26_v27_d3_oracle_vs_sampled_best_val.md \
    analyses/round26_v27_d3_oracle_vs_sampled_final.json \
    analyses/round26_v27_d3_oracle_vs_sampled_final.md ; do
    if [[ -e "$f" ]]; then
        ITEMS+=("$f")
    else
        echo "  [skip] missing: $f"
    fi
done

# Design + review docs (so the local has the full context after extraction).
for f in \
    analyses/2026-05-24_round26_v27_motion_faithful_patch.md \
    analyses/2026-05-24_claude_round26_stage2_strategy_review.md \
    analyses/2026-05-24_round25_reopen_training_strategy.md \
    analyses/2026-05-24_stage2_training_strategy_review.md ; do
    [[ -e "$f" ]] && ITEMS+=("$f") || echo "  [skip] missing: $f"
done

# Stage logs (PREP / TRAIN / EVAL progress + errors).
for f in runs/round26_v27_train/*.log ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

# v27 training run dir (metrics.jsonl, train log, config snapshot, NO .pt).
RUN_DIR="runs/training/stageB_anchordiff_v27_stage2_anchoraware_FULL_DATA"
if [[ -d "${RUN_DIR}" ]]; then
    for f in "${RUN_DIR}"/metrics.jsonl \
             "${RUN_DIR}"/*.log \
             "${RUN_DIR}"/*.txt \
             "${RUN_DIR}"/*.yaml \
             "${RUN_DIR}"/*.json ; do
        # Exclude any accidental .pt by name pattern just in case the glob
        # ever included one — the glob above never matches *.pt anyway.
        [[ -e "$f" && "$f" != *.pt ]] && ITEMS+=("$f")
    done
else
    echo "  [skip] missing v27 run dir: ${RUN_DIR}"
fi

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no Round-26 result files found. Did the training + eval actually run?"
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
echo "To pull to local Windows:"
echo
echo "  # via scp (Git Bash / WSL on Windows):"
echo "  scp gpu-server-1@<host>:${ABS_PATH} 'E:\\Project\\2026-04-13\\'"
echo
echo "  # via rsync:"
echo "  rsync -avz gpu-server-1@<host>:${ABS_PATH} 'E:/Project/2026-04-13/'"
echo
echo "  # then unpack on local (PowerShell or Git Bash from project root):"
echo "  tar xzf ${OUT}"
echo "================================================================"
