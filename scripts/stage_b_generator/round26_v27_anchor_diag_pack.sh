#!/usr/bin/env bash
# Round-26 v27 per-part anchor diagnostic — pack outputs into a single
# tarball for transfer to local.
#
# Includes (~few MB):
#   - analyses/round26_v27_anchor_diag_final/{anchor_stats.json, anchor_summary.md, *.png}
#   - analyses/round26_v27_anchor_diag_best_val/...
#   - analyses/round26_r23_anchor_diag_48clips/...
#   - runs/round26_v27_anchor_diag/*.log
#
# Usage:
#   bash scripts/stage_b_generator/round26_v27_anchor_diag_pack.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round26_v27_anchor_diag_${STAMP}.tar.gz"

ITEMS=()

for diag_dir in \
    analyses/round26_v27_anchor_diag_final \
    analyses/round26_v27_anchor_diag_best_val \
    analyses/round26_r23_anchor_diag_48clips ; do
    if [[ -d "${diag_dir}" ]]; then
        for f in "${diag_dir}"/anchor_stats.json \
                 "${diag_dir}"/anchor_summary.md \
                 "${diag_dir}"/*.png ; do
            [[ -e "$f" ]] && ITEMS+=("$f")
        done
    else
        echo "  [skip] missing diag dir: ${diag_dir}"
    fi
done

for f in runs/round26_v27_anchor_diag/*.log ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no Round-26 anchor diag outputs found."
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
echo "Pull to local:"
echo "  scp gpu-server-1@<host>:${ABS_PATH} 'E:\\Project\\2026-04-13\\'"
echo "Unpack on local (PowerShell or Git Bash from project root):"
echo "  tar xzf ${OUT}"
echo "================================================================"
