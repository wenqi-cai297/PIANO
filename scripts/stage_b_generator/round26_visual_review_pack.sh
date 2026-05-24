#!/usr/bin/env bash
# Round-26 visual review — pack outputs into a single tarball.
#
# Includes (~200-400 MB, depending on video size):
#   - analyses/round26_visual_review/r23_baseline/clip*_{gt,pred}.mp4 + summary.md
#   - analyses/round26_visual_review/v27_final/clip*_{gt,pred}.mp4    + summary.md
#   - analyses/round26_visual_review_selection.json
#   - analyses/round26_air_grab_analysis.{json,md}
#   - runs/round26_visual_review/*.log
#
# Usage:
#   bash scripts/stage_b_generator/round26_visual_review_pack.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round26_visual_review_${STAMP}.tar.gz"

ITEMS=()

# Video output dirs
for dir in \
    analyses/round26_visual_review/r23_baseline \
    analyses/round26_visual_review/v27_final ; do
    if [[ -d "${dir}" ]]; then
        for f in "${dir}"/*.mp4 "${dir}"/summary.md ; do
            [[ -e "$f" ]] && ITEMS+=("$f")
        done
    else
        echo "  [skip] missing render dir: ${dir}"
    fi
done

# Selection + analysis docs (small; convenient to bundle)
for f in \
    analyses/round26_visual_review_selection.json \
    analyses/round26_air_grab_analysis.json \
    analyses/round26_air_grab_analysis.md ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

# Stage logs
for f in runs/round26_visual_review/*.log ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no Round-26 visual review outputs found."
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
echo "Unpack locally (PowerShell or Git Bash from project root):"
echo "  tar xzf ${OUT}"
echo "================================================================"
