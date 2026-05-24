#!/usr/bin/env bash
# Round-26 temporal diagnostics — pack into a single tarball.
# Small text + JSON output (~few MB).

set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="round26_temporal_diag_${STAMP}.tar.gz"

ITEMS=()

for dir in \
    analyses/round26_sustained_contact_v27_final \
    analyses/round26_sustained_contact_r23 \
    analyses/round26_sustained_contact_gt_reference \
    analyses/round26_gait_v27_final \
    analyses/round26_gait_r23 \
    analyses/round26_gait_gt_reference ; do
    if [[ -d "${dir}" ]]; then
        for f in "${dir}"/*.json "${dir}"/*.md ; do
            [[ -e "$f" ]] && ITEMS+=("$f")
        done
    else
        echo "  [skip] missing: ${dir}"
    fi
done

for f in runs/round26_temporal_diag/*.log ; do
    [[ -e "$f" ]] && ITEMS+=("$f")
done

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "ERROR: no temporal diag outputs found."
    exit 1
fi

echo "Packing ${#ITEMS[@]} files into ${OUT} ..."
tar czf "${OUT}" "${ITEMS[@]}"

SIZE=$(du -h "${OUT}" | cut -f1)
ABS=$(readlink -f "${OUT}")
echo
echo "================================================================"
echo "Tarball ready: ${OUT}  (${SIZE})"
echo "Absolute path: ${ABS}"
echo
echo "File inventory:"
tar tzf "${OUT}" | sed 's/^/  /'
echo
echo "================================================================"
echo "Pull to local:"
echo "  scp gpu-server-1@<host>:${ABS} 'E:\\Project\\2026-04-13\\'"
echo "Unpack:"
echo "  tar xzf ${OUT}"
echo "================================================================"
