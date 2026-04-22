"""Probe ``_detect_mesh_up_axis`` across every InterAct object mesh.

Driven by the v5 regression: chairs sitting 49.6% → 39.5% and imhd
sitting 0.66% → 3.05% (false positive) after auto-detect up-axis landed
in ``edf2bb3``. The hypothesis is that face-area argmax silently picks
a wrong axis when the mesh has a prominent backrest/side-panel (chairs)
or is elongated without a real "up" (imhd bats, brooms). This probe
enumerates all 106 InterAct objects, reports the detected +axis along
with a dominance ratio and the raw per-axis surface areas, so we can
see exactly which meshes regressed and how marginal each pick is.

Output: JSON per-object + a console summary of non-+Y detections.

Usage on the server:
    python scripts/data/probe_mesh_up_axis.py \\
        --interact-dir /media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/InterAct \\
        --output runs/checks/up_axis_probe/$(date +%Y-%m-%d_%H%M%S)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh

from piano.data.pseudo_labels.extract_support import _detect_mesh_up_axis


SUBSETS = ("chairs", "imhd", "neuraldome", "omomo_correct_v2")
AXIS_NAMES = ("+X", "+Y", "+Z")


def probe_one(mesh_path: Path, threshold: float = 0.7) -> dict:
    mesh = trimesh.load(mesh_path, force="mesh")
    fa = mesh.area_faces
    fn = mesh.face_normals

    pos_areas = np.array([
        float(fa[fn[:, axis] > threshold].sum()) for axis in range(3)
    ])
    neg_areas = np.array([
        float(fa[fn[:, axis] < -threshold].sum()) for axis in range(3)
    ])

    sorted_pos = np.sort(pos_areas)[::-1]
    dom_ratio = float(sorted_pos[0] / max(sorted_pos[1], 1e-6))

    detected = _detect_mesh_up_axis(mesh, threshold=threshold)
    detected_axis = int(np.argmax(detected))

    return {
        "extents": mesh.extents.tolist(),
        "num_faces": int(len(fn)),
        "total_area": float(fa.sum()),
        "pos_area": {AXIS_NAMES[i]: float(pos_areas[i]) for i in range(3)},
        "neg_area": {AXIS_NAMES[i].replace("+", "-"): float(neg_areas[i]) for i in range(3)},
        "detected_up_axis": AXIS_NAMES[detected_axis],
        "dominance_ratio": dom_ratio,
    }


def _find_mesh(obj_dir: Path) -> Path | None:
    """InterAct ships ``<obj>/<obj>.obj`` plus optional simplified variants."""
    canonical = obj_dir / f"{obj_dir.name}.obj"
    if canonical.exists():
        return canonical
    candidates = sorted(obj_dir.glob("*.obj"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--interact-dir", type=Path, required=True,
        help="InterAct root directory (contains chairs/, imhd/, ...)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory (will be created)",
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    per_subset: dict[str, dict[str, dict]] = {}
    count_by_axis: dict[str, dict[str, int]] = {}

    for subset in SUBSETS:
        subset_dir = args.interact_dir / subset / "objects"
        if not subset_dir.exists():
            print(f"[skip] {subset}: {subset_dir} not found")
            continue

        objs: dict[str, dict] = {}
        axis_counts = {a: 0 for a in AXIS_NAMES}
        for obj_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            mesh_path = _find_mesh(obj_dir)
            if mesh_path is None:
                continue
            try:
                info = probe_one(mesh_path, threshold=args.threshold)
            except Exception as e:
                print(f"  [warn] {subset}/{obj_dir.name}: {e}")
                continue
            objs[obj_dir.name] = info
            axis_counts[info["detected_up_axis"]] += 1

        per_subset[subset] = objs
        count_by_axis[subset] = axis_counts
        print(f"[{subset}] {len(objs)} objects  {axis_counts}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "threshold": args.threshold,
        "count_by_axis": count_by_axis,
        "per_subset": per_subset,
    }
    out_json = args.output / "probe.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nJSON: {out_json}")

    # Highlight non-+Y detections — those are the suspicious ones.
    print("\n--- non-Y-up detections (potential regressors) ---")
    any_found = False
    for subset, objs in per_subset.items():
        for name, info in objs.items():
            if info["detected_up_axis"] != "+Y":
                any_found = True
                pos = info["pos_area"]
                print(
                    f"  [{subset}] {name:30s} "
                    f"detected={info['detected_up_axis']}  "
                    f"dom={info['dominance_ratio']:.2f}  "
                    f"extents={[round(x, 2) for x in info['extents']]}  "
                    f"pos=+X:{pos['+X']:.1f} +Y:{pos['+Y']:.1f} +Z:{pos['+Z']:.1f}"
                )
    if not any_found:
        print("  (none — every object was picked as +Y)")

    # Also show marginal +Y picks — dominance < 1.5 means the auto-detect
    # is effectively guessing, even when it picks the "right" axis.
    print("\n--- marginal +Y detections (dominance < 1.5) ---")
    marginal = False
    for subset, objs in per_subset.items():
        for name, info in objs.items():
            if info["detected_up_axis"] == "+Y" and info["dominance_ratio"] < 1.5:
                marginal = True
                pos = info["pos_area"]
                print(
                    f"  [{subset}] {name:30s} "
                    f"dom={info['dominance_ratio']:.2f}  "
                    f"pos=+X:{pos['+X']:.1f} +Y:{pos['+Y']:.1f} +Z:{pos['+Z']:.1f}"
                )
    if not marginal:
        print("  (none)")


if __name__ == "__main__":
    main()
