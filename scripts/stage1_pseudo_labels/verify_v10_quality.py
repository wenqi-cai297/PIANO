"""Aggregate-stats + integrity checks for a freshly-extracted pseudo-
label run (v10 specifically — checks the new ``contact_target_xyz_gt``
field and the wider RELEASE phase distribution that the v10 extractor
introduced).

Reports per subset:

    - Field presence: confirm ``contact_target_xyz_gt`` is in every npz.
    - xyz GT sanity: shape, dtype, finite, magnitude (typical 0.05-0.5 m).
    - Phase distribution: counts + % per class. v10 should show
      ``release`` at ~25-35% (was ~12% in v9 because of the 10-frame cap).
    - Support distribution.
    - Contact rate per body part.
    - Cleaning report status (kept / dropped / per-reason counts).
    - Comparison vs v9 expectation table at the end.

Usage (server):
    python scripts/stage1_pseudo_labels/verify_v10_quality.py \\
        --piano-root /media/gpu-server-1/4TB_for_data/Cai/datasets/InterAct/piano \\
        --subsets chairs imhd neuraldome omomo_correct_v2

Run after ``rerun_pseudo_labels_interact.sh`` + ``clean_pseudo_labels.py``
to validate the v10 extraction worked correctly before committing to a
re-train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASE_NAMES = ("approach", "pre_contact", "stable_contact", "manipulation", "release")
SUPPORT_NAMES = ("both_feet", "single_foot", "sitting", "hand_support")
BODY_PART_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis")

# v9 → v10 expected delta (rough). Use to flag "looks weird" cases.
V9_PHASE_PCT_TYPICAL = {
    "approach": (15.0, 25.0),
    "pre_contact": (0.2, 1.0),
    "stable_contact": (12.0, 22.0),
    "manipulation": (45.0, 60.0),
    "release": (10.0, 14.0),  # v9 was here; v10 should be > 20
}
V10_PHASE_PCT_EXPECTED = {
    "release": (20.0, 40.0),  # widened by extractor
}


def _summarise_subset(piano_root: Path, subset: str) -> dict:
    pl_dir = piano_root / subset / "pseudo_labels"
    if not pl_dir.exists():
        return {"subset": subset, "error": f"missing pseudo_labels dir at {pl_dir}"}

    npz_files = sorted(pl_dir.glob("*.npz"))
    if not npz_files:
        return {"subset": subset, "error": "no npz files"}

    n = len(npz_files)
    has_xyz_gt = 0
    xyz_finite = 0
    xyz_magnitudes: list[float] = []   # mean per-clip L2 norm
    phase_counts = np.zeros(len(PHASE_NAMES), dtype=np.int64)
    support_counts = np.zeros(len(SUPPORT_NAMES), dtype=np.int64)
    contact_pos_counts = np.zeros(len(BODY_PART_NAMES), dtype=np.int64)
    contact_total = 0
    bad_files: list[str] = []

    for path in npz_files:
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:  # pragma: no cover — corrupt npz
            bad_files.append(f"{path.name}: load failed ({e})")
            continue

        if "contact_target_xyz_gt" in data.files:
            has_xyz_gt += 1
            xyz = data["contact_target_xyz_gt"]                          # (T, 5, 3)
            if np.isfinite(xyz).all():
                xyz_finite += 1
            else:
                bad_files.append(f"{path.name}: contact_target_xyz_gt has NaN/Inf")
            xyz_magnitudes.append(float(np.linalg.norm(xyz, axis=-1).mean()))

        if "phase" in data.files:
            ph = data["phase"]
            phase_counts += np.bincount(ph, minlength=len(PHASE_NAMES))[:len(PHASE_NAMES)]

        if "support" in data.files:
            su = data["support"]
            support_counts += np.bincount(su, minlength=len(SUPPORT_NAMES))[:len(SUPPORT_NAMES)]

        if "contact_state" in data.files:
            cs = data["contact_state"]                                   # (T, 5)
            contact_pos_counts += (cs > 0.5).sum(axis=0)
            contact_total += cs.shape[0]

    phase_pct = (phase_counts / max(phase_counts.sum(), 1)) * 100.0
    support_pct = (support_counts / max(support_counts.sum(), 1)) * 100.0
    contact_pct = (contact_pos_counts / max(contact_total, 1)) * 100.0

    cr_path = piano_root / subset / "cleaning_report.json"
    cr = json.loads(cr_path.read_text()) if cr_path.exists() else None

    return {
        "subset": subset,
        "n_npz": n,
        "has_xyz_gt": has_xyz_gt,
        "xyz_finite": xyz_finite,
        "xyz_mean_l2_per_clip_avg_m": (
            float(np.mean(xyz_magnitudes)) if xyz_magnitudes else None
        ),
        "phase_pct": dict(zip(PHASE_NAMES, [float(x) for x in phase_pct])),
        "support_pct": dict(zip(SUPPORT_NAMES, [float(x) for x in support_pct])),
        "contact_pct_per_part": dict(zip(BODY_PART_NAMES, [float(x) for x in contact_pct])),
        "cleaning": cr,
        "bad_files_sample": bad_files[:5],
        "n_bad_files": len(bad_files),
    }


def _print_subset(r: dict) -> None:
    print()
    print("=" * 78)
    print(f"Subset: {r['subset']}")
    print("=" * 78)
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return

    print(f"  npz files:                    {r['n_npz']}")
    if r['n_npz']:
        coverage = 100.0 * r['has_xyz_gt'] / r['n_npz']
        finite = 100.0 * r['xyz_finite'] / max(r['has_xyz_gt'], 1)
        tag = "" if coverage > 99.9 else "  [WARN — re-extract may not have completed]"
        print(f"  contact_target_xyz_gt present: {r['has_xyz_gt']} ({coverage:.1f}%){tag}")
        print(f"  xyz finite check:             {r['xyz_finite']} / {r['has_xyz_gt']} ({finite:.1f}%)")
        if r['xyz_mean_l2_per_clip_avg_m'] is not None:
            mag_m = r['xyz_mean_l2_per_clip_avg_m']
            mag_tag = ("" if 0.05 < mag_m < 1.0 else
                       "  [WARN — magnitude outside typical 5-100 cm range]")
            print(f"  xyz mean magnitude (object-local): {mag_m*100:.1f} cm{mag_tag}")
        if r['n_bad_files']:
            print(f"  bad files:                    {r['n_bad_files']}")
            for s in r['bad_files_sample']:
                print(f"    - {s}")

    print()
    print("  Phase distribution:")
    for name, pct in r['phase_pct'].items():
        lo, hi = V9_PHASE_PCT_TYPICAL.get(name, (None, None))
        v10 = V10_PHASE_PCT_EXPECTED.get(name)
        tag = ""
        if v10 is not None:
            if not (v10[0] <= pct <= v10[1]):
                tag = f"  [WARN — v10 expected {v10[0]:.0f}-{v10[1]:.0f}%]"
        elif lo is not None and not (lo <= pct <= hi):
            tag = f"  [WARN — v9-typical {lo:.0f}-{hi:.0f}%]"
        print(f"    {name:<16s} {pct:>6.2f}%{tag}")

    print("  Support distribution:")
    for name, pct in r['support_pct'].items():
        print(f"    {name:<16s} {pct:>6.2f}%")

    print("  Contact rate per body part (frames > 0.5):")
    for name, pct in r['contact_pct_per_part'].items():
        print(f"    {name:<16s} {pct:>6.2f}%")

    if r['cleaning']:
        c = r['cleaning']
        print()
        print("  Cleaning report:")
        n_in = c.get("num_in_metadata", c.get("total"))
        kept = c.get("num_kept", c.get("kept"))
        dropped = c.get("num_dropped", c.get("dropped"))
        print(f"    in:      {n_in}")
        print(f"    kept:    {kept}  ({100.0*kept/max(n_in,1):.1f}%)")
        print(f"    dropped: {dropped}")
        reasons = c.get("reason_counts", c.get("per_reason", {}))
        if reasons:
            for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print(f"      {k:<32s} {v}")
    else:
        print("  (no cleaning_report.json found — run clean_pseudo_labels.py)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--piano-root", type=Path, required=True,
        help="root containing one dir per subset (chairs/, imhd/, ...)",
    )
    parser.add_argument(
        "--subsets", nargs="+",
        default=["chairs", "imhd", "neuraldome", "omomo_correct_v2"],
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional JSON output path (default: <piano-root>/v10_quality.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.piano_root / "v10_quality.json"

    all_summaries = []
    for sub in args.subsets:
        r = _summarise_subset(args.piano_root, sub)
        _print_subset(r)
        all_summaries.append(r)

    # Aggregate
    print()
    print("=" * 78)
    print("Cross-subset summary (the most-watched signals)")
    print("=" * 78)
    print(f"{'subset':<18}  {'n_npz':>7}  {'%xyz_gt':>8}  {'%release':>9}  {'%sitting':>9}  {'cleaned_kept%':>15}")
    for r in all_summaries:
        if "error" in r:
            print(f"  {r['subset']:<16s}  ERROR")
            continue
        xyz_cov = 100.0 * r['has_xyz_gt'] / max(r['n_npz'], 1)
        release_pct = r['phase_pct'].get('release', 0)
        sitting_pct = r['support_pct'].get('sitting', 0)
        kept_pct = (100.0 * r['cleaning']['num_kept'] /
                    max(r['cleaning']['num_in_metadata'], 1)) if r.get('cleaning') else float('nan')
        print(f"  {r['subset']:<16s}  {r['n_npz']:>7d}  {xyz_cov:>7.1f}%  {release_pct:>8.1f}%  {sitting_pct:>8.1f}%  {kept_pct:>13.1f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
