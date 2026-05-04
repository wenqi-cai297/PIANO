"""Run the full v13_centered extraction validation in one shot.

To be invoked once the 4-subset re-extraction (v13_centered) is complete.
Produces:

  1. Per-class survey: v13_centered vs v12_strict
     -> analyses/2026-05-05_v13_vs_v12/
  2. Per-class survey: v13_centered vs v11
     -> analyses/2026-05-05_v13_vs_v11/
  3. Per-part global positive rates for v13_centered
     -> analyses/2026-05-05_v13_positive_rates.json + .md

Pass criteria (printed at the end):

  - neuraldome/box: v13 hand contact ≥ 0.30 (was 0.000 v12)
  - neuraldome/case: v13 hand contact ≥ 0.20 (was 0.000 v12)
  - neuraldome/trolleycase: v13 hand contact ≥ 0.30 (was 0.000 v12)
  - neuraldome/tennis: v13 hand contact ≥ 0.20 (was 0.057 v12)
  - chairs/chair: v13 hand contact within ±0.05 of v12 (was 0.400 v12, must
    not regress)
  - omomo/suitcase, plasticbox, smallbox, largebox: hand contact within ±0.05
    of v12 (these meshes are already centered; should be near-identical)
  - foot positive rate (any-foot, weighted): ≥ 0.045 (was 0.045 v12 — small
    increase expected, severe drop signals chair foot regression worse than
    expected)

Usage:
    python scripts/stage1_pseudo_labels/validate_v13_extraction.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PIANO_ROOT = Path("E:/Project/Datasets/InterAct/piano")
SUBSETS = ("imhd", "neuraldome", "omomo_correct_v2", "chairs")
OUT_BASE = Path("analyses")


def _per_part_positive_rate(label_subdir: str) -> dict:
    """Frame-weighted positive rate for v13_centered across all 4 subsets."""
    parts = ("l_hand", "r_hand", "l_foot", "r_foot", "pelvis")
    n_clips = 0
    tot = {p: 0 for p in parts}
    pos = {p: 0 for p in parts}
    for subset in SUBSETS:
        d = PIANO_ROOT / subset / "pseudo_labels" / label_subdir
        files = list(d.glob("*.npz"))
        for f in files:
            try:
                arr = np.load(f, allow_pickle=False)
                if "contact_state" not in arr.files:
                    continue
                cs = arr["contact_state"]
                for i, p in enumerate(parts):
                    tot[p] += cs.shape[0]
                    pos[p] += int(cs[:, i].sum())
                n_clips += 1
            except Exception:
                pass
    rates = {p: (pos[p] / tot[p]) if tot[p] > 0 else 0.0 for p in parts}
    rates["any_hand"] = (pos["l_hand"] + pos["r_hand"]) / (2 * tot["l_hand"]) if tot["l_hand"] > 0 else 0.0
    rates["any_foot"] = (pos["l_foot"] + pos["r_foot"]) / (2 * tot["l_foot"]) if tot["l_foot"] > 0 else 0.0
    return {"label": label_subdir, "n_clips": n_clips, "rates": rates,
            "totals": tot, "positives": pos}


def _expected_pass_check(by_class_csv: Path, label_subdir: str) -> list[tuple[str, bool, str]]:
    """Verify per-class predictions (the cross-walk table from the diagnostic
    doc).

    Returns list of (check_name, passed, message) tuples.
    """
    import csv
    by_class: dict[tuple[str, str], dict] = {}
    with open(by_class_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_class[(row["subset"], row["keyword"])] = row
    # Note: in the survey script columns labeled v12_* hold label-A (here =
    # v13_centered), columns labeled v11_* hold label-B (here = v12_strict).
    checks = [
        # (subset, keyword, min_v13_hand, label, comment)
        ("neuraldome", "box",         0.30, "Mode A primary fix", "was 0.000 v12"),
        ("neuraldome", "case",        0.20, "Mode A primary fix", "was 0.000 v12"),
        ("neuraldome", "trolleycase", 0.30, "Mode A primary fix", "was 0.000 v12"),
        ("neuraldome", "tennis",      0.20, "Mode B (neuraldome side)", "was 0.057 v12"),
    ]
    results: list[tuple[str, bool, str]] = []
    for subset, kw, min_hand, label, ctx in checks:
        row = by_class.get((subset, kw))
        if row is None:
            results.append((f"{subset}/{kw}", False, f"MISSING: no class row"))
            continue
        v13_hand = float(row["v12_any_hand_frac"])  # v12_* col holds label-A in our setup
        passed = v13_hand >= min_hand
        results.append((
            f"{subset}/{kw} ({label})",
            passed,
            f"v13_hand={v13_hand:.3f} (need ≥{min_hand}, {ctx})",
        ))

    # Stability checks (must not regress on stable classes):
    stable_checks = [
        ("chairs", "chair", 0.400, 0.05, "biggest single class, 1479 clips"),
        ("omomo_correct_v2", "suitcase", 0.453, 0.05, "control, mesh already centered"),
        ("omomo_correct_v2", "plasticbox", 0.401, 0.05, "control, mesh already centered"),
        ("omomo_correct_v2", "smallbox", 0.324, 0.05, "control"),
        ("omomo_correct_v2", "largebox", 0.365, 0.05, "control"),
    ]
    for subset, kw, v12_baseline, tol, ctx in stable_checks:
        row = by_class.get((subset, kw))
        if row is None:
            results.append((f"{subset}/{kw}", False, "MISSING"))
            continue
        v13_hand = float(row["v12_any_hand_frac"])
        diff = abs(v13_hand - v12_baseline)
        passed = diff <= tol
        results.append((
            f"{subset}/{kw} (stable)",
            passed,
            f"v13_hand={v13_hand:.3f} vs v12 baseline {v12_baseline:.3f}, |Δ|={diff:.3f} (need ≤{tol}, {ctx})",
        ))

    return results


def main() -> int:
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # ---- Survey 1: v13 vs v12 ----
    out_v13_v12 = OUT_BASE / "2026-05-05_v13_vs_v12"
    print(f"\n[1/3] Surveying v13_centered vs v12_strict -> {out_v13_v12}")
    res = subprocess.run([
        sys.executable, "scripts/stage1_pseudo_labels/survey_label_quality_per_class.py",
        "--label-a", "v13_centered",
        "--label-b", "v12_strict",
        "--output-dir", str(out_v13_v12),
    ], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"FAILED: {res.stderr}")
        return 1

    # ---- Survey 2: v13 vs v11 ----
    out_v13_v11 = OUT_BASE / "2026-05-05_v13_vs_v11"
    print(f"\n[2/3] Surveying v13_centered vs v11 -> {out_v13_v11}")
    res = subprocess.run([
        sys.executable, "scripts/stage1_pseudo_labels/survey_label_quality_per_class.py",
        "--label-a", "v13_centered",
        "--label-b", "v11",
        "--output-dir", str(out_v13_v11),
    ], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"FAILED: {res.stderr}")
        return 1

    # ---- Per-part positive rates ----
    print("\n[3/3] Computing per-part positive rates ...")
    rates_v12 = _per_part_positive_rate("v12_strict")
    rates_v13 = _per_part_positive_rate("v13_centered")
    rates_path = OUT_BASE / "2026-05-05_v13_positive_rates.json"
    rates_path.write_text(json.dumps(
        {"v12_strict": rates_v12, "v13_centered": rates_v13}, indent=2,
    ), encoding="utf-8")

    print(f"\nv12_strict   ({rates_v12['n_clips']} clips):")
    for p, r in rates_v12["rates"].items():
        print(f"  {p}: {r:.4f}")
    print(f"\nv13_centered ({rates_v13['n_clips']} clips):")
    for p, r in rates_v13["rates"].items():
        print(f"  {p}: {r:.4f}")

    # ---- Pass / fail checks ----
    print("\n=== Validation checks ===")
    checks = _expected_pass_check(out_v13_v12 / "by_class.csv", "v13_centered")
    n_pass = sum(1 for _, ok, _ in checks if ok)
    n_total = len(checks)
    for name, ok, msg in checks:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}: {msg}")
    print(f"\n{n_pass}/{n_total} checks passed.")
    return 0 if n_pass == n_total else 2


if __name__ == "__main__":
    sys.exit(main())
