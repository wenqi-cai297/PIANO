"""One-shot comparison: Round-10 vs Round-12 yaw unwrapped ranges per clip.

Confirms the rot6d helper fix produces sign-flipped yaw ranges (since the
bug was a pure sign flip on the angle channel). Same magnitude, same
absolute range, mirrored signs.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    r10 = json.loads(Path("analyses/2026-05-20_coarse_representation_extraction_audit.json").read_text("utf-8"))
    r12 = json.loads(Path("analyses/2026-05-22_coarse_representation_extraction_audit_round12.json").read_text("utf-8"))
    print(f"Round-10 clips: {r10['n_clips']}   Round-12 clips: {r12['n_clips']}")
    n = min(len(r10["per_clip"]), len(r12["per_clip"]))
    print(f"Comparing first {n} clips:")
    print("  subset    seq_id            r10_yaw_range  r12_yaw_range  r10_lo/hi          r12_lo/hi          sign_flip?")
    flips = 0
    matches = 0
    for c10, c12 in list(zip(r10["per_clip"], r12["per_clip"]))[:n]:
        lo10, hi10 = c10["yaw_unwrapped_min_max"]
        lo12, hi12 = c12["yaw_unwrapped_min_max"]
        # Sign flip means: lo12 ≈ -hi10 and hi12 ≈ -lo10 (so [a,b] -> [-b,-a]).
        sign_flip = (abs(lo12 + hi10) < 0.05 and abs(hi12 + lo10) < 0.05)
        same = (abs(lo12 - lo10) < 0.05 and abs(hi12 - hi10) < 0.05)
        tag = "FLIP" if sign_flip else ("SAME" if same else "OTHER")
        if sign_flip:
            flips += 1
        if same:
            matches += 1
        print(
            f"  {c10['subset']:8s}  {c10['seq_id'][:18]:18s}  "
            f"{hi10 - lo10:+7.3f}        {hi12 - lo12:+7.3f}        "
            f"[{lo10:+6.3f},{hi10:+6.3f}]  [{lo12:+6.3f},{hi12:+6.3f}]  {tag}"
        )
    print(f"\nTotal sign-flipped: {flips}/{n}   identical: {matches}/{n}")


if __name__ == "__main__":
    main()
