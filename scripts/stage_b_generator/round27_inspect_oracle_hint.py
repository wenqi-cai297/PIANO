"""Smoke test for Tier-0A oracle interaction hint (Round-27, Commit 1).

Loads a few clips through ``HOIDataset`` with
``use_oracle_interaction_hint=True`` and prints per-clip + aggregate
statistics for each hint variant ({"hand", "foot", "full"}).

Run after editing ``src/piano/data/interaction_hint.py`` or the
``HOIDataset`` plumbing to confirm:

1. The hint tensor materialises with the expected shape ``(T, D)``.
2. Hand contact fraction is non-trivially nonzero on contact-heavy
   clips (hand contact ≥ ~10% of frames on lifting / carrying actions).
3. Foot stance fraction is high (~80%+) on walking / sitting clips.
4. Object-local hand offset norm sits in a sensible range
   (most contact frames: |r| < 0.5 m after clamp to 1.0).
5. No NaN / Inf.

Usage
-----
::

    conda run --no-capture-output -n piano python \
        scripts/stage_b_generator/round27_inspect_oracle_hint.py \
        --root E:/Project/Datasets/InterAct/piano_official_process_4/chairs \
        --pseudo-label-subdir pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker \
        --n-clips 10 --variant full
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _print_hint_stats(
    hint: np.ndarray,
    variant: str,
    seq_len: int,
    seq_id: str,
) -> None:
    """Print per-clip stats restricted to the first ``seq_len`` frames."""
    h = hint[:seq_len]
    print(f"\n--- {seq_id}  (T={seq_len}, D={h.shape[1]}, variant={variant}) ---")

    if variant in {"hand", "full"}:
        hand_contact = h[:, :2]
        rel_offset = h[:, 2:8].reshape(seq_len, 2, 3)
        for hand_i, hand_name in enumerate(("L_hand", "R_hand")):
            frac = float(hand_contact[:, hand_i].mean())
            contact_frames = hand_contact[:, hand_i] > 0.5
            if contact_frames.any():
                offset_norm = np.linalg.norm(
                    rel_offset[contact_frames, hand_i], axis=-1,
                )
                print(
                    f"  {hand_name}: contact_frac={frac:.3f}  "
                    f"offset_norm(contact frames)={offset_norm.mean():.3f}±"
                    f"{offset_norm.std():.3f}  "
                    f"max={offset_norm.max():.3f}"
                )
            else:
                print(f"  {hand_name}: contact_frac={frac:.3f}  (no contact)")

    if variant in {"foot", "full"}:
        foot_off = 0 if variant == "foot" else 8
        foot_stance = h[:, foot_off:foot_off + 2]
        ankle_h = h[:, foot_off + 2:foot_off + 4]
        walk = h[:, foot_off + 4:foot_off + 5]
        for foot_i, foot_name in enumerate(("L_foot", "R_foot")):
            stance_frac = float((foot_stance[:, foot_i] > 0.5).mean())
            print(
                f"  {foot_name}: stance_frac(>0.5)={stance_frac:.3f}  "
                f"stance_mean={float(foot_stance[:, foot_i].mean()):.3f}  "
                f"ankle_h_norm_mean={float(ankle_h[:, foot_i].mean()):.3f}"
            )
        print(
            f"  walking_frac={float(walk.mean()):.3f}  "
            f"({int(walk.sum())}/{seq_len} frames)"
        )

    print(
        f"  hint stats: mean={h.mean():.4f}  std={h.std():.4f}  "
        f"min={h.min():.4f}  max={h.max():.4f}  "
        f"finite={'all' if np.isfinite(h).all() else 'NOT ALL'}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to a single subset root, e.g. .../piano_official_process_4/chairs",
    )
    p.add_argument(
        "--pseudo-label-subdir",
        type=str,
        default="pseudo_labels/v18_h10_f05_pelvis20_official_semantic_marker",
    )
    p.add_argument(
        "--variant", choices=("hand", "foot", "full"), default="full",
    )
    p.add_argument("--n-clips", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seq-length", type=int, default=196)
    p.add_argument("--fps", type=float, default=20.0)
    args = p.parse_args()

    # Defer torch import until after argparse so --help is fast.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    import torch  # noqa: F401  (needed by HOIDataset)
    from piano.data.dataset import HOIDataset
    from piano.data.interaction_hint import hint_dim

    root = Path(args.root)
    pseudo_dir = root / args.pseudo_label_subdir
    if not pseudo_dir.exists():
        print(
            f"[warn] pseudo-label dir {pseudo_dir} does not exist — using default",
            file=sys.stderr,
        )

    ds = HOIDataset(
        root=root,
        pseudo_label_dir=pseudo_dir if pseudo_dir.exists() else None,
        max_seq_length=int(args.max_seq_length),
        motion_representation="smpl_pose_135_plan",
        surface_obj_pose=True,
        use_oracle_interaction_hint=True,
        oracle_hint_variant=args.variant,
        oracle_hint_fps=float(args.fps),
    )

    expected_dim = hint_dim(args.variant)
    print(
        f"[inspect_oracle_hint] root={root.name}  variant={args.variant}  "
        f"D={expected_dim}  dataset_size={len(ds)}"
    )

    rng = np.random.default_rng(int(args.seed))
    n = min(int(args.n_clips), len(ds))
    idxs = rng.choice(len(ds), size=n, replace=False)

    all_hand_contact_fracs: list[float] = []
    all_foot_stance_fracs: list[float] = []
    all_walking_fracs: list[float] = []
    all_offset_norms: list[float] = []

    for idx in idxs:
        sample = ds[int(idx)]
        if "oracle_interaction_hint" not in sample:
            print(
                "[error] sample missing 'oracle_interaction_hint' — check "
                "that contact_state pseudo-label exists for this clip.",
                file=sys.stderr,
            )
            continue
        hint = sample["oracle_interaction_hint"].numpy()
        seq_len = int(sample["seq_len"].item())
        seq_id = sample.get("seq_id", f"idx_{idx}")

        assert hint.shape == (int(args.max_seq_length), expected_dim), (
            f"unexpected hint shape {hint.shape} for variant={args.variant}"
        )
        _print_hint_stats(hint, args.variant, seq_len, str(seq_id))

        h = hint[:seq_len]
        if args.variant in {"hand", "full"}:
            all_hand_contact_fracs.append(float(h[:, :2].mean()))
            rel = h[:, 2:8].reshape(seq_len, 2, 3)
            contact = h[:, :2] > 0.5
            for hand_i in range(2):
                if contact[:, hand_i].any():
                    all_offset_norms.extend(
                        float(x) for x in np.linalg.norm(
                            rel[contact[:, hand_i], hand_i], axis=-1,
                        )
                    )
        if args.variant in {"foot", "full"}:
            foot_off = 0 if args.variant == "foot" else 8
            stance = h[:, foot_off:foot_off + 2]
            walk = h[:, foot_off + 4:foot_off + 5]
            all_foot_stance_fracs.append(float((stance > 0.5).mean()))
            all_walking_fracs.append(float(walk.mean()))

    print("\n=== Aggregate across sampled clips ===")
    if all_hand_contact_fracs:
        a = np.array(all_hand_contact_fracs)
        print(
            f"hand_contact_frac: mean={a.mean():.3f}  std={a.std():.3f}  "
            f"min={a.min():.3f}  max={a.max():.3f}  n={len(a)}"
        )
    if all_offset_norms:
        a = np.array(all_offset_norms)
        print(
            f"object_local_offset_norm (contact frames): mean={a.mean():.3f}  "
            f"std={a.std():.3f}  p50={np.median(a):.3f}  "
            f"p95={np.quantile(a, 0.95):.3f}  max={a.max():.3f}  n={len(a)}"
        )
    if all_foot_stance_fracs:
        a = np.array(all_foot_stance_fracs)
        print(
            f"foot_stance_frac(>0.5): mean={a.mean():.3f}  std={a.std():.3f}  "
            f"min={a.min():.3f}  max={a.max():.3f}  n={len(a)}"
        )
    if all_walking_fracs:
        a = np.array(all_walking_fracs)
        print(
            f"walking_frac:       mean={a.mean():.3f}  std={a.std():.3f}  "
            f"min={a.min():.3f}  max={a.max():.3f}  n={len(a)}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
