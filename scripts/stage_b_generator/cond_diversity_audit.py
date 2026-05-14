"""cond_diversity_audit.py — audit whether GT z_int and plan actually carry
discriminative signal across text-similar clips.

Hypothesis being tested (per user 2026-05-11):
> "If clips with the same text description have nearly identical GT z_int
> and plan, the model fundamentally cannot discriminate them — even a
> perfect model would produce 'average HOI motion' regardless. This would
> explain why the model produces frozen, semantically-clustered motion."

Methodology:
1. Load the same N=10 train subsample the training uses.
2. For every clip, record:
   - text caption
   - per-frame z_int components (contact_state, contact_target_xyz, phase, support)
   - plan summaries (anchor count, body parts active, target_world spread)
3. Group clips by exact text match. Report cluster sizes.
4. For the largest text-clusters, compute INTRA-cluster pairwise diffs of
   z_int and plan, vs INTER-cluster pairwise diffs as a baseline.
5. Surface concrete clip-id examples + their z_int/plan summaries side by side.

Result: writes a markdown report to analyses/2026-05-11_cond_diversity_audit.md
with quantitative comparisons + per-cluster sample triples.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_condition_diagnostics import _build_dataset  # type: ignore[import-not-found]
from piano.data.dataset import collate_hoi


def _z_int_summary(batch: dict, t_idx: int = None) -> dict:
    """Per-clip summary of z_int components.

    Returns dict with:
      - contact_state_active_frac per part : (5,) — fraction of frames where contact_state > 0.5
      - contact_target_norm: scalar — average L2 norm of contact_target_xyz across frames
      - phase_dist: (3,) — fraction of frames in each phase class
      - support_dist: (3,) — fraction of frames in each support class
      - num_valid_frames: int
    """
    seq_len = int(batch["seq_len"].item())
    cs = batch["contact_state"][0, :seq_len].numpy()       # (T, 5)
    ctx = batch["contact_target_xyz"][0, :seq_len].numpy() # (T, 5, 3)
    phase = batch["phase"][0, :seq_len].numpy()            # (T,) int
    support = batch["support"][0, :seq_len].numpy()        # (T,) int

    return {
        "contact_state_active_frac": (cs > 0.5).mean(axis=0).tolist(),  # 5-vec
        "contact_target_norm_mean": float(np.linalg.norm(ctx, axis=-1).mean()),
        "phase_dist": [
            float((phase == 0).mean()),
            float((phase == 1).mean()),
            float((phase == 2).mean()),
        ],
        "support_dist": [
            float((support == 0).mean()),
            float((support == 1).mean()),
            float((support == 2).mean()),
        ],
        "num_valid_frames": seq_len,
    }


def _plan_summary(batch: dict) -> dict:
    """Per-clip summary of the InteractionPlan."""
    anchor_mask = batch["plan_anchor_mask"][0].bool().numpy()  # (K,)
    n_anchors = int(anchor_mask.sum())
    anchor_time = batch["plan_anchor_time"][0].numpy()  # (K,)
    anchor_part = batch["plan_anchor_part"][0].numpy()  # (K, P)
    anchor_target_world = batch["plan_anchor_target_world"][0].numpy()  # (K, P, 3)
    anchor_phase = batch["plan_anchor_phase"][0].numpy()  # (K,)
    anchor_support = batch["plan_anchor_support"][0].numpy()  # (K,)

    valid_times = anchor_time[anchor_mask]
    valid_parts = anchor_part[anchor_mask]
    valid_targets = anchor_target_world[anchor_mask]
    valid_phases = anchor_phase[anchor_mask]
    valid_supports = anchor_support[anchor_mask]

    # Dominant body part per anchor: argmax of anchor_part
    if n_anchors > 0:
        dom_parts = valid_parts.argmax(axis=-1).tolist()
        # active part one-hot fingerprint (which body parts ever get an anchor)
        any_active = (valid_parts > 0).any(axis=0).tolist()
        # mean target_world across all (anchor, part) where part is active
        active_mask = valid_parts > 0  # (n, P)
        if active_mask.any():
            target_centroid = (
                (valid_targets * active_mask[..., None]).sum(axis=(0, 1))
                / max(active_mask.sum(), 1)
            ).tolist()
            target_spread = float(
                np.linalg.norm(
                    valid_targets[active_mask] - np.array(target_centroid),
                    axis=-1,
                ).mean()
            )
        else:
            target_centroid = [0.0, 0.0, 0.0]
            target_spread = 0.0
    else:
        dom_parts = []
        any_active = [False] * 5
        target_centroid = [0.0, 0.0, 0.0]
        target_spread = 0.0

    return {
        "n_anchors": n_anchors,
        "anchor_times": valid_times.tolist(),
        "dom_parts": dom_parts,
        "any_active_part": any_active,
        "phase_seq": valid_phases.tolist(),
        "support_seq": valid_supports.tolist(),
        "target_centroid_world": target_centroid,
        "target_spread_cm": target_spread * 100.0,
    }


def _z_int_pairwise_diff(a: dict, b: dict) -> dict:
    """Per-pair scalar diff metrics on z_int summaries."""
    return {
        "contact_state_l1": float(np.abs(
            np.array(a["contact_state_active_frac"])
            - np.array(b["contact_state_active_frac"])
        ).sum()),
        "phase_dist_l1": float(np.abs(
            np.array(a["phase_dist"]) - np.array(b["phase_dist"])
        ).sum()),
        "support_dist_l1": float(np.abs(
            np.array(a["support_dist"]) - np.array(b["support_dist"])
        ).sum()),
        "target_norm_diff": abs(a["contact_target_norm_mean"] - b["contact_target_norm_mean"]),
    }


def _plan_pairwise_diff(a: dict, b: dict) -> dict:
    """Per-pair diff on plan summaries."""
    return {
        "n_anchors_diff": abs(a["n_anchors"] - b["n_anchors"]),
        "active_part_l1": int(np.abs(
            np.array(a["any_active_part"], dtype=int)
            - np.array(b["any_active_part"], dtype=int)
        ).sum()),
        "target_centroid_dist_cm": float(np.linalg.norm(
            np.array(a["target_centroid_world"])
            - np.array(b["target_centroid_world"])
        ) * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/anchordiff_v12_dit_block_FULL_N10.yaml"),
        help="Use the same data section as the training config.",
    )
    parser.add_argument(
        "--bucket", default="train", choices=["train", "val"],
    )
    parser.add_argument(
        "--md", type=Path,
        default=Path("analyses/2026-05-11_cond_diversity_audit.md"),
    )
    parser.add_argument(
        "--top-k-clusters", type=int, default=5,
        help="How many of the largest text-clusters to surface in the report.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Reuse the same dataset construction as plan_condition_diagnostics
    # (which mirrors the trainer's _build_dataset).
    full_dataset = _build_dataset(cfg, args.bucket, augment=False)
    overfit_n_clips = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n_clips > 0:
        full_dataset = Subset(full_dataset, list(range(min(overfit_n_clips, len(full_dataset)))))

    print(f"Loaded {len(full_dataset)} clips ({args.bucket} bucket)")

    # Iterate one clip at a time (batch_size=1) and collect summaries.
    loader = DataLoader(
        full_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )

    clips: list[dict] = []
    for i, batch in enumerate(loader):
        text = batch["text"][0] if isinstance(batch["text"], list) else str(batch["text"][0])
        z_summary = _z_int_summary(batch)
        plan_summary = _plan_summary(batch)
        clips.append({
            "idx": i,
            "text": text,
            "z_int": z_summary,
            "plan": plan_summary,
        })
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(full_dataset)}")
    print(f"Done: {len(clips)} clips")

    # Group by exact text.
    text_clusters: dict[str, list[int]] = defaultdict(list)
    for c in clips:
        text_clusters[c["text"]].append(c["idx"])

    # Sort clusters by size descending.
    sorted_clusters = sorted(
        text_clusters.items(), key=lambda kv: -len(kv[1]),
    )

    # ---- Aggregate text statistics ----
    unique_texts = len(sorted_clusters)
    largest_size = len(sorted_clusters[0][1]) if sorted_clusters else 0
    singletons = sum(1 for _, idxs in sorted_clusters if len(idxs) == 1)

    # ---- For top-k largest clusters, compute intra-cluster diffs ----
    top_k = sorted_clusters[: args.top_k_clusters]
    intra_diffs_per_cluster = []
    for text, idxs in top_k:
        if len(idxs) < 2:
            continue
        cluster_clips = [clips[i] for i in idxs]
        # All pairs within cluster
        z_diffs = []
        plan_diffs = []
        for i in range(len(cluster_clips)):
            for j in range(i + 1, len(cluster_clips)):
                z_diffs.append(_z_int_pairwise_diff(
                    cluster_clips[i]["z_int"], cluster_clips[j]["z_int"],
                ))
                plan_diffs.append(_plan_pairwise_diff(
                    cluster_clips[i]["plan"], cluster_clips[j]["plan"],
                ))
        intra_diffs_per_cluster.append({
            "text": text,
            "n_clips": len(cluster_clips),
            "n_pairs": len(z_diffs),
            "z_int_mean_diff": {
                k: float(np.mean([d[k] for d in z_diffs]))
                for k in z_diffs[0].keys()
            },
            "plan_mean_diff": {
                k: float(np.mean([d[k] for d in plan_diffs]))
                for k in plan_diffs[0].keys()
            },
            "sample_clip_idxs": idxs[:3],  # first 3 for verbatim inspection
        })

    # ---- Random inter-cluster baseline ----
    # Sample 200 random pairs from DIFFERENT text clusters.
    rng = np.random.RandomState(42)
    inter_z_diffs = []
    inter_plan_diffs = []
    if len(sorted_clusters) >= 2:
        n_clips = len(clips)
        n_pairs_target = 200
        attempts = 0
        while len(inter_z_diffs) < n_pairs_target and attempts < n_pairs_target * 10:
            i, j = rng.choice(n_clips, 2, replace=False)
            if clips[i]["text"] != clips[j]["text"]:
                inter_z_diffs.append(_z_int_pairwise_diff(
                    clips[i]["z_int"], clips[j]["z_int"],
                ))
                inter_plan_diffs.append(_plan_pairwise_diff(
                    clips[i]["plan"], clips[j]["plan"],
                ))
            attempts += 1

    inter_z_mean = {
        k: float(np.mean([d[k] for d in inter_z_diffs]))
        for k in inter_z_diffs[0].keys()
    } if inter_z_diffs else {}
    inter_plan_mean = {
        k: float(np.mean([d[k] for d in inter_plan_diffs]))
        for k in inter_plan_diffs[0].keys()
    } if inter_plan_diffs else {}

    # ---- Markdown report ----
    md = []
    md.append("# Cond diversity audit — text vs z_int + plan signal")
    md.append("")
    md.append(f"**Date:** 2026-05-11")
    md.append(f"**Config:** `{args.config}`")
    md.append(f"**Bucket:** {args.bucket}")
    md.append(f"**Total clips analysed:** {len(clips)}")
    md.append("")
    md.append("## §1 Text uniqueness")
    md.append("")
    md.append(f"- Unique text strings: **{unique_texts}** ({unique_texts/len(clips)*100:.1f}% of clips)")
    md.append(f"- Singleton texts (text appears in exactly 1 clip): **{singletons}** ({singletons/len(clips)*100:.1f}%)")
    md.append(f"- Largest text cluster: **{largest_size} clips**")
    md.append("")
    md.append("Top-10 most common texts (cluster size):")
    md.append("")
    md.append("| rank | n_clips | text |")
    md.append("|---|---|---|")
    for i, (text, idxs) in enumerate(sorted_clusters[:10]):
        short = text[:80] + ("..." if len(text) > 80 else "")
        md.append(f"| {i+1} | {len(idxs)} | {short!r} |")
    md.append("")
    md.append(
        f"**Reading:** if singletons are >90% of clips, text alone is a unique ID — "
        f"no two clips share a text, so 'text-similar' is meaningless. "
        f"Got {singletons/len(clips)*100:.1f}% singletons here."
    )
    md.append("")

    md.append("## §2 Intra-cluster vs inter-cluster diff (the load-bearing comparison)")
    md.append("")
    md.append(
        "For each of the largest text-clusters, we compute the **mean pairwise "
        "z_int / plan diff** among clips WITHIN the cluster. We compare against "
        "the **mean pairwise diff** between random clips from DIFFERENT text "
        "clusters (200-sample baseline)."
    )
    md.append("")
    md.append(
        "Interpretation: if intra-cluster diff ≈ inter-cluster diff, then same "
        "text → same z_int/plan is NOT true — the conditioning carries genuinely "
        "different signal even when text is identical. If intra ≪ inter, then "
        "text duplication does mean conditioning duplication — model can't "
        "discriminate clips that share text."
    )
    md.append("")
    md.append("### z_int diff (lower = more similar)")
    md.append("")
    md.append("| metric | inter-cluster (random pairs, different text) | " +
              " | ".join([f"cluster {i+1} ({len(c['sample_clip_idxs'])} clips intra)" for i, c in enumerate(intra_diffs_per_cluster)]) + " |")
    md.append("|---|" + "---|" * (len(intra_diffs_per_cluster) + 1))
    for key in ["contact_state_l1", "phase_dist_l1", "support_dist_l1", "target_norm_diff"]:
        row = [key, f"{inter_z_mean.get(key, 0):.4f}"]
        for c in intra_diffs_per_cluster:
            row.append(f"{c['z_int_mean_diff'].get(key, 0):.4f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("### plan diff (lower = more similar)")
    md.append("")
    md.append("| metric | inter-cluster | " +
              " | ".join([f"cluster {i+1}" for i in range(len(intra_diffs_per_cluster))]) + " |")
    md.append("|---|" + "---|" * (len(intra_diffs_per_cluster) + 1))
    for key in ["n_anchors_diff", "active_part_l1", "target_centroid_dist_cm"]:
        row = [key, f"{inter_plan_mean.get(key, 0):.4f}"]
        for c in intra_diffs_per_cluster:
            row.append(f"{c['plan_mean_diff'].get(key, 0):.4f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## §3 Verbatim samples for top clusters")
    md.append("")
    for c in intra_diffs_per_cluster:
        md.append(f"### Cluster: {c['n_clips']} clips share text:")
        md.append(f"> `{c['text']}`")
        md.append("")
        for clip_idx in c["sample_clip_idxs"]:
            clip = clips[clip_idx]
            md.append(f"**Clip {clip_idx}:**")
            md.append(f"- z_int.contact_state_active_frac (5 parts): "
                      f"{[f'{v:.2f}' for v in clip['z_int']['contact_state_active_frac']]}")
            md.append(f"- z_int.phase_dist: {[f'{v:.2f}' for v in clip['z_int']['phase_dist']]}")
            md.append(f"- z_int.support_dist: {[f'{v:.2f}' for v in clip['z_int']['support_dist']]}")
            md.append(f"- z_int.contact_target_norm_mean: {clip['z_int']['contact_target_norm_mean']:.3f}")
            md.append(f"- plan.n_anchors: {clip['plan']['n_anchors']}")
            md.append(f"- plan.any_active_part (5 parts): {clip['plan']['any_active_part']}")
            md.append(f"- plan.target_centroid_world (cm): "
                      f"{[f'{v*100:.1f}' for v in clip['plan']['target_centroid_world']]}")
            md.append(f"- plan.target_spread_cm: {clip['plan']['target_spread_cm']:.2f}")
            md.append("")

    md.append("## §4 Interpretation")
    md.append("")
    md.append("**Question 1: Is text a unique ID per clip?**")
    md.append(f"- {singletons}/{len(clips)} ({singletons/len(clips)*100:.1f}%) clips have unique text. ")
    if singletons / len(clips) > 0.95:
        md.append("  ≈ text IS unique — 'text-similar' check is mostly N/A. The audit's main lever is the rare clusters that do share text.")
    else:
        md.append("  text duplication is significant.")
    md.append("")
    md.append("**Question 2: When text is the same, are z_int / plan the same?**")
    md.append(
        "- If intra-cluster z_int / plan diffs are ≪ inter-cluster diffs → "
        "text DOES predict z_int and plan duplication; model can't discriminate. "
        "If intra ≈ inter → conditioning carries discriminative signal beyond text."
    )
    md.append("")
    md.append("Compute intra/inter ratios per metric and read off above tables.")
    md.append("")

    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.md}")


if __name__ == "__main__":
    main()
