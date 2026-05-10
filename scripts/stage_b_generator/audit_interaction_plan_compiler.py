"""Run the InteractionPlanCompiler over a validation subset and emit
aggregate statistics + a Markdown report + per-clip plots.

Usage::

    python scripts/stage_b_generator/audit_interaction_plan_compiler.py \\
        --config configs/training/anchordiff_v9_4_hardobs_overfit.yaml \\
        --bucket val \\
        --max-clips 200 \\
        --output analyses/2026-05-10_interaction_plan_compiler_audit.md \\
        --plot-dir analyses/2026-05-10_interaction_plan_compiler_audit_plots

Reads pseudo-labels via the same path the trainer uses (``HOIDataset``);
the dense ``z_int`` evidence (contact_state, contact_target_xyz, phase,
support) is consumed by the compiler exactly as Stage B will see it at
training time.

The audit is part of the method gate (see
``analyses/piano_interaction_plan_pipeline_reframe_for_claude_code.md``
§9.1 — average anchors per clip ∈ [3, 12], <5% zero-anchor clips,
anchor times not collapsed into a single region).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from piano.data.dataset import HOIDataset, build_subject_split, extract_subject_id
from piano.data.interaction_plan_compiler import (
    ANCHOR_TYPE_ONSET,
    ANCHOR_TYPE_PHASE_CHANGE,
    ANCHOR_TYPE_RELEASE,
    ANCHOR_TYPE_STABLE,
    ANCHOR_TYPE_SUPPORT_CHANGE,
    CompilerStats,
    InteractionPlanCompilerConfig,
    NUM_ANCHOR_TYPES,
    NUM_PARTS_DEFAULT,
    compile_interaction_plan,
    update_stats,
)
from piano.utils.io_utils import load_json


_TYPE_NAMES = {
    ANCHOR_TYPE_ONSET: "onset",
    ANCHOR_TYPE_STABLE: "stable",
    ANCHOR_TYPE_RELEASE: "release",
    ANCHOR_TYPE_PHASE_CHANGE: "phase_change",
    ANCHOR_TYPE_SUPPORT_CHANGE: "support_change",
}
_PART_NAMES = ("L_hand", "R_hand", "L_foot", "R_foot", "pelvis")


def _collect_subject_filter(cfg, bucket: str) -> set | None:
    subj_cfg = cfg.data.get("subject_split")
    if subj_cfg is None or not subj_cfg.get("enabled", False):
        return None
    keys: set[tuple[str, str]] = set()
    for entry in cfg.data.datasets:
        meta_path = Path(entry.root) / "metadata_clean.json"
        if not meta_path.exists():
            meta_path = Path(entry.root) / "metadata.json"
        for m in load_json(meta_path):
            sid = extract_subject_id(Path(entry.root).name, m.get("seq_id", ""))
            if sid is not None:
                keys.add((Path(entry.root).name, sid))
    splits = build_subject_split(
        sorted(keys),
        train_pct=subj_cfg.train_pct,
        val_pct=subj_cfg.val_pct,
        seed=subj_cfg.seed,
    )
    if bucket == "all":
        return None
    return splits[bucket]


def _phase_to_softmax(idx: np.ndarray, num_classes: int) -> np.ndarray:
    arr = np.zeros((len(idx), num_classes), dtype=np.float32)
    safe = np.clip(idx, 0, num_classes - 1).astype(np.int64)
    arr[np.arange(len(idx)), safe] = 1.0
    return arr


def _maybe_plot_clip(
    seq_id: str,
    seq_len: int,
    contact_smooth: np.ndarray,
    phase_label: np.ndarray,
    support_label: np.ndarray,
    plan: dict,
    out_dir: Path,
) -> None:
    """Per-clip diagnostic plot: contact prob / labels / anchor timeline.

    Skipped silently if matplotlib isn't available — the main report
    doesn't depend on plots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t = np.arange(seq_len)

    # Contact probability per part
    for p in range(contact_smooth.shape[1]):
        axes[0].plot(t, contact_smooth[:seq_len, p], label=_PART_NAMES[p], lw=1.0)
    axes[0].set_ylabel("contact prob")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right", fontsize=7, ncol=5)
    axes[0].set_title(f"{seq_id}  (seq_len={seq_len})")

    # Phase / support labels
    axes[1].plot(t, phase_label[:seq_len], lw=1.0, label="phase", color="tab:blue")
    axes[1].plot(t, support_label[:seq_len], lw=1.0, label="support", color="tab:orange")
    axes[1].set_ylabel("label idx")
    axes[1].legend(loc="upper right", fontsize=7)

    # Anchor timeline
    mask = plan["anchor_mask"]
    times = plan["anchor_time"][mask]
    types = plan["anchor_type"][mask]
    confs = plan["anchor_conf"][mask]
    type_color = {0: "tab:red", 1: "tab:green", 2: "tab:purple", 3: "tab:cyan", 4: "tab:olive"}
    for ti, ty, cf in zip(times, types, confs):
        axes[2].axvline(int(ti), color=type_color.get(int(ty), "k"), alpha=0.3 + 0.6 * float(cf))
    legend_handles = [
        plt.Line2D([0], [0], color=type_color[k], label=_TYPE_NAMES[k])
        for k in sorted(type_color)
    ]
    axes[2].legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=5)
    axes[2].set_xlim(0, seq_len)
    axes[2].set_ylabel("anchors")
    axes[2].set_xlabel("frame")

    fig.tight_layout()
    fig.savefig(out_dir / f"{seq_id.replace('/', '_')}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bucket", choices=["train", "val", "all"], default="val")
    parser.add_argument("--max-clips", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--num-plot-samples", type=int, default=8)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    subj_filter = _collect_subject_filter(cfg, args.bucket)

    pseudo_label_subdir = cfg.data.get("pseudo_label_subdir", None)

    datasets: list[HOIDataset] = []
    for entry in cfg.data.datasets:
        sub_dir = (
            str(Path(entry.root) / pseudo_label_subdir) if pseudo_label_subdir else None
        )
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=int(cfg.data.max_seq_length),
            subject_id_filter=subj_filter,
            subsample_n_per_object=cfg.data.get("subsample_n_per_object", None),
            subsample_seed=int(cfg.data.get("subsample_seed", 42)),
            support_collapse_hand_support=bool(
                cfg.data.get("support_collapse_hand_support", True)
            ),
            surface_obj_pose=False,
            motion_representation="motion_263",
        )
        datasets.append(ds)

    compiler_cfg = InteractionPlanCompilerConfig()
    stats = CompilerStats()
    per_clip_anchor_counts: list[int] = []
    per_clip_seq_len: list[int] = []
    plot_dir = args.plot_dir
    n_plotted = 0
    plot_samples_target = args.num_plot_samples
    n_processed = 0

    for ds in datasets:
        for entry in ds.metadata:
            if n_processed >= args.max_clips:
                break
            seq_id = entry["seq_id"]
            label_path = ds.pseudo_label_dir / f"{seq_id}.npz"
            motion_path = ds.root / "motions" / f"{seq_id}.npz"
            if not label_path.exists() or not motion_path.exists():
                continue
            label = np.load(label_path, allow_pickle=False)
            motion = np.load(motion_path, allow_pickle=False)
            if "contact_state" not in label.files or "object_positions" not in motion.files:
                continue
            contact = label["contact_state"].astype(np.float32)
            target_local = (
                label["contact_target_xyz_gt"].astype(np.float32)
                if "contact_target_xyz_gt" in label.files
                else label["contact_target"].astype(np.float32)
            )
            phase = label["phase"].astype(np.int64)
            support = label["support"].astype(np.int64)
            obj_pos = motion["object_positions"].astype(np.float32)
            obj_rot = motion["object_rotations"].astype(np.float32)
            seq_len = min(len(contact), len(obj_pos), int(cfg.data.max_seq_length))

            # Phase / support: GT integers → one-hot softmax.
            num_phase = int(cfg.model.z_int.phase_classes) if "model" in cfg else 3
            num_sup = int(cfg.model.z_int.support_classes) if "model" in cfg else 3
            if cfg.data.get("support_collapse_hand_support", True):
                support = support.copy()
                support[support == 3] = 0
            phase_soft = _phase_to_softmax(phase[:seq_len], num_phase)
            support_soft = _phase_to_softmax(support[:seq_len], num_sup)

            # Some legacy npzs may have contact_target with shape (T, P, K).
            # The compiler expects (T, P, 3). Skip if the shape mismatches.
            if target_local.ndim != 3 or target_local.shape[-1] != 3:
                continue

            plan = compile_interaction_plan(
                contact_prob=contact[:seq_len],
                target_local=target_local[:seq_len],
                phase_softmax=phase_soft,
                support_softmax=support_soft,
                object_pos_world=obj_pos[:seq_len],
                object_rot_world_aa=obj_rot[:seq_len],
                seq_len=seq_len,
                cfg=compiler_cfg,
            )
            update_stats(stats, plan, seq_len)
            per_clip_anchor_counts.append(int(plan["anchor_mask"].sum()))
            per_clip_seq_len.append(seq_len)
            n_processed += 1

            if plot_dir and n_plotted < plot_samples_target:
                from piano.data.interaction_plan_compiler import smooth_contact as _sc
                _maybe_plot_clip(
                    f"{ds.root.name}/{seq_id}",
                    seq_len,
                    _sc(contact[:seq_len], compiler_cfg.contact_smooth_window),
                    phase[:seq_len],
                    support[:seq_len],
                    plan,
                    plot_dir,
                )
                n_plotted += 1
        if n_processed >= args.max_clips:
            break

    # ---------------------------------------------------------------------
    # Write report
    # ---------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    avg_anchors = (
        sum(per_clip_anchor_counts) / max(len(per_clip_anchor_counts), 1)
    )
    avg_time_norm = (
        stats.anchor_time_normalized_sum / max(stats.anchor_time_normalized_n, 1)
    )
    zero_anchor_pct = (
        100.0 * stats.n_zero_anchor_clips / max(stats.n_clips, 1)
    )

    pass_avg = 3.0 <= avg_anchors <= 12.0
    pass_zero_anchor = zero_anchor_pct < 5.0
    # Anchor times should be spread, not concentrated in one third of
    # the clip. avg_time_norm ≈ 0.5 means the population average is
    # roughly mid-clip, which is what we expect. We flag if the
    # population avg is < 0.25 or > 0.75 (collapse to one end).
    pass_time_spread = 0.25 <= avg_time_norm <= 0.75

    md = []
    md.append(f"# Interaction Plan Compiler audit\n")
    md.append(f"**Date:** 2026-05-10  \n")
    md.append(f"**Config source:** `{args.config}`  \n")
    md.append(f"**Bucket:** `{args.bucket}`  \n")
    md.append(f"**Clips processed:** {stats.n_clips}\n")
    md.append("\n## Pass gates (per spec §9.1)\n")
    md.append(f"- Average anchors per clip: **{avg_anchors:.2f}** (target ∈ [3, 12])  → "
              f"{'✓' if pass_avg else '✗'}")
    md.append(f"- Zero-anchor clip rate: **{zero_anchor_pct:.1f}%** (target < 5%)  → "
              f"{'✓' if pass_zero_anchor else '✗'}")
    md.append(f"- Avg normalized anchor time (population): **{avg_time_norm:.3f}** "
              f"(target ∈ [0.25, 0.75]; mid-clip mean)  → "
              f"{'✓' if pass_time_spread else '✗'}")
    md.append("")
    md.append("\n## Anchor count histogram\n")
    md.append("| n_anchors | clips |")
    md.append("|---|---|")
    for n, c in enumerate(stats.anchor_count_histogram):
        md.append(f"| {n} | {c} |")
    md.append("\n## Segment count histogram\n")
    md.append("| n_segments | clips |")
    md.append("|---|---|")
    for n, c in enumerate(stats.segment_count_histogram):
        md.append(f"| {n} | {c} |")
    md.append("\n## Anchor type distribution\n")
    md.append("| type | count |")
    md.append("|---|---|")
    for tid, name in _TYPE_NAMES.items():
        md.append(f"| {name} | {stats.anchor_type_counts[tid]} |")
    md.append("\n## Anchor body-part distribution\n")
    md.append("| part | activations |")
    md.append("|---|---|")
    for p, name in enumerate(_PART_NAMES):
        if p < len(stats.anchor_part_counts):
            md.append(f"| {name} | {stats.anchor_part_counts[p]} |")
    md.append("\n## Compiler config used\n")
    md.append("```yaml")
    for f in compiler_cfg.__dataclass_fields__:
        md.append(f"{f}: {getattr(compiler_cfg, f)}")
    md.append("```")
    if plot_dir:
        md.append(f"\n## Per-clip plots\n")
        md.append(f"Wrote {n_plotted} sample plots to `{plot_dir}/`. Each plot shows: "
                  "smoothed contact probability per body part, phase / support GT "
                  "labels, and the compiled anchor timeline color-coded by anchor type.")

    args.output.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote report to {args.output}")
    print(f"Pass gates: avg={pass_avg}  zero_anchor={pass_zero_anchor}  time_spread={pass_time_spread}")
    print(f"Avg anchors per clip: {avg_anchors:.2f} | Zero-anchor: {zero_anchor_pct:.1f}%")


if __name__ == "__main__":
    main()
