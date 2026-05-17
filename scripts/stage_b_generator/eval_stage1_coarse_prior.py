"""Round-12 evaluation: per-subset Coarse-v1 audit for Stage-1 prior.

Loads a Stage-1 checkpoint, samples Coarse-v1 trajectories conditioned
on cached CLIP text + init Coarse-v1 (frame 0), and reports per-subset
coarse metrics — without using any object/plan/contact information.

Primary metrics (per Codex review §8):

- root velocity        : mean |Δroot| per frame
- root acc p95         : p95 of |Δ²root|
- root jerk p95        : p95 of |Δ³root|
- yaw range            : (max − min) of unwrapped yaw
- yaw velocity         : mean |Δyaw|
- pelvis rot6d velocity: mean |Δrot6d| (Frobenius on rot6d channels)
- spine3 rot6d velocity: mean |Δrot6d|
- head height range / velocity
- shoulder center height range / velocity

Per subset: chairs, imhd, neuraldome, omomo_correct_v2.

xGT = generated metric / GT metric on the same clip set (closer to 1.0
is better). M2/M3 contact metrics are deliberately NOT reported —
those require interaction alignment, which is Stage-2 territory.

Usage
-----

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/eval_stage1_coarse_prior.py \
        --ckpt runs/training/stage1_coarse_prior_s1a_sanity1000/final.pt \
        --tag s1a_sanity1000 --num-clips-per-subset 6 --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from piano.models.coarse_motion_prior import (
    CoarsePriorConfig, CoarsePriorDenoiserConfig, CoarsePriorDiff,
)
from piano.models.motion_anchordiff import DiffusionConfig


# ============================================================================
# Channel layout (sync with trainer)
# ============================================================================
COARSE_DIM = 23
ROOT_TRANS_DIMS = slice(0, 3)        # [0:3] xz, y
ROOT_VEL_DIMS = slice(3, 6)
YAW_SC_DIMS = slice(6, 8)
YAW_VEL_DIM = 8
PELVIS_ROT_DIMS = slice(9, 15)
SPINE3_ROT_DIMS = slice(15, 21)
HEAD_HT_DIM = 21
SHOULDER_HT_DIM = 22


def _per_clip_metrics(coarse: np.ndarray) -> dict[str, float]:
    """coarse: (T, 23). Returns scalar metrics for the clip.

    Round-13 expansion: separately report the **stored** yaw velocity
    channel (``coarse[:, 8]``) and the **derived** yaw velocity
    (finite-difference of unwrapped ``atan2(yaw_sin, yaw_cos)``), plus
    a consistency error between the two. The stored vs derived split
    catches model drift where the model emits a ``yaw_vel`` channel
    that no longer matches its own ``yaw_sin/cos`` derivative.
    """
    T = coarse.shape[0]
    out: dict[str, float] = {}

    # Root translation derivatives (use stored root_local_trans channels)
    root = coarse[:, ROOT_TRANS_DIMS]                            # (T, 3)
    if T >= 2:
        dr = np.diff(root, axis=0)                               # (T-1, 3)
        out["root_vel_mean_abs"] = float(np.mean(np.linalg.norm(dr, axis=-1)))
        if T >= 3:
            d2r = np.diff(dr, axis=0)
            out["root_acc_p95"] = float(np.percentile(np.linalg.norm(d2r, axis=-1), 95))
        else:
            out["root_acc_p95"] = 0.0
        if T >= 4:
            d3r = np.diff(np.diff(dr, axis=0), axis=0)
            out["root_jerk_p95"] = float(np.percentile(np.linalg.norm(d3r, axis=-1), 95))
        else:
            out["root_jerk_p95"] = 0.0
    else:
        out["root_vel_mean_abs"] = 0.0
        out["root_acc_p95"] = 0.0
        out["root_jerk_p95"] = 0.0

    # Yaw range + velocity.
    # The Round-10 extractor stores BOTH yaw sin/cos (state) and yaw_vel
    # (stored derivative). The model can in principle emit a stored
    # yaw_vel that diverges from diff(unwrap(atan2(sin, cos))). We
    # surface all three quantities so the consistency can be audited
    # post-hoc.
    yaw_sin = coarse[:, 6]
    yaw_cos = coarse[:, 7]
    yaw_raw = np.arctan2(yaw_sin, yaw_cos)
    yaw_unwrapped = np.unwrap(yaw_raw)
    out["yaw_range"] = float(yaw_unwrapped.max() - yaw_unwrapped.min())

    yaw_vel_stored = coarse[:, YAW_VEL_DIM]                          # (T,)
    out["yaw_vel_stored_mean_abs"] = float(np.mean(np.abs(yaw_vel_stored)))

    if T >= 2:
        # Derive yaw_vel from the sin/cos state. Match the extractor's
        # convention by using prepend so the result has length T.
        yaw_vel_from_sc = np.diff(
            yaw_unwrapped, prepend=yaw_unwrapped[:1],
        )
        out["yaw_vel_from_sincos_mean_abs"] = float(np.mean(np.abs(yaw_vel_from_sc)))
        # Consistency: how far the stored yaw_vel channel is from the
        # sin/cos derivative. For GT this is ~0 by construction
        # (extractor builds yaw_vel from the same unwrap); for a model
        # sample it measures internal consistency.
        out["yaw_vel_consistency_error_mean_abs"] = float(
            np.mean(np.abs(yaw_vel_stored - yaw_vel_from_sc))
        )
        # Legacy alias kept for backward compat with Round-12 audit JSONs.
        out["yaw_vel_mean_abs"] = out["yaw_vel_from_sincos_mean_abs"]
    else:
        out["yaw_vel_from_sincos_mean_abs"] = 0.0
        out["yaw_vel_consistency_error_mean_abs"] = 0.0
        out["yaw_vel_mean_abs"] = 0.0

    # Pelvis / spine3 rot6d velocity
    pelvis_rot = coarse[:, PELVIS_ROT_DIMS]
    spine3_rot = coarse[:, SPINE3_ROT_DIMS]
    if T >= 2:
        out["pelvis_rot6d_vel_mean"] = float(
            np.mean(np.linalg.norm(np.diff(pelvis_rot, axis=0), axis=-1))
        )
        out["spine3_rot6d_vel_mean"] = float(
            np.mean(np.linalg.norm(np.diff(spine3_rot, axis=0), axis=-1))
        )
    else:
        out["pelvis_rot6d_vel_mean"] = 0.0
        out["spine3_rot6d_vel_mean"] = 0.0

    # Head / shoulder height
    head_h = coarse[:, HEAD_HT_DIM]
    shoulder_h = coarse[:, SHOULDER_HT_DIM]
    out["head_height_range"] = float(head_h.max() - head_h.min())
    out["shoulder_height_range"] = float(shoulder_h.max() - shoulder_h.min())
    if T >= 2:
        out["head_height_vel_mean"] = float(np.mean(np.abs(np.diff(head_h))))
        out["shoulder_height_vel_mean"] = float(np.mean(np.abs(np.diff(shoulder_h))))
    else:
        out["head_height_vel_mean"] = 0.0
        out["shoulder_height_vel_mean"] = 0.0
    return out


def _object_relative_metrics(
    coarse: np.ndarray,                    # (T, 23) denormalized Coarse-v1
    obj_traj: np.ndarray,                  # (T, 9) denormalized obj_pos (3) + obj_rot6d (6)
) -> dict[str, float]:
    """Object-relative metrics that use ALL valid frames — no contact-label
    dependency (per Round-17 §9.3 + SUGGESTION.md §"Object-relative eval
    must not require contact labels as a primary metric").

    Operates in the frame documented by the active cache contract. Under
    Round-18-fix, both inputs are in root0-relative, world-axis coordinates
    (cache field name: `obj_traj_root0_world`). Both inputs are
    valid-truncated by the caller. Metric formulae are frame-agnostic
    (norms + dot products), so the same code works against the legacy
    `obj_traj_canonical` field too.
    """
    T = coarse.shape[0]
    out: dict[str, float] = {}
    if T < 2 or obj_traj.shape[0] < 2:
        return out
    root = coarse[:, ROOT_TRANS_DIMS]      # (T, 3) root_local_trans
    obj_com = obj_traj[:, 0:3]             # (T, 3) obj position
    valid_T = min(root.shape[0], obj_com.shape[0])
    root = root[:valid_T]
    obj_com = obj_com[:valid_T]
    # 1) Root-object distance trajectory: ||root(t) - obj(t)|| per frame.
    dist = np.linalg.norm(root - obj_com, axis=-1)       # (T,)
    out["root_obj_dist_mean"] = float(np.mean(dist))
    out["root_obj_dist_p50"] = float(np.median(dist))
    out["root_obj_dist_p95"] = float(np.percentile(dist, 95))
    out["root_obj_dist_range"] = float(dist.max() - dist.min())
    # 2) Facing-to-object alignment: cosine(human facing vector, root→obj direction in XZ).
    # Human facing comes from coarse's yaw sin/cos. `forward_xz = (yaw_sin, yaw_cos)`
    # is the unit-vector direction the body is facing in the X-Z plane.
    yaw_sin = coarse[:valid_T, 6]
    yaw_cos = coarse[:valid_T, 7]
    forward_xz = np.stack([yaw_sin, yaw_cos], axis=-1)
    # Normalize defensively (yaw sin/cos should be unit-length but
    # generated samples may drift).
    fn = np.linalg.norm(forward_xz, axis=-1, keepdims=True)
    forward_xz = forward_xz / np.clip(fn, 1e-6, None)
    # Direction from root to object in X-Z plane.
    to_obj_xz = (obj_com - root)[:, [0, 2]]              # (T, 2)
    ton = np.linalg.norm(to_obj_xz, axis=-1, keepdims=True)
    to_obj_xz = to_obj_xz / np.clip(ton, 1e-6, None)
    cos_align = (forward_xz * to_obj_xz).sum(axis=-1)    # (T,) in [-1, 1]
    out["facing_to_obj_cos_mean"] = float(np.mean(cos_align))
    out["facing_to_obj_cos_p95"] = float(np.percentile(cos_align, 95))
    return out


METRIC_KEYS = (
    "root_vel_mean_abs", "root_acc_p95", "root_jerk_p95",
    "yaw_range",
    "yaw_vel_from_sincos_mean_abs",
    "yaw_vel_stored_mean_abs",
    "yaw_vel_consistency_error_mean_abs",
    "pelvis_rot6d_vel_mean", "spine3_rot6d_vel_mean",
    "head_height_range", "head_height_vel_mean",
    "shoulder_height_range", "shoulder_height_vel_mean",
)


# ============================================================================
# Checkpoint loading
# ============================================================================


def _build_model_from_ckpt(
    ckpt: dict[str, Any], *, prefer_ema: bool = True,
) -> tuple[CoarsePriorDiff, bool]:
    """Build the Stage-1 model from a checkpoint. Returns ``(model, loaded_ema)``.

    Round-18: when ``prefer_ema=True`` AND the ckpt has an ``"ema"`` state dict,
    the EMA copy is loaded into the model and the LIVE state dict is ignored.
    This matches MDM-family practice — sampling-time inference uses EMA weights.
    """
    cfg_d = ckpt["config"]
    diff = DiffusionConfig(
        num_steps=int(cfg_d["model"]["diffusion"]["num_steps"]),
        schedule=str(cfg_d["model"]["diffusion"]["schedule"]),
        objective="ddpm",
        prediction_target="x0",
    )
    den_d = cfg_d["model"]["denoiser"]
    den = CoarsePriorDenoiserConfig(
        coarse_dim=int(den_d["coarse_dim"]),
        text_dim=int(den_d["text_dim"]),
        init_pose_dim=int(den_d["init_pose_dim"]),
        d_model=int(den_d["d_model"]),
        n_layers=int(den_d["n_layers"]),
        n_heads=int(den_d["n_heads"]),
        ff_mult=int(den_d["ff_mult"]),
        dropout=float(den_d.get("dropout", 0.1)),
        max_seq_length=int(den_d["max_seq_length"]),
        attention_mode=str(den_d["attention_mode"]),
        block_size=int(den_d.get("block_size", 16)),
        obj_traj_dim=int(den_d.get("obj_traj_dim", 0)),
        obj_traj_hint_hidden_mult=int(den_d.get("obj_traj_hint_hidden_mult", 1)),
    )
    model = CoarsePriorDiff(CoarsePriorConfig(diffusion=diff, denoiser=den))
    # Load LIVE state first so any keys missing from the EMA partial overlay
    # still get initialized (EMA only holds requires_grad parameters, not
    # buffers like positional encoding).
    model.load_state_dict(ckpt["model"], strict=True)
    loaded_ema = False
    if prefer_ema and "ema" in ckpt:
        ema_sd = ckpt["ema"]
        # Apply EMA over the live model in-place for the trainable params
        # the EMA tracked.
        own = model.state_dict()
        for k, v in ema_sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v.to(own[k].dtype)
        model.load_state_dict(own, strict=True)
        loaded_ema = True
    return model, loaded_ema


# ============================================================================
# Eval driver
# ============================================================================


def _per_subset_balanced_indices(
    records: list[dict[str, Any]], n_per_subset: int, seed: int,
) -> list[int]:
    """Pick up to n_per_subset clips per subset, deterministic by seed.

    Smoke / fallback selection mode. The official paired comparison
    must use ``--selection-json`` (see ``_indices_from_selection_json``)
    so both S1-A and S1-B are evaluated on the SAME clip set.
    """
    by_subset: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_subset[r["subset"]].append(i)
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for subset in ("chairs", "imhd", "neuraldome", "omomo_correct_v2"):
        idxs = by_subset.get(subset, [])
        if not idxs:
            continue
        if len(idxs) > n_per_subset:
            idxs = list(rng.choice(idxs, size=n_per_subset, replace=False))
        keep.extend(int(i) for i in idxs)
    return keep


def _indices_from_selection_json(
    records: list[dict[str, Any]],
    selection_path: Path,
) -> tuple[list[int], list[dict[str, str]], list[dict[str, str]]]:
    """Match a Round-9-style selection JSON against the current Stage-1 cache.

    Selection JSON schema (subset of Round-9
    ``analyses/2026-05-19_subset_balanced_failure_selection.json``):

        {"selected": [{"subset": str, "seq_id": str, ...}, ...]}

    Returns ``(matched_indices, matched_records, missing_records)``.
    Missing means a (subset, seq_id) pair from the selection that is
    not present in the current cache manifest. The caller decides
    whether to fail hard or just warn.
    """
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    entries = payload.get("selected", payload)
    if not isinstance(entries, list) or not entries:
        raise SystemExit(
            f"[selection] {selection_path} has no 'selected' entries"
        )

    # Build a (subset, seq_id) -> manifest index lookup.
    lookup: dict[tuple[str, str], int] = {}
    for i, r in enumerate(records):
        lookup[(r["subset"], r["seq_id"])] = i

    matched_indices: list[int] = []
    matched_records: list[dict[str, str]] = []
    missing_records: list[dict[str, str]] = []
    for e in entries:
        subset = str(e.get("subset", ""))
        seq_id = str(e.get("seq_id", ""))
        key = (subset, seq_id)
        if key in lookup:
            matched_indices.append(lookup[key])
            matched_records.append({"subset": subset, "seq_id": seq_id})
        else:
            missing_records.append({"subset": subset, "seq_id": seq_id})
    return matched_indices, matched_records, missing_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("cache/stage1_coarse_v1_round12"))
    parser.add_argument(
        # Round-13 follow-up: default=None so we can tell if the user
        # explicitly passed --split. If --selection-json is provided
        # and --split is absent, we resolve split from the JSON's
        # "bucket" field (Option A in the round prompt).
        "--split", choices=["val", "train"], default=None,
        help="Manifest to evaluate against. If omitted and "
             "--selection-json is provided, the split is auto-resolved "
             "from the selection JSON's 'bucket' field. If omitted "
             "without a selection JSON, defaults to 'val'.",
    )
    parser.add_argument("--num-clips-per-subset", type=int, default=6)
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated seeds for the sampler.")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="Seed for clip selection (independent of sampler seed). "
                             "Only consulted in random per-subset fallback mode.")
    parser.add_argument(
        "--selection-json", "--selection-file", dest="selection_json",
        type=Path, default=None,
        help="Path to a Round-9-style selection JSON (e.g. "
             "analyses/2026-05-19_subset_balanced_failure_selection.json). "
             "When provided, the eval uses the EXACT (subset, seq_id) pairs "
             "from this file instead of a random per-subset balanced selection. "
             "Required for paired S1-A vs S1-B comparison.",
    )
    parser.add_argument(
        "--strict-selection", action="store_true",
        help="If set, exit with non-zero code when any (subset, seq_id) in "
             "--selection-json is missing from the current cache manifest. "
             "Default behaviour is to warn loudly and continue with the "
             "matched subset.",
    )
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional label baked into the output JSON filename.")
    parser.add_argument("--max-T", type=int, default=196)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-ddpm-steps", type=int, default=None,
        help="Override DDPM steps for faster smoke eval. Default = ckpt config value.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analyses"))
    # Round-18 additions.
    parser.add_argument(
        "--no-prefer-ema", action="store_true",
        help="If set, load LIVE weights even when the ckpt has an EMA copy. "
             "Default (preferred) is to use EMA weights for sampling.",
    )
    parser.add_argument(
        "--inpaint-frame0", action="store_true",
        help="Force frame 0 of the generated sample to equal the conditioned "
             "init pose via RePaint-style x_t replacement. Round-18 sampler "
             "extension.",
    )
    parser.add_argument(
        "--cfg-scale-text", type=float, default=1.0,
        help="Classifier-free guidance scale for the text branch at sampling. "
             "1.0 = no guidance (default). Higher = stronger text adherence.",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Round-13 final polish: resolve --split BEFORE any checkpoint or
    # cache load. The strict-selection split-mismatch guard now genuinely
    # short-circuits "before any model load" as advertised in the
    # Round-13 follow-up report. Resolution only depends on
    # args.selection_json and args.split — no checkpoint state needed.
    #
    # Rules:
    #   1. user explicit --split → use that;
    #      if selection JSON bucket disagrees, warn;
    #      with --strict-selection, hard-fail (exit 3).
    #   2. user omitted --split AND selection JSON provided
    #      → use JSON bucket.
    #   3. otherwise fall back to "val".
    user_split = args.split
    split_resolution_source = "user_cli" if user_split is not None else None
    selection_bucket: str | None = None
    if args.selection_json is not None:
        try:
            sel_payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
            bucket = sel_payload.get("bucket", None)
            if isinstance(bucket, str) and bucket in ("train", "val"):
                selection_bucket = bucket
        except Exception as e:
            print(f"[eval] WARNING — could not read 'bucket' from {args.selection_json}: {e!r}")
    if user_split is None:
        if selection_bucket is not None:
            split = selection_bucket
            split_resolution_source = "selection_json_bucket"
            print(
                f"[eval] --split auto-resolved to {split!r} from "
                f"{args.selection_json}'s bucket field"
            )
        else:
            split = "val"
            split_resolution_source = "fallback_default"
            print("[eval] --split not provided and no selection JSON bucket; falling back to 'val'")
    else:
        split = user_split
        if (
            selection_bucket is not None
            and selection_bucket != user_split
        ):
            msg = (
                f"[eval] WARNING — user passed --split {user_split!r} but "
                f"selection JSON bucket is {selection_bucket!r}. The "
                f"selection might be against a different split than the "
                f"one being evaluated."
            )
            print(msg)
            if args.strict_selection:
                print(
                    "[eval] --strict-selection set → split mismatch is fatal. "
                    "Exiting BEFORE checkpoint/model load."
                )
                args.output_dir.mkdir(parents=True, exist_ok=True)
                tag = args.tag or args.ckpt.parent.name
                strict_path = args.output_dir / f"2026-05-23_stage1_eval_{tag}_strict_split_mismatch.json"
                strict_path.write_text(json.dumps({
                    "ckpt": str(args.ckpt),
                    "selection_json": str(args.selection_json),
                    "user_split": user_split,
                    "selection_bucket": selection_bucket,
                    "reason": "split_mismatch_with_strict_selection",
                }, indent=2), encoding="utf-8")
                print(f"[eval] wrote {strict_path}")
                return 3

    # Now (and only now) load the checkpoint + build the model.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, loaded_ema = _build_model_from_ckpt(
        ckpt, prefer_ema=not args.no_prefer_ema,
    )
    if loaded_ema:
        print("[eval] loaded EMA weights (Round-18; pass --no-prefer-ema to use LIVE)")
    elif "ema" in ckpt and args.no_prefer_ema:
        print("[eval] ckpt has EMA but --no-prefer-ema set; using LIVE weights")
    model.eval()
    # Round-18: detect obj_traj requirement from the loaded model config.
    model_obj_traj_dim = int(model.cfg.denoiser.obj_traj_dim)

    if args.num_ddpm_steps is not None:
        # Quick override for smoke-eval speed: rebuild diffusion with fewer steps.
        from piano.models.motion_anchordiff import (
            DiffusionConfig as _DC, GaussianDiffusion as _GD,
        )
        new_cfg = _DC(
            num_steps=int(args.num_ddpm_steps), schedule="cosine",
            objective="ddpm", prediction_target="x0",
        )
        model.diffusion = _GD(new_cfg)
    device = torch.device(args.device)
    model = model.to(device)

    # Load cache manifest + CLIP text embeddings + normalization.
    cache_root = args.cache_root
    manifest = [
        json.loads(line)
        for line in (cache_root / f"manifest_{split}.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    clip_npz = np.load(cache_root / "text_embeddings_clip_vit_b32.npz", allow_pickle=True)
    clip_emb = clip_npz["embeddings"]
    text_index = json.loads((cache_root / "text_embeddings_index.json").read_text("utf-8"))["index"]
    norm = json.loads((cache_root / "normalization_train.json").read_text("utf-8"))
    mean = np.asarray(norm["global"]["mean"], dtype=np.float32)
    std = np.asarray(norm["global"]["std_clamped"], dtype=np.float32)
    # Round-18 + Round-18-fix: obj_traj normalization stats.
    # New cache uses `obj_traj_root0_world`; legacy cache used
    # `obj_traj_canonical`. Probe new name first, then fall back.
    global_block = norm.get("global", {})
    obj_block = global_block.get("obj_traj_root0_world", None)
    obj_field_name: str | None = None
    if obj_block is not None:
        obj_field_name = "obj_traj_root0_world"
    else:
        obj_block = global_block.get("obj_traj_canonical", None)
        if obj_block is not None:
            obj_field_name = "obj_traj_canonical"
    if obj_block is not None:
        obj_mean = np.asarray(obj_block["mean"], dtype=np.float32)
        obj_std = np.asarray(obj_block["std_clamped"], dtype=np.float32)
    else:
        obj_mean = None
        obj_std = None
    if model_obj_traj_dim > 0 and obj_mean is None:
        raise SystemExit(
            f"[eval] model needs obj_traj (obj_traj_dim={model_obj_traj_dim}) but "
            f"cache at {cache_root} has no obj_traj normalization stats. "
            f"Point --cache-root at an objtraj cache."
        )
    if model_obj_traj_dim == 0 and obj_mean is not None:
        print(
            f"[eval] WARNING — cache has obj_traj ({obj_field_name}) but model "
            f"has obj_traj_dim=0; the obj_traj fields will be loaded but not "
            f"consumed (object-free model)."
        )
    if obj_field_name is not None:
        print(f"[eval] obj_traj cache field = {obj_field_name}")

    # Selection: prefer fixed Round-9-style selection JSON when provided,
    # else fall back to random per-subset balanced selection (smoke).
    matched_records: list[dict[str, str]] = []
    missing_records: list[dict[str, str]] = []
    selection_mode = "random_balanced"
    selection_source: str | None = None
    if args.selection_json is not None:
        selection_mode = "selection_json"
        selection_source = str(args.selection_json)
        keep_idx, matched_records, missing_records = _indices_from_selection_json(
            manifest, args.selection_json,
        )
        n_total = len(matched_records) + len(missing_records)
        print(
            f"[eval] selection_json={args.selection_json}  "
            f"matched={len(matched_records)}/{n_total}  "
            f"missing={len(missing_records)}"
        )
        if missing_records:
            print("[eval] WARNING — selection has clips not in this cache:")
            for m in missing_records:
                print(f"   - {m['subset']}/{m['seq_id']}")
            if args.strict_selection:
                print("[eval] --strict-selection set → exiting with non-zero code")
                # Still save a record so the caller can audit which clips were missing.
                args.output_dir.mkdir(parents=True, exist_ok=True)
                tag = args.tag or args.ckpt.parent.name
                strict_path = args.output_dir / f"2026-05-23_stage1_eval_{tag}_strict_missing.json"
                strict_path.write_text(json.dumps({
                    "ckpt": str(args.ckpt),
                    "selection_json": selection_source,
                    "missing_records": missing_records,
                    "matched_count": len(matched_records),
                    "missing_count": len(missing_records),
                }, indent=2), encoding="utf-8")
                print(f"[eval] wrote {strict_path}")
                return 2
            print(
                "[eval] continuing with matched subset only "
                "(use --strict-selection to fail hard)"
            )
    else:
        keep_idx = _per_subset_balanced_indices(
            manifest, n_per_subset=args.num_clips_per_subset, seed=args.selection_seed,
        )
        print(
            f"[eval] selected {len(keep_idx)} clips from {split} manifest "
            f"(target up to {args.num_clips_per_subset}/subset, random per-subset balanced)"
        )
    if not keep_idx:
        print("[eval] no clips selected — exiting")
        return 1

    # ---------------- run eval ---------------- #
    per_clip: list[dict[str, Any]] = []
    t_start = time.time()
    for clip_i, idx in enumerate(keep_idx):
        r = manifest[idx]
        npz = np.load(cache_root / r["npz_path"], allow_pickle=False)
        gt = npz["coarse_v1"].astype(np.float32)                      # (T, 23)
        T = min(int(r["seq_len"]), gt.shape[0], args.max_T)
        gt = gt[:T]
        init = npz["init_coarse_v1"].astype(np.float32)
        init_norm = (init - mean) / std
        text = r.get("text", "")
        text_row = text_index.get(text, None)
        text_pool = (
            clip_emb[int(text_row)].astype(np.float32)
            if text_row is not None
            else np.zeros((512,), dtype=np.float32)
        )

        # Round-18 + Round-18-fix: load obj_traj from the clip npz when
        # the active cache stores it. Probes the new field name first,
        # then the legacy one, so both cache flavors evaluate cleanly.
        obj_traj_raw: np.ndarray | None = None
        obj_traj_norm: np.ndarray | None = None
        for cand_field in ("obj_traj_root0_world", "obj_traj_canonical"):
            if cand_field in npz.files:
                obj_traj_raw = npz[cand_field].astype(np.float32)[:T]  # (T, 9)
                break
        if obj_traj_raw is not None and obj_mean is not None and obj_std is not None:
            obj_traj_norm = (obj_traj_raw - obj_mean) / obj_std

        gt_metrics = _per_clip_metrics(gt)
        gt_obj_metrics = (
            _object_relative_metrics(gt, obj_traj_raw)
            if obj_traj_raw is not None else {}
        )

        # Sample for each requested seed.
        for seed in seeds:
            torch.manual_seed(seed)
            valid_mask = torch.ones(1, T, dtype=torch.bool, device=device)
            cond = {
                "text_pool": torch.from_numpy(text_pool).unsqueeze(0).to(device),
                "init_coarse": torch.from_numpy(init_norm).unsqueeze(0).to(device),
                "valid_mask": valid_mask,
            }
            if model_obj_traj_dim > 0:
                if obj_traj_norm is None:
                    raise SystemExit(
                        f"[eval] model needs obj_traj for clip {r['seq_id']} but "
                        "cache does not expose obj_traj for this clip"
                    )
                cond["obj_traj"] = torch.from_numpy(
                    obj_traj_norm,
                ).unsqueeze(0).to(device)
            with torch.no_grad():
                gen_norm = model.sample(
                    shape=(1, T, COARSE_DIM), cond=cond,
                    cfg_scale=float(args.cfg_scale_text), device=device,
                    inpaint_frame0=bool(args.inpaint_frame0),
                )
            gen_norm_np = gen_norm.squeeze(0).cpu().numpy()
            gen = gen_norm_np * std + mean                              # denormalize
            if not np.isfinite(gen).all():
                print(f"  [warn] non-finite sample for clip {r['seq_id']} seed {seed}")
            gen_metrics = _per_clip_metrics(gen)
            ratios = {
                f"xGT.{k}": (gen_metrics[k] / gt_metrics[k]) if gt_metrics[k] > 1e-6 else float("nan")
                for k in METRIC_KEYS
            }
            # Round-18: object-relative metrics (skip if no obj_traj available).
            gen_obj_metrics = (
                _object_relative_metrics(gen, obj_traj_raw)
                if obj_traj_raw is not None else {}
            )
            obj_xGT = {}
            for k in gt_obj_metrics:
                gv = gen_obj_metrics.get(k, float("nan"))
                gtv = gt_obj_metrics.get(k, float("nan"))
                obj_xGT[f"xGT.obj.{k}"] = (
                    float(gv / gtv) if gtv is not None and abs(gtv) > 1e-6 else float("nan")
                )
            per_clip.append({
                "clip_idx_in_manifest": int(idx),
                "subset": r["subset"],
                "seq_id": r["seq_id"],
                "seed": int(seed),
                "T": int(T),
                "gt": gt_metrics,
                "gen": gen_metrics,
                "xGT": ratios,
                "gen_finite": bool(np.isfinite(gen).all()),
                "gt_obj_metrics": gt_obj_metrics,
                "gen_obj_metrics": gen_obj_metrics,
                "obj_xGT": obj_xGT,
            })
        if (clip_i + 1) % 4 == 0:
            print(
                f"  [eval] {clip_i + 1}/{len(keep_idx)} clips done   "
                f"(elapsed {time.time() - t_start:.1f}s)"
            )
    elapsed = time.time() - t_start
    print(f"[eval] sampling done in {elapsed:.1f}s")

    # Per-subset summary (mean across clips × seeds).
    per_subset: dict[str, dict[str, dict[str, float]]] = {}
    for subset in ("chairs", "imhd", "neuraldome", "omomo_correct_v2"):
        rows = [r for r in per_clip if r["subset"] == subset]
        if not rows:
            continue
        agg: dict[str, dict[str, float]] = {"gen_mean": {}, "gt_mean": {}, "xGT_mean": {}}
        for k in METRIC_KEYS:
            agg["gen_mean"][k] = float(np.mean([r["gen"][k] for r in rows]))
            agg["gt_mean"][k] = float(np.mean([r["gt"][k] for r in rows]))
            vals = [r["xGT"][f"xGT.{k}"] for r in rows]
            vals = [v for v in vals if np.isfinite(v)]
            agg["xGT_mean"][k] = float(np.mean(vals)) if vals else float("nan")
        # Round-18: object-relative metric aggregates (skip if no obj_traj
        # available for this subset's clips).
        obj_keys: set[str] = set()
        for r in rows:
            obj_keys.update(r.get("gt_obj_metrics", {}).keys())
        if obj_keys:
            agg["gen_obj_mean"] = {}
            agg["gt_obj_mean"] = {}
            agg["obj_xGT_mean"] = {}
            for k in sorted(obj_keys):
                gen_vals = [r["gen_obj_metrics"].get(k, float("nan")) for r in rows
                            if "gen_obj_metrics" in r]
                gt_vals = [r["gt_obj_metrics"].get(k, float("nan")) for r in rows
                           if "gt_obj_metrics" in r]
                xg_vals = [r["obj_xGT"].get(f"xGT.obj.{k}", float("nan")) for r in rows
                           if "obj_xGT" in r]
                gen_vals_f = [v for v in gen_vals if np.isfinite(v)]
                gt_vals_f = [v for v in gt_vals if np.isfinite(v)]
                xg_vals_f = [v for v in xg_vals if np.isfinite(v)]
                agg["gen_obj_mean"][k] = float(np.mean(gen_vals_f)) if gen_vals_f else float("nan")
                agg["gt_obj_mean"][k] = float(np.mean(gt_vals_f)) if gt_vals_f else float("nan")
                agg["obj_xGT_mean"][k] = float(np.mean(xg_vals_f)) if xg_vals_f else float("nan")
        per_subset[subset] = agg

    # Output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.ckpt.parent.name
    out_path = args.output_dir / f"2026-05-22_stage1_eval_{tag}.json"
    payload = {
        "ckpt": str(args.ckpt),
        "tag": tag,
        "split": split,
        "split_resolution_source": split_resolution_source,
        "selection_bucket": selection_bucket,
        "user_split_arg": user_split,
        "seeds": seeds,
        "num_clips_per_subset": int(args.num_clips_per_subset),
        "n_clips": int(len(keep_idx)),
        "num_ddpm_steps_override": args.num_ddpm_steps,
        "elapsed_seconds": float(elapsed),
        "per_clip": per_clip,
        "per_subset": per_subset,
        "metric_keys": list(METRIC_KEYS),
        # Round-13: persist selection mode so paired analyses know
        # whether two eval JSONs are actually comparable.
        "selection_mode": selection_mode,
        "selection_source": selection_source,
        "selection_matched": matched_records,
        "selection_missing": missing_records,
        # Round-18 metadata for downstream paired analyses + tracking
        # whether EMA / frame-0 inpainting / obj_traj were active.
        "round18_meta": {
            "model_obj_traj_dim": int(model_obj_traj_dim),
            "loaded_ema": bool(loaded_ema),
            "inpaint_frame0": bool(args.inpaint_frame0),
            "cfg_scale_text": float(args.cfg_scale_text),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"[eval] wrote {out_path}")
    # Compact per-subset table
    print()
    print(f"{'subset':18s}  " + " ".join(f"{k:>22s}" for k in METRIC_KEYS))
    print(f"{'':18s}  " + " ".join("  ".join([f"{'xGT':>10s}", f"{'gen':>10s}"]) for _ in METRIC_KEYS))
    for subset, agg in per_subset.items():
        bits = []
        for k in METRIC_KEYS:
            bits.append(f"{agg['xGT_mean'][k]:>10.3f}  {agg['gen_mean'][k]:>10.4f}")
        print(f"{subset:18s}  " + " ".join(bits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
