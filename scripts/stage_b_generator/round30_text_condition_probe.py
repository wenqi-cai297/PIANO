"""Round-30 E1 — Text-condition usability probe.

Per analyses/2026-05-29_round30_idle_local_detail_diagnosis_plan.md §E1.
Decides whether ``text`` is alive in A1; if dead (H5), the entire round
exits at this gate and goes to D4 (text-architecture rework).

Five perturbations:
    baseline             — original text
    text_zero            — zeros_like(text_features)
    text_swap_neutral    — replace text with "a person stands still"
    text_swap_antonym    — replace text with "a person walks rapidly"
    text_shuffle_token   — permute non-padded text tokens within each clip

For each (clip, perturbation) sample the model with a fixed seed and
compare predicted joints to baseline. We report:
  * pred_delta_joints_cm on the 8-joint upper-body subset
  * the same on the full 22-joint set (for orientation)

Run twice: once on ILD val subset, once on control val subset (built by
round30_build_ild_subset.py). Outputs are written under the same dir.

Outputs:
    <output-dir>/text_probe_stats.json
    <output-dir>/text_probe_summary.md

The probe is single-clip (batch_size=1) like round29_cond_usage_probe,
but ``text_features`` is recomputed by re-running CLIP on a swapped
string; ``encode_text_per_token`` is cheap on a single sentence so this
adds < 1% wall-clock.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "stage_b_generator"
sys.path.insert(0, str(SCRIPTS))
# Reuse cm-conversion + key-joint plumbing from the cond-usage probe.
from round29_cond_usage_probe import (  # noqa: E402
    KEY_JOINT_INDICES,
    _per_joint_delta_cm,
)

# Upper-body 8-joint subset (must match round30_build_ild_subset.py).
UPPER_BODY_JOINT_INDICES: tuple[int, ...] = (
    12, 16, 17, 18, 19, 3, 6, 9,
)
UPPER_BODY_JOINT_NAMES: tuple[str, ...] = (
    "neck", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "spine1", "spine2", "spine3",
)


SWAP_NEUTRAL_TEXT = "a person stands still"
SWAP_ANTONYM_TEXT = "a person walks rapidly"

PERTURBATIONS_DEFAULT: tuple[str, ...] = (
    "baseline",
    "text_zero",
    "text_swap_neutral",
    "text_swap_antonym",
    "text_shuffle_token",
)


@dataclass(slots=True)
class TextPerturbationResult:
    """Per-clip per-perturbation row."""
    variant_id: str
    subset: str
    seq_id: str
    perturbation: str
    upper_body_mean_cm: float
    upper_body_p95_cm: float
    full_body_mean_cm: float
    full_body_p95_cm: float


def _upper_body_delta_cm(
    base_joints: np.ndarray, pert_joints: np.ndarray,
) -> tuple[float, float]:
    """Mean + p95 over (valid frames × 8 upper-body joints) in cm."""
    err = _per_joint_delta_cm(base_joints, pert_joints)            # (T, 22)
    if err.size == 0:
        return float("nan"), float("nan")
    sub = err[:, list(UPPER_BODY_JOINT_INDICES)]                   # (T, 8)
    return float(sub.mean()), float(np.percentile(sub, 95))


def _full_body_delta_cm(
    base_joints: np.ndarray, pert_joints: np.ndarray,
) -> tuple[float, float]:
    err = _per_joint_delta_cm(base_joints, pert_joints)
    if err.size == 0:
        return float("nan"), float("nan")
    return float(err.mean()), float(np.percentile(err, 95))


def _shuffle_text_tokens(
    text_features: "torch.Tensor",
    key_padding_mask: "torch.Tensor",
    rng: np.random.RandomState,
) -> "torch.Tensor":
    """Permute the non-padded token positions within each batch row.

    text_features : (B, 77, D); key_padding_mask : (B, 77) bool, True at
    padded positions.
    """
    import torch
    out = text_features.clone()
    for b in range(text_features.shape[0]):
        valid = (~key_padding_mask[b]).nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() < 2:
            continue
        perm_np = rng.permutation(valid.cpu().numpy())
        perm = torch.from_numpy(perm_np).to(valid.device)
        out[b, valid] = text_features[b, perm]
    return out


def _build_perturbed_cond(
    base_cond: dict,
    pert: str,
    *,
    text_zero_template: "torch.Tensor | None",
    swap_text_features_cache: dict[str, "torch.Tensor"],
    base_text_kpm: "torch.Tensor",
    swap_text_kpm_cache: dict[str, "torch.Tensor"],
    rng: np.random.RandomState,
) -> dict:
    """Return a copy of base_cond with only the ``text`` key perturbed.

    For swap perturbations, ``swap_text_features_cache`` contains the
    pre-computed text features for the SWAP_* strings (computed once at
    main() startup against the same clip_model).
    """
    import torch
    out = dict(base_cond)
    if "text" not in base_cond:
        raise KeyError(
            "round30 text probe expects 'text' in cond. This config has "
            "cfg.model.denoiser.text_dim == 0 — there is no text condition "
            "to probe."
        )
    if pert == "baseline":
        return out
    if pert == "text_zero":
        out["text"] = torch.zeros_like(base_cond["text"])
        return out
    if pert in ("text_swap_neutral", "text_swap_antonym"):
        key = "neutral" if pert == "text_swap_neutral" else "antonym"
        out["text"] = swap_text_features_cache[key].to(
            device=base_cond["text"].device,
            dtype=base_cond["text"].dtype,
        )
        return out
    if pert == "text_shuffle_token":
        out["text"] = _shuffle_text_tokens(
            base_cond["text"], base_text_kpm, rng,
        )
        return out
    raise AssertionError(f"unreachable perturbation {pert!r}")


def _aggregate(
    rows: list[TextPerturbationResult],
) -> dict[str, dict[str, Any]]:
    """Aggregate over clips per perturbation.

    Returns:
        { perturbation: { 'n_clips': int,
                          'upper_body_mean_cm': float | None,
                          'upper_body_p95_cm': float | None,
                          'full_body_mean_cm': ... } }
    """
    by_pert: dict[str, list[TextPerturbationResult]] = {}
    for r in rows:
        by_pert.setdefault(r.perturbation, []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for pert, prows in by_pert.items():
        ub_means = [r.upper_body_mean_cm for r in prows
                    if math.isfinite(r.upper_body_mean_cm)]
        ub_p95s = [r.upper_body_p95_cm for r in prows
                   if math.isfinite(r.upper_body_p95_cm)]
        fb_means = [r.full_body_mean_cm for r in prows
                    if math.isfinite(r.full_body_mean_cm)]
        fb_p95s = [r.full_body_p95_cm for r in prows
                   if math.isfinite(r.full_body_p95_cm)]
        out[pert] = {
            "n_clips": len(prows),
            "upper_body_mean_cm": float(np.mean(ub_means)) if ub_means else None,
            "upper_body_p95_cm": float(np.mean(ub_p95s)) if ub_p95s else None,
            "full_body_mean_cm": float(np.mean(fb_means)) if fb_means else None,
            "full_body_p95_cm": float(np.mean(fb_p95s)) if fb_p95s else None,
        }
    return out


def _gate_verdict(
    ild_agg: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply E1 decision gates (see plan §E1)."""
    z = ild_agg.get("text_zero", {})
    sn = ild_agg.get("text_swap_neutral", {})
    sa = ild_agg.get("text_swap_antonym", {})
    z_ub = z.get("upper_body_mean_cm")
    sn_ub = sn.get("upper_body_mean_cm")
    sa_ub = sa.get("upper_body_mean_cm")

    reasons: list[str] = []
    label = "unknown"
    if z_ub is None:
        return {
            "label": "unknown",
            "reasons": ["text_zero result missing"],
            "recommended_next": "investigate",
        }
    reasons.append(f"text_zero upper-body delta = {z_ub:.2f} cm")
    if sn_ub is not None:
        reasons.append(f"text_swap_neutral = {sn_ub:.2f} cm")
    if sa_ub is not None:
        reasons.append(f"text_swap_antonym = {sa_ub:.2f} cm")

    swap_avg = None
    if sn_ub is not None and sa_ub is not None:
        swap_avg = (sn_ub + sa_ub) / 2.0

    if z_ub < 2.0 and (swap_avg is None or swap_avg < 2.0):
        label = "text_dead"
        reasons.append(
            "text_zero AND swap both < 2 cm → text effectively dead (H5)."
        )
        return {
            "label": label, "reasons": reasons,
            "recommended_next": "STOP — go to D4 (text architecture rework)",
        }
    if z_ub >= 5.0 and sa_ub is not None and sn_ub is not None:
        if sa_ub > sn_ub * 1.30:
            label = "text_semantically_alive"
            reasons.append(
                f"swap_antonym {sa_ub:.2f} cm > 1.30 × swap_neutral "
                f"{sn_ub:.2f} cm → text has semantic resolution"
            )
            return {
                "label": label, "reasons": reasons,
                "recommended_next": "continue to E2 + E3",
            }
        label = "text_responds_but_aspecific"
        reasons.append(
            "text reacts to zeroing but swap_antonym ≈ swap_neutral → "
            "text lacks semantic resolution"
        )
        return {
            "label": label, "reasons": reasons,
            "recommended_next": "continue to E2 + E3, but flag as caveat",
        }
    label = "text_partial"
    return {
        "label": label, "reasons": reasons,
        "recommended_next": "continue to E2 + E3, but flag as caveat",
    }


def _write_summary_md(
    out_path: Path,
    *,
    variant_id: str,
    ckpt: str,
    ild_n: int,
    control_n: int,
    ild_agg: dict[str, dict[str, Any]],
    control_agg: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    perturbations: list[str],
) -> None:
    def _fmt(x):
        if x is None:
            return "—"
        try:
            f = float(x)
        except (TypeError, ValueError):
            return str(x)
        if not math.isfinite(f):
            return "—"
        return f"{f:.2f}"

    L: list[str] = []
    a = L.append
    a(f"# Round-30 E1 text-condition probe — `{variant_id}`")
    a("")
    a(f"**Ckpt:** `{ckpt}`")
    a(f"**ILD val clips:** {ild_n}    **Control val clips:** {control_n}")
    a(f"**Perturbations:** {', '.join(perturbations)}")
    a("")
    a("## E1 verdict")
    a("")
    a(f"- **Label:** `{gate.get('label', 'unknown')}`")
    a(f"- **Recommended next:** {gate.get('recommended_next', '—')}")
    for r in gate.get("reasons", []):
        a(f"- {r}")
    a("")
    a("## ILD val — upper-body delta vs baseline (cm)")
    a("")
    a("| perturbation | n | upper_mean | upper_p95 | full_mean | full_p95 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for pert in perturbations:
        row = ild_agg.get(pert, {})
        a(
            f"| {pert} | {row.get('n_clips', 0)} | "
            f"{_fmt(row.get('upper_body_mean_cm'))} | "
            f"{_fmt(row.get('upper_body_p95_cm'))} | "
            f"{_fmt(row.get('full_body_mean_cm'))} | "
            f"{_fmt(row.get('full_body_p95_cm'))} |"
        )
    a("")
    a("## Control val — upper-body delta vs baseline (cm)")
    a("")
    a("| perturbation | n | upper_mean | upper_p95 | full_mean | full_p95 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for pert in perturbations:
        row = control_agg.get(pert, {})
        a(
            f"| {pert} | {row.get('n_clips', 0)} | "
            f"{_fmt(row.get('upper_body_mean_cm'))} | "
            f"{_fmt(row.get('upper_body_p95_cm'))} | "
            f"{_fmt(row.get('full_body_mean_cm'))} | "
            f"{_fmt(row.get('full_body_p95_cm'))} |"
        )
    a("")
    a("## How to read")
    a("")
    a("- `upper_mean` = mean Euclidean delta on the 8 upper-body joints "
      "(neck, L/R shoulder, L/R elbow, spine1/2/3) over valid frames, "
      "in cm vs the baseline sample with the same seed.")
    a("- `text_zero` is the canonical 'is text alive at all' test.")
    a("- Compare ILD vs control to see whether the model uses text more "
      "in idle clips. If text affects only control (contact / walking) "
      "clips and not ILD, text is alive globally but not engaged for the "
      "ILD failure mode — equivalent to dead in our scope.")
    a("")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _load_selection(p: Path) -> set[tuple[str, str]]:
    obj = json.loads(p.read_text("utf-8"))
    selected = obj.get("selected") or obj.get("clips") or []
    return {(e["subset"], e["seq_id"]) for e in selected}


def _run_one_subset(
    *,
    subset_name: str,                       # "ild" or "control" — label
    selection: set[tuple[str, str]],
    cfg,
    args,
    model,
    object_encoder,
    clip_model,
    stage1_norm,
    swap_text_features_cache: dict,
    device,
) -> tuple[list[TextPerturbationResult], dict[str, dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader
    from piano.data.dataset import collate_hoi
    from piano.inference.diagnostic_helpers import (
        _build_cond, _build_dataset, _fk_22joints,
    )
    from piano.utils.clip_utils import encode_text_per_token

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_hoi, num_workers=0,
    )
    rng = np.random.RandomState(args.seed)
    rows: list[TextPerturbationResult] = []
    seen: set[tuple[str, str]] = set()

    for batch in loader:
        subset = str(batch["subset"][0]); seq_id = str(batch["seq_id"][0])
        if (subset, seq_id) not in selection:
            continue
        seen.add((subset, seq_id))
        with torch.no_grad():
            cond, T = _build_cond(
                batch, model, object_encoder, clip_model, cfg, device,
                stage1_norm=stage1_norm,
            )
        seq_len = int(batch["seq_len"][0].item())
        valid_T = min(T, seq_len)
        rest_offsets = batch["rest_offsets"].to(device).float()

        # For text_shuffle_token we need the key_padding_mask of THIS clip.
        # Re-encode the original text to get it.
        _, base_text_kpm = encode_text_per_token(
            clip_model, batch["text"], device,
        )

        def _sample(cond_in):
            torch.manual_seed(args.seed)
            with torch.no_grad():
                return model.sample(
                    cond=cond_in, seq_length=T, cfg_scale=args.cfg_scale,
                )

        pred_base = _sample(cond)
        base_joints = _fk_22joints(pred_base, rest_offsets)[
            0, :valid_T
        ].cpu().numpy()

        for pert in args.perturbations:
            if pert == "baseline":
                rows.append(TextPerturbationResult(
                    variant_id=args.variant_id, subset=subset, seq_id=seq_id,
                    perturbation=pert,
                    upper_body_mean_cm=0.0, upper_body_p95_cm=0.0,
                    full_body_mean_cm=0.0, full_body_p95_cm=0.0,
                ))
                continue
            cond_pert = _build_perturbed_cond(
                cond, pert,
                text_zero_template=None,
                swap_text_features_cache=swap_text_features_cache,
                base_text_kpm=base_text_kpm,
                swap_text_kpm_cache={},
                rng=rng,
            )
            pred_pert = _sample(cond_pert)
            pert_joints = _fk_22joints(pred_pert, rest_offsets)[
                0, :valid_T
            ].cpu().numpy()
            ub_m, ub_p = _upper_body_delta_cm(base_joints, pert_joints)
            fb_m, fb_p = _full_body_delta_cm(base_joints, pert_joints)
            rows.append(TextPerturbationResult(
                variant_id=args.variant_id, subset=subset, seq_id=seq_id,
                perturbation=pert,
                upper_body_mean_cm=ub_m, upper_body_p95_cm=ub_p,
                full_body_mean_cm=fb_m, full_body_p95_cm=fb_p,
            ))

        if len(seen) % 4 == 0:
            print(
                f"  [text_probe:{subset_name}] {len(seen)}/{len(selection)} "
                f"clips, {len(rows)} rows so far"
            )
        if len(seen) >= len(selection):
            break

    missing = selection - seen
    if missing:
        print(
            f"[text_probe:{subset_name}] WARNING: {len(missing)} selected "
            f"clips not found in dataset; first 3: {list(missing)[:3]}",
            file=sys.stderr,
        )

    agg = _aggregate(rows)
    return rows, agg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round-30 E1 text-condition probe (ILD + control val).",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--ild-selection", type=Path, required=True,
        help="selection_val.json from round30_build_ild_subset.py",
    )
    parser.add_argument(
        "--control-selection", type=Path, required=True,
        help="selection_control.json from round30_build_ild_subset.py "
             "(this script reads only the val-bucket subset of it)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="val", choices=["train", "val"])
    parser.add_argument("--variant-id", default="r29_ns_a1_c41_s4_g1")
    parser.add_argument(
        "--perturbations", default=",".join(PERTURBATIONS_DEFAULT),
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.perturbations = [
        p.strip() for p in args.perturbations.split(",") if p.strip()
    ]
    for p in args.perturbations:
        if p not in PERTURBATIONS_DEFAULT:
            raise SystemExit(f"unknown perturbation {p!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Deferred imports — pure helpers stay torch-free.
    import torch
    from omegaconf import OmegaConf
    from piano.inference.diagnostic_helpers import (
        _build_model, _stage1_norm_for_cfg, extract_train_time_meta,
    )
    from piano.utils.clip_utils import encode_text_per_token, load_clip_text_encoder

    cfg = OmegaConf.load(args.config)
    if int(cfg.model.denoiser.get("text_dim", 0)) <= 0:
        raise SystemExit(
            f"FATAL: config {args.config} has text_dim == 0; text is not "
            "wired into the model. The text probe is meaningless here."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, object_encoder = _build_model(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_meta = extract_train_time_meta(state)
    model.load_state_dict(state.get("model", state))
    if "object_encoder" in state:
        object_encoder.load_state_dict(state["object_encoder"])
    elif (
        "extra_modules" in state
        and "object_encoder" in state["extra_modules"]
    ):
        object_encoder.load_state_dict(
            state["extra_modules"]["object_encoder"]
        )
    else:
        raise SystemExit(
            f"FATAL: ckpt {args.ckpt} missing object_encoder state."
        )
    clip_model = load_clip_text_encoder(
        device=device,
        model_name=str(cfg.model.text_encoder.clip_version),
        download_root=str(cfg.model.text_encoder.get(
            "download_root", "cache/clip")),
    )
    stage1_norm = _stage1_norm_for_cfg(cfg, device)
    model.eval()

    # Pre-compute swap text features once (same clip_model, deterministic).
    swap_features: dict[str, torch.Tensor] = {}
    for key, text in (("neutral", SWAP_NEUTRAL_TEXT),
                      ("antonym", SWAP_ANTONYM_TEXT)):
        feats, _ = encode_text_per_token(clip_model, [text], device)
        swap_features[key] = feats.float().cpu()      # (1, 77, D) — moved to device per use
    print(f"[text_probe] swap features cached: "
          f"neutral={tuple(swap_features['neutral'].shape)}, "
          f"antonym={tuple(swap_features['antonym'].shape)}")

    ild_sel = _load_selection(args.ild_selection)
    control_sel = _load_selection(args.control_selection)
    print(f"[text_probe] ILD selection: {len(ild_sel)} clips")
    print(f"[text_probe] control selection: {len(control_sel)} clips")

    # ILD pass.
    print(f"[text_probe] ── ILD val pass ──")
    ild_rows, ild_agg = _run_one_subset(
        subset_name="ild", selection=ild_sel,
        cfg=cfg, args=args, model=model, object_encoder=object_encoder,
        clip_model=clip_model, stage1_norm=stage1_norm,
        swap_text_features_cache=swap_features, device=device,
    )

    print(f"[text_probe] ── control val pass ──")
    control_rows, control_agg = _run_one_subset(
        subset_name="control", selection=control_sel,
        cfg=cfg, args=args, model=model, object_encoder=object_encoder,
        clip_model=clip_model, stage1_norm=stage1_norm,
        swap_text_features_cache=swap_features, device=device,
    )

    gate = _gate_verdict(ild_agg)

    out = {
        "variant_id": args.variant_id,
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "ild_selection": str(args.ild_selection),
        "control_selection": str(args.control_selection),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "n_ild_clips": len(ild_sel),
        "n_control_clips": len(control_sel),
        "perturbations": args.perturbations,
        "train_time": train_meta,
        "ild_aggregate": ild_agg,
        "control_aggregate": control_agg,
        "gate": gate,
        "ild_rows": [
            {
                "subset": r.subset, "seq_id": r.seq_id,
                "perturbation": r.perturbation,
                "upper_body_mean_cm": r.upper_body_mean_cm,
                "upper_body_p95_cm": r.upper_body_p95_cm,
                "full_body_mean_cm": r.full_body_mean_cm,
                "full_body_p95_cm": r.full_body_p95_cm,
            }
            for r in ild_rows
        ],
        "control_rows": [
            {
                "subset": r.subset, "seq_id": r.seq_id,
                "perturbation": r.perturbation,
                "upper_body_mean_cm": r.upper_body_mean_cm,
                "upper_body_p95_cm": r.upper_body_p95_cm,
                "full_body_mean_cm": r.full_body_mean_cm,
                "full_body_p95_cm": r.full_body_p95_cm,
            }
            for r in control_rows
        ],
    }
    out_json = args.output_dir / "text_probe_stats.json"
    out_md = args.output_dir / "text_probe_summary.md"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_summary_md(
        out_md,
        variant_id=args.variant_id, ckpt=str(args.ckpt),
        ild_n=len(ild_sel), control_n=len(control_sel),
        ild_agg=ild_agg, control_agg=control_agg, gate=gate,
        perturbations=args.perturbations,
    )
    print(f"[text_probe] wrote {out_json}")
    print(f"[text_probe] wrote {out_md}")
    print(f"[text_probe] verdict label: {gate['label']!r}")
    print(f"[text_probe] recommended next: {gate['recommended_next']}")
    # Exit 2 on text_dead so the launcher can short-circuit.
    if gate.get("label") == "text_dead":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
