"""Smoke test for the Round-29 Stage-2 condition + injection ablation matrix.

Verifies (prompt §5.6):
    1. Every generated config can instantiate a denoiser.
    2. Every condition builder returns finite tensors at the right shape.
    3. Shape/dim metadata equals the manifest.
    4. A synthetic forward pass works for every injection mode (J0-J4).
    5. Missing required condition keys raise clear errors.

This is the gating check before training. Run with ``--dry-run`` to
skip the slowest sub-checks (full model forward); CI / pre-flight uses
``--dry-run`` and the trainer launcher promotes it to a full check.

Reviewed prompt section 5.6 before implementing this group: yes.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from piano.data.stage2_oracle_conditions import (  # noqa: E402
    BODY_VARIANT_DIMS,
    COARSE_VARIANT_DIMS,
    INTERACTION_VARIANT_DIMS,
    SUPPORT_VARIANT_DIMS,
    build_body_refinement_condition,
    build_coarse_condition,
    build_interaction_condition,
    build_stage2_condition_bundle,
    build_support_condition,
)
from piano.models.motion_anchordiff import (  # noqa: E402
    AnchorDenoiser,
    AnchorDenoiserConfig,
    ZIntDims,
)
from piano.models.round29_cond_injection import (  # noqa: E402
    Round29CondInjectionConfig,
    Round29CondInjectionModule,
)


MANIFEST = ROOT / "analyses" / "round29_stage2_cond_ablation_manifest.json"


def _synth_joints(T: int = 32, seed: int = 0) -> np.ndarray:
    """Synthetic SMPL-22 joints (T, 22, 3) with some motion."""
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.05, size=(22, 3)).astype(np.float32)
    base[0] = [0.0, 0.9, 0.0]  # pelvis at hip height
    # Force shoulders / hips to be wide enough that _facing_angle_y is stable.
    base[16] = [-0.2, 1.3, 0.0]
    base[17] = [+0.2, 1.3, 0.0]
    base[1] = [-0.1, 0.8, 0.0]
    base[2] = [+0.1, 0.8, 0.0]
    base[7] = [-0.1, 0.05, 0.0]
    base[8] = [+0.1, 0.05, 0.0]
    base[20] = [-0.4, 1.2, 0.1]
    base[21] = [+0.4, 1.2, 0.1]
    base[12] = [0.0, 1.5, 0.0]
    out = np.zeros((T, 22, 3), dtype=np.float32)
    for t in range(T):
        delta = rng.normal(scale=0.02, size=(22, 3)).astype(np.float32) * (t / T)
        out[t] = base + delta
        out[t, 0, 0] = 0.02 * t   # pelvis walks slowly along x
    return out


def _synth_object(T: int = 32, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 7)
    obj_pos = rng.normal(scale=0.5, size=(T, 3)).astype(np.float32) + np.array(
        [0.0, 0.5, 0.5], dtype=np.float32,
    )
    obj_rot = rng.normal(scale=0.1, size=(T, 3)).astype(np.float32)
    return obj_pos, obj_rot


def _synth_contact(T: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 13)
    cs = (rng.random((T, 5)) > 0.6).astype(np.float32)
    return cs


def check_builders() -> dict[str, str]:
    """(1) condition builders return finite tensors at right shape."""
    out: dict[str, str] = {}
    T = 24
    joints = _synth_joints(T)
    obj_pos, obj_rot = _synth_object(T)
    contact = _synth_contact(T)

    for cv, expected_dim in COARSE_VARIANT_DIMS.items():
        arr, info = build_coarse_condition(joints, cv)
        assert arr.shape == (T, expected_dim), (cv, arr.shape, expected_dim)
        assert info["finite_frac"] == 1.0, (cv, info)
        # Frame-0 invariant for the key-joint deltas.
        if expected_dim >= 15:
            assert np.abs(arr[0, :15]).max() < 1e-5, (cv, arr[0, :15])
        out[f"coarse/{cv}"] = f"OK shape={tuple(arr.shape)}"

    for iv, expected_dim in INTERACTION_VARIANT_DIMS.items():
        arr, info = build_interaction_condition(
            joints, obj_pos, obj_rot, contact, variant=iv,
        )
        assert arr.shape == (T, expected_dim), (iv, arr.shape, expected_dim)
        assert info["finite_frac"] == 1.0, (iv, info)
        out[f"interaction/{iv}"] = f"OK shape={tuple(arr.shape)}"

    for sv, expected_dim in SUPPORT_VARIANT_DIMS.items():
        arr, info = build_support_condition(joints, variant=sv)
        assert arr.shape == (T, expected_dim), (sv, arr.shape, expected_dim)
        assert info["finite_frac"] == 1.0, (sv, info)
        out[f"support/{sv}"] = (
            f"OK shape={tuple(arr.shape)} phase_valid={info.get('phase_valid_frame_frac', 0):.2f} "
            f"footstep_valid={info.get('footstep_target_valid_frame_frac', 0):.2f}"
        )

    for bv, expected_dim in BODY_VARIANT_DIMS.items():
        arr, info = build_body_refinement_condition(joints, variant=bv)
        assert arr.shape == (T, expected_dim), (bv, arr.shape, expected_dim)
        assert info["finite_frac"] == 1.0, (bv, info)
        out[f"body/{bv}"] = f"OK shape={tuple(arr.shape)} active={info.get('active_joint_frac', 0):.2f}"

    return out


def check_bundle() -> str:
    """Round-trip through ``build_stage2_condition_bundle`` for FULL-DENSE."""
    T = 32
    joints = _synth_joints(T)
    obj_pos, obj_rot = _synth_object(T)
    contact = _synth_contact(T)
    bundle = build_stage2_condition_bundle(
        joints, coarse_variant="C41-current",
        interaction_variant="I3-contact-offset-masked",
        support_variant="S4-S1-phase-footstep",
        body_variant="B4-lowpass-residual-mask",
        object_positions=obj_pos, object_rotations=obj_rot,
        contact_state=contact,
    )
    assert bundle.coarse_extra is not None and bundle.coarse_extra.shape == (T, 18)
    assert bundle.interaction is not None and bundle.interaction.shape == (T, 8)
    assert bundle.support is not None and bundle.support.shape == (T, 13)
    assert bundle.body_refine is not None and bundle.body_refine.shape == (T, 20)
    return "OK"


def check_injection_module() -> dict[str, str]:
    """(4) Synthetic forward pass for every injection mode."""
    out: dict[str, str] = {}
    B, T, D = 2, 16, 128

    def _make(mode: str, per_family_modes=None) -> Round29CondInjectionModule:
        cfg = Round29CondInjectionConfig(
            coarse_extra_dim=18,
            interaction_dim=8,
            support_dim=13,
            body_refine_dim=20,
            injection_mode=mode,
            gate_bias_init=-1.0,
            per_family_modes=per_family_modes,
        )
        mod = Round29CondInjectionModule(cfg, d_model=D)
        mod.configure_adapter_layers(n_layers=4)
        return mod

    cond = {
        "stage2_coarse_extra": torch.randn(B, T, 18),
        "stage2_interaction":  torch.randn(B, T, 8),
        "stage2_support":      torch.randn(B, T, 13),
        "stage2_body_refine":  torch.randn(B, T, 20),
    }
    c_summary = torch.randn(B, D)
    h_in = torch.zeros(B, T, D)

    for mode in ("input_add", "gated_input", "adapter_only", "input_add_adapter"):
        mod = _make(mode)
        h = mod.apply_input_injection(h_in, cond, c_summary=c_summary)
        if mode == "adapter_only":
            # No input add expected.
            assert torch.allclose(h, h_in), f"adapter_only must not modify h: got max delta {(h-h_in).abs().max():.3e}"
        else:
            # Zero-init last Linear -> step-0 forward is bit-exact zero.
            assert torch.allclose(h, h_in, atol=0.0), (mode, (h - h_in).abs().max())
        # Per-layer adapter pass.
        seq = torch.cat([torch.zeros(B, 1, D), h], dim=1)
        for layer_idx in range(4):
            seq2 = mod.apply_per_layer_adapter(seq, layer_idx=layer_idx, motion_token_start=1)
            assert seq2.shape == seq.shape
            seq = seq2
        out[f"injection/{mode}"] = "OK"

    # J4 typed
    mod = _make("typed", per_family_modes={
        "coarse_extra": "input_add",
        "interaction":  "gated_input",
        "support":      "adapter_only",
        "body_refine":  "input_add_adapter",
    })
    h = mod.apply_input_injection(h_in, cond, c_summary=c_summary)
    assert torch.allclose(h, h_in, atol=0.0)
    out["injection/typed"] = "OK"

    # (5) Missing keys raise KeyError.
    mod = _make("input_add")
    short_cond = {k: v for k, v in cond.items() if k != "stage2_interaction"}
    try:
        mod.apply_input_injection(h_in, short_cond, c_summary=c_summary)
    except KeyError as exc:
        out["injection/missing-key"] = f"OK (raised KeyError: {exc.args[0][:40]}...)"
    else:
        raise AssertionError("Expected KeyError when stage2_interaction missing")

    return out


def check_config_loads(manifest_path: Path | None) -> dict[str, str]:
    """(1) Every generated config can be parsed (YAML well-formed)."""
    out: dict[str, str] = {}
    if manifest_path is None or not manifest_path.exists():
        out["manifest"] = "SKIP — no manifest on disk"
        return out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for v in manifest["variants"]:
        cfg_path = ROOT / v["config_path"]
        if not cfg_path.exists():
            out[v["variant_id"]] = f"SKIP — config missing at {cfg_path}"
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            out[v["variant_id"]] = f"FAIL YAML parse: {exc}"
            continue
        # (3) Manifest dims must equal config dims.
        d = cfg["model"]["denoiser"]
        expected = v["expected_dense_dims"]
        for fam_key, mfest_key in (
            ("r29_coarse_extra_dim", "coarse_extra"),
            ("r29_interaction_dim", "interaction"),
            ("r29_support_dim", "support"),
            ("r29_body_refine_dim", "body_refine"),
        ):
            if int(d[fam_key]) != int(expected[mfest_key]):
                out[v["variant_id"]] = (
                    f"FAIL dim mismatch {fam_key}: cfg={d[fam_key]} manifest={expected[mfest_key]}"
                )
                break
        else:
            out[v["variant_id"]] = "OK YAML + dims match manifest"
    return out


def check_model_forward() -> str:
    """(1) Denoiser can be constructed with the R29 branch active and
    accepts a synthetic forward pass.

    Build a minimal AnchorDenoiserConfig (v12 path) with R29 enabled,
    construct the model and run a forward through ``_forward_v12``
    using minimal cond tensors.
    """
    B, T, D = 1, 12, 64
    cfg = AnchorDenoiserConfig(
        motion_dim=135,
        z_int=ZIntDims(num_parts=5, phase_classes=3, support_classes=3),
        object_traj_dim=9,
        init_pose_dim=66,
        text_dim=512,
        object_token_dim=256,
        object_num_tokens=128,
        use_round29_cond_injection=True,
        r29_coarse_extra_dim=18,
        r29_interaction_dim=8,
        r29_support_dim=13,
        r29_body_refine_dim=20,
        r29_injection_mode="input_add_adapter",
        d_model=D,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=T,
    )
    model = AnchorDenoiser(cfg)
    model.eval()

    cond = {
        "z_int": torch.zeros(B, T, cfg.z_int.total),
        "object_world_traj": torch.zeros(B, T, cfg.object_traj_dim),
        "init_pose": torch.zeros(B, cfg.init_pose_dim),
        "text": torch.zeros(B, 4, cfg.text_dim),
        "object_tokens": torch.zeros(B, cfg.object_num_tokens, cfg.object_token_dim),
        "stage2_coarse_extra": torch.randn(B, T, 18),
        "stage2_interaction":  torch.randn(B, T, 8),
        "stage2_support":      torch.randn(B, T, 13),
        "stage2_body_refine":  torch.randn(B, T, 20),
    }
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    if out.shape != (B, T, cfg.motion_dim):
        raise AssertionError(f"forward shape {out.shape} != {(B, T, cfg.motion_dim)}")
    # Zero-init guarantee: step-0 forward should be (numerically) zero.
    max_abs = float(out.abs().max())
    return f"OK shape={tuple(out.shape)} step0_max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------
# Strict preflight (Codex post-review §P2)
# ---------------------------------------------------------------------------

def _resolve_strict_targets(manifest_path: Path) -> list[dict]:
    """Pick one variant per unique (C, I, S, B, injection_mode) combo so
    we don't redundantly re-instantiate identical-shape models."""
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen: set[tuple] = set()
    picked: list[dict] = []
    for v in manifest["variants"]:
        key = (
            v["coarse_variant"], v["interaction_variant"],
            v["support_variant"], v["body_variant"],
            v["injection_mode"], tuple(sorted((v["per_family_modes"] or {}).items())),
        )
        if key in seen:
            continue
        seen.add(key)
        picked.append(v)
    return picked


def check_strict_instantiate_configs(manifest_path: Path) -> dict[str, str]:
    """For each unique config shape, instantiate model + (optional) dataset
    from the generated YAML. Hard-fail if any config errors out at build
    time. Dataset instantiation is best-effort — when local data roots
    are missing we report SKIP with the missing path."""
    from omegaconf import OmegaConf

    out: dict[str, str] = {}
    targets = _resolve_strict_targets(manifest_path)
    if not targets:
        out["manifest"] = "SKIP — no manifest on disk"
        return out

    # Lazy imports — strict mode is slow.
    from piano.training.train_anchordiff import _build_dataset
    from piano.models.motion_anchordiff import (
        AnchorDenoiser,
        AnchorDenoiserConfig,
        ZIntDims,
    )

    for v in targets:
        vid = v["variant_id"]
        cfg_path = ROOT / v["config_path"]
        if not cfg_path.exists():
            out[vid] = f"FAIL config missing at {cfg_path}"
            continue
        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception as exc:  # noqa: BLE001
            out[vid] = f"FAIL OmegaConf.load: {exc}"
            continue

        # (a) Instantiate model from config.
        try:
            d = cfg.model.denoiser
            z_dims = ZIntDims(
                num_parts=int(cfg.model.z_int.num_parts),
                phase_classes=int(cfg.model.z_int.phase_classes),
                support_classes=int(cfg.model.z_int.support_classes),
            )
            denoiser_cfg = AnchorDenoiserConfig(
                motion_dim=int(d.motion_dim),
                z_int=z_dims,
                object_traj_dim=int(d.object_traj_dim),
                init_pose_dim=int(d.init_pose_dim),
                text_dim=int(d.text_dim),
                object_token_dim=int(d.object_token_dim),
                object_num_tokens=int(d.object_num_tokens),
                stage1_coarse_dim=int(d.get("stage1_coarse_dim", 0)),
                use_round29_cond_injection=bool(
                    d.get("use_round29_cond_injection", False)
                ),
                r29_coarse_extra_dim=int(d.get("r29_coarse_extra_dim", 0)),
                r29_interaction_dim=int(d.get("r29_interaction_dim", 0)),
                r29_support_dim=int(d.get("r29_support_dim", 0)),
                r29_body_refine_dim=int(d.get("r29_body_refine_dim", 0)),
                r29_injection_mode=str(d.get("r29_injection_mode", "input_add")),
                r29_gate_bias_init=float(d.get("r29_gate_bias_init", -1.0)),
                r29_per_family_modes=(
                    dict(d.get("r29_per_family_modes"))
                    if d.get("r29_per_family_modes") is not None else None
                ),
                d_model=int(d.d_model),
                n_layers=int(d.n_layers),
                n_heads=int(d.n_heads),
                ff_mult=int(d.ff_mult),
                dropout=float(d.dropout),
                max_seq_length=int(cfg.data.max_seq_length),
            )
            _ = AnchorDenoiser(denoiser_cfg)
        except Exception as exc:  # noqa: BLE001
            out[vid] = f"FAIL model build: {exc}"
            continue

        # (b) Instantiate dataset if dataset roots exist locally.
        dataset_status = "model OK"
        try:
            roots_ok = True
            for ds in cfg.data.datasets:
                if not Path(str(ds.root)).exists():
                    roots_ok = False
                    dataset_status += f" / SKIP dataset (root missing: {ds.root})"
                    break
            if roots_ok:
                try:
                    dataset = _build_dataset(cfg, bucket="train", augment=False)
                    if len(dataset) == 0:
                        dataset_status += " / SKIP dataset (empty after filtering)"
                    else:
                        sample = dataset[0]
                        # Verify active stage2_* keys appear in the sample.
                        missing_keys = []
                        for fam_key, dim_key in (
                            ("stage2_coarse_extra", "r29_coarse_extra_dim"),
                            ("stage2_interaction",  "r29_interaction_dim"),
                            ("stage2_support",      "r29_support_dim"),
                            ("stage2_body_refine",  "r29_body_refine_dim"),
                        ):
                            expected_dim = int(d.get(dim_key, 0))
                            if expected_dim > 0:
                                if fam_key not in sample:
                                    missing_keys.append(fam_key)
                                elif sample[fam_key].shape[-1] != expected_dim:
                                    missing_keys.append(
                                        f"{fam_key} dim {sample[fam_key].shape[-1]} != {expected_dim}"
                                    )
                        if missing_keys:
                            out[vid] = f"FAIL sample missing keys: {missing_keys}"
                            continue
                        dataset_status += f" / dataset sample OK (T={sample['motion'].shape[0]})"
                except Exception as exc:  # noqa: BLE001
                    out[vid] = f"FAIL dataset build/sample: {exc}"
                    continue
        except Exception as exc:  # noqa: BLE001
            out[vid] = f"FAIL dataset preflight: {exc}"
            continue

        out[vid] = f"OK {dataset_status}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the model-forward and strict checks (fastest).")
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "Strict preflight: for every unique (C, I, S, B, inject) "
            "combo in the manifest, instantiate the model from the "
            "generated YAML and (when dataset roots exist locally) "
            "pull one sample to verify the typed stage2_* keys + shapes. "
            "The launcher should call this before training."
        ),
    )
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()

    print("=== R29 Smoke Test ===")
    failures: list[str] = []

    def _section(name: str, fn) -> None:
        print(f"\n[{name}]")
        try:
            res = fn()
            if isinstance(res, dict):
                for k, v in res.items():
                    print(f"  - {k}: {v}")
                    if str(v).startswith("FAIL"):
                        failures.append(f"{name}/{k}: {v}")
            else:
                print(f"  - {res}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"  FAIL: {exc}")
            traceback.print_exc()

    _section("builders", check_builders)
    _section("bundle", check_bundle)
    _section("injection_module", check_injection_module)
    _section("config_files", lambda: check_config_loads(Path(args.manifest)))
    if not args.dry_run:
        _section("model_forward", check_model_forward)
    else:
        print("\n[model_forward] SKIPPED (--dry-run)")

    if args.strict:
        _section(
            "strict_instantiate_configs",
            lambda: check_strict_instantiate_configs(Path(args.manifest)),
        )

    print()
    if failures:
        print(f"=== {len(failures)} FAILURE(S) ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
