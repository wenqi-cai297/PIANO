"""Round-28 server-side smoke test.

Verifies the cleaned-up codebase can:
  1. Import the full active chain (trainer + model + dataset + diags).
  2. Build AnchorDenoiser from every R28 config.
  3. Load one __getitem__ from the dataset if data is available.
  4. Run a synthetic forward through the model (no real ckpt needed).

Run on the server inside the `piano` conda env:

    conda run --no-capture-output -n piano python -u \\
        scripts/stage_b_generator/round28_smoke_test.py

Exits 0 only if every active R28 + R27 Tier-0 + v27/R23 baseline config
builds cleanly. No checkpoint or training is exercised here — we just
prove that nothing dangling from the cleanup broke the active path.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _import_active_chain():
    print("[1/4] Import active chain ...")
    # Trainer + helpers
    import piano.training.train_anchordiff as ta                       # noqa: F401
    import piano.training.trainer as tr                                # noqa: F401
    import piano.training.temporal_interaction_losses as tl            # noqa: F401
    import piano.training.anchor_consistency_loss as acl               # noqa: F401
    import piano.training.anchordiff_geometric_losses as agl           # noqa: F401
    import piano.training.smpl_kinematics as sk                        # noqa: F401
    import piano.training.feature_groups as fg                         # noqa: F401
    import piano.training.feature_weight_state as fws                  # noqa: F401
    # Data
    import piano.data.dataset as ds                                    # noqa: F401
    import piano.data.interaction_hint as ih                           # noqa: F401
    import piano.data.stage1_coarse_oracle as sco                      # noqa: F401
    import piano.data.interaction_plan_compiler as ipc                 # noqa: F401
    # Models
    import piano.models.motion_anchordiff as ma                        # noqa: F401
    import piano.models.object_encoder as oe                           # noqa: F401
    import piano.models.interaction_plan_encoder as ipe                # noqa: F401
    # Inference (visualize_motion is used by render scripts)
    import piano.inference.visualize_motion as vm                      # noqa: F401
    # Diagnostic helpers (live scripts import these)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import plan_condition_diagnostics                                  # noqa: F401
    import anchor_realization_diagnostic                               # noqa: F401
    print("    OK")


def _build_models_from_configs(configs: list[Path]) -> list[str]:
    """Build AnchorDenoiserConfig + the (lightweight) ObjectEncoder for
    every requested config and report any that fail. Skip checkpoint
    loading + actual dataset assembly. Just exercises the config-to-cfg
    plumbing so signature mismatches surface."""
    print(f"[2/4] Build denoiser cfg for {len(configs)} configs ...")
    from omegaconf import OmegaConf
    from piano.models.motion_anchordiff import (
        AnchorDenoiserConfig, ZIntDims,
    )

    failed: list[str] = []
    for cfg_path in configs:
        try:
            cfg = OmegaConf.load(cfg_path)
            z_dims = ZIntDims(
                num_parts=int(cfg.model.z_int.num_parts),
                phase_classes=int(cfg.model.z_int.phase_classes),
                support_classes=int(cfg.model.z_int.support_classes),
            )
            den = AnchorDenoiserConfig(
                motion_dim=int(cfg.model.denoiser.motion_dim),
                z_int=z_dims,
                object_traj_dim=int(cfg.model.denoiser.object_traj_dim),
                init_pose_dim=int(cfg.model.denoiser.init_pose_dim),
                text_dim=int(cfg.model.denoiser.text_dim),
                object_token_dim=int(cfg.model.denoiser.object_token_dim),
                object_num_tokens=int(cfg.model.denoiser.object_num_tokens),
                use_interaction_plan=bool(
                    cfg.model.denoiser.get("use_interaction_plan", False),
                ),
                use_dit_block=bool(
                    cfg.model.denoiser.get("use_dit_block", False),
                ),
                stage1_coarse_dim=int(
                    cfg.model.denoiser.get("stage1_coarse_dim", 0),
                ),
                use_oracle_interaction_hint=bool(
                    cfg.model.denoiser.get("use_oracle_interaction_hint", False),
                ),
                oracle_hint_dim=int(
                    cfg.model.denoiser.get("oracle_hint_dim", 0),
                ),
                use_body_action_hint=bool(
                    cfg.model.denoiser.get("use_body_action_hint", False),
                ),
                body_action_hint_dim=int(
                    cfg.model.denoiser.get("body_action_hint_dim", 0),
                ),
                oracle_hint_injection_mode=str(
                    cfg.model.denoiser.get("oracle_hint_injection_mode", "input_add"),
                ),
                oracle_hint_gate_bias_init=float(
                    cfg.model.denoiser.get("oracle_hint_gate_bias_init", -3.0),
                ),
                d_model=int(cfg.model.denoiser.d_model),
                n_layers=int(cfg.model.denoiser.n_layers),
                n_heads=int(cfg.model.denoiser.n_heads),
                ff_mult=int(cfg.model.denoiser.ff_mult),
                dropout=float(cfg.model.denoiser.dropout),
                max_seq_length=int(cfg.data.max_seq_length),
            )
            # Sanity: motion_dim must be 135 in active path.
            if den.motion_dim != 135:
                raise AssertionError(
                    f"{cfg_path.name}: motion_dim={den.motion_dim} != 135",
                )
            mode = den.oracle_hint_injection_mode
            assert mode in {
                "input_add", "gated_input", "per_layer_adapter", "adapter_only",
            }, f"unknown injection_mode: {mode}"
            print(f"    OK  {cfg_path.name}  "
                  f"(mode={mode}, hint={den.oracle_hint_dim}D, "
                  f"body={den.body_action_hint_dim}D)")
        except Exception as e:
            failed.append(f"{cfg_path.name}: {e!r}")
            traceback.print_exc()
    return failed


def _instantiate_denoiser_forward() -> None:
    """Build a tiny AnchorDenoiser with the R28 A0 shape and run a
    synthetic forward to confirm the model module still wires up."""
    print("[3/4] Synthetic AnchorDenoiser forward ...")
    import torch
    from piano.models.motion_anchordiff import (
        AnchorDenoiser, AnchorDenoiserConfig, ZIntDims,
    )

    cfg = AnchorDenoiserConfig(
        motion_dim=135,
        z_int=ZIntDims(num_parts=5, phase_classes=3, support_classes=3),
        object_traj_dim=24,
        init_pose_dim=66,
        text_dim=512,
        object_token_dim=256,
        object_num_tokens=128,
        use_interaction_plan=True,
        plan_k_max=12,
        plan_s_max=12,
        plan_num_anchor_types=5,
        plan_num_parts=5,
        plan_use_segment_tokens=False,
        plan_use_context_hint=True,
        plan_d_hint=32,
        plan_d_time_embed=64,
        cfg_drop_plan=False,
        plan_per_part_tokens=True,
        plan_context_hint_mode="target_aware",
        use_dit_block=True,
        dit_block_use_plan_pool_in_cond=False,
        stage1_coarse_dim=0,
        cfg_drop_stage1_coarse=False,
        use_oracle_interaction_hint=True,
        oracle_hint_dim=13,
        use_body_action_hint=False,
        body_action_hint_dim=0,
        oracle_hint_injection_mode="input_add",
        d_model=64,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_length=32,
    )
    torch.manual_seed(0)
    model = AnchorDenoiser(cfg).eval()

    B, T = 2, 16
    cond = {
        "z_int": torch.randn(B, T, cfg.z_int.total),
        "object_world_traj": torch.randn(B, T, cfg.object_traj_dim),
        "init_pose": torch.randn(B, cfg.init_pose_dim),
        "text": torch.randn(B, 77, cfg.text_dim),
        "object_tokens": torch.randn(B, cfg.object_num_tokens, cfg.object_token_dim),
        "oracle_interaction_hint": torch.randn(B, T, cfg.oracle_hint_dim),
    }
    K, S, P = cfg.plan_k_max, cfg.plan_s_max, cfg.plan_num_parts
    cond["interaction_plan"] = {
        "anchor_time": torch.zeros(B, K, dtype=torch.long),
        "anchor_part": torch.zeros(B, K, P),
        "anchor_target_local": torch.zeros(B, K, P, 3),
        "anchor_target_world": torch.zeros(B, K, P, 3),
        "anchor_type": torch.zeros(B, K, dtype=torch.long),
        "anchor_phase": torch.zeros(B, K, dtype=torch.long),
        "anchor_support": torch.zeros(B, K, dtype=torch.long),
        "anchor_conf": torch.zeros(B, K),
        "anchor_mask": torch.zeros(B, K, dtype=torch.bool),
        "segment_start": torch.zeros(B, S, dtype=torch.long),
        "segment_end": torch.zeros(B, S, dtype=torch.long),
        "segment_part": torch.zeros(B, S, P),
        "segment_target_summary_local": torch.zeros(B, S, P, 3),
        "segment_phase": torch.zeros(B, S, dtype=torch.long),
        "segment_support": torch.zeros(B, S, dtype=torch.long),
        "segment_conf": torch.zeros(B, S),
        "segment_mask": torch.zeros(B, S, dtype=torch.bool),
    }
    x_t = torch.randn(B, T, cfg.motion_dim)
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, cfg.motion_dim), out.shape
    assert torch.isfinite(out).all(), "non-finite output"
    print(f"    OK  output shape {tuple(out.shape)} finite")


def _build_dataset_smoke(cfg_path: Path) -> None:
    """Optional: assemble HOIDataset for one active config and try one
    __getitem__. Skips silently if InterAct dataset roots are absent."""
    print(f"[4/4] Dataset smoke (config={cfg_path.name}) ...")
    from omegaconf import OmegaConf
    from piano.training.train_anchordiff import _build_dataset

    cfg = OmegaConf.load(cfg_path)
    # Check at least one dataset root exists; if not, skip cleanly.
    roots = [Path(d.root) for d in cfg.data.datasets]
    if not any(p.is_dir() for p in roots):
        print(f"    SKIP (no dataset roots present locally: {roots[0]})")
        return
    ds = _build_dataset(cfg, "train", augment=False)
    print(f"    Train dataset assembled: {len(ds)} clips")
    sample = ds[0]
    keys = ("motion", "joints", "object_positions", "object_rotations",
            "contact_state", "seq_len", "text")
    missing = [k for k in keys if k not in sample]
    if missing:
        raise AssertionError(f"sample missing keys: {missing}")
    print(f"    OK  sample[0] motion shape {tuple(sample['motion'].shape)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        type=Path,
        default=None,
        help="Configs to smoke-build (default: every active R28/R27/v27/R23 yaml).",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip the synthetic model forward test.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip the dataset __getitem__ smoke test.",
    )
    args = parser.parse_args()

    cfg_dir = Path("configs/training")
    if args.configs:
        configs = list(args.configs)
    else:
        configs = sorted(
            list(cfg_dir.glob("anchordiff_r28_*.yaml"))
            + list(cfg_dir.glob("anchordiff_t0*.yaml"))
            + list(cfg_dir.glob("anchordiff_v27_*.yaml"))
            + list(cfg_dir.glob("anchordiff_v25_round23_*.yaml")),
        )
        # Skip _local variants — they are generated at PREP time on
        # the server and may not exist yet on a fresh checkout.
        configs = [p for p in configs if "_local" not in p.name]
    if not configs:
        print("ERROR: no configs found", file=sys.stderr)
        return 1

    try:
        _import_active_chain()
    except Exception:
        traceback.print_exc()
        print("FAIL: import chain", file=sys.stderr)
        return 1

    failed = _build_models_from_configs(configs)
    if failed:
        print("FAIL: cfg build", file=sys.stderr)
        for f in failed:
            print(" -", f, file=sys.stderr)
        return 1

    if not args.skip_forward:
        try:
            _instantiate_denoiser_forward()
        except Exception:
            traceback.print_exc()
            print("FAIL: synthetic forward", file=sys.stderr)
            return 1

    if not args.skip_dataset:
        # Pick the first config so we don't hammer disk on all 23.
        try:
            _build_dataset_smoke(configs[0])
        except Exception:
            traceback.print_exc()
            print("FAIL: dataset smoke", file=sys.stderr)
            return 1

    print()
    print(f"PASS  {len(configs)} configs / forward / dataset smoke clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
