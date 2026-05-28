"""Smoke test: PB1 forward + backward must produce gradients on
cond_summary_mlp params.

Not a pytest — meant to run manually before kicking off the real train.
The test ensures: (1) model builds from the actual PB1 YAML, (2) a
forward + backward on synthetic data produces non-NaN loss and
non-zero gradients on the new cond_summary_mlp.

Usage: python tests/_smoke_pb1_train_step.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    cfg_path = ROOT / "configs" / "training" / "anchordiff_r29_pb_a1_adaln_s4.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    d = cfg["model"]["denoiser"]

    from piano.models.motion_anchordiff import (
        AnchorDenoiser, AnchorDenoiserConfig,
    )

    denoiser_cfg = AnchorDenoiserConfig(
        motion_dim=d["motion_dim"],
        object_traj_dim=d["object_traj_dim"],
        init_pose_dim=d["init_pose_dim"],
        text_dim=d["text_dim"],
        object_token_dim=d["object_token_dim"],
        object_num_tokens=d["object_num_tokens"],
        stage1_coarse_dim=d["stage1_coarse_dim"],
        use_round29_cond_injection=d["use_round29_cond_injection"],
        r29_coarse_extra_dim=d["r29_coarse_extra_dim"],
        r29_interaction_dim=d["r29_interaction_dim"],
        r29_support_dim=d["r29_support_dim"],
        r29_body_refine_dim=d["r29_body_refine_dim"],
        r29_injection_mode=d["r29_injection_mode"],
        r29_per_family_modes=d.get("r29_per_family_modes"),
        r29_zero_init_adapters=d["r29_zero_init_adapters"],
        r29_use_cond_adaln=d["r29_use_cond_adaln"],
        r29_adaln_families=list(d["r29_adaln_families"]),
        r29_adaln_pool=d["r29_adaln_pool"],
        d_model=d["d_model"],
        n_layers=d["n_layers"],
        n_heads=d["n_heads"],
        ff_mult=d["ff_mult"],
        dropout=d["dropout"],
        max_seq_length=cfg["data"]["max_seq_length"],
    )

    print(f"[smoke] building denoiser ({denoiser_cfg.d_model}d / "
          f"{denoiser_cfg.n_layers}L / r29_use_cond_adaln={denoiser_cfg.r29_use_cond_adaln})")
    model = AnchorDenoiser(denoiser_cfg)
    model.train()

    # Verify the new submodule is registered + zero-init.
    cs = model.v12_cond_summary
    assert cs.use_cond_summary_mlp, "PB1 should have cond_summary_mlp"
    assert cs.cond_summary_mlp is not None
    w = cs.cond_summary_mlp[-1].weight
    b = cs.cond_summary_mlp[-1].bias
    print(f"[smoke] cond_summary_mlp last Linear: weight.norm={w.norm():.4e}, bias.norm={b.norm():.4e}")
    assert w.norm().item() == 0.0
    assert b.norm().item() == 0.0

    # Build a tiny synthetic batch (T trimmed to keep the test fast).
    B, T = 2, 16
    x_t = torch.randn(B, T, denoiser_cfg.motion_dim)
    t = torch.randint(0, 1000, (B,))
    cond = {
        "object_world_traj": torch.randn(B, T, 9),
        "object_tokens":     torch.randn(B, denoiser_cfg.object_num_tokens, denoiser_cfg.object_token_dim),
        "text":              torch.randn(B, 4, denoiser_cfg.text_dim),
        "init_pose":         torch.randn(B, denoiser_cfg.init_pose_dim),
        "stage1_coarse":     torch.randn(B, T, denoiser_cfg.stage1_coarse_dim),
        "stage2_coarse_extra": torch.randn(B, T, denoiser_cfg.r29_coarse_extra_dim),
        "stage2_support":      torch.zeros(B, T, denoiser_cfg.r29_support_dim),
    }
    # Make support's walking_mask non-trivially non-zero so the
    # support_walking_mean pool produces a non-zero weighted sum (i.e.
    # the new code path is actually exercised under the synthetic batch).
    cond["stage2_support"][:, :, 4] = 1.0                    # walking_mask = 1 everywhere
    cond["stage2_support"][:, :, :4] = torch.randn(B, T, 4)  # stance/height channels
    cond["stage2_support"][:, :, 5:] = torch.randn(B, T, 8)  # phase + footstep channels

    print(f"[smoke] forward + backward (B={B}, T={T})")
    out = model(x_t, t, cond, cond_drop_mask=None)
    assert out.shape == (B, T, denoiser_cfg.motion_dim), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in output"
    print(f"[smoke] output stats: mean={out.mean():.4e}, std={out.std():.4e}, "
          f"abs_max={out.abs().max():.4e}")

    # Step 0: output should be 0 (V12FinalLayer zero-init + AdaLN-Zero
    # blocks + cond_summary_mlp zero-init).
    assert out.abs().max().item() == 0.0, (
        f"step-0 output should be zero (zero-init invariant); got abs_max={out.abs().max()}"
    )

    # First: at init, output is exactly 0 (all zero-init paths), so grad
    # is 0 everywhere. That's expected — the test of bit-identical
    # step-0 forward is what locks PB1's invariant.
    # To check that gradients DO flow under realistic training conditions,
    # de-zero the V12FinalLayer + AdaLN gates and re-forward. This
    # simulates "the model has trained for a few steps."

    print("[smoke] simulating mid-training: de-zero V12FinalLayer + AdaLN gates")
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.v12_final_layer.linear.weight)
        torch.nn.init.xavier_uniform_(model.v12_final_layer.adaLN_modulation[-1].weight)
        for block in model.v12_blocks:
            torch.nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)

    out = model(x_t, t, cond, cond_drop_mask=None)
    target = torch.randn_like(out)
    loss = (out - target).pow(2).mean()
    print(f"[smoke] mid-training loss: {loss.item():.4e}")
    assert torch.isfinite(loss), "non-finite loss"
    model.zero_grad()
    loss.backward()

    # DEBUG: dump grad norms for every param to find the issue.
    print("[smoke] r29_inject grad norms:")
    for name, p in model.r29_inject.named_parameters():
        gn = "None" if p.grad is None else f"{p.grad.norm().item():.4e}"
        print(f"    {name}: grad_norm={gn}")
    print("[smoke] v12_input_proj grad norms:")
    for name, p in model.v12_input_proj.named_parameters():
        gn = "None" if p.grad is None else f"{p.grad.norm().item():.4e}"
        print(f"    {name}: grad_norm={gn}")
    print("[smoke] v12_cond_summary grad norms:")
    for name, p in model.v12_cond_summary.named_parameters():
        gn = "None" if p.grad is None else f"{p.grad.norm().item():.4e}"
        print(f"    {name}: grad_norm={gn}")

    # Old A1 path — proj.support[-1].weight (final Linear) should have a
    # non-zero grad. Note: proj.support[0].weight grad is 0 at init because
    # it's gated by the zero-init final Linear of proj.support — same
    # behavior as the original A1 code.
    final_w = model.r29_inject.proj.support[-1].weight
    assert final_w.grad is not None and final_w.grad.norm() > 0, (
        f"proj.support[-1].weight has no grad"
    )
    print(f"[smoke] proj.support[-1].weight grad norm: {final_w.grad.norm():.4e}")

    # PB1 path — cond_summary_mlp final Linear bias must have grad
    # (the final Linear weight grad starts at 0 because the input to it
    # passes through SiLU(0)=0 once everything zero-init upstream, but
    # the bias grad is always reachable).
    cs_b = model.v12_cond_summary.cond_summary_mlp[-1].bias
    assert cs_b.grad is not None and cs_b.grad.norm() > 0, (
        f"cond_summary_mlp[-1].bias has no grad (PB1 path not connected!)"
    )
    print(f"[smoke] cond_summary_mlp[-1].bias grad norm: {cs_b.grad.norm():.4e}")

    print("[smoke] PB1 forward + backward OK — PB1 path gradients flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
