"""Smoke-test the v11 encoder modes.

Run via:
    conda run -n piano --no-capture-output python \
      scripts/stage_b_generator/_smoke_v11_encoder.py
"""
import torch
import numpy as np
from piano.data.interaction_plan_compiler import (
    compile_interaction_plan, InteractionPlanCompilerConfig, collate_interaction_plans,
)
from piano.models.interaction_plan_encoder import (
    InteractionPlanEncoder, InteractionPlanEncoderConfig, PlanCrossAttentionBlock,
)


def main() -> None:
    cfg_compile = InteractionPlanCompilerConfig()
    T = 100
    contact = np.zeros((T, 5), dtype=np.float32)
    contact[10:40, 0] = 1.0
    contact[40:65, 1] = 1.0
    contact[5:80, 4] = 1.0  # pelvis active throughout
    target = np.random.randn(T, 5, 3).astype(np.float32) * 0.1
    phase = np.zeros((T, 3), dtype=np.float32)
    phase[:50, 0] = 1.0
    phase[50:, 1] = 1.0
    support = np.zeros((T, 3), dtype=np.float32)
    support[:, 0] = 1.0
    obj_p = np.tile(np.array([1.0, 0.0, 2.0], dtype=np.float32), (T, 1))
    obj_r = np.zeros((T, 3), dtype=np.float32)

    p1 = compile_interaction_plan(contact, target, phase, support, obj_p, obj_r, T, cfg_compile)
    p2 = compile_interaction_plan(contact, target, phase, support, obj_p, obj_r, T, cfg_compile)
    batched = collate_interaction_plans([p1, p2])
    plan_t = {k: torch.from_numpy(v) for k, v in batched.items()}

    print("=== v10 baseline (per-anchor, time_only hint) ===")
    enc_cfg = InteractionPlanEncoderConfig(
        d_model=512, per_part_tokens=False, context_hint_mode="time_only",
    )
    encoder = InteractionPlanEncoder(enc_cfg)
    plan_tokens, plan_mask, hint = encoder(plan_t, T=T)
    print(f"plan_tokens {plan_tokens.shape}  mask sum/clip = {plan_mask.sum(1).tolist()}")
    print(f"hint        {hint.shape}")
    print(f"finite hint: {torch.isfinite(hint).all().item()}")

    print("\n=== v11 per-part tokens, target_aware hint ===")
    enc_cfg2 = InteractionPlanEncoderConfig(
        d_model=512, per_part_tokens=True, context_hint_mode="target_aware",
    )
    encoder2 = InteractionPlanEncoder(enc_cfg2)
    plan_tokens2, plan_mask2, hint2 = encoder2(plan_t, T=T)
    print(f"plan_tokens {plan_tokens2.shape}  (expected B=2, K_max*P=60)")
    print(f"valid_per_clip = {plan_mask2.sum(1).tolist()}")
    print(f"hint        {hint2.shape}")
    print(f"finite tokens: {torch.isfinite(plan_tokens2).all().item()}")
    print(f"finite hint:   {torch.isfinite(hint2).all().item()}")

    # Check the cross-attn still works with the new mask shape
    print("\n=== Cross-attention forward (per-part tokens) ===")
    xa = PlanCrossAttentionBlock(d_model=512, n_heads=4)
    motion = torch.randn(2, T, 512)
    out = xa(motion, plan_tokens2, plan_mask2)
    print(f"xattn out {out.shape}  finite={torch.isfinite(out).all().item()}")

    print("\n=== v11 hint mode 'off' ===")
    enc_cfg3 = InteractionPlanEncoderConfig(
        d_model=512, per_part_tokens=True, context_hint_mode="off",
    )
    encoder3 = InteractionPlanEncoder(enc_cfg3)
    plan_tokens3, plan_mask3, hint3 = encoder3(plan_t, T=T)
    print(f"plan_tokens {plan_tokens3.shape}  hint={hint3}")


if __name__ == "__main__":
    main()
