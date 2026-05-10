"""Quick smoke-test for InteractionPlanEncoder. Not a unit test.

Run via the project conda env:

    conda run -n piano --no-capture-output python \
        scripts/stage_b_generator/_smoke_plan_encoder.py
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
    contact[60:80, 1] = 1.0
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

    enc_cfg = InteractionPlanEncoderConfig(d_model=512, use_segment_tokens=False)
    encoder = InteractionPlanEncoder(enc_cfg)
    plan_tokens, plan_mask, hint = encoder(plan_t, T=T)
    print("plan_tokens", plan_tokens.shape, plan_tokens.dtype)
    print("plan_mask  ", plan_mask.shape, plan_mask.dtype)
    print("hint       ", hint.shape if hint is not None else None)
    print("valid_per_clip", plan_mask.sum(dim=1).tolist())

    xa = PlanCrossAttentionBlock(d_model=512, n_heads=4)
    motion = torch.randn(2, T, 512)
    out = xa(motion, plan_tokens, plan_mask)
    print("xattn out  ", out.shape, "  finite=", torch.isfinite(out).all().item())

    enc_cfg2 = InteractionPlanEncoderConfig(d_model=512, use_segment_tokens=True)
    encoder2 = InteractionPlanEncoder(enc_cfg2)
    pt2, pm2, _ = encoder2(plan_t, T=T)
    print("segtok pt2 ", pt2.shape, "pm2", pm2.shape)


if __name__ == "__main__":
    main()
