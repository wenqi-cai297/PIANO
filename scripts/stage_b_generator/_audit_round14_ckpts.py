"""Round-14 post-training audit: verify all 12 ckpts have correct
provenance (cache_root + seed + checkpoint_name) and that S1-A vs S1-B
step-0 loss agreement is preserved per seed pair (RNG identity).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


RUNS = Path("runs/training")
SEEDS = [42, 43, 44, 45, 46, 47]
MODES = ["s1a", "s1b"]
EXPECTED_CACHE = "cache/stage1_coarse_v1_full"


def main() -> int:
    print(f"{'run':<28s}  {'ck-seed':>7s}  {'cfg-seed':>8s}  {'cache_root':<32s}  {'step0_loss':>11s}  {'final_loss':>11s}  {'final_mse':>10s}  {'attn_mode':<13s}")
    print("-" * 130)
    by_seed: dict[int, dict[str, dict]] = {}
    fails: list[str] = []
    for mode in MODES:
        for seed in SEEDS:
            tag = f"stage1_{mode}_seed{seed}"
            run_dir = RUNS / tag
            ckpt_path = run_dir / "final.pt"
            log_path = run_dir / "loss_log.json"
            if not ckpt_path.exists():
                fails.append(f"missing ckpt: {ckpt_path}")
                continue
            if not log_path.exists():
                fails.append(f"missing log: {log_path}")
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            log = json.loads(log_path.read_text(encoding="utf-8"))
            ck_seed = ckpt.get("seed", None)
            ck_cfg_seed = ckpt.get("config", {}).get("training", {}).get("seed", None)
            ck_cache = ckpt.get("cache_root", None)
            attn_mode = ckpt.get("config", {}).get("model", {}).get("denoiser", {}).get("attention_mode", "?")
            ck_name = ckpt.get("checkpoint_name", None)
            steps = log["steps"]
            step0 = steps[0]["loss"] if steps else float("nan")
            final = steps[-1]["loss"] if steps else float("nan")
            final_mse = steps[-1]["mse"] if steps else float("nan")
            print(
                f"{tag:<28s}  {ck_seed!s:>7s}  {ck_cfg_seed!s:>8s}  "
                f"{(ck_cache or '?'):<32s}  {step0:>11.4f}  {final:>11.4f}  "
                f"{final_mse:>10.4f}  {attn_mode:<13s}"
            )
            # Provenance checks
            if ck_seed != seed:
                fails.append(f"{tag}: top-level seed={ck_seed} != expected {seed}")
            if ck_cfg_seed != seed:
                fails.append(f"{tag}: config.training.seed={ck_cfg_seed} != expected {seed}")
            if ck_cache != EXPECTED_CACHE.replace("/", "\\") and ck_cache != EXPECTED_CACHE:
                fails.append(f"{tag}: cache_root={ck_cache!r} != expected {EXPECTED_CACHE!r}")
            if ck_name != "final.pt":
                fails.append(f"{tag}: checkpoint_name={ck_name!r} != 'final.pt'")
            by_seed.setdefault(seed, {})[mode] = {
                "step0": step0,
                "final": final,
                "final_mse": final_mse,
            }

    # Paired step-0 identity: for each seed, S1-A step0 should equal S1-B step0
    # bit-exactly (same RNG + same init + same first batch + zero-init head
    # makes the step-0 loss identical regardless of attention_mode).
    print()
    print("Paired step-0 RNG identity (S1-A vs S1-B per seed; must be ~0):")
    for seed in SEEDS:
        pair = by_seed.get(seed, {})
        if "s1a" in pair and "s1b" in pair:
            diff = abs(pair["s1a"]["step0"] - pair["s1b"]["step0"])
            mark = "OK" if diff < 1e-3 else "DRIFT"
            print(
                f"  seed {seed}: S1-A step0 = {pair['s1a']['step0']:.4f}  "
                f"S1-B step0 = {pair['s1b']['step0']:.4f}  "
                f"|diff| = {diff:.6f}  [{mark}]"
            )
            if diff > 1e-3:
                fails.append(f"seed {seed}: step-0 paired drift = {diff:.6f}")

    print()
    if fails:
        print("AUDIT FAILURES:")
        for f in fails:
            print("  " + f)
        return 1
    print("AUDIT PASS — all 12 ckpts have consistent provenance + paired RNG identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
