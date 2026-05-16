"""Round-13 final polish: inspect the smoke ckpt + loss_log for seed
provenance agreement.

Confirms the three places the resolved seed is recorded all carry the
same value (43 in the smoke test):

- checkpoint["seed"] (top-level)
- checkpoint["config"]["training"]["seed"]
- loss_log["seed"]

Also confirms the smoke run wrote `smoke_final.pt`, NOT `final.pt`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


RUN_DIR = Path("runs/training/_round13_final_polish_seed_ckpt_test")


def main() -> int:
    if not RUN_DIR.exists():
        print(f"[inspect] FAIL — run dir {RUN_DIR} missing")
        return 1

    ckpt_path = RUN_DIR / "smoke_final.pt"
    final_path = RUN_DIR / "final.pt"
    loss_log_path = RUN_DIR / "loss_log.json"
    if not ckpt_path.exists():
        print(f"[inspect] FAIL — smoke_final.pt missing at {ckpt_path}")
        return 1
    if final_path.exists():
        print(f"[inspect] FAIL — unexpected final.pt at {final_path}")
        return 1
    if not loss_log_path.exists():
        print(f"[inspect] FAIL — loss_log.json missing at {loss_log_path}")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    top_seed = ckpt.get("seed", None)
    cfg = ckpt.get("config", {}) or {}
    cfg_training = cfg.get("training", {}) or {}
    cfg_seed = cfg_training.get("seed", None)
    log = json.loads(loss_log_path.read_text(encoding="utf-8"))
    log_seed = log.get("seed", None)
    log_ckpt_name = log.get("checkpoint_name", None)
    log_cache_root = log.get("cache_root", None)

    print(f"[inspect] ckpt top-level seed        = {top_seed!r}")
    print(f"[inspect] ckpt config.training.seed  = {cfg_seed!r}")
    print(f"[inspect] loss_log.json seed         = {log_seed!r}")
    print(f"[inspect] loss_log.json checkpoint   = {log_ckpt_name!r}")
    print(f"[inspect] loss_log.json cache_root   = {log_cache_root!r}")

    expected_seed = 43
    if top_seed != expected_seed or cfg_seed != expected_seed or log_seed != expected_seed:
        print("[inspect] FAIL — seed provenance disagreement")
        return 1
    if log_ckpt_name != "smoke_final.pt":
        print(f"[inspect] FAIL — loss_log.checkpoint_name={log_ckpt_name!r}, expected 'smoke_final.pt'")
        return 1

    print("[inspect] PASS — seed provenance consistent at top-level, config, and loss_log")
    print("[inspect] PASS — no final.pt produced; smoke artefact is correctly named smoke_final.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
