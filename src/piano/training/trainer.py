"""Shared Accelerate-based training loop.

Provides reusable training infrastructure for all three stages.
Each stage script builds its own model/optimizer/data, then calls
``run_training_loop`` with the appropriate ``step_fn``.

Usage:
    accelerate launch training/train_predictor.py --config configs/training/predictor.yaml
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Callable

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from piano.utils.io_utils import ensure_dir


def build_optimizer(
    params: Any,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """Build AdamW optimizer (single-group, decay applied uniformly).

    Prefer ``build_optimizer_with_decay_groups`` when training a
    Transformer: weight decay on LayerNorm / BatchNorm weights and
    biases is empirically harmful (ViT, T5, GPT-2, MoMask, PointNeXt).
    """
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def build_optimizer_with_decay_groups(
    modules: list[torch.nn.Module],
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """AdamW with ViT/T5-style weight-decay exclusion.

    Convention (Loshchilov & Hutter ICLR'19 + ViT / T5 / GPT-2 /
    MoMask / PointNeXt): exclude LayerNorm / BatchNorm weights, biases,
    and learned positional / time embeddings from weight decay — they
    have no well-defined notion of magnitude-based regularisation.

    We detect "no-decay" params by:
      1. ``param.ndim <= 1`` — biases and norm weights (all 1-D).
      2. Name ends in ``.bias``.
      3. Name matches a positional / time / embedding convention
         (``time_tokens``, ``pos_encoding``, ``.embed``).

    Params are pooled across the provided modules (the predictor and
    the object encoder share an optimizer).
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    def _is_no_decay(name: str, p: torch.nn.Parameter) -> bool:
        if p.ndim <= 1:
            return True
        if name.endswith(".bias"):
            return True
        lname = name.lower()
        if "time_tokens" in lname or "pos_encoding" in lname or ".embed" in lname:
            return True
        return False

    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            (no_decay if _is_no_decay(name, param) else decay).append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr, betas=betas)


def build_scheduler(
    optimizer: Any,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
) -> SequentialLR:
    """Build cosine annealing scheduler with linear warmup."""
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def run_training_loop(
    accelerator: Accelerator,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step_fn: Callable[..., dict[str, torch.Tensor]],
    num_epochs: int,
    output_dir: str | Path,
    log_every: int = 50,
    save_every_epochs: int = 10,
    max_grad_norm: float = 1.0,
    wandb_run: Any = None,
) -> None:
    """Generic training loop used by all stages.

    Parameters
    ----------
    accelerator : HuggingFace Accelerator instance
    model : the model (already prepared by accelerator)
    dataloader : training dataloader (already prepared)
    optimizer : optimizer (already prepared)
    scheduler : LR scheduler
    step_fn : callable(model, batch) -> dict with "loss" key and optional metric keys
    num_epochs : number of training epochs
    output_dir : where to save checkpoints
    log_every : log metrics every N steps
    save_every_epochs : save checkpoint every N epochs
    max_grad_norm : gradient clipping norm
    wandb_run : optional wandb run for logging
    """
    output_dir = ensure_dir(output_dir)
    global_step = 0

    # Backward-compatible hand-off of optimizer-step count to step_fn.
    # New stage scripts (predictor) declare ``global_step`` so their
    # step_fn can warm up physical priors; legacy skeletons keep the
    # old (model, batch) signature and are invoked unchanged.
    step_fn_params = inspect.signature(step_fn).parameters
    pass_global_step = "global_step" in step_fn_params

    accelerator.print(f"Training for {num_epochs} epochs")
    accelerator.print(f"  Batches per epoch: {len(dataloader)}")
    accelerator.print(f"  Output: {output_dir}")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses: dict[str, float] = {}
        epoch_start = time.time()

        for batch in dataloader:
            with accelerator.accumulate(model):
                # Forward + loss
                if pass_global_step:
                    loss_dict = step_fn(model, batch, global_step=global_step)
                else:
                    loss_dict = step_fn(model, batch)
                loss = loss_dict["loss"]

                # Backward
                accelerator.backward(loss)
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Accumulate metrics
            for key, val in loss_dict.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                epoch_losses[key] = epoch_losses.get(key, 0.0) + val

            global_step += 1

            # Logging
            if global_step % log_every == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                msg = f"  step {global_step} | lr={lr:.2e}"
                for key, val in loss_dict.items():
                    v = val.item() if isinstance(val, torch.Tensor) else val
                    msg += f" | {key}={v:.4f}"
                accelerator.print(msg)

                if wandb_run is not None:
                    log_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
                    log_dict["lr"] = lr
                    log_dict["epoch"] = epoch
                    wandb_run.log(log_dict, step=global_step)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        n_batches = len(dataloader)
        if accelerator.is_main_process:
            avg = {k: v / n_batches for k, v in epoch_losses.items()}
            msg = f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.0f}s)"
            for key, val in avg.items():
                msg += f" | {key}={val:.4f}"
            accelerator.print(msg)

        # Save checkpoint
        if (epoch + 1) % save_every_epochs == 0:
            _save_checkpoint(accelerator, model, optimizer, epoch, global_step, output_dir)

    # Final save
    _save_checkpoint(accelerator, model, optimizer, num_epochs - 1, global_step, output_dir, name="final")
    accelerator.print("Training complete.")


def _save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    output_dir: Path,
    name: str | None = None,
) -> None:
    """Save model checkpoint (main process only)."""
    if not accelerator.is_main_process:
        return

    ckpt_name = name or f"epoch_{epoch+1:04d}"
    ckpt_path = output_dir / f"{ckpt_name}.pt"

    unwrapped = accelerator.unwrap_model(model)
    torch.save(
        {
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        ckpt_path,
    )
    accelerator.print(f"  Saved checkpoint: {ckpt_path}")


def main() -> None:
    """CLI entrypoint for ``piano-train``. Dispatches to stage-specific scripts."""
    import argparse

    parser = argparse.ArgumentParser(description="PIANO training dispatcher")
    parser.add_argument("stage", choices=["predictor", "generator", "joint"],
                        help="Training stage to run")
    parser.add_argument("--config", type=str, required=True, help="Config yaml path")
    args = parser.parse_args()

    if args.stage == "predictor":
        from piano.training.train_predictor import run as run_predictor
        run_predictor(args.config)
    elif args.stage == "generator":
        from piano.training.train_generator import run as run_generator
        run_generator(args.config)
    elif args.stage == "joint":
        from piano.training.train_joint import run as run_joint
        run_joint(args.config)


if __name__ == "__main__":
    main()
