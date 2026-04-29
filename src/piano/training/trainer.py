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
    kendall_lr: float | None = None,
) -> AdamW:
    """AdamW with ViT/T5-style weight-decay exclusion + optional
    Kendall-log-var fast lr group.

    Convention (Loshchilov & Hutter ICLR'19 + ViT / T5 / GPT-2 /
    MoMask / PointNeXt): exclude LayerNorm / BatchNorm weights, biases,
    and learned positional / time embeddings from weight decay — they
    have no well-defined notion of magnitude-based regularisation.

    We detect "no-decay" params by:
      1. ``param.ndim <= 1`` — biases and norm weights (all 1-D).
      2. Name ends in ``.bias``.
      3. Name matches a positional / time / embedding convention
         (``time_tokens``, ``pos_encoding``, ``.embed``).

    **Kendall log-var fast group** (when ``kendall_lr`` is set):
    parameters whose name contains ``kendall.log_vars`` are routed to
    a separate group with the supplied higher lr. v3 found that
    Kendall's per-task log-variance scalars need ~100× the main lr to
    converge in a 100-epoch run — at lr=1e-4 they crawled from 0 to
    -0.25 over 6300 steps, vs the predicted equilibrium of ≈ -3.3 for
    the target task. Without this fix the multi-task auto-balancing
    is functionally inactive.

    Params are pooled across the provided modules (the predictor +
    the object encoder + the criterion when Kendall is on).
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    kendall_logvars: list[torch.nn.Parameter] = []
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

    def _is_kendall_logvar(name: str) -> bool:
        return "kendall.log_vars" in name

    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if kendall_lr is not None and _is_kendall_logvar(name):
                kendall_logvars.append(param)
            elif _is_no_decay(name, param):
                no_decay.append(param)
            else:
                decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]
    if kendall_logvars:
        param_groups.append({
            "params": kendall_logvars,
            "weight_decay": 0.0,
            "lr": kendall_lr,
        })
    return AdamW(param_groups, betas=betas)


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
    extra_modules: dict[str, torch.nn.Module] | None = None,
    val_dataloader: DataLoader | None = None,
    val_every_epochs: int = 0,
    val_best_key: str = "loss",
    contact_eval_fn: Callable[[], dict[str, float]] | None = None,
    contact_best_key: str = "mean_min_dist",
    train_report_keys: list[str] | tuple[str, ...] | None = None,
    val_report_keys: list[str] | tuple[str, ...] | None = None,
    contact_report_keys: list[str] | tuple[str, ...] | None = None,
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
    extra_modules : optional ``{name: nn.Module}`` dict of additional
        trainable modules to persist into each checkpoint alongside the
        main ``model``. Example: ``{"object_encoder": object_encoder}``
        for Stage A, where the main ``model`` is the predictor and the
        encoder is a peer module whose weights are just as critical
        for inference.
    val_dataloader : optional DataLoader of held-out clips. When
        provided together with ``val_every_epochs > 0``, the same
        ``step_fn`` is re-run on this loader every N epochs (with
        grads disabled and model.eval()) to measure val loss. A
        ``best_val.pt`` checkpoint is written whenever the total val
        loss improves. Does NOT interrupt training — run_training_loop
        always goes to ``num_epochs`` so the final checkpoint is also
        saved.
    val_every_epochs : how often (in epochs) to evaluate val. 0 or
        negative disables val entirely.
    val_best_key : which key from step_fn's loss_dict to minimise
        for best-val selection. Defaults to ``"loss"`` (total). Use
        e.g. ``"loss_target"`` to select on a specific component.
    contact_eval_fn : optional ``Callable[[], dict[str, float]]`` that
        returns a contact metrics dict (e.g.
        ``{"mean_min_dist": 0.16, "n_clips": 20}``) when called with no
        args. Invoked at the same cadence as ``val_dataloader``. Stage B
        uses this to save ``best_contact.pt`` alongside ``best_val.pt``,
        because val_loss and contact distance are empirically
        decoupled (see
        ``analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md``).
        ``None`` disables contact eval (Stage A predictor doesn't need it).
    contact_best_key : which key in the contact metrics dict to minimise.
        Defaults to ``"mean_min_dist"``.
    train_report_keys, val_report_keys, contact_report_keys : optional
        allow-lists for console/wandb reporting. The full metric dict is
        still computed internally and validation/checkpoint decisions still
        use ``val_best_key`` / ``contact_best_key``. ``None`` preserves the
        historical "report everything" behaviour.
    """
    output_dir = ensure_dir(output_dir)
    global_step = 0
    best_val_loss: float = float("inf")
    best_contact_dist: float = float("inf")

    # Backward-compatible hand-off of optimizer-step count to step_fn.
    # New stage scripts (predictor) declare ``global_step`` so their
    # step_fn can warm up physical priors; legacy skeletons keep the
    # old (model, batch) signature and are invoked unchanged.
    step_fn_params = inspect.signature(step_fn).parameters
    pass_global_step = "global_step" in step_fn_params

    val_enabled = val_dataloader is not None and val_every_epochs > 0

    accelerator.print(f"Training for {num_epochs} epochs")
    accelerator.print(f"  Batches per epoch: {len(dataloader)}")
    if val_enabled:
        accelerator.print(
            f"  Val every {val_every_epochs} epochs on {len(val_dataloader)} batches; "
            f"best-ckpt key = {val_best_key!r}",
        )
    accelerator.print(f"  Output: {output_dir}")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses: dict[str, float] = {}
        epoch_counts: dict[str, int] = {}
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

            # Accumulate metrics. Skip non-scalar tensors (diagnostic
            # outputs like predicted token IDs) — they're allowed in
            # the step_fn's return dict for downstream consumers but
            # don't fit the per-step ``+= val.item()`` reduction.
            # Older Stage A step_fns only returned scalars, so this
            # branch is a no-op there; Stage B onwards may include
            # per-token diagnostics.
            for key, val in loss_dict.items():
                if isinstance(val, torch.Tensor):
                    if val.numel() != 1:
                        continue
                    val = val.item()
                epoch_losses[key] = epoch_losses.get(key, 0.0) + val
                epoch_counts[key] = epoch_counts.get(key, 0) + 1

            global_step += 1

            # Per-step console logging (live monitoring only — not
            # pushed to wandb: per-step values are too noisy to read a
            # trend from. wandb gets the epoch-averaged values at the
            # end of each epoch below, matching the terminal "Epoch
            # N/M" summary line.
            if global_step % log_every == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                msg = f"  step {global_step} | lr={lr:.2e}"
                for key, val in _select_report_metrics(
                    loss_dict, train_report_keys,
                ).items():
                    if isinstance(val, torch.Tensor):
                        if val.numel() != 1:
                            continue
                        v = val.item()
                    else:
                        v = val
                    msg += f" | {key}={v:.4f}"
                accelerator.print(msg)

        # Epoch summary — printed to console AND pushed to wandb as one
        # data point per epoch (x-axis = epoch number). ``epoch_losses``
        # was accumulated per batch during the epoch loop; averaging by
        # ``n_batches`` gives the same smoothed curve the terminal
        # prints, which is the right granularity for trend reading on
        # a small-dataset training run.
        epoch_time = time.time() - epoch_start
        n_batches = len(dataloader)
        if accelerator.is_main_process:
            avg = {
                k: v / max(epoch_counts.get(k, n_batches), 1)
                for k, v in epoch_losses.items()
            }
            avg_report = _select_report_metrics(avg, train_report_keys)
            lr = optimizer.param_groups[0]["lr"]
            msg = f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.0f}s)"
            for key, val in avg_report.items():
                msg += f" | {key}={val:.4f}"
            accelerator.print(msg)

            if wandb_run is not None:
                log_dict = dict(avg_report)
                log_dict["lr"] = lr
                log_dict["epoch"] = epoch + 1
                log_dict["epoch_time_sec"] = epoch_time
                wandb_run.log(log_dict, step=epoch + 1)

        # Val pass + best-val checkpoint. Runs on all ranks (Accelerate
        # splits the val dataloader across them); losses are reduced
        # across ranks so the comparison is deterministic.
        if val_enabled and (epoch + 1) % val_every_epochs == 0:
            val_means = _run_validation(
                accelerator=accelerator,
                model=model,
                val_dataloader=val_dataloader,
                step_fn=step_fn,
                pass_global_step=pass_global_step,
                global_step=global_step,
            )
            if accelerator.is_main_process:
                val_report = _select_report_metrics(val_means, val_report_keys)
                msg = f"  Val @ epoch {epoch+1}"
                for key, val in val_report.items():
                    msg += f" | val_{key}={val:.4f}"
                accelerator.print(msg)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val_{k}": v for k, v in val_report.items()},
                        step=epoch + 1,
                    )

            cur_val = val_means.get(val_best_key, float("inf"))
            if cur_val < best_val_loss:
                best_val_loss = cur_val
                _save_checkpoint(
                    accelerator, model, optimizer, epoch, global_step, output_dir,
                    name="best_val", extra_modules=extra_modules,
                )
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  ↑ new best val {val_best_key}={cur_val:.4f} "
                        f"(epoch {epoch+1})",
                    )

            # Contact-aware checkpointing (B1 from
            # analyses/2026-04-28_v0_3_delta_retrain_and_v0_5_contact.md).
            # Stage B's training objective (masked-CE) is empirically
            # decoupled from the ship metric (geometric body-to-object
            # distance) — v0.4→v0.5 same-arch ablation showed CE↓ but
            # contact↑. So `best_val.pt` selected by val_loss is the
            # wrong checkpoint for shipping. We additionally save
            # `best_contact.pt` selected by the contact metric. Only
            # the main process generates + measures, then broadcasts
            # the float so all ranks save consistently.
            if contact_eval_fn is not None:
                if accelerator.is_main_process:
                    contact_metrics = contact_eval_fn()
                    cur_contact = float(contact_metrics.get(
                        contact_best_key, float("inf"),
                    ))
                else:
                    contact_metrics = {}
                    cur_contact = float("inf")
                # Broadcast cur_contact across ranks so the best-ckpt
                # decision matches on every rank. Accelerate's
                # gather/broadcast is the lightest tool here.
                cur_contact_t = torch.tensor(
                    [cur_contact], device=accelerator.device, dtype=torch.float32,
                )
                if accelerator.num_processes > 1:
                    torch.distributed.broadcast(cur_contact_t, src=0)
                cur_contact = float(cur_contact_t.item())

                if accelerator.is_main_process:
                    contact_report = _select_report_metrics(
                        contact_metrics, contact_report_keys,
                    )
                    msg = f"  Contact @ epoch {epoch+1}"
                    for k, v in contact_report.items():
                        msg += f" | contact_{k}={v:.4f}"
                    accelerator.print(msg)
                    if wandb_run is not None:
                        wandb_run.log(
                            {f"contact_{k}": v for k, v in contact_report.items()},
                            step=epoch + 1,
                        )

                if cur_contact < best_contact_dist:
                    best_contact_dist = cur_contact
                    _save_checkpoint(
                        accelerator, model, optimizer, epoch, global_step, output_dir,
                        name="best_contact", extra_modules=extra_modules,
                    )
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"  ↑ new best contact "
                            f"{contact_best_key}={cur_contact:.4f} "
                            f"(epoch {epoch+1})",
                        )

        # Save checkpoint
        if (epoch + 1) % save_every_epochs == 0:
            _save_checkpoint(
                accelerator, model, optimizer, epoch, global_step, output_dir,
                extra_modules=extra_modules,
            )

    # Final save
    _save_checkpoint(
        accelerator, model, optimizer, num_epochs - 1, global_step, output_dir,
        name="final", extra_modules=extra_modules,
    )
    accelerator.print("Training complete.")


def _select_report_metrics(
    metrics: dict[str, Any],
    keys: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Return only the metrics intended for human-facing reporting."""
    if keys is None:
        return dict(metrics)
    return {key: metrics[key] for key in keys if key in metrics}


@torch.no_grad()
def _run_validation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    step_fn: Callable[..., dict[str, torch.Tensor]],
    pass_global_step: bool,
    global_step: int,
) -> dict[str, float]:
    """Run one full pass over val_dataloader, returning mean losses.

    Reuses the training ``step_fn`` (no special val path). ``model.eval()``
    is set on entry and restored to ``train()`` on exit so dropout etc.
    behave correctly for val. Priors are passed ``global_step`` from the
    current training state so the val total is apples-to-apples with the
    corresponding train iteration.

    Under DDP, Accelerate prepared the val loader to shard batches
    across ranks. We accumulate sums per rank and reduce with mean at
    the end so each reported number is the global mean.
    """
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    device = accelerator.device
    try:
        for batch in val_dataloader:
            if pass_global_step:
                ld = step_fn(model, batch, global_step=global_step)
            else:
                ld = step_fn(model, batch)
            for k, v in ld.items():
                if isinstance(v, torch.Tensor):
                    # Detach + float so the running sum is a scalar tensor
                    # on-device; avoids CPU syncs per batch.
                    s = sums.get(k)
                    if s is None:
                        sums[k] = v.detach().float().mean()
                    else:
                        sums[k] = s + v.detach().float().mean()
            n_batches += 1
    finally:
        model.train()

    means: dict[str, float] = {}
    for k, total in sums.items():
        local_mean = total / max(n_batches, 1)
        global_mean = accelerator.reduce(local_mean, reduction="mean")
        means[k] = float(global_mean.item())
    return means


def _save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    output_dir: Path,
    name: str | None = None,
    extra_modules: dict[str, torch.nn.Module] | None = None,
) -> None:
    """Save model checkpoint (main process only).

    ``extra_modules`` is a flat ``{name: nn.Module}`` mapping — each
    entry's unwrapped state_dict is stored under its name as a
    top-level key alongside ``model``. Required for any stage whose
    inference path needs more than just the main ``model`` (Stage A:
    predictor + object_encoder).
    """
    if not accelerator.is_main_process:
        return

    ckpt_name = name or f"epoch_{epoch+1:04d}"
    ckpt_path = output_dir / f"{ckpt_name}.pt"

    unwrapped = accelerator.unwrap_model(model)
    payload: dict[str, Any] = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if extra_modules:
        for mod_name, mod in extra_modules.items():
            payload[mod_name] = accelerator.unwrap_model(mod).state_dict()

    torch.save(payload, ckpt_path)
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
