"""Round-29 — inspect I3 `hand_contact` channel distribution.

Per analyses/2026-05-27_round29_loss_strategy_codex_review.md §"I3 soft-mask concern":

> Before the next run, log these fractions on the 48-clip subset:
>     frac(0.5 < hand_contact < 0.95)
>     frac(hand_contact >= 0.95)
>     frac(hand_contact == 1.0)
> If fractional contact is common, weight tuning is only a partial fix.

This is a read-only diagnostic. It does NOT load any model. It just
iterates the 48-clip subset's batches, pulls the I3 condition tensor
that ``HOIDataset`` emits for variant ``I3-contact-offset-masked``, and
prints a histogram of the ``hand_contact`` channel (= contact_state[:, :2]
clamped to [0, 1]).

Usage:
    python scripts/stage_b_generator/round29_inspect_i3_hand_contact_distribution.py \\
        --config configs/training/anchordiff_r29_ls_a2_relbeh_v2_anchor0_low.yaml \\
        --selection-json analyses/round27_tier0_train_indices_48_balanced.json \\
        --bucket train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plan_condition_diagnostics import _build_dataset  # noqa: E402

from piano.data.dataset import collate_hoi  # noqa: E402
from piano.utils.io_utils import load_json  # noqa: E402


def _load_selection_pairs(sel_path: Path) -> set[tuple[str, str]]:
    if sel_path is None or not sel_path.exists():
        return set()
    data = load_json(sel_path)
    clips = data.get("clips", [])
    return {(str(c["subset"]), str(c["seq_id"])) for c in clips}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the distribution of the I3 hand_contact channel "
            "(soft 0-1) over the 48-clip subset."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--selection-json", type=Path,
        default=Path("analyses/round27_tier0_train_indices_48_balanced.json"),
    )
    parser.add_argument("--bucket", choices=["train", "val"], default="train")
    parser.add_argument(
        "--output", type=Path,
        default=Path("analyses/round29_i3_hand_contact_distribution.json"),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if str(cfg.data.get("r29_interaction_variant", "")) != "I3-contact-offset-masked":
        print(
            "[i3-inspect] WARN: config does not use I3-contact-offset-masked "
            "(got r29_interaction_variant="
            f"{cfg.data.get('r29_interaction_variant')!r})",
            file=sys.stderr,
        )
    sel_pairs = _load_selection_pairs(args.selection_json)
    print(f"[i3-inspect] selection: {len(sel_pairs)} clips")

    dataset = _build_dataset(cfg, args.bucket, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_hoi, num_workers=0)

    # Aggregators over all selected clip-frames (per-hand flat list).
    all_hc: list[np.ndarray] = []
    n_clips = 0
    for batch in loader:
        subset = str(batch["subset"][0])
        seq_id = str(batch["seq_id"][0])
        if sel_pairs and (subset, seq_id) not in sel_pairs:
            continue
        if "stage2_interaction" not in batch:
            print(
                "[i3-inspect] ERROR: batch has no stage2_interaction. "
                "Check that data.r29_interaction_variant is set.",
                file=sys.stderr,
            )
            return 2
        n_clips += 1
        si = batch["stage2_interaction"][0].cpu().numpy()                  # (T, 8)
        seq_len = int(batch["seq_len"][0].item())
        hc = si[:seq_len, 0:2]                                              # (T_valid, 2)
        all_hc.append(hc.reshape(-1))

    if not all_hc:
        print("[i3-inspect] no clips matched selection.")
        return 1

    hc_flat = np.concatenate(all_hc)                                        # (N,)
    n_total = int(hc_flat.size)

    # Bands per Codex review.
    frac_zero = float((hc_flat == 0.0).mean())
    frac_partial = float(((hc_flat > 0.0) & (hc_flat < 0.5)).mean())
    frac_soft_high = float(((hc_flat >= 0.5) & (hc_flat < 0.95)).mean())
    frac_near_one = float(((hc_flat >= 0.95) & (hc_flat < 1.0)).mean())
    frac_exact_one = float((hc_flat == 1.0).mean())

    # Histogram of nonzero values for visual sanity.
    nonzero = hc_flat[hc_flat > 0.0]
    bins = np.linspace(0.0, 1.0, 11)                                        # 10 bins
    hist, _ = np.histogram(nonzero, bins=bins)

    summary = {
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "bucket": args.bucket,
        "n_clips_scanned": n_clips,
        "n_hand_frames_total": n_total,
        "fractions": {
            "hand_contact == 0.0": frac_zero,
            "0.0 < hand_contact < 0.5": frac_partial,
            "0.5 <= hand_contact < 0.95": frac_soft_high,
            "0.95 <= hand_contact < 1.0": frac_near_one,
            "hand_contact == 1.0": frac_exact_one,
        },
        "nonzero_histogram_bins_0_to_1_in_10": bins.tolist(),
        "nonzero_histogram_counts": hist.tolist(),
    }

    print(f"\n[i3-inspect] N clips scanned: {n_clips}")
    print(f"[i3-inspect] N hand-frames total: {n_total} (L+R flattened)")
    print(f"[i3-inspect] frac hand_contact == 0.0:        {frac_zero*100:6.2f}%")
    print(f"[i3-inspect] frac 0.0 < hc < 0.5:             {frac_partial*100:6.2f}%")
    print(f"[i3-inspect] frac 0.5 <= hc < 0.95 (SOFT):    {frac_soft_high*100:6.2f}%")
    print(f"[i3-inspect] frac 0.95 <= hc < 1.0:           {frac_near_one*100:6.2f}%")
    print(f"[i3-inspect] frac hand_contact == 1.0:        {frac_exact_one*100:6.2f}%")
    print(f"\n[i3-inspect] Codex's question: 'frac(0.5 < hc < 0.95)' = {frac_soft_high*100:.2f}%")
    print(f"[i3-inspect] If this is > 1-2%, the soft-mask scaling artifact")
    print(f"[i3-inspect] is a non-trivial fraction of the loss signal; consider")
    print(f"[i3-inspect] thresholding hand_contact >= 0.95 or unmasking the I3 target.")
    print(f"[i3-inspect] If < 1%, weight tuning alone is the right call.")

    import json
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
