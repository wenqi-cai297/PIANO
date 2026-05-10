"""Print the seq_id of clip 0 from the overfit-mode dataset construction.

Used to look up the exact clip the v10 overfit trainer trained on, so
the visualizer can render the same clip.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, Subset

from piano.data.dataset import HOIDataset, build_subject_split, extract_subject_id
from piano.utils.io_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bucket", default="train")
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    keys: set[tuple[str, str]] = set()
    for entry in cfg.data.datasets:
        meta = load_json(Path(entry.root) / "metadata_clean.json")
        for m in meta:
            sid = extract_subject_id(Path(entry.root).name, m.get("seq_id", ""))
            if sid is not None:
                keys.add((Path(entry.root).name, sid))
    splits = build_subject_split(
        sorted(keys),
        train_pct=int(cfg.data.subject_split.train_pct),
        val_pct=int(cfg.data.subject_split.val_pct),
        seed=int(cfg.data.subject_split.seed),
    )
    subj_filter = splits[args.bucket]

    datasets = []
    for entry in cfg.data.datasets:
        sub_dir = str(Path(entry.root) / cfg.data.pseudo_label_subdir)
        ds = HOIDataset(
            root=entry.root,
            pseudo_label_dir=sub_dir,
            max_seq_length=int(cfg.data.max_seq_length),
            subject_id_filter=subj_filter,
            subsample_n_per_object=int(cfg.data.subsample_n_per_object),
            subsample_seed=int(cfg.data.subsample_seed),
            support_collapse_hand_support=True,
            surface_obj_pose=True,
            force_world_frame=True,
            motion_representation=str(cfg.data.motion_representation),
        )
        datasets.append(ds)
    combined = ConcatDataset(datasets)

    overfit_n = int(cfg.data.get("overfit_n_clips", 0))
    if overfit_n > 0:
        idx_list = list(range(min(overfit_n, len(combined))))
    else:
        idx_list = list(range(min(args.n, len(combined))))

    for i in idx_list:
        sample = combined[i]
        print(f"clip[{i}]: subset={sample['subset']}  seq_id={sample['seq_id']}")


if __name__ == "__main__":
    main()
