# HOIDataset Can Load Preprocessed OMOMO — 2026-04-19

## Context

After preprocessing OMOMO into PIANO format, we verified that `HOIDataset`
(`src/piano/data/dataset.py`) + `collate_hoi` can load and batch the output
without shape or key mismatches.

- Runner: `scripts/server/check_hoi_dataset.sh` (`piano-check-hoi-dataset`)
- Data: `/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/piano`

## Results

```
Dataset size: 4919 sequences (all splits)
Metadata: 4919 entries; 4838 with non-empty text; splits {train, test}

Single-sample tensor shapes (max_seq_length=196):
  motion      (196, 263)       float32
  joints      (196, 22, 3)     float32
  object_pc   (1024, 3)        float32
  seq_len     ()               int64
  text        str

Batched (B=4) shapes via collate_hoi:
  motion      (4, 196, 263)    float32
  joints      (4, 196, 22, 3)  float32
  object_pc   (4, 1024, 3)     float32
  seq_len     (4,)             int64
  text        list[4] of str
```

## Observations

- All sequences correctly padded to `max_seq_length=196` (HumanML3D max).
- `seq_len` tensor carries the true (pre-padding) length.
- String `text` field survives batching via `collate_hoi` as a Python list.
- Pseudo-label fields (`contact_state`, `contact_target`, `phase`, `support`)
  are absent — as expected, they are only present after pseudo-label
  extraction runs in a later step.

## Diagnosis

The pipeline from preprocessing → dataloader works end-to-end. Batch
collation handles the heterogeneous types (tensors + list of strings)
correctly.

## Implications

- Training can consume this dataset without any additional glue code.
- Pseudo-label extraction can process the same output directory; its
  results will be merged into the sample dict by `HOIDataset` at load time
  (via `pseudo_label_dir` kwarg).
- Our preprocessing → dataloader contract is stable. Same pattern will
  apply to InterAct data when it arrives — we just need a new
  `preprocess_interact.py` that writes files into the same layout.

## Action Items (→ PLAN.md)

- [x] Dataloader verified
- [ ] Next: extract pseudo-labels for all 4919 sequences
- [ ] Next: run end-to-end inference smoke test to verify the full model chain
