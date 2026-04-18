# PIANO — Experiment Analysis Index

Index of all experiment analyses. Each row points to a file in `analyses/`
containing the detailed writeup. Sorted by date (most recent first).

**Workflow:**
1. User downloads experiment results locally
2. Claude reads results and writes a detailed analysis to `analyses/YYYY-MM-DD_<topic>.md`
3. A summary row is added to the table below
4. `PLAN.md` is updated with next steps based on the finding
5. Commit all three (`ANALYSIS.md`, `analyses/*.md`, `PLAN.md`) together

---

## Analysis Log

| Date | File | Topic | Key Finding | Action Taken |
|------|------|-------|-------------|--------------|
| 2026-04-19 | [momask_weight_loading](analyses/2026-04-19_momask_weight_loading.md) | MoMask pretrained weights load verification | VQ-VAE needs `mu=0.99` in opt; ResidualTransformer needs `share_weight=True` (from `_sw` suffix). All three checkpoints load cleanly after fixes. | Added missing fields to `build_momask_opt`; set `share_weight=True` default in adapter |
| 2026-04-19 | [omomo_data_inspection](analyses/2026-04-19_omomo_data_inspection.md) | CHOIS processed_data format check | `.p` files are joblib pickles (not vanilla); `obj_trans` has trailing singleton axis; 16 betas; no rest-pose meshes for vacuum/mop | Switched to `joblib.load`; added joblib dep; documented field conventions |
| 2026-04-19 | [omomo_preprocessing](analyses/2026-04-19_omomo_preprocessing.md) | SMPL-X FK → HumanML3D 263-dim on 5882 sequences | 4919 sequences preprocessed in ~1 min on A6000 (92 seq/sec); found + fixed SMPL-X zero-buffer batch-size bug | Explicit batch-sized zero tensors for unused SMPL-X params; downsample 30→20 fps via linear interp |
| 2026-04-19 | [hoi_dataset_verification](analyses/2026-04-19_hoi_dataset_verification.md) | HOIDataset + collate_hoi on preprocessed data | 4919 sequences, shapes correct, 4838 with text, collate handles str + tensors cleanly | No code changes needed; pipeline is stable |

---

## Analysis File Naming Convention

`analyses/YYYY-MM-DD_<short-topic-slug>.md`

Examples:
- `analyses/2026-05-01_pseudo_label_quality.md`
- `analyses/2026-05-15_stage_a_predictor_smoke_test.md`
- `analyses/2026-06-01_stage_b_cfg_scale_sweep.md`
- `analyses/2026-07-10_ass_ablation_vs_momask_baseline.md`

---

## Analysis File Template

Each analysis file should contain:

```markdown
# <Topic> — <Date>

## Context
What experiment was run, which commit, which config, which data.

## Results
Concrete numbers: loss curves, metrics, sample counts.
Link to any artifacts (checkpoints, logs, figures).

## Observations
What's notable in the results — both expected and unexpected.

## Diagnosis
If something went wrong, root cause. If something went well, what drove it.

## Implications
What does this mean for the project direction?

## Action Items (→ goes to PLAN.md)
Concrete next steps derived from this analysis.
```
