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
| _(no analyses yet — project pre-training phase)_ | | | | |

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
