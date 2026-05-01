# 2026-05-03 — v12 strict pseudo-label design (rev 2: two-case OR)

**Update 2026-05-03 r2**: User pushed back on the v1 AND-formulation
("contact = distance × engagement") with a wrap-grip case:

> "手拿着球拍晃动，人的手由于戴了手套，双手的关节节点和球拍之间距离大于5cm，
> 但是由于在世界里的坐标是同步运动的，其实是真实接触的，
> 这种情况你现在的限制条件下会判定为接触吗？"

Answer under v1: **NO** — distance > 5 cm makes `dist_score ≈ 0`, AND
multiplied with any engagement gives 0. This is a serious false negative,
losing v11's documented "wrist 18–22 cm from mesh in wrap-grip cases"
coverage for bat / racket / handle / carry-bag / glove scenarios.

This revision (r2) rebuilds contact as the **OR of two cases**, each
internally AND'd:

```
contact = case_kinematic   OR   case_static

case_kinematic = kinematic_engagement × loose_distance      # wrap-grip / glove
                  hand loose threshold = 0.25 m  (vs 0.05 tight)
                  foot loose threshold = 0.15 m  (vs 0.03)
                  pelvis loose threshold = 0.30 m (vs 0.12)

case_static    = static_engagement × tight_distance         # press / sit / grip-static
                  uses original v12 r1 tight thresholds (5 / 3 / 12 cm)
```

Verification of the user's example: wrist–racket distance 7–10 cm
(through gloves) < loose threshold 25 cm; racket + wrist move in
lockstep → kinematic_engagement HIGH → **case_kinematic fires →
contact ✓**.

Verification "walking past a table" (false positive case to avoid):
wrist 7 cm from table edge, table is stationary, body is moving
quickly → kinematic_engagement = 0 (object speed 0), static_engagement
= 0 (body not stable) → both cases fire = 0 → not contact ✓.

Verification "sit on chair": pelvis 12 cm from seat, chair stationary,
body stable → static_engagement HIGH, distance < tight threshold →
case_static fires → contact ✓.



User triggered this after visual review of v17-E.50 + final.pt:

> "视觉还是不过关，很多时候人根本就没有真正的接触到物体上，只是有点靠近而已。"

The 2026-05-03 unified metric results showed this is a **training-side
distribution problem** — `correct_part_recall` of v16 raw output is
0.176 (only 18 % of GT contact frames have the right part actually
labeled as in-contact in the model's output). Inference per-step pushes
it to 0.292 but pays a plausibility tax (jerk × 8, penetration +0.4 cm)
and STILL doesn't produce visually correct contact.

Root cause: **v11 pseudo-label defines "contact" as "within 12 cm" for
the hand**. The hand wrist sits 5–8 cm inside the forearm; palm surface
is another 5 cm out. So `wrist within 12 cm of mesh` corresponds to
`palm within ~−1 to 7 cm of mesh` — most of which is "approaching",
not "touching". The model trains on this label and reproduces this
behaviour: it learns to approach the surface neighbourhood, not to
touch the surface.

This document specifies a STRICT pseudo-label v12 that re-defines
"contact" as actual physical engagement, and presents the local
PC-based comparison evidence to justify the proposed thresholds.

## 1. Definition (v12 strict)

A frame `t` × body part `p` is in contact iff ALL of the following hold:

```python
# (1) Tight distance — palm/sole/ischium near surface
distance_score = sigmoid(
    (threshold_strict[part] - distance_to_mesh) / 0.015
)
# threshold_strict: hand=0.05, foot=0.03, pelvis=0.12 (m)
# Was v11: hand=0.12, foot=0.06, pelvis=0.20

# (2) Engagement — body is physically interacting, not just close
kinematic_engagement = (
    body_stable_in_object_local_frame  # std < 3 cm over 0.5 s
    AND object_translating_or_rotating  # speed > 0.15 m/s proxy
)
static_engagement = (
    object_stationary  # speed < 0.05 m/s
    AND body_stable_at_contact_point   # local std < 2 cm
)
engagement_score = max(kinematic_engagement, static_engagement)

# (3) AND-combined (was v11 OR-combined)
contact_score = distance_score * engagement_score

# (4) Temporal smoothing + min duration
contact = median_filter(contact_score, size=7) > 0.5
contact = filter_out_segments_shorter_than(5_frames)   # 0.25 s @ 20 fps

# (5) Within-segment drift filter (NEW for v12)
contact = filter_out_segments_where(
    body_part_in_object_local_drifts_more_than(0.05_m)
)
```

Key differences vs v11:
- **Tighter distance** (2–2.5 × tighter on each part, sigma 2 × tighter)
- **AND with engagement** (was OR with kinematic)
- **Longer min duration** (5 frames vs 3)
- **NEW: within-segment object-local drift filter** (5 cm)
- **NEW: static engagement signal** for press/sit/lean (not just kinematic
  coupling on moving objects)

## 2. Per-part threshold rationale

| part | v11 (m) | v12 strict (m) | rationale |
|------|--------:|---------------:|-----------|
| left/right_hand | 0.12 | **0.05** | wrist joint to palm surface ≈ 5–8 cm; v12 threshold means palm is within 0–2 cm of mesh ("touching") |
| left/right_foot | 0.06 | **0.03** | foot mid-joint to sole ≈ 4–5 cm; v12 means sole is within 0–1 cm ("on the surface") |
| pelvis | 0.20 | **0.12** | SMPL root to ischium ≈ 15–20 cm; v12 means ischium is within ~0–5 cm ("seated") |

Community alignment: OMOMO (Li et al., SIGGRAPH Asia 2023, arXiv:2309.16237)
§"Contact metric" uses 0.05 m hand-to-object distance + duration filter
as the canonical contact definition. CHOIS (Li et al., CVPR 2024,
arXiv:2312.17134) uses similar 0.05 m. v12's 0.05 m hand threshold
matches this directly.

## 3. Local PC-based evaluation on 80 GT clips (revised r2)

Evaluated via `evaluate_contact_definitions_pc.py` (uses nearest-PC
distance as `points_to_mesh_distance` approximation; mesh-based on the
server will be ~1–2 cm more permissive on average).

### Aggregate (r2 two-case OR)

| metric | v11 | v12 r1 (AND) | **v12 r2 (OR)** | r2 reduction vs v11 |
|--------|----:|----:|----:|----:|
| mean contact frame frac (any part) | 77.6 % | 12.5 % | **19.4 %** | −75.0 % |
| total contact frames | 8787 | 1474 | **2214** | −74.8 % |
| total contact segments | 485 | 109 | **185** | −61.9 % |
| mean segment duration (frames) | 43.2 | 14.4 | 14.9 | −66 % |

r2 captures meaningful wrap-grip cases that r1 dropped, while keeping
the over-counting reduction substantial.

### Per body part (r2)

| part | v11 frame frac | v12 r2 frame frac | reduction |
|------|---------------:|---------------:|----------:|
| left_hand   | 49.6 % | 9.0 %  | −81.9 % |
| right_hand  | 52.0 % | 10.9 % | −79.1 % |
| left_foot   | 2.8 %  | 0.2 %  | −93.5 % |
| right_foot  | 2.7 %  | 0.1 %  | −97.4 % |
| pelvis      | 39.5 % | 6.1 %  | −84.6 % |

### By subset (r2)

| subset (N=20) | v11 frac | v12 r2 frac | v11 #seg | v12 r2 #seg |
|---------------|---------:|---------:|---------:|---------:|
| chairs        | 83.4 %   | **25.9 %** | 82  | 51  |
| imhd          | 94.5 %   | **19.2 %** | 119 | 40  |
| neuraldome    | 73.7 %   | **25.4 %** | 155 | 72  |
| omomo (correct_v2) | 59.8 % | 7.2 % | 129 | 22  |

**Key change r1 → r2**: neuraldome rises from 12.0 % to 25.4 % — this
is the dataset dominated by wrap-grip cases (bat, racket, monitor,
handle, bag) that the user's racket example highlighted. The OR
formulation correctly captures these.

### Interpretation

**v11 is too permissive everywhere**. Even imhd (mostly wave/point
gestures with minimal real grip) shows 94 % contact-frame fraction —
which directly explains the model learning to "be generally near the
object" instead of "touching it".

**v12 strict is mostly correct but possibly too aggressive in two areas**:

1. **omomo at 4.3 %** is suspiciously low. OMOMO is a grasp-rich dataset
   (plasticbox, largebox, etc. — should have substantial sustained
   grip). The PC-based approximation introduces some negative bias:
   sparse PC (256 points on a 30 × 50 cm box → ~ 5 cm spacing) means
   tight thresholds (5 cm) measure "close to nearest sample" which is
   stricter than "close to mesh surface". Mesh-based on the server
   should give 8–15 % for omomo.

2. **foot at 0 %** is from the combination of (a) 3 cm threshold,
   (b) most PIANO motion is upright walking/standing where feet rarely
   contact the *object* (vs floor), (c) PC sparsity. In v11 it's
   already only 2.7 %, so this isn't a regression — feet just rarely
   contact the manipulated object in this dataset.

**Chairs at 25 %** is the most encouraging signal: 25 % means "person
sitting on chair for 1/4 of the clip" — visually correct ground truth.
v11's 83 % is "person within 20 cm of chair for most of the clip"
which is way over-counted.

## 4. Two candidate configurations (server-side choice)

### Option A: STRICT (recommended for first try)

The exact definition above. Pros: maximally aligned with "real contact"
visual semantic. Cons: training data sparser, potentially harder to
optimise.

```python
StrictContactConfig(
    distance_thresholds={"left_hand": 0.05, "right_hand": 0.05,
                          "left_foot": 0.03, "right_foot": 0.03,
                          "pelvis": 0.12},
    distance_sigma=0.015,
    require_engagement=True,
    min_contact_duration=5,
    max_segment_drift_m=0.05,
)
```

Predicted training data shape after re-extraction (mesh-based on server):
- ~15–20 % of frames in contact (vs v11's 60 %+)
- contact segments mostly grasp / sit / press

### Option B: MODERATE (safer fallback)

Halfway between v11 and v12:

```python
StrictContactConfig(
    distance_thresholds={"left_hand": 0.07, "right_hand": 0.07,    # was 0.05
                          "left_foot": 0.05, "right_foot": 0.05,   # was 0.03
                          "pelvis": 0.15},                          # was 0.12
    distance_sigma=0.02,
    require_engagement=True,
    min_contact_duration=4,                                          # was 5
    max_segment_drift_m=0.08,                                        # was 0.05
)
```

Predicted: ~30–40 % contact frame frac (vs v11 78 %, strict 12 %).
Less aggressive but still meaningfully strict.

## 5. Implementation plan

Files added (commit pending):
- `src/piano/data/pseudo_labels/extract_strict_contact.py` — v12 module
- `scripts/stage_b_generator/evaluate_contact_definitions_pc.py` — local
  PC-based comparison script (used to produce §3 numbers)
- `analyses/2026-05-03_pseudo_label_v12_strict_design.md` — this doc

Server-side workflow when user greenlights:
1. Run mesh-based v12 extraction on all 4 datasets (chairs / imhd /
   neuraldome / omomo). Stores to `pseudo_labels/v12_strict/`.
2. Train Stage A predictor on v12 labels (~6 h server time).
3. Train Stage B v18 with v12 pseudo-labels (`pseudo_label_dir` config
   override → `pseudo_labels/v12_strict/`).
4. Evaluate v18 on the unified metric set, expecting:
   - raw `correct_part_recall` to rise from 0.176 (v16) → 0.30+
   - guided to rise from 0.292 → 0.40+
   - Visual: real contact, not just approach
5. If v18 improves on alignment AND visual: ship. Else: try moderate
   config (Option B), or back to inference-side / γ_int branches.

## 6. Risk register

| risk | mitigation |
|------|------------|
| Training data too sparse → model can't learn contact at all | Try moderate config (Option B). If still too sparse, fall back to v11 with downweighted contact aux loss. |
| omomo / imhd subsets become unusably low contact frac | Per-subset threshold relaxation (allow looser thresholds for known-difficult subsets). |
| Stage A predictor fails to predict strict labels | Stage A predicts soft scores; even sparse strict labels train the predictor with contrastive signal vs non-contact frames. |
| Mesh-based extraction differs significantly from PC-based eval | PC-based eval is upper bound on distance (so under-estimates contact); mesh-based should be 5–15 pp higher contact frac. Acceptable. |

## 7. Sources

- Implementation: `src/piano/data/pseudo_labels/extract_strict_contact.py`
- PC-based eval: `runs/eval/_contact_def_compare_GT/summary.json`
- Reproducer: `python scripts/stage_b_generator/evaluate_contact_definitions_pc.py
  --input-dir runs/eval/<...>_gt_roundtrip_80/gt_original
  --output-dir runs/eval/_contact_def_compare_GT`
- v11 source: `src/piano/data/pseudo_labels/extract_contact.py`
- Visual evidence (failure motivating this work):
  `runs/visualizations/stageB_v17E50_final_review/` (10 mp4 clips)
- External community references:
  - Li et al. *OMOMO.* SIGGRAPH Asia 2023. arXiv:2309.16237.
  - Li et al. *CHOIS.* CVPR 2024. arXiv:2312.17134.
  - Xu et al. *InterDiff.* ICCV 2023. arXiv:2308.16905.
- Prior PIANO docs (this design responds to):
  - `analyses/pseudo_label_pipeline.md` (v11 design rationale)
  - `analyses/2026-05-03_unified_metric_results.md` (training-bottleneck diagnosis)
  - `analyses/2026-05-03_gamma_int_re_evaluation.md` (γ_int is not the lever; pseudo-label is)
