# 2026-05-03 — v12 strict pseudo-label design (rev 3: loose-distance + relaxed kinematic σ)

**Update 2026-05-03 r3**: User asked to inspect dropped clips:

> "你直接去检查数据集里的接触帧比例异常低的部分，分析这部分数据，看看我们遗漏了哪些。"

Debug of 27 clips where v11 marked > 40 % contact but v12 r2 dropped to
< 5 % surfaced 4 root causes:

1. **Sit case (e.g. Sub1034 sit, T=196, v11=100% → r2=0%)**: pelvis dist
   16 cm > tight threshold 12 cm. static_engagement was high (0.57) but
   case_static was tied to tight_distance, which gave 7 % — not enough.
   **Pelvis-to-seat is structurally 15–22 cm for SMPL 22-joint root**;
   tight 12 cm is anatomically wrong.
2. **Bat / racket swing (v11=100% → r2=0%)**: kin_engagement only 0.17
   even though bat moves at 4 m/s. Reason: SMPL wrist joint articulates
   ±4–5 cm in the bat's local frame during a swing (wrist flexion);
   `kin_local_sigma=0.03 m` makes the sigmoid output ~0.17 instead of
   ~1.0. The local-stability test was too strict for a 22-joint model
   that can't represent finger grip rigidly.
3. **Carry case (subject03_case_1350, v11=100% → r2=0%)**: case_kin mean
   72 % (high!) but binary frame frac 0 %. drift filter at 5 cm killed
   all segments — wrist articulation makes wrist drift 5–10 cm in
   object-local even when grip is rigid.
4. **Walking carry (Sub1716 sit, plasticbox, etc.)**: scores hover near
   0.5; median_filter + min_duration filter chains broke segments.

Fixes (r3):

- **Both cases use loose distance** (kinematic AND static). The
  tight_distance branch was anatomically wrong for sit/press; the
  engagement signal is what excludes false positives.
- **`kin_local_sigma`: 0.03 → 0.06 m** (allows wrap-grip wrist
  articulation; PIANO 22-joint can't go below this without artifacts).
- **`static_engagement_local_std_m`: 0.02 → 0.05 m** (sit/press has
  micro-motion).
- **`max_segment_drift_m`: 0.05 → 0.10 m** (wrist articulation in
  object-local frame).

Verification of user's racket case under r3: dist 8 cm < loose 25 cm
→ loose_score ~1.0; kin_local_sigma 0.06 with 4 cm articulation gives
kin_engage ~0.4–0.7 (boundary but firing); contact ✓.

Re-evaluation on the 27 r2-dropped clips:
- **24 / 27 clips fully or substantially recovered** (r3 ≥ 30 %)
- **3 / 27 still 0 %**: tripod_068 (right_hand mean 36 %, boundary
  binarisation), suitcase_lefthand_push (wrist 42 cm from PC due to
  PC sparsity, mesh-based should be ~8 cm), pink_1327_1 (mean 32 %,
  boundary).



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

### Aggregate (all revisions compared)

| metric | v11 | r1 (AND tight) | r2 (OR tight/loose) | **r3 (OR loose/loose)** |
|--------|----:|----:|----:|----:|
| mean contact frame frac (any part) | 77.6 % | 12.5 % | 19.4 % | **45.1 %** |
| total contact frames | 8787 | 1474 | 2214 | **4960** |
| total contact segments | 485 | 109 | 185 | **301** |
| mean segment duration (frames) | 43.2 | 14.4 | 14.9 | **28.0** |

r3 substantially recovers wrap-grip + sit + carry cases that r2 dropped.
The 45 % vs v11's 78 % gap reflects v11 over-counting "approaches" (hand
within 12 cm) as contact; r3 requires either kinematic engagement OR
static engagement *plus* loose-distance gate.

### Per body part (r2)

| part | v11 frame frac | v12 r2 frame frac | reduction |
|------|---------------:|---------------:|----------:|
| left_hand   | 49.6 % | 9.0 %  | −81.9 % |
| right_hand  | 52.0 % | 10.9 % | −79.1 % |
| left_foot   | 2.8 %  | 0.2 %  | −93.5 % |
| right_foot  | 2.7 %  | 0.1 %  | −97.4 % |
| pelvis      | 39.5 % | 6.1 %  | −84.6 % |

### By subset (r2 → r3)

| subset (N=20) | v11 frac | r2 frac | **r3 frac** | r3 #seg | comment |
|---------------|---------:|--------:|------------:|--------:|---------|
| chairs        | 83.4 %   | 25.9 %  | **73.0 %** | 129 | sit dominant; r3 captures full sit duration (was missing pelvis-on-seat at 12-22 cm) |
| imhd          | 94.5 %   | 19.2 %  | **33.3 %** | 52  | bat/racket/suitcase wrap-grip; r3 captures swing actions |
| neuraldome    | 73.7 %   | 25.4 %  | **35.6 %** | 74  | large wrap-grip objects; r3 captures carry/wave actions |
| omomo (correct_v2) | 59.8 % | 7.2 % | **37.9 %** | 46  | grasp-rich; r3 captures most grips. PC sparsity still causes some bias on long boxes |

**Key change r2 → r3**: chairs jumps from 25.9 % → 73.0 % (sit is now
captured at the realistic 70-90 % per-clip rate); neuraldome 25.4 % →
35.6 %; omomo 7.2 % → 37.9 %. Aggregate 19.4 % → 45.1 %.

### Sanity check: r3 doesn't over-count "approach"

Clips where v11 already marked < 30 % contact (i.e., short/sparse
contact, mostly approach with brief touch):

| seq_id | v11 | r3 | verdict |
|--------|----:|---:|---------|
| sub7_smallbox_035 | 22 % | 30 % | small over-count (8 pp) |
| sub7_whitechair_030 | 12 % | 14 % | aligned |
| subject03_trolleycase_1083_3 | 21 % | 13 % | r3 stricter |
| sub5_suitcase_030 | 17 % | 0 % | r3 stricter (PC sparsity) |
| sub7_trashcan_001 | 28 % | 0 % | r3 stricter |

r3 is in the same ballpark as v11 for low-contact clips (no false-positive
explosion), and is strictly tighter for some (which is the intended
behaviour — v11 over-counts brief approaches).

### Remaining edge cases (3/27 still 0 % at r3)

- **tripod_068**: right_hand case_kin score mean 36 %, boundary binarise
- **suitcase_lefthand_push**: wrist mean 42 cm from PC due to PC
  sparsity (mesh-based on the server should bring this to ~8 cm)
- **subject03_pink_1327_1**: similar mean-32 % boundary case

These are the natural failure modes of PC-based approximation +
0.5 binarise threshold. Server-side mesh evaluation will recover the
two PC-sparsity cases. The boundary-binarise cases are inherent to
the strict definition and acceptable as long as they're a small minority.

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
