# PhysInteract: Object-Adaptive Human Motion Generation via Structured Interaction Latents

## Project Specification

---

## 1. Problem Statement

### 1.1 Core Problem: Object-Adaptive Interaction

In the real world, the same semantic action produces fundamentally different motor strategies depending on object properties. Picking up a small cup vs. a heavy box involves different approach trajectories, grip configurations, body postures, contact timing, and force distribution. Sitting on a high stool vs. a low sofa requires different descent strategies, support transitions, and balance control.

**This object-adaptive behavior is largely ignored by existing methods:**

- **End-to-end text-to-motion models** (MDM, MLD, MotionDiffuse) generate motions from text alone. They are entirely blind to object properties — "pick up" always produces the same motion regardless of what is being picked up.
- **Scene-aware methods** (HUMANISE, SceneDiffuser, TeSMo) condition on scene geometry but treat objects as static spatial constraints, not as entities whose properties should modulate motor strategy.
- **Affordance-based two-stage methods** (Move as You Say, CVPR 2024 Highlight) predict *where* interaction happens (spatial affordance maps) but not *how* it unfolds over time. The affordance map does not change when object weight, size, or shape changes — it only encodes location.
- **Contact-guided methods** (CG-HOI, Text2HOI) model contact as a spatial prior but lack temporal structure (when contacts form/break) and attribute sensitivity (how contacts change with object properties).

**No existing method explicitly models the causal chain:** `object properties → interaction strategy → motion trajectory`.

### 1.2 Why This Matters

Object-adaptive interaction is not a niche requirement — it is a fundamental aspect of physically plausible motion. Without it:

- Generated motions look "template-like": the same kinematic pattern applied to all objects
- Physical violations increase when the template doesn't fit the specific object (e.g., reaching distance mismatch, unsupported posture)
- Downstream applications (robotics, VR, animation) cannot use generated motions because they don't respect object-specific constraints

### 1.3 Our Claim

We argue that the missing piece is not better motion decoders or more data, but an explicit **structured interaction latent** that:

1. Encodes *how* interaction unfolds temporally (contact timing, phases, support transitions)
2. Is *causally downstream* of object properties (different objects → different latents → different motions)
3. Provides *decomposed, editable* control over interaction (change contact target without changing phase; change timing without changing contact pattern)

This turns object-adaptive behavior from something the model must implicitly discover into something explicitly supervised and structurally guaranteed.

---

## 2. Approach Overview

A two-stage generation framework:

```
text + object_properties + init_pose  -->  Interaction Predictor  -->  z_interaction
                                                                           |
                                                                           v
                                                                Motion Generator  -->  motion sequence
```

Instead of learning `(text, object) -> motion` end-to-end, we learn:

1. `(text, object_properties, init_pose) -> z_interaction` — predict a structured, temporally-resolved interaction plan that is sensitive to object attributes
2. `(z_interaction, text) -> motion` — generate motion conditioned on this interaction plan

**Key principles:**

- **Interaction structure, not motion, is the first-class variable.** Motion is a consequence of interaction decisions.
- **Object properties modulate interaction, not motion directly.** A heavier object changes the interaction plan (longer pre-contact, slower manipulation, more stable support), which in turn changes the motion. This two-hop causal chain is what makes the adaptation interpretable and controllable.
- **The latent is structured and decomposed,** enabling independent editing of contact targets, timing, phases, and support — a capability that monolithic affordance maps or end-to-end models cannot provide.

---

## 3. Interaction Latent Definition

The interaction latent `z_int` consists of per-frame structured variables:

| Variable | Type | Shape (per frame) | Description |
|----------|------|--------------------|-------------|
| `contact_state` | soft binary | `[B]` where B=5 body parts | Whether each body part contacts object/scene |
| `contact_target` | categorical (soft) | `[B, K]` where K=16~32 patches | Which object surface patch is contacted |
| `interaction_phase` | categorical | `[P]` where P=5~6 phases | Coarse temporal phase of interaction |
| `support_state` | categorical | `[S]` where S=4~5 states | Body support configuration |

**Body parts (B=5):** left_hand, right_hand, left_foot, right_foot, pelvis

**Phases (P=5):** approach, pre-contact, stable-contact, manipulation, release

**Support states (S=4):** both_feet, single_foot, sitting, hand_support

---

## 4. Tech Stack

### 4.1 Frozen / Pretrained Components

| Component | Choice | Source |
|-----------|--------|--------|
| Text encoder | CLIP ViT-L/14 | OpenAI CLIP (frozen) |
| Motion backbone | MLD (Motion Latent Diffusion) | [github.com/ChenFengYe/motion-latent-diffusion](https://github.com/ChenFengYe/motion-latent-diffusion) |
| Motion VAE | MLD pretrained VAE | MLD checkpoint |
| Body model | SMPL (22 joints) | [smplx](https://github.com/vchoutas/smplx) library |
| Motion representation | HumanML3D 263-dim | MLD built-in |

### 4.2 Trainable Components

| Component | Architecture | Params (approx) |
|-----------|-------------|-----------------|
| Object encoder | PointNet++ (PyTorch3D) | ~2M |
| Interaction predictor | Temporal Transformer (L=6, d=512) | ~25M |
| Interaction cross-attention (added to MLD denoiser) | Cross-attention layers | ~5M |
| Interaction extractor (for consistency loss) | Lightweight Transformer (L=3, d=256) | ~5M |

### 4.3 Libraries & Tools

| Purpose | Library |
|---------|---------|
| Deep learning | PyTorch 2.x |
| Training framework | HuggingFace Accelerate |
| Point cloud ops | PyTorch3D |
| Mesh processing | trimesh |
| Body model | smplx |
| Signal processing | scipy (median_filter, savgol_filter) |
| Phase refinement | hmmlearn |
| Surface clustering | scikit-learn (KMeans) / Open3D (FPS) |
| Diffusion scheduling | diffusers (DDPMScheduler / DDIMScheduler) or standalone implementation |
| Evaluation | Custom (based on HumanML3D eval protocol) |
| Logging | wandb |
| Config | OmegaConf (yaml configs) |

### 4.4 Training Framework: Accelerate

We use HuggingFace Accelerate instead of PyTorch Lightning for the following reasons:

- **Multi-GPU ready**: single-GPU code auto-scales to multi-GPU (DDP) with zero code change via `accelerate launch`
- **Lighter weight**: no complex callback/hook system, just a thin wrapper around native PyTorch training loops
- **Mixed precision**: built-in bf16/fp16 support via one config flag
- **Gradient accumulation**: built-in support, useful when batch size is limited by VRAM
- **MLD decoupling**: since we extract MLD's VAE and denoiser as standalone modules (not using MLD's Lightning training loop), Accelerate gives us full control over the training loop

Usage pattern:
```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2)
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)

for batch in dataloader:
    with accelerator.accumulate(model):
        loss = compute_loss(model, batch)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

Launch:
```bash
# Single GPU
accelerate launch training/train_predictor.py --config configs/predictor.yaml

# Multi-GPU (auto-detected)
accelerate launch --multi_gpu training/train_predictor.py --config configs/predictor.yaml
```

---

## 5. Data

### 5.1 Datasets

| Dataset | Role | Content |
|---------|------|---------|
| HumanML3D | Motion prior pretraining (use MLD pretrained weights) | 14k text-motion pairs, SMPL 22 joints |
| InterAct (CVPR 2025) | Primary HOI training data + pseudo-label extraction | 30.7h, standardized HOI, SMPL-X |
| OMOMO (SIGGRAPH Asia 2023) | Supplementary HOI data | 10h, 15 objects, SMPL-X |
| GRAB (ECCV 2020) | Fine-grained hand-object contact supervision | 51 objects, SMPL-X with hands |

### 5.2 Data Preprocessing

SMPL-X data from InterAct/OMOMO/GRAB must be **downsampled to SMPL 22 joints** to align with MLD's HumanML3D motion representation. This is a lossy conversion (hand fingers are dropped) but enables full reuse of MLD pretrained weights.

Pipeline:
1. Load SMPL-X sequences
2. Extract 22 SMPL joint positions + root orientation
3. Convert to HumanML3D 263-dim format (root velocity, joint positions, joint velocities, foot contact)
4. Normalize using HumanML3D statistics (mean/std from MLD)

---

## 6. Pseudo-label Extraction

All pseudo-labels are extracted offline before training. No GPU required.

### 6.1 Contact State

```python
for each frame t:
    for each body_part b in [left_hand, right_hand, left_foot, right_foot, pelvis]:
        d = min_distance(joint[b], object_surface)       # trimesh ProximityQuery
        v = relative_velocity(joint[b], object)           # finite difference
        contact[t, b] = sigmoid((tau_d - d) / sigma_d) * sigmoid((tau_v - |v|) / sigma_v)

# temporal smoothing: scipy.ndimage.median_filter(contact, size=5)
# minimum duration filter: discard contact events < 3 frames
```

**Thresholds (initial):**
- `tau_d` = 0.02m (2cm)
- `sigma_d` = 0.005m
- `tau_v` = 0.1 m/s
- `sigma_v` = 0.02 m/s

### 6.2 Contact Target

```python
# Pre-compute: FPS sample K=16 patch centers on object mesh
patch_centers = farthest_point_sample(object_vertices, K=16)

for each frame t where contact[t, b] > 0.5:
    nearest_point = closest_point_on_mesh(joint[b], object_mesh)
    target[t, b] = soft_assignment(nearest_point, patch_centers, sigma=0.01)
```

### 6.3 Interaction Phase

State machine with HMM refinement:

```python
# Heuristic initial assignment:
d_hand_obj = distance(hand_joints, object_center)  # per frame

phase[t] =:
    "approach"       if d_hand_obj[t] > tau_far and d_hand_obj decreasing
    "pre-contact"    if tau_near < d_hand_obj[t] < tau_far
    "stable-contact" if contact and object_velocity < epsilon
    "manipulation"   if contact and object_velocity >= epsilon
    "release"        if contact transitions from True to False

# Refinement: fit HMM (hmmlearn.GaussianHMM, n_components=5) on
# features = [d_hand_obj, contact_state, object_velocity]
# Use heuristic labels as initialization
```

### 6.4 Support State

```python
left_foot_contact  = contact[t, "left_foot"] > 0.5
right_foot_contact = contact[t, "right_foot"] > 0.5
pelvis_contact     = contact[t, "pelvis"] > 0.5
hand_contact       = any(contact[t, h] > 0.5 for h in ["left_hand", "right_hand"])

support[t] = :
    "both_feet"     if left_foot_contact and right_foot_contact
    "single_foot"   if exactly one foot contact
    "sitting"       if pelvis_contact
    "hand_support"  if hand_contact and not pelvis_contact and not both_feet
```

---

## 7. Model Architecture

### 7.1 Interaction Predictor

```
Inputs:
  text_emb:    [d_text]         from CLIP (frozen)
  init_pose:   [d_pose]         initial body state, MLP projected to d=512
  object_pc:   [N_pts, 3+feat]  object point cloud, PointNet++ -> [M, d]

Architecture:
  time_tokens = learnable_positional_embedding(T)         # [T, 512]
  object_tokens = PointNetPP(object_pc)                   # [M, 512]

  Transformer Decoder (L=6, d=512, heads=8):
    self-attention:  time_tokens attend to each other
    cross-attention: time_tokens attend to object_tokens
    conditioning:    text_emb via AdaLN (adaptive layer norm)
    init injection:  init_pose added to first time token

Outputs (per frame):
  contact_head:  Linear(512, B)     -> sigmoid  -> contact_state   [T, B]
  target_head:   Linear(512, B*K)   -> softmax  -> contact_target  [T, B, K]
  phase_head:    Linear(512, P)     -> softmax  -> phase           [T, P]
  support_head:  Linear(512, S)     -> softmax  -> support_state   [T, S]
```

### 7.2 Conditioned Motion Generator (Modified MLD)

Modifications to MLD's denoiser (a Transformer-based network):

```
Original MLD denoiser:
  Input:  noisy_latent_z + timestep_emb
  Cond:   text_emb via cross-attention
  Output: predicted noise / denoised latent

Our modification — add interaction cross-attention:
  For each Transformer block in denoiser:
    1. self-attention(z)                         # original
    2. cross-attention(z, text_emb)              # original
    3. cross-attention(z, interaction_tokens)     # NEW: attend to z_int
    4. feedforward(z)                            # original

  interaction_tokens = MLP(concat(contact_state, contact_target, phase, support))
                     -> [T, d_model]
```

Interaction tokens are projected to match MLD's hidden dimension and injected via standard cross-attention with separate learned Q/K/V projections.

### 7.3 Interaction Extractor (for consistency loss)

```
Input:  motion sequence [T, d_motion]
Architecture: Lightweight Transformer (L=3, d=256, heads=4)
Output: predicted interaction labels (same format as pseudo-labels)
```

Trained jointly during Stage 4 to enforce that generated motion is consistent with its conditioning interaction latent.

---

## 8. Training

### 8.1 Stages

| Stage | What | Data | Trainable | Duration (1x A100) |
|-------|------|------|-----------|---------------------|
| 0 | Download MLD pretrained weights | - | - | - |
| 1 | Pseudo-label extraction | InterAct + OMOMO + GRAB | - (CPU only) | ~hours |
| 2 | Train Interaction Predictor | HOI data + pseudo-labels | Predictor + Object Encoder | 1-2 days |
| 3 | Train Conditioned Generator | HOI data + pred/GT z_int | MLD denoiser (finetune) + cross-attn layers | 2-3 days |
| 4 | Joint finetune + consistency | HOI data | All trainable + Extractor | 1-2 days |

### 8.2 Loss Functions

**Stage 2 — Interaction Predictor:**

```
L_predictor = lambda_c  * BCE(pred_contact, gt_contact)          # contact state
            + lambda_t  * CE(pred_target, gt_target)             # contact target
            + lambda_p  * CE(pred_phase, gt_phase)               # interaction phase
            + lambda_s  * CE(pred_support, gt_support)           # support state
```

Initial lambdas: `lambda_c=1.0, lambda_t=0.5, lambda_p=0.5, lambda_s=0.5`

**Stage 3 — Motion Generator:**

```
L_generator = L_diffusion                                        # MLD's standard DDPM loss
            + lambda_smooth * L_velocity_smoothness              # acceleration penalty
```

**Stage 4 — Joint Finetune:**

```
L_joint = L_predictor
        + L_generator
        + lambda_cons * L_consistency                            # extractor(generated_motion) ≈ z_int
```

`L_consistency = BCE(extracted_contact, input_contact) + CE(extracted_phase, input_phase) + ...`

### 8.3 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate (Predictor) | 1e-4 |
| Learning rate (Generator finetune) | 5e-5 |
| LR scheduler | Cosine annealing with warmup (1000 steps) |
| Batch size (per GPU) | 64 (A100) / 32 (3090) |
| Gradient accumulation steps | 2 (effective batch = per_gpu × num_gpu × accum) |
| Sequence length | 60-120 frames (2-4 sec @ 30fps) |
| Diffusion steps (train) | 1000 |
| Diffusion steps (inference) | 50 (DDIM) |
| Classifier-free guidance | p_uncond=0.1, guidance_scale=7.5 |
| Mixed precision (Accelerate) | bf16 |
| Gradient checkpointing | Enabled for Generator |

### 8.4 Accelerate Configuration

```yaml
# accelerate_config.yaml (generated via `accelerate config` or written manually)
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU    # or NO for single GPU
mixed_precision: bf16
num_machines: 1
num_processes: 1               # set to number of GPUs
```

---

## 9. Weak Physical Priors

Applied as regularization losses during interaction predictor training:

### 9.1 Reachability Prior

```
# If contact predicted for hand, check target is within arm reach
arm_length = approximate_arm_length(body_params)
hand_to_target_dist = ||hand_joint - contact_target_point||
L_reach = max(0, hand_to_target_dist - arm_length) ** 2
```

### 9.2 Contact Persistence Prior

```
# Penalize single-frame contact flickers
contact_diff = |contact[t] - contact[t-1]|
L_persist = mean(contact_diff) * lambda_persist
# Alternatively: penalize contact segments shorter than min_duration=3 frames
```

### 9.3 Support Consistency Prior

```
# Support state should not oscillate rapidly
support_change = (support[t] != support[t-1]).float()
L_support_smooth = mean(support_change) * lambda_support
```

### 9.4 Phase Monotonicity Prior

```
# Phase transitions should be mostly forward (approach -> contact -> manipulation -> release)
# Soft penalty for backward phase transitions
phase_idx = argmax(pred_phase, dim=-1)
backward = max(0, phase_idx[t-1] - phase_idx[t])
L_phase_mono = mean(backward) * lambda_phase
```

---

## 10. Inference Pipeline

```
Input: text_prompt, object_point_cloud, initial_pose

Step 1: Encode inputs
  text_emb = CLIP.encode(text_prompt)
  obj_tokens = PointNetPP(object_point_cloud)
  pose_emb = MLP(initial_pose)

Step 2: Predict interaction latent
  z_int = InteractionPredictor(text_emb, obj_tokens, pose_emb)
  # z_int = {contact_state, contact_target, phase, support}  [T, ...]

Step 3: Generate motion
  interaction_tokens = MLP(flatten(z_int))
  motion_latent = MLD.sample(
      conditions=[text_emb, interaction_tokens],
      num_steps=50,  # DDIM
      guidance_scale=7.5
  )
  motion = MLD.decode(motion_latent)

Step 4: Post-process (optional)
  # Standard: root trajectory integration, FK to get joint positions
  # No heavy IK or physics post-processing needed

Output: motion_sequence [T, 263] in HumanML3D format
        -> can be converted to SMPL mesh via smplx
```

---

## 11. Evaluation

### 11.1 Standard Motion Metrics (from MLD eval)

| Metric | Measures |
|--------|----------|
| FID | Distribution quality |
| R-Precision (top1/2/3) | Text-motion alignment |
| MM-Dist | Text-motion matching distance |
| Diversity | Motion variety |
| MultiModality | Variation for same text |

### 11.2 Physical / Interaction Metrics (custom)

| Metric | Definition |
|--------|------------|
| Penetration rate | % frames with body-object penetration (distance < 0) |
| Contact precision | Among predicted contacts, % that are geometrically valid |
| Contact recall | Among GT contacts, % that are predicted |
| Foot sliding | Mean foot velocity when foot is in ground contact |
| Support consistency | % frames where support state is physically valid |
| Phase accuracy | Agreement between generated motion's extracted phase and predicted phase |

### 11.3 Object-Adaptive Evaluation (core novelty metrics)

This is the **primary evaluation axis** that differentiates us from all prior work.

**11.3.1 Attribute Sensitivity Score (ASS)**

For the same text prompt, generate motions with objects of varying attributes (e.g., small/medium/large box, light/heavy object). Measure whether the generated motion strategy changes meaningfully:

```
ASS = mean over attribute pairs (a_i, a_j):
    || MotionFeature(gen(text, obj_ai)) - MotionFeature(gen(text, obj_aj)) || /
    || AttributeFeature(a_i) - AttributeFeature(a_j) ||
```

Motion features include: approach speed, pre-contact duration, grasp width, CoM lowering, manipulation velocity. A higher ASS means the model is more responsive to object property changes.

**11.3.2 Attribute-Strategy Consistency (ASC)**

Verify that the direction of motion change is physically correct:
- Heavier object → slower manipulation, longer pre-contact, lower CoM
- Larger object → wider grasp, more body lean, adjusted approach angle
- Higher surface → more arm elevation, different support transition

Evaluated via a set of predefined physical rules, reported as % of rule-compliant generations.

**11.3.3 Cross-Object Generalization**

Train on a subset of objects, test on held-out objects with known attributes. Report ASS and ASC on unseen objects to demonstrate that the model learns attribute-to-strategy mappings, not object-specific templates.

### 11.4 Controllability & Editability Tests

| Test | Method | What it shows |
|------|--------|---------------|
| Latent sensitivity | Perturb z_int, measure motion change | Generator actually uses z_int |
| Contact target swap | Fix phase/support, change contact target | Decomposed control over where |
| Phase retiming | Fix contacts, stretch/compress phase timing | Decomposed control over when |
| Phase-contact recombination | Take phase from motion A, contacts from motion B | Compositional generation |
| Attribute interpolation | Interpolate object embeddings, observe smooth strategy change | Continuous adaptation |

### 11.5 Baselines

| Baseline | Description | What comparison shows |
|----------|-------------|----------------------|
| MLD (text-only) | No object or interaction information | Value of object-aware generation |
| MLD + object concat | Object features concatenated to text, no interaction latent | Value of structured latent vs. naive conditioning |
| CG-HOI | Contact-guided HOI generation | Value of temporal structure beyond contact |
| Text2HOI | Contact map then conditioned generation | Value of phase/support beyond contact |
| Move as You Say (if applicable) | Affordance map as intermediate | Value of temporal interaction latent vs. spatial affordance |
| Ours (end-to-end) | Remove interaction latent, direct text+obj→motion | Value of two-stage decomposition |
| Ours (w/o phase) | Remove phase variable | Phase contribution |
| Ours (w/o support) | Remove support variable | Support contribution |
| Ours (w/o target) | Remove contact target | Target contribution |
| Ours (w/o object attr) | Remove object attribute embedding | Object-adaptive contribution |
| Ours (w/o priors) | Remove all weak physical priors | Prior contribution |
| Ours (w/o consistency) | Remove consistency loss | Consistency loss contribution |

---

## 12. Project Structure

```
physinteract/                            # Repository root
│
├── SPEC.md                              # This specification document
├── README.md                            # (create when ready to release)
├── pyproject.toml                       # Package metadata, deps, CLI entrypoints
├── environment.yml                      # Conda env for one-command server setup
├── .gitignore
│
├── configs/                             # All configuration (OmegaConf yaml)
│   ├── accelerate_config.yaml           # Accelerate distributed / mixed-precision
│   ├── training/
│   │   ├── predictor.yaml               # Interaction predictor training hparams
│   │   ├── generator.yaml               # Motion generator finetune hparams
│   │   └── joint_finetune.yaml          # Joint finetune hparams
│   └── model/
│       ├── interaction_predictor.yaml   # Predictor architecture config
│       ├── motion_generator.yaml        # Modified MLD denoiser config
│       └── object_encoder.yaml          # PointNet++ config
│
├── src/                                 # Installable Python package
│   └── physinteract/
│       ├── __init__.py                  # Package version
│       │
│       ├── data/                        # Data processing & datasets
│       │   ├── __init__.py
│       │   ├── preprocess_smplx.py      # SMPL-X → SMPL 22-joint conversion
│       │   ├── humanml3d_repr.py        # Convert to HumanML3D 263-dim format
│       │   ├── dataset.py               # PyTorch Dataset classes
│       │   └── pseudo_labels/           # Pseudo-label extraction (CPU-only)
│       │       ├── __init__.py
│       │       ├── extract_contact.py   # Contact state extraction
│       │       ├── extract_target.py    # Contact target region extraction
│       │       ├── extract_phase.py     # Interaction phase extraction
│       │       ├── extract_support.py   # Support state extraction
│       │       ├── refine_phase_hmm.py  # HMM-based phase refinement
│       │       └── run_all.py           # Full extraction pipeline entrypoint
│       │
│       ├── models/                      # Model definitions
│       │   ├── __init__.py
│       │   ├── interaction_predictor.py # Temporal Transformer predictor
│       │   ├── object_encoder.py        # PointNet++ wrapper (PyTorch3D)
│       │   ├── interaction_cross_attn.py # Cross-attn layer for MLD denoiser
│       │   ├── interaction_extractor.py # Lightweight extractor (consistency)
│       │   ├── motion_generator.py      # MLD VAE + modified denoiser
│       │   └── diffusion.py             # DDPM / DDIM scheduling utilities
│       │
│       ├── training/                    # Training logic (Accelerate-based)
│       │   ├── __init__.py
│       │   ├── trainer.py               # Shared Accelerate training loop
│       │   ├── train_predictor.py       # Stage A: predictor training
│       │   ├── train_generator.py       # Stage B: generator finetune
│       │   ├── train_joint.py           # Stage C: joint finetune
│       │   ├── losses.py                # All loss functions
│       │   └── priors.py               # Weak physical prior losses
│       │
│       ├── evaluation/                  # Evaluation modules
│       │   ├── __init__.py
│       │   ├── motion_metrics.py        # FID, R-Precision, MM-Dist, etc.
│       │   ├── physics_metrics.py       # Penetration, contact, foot sliding
│       │   └── controllability.py       # ASS, ASC, latent sensitivity
│       │
│       ├── inference/                   # Inference pipeline
│       │   ├── __init__.py
│       │   ├── generate.py              # Full text+object → motion pipeline
│       │   └── visualize.py             # Motion → SMPL mesh visualization
│       │
│       └── utils/                       # Shared utilities
│           ├── __init__.py
│           ├── io_utils.py              # File I/O helpers (JSONL, npz, etc.)
│           ├── geometry.py              # Point cloud / mesh distance helpers
│           └── smpl_utils.py            # SMPL/SMPL-X body model helpers
│
├── scripts/                             # Operational scripts (not part of package)
│   ├── data/
│   │   └── download_datasets.sh         # Download InterAct, OMOMO, GRAB
│   ├── server/
│   │   ├── setup_env.sh                 # Environment setup on new server
│   │   ├── download_mld_weights.sh      # Fetch MLD pretrained checkpoints
│   │   ├── run_pseudo_labels.sh         # Full pseudo-label extraction
│   │   ├── run_train_predictor.sh       # accelerate launch wrapper
│   │   ├── run_train_generator.sh       # accelerate launch wrapper
│   │   ├── run_train_joint.sh           # accelerate launch wrapper
│   │   └── run_eval.sh                  # Full evaluation pipeline
│   └── inference/
│       └── sample_motions.sh            # Batch inference script
│
├── tests/                               # Unit & integration tests
│   ├── test_pseudo_labels.py
│   ├── test_models.py
│   └── test_losses.py
│
├── docs/                                # Design docs & notes
│
└── runs/                                # All outputs (gitignored)
    ├── pseudo_labels/                   # Extracted pseudo-labels
    ├── training/                        # Checkpoints, logs per run
    │   ├── predictor/<timestamp>/
    │   ├── generator/<timestamp>/
    │   └── joint/<timestamp>/
    └── eval/                            # Evaluation results
```

### 12.1 Package Installation

The project is a standard setuptools package. Editable install for development:

```bash
pip install -e ".[wandb]"
```

This makes `physinteract` importable everywhere and registers CLI entrypoints.

### 12.2 Code Conventions

Following the patterns established in the 2026-03-25 reference project:

| Convention | Detail |
|------------|--------|
| Module headers | `from __future__ import annotations` + module-level docstring |
| Type hints | Full annotations, Python 3.10+ syntax (`str \| None`) |
| Dataclasses | `@dataclass(slots=True)` for config / record types |
| Path handling | `pathlib.Path` throughout, no raw string paths |
| Imports | Grouped: stdlib → third-party → local, separated by blank lines |
| Naming | Files/functions: `snake_case`. Classes: `PascalCase` |
| Docstrings | Module-level explains purpose/design. Functions explain *why*, not just *what* |
| Comments | Sparse but strategic — explain non-obvious design decisions |
| I/O | Incremental writes for long jobs, resume support where applicable |
| Configs | OmegaConf yaml files under `configs/`, never hardcoded |
| Outputs | All artifacts go to `runs/`, never tracked in git |

---

## 13. Environment Setup

### 13.1 environment.yml

```yaml
name: physinteract
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - torch>=2.0
    - torchvision
    - accelerate
    - transformers
    - diffusers
    - einops
    - omegaconf
    - trimesh
    - smplx
    - scipy
    - scikit-learn
    - hmmlearn
    - open3d
    - matplotlib
    - tqdm
    - wandb
```

### 13.2 pyproject.toml

```toml
[project]
name = "physinteract"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torchvision",
    "accelerate",
    "transformers",
    "diffusers",
    "einops",
    "omegaconf",
    "trimesh",
    "smplx",
    "scipy",
    "scikit-learn",
    "hmmlearn",
    "tqdm",
    "numpy",
]

[project.optional-dependencies]
wandb = ["wandb"]
viz = ["matplotlib", "open3d"]
dev = ["pytest"]

[project.scripts]
physinteract-train = "physinteract.training.trainer:main"
physinteract-eval = "physinteract.evaluation.motion_metrics:main"
physinteract-generate = "physinteract.inference.generate:main"
physinteract-pseudo-labels = "physinteract.data.pseudo_labels.run_all:main"

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### 13.3 Quick Start

```bash
# Option A: conda (recommended for new servers)
conda env create -f environment.yml
conda activate physinteract

# Option B: pip only
pip install -e ".[wandb,viz,dev]"

# PyTorch3D (build from source if pip fails)
pip install pytorch3d

# Configure accelerate
accelerate config

# Verify
python -c "from physinteract import __init__; print('OK')"
```

### 13.4 .gitignore

```
__pycache__/
*.pyc
*.egg-info/
.venv/
runs/
checkpoints/
*.ckpt
*.pt
*.pth
wandb/
.DS_Store
```

---

## 14. Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Environment + data | MLD running, datasets downloaded, SMPL-X->SMPL conversion done |
| 3 | Pseudo-labels | All pseudo-labels extracted, quality verified via visualization |
| 4-5 | Interaction Predictor | Trained predictor, accuracy metrics on held-out set |
| 6-7 | Motion Generator | MLD finetuned with interaction conditioning, qualitative results |
| 8 | Joint finetune | Full pipeline end-to-end, consistency loss working |
| 9-10 | Evaluation | All metrics computed, ablations done, visualizations ready |

---

## 15. Differentiation from Prior Work

### 15.1 Why This Is Not "Move as You Say with More Variables"

| Dimension | Move as You Say (CVPR 2024) | Ours |
|-----------|----------------------------|------|
| **Problem framing** | Where in the scene to interact | How interaction unfolds given object properties |
| **Intermediate repr** | Spatial affordance map (static, scene-level) | Temporal interaction sequence (dynamic, per-frame) |
| **Object awareness** | None — affordance doesn't change with object | Core — object attributes directly modulate interaction latent |
| **Temporal structure** | No — affordance is a single spatial prediction | Yes — contact/phase/support evolve over time |
| **Editability** | Cannot edit sub-components independently | Decomposed: edit contact, phase, target, support separately |
| **Evaluation** | Standard motion metrics + scene plausibility | + Attribute Sensitivity Score + Attribute-Strategy Consistency |

The structural resemblance (both are two-stage with an intermediate representation) is intentional — two-stage is a proven design. The contribution is not the architecture pattern but **what the intermediate representation encodes and what new capability it enables**.

### 15.2 Contributions (Paper Framing)

1. **Problem contribution**: We identify object-adaptive interaction as a neglected but fundamental aspect of physically plausible motion generation. We show that existing methods — including affordance-based two-stage approaches — produce attribute-insensitive motions.

2. **Representation contribution**: We propose a structured interaction latent decomposed into contact state, contact target, interaction phase, and support cue. Unlike monolithic affordance maps, this representation is temporal, decomposed, and causally linked to object properties.

3. **Method contribution**: We show that pseudo interaction labels extracted from existing HOI data, combined with weak physical priors, are sufficient to supervise this latent without expensive physics simulation.

4. **Evaluation contribution**: We introduce Attribute Sensitivity Score (ASS) and Attribute-Strategy Consistency (ASC) as new metrics for evaluating object-adaptive motion generation — an evaluation axis absent from prior work.

### 15.3 Anticipated Reviewer Questions and Answers

**Q: "This is just Move as You Say with richer features."**
A: Move as You Say predicts *where* (spatial affordance). We predict *how and when* (temporal interaction plan). More importantly, Move as You Say's affordance does not change with object properties — our latent does. This enables a fundamentally new capability (object-adaptive generation) that we validate with dedicated metrics (ASS/ASC).

**Q: "The two-stage decomposition is not novel."**
A: We agree — two-stage is a design choice, not a contribution. Our contribution is the specific structure of the interaction latent and the object-adaptive capability it enables. We show via ablation that a naive two-stage (e.g., contact-only intermediate) does not achieve the same object sensitivity.

**Q: "Pseudo-labels are noisy, how do you know the latent is meaningful?"**
A: (1) Soft labels + temporal smoothing reduce noise. (2) Weak physical priors constrain the latent to physically plausible regions. (3) Consistency loss ensures the generator actually uses the latent. (4) Controllability tests (Section 11.4) directly verify that editing the latent produces corresponding motion changes.

**Q: "Why not use a physics simulator?"**
A: Physics-in-the-loop methods (InterPhys, CooHOI) achieve strong physical plausibility but require 4-8x A100 for weeks and complex RL pipelines. Our approach achieves object-adaptive behavior with pseudo-labels on a single GPU in days. We target a different point on the compute-capability tradeoff curve — and we argue that for many applications, object-adaptive motion strategy is more important than exact physical fidelity.

---

## 16. Risk Mitigation & Reviewer Defense

| Risk | Mitigation |
|------|------------|
| Pseudo-label noise | Soft labels + temporal smoothing + region-level (not point-level) supervision |
| Generator ignores z_int | Consistency loss + classifier-free guidance on interaction tokens + dropout training |
| SMPL-X to SMPL information loss | Acceptable for full-body tasks; hand-level tasks deferred to v2 |
| InterAct data format issues | Start with OMOMO (smaller, well-documented) as sanity check, then scale to InterAct |
| MLD codebase compatibility | Pin MLD commit hash; test integration early in Week 1 |
| Reviewer says "incremental over Move as You Say" | Frame paper around object-adaptive problem (not two-stage architecture); lead with ASS/ASC metrics; show Move as You Say cannot do attribute-sensitive generation |
| Attribute sensitivity not significant in experiments | Ensure training data covers diverse object attributes; if not, augment by scaling object meshes and adjusting pseudo-labels accordingly |
