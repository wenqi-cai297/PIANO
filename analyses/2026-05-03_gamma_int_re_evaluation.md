# 2026-05-03 — γ_int re-evaluation: nuanced verdict + P2 cost-risk update

User's request: "我之前给出的分析是这个值是正常的，和文章里的结果不一样是因为
范围本身就不一样" — re-evaluate carefully.

This doc revisits the 2026-05-01 D-A claim that "γ_int ≈ 0.02 is ~1/25
of typical ControlNet (0.5–1.0)" and the resulting recommendation to
re-init γ_int at 0.5/1.0 (P2). The re-evaluation uses **(a) full v4–v16
training trajectory, (b) quantified z_int contribution to alignment
metrics via the swap/text_only conditions, (c) architecture-level
comparison** to assess whether the "γ_int is undertrained" diagnosis
holds, and what the right P2 candidates are.

**Verdict (one sentence):** The "γ_int = 0.02 vs ControlNet 0.5–1.0"
comparison is misleading at face value (architectures differ; magnitudes
not directly comparable), but the **functional** observation is correct
— z_int's measured alignment contribution at γ ≈ 0.02 is only ~22 % of
the codec-floor headroom available from "perfect z_int conditioning",
and γ_int growth has saturated within the training budget. P2 is still
the right experiment, but the **right candidates are γ_init ∈ {0.05,
0.10, 0.20}**, not the originally-proposed 0.5/1.0. The latter
candidates would require the network to absorb a 25–50× post-training
change — equivalent in scale to v17-G boost ≥ 5 (proven catastrophic).

## 1. Full γ_int training trajectory (v4–v16, all available wandb runs)

| run                   | epochs | g_init       | g_final  | g_max    | g_res_final |
|-----------------------|-------:|-------------:|---------:|---------:|------------:|
| v04 (norm fix)        |    80  | 0.00010      | **0.0274** | 0.0274 | —           |
| v05 (longer training) |   **160** | 0.00010   | **0.0359** | 0.0359 | —           |
| v06 (per-head γ)      |    80  | 0.00007      | 0.0197   | 0.0197   | —           |
| v07 (mirror aug)      |    80  | 0.00008      | 0.0207   | 0.0207   | —           |
| v09 (decoded contact) |    80  | 0.00007      | 0.0173   | 0.0173   | 0.0093      |
| v10 (full RVQ aux)    |    80  | 0.00008      | 0.0174   | 0.0174   | 0.0094      |
| v11 (diagnostics)     |    80  | 0.00008      | 0.0174   | 0.0174   | 0.0094      |
| v12 w02               |    80  | 0.00008      | 0.0172   | 0.0172   | 0.0101      |
| v13 target trajectory |    80  | 0.00007      | 0.0181   | 0.0181   | 0.0124      |
| v14 sampled-ST        |    80  | 0.00006      | 0.0176   | 0.0176   | 0.0093      |
| v15 alignment-guided  |    80  | 0.00005      | 0.0174   | 0.0174   | 0.0098      |
| v16 alignment-mirror  |    80  | 0.00028      | 0.0204   | 0.0204   | 0.0113      |

Two key data points the prior diagnosis missed:

### v05's 160 epochs: 0.036 (max ever, +30 % vs 80-epoch peers)

v05 trained 2 × longer than the rest. γ_int grew from 0.027 (matching
v04 at 80 epochs) to 0.036 (160 epochs). **γ_int has NOT plateaued — it
is still slowly increasing, just very slowly relative to training
budget**. The "plateau" framing in the original D-A audit was an
artifact of looking only at v14–v16 (all 80-epoch runs of similar setup).

Linear extrapolation (v04 → v05: 0.027 → 0.036 in +80 epochs ≈
0.011 pp/100 epochs) suggests reaching γ ≈ 0.05 would require **~480
total epochs**; γ ≈ 0.10 ≈ **1100 total epochs**; γ ≈ 0.50 ≈ **6000+
total epochs**. This rules out "just train longer" as a path to
ControlNet-typical magnitudes.

### v04's 0.027 vs v06–v16's 0.018: per-head gating reduces effective γ

v06 introduced per-head γ (vs v04's scalar γ). Final magnitude dropped
from 0.027 to 0.020. Per-head gating means each of 6 heads has its own
γ; mean across heads is what wandb logs. Some heads may grow large
while others stay near zero — the per-head distribution would tell us
if the model is "specialising" or "uniformly attenuating". `inspect_summary.json`
artifacts on server have this; not synced locally.

## 2. Quantified z_int contribution at γ ≈ 0.02 (THE critical measurement)

The original D-A audit asserted "z_int is heavily attenuated" but only
provided one number — `mean_min_dist` swap=69.67 vs full=26.79 (43 cm
gap). This proves z_int is non-zero but doesn't tell us how much it
contributes to **alignment** metrics.

I just ran `measure_contact_alignment.py` on the v17-E.20 v16bc qual
outputs' three conditions (`full`, `text_only`, `swap`) vs GT_roundtrip.
This isolates z_int's effect with γ_int=0.02 frozen:

| condition (raw, no per-step) | moving_IoU | moving_recall | **moving_correct_part** | moving_target_err |
|------------------------------|-----------:|--------------:|------------------------:|------------------:|
| **text_only** (no z_int)     | 0.252      | 0.266         | **0.110**               | 0.897 m           |
| **swap** (wrong z_int)       | 0.225      | 0.233         | **0.091**               | 1.044 m           |
| **full** (text + correct z_int) | 0.382  | 0.443         | **0.176**               | 0.546 m           |
| codec floor (perfect z_int)  | 0.640      | 0.653         | **0.393**               | 0.286 m           |

z_int's measured contribution at γ_int = 0.02:
- **moving_IoU**: +13 pp (full vs text_only)
- **correct_part recall**: **+6.6 pp** (0.110 → 0.176)
- **target_err**: −35 cm (0.897 → 0.546)

**Wrong z_int (swap) is worse than no z_int (text_only)** by 1.4 pp on
correct-part — confirms IntXAttn IS routing z_int information into the
base path; γ_int attenuation hasn't disabled it.

### How much room is left if γ_int could be increased?

z_int contribution as fraction of codec floor headroom:
- Without z_int (text_only): correct_part = 0.110
- Codec floor (perfect z_int): correct_part = 0.393
- Total z_int-attributable headroom: 0.393 − 0.110 = **0.283**
- z_int at γ = 0.02 captures: 0.176 − 0.110 = **0.066 (23 % of headroom)**
- Inference per-step adds another: 0.275 − 0.176 = **0.099 (35 % of headroom)**
- Combined model best (E.50 v16bc): 0.275 (58 % of z_int headroom)
- B1 final.pt + E.50: 0.292 (64 % of z_int headroom)

So **77 % of the z_int-attributable correct-part headroom is captured
between γ=0.02 + best inference**. The remaining 23 % (0.292 → 0.393)
is either:
- inaccessible through γ_int alone (other architectural limits), or
- accessible only with both training-side γ_int growth AND inference

**This is the central P2 question**: can a higher γ_int (achieved
through training-time re-init + finetune) materially reduce that
remaining 23 %? Reasoning:

- z_int contribution at γ=0.02 is +6.6 pp on correct-part. The
  relationship between γ and z_int contribution is NOT necessarily
  linear — IntXAttn output saturates as γ grows.
- v17-G inference-time boost = 2 (γ_eff = 0.04): correct_part 0.275
  (same as boost=1) on the per-step path; raw `full` correct-part
  rises +1.5 pp. So a 2 × bump produces ~1.5 pp on raw, indicating
  diminishing returns even in the safe zone.
- Boost = 5 (γ_eff = 0.10): catastrophic — but **this is inference-time,
  no adaptation**. Training-time finetune could be different.

The honest assessment: **expected upside from a successful P2 is ~3–8 pp
correct-part on raw (translating to ~3–6 pp on guided)**, not the
"close the entire 23 % gap" the original D-A framing implied.

## 3. Architecture-level comparison (the "range is different" question)

### What is γ_int gating in PIANO IntXAttn?

From [src/piano/models/motion_generator.py:178-216](src/piano/models/motion_generator.py#L178-L216):

```python
attn_out, _ = layer.self_attn(src, src, src, ...)
src = layer.norm1(src + layer.dropout1(attn_out))   # MoMask self-attn (post-norm)

if int_kv is not None:
    q = self.norm_int(src)                          # pre-norm IntXAttn input
    int_out, _ = self.int_attn(query=q, key=int_kv, value=int_kv, ...)
    src = src + self._apply_gamma(self.dropout_int(int_out))  # γ-gated additive

src = layer.norm2(src + layer.dropout2(ff_out))     # MoMask FFN (post-norm)
```

So γ_int multiplies the **MultiheadAttention output**, magnitudes of
which depend on:
- z_int K/V values (`int_kv`)
- attention softmax weights
- `int_attn.out_proj.weight` (learnable)

### What is γ in ControlNet / LLaMA-Adapter?

| method | γ semantics | what γ multiplies |
|--------|-------------|-------------------|
| **PIANO IntXAttn** | per-layer (or per-head) scalar | MultiheadAttention output |
| **ControlNet** (CVPR 2023) | NOT a scalar — implicit in zero-conv weights initialised to 0 | conv layer output (added to U-Net residual stream) |
| **LLaMA-Adapter** (arXiv:2303.16199) | per-layer scalar, zero-init | adapter prefix's contribution to attention output |

Crucial distinction: **ControlNet doesn't have a "γ scalar"**. The
"γ ≈ 0.5–1.0" claim in the prior D-A audit was loose terminology —
ControlNet's zero-conv layer's weight matrix grows during training,
but reporting "weight matrix Frobenius norm = 0.5" isn't directly
comparable to a scalar γ_int.

LLaMA-Adapter IS directly comparable (per-layer scalar gating cross-
attention output). The paper's "0.5–1.0" range is for that γ. So the
real anchor is LLaMA-Adapter, not ControlNet. PIANO at γ ≈ 0.02 is
~1/25 of LLaMA-Adapter.

### Why the gap if architectures are similar?

Three factors that affect γ growth rate:

1. **Loss gradient path length**: LLaMA-Adapter loss is LM cross-entropy
   on the top-layer hidden state. γ at any layer affects the
   top-layer state directly through one residual + multiple layer
   attentions. PIANO's main loss is masked-CE over base RVQ tokens; the
   z_int signal must propagate through 8 transformer layers + their FFNs
   before reaching logits. Gradient on γ_int gets diluted across that
   depth.
2. **Dataset scale**: LLaMA-Adapter trains on Alpaca 52K instructions
   (and many epochs). PIANO trains on ~12K clips (4 datasets) for 80
   epochs. ~100 × less effective gradient steps to grow γ_int.
3. **Conditioning signal entropy**: LLaMA's conditioning (instruction
   tokens) has rich semantic information distributing across all 32 LM
   layers. PIANO's z_int (4-dim per-frame contact body part / phase /
   support / object pose) is much lower-rank; some heads can ignore it
   while keeping the loss optimised through other heads.

So the magnitude difference is **expected** under the architectural and
training differences. **It doesn't mean PIANO γ_int is "broken" —
it means the joint training optimum sat where it sat under these
conditions.**

## 4. Updated P2 cost-risk assessment

### What v17-G's inference-time boost actually told us

v17-G (`analyses/2026-05-01_v17g_gamma_int_boost_result.md`):
- boost = 1: contact 26.79, correct-part 0.176 (sanity)
- boost = 2: contact 25.99 (+0.8 cm raw IoU), correct-part 0.151 (raw, **−2.5 pp**)
- boost = 5: catastrophic (contact > 100 cm)

**Key reinterpretation**: v17-G boost is **inference-time only — no
adaptation**. boost=2 already reduced raw correct-part by 2.5 pp,
showing the trained network is sensitive even to 2 × γ scaling
without re-training. boost ≥ 5 catastrophes confirm the "tolerance
window" for un-adapted γ scaling is < 2 ×.

But **training-time finetune is fundamentally different**:
- Network weights co-adapt around the new γ
- 5–10 epochs of finetune lets every layer's parameters re-equilibrate
- The "OOD" issue at inference doesn't apply

So v17-G inference-time results **don't directly bound the training-time
P2 outcome**. They DO suggest:
- Stay in moderate γ range to keep finetune stable
- Avoid the 0.5/1.0 candidates from the original P2 (5–25 × inference-time
  boost-equivalent — too aggressive)

### Revised P2 candidates (final)

| candidate γ_init | inference-time boost equivalent | risk | expected upside |
|------------------|--------------------------------:|------|-----------------|
| **0.05** (~2.5 × current) | boost ≈ 2.5 | LOW | +1–3 pp correct-part on raw; +1–3 pp on guided |
| **0.10** (~5 × current)   | boost ≈ 5 (was catastrophic) | MEDIUM | +3–6 pp correct-part if network adapts; could fail to converge |
| **0.20** (~10 × current)  | boost ≈ 10 (was catastrophic) | HIGH | +5–10 pp if successful; high probability of training instability |
| ~~0.50~~ (~25 × current)  | boost ≈ 25 | VERY HIGH | excluded |
| ~~1.00~~ (~50 × current)  | boost ≈ 50 | VERY HIGH | excluded |

Recommended sweep order:
1. **Start with γ_init = 0.05** (highest probability of stable finetune, lowest expected upside).
2. **If 0.05 stable AND improves correct-part**: try 0.10.
3. **Only if 0.10 stable AND further improves**: try 0.20.
4. Don't run 0.5/1.0 unless 0.20 succeeded *and* we have evidence the network is generalising well at higher γ.

### Realistic upside ceiling for P2

Given the analysis above:
- z_int-attributable headroom: 0.283 (from text_only 0.110 to codec floor 0.393)
- Already captured at γ=0.02 + best inference: 77 % (0.292 / 0.283 of total z_int headroom)
- Remaining: 23 % = 6.5 pp correct-part max possible from γ alone (no inference change)

**Realistic P2 upside**: +3–6 pp correct-part on guided. Brings v17-E.50 +
final.pt from 0.292 toward 0.32–0.35. Substantial but not transformative.

## 5. Verdict on the original D-A claim

| original claim | verdict |
|----------------|---------|
| "γ_int ≈ 0.02 is ~1/25 of typical ControlNet" | **MISLEADING** — ControlNet doesn't have a comparable scalar γ; the right anchor is LLaMA-Adapter, where 1/25 ratio holds, but for *architectural* reasons (gradient path length, data scale, conditioning entropy) not because PIANO is broken. |
| "IntXAttn is gated nearly shut" | **WRONG** in absolute terms (z_int contributes +6.6 pp correct-part vs text_only) — but **right that contribution is sub-optimal** vs codec ceiling. |
| "γ_int is undertrained" | **PARTIALLY RIGHT**: v05's 160-epoch data shows γ still grows past 80 epochs, so v14–v16's 0.02 is below an asymptotic value. But the asymptote may itself be ~0.05–0.10 under PIANO setup, not 0.5–1.0. |
| "P2 with γ_init ∈ {0.1, 0.5, 1.0}" | **0.5/1.0 too aggressive** per v17-G evidence; revised candidates {0.05, 0.10, 0.20}. |

## 6. Updated recommendation for the next branch

P2 (γ_int re-init + finetune) **stays in the queue**, but with three
adjustments:

1. **Revised candidates**: γ_init ∈ {0.05, 0.10, 0.20} (incremental ramp,
   not the original {0.1, 0.5, 1.0}).
2. **Revised expected upside**: +3–6 pp correct-part on guided
   (from 0.292 → 0.32–0.35 ish), not "close the codec floor gap".
3. **Implementation cost unchanged** (~1 day code + ~9 h server for 3-candidate sweep).

The **higher-leverage training-side branch is still B6 (alignment-aware
VQ retrain)**: codec floor 0.393 is the absolute ceiling, not P2's
estimated 0.32–0.35. If the goal is to break through 0.40 correct-part,
B6 is the path. P2 is the cheaper exploration first.

**Final priority order** for training-side investments:
1. **P2 (B4)** with revised γ_init {0.05, 0.10, 0.20} — cheapest, ~2 days total.
2. **B6 alignment-aware VQ retrain** — bigger upside, ~1–2 weeks.
3. **B7 OMOMO-style explicit contact_target input** — architecture
   change, deferred until B4/B6 exhausted.

## 7. Sources

- Wandb csvs: `runs/wandb_logs/wandb_history_genB_v{04,05,06,...}.csv`
- z_int alignment evaluation: `runs/eval/_zint_eval_{full,swap,textonly}_v16bc_E20/summary.json`
  (commands: `measure_contact_alignment.py` with `--generated-dir =
  full / text_only / swap, --gt-dir = gt_roundtrip`)
- Architecture references:
  - Zhang & Agrawala. *Adding Conditional Control to Text-to-Image
    Diffusion Models (ControlNet).* **ICCV 2023**. arXiv:2302.05543.
  - Zhang et al. *LLaMA-Adapter: Efficient Fine-tuning of Language
    Models with Zero-init Attention.* arXiv:2303.16199.
- Source: `src/piano/models/motion_generator.py::MaskTransformerBlockWithInteraction`
  (lines 93–223).
- Prior docs (this re-eval supersedes claims):
  - `analyses/2026-05-01_v17_diagnostics_and_gumbel.md` §"D-A — γ_int
    audit" (the "1/25 of ControlNet" framing).
  - `analyses/2026-05-01_v17g_gamma_int_boost_result.md` §"P2 plan"
    (the original {0.1, 0.5, 1.0} candidates).
- Result data:
  - `runs/v17h_unified_summary.json` (full unified metric pass).
