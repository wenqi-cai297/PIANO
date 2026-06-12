# R43 后方法瓶颈复盘与 Pipeline 审查任务书（给 Claude Code）

日期：2026-06-12  
撰写者：Codex  
面向：Claude Code  
目的：不要继续围绕 R43 P0 的局部参数小修小补。请基于本文重新审查整个 coarse-to-fine pipeline，定位真正的 bottleneck：是 Stage1 生成分布、阶段分解接口、条件信息不足、采样/选择机制，还是训练/推理实现不一致。

## 0. 先给结论

我不认为 R43 P0 的失败应该被解读为“Stage1.5 mixed 训练的某个细节没调好”。它更像是在暴露一个更上游的问题：

**Stage1 把多模态 motion generation 的 mode collapse 压缩到了 23-D coarse plan 上。Stage1.5 和 Stage2 在 oracle 条件下可以工作，但 generated Stage1 coarse 的分布不够像 oracle，且缺少/压扁了 downstream 所需的 mode 与 dynamics 信息。**

因此下一步不应该优先继续调 `stage1p5_generated_prob`、cond aug sigma、训练 epoch、loss 权重这些局部 knob。那些改动可能保护某个 ablation cell，但大概率不能解决 GG（generated Stage1 + generated Stage1.5）失败。

请你做一轮“pipeline bottleneck review”，用代码和实验结果回答：

1. Stage1 生成的 23-D plan 是否真的覆盖了 oracle plan 的分布与动态？
2. 当前 23-D Stage1 表示是否足以承载 Stage1.5/PB1 需要的 motion mode？
3. 如果 Stage1 cache 已经 collapse，Stage1.5 mixed training 是否只是把下游适配到错误输入？
4. 训练和推理的条件流是否完全一致，有没有 hidden oracle 或 convention mismatch？
5. 如果要跳出瓶颈，应该改 sampling/selection、加 inference-feasible mode condition、改 Stage1 表示，还是重构分解方式？

## 1. 必须先读的文件和结果

请按这个顺序读，不要跳过。读完每个 stage 后做一次自己的 code review 笔记，再进入下一 stage。

### Stage A：先读负面结果和历史证据

- `analyses/2026-06-12_r43_p0_verdict_negative_for_codex.md`
- `analyses/2026-06-01_stage1_underdetermination_for_codex.md`
- `analyses/2026-06-01_round38_verdict_for_codex.md`
- `analyses/2026-06-02_r41_calibration_verdict_for_codex.md`
- tarball:
  - `analyses/round42_cond_2x2_20260611_205914.tar.gz`
  - `analyses/round43_p0_results_20260612_084727.tar.gz`
  - `analyses/round41_cascade_results_20260611_201636.tar.gz`

关键数值证据：

| diag | OO | GO | OG | GG |
| --- | ---: | ---: | ---: | ---: |
| R42 drift mean cm | 7.52 | 16.98 | 11.94 | 39.34 |
| R43 P0 drift mean cm | 7.48 | 16.98 | 34.85 | 41.51 |
| R42 pelvis mean cm | 3.21 | 4.28 | 5.48 | 33.91 |
| R43 P0 pelvis mean cm | 3.17 | 4.35 | 27.69 | 34.26 |

我的解读：

- OO 稳定，说明 PB1 在 oracle Stage1 + oracle Stage1.5 条件下是好的。
- GO 稳定在 16.98，说明“generated Stage1 + oracle Stage1.5”已经有明显上游误差，但不是最坏。
- R43 P0 把 OG 从 11.94 打到 34.85，说明 mixed Stage1.5 适配 generated cache 后严重伤害 oracle compatibility。
- GG 从 39.34 到 41.51，没有改善。也就是说 R43 P0 既没有解决 generated pipeline，也破坏了 oracle Stage1 compatibility。
- R43 cache audit 已经报告 14/23 个 Stage1 channel 分布异常，其中很多 std 远小于 1。这不是一个 benign perturbation；这是 generated Stage1 cache 本身不够像训练时的 oracle z-space。

R43 P0 的 val loss 下降不能作为正向证据。它只说明 Stage1.5 能拟合当前训练目标，不说明它输出的 C41/S4 对 PB1 downstream 有用。

### Stage B：读 pipeline 实现，不要只读脚本

请重点读以下文件，并在 review 文档中引用具体代码行。

Stage1 target 和 cascade loss：

- `src/piano/training/train_stage1.py`
  - `extract_coarse_v1_batched(...)` 和 z-score target 的位置：约 `474-482`
  - diffusion x0 target 是 23-D z-space：约 `534-555`
  - velocity / raw dynamics loss：约 `563-585`, `805-824`
  - R41 cascade through frozen PB1：约 `849-1012`

Stage1 表示：

- `src/piano/data/stage1_coarse_oracle.py`
- `src/piano/models/stage1_trajectory.py`

当前 23-D coarse layout 是：

- `[0:3]` root local x,z,y
- `[3:6]` root velocity x,z,y
- `[6:9]` yaw sin, yaw cos, yaw_vel
- `[9:15]` pelvis rot6d
- `[15:21]` spine3 rot6d
- `[21]` head height
- `[22]` shoulder center height

这 23-D 不包含 wrists、knees、feet、explicit contact、phase、stance、hand-object local geometry。它是一个很窄的抽象 plan。请认真判断：这个 plan 是否足以 disambiguate Stage1.5/PB1 需要的 motion mode？

Stage1.5 condition source：

- `src/piano/training/train_stage1p5.py`
  - oracle Stage1 计算与 cond source selection：约 `242-267`
  - `cond["stage1_coarse"]` 进入模型：约 `277`
  - x0 是 raw C41/S4 target：约 `319-339`
  - non-oracle condition 约束与 config validation：约 `843-888`
- `src/piano/training/stage1p5_cond_sources.py`
  - generated cache layout：约 `45`
  - shape/NaN validation：约 `125`
  - mixed oracle/generated selection：约 `189-207`

PB1 / Stage2 condition usage：

- `src/piano/training/train_anchordiff.py`
  - PB1 training 中从 GT motion 抽 oracle `stage1_coarse` 并 z-score：约 `387-406`
  - PB1 还吃 `stage2_coarse_extra` C41 和 `stage2_support` S4。

R43 scripts / audit：

- `scripts/stage_a_generator/run_round43_p0_pipeline.sh`
  - Step 3 调用 cache audit，但没有强制 `--fail-on-warnings`。
- `scripts/stage_a_generator/round43_p0_cache_audit.py`
  - 当前 distribution mismatch 默认只是 warning；只有传 `--fail-on-warnings` 才 hard fail。

我的代码层面担忧：

- R43 pipeline 已经看到了 Stage1 cache 的分布 collapse，但 audit 没有阻止后续 Stage1.5 训练。这个 guardrail 要修，但它只是防止浪费训练，不是方法解。
- R41 Stage1 cascade A2 在结果 tarball 里的 `configs/training/stage1_r41_a2_world_vel.yaml` 显示 `cascade.w_total: 1.0`，不是一个强校准后的 cascade objective。A2 的 direct downstream 接近 R42 GO，而不是解决 Stage1。
- Stage1 model 的条件主要是 object trajectory/object tokens/text/init pose；它没有额外 mode variable。面对同一 object/text/path 对应多个合理 human mode 时，它天然会被 MSE/x0 regression 拉向平均模态。

## 2. 请你检查的核心假设

### 假设 1：Stage1 生成分布是主瓶颈

证据：

- R35/R41 OOD audit 都显示 Stage1 generated coarse 的 std/dynamics 低于 oracle。
- R43 generated cache audit 直接显示 14/23 channel mean/std gap。
- R42 GO 已经显著差于 OO；GG 更差。
- R43 适配 Stage1.5 到 generated cache 后，GG 没改善，OG 反而崩。

请你不要只看 drift mean。请看：

- per-channel mean/std gap
- group gap：root velocity、yaw、pelvis rot6d、spine rot6d、height
- temporal dynamics：velocity/acceleration distribution
- sample diversity 和 sample quality 的关系
- Stage1 generated cache 是否只是 first-order calibration 问题，还是 semantic/mode 错误

### 假设 2：Stage1 23-D 表示太窄，压掉了下游需要的 mode 信息

Stage1.5/PB1 需要生成 fine contact 和 full motion，但 Stage1 只给 root/yaw/pelvis/spine/height。即使 oracle 23-D 可以让 PB1 工作，也不代表 generated 23-D 容易学习。oracle 23-D 可能携带了很多由 GT motion 隐含编码的微妙 mode；generated Stage1 如果学成平均，就失去这些隐含 cue。

请重点判断：

- 23-D oracle 的哪些 channel 对 PB1 downstream 最敏感？
- R43 collapsed 的 channel 是否正好是 PB1 敏感 channel？
- 是否需要 Stage1 显式输出 mode-controlling variables，例如 phase/stance/contact plan/high-level posture archetype/facing mode？
- 是否应该让 Stage1 输出更接近 downstream 所需的 intermediate，而不是只输出 root/pelvis/spine 的平均 plan？

### 假设 3：Stage1.5 mixed training 不是解法，而是 exposure-bias patch

Stage1.5 mixed training 的前提是 generated Stage1 cache 只是轻微 OOD，仍然保留有效 mode 信息。R43 P0 的结果说明这个前提不成立。

如果输入 cache 已经 collapsed，Stage1.5 学会“吃 collapsed Stage1”可能只会：

- 破坏 oracle Stage1 compatibility；
- 学到对 PB1 不可用的 C41/S4；
- 让 val loss 看起来不错但 downstream 变差；
- 把错误从 Stage1 显式传播到 Stage1.5。

请你审查：Stage1.5 是否应该暂时停止 generated-cache retraining，直到 Stage1 source cache 过硬 audit。

## 3. 你要做的 review / diag 计划

下面不是让你直接做大规模训练，而是先定位 bottleneck。每完成一个 stage，请做一次 code review，确认结果和脚本没有 bug，再继续。

### Stage 0：结果与 provenance 核验

目标：确认我们不是在比较错 config、错 checkpoint、错 tarball。

请核验：

- R42/R43 的 diag configs 是否对应同一个 PB1 checkpoint。
- R43 P0 Stage1.5 config 是否确实是 `stage1p5_stage1_cond_source: mixed` 和 `stage1p5_generated_prob: 0.8`。
- R43 generated cache 是否来自 A2 Stage1，而不是旧 checkpoint。
- R41 A2 的 actual config 是否确实 `cascade.w_total: 1.0`，并在 review 中说明这意味着什么。
- 当前 repo 中有些 R41 configs 可能只在 tarball 或 `analyses/configs/...` 中，不一定在 `configs/training/` 里。请注意 provenance，不要假设工作树里的文件就是当时训练用的文件。

输出：

- 一个小表：experiment、checkpoint、config、tarball、关键参数。
- 如果发现任何 provenance mismatch，先停下来写明，不要继续做策略判断。

### Stage 1：训练/推理 contract review

目标：确认不是实现 mismatch 造成的假 bottleneck。

请逐项比对：

- Stage1 training 的 cond 构造 vs `sample_substitute_conds_cli.py` / cache generation 的 cond 构造。
- Stage1 inference 输出是否 z-scored，cache 是否保存 z-space，Stage1.5 loader 是否按 z-space 读。
- Stage1.5 training 的 oracle/generated/mixed selection 是否和 R42/R43 diag 使用的 condition source 一致。
- PB1 training 用 oracle `stage1_coarse` 的 z-score 统计，diag 替换 generated cond 时是否用同一套 stats。
- object trajectory、frame index、padding/mask、canonical/world/root0 convention 是否跨阶段一致。
- `init_pose` F1/F2 是否是合法 inference-time condition。尤其要问：如果真实应用没有 GT initial pose，那么 F1 也是 hidden oracle；如果真实任务允许用户给初始姿态，则 F1 合法。

输出：

- 一张 “condition contract map”：每个 stage 输入什么、从哪里来、训练时是什么、推理时是什么。
- 所有 mismatch 必须标 P0/P1/P2。

### Stage 2：Stage1 source / sampler 诊断，不训练 Stage1.5

目标：先判断 Stage1 是否能产生可用的 source distribution。

请对多个 Stage1 source 生成 cache，并运行同一个 audit：

- V8/V6 baseline（如果 checkpoint 可用）
- R41 A0/A1/A2/A3/A4（如果 checkpoint 可用）
- R40/R41 其他代表性 variants（如果 checkpoint 可用）

对每个 source 记录：

- generated cache audit：bad channel 数、mean/std gap、group gap
- Stage1 OOD summary：std ratio、velocity ratio、acceleration ratio
- K-sample diversity summary
- GO downstream drift（generated Stage1 + oracle Stage1.5）
- 如果便宜，再做 GG，但不要一开始就全量。

重要：K-sample diversity 不能直接等于好。A2 可以有 sample diversity，但 aggregate distribution 仍 collapse。你要检查“多样但不对”还是“根本不多样”。

建议成功门槛：

- 不要用单一 drift mean 决策。
- generated cache bad channel 数应显著低于 R43 A2 的 14/23。
- root velocity、yaw、pelvis、height 这些 group 的 std/dynamics 不能明显低于 oracle。
- GO 至少要明显优于 R42/R43 的 16.98，或者在同等 GO 下有更好的 distribution audit。

如果所有 source 都不过 audit，请不要继续 Stage1.5 mixed 训练。那说明要改 Stage1 或分解接口。

### Stage 3：定位 collapse 类型的诊断

目标：区分“calibration 问题”“sampling/selection 问题”和“模型/表示问题”。

请实现或复用最小诊断，不要直接大改训练。

#### 3.1 Channel-wise affine calibration 诊断

对 generated Stage1 cache 做 per-channel mean/std affine correction，让它的一阶统计匹配 oracle train stats。然后跑 GO/GG 或至少 GO。

这不是最终方案，只是诊断：

- 如果 affine calibration 明显改善，说明 bottleneck 有很强的 calibration / normalization component。
- 如果没改善，说明问题不是一阶统计，而是 temporal/semantic/mode 内容错误。

注意不要把这个当作正式方法上线，因为它可能把不合理 trajectory 强行拉到 oracle stats。

#### 3.2 Best-of-K oracle selection 诊断

对每个 sample 生成 K 个 Stage1 coarse。用 oracle metric 选 best one，再跑 GO/GG。

这也是诊断，不是可部署方案：

- 如果 best-of-K 大幅改善，说明 Stage1 sampler 里存在好 mode，但缺少 inference-time selection/reranking。
- 如果 best-of-K 也不改善，说明 Stage1 模型没有生成足够好的 mode，不能靠 sampler tweak 解决。

请同时记录 random-K 的均值/方差，不要只报告 best。

#### 3.3 Stage1 sampler / CFG sweep

如果现有采样支持 DDIM/DDPM/cfg scale，请扫：

- sampler：DDIM eta=0、DDPM deterministic、DDPM stochastic（按现有支持）
- cfg scale：1.0、1.5、2.0、3.0
- seeds：至少 3 个

评价不是只看 downstream，还要看 cache audit。CFG 可能让 motion 更 sharp，也可能让分布更偏。

### Stage 4：策略决策树

请基于前面诊断选择路线，不要提前定论。

#### 路线 A：Stage1 能生成好 mode，但缺少选择机制

证据：

- K-sample best-of-K 明显改善；
- random samples 中存在接近 oracle 的 candidate；
- aggregate audit 不是灾难性 collapse。

下一步：

- 设计 inference-feasible reranker / energy / self-consistency score。
- 用 object-contact plausibility、PB1 predicted consistency、trajectory smoothness、contact prior 等不依赖 GT 的指标选样。
- 之后再考虑 Stage1.5 轻量 mixed 适配。

#### 路线 B：Stage1 输入条件不足，导致多模态困惑

证据：

- 同样 object/text/init_pose 对应多个合理 Stage1 oracle；
- generated Stage1 平均化；
- 加强 sampling 也不能稳定得到好 mode。

下一步：

- 加 inference-feasible mode condition，而不是继续 MSE 强拟合平均。
- 候选 mode variable：
  - gait phase / stance side
  - contact plan / hand-object side
  - posture archetype
  - coarse action submode
  - facing/turning mode
- 这要像之前解决 gait 问题那样，把“哪个脚先动都可以”的多模态因素显式化，让模型自然落在某个 mode，而不是平均。

#### 路线 C：Stage1 表示太窄

证据：

- oracle 23-D 对 PB1 有用，但 generated 23-D 无法携带足够 cue；
- PB1 敏感 channel 正好是 Stage1 collapse channel；
- Stage1.5/PB1 需要 contact/phase/limb-mode，但 Stage1 不输出。

下一步：

- 重新设计 intermediate plan，使 Stage1 输出更接近 downstream 所需变量。
- 可能把 Stage1 与 Stage1.5 合并为一个 latent plan generator，直接输出 23-D + C41/S4 的 coherent plan。
- 或让 Stage1 输出 additional mode/control channels，再由 Stage1.5 refine。

#### 路线 D：实现 mismatch

证据：

- condition convention、z-score、mask、frame、cache path、checkpoint provenance 有 bug。

下一步：

- 先修实现，不做方法判断。
- 修完重跑最小 OO/GO/OG/GG。

## 4. 允许先修的小 guardrail

这个可以先做，但请明确它不是方法突破：

- 在 `scripts/stage_a_generator/run_round43_p0_pipeline.sh` 的 cache audit 调用中加入 `--fail-on-warnings`，或提供一个明确的 `--strict-audit` 默认开启。
- 在 `round43_p0_cache_audit.py` 中把 distribution warning 的返回码策略改清楚：训练 generated-cache Stage1.5 时，bad channel 过多必须 hard fail。
- 把 bad channel 表打进最终 pack，避免只看 downstream summary。

这能防止我们再次在明显 collapsed 的 cache 上训练 Stage1.5，但不能解决 Stage1 collapse 本身。

## 5. 不要做的事

请避免这些方向，除非前面的诊断明确支持：

- 不要直接再跑一个 `generated_prob=0.2/0.5/0.8` 的 Stage1.5 grid，把它当主线。
- 不要把 cond aug sigma 从 0.02 改到 0.01/0.05 当主线。
- 不要用 Stage1.5 val loss 或 C41/S4 MSE 证明方法有效。
- 不要因为 A2 的 GO 比 GG 好，就说 A2 已经解决 Stage1。GO=16.98 仍然明显差于 OO=7.5。
- 不要因为 K-sample diversity 存在，就说 Stage1 没 collapse。要看样本是否覆盖 oracle distribution 和 downstream useful modes。
- 不要在 Stage1 source cache 没过 audit 之前启动新的 Stage1.5 mixed 长训练。

## 6. 你最终要返回给 Codex 的文档

请写：

`analyses/YYYY-MM-DD_r44_pipeline_bottleneck_review_return_for_codex.md`

文档必须包含：

1. 你读过的文件清单，带关键代码行引用。
2. R41/R42/R43 的结果表和你的复核结论。
3. 一张完整 condition flow map：Stage1、Stage1.5、PB1 的 train/inference 输入分别是什么。
4. 所有发现的问题，按 P0/P1/P2 排序。
5. 对四个 bottleneck 假设的证据判断：
   - Stage1 generated distribution
   - Stage1 representation sufficiency
   - Stage1.5 exposure-bias patch
   - implementation mismatch
6. 你建议的下一轮实验，不超过 3 个主实验。每个实验必须写：
   - 目的
   - 要验证/证伪的假设
   - 需要跑的脚本
   - 成功/失败判据
   - 如果失败，下一步怎么 pivot
7. 如果你改了任何代码，先做自己的 code review，再 commit push，并给服务器操作步骤。

## 7. 我的当前判断

我倾向于把 R44 定义为“Stage1 bottleneck localization”，而不是“Stage1.5 adaptation round”。

最值得优先做的不是再训练，而是这三个诊断：

1. **Stage1 source/sampler audit matrix**：确认有没有任何现有 Stage1 checkpoint/sampler 能产生接近 oracle 的 23-D distribution。
2. **Best-of-K oracle selection diagnostic**：判断好 mode 是否已经存在于 Stage1 samples 中。
3. **Affine calibration diagnostic**：判断问题是否主要是一阶分布错位，还是更深的 temporal/semantic/mode 错误。

如果这三个诊断都指向 Stage1 没有生成可用 mode，那么我们应该停止围绕 Stage1.5 微调，把精力转向 Stage1 的条件设计、mode variable、intermediate representation，甚至重新考虑 Stage1+1.5 是否应该合成一个 coherent plan generator。

