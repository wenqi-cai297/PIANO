# Restart Prompt

Paste the text below verbatim after `/clear` to hand a fresh Claude
instance enough context to match the state the previous instance had.
The cost is a few minutes of reading; the benefit is that the new
instance won't fumble into already-ruled-out paths or misread which
long-running jobs are live.

Keep this file up to date when the project's "what to read to
onboard" set changes. For routine edits the prompt doesn't need to
move; update only when a tier shifts (new must-read analysis, new
key module, new reference review, etc.).

---

```
我们在 PIANO 项目 pseudo-label 抽取的中段。在动手任何事情之前，按下面的顺序
读全所有文件（不是摘要，是逐字读），然后按指定格式汇报，等我确认。

## Tier 1 — 当前状态（必读）
- PROGRESS.md（特别看 §0 Active Long Runs、§1 Module Status、§4 Commit History 最近 5 行）
- PLAN.md（特别看 §1.2 Data Preparation、§3.1 P1/P2 items 表）
- ANALYSIS.md（索引表，看哪些 analysis 是近期的）
- 跑 `git log -n 20 --oneline` 看最近 commit 序列

## Tier 2 — 关键决策上下文（必读）
- analyses/2026-04-20_codex_review_p0_fixes.md
- analyses/2026-04-20_pseudo_label_stats_v1_diagnosis.md
- analyses/2026-04-21_text_annotation_probe_dead_end.md
- SUGGESTION.md（Codex 对 pseudo-label 的整体 review）
- support.md（Codex 对 support label 的深度批评）

## Tier 3 — 代码当前值（读关键行）
- src/piano/data/pseudo_labels/extract_contact.py  ← 看 DEFAULT_DISTANCE_THRESHOLDS 和注释
- src/piano/data/pseudo_labels/stats.py            ← 看 aggregate_stats + make_quality_flags
- src/piano/data/pseudo_labels/run_all.py          ← 看 run_pipeline 流程
- src/piano/data/preprocess_interact.py            ← 看 preprocess_sequence 保存了哪些字段

## Tier 4 — 总体设计（扫一眼，不必精读）
- SPEC.md §1-3（problem statement + 两段式管线 + interaction latent 定义）

## 读完后按下面九点回我

1. **当前在跑什么**：§0 表格里 🔄 的行是什么、ETA、预计什么时候出结果
2. **最近 5 个 commit**：hash + 一句话说明作用
3. **阈值现状 + 依据**：hand/foot/pelvis 三组值，每组背后有几重证据
4. **已放弃的路径**：stricter-prior via text window 为什么不做了（一句话）
5. **v2 pass bar**：4 个 subset 各自的硬判定标准
6. **v2 之后的下一步**：按 PLAN.md 应该做什么
7. **用户偏好 3 条**：从 MEMORY 里抽最重要的三条（提示：auto-update docs / verify claims / active runs 是近期加的）
8. **整体架构复述**：用一段话复述你对 PIANO 整体 two-stage 架构的理解，以及 z_int 为什么是 per-frame 四元组，把 SPEC 的核心主张也过一遍
9. **你不确定的**：读完还有哪些地方不清楚？

在我对你的汇报确认之前，不要改代码、不要起长跑任务、不要 commit。
```

---

## 为什么要九点

- 1-2 定位"现在在哪"
- 3-6 恢复决策记忆（为什么阈值是这个值、什么路径已排除、什么是 pass bar、下一步是什么）
- 7 偏好层面（包括 memory 已经自动加载的规则）
- 8 verbal recap 强制新实例复述 SPEC 核心主张，让它对"为什么 PIANO 不是 Move-as-You-Say+"、"为什么 z_int 要结构化"这种 framing-level 问题有 active model
- 9 留出诚实出口 — 读完还不清楚的地方直接说，不要装懂

## 使用建议

- 跑了长任务刚完成后发回结果时，把新的 summary.json 贴到 prompt 后面，新实例一次性拿到状态 + 结果数据
- 长期维护：当新加了一个重要的 analysis 或换了关键模块，更新 Tier 2 / Tier 3 的清单
