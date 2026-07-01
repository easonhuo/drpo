# Historical conclusions and provenance

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `history_provenance`
- Responsibility: Preserve restoration strategy, superseded statements, historical evidence boundaries, and source recovery rules.
- Dependencies: `global_core_governance`
- Content-contract topics: none
- Owned source blocks: 2
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: none
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000100:START -->
# 8. v4-v10 版本审计与恢复策略

| 版本 | 主要新增 | 是否保持累加 | 当前处理 |
|---|---|---|---|
| v4 | 锁定共识、两套环境职责、因果链、稳定外推计划、完整 related work、论文重构与文件索引 | 是 | 全部保留 |
| v5 | fixed-advantage 最小机制设定；明确未来统一环境是必做重构 | 是 | 全部保留 |
| v6 | sigma 方向修正；expected Fisher 纠错；均值—方差联合分析；原始代码诊断 | 是 | 全部保留并标记修正原论文 |
| v7 | 所谓统一 benchmark、三个 protocols、正式结果与代码索引 | 大体累加，但“统一环境”命名不准确 | 作为恢复底稿；相关结果降为旧分离环境开发证据 |
| v8 | 内容压缩、categorical 结果、方法筛选 | 否；发生破坏性删减 | 有效新增合入；删除动作不继承 |
| v9 | exponential-family 统一、自审、神经网络/critic 边界 | 在 v8 不完整底稿上新增 | 理论作为核心 patch 合入；过度完成声明撤回 |
| v10 | Hopper learned-critic probe | 独立报告 | 合入外部验证状态；明确 600-step 限制 |
| saturation audit | 部分稳定外推子实验长程复核 | 独立报告 | 合入收敛规范与修正结果；不得冒充全部实验审计 |

<!-- STAGE4B-SOURCE-BLOCK:B000100:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000101:START -->
## 8.1 为什么 v8 删除是不合理的

当时将“压缩重复”错误实现为“删除历史环境、related work、实验 provenance 和多版计划”。这些内容对于论文定位、避免重复劳动和追踪结论来源都是必要的。正确方式应是：

- 正文只突出当前主结果；
- 历史内容移动到明确附录；
- 错误结论保留替代记录；
- related work 不得从研究主文档删除；
- performance/结果表必须保留来源、环境、训练步数和状态标签。

本恢复版以 v7 全量内容为底稿，后续补丁只增加或标记替代，不继承 v8 的破坏性删除。

---

<!-- STAGE4B-SOURCE-BLOCK:B000101:END -->
