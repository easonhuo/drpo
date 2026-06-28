# Stage 4 语义上下文与无损拆分 Shadow 规范

**治理 claim：** `GOV-HANDOFF-INDEX-01`
**阶段：** Stage 4 设计规范；当前只允许设计、静态清点和 shadow candidate 验证。
**权威边界：** `docs/handoff.md` 仍是唯一研究 Master。本规范不是第二份研究 Master，不承载新的科学结论，也不得改变实验状态、门禁、变量或执行顺序。

## 1. 目标

Stage 4 的目标不是立即拆掉正式 handoff，而是在不切换权威的前提下，生成并验证一套可回滚的 shadow candidate：

1. 精简的当前 handoff candidate；
2. 按语义功能组织的研究模块 candidate；
3. 不可变、可索引的完整历史 candidate；
4. claim / experiment 节点图与结论谱系 candidate；
5. 面向目标任务的最小充分上下文 candidate。

Stage 4 只验证“如果未来切换，是否无损且足够”。Stage 5 才能执行正式 authority cutover。

## 2. 非目标

Stage 4 不做以下事情：

- 不直接替换 `docs/handoff.md`；
- 不修改启动协议使其默认读取 shadow candidate；
- 不让 compact candidate 成为研究权威；
- 不把旧结论视为无用内容并删除；
- 不按 Markdown 章节号机械切成大量碎片；
- 不让依赖图成为与 handoff 竞争的第二研究 Master；
- 不因节省 token 而省略不确定但可能必要的上游结论；
- 不在本阶段切换为 Delta-first authoritative mode；
- 不改变任何科研实验、registry、seeds、阈值、终态审计或方法职责。

## 3. 核心原则

### 3.1 语义模块，而不是机械章节切分

模块按照研究功能和高内聚关系组织。推荐的候选模块包括：

- 全局研究核心与治理边界；
- 连续远场动力学 E1-E4；
- 离散动力学 E5-E6；
- Hopper 外部验证 E7；
- Countdown 外部验证 E8；
- 方法族、理论比较与相关工作；
- 当前执行状态与门禁；
- 历史结论谱系与 provenance。

一个模块可以覆盖多个现有章节；一个章节也可以映射到多个语义节点。不得把“文件切得更小”本身当作成功标准。

### 3.2 研究对象建模为类型化节点

最小节点类型：

- `question`
- `hypothesis`
- `claim`
- `assumption`
- `experiment`
- `evidence`
- `method`
- `limitation`
- `alternative`
- `open_issue`

最小关系类型：

- `depends_on`：理解或执行当前节点所必需的逻辑上游；
- `tests`：实验直接检验某 claim；
- `supports`：提供支持证据但不等价于严格证明；
- `contradicts`：与现有 claim 冲突；
- `supersedes`：新结论取代旧结论的当前权威地位；
- `external_validates`：外部环境验证可迁移性；
- `does_not_replace`：明确外部实验不能替代受控识别；
- `motivates`：提供研究动机但不是逻辑前提。

不得把所有关系都压缩成模糊的“相关”。

### 3.3 依赖闭包，而不是按名字检索

目标任务的上下文必须包含：

1. 固定加载的项目总规则；
2. 目标节点；
3. 所有直接 `depends_on` 节点；
4. 上述节点的传递依赖闭包；
5. 与目标节点直接相关的限制、纠错谱系和 provenance；
6. 仍不确定但可能影响结论的上游节点。

例如研究 E7 时，不能只加载 Hopper 模块；必须加载它所依赖的 E1-E4 连续机制链、Gaussian 方差方向修正、collapse taxonomy，以及 E7 自身协议与历史 probe。

### 3.4 研究充分性优先于 token 节省

依赖图不确定时采用保守策略：

- 宁可多加载，不可少加载；
- 不允许仅因内容较旧、近期未讨论或位于 archive 就排除；
- AI 初次提取的依赖图必须可人工审计、可纠正；
- 依赖边随未来 Delta 持续维护，不能视为一次生成后永久正确；
- 上下文装配器必须报告所加载节点、边和遗漏警告。

### 3.5 旧结论保留可发现性

“被替代”不等于“永久无用”。重要历史结论必须保留谱系卡，至少包含：

```yaml
claim_id: EXAMPLE-CLAIM
current_statement: 当前权威结论
historical_variants:
  - summary: 旧结论的高度凝练摘要
    status: superseded
    superseded_by: EXAMPLE-CLAIM
    reason: 被替代的原因和证据
    keywords: [关键词, 同义词, 旧术语]
    reopen_conditions: [在何种新条件下值得重新调查]
    archive_pointer: docs/handoff_history/archive/...
```

任何结论只要没有明确 `superseded_by`，就不能仅因时间久或近期未使用而自动归档为过期。

## 4. 候选产物

Stage 4 的 shadow candidate 建议统一放在一个明确的非权威根目录下，例如：

```text
docs/handoff_shadow/stage4/
  CURRENT_CANDIDATE.md
  modules/
  history/INDEX.yaml
  history/archive/
  graph/NODES.yaml
  graph/EDGES.yaml
  graph/CLAIM_LINEAGE.yaml
  context_packs/
  reports/
```

具体路径可在 Stage 4A 实现前冻结，但必须满足：

- 与正式 `docs/handoff.md` 明确隔离；
- 文件名显式标记 candidate/shadow；
- 默认启动协议不得读取这些文件；
- 任一失败可整目录删除并回退到现有 handoff；
- 不能把同一科学事实在多个候选文件中变成互相竞争的权威副本。

## 5. 分阶段实施

### Stage 4A：结构登记与静态清点

只建立规范、schema、静态 inventory 和 validator，不拆正式 handoff。

必须完成：

- 语义模块清单；
- 节点和边 schema；
- claim lineage schema；
- heading / claim / experiment inventory；
- 依赖闭包规则；
- 模糊分类 fail-closed 规则；
- 静态 validator 和单元测试。

验收问题：**原 handoff 中有哪些对象，我们准备怎样组织？**

### Stage 4B：无损拆分 candidate

生成非权威候选：current、modules、history、index、lineage 和 graph。

机械门禁至少包括：

- 原 handoff 完整字节进入 current 或 archive 的可验证映射；
- 每个旧顶级标题都有 current/archive 去向；
- 当前锁定 claim、实验状态和门禁均存在；
- 每条被替代结论都有摘要、替代关系和 archive pointer；
- 相同输入产生相同输出；
- 索引不存在重复、悬空引用或未解析冲突；
- 分类无法确定时失败，不得自动猜测。

验收问题：**能否安全地试拆，而不丢历史和当前事实？**

### Stage 4C：上下文装配与真实 shadow 验证

针对真实任务生成上下文包，并与完整 handoff 下的判断对照。

必须覆盖：

- 至少一个连续机制任务；
- 至少一个外部验证任务；
- 一个包含 superseded claim 的任务；
- 一个多模块传递依赖闭包；
- 一个故意漏边的 mutation case；
- 一个依赖不确定时过度包含的 case。

每个 context pack 必须记录：

- target node；
- loaded node IDs；
- traversed edges；
- global core version；
- source hashes；
- unresolved dependency warnings；
- 与 full-context 对照的差异报告。

验收问题：**拆开以后，研究质量和治理判断是否保持？**

## 6. Stage 4 与 Stage 5 的边界

Stage 4：

- 允许生成拆分 candidate；
- 允许生成依赖图和 context pack；
- 只在 shadow 路径比较；
- 正式 handoff 仍由人工直接维护；
- 禁止 authority cutover。

Stage 5：

- 只能消费已经通过 Stage 4A/B/C 验证的结构；
- 不应在切换时重新发明 schema 或模块边界；
- 经独立用户批准后，才可切换 compact handoff、history index 和 Delta-first 生成路径；
- 必须保留一键回滚到原 handoff 的路径。

简化表述：

> Stage 4 是拆分演习与质量验证；Stage 5 才是正式搬家和权威切换。

## 7. 验收与风险控制

Stage 4 开发必须拆分为独立可验收更新，不允许一次性实现并切换：

1. 4A schema / inventory；
2. 4B candidate renderer / lossless audit；
3. 4C context packer / real shadow validation；
4. 单独的 Stage 5 cutover proposal。

主要风险与控制：

- **漏依赖：** 传递闭包、mutation 测试、保守过度包含；
- **旧结论遗忘：** lineage 摘要、关键词、reopen conditions；
- **过度碎片化：** 语义模块高内聚约束；
- **第二权威：** candidate-only 路径和启动协议隔离；
- **自动误分类：** 不确定即 fail closed；
- **切换过早：** Stage 5 独立授权和回滚计划；
- **治理复杂度膨胀：** 只保留必要节点/关系类型，不建设通用知识库平台。

## 8. 初始设计登记时的授权边界（历史）

本规范的登记只授权：

- Stage 3 进入 feature-frozen / bugfix-only 维护状态，同时继续真实 shadow observation；
- Stage 4 的设计、静态清点和实现前协议冻结；
- 后续按 4A、4B、4C 分步提出独立更新。

本规范不授权：

- 实际启用 Stage 4 candidate 作为默认上下文；
- 修改 `docs/handoff.md` 的权威地位；
- 启动 Stage 5；
- 自动编辑或提交正式 handoff；
- 修改任何科学实验状态或执行顺序。

## 9. 当前 Stage 4A 并行实现授权

在 `GOV-STAGE4A-PARALLEL-IMPLEMENTATION-2026-06-28` 下，Stage 4 的前置关系更新为：Stage 3 已实现并进入 `feature_frozen_bugfix_only` 即足以授权 Stage 4A shadow implementation；Stage 3 继续独立执行真实 observation 和 20-update／7-day Full Acceptance。

当前允许：

- 冻结 Stage 4A schema；
- 建立 heading / claim / experiment 静态 inventory；
- 实现只读、确定性的 Stage 4A validator；
- 增加 fail-closed mutation tests；
- 在明确的 shadow 路径保存 Stage 4A 静态产物。

当前仍禁止：

- 生成或启用 Stage 4B 的拆分 candidate；
- 实现或启用 Stage 4C context packer；
- 修改默认启动上下文；
- 替换、覆盖或自动提交 `docs/handoff.md`；
- 任何 Stage 5 或 Delta-first authority cutover。

Stage 4B 只有在独立 Stage 4A acceptance 更新通过后才能授权。Stage 4C 只有在独立 Stage 4B acceptance 更新通过后才能授权。Stage 5 必须同时等待 Stage 3 shadow validation 和 Stage 4 lossless validation。

## 10. Stage 4A 冻结实现布局（2026-06-28）

`GOV-STAGE4A-PARALLEL-IMPLEMENTATION-2026-06-28` 下的 Stage 4A 实现路径冻结为：

```text
docs/handoff_shadow/stage4/
  README.md
  schema/STAGE4A_SCHEMA.yaml
  inventory/MODULES.yaml
  inventory/HEADINGS.yaml
  inventory/CLAIMS.yaml
  inventory/EXPERIMENTS.yaml
scripts/validate_stage4a_inventory.py
tests/test_stage4a_inventory.py
```

该布局只登记 schema 与静态 inventory，不生成 Stage 4B 的
`CURRENT_CANDIDATE.md`、`modules/`、`history/` 或 graph 节点/边 candidate，
也不生成 Stage 4C `context_packs/`。出现上述后续阶段产物时，Stage 4A
validator 必须 fail closed。

冻结的最小规则如下：

- node types 与 relation types 严格采用本规范第 3.2 节的集合；未知类型拒绝；
- heading inventory 对 `docs/handoff.md` 的全部 Markdown headings 做有序、逐项、
  source-hash 绑定的静态登记；重复标题使用 occurrence 和独立 heading ID 区分；
- claim inventory 是非权威 locator/lineage 清单，必须绑定原文 heading、精确 anchor
  与 SHA-256；被替代 claim 必须登记 reciprocal `supersedes/superseded_by`、archive
  pointer 和 reopen conditions；
- experiment inventory 必须与 `experiments/registry.yaml` 的正式 experiment 列表
  一一对应，并引用已登记 claim 与 handoff heading；
- 单模块和已人工消歧的多模块分类均可接受；多模块分类必须给出 rationale；任何
  unresolved/automatic guess、重复 ID、悬空引用、lineage cycle 或 source drift 均拒绝；
- 依赖闭包只冻结规则，不在 Stage 4A 实现 context packer：未来只沿 `depends_on`
  做传递闭包，并附带直接 limitation、lineage 和 provenance；不确定依赖必须保守
  过度包含并报警，缺失依赖直接失败。

Stage 4A 代码和 inventory 完成不等于 Stage 4A 已验收，也不会自动解锁 Stage 4B。
Stage 4B 仍须通过独立 acceptance 更新获得授权；`docs/handoff.md` 继续是唯一权威
研究 Master。
