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

## 11. Stage 4A.1 动态语义图扩展（2026-06-29）

在 `GOV-STAGE4A-DYNAMIC-SEMANTIC-GRAPH-2026-06-29` 下，Stage 4A 的静态
inventory 保留为 bootstrap snapshot，但不再被视为最终的持续维护方式。本扩展仍属于
Stage 4A shadow implementation，不授权 Stage 4B handoff candidate、Stage 4C context
packer 或任何 authority cutover。

### 11.1 四层分离

Stage 4 动态层必须分为：

1. **Research Semantic Kernel**：跨项目复用的 node、edge、lifecycle、review 和 fail-closed 规则；
2. **Versioned Project Semantic Profile**：由项目内容逐步形成并可版本化演化的模块、术语和推断规则；
3. **Human-approved Overrides**：只记录真正有歧义的语义决定、模块 split/merge/supersedes 和人工接受或拒绝；
4. **Generated Graph and Views**：由相同 canonical graph 数据自动生成的 nodes、edges、review queue、Mermaid 和 DOT 视图。

引擎不得硬编码 `E1--E18`、Hopper、Countdown 或任一 DRPO 模块。新项目允许从一个问题、
一个假设和一个实验开始；后续实验通过 registry 自动发现，模块通过 profile 逐步增长，而不是
预分配未来业务结构。

### 11.2 动态更新原则

- 机械发现全自动：Markdown headings、registry experiments、source hashes、stable IDs、重复和悬空引用；
- 语义建议自动：模块归属、claim relation、模块拆分或合并候选；
- 歧义决定人工批准：未确认建议只能进入 review queue，不得进入 accepted dependency closure；
- 节点、边和模块均有 lifecycle；rename/split/merge/supersede 不得破坏旧 ID 和历史 provenance；
- builder 必须离线、确定性、无网络和无在线 LLM 调用。AI 产生的建议通过可审计 proposal/override 输入进入；
- Stage 3 Delta 是未来首选增量触发源；全量确定性 rebuild 是回归和恢复路径。

### 11.3 Graph 与可视化

逻辑 graph 的 canonical 数据与显示格式必须分离。仓库只人工维护 kernel、project profile 和
少量 overrides；`NODES.yaml`、`EDGES.yaml`、review queue、Mermaid Markdown 与 Graphviz DOT
全部是生成产物，不得手工编辑。

同一 graph 至少生成：

- 模块概览和依赖图；
- claim--experiment 关系图；
- supersedes 结论演化图；
- 可机器读取的完整 DOT 图；
- source/profile/kernel/graph hash manifest。

任一 accepted edge 或 module lifecycle 变化都必须改变 graph hash 并同步改变相应视图。graph
hash 已变化但视图未更新时，validator 必须失败。大图只作完整机器视图；人类阅读默认使用过滤后的
局部视图，避免形成不可读的单一“蜘蛛网”。

### 11.4 验收门禁

Stage 4A.1 至少验证：

- 相同输入两次生成 byte-for-byte 一致；
- 新增实验时不修改 Python，引擎能自动发现；
- 新项目只有 Q1/H1/E1 时也能构图，不依赖 DRPO 预设实验编号；
- 未知或歧义语义进入 review queue，增加 override 后可确定性消解；
- module split/merge/supersede 保留旧 module identity；
- 修改 accepted edge 后 graph hash 和可视化同步变化；
- duplicate ID、dangling edge、unknown type、supersedes cycle、stale source、手工篡改视图均 fail closed；
- 原 Stage 4A inventory validator、治理 stage validator 和现有测试继续通过。

Stage 4A.1 的完成仍不等于 Stage 4A acceptance；Stage 4B 继续由独立 acceptance 更新解锁。

## 12. Stage 4A.2 动态治理闭环加固（2026-06-29）

`GOV-STAGE4A-DYNAMIC-GRAPH-HARDENING-2026-06-29` 只修补 Stage 4A.1
动态语义图的已识别治理缺口，不授权 Stage 4B、Stage 4C 或 authority cutover。
跨项目复用与 Stage 3 Delta adapter 继续作为后续目标，不作为本轮验收门禁。

### 12.1 已拒绝语义候选必须持久化

`rejected_candidates` 是人工否决记录，而不是临时过滤器。每条记录必须绑定由
`kind + object_id + reason + candidates` 确定性计算的 `review_id`，并保存
`rationale` 与 `decision_version`。当同一候选再次出现时，builder 必须：

- 不再把它放入 pending review queue；
- 在生成的 `REVIEW_QUEUE.yaml` 中保留 `state: rejected` 的可审计决定；
- 标记它是再次匹配当前候选，还是仅作为历史决定保留；
- 将 rejected decision 纳入 canonical graph hash，防止静默丢失。

候选语义签名发生真实变化时应产生新的 review ID；旧拒绝不能模糊匹配或自动扩张
到不同的研究语义。

### 12.2 模块 lifecycle 通过小型 override 演化

模块不是一次性固定分类，也不得由每次运行重新聚类。经人工批准后，
`module_lifecycle_changes` 可执行：

- `rename`：稳定 module ID 不变，只改变显示名称；
- `supersede`：旧模块保留并标记为 superseded；
- `split`：一个旧模块被多个新模块替代；
- `merge`：多个旧模块被一个新模块替代。

所有操作必须显式给出 source/target module IDs、变更前后版本和 rationale。
旧 module ID 不得删除；split/merge/supersede 必须生成 reciprocal lineage，
并在节点属性和 `supersedes` edges 中同时可审计。可视化和 graph hash 必须随之更新。

### 12.3 Profile、override 与 module 版本必须强制执行

版本号不是说明性字段。builder 必须使用前一版 `GRAPH_MANIFEST.json` 的语义
fingerprint 强制以下规则：

- project profile 语义改变时，`profile_version` 必须递增；
- accepted/rejected override 或 lifecycle 决策改变时，`override_version` 必须递增；
- module 的名称、用途、依赖或 lifecycle 语义改变时，该 module version 必须递增；
- 已存在 module 不得从 profile 中破坏性移除；只能保留并 supersede；
- fingerprint 算法自身必须版本化；旧算法 manifest 只允许一次明确迁移，之后严格校验。

版本门禁既约束正常构建，也必须有 mutation tests 覆盖“不升版本修改语义”、
“无 lineage 删除模块”和“split/merge 未递增所有 touched module version”。

### 12.4 本轮验证边界

本轮必须证明 DRPO 项目内的动态治理闭环，而不宣称已经实现完整跨项目产品化：

- rejected candidate 不重复进入 review；
- rename/split/merge 保留稳定 ID 和历史 lineage；
- profile、override、module 版本门禁 fail closed；
- canonical graph、manifest 与所有可视化保持同步；
- 原 Stage 4A bootstrap inventory 与 Stage 4A.1 deterministic graph 测试不回归。

Stage 3 Delta 的直接消费、AI proposal adapter、跨项目真实仓库验证和长期 shadow
precision/recall 仍需后续独立实现与验收。Stage 4B 继续被 Stage 4A 独立 acceptance
阻塞。

## 13. Stage 4A Minimal Context Core（2026-06-29）

`GOV-STAGE4A-MINIMAL-CONTEXT-CORE-2026-06-29` 将 Stage 4 的当前主开发路径
收缩为一个可确定性验证的上下文编译闭环。该更新不删除既有 Stage 4A inventory 或
动态语义图；后者继续保留为历史 shadow implementation，但不作为本轮 Context Builder
的运行时依赖。

### 13.1 唯一目标与冻结边界

本轮只回答：如何把持续增长的 `docs/handoff.md` 映射为稳定研究模块，并在处理目标
任务时只加载该模块及其传递依赖。

冻结边界如下：

- `docs/handoff.md` 与 `experiments/registry.yaml` 继续是权威输入；
- 不物理拆分 handoff，不反向自动写回 handoff；
- 模块边界按独立研究职责显式登记，不按每次运行重新聚类；
- 核心关系只有 `depends_on`；
- 不实现 AI proposal adapter、Stage 3 Delta adapter、跨项目产品化或 authority cutover；
- 不实现模块 lifecycle 状态机、多层人工版本号或生成 manifest 充当历史账本；
- Stage 4B、Stage 4C 与 Stage 5 继续保持原门禁。

### 13.2 内容自动更新，结构只建议不自动执行

已有模块的 mapped source 变化时，builder 自动更新该模块快照。增加、删除、拆分、
合并、改变职责或重接依赖都属于结构变化：工具只能生成确定性建议报告，不能修改
`MODULES.yaml` 或 `DEPENDENCIES.yaml`。结构变化必须经用户或研究负责人明确批准后，
通过普通 Git 更新落库。

### 13.3 简单 dirty-module rebuild

每个模块的 source 内容、模块标题与职责共同形成 `source_hash`：

- source hash 不变且生成快照逐字节一致时，不重写该模块；
- source hash 变化或快照缺失时，只刷新该模块；
- 小型依赖图、闭包检查、可视化、索引和结构建议每次从当前配置全量重算；
- 旧索引丢失只导致必要输出重新生成，不影响正确性；生成文件从不成为 authority。

该机制不是 Stage 3 Delta 驱动的复杂增量状态机，也不维护局部更新传播缓存。

### 13.4 Source mapping 与稳定性

Minimal Core 仅支持四种显式 source：

1. `markdown_range`：由两个必须唯一匹配的精确行界定；
2. `marker_block`：读取一个明确的 `HANDOFF-DELTA-BLOCK`；
3. `marker_blocks_matching`：按已登记 experiment ID 前缀自动吸收后续相关 delta block；
4. `registry_entries`：按明确 experiment ID 提取 registry 条目。

任何边界零匹配、多匹配、反向范围、未知 experiment ID、悬空依赖或依赖环都必须
fail closed。Builder 不使用“同名标题第几次出现”作为身份，也不自行改变模块粒度。

### 13.5 Context Pack 与验收

给定目标模块，Context Builder 按确定性拓扑顺序输出全部传递依赖，再输出目标模块。
Context Pack 是临时生成的非权威输入，不反向修改模块、handoff 或 registry。

本轮至少验证：

- E4-TAPER pack 包含治理、理论、终态、E1--E3 因果基础和 E4 模块，不包含 E7/E8；
- E7 pack 包含连续机制和外部验证边界，不包含 Countdown；
- E8 pack 包含 categorical E5/E6 基础，不包含 Hopper 与 E4-TAPER；
- 新模块只改 YAML 即可加入，不修改 Python；
- 单模块 source 变化只重写对应模块快照；
- 依赖变化同步改变闭包和两种可视化；
- 相同输入重复构建得到逐字节相同输出；
- 自动建议不修改正式模块或依赖配置；
- handoff 与 registry 在 build 前后逐字节不变。

通过这些工程验收只证明按依赖加载的 shadow 闭环可用，不证明初始模块粒度已经最优，
也不构成 Stage 4A 最终 acceptance 或 authority cutover。

### 13.6 Minimal Context Core 语义完整性加固（2026-06-29）

本轮加固修复一个高风险但不易从依赖图中发现的问题：原 builder 只验证“模块存在、
source 可读取、依赖闭包正确”，却没有验证模块内容是否真的覆盖其 `responsibility`。
因此 `terminal_audit` 虽然声明负责收敛、持续漂移、任务性能崩溃、support/variance
boundary 与 NaN/Inf 分报，实际 source 却只包含统一收敛窗口。只要目标模块依赖了
`terminal_audit`，结构验收仍会通过，缺失语义只能在后续实验误报时才被发现。

加固后的规则如下：

1. `MODULES.yaml` 通过顶层 `semantic_contract_required_modules` 锁定必须具备语义契约的
   高风险共享模块；这些模块的整个 `content_contract` 若被删除，builder 立即 fail closed，
   不能依赖测试人员事后发现。每个 required topic 必须给出稳定 `topic_id`、职责描述、
   可接受的权威文本锚点和允许提供证据的 source label 范围。
2. Builder 不再只做整模块关键词搜索，而是为每个 topic 生成确定性的
   `topic -> matched phrase -> authoritative source` 证据。证据写入模块正文、`source_hash` 和
   `MODULE_INDEX.json`；锚点出现在错误 source 中也不能蒙混通过。`terminal_audit` 显式映射
   统一实验验收表、收敛/持续漂移规则，以及任务崩溃、support/variance boundary、NaN/Inf
   三类事件分离的权威段落，并登记七项 required topics。
3. 所有 handoff source 记录物理行区间。若 broad `markdown_range` 已覆盖某个 marker block，
   只保留一份；完全包含的重复 source 会被确定性去重并记录在 module index，部分重叠则
   fail closed，避免静默剪裁或重复。
4. 未映射检测同时扫描 canonical `experiments` 与
   `development_experiment_registrations`。formal development registration 使用高优先级
   suggestion，pilot/development entry 使用普通 suggestion；两者都只建议，不自动改结构。
5. 经用户明确批准，原混合 E4 模块拆为 `continuous_e4_extrapolation` 与
   `continuous_e4_taper`。后者依赖前者；基础 E4 任务不再加载全部 taper follow-up，
   taper pack 仍保留完整机制基础。

上述加固只提高 Stage 4A shadow Context Builder 的可审计性和 fail-closed 能力。
`docs/handoff.md` 与 `experiments/registry.yaml` 继续是唯一权威输入，Stage 4B/4C、
Stage 5 和 authority cutover 仍保持阻塞。

### 13.7 Stage 4A final acceptance closure（2026-06-30）

Stage 4A 的静态 inventory、动态语义图与 Minimal Context Core 通过统一的
`governance_stage4a_acceptance_spec.md` 进行独立验收。验收入口必须同时执行正向完整性、
确定性/no-op、权威隔离、六个 context closure、三个高风险共享模块的 source-scoped
semantic contracts，以及故障注入矩阵。

验收证据写入 `docs/governance_stage4a_acceptance/`，并以 `AFTER_IMAGE.json` 冻结
Stage 4A 实现。验收通过后只允许 bugfix、compatibility 或 clarification 类修改；职责变化
或架构扩展需要显式 reopen/新授权。

通过 Stage 4A 只把 `stage_4b_lossless_candidate` 调整为 `ready_for_authorization`，不等于
Stage 4B 已启动。Stage 4C、Stage 5 与 authority cutover 继续保持阻塞，`docs/handoff.md`
和 `experiments/registry.yaml` 继续是权威输入。
