# DRPO 编码前上下文闭合与修改面收敛流程（试行版）

- Governance claim：`GOV-DEV-CHANGE-SURFACE-CLOSURE-PILOT-01`
- 状态：`documented_pilot_not_yet_executed`
- 首次试运行任务：`EXT-H-E7-SQEXP-GAE-01`
- 文档基线：`main@dd355642df2ba2479715c9620749c1e9f75f76ba`
- 科学状态影响：无
- 默认开发流程影响：无；试运行、复盘并经用户另行批准前，不升级为仓库默认规则

## 1. 文档职责与权威边界

本文件只定义一次**人工试运行流程**，目标是在写代码前闭合相关代码上下文和修改面，防止局部功能逐步膨胀成平行子系统。

本文件不是第二份研究主文档：

- `docs/handoff.md` 仍是唯一研究 Master；
- `experiments/registry.yaml` 仍是科学实验登记表；
- 不修改任何科学 claim、实验状态、数据、seeds、阈值、训练预算、收敛标准或执行顺序；
- 不重新开启既有 governance pipeline 的关闭阶段；
- 不授权新增 validator、runner、framework、registry 或自动化服务；
- 第一次仅用于 `EXT-H-E7-SQEXP-GAE-01` 的受控人工试运行。

本文中的阶段编号为 `CS0`–`CS7`，与 `docs/governance_pipeline_stage_status.yaml` 中的治理 `stage_0`–`stage_5` 无关。后续对话中的“Stage 0”默认指 `CS0`。

## 2. 要解决的问题

问题不只是“代码行数太多”，而是**仓库上下文未闭合就开始实现**：

```text
只读到局部代码
-> 立即开始改
-> 写到一半发现新依赖
-> 不清楚现有 owner
-> 新建 wrapper / adapter / runner
-> 新组件继续产生 config / test / audit / lifecycle
-> 局部算法修改变成新子系统
```

行数门禁只能拦截晚期症状。本流程把门禁前移：

> 在相关代码子图、现有职责 owner 和预计修改面闭合前，不得写入生产代码。

## 3. 核心原则

### 3.1 读取相关代码子图，而不是通读十万行仓库

每个任务必须先闭合以下任务相关链路：

```text
entrypoint
-> config / branch expansion
-> bootstrap / runtime injection
-> trainer / data flow
-> algorithm owner
-> direct helpers
-> artifact provenance
-> tests / executable command contract
```

边界由职责和调用关系确定，不由任意文件数量确定。

### 3.2 现有 owner 默认拥有职责权威

遇到新问题时默认顺序为：

```text
定位现有 owner -> 完整阅读 -> 最小扩展
```

而不是：

```text
不熟悉旧代码 -> 新建局部 wrapper -> 重建外层生命周期
```

“没有看懂现有 owner”不能作为“仓库里没有 owner”的证据。

### 3.3 方案必须提前暴露真实代码量

“增加 GAE 支持”之类自然语言方案不足以批准实施。编码前必须给出绑定 base commit 的完整 prospective unified diff，但不得应用到实现分支。

### 3.4 任何扩张都必须重新审批

新增文件、职责、抽象层、修改文件或行数预算，必须回退到前一阶段重新审计，不能在实现过程中顺手吸收。

### 3.5 流程本身必须保持轻量

第一次只进行人工试运行，不开发新的流程引擎。`CS0`–`CS3` 原则上只保存在一个 durable plan record 中，不为每个阶段创建单独文件。

## 4. 基本定义

### 4.1 职责角色

- **Entry owner**：任务实际执行的命令、模块或函数。
- **Data-flow owner**：加载、对齐、采样或传递数据的现有组件。
- **Algorithm owner**：真正计算目标算法行为的函数或类。
- **Lifecycle owner**：现有 runner、resume、aggregation、artifact 或 delivery 组件。
- **Artifact-provenance owner**：绑定 prepared inputs、checkpoint 或缓存数组身份的 manifest/组件。
- **Test owner**：已保护相关接口或行为的现有测试文件。
- **Historical owner**：职责被冻结、不能被 successor 悄悄改写的历史文件。

### 4.2 阅读等级

- `FULL_READ`：完整阅读当前文件。所有候选生产修改文件必须达到此级别。
- `TARGETED_READ`：完整阅读相关函数/类、imports、直接调用上下文和接口约束。
- `SEARCH_ONLY`：只确认文件或符号存在；不能据此修改文件或宣称已理解其职责。

### 4.3 修改面术语

- **Candidate change surface**：写逐行方案前可能需要修改的文件和符号。
- **Approved change surface**：`CS1` 通过后冻结的文件白名单。
- **Prospective patch**：基于锁定 base 生成、尚未应用的完整 diff。
- **Plan delta**：只有真实测试/liveness 暴露遗漏后才能提出的增量 prospective diff。
- **Responsibility expansion**：新增未获批准的 trainer、runner、loader、sampler、resume、aggregation、artifact、registry 或通用 framework 职责。

## 5. 角色分工与持久记录

### 5.1 Reviewer / gatekeeper

负责 `CS0`–`CS3` 和 `CS7`：

- 锁定 base、任务身份和评价标准；
- 审核代码子图和职责 owner；
- 审批文件、职责和预算；
- 审批 prospective patch 与 plan delta；
- 不把静态测试或 liveness 解释为科学结果。

### 5.2 Implementation agent

- `CS0`–`CS2` 可以只读检查，但不得修改生产代码；
- `CS4` 只能应用已批准 patch；
- 测试失败时先报告，不得自行扩大修复范围；
- 不得自批新增文件、职责或预算。

当前 DRPO 分工中，GLM/其他 coding agent 可负责实现，ChatGPT 负责设计、review 和 merge gate，除非用户另行调整。

### 5.3 单一 durable plan record

`CS0`–`CS3` 只保存在一个持久记录中，可选：

- 一个 GitHub issue / Draft PR body；或
- 一个 reviewer-authored plan 文档。

实施代码的“禁止新增文件”预算不计算这一份 reviewer plan，但 plan 不能在 prospective patch 之外夹带可执行生产代码。

## 6. `CS0`–`CS7` 执行流程

每一阶段只能返回 `PASS`、`REVISE`、`BLOCKED` 或 `REJECTED`。前一阶段未 `PASS`，不得进入后一阶段。

### CS0 — Repository Context Closure

**目标**：在提出修改前，证明相关 current-main 代码如何真实运行。

**只允许**：read、search、compare、追踪 imports/callers、检查测试和历史候选。

**禁止**：改文件、建实现分支、写生产代码、建替代模块、提前写精确 patch。

**必须产出**：

1. repo、branch、完整 base SHA；
2. experiment ID / engineering claim 与当前状态；
3. 剩余不确定性；
4. 任务调用和数据流图；
5. 相关文件清单与阅读等级；
6. entry/data/algorithm/lifecycle/artifact/test/history owner 地图；
7. 可复用组件；
8. 精确功能缺口；
9. 明确排除的职责；
10. 未闭合的上下文边。

**通过条件**：核心路径的每一条直接边都结束于：

- 已审计的相关 owner；或
- 本任务不修改的稳定边界。

generated command → parser、runtime monkey-patch/dynamic import、prepared-artifact identity 都属于可能影响真实执行的直接边。候选生产文件不能停留在 `SEARCH_ONLY`。

发现未审计的直接 caller、dependency、runtime patch 或历史 owner，立即重开 `CS0`。

### CS1 — Change-Surface Closure

**目标**：把 `CS0` 事实收敛为最小修改面。

**必须产出**：

1. 生产文件白名单；
2. test/config/doc 白名单；
3. 每个文件要修改的 symbol/function；
4. 每项修改对应的 `CS0` 缺口；
5. 必须保持不变的文件；
6. 允许和禁止的职责；
7. 新文件数量，默认 `0`；
8. production/test/config/total 行数与 churn 预算；
9. 若不是表面最小方案，说明更小方案及其失败原因；
10. 静态测试和真实 liveness 目标。

**新文件例外**只有在以下条件全部满足后才能申请：

- 现有 owner 无法承载而不破坏冻结历史边界；
- 最近的现有 owner 和最小扩展方案已完整阅读；
- 新文件只承担一个有界职责，不复制生命周期；
- 用户在 `CS2` 前明确批准。

每一项修改必须追溯到一个 `CS0` 缺口；禁止用未归属的通用 framework 解决局部问题。

### CS2 — Line-Level Prospective Patch

**目标**：在仓库实际修改前暴露完整实现。

**必须产出**：

1. 绑定 base 的完整 unified diff；
2. 精确文件路径和 hunks；
3. 每个 hunk 的目的；
4. 新增/删除的 imports、signatures、branches、checks 和 test assertions；
5. 分类后的 additions、deletions、total churn；
6. 未修改路径和兼容性说明；
7. 测试与 liveness 命令；
8. 明确声明 patch 尚未应用；
9. 环境允许时，在 disposable checkout 上执行 `git apply --check`，不得创建实现 commit/branch。

patch 必须代码完整，不能包含伪代码、省略号、占位符或“实现时再决定”。

超出 `CS1` 预算、出现未批准文件或职责，自动停止并返回 `REVISE`。

### CS3 — Pre-Implementation Review Gate

按以下顺序 review：

1. base 与任务身份；
2. `CS0` 上下文闭合；
3. 文件/职责白名单；
4. 新文件与抽象层；
5. 行数与 churn；
6. 复用和复制；
7. 测试覆盖；
8. liveness 是否真正经过修改行；
9. 科学变量和治理不变量。

以下任一项直接拒绝：

- 候选生产文件未 `FULL_READ`；
- 未批准的新生产文件；
- 修改白名单外文件；
- 算法任务新增生命周期职责；
- 复制 trainer/runner/loader/sampler/resume/aggregation/artifact；
- 为单一功能新增 generic factory/registry/plugin/callback/adapter framework/base class；
- 拆 commit/PR 规避累计预算；
- 弱化或重写旧 regression test 隐藏行为删除；
- 无法说明真实 liveness 如何到达修改代码。

决策只能是：

- `APPROVED_FOR_APPLICATION`；
- `NEEDS_PLAN_REVISION`；
- `REOPEN_CS0`；
- `REJECTED_OVERSCOPE`。

只有第一种允许进入 `CS4`。

### CS4 — Apply Approved Patch

- 批准后才从锁定 base 创建 dev branch；
- 只应用已批准 patch；
- 禁止顺手 refactor、全局格式化、cleanup、文档扩张或无关修复；
- 遇到冲突或意外接口时停止，不得现场重新设计；
- 记录 commit 和实际 diff。

实际需要与批准 patch 不一致时，局部遗漏回 `CS2`；架构理解错误回 `CS0`。

### CS5 — Static and Focused Validation

运行最小但完整的适用集合：

- compile / syntax；
- Ruff 或仓库标准 lint；
- affected owner 的现有 focused tests；
- `CS2` 批准的新回归测试；
- 涉及执行 plumbing 时的 command-contract/parser-path 测试。

失败先分类：

1. 实际代码没有按批准 patch 应用；
2. 已批准职责内遗漏局部分支；
3. `CS0`/`CS1` 架构或修改面不完整；
4. 环境/依赖 blocker。

只有第 2 类可以走小型 plan delta；第 3 类必须重开 `CS0`。

### CS6 — Real Liveness and Plan Delta

liveness 合同必须在 `CS1` 冻结。首次 GAE 试运行至少包括：

- 一次真实 A2C + one-step TD actor update；
- 一次真实 PPO-K4 + GAE actor update；
- 验证 intended prepared advantage 被消费；
- 协议要求 frozen critic 时验证 critic 未改变。

真实失败提出 `PLAN_DELTA_N` 时必须记录：

1. exact command 和 commit；
2. error、traceback/log；
3. root cause；
4. 原方案为什么遗漏；
5. 完整增量 prospective diff；
6. 增量和累计预算；
7. 是否改变文件或职责边界。

plan delta 一旦需要新文件、新职责、第三个生产文件或架构变化，就不是小修，必须重开 `CS0`/`CS1`。

### CS7 — Plan–Actual and Delivery Audit

比较：

```text
approved prospective patch
+ approved plan deltas
vs.
base-to-head actual diff
```

必须检查：

- 实际文件集合；
- 未批准新文件/职责；
- production/test/config 行数与 churn；
- generated/vendor/test/config/shell 中是否藏有未批准改动；
- 测试和 liveness 是否绑定 reviewed head；
- 未执行测试和剩余失败是否明确列出；
- 没有独立证据和治理批准时，科学状态保持不变。

结论只能是：

- `MERGE_REVIEW_ELIGIBLE`；
- `REQUEST_CHANGES`；
- `REJECT_IMPLEMENTATION`；
- `BLOCKED_BY_UNEXECUTED_LIVENESS`。

该结论不等于 merge 授权；仍需用户明确批准。

## 7. 回退矩阵

| 新发现 | 必须动作 |
|---|---|
| 未审计 caller/dependency 或 owner 判断错误 | 重开 `CS0` |
| 同一职责内需要额外现有文件 | 重开 `CS1` |
| prospective diff 超预算 | 修改 `CS1` 或拒绝 |
| 需要新生产文件或生命周期职责 | 重开 `CS0`/`CS1` 并显式审批 |
| static test 暴露已批准 owner 内的局部分支遗漏 | 提交小型 `CS2` plan delta |
| liveness 暴露架构/接口理解错误 | 重开 `CS0` |
| actual diff 与批准 patch 不一致 | 立即停止，禁止事后补写方案 |
| application/final review 前 `main` 前进 | 重新验证上下文并重生成 prospective patch |

## 8. 防绕过规则

所有预算按锁定 base 到最终 reviewed head 累计计算。以下行为不能降低 scope：

- 拆 commit 或 PR；
- 把生产逻辑藏进 tests、config、shell、generated 或 docs；
- 先删除再重写以降低净新增；
- 把多个职责塞进一个旧文件以规避新文件禁令；
- 复制现有实现后改名；
- 把结构扩张称为 temporary / pilot-only；
- 弱化旧测试；
- 把 `SEARCH_ONLY` 声称为完整 review；
- minify、长行压缩或合并无关逻辑规避行数统计。

必须至少报告：

- production additions/deletions；
- test additions/deletions；
- config/shell/doc additions/deletions；
- total churn；
- modified/new file 数量。

## 9. 首次 GAE 试运行约束

### 9.1 唯一允许新增的职责

> 在现有 canonical E7 A2C/PPO actor path 中消费已验证的 one-step TD 或 trajectory GAE advantage，并保持登记的 frozen-critic 行为。

禁止新增 trainer、loader、sampler、runner、resume、aggregation、artifact、registry、generic adapter 或 experiment-lifecycle framework。

### 9.2 初始预算

以下只用于本次 pilot，不是全仓默认值：

- 新实现文件：`0`；
- 现有生产文件：目标 `<=2`，硬停止 `2`，除非 `CS1` 证明并获批例外；
- 现有测试文件：目标 `<=2`；
- production additions：目标 `<=100`，硬停止 `150`；
- all additions：硬停止 `300`；
- total churn：硬停止 `450`；
- 单个小型 plan delta：目标 `<=30` production lines；
- 连续两次修复 delta 仍失败：重开 `CS0`，禁止继续打补丁。

所有统计均为 base-to-head 累计。

### 9.3 历史候选的使用方式

未合并 GAE PR 只能作为证据和反例，不能作为架构权威。必须先根据 current `main` 独立确定 owner 和最小修改面，再检查候选中是否有可复用代码。

算法局部任务若新增独立 protocol、matrix、runner、artifact、aggregation、terminal-audit owner，应在 `CS1`/`CS3` 拒绝，除非逐项证明仓库中确实没有现有 owner。

## 10. 文档五轮 review 记录

| Review | 结果 | 本轮实质修正 |
|---|---|---|
| Architecture | PASS | 强制闭合 runtime command/parser、dynamic patch 和 artifact provenance；所有候选生产文件必须 `FULL_READ`。 |
| Operational | PASS | 为每阶段补齐产物、退出和回退条件；增加单一 durable plan record 与 disposable `git apply --check`。 |
| Adversarial anti-bloat | PASS | 增加 base-to-head 累计统计、禁止拆 PR、禁止藏代码/弱化测试/minify 绕过。 |
| DRPO governance compatibility | PASS | 将流程编号改为 `CS0`–`CS7`，避免与治理 stage 混淆；保持 handoff/registry/science 状态不变。 |
| Cost and proportionality | PASS | 第一次只做人工 pilot，不开发自动化；阶段材料限制为一个 plan record，并记录流程成本。 |

当前没有未解决的高严重度设计问题。剩余不确定性只能通过实际 GAE 试运行回答：该流程能否显著减少修改面和返工，同时不过度增加前期时间。

## 11. 试运行成功标准

1. `CS0` 在编码前闭合 current-main 相关代码子图；
2. `CS1` 冻结有界文件和职责白名单；
3. `CS2` 在应用前展示完整 diff；
4. `CS3` 能拒绝未批准 owner、文件或预算扩张；
5. 实际实现不创建独立 trainer/runner/lifecycle stack；
6. actual diff 等于 approved patch + approved deltas；
7. A2C/TD 与 PPO-K4/GAE 真实 liveness 到达修改路径；
8. static test 或 one-step liveness 不被宣称为科学结果；
9. 记录开发时间和返工；
10. 拦截价值高于新增流程成本。

建议记录：

```text
cs0_elapsed
cs1_elapsed
cs2_elapsed
first_liveness_elapsed
files_initially_proposed
files_finally_modified
unplanned_files
prospective_production_lines
actual_production_lines
plan_delta_count
cs0_reopen_count
real_liveness_failures
prevented_new_file_requests
```

## 12. 非目标

本流程不：

- 保证第一次 prospective patch 必然正确；
- 要求通读整个仓库；
- 永久禁止所有新文件；
- 替代 tests、real liveness、CI、terminal audit 或科学 review；
- 授权开始实现或运行 `EXT-H-E7-SQEXP-GAE-01`；
- 关闭、合并、替换或修改现有 GAE PR；
- 创建新的治理 framework/automation service；
- 建立全仓通用行数上限；
- 把 smoke test 当成实验结果。

## 13. 当前状态与下一步

```text
workflow_document: five_pass_review_complete
pilot_default_route: not_activated
scientific_code_changed: false
experiment_started: false
formal_cs0_restart: pending_user_acceptance_of_this_document
```

本文档获确认后，从届时最新 `main` 重新开始正式 `CS0`。此前探索性 `CS0` 结论只能作为待复核的 source list 和 hypothesis，不能直接继承为正式 `CS0` 结果。
