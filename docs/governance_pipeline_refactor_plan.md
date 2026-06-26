# DRPO 治理与多 Session 集成重构方案（工作设计稿）

**治理 claim：** `GOV-RULE-MIGRATION-01`、`GOV-FORMAL-ENTRYPOINT-01`、`GOV-HANDOFF-INDEX-01`、`GOV-UPDATE-BUNDLE-01`
**文档性质：** 工作设计稿，不是第二份研究 Master，不改变任何实验状态、冻结参数或科学结论。
**当前权威来源：** `AGENTS.md`、`docs/handoff.md`、`experiments/registry.yaml` 仍保持原有优先级。

## 1. 目标与非目标

本重构要解决四个问题：

1. 多个 session 基于不同 main commit 生成更新包时，即使代码不冲突也经常整体失效；
2. `docs/handoff.md` 是高频冲突热点；
3. 关键规则既不能只依赖模型记忆，也不能在多个训练脚本中重复实现；
4. 长期累积的治理内容可能挤占研究推理所需的有效上下文。

当前阶段不做以下事情：

- 不删除或迁移 `AGENTS.md` 中的现有规则；
- 不拆分现有 handoff；
- 不切换 handoff 的权威生成路径；
- 不改变正式实验、artifact、checkpoint 或大小限制；
- 不修改 C-U1、D-U1、Hopper、Countdown 的实验职责和冻结配置。

## 2. 统一打包通道：训练文件不自行实现 artifact 规则

新的训练文件只负责训练逻辑和约定输出，不自行执行 `zip -r`，也不复制 25 MiB / 10 MiB、checksum、symlink、sidecar 等逻辑。

统一执行路径应为：

```text
registered experiment
        ↓
scripts/run_experiment_guard_hardened.py
        ↓
experiment entrypoint（任意新的训练文件）
        ↓
scripts/package_experiment_hardened.py
        ↓
scripts/verify_experiment_package_hardened.py
        ↓
scripts/artifact_protocol_hardened.py（共享实现）
```

因此“新训练文件默认遵守规则”的含义不是把相同检查复制到每个文件，而是任何正式入口都必须经过唯一的 canonical channel。

## 3. AGENTS、协议文档与 guard 为什么要分层

三者不是三个等价 prompt。

### 3.1 `AGENTS.md`：常驻的路由器和不可绕过原则

`AGENTS.md` 会被 agent 启动时读取，职责是：

- 告知哪些规则是仓库级硬约束；
- 告知当前任务必须读取哪些权威文件；
- 明确禁止绕过统一 guard / package / verify 路径；
- 保留 25 MiB / 10 MiB 等跨任务默认硬限制；
- 明确验证失败时不得交付或宣称完成。

它不应成为每一种错误分支和底层实现算法的唯一载体。

### 3.2 专用协议：完整合同

`docs/formal_experiment_artifact_protocol.md` 记录完整的生命周期、sidecar、原子发布、失败恢复、路径安全和 provenance 语义。只有正式实验或 artifact 任务需要详细读取它。

### 3.3 guard / hardened core：可执行门禁

`run_experiment_guard_hardened.py` 和 `artifact_protocol_hardened.py` 是程序，不是依赖模型遵守的自然语言 prompt。即使 agent 忘记某条低层细节，代码也应 fail closed。

分层后的收益是：

- 根提示仍保留不可绕过的硬规则；
- 详细合同不丢失；
- 实际执行不依赖模型是否“记住”；
- 新训练文件只需接入统一通道，而不是复制检查代码。

## 4. 防绕过的四层结构

任何正式训练入口需要同时满足：

1. **Prompt 层：** `AGENTS.md` 明确禁止绕过 canonical guard；
2. **登记层：** registry 声明 experiment ID、entrypoint、supervisor、packager 和 artifact policy；
3. **代码层：** guard / package / verify 共用 hardened core，缺失时 fail closed；
4. **测试层：** governance validator 检查 formal entrypoint 的登记和统一通道引用。

单靠任何一层都不充分：

- 只有 prompt，agent 可能遗漏；
- 只有代码，未经过该代码的入口仍可能绕过；
- 只有 registry，没有运行时强制；
- 只有测试，无法覆盖运行中变更和故障路径。

`GOV-FORMAL-ENTRYPOINT-01` 的正式门禁将在 registry 迁移准备完成后启用。第一阶段只建立规则迁移安全网，不让新检查破坏现有运行。

## 5. handoff 当前状态与历史的推荐结构

用户提出的“当前 handoff 与版本历史分开”是合理方向，但需要机器索引，而不能只拆成很多无法检索的 Markdown。

推荐三层结构：

```text
docs/handoff.md                         # 当前唯一 Master；每个新 session 完整读取
docs/handoff_history/INDEX.yaml         # 机器可读索引
docs/handoff_history/archive/*.md       # 不可破坏的历史原文分片
```

### 5.1 当前 Master

`docs/handoff.md` 只保留：

- 第 0 节启动协议；
- 当前锁定结论；
- 当前术语覆盖；
- 当前实验状态和门禁；
- 当前执行顺序；
- 当前代码入口；
- 未解决问题；
- 历史索引入口。

它仍然每个 session 完整读取，不取消启动协议。

### 5.2 历史索引

`INDEX.yaml` 的每条记录至少包含：

```yaml
record_id: HANDOFF-HIST-CU1-E3-001
file: docs/handoff_history/archive/v21-v24.md
version_range: [21, 24]
date_range: [2026-06-24, 2026-06-25]
experiment_ids: [C-U1-E3-ADAM-RERUN]
claim_ids: [far_field_causal_path]
topics: [optimizer, terminal_audit, support_contraction]
headings: ["E3 Adam rerun", "terminal audit"]
superseded_by: null
source_sha256: "..."
```

查询时优先按 `experiment_id` / `claim_id` / `topic` 过滤，再读取少量对应历史分片，而不是逐个打开所有历史文件。

### 5.3 第 0 节只保留查询入口，不维护巨大人工目录

第 0 节应说明：

- 哪些问题必须查历史；
- 索引文件路径；
- 推荐查询命令；
- 当前结论优先于未显式恢复的旧结论。

具体记录目录由生成器维护，不要求每次手工向第 0 节追加几百条链接。

### 5.4 关于“读取后释放”

模型在同一对话中不能像进程一样保证把旧文本物理释放出上下文。但按需检索仍有价值：

- 启动时不注入全部历史；
- 只读取当前问题相关的少量分片；
- 后续轮次不再重复加载该分片；
- 长对话压缩时，历史细节更容易被压成引用和结论，而不是长期占据显式文本。

因此应理解为“避免重复注入和降低活跃注意力”，不是严格的内存卸载。

## 6. 如何证明每条规则迁移后没有丢失

第一阶段新增 `docs/governance_rule_inventory.yaml` 和 validator。证明链分为五层。

### 6.1 稳定 rule ID

每条准备迁移的规范项获得唯一 `rule_id`，以后位置变化但 ID 不变。

### 6.2 原文指纹和完整覆盖

inventory 记录：

- 原始文件；
- Markdown section；
- 项目序号；
- 规范原文；
- 归一化文本 SHA-256。

validator 从被跟踪 section 重新提取所有顶层 bullet，要求：

- section 中每一条 bullet 恰好对应一个 rule ID；
- inventory 不得遗漏、重复或引用不存在的 bullet；
- 修改原文后必须同步更新 inventory，不能静默漂移。

这比仅靠人工写“已迁移”更强，因为它可以机械证明迁移范围内的每条源规则都被登记。

### 6.3 目的地与职责映射

每条规则记录当前权威位置、详细协议位置、代码 enforcement 和 tests。迁移时必须写明：

- `migration.status: migrated`；
- 新位置；
- 原位置是否保留短版指针；
- 迁移 commit；
- 验证命令和结果。

### 6.4 双轨期

首次迁移不立即删除原文：

1. 新位置先加入；
2. validator 同时检查旧位置与新位置；
3. 运行至少一个真实更新周期；
4. 通过后才将旧位置压缩为短版硬约束和导航指针。

### 6.5 机器可执行规则必须绑定代码与测试

标记为 `machine_enforceable: true` 的规则必须至少有：

- 一个实际 enforcement path；
- 一个测试文件；
- 不允许只剩自然语言说明。

25 MiB / 10 MiB、symlink、安全路径、原子发布、统一 hardened core 等属于此类。

## 7. 渐进实施顺序

### Stage 0：规则迁移安全网（本更新）

- 新增工作设计稿；
- 新增治理规则 inventory；
- 新增 inventory validator；
- 新增单元测试；
- 不迁移、不删除任何现有规则。

### Stage 1：Git commit bundle 与自动三方集成

- 保留现有 `update.patch`；
- 新增小型 `change.bundle` 和 `PATCH_COMMIT.txt`；
- base 相等时沿用旧路径；
- main 前进但 base 是祖先时尝试隔离 worktree 中的 cherry-pick；
- 真冲突或测试失败则停止；
- 记录 `APPLY_REPORT.json`。

### Stage 2：`HANDOFF_DELTA.yaml` shadow mode

- 包仍直接修改 handoff；
- 同时携带结构化 delta；
- 生成 `handoff.generated.candidate.md`；
- 只比较，不替换正式 handoff；
- 验证幂等、顺序无关、去重和冲突拒绝。

### Stage 3：无损历史拆分 shadow mode

- 原 handoff 完整归档并记录 SHA-256；
- 生成 compact candidate；
- 生成历史 `INDEX.yaml`；
- 检查每个旧顶级标题都有 current 或 archive 映射；
- 不立即切换启动入口。

### Stage 4：受控切换与 AGENTS 精简

- 只有前述 shadow 验证稳定后，才切换 handoff 生成路径；
- AGENTS 只压缩已被 inventory 覆盖、已有目标位置和测试的实现细节；
- 仓库级硬约束继续常驻，包括统一通道、25 MiB / 10 MiB、禁止绕过和 fail-closed 原则。

## 8. 验收与回滚

每一阶段必须满足：

- `git apply --check`；
- package 自带测试；
- `git diff --check`；
- 修改文件集合与 manifest 一致；
- 旧权威文件未被破坏性删除；
- 新功能有独立关闭开关或 shadow mode；
- 任一门禁失败时保持旧路径可用。

handoff 切换前还必须满足：

- 历史总字节不减少，只允许移动和增加索引；
- 当前锁定结论、状态和门禁结构化比较一致；
- 相同 delta 重放结果相同；
- 真实语义冲突停止，而不是自动选边。

## 9. 预期增益

- 普通 stale-base 更新包无需整包重做；
- handoff 的纯文本追加冲突显著减少；
- 新训练文件只接入统一通道，避免复制和漂移；
- 规则迁移有可审计证据，不依赖记忆；
- 每次 session 仍完整读取当前 handoff，但不再加载全部历史；
- 简单理论问题减少无关 artifact 细节占用，同时正式任务的硬门禁更强。

## 10. 当前阶段边界

本设计稿和 Stage 0 安全网不改变已有规则的权威性。若 inventory 与 `AGENTS.md` 或 handoff 冲突，仍以原权威文件为准。inventory 当前只覆盖明确登记的迁移候选 section；扩大迁移范围前，必须先把对应 section 纳入完整覆盖检查。

## 11. Stage 0.1：逐规则 assurance 与精确测试证据

Stage 0 证明被跟踪规则没有被静默删除、增加、改写或无目的地迁移，但它只要求机器规则声明测试**文件**。仅验证文件存在，无法证明该文件中确有一个可收集、可运行并与规则绑定的测试。

Stage 0.1 增加 `docs/governance_rule_assurance.yaml`，并要求 inventory 中的每一个 rule ID 恰好拥有一种 assurance：

- `machine`：规则由代码直接执行，必须登记实现路径、精确 pytest node 和覆盖级别；
- `review`：规则依赖治理或科研判断，必须登记触发条件和审查证据；
- `structural`：规则可以通过 schema、索引、哈希或集合完整性验证，必须登记结构检查项。

机器 assurance 的精确节点使用如下形式：

```text
 tests/test_experiment_artifact_hardening.py::test_external_result_symlink_is_rejected
```

validator 必须依次确认：

1. assurance 与 inventory 的 rule ID 集合完全相等；
2. `machine_enforcement.required` 与 `assurance.type: machine` 一致；
3. implementation path 存在，且已经登记在 inventory 的 implementation 集合中；
4. pytest node 的文件存在，且已经登记在 inventory 的 test 集合中；
5. Python AST 中存在精确的测试函数或类方法；
6. 使用 `--collect-pytest` 时，pytest 可以收集全部唯一节点；
7. 使用 `--run-machine-tests` 时，全部节点实际执行成功；
8. 输出逐规则 JSON 报告，而不只给出一个总布尔值。

覆盖级别分为：

- `direct`：节点直接断言该规则的行为；
- `grouped`：端到端测试覆盖一组 package contract，但尚未为该条规则建立独立断言。

`grouped` 不是伪装成完整覆盖。它必须带 `coverage_note`，并在报告中单独计数，使后续专门测试的缺口保持可见。当前 patch contract 中少数规则仍属于 grouped coverage；Stage 0.1 的目标是把这一事实显式化，而不是在没有证据时声称已经一对一验证。

规范命令：

```bash
python3 scripts/validate_governance_rule_inventory.py --repo-root .
python3 scripts/validate_governance_rule_inventory.py --repo-root . --collect-pytest
python3 scripts/validate_governance_rule_inventory.py --repo-root . --run-machine-tests \
  --report-out governance_assurance_report.json
```

第一条进行快速结构和 AST 检查；第二条证明节点可被 pytest 收集；第三条实际运行机器 assurance 的全部唯一测试节点。任何一步失败均应 fail closed。

## 12. Stage 0.2：紧凑输出与 direct package-contract coverage

Stage 0.1 保留了完整逐规则 JSON 报告，但默认把整份报告写到终端。该信息对审计有价值，却不适合作为每次治理检查的默认对话输出：终端内容可能被模型或自动化直接带入上下文，造成与规则数量近似线性增长的 token 开销。

Stage 0.2 将输出职责拆开：

- 默认终端只打印固定长度的通过摘要，包括规则总数、assurance 类型、direct/grouped 覆盖数、pytest 节点数和运行状态；
- `--report-out` 继续把完整逐规则 JSON 写入文件，审计信息不丢失；
- `--verbose` 显式请求时才在终端打印完整 JSON；
- 失败仍然 fail closed，并只展开直接导致失败的规则或测试节点。

这种优化不降低安全等级。validator 仍在内存中构造并验证完整报告，改变的只是正常成功路径的展示方式。因此运行测试主要消耗本地 CPU 与磁盘，而不会要求模型反复阅读全部规则。

Stage 0.1 中三条更新包契约规则仍标记为 `grouped`：`update.patch`、`CHANGE_SUMMARY.md` 和 `TEST_COMMANDS.sh`。`grouped` 表示一个端到端测试整体经过了这些文件，但没有一个独立测试节点直接断言该条规则；它是可见的待补证据，不等于失败，也不等于一对一覆盖。

Stage 0.2 新增 `tests/test_update_package_contract.py`，分别验证：

1. `update.patch` 是必需成员、正式治理包中不得为空，并包含实际修改；
2. `CHANGE_SUMMARY.md` 是必需成员，自动生成内容绑定 experiment ID、base commit 和修改文件；
3. `TEST_COMMANDS.sh` 是必需成员，保留可执行模式、包含 `set -euo pipefail`，并拒绝占位命令。

完成后 11 条 machine assurance 均为 `direct`，`grouped_machine_rules` 必须为 0。若以后新增 grouped coverage，仍必须显式记录 `coverage_note`，以防尚未补齐的证据被误报为完整覆盖。
