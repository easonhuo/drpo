# DRPO 治理与多 Session 集成重构方案（工作设计稿）

**治理 claim：** `GOV-RULE-MIGRATION-01`、`GOV-FORMAL-ENTRYPOINT-01`、`GOV-HANDOFF-INDEX-01`、`GOV-UPDATE-BUNDLE-01`、`GOV-UPDATE-FAST-GATE-01`、`GOV-PIPELINE-STAGE-CLOSURE-01`
**文档性质：** 工作设计稿，不是第二份研究 Master，不改变任何实验状态、冻结参数或科学结论。
**当前权威来源：** `AGENTS.md`、`docs/handoff.md`、`experiments/registry.yaml` 仍保持原有优先级。

## 0. 当前规范阶段编号与关闭状态

本节是当前唯一规范编号。第 7 节保留的是早期历史实施计划；它的旧 Stage 2/3/4 编号已由第 14.1 节顺延，不得再用于判断当前实施状态。机器可读状态与关闭门禁位于 `docs/governance_pipeline_stage_status.yaml`，由 `scripts/validate_governance_pipeline_stage_status.py` fail closed 校验。

| 当前阶段 | 责任 | 状态 | 历史编号关系 |
|---|---|---|---|
| Stage 0 | 规则迁移安全网与 assurance | closed | 未调整 |
| Stage 1 | bundle-backed 更新包、隔离三方集成、测试与诊断闭环 | closed / maintenance-only | 未调整 |
| Stage 2 | 正式实验 guard/package/verify 唯一通道与防绕过门禁 | closed / maintenance-only | 后插入的当前 Stage 2 |
| Stage 3 | `HANDOFF_DELTA.yaml` shadow mode | shadow active / feature-frozen / observation ongoing | 原 Stage 2 |
| Stage 4 | 无损 handoff 历史拆分与语义上下文 shadow mode | active：仅 Stage 4A shadow implementation 已授权 | 原 Stage 3 |
| Stage 5 | 受控切换与 AGENTS 精简 | blocked by Stage 4 | 原 Stage 4 |

`closed / maintenance-only` 不等于永远禁止修复。Stage 1/2 只接受 bugfix、安全修复、兼容性修复和不改变职责的文档澄清；新功能、架构扩张、职责变化或默认策略变化必须先登记新的治理 claim、取得用户明确批准、创建 reopen authorization 和回滚计划。受保护核心文件采用 SHA-256 after-image 与授权记录绑定，任何未授权修改都会使治理 validator 失败。

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

> **历史编号说明：** 本节保留最初计划，不破坏性重写。自第 14.1 节起，原 Stage 2 `HANDOFF_DELTA.yaml` shadow mode 顺延为当前 Stage 3，原 Stage 3/4 顺延为当前 Stage 4/5。当前判断必须使用第 0 节和机器 ledger。

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

## 13. Stage 1：仓库托管的 drpo-update 与 Git bundle 三方集成

### 13.1 本地工具进入仓库

`tools/drpo-update/` 是本地 `drpo-update` 的唯一规范源码。该源码来自用户上传的本地安装包；用户已通过 `cmp` 和 SHA-256 独立确认上传可执行文件与 `/Users/easonhuo/bin/drpo-update` 完全一致。原始可执行文件 SHA-256 为：

```text
f344b0cffc163ecdeb80ec8d07b564d00c3538ad22e03887f87bd1ce2a85f4f3
```

安装器默认使用从 `~/bin/drpo-update` 指向仓库源码的符号链接。这样本地命令不再是脱离仓库的孤立副本，后续仓库更新会自动同步到已安装命令；仍保留 `--copy` 兼容模式。

### 13.2 更新包双表示

Stage 1 保留现有审计成员：

```text
BASE_COMMIT.txt
update.patch
CHANGE_SUMMARY.md
TEST_COMMANDS.sh
modified_files/
```

并允许新增一对不可拆分的 Git 成员：

```text
change.bundle
PATCH_COMMIT.txt
```

`PATCH_COMMIT.txt` 必须是完整 40 位 SHA。该 commit 必须恰有一个 parent，且 parent 必须严格等于 `BASE_COMMIT.txt`。`change.bundle` 与 `update.patch` 不是两个可任选的真相：验证器必须把 patch 应用到 base 后计算 Git tree，并要求它与 patch commit 的 tree 完全一致。

当前 Stage 1 引导更新包本身仍使用旧版 exact-base patch 路径，因为用户电脑上的旧 helper 尚不理解 Git bundle。该包应用并提交后，运行一次仓库内安装器即可切换到 Stage 1 helper；此后新包可以携带 bundle pair。

### 13.3 隔离集成与失败语义

新版 helper 对 legacy 和 bundle 路径都使用临时 worktree。真实 `main` 在以下条件全部成立前保持不变：

1. 包结构安全且完整；
2. base 与 bundle identity 验证通过；
3. patch/tree 等价验证通过；
4. 三方 cherry-pick 或 exact-base apply 成功；
5. `TEST_COMMANDS.sh` 或回退测试通过；
6. `git diff --check` 通过；
7. 用户确认或使用 `--yes`。

若 package base 等于当前 main，legacy patch 和 bundle 都可执行。若 base 已落后但仍是 current main 的祖先，则只有 bundle 路径允许自动三方集成。若 base 不是祖先、发生真实冲突、patch 与 bundle 不一致或测试失败，helper 必须 fail closed，并保持真实 main 干净且未移动。

### 13.4 应用审计

每次执行都在 `~/.config/drpo-update/reports/` 写入结构化报告，记录：

- package original base；
- application 前的 main HEAD；
- patch commit；
- integration mode；
- tests；
- conflict files；
- integrated commit；
- push 状态；
- failure detail。

报告不进入研究结果主文档，也不默认打印完整 JSON，避免治理审计反向膨胀对话 token。

## 14. Stage 2：统一正式实验通道与防绕过门禁

### 14.1 阶段编号调整

早期计划把 handoff delta shadow mode 称为 Stage 2。实施过程中，更新包过期问题已经由 Stage 1 收口，而“新训练入口能否默认继承统一 artifact 规则”成为更直接的治理风险。因此从本节起调整执行顺序，但不删除第 7 节的历史计划：

- 当前 Stage 2：统一正式实验 guard/package/verify 通道与防绕过门禁；
- 原 Stage 2 handoff delta shadow mode 顺延为 Stage 3；
- 原 Stage 3 无损历史拆分顺延为 Stage 4；
- 原 Stage 4 受控切换与 AGENTS 精简顺延为 Stage 5。

该调整只改变治理工作的实施顺序，不改变任何科学实验职责、冻结参数或结果状态。

### 14.2 唯一正式执行通道

`experiments/registry.yaml` 新增顶层 `formal_execution_channel`，将以下文件登记为唯一规范通道：

```text
scripts/run_experiment_guard_hardened.py
scripts/package_experiment_hardened.py
scripts/verify_experiment_package_hardened.py
scripts/artifact_protocol_hardened.py
docs/formal_experiment_artifact_protocol.md
```

新的训练文件只负责训练、指标和规定的 checkpoint 输出，不自行复制 25 MiB / 10 MiB、symlink、checksum、原子发布或失败恢复逻辑。正式运行的 artifact 所有权属于统一通道。所有 registry 实验必须显式声明 `execution_class`：

- `formal`：当前或未来可能产生正式证据；
- `pilot`：只产生 pilot 或开发证据；
- `historical_formal`：Stage 2 之前已经完成并保留 provenance 的正式运行；
- `superseded`：保留历史但不得重新启动的旧协议。

正式条目必须声明 canonical channel、guard、packager、verifier、hardened core、artifact protocol、entrypoint 状态和 runner archive policy。已规划但尚未实现的正式入口只能处于 blocked 状态，不能伪装为可执行入口。

### 14.3 防绕过 validator

`scripts/validate_formal_execution_channel.py` 负责：

1. 验证三个公共 wrapper 都导入同一个 hardened core，并在 core 缺失时 fail closed；
2. 锁定默认主包 25 MiB、单文件 10 MiB、persistent-local 大文件索引和 sidecar 默认关闭；
3. 要求每个正式实验显式绑定统一通道；
4. 要求 active formal entrypoint 的 launch template 经过 canonical guard 并绑定精确 experiment ID；
5. 静态拒绝新正式 Python entrypoint 直接创建 ZIP/TAR/7z 等 archive；
6. 区分 formal、pilot、historical formal 和 superseded，防止 pilot 冒充正式结果；
7. 默认只输出紧凑摘要，完整逐实验报告写入文件，避免治理输出反向膨胀 token。

该 validator 不能替代 artifact hardened core。它负责证明“入口是否被正确接线”，hardened core 负责真正执行路径、大小、symlink、checksum、发布和失败恢复规则。

### 14.4 现有 C-U1 recovery checkpoint 的窄例外

`src/drpo/drpo_cu1_e1_e4_oneclick.py` 在 Stage 2 之前已经包含每五个 seed 写一次 recovery checkpoint ZIP 的逻辑。该 ZIP 是恢复检查点，不是最终 formal artifact。为避免在本阶段同时重写已登记的 C-U1 runner，registry 只为 `C-U1-E4-CONV-01` 登记一个精确 legacy exception：

```text
CU1-RECOVERY-CHECKPOINT-LEGACY-01
```

例外仅允许既有 entrypoint、既有 experiment ID 和 `recovery_checkpoint_only` 范围；新训练入口不得复制。最终/raw-complete 包仍必须由 canonical hardened channel 生成。未来若重构 C-U1 checkpoint API，应先移除直接 archive 写入，再删除该例外，而不是扩大例外范围。

### 14.5 Stage 2A 的边界

本阶段只完成 registry schema、validator、规则 assurance 和负向测试：

- 不启动任何实验；
- 不修改 seeds、数据规模、阈值、优化器或终态标准；
- 不删除 AGENTS 中现有硬规则；
- 不修改 artifact 大小预算；
- 不把历史正式运行倒推为经过当前 guard；
- 不开始 handoff 历史拆分。

Stage 2A 验收后，新增 formal entrypoint 若未接入统一通道、使用自定义 packager、直接创建 formal archive 或缺少明确 execution class，治理测试必须失败。

## 15. Stage 1D：更新应用计时与 fail-closed fast gate

Stage 1 已解决 stale base 的自动三方集成，但“无冲突更新究竟慢在哪里”此前没有结构化数据，且所有包只能依靠各自的 `TEST_COMMANDS.sh` 决定测试范围。Stage 1D 在不取消测试的前提下增加两项能力：

1. `drpo-update` 在 `APPLY_REPORT` 中记录 package extraction、repository preflight、fetch/merge、base resolution、bundle verification、integration、package tests、repository test gate、review、main fast-forward、push 和 total 的耗时；
2. 由当前真实 `main` 中的 `tools/drpo-update/test_impact_map.json` 根据 candidate diff 选择 focused fast gate 或 full suite。

测试选择必须满足以下安全边界：

- `TEST_COMMANDS.sh` 仍然先执行，保留包级专用验证；
- 已知 low/medium-risk 文件只运行变更 Python 文件的 compile/Ruff、映射 validator 和映射 pytest targets；
- shared artifact core、正式训练代码、依赖、AGENTS、测试选择控制面等 high-risk 文件强制 full suite；
- 任何未被 impact map 覆盖的新路径默认 full suite，不得静默跳过；
- 显式 `--test-mode fast` 不能降级 high-risk 或 unknown-path 决策；
- impact map 从应用前真实 `main` 读取，candidate 不能先修改 map 再利用新 map 弱化自己的门禁；
- 任一 package test、selected test、Ruff、compile 或 validator 失败，真实 `main` 保持不动。

该机制优化的是“相关测试选择”和可观测性，不把“无文本冲突”错误等同于“无需测试”。包生成时验证的是 `base + patch`，stale integration 实际验证的是 `new main + patch`，两者属于不同软件状态，仍必须至少经过 focused integration gate。

## 16. Stage 1E：生产端全仓验收与失败诊断闭环

Stage 1E 补齐 Stage 1D 之后仍存在的两个低摩擦缺口：更新包生产端未必先在
exact-base 完整仓库中执行全仓门禁，以及用户端失败后仍需人工整理日志和仓库
状态。该阶段不改变任何科学实验、冻结配置或结果状态。

### 16.1 生产端交付硬门禁

任何代码更新包在交付前必须基于已确认的完整 Git commit object：

1. 在未修改 exact base 上记录 baseline 门禁结果；
2. 在独立干净 checkout 中按用户侧等价路径应用 `update.patch`；
3. 执行包内 `TEST_COMMANDS.sh`；
4. 聚合执行 compile、Shell syntax、formal-channel validator、governance
   validator、全仓 pytest 与全仓 Ruff；
5. 执行 `git diff --check`、modified-file identity 和 executable-mode 检查；
6. 任一候选新增失败时禁止交付。

baseline 若存在失败，必须明确记录原失败集合；不能把 base 环境缺失或原有测试
债务误报为候选成功。用户侧 `drpo-update` 仍执行最终集成复核，但不应成为普通
源码回归的首次发现位置。

### 16.2 聚合门禁

`tools/drpo-update/test_selection.py` 不再在第一个独立门禁失败时立即退出。完整
门禁依次尝试：

- Python compileall；
- updater Shell syntax；
- formal execution channel validator；
- governance rule inventory validator；
- full pytest；
- full Ruff。

fast gate 同样聚合 changed-file compile/Ruff、映射 validators 与映射 pytest。
每个命令单独记录 return code 和完整日志，全部结束后统一报告失败集合。缺少 Ruff
等必需执行文件按门禁失败处理，不再静默跳过。

### 16.3 自动诊断包

`drpo-update` 任一失败路径默认原子生成：

```text
~/Downloads/DRPO_DIAGNOSTIC_<HEAD>_<TIMESTAMP>_<ID>.zip
```

可通过 `--diagnostic-dir` 或 `DRPO_UPDATE_DIAGNOSTIC_DIR` 覆盖，但默认不得写入
Desktop。诊断包至少包含原更新包、apply report、完整测试日志、当前仓库 Git
bundle、候选 identity/diff、冲突 base/ours/theirs/worktree 四份材料、Git 状态与
refs、依赖/系统信息和 SHA-256 manifest。

冲突和测试失败时，真实 `main` 在诊断包完成前保持不动；诊断包必须在临时
worktree 清理前生成。用户失败后的唯一人工动作是上传这一份 ZIP，不再手工执行
`git bundle create` 或逐段复制日志。

### 16.4 验收路径

Stage 1E 自动测试必须覆盖：

1. stale ancestral bundle 非冲突成功；
2. 真实三方冲突，主分支不动且诊断包含四方冲突材料；
3. package test 失败后仍执行 repository gate，并生成完整日志；
4. full gate 首项失败后后续独立命令仍执行；
5. 默认诊断目录严格为 `~/Downloads`；
6. 失败后仓库 clean、HEAD 未移动；成功后才 fast-forward。

## 17. Stage 1F：Bundle-default 生产闭环与成功提交快照

Stage 1F 关闭 Stage 1 在“消费端已支持 bundle、生产端却仍默认手工 patch-only”上的迁移缺口。该阶段只修改代码更新基础设施，不启动实验、不改变实验状态，也不修改冻结科学变量。

### 17.1 新包生产规范

1. 新生成的代码更新包必须由 `scripts/package_update.py` 构建。
2. canonical 包必须同时包含 `change.bundle`、`PATCH_COMMIT.txt` 和 `UPDATE_PACKAGE_MANIFEST.json`；缺任一项即为生产端硬失败。
3. `PATCH_COMMIT.txt` 指向的提交必须以 `BASE_COMMIT.txt` 为唯一父提交；bundle tree、`update.patch` 应用结果、`modified_files/` 完整 after-image 和 executable mode 必须一致。
4. 历史 patch-only exact-base 包仍可由 `drpo-update` 消费，但 canonical producer 不再生成这种 legacy package。
5. canonical verifier 为 `scripts/verify_update_package.py`。生产端验证失败时禁止交付。

### 17.2 成功 push 后导出 main bundle

只有在候选测试通过、本地 `main` fast-forward、`git push origin main` 成功且随后 `git ls-remote origin refs/heads/main` 与本地 HEAD 完全一致后，`drpo-update` 才自动在 `~/Downloads` 原子生成：

```text
DRPO_MAIN_<12-char-SHA>.bundle
DRPO_MAIN_<12-char-SHA>.bundle.sha256
DRPO_MAIN_LATEST.bundle
DRPO_MAIN_LATEST.bundle.sha256
```

`--main-bundle-dir` 或 `DRPO_UPDATE_MAIN_BUNDLE_DIR` 可覆盖目录；默认不得使用 Desktop。`--no-push` 不生成正式 main bundle；`--no-export-main-bundle` 可显式关闭成功后的导出。push 成功但导出失败时，提交已存在于本地和远端，apply report 必须使用单独状态并自动生成诊断 ZIP，不得误报为“main 未修改”。

### 17.3 本机 doctor

`drpo-update --doctor` 在临时 synthetic repositories 中运行 bundle exact/stale、真实冲突、失败诊断、producer/verifier 与 post-push export 的事务测试，同时执行 Python compile 和 Shell syntax。doctor 不修改真实 `main`、不创建真实提交、不 push。

### 17.4 验收矩阵

Stage 1F 交付前至少覆盖：

1. canonical producer 总是生成 bundle pair 和 manifest；
2. producer 对缺失 after-image、错误 mode、patch/bundle 不一致 fail closed；
3. producer verifier 拒绝新生产的 patch-only 包；
4. updater 继续接受历史 exact-base legacy 包；
5. bundle-backed exact-base 与 stale ancestral 无冲突路径成功；
6. stale 真实冲突与测试失败保持原 main 不动并生成诊断 ZIP；
7. 成功 push 后远端 SHA 二次确认并生成 versioned/latest bundle 与 SHA-256；
8. `--no-push` 和 `--no-export-main-bundle` 均不生成正式 main bundle；
9. `drpo-update --doctor` 全部通过；
10. candidate 全仓 pytest、Ruff、compile、Shell syntax、治理 validators 与 `git diff --check` 通过。

## 18. Stage 1/2 冻结式 Closure（2026-06-27）

### 18.1 Closure 的载体与权威边界

Closure 不是新建第二份研究 Master，也不是只写一句“已经完成”。它由四层共同体现：

1. 本文档保存人可读的职责、验收证据、遗留边界和阶段编号；
2. `docs/governance_pipeline_stage_status.yaml` 保存机器可读状态、受保护文件 after-image 和 reopen 条件；
3. `docs/governance_stage_authorizations/` 保存用户批准、治理 claim、变更类别、授权状态和授权文件哈希；
4. `scripts/validate_governance_pipeline_stage_status.py`、pytest 和 update impact map 共同形成 fail-closed 门禁。

`AGENTS.md` 与 `docs/handoff.md` 第 0 节只保留入口指针和不可绕过原则，避免形成第三份重复清单。

### 18.2 Stage 1 关闭结论

Stage 1 的责任边界是代码更新包生产、验证、隔离集成、测试、诊断和成功 push 后的 main bundle 导出。关闭时已具备：

- canonical bundle-backed producer 与 verifier；
- exact-base 和 stale-ancestral 隔离集成；
- patch、bundle、after-image 和 executable mode 等价验证；
- package test、fast/full repository gate、聚合失败日志和 `git diff --check`；
- 冲突或测试失败时真实 `main` 不移动，并自动生成诊断 ZIP；
- push 后远端 SHA 二次确认与 versioned/latest main bundle 导出；
- `drpo-update --doctor` 事务测试；
- 默认 symlink 安装与显式 `--copy` 安装都能完整部署 wrapper、Python runtime 和 test selector；
- recovery/raw-complete 证据包与代码更新包的类型区分。

Stage 1 状态关闭为 `closed_maintenance_only`。历史 patch-only exact-base 消费兼容继续保留，但不得恢复为新包生产格式。

### 18.3 Stage 2 关闭结论与 E6 生产验收

当前 Stage 2 的责任边界是正式实验统一 guard/package/verify 通道，不是 handoff delta。关闭时已具备：

- registry 中唯一 `hardened-v1` channel；
- formal/pilot/historical_formal/superseded 执行类别；
- clean commit、origin/main、源文件和输出目录运行前门禁；
- 前台 supervisor、heartbeat、stale escalation、退出和结束 provenance；
- raw-complete、failed、checkpoint、final 等 package kind；
- safe path、symlink、checksum、size budget、原子发布和旧 artifact 保留；
- formal entrypoint 接线、防自定义 archive、防 blocked/active 状态矛盾的 validator；
- 任务性能崩溃、支持/方差边界和 NaN/Inf 分报要求。

`D-U1-E6-SEMANTIC-LONGRUN-01` 是首个完整生产验收：clean commit `eb5e12626026854f44f4698dbc8ed8829e74e0b0` 上完成 360/360 formal runs、2x terminal audit、canonical raw-complete package 与 durable delivery；随后 compact closure 进入 `main` commit `ff2afe443167154eae5de7871cda83f3aba9a89e`。该实验同时暴露并修复了 raw-complete recovery package 被误交给 `drpo-update` 时诊断不清的问题。Stage 2 因而关闭为 `closed_maintenance_only`。

### 18.4 已知遗留项与非阻塞边界

- E6 首个 repository closure ZIP 的 SHA-256 未在仓库证据中记录；保持 `null` 并显式登记 `not_recorded`，不得推测。该缺口不影响已应用 commit、科学结果或 Stage 2 通道验收。
- Stage 1/2 的维护门禁不能替代用户对治理授权的真实性审查；机器门禁负责要求授权记录、状态和 after-image 自洽，不声称提供密码学身份认证。
- 新实验继续允许更新 registry 和科学 runner；关闭保护的是 Stage 1/2 的核心职责实现，不冻结科研路线。
- Stage 3 尚未实现。本 closure 包不包含 `HANDOFF_DELTA.yaml` schema、合并器或 candidate handoff 生成。

### 18.5 关闭后的变更规则

Stage 1/2 受保护核心文件必须与 ledger 中 SHA-256 一致。维护修改必须：

1. 使用允许的 change class；
2. 登记新的 authorization record、治理 claim、用户批准来源和授权 after-image；
3. 更新 ledger 中对应文件哈希与 `authorized_by`；
4. 运行 stage-status validator、相关 pytest、全仓门禁和 canonical package verifier。

新功能、架构扩张、职责变化或默认策略变化必须先将目标阶段通过 `reopen` authorization 显式重开，并包含回滚计划。直接编辑 ledger 状态、只更新哈希或只修改文件都会被测试拒绝。

### 18.6 下一阶段

当前 Stage 3 为 `ready_not_started`。下一更新应独立实现 `HANDOFF_DELTA.yaml` shadow mode，并保持“直接修改正式 handoff + 同时生成 candidate + 只比较不切换”的双轨边界；Stage 1/2 closure 不与 Stage 3 实现混包。

## 19. Stage 3.1：`HANDOFF_DELTA.yaml` Shadow Mode 实现与运行协议（2026-06-27）

### 19.1 当前阶段边界

Stage 3 由 `ready_not_started` 迁移为 `shadow_active`。本阶段仍采用双轨：人工直接修改的 `docs/handoff.md` 是唯一正式结果，结构化 delta 只负责从 exact base replay 出 candidate 并做比较。candidate 不得覆盖、替换或自动提交正式 handoff；authority cutover 必须由后续独立、用户明确批准的 stage transition 完成。

Stage 1/2 的 closure 继续有效。为了让现有更新器自动选择 Stage 3 gate，本版仅临时 reopen Stage 1 的 test-impact integration surface，加入 `handoff_delta_shadow` 测试组后立即重新关闭；不修改 bundle producer、三方集成、push 或正式实验 artifact 逻辑。

### 19.2 版本 1 的结构

Stage 3.1 使用七层结构：

1. `HANDOFF_DELTA.yaml`：声明 exact base、操作、registry 变更和预期 after-image；
2. schema/provenance validator：拒绝未知字段、短 SHA、base/hash 不一致和不安全路径；
3. Markdown heading index：以完整 heading path 唯一定位目标；
4. append-oriented operation engine：只允许 `replace_heading`、`insert_after_heading`、`append_to_section`；
5. deterministic renderer：相同 base + delta 必须产生相同 candidate，重复 replay 必须幂等；
6. safety validators：历史 experiment/claim ID 守恒、registry state-machine、candidate/manual exact match；
7. shadow report：记录 hash、affected selectors、timing 和是否使用 network/LLM。

版本 1 明确禁止任意 `replace_text`、整段自由替换和破坏性删除。需要纠错时仍采用“旧结论—问题—新证据—新结论”的 append/supersession 方式；未来若确需新增 destructive-capable operation，必须升级 schema 主版本并重新执行 Full Acceptance。

### 19.3 Delta 位置和单更新约束

每个涉及 handoff 或 registry 的更新必须新增且只新增一个：

```text
docs/handoff_deltas/<update_id>/HANDOFF_DELTA.yaml
```

目录名必须等于 `update_id`。版本 1 每个 update package 最多一个 delta，以避免同一提交内多 delta composition 复杂性；多 session 独立 delta 的 commutativity 和冲突检查由 pair/replay 测试覆盖。成功更新保留 delta 和 `SHADOW_REPORT.json`，不默认保存整份 candidate；失败、Full Acceptance 或 golden replay 才保存完整 candidate。

### 19.4 Fast / Standard / Full 固定触发

**Fast Gate** 在每次修改 `docs/handoff.md`、`experiments/registry.yaml`、delta、policy、state machines 或 Stage 3 renderer 时运行。它必须本地、确定性、无网络、无 LLM 阻塞判断；目标 p95 `<=5s`，硬上限 `15s`。

**Standard Regression** 在 schema、renderer、state machine、conflict rule、parser/index 或新增 operation 时运行，目标 `<=60s`。普通只更新某个实验状态或结果的 delta 不重复跑 Standard。

**Full Acceptance** 在以下任一条件触发：

1. shadow activation 前；
2. authority cutover 前；
3. schema 主版本升级；
4. renderer/state-machine 架构级变化；
5. 累计 20 次成功相关更新；
6. 距离上次 Full 已 7 天，且期间至少发生一次相关更新；
7. critical semantic mismatch 修复完成；
8. 正式 cutover 后累计 50 次 delta 或重大版本升级。

Full 目标 `<=15min`，覆盖全量 replay、mutation、idempotence、独立操作交换性、冲突矩阵、历史守恒和人工审查 critical differences。没有相关变更时，7 天兜底不触发空跑。

### 19.5 为什么禁止 LLM 作为阻塞式门禁

Fast/Standard 的 pass/fail 只能由可重复的本地规则决定。LLM 可能因模型版本、网络、限流或上下文变化产生不同判断，因此只能在 Full Acceptance 中作为非阻塞 warning producer；critical difference 最终由确定性检查或人工裁决。更新器不得因外部模型不可用而无法应用一个本地 Git 包。

### 19.6 增量渲染和存储

renderer 只触碰 delta 声明的 heading 或 section。未声明部分直接沿用 base bytes；候选最终仍与人工 handoff 做全文件 exact-match，因此任何未声明手工改动都会失败。正常通过只保留 delta、报告、hash 和 affected selectors，不保存重复的完整 candidate；失败时才写 full candidate 以便诊断。

### 19.7 Bootstrap 自举与状态边界

本版自身由 `GOV-STAGE3-SHADOW-BOOTSTRAP-2026-06-27/HANDOFF_DELTA.yaml` 从 `40a942b5305e7de35acc16add2b6cc6b798a4508` replay 生成 v50 candidate，并要求与人工 v50 handoff 字节级一致。该 bootstrap、单元测试和 Full tier 通过只允许把 Stage 3 标为 `shadow_active`，不允许标为 `shadow_validated`，更不允许切换权威路径。Stage 3 closure 仍需真实更新周期和预注册 coverage 完成。

## 20. Stage 3.2：真实 Observation 记账、报告持久化与验收触发自动化（2026-06-27）

### 20.1 真实观察与 provenance 语义

`DU1-E6-SEMANTIC-GAP-FORMAL-2026-06-27` 是第一个真实 shadow observation，`EXT-C-E8-V4.3-DYNAMIC-CONTROL-2026-06-27` 是第二个。旧 `SHADOW_REPORT.json` 中的 `head_commit` 来自隔离验证 worktree，不代表最终 repository commit；Stage 3.2 不重写该历史证据，而是在读取时将其解释为 legacy validation-worktree head，并从 Git 历史导出真实 repository commit。V4.3 的历史 schema-v1 delta 在本版补写确定性 sibling report。

新报告使用 `validation_worktree_head`，不再输出含义模糊的 `head_commit`。最终 repository commit 由 observation audit 从 delta 首次进入仓库的提交中推导，因此不要求在尚未生成最终 commit 时预知其 SHA。

### 20.2 派生式 observation ledger

不新增需要每次人工递增的计数文件。不可变 delta 目录、sibling report 与 Git history 共同构成事实 ledger；`corpus-check` 在每个 observation 的历史 repository after-image 上执行 replay，验证 candidate/manual、stored report、幂等、历史守恒和 registry 断言，并输出 bootstrap/real 数量、真实 repository commit、validation-worktree head 和耗时统计。

该设计避免“报告已成功但计数 ledger 忘记更新”的双写漂移。Stage ledger 只登记 observation ledger 的派生模式、入口和权威边界，不复制动态计数。

### 20.3 报告持久化与重新验证

每个新 delta 必须有 sibling `SHADOW_REPORT.json`。作者用 `record-report` 单命令生成；普通 `auto-check` 不覆盖报告，而是重新 replay，并比较排除 timing/provenance 后的确定性语义投影。缺失报告、报告 stale、hash 或 registry assertion 不一致均 fail closed。 pre-v2 历史 observation 的 report-only 补写按其 repository after-image 复验，不计作第二个新 delta；同一更新仍只允许一个新增或修改的 `HANDOFF_DELTA.yaml`。

正常成功仍不保存完整 candidate。失败、golden replay 或 Full Acceptance 才保存完整候选，保持热路径低成本。

### 20.4 Delta schema v2 与 registry 完整覆盖

schema v2 保留原三种 append-oriented Markdown operation，但加强 registry 协议：

- `transition`：必须满足注册状态机；
- `add_entity`：新增实验必须显式登记并绑定 evidence；
- `update_field`：非状态字段必须给出 exact before/after、理由与 evidence；
- experiment 删除禁止；
- 所有实际 registry diff 必须被断言恰好覆盖一次，未声明变更和虚假断言均拒绝。

bootstrap、observation #1 与 V4.3 observation #2 作为不可变历史继续使用 schema v1 allowlist；新 update ID 禁止再使用 v1。旧 v1 的 exact registry after-image 仍由 SHA-256 绑定，但报告明确标记为 legacy partial assertions，不能冒充 v2 完整语义覆盖。

### 20.5 Full Acceptance 自动触发

最新成功 Full report 必须记录覆盖的 real observation update IDs。`acceptance-status` 从该 coverage 与当前 observation corpus 计算未覆盖更新：

- 未覆盖成功更新达到 20 次，Full due；
- 距上次 Full 达到 7 天且存在未覆盖更新，Full due；
- 没有未覆盖更新时不因日历经过而空跑。

普通 `auto-check` 在 overdue 时阻塞；Full runner 可使用显式 `--allow-full-due` 完成修复性验收，成功后持久化新的 coverage report。schema major、renderer/state-machine 架构变化、critical mismatch 修复和 authority cutover 前仍是事件触发的强制 Full。

### 20.6 当前状态边界

Stage 3 保持 `shadow_active`，acceptance state 更新为 `real_shadow_observation_active`。本版属于 schema/renderer/acceptance architecture 变化，因此自身必须执行 Full Acceptance；应用后构成第三个真实 observation。该证据仍不足以切换 authority：后续必须继续覆盖真实 `append_to_section`、多 session 独立交换、同 entity 冲突和更多 registry 变更类型。

## 21. Stage 3 功能冻结与 Stage 4 Shadow 候选设计登记（2026-06-28）

### 21.1 Stage 3 收尾边界

Stage 3 的核心实现已经完成并进入功能冻结：`feature_frozen_bugfix_only`。后续只接受 bugfix、安全修复、兼容性修复和不改变职责的文档澄清；不得继续扩张 schema、operation、report 体系或 authority 职责。Stage 3 仍保持 `shadow_active`，继续累积真实 observation，并继续执行 20 次更新／7 天和事件触发的 Full Acceptance。功能冻结不等价于 `shadow_validated`，也不授权 authority cutover。

本次持久化 Full Acceptance checkpoint 覆盖当前 8 个真实 observation 和 1 个 bootstrap observation，关闭 `c2ad7d5...` 所修复的 report-equivalence critical semantic mismatch 的验收记录缺口。该 checkpoint 证明现有 observation corpus 可重放，不代表 Stage 3 的预注册时间／数量门槛已经完成。

### 21.2 Stage 4 的准确边界

Stage 4 的准确名称应理解为“无损拆分候选的 shadow 验证”，不是正式拆分和搬家。它可以在隔离的 candidate 路径中生成 compact current、语义模块、完整历史、索引、claim lineage、dependency graph 和 context pack，但：

- `docs/handoff.md` 仍是唯一研究 Master；
- 默认启动协议仍读取现有 handoff；
- shadow candidate 不得替换、覆盖或自动提交正式 handoff；
- Stage 5 才能在独立用户授权后执行 authority cutover。

### 21.3 Stage 4 的三步实施

Stage 4 按独立更新分为：

1. **Stage 4A：结构登记与静态清点。** 冻结语义模块、节点、关系、claim lineage、依赖闭包和 fail-closed 规则；建立 inventory、validator 和测试。暂不生成正式拆分结果。
2. **Stage 4B：无损拆分 candidate。** 在非权威路径生成 current/modules/history/index/graph，并机械检查字节、标题、claim、状态、门禁、确定性和索引完整性。
3. **Stage 4C：上下文装配与真实 shadow 验证。** 对真实任务加载目标节点的传递依赖闭包，与完整 handoff 下的研究判断对照，验证不会因压缩上下文降低研究质量。

不得把 4A、4B、4C 和 Stage 5 cutover 一次性混入同一更新。

### 21.4 语义模块与 Research Claim Graph

Stage 4 按研究功能拆分，而不是按 Markdown 章节号机械切割。节点至少区分 question、hypothesis、claim、assumption、experiment、evidence、method、limitation、alternative 和 open issue；关系至少区分 `depends_on`、`tests`、`supports`、`contradicts`、`supersedes`、`external_validates`、`does_not_replace` 和 `motivates`。

上下文装配必须加载目标节点、项目总规则和全部传递依赖闭包。研究 E7 时，E1-E4 连续机制链、Gaussian 方差方向修正和 collapse taxonomy 属于必要上游，不能只按“E7”关键词裁剪。依赖不确定时宁可多加载，禁止为了节省 token 少加载。

### 21.5 旧结论的可发现性

被替代结论不再作为当前权威使用，但必须保留：

- 高度凝练摘要；
- 旧术语和关键词；
- `superseded_by` 谱系；
- 被替代的理由和证据；
- 在何种新 regime 下值得重新检查的 `reopen_conditions`；
- 完整 archive pointer。

任何结论只要没有明确 `superseded_by`，就不能仅因为时间久或近期未使用而自动判为过期。当前、未解决、被替代和纯历史 provenance 必须分开标记。

### 21.6 规范入口和非授权事项

完整 Stage 4 设计位于 `docs/governance_stage4_semantic_context_spec.md`。本节和该规范只登记未来实现合同，不授权：

- 启用 shadow candidate 作为默认上下文；
- 正式拆分或替换 handoff；
- 启动 Stage 5；
- Delta-first authority cutover；
- 修改任何科研实验状态、参数、门禁或执行顺序。

## 22. Stage 4A 与 Stage 3 Observation 并行授权（2026-06-28）

### 22.1 前置关系修正

第 21 节记录的是 Stage 4 初始设计登记时的保守门禁：整个 Stage 4 implementation 等待 Stage 3 shadow validation 完成。后续审查确认，这个串行关系过强。Stage 3 与 Stage 4 的技术职责不同：Stage 3 验证结构化 Delta 修改链；Stage 4 在隔离路径验证知识拆分与上下文装配。Stage 4 shadow 开发不会替换正式 handoff，也不会停止 Stage 3 observation。

因此，Stage 4A 的实现前置条件修正为：

- Stage 3 核心实现已经完成；
- Stage 3 已进入 `feature_frozen_bugfix_only`；
- Stage 3 继续保持 `shadow_active` 并正常累积 observation；
- `docs/handoff.md` 继续是唯一研究 Master；
- Stage 4 所有输出继续只存在于明确的 shadow/candidate 路径。

Stage 3 的 20-update／7-day Full Acceptance 观察门槛不再阻塞 Stage 4A 开发，但仍然是 Stage 5 authority cutover 的必要前置证据。

### 22.2 当前只授权 Stage 4A

本次只授权 `stage_4a_schema_inventory`：schema、静态 inventory、validator 与单元测试。不得在同一更新中提前实现 Stage 4B renderer、Stage 4C context packer 或 Stage 5 cutover。

固定相位门禁为：

1. Stage 4A：`authorized`；
2. Stage 4B：`blocked_by_stage_4a_acceptance`；
3. Stage 4C：`blocked_by_stage_4b_acceptance`；
4. Stage 5：同时等待 `stage_3_shadow_validation` 与 `stage_4_lossless_validation`。

Stage 4A 验收必须由独立更新登记；实现完成本身不能自动解锁 Stage 4B。

### 22.3 不变边界

本次授权不改变：

- Stage 3 的 `shadow_active`、feature freeze、报告持久化或 Full Acceptance 规则；
- `docs/handoff.md` 的唯一权威地位；
- 默认新会话启动协议；
- 任何科研实验状态、参数、seeds、阈值、终态审计或执行顺序；
- Stage 4B、Stage 4C 或 Stage 5 的实现和 authority 权限。

## 23. Pipeline Phase 2 Backlog：handoff 模块化与低风险辅助功能

本节登记二期规划，不改变当前 `docs/handoff.md` 的唯一研究 Master 地位，不授权立即实现，也不修改任何实验状态、冻结参数、seeds、阈值或执行顺序。

### 23.1 二期主挑战：handoff 拆分与模块化

Pipeline Phase 2 的最大挑战是将当前超长 `docs/handoff.md` 逐步拆分为可索引、可验证、可按需装配的模块体系，同时保持完整历史、当前结论、术语覆盖、实验门禁和执行顺序不丢失。

二期 handoff 模块化必须继续遵守：

1. `docs/handoff.md` 在正式切换前仍是唯一研究 Master；
2. 拆分候选必须先在 shadow/candidate 路径中验证；
3. 当前结论、历史 supersession、claim lineage、证据、限制和 open issue 必须可追溯；
4. 上下文装配必须加载目标节点及其传递依赖，不能只按关键词裁剪；
5. 切换 authority 必须另有独立授权、验收和回滚计划。

### 23.2 变更文档与状态分类

二期需要维护可读的 pipeline backlog/change document，用于区分：

- `implemented`：已实现并通过验收；
- `shadow-only`：只在 shadow/candidate 路径运行；
- `partial`：部分完成但不能视为闭环；
- `paused`：暂停，不进入当前主流程；
- `reverted`：曾实现但已回滚；
- `future backlog`：未来可能实现但尚未授权。

每个未来小功能都必须单独登记范围、是否触碰核心更新器、是否触碰 handoff/registry、是否影响治理门禁，以及对应的回滚方案。

### 23.3 暂停项：`.drpoupdate` macOS 双击入口

`.drpoupdate` Finder/macOS 双击 App 已暂停并归入 Phase 2 backlog。它不得进入当前 active update path。

未来如重新评估，只能作为 old CLI 的薄壳 wrapper：

1. 只接收用户双击的包路径并调用既有 `tools/drpo-update` CLI；
2. 不修改 `drpo-update` 核心语义；
3. 不修改 `BASE_COMMIT`、stale-package recovery、bundle-backed integration 或治理门禁；
4. 不和 post-push bundle export、preflight 语义变化或核心 updater 重构混包；
5. 必须先证明 old-CLI equivalence、旧 base 非冲突兼容、真实冲突保护和可回滚性；
6. 未通过这些回归测试前不得 push。

### 23.4 默认包交付策略

后续 DRPO 更新包默认必须是 canonical bundle-backed package。Patch-only runnable package 不再作为正常交付格式，即使生成时已经拿到最新 `main`，也不能假设用户应用前 `main` 不会被其他提交推进。

如果无法生成并验证 canonical bundle-backed package，交付物只能是方案、非运行 diff 草案或对最新 Git bundle/diagnostic 的明确需求；不得提供一个可能因普通 `main` 前进而立即失效的 runnable patch-only 包。

### 23.5 图表生成路由与在线轮询规则

论文图、实验图和 paper figure 的默认执行方式是代码绘图，而不是图片生成模型。该规则同步登记在 handoff v76 常驻入口中，以降低“画图/图片/Figure”误触发 image generation 的概率。

在线轮询属于跨项目通用执行语义：用户要求“轮询/盯着跑/跑完再汇报/等终态再说”等表达时，必须解释为同一轮内阻塞式工具检查到终态或明确失败，而不是后台想象监控或下次再查。该规则同步登记在 handoff v76，因为它影响长跑实验、pipeline、训练和诊断任务的默认行为。
