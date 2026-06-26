# DRPO 治理与多 Session 集成重构方案（工作设计稿）

**治理 claim：** `GOV-RULE-MIGRATION-01`、`GOV-FORMAL-ENTRYPOINT-01`、`GOV-HANDOFF-INDEX-01`、`GOV-UPDATE-BUNDLE-01`、`GOV-UPDATE-FAST-GATE-01`
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
