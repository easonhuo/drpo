# DRPO Dev Branch → Main 集成 Pipeline 构建交接说明

**建议治理 claim：** `GOV-DEV-BRANCH-INTEGRATION-01`  
**文档性质：** 构建设计交接稿，不是研究 Master，不改变当前实验状态、冻结变量、默认合并策略或任何已关闭治理 Stage 的职责。  
**权威来源：** `AGENTS.md`、`docs/handoff.md`、`experiments/registry.yaml`、`docs/governance_pipeline_stage_status.yaml`。  
**编写背景：** 2026-07-11 对 `dev/e8-oracle-bank-v2` 的真实集成暴露出缺失的标准化 dev→main 科学实验集成通道。  
**面向读者：** 后续专门负责构建该 pipeline 的 ChatGPT/session。  

---

## 0. 启动要求

开始任何实现前必须：

1. 读取 `AGENTS.md`。
2. 首先读取 `docs/handoff.md` 第 0 节。
3. 读取 `experiments/registry.yaml`。
4. 读取 `docs/governance_pipeline_stage_status.yaml`。
5. 读取 `docs/governance_pipeline_refactor_plan.md`。
6. 读取 `docs/agents/glm_dev_agent.md`。
7. 检查当时最新 `main` SHA 和目标 dev 分支 SHA，不能使用本文件记录的历史 SHA 作为新实现基线。
8. 在改代码前判断本工作属于：
   - 已关闭 Stage 的 documentation clarification / bugfix / compatibility fix；或
   - 新功能、架构扩张、职责变化、默认策略变化。
9. 后一种情况必须先登记治理 claim、取得用户明确批准、创建 authorization 和 rollback plan，再开始实现。

本文件只传递问题、经验、目标架构和验收标准，不构成 reopen authorization。

---

## 1. 问题定义

DRPO 已经固化了：

- dev agent 只在 dev 分支实现和跑实验；
- reviewer 负责研究设计、审查、结果定级和 merge；
- schema-v3 handoff delta 是生产写入权威；
- registry、handoff、materialization report 和 generated views 有严格治理约束；
- 正式实验、结果 provenance 和 terminal audit 有独立门禁。

但仓库尚缺少一条标准化、机器执行的流程，将一个长期 dev 分支上的已审查代码与实验结果，安全集成到不断前进的 `main`。

当前缺口不是普通 Git merge，而是以下组合问题：

```text
long-lived dev branch
+ current main drift
+ selective code import
+ scientific result review
+ registry classification
+ schema-v3 delta normalization
+ immutable materialization history
+ generated-view refresh
+ tests and governance gates
+ exactly one formal PR
```

目标是把它从“每个 session 临时推理和试错”变成“固定 manifest + 固定状态机 + fail-closed 工具”。

---

## 2. 真实试错案例与已确认经验

### 2.1 案例边界

2026-07-11 的 E8 集成：

- 初始 dev 分支：`dev/e8-oracle-bank-v2`
- 被审查的运行代码历史 commit：`64a2fa2d031b0cde2cb22482ce7a1842e72172b5`
- 后续 converter 修复后的 dev commit：`fe214f010bd5fec1e0e6a83f8297132a9ae8882b`
- 当时 current-main 基线：`8759c762ac478053ab392590c862b6fb14c6d713`
- ready commit：`ab30ec38dc3f50e9e4a4e403ebff2730e59fe46f`
- 最终 main merge commit：`c9699d6d93f17d28164dd7404929a5a09c0fad3e`
- 临时 PR：#18，关闭且未合并
- 正式 PR：#19，已合并
- 实验：`EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01`

这些 SHA 仅用于复盘，不得作为未来实现的固定基线。

### 2.2 暴露的问题

#### A. dev 治理文件基于旧 main

长期 dev 分支会携带旧版：

- `experiments/registry.yaml`
- `docs/handoff.md`
- `docs/handoff_deltas/**`
- `docs/handoff_shadow/**/generated/**`

如果整体 merge，可能覆盖或冲突于 main 上后来登记的实验与治理内容。

**结论：** 默认不得整体 merge 长期 dev 分支；必须从最新 main 创建 clean integration branch，只导入 allowlist 文件。

#### B. registry 条目插入错误区域

E8 集成中，新实验一度被插到 `development_experiment_registrations` 之后。YAML 仍可解析，但治理 validator 认为它不属于正式 `experiments` 集合。

**结论：** 不得依赖字符串寻找“下一个 `- id:`”来插入 registry；必须通过 YAML AST / 明确 schema path 操作，并验证 added entity 的结构归属。

#### C. fail-closed 校验顺序错误

`v2_bank_convert.py` 原先先加载 tokenizer/model stack，再检查源数据重复、空 bank、超过上限等完整性问题。无模型环境的单元测试因此在真正的数据校验之前失败。

**结论：** 任何可静态完成的完整性、scope、manifest、路径和 provenance 校验必须先于重依赖初始化、CUDA、模型加载和网络访问。

#### D. delta 与 materialization report 的不可变历史

schema-v3 authority 要求：

- `HANDOFF_DELTA.yaml` 只能首次加入一次；
- sibling `MATERIALIZATION_REPORT.json` 必须与 delta 在同一首次提交出现；
- 之后二者都不可再次修改。

E8 集成中，先提交 delta、后提交 report 会触发：

```text
materialization report is not immutable
```

最终正确做法是：

1. 先形成 source commit；
2. trusted normalization 生成 report、handoff 和 generated views；
3. 将生成结果 `git commit --amend --no-edit` 回 source commit；
4. 再执行 committed-history verify。

**结论：** normalization 必须是原子 amend 流程，不是第二个 commit。

#### E. 把 CI 当作在线脚本编辑器

试错阶段创建过：

- 临时 integration workflow；
- workflow patcher；
- failure reporter；
- artifact log exporter；
- 多个中间 PR。

虽然最终能定位问题，但这是错误的常规架构。

**结论：** 正式流程不得在运行中自修改 workflow；不得为一次集成创建“执行 PR + patch PR + 最终 PR”。正常路径只有一个 integration PR。

#### F. 日志获取路径不标准

部分 Actions 失败只能通过新增 workflow 导出日志，增加了大量迭代。

**结论：** 新工具必须原生写入结构化诊断文件和单一 artifact；所有失败在退出前给出稳定 error code、阶段、命令、stdout/stderr tail、工作树状态和可恢复步骤。

#### G. current main 在集成期间前进

固定旧 base 会造成集成脚本在最后 push 时失效，或错误地在旧 main 上生成 registry/delta。

**结论：** 建立 integration branch 时锁定 main SHA；最终 push/merge 前再次检查。main 漂移时必须重新从新 main 生成集成提交，禁止直接强推旧生成视图。

#### H. 科学结果与工程合并是两个审查层

E8 结果本身是：

- dirty-worktree pilot；
- single seed per cell；
- 不同 seed offset；
- 不同 early-stop horizon；
- 无正式方法排名。

但代码、结果摘要和 pilot 定级仍可以进入 main。

**结论：** pipeline 不能把“可归档”误等同于“正式结果成立”。必须独立处理：

1. code integration eligibility；
2. evidence classification；
3. claim support level；
4. merge readiness。

---

## 3. 固定角色边界

### 3.1 本地 JRM / GLM / Claude Code dev agent

只负责：

- 按冻结 scope 在指定 dev 分支实现；
- 跑 unit/static/liveness/experiment；
- 保存原始结果、日志、失败证据和 provenance；
- 推送 dev 分支；
- 提供精确 dev SHA、base SHA、结果路径和命令。

不得负责：

- 最终实验设计；
- 修改 claim；
- 解释最终方法排名；
- 修改最终 `docs/handoff.md`；
- 最终 registry 定级；
- current-main normalization；
- merge main；
- 处理治理冲突。

已有详细角色协议：`docs/agents/glm_dev_agent.md`。

### 3.2 Reviewer / Integrator session

负责：

- 检查 dev diff 与真实代码实现；
- 检查冻结变量漂移；
- 审查结果包和 provenance；
- 区分 smoke / pilot / formal / closure；
- 区分 task collapse、support/boundary、NaN/Inf；
- 决定导入文件 allowlist；
- 从最新 main 创建 integration branch；
- 编写 registry event、compact result summary、handoff delta；
- 执行 trusted normalization；
- 审查 gates；
- 创建唯一正式 PR；
- 决定并执行 merge。

### 3.3 Pipeline 工具

工具只负责工程与治理确定性，不负责科学判断。

工具不得自动决定：

- 某方法更优；
- pilot 可升级为 formal；
- fixed horizon 等于收敛；
- best checkpoint 可替代 terminal；
- Countdown/Hopper 可替代 C-U1/D-U1 机制识别。

---

## 4. 目标状态机

建议状态：

```text
RECEIVED
  ↓
SOURCE_LOCKED
  ↓
SCIENTIFIC_REVIEWED
  ↓
FILESET_APPROVED
  ↓
INTEGRATION_PREPARED
  ↓
DELTA_VALIDATED
  ↓
NORMALIZED_UNCOMMITTED
  ↓
ATOMIC_COMMIT_CREATED
  ↓
TARGETED_GATES_PASSED
  ↓
FULL_GATES_PASSED
  ↓
PR_READY
  ↓
MERGED
```

失败状态：

```text
BLOCKED_SOURCE_DRIFT
BLOCKED_SCOPE_VIOLATION
BLOCKED_PROVENANCE
BLOCKED_SCIENTIFIC_REVIEW
BLOCKED_REGISTRY
BLOCKED_DELTA
BLOCKED_NORMALIZATION
BLOCKED_IMMUTABILITY
BLOCKED_TARGETED_TEST
BLOCKED_FULL_TEST
BLOCKED_MAIN_DRIFT
```

每个状态都必须写入机器可读 transaction report，不能只存在于聊天消息。

---

## 5. 核心不变量

实现必须机械保证：

1. **最新 main 起点**：integration branch 从已锁定 current main 创建。
2. **不可变 dev 输入**：dev SHA 固定；流程中 dev branch 漂移不改变已审查对象。
3. **selective import**：只导入 allowlist；默认拒绝 dev 的 handoff、registry、generated views 和未知文件。
4. **无直接 handoff 编辑**：任何 handoff/registry 更新必须走单个 schema-v3 delta。
5. **原子首次提交**：delta、materialization report、materialized handoff、registry 和 refreshed generated views 在同一最终 commit 中。
6. **单一正式 PR**：一个 integration branch 对应一个正式 PR。
7. **无 workflow 自修改**：CI 不写回或改写其自身 workflow。
8. **先轻后重**：静态校验在模型/CUDA/网络初始化之前。
9. **科学状态显式**：result status、限制和 terminal-audit 状态必须登记。
10. **main freshness**：push 和 merge 前重新验证 main 未漂移；漂移则重建。
11. **历史保留**：不破坏性删除旧实验、旧结果或旧结论。
12. **失败可恢复**：失败保留诊断、阶段和输入 SHA，不产生半成品正式 PR。

---

## 6. 建议 manifest

建议新增模板：

`docs/templates/dev_integration_manifest.yaml`

示例：

```yaml
schema_version: 1
integration_id: EXT-C-E8-V2-PILOT-INTEGRATION-2026-07-11

source:
  repository: easonhuo/drpo
  main_sha: <resolved-current-main-sha>
  dev_branch: dev/e8-oracle-bank-v2
  dev_sha: <reviewed-dev-sha>
  result_commit_sha: <commit-that-produced-results>
  result_git_dirty: true

subject:
  experiment_ids:
    - EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01
  governance_claims: []
  evidence_level: pilot
  reviewer_decision_file: <path>

files:
  allowlist:
    - src/drpo/countdown_e8_oracle_offline_v2_matrix.py
    - scripts/run_countdown_e8_oracle_offline_v2_matrix.py
    - scripts/v2_bank_convert.py
    - scripts/v2_bank_smoke.py
    - scripts/v2_sft.py
    - tests/test_countdown_e8_oracle_offline_v2_matrix.py
  forbidden_from_dev:
    - docs/handoff.md
    - experiments/registry.yaml
    - docs/handoff_deltas/**
    - docs/handoff_shadow/**/generated/**

registration:
  registry_mode: add_entity
  result_summary_path: experiments/results/<id>/RESULT_SUMMARY.json
  handoff_update_id: <schema-v3-update-id>
  handoff_heading_path:
    - 0. 研究与执行原则（每次新会话首先阅读）
    - 0.1 当前执行门禁

checks:
  targeted:
    - python -m compileall -q <paths>
    - ruff check <paths>
    - pytest -q <target-tests>
  full:
    - python scripts/handoff_authority.py verify --repo-root . --json
    - python scripts/validate_governance_pipeline_stage_status.py --repo-root .
    - python -m pytest -q
    - ruff check .

merge:
  require_single_commit: true
  require_single_formal_pr: true
  merge_method: squash
  fail_on_main_drift: true
```

Manifest 中不得自动推导科学结论。`reviewer_decision_file` 应由 reviewer 明确提供。

---

## 7. 推荐 CLI

建议入口：

```bash
python scripts/integrate_dev_branch.py prepare \
  --manifest docs/integrations/<integration-id>.yaml
```

```bash
python scripts/integrate_dev_branch.py normalize \
  --transaction-dir <path>
```

```bash
python scripts/integrate_dev_branch.py gate \
  --transaction-dir <path> \
  --tier targeted
```

```bash
python scripts/integrate_dev_branch.py finalize \
  --transaction-dir <path>
```

```bash
python scripts/integrate_dev_branch.py status \
  --transaction-dir <path> \
  --json
```

不要把所有动作塞进一个不可审计的巨大命令。CLI 可以提供 `run` convenience command，但底层阶段必须独立、幂等、可恢复。

---

## 8. 推荐实现算法

### Phase A：锁定输入

1. 从远程解析 `main` SHA。
2. 解析 dev branch SHA。
3. 验证 manifest 中 SHA 与远程一致。
4. 保存 `SOURCE_LOCK.json`。
5. 验证 dev SHA 可达、结果 commit 可达。
6. 验证结果 commit 与 dev HEAD 的关系；若不相同，必须显式记录。

### Phase B：审查输入

1. 计算 `main...dev_sha` diff。
2. 与 allowlist/forbidden list 比较。
3. 生成 `SCOPE_AUDIT.json`。
4. 检查科学变量和配置漂移。
5. 检查结果包：commit、dirty、seeds、horizon、terminal audit、checksums。
6. 缺少 reviewer decision 时停止。

### Phase C：创建 integration worktree/branch

1. 从锁定 main SHA 创建 clean worktree。
2. 新建一次性 integration branch。
3. 从 dev SHA 逐文件提取 allowlist。
4. 不 cherry-pick 整个 dev 历史，除非 reviewer 明确批准。
5. 对文件模式和删除操作显式审计。

### Phase D：生成注册内容

1. 通过 YAML parser 更新 registry，不做脆弱字符串插入。
2. 写 compact result summary。
3. 写 schema-v3 delta。
4. 记录所有 evidence paths。
5. 运行 registry structural diff，确认 added/changed entities 与声明一致。

### Phase E：source commit

1. stage allowlist、registry、result summary、delta。
2. `git diff --cached --check`。
3. 创建 source commit。
4. 记录 source commit SHA。

### Phase F：trusted normalization

1. trusted repo 固定到锁定 main SHA。
2. target repo 固定到 source commit。
3. 调用 current-main trusted `handoff_authority.py normalize`。
4. 生成 handoff、materialization report、generated views。
5. 运行 no-op / history-preservation / idempotence / registry checks。
6. **将生成产物 amend 回 source commit**。
7. 记录最终 atomic commit SHA。

### Phase G：门禁

顺序固定：

1. `git diff --check`
2. compile
3. Ruff targeted
4. pytest targeted
5. handoff authority verify
6. governance stage validator
7. formal channel validator（若适用）
8. tiered/full pytest
9. full Ruff
10. worktree clean check

### Phase H：PR 与 merge

1. 再次解析 remote main SHA。
2. 与锁定 main SHA 不一致则 `BLOCKED_MAIN_DRIFT`，重建而非强推。
3. push integration branch。
4. 创建唯一正式 PR。
5. PR body 自动包含：输入 SHA、文件、experiment/claim、状态、测试、限制。
6. CI 完成后 reviewer 最终审查。
7. 使用 expected head SHA merge。
8. 验证 main 新 SHA、registry、handoff 和结果摘要可读取。
9. 写 `MERGE_CLOSURE.json`。

---

## 9. 错误分类与诊断合同

每个失败必须有稳定 error code：

| Code | 含义 | 默认恢复动作 |
|---|---|---|
| `SOURCE_DRIFT` | main/dev SHA 与锁定值不一致 | 重新锁定并重建 |
| `SCOPE_VIOLATION` | 出现未授权文件或冻结变量改动 | 返回 reviewer/dev 修复 |
| `PROVENANCE_INCOMPLETE` | 缺 commit、dirty snapshot、checksum 等 | 补证据，不继续注册 |
| `SCIENTIFIC_REVIEW_MISSING` | 缺少证据等级/结论审查 | reviewer 补 decision |
| `REGISTRY_STRUCTURE_ERROR` | entity 不在正确 schema path | 修 registry generator |
| `DELTA_VALIDATION_ERROR` | schema-v3 delta 失败 | 修 delta，不运行 normalize |
| `NORMALIZATION_ERROR` | trusted normalization 失败 | 保留 target/trusted 状态与日志 |
| `IMMUTABILITY_ERROR` | delta/report 历史不原子 | 重新生成 atomic commit |
| `TARGETED_TEST_FAILURE` | 任务相关测试失败 | 修代码/测试 |
| `FULL_TEST_FAILURE` | 全仓门禁失败 | 分类影响后修复 |
| `MAIN_DRIFT` | PR 前 main 已前进 | 自动重建 integration branch |
| `MERGE_RACE` | expected head/base 不再匹配 | 停止，重新审计 |

结构化诊断至少包含：

```json
{
  "status": "FAIL",
  "error_code": "REGISTRY_STRUCTURE_ERROR",
  "phase": "registration",
  "main_sha": "...",
  "dev_sha": "...",
  "atomic_commit_sha": null,
  "command": ["..."],
  "returncode": 1,
  "stdout_tail": "...",
  "stderr_tail": "...",
  "changed_paths": ["..."],
  "recovery": ["..."]
}
```

禁止失败后只输出一句“CI failed”。

---

## 10. 建议新增文件

下一 session 应先设计，再决定最终路径。推荐候选：

```text
docs/dev_branch_integration_protocol.md
docs/templates/dev_integration_manifest.yaml
docs/integrations/README.md
scripts/integrate_dev_branch.py
scripts/validate_dev_integration.py
tests/test_dev_branch_integration.py
.github/workflows/dev-integration-gate.yml
```

还应考虑是否补充：

```text
docs/agents/reviewer_integrator.md
```

不要立即修改 `AGENTS.md` 或默认策略。先完成工具、测试和真实 shadow 观察，再决定是否将短版路由规则加入 `AGENTS.md`。

---

## 11. 测试计划

### 11.1 单元测试

至少覆盖：

- allowlist 导入；
- forbidden file 拒绝；
- 文件模式保持；
- registry 正确 insertion path；
- registry fake add_entity 拒绝；
- duplicate experiment ID 拒绝；
- main/dev SHA drift；
- dirty result provenance；
- missing reviewer decision；
- delta/report 同 commit；
- report 第二次修改拒绝；
- generated views 不从 dev 复制；
- source commit amend 后 verify；
- no-op normalization；
- failure artifact 完整性。

### 11.2 集成测试

构造本地 fixture：

1. main 与 dev 从共同基点分叉；
2. main 新增 registry entity；
3. dev 新增科学代码并携带旧 registry；
4. integration 只导入代码；
5. registry 同时保留 main 新 entity 并加入 dev experiment；
6. normalization 生成原子 commit；
7. authority verify PASS。

### 11.3 故障注入

至少注入：

- registry 插入 `development_experiment_registrations` 之后；
- delta 与 report 分 commit；
- dev SHA 在审查后移动；
- main 在 finalize 前移动；
- result commit 与执行 commit 不一致；
- `git_dirty=true` 却声明 formal；
- terminal audit 缺失却声明 convergence；
- workflow 尝试修改自身；
- 第二个正式 PR 被创建。

### 11.4 真实 shadow 观察

在启用默认路径前，至少选择两个真实但低风险的 dev 分支完成 shadow：

- 一个 code-only 集成；
- 一个 code + registry + handoff delta + pilot result 集成。

shadow 期间工具只生成候选和报告，不自动 merge。

---

## 12. 分阶段实施建议

### Stage A：文档和 dry-run

- 注册治理 claim；
- 冻结 scope 和 rollback；
- 定义 manifest、状态机和错误码；
- 实现 read-only plan/diff/audit；
- 不写 GitHub，不改默认流程。

### Stage B：本地 integration worktree

- 实现 selective import；
- 实现 registry AST 更新；
- 实现 source commit + normalize + amend；
- 只生成本地 ready commit；
- 不自动开 PR。

### Stage C：shadow PR

- 自动 push ready branch；
- 自动生成 PR body；
- 仍需 reviewer 手工开/合并；
- 记录耗时、拦截率和误报。

### Stage D：受控默认化

只有在：

- 两个以上真实 shadow PASS；
- 无治理回归；
- 失败诊断可用；
- rollback rehearsal PASS；
- 用户再次明确批准；

之后，才考虑加入 `AGENTS.md` 的短版默认路由或 CI blocking gate。

---

## 13. 明确非目标

本 pipeline 不负责：

- 自动设计实验；
- 自动选择最佳方法；
- 自动升级结果状态；
- 自动生成论文结论；
- 自动合并未审查的 dev 分支；
- 取代 formal experiment guard/package/verify；
- 取代 schema-v3 handoff authority；
- 修改 Stage 1 update-package pipeline 的既有职责；
- 绕过 reviewer。

它是“reviewed dev evidence 到 current main 的安全集成层”，不是新的研究 agent。

---

## 14. 验收标准

第一版至少满足：

1. 从 current main 和固定 dev SHA 生成一个 clean integration commit。
2. dev 的旧 handoff/registry/generated views 不会进入候选。
3. registry 新 entity 位于正确 schema 集合。
4. delta 与 materialization report 在同一首次 commit。
5. committed authority verify PASS。
6. targeted tests 和治理 validator PASS。
7. main 漂移时 fail closed。
8. 失败产生结构化诊断。
9. 全流程只创建一个正式 PR。
10. 不依赖修改 CI workflow 来推进 transaction。
11. 不改变任何科学变量或结果状态。
12. shadow 实例总耗时显著低于 2026-07-11 的约四小时试错流程；建议目标为：
    - 人工处理 20--40 分钟；
    - 正常端到端 40--75 分钟；
    - 有真实冲突时不超过 90 分钟。

耗时只是观测指标，不能为了提速跳过门禁。

---

## 15. 下一 session 的第一批具体任务

1. 重新读取当时最新 main 上所有权威文件。
2. 提出 claim、scope、allowed files、forbidden files、rollback。
3. 判断是否需要 reopen Stage 1，或应作为新的独立 integration stage/工具。
4. 先产出 manifest schema 与 transaction state schema。
5. 先写 failure taxonomy 和 tests，再写 Git 操作。
6. 复用 `scripts/handoff_authority.py`，不要复制其 normalization 逻辑。
7. 复用现有 test-selection 与 governance validator，不建立第二套全仓规则。
8. 用本地 fixture 重放 E8 失败案例：
   - 错误 registry 区域；
   - report 非原子；
   - main drift；
   - old generated views；
   - heavy dependency before validation。
9. 完成 read-only dry-run 后，再申请进入写入阶段。
10. 所有实现与真实 shadow 结果通过 dev branch + reviewer gate 合并。

---

## 16. 参考文件

- `AGENTS.md`
- `docs/handoff.md`
- `experiments/registry.yaml`
- `docs/governance_pipeline_stage_status.yaml`
- `docs/governance_pipeline_refactor_plan.md`
- `docs/agents/glm_dev_agent.md`
- `scripts/handoff_authority.py`
- `scripts/handoff_delta_shadow.py`
- `tests/stage5_candidate_integration.py`
- `scripts/select_update_tests.py`
- `.github/workflows/pr-gate-log.yml`
- E8 集成 PR #18 与 #19 的历史记录

---

## 17. 最终交接结论

2026-07-11 的 E8 集成已经证明：现有底层治理能力足以完成安全集成，但缺少一个将这些能力按正确顺序组合起来的标准 transaction layer。

下一 session 不应重新发明科学审查、handoff authority、formal artifact 或全仓测试系统。它应只构建一个薄而严格的 orchestration layer：

```text
immutable reviewed inputs
→ selective import on current main
→ explicit scientific classification
→ schema-v3 registration
→ trusted normalization
→ atomic amend
→ deterministic gates
→ one PR
→ verified merge closure
```

成功标准不是“自动化更多”，而是让正确路径更短、错误路径更早失败、所有失败都可诊断、所有科学判断仍由 reviewer 控制。
