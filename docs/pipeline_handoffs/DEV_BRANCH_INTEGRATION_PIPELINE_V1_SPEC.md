# DRPO Dev Branch → Main 集成 Pipeline V1 实施合同

**治理 claim：** `GOV-DEV-BRANCH-INTEGRATION-01`  
**实施阶段：** `v1_local_transaction`  
**基线：** `d94eb5d7231653f557e66c6ae0b1cc4fa008ef27`  
**批准记录：** `user_approved_2026_07_11_finalize_lightweight_v1_documents_then_begin_development`  
**文档性质：** 当前实现合同。它不替代 `docs/handoff.md`，不改变任何科学实验状态，也不修改已关闭治理 Stage 的职责。

## 1. 与原 build brief 的关系

`DEV_BRANCH_INTEGRATION_PIPELINE_BUILD_BRIEF.md` 保留为长期问题复盘和完整能力蓝图。本文件收敛第一版真正允许开发和验收的范围。

发生冲突时：

1. `AGENTS.md`、`docs/handoff.md`、`experiments/registry.yaml`、`docs/governance_pipeline_stage_status.yaml` 仍是上位权威；
2. 对 V1 的实现范围、非目标、状态机和验收，以本文件为准；
3. 原 build brief 中的自动 PR、自动 merge、完整 PR 生命周期和并发调度均视为未来路线，不是 V1 要求。

## 2. 架构判断

V1 采用“小型正式 transaction framework”，不是临时复制脚本，也不是重型平台。

核心原则是：

> 轻框架，重不变量；少造新系统，多复用现有系统。

V1 不引入数据库、后台服务、Web UI、工作流语言或新的科学判断引擎。它只负责把 reviewer 已批准的输入，按固定顺序交给现有 Git、handoff authority、registry 校验和测试选择能力。

## 3. V1 目标

V1 将一个固定的 reviewed dev commit，安全转换为基于 current main 的本地 ready commit，并生成可审计 transaction 记录。

正常路径：

```text
reviewed request
→ immutable source lock
→ scope audit
→ clean current-main worktree
→ selective import
→ reviewer-provided registration inputs
→ trusted normalization
→ atomic amend
→ required gates
→ local ready commit
```

V1 到 `READY` 为止，不自动 push、不自动开 PR、不自动 merge。

## 4. V1 明确非目标

V1 不做：

- 自动设计实验或修改 claim；
- 自动把 smoke/pilot 升级为 formal；
- 自动判断方法排名或收敛；
- 自动生成论文结论；
- 自动 push、开 PR、等待 CI、刷新 PR 或 merge；
- 新建或自修改 GitHub Actions workflow；
- 多仓库、多租户、任务队列或并发调度平台；
- 数据库、服务端守护进程或 Web UI；
- 可在 manifest 中任意执行 shell 的工作流语言；
- 新写一套 handoff renderer、registry authority、formal artifact channel 或全仓测试规则；
- 修改 `AGENTS.md`、`docs/handoff.md`、`experiments/registry.yaml`、现有 handoff authority 核心或任何 Stage 1/2/5 受保护文件，除非后续单独授权。

## 5. 核心与 adapter 边界

### 5.1 核心只负责

- request 和 reviewer decision 的 schema 校验；
- source/main/dev/result commit 的锁定与一致性检查；
- scope、路径、文件操作和 provenance 审计；
- transaction 状态推进；
- 幂等性、attempt 隔离和结构化诊断；
- 调用既有 adapter；
- 记录命令、返回码、摘要、SHA 和产物路径。

### 5.2 adapter 负责

- Git：ref 解析、worktree、blob 提取、diff、commit、amend；
- registry：通过 YAML parser 在明确 schema path 上修改；
- handoff：调用 `scripts/handoff_authority.py`，不复制 normalization 逻辑；
- gates：调用 `scripts/select_update_tests.py`、治理 validator 和适用的 formal-channel validator；
- reviewer input：读取人工批准的 evidence classification，不自行推导科学结论。

## 6. V1 状态机

成功状态只保留：

```text
RECEIVED
→ SOURCE_LOCKED
→ REVIEWED
→ PREPARED
→ NORMALIZED
→ REQUIRED_GATES_PASSED
→ READY
```

通用非成功状态：

```text
BLOCKED
ABORTED
STALE
```

详细原因通过稳定 `error_code` 表达，不为每个错误膨胀一个永久状态。

`FULL_GATES_PASSED` 不作为固定状态。不同改动运行的 gate 集合由现有 test selector 和治理规则决定，统一登记为 `REQUIRED_GATES_PASSED`。

## 7. 输入与机器产物分离

### 7.1 人工输入

`INTEGRATION_REQUEST.yaml`：

- integration ID；
- repository；
- expected main ref；
- dev branch 和 reviewed dev SHA；
- result-producing SHA 与 dirty/provenance 信息；
- experiment IDs / governance claims；
- 文件操作 allowlist；
- reviewer decision 路径；
- registration intent；
-期望 gate tier。

`REVIEW_DECISION.yaml`：

- code integration eligibility；
- evidence level；
- claim support level；
- result status；
- terminal audit 状态；
- task-performance collapse、support/boundary、NaN/Inf 分类；
-允许导入的文件和注册内容；
-限制与未解决问题；
- reviewer identity/token 和内容哈希。

### 7.2 机器生成

- `SOURCE_LOCK.json`；
- `SCOPE_AUDIT.json`；
- `TRANSACTION.json`；
- `GATE_REPORT.json`；
- `DIAGNOSTIC.json`；
- `READY_COMMIT.json`。

机器产物不得由用户输入覆盖。`SOURCE_LOCK.json` 一旦生成即不可原地改写；输入变化必须创建新的 attempt。

## 8. 文件操作合同

allowlist 必须描述精确操作，而不仅是路径前缀：

- `add` / `modify` / `delete` / `rename`；
- source path；
- destination path；
- expected blob SHA；
- expected mode；
- 可选 expected old blob SHA。

系统级永久拒绝：

- 未经单独授权的 `docs/handoff.md`、`experiments/registry.yaml`、`docs/handoff_deltas/**` 和 generated views；
- symlink、submodule/gitlink、路径穿越、绝对路径；
- repository root escape；
- case-fold collision；
- 同一目标路径的重复操作；
- 未声明 rename/delete；
- 未批准的 executable-bit 变化；
- manifest 与实际 blob/mode 不一致。

Git LFS pointer 若出现，必须显式登记，V1 不自动下载大文件。

## 9. source freshness 与 attempt 语义

至少执行三次 main freshness 检查：

1. plan/source-lock 时；
2. 创建 integration worktree 前；
3. finalize ready commit 前。

V1 不负责 PR 阶段，因此 push/merge freshness 属于未来 publish adapter。

main 或 dev ref 漂移时：

- 当前 attempt 标为 `STALE`；
- 保留全部报告；
- 从新的 authoritative ref 创建新 attempt；
- 不在旧 attempt 中偷偷替换 SHA；
- 不复用旧 normalization 产物。

## 10. CLI 合同

建议入口：

```bash
python3 scripts/integrate_dev_branch.py plan \
  --request <INTEGRATION_REQUEST.yaml> \
  --transaction-root <dir>
```

```bash
python3 scripts/integrate_dev_branch.py prepare \
  --transaction-dir <attempt-dir>
```

```bash
python3 scripts/integrate_dev_branch.py normalize \
  --transaction-dir <attempt-dir>
```

```bash
python3 scripts/integrate_dev_branch.py gate \
  --transaction-dir <attempt-dir>
```

```bash
python3 scripts/integrate_dev_branch.py finalize \
  --transaction-dir <attempt-dir>
```

```bash
python3 scripts/integrate_dev_branch.py status \
  --transaction-dir <attempt-dir> \
  --json
```

底层阶段必须独立、可检查、幂等。未来可增加 `run` convenience command，但它只能顺序调用同一批公开阶段。

## 11. Gate 策略

V1 不维护第二份测试影响图。

顺序：

1. request/schema/path/static preflight；
2. `git diff --check`；
3. compile；
4. targeted Ruff/pytest；
5. `handoff_authority.py verify`；
6. governance stage validator；
7. formal-channel validator（适用时）；
8. `select_update_tests.py` 选择的 required gates；
9. worktree clean 和 HEAD/provenance 复核。

每个 gate 记录：

- selector 原因；
-命令和工作目录；
-开始/结束时间；
-返回码；
- stdout/stderr tail；
-是否首次拦截该 transaction；
-失败归类和恢复建议。

## 12. 错误合同

V1 至少提供：

- `REQUEST_INVALID`；
- `REVIEW_DECISION_INVALID`；
- `SOURCE_UNRESOLVED`；
- `SOURCE_DRIFT`；
- `PROVENANCE_INCOMPLETE`；
- `SCOPE_VIOLATION`；
- `UNSAFE_PATH`；
- `BLOB_OR_MODE_MISMATCH`；
- `REGISTRY_STRUCTURE_ERROR`；
- `DELTA_VALIDATION_ERROR`；
- `NORMALIZATION_ERROR`；
- `IMMUTABILITY_ERROR`；
- `GATE_FAILURE`；
- `WORKTREE_DIRTY`；
- `HEAD_DRIFT`；
- `INTERNAL_ERROR`。

所有失败必须生成 `DIAGNOSTIC.json`，不能只打印一句错误。

## 13. V1 实施范围

### Batch 1：read-only core

- request/reviewer-decision schema；
- `plan`、`status`；
- remote/local ref 解析；
- immutable source lock；
- diff/scope/path/provenance audit；
- transaction/diagnostic 模型；
-本地 bare-repo fixtures 和 fault injection。

### Batch 2：local write path

- clean worktree；
- selective import；
- blob/mode/delete/rename 审计；
- registry AST 更新；
- source commit；
- trusted normalize + atomic amend；
- required gates；
- local ready commit。

### Batch 3：shadow hardening

- 一个 code-only shadow；
- 一个 code + registry + delta + pilot-summary shadow；
- rollback rehearsal；
-拦截率、误报、耗时和独特 blocker 统计；
-修复真实 shadow 暴露的问题。

自动 push/PR/merge 不属于这三个 batch。

## 14. V1 验收

V1 只有同时满足以下条件才可称为完成：

1. current main、dev SHA、result SHA 被明确锁定；
2. 未批准文件、路径和文件模式变化 fail closed；
3. dev 的旧 handoff/registry/generated views 不会被导入；
4. registry 修改位于正确 schema path；
5. delta/report/materialized outputs 以原子 amend 形成首次提交；
6. committed authority verify 通过；
7. required gates 通过；
8. main/dev 漂移生成新 attempt，不篡改旧记录；
9.失败总有结构化诊断；
10.两个真实 shadow 完成且无科学状态误升级；
11. rollback rehearsal 通过；
12. V1 未修改默认 GitHub merge 流程或任何已关闭 Stage 的受保护职责。

## 15. 回滚

在 V1 成为默认路径之前：

- 删除本地 transaction/worktree 即可停止，不改变 main；
- 已提交的工具 PR 通过 revert 回滚；
-保留 scope、authorization、测试报告和 shadow 诊断作为历史；
-不删除任何实验结果、handoff delta、materialization report 或治理 closure 记录；
-失败 transaction 只关闭，不改写其历史状态。

## 16. 未来扩展门槛

以下能力必须由真实痛点触发，并单独设计、授权和验收：

- publish/push adapter；
-单一 PR 创建与刷新；
- GitHub CI 轮询；
- main drift 后同一 PR head 重建；
- merge closure；
-并发 transaction locking；
- gate cache；
-多仓库抽象。

V1 shadow 完成前，不把该工具写入 `AGENTS.md` 默认路由，也不新增 blocking workflow。

## 17. 当前未确定事项

- GitHub connector、原生 Git 和本地训练服务器之间的 authoritative ref 获取优先级需要在 Batch 1 用 fixture 和真实仓库各验证一次；
- registry AST 的最小 mutation API 应在不复制 handoff authority 职责的前提下确定；
- code + delta shadow 的运行时间取决于现有全仓 gate，不能在文档阶段承诺固定分钟数；
- publish/PR/merge 是否值得自动化，只能由 V1 shadow 的人工耗时和失败分布决定。
