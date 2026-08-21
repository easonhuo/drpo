# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 Runbook V2

## 0. 身份、状态与使用边界

实验 ID：`EXT-C-E8-MULTITASK-EXP-COLDSTART-01`

RunSpec ID：`E8_MULTITASK_EXP_COLDSTART_20260820_02`

冻结科学/执行实现：`452e9fefdca94e25d4cae2422ea2e0ada8ec01fe`

授权执行分支：`dev/e8-multitask-exp-coldstart-01`，对应远端 ref 为 `refs/heads/dev/e8-multitask-exp-coldstart-01`。

当前科学状态：`pilot / not_run`。本轮分支执行属于 pilot，`formal_evidence_allowed: false`；不得把 smoke、liveness、有限步工程检查或本次 pilot 自动升级为 formal result。

本 Runbook 只描述当前执行协议。只有 schema-v3 registration 已刷新、registry 指向上述 implementation freeze 与 RunSpec、且本实验在 `runspecs/ready/` 下恰好存在一个匹配 READY RunSpec 后，才允许启动。

旧版 Runbook 已封存为：
`docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK_20260818_SUPERSEDED.md`。

## 1. 冻结科学协议

不得在服务器现场修改以下科研身份：

- 9 个任务：Countdown + Word Sorting、Spiral Matrix、Mini Sudoku、Maze、Word Ladder、Knights & Knaves、Graph Coloring、WikiSQL；
- 总计 208 cells；
- 固定 1200 optimizer updates，无 early stopping；
- fresh LoRA / zero-update initialization 语义不变，无 SFT warm start；
- Countdown 继续只承担回归/外部有效性 sentinel，不以随机性能数值阻断 transfer tasks；
- transfer-task negative bank 继续来自 July-29 P0 deterministic verified-wrong candidate universe；
- 每个 prompt 选择 16 个负例，严格保留 source P0 error-class sequence，并在每个 error class 内按 frozen-reference mean-token surprisal/rank 拉开 near-to-far 覆盖；
- reference rank/surprisal 只用于 bank provenance 与诊断，不进入训练权重；Exp taper 仍在每次 update 按当前 policy surprisal 重新计算；
- Countdown 保持 256/80、evaluation batch 8、Greedy 500、Pass@8 500，并保留 Pass@64 辅助诊断；8 个 transfer tasks 保持 512/128、evaluation batch 16、Greedy 500、Pass@8 128，不运行隐藏 Pass@64；
- LR、optimizer、warmup、max-grad-norm、LoRA r/alpha/dropout、数据 split、task prompt/verifier、采样 temperature/top-p、Exp 公式、task-local coefficient grids 与全部冻结 seeds 不变；
- 不新增 remoteness 数值阈值，不新增“峰值必须超过 Positive-only”等结果门禁；
- task-performance degradation、support/structure boundary event 与 NaN/Inf numerical failure 分开报告；
- terminal aggregate 与 terminal audit 完成前，不做稳态、崩溃或方法排名结论。

## 2. 当前调度语义

调度是 **shared dynamic slot queue**，不是 hard-wave scheduler：

- 8 GPUs × 2 persistent slots/GPU = **16 个固定并发 slot**；
- 208 个冻结 cells 进入同一个 pending queue；
- 任意 slot 完成一个 cell 后立即领取下一个 pending cell，不等待其他 slot；
- **不存在 16-cell hard wave barrier**；
- `13 × 16` 只保留为 nominal audit/recovery geometry，不是调度边界，也不是科研门禁；
- OOM 或 cell failure 必须保留证据并 fail closed；禁止自动修改 batch、coefficient、loss、数据或其他科研参数；
- 某个 task 的全部冻结 cells 一旦完成，可以立即发布该 task 的确定性 task-local snapshot，不需要等待 nominal batch 或全部 208 cells；
- 最终权威仍是 208-cell terminal aggregate + terminal audit。

## 3. 唯一允许的顶层执行入口

服务器操作员/本地 AI 不直接运行 bootstrap。顶层必须走 RunSpec lane executor，并显式绑定 lane 与 RunSpec ID：

```bash
python scripts/agent/run_lane.py --lane e8 --run-id E8_MULTITASK_EXP_COLDSTART_20260820_02 --once
```

READY RunSpec 内部显式设置：

```text
E8_COLDSTART_TARGET_REF=refs/heads/dev/e8-multitask-exp-coldstart-01
E8_COLDSTART_RUN_CLASS=pilot
E8_COLDSTART_REQUIRE_ORIGIN_MAIN=0
```

然后调用：

```bash
bash scripts/bootstrap_e8_multitask_exp_coldstart.sh full
```

这里的 `full` 表示执行完整 cold-start workload，不再等价于“必须选择 main”。bootstrap 会 fetch RunSpec 指定的 authoritative branch ref，并要求 **selected source HEAD == remote target-ref HEAD**；不一致时在创建隔离 worktree 和接触训练前 fail closed。

默认未显式覆盖时仍是 `refs/heads/main`；runner 默认仍是 `formal` 且强制 origin/main match。只有本 RunSpec 明确声明的 branch pilot 使用 `pilot + no-main-match`，因此没有放宽默认 formal 路径。

## 4. 启动前 fail-closed 检查

在创建训练环境、下载模型或接触 GPU 前必须全部满足：

1. `experiments/registry.yaml` 中恰好一个 `EXT-C-E8-MULTITASK-EXP-COLDSTART-01`；
2. registry 为 `execution_class: pilot`、`result_status: not_run`、`implementation_state: implemented`；
3. registry 的 `implementation_commit` 精确等于 `452e9fefdca94e25d4cae2422ea2e0ada8ec01fe`；
4. registry 的 `runspec_id` 精确等于 `E8_MULTITASK_EXP_COLDSTART_20260820_02`；
5. `runspecs/ready/` 对本实验恰好只有该 READY RunSpec；
6. RunSpec `repo_commit` 精确等于上述 freeze，并且它是执行 HEAD 的祖先；
7. RunSpec 中全部 protected paths 自该 freeze 后保持不变；
8. 当前 checkout 干净，origin 为 `easonhuo/drpo`；
9. RunSpec target ref 精确为 `refs/heads/dev/e8-multitask-exp-coldstart-01`，bootstrap 实际解析出的远端 commit 与当前 selected source HEAD 完全一致；
10. branch pilot 使用 `E8_COLDSTART_RUN_CLASS=pilot` 与 `E8_COLDSTART_REQUIRE_ORIGIN_MAIN=0`；任何 `formal` 执行若试图关闭 origin/main match 必须被 runner 拒绝；
11. handoff/registry schema-v3 authority、formal execution channel 和 repository governance gates 全部通过。

任一条件不满足都必须停止，不允许在服务器现场修改科研参数、RunSpec 或 source identity 后继续。

## 5. 恢复、结果与解释

运行中断时只允许使用已有 guard/recovery 协议恢复已经身份校验通过的完成 cell；部分训练 cell 不视为完成 cell。branch pilot 的 recovery checkpoint 与 delivery preflight 沿用同一个 `pilot / no-origin-main-match` 身份，不得在恢复时切回 main 或另一 commit。

每个已完成 task 可以先读取其 task-local snapshot 做过程分析，但它不替代最终 208-cell 汇总。任何 smoke test、self-test、liveness、静态检查或有限工程 pilot 都不能被表述为正式科学结果。

本实验只测试冻结九任务 cold-start 协议下，保留 July-29 source error-direction sequence 且在类内覆盖 frozen-reference surprisal 后，transfer-task Exp response curve 相对 Positive-only 如何变化。**不预设 improvement、convergence 或 universal best coefficient。**
