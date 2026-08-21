# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 Runbook V2

## 0. 身份、状态与使用边界

实验 ID：`EXT-C-E8-MULTITASK-EXP-COLDSTART-01`

候选 RunSpec ID：`E8_MULTITASK_EXP_COLDSTART_20260820_02`

冻结科学实现：`031239031b721314e565da32bd254b46ed28f4db`

该 freeze 是从当前 `main@ea2c6d60084acd3e87d6f4118161747364bfe9b4` 重新构造的干净实现提交；4 个科研/执行文件的 Git blob 与此前审查通过的 `4702c2e97af8df0f26a9e1ceb5a45084cdc1235d` 完全一致，重建只移除了旧 registration/handoff 历史对 V1 scope audit 的污染，没有改科研实现字节。

当前科学状态：`pilot / not_run`

本 Runbook 只描述当前候选执行协议。它本身不激活实验，也不构成科学结果。只有当 schema-v3 registration 已刷新、registry 指向上述 RunSpec、该实验在 `runspecs/ready/` 下恰好存在一个匹配的 READY RunSpec、来源与 provenance 门禁全部通过之后，才允许进入执行。

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

服务器操作员/本地 AI 不应把 bootstrap 脚本当作顶层正式执行命令直接运行。顶层执行必须走 RunSpec lane executor：

```bash
python scripts/agent/run_lane.py --once
```

当且仅当 `E8_MULTITASK_EXP_COLDSTART_20260820_02` 已经通过注册并成为本实验唯一 READY RunSpec 时，lane executor 才可 claim 它。该 RunSpec 内部仍调用已经审查的：

```bash
bash scripts/bootstrap_e8_multitask_exp_coldstart.sh full
```

bootstrap / runner 是受 RunSpec provenance 约束的下层入口，不得绕过 RunSpec claim、package、delivery 和 source-identity gate 独立启动。

## 4. 启动前 fail-closed 检查

在创建训练环境、下载模型或接触 GPU 前必须全部满足：

1. `experiments/registry.yaml` 中恰好一个 `EXT-C-E8-MULTITASK-EXP-COLDSTART-01`；
2. registry 仍为 `execution_class: pilot`、`result_status: not_run`、`implementation_state: implemented`；
3. registry 的 `implementation_commit` 精确等于 `031239031b721314e565da32bd254b46ed28f4db`；
4. registry 的 `runspec_id` 精确等于 `E8_MULTITASK_EXP_COLDSTART_20260820_02`；
5. `runspecs/ready/` 对本实验恰好只有这一个 READY RunSpec；
6. RunSpec `repo_commit` 精确等于上述冻结实现，并且它必须是执行 HEAD 的祖先；
7. RunSpec 中全部 protected paths 自该实现 freeze 后保持不变；
8. checkout 干净，origin 为 `easonhuo/drpo`；
9. `full` bootstrap 不允许把当前 source checkout 静默切换到另一 commit；source identity 不一致必须停止；
10. handoff/registry schema-v3 authority、formal execution channel 和 repository governance gates 全部通过。

任一条件不满足都必须停止，不允许在服务器现场“修一下再跑”。

## 5. 恢复、结果与解释

运行中断时只允许使用已有 guard/recovery 协议恢复已经身份校验通过的完成 cell；部分训练 cell 不视为完成 cell。不得通过恢复流程改变科研参数或 RunSpec。

每个已完成 task 可以先读取其 task-local snapshot 做过程分析，但它不替代最终 208-cell 汇总。任何 smoke test、self-test、liveness、静态检查或有限工程 pilot 都不能被表述为正式科学结果。

本实验只测试冻结九任务 cold-start 协议下，保留 July-29 source error-direction sequence 且在类内覆盖 frozen-reference surprisal 后，transfer-task Exp response curve 相对 Positive-only 如何变化。**不预设 improvement、convergence 或 universal best coefficient。**
