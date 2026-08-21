# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 Runbook V2

## 0. 身份、状态与使用边界

实验 ID：`EXT-C-E8-MULTITASK-EXP-COLDSTART-01`

Run ID：`E8_MULTITASK_EXP_COLDSTART_20260820_02`

冻结科学/执行实现：`452e9fefdca94e25d4cae2422ea2e0ada8ec01fe`

授权执行分支：`dev/e8-multitask-exp-coldstart-01`，对应远端 ref 为 `refs/heads/dev/e8-multitask-exp-coldstart-01`。

当前科学状态：`pilot / not_run`。本轮是 branch pilot，`formal_evidence_allowed: false`；不得把 smoke、liveness、工程自检或本次 pilot 自动升级为 formal result。

本 pilot 的启动链已明确取消以下三个前置治理依赖：

1. READY RunSpec activation；
2. registry execution-gate / implementation-identity activation；
3. 为满足前两项而执行的 Stage-5 schema-v3 registration transaction。

这些历史文件继续保留作 provenance，不再是本 pilot 的启动许可证，也不得因此修改或删除历史 handoff / registry 记录。真正保留的执行锁是：精确远端 branch commit、clean checkout、冻结科研配置/代码、GPU/模型/数据 preflight、foreground guard、recovery、terminal audit 与结果 provenance。

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

## 3. 唯一顶层执行入口

服务器先确保已有 `drpo` checkout 位于授权分支且与远端同步：

```bash
git fetch origin dev/e8-multitask-exp-coldstart-01
git checkout dev/e8-multitask-exp-coldstart-01
git merge --ff-only origin/dev/e8-multitask-exp-coldstart-01
```

然后只运行：

```bash
E8_COLDSTART_RUN_ID=E8_MULTITASK_EXP_COLDSTART_20260820_02 \
E8_COLDSTART_TARGET_REF=refs/heads/dev/e8-multitask-exp-coldstart-01 \
  bash scripts/bootstrap_e8_multitask_exp_coldstart.sh full
```

bootstrap 的 `full` 通过上面的显式 `E8_COLDSTART_TARGET_REF` 锁定授权分支；runner 默认身份已经改为 `pilot`，且默认 `E8_COLDSTART_REQUIRE_ORIGIN_MAIN=0`。不再通过 `scripts/agent/run_lane.py`、READY RunSpec、registry activation 或 Stage-5 registration 才能进入训练。

`full` 仍然要求 selected source HEAD 与远端授权分支 HEAD 完全一致，然后创建/复用隔离 checkout；因此取消治理许可证不等于允许跑任意本地代码。

## 4. 启动前仍保留的硬检查

在真正训练前必须满足：

1. 当前 source commit 是远端 `dev/e8-multitask-exp-coldstart-01` 的精确 HEAD；
2. checkout 完全干净，origin 为 `easonhuo/drpo`；
3. experiment/run identity 分别为 `EXT-C-E8-MULTITASK-EXP-COLDSTART-01` 与 `E8_MULTITASK_EXP_COLDSTART_20260820_02`；
4. runner 以 `pilot` 执行，且不要求 origin/main match；
5. 8 张 CUDA GPU 可见且满足显存要求，运行盘满足空间要求；
6. 固定 Qwen revision 可取得，runtime/依赖检查通过；
7. frozen scientific config 与 canonical cold-start source audit 通过；
8. prepare/qualification、liveness、shared dynamic queue、recovery、terminal aggregate/audit 和 durable package 路径保持原样。

registry、READY RunSpec 和 schema-v3 registration 的状态不再参与上述启动判定。它们可以陈旧，但不得被错误解释为当前 pilot 的科学身份来源。

## 5. 恢复、结果与解释

运行中断时只允许使用已有 guard/recovery 协议恢复已经身份校验通过的完成 cell；部分训练 cell 不视为完成 cell。branch pilot 的 recovery checkpoint 与 delivery preflight 沿用同一个 `pilot / no-origin-main-match` 身份，不得在恢复时切回 main 或另一 commit。

每个已完成 task 可以先读取其 task-local snapshot 做过程分析，但它不替代最终 208-cell 汇总。任何 smoke test、self-test、liveness、静态检查或有限工程 pilot 都不能被表述为正式科学结果。

本实验只测试冻结九任务 cold-start 协议下，保留 July-29 source error-direction sequence 且在类内覆盖 frozen-reference surprisal 后，transfer-task Exp response curve 相对 Positive-only 如何变化。**不预设 improvement、convergence 或 universal best coefficient。**
