# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 运行交接

## 1. 结论与边界

这次八任务实验不使用 SFT warm start，也不加载任何外部 reference adapter。
Positive-only 与 Exp 都从同一个 Qwen 预训练 base 加同分布的 fresh LoRA 开始。

新代码不实现第二份 DRPO loss。它只负责：

1. 把八个任务的数据转换为老 Countdown cold-start core 已有的输入格式；
2. 把老 evaluator 替换为对应任务的 verifier；
3. 组织 160 个 cell 的动态 GPU 队列；
4. 汇总完整结果和画图用的小 CSV。

任务为：Countdown、Word Sorting、Mini Sudoku、Maze、Word Ladder、Knights and
Knaves、Graph Coloring、WikiSQL。Spiral Matrix 因旧实验全网格均为 100% 而排除。

每个任务运行 1 个 Positive-only 和 19 个 Exp lambda，共
`8 × (1 + 19) = 160` 个 cell。所谓“十波”仅表示 `160 / 16 = 10` 个名义批次，
不是同步屏障。

## 2. 老代码同源审计

runner 在 prepare、calibrate 和每个训练子进程开始时校验以下 Git blob SHA；任一
文件不同就 fail closed：

| 责任 | 原始文件 | Git blob SHA |
|---|---|---|
| 公共 arena、fresh LoRA、Positive-only 内核 | `src/drpo/countdown_qwen_arena_onefile.py` | `d8a04f3ae3edd08042aa1004b4cbf927fc5cea72` |
| 老 base cold-start 包装 | `src/drpo/countdown_e8_base_rl_replay.py` | `951f81a909fa73cb5af6a187b07691fd67921bd0` |
| 老 taper loss、optimizer、scheduler、early stop | `src/drpo/countdown_e8_oracle_offline_v2_taper_sweep.py` | `29825a4c5f496aff42470438efc1f7851993786e` |
| 老 hardened taper runtime | `src/drpo/countdown_e8_oracle_offline_v2_taper_runtime.py` | `88a51c8ad39764d71682118f9fd04db63d890e5f` |
| 老 base 配置 | `configs/countdown_e8_base_rl_replay_0p5b.yaml` | `10f27f32719298376bdc7be7e01023626c6ad3f8` |
| 老 taper 配置模板 | `configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml` | `bb941f3eb6fc08bc4ed82394e99f9f889e2249e7` |

正式调用关系：

- Positive-only：`countdown_e8_base_rl_replay.train_offline_method(..., method="positive_only")`；
- Exp：`countdown_e8_oracle_offline_v2_taper_runtime.worker`，其内部调用老
  `taper_sweep.train_cell`；
- 两条路径最终都调用老 `arena.load_model(..., adapter_path=None,
  trainable_adapter=True)`，因此没有 SFT/reference warm start；
- 八任务层只 monkeypatch `arena.evaluate_rows` 和 posthoc evaluator，训练公式不变。

配置还保留老 cold-start 的主要协议：LoRA `r=32, alpha=64, dropout=0.05`，
1200 上限、400 最小步数、patience 6、delta 0.002、micro-batch 1、gradient
accumulation 8、学习率 `5e-5`、cosine warmup 0.03、gradient clip 1.0、每 100
步验证、500 条 validation、Pass@8/Pass@64、256/16 条 taper 校准。

## 3. 正式运行前硬门

必须满足：

- 代码 PR 已经 review、合并，且 experiment registration / RunSpec 为 READY；
- checkout 的 HEAD 是审核后的精确 commit，worktree 无 tracked 修改；
- 8 张可见 CUDA GPU，每张至少 12 GiB；默认每卡两个固定 slot；
- 至少 80 GiB 可用磁盘；
- 服务器能取得固定的 Qwen revision
  `7ae557604adf67be50417f59c2c2f167def9a775`；
- 不提供、不复制、不寻找任何旧 SFT/reference adapter。

单元测试、静态检查或两步 liveness 不是正式实验结果。

## 4. 一键入口

在仓库根目录执行。先把审核后的完整 commit 写入环境变量：

```bash
export E8_COLDSTART_EXPECTED_COMMIT="<reviewed-full-commit-sha>"
bash scripts/run_e8_multitask_exp_coldstart.sh full
```

`full` 依次完成：runtime setup、数据 prepare/qualification、八任务并行校准、
两步 canonical liveness、160-cell 动态队列、aggregate、terminal audit 和 package。

若希望逐段检查：

```bash
export E8_COLDSTART_EXPECTED_COMMIT="<reviewed-full-commit-sha>"
bash scripts/run_e8_multitask_exp_coldstart.sh setup
bash scripts/run_e8_multitask_exp_coldstart.sh prepare
bash scripts/run_e8_multitask_exp_coldstart.sh plan
bash scripts/run_e8_multitask_exp_coldstart.sh calibrate
bash scripts/run_e8_multitask_exp_coldstart.sh liveness
bash scripts/run_e8_multitask_exp_coldstart.sh run
bash scripts/run_e8_multitask_exp_coldstart.sh finish
```

中断后继续：

```bash
bash scripts/run_e8_multitask_exp_coldstart.sh resume
bash scripts/run_e8_multitask_exp_coldstart.sh finish
```

已完成且 identity 一致的 cell 会直接复用；只有存在目录但没有完整 manifest 的
cell 才会在 `--retry-incomplete` 语义下重跑。不要使用 `--force` 覆盖完整 cell。

## 5. 调度语义

默认 GPU 为 `0,1,2,3,4,5,6,7`，每卡两个固定 slot，共 16 个 worker。所有 worker
共享一个 160-cell queue。任一 cell 完成后，该 slot 立即取下一个 cell，不等待同一
名义批次的其他 15 个 cell。

调度记录位于：

```text
outputs/e8/<RUN_ID>/scheduler/queue_events.jsonl
outputs/e8/<RUN_ID>/scheduler/dynamic_run.json
outputs/e8/<RUN_ID>/logs/<CELL_KEY>.log
```

任一子进程失败时，本次调度 fail closed：正在运行的 cell 收尾，未开始的 cell 保留
为 unscheduled，修复后用 `resume` 继续。

## 6. 输出与传输

默认运行目录：

```text
outputs/e8/E8_MULTITASK_EXP_COLDSTART_20260808_01/
```

`finish` 会打印两个最重要的路径和 SHA-256：

1. 完整 text-first 结果 ZIP：
   `packages/<RUN_ID>_full_results.zip`；
2. 画论文曲线所需的小文件：
   `aggregate/plot_curve_points.csv`。

CSV 每行就是一个任务/方法/lambda 点，包含 best/terminal Pass@8、greedy、valid
rate、best/terminal step、stop reason、seed、源码 commit 和是否旧网格 anchor。
只画图时优先传这个 CSV。ZIP 保存 config、split/calibration identity、完整训练曲线、
日志、cell manifests、aggregate、audit 和 checksums，但按仓库协议排除 LoRA 权重。

只有 `terminal_audit.json` 中
`all_training_and_evaluation_complete=true` 时才允许 package。

## 7. 结果解释

- 主曲线来自这次无 warm-start 的 160 个 cell；
- 旧 72-cell warm-start 结果保留为历史辅助证据，不混入主选择；
- 七个旧 lambda anchor 在每个任务的新网格中重跑，可用于判断 warm-start 和
  cold-start 曲线是否近似，但不能预先假定二者相同；
- 当前只有一个 tuning seed，属于 response-shape pilot；找到稳定内部峰值后仍需 fresh
  seed confirmation；
- best checkpoint、terminal checkpoint、任务性能、valid/structure 诊断和 NaN/Inf
  必须分开报告；early stop 或 1200 上限都不等于收敛证明。

## 8. 当前验证状态

交付时已完成的本地验证应记录在 PR：Ruff、Python compilation、shell syntax、160-cell
plan、动态无屏障队列测试、schema adapter 测试、canonical blob audit、aggregate/CSV
测试和旧 taper 单测。没有 8-GPU/CUDA 的环境不得声称通过正式 liveness 或完成正式
160-cell 实验。
