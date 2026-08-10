# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 本地执行交接

## 0. 本地 AI 的职责

本地 AI 不修改代码、不选择超参数、不寻找旧 adapter，也不解释科学结果。它只执行
本文的确定性命令，并在命令失败时返回失败阶段、日志路径和已有失败包。

唯一仓库：<https://github.com/easonhuo/drpo>
当前审核 PR：<https://github.com/easonhuo/drpo/pull/309>

正式 160-cell 实验只有在 PR 已 review/合并、实验已登记且 RunSpec 为 READY 后才允许
启动。合并前只运行第 1 节的工程自检。

## 1. 从空目录取得源码并运行工程自检

以下整段可直接复制。它会从 PR #309 的远端 head 解析一个不可变完整 SHA，detached
checkout 后运行不需要模型或 GPU 的 placeholder 全流程验收。

```bash
set -euo pipefail

REPO_URL="https://github.com/easonhuo/drpo.git"
WORK_PARENT="${PWD}/drpo-e8-coldstart"
CHECKOUT="${WORK_PARENT}/repo"

test ! -e "${WORK_PARENT}"
mkdir -p "${WORK_PARENT}"
git clone --filter=blob:none --no-checkout "${REPO_URL}" "${CHECKOUT}"
cd "${CHECKOUT}"
git fetch origin refs/pull/309/head
IMPLEMENTATION_COMMIT="$(git rev-parse FETCH_HEAD)"
git checkout --detach "${IMPLEMENTATION_COMMIT}"
test "$(git rev-parse HEAD)" = "${IMPLEMENTATION_COMMIT}"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

export E8_COLDSTART_EXPECTED_COMMIT="${IMPLEMENTATION_COMMIT}"
export E8_COLDSTART_RUNTIME_ROOT="${WORK_PARENT}/runtime"
bash scripts/run_e8_multitask_exp_coldstart.sh self-test
```

该自检不会加载 Qwen、不会调用 CUDA、不会执行优化器更新，也不能产生科研证据。它会
真实执行以下非 GPU 链路：

1. 校验八任务、160 cells、六个老 cold-start Git blob；
2. 构造隔离的小型输入 fixture 并执行 prepare、split 和 canonical schema 转换；
3. 写入明确标记为 `engineering_placeholder_backend=true` 的 calibration/liveness；
4. 启动 16-slot 共享动态队列；
5. 主动注入一次 cell 失败，验证 fail-closed 与 unscheduled 保留；
6. resume 到 160/160，验证第二名义批次在第一批最慢 cell 结束前已经补位；
7. 再运行一次，验证完整 cell 的 manifest 哈希不变；
8. aggregate、terminal audit、生成完整 ZIP 和小 CSV；
9. 重新打开 ZIP，核对安全路径、成员清单和每个 SHA-256；
10. 篡改 ZIP 并确认 verifier 拒绝。

成功终态必须同时出现：

```text
"complete": true
"resume_completed_cells": 160
"repeat_run_preserved_cell_hashes": true
"later_cell_started_before_first_batch_finished": true
"package_reopen_verification_passed": true
"tampered_package_rejected": true
ENGINEERING_SELF_TEST_ROOT=...
```

## 2. 科学协议与代码同源边界

本实验不使用 SFT warm start，也不加载外部 reference adapter。Positive-only 与 Exp
分别直接进入已锁定的老 cold-start 实现：

- Positive-only：`countdown_e8_base_rl_replay.train_offline_method`；
- Exp：`countdown_e8_oracle_offline_v2_taper_runtime.worker`，随后调用老
  `taper_sweep.train_cell`；
- 两条路径最终都以 `adapter_path=None, trainable_adapter=True` 从同一 Qwen base
  创建 fresh LoRA。

runner 会逐字节校验 arena、Positive-only、taper core、hardened runtime 和两份老配置
的 Git blob SHA。八任务层只负责数据 schema、verifier、调度与导出，不复制 loss、
optimizer、scheduler 或 early stop。

固定任务为 Countdown、Word Sorting、Mini Sudoku、Maze、Word Ladder、Knights and
Knaves、Graph Coloring、WikiSQL。每任务 `1 Positive-only + 19 Exp`，共 160 cells。
Spiral Matrix 因旧网格饱和而排除。16 个固定 slot 为 8 GPU × 每卡 2 slot；“十波”
仅是 `160 / 16` 的容量换算，不是同步屏障。

## 3. 正式运行前硬门

正式运行必须同时满足：

- PR #309 及其 stack 已 review 并合并到 `main`；
- `EXT-C-E8-MULTITASK-EXP-COLDSTART-01` 已出现在 `docs/handoff.md` 和
  `experiments/registry.yaml`，formal execution validator 通过；
- `runspecs/ready/` 中恰好有一份本实验 RunSpec，其 `repo_commit` 绑定本次 checkout 的
  完整 Git SHA，entrypoint 调用本脚本的 `full`；
- checkout 完全干净，运行目录、venv、模型和结果均位于仓库外；
- 8 张 CUDA GPU，每张至少 12 GiB；运行盘至少 80 GiB；
- 系统 Python ≥3.10，并已有能识别这些 GPU 的 CUDA-compatible PyTorch；
- 能下载 Qwen revision `7ae557604adf67be50417f59c2c2f167def9a775`；
- 不提供、不复制、不寻找任何旧 SFT/reference adapter。

`full` 会先 fail-closed 检查登记、`execution_gate.state=ready` 和上述 READY RunSpec，未
获准时不会创建环境、下载模型或检查 GPU。通过后再检查源码 SHA、`origin/main`、干净 worktree、GPU、磁盘、
依赖、模型 revision、旧代码 blob 和测试，然后通过
`run_experiment_guard_hardened.py` 前台守护正式链路。任何门未满足都不启动 sweep。

## 4. 合并并登记后的正式一键运行

在新的空目录中执行；不复用第 1 节的 PR checkout 或自检结果。

```bash
set -euo pipefail

REPO_URL="https://github.com/easonhuo/drpo.git"
WORK_PARENT="${PWD}/drpo-e8-coldstart-formal"
CHECKOUT="${WORK_PARENT}/repo"

test ! -e "${WORK_PARENT}"
mkdir -p "${WORK_PARENT}"
git clone --filter=blob:none --no-checkout "${REPO_URL}" "${CHECKOUT}"
cd "${CHECKOUT}"
git fetch origin main
IMPLEMENTATION_COMMIT="$(git rev-parse origin/main)"
git checkout --detach "${IMPLEMENTATION_COMMIT}"
test "$(git rev-parse HEAD)" = "${IMPLEMENTATION_COMMIT}"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

export E8_COLDSTART_EXPECTED_COMMIT="${IMPLEMENTATION_COMMIT}"
export E8_COLDSTART_RUNTIME_ROOT="${WORK_PARENT}/runtime"
bash scripts/run_e8_multitask_exp_coldstart.sh full
```

`full` 的顺序为：登记/READY 门 → setup → prepare/qualification → 八任务并行 calibration →
canonical 两步 liveness → 160-cell 动态队列 → aggregate → terminal audit → package →
解包复验。固定 Qwen、公开数据源和 Python 依赖由脚本准备；CUDA driver/PyTorch 属于
服务器硬件栈，预检失败时不会进入数据或训练阶段。

## 5. 失败与恢复

正式 `full` 在同一个前台 guard 内最多执行三次动态队列 attempt。后一次只复用 identity
一致的完整 cell，并重跑 incomplete/unscheduled cell；禁止 `--force` 覆盖完整 cell。
三次后仍失败时，guard 会保存 heartbeat、日志、失败状态和 recovery artifact，然后以
非零状态退出。

若整个服务器进程被终止，不要直接运行未守护的 `resume`。把 recovery artifact、
`scheduler/dynamic_run.json` 和最后日志交回 reviewer，由 reviewer 以新的 attempt/output
identity 生成恢复指令。不得把 shell 退出、部分 cell 或 raw-complete 冒充实验完成。

## 6. 服务器运行结束只需交回两个文件

成功后脚本打印：

```text
RAW_COMPLETE_RESULTS_ZIP=...
<sha256>  <raw-complete-zip>
PLOT_CSV=...
<sha256>  <plot_curve_points.csv>
```

交回：

1. `RAW_COMPLETE_RESULTS_ZIP`：完整受守护的 `experiment-raw-complete` 证据包；
2. `PLOT_CSV`：论文画曲线所需的 160 行小 CSV。

raw-complete 包包含 config、source provenance、split/calibration identity、queue events、逐 cell
manifest 与日志、aggregate、`RUN_COMPLETE.json`、`run_manifest.json`、
`scientific_run_manifest.json`、terminal audit、包内 inventory 和 checksums，并排除
LoRA/model weights。只有 terminal audit 的 `all_training_and_evaluation_complete=true`
且 hardened verifier 通过时才打印成功路径。

这里故意不把 raw-complete 叫作 `experiment-final`。仓库协议要求 reviewer 回收上述两个
文件、完成结果解释并更新 `docs/handoff.md`、`experiments/registry.yaml` 和紧凑结果后，
再生成带非空 repository closure patch 的 `experiment-final`。本地 AI 不做这一步，也不
能把 raw-complete 冒充最终仓库闭环。

## 7. 结果解释边界

- 主曲线只来自本次无 warm-start 的 160 cells；
- 旧 72-cell warm-start 结果保留为历史辅助证据，不混入新主选择；
- 七个旧 lambda anchor 在新网格中重跑，只用于事后比较曲线结构；
- 当前是单 tuning seed 的 response-shape pilot，仍需 fresh-seed confirmation；
- best、terminal、任务性能、valid/structure 诊断和 NaN/Inf 分开报告；
- early stop、1200 上限、工程自检和两步 liveness 都不等于收敛或正式科学结果。
