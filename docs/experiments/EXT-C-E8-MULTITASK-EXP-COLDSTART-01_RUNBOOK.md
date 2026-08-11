# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 一键执行 Runbook

## 0. 交付目标与不可变边界

这份文件是服务器本地 AI 的唯一操作入口。执行者不需要预先知道 DRPO 在哪里，不需要
手动切分支，也不需要补路径、adapter 或超参数。它只需完整执行第 1 节的一个代码块，
然后等待终态。

唯一仓库：<https://github.com/easonhuo/drpo>

审核实现：<https://github.com/easonhuo/drpo/pull/309>
实验 ID：`EXT-C-E8-MULTITASK-EXP-COLDSTART-01`

科学边界保持不变：

- 无 SFT warm start、无外部/reference adapter；
- Positive-only 与 Exp 直接调用锁定的老 cold-start kernel；
- 8 个任务 ×（1 Positive-only + 19 Exp）= 160 cells；
- 8 GPU × 每卡 2 slot，共 16 个共享动态 slot，无同步波次屏障；
- 旧 warm-start 结果不混入本次主曲线；
- 工程自检、liveness 和有限步训练不冒充收敛或正式方法排名。

正式模式默认运行 `full`。如果 PR/stack 尚未合并、实验未登记、execution gate 不是
`ready`，或没有唯一 READY RunSpec 绑定本次 `main` SHA，入口会在创建训练环境、
下载模型或接触 GPU 之前停止。这是预期的 fail-closed 行为，不允许绕过。

## 1. 唯一需要执行的代码块

本地 AI 必须把下面代码块作为一个完整 Bash 脚本一次性执行；不要逐行复制，不要自行
替换变量，不要在旧 checkout 内直接 `pull`、`checkout`、`stash`、`reset` 或
`clean`。

<!-- ONE_CLICK_BOOTSTRAP_START -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-COLDSTART-01"
EXPECTED_REPOSITORY="https://github.com/easonhuo/drpo.git"
MODE="${E8_COLDSTART_EXECUTION_MODE:-${1:-full}}"
CURRENT_STAGE="bootstrap_init"
BOOTSTRAP_STATUS="initializing"
BOOTSTRAP_OWNS_ROOT=0

fail() {
  echo "ERROR: $*" >&2
  if [[ "${BOOTSTRAP_OWNS_ROOT}" -eq 1 ]] && declare -F write_state >/dev/null; then
    BOOTSTRAP_STATUS="failed"
    write_state "${BOOTSTRAP_STATUS}"
    echo "BOOTSTRAP_FAILED_STAGE=${CURRENT_STAGE}" >&2
    echo "BOOTSTRAP_STATE=${STATE_FILE}" >&2
  fi
  exit 2
}

case "${MODE}" in
  full|self-test) ;;
  *) fail "usage: $0 {full|self-test}" ;;
esac

if [[ -n "${E8_COLDSTART_BOOTSTRAP_PARENT:-}" ]]; then
  BOOTSTRAP_PARENT="${E8_COLDSTART_BOOTSTRAP_PARENT}"
elif [[ -d /root && -w /root ]]; then
  BOOTSTRAP_PARENT="/root"
elif [[ -d /workspace && -w /workspace ]]; then
  BOOTSTRAP_PARENT="/workspace"
else
  BOOTSTRAP_PARENT="$(pwd -P)"
fi

BOOTSTRAP_ROOT="${E8_COLDSTART_BOOTSTRAP_ROOT:-${BOOTSTRAP_PARENT}/drpo-e8-coldstart-${MODE}}"
STATE_FILE="${BOOTSTRAP_ROOT}/BOOTSTRAP_STATE.env"
CHECKOUT="${BOOTSTRAP_ROOT}/repo"
RUNTIME_ROOT="${BOOTSTRAP_ROOT}/runtime"

write_state() {
  local status="$1"
  mkdir -p "${BOOTSTRAP_ROOT}"
  {
    printf 'experiment_id=%q\n' "${EXPERIMENT_ID}"
    printf 'mode=%q\n' "${MODE}"
    printf 'status=%q\n' "${status}"
    printf 'stage=%q\n' "${CURRENT_STAGE}"
    printf 'source_repo=%q\n' "${SOURCE_REPO:-}"
    printf 'source_remote=%q\n' "${SOURCE_REMOTE:-}"
    printf 'target_ref=%q\n' "${TARGET_REF:-}"
    printf 'target_commit=%q\n' "${TARGET_COMMIT:-}"
    printf 'checkout=%q\n' "${CHECKOUT}"
    printf 'runtime_root=%q\n' "${RUNTIME_ROOT}"
  } >"${STATE_FILE}"
}

on_error() {
  local status="$?"
  BOOTSTRAP_STATUS="failed"
  write_state "${BOOTSTRAP_STATUS}"
  echo "BOOTSTRAP_FAILED_STAGE=${CURRENT_STAGE}" >&2
  echo "BOOTSTRAP_STATE=${STATE_FILE}" >&2
  exit "${status}"
}

[[ ! -e "${BOOTSTRAP_ROOT}" ]] || \
  fail "bootstrap root already exists; refusing to overwrite or reuse it: ${BOOTSTRAP_ROOT}"
mkdir -p "${BOOTSTRAP_ROOT}"
BOOTSTRAP_OWNS_ROOT=1
write_state "${BOOTSTRAP_STATUS}"
trap on_error ERR

is_canonical_origin() {
  local url="$1"
  [[ "${url}" =~ ^https://([^/@]+(:[^/@]*)?@)?github\.com/easonhuo/drpo(\.git)?/?$ ]] ||
    [[ "${url}" =~ ^git@github\.com:easonhuo/drpo(\.git)?$ ]] ||
    [[ "${url}" =~ ^ssh://git@github\.com/easonhuo/drpo(\.git)?/?$ ]]
}

canonical_checkout() {
  local candidate="$1"
  [[ -d "${candidate}" ]] || return 1
  git -C "${candidate}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1
  local origin_url
  origin_url="$(git -C "${candidate}" remote get-url origin 2>/dev/null)" || return 1
  is_canonical_origin "${origin_url}" || return 1
  printf '%s\n' "$(cd "${candidate}" && pwd -P)"
}

CURRENT_STAGE="discover_canonical_checkout"
declare -a candidates=()
declare -A seen=()

add_candidate() {
  local candidate="$1"
  [[ -n "${candidate}" ]] || return 0
  [[ -z "${seen[${candidate}]:-}" ]] || return 0
  seen["${candidate}"]=1
  candidates+=("${candidate}")
}

add_candidate "${E8_COLDSTART_EXISTING_REPO:-}"
add_candidate "/root/drpo"
add_candidate "/root/d4rl2"
add_candidate "/workspace/drpo"
add_candidate "/data/drpo"
add_candidate "/mnt/data/drpo"
add_candidate "$(pwd -P)"

SOURCE_REPO=""
SOURCE_REMOTE=""
for candidate in "${candidates[@]}"; do
  if resolved="$(canonical_checkout "${candidate}")"; then
    SOURCE_REPO="${resolved}"
    SOURCE_REMOTE="origin"
    break
  fi
done

if [[ -z "${SOURCE_REPO}" ]]; then
  IFS=: read -r -a search_roots <<< \
    "${E8_COLDSTART_SEARCH_ROOTS:-/root:/home:/workspace:/data:/mnt}"
  for search_root in "${search_roots[@]}"; do
    [[ -d "${search_root}" ]] || continue
    while IFS= read -r -d '' git_marker; do
      candidate="$(dirname "${git_marker}")"
      if resolved="$(canonical_checkout "${candidate}")"; then
        SOURCE_REPO="${resolved}"
        SOURCE_REMOTE="origin"
        break 2
      fi
    done < <(
      find "${search_root}" -maxdepth 5 \
        \( -type d -o -type f \) -name .git -print0 2>/dev/null
    )
  done
fi

if [[ -z "${SOURCE_REPO}" ]]; then
  CURRENT_STAGE="clone_fallback"
  SOURCE_REPO="${BOOTSTRAP_ROOT}/source-cache"
  git clone --filter=blob:none --no-checkout "${EXPECTED_REPOSITORY}" "${SOURCE_REPO}"
  SOURCE_REMOTE="origin"
fi

CURRENT_STAGE="verify_selected_origin"
origin_url="$(git -C "${SOURCE_REPO}" remote get-url "${SOURCE_REMOTE}")"
is_canonical_origin "${origin_url}" || fail "selected checkout does not have canonical origin"

if [[ "${MODE}" == "full" ]]; then
  TARGET_REF="refs/heads/main"
  LOCAL_FETCH_REF="refs/remotes/origin/main"
else
  TARGET_REF="refs/pull/309/head"
  LOCAL_FETCH_REF="refs/e8-coldstart-bootstrap/pr-309-head"
fi

CURRENT_STAGE="fetch_authoritative_ref"
git -C "${SOURCE_REPO}" fetch --no-tags --force "${SOURCE_REMOTE}" \
  "${TARGET_REF}:${LOCAL_FETCH_REF}"
TARGET_COMMIT="$(git -C "${SOURCE_REPO}" rev-parse "${LOCAL_FETCH_REF}^{commit}")"
[[ "${TARGET_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "resolved commit is not a full SHA"

REMOTE_COMMIT="$(
  git -C "${SOURCE_REPO}" ls-remote "${SOURCE_REMOTE}" "${TARGET_REF}" |
    awk -v ref="${TARGET_REF}" '$2 == ref {print $1}'
)"
[[ "${REMOTE_COMMIT}" == "${TARGET_COMMIT}" ]] || \
  fail "fetch/authoritative-ref mismatch for ${TARGET_REF}"

CURRENT_STAGE="create_isolated_worktree"
git -C "${SOURCE_REPO}" worktree add --detach "${CHECKOUT}" "${TARGET_COMMIT}"
[[ "$(git -C "${CHECKOUT}" rev-parse HEAD)" == "${TARGET_COMMIT}" ]] || \
  fail "isolated checkout commit mismatch"
[[ -z "$(git -C "${CHECKOUT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail "isolated checkout is not clean"
[[ -f "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" ]] || \
  fail "target commit does not contain the reviewed experiment entrypoint"

BOOTSTRAP_STATUS="prepared"
write_state "${BOOTSTRAP_STATUS}"

CURRENT_STAGE="execute_${MODE}"
export E8_COLDSTART_EXPECTED_COMMIT="${TARGET_COMMIT}"
export E8_COLDSTART_RUNTIME_ROOT="${RUNTIME_ROOT}"
bash "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" "${MODE}"

CURRENT_STAGE="complete"
BOOTSTRAP_STATUS="complete"
write_state "${BOOTSTRAP_STATUS}"
echo "BOOTSTRAP_SOURCE_REPO=${SOURCE_REPO}"
echo "BOOTSTRAP_CHECKOUT=${CHECKOUT}"
echo "BOOTSTRAP_TARGET_COMMIT=${TARGET_COMMIT}"
echo "BOOTSTRAP_STATE=${STATE_FILE}"
```
<!-- ONE_CLICK_BOOTSTRAP_END -->

这一个入口会自动完成：

1. 优先核验 `/root/drpo`、`/root/d4rl2`、当前目录和常见服务器目录；
2. 只接受 `origin` 精确指向 `easonhuo/drpo` 的 Git checkout；
3. 即使找到的旧 checkout 有未提交修改，也只 fetch Git 对象，不改变其 HEAD、分支、
   index 或工作树；
4. 找不到合法 checkout 时才从 GitHub clone 一个 source cache；
5. 通过 `git fetch` 与 `git ls-remote` 双重解析权威 `main` 完整 SHA；
6. 在独立目录创建 detached clean worktree；
7. 把 venv、模型、数据、日志、guard 状态和结果全部放在 worktree 外；
8. 调用正式 `full` 入口，依次执行登记/READY 门、setup、prepare/qualification、
   八任务 calibration、两步 liveness、160-cell 动态队列、自动重试、aggregate、
   terminal audit、hardened package 和重新解包复验；
9. 任一步失败时写出 `BOOTSTRAP_STATE.env`，明确失败 stage、源码 SHA、checkout 和
   runtime 路径，不要求服务器执行者现场判断恢复命令。

默认使用全新目录 `/root/drpo-e8-coldstart-full`（非 root 环境自动选择可写位置）。
若该目录已存在，入口拒绝覆盖或复用，防止重复启动、污染旧证据或误把残留输出当成
本轮结果。

## 2. 代码同源与正式训练路径

本实验不重新实现算法。正式 dispatch 固定为：

- Positive-only → `countdown_e8_base_rl_replay.train_offline_method(..., method="positive_only")`；
- Exp → `countdown_e8_oracle_offline_v2_taper_runtime.worker` →
  `countdown_e8_oracle_offline_v2_taper_sweep.train_cell`；
- 两条路径最终都以 `adapter_path=None, trainable_adapter=True` 从同一 Qwen base
  创建 fresh LoRA。

runner 会逐字节校验 arena、Positive-only wrapper、taper core、hardened runtime 和两份
老配置的 Git blob。八任务层只负责数据 schema、verifier、调度和结果导出，不复制 loss、
optimizer、scheduler、near/far 选择或 early-stop 逻辑。

固定任务是 Countdown、Word Sorting、Mini Sudoku、Maze、Word Ladder、Knights and
Knaves、Graph Coloring、WikiSQL。Spiral Matrix 因旧网格已饱和而排除。19 个 Exp λ
保留旧公共锚点并扩展曲线覆盖；本 Runbook 不允许现场改网格。

## 3. 正式入口的硬门与自动准备

一键入口解析的是执行时权威 `origin/main`，不是聊天记录中的旧 SHA。随后正式 runner
在任何重型操作前同时检查：

- `EXT-C-E8-MULTITASK-EXP-COLDSTART-01` 已进入 handoff 和 registry；
- `implementation_state` 为 implemented，`execution_gate.state=ready`；
- `runspecs/ready/` 恰有一份本实验 RunSpec；
- RunSpec 的 `repo_commit` 等于 detached checkout 的完整 SHA；
- RunSpec entrypoint 调用 `run_e8_multitask_exp_coldstart.sh full`；
- checkout 完全干净且 `origin/main` 与权威远端一致；
- 8 张 CUDA GPU，每张至少 12 GiB，运行盘至少 80 GiB；
- Python ≥3.10，CUDA-compatible PyTorch 可见 8 张卡；
- Qwen revision 固定为
  `7ae557604adf67be50417f59c2c2f167def9a775`；
- 六个老 cold-start source/config Git blob 全部匹配；
- 目标 pytest、formal-channel validator 和依赖检查通过。

固定 Python 依赖、Qwen 模型、公开数据、bank 构造、split、qualification 和 schema 转换
由正式脚本自动准备。CUDA driver 与能识别 GPU 的 PyTorch 属于服务器基础硬件栈；不满足
时预检直接停止，不会进入训练。

## 4. 动态队列、失败与恢复

正式 `full` 始终在 hardened foreground guard 内运行。160 cells 使用 16 个固定 slot
共享队列：任一 slot 完成即领取下一 cell，不等待整批结束。

同一次 guard 内最多运行三次 queue attempt。后续 attempt 只复用 identity 与 manifest
完整匹配的已完成 cell，并只重跑 incomplete/unscheduled cell；禁止 `--force` 覆盖
完整 cell。三次后仍失败时，guard 保存 heartbeat、完整日志、失败状态、未完成 inventory
和 recovery artifact 后非零退出。

如果整个服务器进程被外部终止，不允许直接运行裸 `resume`。执行者只回传
`BOOTSTRAP_STATE.env`、guard failure artifact、`scheduler/dynamic_run.json` 和最后
日志；reviewer 依据不可变 source/output identity 生成下一次恢复指令。部分 cell、
shell 退出或 raw-complete 都不能冒充最终实验完成。

## 5. 成功终态只回传两个文件

成功时终端会打印：

```text
RAW_COMPLETE_RESULTS_ZIP=...
<sha256>  <raw-complete-zip>
PLOT_CSV=...
<sha256>  <plot_curve_points.csv>
```

服务器执行者只需回传：

1. `RAW_COMPLETE_RESULTS_ZIP`：通过 hardened verifier 的
   `experiment-raw-complete` 证据包；
2. `PLOT_CSV`：160 行 `plot_curve_points.csv`。

raw-complete 包包含 config、source provenance、split/calibration identity、queue events、
逐 cell manifest 与日志、aggregate、`RUN_COMPLETE.json`、`run_manifest.json`、
`scientific_run_manifest.json`、terminal audit、inventory 和 SHA-256；不包含 LoRA 或
模型权重。

`raw-complete` 不是 `experiment-final`。reviewer 回收结果、完成解释并通过
handoff/registry delta 关闭仓库证据链后，才能生成 final artifact。服务器本地 AI 不解释
方法排名、不编辑论文、不修改 registry，也不把 raw-complete 宣称为仓库闭环完成。

## 6. 失败时只回传证据，不现场 debug

若一键入口非零退出，执行者不得修改代码或猜下一条命令，只回传终端中打印的：

- `BOOTSTRAP_FAILED_STAGE`；
- `BOOTSTRAP_STATE` 指向的文件；
- 若已经进入 guard，则回传 guard 生成的 failed/recovery artifact 与最后日志。

常见的正确拒绝包括：错误 origin、权威 ref 漂移、已有 bootstrap root、main 未合并登记、
READY RunSpec 缺失或 SHA 不一致、GPU/PyTorch/磁盘不达标、模型 revision 不可获取、
canonical Git blob 漂移。每一种都必须先由 reviewer 判断，不能由本地 AI 绕过。
