# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 一键执行与自动恢复 Runbook

## 0. 交付目标与不可变边界

这份文件是服务器本地 AI 的唯一操作入口。执行者不需要预先知道 DRPO 在哪里，不需要
手动切分支，不需要判断应该从训练、聚合还是打包恢复。完整执行第 1 节唯一代码块后，
脚本会自动发现仓库、创建隔离 worktree、运行门禁并在同一 runtime 内自动恢复。

唯一仓库：<https://github.com/easonhuo/drpo>

审核实现：<https://github.com/easonhuo/drpo/pull/309>

实验 ID：`EXT-C-E8-MULTITASK-EXP-COLDSTART-01`

科学边界保持不变：

- 无 SFT warm start、无外部/reference adapter；
- Positive-only 与 Exp 直接调用锁定的老 cold-start kernel；
- 8 个任务 ×（1 Positive-only + 19 Exp）= 160 cells；
- 8 GPU × 每卡 2 slot，共 16 个共享动态 slot，无同步波次屏障；
- 任务、λ、seed、训练公式、optimizer、阈值、early stop 与结果解释均不得现场修改；
- 工程自检、liveness、恢复验收和有限步训练不冒充收敛或正式方法排名。

正式模式默认运行 `full`。PR/stack 未合并、实验未登记、execution gate 不是
`ready`，或没有唯一 READY RunSpec 绑定执行时的精确 `main` SHA 时，入口会在创建
训练环境、下载模型或接触 GPU 前停止。不得绕过。

## 1. 唯一需要执行的代码块

本地 AI 必须把下面代码块保存为一个完整 Bash 脚本并一次性执行。不要逐行复制，不要
替换超参数，也不要在旧 checkout 内直接 `pull`、`checkout`、`stash`、
`reset` 或 `clean`。

第一次运行与中断后的再次运行使用完全相同的代码块和命令。已有合法 bootstrap root
不会再被拒绝；脚本会验证 experiment、mode、origin、source SHA 与 runtime identity，
然后继续未完成的 attempt。若已有终态包，则只复验并返回，不新建训练。

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
    if [[ -f "${CHECKOUT:-}/scripts/run_e8_multitask_exp_coldstart.sh" ]]; then
      E8_COLDSTART_EXPECTED_COMMIT="${TARGET_COMMIT:-}" \
        E8_COLDSTART_RUNTIME_ROOT="${RUNTIME_ROOT:-${BOOTSTRAP_ROOT:-.}/runtime}" \
        E8_COLDSTART_BOOTSTRAP_STATE="${STATE_FILE}" \
        bash "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" \
          diagnose "bootstrap_${CURRENT_STAGE}" || true
    fi
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
  if [[ -f "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" ]]; then
    E8_COLDSTART_EXPECTED_COMMIT="${TARGET_COMMIT:-}" \
      E8_COLDSTART_RUNTIME_ROOT="${RUNTIME_ROOT}" \
      E8_COLDSTART_BOOTSTRAP_STATE="${STATE_FILE}" \
      bash "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" \
        diagnose "bootstrap_${CURRENT_STAGE}" || true
  fi
  echo "BOOTSTRAP_FAILED_STAGE=${CURRENT_STAGE}" >&2
  echo "BOOTSTRAP_STATE=${STATE_FILE}" >&2
  exit "${status}"
}

RESUME_BOOTSTRAP=0
BOOTSTRAP_WAS_COMPLETE=0
if [[ -e "${BOOTSTRAP_ROOT}" ]]; then
  [[ -d "${BOOTSTRAP_ROOT}" ]] || fail "bootstrap root exists but is not a directory"
  [[ -f "${STATE_FILE}" ]] || fail "existing bootstrap root has no identity state: ${STATE_FILE}"
  [[ -d "${CHECKOUT}" ]] || fail "existing bootstrap root has no isolated checkout: ${CHECKOUT}"
  grep -Fqx "experiment_id=${EXPERIMENT_ID}" "${STATE_FILE}" || \
    fail "existing bootstrap state belongs to another experiment"
  grep -Fqx "mode=${MODE}" "${STATE_FILE}" || \
    fail "existing bootstrap state was created for another mode"
  RESUME_BOOTSTRAP=1
  if grep -Fqx "status=complete" "${STATE_FILE}"; then
    BOOTSTRAP_WAS_COMPLETE=1
  fi
  BOOTSTRAP_STATUS="recovering"
else
  mkdir -p "${BOOTSTRAP_ROOT}"
fi
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

if [[ "${RESUME_BOOTSTRAP}" -eq 1 ]]; then
  SOURCE_REPO="${CHECKOUT}"
  SOURCE_REMOTE="origin"
else
  SOURCE_REPO=""
  SOURCE_REMOTE=""
  for candidate in "${candidates[@]}"; do
    if resolved="$(canonical_checkout "${candidate}")"; then
      SOURCE_REPO="${resolved}"
      SOURCE_REMOTE="origin"
      break
    fi
  done
fi

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
if [[ "${BOOTSTRAP_WAS_COMPLETE}" -eq 1 ]]; then
  TARGET_COMMIT="$(git -C "${CHECKOUT}" rev-parse HEAD)"
else
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
fi

CURRENT_STAGE="create_isolated_worktree"
if [[ "${RESUME_BOOTSTRAP}" -eq 0 ]]; then
  git -C "${SOURCE_REPO}" worktree add --detach "${CHECKOUT}" "${TARGET_COMMIT}"
fi
[[ "$(git -C "${CHECKOUT}" rev-parse HEAD)" == "${TARGET_COMMIT}" ]] || \
  fail "existing isolated checkout no longer matches authoritative ${TARGET_REF}; local AI review is required"
[[ -z "$(git -C "${CHECKOUT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail "isolated checkout is not clean"
[[ -f "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" ]] || \
  fail "target commit does not contain the reviewed experiment entrypoint"

BOOTSTRAP_STATUS="prepared"
write_state "${BOOTSTRAP_STATUS}"

CURRENT_STAGE="execute_${MODE}"
export E8_COLDSTART_EXPECTED_COMMIT="${TARGET_COMMIT}"
export E8_COLDSTART_RUNTIME_ROOT="${RUNTIME_ROOT}"
export E8_COLDSTART_BOOTSTRAP_STATE="${STATE_FILE}"
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

## 2. 一次执行会完成什么

入口按以下顺序自动工作：

1. 优先核验 `/root/drpo`、`/root/d4rl2`、当前目录和常见服务器目录；
2. 只接受 `origin` 精确指向 `easonhuo/drpo` 的 Git checkout；
3. 旧 checkout 即使是 dirty，也只用于 fetch Git 对象，不改变其 HEAD、分支、index
   或工作树；找不到合法 checkout 才 clone source cache；
4. 通过 `git fetch` 与 `git ls-remote` 双重解析权威 ref，并创建 detached clean
   worktree；
5. 把 venv、模型、输入、日志、attempt、checkpoint 与结果全部放在 worktree 外；
6. 在任何重型操作前执行 registry、execution gate、READY RunSpec、source blob、
   origin/main、GPU、磁盘、依赖与测试门；
7. 完成 prepare/qualification、八任务 calibration、两步 liveness、160-cell 动态队列、
   aggregate、terminal audit、finalize、23 MiB delivery preflight、hardened guard 打包和
   独立复验；
8. 成功时打印 raw-complete ZIP、plot CSV 与最新恢复 checkpoint 的绝对路径和 SHA-256。

## 3. 自动恢复语义

每个正式 attempt 都在新的 hardened guard root 中运行，例如：

```text
runtime/guard/E8_MULTITASK_EXP_COLDSTART_20260808_01/attempt-001
runtime/guard/E8_MULTITASK_EXP_COLDSTART_20260808_01/attempt-002
```

失败后不会修改、覆盖或删除旧 attempt。自动恢复会：

- 对 prepare、calibration、liveness、cells、aggregate、audit 与 finalize 分阶段审计；
- 从最近一个具备完整 prepare/source 身份的 attempt，只导入 identity、config、精确
  40 位 source commit、evaluation 与文件完整性均通过的 completed cells；
- 使用同一文件系统硬链接复用完整输入、adapter 与原始结果，避免复制大 checkpoint；
- 将 incomplete、corrupt、identity mismatch 或 NaN/Inf cell 排除，不伪造 complete；
- 为每个导入 cell 保存源 manifest SHA-256 与新 manifest SHA-256；
- 从第一个不完整阶段继续；aggregate/audit/finalize/打包失败不会重新训练 completed
  cells；
- 单次 queue 内保留三次 incomplete-cell retry；guard attempt 失败后默认再自动创建
  最多三个新 attempt；
- 使用 `flock` 保证同一 runtime 只有一个 writer；原进程仍活着时拒绝启动第二队列；
- 已有完整 raw-complete artifact 时只做独立验证并返回，重复执行不会新跑实验。

锁定老 kernel 没有保存完整 optimizer、scheduler、RNG 与 dataloader 状态，因此不宣称
单 cell 内精确断点续训。一个 cell 在最终 manifest 写出前中断时，只重跑该 cell；其他
完整 cell 均复用。正常机器/进程中断最多影响当时运行中的 16 个 cell，不会从 160 个
cell 全部重跑。

## 4. 运行期间 checkpoint 与打包保护

动态队列默认每新增 5 个完整 cell，就原子更新：

```text
runtime/recovery/E8_MULTITASK_EXP_COLDSTART_20260808_01/latest_checkpoint.zip
```

该 `experiment-checkpoint` 包使用仓库 hardened producer，包含：

- 完整 cell inventory；
- 每个 cell manifest 的路径与 SHA-256；
- persistent-local canonical output / terminal adapter 定位；
- source commit、config hash、completed/missing inventory；
- 独立 checksums 与 source snapshot。

新 checkpoint 完成内部验证前不会替换上一份有效包。若配置了一个已经挂载的异机持久
目录，可在运行前设置 `E8_COLDSTART_RECOVERY_MIRROR=/absolute/mounted/path`；每个
新 checkpoint 会校验后原子镜像到该目录。未配置 mirror 时，只能保证服务器本地磁盘
恢复，不能抵御整机或磁盘永久丢失，本地 AI 必须明确报告该剩余风险。

正式 guard 返回成功前，会先用同一 hardened packager 以 23 MiB 上限真实预打包，给
最终 25 MiB 硬上限留下余量。若累计日志导致超限，脚本以可中断事务生成带 SHA-256
inventory 的完整 `tar.gz` 日志归档，在 `logs/tails` 保留每份日志最后 64 KiB，再重新预检。
非体积原因或压缩后仍失败时禁止声称完成，并进入第 5 节本地 AI 处置。

## 5. 自动恢复仍失败时，本地 AI 必须继续处理

脚本耗尽安全自动恢复次数、检测到并发 writer、source/main 漂移、hardlink 跨文件系统、
数据损坏或不可自动分类的打包错误时，会生成：

```text
runtime/recovery/E8_MULTITASK_EXP_COLDSTART_20260808_01/LOCAL_AI_RECOVERY.md
runtime/recovery/E8_MULTITASK_EXP_COLDSTART_20260808_01/LOCAL_AI_RECOVERY_BUNDLE.zip
```

正在执行本 Runbook 的本地 AI 不应立刻把调试退给用户。它必须先读取 prompt 与 bundle，
按其中顺序检查 attempt state、recovery plan、scheduler inventory、package status 和
supervised log tail，然后首先重跑同一代码块。只有以下情况才停止并回报 reviewer：

- 权威 `main` 已变化，无法继续满足原 READY RunSpec/source identity；
- manifest、adapter 或数据哈希损坏；
- 服务器磁盘/文件系统损坏或跨设备无法 hardlink，且安全复制空间不足；
- 唯一修复会改变被冻结的科学变量或锁定 kernel；
- 三次最小工程修复后仍是同一不可恢复错误。

服务器若已经有自主本地 AI wrapper，可在首次执行前把
`E8_COLDSTART_LOCAL_AI_COMMAND` 设为该 wrapper 的绝对可执行文件路径。普通自动恢复耗尽
后，父进程会在仍持有 single-writer lock 时调用它；第一个位置参数是
`LOCAL_AI_RECOVERY.md`，第二个是 `LOCAL_AI_RECOVERY_BUNDLE.zip`，同时通过
`DRPO_LOCAL_AI_PROMPT`、`DRPO_LOCAL_AI_BUNDLE`、`DRPO_LOCAL_AI_REPOSITORY`、
`DRPO_LOCAL_AI_RUNTIME` 与 `DRPO_LOCAL_AI_FAILURE_STAGE` 提供同一上下文。wrapper 只可在
完成不改变科研边界的工程修复后返回 0；父进程随后只允许一个新的 hardened guard
attempt。默认超时 900 秒，每个 runtime 最多调用三次，调用状态与输出均写入恢复包。
未配置 wrapper 时不会猜测服务器使用哪一种本地 AI CLI，而是保留完整 bundle 供现有
本地 AI 直接读取。

本地 AI 不得删除旧 attempt，不得手工写 complete manifest，不得用裸 `resume` 绕过
guard，不得修改实验协议以让流程“跑通”。

## 6. 成功终态与回传

成功时终端必须同时出现：

```text
RAW_COMPLETE_RESULTS_ZIP=...
<sha256>  <raw-complete-zip>
PLOT_CSV=...
<sha256>  <plot_curve_points.csv>
RECOVERY_CHECKPOINT=...
<sha256>  <latest_checkpoint.zip>
```

服务器只需回传前两个正式交付文件：

1. `RAW_COMPLETE_RESULTS_ZIP`
2. `PLOT_CSV`

`RECOVERY_CHECKPOINT` 是中断恢复证据，不是论文结果，也不传给 `drpo-update`。真正的
`experiment-final` 仍由 reviewer 回收 raw-complete 结果、完成解释和 handoff/registry
仓库闭环后生成。

## 7. 绝对禁止

- 不得把 `self-test`、liveness、checkpoint 或 failed artifact 当成科学结果；
- 不得在 PR 分支上绕过正式登记与 READY RunSpec 运行；
- 不得访问 sealed test 来选 λ；
- 不得混入旧 warm-start adapter 或旧 warm-start 曲线；
- 不得删除历史 attempt、failed artifact、日志归档或 recovery inventory；
- 不得把 task-performance degradation、support/structure boundary 与 NaN/Inf 混报。
