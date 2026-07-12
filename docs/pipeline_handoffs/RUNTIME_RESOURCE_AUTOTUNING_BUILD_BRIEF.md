# DRPO 运行时资源自标定与并发调度 Pipeline：分阶段构建交接说明

**治理 claim：** `GOV-RUNTIME-RESOURCE-AUTOTUNE-01`  
**文档版本：** v2 phased plan  
**文档性质：** 构建设计交接稿，不是研究 Master；不改变任何实验状态、科学变量、方法数量、seeds、阈值、训练 horizon、停止标准或论文结论。  
**权威来源：** `AGENTS.md`、`docs/handoff.md`、`experiments/registry.yaml`、`docs/governance_pipeline_stage_status.yaml`、`runspecs/README.md`。  
**一期目标：** 先交付一个足够小、可审计、可回滚的 DRPO 内核，使后续 E7 CPU 与 E8 GPU 新任务能够 opt-in 使用。  
**长期目标：** 在一期真实使用稳定后，再逐步扩展到 E3/E4/E9、通用训练任务以及独立的跨项目 SDK。  
**本版基线：** 本文档 v2 重建时的 `main` 为 `32aa4fc02a4faadded897fd14aeda2d51a1151bc`；后续实现必须重新解析当时最新 `main`，不得把该 SHA 当作永久基线。  
**命名约束：** 本方案统一称为 **runtime resource autotuning / 运行时资源自标定**。仓库已有 `countdown_e8_capacity_diag.py`，其中 “capacity” 指模型或训练方案容量诊断，属于科学实验，不得与本方案混用。

---

## 0. 后续实现 session 的强制启动要求

开始任何实现前必须：

1. 读取 `AGENTS.md`。
2. 首先读取 `docs/handoff.md` 第 0 节，并继承最新锁定结论、术语、实验门禁和执行顺序。
3. 读取 `experiments/registry.yaml`。
4. 读取 `docs/governance_pipeline_stage_status.yaml`。
5. 读取 `runspecs/README.md`、`scripts/agent/run_lane.py`、`scripts/agent/run_claimed_runspec.py`、RunSpec schema/validator 及相关 tests。
6. 检查当时最新 `main` SHA、目标 dev branch SHA，以及服务器上 E7/E8 的真实运行状态；不得中断、修改或复用未完成任务的结果目录。
7. 检查现有 E7 CPU runner、E8 GPU runner、线程环境、数据读取方式、branch identity、artifact 和 provenance 逻辑，不能只按本文档想象实现。
8. 判断实现是否触及 `canonical_formal_experiment_channel` 或其他 `closed_maintenance_only` stage 的 protected responsibility。
9. 若属于 new feature、architecture expansion、responsibility change 或 default-policy change，必须在实现前完成：
   - 用户明确授权；
   - 注册治理 claim；
   - 新建 authorization record；
   - 写 rollback plan；
   - 通过 governance stage validator。
10. 在实质性工作前报告：
    - 仓库和分支；
    - current `main` SHA；
    - 当前实验状态；
    - 对应 claim；
    - 计划修改文件；
    - 已知不确定性。

本文档只传递设计、边界、实施顺序和验收标准，不自动构成 closed-stage reopen authorization，也不自动授权修改正在运行的实验。

---

## 1. 本版决策：先做小内核，再逐步平台化

### 1.1 一期不做“大而全平台”

一期只解决一个明确问题：

> 在科学矩阵已经冻结之后，针对一组相互独立或可排队的训练 branch，用不超过约十分钟的前置测量，找到受 CPU、RAM、GPU、显存和 I/O 共同约束的安全并发度，并把决策写入 runtime provenance。

一期的产品边界是：

- 服务 DRPO 当前最急迫的 E7 CPU 与 E8 GPU 新任务；
- 不接管正在运行的 E7/E8 进程；
- 不重写现有训练器；
- 不实现多机集群调度平台；
- 不自动修改 batch、模型、方法、seed、horizon 或其他科学变量；
- 不要求第一版就能零适配地支持所有项目。

### 1.2 一期仍要为通用化保留正确边界

虽然一期落在 DRPO 仓库内，但代码必须按以下边界组织：

```text
generic core
    + resource backends
    + thin workload adapters
    + DRPO / RunSpec integration
```

通用 core 不得包含 E7/E8 的科学逻辑。E7/E8 只通过薄 adapter 声明：

- 哪些字段影响资源；
- 如何启动 bounded probe；
- 如何读取 throughput；
- 如何读取 RAM/显存和健康状态；
- 哪些 schedule 合法；
- 如何把选择结果传给现有 runner。

这样一期可以控制开发量，二期又不需要推倒重来。

### 1.3 长期演进方向

稳定后按顺序扩展：

1. E3/E4 等历史多 seed 或多方法任务；
2. E9 和未来 DRPO runner；
3. generic subprocess-grid adapter；
4. generic PyTorch / Hugging Face adapter；
5. 独立安装的跨项目 package；
6. 最后才考虑多机、异构资源和更复杂调度。

不得在一期同时实现上述全部能力。

---

## 2. 优化收益主要由任务规模决定

### 2.1 核心判断

资源自标定的价值不是由单个任务“看起来很大”决定，而主要取决于：

- total branches / tasks 数量；
- 每个 branch 的持续时间；
- 是否有多个 seeds、methods、datasets 或 coefficients；
- workload 是否会重复运行；
- 固定并发是否明显低于或高于机器有效容量；
- probe 成本能否被后续节省的 wall-clock 摊薄。

任务越多、单任务越长、重复次数越多，自标定的潜在收益越大。

任务很少时，即使找到了更优并发，节省时间也可能不足以覆盖 probe 成本。此时正确行为不是强制探测，而是显式使用 verified cache、保守 fixed schedule 或小任务豁免。

### 2.2 ROI gate

在运行 probe 前先做成本收益门禁：

```text
expected_savings
    = estimated_runtime_at_fallback
    - estimated_runtime_at_candidate

probe_is_worthwhile
    only if expected_savings > probe_cost × minimum_payback_ratio
```

一期不要求精确预测，但至少使用：

- branch count；
- 单 branch 粗略时长；
- fallback 并发；
- 历史 profile；
- probe hard time budget。

建议默认：

```yaml
roi_policy:
  minimum_branch_count: adapter_defined
  minimum_expected_run_seconds: adapter_defined
  maximum_probe_fraction_of_expected_run: 0.10
  minimum_payback_ratio: 2.0
```

含义是：

- probe 默认不应超过预计正式总时长的 10%；
- 预计收益至少覆盖 probe 成本的 2 倍；
- adapter 可以按 workload 调整 branch-count 门槛；
- 任何豁免都必须写入 `RUNTIME_SELECTION.json`，不能静默跳过。

### 2.3 小任务处理

以下情况可直接跳过 cold probe：

- branch 数量低于 adapter 预登记阈值；
- 预计完整任务时间短于 cold-probe budget；
- exact matching profile 已通过 60–120 秒短验证；
- 合法调度自由度为零；
- 资源池已被其他正式任务占用，无法获得代表性测量窗口。

跳过 probe 不代表不记录资源。仍必须写明：

```text
mode = fixed | cached | exempt
reason = ...
selected_schedule = ...
```

---

## 3. 资源模型：CPU、RAM、GPU、显存和 I/O 共同决定上限

### 3.1 安全并发不是 CPU 单变量

一期必须把安全并发定义为多种约束的最小值：

```text
safe_concurrency
    = min(
        cpu_limit,
        host_memory_limit,
        gpu_slot_limit,
        gpu_memory_limit,
        io_limit,
        file_descriptor_limit,
        machine_policy_limit
      )
```

任何一个维度先到达上限，都必须停止增加并发。

因此：

- CPU 还有余量但 RAM 不足，不能增加 workers；
- GPU utilization 不高但显存接近峰值，不能同卡加进程；
- RAM 足够但 swap thrash 或 I/O 饱和，也不能继续增加；
- 物理机器资源很多但 cgroup/cpuset 限制较小，必须按实际可见限制计算。

### 3.2 Host RAM 是一期硬约束

一期必须采集并记录：

- physical RAM；
- cgroup/container memory limit；
- `MemAvailable`；
- baseline RSS；
- per-branch RSS/PSS 或可获得的近似；
- peak resident memory；
- page cache 变化；
- shared memory / `/dev/shm`；
- pinned memory；
- tmpfs 使用；
- swap total/free；
- swap-in / swap-out rate；
- OOM kill / process exit evidence。

候选并发的内存可行性至少按以下保守关系估计：

```text
required_host_memory
    = baseline_reserved
    + concurrent_branches × per_branch_peak_increment
    + fragmentation_and_uncertainty_headroom
```

不能只用平均 RSS。优先使用：

- probe 窗口中的 peak；
- 多个采样窗口的高分位；
- workload phase 的最大值；
- 至少 10%–20% 的 host-memory headroom，具体值由 policy 配置。

### 3.3 Swap 不是可用容量扩展

一期默认：

- 发生持续 swap-in/swap-out 的候选判为不健康；
- swap 明显上升时不继续扩大并发；
- OOM、kernel kill、allocator failure 或 worker exit 的候选直接排除；
- 不能因为进程尚未退出，就把严重 memory pressure 视为可接受。

`swap_used > 0` 本身不一定失败，因为机器可能已有历史 swap 占用；判断重点是 probe 期间的增量和 swap activity。

### 3.4 GPU 显存也是硬约束

GPU adapter 必须采集：

- total / free VRAM；
- process-level allocated/reserved memory；
- peak allocated/reserved memory；
- CUDA OOM；
- allocator fragmentation signal；
- device utilization；
- power / temperature；
- 当前设备上其他进程占用；
- MIG、exclusive mode 或其他 device policy；
- pinned host memory；
- training、generation、evaluation、checkpoint/reload 的阶段峰值。

一期默认每张 GPU 最多一个训练进程。即使训练阶段显存有余量，只要 generation/evaluation 阶段可能 OOM，也不得允许同卡多进程。

### 3.5 阶段峰值

一个 workload 的资源 profile 至少按以下阶段记录：

```text
startup / model load
training steady state
evaluation
generation or rollout
checkpoint save/reload
terminal evaluation
```

production schedule 必须满足所有必经阶段的最大需求，而不是只满足最稳定的 training window。

一期不做复杂的 phase-dynamic rescheduling。它只使用最坏必经阶段决定静态安全上限。

### 3.6 其他容量约束

一期至少监控：

- CPU user/system/idle/iowait；
- load average；
- disk read/write throughput；
- disk utilization / queue；
- open file descriptors；
- process count；
- `/dev/shm`；
- queue depth；
- branch failure rate。

NUMA、CPU affinity、BLAS threads、dataloader threads 在一期只检测和记录，不作为自动搜索变量，除非后续单独授权。

---

## 4. 必须分开的四类身份

### 4.1 Scientific identity

描述“实验验证什么”，包括：

- experiment ID / claim；
- datasets / environments；
- methods；
- seeds；
- scientific hyperparameters；
- horizon；
- evaluation protocol；
- terminal audit requirements。

它决定结果能否比较和支持何种 claim。

### 4.2 Workload resource fingerprint

描述“一个 branch 如何消耗资源”，包括：

- entrypoint / algorithm family；
- update structure；
- model architecture；
- parameterization；
- batch / sequence length；
- precision / quantization；
- gradient checkpointing；
- data access mode；
- evaluation/generation path；
- fixed thread environment；
- adapter version。

它不包含 total branch count、seed 或完整科学矩阵。

### 4.3 Machine/runtime fingerprint

分为两层：

**Static machine/software fingerprint**

- CPU model；
- actual cpuset；
- physical cores / SMT / NUMA；
- host RAM 与 cgroup limit；
- GPU type/count/MIG/exclusive mode；
- driver/CUDA/PyTorch/Python/kernel；
- filesystem/mount type；
- adapter/core version。

**Dynamic load snapshot**

- 当前可用 RAM；
- 当前 GPU 占用；
- 当前 CPU/load；
- swap activity；
- I/O background load；
- 其他进程干扰。

静态 fingerprint 决定 profile 是否可能复用；动态 snapshot 决定本次是否必须短验证、降级或暂缓。

### 4.4 Runtime selection / execution identity

描述本次实际如何运行：

- selected concurrency；
- device allocation；
- queue order；
- cache/probe provenance；
- safety headroom；
- fallback；
- schedule epoch；
- runtime monitor；
- actual schedule 与 selected schedule 是否一致。

建议分别保存：

```text
scientific_identity_sha256
resource_fingerprint_sha256
runtime_selection_sha256
execution_identity_sha256
```

runtime schedule 必须进入 provenance，但不得改变 scientific identity。

---

## 5. 字段分级

### 5.1 Hard fields：变化后必须重新 cold probe

典型字段：

- trainer entrypoint / algorithm family；
- actor/critic 更新次数或其他 update structure；
- 模型宽度、深度、hidden size、参数量级；
- LoRA、QLoRA、full-parameter；
- batch、micro batch、gradient accumulation；
- sequence length / max tokens；
- precision、quantization、gradient checkpointing；
- negative sample count、replay fanout、auxiliary heads；
- HDF5、mmap、内存缓存等 data access mode；
- evaluation / generation 是否启用；
- fixed thread environment；
- GPU device type/count；
- CPU/cpuset、RAM/cgroup limit、NUMA；
- 单 branch 是否独占 GPU；
- adapter/core schema version。

### 5.2 Soft fields：允许复用，但必须短验证

典型字段：

- dataset ID 或有限维度变化；
- evaluation interval / episodes；
- checkpoint frequency；
- logging frequency；
- mount/path 变化；
- 同机背景负载；
- adapter 明确允许的小范围 shape 变化。

Soft 容差必须由 adapter 声明，不能由 core 猜测。

### 5.3 Ignored scientific fields：不触发资源重新标定

在计算路径不变时通常包括：

- seed；
- training steps / horizon；
- alpha；
- taper coefficient；
- reward/loss scalar weight；
- threshold；
- display label；
- branch order；
- total seeds、methods、branches。

例外：若字段会改变控制流、张量规模、更新次数或 generation 路径，必须升级为 hard/soft field。

### 5.4 Total branch count 的特殊位置

total branch count 不进入单 branch resource fingerprint，但必须进入本次 lifecycle metadata，因为它影响：

- 是否值得 probe；
- 预计 wall-clock；
- queue depth；
- disk/artifact 预算；
- memory leak 累积风险；
- profile 的经济价值。

---

## 6. 一期最小架构

### 6.1 Core

一期 core 负责：

- descriptor normalization；
- workload/machine fingerprint；
- cache validation；
- bounded candidate search；
- memory feasibility；
- throughput statistics；
- conservative selection；
- standard artifacts；
- atomic cache write；
- lock 和 stale-entry handling。

core 不得 import E7/E8 科学模块。

### 6.2 Backends

一期只实现：

```text
cpu_process_pool
cuda_device_pool
```

CPU backend 负责：

- CPU/load/iowait；
- RAM/swap/shared-memory；
- process-level metrics；
- active subprocess slots。

CUDA backend 负责：

- device inventory；
- VRAM；
- utilization；
- one-process-per-device slot；
- phase peak；
- OOM evidence。

### 6.3 Workload adapters

一期首批：

```text
synthetic_resource_adapter_v1
e7_canonical_cpu_v1
e8_countdown_cuda_v1
```

adapter 接口至少包含：

```python
class WorkloadAdapter:
    def resource_projection(self, config): ...
    def candidate_schedules(self, machine, policy): ...
    def launch_probe(self, schedule, work_dir, time_budget): ...
    def read_metrics(self, work_dir): ...
    def validate_health(self, metrics): ...
    def production_overrides(self, schedule): ...
```

### 6.4 DRPO integration

一期把 autotuner 作为 **preflight selection layer**：

```text
RunSpec claim
→ resource preflight
→ RUNTIME_SELECTION.json
→ pass explicit args/env to existing runner
→ existing runner owns branch plan/resume/execution
```

不得另造一个与 RunSpec、E7 runner、E8 runner竞争的通用任务队列系统。

### 6.5 建议一期文件布局

```text
src/drpo/runtime_resources/
  __init__.py
  models.py
  fingerprint.py
  machine.py
  memory.py
  cache.py
  probe.py
  selection.py
  artifacts.py
  backends/
    cpu.py
    cuda.py
  adapters/
    synthetic.py
    e7_canonical_cpu.py
    e8_countdown_cuda.py

scripts/agent/probe_runtime_resources.py
configs/runtime_resource_autotune_policy.yaml
schemas/runtime_resource_profile.schema.json
schemas/runtime_resource_selection.schema.json
docs/runtime_resource_autotuning.md

tests/test_runtime_resource_fingerprint.py
tests/test_runtime_resource_memory.py
tests/test_runtime_resource_probe.py
tests/test_runtime_resource_selection.py
tests/test_runtime_resource_cache.py
tests/test_runtime_resource_e7_adapter.py
tests/test_runtime_resource_e8_adapter.py
tests/test_runtime_resource_runspec.py
```

一期仍位于 DRPO 仓库内，不立即拆成独立 PyPI package。core 的 import 边界和 schema 必须支持二期抽取。

---

## 7. 时间预算：必须在十来分钟内结束

### 7.1 Cache hit

目标额外开销：

```text
static fingerprint check: seconds
short validation: 60–120 seconds
total: 1–2 minutes
```

### 7.2 Cold probe

默认 SLA：

```yaml
probe_time_policy:
  default_budget_seconds: 600
  absolute_hard_cap_seconds: 900
  cache_validation_seconds: 120
```

正常 cold probe 目标 7–10 分钟；任何情况下不得无界运行几个小时。

建议流程：

1. 机器与缓存检查：数秒；
2. baseline schedule：约 1–2 分钟；
3. 2–3 个递增候选：约 3–5 分钟；
4. knee 附近复测：约 1–2 分钟；
5. selection/artifact：数秒。

达到以下任一条件提前结束：

- throughput 增益低于阈值；
- memory/VRAM 逼近安全线；
- swap、I/O 或 tail latency 越界；
- OOM/failure；
- 测量波动过大且剩余预算不足；
- 已达到 policy 上限；
- 达到 hard time budget。

### 7.3 极慢启动 workload

若模型加载或初始化本身接近 probe budget：

- 尽量在一个存活进程内测试多个合法 schedule；
- 优先使用历史 profile + short validation；
- 记录 startup cost；
- 超过硬预算后使用保守 fallback；
- 不为追求完美峰值无限延长 probe。

### 7.4 时间预算与任务规模联动

短任务不得默认花十分钟标定。probe budget 应满足：

```text
probe_budget
    <= min(
        configured_hard_cap,
        expected_total_runtime × maximum_probe_fraction
      )
```

若不满足，进入 cache/fixed/exempt 路径。

---

## 8. Probe 搜索与选择

### 8.1 先做合法性和内存上界筛选

在启动真实候选前：

1. 读取 actual cpuset/cgroup/device limits；
2. 估计单 branch peak RAM/VRAM；
3. 生成不超过 hard capacity 的候选；
4. 排除违反 one-process-per-GPU、memory headroom 或 machine policy 的候选。

### 8.2 候选搜索

CPU 可以使用 bounded geometric search：

```text
baseline → approximately 2× → approximately 3× → local refinement
```

GPU 一期只在合法 device slots 上搜索：

```text
1 GPU slot → subset → all allowed GPU slots
```

具体数值由 machine 和 adapter 生成，不能把 `60/120/180/240` 或 `1/2/4/8` 写成跨机器常数。

### 8.3 每档测量

每个候选至少：

- warm-up；
- 2–3 个短 measurement windows；
- aggregate throughput；
- per-branch throughput；
- RSS/available memory/swap；
- GPU peak memory；
- I/O；
- failure/OOM；
- measurement variability。

选择统计量优先使用：

- median throughput；
- p10/p90 或 coefficient of variation；
- conservative lower bound。

高噪声候选在预算允许时复测；预算不足时宁可选择较小候选。

### 8.4 选择规则

1. 排除失败、OOM、持续 swap、严重 I/O wait、显著 tail slowdown 或 phase-memory 不安全的候选。
2. 计算健康候选的 conservative aggregate throughput。
3. 找到接近峰值的最小资源点。
4. 使用一个统一的保守规则，不同时重复“95%–97% 峰值”与无条件再减 10% 两次保守。
5. 输出：

```text
measured_peak_schedule
production_schedule
fallback_schedule
selection_reason
```

建议默认使用：

```yaml
selection:
  near_peak_fraction: 0.95
  choose_smallest_near_peak: true
  additional_headroom: adapter_defined
```

`additional_headroom` 只用于未被 near-peak 规则覆盖的 memory/variance 风险，不能机械重复降档。

---

## 9. E7 CPU 一期接入

### 9.1 当前迁移事实

现有 E7 路径把 `max_workers = 60` 写入：

- config；
- validator；
- normalized argv；
- execution plan；
- RunSpec success criteria；
- 部分 run identity/provenance。

因此一期不能只在外面计算一个新数字。必须明确拆分：

```text
scientific identity
runtime selection
execution identity
```

并对旧 fixed-60 RunSpec 保持原样，不得静默改变已登记或已启动任务。

### 9.2 E7 一期只调一个变量

一期 E7 adapter 只搜索：

```text
active subprocess count
```

以下保持冻结：

- `OMP_NUM_THREADS`；
- `MKL_NUM_THREADS`；
- `OPENBLAS_NUM_THREADS`；
- dataloader threads；
- CPU affinity；
- NUMA placement；
- batch；
- scientific matrix；
- branch order semantics。

这样可以最大限度降低数值轨迹和实现复杂度风险。

### 9.3 E7 内存门禁

候选必须同时满足：

- host available-memory headroom；
- per-branch peak RSS headroom；
- no OOM/worker death；
- no sustained swap activity；
- no excessive disk pressure；
- queue/resume state remains valid。

### 9.4 E7 使用顺序

1. synthetic core 通过；
2. 在独立空闲窗口运行 E7 shadow probe；
3. 只生成建议，不接管在跑任务；
4. 与固定 60 workers 做 throughput 和 memory 对比；
5. 通过科学不变性与 resume 测试后，下一份新 RunSpec 才能 opt-in auto；
6. 旧 RunSpec 继续 fixed。

---

## 10. E8 GPU 一期接入

### 10.1 一期调度变量

一期 E8 adapter 只选择：

```text
active GPU device slots
```

默认：

```text
max_processes_per_device = 1
```

一期不做同卡多训练进程，不自动修改：

- micro batch；
- gradient accumulation；
- LoRA/full-parameter；
- precision；
- sequence length；
- generation parameters；
- scientific matrix。

### 10.2 E8 可能没有很大优化空间

当：

- tasks 数量不多；
- 每个任务天然独占一张 GPU；
- GPU 数量与并发任务相等；
- 所有 GPU 已被充分使用；

autotuner 可能只确认当前 schedule，而不会产生明显加速。这是正确结果，不是功能失败。

当 tasks 很多、GPU 数有限，收益主要来自：

- 自动选择可用设备集合；
- 避免错误同卡并发；
- 根据 phase peak 阻止 OOM；
- 缓存验证和快速复用；
- 减少空闲 slot 和人工猜测。

### 10.3 E8 phase-aware gate

必须至少验证：

- model load；
- training；
- validation；
- generation/rollout；
- checkpoint/terminal eval。

只测 training steady-state 不足以宣布 schedule 安全。

---

## 11. RunSpec 接入策略

### 11.1 保留 fixed 语义

现有 RunSpec v1 的固定 worker/device 语义不能被偷偷重解释。

一期接入应明确三种模式：

```yaml
runtime_resources:
  mode: fixed | auto | exempt
```

- `fixed`：使用预登记 schedule；
- `auto`：运行 cache validation / bounded probe；
- `exempt`：小任务或无合法优化自由度，必须有机器可验证理由。

### 11.2 版本迁移

实现时优先考虑：

- RunSpec v2 承载 autotune；
- v1 保持旧 fixed 行为；
- 或在经过 schema review 后，为 v1 增加完全 opt-in、默认关闭且向后兼容的扩展。

不得让旧 RunSpec 因升级 validator 而改变运行行为。

### 11.3 示例

CPU：

```yaml
runtime_resources:
  mode: auto
  adapter: e7_canonical_cpu_v1
  cache_policy: validate_then_reuse
  probe_budget_seconds: 600
  absolute_probe_cap_seconds: 900
  host_memory_headroom_fraction: 0.15
  fallback_workers: 60
```

GPU：

```yaml
runtime_resources:
  mode: auto
  adapter: e8_countdown_cuda_v1
  device_ids: [0, 1, 2, 3]
  max_processes_per_device: 1
  probe_budget_seconds: 600
  gpu_memory_headroom_fraction: 0.12
  fallback_device_slots: 4
```

### 11.4 Validator 必须拒绝

- adapter 不存在；
- schedule 超出 actual machine/cgroup/device policy；
- memory headroom 无法计算；
- probe 修改科学变量；
- probe 使用 formal held-out seeds；
- probe 与 formal output root 重叠；
- production selection 未写 provenance；
- auto 静默退回 fixed；
- actual executor 参数与 selection 不一致；
- 新 RunSpec 绕过已要求的 autotune 流程。

---

## 12. 标准 artifacts 与缓存

每次资源决策至少生成：

```text
RESOURCE_FINGERPRINT.json
MACHINE_FINGERPRINT.json
RESOURCE_PROBE.json
RESOURCE_PROFILE.json
RUNTIME_SELECTION.json
RESOURCE_MONITOR.jsonl
```

### 12.1 `RESOURCE_FINGERPRINT.json`

包含：

- normalized hard/soft fields；
- resource hash；
- adapter/core version；
- scientific identity reference；
- 不把完整科学配置混进 resource hash。

### 12.2 `MACHINE_FINGERPRINT.json`

包含：

- static machine/software fingerprint；
- cgroup/cpuset；
- RAM/swap；
- GPU/MIG/device policy；
- filesystem；
- dynamic load snapshot reference。

### 12.3 `RESOURCE_PROBE.json`

包含：

- candidate schedules；
- time budget；
- warm-up/measurement windows；
- throughput distribution；
- host-memory metrics；
- GPU-memory metrics；
- I/O/swap；
- failures/OOM；
- stop reason；
- probe isolation proof。

### 12.4 `RESOURCE_PROFILE.json`

包含：

- measured peak；
- selected production schedule；
- fallback；
- memory/VRAM headroom；
- phase profiles；
- validity conditions；
- TTL；
- profile schema/version。

### 12.5 `RUNTIME_SELECTION.json`

每次正式 run 都必须有：

- fixed/auto/exempt；
- cache hit/miss；
- validation result；
- ROI decision；
- selected/fallback schedule；
- exact profile hash；
- relation to scientific/execution identity；
- schedule epoch；
- any contraction/backoff reason。

### 12.6 Cache safety

server-local cache 建议放在：

```text
.runspec_state/resource_profiles/
```

必须实现：

- atomic write；
- lock；
- schema validation；
- symlink/path safety；
- stale/partial entry rejection；
- adapter/core version binding；
- machine fingerprint binding；
- TTL；
- 不覆盖最后一个 verified profile。

跨 workspace 共享必须显式设计，不能默认多个 checkout 无锁共写。

---

## 13. 运行期监控与安全收缩

一期正式运行持续记录：

- aggregate/per-branch throughput；
- CPU/load/iowait；
- RSS/available memory；
- swap activity；
- disk throughput/utilization；
- GPU utilization/VRAM/power/temperature；
- queue depth；
- active/completed/failed；
- actual schedule；
- schedule epoch。

一期允许的在线动作仅限**安全收缩**：

- 停止接纳新的 branch；
- 当前 branch 正常结束后减少后续 slots；
- 记录新的 schedule epoch；
- 触发人工/runner-defined fallback。

一期禁止：

- kill 正在运行的健康 branch；
- 修改 `ThreadPoolExecutor._max_workers` 私有字段；
- 无原子 claim 时启动第二个 scheduler；
- 未授权在线扩容；
- 在同一 work directory 启动竞争写入者。

---

## 14. Probe 与科学实验隔离

资源 probe 不是实验结果，必须：

- 不使用 formal held-out seeds；
- 不进入 method ranking；
- 不写正式 aggregate；
- 不覆盖正式 branch 目录；
- 不复用 probe checkpoint 作为正式初始化；
- 不改变 methods/data/batch/horizon/threshold/evaluation；
- 不把 throughput 当作收敛证据；
- 将 OOM/failure 解释为 schedule/resource failure，而不是算法科学失败。

建议：

- 独立 `resource_probe` seed namespace；
- 独立 work directory；
- 独立 artifact type；
- TTL；
- 只保留小型 metrics/provenance，不保留大模型权重。

---

## 15. 分阶段开发计划

### Phase 0：方案与治理锁定

本 PR 只交付设计文档。

后续实现前交付：

- approved scope；
- governance claim；
- stage impact classification；
- authorization record；
- rollback plan；
- exact file allowlist。

当前用户对本文档的修订授权不自动等于实现或 default-policy cutover 授权。

### Phase 1A：最小 core + synthetic（先做）

目标：不接真实 E7/E8 runner，完成最小可验证内核。

交付：

- typed models/schema；
- machine/RAM/GPU discovery；
- fingerprint；
- cache；
- bounded probe；
- robust selection；
- memory/swap/OOM gate；
- synthetic adapter；
- CLI；
- unit/integration tests；
- standard artifacts。

范围限制：

- 不改 RunSpec 默认；
- 不改 E7/E8 runner；
- 不碰 formal output；
- 不接管当前任务。

预计开发量：约 3–5 人日。

### Phase 1B：E7/E8 shadow adapters

目标：使用真实计算路径做短 probe，但先只输出 recommendation。

交付：

- E7 active-subprocess adapter；
- E8 one-process-per-GPU adapter；
- phase-memory metrics；
- real shadow profile；
- fixed-vs-selected engineering comparison；
- scientific invariance checks；
- no-interference evidence。

预计开发量：约 3–6 人日。

若服务器有正在运行的正式任务，只开发和跑 synthetic；真实 shadow 等待隔离资源或空闲窗口，不中断现有任务。

### Phase 1C：下一批新 RunSpec opt-in

目标：让**尚未启动的新 E7/E8 RunSpec**显式使用 auto。

交付：

- RunSpec schema/version decision；
- validator；
- preflight hook；
- exact runtime selection injection；
- resume/failure tests；
- provenance；
- rollback to fixed。

预计开发量：约 2–4 人日。

一期总开发量：约 8–15 人日。它比跨项目平台小得多，但仍包括真实 adapter、内存门禁、测试和治理，不能按一个简单脚本估算。

### Phase 2：DRPO 内部通用化

通过一期真实使用后扩展：

- E3/E4 多 seed adapter；
- E9 adapter contract；
- generic subprocess-grid adapter；
- adapter registration；
- 统一文档；
- 5–8 个真实任务的 observation ledger；
- 根据真实 blocker/效率证据优化 gate。

预计新增：约 8–15 人日。

### Phase 3：跨项目 SDK

只有 Phase 2 稳定后再做：

- 从 `src/drpo` 抽取独立 package；
- plugin discovery；
- generic PyTorch/Hugging Face adapters；
- versioned public schemas；
- install/upgrade path；
- cross-project examples；
- compatibility matrix；
- standalone docs/tests。

预计新增：约 10–20 人日。

### Phase 4：重型能力，另立 claim

不属于当前方案自动授权：

- 多机 distributed scheduler；
- Slurm/Kubernetes integration；
- heterogeneous weighted scheduler；
- same-GPU multi-process；
- phase-dynamic rescheduling；
- online arbitrary scale-up；
- 自动修改 batch/model parallelism。

---

## 16. 一期 Demo 与测试

### 16.1 Deterministic synthetic adapter

必须模拟：

- 低并发近线性扩展；
- knee 后 throughput 饱和；
- 过载后单任务变慢；
- CPU-bound；
- I/O-bound；
- host-memory-bound；
- GPU-memory-bound；
- swap thrash；
- OOM/failure；
- noisy measurements；
- cache hit；
- soft drift；
- hard invalidation；
- phase peak。

### 16.2 Unit tests

至少覆盖：

- ignored scientific field 不改变 resource fingerprint；
- hard field 改变触发 cold probe；
- soft drift 触发 short validation；
- stale/foreign machine cache 被拒绝；
- cgroup/cpuset 优先于物理机器总量；
- host RAM 限制低于 CPU 限制时由 RAM 决定并发；
- GPU VRAM 限制低于 device-count 时由 VRAM 决定；
- swap-thrash candidate 被排除；
- OOM/failure candidate 被排除；
- cold search 找到 synthetic knee；
- noisy candidate 使用稳健统计；
- small-task ROI gate 跳过 cold probe；
- probe hard cap 生效；
- runtime schedule 不进入 scientific identity；
- runtime selection 完整进入 provenance；
- atomic cache/lock；
- symlink/path safety；
- fixed/auto/exempt schema；
- resume 不重复 claim；
- schedule contraction 只影响未来 branch。

### 16.3 Scientific invariance test

对同一个固定 seed/config：

```text
schedule A: concurrency = 1
schedule B: concurrency = N
```

必须验证：

- scientific identity 完全一致；
- branch config 完全一致；
- 允许的 runtime fields 是唯一差异；
- 科学输出在预登记确定性/数值容差内等价；
- branch ordering 不改变 seed assignment；
- resume 后结果不重算或错配。

### 16.4 真实 E7 acceptance

至少：

- 完成短 probe；
- 记录 throughput curve；
- 记录 peak RSS/available RAM/swap；
- 与 fixed 60 对比；
- 不污染正式目录；
- 不修改正在运行的 RunSpec；
- recommendation 可解释；
- 后续 opt-in run 能按 selection 启动和 resume。

### 16.5 真实 E8 acceptance

至少：

- 一次真实 GPU short probe；
- one-process-per-GPU；
- 记录 training 与 evaluation/generation peak VRAM；
- OOM candidate fail closed；
- 不修改 scientific batch/LoRA/precision；
- 无真实 GPU 验证前不得宣布 E8 production-ready。

---

## 17. Fail-closed、fallback 与 rollback

### 17.1 Fail-closed

以下情况不得自动开始大规模正式 run：

- fingerprint 无法生成；
- adapter/RunSpec 不匹配；
- actual cgroup/cpuset/device limit 无法确认；
- RAM/VRAM headroom 无法确认；
- cache schema/machine identity 不匹配；
- probe 使用 formal seed 或正式 work directory；
- 所有候选失败/OOM/swap thrash；
- selected schedule 超出 policy；
- artifacts 写入失败；
- actual schedule 与 selection 不一致；
- provenance 无法绑定 commit；
- branch 已被其他 executor claim；
- 当前背景负载使测量不具代表性且无 verified fallback。

### 17.2 Fallback

允许：

- 上一次 verified production schedule；
- adapter 固定保守值；
- 小任务 fixed/exempt；
- probe 超时后的保守 candidate；
- 运行期停止接纳新 branch 并缩减后续 slots。

Fallback 必须写入 `RUNTIME_SELECTION.json`，包含原因和证据。

### 17.3 Rollback

必须能够：

- RunSpec 设回 `mode: fixed`；
- 恢复旧 fixed-worker/device 路径；
- 停用 preflight hook；
- 保留但不使用 resource profiles；
- 不改变 scientific identity；
- 不删除历史 cache、profile、selection 或失败 evidence；
- 不修改已完成或正在运行实验。

---

## 18. 一期验收标准

一期只有同时满足以下条件，才能称为 DRPO opt-in ready：

1. cold probe 正常目标 7–10 分钟，hard cap 不超过 15 分钟。
2. cache hit validation 目标 1–2 分钟。
3. task-count/expected-runtime ROI gate 能正确跳过不值得 probe 的小任务。
4. CPU、host RAM、swap、GPU、VRAM、I/O 都进入 capacity decision。
5. host-memory-bound 和 GPU-memory-bound synthetic 测试都能把并发降到安全值。
6. E7 一期只改变 active subprocess count。
7. E8 一期保持 one-process-per-GPU。
8. 固定线程、batch、model、seed、horizon 和科学矩阵不被 autotuner 修改。
9. resource schedule 不改变 scientific identity。
10. profile/selection/artifacts 可审计、可 resume、可 rollback。
11. E7/E8 真实 shadow 未污染正式结果。
12. fixed/auto/exempt 语义明确，旧 RunSpec 行为不变。
13. targeted tests、全仓 tests、handoff authority、formal channel validator 和 governance stage validator 按修改范围通过。
14. 没有真实硬件证据的 backend 不得宣布 production-ready。
15. default-policy cutover 尚未自动发生。

---

## 19. 一期明确不做

- 根据机器容量自动增加 seeds、methods、datasets 或 coefficients；
- 修改 scientific batch、模型、训练 horizon；
- 调整 optimizer 或 loss；
- 自动搜索 BLAS/OpenMP/dataloader/affinity/NUMA；
- 同卡多训练进程；
- 在线 kill 健康 branch；
- 在线任意扩容；
- 修改私有 executor 字段；
- 新建第二套 branch claim/queue 系统；
- 自动判断方法排名、收敛、稳态或论文 claim；
- 一步到位实现跨项目平台；
- 多机、Slurm、Kubernetes 或 weighted heterogeneous scheduling。

---

## 20. 本方案锁定的设计结论

除非用户重新决策，后续实现默认继承：

1. 优化收益主要由 branch 数量、单 branch 时长和重复运行频率决定。
2. 小任务可以显式 fixed/cached/exempt，不强制 cold probe。
3. RAM、swap、共享内存和 GPU 显存都是硬容量约束，不是附加监控项。
4. 安全并发由 CPU、RAM、GPU、VRAM、I/O 和 policy 中最先到达的约束决定。
5. 一期先服务 E7/E8，不一步到位做跨项目平台。
6. 一期 core 与 adapter 分层，为二期抽取保留边界。
7. 一期 E7 只调 active subprocess count。
8. 一期 E8 只调 active GPU slots，并保持 one-process-per-GPU。
9. cold probe 正常目标 7–10 分钟，hard cap 15 分钟；cache hit 1–2 分钟。
10. 极慢启动 workload 超时后使用 verified/fixed conservative fallback，不无限延长。
11. scientific matrix 先冻结；资源系统只决定并行和排队。
12. resource schedule 属于 runtime provenance，不属于 scientific identity。
13. 不接管、修改或中断正在运行的 E7/E8。
14. 旧 fixed RunSpec 保持旧语义；auto 先 opt-in。
15. default-policy cutover 必须在真实 E7/E8 验证后另行授权。
16. 跨项目 SDK 是 Phase 3，不是一期验收前置条件。

---

## 21. 下一 implementation session 的首个任务

下一 session 不应直接修改 E7 的固定 60 workers，也不应尝试接管当前 E7/E8 运行。首个实现任务应是：

```text
基于当时 current main，完成 GOV-RUNTIME-RESOURCE-AUTOTUNE-01 的
approved scope、stage-impact、authorization 与 rollback 判断。

随后只实现 Phase 1A：
- generic typed core；
- machine/cgroup/cpuset discovery；
- host RAM/swap 与 GPU/VRAM inventory；
- fingerprint；
- atomic profile cache；
- bounded probe；
- robust conservative selection；
- synthetic adapter；
- standard artifacts；
- targeted tests。

不得在 Phase 1A 修改 E7/E8 runner、RunSpec 默认行为、
正式实验目录、科学变量或正在运行的任务。
```

Phase 1A 审查通过后，再为 E7/E8 分别提交 Phase 1B shadow-adapter PR；真实 shadow 通过后，才进入 Phase 1C opt-in RunSpec 集成。
