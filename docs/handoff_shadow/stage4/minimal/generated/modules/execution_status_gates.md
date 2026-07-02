# Current execution status and gates

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `execution_status_gates`
- Responsibility: Provide the current experiment evidence states, execution ordering, and formal launch gates.
- Content contract topics: `formal_vs_development_evidence`, `single_registered_execution_order`, `blocked_requires_protocol_or_predecessor`, `current_formal_route`, `no_unregistered_experiment`
- Deduplicated overlapping source chunks: 0
- Source hash: `691bba82b3ffad8a9373e9db2fd5b052b79406b639eef9eed4ff985bbe28e2ca`

## Content contract evidence

| Topic | Required semantic responsibility | Authoritative source | Matched phrase |
|---|---|---|---|
| formal_vs_development_evidence | Keep development, smoke, and static evidence separate from formal scientific results. | docs/handoff.md: # 5. 当前真实完成状态 -> # 7. 变量治理 | 只提供 development 证据 |
| single_registered_execution_order | Preserve the registered global execution order instead of selecting an easier route ad hoc. | docs/handoff.md: # 5. 当前真实完成状态 -> # 7. 变量治理 | 接下来唯一执行顺序 |
| blocked_requires_protocol_or_predecessor | Preserve explicit predecessor and protocol gates for blocked experiments. | docs/handoff.md: # 5. 当前真实完成状态 -> # 7. 变量治理 | review-required + blocked |
| current_formal_route | Keep the current formal route anchored to the latest handoff override. | docs/handoff.md: # 5. 当前真实完成状态 -> # 7. 变量治理 | 当前直接进入已实现且 registry 为 ready/active 的 `EXT-H-E7-Q2` |
| no_unregistered_experiment | Forbid unregistered experiments or silent changes to the route. | docs/handoff.md: # 5. 当前真实完成状态 -> # 7. 变量治理 | 任何新增实验必须先说明它补哪一个 claim |

## Source 1: docs/handoff.md: # 5. 当前真实完成状态 -> # 7. 变量治理

# 5. 当前真实完成状态

| 实验 | 旧环境结果 | 真正统一环境状态 | 论文可用状态 |
|---|---|---|---|
| E1 | Product 环境已完成，逻辑严密 | **C-U1 正式 20-seed 已完成**：positive-trained full-gradient far/near 9.093×，aggregate 10.072×；advantage 1.000× | 正式机制识别完成；数值倍率仅限本环境 |
| E2 | 零散 positive-only 曲线 | **C-U1 正式 20-seed 已完成**：20/20 通过稳态与 2× 延长审计；phantom gradient 增长 28.93× | 正式长期验证完成 |
| E3 | Product/Collapse 环境与旧 SGD C-U1 结果保留 provenance | **`C-U1-E3-ADAM-RERUN` 已完成并交付**：固定方差 Baseline/Near-zero 20/20 任务崩溃，Far-zero/Far-cap 0/20；可学习方差 Baseline/Near-zero 20/20 support contraction，远场控制 0/20；NaN/Inf 0/220 | **已长期验证，论文可用**；主文采用四方法 fixed-variance 因果链，learnable-variance 作互补 panel/附录 |
| E4 | 独立 Extrapolation 环境；部分长程审计 | **`C-U1-E4-ADAM-RERUN` 已完成并交付**：有限步 reward 相变、过强压力任务崩溃、learnable-variance support contraction 与 4000-step controls 均完成；受益分支未通过 20/20 双 residual audit | **有限训练步数验证**；可用于有限步相变图与失稳分支，暂不可写成稳定有益 fixed point |
| E4-CONV | 无历史独立环境结果 | **4000-step 正式运行已完成**：`0.75/1.00/1.25` 目标状态分别为 15/20、16/20、15/20，剩余均 inconclusive，0 个明确相反终态，60/60 科学角色未反转 | **已长期验证（用户确认闭环）**；原 18/20 门禁未通过的事实继续保留，不等同于 20/20 fixed-point 认证 |
| E4-TAPER | seeds 0--4 独立复制实现 pilot | **正式 seeds 70--89 已完成 220/220 runs**：quadratic vs linear 在主 rho=0.25 上 20/20 更强抑制远场且 20/20 reward 更高；200 controlled/positive runs 到 8000 steps 仍无稳定候选，20 unweighted runs 触发 support boundary | **有限训练步数验证**；机制阶数 claim 可用，稳定终态和 universal method ranking 不可声称 |
| E5 | 历史解析、direct-softmax 与 20-seed 因果结果保留；旧 runner/raw artifact 未入库 | **`D-U1-E5-LONGRUN-RERUN` 已完成**：direct-softmax 参照通过，120/120 长程因果 runs 全部分类且 120/120 复现历史 qualitative class，NaN/Inf 0/120 | **已长期验证**；受控 categorical 排斥、支持边界和 near/far 因果链可用于论文，E6 语义泛化仍未完成 |
| E6 | unordered semantic categorical pilot/focused runner 与 formal runner/config 均已实现 | **`D-U1-E6-SEMANTIC-LONGRUN-01` 已完成 360/360 formal runs**：E6-A 受控 local negatives 在 alpha 0.25/0.50 上 20/20 胜过 positive-only，alpha 0.75 出现 20/20 reward 反转；E6-B task collapse 0、support boundary 120；E6-C aligned 在四方法上均 20/20 胜过 shuffled | **已长期验证**；可用于 positive-only ceiling、受控负梯度非单调收益、支持边界分离与 semantic-alignment 排他性，不能称 OOD 或 universal method ranking |
| E6-TAPER | 无正式结果 | `D-U1-E6-TAPER-01` 的 predecessor delivery 已满足，但 semantic remoteness coordinate、paired protocol、新 untouched seeds 与独立 runner 尚未冻结/实现 | 未完成、review-required + blocked；不得套用 Gaussian 二次界或自动启动 |
| E7-MECH | Hopper learned-critic 600-step probe | `EXT-H-E7-Q2` runner/config/operator/test 已在 commit `f64452a7452274a183b03c87c39b847039230c00` 实现；formal launch 仍等待 E6-TAPER 交付 | 旧 probe 仅有限步；新实现科学状态仍为 not_run/blocked |
| E7-BENCH | 无 9-task 主表 | `EXT-H-E7-BENCH-01` 已登记 D4RL MuJoCo locomotion 9-task scope | 未完成、blocked；等待 E7-MECH 与 bandit shortlist |
| E8-MECH | v4.2 平衡离线集与动态诊断实现已登记；V4.1 off-protocol 单 seed 仅保留开发 provenance | V4.2 未在 clean committed source 上完成真实 Qwen/CUDA/BF16-LoRA 运行 | 尚未运行；不得把静态/CPU 测试或 V4.1 off-protocol pilot 当正式结果 |
| E8-SCALE | 无规模结果 | `EXT-C-E8-SCALE-01` 已 planned 登记 3B 主结果与 7B 冻结确认 | 未完成、blocked；精确数据规模/seeds 待运行前冻结 |

---

<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-completion-status:START -->
**v55 E6 Semantic-Gap 结果补充：** `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 已完成 100/100 runs。32k 时 `alpha=0.25/0.50` 均 20/20 胜过 Positive-only；`alpha=1.0` 相对差距随 8k→32k 由 `-0.013741` 扩大至 `-0.061085`，20/20 失败。由于仅 45/100 terminal plateau，论文可用状态限定为有限 horizon trajectory 与 paired finite-step claim，不允许全方法稳态排名。三类失效事件分别为 0/100、0/100、0/100。
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-completion-status:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-completion-status:START -->
**v56 E6 父实验关闭判断：** E6 已达到当前论文所需的机制与泛化证据闭环：主语义 long-run 给出 Positive-only ceiling、适度负信号收益、过强压力反转与 semantic-alignment 排他性；semantic-gap successor 复现中等 alpha 收益和 `alpha=1` 随 horizon 扩大的退化；conditional-gap stress diagnostic 证明更强支持缺口下局部收益与 overall trade-off、强压力任务崩溃及控制救援。关闭的是上述父 claim，不是把所有子运行宣称为稳态，也不是冻结 universal 方法排名。
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-completion-status:END -->

# 6. 接下来唯一执行顺序

1. E1/E2/E3、E4、E4-CONV 与 E5 的既有科学状态和历史证据保持不变；原 E4 18/20 门禁失败事实继续披露。
2. **`C-U1-E4-TAPER-01` 已完成正式运行与交付。** 当前科学状态是有限训练步数验证：主 paired mechanism-order claim 得到支持，但 2× 终态门禁未通过；不得自动延长或升级为 long-run validated。
3. `D-U1-E6-SEMANTIC-PILOT-01` 已完成并交付，但只提供 development 证据，不产生论文级方法排名，也不自动冻结 E6 正式参数。
4. `D-U1-E6-SEMANTIC-LONGRUN-01` 已完成 360/360 formal runs、2x 终态审计、raw evidence 和仓库闭环；禁止复用 held-out seeds 10--29 调参或无新登记重跑。
5. 下一步先为 `D-U1-E6-TAPER-01` 单独冻结 semantic remoteness coordinate、paired method protocol、新 untouched seeds，并实现 formal runner；用户审阅前不得运行，也不得把 Gaussian 标准化距离或二次临界界直接搬到 categorical。
6. E6-TAPER 交付后，解除已实现的 `EXT-H-E7-Q2`（E7-MECH）formal gate 并启动正式运行；它只回答 Hopper learned-critic 下二次 log-scale 远场区是否真实激活并传导。
7. E7-MECH 交付后，实施 `EXT-H-E7-BENCH-01`（E7-BENCH）：D4RL MuJoCo locomotion 9 tasks，方法 shortlist/超参从受控实验冻结，不做按任务方法族重选。
8. E7-BENCH 交付后，运行 `EXT-C-E8-V4.2`（E8-MECH）真实 Qwen pilot；代码/CPU smoke 不构成实验结果。
9. E8-MECH 交付后，冻结更大固定 Countdown 数据、3B 主模型与 7B 确认协议，实施 `EXT-C-E8-SCALE-01`。
10. SBRC/Hybrid 和 entropy/target-entropy controls 仍是后续安全层与排他性消融；未另行冻结前，不插入 Linear/Quadratic/Exp 核心顺序比较，也不预设优胜。

任何新增实验必须先说明它补哪一个 claim、是否替代现有实验、是否进入本文档。

---

<!-- HANDOFF-DELTA-BLOCK:section_end:v52-execution-order-override:START -->
11. **v52 执行覆盖：** 当锁定路线进入 E8-MECH 时，执行 `EXT-C-E8-V4.3` 而不是 V4.2；当前只完成注册和代码实现，真实 Qwen/CUDA pilot 仍为 not_run。
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-execution-order-override:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-execution-order:START -->
12. **v55 执行覆盖：** Semantic-Gap 正式结果已闭环，不再等待该 successor 的 delivery。下一项仍不是直接运行 E6-TAPER，而是先冻结其 semantic remoteness coordinate、paired method protocol、全新 untouched held-out seeds，并实现独立 formal runner；完成用户审阅和 registry activation 前禁止启动。
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-execution-order:START -->
13. **v56 执行覆盖：** E6 父 claim 已关闭，`D-U1-E6-TAPER-01` 改为可选非门禁 future study；当前直接进入已实现且 registry 为 ready/active 的 `EXT-H-E7-Q2`（E7-MECH）。E7-Q2 仍为 not_run，必须先完成正式运行、终态审计、打包与交付；其后才允许冻结并实施 `EXT-H-E7-BENCH-01`。E8-MECH/V4.3 与 E8-SCALE 的相对顺序不变。
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-bank-execution-order:START -->
14. **v57 执行覆盖：** v56 的 formal 顺序不变，`EXT-H-E7-Q2` 仍是下一正式实验。用户批准的 V4.4 作为 single-seed focused pilot 可独立执行，但必须先完成自身 best/terminal audit 与结果交付，才允许讨论 online off-policy successor；不得一次性同时改变 negative-bank 密度和数据在线刷新机制。
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-bank-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-execution-order:START -->
15. **v59 执行覆盖：** formal 顺序仍由 v56/v58 控制，E7-Q2 优先级不变。V4.5 可作为独立 pilot 执行，但必须复用并校验 V4.4 frozen inputs，按 Stage A alpha、Stage B lambda、untouched-seed confirmation 顺序完成；test 只能在 selection 冻结后运行，结果必须 best/terminal 与三类事件分报后再交付。
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-local-execution-order:START -->
16. **v60 E4-TAPER 内部执行序列：** 本条只登记 TAPER track 的依赖顺序，不自动改写 v56--v59 的全局外部实验路线。选择推进该 track 时，必须按 `NEAR-RETENTION-01 -> BUDGET-MATCH-01 -> CONV-01 -> CONFIRM-01` 逐项完成 protocol freeze、实现、正式运行、终态审计、打包和交付；下一项不得在前一项交付前启动。四项目前均 blocked，第一步是另行冻结 Near-retention protocol，而不是直接运行或延长旧 E4-TAPER。
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-local-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-execution-order:START -->
17. **v61 E4-TAPER 内部执行覆盖：** `NEAR-RETENTION-01` 已从 blocked 迁移为 implemented + ready + active + not_run，允许作为当前 TAPER track 的下一项正式运行。运行必须使用 hardened guard、正式 seeds 90--109、development-only coefficient calibration 和每 5 seeds checkpoint index；raw-complete 后仍需终态审计、canonical packaging 与交付。`BUDGET-MATCH-01` 在该交付之前继续 blocked，Long-run 与 Confirmation 顺序不变。
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-execution-order:START -->
18. **v62 Countdown 执行覆盖：** formal 主顺序继续由 v56/v58/v61 控制；`EXT-H-E7-Q2` 优先级不变。V4.6 允许作为独立 guarded pilot 执行，顺序固定为 predecessor/input hash audit -> 四 cell paired training -> 全部训练结束后 test evaluation -> 2×2 paired effect/interaction -> terminal audit -> canonical artifact delivery。任何 online phase 都必须保留 collector manifest、round JSONL、fresh/stale mix 与实际 selected-bank diagnostics；smoke 或单 seed 不得称实验结果。
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-execution-order:START -->
18. **v63 E4-TAPER 内部执行覆盖：** `NEAR-RETENTION-01` 已完成正式矩阵并沉淀为有限训练步数验证；当前下一项是已冻结且已实现的 `BUDGET-MATCH-01`，正式 seeds 固定为 110--129，只允许按每一步 Adam 之前的 raw negative-gradient L2 norm 做 paired budget matching。`CONV-01` 与 `CONFIRM-01` 虽已完整登记输入输出契约、shortlist 规则和 untouched seeds，但继续 blocked；必须等待 Budget-Match 正式结果完成终态审计、打包、交付并冻结 shortlist 后，才允许实现和启动 Convergence，Confirmation 仍为最后一步。
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-execution-order:START -->
19. **v66 E4-TAPER 内部执行覆盖：** `BUDGET-MATCH-01` 已完成正式矩阵、终态审计和闭环交付，禁止无新登记重跑。`CONV-01` 仍不允许启动：先用既定规则在独立更新中生成并校验 `FROZEN_CONVERGENCE_SHORTLIST.json`，再实现读取 run_003 exact actor + Adam optimizer state 的 continuation runner；只有这两项通过并另行 activation 后才能运行。`CONFIRM-01` 与 seeds `130--149` 继续保持最后一道防火墙。
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v70-du1-e6-cartesian-taper-execution-order:START -->
10. **v70 D-U1 successor 覆盖：** `D-U1-E6-CARTESIAN-TAPER-01` 作为一个联合 formal experiment 执行，顺序固定为 environment/preflight audit → E6-Cartesian mechanism methods → preregistered TAPER methods → paired aggregation → 2× terminal audit → hardened packaging/delivery。禁止先查看正式机制结果后修改 taper family、retention、seeds 或阈值；原 `D-U1-E6-TAPER-01` 不再作为独立 runnable experiment。
<!-- HANDOFF-DELTA-BLOCK:section_end:v70-du1-e6-cartesian-taper-execution-order:END -->
