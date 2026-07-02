# Categorical D-U1 E6 shared-semantic generalization

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `categorical_e6_generalization`
- Responsibility: Cover positive-only ceiling, controlled local-negative benefit, support-boundary separation, semantic alignment, and structured support-gap successors.
- Dependencies: `global_core_governance`, `theory_methods_related_work`, `terminal_audit`, `categorical_e5_mechanism`
- Content-contract topics: none
- Owned source blocks: 5
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `D-U1-E6-SEMANTIC-LONGRUN-01`, `D-U1-E6-SEMANTIC-GAP-LONGRUN-01`, `D-U1-E6-CONDITIONAL-GAP-01`, `D-U1-E6-CARTESIAN-TAPER-01`
- `development_experiment_registrations`: `D-U1-E6-CONDITIONAL-GAP-DEV-01`, `D-U1-E6-SEMANTIC-PILOT-01`, `D-U1-E6-SEMANTIC-FOCUSED-DEV-01`, `D-U1-E6-TAPER-01`

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000002:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v50-stage3-shadow-bootstrap:START -->
> **v50 增量登记：治理 Pipeline Stage 3 `HANDOFF_DELTA.yaml` shadow mode 启动（不删除 v49 及更早内容）**
>
> - Stage 1 与当前 Stage 2 的 `closed_maintenance_only` 状态保持不变；本版只启动当前 Stage 3，不改变任何科学实验变量、seeds、阈值、结果或执行顺序。`D-U1-E6-CONDITIONAL-GAP-01` 继续保持 **not_run + implemented + ready + active**。
> - Stage 3 状态由 `ready_not_started` 迁移为 `shadow_active`。`docs/handoff.md` 在整个 shadow 期继续是唯一权威研究 Master；结构化 delta 只生成 candidate 并与人工 handoff 比较，不得替换正式 handoff。
> - 新增 `docs/handoff_delta_protocol.md`、机器策略 `docs/handoff_delta_policy.yaml`、状态机 `docs/handoff_delta_state_machines.yaml`、确定性入口 `scripts/handoff_delta_shadow.py` 与三级验收入口 `scripts/run_handoff_delta_acceptance.py`。版本 1 只允许 heading rename、heading 后插入和 section 末尾 append，不允许任意文本替换或破坏性删除。
> - Fast Gate 对每个 handoff / registry / delta 相关更新执行本地确定性 replay、base/hash、幂等、历史 ID、registry transition 与 candidate/manual exact-match 检查；禁止网络和 LLM 作为阻塞式 oracle。目标 p95 不超过 5 秒，硬上限 15 秒。
> - Standard Regression 在 schema、renderer、状态机、冲突规则、parser/index 或 operation 变化时运行，目标 60 秒。Full Acceptance 在 shadow 激活前、authority cutover 前、schema 主版本或架构变化、累计 20 次相关更新、7 天内发生相关更新的兜底周期、critical mismatch 修复后运行，目标 15 分钟。
> - 本更新使用 `GOV-STAGE3-SHADOW-BOOTSTRAP-2026-06-27/HANDOFF_DELTA.yaml` 自举 replay；candidate 与人工 v50 handoff 必须字节级一致。该自举通过只证明实现与门禁可运行，不等于 Stage 3 已验收或可以切换权威路径。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v50-stage3-shadow-bootstrap:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000002:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000003:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v51-du1-e6-semantic-gap-formal:START -->
> **v51 增量登记：`D-U1-E6-CONDITIONAL-GAP-01` 结果闭环与 `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 最小改动 formal freeze（不删除 v50 及更早内容）**
>
> - `D-U1-E6-CONDITIONAL-GAP-01` 已在 clean scientific run commit `7a70278f3d6061379c81f33e82d93ead86484908` 上完成 frozen matrix `200/200` runs、terminal audit 与 raw-complete artifact。三类事件严格分报：task-performance collapse `77/200`、support/temperature boundary `0/200`、NaN/Inf numerical failure `0/200`。全实验只有 `49/200` terminal plateau，`151/200` 为 persistent-drift-or-inconclusive，因此科学状态只登记为 **有限训练步数验证（finite-step validated）**，禁止升级为 long-run validated，并禁止稳态排名。
> - 该 group-based conditional-gap 实验保留为大缺口与强压力的 stress diagnostic。其 structured-gap local `alpha=0.5` 虽提高 withheld-block reward，却降低 overall reward；`alpha=1.5` 与 far-pressure stress 不属于后续正式方法域。该结果不得推翻旧 `D-U1-E6-SEMANTIC-LONGRUN-01` 已锁定的“适度负优势可在 overall reward 上超过 Positive-only”结论。
> - 新实验 `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 是旧 64-action shared-semantic E6 的正式最小改动 successor。继续使用 6D state、64 个 4D semantic actions、旧 `t_plus/t_star/t_minus` reward 几何、4 positive / 1 local negative / 4 far negatives、固定相等 advantage、共享 `SemanticPolicy`、fixed concentration `8.0`、Adam `lr=1e-3`、batch 128。唯一核心环境干预是在 exactly 50% contexts 上，将按该状态 reward similarity 排名前 25% 的 16 个动作从 positive/local/far 三类日志角色中移除；完整 reward oracle 与 64-action evaluation space 保持不变，并要求每个 action 在全局日志中仍然出现。
> - train/test contexts 继续独立采样自同一 `N(0,I_6)`，因此只称 same-distribution held-out-context generalization / structured state-action support gap，不称 state-distribution OOD generalization。
> - 临时 sandbox 未写入 registry、未使用正式 seeds，也不构成正式结果。它只用于 candidate qualification：64-action 最小改动环境复现了 `Positive-only ceiling -> intermediate-alpha benefit -> stronger-alpha reversal`；长 horizon 诊断显示 `alpha=1` 相对 Positive-only 的 overall reward 差距会随训练延长扩大。
> - 正式 alpha 域冻结为 `[0.0,0.25,0.5,0.75,1.0]`。`alpha=0` 是 Positive-only，`alpha=1` 是不抑制原始负梯度；`alpha>1` 不属于方法域，也不进入正式实验。唯一主指标为 overall expected semantic reward，并登记 paired difference vs Positive-only 与 4k/8k/16k/24k/32k trajectory；hidden-optimal probability、support 与 gradient 只作诊断。
> - 正式协议使用 untouched seeds `150--169`，禁止 sandbox seeds `900--909` 进入正式聚合；5 个 alpha × 20 seeds，共 `100` method-seed runs。最大 `32000` steps、每 `200` steps evaluation，每 5 seeds 写 persistent-local checkpoint；terminal windows 为 `16000--24000` 与 `24000--32000`。
> - task-performance collapse、support/temperature boundary 与 NaN/Inf numerical failure 继续分报。完整运行和登记窗口审计完成可形成有限步正式证据；只有全部登记 runs 达到 formal terminal plateau 时，才允许声称稳态方法排名。若仍持续漂移，必须报告 trajectory 与 finite-step status，禁止预设最佳 alpha。
> - 代码复用 `src/drpo/du1_e6_semantic.py`，新增最小差异 validator `src/drpo/du1_e6_semantic_gap.py`、formal entrypoint `src/drpo/du1_e6_semantic_gap_longrun.py`、冻结配置 `configs/du1_e6_semantic_gap_longrun.yaml` 和 hardened wrapper `scripts/run_du1_e6_semantic_gap_longrun.py`。应用并提交本版后，实验状态为 **implemented + ready + active + not_run**；正式训练不得在 dirty worktree 或未匹配 `origin/main` 时启动。
> - 本更新重基于当前 `main` commit `1fa7f04d4830e4d562ab147dbb11dfa8cecc9b5d`，并保留治理 Pipeline Stage 3 shadow mode 的全部新内容。`D-U1-E6-TAPER-01` 在本 successor terminal-audited、packaged、delivered 之前继续 blocked。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v51-du1-e6-semantic-gap-formal:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000003:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000007:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v55-du1-e6-semantic-gap-result-closure:START -->
> **v55 增量登记：`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 正式结果、终态审计与仓库闭环（不删除 v54 及更早内容）**
>
> - 冻结协议在科学运行 commit `0907c3c0e76fc836c2bf2b752abf554c17f79f22` 上完成 `100/100` method-seed runs；正式 seeds 为 `150--169`，sandbox seeds `900--909` 未进入正式聚合。raw-complete 包 SHA-256 为 `65630159ef85c665a3a0ac0eba5cbf751ecb77a929f267423f7a6d9a8e5c4fbf`。
> - 所有 required outputs 与 terminal audits 均存在且被接受，并完成预注册的 2× horizon 扩展到 32000 steps；但只有 `45/100` runs 达到 formal terminal plateau，`55/100` 为 persistent-drift-or-inconclusive。因此科学状态只升级为 **有限训练步数验证（finite-step validated）**，不得形成全 alpha 稳态方法排名。
> - 三类事件严格分报且均为 0：task-performance collapse `0/100`、support/temperature boundary `0/100`、NaN/Inf numerical failure `0/100`。这不等于全部方法已收敛。
> - 32k 时 Positive-only reward 为 `0.741309`；`alpha=0.25` 与 `alpha=0.50` 分别为 `0.766269` 和 `0.765975`，相对 Positive-only 的 paired mean gains 为 `+0.024960` 与 `+0.024666`，均为 `20/20` seeds 胜出。
> - `alpha=1.0` 相对 Positive-only 的差距从 4k 的 `+0.003943` 转为 8k/16k/24k/32k 的 `-0.013741/-0.039167/-0.053227/-0.061085`，自 8k 起均为 `0/20` 胜出；`alpha=0.75` 到 32k 为 `-0.001978`、9 胜 11 负且 0/20 plateau，属于有限 horizon 反转与持续漂移证据，不是稳态排名。
> - 该结果支持：Positive-only 存在有限 horizon overall-reward ceiling、适度保留负梯度可改善同分布 held-out-context reward、完全不抑制的原始负梯度会产生随 horizon 扩大的任务退化。它不支持 universal alpha optimum、跨任务方法优越性或 categorical policy 的 Gaussian 二次远场律。
> - 训练/测试 contexts 仍独立采样自同一分布；只能称 **same-distribution held-out-context generalization / structured state-action support gap**，不得称 state-distribution OOD generalization。
> - 用户上传的 raw-complete ZIP 是不可变实验/恢复证据，不是 repository update；`drpo-update` 在 `package_extract` 阶段拒绝它是预期行为。仓库只纳入 compact summaries、terminal audit、provenance 与 artifact index，33.6 MB trajectory 保持 persistent-local 索引。
> - `D-U1-E6-TAPER-01` 的 semantic-gap successor delivery blocker 已满足并移除，但 semantic remoteness coordinate、paired protocol、新 untouched seeds 与独立 formal runner 仍未冻结/实现；其状态继续是 **not_run + not_implemented + review-required/blocked**，不得自动启动。
> - 本仓库闭环更新基于 `main` commit `fa225510e3e3e4616f36d8f586611aa6af79bf6e`；未重跑正式实验，也未修改冻结变量、seeds、阈值或训练协议。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v55-du1-e6-semantic-gap-result-closure:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000007:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000008:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v56-e6-parent-closure-route-release:START -->
> **v56 增量登记：E6 父 claim 关闭与 E7-MECH 路线解锁（不删除 v55 及更早内容）**
>
> - 用户在确认 `main` commit `e70f0d84256cdeb6ebbf80b0495a043582787bf6` 已提交后，批准对 **E6 父实验/父 claim** 做范围受限关闭。关闭依据是：`D-U1-E6-SEMANTIC-LONGRUN-01` 的 `360/360` long-run validated 主结果、`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 的 `100/100` finite-step robustness successor，以及 `D-U1-E6-CONDITIONAL-GAP-01` 的 `200/200` finite-step stress diagnostic。
> - 本次关闭锁定五项论文可用结论：Positive-only 在共享语义 categorical 环境存在 imitation ceiling；适度受控负信号可改善同分布 held-out-context / unseen-action 表现；过强或不抑制负压力会出现性能反转或随 horizon 扩大的退化；semantic alignment 是观察到的未见动作迁移的重要排他性条件；任务性能崩溃、support/temperature boundary 与 NaN/Inf 必须继续分报。
> - 本次关闭不把两个 gap 子实验升级为 long-run validated，也不声称全 alpha 稳态排名、universal alpha optimum、state-distribution OOD generalization、categorical policy 的 Gaussian 二次远场律或跨任务方法优越性。`45/100` semantic-gap plateau 与 `49/200` conditional-gap plateau 的终态边界原样保留。
> - `D-U1-E6-TAPER-01` 降为**可选、独立、非门禁**的方法形状比较：它仍是 `not_run + not_implemented + blocked`，若未来执行，必须另行冻结 semantic remoteness coordinate、paired protocol、全新 untouched seeds 与独立 runner；但它不再是 E6 父 claim 关闭或 E7-MECH 启动的前置条件。
> - `EXT-H-E7-Q2` 由 `blocked/blocked` 迁移为 **ready/active**，科学状态仍为 `not_run`。该迁移只开放已经冻结和实现的 Hopper mechanism formal protocol，不代表 E7 已运行或已有结果。`EXT-H-E7-BENCH-01` 继续 blocked，但依赖收缩为 E7-Q2 交付和随后冻结 shortlist，不再依赖可选 E6-TAPER。
> - 本更新只修改研究治理、路线和相应测试/操作说明；未重跑实验，未更改任何冻结变量、数据规模、seeds、阈值、收敛标准或方法职责。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v56-e6-parent-closure-route-release:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000008:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000064:START -->
## 3.7.3 E6 共享语义 pilot `D-U1-E6-SEMANTIC-PILOT-01`

1. **实验职责：** E6 不重复 E5 的 direct-softmax/support-boundary 结论。它检验共享 semantic representation 下，受控 local negative 是否能把策略从 positive demonstrations 推向训练中未展示的 hidden optimal action，并检验 far pressure 是否导致 task failure 或 support/temperature boundary。
2. **状态与术语：** train/test contexts 独立采样自相同 `N(0,I_6)`，因此只报告同分布 held-out-context / unseen-state generalization；本实验没有显式 state distribution shift，不使用 OOD generalization。
3. **开发身份与当前状态：** experiment ID 为 `D-U1-E6-SEMANTIC-PILOT-01`，seeds 固定为 0--4，科学状态为 `pilot`。105/105 development runs 已完成、审计并交付；这不升级为 formal long-run。独立 formal ID `D-U1-E6-SEMANTIC-LONGRUN-01` 已在 untouched seeds `10--29` 上完成 360/360 runs、终态审计和交付，科学状态为 `long_run_validated`。
4. **实现入口：** `src/drpo/du1_e6_semantic.py`；开发配置 `configs/du1_e6_semantic_pilot.yaml`；实现说明 `src/drpo/README_DU1_E6_SEMANTIC.md`。runner 只写普通 JSON/JSONL/CSV/YAML，不自行打包。
5. **E6-A：** fixed concentration 下比较 positive-only 与 local-negative alpha scan，观察 hidden-optimal probability、positive-support probability、expected semantic reward 与 normalized semantic extrapolation。alpha 网格是开发值，不是正式冻结值。
6. **E6-B：** learnable concentration 下比较 `positive_only / local_only / uncontrolled / near_zero / far_zero / far_cap / budget_matched_global`。`local_only` 与 `far_zero` 在数学更新上同义但保留不同协议语义；Far-cap 与 Global 只匹配 raw controlled-negative norm。
7. **E6-C：** 对同一 reward-side catalogue、hidden optimum、demonstrations、negative sets 与 fixed advantages，只独立置换 policy-side action embeddings。若 suppression 仍存在但 hidden-optimum 改善消失，才支持 shared semantic alignment 是外推收益的必要通道；pilot 不构成正式结论。
8. **配对与审计结果：** 同一 seed 内共享网络初始化与 minibatch index stream；105/105 runs 完整。任务性能崩溃为 0/105，support/temperature boundary 为 56/105，NaN/Inf 为 0/105；三类事件继续分报。
9. **终态边界：** fixed-concentration 的 30/30 runs 均未通过两尾窗 provisional plateau，可学习 concentration 的负压力分支普遍触发 support boundary；正式 2x 延长未执行。因此本 pilot 不能升级为 long-run validated、稳定 fixed point 或正式方法排名。
10. **下一门禁：** focused-development freeze 已由用户批准，formal runner/config 已实现并激活。应用本版后直接启动 E6 long-run；运行完成后必须先做 terminal audit、durable packaging 和 delivery，随后才可进入 E6-TAPER。E4-TAPER 与 E6 的科学职责和输出仍相互独立。

---

<!-- STAGE4B-SOURCE-BLOCK:B000064:END -->
