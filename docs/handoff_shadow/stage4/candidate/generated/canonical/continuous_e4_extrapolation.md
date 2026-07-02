# Continuous C-U1 E4 stable extrapolation and phase transition

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `continuous_e4_extrapolation`
- Responsibility: Cover the positive-only ceiling, bounded extrapolation, phase transition, fixed or learnable variance outcomes, and E4 convergence closure before taper-family follow-ups.
- Dependencies: `global_core_governance`, `theory_methods_related_work`, `terminal_audit`, `continuous_mechanism_e1_e3`
- Content-contract topics: none
- Owned source blocks: 5
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `C-U1-E4-ADAM-RERUN`, `C-U1-E4-CONV-01`
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000054:START -->
### 3.6.2 E4：稳定外推—相变—远场控制

1. **正式 seeds：** 开发 seeds 5–9 只用于确定扫描区间、学习率和 far-pressure 强度；正式 held-out seeds 50–69，所有方法配对。
2. **共同初始化：** 从同一 positive-only 饱和策略开始；固定方差主分支使用解析 `sigma=0.190394`，可学习方差分支保留 state-conditioned log-std。
3. **有益局部负信号：** 仅使用每状态第 0 个负动作 `a_minus=a_plus-0.50u`，其排斥方向与真实 improvement direction `a_star-a_plus` 对齐。局部目标为 `L_pos + alpha_local L_minus`。
4. **固定方差强度扫描：** 扫描 `alpha_local` 从 0 到超过解析临界值 `alpha_c=A_pos/|A_neg|≈1.693`；报告解析 signed target、经验归一化外推位移、test reward、终态类别和 2× horizon 审计。最低目标是复现 positive-only ceiling、越过 `a_plus`、在 `归一化外推位移≈1` 附近达到未见最优、随后过度外推和临界漂移。
5. **可学习方差扫描：** 在同一局部目标上扫描更细的低 alpha 区间，检验二阶矩可行性边界是否早于固定方差均值边界；方差越界与任务 reward 失效分别报告。
6. **远场压力：** 将其余 7 个等 advantage 轮廓动作定义为额外 far-pressure，目标写成 `L_pos + alpha_local L_minus + lambda_far L_far`；`alpha_local` 固定在固定方差近最优区间，`lambda_far` 由开发 seeds 预注册为能稳定触发性能反转但不依赖 NaN 的最小值。
7. **控制方法：** 比较 `positive_only`、`local_only`、`uncontrolled_all`、`far_zero/local_oracle`、`far_cap` 与 `budget_matched_global`。Far-cap 只缩放 far 分量；budget-matched global 将全部负梯度统一缩放到与 Far-cap 相同的 post-control norm，以排除“仅仅总梯度更小”。
8. **方向与影响诊断：** 逐负动作报告其梯度与真实 improvement update 的 cosine、score norm、全参数 influence；检验局部有益方向与远场低/反向 utility 是否同时伴随更大 influence。
9. **正式验收：** （a）20/20 或统计显著多数策略越过 `a_plus`；（b）held-out `a_star` reward 高于 positive-only；（c）reward 对负推力呈倒 U 型或存在明确相变；（d）Far-cap 在远场压力下恢复有益外推且不崩溃；（e）相对等预算 global 的差异用 paired bootstrap CI 报告，不预设 Distance 必然胜出。

<!-- STAGE4B-SOURCE-BLOCK:B000054:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000055:START -->
### 3.6.3 E4 数值配置冻结与一次执行流程纠正

开发 seeds 5–9 得到以下预注册配置：

- **固定方差局部强度网格：** `alpha_local ∈ {0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75}`。其中解析均值临界值为 `alpha_c≈1.693`；1.50 用于观察有限但严重过度外推的稳态，1.75 用于观察固定点消失后的持续漂移。
- **可学习方差局部强度网格：** `alpha_local ∈ {0, 0.10, 0.20, 0.30, 0.35, 0.38, 0.40, 0.50}`。解析二阶矩可行边界约为 `alpha_sigma≈0.381`，因此 0.38/0.40 跨越该边界。
- **优化与终态审计：** 有有限解析内部解的配置先运行 200-step minibatch SGD，随后执行全数据 LBFGS stationary audit，再进行等长 200-step continuation，最后对同一目标重新 stationary audit。无内部解的配置运行 2000+2000-step 长程 SGD，不使用 LBFGS。
- **残差判据细化：** signed objective 的正负分量可各自很大并在固定点相消，因此正式使用 `||g_total||/(||g_pos||+||g_neg||)<2e-3` 作为净动力场归一化残差；`alpha=0` 单独要求 absolute norm `<1e-3`。这是对第 3.5(10) 绝对阈值的必要尺度化细化，原阈值不删除。
- **远场压力与控制：** `alpha_local=1.0`、`lambda_far=1.0`，Far-cap 约束 far weighted-gradient norm 不超过 local weighted-gradient norm 的 `0.05`。开发 seed 上该配置使 uncontrolled_all 发生有限数值下的任务崩溃，而 Far-cap 保留正向外推。Budget-matched global 的 post-control negative norm 与 Far-cap 精确匹配。
- **方向诊断：** 在 positive-only 初始化处，第 0 个负动作与真实 improvement update 的 cosine 为 1；最远第 4 个动作 cosine 为 -1，且其全参数 update norm 约为近场的 3.8 倍。正式结果使用 20 seeds 汇总，不把单 seed 数值当作结论。

**执行流程纠正：** 在本小节写入前曾误启动固定方差正式 driver，产生 12 个未完成结果。发现“精确网格尚未先回写文档”后立即停止；这些文件未删除，整体移动到 `e4_pre_freeze_fixed_pilot_091632/`，只作 provenance，不进入正式统计。正式 E4 必须在本小节冻结后从空目录重新运行。

<!-- STAGE4B-SOURCE-BLOCK:B000055:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000056:START -->
### 3.6.4 E4 控制分支的精确长程配置

- `positive_only` 与 `local_only(alpha=1.0)` 直接复用同 seeds 的正式局部扫描结果，不重复训练。
- 新增长程方法只有 `uncontrolled_all`、`far_cap`、`budget_matched_global`；共同使用 `alpha_local=1.0`、`lambda_far=1.0`、Far-cap ratio `0.05`、固定 `sigma=0.190394`、SGD `lr=5e-4`。
- 训练 4000 steps，每 100 steps 评估；2000 steps 是候选 horizon，4000 steps 是 2× extension。报告 reward、归一化外推位移、净更新残差、任务崩溃 onset、数值有限性及方法排序是否在后半程反转。
- `budget_matched_global` 在每一步将原始全部负梯度统一缩放，使其 post-control norm 与 Far-cap 完全相同；允许缩放系数大于 1，因为原始 local/far 分量可能方向抵消。该对照匹配的是实际净负梯度预算，而不是预设“只能缩小”。
- 正式方向诊断在 positive-only 初始化处对 8 个等 advantage 负动作分别计算全参数 update norm、标准化距离及与真实 improvement update 的 cosine；20 seeds 配对汇总。

<!-- STAGE4B-SOURCE-BLOCK:B000056:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000057:START -->
### 3.6.5 v29 统一 Adam 执行覆盖（当前有效协议）

本节覆盖 3.6.3、3.6.4 和 11.4 中的 SGD/LBFGS 执行细节；旧内容保留作 provenance。

1. E3 fixed、E3 learnable、E4 fixed、E4 learnable 与 E4 control 的训练优化器统一为 Adam，`betas=(0.9,0.999)`、`eps=1e-8`；沿用已冻结的各分支 learning rate、alpha、seeds、数据、步数上限和任务阈值，不借优化器迁移反向调参。
2. E3/E4 初始化固定为同 seed 2000-step positive-only Adam checkpoint。E2 的 LBFGS、2× continuation 和 adaptive polish 仅做 E2 终态审计。
3. E4 有有限解析内部解的配置先做 200-step Adam、全数据 residual audit、等长 200-step Adam continuation、第二次 residual audit；audit 只测量同一目标的净动力场，不再用 LBFGS 改写参数。无内部解配置按原上限做 Adam 长程并报告持续漂移或首次支持收缩。
4. Learnable-variance 每一步在完整 4096 train states 上做首次事件审计。`support_contraction`、task-performance collapse、parameter/log-sigma/sigma-output NaN/Inf 分开；任何 `unexpected_support_expansion` 都是失败诊断，不进入方法排名。
5. E3/E4 输出必须同时包含 raw total/negative gradient norm 与 Adam parameter-update norm。Raw-gradient matched control 仍用于机制对照，但论文不得称其为 actual-update matched，除非另行登记并实现 Adam update-level calibration。
6. 主文只保留最短因果链和倒 U 型相变；Global、Far-to-near、budget-matched controls 进入附录，不把优化器细节拆成多条主叙事。
7. 正式命令必须按 stage 分开执行；`--stage all` 只允许 smoke。

<!-- STAGE4B-SOURCE-BLOCK:B000057:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000058:START -->
### 3.6.6 `C-U1-E4-CONV-01` 长程终态确认（v33 当前有效协议）

1. **实验职责：** 仅确认原 E4 固定方差 `alpha=0.75/1.00/1.25` 的长期状态是否反转。它不重跑可学习方差、控制方法、`alpha=1.50/1.75`，也不新增方法排名。
2. **Positive-only 边界：** 不追加运行 `alpha=0`。E2 承担 positive-only 完整动力学；原 E4 的 `alpha=0` 只保留为相变扫描左端 control。
3. **冻结执行：** seeds 50--69；从同 seed 的 2000-step positive-only Adam checkpoint 重新开始；固定方差、Adam、学习率、batch、advantage、数据和 RNG 与 `C-U1-E4-ADAM-RERUN` 完全一致。
4. **训练与审计：** 每个 alpha 运行 4000 steps；full-state audits 为 `400/800/1600/2400/3200/4000`；终态窗口为 `2000--3000` 和 `3000--4000`。
5. **稳定判据：** W2 位移变化绝对值 `<=0.02`，W2 reward 变化绝对值 `<=0.01`，raw full-data gradient 与 Adam update 的 W2/W1 中位比均 `<=1.25`，且长期科学角色不反转。
6. **Runaway 判据：** 两个窗口的位移均增加，W2 位移增量 `>0.05`，且 raw gradient 或 Adam update 的 W2/W1 中位比 `>1.25`。其余登记 `terminal_state_inconclusive`。
7. **残差口径：** 继续记录 full-data normalized residual，但 `2e-3` 不再是硬 gate，不为通过门禁而改学习率、optimizer、batch、threshold 或训练长度。
8. **目标状态与汇总：** `0.75/1.00 -> stable_beneficial_extrapolation`；`1.25 -> stable_over_extrapolation`。每个 alpha 至少 18/20 达标，余下只允许 inconclusive。
9. **持久化：** 每 5 seeds 生成 checkpoint 包；正式结束后必须独立报告任务性能、support/variance boundary 和 NaN/Inf，并完成终态审计与 durable delivery。

---

<!-- STAGE4B-SOURCE-BLOCK:B000058:END -->
