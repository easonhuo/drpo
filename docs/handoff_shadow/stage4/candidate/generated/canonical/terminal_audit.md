# Terminal-state and collapse audit rules

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `terminal_audit`
- Responsibility: Define convergence, persistent drift, task-performance collapse, support or variance boundaries, and numerical failure reporting.
- Dependencies: `global_core_governance`
- Content-contract topics: `convergence_or_persistent_drift`, `two_x_continuation`, `false_plateau_checks`, `task_performance_collapse`, `support_or_variance_boundary`, `nan_inf_numerical_failure`, `separate_failure_reporting`
- Owned source blocks: 4
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: none
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000080:START -->
# 4. 论文机制实验总表与验收标准

| ID | 实验 | 核心问题 | 是否要求训练饱和 | 正式验收 |
|---|---|---|---|---|
| E1 | 瞬时梯度来源隔离 | 相同 advantage 时，远样本梯度是否更大 | 否 | reward/advantage 跨距离误差接近 0；far/near score 与全参数梯度；20 seeds |
| E2 | Positive-only 完整动力学 | 正拟合是否使固定负样本远场化与梯度增长；最终是否平台 | 是 | mu、sigma、正样本 loss、负样本距离、phantom gradient 和梯度比均通过停止标准 |
| E3 | Joint + Near/Far 因果干预 | 远场异常负梯度是否是 drift/collapse 主路径 | 是 | Baseline/Near-zero/Far-zero/Far-cap/Global/Far-to-near；早期时序 + 长期结果；20 held-out seeds |

**E3 结果状态必须拆分（v12 新增协议细化）：**

1. **任务效果崩溃、数值训练仍可运行**：evaluation reward 显著失效，但 loss、梯度和参数仍为有限值，训练没有 NaN/Inf。
2. **任务效果与数值训练同时崩溃**：除 reward 失效外，还出现非有限 loss/gradient/parameter、方差触底或优化器无法继续。

该区分在早期讨论中存在概念基础，但旧 E3 表格和正式输出没有明确登记；是在用户本轮提醒后才补入实验协议。因此不能声称旧实验已经完整报告了两类崩溃。
| E4 | 稳定外推与泛化 | 受控负梯度能否突破 positive-only 上限，远场是否反转为有害 | 是 | 策略越过 a_plus、接近 a_star；训练分布内/同分布 held-out-context reward；强度扫描；控制恢复；固定/可学习方差 |
| E4-TAPER | 距离衰减阶数 | 同一标准化距离上二次 reciprocal 是否比线性 reciprocal 更强压制远场负梯度；是否改善任务效果 | 是 | 20 paired seeds；主 rho=0.25；实际全参数 far/near ratio；held-out-context reward；2× 终态审计；三类失效分报 |
| E5 | Categorical 排斥与支持边界 | 有界 logit score 下重复负更新如何把概率推向边界 | 解析 + 长期 | direct-softmax 解析、概率衰减、rare/common 干预 |
| E6 | 共享语义 categorical 外推 | 负梯度能否利用共享表示改善未见动作且避免 support collapse | 是 | unordered semantic actions；E6 pilot 只冻结协议；long-run 承担 E6-A/B/C 与语义置乱排他性 |
| E6-TAPER | categorical 方法迁移 | 同一 semantic remoteness 上 Linear/Quadratic/Exp 是否兼顾未见动作收益与支持稳定 | 是 | paired stream；distance definition 冻结；positive-only/uncontrolled/global-alpha controls；不声称 Gaussian 二次界 |
| E7-MECH | Hopper learned-critic | 真实数据是否进入并受 Gaussian 二次 log-scale 远场区影响 | 是 | 优势匹配；mean/log-scale 分解；full-parameter 传导；长期 Near/Far/Far-cap/Global；终态审计 |
| E7-BENCH | D4RL MuJoCo locomotion | bandit 中冻结的方法是否在 9 个公共连续控制任务上改善 normalized return 与稳定性 | 是 | Hopper/Walker2d/HalfCheetah × medium/replay/expert；多 seed；平均排名；三类失效分报 |
| E8-MECH | Countdown/Qwen 0.5B | Transformer 中固定负优势 near/far 机制迁移与小规模方法信号 | base-first 门禁；必要时最小 SFT fallback；best + terminal/last-finite 审计 | 0.5B BF16-LoRA pilot；固定 A=-1 probe；Positive-only/Controlled/Uncontrolled/Global-matched |
| E8-SCALE | Countdown 大模型/大数据 | 冻结方法在更大固定数据和 3B/7B 模型上是否保持效果 | 是 | 3B 主结果；7B 冻结确认；不重新筛方法族；性能、支持/熵边界、NaN/Inf 分报 |

<!-- STAGE4B-SOURCE-BLOCK:B000080:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000081:START -->
## 4.1 动力学实验统一收敛标准

不能用固定的 500/1000/10000 步替代收敛判断。所有 E2/E3/E4/E6/E7 需要：

1. 预先定义最大训练步数；
2. 连续多个评估窗口中，核心状态量斜率低于阈值；
3. 更新向量/梯度场残差足够小，或明确持续 runaway；
4. 将训练步数延长至少 2 倍，状态分类、主要结论和方法排序不反转；
5. 检查是否由 clamp、temperature floor 或数值溢出造成假平台。

---

<!-- STAGE4B-SOURCE-BLOCK:B000081:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000113:START -->
### 可学习方差：远场路径提前触发支持收缩

`alpha=0.15, Adam lr=5e-4, max 2000 steps, no variance clamp`。边界审计覆盖完整 4096 个训练状态，并以第一次负向越界作为科学事件。

| 方法 | 首个事件或终态 reward mean [95% CI] | support contraction | onset median | unexpected expansion | NaN/Inf |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.603254 [0.593897, 0.612491] | 20/20 | 73 | 0/20 | 0/20 |
| Near-zero | 0.601992 [0.592785, 0.611119] | 20/20 | 73 | 0/20 | 0/20 |
| Far-zero | 0.652887 [0.651945, 0.653841] | 0/20 | — | 0/20 | 0/20 |
| Far-cap | 0.652625 [0.651661, 0.653619] | 0/20 | — | 0/20 | 0/20 |
| Global-scale | 0.642867 [0.641859, 0.643927] | 0/20 | — | 0/20 | 0/20 |

该分支的第一事件是支持/方差收缩，不是“方差爆炸”，也不是 NaN/Inf。Adam 下未复现旧 plain-SGD 的正向一步过冲：100 个 learnable method-seed runs 中 unexpected expansion 为 0；固定与可学习方差合计 220 个 method-seed runs 中 NaN/Inf 为 0。

<!-- STAGE4B-SOURCE-BLOCK:B000113:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000114:START -->
### 因果结论、论文位置与边界

E3 在同一个 C-U1 环境中闭合两条互补路径：固定方差下，远场异常负影响传导为均值漂移和任务性能崩溃；可学习方差下，它更早传导为支持收缩。Near-zero 在两条路径上均不能救援，Far-zero 与 Far-cap 均能救援。因此该结果可以进入论文的受控因果实验：fixed-variance 四方法对照进主文，learnable-variance 进互补 panel 或附录。

该实验不回答显式状态分布偏移，不得称 OOD；不证明所有真实任务都由该机制崩溃；不证明 Distance、Global-scale 或 Far-to-near 跨任务必然更优。任务性能崩溃、support/variance boundary 与 NaN/Inf 必须继续分开报告。

结果索引：`outputs/cu1_e3_adam/RESULT_SUMMARY.md`、`fixed_variance_aggregate.csv`、`learnable_variance_aggregate.csv`、`TERMINAL_AUDIT.md`、`ARTIFACT_INDEX.json`。完整 raw trajectories 与 checkpoints 位于已交付 artifact `DRPO_CU1_E3_ADAM_AC286A4_FINAL.zip`，SHA-256 为 `2b8bfdbe6f33ed1db9dc1e59f6e9fbdb6c224c7b31b1326a7f2fbaeeaaaf522b`。

<!-- STAGE4B-SOURCE-BLOCK:B000114:END -->
