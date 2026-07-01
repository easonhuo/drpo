# Continuous C-U1 source and causal mechanism E1-E3

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `continuous_mechanism_e1_e3`
- Responsibility: Cover continuous far-field gradient-source identification and causal transmission into drift, task collapse, and variance or support contraction.
- Dependencies: `global_core_governance`, `theory_methods_related_work`, `terminal_audit`
- Content-contract topics: none
- Owned source blocks: 9
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `C-U1-E3-ADAM-RERUN`, `C-U1-E1-COMP-01`
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000043:START -->
# 3. 连续统一环境 C-U1 的详细设计

<!-- STAGE4B-SOURCE-BLOCK:B000043:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000044:START -->
## 3.1 状态与动作

- 状态：`s in R^6`；训练集和测试集分别采样，使用同一生成函数。
- 动作：`a in R^2`；策略为 state-conditioned Gaussian，均值与方差共同学习。
- 每个状态产生 state-dependent 的 `a_plus(s)`、`a_star(s)`、任务方向和正交方向。

<!-- STAGE4B-SOURCE-BLOCK:B000044:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000045:START -->
### 3.1.1 “context/state”在小网络中的具体含义

这里的 context 不是自然语言上下文，而是输入给 MLP 的 6 维数值向量。每一个状态 `s` 代表一个不同的一步决策条件；环境通过固定生成函数把 `s` 映射为该条件下的 `a_plus(s)`、`a_star(s)` 和奖励地形。小网络学习的是函数 `s -> (mu(s), sigma(s))`，而不是记忆一个全局动作。

- **训练状态**：其 state-action-reward 样本参与参数更新。
- **测试状态**：由同一状态生成分布独立采样，但完全不参与训练，用于检查 MLP 是否学到状态到动作的映射，而不是仅记住训练状态。
- **一个状态不等于一个样本**：同一状态下可构造多个正动作、多个负动作和额外梯度探针，因此 transition 数等于“状态数 × 每状态动作数”。
- 当前环境原型使用 1024 个基础状态做不变量检查；上一轮提出的 4096 train / 4096 test、每状态 4 正 / 8 负只是**正式配置提案**，尚未获得用户确认，也尚未用于正式训练。

训练/测试状态拆分的唯一目的，是验证 state-conditioned 网络对未见数值输入的函数泛化。E1 的距离—梯度来源识别主要按状态聚合，不把同一状态的多个复制动作当作独立样本。

<!-- STAGE4B-SOURCE-BLOCK:B000045:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000046:START -->
## 3.2 数据与奖励

Ground-truth reward 由动作到 `a_star(s)` 的二维距离决定，因此 `a_star` 是唯一最优动作。正样本分布位于 `a_plus` 周围；负样本位于经过 `a_minus` 的等奖励轮廓。等奖励轮廓上的所有负样本 reward/advantage 精确相同，但相对当前策略的距离不同。

<!-- STAGE4B-SOURCE-BLOCK:B000046:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000047:START -->
## 3.3 同一环境如何支持四个实验

- E1 直接读取同一状态下等 advantage 的轮廓负样本，比较距离与梯度；
- E2 仅应用正样本梯度，轮廓负样本只作为 phantom gradient 监测对象；
- E3 应用正负梯度，并按当前策略距离动态划分 near/far 进行干预；
- E4 重点使用 `a_minus` 及其邻近负样本提供指向 `a_star` 的有益排斥，再加入远场轮廓样本观察从外推到失稳的转折。

<!-- STAGE4B-SOURCE-BLOCK:B000047:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000048:START -->
## 3.4 需要预先讨论而不能擅自决定的设计项

1. 负轮廓角度数量与距离范围；
2. positive residual spread 是否固定以及是否加入 state-dependent 噪声；
3. advantage 使用固定真实 reward-baseline，还是增加 learned-critic 附录；
4. E4 中使用全部负轮廓还是只使用方向一致的近场子集作为有益负信号；
5. 训练步数、停止标准与正式 seeds。

在这些项目冻结前只做 invariant/smoke test，不宣称正式结果。


<!-- STAGE4B-SOURCE-BLOCK:B000048:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000049:START -->
## 3.5 v13 冻结后的 C-U1 正式配置

用户已授权冻结以下配置并开始正式执行：

1. **状态与数据量：** `s ~ N(0,I_6)`；4096 个训练状态与 4096 个独立测试状态。每个状态构造 4 个正动作和 8 个负动作。训练按 state minibatch 取样，并同时读取该状态对应动作，避免把同状态复制动作当作独立 context。
2. **任务几何：** `a_star(s)=a_plus(s)+0.70 u(s)`；有益近场负动作 `a_minus(s)=a_plus(s)-0.50 u(s)`。8 个负动作位于以 `a_star` 为圆心、半径 1.20 的等奖励圆上，包含 `a_minus`，因此其 reward/advantage 在每个状态内严格相等，但相对策略距离不同。
3. **正样本条件残差：** 4 个正动作位于 `a_plus ±0.18u` 与 `a_plus ±0.18v`；该非零 residual spread 允许 positive-only 的 Gaussian 方差存在内部有限目标，避免把确定性 MLE 的方差坍缩误当作远场机制。
4. **奖励和 advantage：** `R(s,a)=exp(-||a-a_star(s)||^2/(2*0.75^2))`；固定 baseline 为 0.40。所有 advantage 在训练前计算并冻结。负动作 advantage 跨轮廓数值误差须低于 `1e-6`，所有正动作 advantage >0，所有负动作 advantage <0。
5. **策略：** 共享两层 MLP，state-conditioned 2D Gaussian mean 与标量 log-standard-deviation head；不使用人为方差 clamp。`log_sigma<-12` 是 support/variance contraction 边界事件；参数、log-sigma 或 sigma 输出的 NaN/Inf 单独记为数值失败；`log_sigma>12` 只能记为 unexpected positive-boundary event，不构成理论中的方差扩张分支。
6. **目标归一化：** 正、负部分分别按组取均值，更新写为 `g = g_pos + alpha*g_neg`，使 alpha 表示负向总质量相对强度，不由 4/8 样本数量机械决定。
7. **Near/Far：** 依据当前策略下标准化动作距离动态划分，正式阈值 `d=5.0`；阈值稳健性在开发集检查 `4.0/6.0`。Near/Far mask 只用于干预，不回传距离权重梯度。
8. **E4 有益负信号：** 只使用轮廓中位于 `a_minus` 方向的近场动作作为方向可靠负信号；其余轮廓动作作为远场压力源。这样 E4 检验的是“有益局部排斥 + 额外远场压力”的转折，不把方向相反的负动作混入有益外推定义。
9. **seeds：** 0–4 仅用于回归、阈值和 alpha 相变定位。E1/E2 使用 held-out 10–29；由于 E3 smoke 曾意外查看 seed 10，E3 为保持严格盲测改用 held-out 30–49。所有方法在各实验内部配对相同 seeds。
10. **收敛与终态：** 每 100 steps 评估；E3/E4 论文主训练统一使用 Adam，并分别记录 raw gradient norm 与 Adam parameter-update norm。稳定候选需通过全数据净动力场残差和 2× continuation 审计，且状态分类不反转；持续漂移则报告斜率、reward 失效时间和数值状态。最大步数按各 protocol 配置记录，不用固定步数冒充稳态。

该配置替代第 3.4 节中的“待讨论”状态；第 3.4 节保留作为决策 provenance，不删除。

<!-- STAGE4B-SOURCE-BLOCK:B000049:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000050:START -->
## 3.6 v13 执行期勘误与 E4 正式协议冻结

<!-- STAGE4B-SOURCE-BLOCK:B000050:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000051:START -->
### 3.6.1 正样本几何勘误（不破坏性覆盖）

- **原登记：** 第 3.5(3) 写成“四个正动作位于 `a_plus ±0.18u` 与 `a_plus ±0.18v`”。
- **问题：** 该写法与第 3.5(4) 的等 reward 设定、已经运行的代码和 E1/E2 结果不一致；这些四点相对 `a_star` 的距离并不严格相等。
- **正式实现与修正：** 四个正动作位于以 `a_star` 为圆心、半径 0.75 的等 reward 圆上，角度为 `pi±theta_1` 与 `pi±theta_2`，其中 `theta_1=0.20`，`theta_2` 由质心精确等于 `a_plus` 的方程确定。其条件残差总二阶矩为 `0.75^2-0.70^2=0.0725`，二维共享标准差的 positive-only 解析目标为 `sqrt(0.0725/2)=0.190394`。
- **证据：** C-U1 invariant、E1 与 E2 均使用该等 reward 实现；E2 的 20-seed 最终平均 `sigma=0.190419`，与解析值一致。
- **处理：** 第 3.5(3) 作为错误 provenance 保留，本节为正式替代记录；后续实验不改动已运行的数据生成器。

<!-- STAGE4B-SOURCE-BLOCK:B000051:END -->
