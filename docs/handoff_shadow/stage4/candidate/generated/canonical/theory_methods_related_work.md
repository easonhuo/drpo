# Theory, method families, and related work

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `theory_methods_related_work`
- Responsibility: Provide the shared repulsive-dynamics theory, method definitions, comparison boundaries, and literature positioning.
- Dependencies: `global_core_governance`
- Content-contract topics: none
- Owned source blocks: 21
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: none
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000211:START -->
# Part III. v9 Exponential-Family 核心理论补丁（完整保留）

> 本节保留 v9 理论正文。它是原 DRPO repulsive dynamics 的统一数学升级，不是另起一套与原理论无关的框架。涉及过多符号的部分在后续论文精简时调整，但研究主文档不删除。

<!-- STAGE4B-SOURCE-BLOCK:B000211:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000212:START -->
# 2. 大一统理论：Repulsive Signed-Moment Dynamics

<!-- STAGE4B-SOURCE-BLOCK:B000212:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000213:START -->
## 2.1 研究对象、记号与结论层级

对每个状态 s 条件化后，把正优势和负优势样本分别写成加权分布 P₊(a\|s)、P₋(a\|s)。令正质量 p(s)=E\[A₊\|s\]，负质量 q(s)=E\[(-A)₊\|s\]；全局 α、样本权重或方法控制均被吸收到 q 和 P₋ 中。基础理论先假设 actor step 内 advantage stop-gradient，随后再讨论 value/Q 随时间变化。

$$
J(\theta)=\mathbb{E}_{\mathcal D}[A(s,a)\log\pi_\theta(a\mid s)],\qquad F(\theta)=\nabla_\theta J(\theta)=\mathbb{E}_{\mathcal D}[A\nabla_\theta\log\pi_\theta(a\mid s)]
$$

理论分成三层：第一层是任意可微策略都成立的单样本 surprisal 递推；第二层是在正则最小指数族中成立的 signed-moment 平衡定理；第三层才是 Gaussian、categorical、神经网络与具体控制方法的分叉推论。这样既保留 general 形式，也避免把 expected Fisher 当成固定样本动力学。

<!-- STAGE4B-SOURCE-BLOCK:B000213:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000214:START -->
## 2.2 定理 1：单个负优势更新必然提高该样本 surprisal

令 z=(s,a)，Sθ(z)=−logπθ(a\|s)，gθ(z)=∇θlogπθ(a\|s)。对固定负优势 A(z)=−c\<0，单样本梯度上升为：

$$
\theta^+=\theta-hc\,g_\theta(z),\qquad h>0
$$

对 Sθ 做二阶 Taylor 展开，存在位于 θ 与 θ⁺ 之间的 θ̃，使：


$$
S_{\theta^{+}}(z)-S_\theta(z)=hc\lVert g_\theta(z)\rVert^2+\frac12 h^2c^2 g_\theta(z)^\top\!\left[\nabla^2 S_{\tilde\theta}(z)\right]g_\theta(z)
$$


若该线段上 ‖∇²S‖op≤L，则：


$$
S_{\theta^{+}}-S_\theta\ge hc\lVert g_\theta\rVert^2\left(1-\frac12hcL\right)
$$


因此当 hcL\<2 时，surprisal 严格增加。连续时间梯度流 θ̇=−c gθ 下更有精确恒等式：


$$
\frac{dS_\theta(z)}{dt}=c\lVert g_\theta(z)\rVert^2\ge 0
$$


这一定理是连续与离散的共同主干：负更新不是“静态降低概率”，而是把同一样本沿当前策略的 score geometry 持续推向更低支持。

<!-- STAGE4B-SOURCE-BLOCK:B000214:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000215:START -->
## 2.3 批量更新：自项、跨样本干涉与方向一致性

单样本单调性不能无条件提升为“batch 中每个负样本 surprisal 都单调增加”。令 batch field F=ΣⱼAⱼgⱼ，则样本 i 的首阶变化为：


$$
\Delta S_i=-h g_i^\top F+O(h^2)=h|A_i|\lVert g_i\rVert^2-h\sum_{j\ne i}A_j\langle g_i,g_j\rangle+O(h^2)
$$


第一项是负样本自身的确定性排斥；第二项是正负样本共享参数带来的 interference。远场风险因此不仅取决于单样本 scale，还取决于梯度方向是否相干。本文实验中的 aggregate amplification 正是在单样本 score 放大之外叠加了 coherence。


$$
\text{Repulsive influence}\approx\text{negative mass}\times\text{score scale}\times\text{directional coherence}\times\text{repeated reuse}
$$


<!-- STAGE4B-SOURCE-BLOCK:B000215:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000216:START -->
## 2.4 定理 2：正则最小指数族中的 signed-moment 平衡

考虑固定状态下的正则最小指数族：


$$
\pi_\eta(a)=h(a)\exp\!\left\{\eta^\top T(a)-\psi(\eta)\right\}
$$


令 t₊=E\_{P₊}\[T(a)\]、t₋=E\_{P₋}\[T(a)\]，w=p−q。则 signed policy objective 可精确写为：


$$
J(\eta)=(p t_+-q t_-)^\top\eta-(p-q)\psi(\eta)+C=w\left[\tau^\top\eta-\psi(\eta)\right]+C
$$



$$
\tau=\frac{p t_+-q t_-}{p-q}
$$


其梯度和 Hessian 为：

$$
\nabla_\eta J=w[\tau-m(\eta)],\qquad m(\eta)=\mathbb E_{\pi_\eta}[T(a)]
$$


$$
\nabla_\eta^2J=-w\,\operatorname{Cov}_{\pi_\eta}[T(a)]
$$


由此得到统一结论：

- 若 w\>0 且 signed target τ 位于指数族 mean-parameter domain 的内部，则存在唯一有限平衡 η\*，满足 m(η\*)=τ；在可识别子空间上 Hessian 负定，平衡局部渐近稳定。

- 若 τ 位于 mean-domain 边界，最优分布只能在边界上实现，通常需要自然参数趋于无穷；这对应 Gaussian 的零方差边界或 categorical 的零概率支持。

- 若 τ 落在可行域之外，或 w≤0，则不存在有限内部平衡；目标可能无界，或动力学向参数/分布边界逃逸，具体表型由策略族决定。

- 离散 Euler 更新在平衡附近的充分步长条件是 ρ(I+hJ)\<1；指数族自然参数下可写为 h \< 2/\[w λmax(Covπ\*\[T\])\]。

这个定理把“稳定外推”和“崩溃”统一成一个几何问题：负优势把正样本的 moment target 沿远离负样本的方向外推；只要外推后的 signed target 仍位于可行 moment 域内，就存在稳定解；一旦越界，内部固定点消失。

| **策略族**                   | **充分统计 T(a)** | **mean-domain** | **越界表型**                                                   |
|------------------------------|-------------------|-----------------|----------------------------------------------------------------|
| 固定方差 Gaussian            | a                 | 整个实数空间    | p≤q 时均值漂移或 runaway                                       |
| 可学习方差 Gaussian          | (a, a²)           | m₂\>m₁²         | signed variance≤0，σ→0 或联合失稳                              |
| full softmax categorical     | one-hot eₐ        | 概率单纯形      | 某些 signed probability≤0，logit gap→∞                         |
| feature / energy categorical | 动作特征 φ(a)     | 特征凸包内部    | 目标 feature moment 越界或贴边，support / temperature collapse |

<!-- STAGE4B-SOURCE-BLOCK:B000216:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000217:START -->
## 2.5 Gaussian 推论 A：固定方差下的稳定外推与均值相变

对 π=N(μ,σ²)，固定 σ。设正负动作均值为 m₊、m₋，有效质量为 p、q。均值动力学为：


$$
\dot\mu=\frac{p(m_+-\mu)-q(m_--\mu)}{\sigma^2}
$$


当 p\>q 时存在稳定点：

$$
\mu^*=\frac{pm_+-qm_-}{p-q},\qquad \mu^*-m_+=\frac{q(m_+-m_-)}{p-q}
$$

若 m₋\<m₊，负样本位于正样本另一侧，则 μ\*\>m₊：负梯度把策略稳定推到最佳正样本支持之外。若真实最优为 a\*\>m₊，使 μ\*=a\* 的最优负质量为：


$$
q_{\mathrm{opt}}=p\frac{a^*-m_+}{a^*-m_-}<p
$$


因此任务最优点严格位于动力学临界点 qcrit=p 之前。离散更新的误差满足：


$$
\mu_{t+1}-\mu^*=\left[1-\frac{h(p-q)}{\sigma^2}\right](\mu_t-\mu^*)
$$


稳定步长要求 0\<h(p−q)/σ²\<2。q=p 时若 m₊≠m₋，吸引与排斥曲率抵消，出现持续漂移；q\>p 时均值固定点失去稳定性并产生 runaway。

<!-- STAGE4B-SOURCE-BLOCK:B000217:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000218:START -->
## 2.6 Gaussian 推论 B：可学习方差的联合稳态与提前失稳

令 ξ=logσ，正负条件方差分别为 v₊、v₋，并定义 M±(μ)=v±+(μ−m±)²。精确动力学为：


$$
\dot\mu=\frac{p(m_+-\mu)-q(m_--\mu)}{\sigma^2}
$$



$$
\dot\xi=\frac{pM_+(\mu)-qM_-(\mu)}{\sigma^2}-(p-q)
$$


联合内部固定点为：


$$
\mu^*=\frac{p m_+-q m_-}{p-q}
$$



$$
\sigma^{2*}=\frac{pM_+(\mu^*)-qM_-(\mu^*)}{p-q}
$$


将其化成 signed variance 可得到更清晰的可行性条件。令 Δ=m₊−m₋：


$$
\sigma^{2*}=\frac{p v_+-q v_-}{p-q}-\frac{pq\Delta^2}{(p-q)^2}
$$


因此联合稳态需要 p\>q 且 σ²\*\>0。第二个条件通常更严格，使方差边界早于均值边界。令 C=v₊+v₋+Δ²，v₋\>0 时较小正根为：


$$
q_{\mathrm{var}}=p\frac{C-\sqrt{C^2-4v_+v_-}}{2v_-}
$$


若 v₋=0，则极限为：


$$
q_{\mathrm{var}}=p\frac{v_+}{v_++\Delta^2}
$$


在联合固定点处，(μ,ξ) 动力学 Jacobian 恰好对角化：


$$
J_F(\mu^*,\xi^*)=\operatorname{diag}\!\left(-\frac{p-q}{\sigma^{2*}},-2(p-q)\right)
$$


所以只要内部解存在且 p\>q，均值和 log-std 都局部稳定；实验中观察到的“方差先坍缩”不是固定点不稳定，而是 signed target 先离开 Gaussian 可行 moment 域，使有限固定点直接消失。

<!-- STAGE4B-SOURCE-BLOCK:B000218:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000219:START -->
## 2.7 Gaussian 推论 C：方差四象限、单样本 MLE 与远场幅度放大

$$
\frac{\partial\log\pi}{\partial\xi}=z^2-1,\qquad z=\frac{a-\mu}{\sigma}
$$

| **advantage** | **\|z\|\<1**          | **\|z\|\>1**            |
|---------------|-----------------------|-------------------------|
| A\>0          | σ下降：集中到近正样本 | σ上升：覆盖远正样本     |
| A\<0          | σ上升：摊薄近负样本   | σ下降：压缩远负样本支持 |

单个确定性正样本的 Gaussian log-likelihood 没有有限最大值：μ→a 后仍有 logπ(a)=−logσ+C→+∞，故 σ→0。只有拟合均值后仍存在非零条件残差，或加入 entropy/KL/σ-min，positive-only 才有有限方差稳态。

原 sign-only Hessian 论证的问题在此处最清楚。固定样本的 negative-log-likelihood Hessian 为：


$$
H_{\mathrm{sample}}=\begin{bmatrix}\sigma^{-2}&2(a-\mu)\sigma^{-2}\\2(a-\mu)\sigma^{-2}&2(a-\mu)^2\sigma^{-2}\end{bmatrix}
$$


$$
\det(H_{\mathrm{sample}})=-\frac{2(a-\mu)^2}{\sigma^4}<0\qquad(a\ne\mu)
$$

它是不定矩阵；只有对 a~π 取期望后才得到 Fisher / expected Hessian diag(σ⁻²,2)≻0。因此不能由 expected SPD 推出固定 off-policy 样本在 (μ,ξ) 每个方向都统一扩张。正确结论是：负样本始终排斥均值，但方差方向由 z²−1 决定。

远场幅度分叉仍然成立。Gaussian score 为：

$$
g_\mu=\frac{a-\mu}{\sigma^2}=\frac{z}{\sigma},\qquad g_\xi=z^2-1
$$


$$
\lVert g\rVert^2=\frac{z^2}{\sigma^2}+(z^2-1)^2
$$


固定 σ 且只重复一个负样本时，δₜ=μₜ−a 满足精确递推 δₜ₊₁=(1+hc/σ²)δₜ，故均值距离和 mean-score 关于训练步数几何增长。可学习方差时，远场负样本同时使 μ 远离、σ 收缩，通常进一步放大标准化距离；但不应再无条件声称 μ 与 σ 都“expand”。

<!-- STAGE4B-SOURCE-BLOCK:B000219:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000220:START -->
## 2.7A Gaussian 远场负梯度的二次临界衰减定理（v30，已解析证明）

本节只证明同一 Gaussian 标准化距离上的控制阶数，不把 surprisal 替换成距离，也不把任务 reward 排名写进定理。考虑动作维数为 `D` 的 isotropic Gaussian：

$$
\pi_\theta(a\mid s)=\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2I_D),\qquad \xi_\theta(s)=\log\sigma_\theta(s).
$$

定义当前 C-U1 Near/Far 使用的标准化距离

$$
d=d_\theta(s,a)=\frac{\lVert a-\mu_\theta(s)\rVert_2}{\sigma_\theta(s)}.
$$

固定一个负优势样本 `A(s,a)=-c<0`，并令 policy-output 坐标为 `y=(mu,xi)`。其负梯度幅度等于 `c` 乘以 Gaussian score 幅度。精确公式为

$$
\nabla_\mu\log\pi_\theta(a\mid s)=\frac{a-\mu}{\sigma^2},\qquad
\frac{\partial\log\pi_\theta(a\mid s)}{\partial\xi}=d^2-D,
$$

从而

$$
\boxed{
\lVert g_y^-(s,a)\rVert_2^2
=c^2\left[\frac{d^2}{\sigma^2}+(d^2-D)^2\right].
}
$$

<!-- STAGE4B-SOURCE-BLOCK:B000220:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000221:START -->
### 定理 3（pre-boundary 区域中的二次远场界）

假设负优势幅度满足 `0<c_min<=c<=c_max<infinity`，并且只在 support/variance boundary 之前的正则区域讨论，即存在 `sigma_min>0` 使 `sigma_theta(s)>=sigma_min`。令 `d>=d_0=max{1,sqrt(2D)}`，则存在与 `d` 无关的正常数 `C_1,C_2`，使

$$
C_1d^2\le \lVert g_y^-(s,a)\rVert_2\le C_2d^2.
$$

可取

$$
C_1=\frac{c_{\min}}{2},\qquad
C_2=c_{\max}\sqrt{1+\sigma_{\min}^{-2}}.
$$

**证明。** 当 `d^2>=2D` 时，`d^2-D>=d^2/2`，故 log-scale 分支直接给出

$$
\lVert g_y^-\rVert_2\ge c|d^2-D|\ge\frac{c_{\min}}2d^2.
$$

另一方面，`0<=d^2-D<=d^2`，且 `d>=1`、`sigma>=sigma_min`，因此

$$
\frac{d^2}{\sigma^2}\le\frac{d^4}{\sigma_{\min}^2},\qquad(d^2-D)^2\le d^4.
$$

代回精确范数式，得到

$$
\lVert g_y^-\rVert_2^2\le c_{\max}^2d^4(1+\sigma_{\min}^{-2}).
$$

开平方即得上界，故 `||g_y^-||=Theta(d^2)`。证毕。

该定理说的是**固定时刻、同一标准化距离上的单样本输出梯度**。它不等同于 advantage 自身二次增长，也不声称神经网络全参数梯度无条件具有严格二次下界。

<!-- STAGE4B-SOURCE-BLOCK:B000221:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000222:START -->
### 定理 4（reciprocal-polynomial 的二次临界阶）

令距离权重 stop-gradient，并定义

$$
w_p(d)=\frac{1}{1+\lambda(d/d_{\mathrm{ref}})^p},\qquad \lambda>0,\ p\ge0.
$$

在定理 3 的条件下，

$$
\boxed{\lVert w_p(d)g_y^-(s,a)\rVert_2=\Theta(d^{2-p}).}
$$

因此

$$
p<2\Rightarrow \lVert w_pg_y^-\rVert_2\to\infty,
$$

$$
p=2\Rightarrow 0<\liminf_{d\to\infty}\lVert w_pg_y^-\rVert_2\le\limsup_{d\to\infty}\lVert w_pg_y^-\rVert_2<\infty,
$$

$$
p>2\Rightarrow \lVert w_pg_y^-\rVert_2\to0.
$$

**证明。** 当 `p=0` 时，`w_0(d)=1/(1+lambda)` 为正常数，结论直接由定理 3 得到。以下设 `p>0`。对充分大的 `d`，`lambda(d/d_ref)^p>=1`，于是

$$
\frac{d_{\mathrm{ref}}^p}{2\lambda}d^{-p}
\le w_p(d)\le
\frac{d_{\mathrm{ref}}^p}{\lambda}d^{-p}.
$$

与定理 3 的 `C_1d^2<=||g_y^-||<=C_2d^2` 相乘，即得两侧同阶界 `Theta(d^{2-p})`；三种极限由 `2-p` 的符号立即得到。证毕。

**直接推论。**

$$
w_{\mathrm{lin}}(d)=\frac{1}{1+\lambda d/d_{\mathrm{ref}}}\quad\Rightarrow\quad \lVert w_{\mathrm{lin}}g_y^-\rVert=\Theta(d),
$$

仍然无界；

$$
w_{\mathrm{quad}}(d)=\frac{1}{1+\lambda(d/d_{\mathrm{ref}})^2}\quad\Rightarrow\quad \lVert w_{\mathrm{quad}}g_y^-\rVert=\Theta(1),
$$

所以二次 reciprocal 是该正值平滑多项式族中保证有界的最低阶。对

$$
w_{\exp}(d)=e^{-\lambda d/d_{\mathrm{ref}}},
$$

由 `d^2e^{-lambda d/d_ref}->0`，加权影响趋零。

<!-- STAGE4B-SOURCE-BLOCK:B000222:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000223:START -->
### 命题 5（同一参考衰减下的有限距离选择性）

固定 `rho in (0,1)`，令 `lambda=rho^{-1}-1`、`u=d/d_ref`，则

$$
w_{\mathrm{lin}}(u)=\frac1{1+\lambda u},\qquad
w_{\mathrm{quad}}(u)=\frac1{1+\lambda u^2},
$$

并且二者均满足 `w(1)=rho`。因为 `u^2<u` 当 `0<u<1`，而 `u^2>u` 当 `u>1`，所以

$$
0<u<1\Rightarrow w_{\mathrm{quad}}(u)>w_{\mathrm{lin}}(u),
$$

$$
u=1\Rightarrow w_{\mathrm{quad}}(u)=w_{\mathrm{lin}}(u)=\rho,
$$

$$
u>1\Rightarrow w_{\mathrm{quad}}(u)<w_{\mathrm{lin}}(u).
$$

因此在相同 `d_ref` 和参考强度下，二次方法同时具有“近场保留更多、远场压制更强”的解析性质。该命题不需要渐近极限，但仍不推出任务 reward 必然更高。

<!-- STAGE4B-SOURCE-BLOCK:B000223:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000224:START -->
### 对角 Gaussian 推论

对一般 diagonal Gaussian，令

$$
z_j=\frac{a_j-\mu_j}{\sigma_j},\qquad d=\lVert z\rVert_2,
$$

则每个 log-scale 分量为

$$
\frac{\partial\log\pi}{\partial\xi_j}=z_j^2-1.
$$

由 Cauchy--Schwarz 不等式，

$$
\frac{d^2}{\sqrt D}\le \lVert z^{\odot2}\rVert_2\le d^2.
$$

因此

$$
\frac{d^2}{\sqrt D}-\sqrt D
\le
\lVert z^{\odot2}-\mathbf 1\rVert_2
\le
 d^2+\sqrt D.
$$

当 `d^2>=2D` 时，log-scale 联合分支被上下界为常数倍 `d^2`。若各维 `sigma_j>=sigma_min>0`，mean 分支也至多为 `O(d^2)`，故 diagonal Gaussian 的联合 policy-output 梯度仍为 `Theta(d^2)`，定理 4 的临界阶 `p=2` 不变。该推论覆盖后续 state-conditioned diagonal Gaussian 外部验证；tanh-squashed actor 必须在 frozen inverse-squash/base-Gaussian 坐标中使用此距离。

<!-- STAGE4B-SOURCE-BLOCK:B000224:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000225:START -->
### 神经网络 pullback、例外与可声明边界

令 `J_theta(s)=partial(mu_theta,xi_theta)/partial theta`，则

$$
g_\theta^-=J_\theta(s)^\top g_y^-.
$$

若研究区域内 `||J_theta(s)||_op<=M`，则

$$
\lVert w_pg_\theta^-\rVert_2\le M\lVert w_pg_y^-\rVert_2.
$$

故 `p=2` 对实际全参数单样本影响给出充分有界性；但要把 `p=2` 写成全参数空间的必要临界阶，还需 Jacobian 在 log-scale score 方向有统一非退化下界。正式 C-U1 实验直接测量实际全参数梯度，正是为了验证该 pullback 是否保留理论排序。

边界条件必须同时保留：

1. 固定方差时 log-scale 分支不存在，mean score 仅为一次阶，临界多项式阶降为 `p=1`。
2. `p=2` 的**有界上界**只需要 `|A|<=c_max`；“`p<2` 必然无界”的必要性结论还要求沿所讨论的远场序列存在 `|A|>=c_min>0`。C-U1 的等 advantage 设计正是用来隔离这一条件。
3. 若优势幅度本身满足 `|A(d)|=Theta(d^q)`，则总输出梯度阶变为 `Theta(d^{2+q})`，reciprocal-polynomial 的临界阶相应变为 `p=2+q`；当前定理的主情形是 `q=0`。
4. 若允许 `sigma->0` 且不在 pre-boundary 区域设置任何 `sigma_min`，标准化距离的二次 taper只能直接保证 log-scale 分支，不能无条件给出总 mean 分支的统一界；因此 support/variance boundary 必须单独报告。
5. 若权重不 stop-gradient，会额外出现 `grad_theta w`，本定理不适用。
6. `[1-lambda d]_+` 等 clipped-linear 在有限阈值后严格为零，属于 compact-support hard cutoff，不属于 reciprocal-linear 尾部，必须另行分析。
7. 本定理证明控制强度和有界性，不证明 Quadratic、Exp 或其他方法的任务性能排名。


<!-- STAGE4B-SOURCE-BLOCK:B000225:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000226:START -->
## 2.8 Categorical 推论 A：有界单步 score 仍可把策略推到 simplex 边界

对 K 类 full-softmax，logits 为 z，π=softmax(z)。单独重复负更新动作 j，A=−c：


$$
\dot z=c(\pi-e_j)
$$



$$
\frac{d[-\log\pi_j]}{dt}=c\lVert e_j-\pi\rVert^2
$$


direct-logit score 有界：‖eⱼ−π‖≤√2。因此 categorical 不具备 Gaussian 式的单 token 欧氏梯度无界爆炸。但一旦 πⱼ≤ε，Cauchy 不等式给出：


$$
\lVert e_j-\pi\rVert^2\ge\frac{K}{K-1}(1-\varepsilon)^2>0
$$


所以该 token 的 surprisal 至少线性增长，概率至多指数衰减；logit gap 可以趋于无穷，分布被推到概率单纯形边界。动作集合有限并不能阻止 support collapse。

full-softmax 也是指数族，T(a)=eₐ。signed target 为：


$$
\pi^*=\frac{p r_+-q r_-}{p-q}
$$


若某个分量为 0，有限 logits 无法达到，只能令对应 logit→−∞；若某个分量为负，则 target 已离开 simplex，不存在内部解。由此得到离散版的精确 support-feasibility 边界。

Entropy 不是这一动力学的充分统计量：抑制高概率负动作时 entropy 可以先升高，抑制低概率负动作时 entropy 可直接下降；两种路径都可能最终损伤任务支持。因此 entropy control 是必要 baseline，但不能替代对具体危险负更新的选择性诊断。

<!-- STAGE4B-SOURCE-BLOCK:B000226:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000227:START -->
## 2.9 Categorical 推论 B：未见动作外推为何需要语义结构，而不需要“动作有序”

对完全饱和的独立 logits，训练中从未出现的动作在经验 signed target 中通常为 0；纯最大似然/负强化不会凭空知道应把概率放到哪个未见动作。方向性外推必须来自共享参数、预训练先验或动作特征，而不是 token ID 顺序。

更一般地，令动作拥有任意编号和语义特征 φ(a)，使用 energy policy：


$$
\pi_\eta(a\mid s)\propto\exp\!\left\{\eta(s)^\top\phi(a)\right\}
$$


它仍是指数族，稳定点满足：


$$
\mathbb E_{\pi^*}[\phi(a)]=\frac{p\mathbb E_+[\phi(a)]-q\mathbb E_-[\phi(a)]}{p-q}
$$


负样本把目标 feature moment 推离坏动作特征；指数族的最大熵投影会把概率重新分配给具有相似语义、但可能未在正样本中出现的动作。若随机打乱 feature 与 reward 的对应关系，这种 task gain 应消失，而 support suppression 仍然存在。于是“结构破坏”对照不是为有序动作辩护，而是区分两个命题：通用的支持压制不需要结构；有益的未见动作外推需要可泛化结构。

一维 ordinal catalogue 仅保留为可解析的 T=(x,x²) 桥梁；generic categorical 的主要证据应使用随机动作 ID + semantic embedding，而不是人为数轴。

<!-- STAGE4B-SOURCE-BLOCK:B000227:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000228:START -->
## 2.10 神经网络共享参数：指数族输出场的 pullback

令网络输出自然参数 ηθ(s)，Jacobian 为 Jθ(s)=∂ηθ(s)/∂θ。输出空间残差为 r_s(η)=p_s t₊(s)−q_s t₋(s)−(p_s−q_s)m(η)。参数场为：


$$
F_\theta=\mathbb E_s\!\left[J_\theta(s)^\top r_s(\eta_\theta(s))\right]
$$


若存在可实现的 moment-matching 解，使每个相关状态 r_s=0，则网络二阶项在固定点消失，局部 Jacobian 为：


$$
J_F(\theta^*)=-\mathbb E_s\!\left[(p_s-q_s)J_\theta(s)^\top\operatorname{Cov}_{\pi^*}[T]J_\theta(s)\right]
$$


在 p_s\>q_s 且聚合 feature-Fisher 对可训练参数子空间满秩时，该矩阵负定，得到局部稳定性。若多个状态的 signed targets 不能被同一网络同时实现，或固定点残差不为零，网络二阶项重新出现；此时只能使用一般 signed-field Jacobian，而不能声称全局凸性或唯一解。

这一推导说明矩阵形式完全可以保留：真正 general 的对象是 signed off-policy field Jacobian，而不是把 on-policy expected Fisher 直接当作固定样本转移矩阵。

<!-- STAGE4B-SOURCE-BLOCK:B000228:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000229:START -->
## 2.11 方法推论：Global α、Exp-remoteness 与 stability budget

Global α 只改变总负质量 q，简单、稳定，但会无差别削弱近场有用信息。选择性方法令负样本权重依赖当前 policy-relative remoteness。定义连续/离散统一的 remoteness：

$$
S_i=-\log\pi_\theta(a_i\mid s_i),\qquad c_\lambda(S_i)=\exp\{-\lambda(S_i-S_0)_+\}
$$

实现时对 cλ stop-gradient，保证它是纯重权而不是额外可微正则。单负样本的首阶 surprisal 速度变为：

$$
\frac{dS}{dt}=|A|c_\lambda(S)\kappa(S),\qquad \kappa(S)=\lVert\nabla\log\pi\rVert^2
$$

若远场 κ(S) 至多多项式增长，或更一般满足 κ(S)≤Cexp(βS)，则 λ\>β 时加权 influence 有界并在远场衰减。固定方差 Gaussian 的 κ 为 O(S)，含 log-variance 的标准化远场为 O(S²)；direct-logit categorical 的 κ≤2。因此 Exp-remoteness 有一个比“梯度关于距离指数增长”更准确的故事：指数 taper 支配有限阶 score growth，并统一为 categorical 中的 π(a)^λ。

更强的 stability-budget 方法直接使用定理 2 的可行性：先经 cλ 重权得到有效 q_c 与 t₋,c，再选择最大的 batch 系数 γ∈\[0,1\]，使 signed target 保持在 mean-domain 的安全内点。

$$
\gamma^*=\max\{\gamma\in[0,1]:p-\gamma q_c\ge\varepsilon_{\mathrm{mass}},\ \operatorname{dist}(\tau(\gamma),\partial\mathcal M)\ge\varepsilon_{\mathrm{geom}}\}
$$

Gaussian 中可用 p−γq_c\>0 与 σ²\*(γ)≥σ²min 两个闭式条件，计算只需 batch reductions；full-softmax 可约束所有 signed probabilities≥ε。一般 feature policy 的凸包距离较难精确计算，因此 SBRC-Lite 只能使用 score/moment proxy，理论保证相应减弱。

<!-- STAGE4B-SOURCE-BLOCK:B000229:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000230:START -->
## 2.12 Learned critic / value network：瞬时适用与移动目标

在 DRPO-Q、IQL 或一般 actor-critic 中，A_t=Qφ(s,a)−Vψ(s) 会随 critic 更新。只要 actor step 使用 A_t.detach()，上述理论对每一步的瞬时 signed field 仍成立；但整个系统变成非自治动力学，不能把固定 advantage 的全局固定点直接照搬。

若每一时刻都存在内部目标 η\*(t)，局部收缩率下界为 m\>0，且目标漂移速度 ‖η̇\*(t)‖≤v，则标准移动平衡分析给出 tracking error 的量级：


$$
\limsup_t\lVert\eta(t)-\eta^*(t)\rVert\le\frac{v}{m}
$$


因此 critic 越慢、稳定裕度越大，actor 越能跟踪；但任何梯度控制都不能修复 critic 给错 advantage 符号的问题，只能限制错误信号被 score geometry 放大的破坏。

<!-- STAGE4B-SOURCE-BLOCK:B000230:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000231:START -->
## 2.13 自我审查：反例挑战、修正与最终可声明边界

| **挑战**                                       | **审查结果**                                                     | **最终处理**                                                            |
|------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------|
| 单负样本 surprisal 是否在 batch 中仍必增？     | 否；跨样本 Gram 项可反转。                                       | 定理限定为单样本/隔离更新；batch 使用 interference 分解。               |
| expected Fisher SPD 能否证明固定样本联合扩张？ | 不能；pointwise Hessian 一般不定。                               | 以 signed field Jacobian 和指数族 Hessian 取代。                        |
| Gaussian 负样本是否总使 σ 增大？               | 否；far negative 使 σ下降，near negative 使 σ上升。              | 保留 z²−1 四象限，删除 both μ and σ expand。                            |
| 正样本非确定是否自动保证有限 σ？               | 仅当拟合状态后仍有非零条件残差。                                 | 把条件残差或 entropy/KL/σ-min 写成必要来源。                            |
| 有限 categorical 是否不会发散？                | 动作有限，但 logit gap 无界，概率可到 simplex 边界。             | 区分 amplitude runaway 与 support runaway。                             |
| rare token 的 direct-logit score 是否无界？    | 否，范数≤√2。                                                    | 只声称持续 suppression；Fisher 内禀范数与 SGD 梯度分开。                |
| 负优势是否必然带来未见动作泛化？               | 否；无结构 independent logits 不知道往哪里分配。                 | 外推需共享表示/动作特征；加入结构破坏对照。                             |
| entropy 是否等价于 support quality？           | 否；同一 entropy 可对应不同任务支持。                            | entropy control 仅作为 baseline，不作为机制替代。                       |
| Exp 是否由“距离指数增长”直接推出？             | 不完全；score 对距离多为线性/二次。                              | 改为指数 taper 支配多项式 score growth 的有界性论证。                   |
| 指数族全局结论能否直接套神经网络？             | 不能；共享网络可能不可实现，且非凸。                             | 只在 realizable fixed point 给 pullback 局部稳定；其余用一般 Jacobian。 |
| Adam / PPO / importance ratio 是否被定理覆盖？ | 当前定理直接覆盖 gradient flow / Euler 和 detached reweighting。 | 其他优化器、ratio clipping 作为经验扩展，不写成严格推论。               |
| information 随距离下降是否已证明？             | 尚未；需要任务结构和方向可靠性假设。                             | 保留为可检验 hypothesis，不列为已证定理。                               |
| 边界/低熵是否必然导致任务 reward collapse？    | 不必；若边界动作恰为最优可提升。                                 | 区分 support collapse 与 task collapse，后者需环境因果实验。            |

自审结论：目前没有发现会推翻主框架的逻辑缺口。可以严格成立的是“单样本排斥恒等式 + 指数族 signed-moment 可行性 + Gaussian/categorical 分叉 + 局部神经网络 pullback”。仍不能升级为定理的是“方向信息必随距离单调下降”“任意真实任务都由该机制唯一导致 collapse”以及“某一种控制在所有任务上必胜”。

---

<!-- STAGE4B-SOURCE-BLOCK:B000231:END -->
