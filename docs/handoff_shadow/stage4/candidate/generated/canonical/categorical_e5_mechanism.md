# Categorical D-U1 E5 repulsion and support boundary

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `categorical_e5_mechanism`
- Responsibility: Cover direct-softmax bounded-score analysis and repeated-update transmission to probability or support boundaries.
- Dependencies: `global_core_governance`, `theory_methods_related_work`, `terminal_audit`
- Content-contract topics: none
- Owned source blocks: 2
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `D-U1-E5-LONGRUN-RERUN`
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000065:START -->
## 3.7.1 D-U1 / D-Diag E5 长程复核 `D-U1-E5-LONGRUN-RERUN`

本实验是 E5 历史结果的正式 provenance 重建与长期复核，不是新的方法排名，也不替代 E6。历史代码和 raw artifact 未进入当前 Git 历史，因此本轮只继承已锁定的科学职责、解析初值、方法角色和 qualitative 参照；所有重建参数均在本节和 registry 中一次性冻结。

1. **D-Diag direct-softmax：** 三动作 full-softmax，目标动作固定负优势 `A=-1`，plain Euler/SGD learning rate `1e-3`，20000 steps。两个精确初态分别匹配旧 handoff 的 `(p0,H0)=(0.8991,0.386)` 与 `(0.0038,0.292)`；保存 target probability、surprisal、entropy、direct-logit score 和 logit gap。
2. **D-U1 causal reconstruction：** 6D contexts 只生成受控 contextual provenance；26 个 action ID 由 semantic offset `[-3,3]` 的随机 permutation 得到，禁止把 ID 顺序解释成语义顺序。positive/near/far 的 offset-spread 分别为 `(0,1.2)`、`(-0.5,0.2)`、`(-2.5,0.2)`，advantage magnitude 固定相等。
3. **优化器与数据：** 每 seed 4096 contexts；positive 4096、near 2048、far 2048 empirical samples；Adam `lr=0.003,betas=(0.9,0.999),eps=1e-8`；正式 seeds 10--29。
4. **方法质量：** `positive_only=(0,0)`、`baseline=(0.25,0.45)`、`near_zero=(0,0.45)`、`far_zero=(0.25,0)`、`far_cap=(0.25,0.03)`、`global_scale=(0.10,0.18)`，元组顺序为 `(near_mass,far_mass)`。不得根据正式结果修改。
5. **终态：** 最大 20000 steps、每 100 steps 评估；W1=`10000--15000`，W2=`15000--20000`。稳定门禁为 W2 `|Δbeta|<=0.02`、`|Δtau|<=0.02`、`|Δreward|<=0.01`、raw-gradient median `<=1e-4`。`tau<=0.05` 或 effective support `<=1.5` 记为 support/temperature boundary；没有内部稳态但 surprisal 继续增长则记 persistent suppression；其他为 inconclusive。
6. **任务阈值：** 每 seed 以同一 seed 的 positive-only terminal reward 为参考，终态 reward 不高于其 20% 记 task-performance collapse。该事件与 support boundary、NaN/Inf 分报。
7. **历史比较：** 旧 20/20 qualitative pattern 是预注册 comparison。正式验收首先要求 120/120 method-seed runs 完整、direct-softmax 数值重建通过、所有终态可审计；是否与历史完全一致必须如实报告，不能作为结果后调参门禁。
8. **执行与 artifact：** canonical hardened guard 负责监督和打包；runner 只写普通 CSV/JSON/PNG/Markdown 和每 5 seeds checkpoint marker，不写 archive。正式运行完成后先交 raw-complete 包，再做 terminal audit 和仓库闭环更新。


<!-- STAGE4B-SOURCE-BLOCK:B000065:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000066:START -->
## 3.7.2 E5 长程复核结果与论文口径

- **运行身份：** `D-U1-E5-LONGRUN-RERUN`，run commit `22c5823d66169eb90c256de342e27c5391e464c3`，formal seeds 10--29，六方法各 20000 steps，120/120 完整。
- **Direct-softmax：** 两个初态均满足 score bound；高概率负动作的 entropy 为 rise-then-fall，低概率负动作 entropy 非增；两者尾段 surprisal/logit-gap slope 均约 `2e-3` per step。该分支证明的是 persistent support suppression，而不是欧氏 logit-gradient amplitude explosion。
- **因果分类：** Baseline/Near-zero 为 task+support 双失败；Far-zero/Far-cap 为两类均救援；Global-scale 保住 task reward 但未保住 support；Positive-only 两类均不失败。每一方法均为 20/20 与历史 qualitative class 一致。
- **事件分离：** task-performance collapse、support/temperature boundary 与 NaN/Inf 继续分开报告。本次三者计数分别依方法变化、支持边界总计 60/120、NaN/Inf 总计 0/120。
- **允许论文表述：** “在该受控 categorical reconstruction 中，bounded direct-logit scores under repeated negative updates still induce monotone surprisal/logit-gap growth and simplex-boundary suppression; selective far-negative removal/capping, but not near-negative removal, breaks the harmful path.”
- **禁止升级：** 不写成旧 runner 逐字节复现、离散欧氏梯度无界、support boundary 等同数值崩溃、E5 已证明未见动作泛化、或 Far-cap/Global-scale 的普遍方法排名。

<!-- STAGE4B-SOURCE-BLOCK:B000066:END -->
