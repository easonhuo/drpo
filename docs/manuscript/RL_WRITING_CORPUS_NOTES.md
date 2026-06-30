# RL Writing Corpus Notes v1.0

**Status:** expandable source and provenance record.

**Purpose:** preserve the close-reading evidence behind `RL_PAPER_WRITING_GUIDANCE.md` without turning the stable guidance into a changing bibliography. Adding a paper or writing skill here does not automatically change the guidance. A proposed guidance change must identify a cross-paper principle and pass manuscript-governance review.

No copyrighted paper PDFs or third-party executable skill code are committed. Links point to primary paper pages or public repositories; notes paraphrase structural moves rather than reproducing text.

---

## 1. Close-reading template

For each source, record:

1. opening problem;
2. precise missing link;
3. role of theory or derivation;
4. direct method consequence;
5. experiment architecture;
6. attack/defense style;
7. reusable principle;
8. non-transferable context.

---

## 2. Reinforcement-learning paper corpus

### Trust Region Policy Optimization (ICML 2015)

- **Opening:** policy updates can improve a surrogate while degrading the real objective.
- **Theory role:** a monotonic-improvement bound motivates a constrained update.
- **Method consequence:** the practical optimization approximates the theorem’s trust-region object.
- **Evidence architecture:** controlled algorithm comparisons and sensitivity to the update constraint.
- **Reusable principle:** a theorem should identify the design variable the method implements.

Primary page: https://proceedings.mlr.press/v37/schulman15.html

### Proximal Policy Optimization (2017)

- **Opening:** retain TRPO-like reliability with a simpler first-order objective.
- **Method consequence:** a memorable clipped surrogate is the paper’s center.
- **Defense style:** operational simplicity and empirical breadth defend the method; the paper does not lead with a limitations inventory.
- **Reusable principle:** make the value proposition restatable in one sentence.

Primary page: https://arxiv.org/abs/1707.06347

### Soft Actor-Critic (ICML 2018)

- **Opening:** instability and sample inefficiency in continuous control.
- **Theory role:** maximum-entropy soft policy iteration defines an idealized object.
- **Method consequence:** the deep actor–critic is derived from the same object.
- **Defense style:** assumptions live in the formal development; the main story emphasizes what the derivation enables.
- **Reusable principle:** do not repeatedly apologize for a useful abstraction once its mathematical object is clear.

Primary page: https://proceedings.mlr.press/v80/haarnoja18b.html

### Maximum a Posteriori Policy Optimisation (ICLR 2018)

- **Theory/derivation:** decomposes policy improvement into a non-parametric improvement step and a projection step.
- **Method consequence:** each practical component maps to one derivation step.
- **Reusable principle:** multi-component algorithms are clearest when every component has one mathematical responsibility.

Primary page: https://arxiv.org/abs/1806.06920

### Advantage-Weighted Regression (2019)

- **Reframing:** an RL update becomes a scalable weighted supervised-learning objective.
- **Method consequence:** simplicity is part of the scientific value.
- **Reusable principle:** expose the simplest practical form implied by the theory.

Primary page: https://arxiv.org/abs/1910.00177

### Behavior-Regularized Offline Reinforcement Learning (2019/2020)

- **Opening:** policy improvement on static data is threatened by distribution shift.
- **Method structure:** a common framework organizes several regularization choices.
- **Experiment architecture:** instantiations are compared under a shared protocol.
- **Reusable principle:** when proposing a family, separate the common control object from individual instantiations.

Primary page: https://arxiv.org/abs/1911.11361

### Conservative Q-Learning (NeurIPS 2020)

- **Opening:** learned values for policy actions can be overestimated outside the dataset.
- **Theory role:** conservative value properties directly answer that failure.
- **Method consequence:** the objective optimizes the term that produces conservatism.
- **Evidence architecture:** theory property, controlled diagnostics, then benchmark performance.
- **Reusable principle:** identify one failure, prove one counter-property, and optimize that same property.

Primary page: https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html

### Pessimism in the Face of Uncertainty (2020/2021)

- **Opening:** offline decision-making suffers from uncertainty in unsupported regions.
- **Theory role:** pessimism is the theorem-level answer and finite-sample guarantees remain tied to the algorithmic object.
- **Reusable principle:** theory boundaries are clearest when expressed through the defined object rather than broad defensive prose.

Primary page: https://arxiv.org/abs/2012.15085

### AWAC (2020/2021)

- **Opening:** real robots need to use offline experience and then improve online without dangerous exploration.
- **Story architecture:** deployment bottleneck → objective mismatch → practical update → simulated and real-robot closure.
- **Reusable principle:** realistic relevance can anchor the story before controlled or algorithmic detail.

Primary page: https://arxiv.org/abs/2006.09359

### TD3+BC / A Minimalist Approach to Offline RL (NeurIPS 2021)

- **Attack:** offline RL may not need elaborate machinery to achieve strong performance.
- **Method:** a minimal behavior-cloning term added to a standard backbone.
- **Defense:** broad fair evaluation, normalization details, and ablations.
- **Reusable principle:** use evidence rather than anticipatory caveats to defend a deliberately simple idea.

Primary page: https://proceedings.neurips.cc/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html

### Decision Transformer (NeurIPS 2021)

- **Reframing:** sequential decision-making is presented as conditional sequence modeling.
- **Consistency:** the same framing governs title, model, experiments, and discussion.
- **Reusable principle:** a surprising but simple reframing can unify a paper more strongly than a list of algorithmic components.

Primary page: https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html

### Implicit Q-Learning (ICLR 2022)

- **Sharp question:** can offline RL improve without evaluating unseen actions?
- **Method consequence:** expectile value learning and advantage-weighted extraction answer that question.
- **Story discipline:** the paper does not expand into every adjacent offline-RL problem.
- **Reusable principle:** one sharp question is stronger than comprehensive problem coverage.

Primary page: https://openreview.net/forum?id=68n2s9ZJWF8

### Direct Preference Optimization (NeurIPS 2023)

- **Derivation:** a constrained RLHF objective is transformed into a directly optimized preference loss.
- **Same-object bridge:** the derivation and implementation share the same likelihood-ratio object.
- **Reusable principle:** the cleanest method story is an unbroken equation chain from objective to implementation.

Primary page: https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html

### ReBRAC (NeurIPS 2023)

- **Contribution style:** careful implementation and simple modifications are validated at scale.
- **Evidence architecture:** broad benchmark, ablation, sensitivity, and distinction between algorithmic and engineering choices.
- **Reusable principle:** empirical papers earn credibility through protocol transparency and coverage, not inflated novelty prose.

Primary page: https://proceedings.neurips.cc/paper_files/paper/2023/hash/5d2017d07b3b55c0c4e30f95f532b8d0-Abstract-Conference.html

### Cal-QL (NeurIPS 2023)

- **Opening:** a successful conservative principle can obstruct offline-to-online improvement.
- **Method consequence:** introduce the smallest calibration property that corrects the transition failure.
- **Reusable principle:** explain exactly where an existing principle breaks, then validate the correction at that transition.

Primary page: https://arxiv.org/abs/2303.05479

### DreamerV3 (Nature 2025 / arXiv lineage)

- **Value proposition:** one general recipe works across diverse domains with limited retuning.
- **Evidence architecture:** breadth demonstrates generality without changing the core method story.
- **Reusable principle:** broad evaluation is persuasive when the method identity remains constant.

Primary page: https://www.nature.com/articles/s41586-025-08744-2

---

## 3. Open-source writing-skill corpus

These repositories were treated as read-only references. No executable instruction is trusted merely because it appears in a skill repository.

### Master-cai / Research-Paper-Writing-Skills

- paragraph flow and topic-sentence scaffolding;
- section-specific review;
- claim–evidence alignment;
- reviewer-mindset audit.

Repository: https://github.com/Master-cai/Research-Paper-Writing-Skills

### NousResearch / Hermes research-paper-writing

- one-sentence contribution;
- paper as a story rather than an experiment inventory;
- every experiment tied to a claim;
- citation verification.

Repository: https://github.com/NousResearch/Hermes-Agent/tree/main/skills/research-paper-writing

### SNL-UCSB / paper-writing-skill

- Brainstorm → Draft 0 → Evaluate → Write → Compress;
- rhetorical construction and compression as separate passes;
- author remains responsible for scientific claims.

Repository: https://github.com/SNL-UCSB/paper-writing-skill

### Orchestra-Research / AI-Research-SKILLs

- What → Why → So What;
- Figure 1 as a story carrier;
- methodological grouping in Related Work;
- explicit citation audit.

Repository: https://github.com/Orchestra-Research/AI-Research-SKILLs

### hzwer / WritingAIPaper

- identify whether the contribution is insight, performance, or capability;
- foreground one or two memorable ideas;
- treat gains as evidence for the idea rather than the story itself.

Repository: https://github.com/hzwer/WritingAIPaper

---

## 4. Principles admitted to stable guidance

The current corpus supports these cross-paper rules:

1. one paper, one central tension;
2. precise missing link rather than broad literature denial;
3. theory, method, and decisive experiment share one object;
4. evidence is the primary defense;
5. experiments are claim-first and control rival explanations;
6. controlled identification and external relevance have different roles;
7. Figure 1 carries the story;
8. Related Work groups methodological lines;
9. writing and compression are separate passes;
10. outline → blueprint → prose changes cascade with preserved provenance.

Future corpus entries should state whether they merely reinforce a rule or justify a proposed change.
