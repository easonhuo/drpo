# DRPO full-paper prose draft v0.1

Parent: `docs/paper_rewrite_blueprint_v0_6.md`. This is an automatically generated scientific-completeness draft. Formal external results remain explicit TBDs. Structured metadata inside each block supports bidirectional synchronization.

# Abstract

<!-- MANUSCRIPT:BEGIN ABSTRACT-P01 -->
## [ABSTRACT-P01] Paper Summary
Parent-Blueprint-SHA256: `51957be2e7338988f3514c5ca693198ac63e9c0b53d6b8b81d1197855062ff99`

**Claim:** Negative feedback is useful for policy improvement, but historical negative actions can become far-field and exert excessive influence; DRPO controls that transition.

**Reader question:** What is the problem, missing link, theory, method, evidence, and implication in one compact sequence?

**Role:** Summarize the full paper without fixed-advantage scope language, split-manuscript framing, or unverified numerical claims.

**Logical moves:**
- Negative feedback as a resource
- badness--distance isolation
- stable extrapolation to equilibrium loss
- DRPO attenuation of the far-field component
- controlled and external evidence

**Evidence use:**
- Theorem 1
- source-isolation protocols
- targeted interventions
- external tasks marked TBD until formal closure

**Body:**

Negative feedback is a central resource in policy optimization: it suppresses known bad behavior and can move a policy beyond the solution obtained from positive examples alone. Yet the same feedback can become destructive when historical negative actions remain in offline data, replay, or stale trajectories after the learner has moved away from them. We identify the missing mechanism by separating sample badness from learner-relative distance or rarity, showing that policy remoteness independently amplifies negative influence. We then develop Repulsive Dynamics, which characterizes a transition from the Positive-only target, through stable extrapolation, to a feasibility boundary and the loss of a finite equilibrium. Based on this analysis, Distributionally Robust Policy Optimization (DRPO) reweights the aggregate negative term with a policy-relative exponential envelope, retaining useful local repulsion while suppressing its far-field tail. Controlled continuous and categorical environments isolate the source and causal transmission of the effect; registered Hopper/D4RL and Countdown experiments are reserved for external validation under learned critics and shared-parameter sequence policies, with formal results remaining TBD until terminal audit. The resulting framework tests a general principle: policy optimization should control far-field negative influence rather than remove negative feedback as a whole.
<!-- MANUSCRIPT:END ABSTRACT-P01 -->

# Introduction

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource
Parent-Blueprint-SHA256: `8e353e1fc6104dba254ce1f43878293d2af65629b39c3db61c81cf6afcfb96ec`

**Claim:** Negative feedback is not merely noise: balanced against positive attraction, it can suppress bad modes and shift a policy beyond the Positive-only target.

**Reader question:** Why should policy optimization retain negative feedback at all?

**Role:** Open with the constructive role of failures and establish Positive-only as a stable but limited reference.

**Logical moves:**
- positive attraction reinforces observed successes
- negative feedback suppresses known bad modes
- balanced repulsion can shift the equilibrium beyond observed positive behavior
- central question: preserve benefit without excessive repulsion

**Evidence use:**
- Theorem 1 stable-extrapolation regime
- controlled Positive-only comparison

**Body:**

Policy optimization learns not only from successful actions but also from actions that should become less likely. Positive updates reinforce observed successes, whereas negative updates suppress known bad modes and shape the boundary between acceptable and unacceptable behavior. This distinction matters because a learner trained only on positive examples is naturally pulled toward the empirical positive target. When negative repulsion is balanced against that attraction, however, the policy can settle beyond the Positive-only target and improve in regions that were not directly demonstrated by the positive data. The central problem is therefore not whether negative feedback should be used, but how to preserve its policy-improvement value without allowing the same repulsion to become dynamically excessive.
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion
Parent-Blueprint-SHA256: `ae2c0bb538f1c34d53e2ec5593ab52dbe7e1e88df8c175bcc222c47180ffed50`

**Claim:** Offline logs, replay, stale actors, and asynchronous trajectories continue to reuse negative actions after the current learner has moved away, turning local feedback into persistent far-field repulsion.

**Reader question:** How does useful negative feedback become dangerous in off-policy learning?

**Role:** Introduce the temporal mechanism shared by offline, replay-based, stale-policy, and asynchronous training.

**Logical moves:**
- negative actions can initially be locally informative
- historical reuse persists after policy movement
- Gaussian scores can grow with standardized distance
- categorical scores are bounded but suppression persists
- useful-local to destructive-far-field transition

**Evidence use:**
- Gaussian score identity
- categorical log-odds dynamics
- external occurrence diagnostics

**Body:**

The difficulty arises when the data are historical. A negative action may initially lie near the learner and provide relevant local boundary information, but offline logs, replay buffers, stale actors, and asynchronous trajectories can continue to reuse that action after the policy has moved away. Its learner-relative distance or rarity then increases while its negative label continues to generate updates. For Gaussian policies, the score can grow with standardized distance; for categorical policies, the direct-logit score is bounded, yet repeated negative updates continue to reduce the action's log-odds. Historical reuse can therefore transform locally useful feedback into persistent far-field repulsion.
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Existing Controls and the Missing Identification Link
Parent-Blueprint-SHA256: `d59ad63dddb81c0b37b4ff10b1cb53b7cfd3aa55a89cd9fb3cbc2668bef3294e`

**Claim:** Existing controls regulate sign, scale, ratio, support, staleness, or data quality, but they neither explain the useful-to-destructive transition nor isolate policy remoteness from sample badness.

**Reader question:** What do existing methods address, what remains unresolved, and why is strict badness--distance isolation necessary?

**Role:** Combine prior-method positioning with the decisive identification control instead of replacing one with the other.

**Logical moves:**
- positive-only, global scaling, clipping, support constraints, low-probability controls, and quality filtering
- their stabilizing value
- unresolved transition from useful local repulsion to destructive far-field influence
- realistic correlation among reward, negative advantage, rarity, and distance
- matched control holding context, semantics, reward, advantage, count, coefficient, and policy stage fixed

**Evidence use:**
- Related-work synthesis
- C-U1 E1 quality--distance factorization
- D-U1 common/rare matched analogue

**Body:**

Existing methods already regulate negative updates in several ways: positive-only objectives remove them, global coefficients reduce their scale, clipping limits individual ratios, behavior constraints restrict support movement, low-probability controls target rare events, and quality filtering removes selected data. These mechanisms can improve stability, but they do not explain why the same negative feedback is useful near the learner and destructive after it becomes far field. The question is difficult to answer from ordinary logs because low reward, large negative advantage, rarity, and policy distance are typically correlated. We therefore isolate policy remoteness while matching context, action quality or semantic role, reward, negative-advantage magnitude, sample count, base coefficient, and policy stage. This control makes distance or rarity an independently testable amplifier rather than a proxy for samples that are simply worse.
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss
Parent-Blueprint-SHA256: `c3f4795a1c47b3be922cb7b52b990055cfb6e5b4bfd0cda1bb8ff4a57eb13607`

**Claim:** Positive attraction and negative repulsion define a phase sequence from the Positive-only target to stable extrapolation, boundary approach, and loss of finite equilibrium.

**Reader question:** What theoretical structure unifies beneficial and harmful negative updates?

**Role:** Preview Theorem 1 and the continuous--categorical distinction without proof details.

**Logical moves:**
- Positive-only target
- moderate negative contribution produces stable extrapolation
- stronger or more outward contribution moves the signed target toward a feasible boundary
- finite equilibrium can disappear
- Gaussian and categorical manifestations differ

**Evidence use:**
- Theorem 1
- Gaussian and categorical corollaries
- E4/E6 phase tests

**Body:**

Repulsive Dynamics explains the full transition at the level of aggregate policy updates. Positive-only optimization targets the positive moment. A moderate negative contribution shifts the finite equilibrium beyond that target, yielding stable extrapolation. As the negative contribution becomes stronger or more outward, the signed target approaches the feasible boundary of the policy family; beyond that boundary, a finite equilibrium no longer exists. Gaussian and categorical policies share this aggregate phase structure, although they express it differently: Gaussian policies can exhibit distance-amplified scores and coupled mean--support dynamics, whereas categorical policies exhibit bounded but persistent probability suppression toward the simplex boundary.
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term
Parent-Blueprint-SHA256: `e83b5f84c401091babf002b2ff344edefeafa9de99bebed75e12ce2efdb98980`

**Claim:** DRPO reweights the aggregate negative contribution with a policy-relative exponential envelope, retaining local negative feedback while making the weighted far-field gradient vanish under finite-order score growth.

**Reader question:** How does the method act on the theoretical failure mechanism?

**Role:** Present DRPO as the current paper's unified method, not as a sequel or a revision appended to an older paper.

**Logical moves:**
- distributional reweighting of the empirical actor update
- exponential distance/surprisal weight
- direct modification of the Theorem 1 negative term
- far-field vanishing proposition
- quality selection and remoteness control are distinct axes

**Evidence use:**
- method equations
- Proposition 2
- controlled method comparisons

**Body:**

DRPO acts directly on the negative term identified by the theory. It replaces the raw negative contribution with a policy-relative exponential weight, so that nearby negative actions retain substantial influence while increasingly remote actions are attenuated. Under finite-order score growth, the weighted far-field gradient converges to zero. This design is naturally interpreted as distributional control of the empirical actor update: quality-based selection and learner-relative remoteness are distinct axes and are evaluated separately rather than being conflated. The method therefore preserves the local negative feedback responsible for extrapolation while preventing the far-field tail from dominating the aggregate update.
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions
Parent-Blueprint-SHA256: `e4e224c5a63a29ca54fc2dff20662eb05f6002e3fa9d20abdcbbbb8766006e4d`

**Claim:** The paper uses an external--controlled--external evidence chain to establish occurrence, identify source and causality, test the phase transition and DRPO control, and close on external task performance.

**Reader question:** How are the claims divided across environments and experiments?

**Role:** End the Introduction with four research questions and four concise contributions.

**Logical moves:**
- RQ1 external occurrence in Hopper and Countdown
- RQ2 matched source isolation plus targeted causal intervention
- RQ3 phase transition, aggregate-term measurement, and controlled method comparison
- RQ4 external task closure
- C-U1/D-U1 controlled roles and Hopper/Countdown external roles
- terminal audit and separate failure events

**Evidence use:**
- environment responsibility table
- registered experiments
- best and terminal metrics

**Body:**

We evaluate this account through four questions. First, do far-field or rare negative updates become disproportionately influential in realistic policy learning? Second, when badness is matched, does policy remoteness independently generate large negative influence and causally transmit into drift or boundary events? Third, does training follow the phase sequence predicted by Theorem~\ref{thm:equilibrium}, and can DRPO preserve useful repulsion while preventing the destructive regime under fair gradient budgets? Fourth, does the resulting control improve external tasks? C-U1 and D-U1 provide controlled continuous and categorical identification, whereas registered Hopper/D4RL and Countdown experiments are assigned the external-validation role and remain TBD until formal terminal audit. The paper contributes a theory of Repulsive Dynamics, matched quality--distance and quality--rarity causal identification, DRPO control of the theorem-level negative term, and a pre-registered external--controlled--external evidence chain.

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{figures/generated/fig1_story.pdf}
\caption{Paper overview. Negative feedback can move a policy beyond the Positive-only target, but historical reuse can turn local feedback into destructive far-field influence. DRPO attenuates that tail. The evidence chain first establishes external occurrence, then performs matched source and causal identification, and finally returns to external task closure. All curves are conceptual; empirical figures are inserted only after formal terminal-audited completion.}
\label{fig:story}
\end{figure*}
<!-- MANUSCRIPT:END INTRO-P06 -->

# Related Work

<!-- MANUSCRIPT:BEGIN RELATED-P01 -->
## [RELATED-P01] Learning from Negative or Suboptimal Behavior
Parent-Blueprint-SHA256: `4c50fe16beea7e93eb7f0f3d1d88ea23de4f3f5f57022ee6e795ff222b0e33eb`

**Claim:** Prior work establishes that negative or suboptimal data can be useful, but does not characterize the learner-relative transition from useful local repulsion to destructive far-field influence.

**Reader question:** How does the paper differ from positive-only, advantage-weighted, and failure-learning work?

**Role:** Credit prior evidence for negative-feedback value and position the new transition.

**Logical moves:**
- positive-only filtering
- advantage-weighted regression
- failure learning and negative reinforcement
- useful information in negative data
- missing dynamics across learner-relative distance

**Evidence use:**
- AWR and related actor fitting
- negative-feedback literature

**Body:**

Advantage-weighted and filtered policy-learning methods use sample quality to emphasize successful behavior or reduce harmful updates~\cite{peng2019advantage,kostrikov2021offline}. Other work on failures and negative reinforcement shows that suppressing known bad behavior can improve decision boundaries, diversity, or long-horizon performance. We build on this evidence rather than treating all negative data as noise. Our focus is the dynamical transition: the same negative action can be informative when it is near the current learner and excessive after repeated off-policy reuse moves it into the far field.
<!-- MANUSCRIPT:END RELATED-P01 -->

<!-- MANUSCRIPT:BEGIN RELATED-P02 -->
## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates
Parent-Blueprint-SHA256: `d9558be882b5c5f54d6e9f4fa64a909a7973b07f1343ae7517ee059d48c3e8ea`

**Claim:** Clipping, importance correction, stale-policy control, and rare-event regulation address mismatch or scale, while this paper connects repeated learner-relative movement to aggregate repulsion and equilibrium loss.

**Reader question:** How does the paper relate to standard off-policy stabilization?

**Role:** Position the far-field mechanism as complementary to ratio, staleness, and probability controls.

**Logical moves:**
- PPO-style clipping
- off-policy correction
- stale-policy and asynchronous updates
- low-probability actions or tokens
- aggregate equilibrium consequence

**Evidence use:**
- PPO and off-policy references
- Countdown diagnostics

**Body:**

Off-policy algorithms commonly use importance correction, clipping, trust regions, or replay design to control distribution mismatch~\cite{schulman2017proximal,haarnoja2018soft}. Recent analyses of stale or low-probability actions emphasize that rare negative events can dominate updates in shared-parameter policies. Repulsive Dynamics complements these views by following a negative action across repeated reuse: learner-relative remoteness changes its score contribution, the contributions aggregate, and the signed target can move to a policy boundary where a finite equilibrium is lost.
<!-- MANUSCRIPT:END RELATED-P02 -->

<!-- MANUSCRIPT:BEGIN RELATED-P03 -->
## [RELATED-P03] Robust Offline Policy Learning
Parent-Blueprint-SHA256: `bd9c2611f0e892be23d67cfd159984c10a06f3c0e240df51b18ab9108fa4f423`

**Claim:** Offline RL controls value extrapolation, policy support, or data quality; DRPO instead targets the far-field component of signed actor updates while retaining useful local negatives.

**Reader question:** How is DRPO positioned against conservative and behavior-regularized offline RL?

**Role:** Distinguish the actor-dynamics object without dismissing established offline-RL solutions.

**Logical moves:**
- CQL and pessimism
- IQL and in-support value learning
- behavior regularization and TD3+BC-style simplicity
- data filtering
- signed actor update as the distinct object

**Evidence use:**
- CQL/IQL citations
- external D4RL evaluation

**Body:**

Conservative and implicit offline-RL methods control extrapolation by pessimistic value estimation, in-support dynamic programming, or explicit behavior regularization~\cite{kumar2020conservative,kostrikov2021offline,fujimoto2019off}. These approaches address central offline-RL failures and provide strong external baselines. DRPO studies a different but compatible object: the signed actor update itself. Its purpose is to retain useful local negative feedback while attenuating the learner-relative far-field component that can dominate the policy dynamics.
<!-- MANUSCRIPT:END RELATED-P03 -->

# Problem Setup

<!-- MANUSCRIPT:BEGIN SETUP-P01 -->
## [SETUP-P01] Signed Actor Update and Influence Factorization
Parent-Blueprint-SHA256: `6bfa404490d2be2b8bd6625bd5d36a7467db893a4acfe5edf6b03080e451a316`

**Claim:** A negative sample's influence factors into advantage severity and policy-score geometry, while count and directional coherence determine aggregation.

**Reader question:** What is the mathematical object shared by theory, method, and experiments?

**Role:** Define the signed update and isolate badness from geometry.

**Logical moves:**
- signed empirical actor field
- positive and negative advantage parts
- per-sample norm factorization
- aggregation factors

**Evidence use:**
- policy-gradient identity
- per-sample and aggregate diagnostics

**Body:**

Let $\nu$ denote the empirical update distribution induced by offline data, replay, or stale trajectories. We study the signed actor field
\begin{equation}
\mathbf F(\theta)=\mathbb E_{(s,a)\sim\nu}\!\left[\widehat A(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
\label{eq:signed-field}
\end{equation}
Writing $A^+=\max(\widehat A,0)$ and $A^-=\max(-\widehat A,0)$ gives $\mathbf F=\mathbf F^+-\mathbf F^-$. For a negative sample,
\begin{equation}
\|g_i^-\|=A_i^-\,\|\nabla_\theta\log\pi_\theta(a_i\mid s_i)\|.
\label{eq:factorization}
\end{equation}
The first factor is sample badness or severity; the second is learner-relative policy geometry. Sample count, directional coherence, and the network Jacobian determine how individual terms aggregate. Equation~\ref{eq:factorization} motivates matching badness while changing only distance or rarity.
<!-- MANUSCRIPT:END SETUP-P01 -->

<!-- MANUSCRIPT:BEGIN SETUP-P02 -->
## [SETUP-P02] Policy-Relative Far Field
Parent-Blueprint-SHA256: `1eb6bc9b869cb57d6dfe5b9ec1e6e26b76602d858134eb3d72bd10edb0942da0`

**Claim:** Far field is a dynamic relation between a historical action and the current learner, realized by standardized distance for Gaussian policies and surprisal for categorical policies.

**Reader question:** How is remoteness defined across continuous and categorical policies?

**Role:** Define family-specific remoteness without claiming identical amplification laws.

**Logical moves:**
- Gaussian standardized or Mahalanobis distance
- categorical surprisal
- sequence normalized token/completion NLL
- dynamic rather than permanent label

**Evidence use:**
- C-U1 distance coordinate
- D-U1 rarity coordinate
- Countdown NLL coordinate

**Body:**

We use $r_\theta(s,a)$ as a family-indexed remoteness coordinate. For a Gaussian policy, $r_\theta$ is a registered standardized or Mahalanobis distance (or an equivalent calibrated negative log-density). For a categorical policy, it is action surprisal $-\log\pi_\theta(a\mid s)$; for sequence policies, the registered analogue is normalized token- or completion-level negative log-likelihood. The notation is shared, but the score-growth law is not assumed to be identical across policy families. Crucially, far field is dynamic: a fixed historical action can become remote as the current learner changes.
<!-- MANUSCRIPT:END SETUP-P02 -->

# Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN THEORY-P01 -->
## [THEORY-P01] Per-Sample Far-Field Dynamics
Parent-Blueprint-SHA256: `60439d6f0f8ccaba3418106f387496d7ae967d31e086511e38680de3a4ab0874`

**Claim:** Gaussian negative updates can amplify with standardized distance, while categorical negative updates remain bounded in direct-logit norm but persistently suppress probability.

**Reader question:** What happens to one negative action under repeated updates?

**Role:** Establish the family-specific local mechanisms before the aggregate theorem.

**Logical moves:**
- Gaussian mean score identity
- corrected variance sign depends on standardized location
- categorical direct-logit score bound
- log-odds suppression

**Evidence use:**
- analytic derivations in appendices B and C

**Body:**

For a Gaussian policy with mean $\mu$ and covariance $\Sigma$,
\begin{equation}
\nabla_\mu\log\pi(a\mid s)=\Sigma^{-1}(a-\mu),
\label{eq:gaussian-score}
\end{equation}
so the mean-score magnitude grows with standardized distance. When covariance is learned, the variance direction depends on standardized location: the dangerous far-field chain is mean repulsion together with support contraction, not universal expansion of both mean and variance. For categorical logits $z$,
\begin{equation}
\nabla_z\log\pi(a\mid s)=e_a-\pi(\cdot\mid s).
\label{eq:categorical-score}
\end{equation}
This direct-logit score is bounded, but repeated negative updates continue to decrease the selected action's log-odds. The shared phenomenon is persistent movement toward a policy-family boundary, not an identical Euclidean gradient law.
<!-- MANUSCRIPT:END THEORY-P01 -->

<!-- MANUSCRIPT:BEGIN THEORY-P02 -->
## [THEORY-P02] Aggregate Positive--Negative Competition
Parent-Blueprint-SHA256: `438f55366d3f2eb4510744bf56d74d7fcfdf2c2ec9e0f6347e7d338802c13f5b`

**Claim:** For an exponential-family policy, the signed actor field is governed by positive and negative update masses and their sufficient-statistic moments.

**Reader question:** How do per-sample effects combine into a tractable equilibrium model?

**Role:** Define the exact signed objective used by Theorem 1 without paper-wide fixed-advantage disclaimers.

**Logical moves:**
- regular minimal exponential family
- positive and negative masses p and q
- moments m+ and m-
- signed field formula

**Evidence use:**
- exponential-family algebra

**Body:**

Consider a regular minimal exponential-family policy
\begin{equation}
\pi_\eta(a)=h(a)\exp\{\eta^\top T(a)-\psi(\eta)\}.
\end{equation}
Let $p,q\ge 0$ denote aggregate positive and negative update mass, and let $\mathbf m_+$ and $\mathbf m_-$ denote their normalized sufficient-statistic moments. The population signed field is
\begin{equation}
\mathbf F(\eta)=p\mathbf m_+-q\mathbf m_--(p-q)\nabla\psi(\eta).
\label{eq:aggregate-field}
\end{equation}
These quantities define the aggregate positive--negative competition analyzed by the equilibrium theorem.
<!-- MANUSCRIPT:END THEORY-P02 -->

<!-- MANUSCRIPT:BEGIN THEORY-P03 -->
## [THEORY-P03] Theorem 1: Stable Extrapolation and Loss of Finite Equilibrium
Parent-Blueprint-SHA256: `1fee22fcf60c0b0176cdad96ac5919e6eff8aec44bfa17937df45789014ff439`

**Claim:** The aggregate field admits a stable finite equilibrium beyond the Positive-only target when the signed moment remains interior, and loses that equilibrium at or beyond the feasible boundary.

**Reader question:** When does negative feedback help, and when does the same force remove a finite equilibrium?

**Role:** Serve as the theoretical hinge connecting useful negative feedback, phase transition, and the method target.

**Logical moves:**
- Positive-only limit
- stable extrapolation formula
- boundary approach
- no finite equilibrium
- local stability statement

**Evidence use:**
- proof in Appendix A
- phase tests in E4/E6

**Body:**

\begin{theorem}[Stable extrapolation and loss of finite equilibrium]
\label{thm:equilibrium}
Assume $p>q$ in~\eqref{eq:aggregate-field}. Define the signed target
\begin{equation}
\mathbf m^\star=\frac{p\mathbf m_+-q\mathbf m_-}{p-q}.
\label{eq:signed-target}
\end{equation}
If $\mathbf m^\star$ lies in the interior of the mean-parameter space, there is a unique finite equilibrium $\eta^\star$ satisfying $\nabla\psi(\eta^\star)=\mathbf m^\star$. Moreover,
\begin{equation}
\mathbf m^\star-\mathbf m_+=\frac{q}{p-q}(\mathbf m_+-\mathbf m_-),
\end{equation}
so negative repulsion moves the equilibrium beyond the Positive-only target in the direction away from the negative moment. As $\mathbf m^\star$ approaches the feasible boundary, the corresponding natural parameter becomes degenerate or unbounded; if the signed target lies outside the feasible mean-parameter space, no finite equilibrium exists. At an interior equilibrium, $J_F(\eta^\star)=-(p-q)\nabla^2\psi(\eta^\star)$ is negative definite.
\end{theorem}
The theorem gives one phase sequence: Positive-only, stable extrapolation, boundary approach, and loss of finite equilibrium. The experiments align this phase classification with separate measurements of task performance, policy-family boundaries, and numerical state.
<!-- MANUSCRIPT:END THEORY-P03 -->

<!-- MANUSCRIPT:BEGIN THEORY-P04 -->
## [THEORY-P04] Testable Predictions
Parent-Blueprint-SHA256: `7915cfbb24cb39d4e0fedcda0b183bf718ccef16b63be3a473ee4d8ca3bf8dae`

**Claim:** The theorem predicts distinct observable regimes and a targeted recovery when the far-field component of the aggregate negative term is attenuated.

**Reader question:** How is Theorem 1 connected to actual experiments?

**Role:** Map each mathematical regime to a pre-specified intervention and terminal observable.

**Logical moves:**
- Positive-only platform
- controlled negative stable platform
- boundary event
- persistent drift or non-vanishing residual
- DRPO recovery

**Evidence use:**
- E4 continuous phase sweep
- E6 categorical phase sweep
- terminal slope/residual
- support and probability boundaries

**Body:**

Theorem~\ref{thm:equilibrium} yields direct experimental predictions. Removing negative updates should produce a stable Positive-only platform. Adding moderate negative influence should move the policy beyond that platform while retaining a finite terminal state. Increasing negative strength or outward moment should approach a covariance, support, or probability boundary; beyond the feasible region, the policy should exhibit persistent drift or a non-vanishing field residual. Finally, selective attenuation of the far-field component should restore a finite regime while retaining part of the extrapolation benefit. We test these predictions in C-U1 and D-U1, reporting task-performance collapse, support or variance boundaries, and NaN/Inf failures as separate outcomes.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/generated/fig2_phase_map.pdf}
\caption{Schematic phase map induced by Theorem~\ref{thm:equilibrium}. The empirical tests separately identify the Positive-only target, stable extrapolation, boundary approach, and persistent drift or loss of a finite equilibrium.}
\label{fig:phase-map}
\end{figure}
<!-- MANUSCRIPT:END THEORY-P04 -->

<!-- MANUSCRIPT:BEGIN THEORY-P05 -->
## [THEORY-P05] Gaussian and Categorical Corollaries
Parent-Blueprint-SHA256: `370c28060a821d2ac01ffc09708830ad5ec3576dcd1c6e9b4d05db2bb1245a64`

**Claim:** Gaussian and categorical policies instantiate the same aggregate boundary principle through different score and feasibility geometries.

**Reader question:** What is shared and what remains family-specific?

**Role:** Prevent an invalid claim that categorical policies exhibit the same unbounded Euclidean gradient explosion as Gaussian policies.

**Logical moves:**
- Gaussian finite mean/covariance and boundary behavior
- categorical finite logits and simplex boundary
- shared aggregate competition
- different amplification laws

**Evidence use:**
- Appendices B and C

**Body:**

For Gaussian policies, an interior signed target corresponds to finite mean and covariance parameters; far-field negative updates can increase standardized displacement and contract support, causing score amplification and boundary approach. For categorical policies, an interior target corresponds to finite logits, whereas a boundary target drives selected probabilities toward zero or one. Direct-logit score norms remain bounded, but suppression persists. Thus the transferable principle is aggregate repulsion toward a feasibility boundary, while the local amplification law remains policy-family specific.
<!-- MANUSCRIPT:END THEORY-P05 -->

# Distributionally Robust Policy Optimization

<!-- MANUSCRIPT:BEGIN METHOD-P01 -->
## [METHOD-P01] Distributional Reweighting of Signed Actor Updates
Parent-Blueprint-SHA256: `c0380743c09054bcdb5574fe03f2b87bb15149c496598d7138b79207c95a6315`

**Claim:** DRPO controls the signed actor-update distribution by exponentially attenuating learner-relative remote negative updates while leaving positive updates unchanged.

**Reader question:** What is the current paper's self-contained DRPO update?

**Role:** Define the main method as one current formulation, without old-versus-new chronology or an unregistered combined quality-weight objective.

**Logical moves:**
- start from the signed actor field defined in the setup
- define the exponential learner-relative weight on negative updates
- state the complete DRPO field used by the main theory and method experiments
- separate this remoteness control from quality-based selection

**Evidence use:**
- connect to Theorem 1 through the weighted aggregate negative contribution
- refer quality-selection derivation to Appendix~\ref{app:optimistic-dro} without chronology

**Body:**

DRPO applies a policy-relative envelope to the negative component of the signed actor field. Let $r_\theta(s,a)\ge 0$ denote the registered remoteness measure for the policy family and experiment. We define
\begin{equation}
w^-_\lambda(s,a)=\exp[-\lambda r_\theta(s,a)],\qquad \lambda\ge 0,
\label{eq:exp-weight}
\end{equation}
and optimize
\begin{equation}
\mathbf F_{\mathrm{DRPO}}(\theta)=\mathbb E_\nu\!\left[A^+\nabla\log\pi_\theta-A^-w^-_\lambda\nabla\log\pi_\theta\right].
\label{eq:drpo-field}
\end{equation}
Positive updates are unchanged, nearby negative actions retain substantial influence, and increasingly remote negative actions are attenuated. Sample quality and learner-relative remoteness remain distinct axes: the matched experiments vary remoteness without changing badness, while Appendix~\ref{app:optimistic-dro} states the separate Optimistic-DRO quality-selection result. The main DRPO update does not equate the two controls.
<!-- MANUSCRIPT:END METHOD-P01 -->

<!-- MANUSCRIPT:BEGIN METHOD-P02 -->
## [METHOD-P02] Direct Bridge to Theorem 1
Parent-Blueprint-SHA256: `8bdef9f82d704e9a88495d79ece41dd997a46dc15484114f08cbcbffa13312f9`

**Claim:** DRPO replaces the same aggregate negative moment that moves the equilibrium, allowing the method to preserve local displacement while reducing boundary-driving far-field mass.

**Reader question:** Which exact theorem object does DRPO modify?

**Role:** Create an unbroken theory--method equation chain.

**Logical moves:**
- uncontrolled negative moment
- weighted negative moment
- controlled signed target
- local retention and far attenuation

**Evidence use:**
- aggregate-moment diagnostics

**Body:**

In the exponential-family analysis, the uncontrolled negative moment is
\begin{equation}
q\mathbf m_-=\mathbb E[A^-T(a)].
\end{equation}
DRPO replaces it by
\begin{equation}
q_\lambda\mathbf m_{-,\lambda}=\mathbb E[A^-e^{-\lambda r_\theta(s,a)}T(a)],
\label{eq:weighted-negative-moment}
\end{equation}
which induces the controlled target
\begin{equation}
\mathbf m_\lambda^\star=\frac{p\mathbf m_+-q_\lambda\mathbf m_{-,\lambda}}{p-q_\lambda}.
\end{equation}
The method therefore modifies the same term that first creates stable extrapolation and later drives the signed target toward the boundary. Small $r_\theta$ preserves substantial negative influence; large $r_\theta$ is selectively attenuated.
<!-- MANUSCRIPT:END METHOD-P02 -->

<!-- MANUSCRIPT:BEGIN METHOD-P03 -->
## [METHOD-P03] Proposition 2: Vanishing Weighted Far-Field Gradient
Parent-Blueprint-SHA256: `9a371646976c5cd4bc3ca164fa9309fcc042b677e4708fb22ed3237d1c13ef94`

**Claim:** An exponential envelope dominates any finite-order score growth, so the weighted negative gradient vanishes in the far field.

**Reader question:** Why use the exponential form rather than an arbitrary taper?

**Role:** Provide the method-level tail guarantee without inventing a utility-decay law.

**Logical moves:**
- finite-order score-growth condition
- exponential times polynomial tends to zero
- no assumption that sample utility decays exponentially

**Evidence use:**
- analytic proposition
- distance-binned gradient diagnostics

**Body:**

\begin{proposition}[Vanishing weighted far-field gradient]
\label{prop:vanishing}
Suppose the policy score satisfies $\|\nabla_\theta\log\pi_\theta(a\mid s)\|\le C(1+r_\theta(s,a))^k$ for finite $C,k$. For any $\lambda>0$,
\begin{equation}
e^{-\lambda r_\theta(s,a)}\|\nabla_\theta\log\pi_\theta(a\mid s)\|\rightarrow 0
\quad\text{as }r_\theta(s,a)\rightarrow\infty.
\end{equation}
\end{proposition}
The exponential form is therefore a gradient-tail envelope derived from the far-field score-growth bound. Directional utility is measured separately in the controlled experiments.
<!-- MANUSCRIPT:END METHOD-P03 -->

<!-- MANUSCRIPT:BEGIN METHOD-P04 -->
## [METHOD-P04] Controls and Ablations
Parent-Blueprint-SHA256: `b68d2859e623e6d1a442893291cb4e3e4bfe4729e38519ba29ff772f88f47d9c`

**Claim:** Positive-only, global scaling, hard thresholds, linear tapers, and exponential DRPO isolate which benefits come from retaining local negatives, selecting distance, and controlling total gradient budget.

**Reader question:** How will the method be tested against simpler controls?

**Role:** Define comparisons without predeclaring a winner or conflating quality and distance thresholds.

**Logical moves:**
- uncontrolled signed baseline
- Positive-only
- global negative scaling
- linear or reciprocal distance taper
- hard distance threshold
- matched raw negative-gradient budgets

**Evidence use:**
- C-U1 method matrix
- D-U1 analogues
- terminal audit

**Body:**

We compare DRPO against an uncontrolled signed baseline, Positive-only learning, non-selective global scaling, registered linear or reciprocal tapers, and a hard threshold on the same remoteness coordinate. A quality-based hard filter is reported separately because it controls a different axis. Where the protocol calls for budget matching, methods receive the same pre-optimizer raw negative-gradient norm and are evaluated with paired seeds. No method ranking is assumed in advance; best and terminal checkpoints, local-negative retention, far-field retention, task performance, boundary events, and numerical failures are all reported.
<!-- MANUSCRIPT:END METHOD-P04 -->

# Experiments

<!-- MANUSCRIPT:BEGIN EXP-P01 -->
## [EXP-P01] Environments and Evidence Roles
Parent-Blueprint-SHA256: `4334c778d20d183f8906ef411816dedbfaa3e3b522cf76c5bcdf71755712aaaa`

**Claim:** C-U1 and D-U1 provide controlled identification, while Hopper/D4RL and Countdown provide external validity; no environment substitutes for another's scientific responsibility.

**Reader question:** Why are these environments sufficient and what does each one prove?

**Role:** Introduce environment construction before reporting results so the claims cannot appear manufactured by an opaque simulator.

**Logical moves:**
- C-U1 6D context and 2D action
- D-U1 shared semantic categorical actor
- same-distribution held-out contexts
- Hopper learned critic
- Countdown shared parameters
- environment responsibility table

**Evidence use:**
- Appendix D full specifications

**Body:**

We use two primary controlled environments and two external families. C-U1 is a nonlinear state-conditioned Gaussian environment with six-dimensional numerical contexts, two-dimensional actions, a known state-dependent optimum, and independently sampled train and test contexts from the same distribution. It supports matched quality--distance probes, near/far intervention, and mean--support terminal audits. D-U1 is a shared-network categorical environment with unordered semantic actions and a known reward structure; it supports matched quality--rarity probes, common/rare intervention, and probability-boundary audits. We report C-U1 using the precise term held-out-context or unseen-state generalization. Hopper/D4RL adds a learned critic and long-horizon continuous control, while Countdown adds shared Transformer parameters and sequence-level verification. External tasks establish relevance; controlled environments establish source and causality.

\input{tables/environment_roles}
<!-- MANUSCRIPT:END EXP-P01 -->

<!-- MANUSCRIPT:BEGIN EXP-P02 -->
## [EXP-P02] RQ1: External Occurrence
Parent-Blueprint-SHA256: `2a09248f48d147e66f3e4503a1618c9f1489e3aed643ae21a2854308637f008e`

**Claim:** Hopper and Countdown test whether far-field or rare negative influence becomes disproportionately large before task degradation in realistic policy learning.

**Reader question:** Does the phenomenon occur outside controlled environments?

**Role:** Use external evidence as a reality anchor without treating it as the causal isolation experiment.

**Logical moves:**
- distance/surprisal bins
- positive/negative imbalance
- temporal ordering
- best and terminal checkpoints
- TBD until formal results close

**Evidence use:**
- EXT-H-E7-Q2 terminal-audited outputs
- Countdown formal outputs

**Body:**

\paragraph{Hopper/D4RL.} We train the registered learned critic, freeze its selected checkpoint, and audit negative actor gradients by current-policy distance. The formal report will include far/near negative-gradient ratios, positive/negative imbalance, policy drift, rollout return, and terminal classification. \textbf{TBD:} insert only terminal-audited formal results from the registered E7 package.

\paragraph{Countdown.} We bin negative tokens or completions by current-policy surprisal and track score magnitude, shared-parameter gradient influence, target-probability change, verifier success, validity, and pass@$k$. \textbf{TBD:} insert only the registered formal result; focused pilots remain provenance rather than final evidence.
<!-- MANUSCRIPT:END EXP-P02 -->

<!-- MANUSCRIPT:BEGIN EXP-P03 -->
## [EXP-P03] RQ2a: Matched Badness--Distance and Badness--Rarity Isolation
Parent-Blueprint-SHA256: `892ab51701e233352de180796bc15e89f7cbf1633e5b7207c483ff54a94ab705`

**Claim:** When quality, advantage, semantics, count, coefficient, and policy stage are matched, learner-relative distance or rarity independently changes negative influence.

**Reader question:** Are far or rare gradients larger because the samples are farther, or merely because they are worse?

**Role:** Close the paper's decisive rival explanation with a transparent matched protocol.

**Logical moves:**
- same context and quality coordinate
- same reward and advantage magnitude
- same count and base weight
- only distance or rarity changes
- score and full-parameter gradients
- direction coherence

**Evidence use:**
- C-U1 E1
- D-U1 matched analogue

**Body:**

The source-isolation protocol holds sample badness fixed. In C-U1, negative actions share the same state, reward, advantage magnitude, quality coordinate, count, coefficient, and policy parameters while varying only learner-relative radius. We report output-space score norms, full-parameter per-sample gradients, aggregate gradients, and directional coherence. In D-U1, common and rare actions are matched in semantic role, reward, negative advantage, count, and base coefficient while initial probability or current surprisal changes. A far/near or rare/common gap under these controls identifies policy remoteness as an independent amplifier; it does not imply that distance is the only factor in real data.
<!-- MANUSCRIPT:END EXP-P03 -->

<!-- MANUSCRIPT:BEGIN EXP-P04 -->
## [EXP-P04] RQ2b: Targeted Causal Transmission
Parent-Blueprint-SHA256: `0cb01b5082a499b88a4bc85194e6696331038201a905de001f033dfe4056d421`

**Claim:** Near/far and common/rare interventions test whether the anomalous remote component is the pathway that transmits negative influence into drift, boundary events, and task collapse.

**Reader question:** Do large far-field updates actually cause the observed instability?

**Role:** Move from source identification to causal intervention with equal-budget controls.

**Logical moves:**
- uncontrolled signed baseline
- near/common removal
- far/rare removal
- far/rare cap
- global equal-budget control
- budget transfer
- separate failure events

**Evidence use:**
- C-U1 E3
- D-U1 causal protocol

**Body:**

We next intervene on the location of negative influence. The continuous comparison includes the uncontrolled signed baseline, Near-zero, Far-zero, Far-cap, a non-selective global control matched to the retained raw negative-gradient budget, and a registered budget-transfer control. The categorical analogue suppresses common or rare negatives under matched counts and budgets. We record the onset order of remote influence, parameter drift, support or probability boundaries, and task performance. A selective far/rare rescue together with an ineffective near/common removal supports the remote component as a causal transmission pathway; task-performance collapse, boundary events, and NaN/Inf remain distinct labels.
<!-- MANUSCRIPT:END EXP-P04 -->

<!-- MANUSCRIPT:BEGIN EXP-P05 -->
## [EXP-P05] RQ3: Phase Transition and DRPO Control
Parent-Blueprint-SHA256: `ec032f73a29a042f59b011fa691a8f8727a6e195a89edfd1875368f83eae61cc`

**Claim:** Strength sweeps map the regimes of Theorem 1, while DRPO and matched controls test whether selective far-field attenuation preserves stable extrapolation.

**Reader question:** When does negative feedback become destructive, and can DRPO control that transition?

**Role:** Unify theorem validation, aggregate-term measurement, and method comparison in one result section.

**Logical moves:**
- Positive-only to stable extrapolation to boundary/drift
- continuous and categorical phase maps
- aggregate negative moment proxy
- distance bins
- paired seeds and raw-budget matching
- best and terminal reporting

**Evidence use:**
- C-U1 E4
- D-U1 E6
- registered taper experiments

**Body:**

We sweep the effective negative strength from zero through controlled repulsion and into the unstable regime. The primary test is not a single endpoint score but the phase sequence predicted by Theorem~\ref{thm:equilibrium}: Positive-only, stable extrapolation, boundary approach, and persistent drift or loss of finite equilibrium. We measure a family-specific empirical proxy for the aggregate negative term in~\eqref{eq:weighted-negative-moment}, together with its distance or rarity bins, direction, and relationship to equilibrium displacement. DRPO is compared with Positive-only, global scaling, and registered distance controls under paired seeds and explicit raw-gradient budgets. Results are reported at both selected and terminal checkpoints. \textbf{TBD:} populate the final table only from the current terminal-audited repository artifacts and preserve each result's formal status.

\input{tables/controlled_results}
<!-- MANUSCRIPT:END EXP-P05 -->

<!-- MANUSCRIPT:BEGIN EXP-P06 -->
## [EXP-P06] RQ4: External Task Closure
Parent-Blueprint-SHA256: `a6edabfbf4df43f216e43a93f8de1d9bf8da20cbe7974a859b2b584d50be958f`

**Claim:** External D4RL and Countdown evaluations test whether controlling remote negative influence improves actual task performance without sacrificing the useful negative signal.

**Reader question:** Does the mechanism-targeted control matter on public and sequence tasks?

**Role:** Close the external--controlled--external evidence chain.

**Logical moves:**
- D4RL locomotion datasets
- Countdown model scales and common data bank
- same initialization and selection rules
- best and terminal metrics
- mechanism diagnostics beside performance

**Evidence use:**
- EXT-H-E7-BENCH-01
- EXT-C-E8-SCALE-01

**Body:**

The final evaluation returns to external tasks. On D4RL locomotion, methods share dataset, critic protocol, initialization, seeds, evaluation horizon, and selection rule; we report normalized return, uncertainty, best and terminal checkpoints, and distance-binned negative influence. On Countdown, methods share the SFT initialization, rollout or replay bank, verifier, seeds, and checkpoint-selection protocol; we report greedy success, pass@$k$, validity, terminal degradation, and rare-negative diagnostics. \textbf{TBD:} the main external tables remain unfilled until the corresponding registered formal experiments are terminal-audited and delivered.

\input{tables/external_results}
<!-- MANUSCRIPT:END EXP-P06 -->

# Implications and Conclusion

<!-- MANUSCRIPT:BEGIN DISC-P01 -->
## [DISC-P01] Negative Feedback Is a Resource with a Dynamical Failure Mode
Parent-Blueprint-SHA256: `32f9a9ee54446c500c7b637c970be9fd2cd786637de28dfd925dc12a7059af47`

**Claim:** The transferable lesson is to control learner-relative far-field negative influence rather than classify all negative feedback as harmful.

**Reader question:** What should readers remember beyond the specific method?

**Role:** State the positive principle rather than a list of disclaimers.

**Logical moves:**
- negative feedback supports extrapolation
- historical reuse changes its relevance and magnitude
- control object is remote negative influence

**Evidence use:**
- theory and controlled evidence

**Body:**

Negative feedback is a policy-improvement resource with a dynamical failure mode. It provides boundary information and can shift a policy beyond the Positive-only target. The failure arises when historical actions remain active after becoming remote from the current learner, so their optimization influence no longer tracks their local relevance. The transferable design principle is therefore to control learner-relative far-field negative influence rather than delete negative feedback as a class.
<!-- MANUSCRIPT:END DISC-P01 -->

<!-- MANUSCRIPT:BEGIN DISC-P02 -->
## [DISC-P02] Continuous and Categorical Synthesis
Parent-Blueprint-SHA256: `2bcdbda91749472f16454d88ecf7ada723e414632bcde5f579806dd2538689b0`

**Claim:** Gaussian and categorical policies share aggregate boundary dynamics while differing in local score growth and failure manifestation.

**Reader question:** What is genuinely unified across policy families?

**Role:** Leave one accurate cross-family synthesis.

**Logical moves:**
- Gaussian distance-amplified score and support dynamics
- categorical bounded persistent suppression
- shared feasibility-boundary transition

**Evidence use:**
- family corollaries
- C-U1/D-U1 evidence

**Body:**

The continuous and categorical analyses share an aggregate structure but not an identical local law. Gaussian policies can amplify mean scores with standardized distance and couple repulsion to support contraction. Categorical direct-logit scores remain bounded, yet repeated negative updates can persistently suppress probability. In both cases, aggregate positive--negative competition determines whether the policy remains at an interior equilibrium or moves toward a feasibility boundary.
<!-- MANUSCRIPT:END DISC-P02 -->

<!-- MANUSCRIPT:BEGIN DISC-P03 -->
## [DISC-P03] Conclusion
Parent-Blueprint-SHA256: `83d6f7446b38c7d67dd58ac91cb59e5601d7e221e66a70cbdb45d754dd473a09`

**Claim:** DRPO preserves useful negative feedback by attenuating the remote component of the aggregate term that drives equilibrium loss.

**Reader question:** What is the final three-sentence contribution?

**Role:** Close with theory, identification, and method consequence.

**Logical moves:**
- stable extrapolation
- badness--distance isolation
- far-field causal pathway
- DRPO recovery

**Evidence use:**
- all main claims

**Body:**

Negative feedback can create stable policy improvement beyond Positive-only learning. By separating badness from policy remoteness, we identify how historical negative actions become far-field, dominate the aggregate update, and move the policy toward a feasibility boundary where a finite equilibrium can disappear. DRPO controls this same negative term, preserving useful local repulsion while suppressing the destructive far-field tail.
<!-- MANUSCRIPT:END DISC-P03 -->

# Proofs for Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN APP-PROOF-P01 -->
## [APP-PROOF-P01] Proof of Theorem 1
Parent-Blueprint-SHA256: `54a72f2d7122c32ab224c78974de584809d251d4cd4965972d6d7787cfcb5c57`

**Claim:** The equilibrium theorem follows from the diffeomorphism between natural and mean parameters in a regular minimal exponential family.

**Reader question:** What are the exact existence, uniqueness, boundary, and stability arguments?

**Role:** Provide the full proof outside the main narrative.

**Logical moves:**
- derive aggregate field
- interior mean-space existence
- boundary divergence
- Jacobian

**Evidence use:**
- standard exponential-family properties

**Body:**

For a regular minimal exponential family, $\nabla\psi$ maps the natural-parameter space diffeomorphically onto the interior of the mean-parameter space. Setting~\eqref{eq:aggregate-field} to zero with $p>q$ gives~\eqref{eq:signed-target}. If $\mathbf m^\star$ is interior, uniqueness follows from strict convexity of $\psi$. The displacement identity follows by subtracting $\mathbf m_+$ from~\eqref{eq:signed-target}. If $\mathbf m^\star$ approaches the boundary, no finite natural parameter realizes it; if it lies outside the closure, no policy in the family realizes the target. Differentiating~\eqref{eq:aggregate-field} gives $J_F=-(p-q)\nabla^2\psi$, which is negative definite at an interior point because the Hessian is the positive-definite covariance of the sufficient statistic. The discrete update is locally stable for step sizes whose transition matrix has spectral radius below one.
<!-- MANUSCRIPT:END APP-PROOF-P01 -->

# Gaussian Mean--Variance Derivations

<!-- MANUSCRIPT:BEGIN APP-GAUSS-P01 -->
## [APP-GAUSS-P01] Corrected Gaussian Mean and Variance Dynamics
Parent-Blueprint-SHA256: `e87897765c2b82fcb5e413fb8ee728f61506daf7a52865b44b0866cc3fe71e73`

**Claim:** Far-field negative advantages repel the mean and tend to contract variance when standardized distance exceeds one.

**Reader question:** What corrects the earlier mean-and-variance expansion account?

**Role:** Record the exact score signs and joint equilibrium condition.

**Logical moves:**
- mean score
- log-standard-deviation score
- four sign/location quadrants
- joint equilibrium

**Evidence use:**
- analytic Gaussian score

**Body:**

For a scalar Gaussian with $\xi=\log\sigma$ and $z=(a-\mu)/\sigma$,
\begin{equation}
\partial_\mu\log\pi=\frac{a-\mu}{\sigma^2},\qquad
\partial_\xi\log\pi=z^2-1.
\end{equation}
A negative advantage reverses both score directions. It always pushes the mean away from the action, but its effect on scale depends on $|z|$: a near negative action with $|z|<1$ increases $\sigma$, whereas a far negative action with $|z|>1$ decreases $\sigma$. Positive updates have the opposite signs. A full Gaussian terminal state must satisfy both mean and variance equations; a stationary mean with continuing covariance movement is not a complete policy equilibrium.
<!-- MANUSCRIPT:END APP-GAUSS-P01 -->

# Categorical Boundary Dynamics

<!-- MANUSCRIPT:BEGIN APP-CAT-P01 -->
## [APP-CAT-P01] Categorical Log-Odds and Boundary Behavior
Parent-Blueprint-SHA256: `035e32cf044d0b5341b4831f35dbc830c1c28a07b59c6cfbfa241d193fa12230`

**Claim:** Repeated negative updates produce persistent log-odds decay even though the direct-logit score norm is bounded.

**Reader question:** Why is categorical instability not an unbounded Euclidean gradient explosion?

**Role:** Derive the probability-boundary mechanism accurately.

**Logical moves:**
- softmax score bound
- log-odds update
- probability decay
- shared-parameter caveat

**Evidence use:**
- categorical algebra

**Body:**

For softmax probabilities $p_j$, the score for selected action $y$ is $e_y-p$, whose Euclidean norm is bounded. Under a negative update with fixed local coefficient, the selected logit decreases relative to competing logits, so $z_y-z_j$ falls approximately linearly while $p_y$ can decay exponentially. The failure mode is therefore persistent support suppression toward the simplex boundary. In a shared network, the parameter-space gradient additionally contains the network Jacobian and cross-action interference, which are measured rather than inferred from the direct-logit bound alone.
<!-- MANUSCRIPT:END APP-CAT-P01 -->

# Controlled Environments

<!-- MANUSCRIPT:BEGIN APP-ENV-P01 -->
## [APP-ENV-P01] C-U1 and D-U1 Construction
Parent-Blueprint-SHA256: `325f783c95fd5408e786ac5ec06c829f3ab9833d03a0e1b204d381c90b494a46`

**Claim:** The controlled environments expose ground truth and permit matched isolation without claiming that construction alone establishes external validity.

**Reader question:** How exactly are the controlled environments generated?

**Role:** Provide enough detail for reproducibility and reviewer inspection.

**Logical moves:**
- C-U1 states/actions/reward/hidden optimum
- train/test same distribution
- D-U1 semantic actions
- matched probes
- dynamic near/far

**Evidence use:**
- environment code and manifests

**Body:**

C-U1 samples numerical contexts $s\sim\mathcal N(0,I_6)$ independently for training and testing. A fixed generator maps each state to a positive action, hidden optimal action, negative quality coordinate, and reward surface in $\mathbb R^2$. Source-isolation probes replicate reward and advantage across radii while retaining the same state and quality coordinate. Near/far membership for causal interventions is computed relative to the current policy according to the registered metric. D-U1 uses a shared state-conditioned categorical network over unordered semantic actions; quality/semantics and rarity are factorized so that common/rare probes can be matched in reward, advantage, count, and coefficient. Complete constants, seeds, and invariants are taken from the registered experiment configurations.
<!-- MANUSCRIPT:END APP-ENV-P01 -->

# Experimental Protocols and Terminal Audits

<!-- MANUSCRIPT:BEGIN APP-PROT-P01 -->
## [APP-PROT-P01] Stopping, Budget Matching, and Terminal Classification
Parent-Blueprint-SHA256: `73bf204f0a903aab2d14cb63085372abe1b5b61afb273b263f4445b20596769d`

**Claim:** All dynamic and ranking claims require fixed protocol ownership, paired seeds, explicit budgets, and terminal audits.

**Reader question:** How are false convergence and unfair comparisons prevented?

**Role:** Centralize experiment governance details.

**Logical moves:**
- maximum horizon
- terminal slopes/residuals
- 2x continuation where registered
- raw pre-optimizer negative-gradient budget
- best and terminal

**Evidence use:**
- handoff and registry

**Body:**

Formal protocols specify maximum horizons, evaluation cadence, paired development and held-out seeds, and a terminal audit before execution. A fixed horizon is not itself called convergence. Depending on the registered experiment, terminal classification uses state slopes, update residuals, boundary checks, and a continuation horizon. Budget-matched comparisons use the pre-optimizer raw negative-gradient $\ell_2$ norm unless another coordinate is explicitly frozen. Adam parameter-update norms are recorded separately. Best-validation and terminal checkpoints are both reported, and task-performance collapse, support or variance boundaries, and NaN/Inf failures remain distinct fields.
<!-- MANUSCRIPT:END APP-PROT-P01 -->

# Additional Results and Failure Taxonomy

<!-- MANUSCRIPT:BEGIN APP-RES-P01 -->
## [APP-RES-P01] Additional Tables, Curves, and Negative Results
Parent-Blueprint-SHA256: `54eb4003ec38b4cadfe2a59a27204a58a7b2d71a5b1808479df794cf3b218697`

**Claim:** Appendix results preserve uncertainty, failed runs, and non-ranking outcomes instead of presenting only favorable endpoints.

**Reader question:** What evidence is needed beyond the main figures?

**Role:** Provide placeholders for complete result deposition.

**Logical moves:**
- per-seed tables
- full trajectories
- sensitivity
- negative results
- failure inventory

**Evidence use:**
- formal artifact packages

**Body:**

This appendix will contain per-seed results, full training trajectories, confidence intervals, sensitivity analyses, and terminal classifications for every main comparison. Failed or inconclusive formal runs are indexed rather than silently removed. \textbf{TBD:} populate these materials from the durable formal packages after repository closure; smoke tests, static checks, and focused pilots are not promoted to formal evidence.
<!-- MANUSCRIPT:END APP-RES-P01 -->

# Implementation and Reproducibility

<!-- MANUSCRIPT:BEGIN APP-REPRO-P01 -->
## [APP-REPRO-P01] Code, Data, and Artifact Provenance
Parent-Blueprint-SHA256: `f78d20725f6c41d4db9233ef761ff7c6f6fe41b4a11db23ef37c4f5df02de282`

**Claim:** Every formal claim is bound to code, configuration, seeds, raw outputs, terminal audit, and a source commit.

**Reader question:** How can the study be reproduced and audited?

**Role:** Connect the paper to repository provenance without embedding live status in the outline.

**Logical moves:**
- repository paths
- run manifests
- checksums
- formal package lifecycle
- Overleaf build

**Evidence use:**
- repository scripts and artifacts

**Body:**

The repository records experiment IDs, configurations, seeds, code entry points, result manifests, raw curves, logs, terminal audits, checksums, and source commits. Formal runs progress through registered, running, raw-complete, terminal-audited, packaged, delivered, and applied states. The paper source is generated from the stable-ID manuscript graph, compiled in the imported ICML/Overleaf project, and packaged with a reproducible build command. Exact repository paths and commit hashes will be inserted at submission freeze.
<!-- MANUSCRIPT:END APP-REPRO-P01 -->

# Optimistic Distributional Selection

<!-- MANUSCRIPT:BEGIN APP-DRO-P01 -->
## [APP-DRO-P01] Optimistic-DRO Quality Selection and Its Distinct Role
Parent-Blueprint-SHA256: `baeb28d00af59485667d08981ad9bb489cdad059ad996e89e0cf5caba9f7b979`

**Claim:** The Optimistic-DRO subdistribution problem yields a quality-based hard-selection solution; this result is part of the current manuscript but is mathematically distinct from learner-relative remoteness tapering.

**Reader question:** What does the Optimistic-DRO result establish in the current paper, and what does it not establish?

**Role:** Integrate the valid distributional-selection derivation into the current manuscript while keeping quality selection mathematically distinct from learner-relative remoteness tapering.

**Logical moves:**
- state the density-ratio-constrained quality-selection problem
- state its hard-selection solution
- explain the exact scope of that result
- separate it from the remoteness envelope used in the main actor update

**Evidence use:**
- provide the self-contained optimization and solution statement
- cross-reference the main matched badness--distance experiments

**Body:**

Let $P$ denote the empirical data distribution and let $R(s,a)$ be the registered quality signal. The Optimistic-DRO selection problem is
\begin{equation}
\begin{aligned}
\max_Q\quad &\mathbb E_Q[R(s,a)]\\
\text{s.t.}\quad &Q\ll P,\qquad 0\le \frac{dQ}{dP}\le \frac{1}{\kappa},\\
&\int dQ=1.
\end{aligned}
\label{eq:optimistic-dro}
\end{equation}
Its optimum allocates the maximum admissible density to the highest-quality region of $P$, with boundary mass only when needed to satisfy normalization. For a continuous reward distribution this is the top-$\kappa$ hard-selection solution. This result establishes a principled quality-selection operator inside the current DRPO manuscript. It does not identify policy remoteness with low quality and does not derive the exponential envelope in Eq.~\eqref{eq:exp-weight}. Quality selection controls which data are emphasized; remoteness tapering controls how a negative sample's influence changes as the learner moves. The source-isolation experiments keep quality fixed precisely so that the second effect can be identified independently.
<!-- MANUSCRIPT:END APP-DRO-P01 -->

# Additional Theoretical and Reporting Clarifications

<!-- MANUSCRIPT:BEGIN APP-CORR-P01 -->
## [APP-CORR-P01] Additional Theoretical and Reporting Clarifications
Parent-Blueprint-SHA256: `256cd0aa0ed75caafc24611ce87617414ee5dffb5f465aad1b6cca233de0de8b`

**Claim:** The manuscript consistently uses corrected Gaussian scale dynamics, the Jacobian of the actual signed field, held-out-context terminology for C-U1, and separate reporting of task, boundary, and numerical failures.

**Reader question:** Which technical conventions must remain consistent across the main text and appendix?

**Role:** Collect the technical conventions that prevent contradictory statements across theory, experiments, and reporting.

**Logical moves:**
- Gaussian mean repulsion with location-dependent scale dynamics
- actual signed-field Jacobian rather than expected Fisher as the stability object
- held-out-context or unseen-state generalization for C-U1
- separate task collapse, policy-family boundary, and NaN/Inf fields
- terminal audit before convergence or long-run language

**Evidence use:**
- corrected Gaussian derivation
- registered reporting protocol
- terminal-audit governance

**Body:**

The manuscript uses the following technical conventions throughout. Far-field negative Gaussian updates combine mean repulsion with location-dependent scale dynamics and commonly support contraction; they do not universally expand both mean and variance. The expected Fisher matrix describes policy geometry but does not by itself determine stability of a fixed off-policy signed field; local stability is evaluated with the Jacobian of the actual aggregate field. C-U1 test states are independently sampled from the same state distribution and are described as held-out-context or unseen-state generalization, not OOD generalization. Task-performance collapse, support or variance boundaries, and NaN/Inf numerical failure are reported separately. Finally, finite-horizon evidence is not relabeled as convergence or long-run validation without the registered terminal audit.
<!-- MANUSCRIPT:END APP-CORR-P01 -->
