# DRPO full-paper prose draft v0.2

Parent: `docs/paper_rewrite_blueprint_v0_6.md`. The first four chapters are publication-quality candidates generated and gated from detailed sentence-unit blueprints; unclosed external results remain explicit TBDs.

This is one DRPO manuscript being rewritten, not an old paper followed by a sequel.

# Abstract

<!-- MANUSCRIPT:BEGIN ABSTRACT-P01 -->
## [ABSTRACT-P01] Paper Summary
Parent-Blueprint-SHA256: `5a95ad38786ad0a4b05469bfbafd8b03d093b1059b997fa0ecfddd00814f362a`

**Claim:** Negative feedback is useful for policy improvement, but historical negative actions can become far-field and exert excessive influence; DRPO controls that transition.

**Reader question:** What is the problem, missing link, theory, method, evidence, and implication in one compact sequence?

**Role:** Summarize the full paper without fixed-advantage scope language, split-manuscript framing, or unverified numerical claims.

**Topic sentence:** Negative feedback is useful for policy improvement, but historical negative actions can become far-field and exert excessive influence; DRPO controls that transition.

**Logical moves:**
- Negative feedback as a resource
- badness--distance isolation
- stable extrapolation to equilibrium loss
- DRPO attenuation of the far-field component
- controlled and external evidence

**Sentence plan:**

**Evidence use:**
- Theorem 1
- source-isolation protocols
- targeted interventions
- external tasks marked TBD until formal closure

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

Negative feedback is a central resource in policy optimization: it suppresses known bad behavior and can move a policy beyond the solution obtained from positive examples alone. Yet the same feedback can become destructive when historical negative actions remain in offline data, replay, or stale trajectories after the learner has moved away from them. We identify the missing mechanism by separating sample badness from learner-relative distance or rarity, showing that policy remoteness independently amplifies negative influence. We then develop Repulsive Dynamics, which characterizes a transition from the Positive-only target, through stable extrapolation, to a feasibility boundary and the loss of a finite equilibrium. Based on this analysis, Distributionally Robust Policy Optimization (DRPO) reweights the aggregate negative term with a policy-relative exponential envelope, retaining useful local repulsion while suppressing its far-field tail. Controlled continuous and categorical environments isolate the source and causal transmission of the effect; registered Hopper/D4RL and Countdown experiments are reserved for external validation under learned critics and shared-parameter sequence policies, with formal results remaining TBD until terminal audit. The resulting framework tests a general principle: policy optimization should control far-field negative influence rather than remove negative feedback as a whole.
<!-- MANUSCRIPT:END ABSTRACT-P01 -->

# Introduction

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource
Parent-Blueprint-SHA256: `65b9023a6654365d29aee02e2a9bee4ffffb38ea6f019e316fcc6de68c7a06d8`

**Claim:** Off-policy policy optimization needs both attraction toward successful behavior and suppression of known failures, but the latter must remain dynamically controlled.

**Reader question:** Why is negative feedback a necessary policy-improvement signal rather than disposable noise?

**Role:** Establish the broad off-policy setting, the complementary roles of positive and negative feedback, and the paper's governing question.

**Topic sentence:** Historical data are useful precisely because they contain both successful behavior to reinforce and failed behavior to suppress.

**Logical moves:**
- Open with the role of historical data in offline, replay-based, recommendation, and post-training policy optimization.
- Explain positive attraction toward observed successful actions.
- Explain why negative feedback suppresses known failures and competing modes.
- Position Positive-only as stable but potentially limited to the observed positive target.
- State the paper's central problem: retain useful negative information without allowing repulsion to become excessive.

**Sentence plan:**
- {"anchors": ["historical data", "off-policy"], "instruction": "Open with the role of historical data in offline, replay-based, recommendation, and post-training policy optimization.", "role": "context"}
- {"anchors": ["positive updates", "successful"], "instruction": "Explain positive attraction toward observed successful actions.", "role": "positive_role"}
- {"anchors": ["negative feedback", "suppress"], "instruction": "Explain why negative feedback suppresses known failures and competing modes.", "role": "negative_role"}
- {"anchors": ["Positive-only", "empirical positive target"], "instruction": "Position Positive-only as stable but potentially limited to the observed positive target.", "role": "limitation"}
- {"anchors": ["central question", "excessive repulsion"], "instruction": "State the paper's central problem: retain useful negative information without allowing repulsion to become excessive.", "role": "question"}

**Evidence use:**
- offline-RL and off-policy policy-optimization literature
- advantage-weighted and critic-regularized actor fitting
- Theorem 3 stable-extrapolation regime

**Citation refs:**
- levine2020offline
- fujimoto2019off
- peng2019advantage
- wang2020critic

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:** 125--220

**Body:**

Off-policy policy optimization improves a current policy by reusing behavior collected by earlier, stale, or heterogeneous policies. This paradigm is central to offline reinforcement learning, replay-based control, recommender systems, and model post-training because it reduces the need for fresh interaction and exploits experience accumulated across policy iterations~\cite{levine2020offline,fujimoto2019off}. Historical data contain two complementary signals. Positive updates increase the likelihood of observed successful actions, whereas negative feedback suppresses known failures and competing modes~\cite{peng2019advantage,wang2020critic}. The Positive-only endpoint, which removes every negative update, therefore yields a useful stability reference but can restrict the learner to attraction toward the empirical positive target. Balanced repulsion can instead reshape the decision boundary and move the aggregate equilibrium beyond that target. The central question is not whether negative feedback should be kept or discarded wholesale, but how to preserve its policy-improvement value without allowing repeated reuse to turn it into excessive repulsion.
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion
Parent-Blueprint-SHA256: `a42ca5714e6a35a77e3e51f4765c8571d29acf9f929bdd92e4d4f88e3c77879e`

**Claim:** Repeated reuse makes negative influence learner-relative: as the current policy moves away from a fixed historical action, the same label can become persistent or self-amplifying far-field repulsion.

**Reader question:** How can locally informative negative feedback become dynamically dangerous?

**Role:** Introduce historical reuse as the temporal mechanism and distinguish Gaussian amplification from categorical persistent suppression.

**Topic sentence:** The difficulty appears when a fixed negative action is reused after the learner has moved.

**Logical moves:**
- Describe offline logs, replay buffers, stale actors, and asynchronous trajectories as fixed or delayed update sources.
- Explain that a negative action can initially supply locally relevant boundary information.
- Show that repulsion increases learner-relative distance while the historical label remains active.
- State that Gaussian mean scores can grow with standardized distance.
- State that categorical direct-logit scores are bounded but repeated suppression persists.
- Name the useful-local to destructive-far-field transition and hand off to prior controls.

**Sentence plan:**
- {"anchors": ["offline logs", "replay"], "instruction": "Describe offline logs, replay buffers, stale actors, and asynchronous trajectories as fixed or delayed update sources.", "role": "historical_setting"}
- {"anchors": ["locally informative", "boundary"], "instruction": "Explain that a negative action can initially supply locally relevant boundary information.", "role": "local_value"}
- {"anchors": ["learner-relative", "distance"], "instruction": "Show that repulsion increases learner-relative distance while the historical label remains active.", "role": "reuse_mechanism"}
- {"anchors": ["Gaussian", "standardized distance"], "instruction": "State that Gaussian mean scores can grow with standardized distance.", "role": "gaussian_case"}
- {"anchors": ["categorical", "bounded", "suppression"], "instruction": "State that categorical direct-logit scores are bounded but repeated suppression persists.", "role": "categorical_case"}
- {"anchors": ["far-field", "transition"], "instruction": "Name the useful-local to destructive-far-field transition and hand off to prior controls.", "role": "transition"}

**Evidence use:**
- Gaussian score identity
- repeated-reuse theorem
- categorical log-odds dynamics

**Citation refs:**
- schulman2017proximal
- haarnoja2018soft

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:** 125--220

**Body:**

The difficulty appears when negative actions are historical rather than freshly sampled. Offline logs, replay buffers, stale actors, and asynchronous trajectories can keep presenting the same action after the current learner has changed~\cite{schulman2017proximal,haarnoja2018soft}. A negative sample may initially lie near the policy and provide locally informative boundary information. Its update then pushes the learner away, increasing the action's learner-relative distance or surprisal while the stored negative label remains active. In Gaussian policies, the mean score can grow with standardized distance, so subsequent reuse produces progressively larger repulsion. In categorical policies, the direct-logit score is bounded, yet repeated negative updates continue to lower the selected action's log-odds and can drive its probability toward the support boundary. Historical reuse therefore creates a policy-family-dependent transition from useful local feedback to destructive far-field influence, rather than merely exposing the learner to one unusually large update.
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Existing Controls and the Missing Identification Link
Parent-Blueprint-SHA256: `a3f25a65426ca9625e7ac0c3073a10c89991a544d596c9479d16ebe158cea6c1`

**Claim:** Existing controls stabilize off-policy learning, but the causal role of learner-relative remoteness remains unidentified when quality, advantage magnitude, rarity, and distance are entangled.

**Reader question:** What do existing controls solve, and what identification gap remains?

**Role:** Acknowledge established stabilization methods, isolate the missing dynamic claim, and motivate matched quality--distance controls.

**Topic sentence:** Current stabilization methods show that negative updates require control, but they do not identify why repeated remoteness changes their influence.

**Logical moves:**
- Enumerate positive-only, global scaling, clipping, behavior constraints, rarity-aware control, and quality filtering.
- Acknowledge that these controls can stabilize learning and should not be dismissed as tricks.
- Specify the missing explanation of the local-benefit to far-field-harm transition.
- Explain that quality, negative advantage, rarity, and distance are correlated in ordinary logs.
- Describe matched controls that hold context, semantics, reward, coefficient, and policy stage fixed.
- Separate source isolation from causal transmission and lead into the theory.

**Sentence plan:**
- {"anchors": ["positive-only", "clipping", "quality filtering"], "instruction": "Enumerate positive-only, global scaling, clipping, behavior constraints, rarity-aware control, and quality filtering.", "role": "method_landscape"}
- {"anchors": ["stabilize", "controls"], "instruction": "Acknowledge that these controls can stabilize learning and should not be dismissed as tricks.", "role": "acknowledged_value"}
- {"anchors": ["useful local", "far-field"], "instruction": "Specify the missing explanation of the local-benefit to far-field-harm transition.", "role": "unresolved_gap"}
- {"anchors": ["advantage magnitude", "rarity", "distance"], "instruction": "Explain that quality, negative advantage, rarity, and distance are correlated in ordinary logs.", "role": "confounding"}
- {"anchors": ["matched", "policy stage"], "instruction": "Describe matched controls that hold context, semantics, reward, coefficient, and policy stage fixed.", "role": "identification"}
- {"anchors": ["source identification", "intervention protocol"], "instruction": "Separate source isolation from causal transmission and lead into the theory.", "role": "transition"}

**Evidence use:**
- related-work synthesis
- C-U1 E1 source isolation
- C-U1 E3 targeted intervention
- D-U1 common/rare analogue

**Citation refs:**
- schulman2017proximal
- kumar2020conservative
- kostrikov2021offline
- peng2019advantage

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:** 145--250

**Body:**

Existing methods already regulate negative or off-policy updates in several principled ways. Positive-only objectives remove negative terms; global coefficients and clipping reduce their scale; trust regions and behavior constraints restrict policy movement; rarity-aware rules target low-probability events; and quality filtering removes selected data~\cite{schulman2017proximal,kumar2020conservative,kostrikov2021offline,peng2019advantage}. These controls can improve stability, but they do not by themselves explain why the same negative action can be useful near the learner and destructive in the far field after repeated reuse. The issue is difficult to identify in ordinary logs because low reward, large negative advantage, rarity, and learner-relative distance are typically correlated. We therefore construct matched controls that use matching context and semantic role, holding negative-advantage magnitude, sample count, base coefficient, and policy stage fixed to isolate policy remoteness. The source-identification protocol tests whether remoteness independently changes gradient scale; a separate intervention protocol tests whether the resulting far-field influence transmits into drift, support-boundary events, and task-performance collapse. This separation motivates an aggregate theory rather than another static weighting heuristic.
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss
Parent-Blueprint-SHA256: `44d6169f68b62c0de7f768ea5867316b95f5a93a45eb409558206ea522947345`

**Claim:** Repulsive Dynamics unifies stable extrapolation, persistent drift, and instability through the aggregate balance of positive and negative update masses and moments.

**Reader question:** What theoretical structure connects beneficial and harmful negative updates?

**Role:** Preview the static score relation, repeated-reuse asymmetry, aggregate equilibrium theorem, and policy-family manifestations.

**Topic sentence:** Repulsive Dynamics separates three layers that are often conflated: static score geometry, repeated sample reuse, and aggregate equilibrium.

**Logical moves:**
- Introduce the three-layer theoretical decomposition.
- Explain finite stable extrapolation beyond the Positive-only target under positive dominance.
- Explain boundary approach, persistent drift, and loss of finite stability.
- Contrast Gaussian unbounded score growth with categorical bounded-gradient boundary degeneration.
- State that negative feedback is neither uniformly beneficial nor uniformly harmful.
- Connect the theoretical failure term to the method design.

**Sentence plan:**
- {"anchors": ["learner-relative remoteness", "aggregate"], "instruction": "Introduce the three-layer theoretical decomposition.", "role": "theory_object"}
- {"anchors": ["stable extrapolation", "Positive-only"], "instruction": "Explain finite stable extrapolation beyond the Positive-only target under positive dominance.", "role": "stable_regime"}
- {"anchors": ["persistent drift", "finite"], "instruction": "Explain boundary approach, persistent drift, and loss of finite stability.", "role": "boundary_regime"}
- {"anchors": ["Gaussian", "categorical", "bounded"], "instruction": "Contrast Gaussian unbounded score growth with categorical bounded-gradient boundary degeneration.", "role": "family_difference"}
- {"anchors": ["neither", "beneficial", "harmful"], "instruction": "State that negative feedback is neither uniformly beneficial nor uniformly harmful.", "role": "implication"}
- {"anchors": ["method", "far-field"], "instruction": "Connect the theoretical failure term to the method design.", "role": "transition"}

**Evidence use:**
- Proposition on remoteness and score response
- reuse theorem
- aggregate equilibrium theorem
- policy-family runaway theorem

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**
- app:proof-score-remoteness
- app:proof-reuse
- app:proof-theorem-equilibrium
- app:proof-family-runaway

**Word budget:** 135--235

**Body:**

Repulsive Dynamics separates three layers that are often conflated. First, learner-relative remoteness determines the score response of a fixed historical action. Second, repeated reuse creates an asymmetry: positive updates make a successful action more compatible with the learner and attenuate its score, whereas negative updates make a rejected action more remote and preserve or amplify its response. Third, positive and negative contributions aggregate into a signed policy field. When positive mass dominates and the signed target remains inside the feasible mean-parameter space, the policy has a finite stable equilibrium beyond the Positive-only target. At the critical balance the field can produce persistent drift, while negative dominance makes any finite stationary point unstable. The terminal manifestation depends on the policy family: Gaussian mean runaway yields unbounded far-field scores, whereas categorical policies approach the simplex boundary with bounded per-sample logit scores. Negative feedback is therefore neither uniformly beneficial nor uniformly harmful; its value depends on aggregate balance and learner-relative location.
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term
Parent-Blueprint-SHA256: `8d810a573b78815f5f64560bcf3e2990c1f9165bc490b2de1b485b1678c48764`

**Claim:** DRPO preserves informative local negative feedback while attenuating the learner-relative far-field tail that drives the unstable aggregate regime.

**Reader question:** How does the method intervene on the mechanism rather than merely discard negative data?

**Role:** Introduce the method target, local-preservation principle, tail guarantee, and distinction between quality selection and remoteness control.

**Topic sentence:** DRPO acts on the far-field component of the signed actor update rather than deleting negative feedback as a class.

**Logical moves:**
- Identify the aggregate negative term as the intervention target.
- Explain why nearby negative actions retain substantial weight.
- Explain nonlinear attenuation with distance or surprisal.
- State the finite-order score-growth tail guarantee.
- Separate remoteness control from quality-based data selection.
- Lead to the layered empirical program without ranking all tapers in advance.

**Sentence plan:**
- {"anchors": ["negative term", "actor update"], "instruction": "Identify the aggregate negative term as the intervention target.", "role": "method_target"}
- {"anchors": ["nearby", "retain"], "instruction": "Explain why nearby negative actions retain substantial weight.", "role": "local_preservation"}
- {"anchors": ["distance", "surprisal", "attenuation"], "instruction": "Explain nonlinear attenuation with distance or surprisal.", "role": "far_attenuation"}
- {"anchors": ["finite-order", "vanish"], "instruction": "State the finite-order score-growth tail guarantee.", "role": "guarantee"}
- {"anchors": ["quality", "separate axes"], "instruction": "Separate remoteness control from quality-based data selection.", "role": "distinction"}
- {"anchors": ["empirical", "ranking"], "instruction": "Lead to the layered empirical program without ranking all tapers in advance.", "role": "transition"}

**Evidence use:**
- DRPO objective
- vanishing weighted far-field proposition
- controlled global and selective controls

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**
- app:proof-far-field

**Word budget:** 120--215

**Body:**

DRPO acts on the mechanism identified by the theory rather than treating every negative sample as toxic. Starting from the signed empirical actor field, it reweights the aggregate negative term by learner-relative distance or surprisal. Nearby negative actions retain substantial weight and can continue to shape local boundaries or suppress competing bad modes. As an action becomes increasingly remote, the weight decays nonlinearly, preventing the far-field tail from dominating the positive attraction. The exponential envelope is chosen for a precise tail property: under finite-order growth of the unweighted score-times-advantage contribution, the weighted contribution vanishes as remoteness increases. This guarantee does not assume that a sample's utility decays exponentially and does not establish a universal ranking over all tapers. Quality-based selection and learner-relative remoteness control are distinct axes; the experiments evaluate them and global scaling under matched conditions rather than presuming that one control must always win.
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions
Parent-Blueprint-SHA256: `c615658d6a84a58ea2917df61fe39de1fbaed263a30c0d336d80bc1dbcac76d1`

**Claim:** The evidence chain assigns occurrence, controlled source identification, causal transmission, phase testing, and external validity to distinct experiments and reports terminal outcomes with a separated failure taxonomy.

**Reader question:** How do the experiments jointly support the paper without conflating controlled mechanisms and external validity?

**Role:** State the four research questions, environment responsibilities, reporting discipline, and contributions.

**Topic sentence:** The empirical program follows the same decomposition as the theory and assigns each claim to a specific environment.

**Logical moves:**
- State the external occurrence question.
- State the matched source and targeted causal question.
- State the phase and control question.
- State the external task-performance question.
- Separate C-U1/D-U1 controlled roles from Hopper/Countdown external validity and mention terminal audit.
- Summarize the theory, identification, method, and evidence contributions.

**Sentence plan:**
- {"anchors": ["First", "realistic"], "instruction": "State the external occurrence question.", "role": "rq1"}
- {"anchors": ["Second", "matched"], "instruction": "State the matched source and targeted causal question.", "role": "rq2"}
- {"anchors": ["Third", "phase"], "instruction": "State the phase and control question.", "role": "rq3"}
- {"anchors": ["Fourth", "external"], "instruction": "State the external task-performance question.", "role": "rq4"}
- {"anchors": ["C-U1", "D-U1", "Hopper", "Countdown"], "instruction": "Separate C-U1/D-U1 controlled roles from Hopper/Countdown external validity and mention terminal audit.", "role": "environment_roles"}
- {"anchors": ["contribute", "theory", "causal"], "instruction": "Summarize the theory, identification, method, and evidence contributions.", "role": "contributions"}

**Evidence use:**
- environment responsibility table
- registered controlled experiments
- terminal-audited result manifests
- external protocols marked TBD until formal closure

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:** 150--270

**Body:**

The empirical program follows the same decomposition as the theory. First, we ask whether far-field or rare negative updates become disproportionately influential in realistic policy learning. Second, using matched quality--distance and quality--rarity controls, we test whether remoteness independently changes negative influence and whether targeted far-field interventions interrupt drift or boundary events. Third, we test the predicted sequence from the Positive-only platform through stable extrapolation to persistent drift or instability, and compare selective attenuation with global and budget-matched controls. Fourth, we evaluate whether the resulting control improves external tasks. C-U1 and D-U1 provide controlled continuous and categorical identification; Hopper/D4RL and Countdown provide external validity and do not replace those controlled mechanisms. Every dynamics claim is terminal-audited, and task-performance collapse, support or variance-boundary events, and NaN/Inf numerical failure are reported separately. We contribute a layered theory of repeated repulsion, matched causal identification of learner-relative remoteness, DRPO control of the far-field negative term, and an evidence chain that keeps mechanism claims distinct from external task claims.
<!-- MANUSCRIPT:END INTRO-P06 -->

# Related Work

<!-- MANUSCRIPT:BEGIN RELATED-P01 -->
## [RELATED-P01] Learning from Negative or Suboptimal Behavior
Parent-Blueprint-SHA256: `b65225ab052a181014fa6ad3110ad4f7a97211e912afa734be58f144b3812022`

**Claim:** Prior work shows both that negative updates can be filtered for stability and that failures can carry useful learning signal; the missing object is how the same negative action changes as it becomes remote from the learner.

**Reader question:** What is already known about learning from positive, negative, and suboptimal behavior?

**Role:** Synthesize filtering and negative-information literatures, state the established fact, and isolate the repeated-remoteness gap.

**Topic sentence:** Policy-learning methods disagree less about whether quality matters than about how negative information should be used.

**Logical moves:**
- Introduce quality-aware actor fitting as the broad family.
- Describe weighting and filtering of low-advantage actions.
- Explain why failed behavior can still suppress undesirable modes or refine boundaries.
- Synthesize the shared conclusion that negative feedback is potentially informative but risky.
- Identify the missing repeated learner-relative transition.
- Position this paper as dynamics and control, not a rejection of prior filtering.

**Sentence plan:**
- {"anchors": ["quality-aware", "actor"], "instruction": "Introduce quality-aware actor fitting as the broad family.", "role": "prior_family"}
- {"anchors": ["advantage-weighted", "filter"], "instruction": "Describe weighting and filtering of low-advantage actions.", "role": "positive_filtering"}
- {"anchors": ["failed", "boundaries"], "instruction": "Explain why failed behavior can still suppress undesirable modes or refine boundaries.", "role": "negative_value"}
- {"anchors": ["informative", "risk"], "instruction": "Synthesize the shared conclusion that negative feedback is potentially informative but risky.", "role": "established_fact"}
- {"anchors": ["repeated", "learner-relative"], "instruction": "Identify the missing repeated learner-relative transition.", "role": "unresolved_gap"}
- {"anchors": ["dynamics", "complement"], "instruction": "Position this paper as dynamics and control, not a rejection of prior filtering.", "role": "positioning"}

**Evidence use:**
- AWR
- IQL
- CRR
- offline policy-fitting literature

**Citation refs:**
- peng2019advantage
- kostrikov2021offline
- wang2020critic
- zhuang2023behavior

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:** 145--250

**Body:**

Quality-aware policy fitting already provides several ways to use or suppress suboptimal behavior. Advantage-weighted regression, implicit value learning, critic-regularized regression, and proximal behavior objectives emphasize actions estimated to be better while reducing the impact of lower-quality data~\cite{peng2019advantage,kostrikov2021offline,wang2020critic,zhuang2023behavior}. Positive-only and hard-filtering variants take the conservative endpoint and remove selected negative updates altogether. At the same time, failed or suboptimal behavior can remain informative: it can suppress competing bad modes, sharpen a decision boundary, or release probability mass for alternatives represented by the model. The established picture is therefore not that negative feedback is simply noise, but that it combines information value with optimization risk. What remains under-characterized is the temporal relation to the current learner. A fixed negative action can begin as relevant local feedback and become increasingly remote after repeated reuse. We study that transition and its aggregate consequences, complementing rather than dismissing prior weighting and filtering methods.
<!-- MANUSCRIPT:END RELATED-P01 -->

<!-- MANUSCRIPT:BEGIN RELATED-P02 -->
## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates
Parent-Blueprint-SHA256: `3b93e0e91b5ed2e1b81e67c9406c2e827790530886cdc71c552d0ad0192f1d48`

**Claim:** Importance correction, clipping, trust regions, stale-policy controls, and rarity-aware rules regulate mismatch or update scale, but usually treat remoteness as a static signal rather than an endogenous repeated-reuse process.

**Reader question:** How does this work differ from established off-policy, stale-policy, and low-probability update controls?

**Role:** Credit mismatch controls, state what they establish, and isolate the dynamic feedback-loop distinction.

**Topic sentence:** Off-policy learning has long controlled mismatch between the behavior distribution and the current policy.

**Logical moves:**
- Describe importance correction, clipping, and trust regions.
- Describe replay-based, stale-policy, and asynchronous reuse.
- Explain why low probability or surprisal can guide selective attenuation.
- State that these methods establish the importance of learner-relative mismatch.
- Distinguish static weighting from remoteness increased by the negative update itself.
- Position the aggregate-equilibrium analysis as complementary.

**Sentence plan:**
- {"anchors": ["importance", "clipping", "trust"], "instruction": "Describe importance correction, clipping, and trust regions.", "role": "mismatch_controls"}
- {"anchors": ["replay", "stale"], "instruction": "Describe replay-based, stale-policy, and asynchronous reuse.", "role": "stale_updates"}
- {"anchors": ["low-probability", "surprisal"], "instruction": "Explain why low probability or surprisal can guide selective attenuation.", "role": "rarity_controls"}
- {"anchors": ["learner-relative", "mismatch"], "instruction": "State that these methods establish the importance of learner-relative mismatch.", "role": "established_fact"}
- {"anchors": ["endogenous", "repeated"], "instruction": "Distinguish static weighting from remoteness increased by the negative update itself.", "role": "dynamic_gap"}
- {"anchors": ["aggregate", "complement"], "instruction": "Position the aggregate-equilibrium analysis as complementary.", "role": "positioning"}

**Evidence use:**
- PPO
- SAC and replay-based control
- offline-RL review
- D4RL external setting

**Citation refs:**
- schulman2017proximal
- haarnoja2018soft
- levine2020offline
- fu2020d4rl

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:** 145--250

**Body:**

Off-policy learning has long controlled mismatch between the behavior distribution and the current policy. Importance correction, PPO-style clipping, and trust regions limit the effect of samples collected by another policy~\cite{schulman2017proximal}. Replay-based algorithms and maximum-entropy off-policy control likewise manage repeated data reuse while the learner evolves~\cite{haarnoja2018soft}. Offline-RL analyses make the same mismatch explicit when no new interaction is available~\cite{levine2020offline,fu2020d4rl}. Low action probability or high surprisal can therefore serve as a useful learner-relative warning signal. These approaches establish that mismatch and rarity matter; our distinction concerns how they arise. We treat remoteness not only as a static property used to choose a weight, but as an endogenous state variable that a negative update increases and subsequent reuse feeds back into the next update. We then connect that per-sample process to aggregate equilibria and targeted causal interventions. This perspective complements clipping and stale-policy control rather than asserting that they are ineffective.
<!-- MANUSCRIPT:END RELATED-P02 -->

<!-- MANUSCRIPT:BEGIN RELATED-P03 -->
## [RELATED-P03] Robust Offline Policy Learning
Parent-Blueprint-SHA256: `4b722b09d9ed3df8894bd38ce0c2241dbed1057662cb1d9d65ef1ed4d7863021`

**Claim:** Conservative value learning, in-support dynamic programming, behavior regularization, and data selection address major offline-RL failures, while DRPO focuses on the signed actor field and selectively preserves local negative information.

**Reader question:** How is DRPO related to conservative and behavior-regularized offline reinforcement learning?

**Role:** Differentiate value extrapolation, policy support, data selection, and signed-actor dynamics while explaining compatibility.

**Topic sentence:** Offline reinforcement learning contains several distinct failure channels, and far-field signed actor dynamics is only one of them.

**Logical moves:**
- Describe pessimistic value learning and extrapolation-error control.
- Describe in-support or implicit value learning.
- Describe behavior regularization and proximal actor constraints.
- Describe filtering or quality selection as another axis.
- Define the signed actor field as this paper's object.
- Explain that DRPO can be combined with value-side safeguards and is not a universal offline-RL replacement.

**Sentence plan:**
- {"anchors": ["pessimistic", "value"], "instruction": "Describe pessimistic value learning and extrapolation-error control.", "role": "pessimism"}
- {"anchors": ["in-support", "implicit"], "instruction": "Describe in-support or implicit value learning.", "role": "in_support_learning"}
- {"anchors": ["behavior", "proximal"], "instruction": "Describe behavior regularization and proximal actor constraints.", "role": "behavior_regularization"}
- {"anchors": ["selection", "quality"], "instruction": "Describe filtering or quality selection as another axis.", "role": "data_selection"}
- {"anchors": ["signed actor", "negative"], "instruction": "Define the signed actor field as this paper's object.", "role": "distinct_object"}
- {"anchors": ["combined", "not replace"], "instruction": "Explain that DRPO can be combined with value-side safeguards and is not a universal offline-RL replacement.", "role": "compatibility"}

**Evidence use:**
- CQL
- IQL
- off-policy without exploration
- CRR and BPPO

**Citation refs:**
- kumar2020conservative
- kostrikov2021offline
- fujimoto2019off
- wang2020critic
- zhuang2023behavior

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:** 145--255

**Body:**

Offline reinforcement learning contains several distinct failure channels, and far-field signed actor dynamics is only one of them. Conservative Q-learning controls value extrapolation by assigning pessimistic values to unsupported actions~\cite{kumar2020conservative}. Implicit Q-learning avoids querying out-of-distribution actions during value improvement and then fits an actor to supported high-value behavior~\cite{kostrikov2021offline}. Behavior-constrained and proximal approaches instead keep the learned policy close to the data or a behavior reference~\cite{fujimoto2019off,wang2020critic,zhuang2023behavior}. Data selection adds another axis by changing which transitions participate in learning. DRPO focuses on the signed actor field after an advantage-like signal has been supplied: it asks how negative actor updates change as historical actions become remote, and how to preserve their local information without allowing their far-field contribution to dominate. This actor-side control can be combined with pessimistic critics, in-support value learning, or behavior regularization. It is not presented as a replacement for those safeguards or as a complete explanation of every offline-RL failure.
<!-- MANUSCRIPT:END RELATED-P03 -->

# Problem Setup

<!-- MANUSCRIPT:BEGIN SETUP-P01 -->
## [SETUP-P01] Signed Actor Update and Influence Factorization
Parent-Blueprint-SHA256: `3648bb2cb4cbca4b6edfb6415a13031ee569478843f6ec1747f10a9cf5938264`

**Claim:** The analysis studies a historical signed actor field whose per-sample negative influence factors into advantage severity and score geometry, while frequency and directional coherence determine aggregation.

**Reader question:** What exact update field is analyzed, what is held fixed during an actor step, and how are positive and negative contributions separated?

**Role:** Define the historical update distribution, fixed actor-step scope, sign decomposition, influence factorization, aggregation variables, and conclusion boundary.

**Topic sentence:** We analyze the actor-side field generated by repeatedly applying signed updates from a historical distribution.

**Logical moves:**
- Define the historical update distribution over state--action pairs.
- Define the signed empirical actor field and discrete update.
- State that the update distribution and advantage-like labels are fixed with respect to theta during the actor step.
- Define positive and negative advantage components.
- Factor negative per-sample magnitude into severity and score geometry.
- Name frequency and directional coherence as aggregate factors.
- Clarify that the field is not assumed to be the exact on-policy policy gradient.

**Sentence plan:**
- {"anchors": ["historical update distribution", "nu"], "instruction": "Define the historical update distribution over state--action pairs.", "role": "historical_distribution"}
- {"anchors": ["mathbf F", "discrete update"], "instruction": "Define the signed empirical actor field and discrete update.", "role": "actor_field"}
- {"anchors": ["treated as fixed", "actor step"], "instruction": "State that the update distribution and advantage-like labels are fixed with respect to theta during the actor step.", "role": "fixed_step_scope"}
- {"anchors": ["A^+", "A^-"], "instruction": "Define positive and negative advantage components.", "role": "sign_split"}
- {"anchors": ["severity", "score geometry"], "instruction": "Factor negative per-sample magnitude into severity and score geometry.", "role": "influence_factorization"}
- {"anchors": ["frequency", "directional coherence"], "instruction": "Name frequency and directional coherence as aggregate factors.", "role": "aggregation"}
- {"anchors": ["not assumed", "on-policy"], "instruction": "Clarify that the field is not assumed to be the exact on-policy policy gradient.", "role": "boundary"}

**Evidence use:**
- Equation signed-field
- Equation influence-factorization
- registered fixed-advantage mechanism scope

**Citation refs:**
- sutton1999policy
- levine2020offline

**Theorem or equation refs:**
- eq:signed-field
- eq:sign-split
- eq:influence-factorization

**Appendix bindings:**

**Word budget:** 190--340

**Body:**

We analyze the actor-side dynamics generated by repeatedly reusing signed feedback from a historical update distribution. Let $\nu$ denote a distribution over state--action pairs, representing an offline dataset, a replay buffer, or trajectories collected by earlier or stale policies, and let $\widehat A(s,a)$ be the advantage-like coefficient supplied to the actor. During one actor step, both $\nu$ and $\widehat A$ are treated as fixed with respect to $\theta$. The resulting empirical actor field and discrete update are
\begin{equation}
\begin{aligned}
\mathbf F(\theta)
&=\mathbb E_{(s,a)\sim\nu}\!\left[\widehat A(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right],\\
\theta_{t+1}&=\theta_t+\eta\mathbf F(\theta_t).
\end{aligned}
\label{eq:signed-field}
\end{equation}
This field is not assumed to equal the exact on-policy policy gradient, because $\nu$ need not match the current state--action distribution~\cite{sutton1999policy,levine2020offline}. Define the positive and negative components
\begin{equation}
\widehat A^+=\max(\widehat A,0),\qquad \widehat A^-=\max(-\widehat A,0),
\label{eq:sign-split}
\end{equation}
so that $\mathbf F$ decomposes into attraction from $\widehat A^+$ and repulsion from $\widehat A^-$. For one negative sample, the update magnitude factorizes as
\begin{equation}
\left\|\widehat A^-\nabla_\theta\log\pi_\theta(a\mid s)\right\|_2
=\widehat A^-\left\|\nabla_\theta\log\pi_\theta(a\mid s)\right\|_2.
\label{eq:influence-factorization}
\end{equation}
The first factor measures disadvantage severity; the second is learner-relative score geometry. At the aggregate level, sample frequency and directional coherence further determine whether these contributions cancel or reinforce. Freezing $\nu$ and $\widehat A$ is therefore a mechanism-isolation device: it asks what the actor score alone can do under historical reuse, without claiming that critic error, importance ratios, or jointly evolving actor--critic dynamics are absent in full algorithms.
<!-- MANUSCRIPT:END SETUP-P01 -->

<!-- MANUSCRIPT:BEGIN SETUP-P02 -->
## [SETUP-P02] Policy-Relative Far Field
Parent-Blueprint-SHA256: `3003e15dd27ceae5b4b21127c0f5d1270442a7a1dbcb0035c8e7c449ec44c8e7`

**Claim:** Learner-relative remoteness is negative log probability under the current policy; it maps to Mahalanobis distance for Gaussian policies and surprisal for categorical policies without implying identical amplification laws.

**Reader question:** How is far field defined across continuous, categorical, and sequence policies?

**Role:** Define remoteness and score response, give policy-family mappings, state their dynamic status, and prevent false continuous--discrete equivalence.

**Topic sentence:** The common coordinate is how unlikely a historical action is under the current learner, not raw action-space distance.

**Logical moves:**
- Define remoteness as negative log probability under the current policy.
- Map Gaussian remoteness to a covariance constant plus squared Mahalanobis distance.
- Map categorical remoteness to selected-action surprisal.
- Describe normalized token or completion NLL for sequence policies.
- Explain that remoteness changes as theta changes even when data are fixed.
- Define squared score response and warn that its law differs by policy family.
- Lead to the static score-remoteness proposition.

**Sentence plan:**
- {"anchors": ["D_theta", "negative log"], "instruction": "Define remoteness as negative log probability under the current policy.", "role": "definition"}
- {"anchors": ["Mahalanobis", "Gaussian"], "instruction": "Map Gaussian remoteness to a covariance constant plus squared Mahalanobis distance.", "role": "gaussian_mapping"}
- {"anchors": ["categorical", "surprisal"], "instruction": "Map categorical remoteness to selected-action surprisal.", "role": "categorical_mapping"}
- {"anchors": ["sequence", "normalized"], "instruction": "Describe normalized token or completion NLL for sequence policies.", "role": "sequence_mapping"}
- {"anchors": ["dynamic", "theta"], "instruction": "Explain that remoteness changes as theta changes even when data are fixed.", "role": "dynamic_status"}
- {"anchors": ["no universal relation", "not assumed"], "instruction": "Define squared score response and warn that its law differs by policy family.", "role": "non_equivalence"}
- {"anchors": ["Proposition", "score"], "instruction": "Lead to the static score-remoteness proposition.", "role": "transition"}

**Evidence use:**
- Equations remoteness and score-response
- Gaussian and categorical specializations

**Citation refs:**

**Theorem or equation refs:**
- eq:remoteness
- eq:score-response

**Appendix bindings:**

**Word budget:** 170--300

**Body:**

To compare continuous and categorical policies without forcing them to share the same amplification law, we use a common learner-relative coordinate. For a fixed context $s$ and historical action $a$, define the negative log probability
\begin{equation}
D_\theta(s,a)=-\log\pi_\theta(a\mid s).
\label{eq:remoteness}
\end{equation}
For a Gaussian policy, $D_\theta$ equals a scale-dependent constant plus one half of the squared Mahalanobis distance from the policy mean. For a categorical policy, it is exactly the selected action's surprisal. In sequence models the same quantity may be evaluated at token level or normalized across a completion, but the normalization must be fixed before comparing examples. Crucially, remoteness is dynamic: the historical action and its label remain fixed while $D_\theta(s,a)$ changes as the learner moves. We separately define the squared score response
\begin{equation}
R_\theta(s,a)=\left\|\nabla_\theta\log\pi_\theta(a\mid s)\right\|_2^2.
\label{eq:score-response}
\end{equation}
The notation is shared, but no universal relation between $D_\theta$ and $R_\theta$ is assumed. Gaussian mean coordinates can have unbounded response, categorical direct-logit scores are bounded, and shared neural parameters introduce an additional Jacobian. Proposition~\ref{prop:score-remoteness} therefore derives the policy-family-specific static relation before we analyze repeated reuse.
<!-- MANUSCRIPT:END SETUP-P02 -->

# Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN THEORY-P01 -->
## [THEORY-P01] Learner-Relative Remoteness and Static Score Response
Parent-Blueprint-SHA256: `c1dc2714761751bce56e5e3fd649ea5b3600150ca78be975ad6f8e57d3bf5935`

**Claim:** At a fixed policy, Gaussian mean-score magnitude grows without bound with Mahalanobis remoteness, whereas categorical selected-logit response increases with surprisal but saturates.

**Reader question:** How does learner-relative remoteness determine the strength of a single score contribution before any repeated update is considered?

**Role:** Define the static geometry, state the Gaussian and categorical proposition, interpret both cases, and prevent a premature divergence claim.

**Topic sentence:** We begin with a static relation: at a fixed learner, remoteness controls the score response of a historical action.

**Logical moves:**
- Explain why static score geometry must be isolated before repeated dynamics.
- Recall remoteness and score response at a fixed context.
- State the Gaussian covariance-eigenvalue bounds and isotropic exact ordering.
- Interpret the Gaussian response as unbounded mean-score growth with remoteness.
- State the categorical selected-logit formula and full-score bound.
- Interpret rarity as stronger but saturating direct-logit suppression.
- State that a static relation alone does not imply global training divergence.
- Lead to repeated reuse of one fixed sample.

**Sentence plan:**
- {"anchors": ["static relation", "fixed learner"], "instruction": "Explain why static score geometry must be isolated before repeated dynamics.", "role": "motivation"}
- {"anchors": ["D_", "R_"], "instruction": "Recall remoteness and score response at a fixed context.", "role": "definitions"}
- {"anchors": ["lambda_{\\min}", "lambda_{\\max}", "isotropic"], "instruction": "State the Gaussian covariance-eigenvalue bounds and isotropic exact ordering.", "role": "gaussian_statement"}
- {"anchors": ["unbounded", "Gaussian mean score"], "instruction": "Interpret the Gaussian response as unbounded mean-score growth with remoteness.", "role": "gaussian_interpretation"}
- {"anchors": ["1-e", "bounded"], "instruction": "State the categorical selected-logit formula and full-score bound.", "role": "categorical_statement"}
- {"anchors": ["saturat", "suppression"], "instruction": "Interpret rarity as stronger but saturating direct-logit suppression.", "role": "categorical_interpretation"}
- {"anchors": ["does not", "divergence"], "instruction": "State that a static relation alone does not imply global training divergence.", "role": "limitation"}
- {"anchors": ["repeated reuse", "next"], "instruction": "Lead to repeated reuse of one fixed sample.", "role": "transition"}

**Evidence use:**
- Proposition score-remoteness
- Appendix proof score-remoteness

**Citation refs:**

**Theorem or equation refs:**
- prop:score-remoteness
- eq:remoteness
- eq:score-response

**Appendix bindings:**
- app:proof-score-remoteness

**Word budget:** 250--450

**Body:**

We begin with a static relation: at a fixed learner, remoteness controls the score response of a historical action. Fix a context and write $u$ for the policy-output coordinates. Recall from Equations~\eqref{eq:remoteness}--\eqref{eq:score-response} that $D_u(a)=-\log\pi_u(a)$ and $R_u(a)=\|\nabla_u\log\pi_u(a)\|_2^2$.
\begin{proposition}[Learner-relative remoteness and score response]
\label{prop:score-remoteness}
For $\pi_\mu=\mathcal N(\mu,\Sigma)$ with fixed $\Sigma\succ0$ and $C_\Sigma=\tfrac d2\log(2\pi)+\tfrac12\log|\Sigma|$,
\begin{equation}
\begin{aligned}
R_\mu(a)&\ge2\lambda_{\min}(\Sigma^{-1})(D_\mu-C_\Sigma),\\
R_\mu(a)&\le2\lambda_{\max}(\Sigma^{-1})(D_\mu-C_\Sigma).
\end{aligned}
\label{eq:gaussian-score-bounds}
\end{equation}
Along any fixed direction the relation is exactly linear in $D_\mu-C_\Sigma$; in the isotropic case $\Sigma=\sigma^2I$, it reduces to $R_\mu(a)=2(D_\mu-C_\Sigma)/\sigma^2$. For a categorical policy with selected-action probability $p_a=e^{-D_z(a)}$,
\begin{equation}
\left|\partial_{z_a}\log\pi_z(a)\right|=1-e^{-D_z(a)},
\qquad \|\nabla_z\log\pi_z(a)\|_2^2\le2.
\label{eq:categorical-score-bound}
\end{equation}
\end{proposition}
The Gaussian mean score is therefore unbounded as Mahalanobis remoteness increases, with the covariance eigenvalues controlling the amplification rate. Under isotropic covariance, remoteness and Gaussian mean-score magnitude have the same global ordering; under anisotropic covariance the ordering remains bounded by the two eigenvalue slopes. Categorical rarity also strengthens selected-logit suppression, but the response saturates: low probability does not imply an unbounded direct-logit gradient. This distinction matters because a categorical policy may still undergo persistent support suppression even though every per-sample score is finite. Proposition~\ref{prop:score-remoteness} is only a comparison at one fixed learner. It does not establish aggregate divergence, task-performance collapse, or NaN/Inf failure; Appendix~\ref{app:proof-score-remoteness} gives the complete derivation. We next study how repeated reuse changes both remoteness and score response over time.
<!-- MANUSCRIPT:END THEORY-P01 -->

<!-- MANUSCRIPT:BEGIN THEORY-P02 -->
## [THEORY-P02] Self-Attenuation and Self-Amplification under Reuse
Parent-Blueprint-SHA256: `f6bb52aa989741e63ea6fb9eeb36d0e36a5d9b051f77844af80c6b994e4c8327`

**Claim:** For a convex negative log-likelihood along the update path, repeated negative reuse increases remoteness and cannot decrease score response, while sufficiently small positive reuse decreases both.

**Reader question:** How does repeatedly reusing one fixed historical action turn static geometry into a feedback process?

**Role:** State the repeated-reuse theorem, its assumptions, family-specific consequences, and its limit as a single-sample result.

**Topic sentence:** Historical reuse converts the static score relation into opposite positive and negative feedback loops.

**Logical moves:**
- Define one fixed action, negative log likelihood, score response, and repeated update.
- State monotone remoteness increase and non-decreasing response for negative reuse.
- State remoteness decrease and non-increasing response for sufficiently small positive reuse.
- Explain where convexity holds and what coordinate system is intended.
- Connect the theorem to unbounded Gaussian mean-score amplification.
- Connect it to bounded but persistent categorical suppression.
- State that individual-sample amplification does not determine the full aggregate trajectory.
- Lead to aggregate positive--negative competition.

**Sentence plan:**
- {"anchors": ["fixed historical action", "D_t", "R_t"], "instruction": "Define one fixed action, negative log likelihood, score response, and repeated update.", "role": "reuse_setup"}
- {"anchors": ["D_{t+1}", "negative reuse"], "instruction": "State monotone remoteness increase and non-decreasing response for negative reuse.", "role": "negative_statement"}
- {"anchors": ["positive reuse", "L-Lipschitz"], "instruction": "State remoteness decrease and non-increasing response for sufficiently small positive reuse.", "role": "positive_statement"}
- {"anchors": ["convexity", "natural coordinates"], "instruction": "Explain where convexity holds and what coordinate system is intended.", "role": "convexity_scope"}
- {"anchors": ["Gaussian", "unbounded"], "instruction": "Connect the theorem to unbounded Gaussian mean-score amplification.", "role": "gaussian_consequence"}
- {"anchors": ["categorical", "persistent"], "instruction": "Connect it to bounded but persistent categorical suppression.", "role": "categorical_consequence"}
- {"anchors": ["does not determine", "aggregate"], "instruction": "State that individual-sample amplification does not determine the full aggregate trajectory.", "role": "limitation"}
- {"anchors": ["aggregate", "balance"], "instruction": "Lead to aggregate positive--negative competition.", "role": "transition"}

**Evidence use:**
- Theorem reuse
- Appendix proof reuse

**Citation refs:**

**Theorem or equation refs:**
- thm:reuse
- eq:reuse-update

**Appendix bindings:**
- app:proof-reuse

**Word budget:** 260--470

**Body:**

Historical reuse converts the static score relation into opposite positive and negative feedback loops. Fix a historical action $a$, write $D(u)=-\log\pi_u(a)$ and $R(u)=\|\nabla_uD(u)\|_2^2$, and consider the repeated update
\begin{equation}
u_{t+1}=u_t-\eta\widehat A\nabla_uD(u_t).
\label{eq:reuse-update}
\end{equation}
The action and its signed coefficient remain fixed over the reuse window; only the learner coordinates change.
\begin{theorem}[Self-attenuation and self-amplification under reuse]
\label{thm:reuse}
Assume that $D$ is differentiable and convex along the update path. If $\widehat A<0$, then every negative reuse step satisfies
\begin{equation}
D_{t+1}\ge D_t+\eta|\widehat A|R_t,
\qquad R_{t+1}\ge R_t.
\end{equation}
If $\widehat A>0$, $\nabla D$ is $L$-Lipschitz, and $0<\eta\widehat A\le1/L$, then positive reuse satisfies
\begin{equation}
D_{t+1}\le D_t-\tfrac{\eta\widehat A}{2}R_t,
\qquad R_{t+1}\le R_t.
\end{equation}
\end{theorem}
The first pair of inequalities says that negative reuse makes the fixed historical action more remote and cannot weaken its subsequent score response. The second pair gives the complementary positive mechanism: sufficiently small positive reuse makes the successful action more compatible with the learner and attenuates its response. Convexity holds for fixed-covariance Gaussian mean coordinates and categorical logits, and for regular exponential families in natural coordinates because $\nabla^2D$ is a covariance matrix. Combining this theorem with Proposition~\ref{prop:score-remoteness} yields unbounded Gaussian mean-score amplification but bounded, persistent categorical suppression. The result does not determine the aggregate policy trajectory: positive samples regain attractive force as the learner moves, and updates from different samples may cancel. Appendix~\ref{app:proof-reuse} proves the inequalities and the score monotonicity. We therefore turn from the single-sample loop to the aggregate balance of positive and negative historical updates.
<!-- MANUSCRIPT:END THEORY-P02 -->

<!-- MANUSCRIPT:BEGIN THEORY-P03 -->
## [THEORY-P03] Aggregate Attraction--Repulsion Equilibria
Parent-Blueprint-SHA256: `c7dc29c5d0560bbc78d289f8a5c0a64794c924e482df2e17c86e8fa1f0cb1637`

**Claim:** In a regular exponential family, positive dominance yields a unique stable finite equilibrium when the signed target is feasible, equality yields persistent drift, and negative dominance admits no stable finite equilibrium.

**Reader question:** When do positive attraction and negative repulsion aggregate into stable extrapolation, drift, or instability?

**Role:** Derive the aggregate field, state all three regimes, interpret extrapolation, give stability conditions, and connect feasibility to boundary loss.

**Topic sentence:** Per-sample amplification does not imply global divergence because the full field balances all positive and negative historical updates.

**Logical moves:**
- Define the exponential-family policy and positive/negative masses and moments.
- Derive the signed aggregate field.
- State unique finite equilibrium under positive dominance and feasibility.
- Interpret the displacement beyond the Positive-only target.
- Explain boundary approach and loss of finite realization.
- State persistent drift when p=q and moments differ.
- State instability of finite stationary points when p<q.
- Give continuous and discrete local stability conditions.
- Lead to policy-family manifestations under negative dominance.

**Sentence plan:**
- {"anchors": ["exponential-family", "p", "q"], "instruction": "Define the exponential-family policy and positive/negative masses and moments.", "role": "aggregate_setup"}
- {"anchors": ["aggregate field", "policy-dependent mean parameter"], "instruction": "Derive the signed aggregate field.", "role": "field_derivation"}
- {"anchors": ["p>q", "unique finite"], "instruction": "State unique finite equilibrium under positive dominance and feasibility.", "role": "positive_regime"}
- {"anchors": ["m^star-m_+", "beyond"], "instruction": "Interpret the displacement beyond the Positive-only target.", "role": "extrapolation_identity"}
- {"anchors": ["boundary", "no finite"], "instruction": "Explain boundary approach and loss of finite realization.", "role": "boundary_case"}
- {"anchors": ["p=q", "persistent drift"], "instruction": "State persistent drift when p=q and moments differ.", "role": "critical_regime"}
- {"anchors": ["p<q", "unstable"], "instruction": "State instability of finite stationary points when p<q.", "role": "negative_regime"}
- {"anchors": ["Jacobian", "step size"], "instruction": "Give continuous and discrete local stability conditions.", "role": "stability"}
- {"anchors": ["policy-family", "manifestations"], "instruction": "Lead to policy-family manifestations under negative dominance.", "role": "transition"}

**Evidence use:**
- Theorem aggregate equilibrium
- Appendix detailed proof

**Citation refs:**

**Theorem or equation refs:**
- thm:equilibrium
- eq:aggregate-field
- eq:signed-target

**Appendix bindings:**
- app:proof-theorem-equilibrium

**Word budget:** 330--620

**Body:**

Per-sample amplification does not imply global divergence because the full field balances all positive and negative historical updates. At a fixed context, consider a regular minimal exponential-family policy $\pi_\eta(a)=h(a)\exp\{\eta^\top T(a)-\psi(\eta)\}$. Let $p=\mathbb E_\nu[\widehat A^+]$ and $q=\mathbb E_\nu[\widehat A^-]$ denote the effective positive and negative masses, and let $m_+$ and $m_-$ be their normalized sufficient-statistic moments. Expanding the signed objective gives the aggregate field
\begin{equation}
\mathbf F(\eta)=pm_+-qm_--(p-q)\nabla\psi(\eta).
\label{eq:aggregate-field}
\end{equation}
This expression separates the fixed signed target $pm_+-qm_-$ from the policy-dependent mean parameter $\nabla\psi(\eta)$. It is the exact population field for the frozen signed-update model.
\begin{theorem}[Aggregate attraction--repulsion equilibria]
\label{thm:equilibrium}
If $p>q$, define
\begin{equation}
m^\star=\frac{pm_+-qm_-}{p-q}.
\label{eq:signed-target}
\end{equation}
When $m^\star$ lies in the interior mean-parameter space, there is a unique finite equilibrium $\eta^\star$ satisfying $\nabla\psi(\eta^\star)=m^\star$. It is locally asymptotically stable in continuous time and locally stable for a sufficiently small discrete step size. Moreover,
\begin{equation}
m^\star-m_+=\frac{q}{p-q}(m_+-m_-),
\end{equation}
so negative feedback moves the equilibrium beyond the Positive-only target in the direction away from the negative moment. As $m^\star$ approaches the feasible boundary, the realizing natural parameter leaves every compact set; outside the feasible set no finite equilibrium exists. If $p=q$ and $m_+\ne m_-$, the field is constant and produces persistent drift. If $p<q$, any finite stationary point is unstable in continuous and discrete time.
\end{theorem}
The positive-dominant regime therefore permits finite stable extrapolation, but only while the signed target remains realizable. The extrapolation identity is geometric: moving beyond $m_+$ does not by itself guarantee greater task utility. At the critical regime $p=q$, attraction and repulsion cancel only in total mass, leaving a nonzero moment difference that drives linear drift. Under negative dominance, the objective becomes locally convex rather than concave around any stationary point, making that point repelling. Formally, the local Jacobian for $p>q$ is $-(p-q)\nabla^2\psi(\eta^\star)$, while the discrete step requires $\rho(I+\alpha J_F)<1$. Appendix~\ref{app:proof-theorem-equilibrium} gives the existence, boundary, and exact step-size arguments. The theorem classifies the aggregate regimes; the next result derives their policy-family manifestations.
<!-- MANUSCRIPT:END THEORY-P03 -->

<!-- MANUSCRIPT:BEGIN THEORY-P04 -->
## [THEORY-P04] Policy-Family Manifestations of Negative Dominance
Parent-Blueprint-SHA256: `36f31d39a9ea1475165604257e4ded3f3e164b9a840ee30f3c2aaccc499a2a3f`

**Claim:** Negative dominance produces unbounded Gaussian mean displacement and far-field scores, but categorical policies approach the simplex boundary while each direct-logit score remains bounded.

**Reader question:** What does aggregate instability look like in Gaussian and categorical policies, and which failure labels are justified?

**Role:** Derive family-specific runaway, distinguish gradient explosion from bounded-gradient boundary degeneration, and preserve reporting boundaries.

**Topic sentence:** The aggregate instability in Theorem 3 is shared, but its observable signature depends on the policy family.

**Logical moves:**
- Restate negative dominance and the shared absence of a stable finite equilibrium.
- Give the Gaussian affine field around the repelling point.
- Derive unbounded mean displacement and fixed-action score.
- Describe gauge-fixed categorical logits and convex ascent.
- State escape to the simplex boundary with bounded per-sample score.
- Name Gaussian gradient-amplitude runaway versus categorical support degeneration.
- State that neither result alone proves task collapse or NaN/Inf failure.
- Lead to empirically testable regimes and separated event reporting.

**Sentence plan:**
- {"anchors": ["negative dominance", "finite stable equilibrium"], "instruction": "Restate negative dominance and the shared absence of a stable finite equilibrium.", "role": "shared_instability"}
- {"anchors": ["aggregate mean field", "mu^\\dagger"], "instruction": "Give the Gaussian affine field around the repelling point.", "role": "gaussian_form"}
- {"anchors": ["to infinity", "score"], "instruction": "Derive unbounded mean displacement and fixed-action score.", "role": "gaussian_consequence"}
- {"anchors": ["gauge", "logits"], "instruction": "Describe gauge-fixed categorical logits and convex ascent.", "role": "categorical_form"}
- {"anchors": ["simplex boundary", "less than or equal"], "instruction": "State escape to the simplex boundary with bounded per-sample score.", "role": "categorical_consequence"}
- {"anchors": ["gradient-amplitude", "support"], "instruction": "Name Gaussian gradient-amplitude runaway versus categorical support degeneration.", "role": "distinction"}
- {"anchors": ["task-performance", "NaN/Inf"], "instruction": "State that neither result alone proves task collapse or NaN/Inf failure.", "role": "reporting_boundary"}
- {"anchors": ["predictions", "separate"], "instruction": "Lead to empirically testable regimes and separated event reporting.", "role": "transition"}

**Evidence use:**
- Theorem family runaway
- Gaussian derivation
- categorical derivation

**Citation refs:**

**Theorem or equation refs:**
- thm:family-runaway
- eq:gaussian-runaway
- eq:categorical-bound

**Appendix bindings:**
- app:proof-family-runaway
- app:gaussian
- app:categorical

**Word budget:** 300--540

**Body:**

Theorem~\ref{thm:equilibrium} shows that negative dominance admits no finite stable equilibrium, but it does not determine how runaway appears in a particular policy family. Gaussian and categorical policies share the aggregate instability while differing sharply in gradient amplitude and boundary behavior. The distinction is structural.
\begin{theorem}[Unbounded Gaussian and bounded-score categorical runaway]
\label{thm:family-runaway}
Assume $p<q$ and let $r=q-p>0$. For a fixed-covariance Gaussian policy, define $\mu^\dagger=(q\mu_--p\mu_+)/r$. The aggregate mean field is
\begin{equation}
\mathbf F_\mu(\mu)=r\Sigma^{-1}(\mu-\mu^\dagger).
\label{eq:gaussian-runaway}
\end{equation}
Unless initialized exactly at $\mu^\dagger$, continuous and discrete trajectories satisfy $\|\mu\|_2\to\infty$, and for every fixed historical action $a$, $\|\nabla_\mu\log\pi_\mu(a)\|_2\to\infty$. For a categorical policy with gauge-fixed logits, every nonstationary aggregate trajectory leaves each compact subset of logit space and has a subsequence approaching the simplex boundary, while
\begin{equation}
\|\nabla_z\log\pi_z(a)\|_2^2=\|e_a-\pi_z\|_2^2\le2.
\label{eq:categorical-bound}
\end{equation}
\end{theorem}
In the Gaussian case, $\mu^\dagger$ is a repelling point and the positive-definite matrix $r\Sigma^{-1}$ expands every nonzero displacement. Increasing separation from any fixed action then converts directly into gradient-amplitude runaway. In the categorical case, fixing the additive-logit gauge reveals strictly convex ascent away from any finite stationary point. The logits become unbounded and probabilities approach the simplex boundary, but each direct-logit score remains less than or equal to $\sqrt{2}$. Categorical degeneration is therefore support suppression, not Gaussian-style gradient explosion. This bounded-score distinction also prevents us from interpreting a probability-boundary event as evidence of floating-point overflow or an unbounded Euclidean token gradient. A learned Gaussian covariance can further amplify standardized distance when support contracts, but covariance learning is not required for the fixed-covariance mean result. Neither theorem branch alone establishes task-performance collapse or NaN/Inf numerical failure; those events require separate empirical tests. Appendix~\ref{app:proof-family-runaway} provides both derivations, with additional Gaussian and categorical details in Appendices~\ref{app:gaussian} and~\ref{app:categorical}. These distinctions lead directly to the predictions below.
<!-- MANUSCRIPT:END THEORY-P04 -->

<!-- MANUSCRIPT:BEGIN THEORY-P05 -->
## [THEORY-P05] Observable Regimes and Experimental Predictions
Parent-Blueprint-SHA256: `6a74778cb71edec435982771633ba710fb416abd0e92a2d261c47ad170536781`

**Claim:** The theory predicts a sequence from a Positive-only platform through stable controlled extrapolation to boundary approach or persistent runaway, with targeted far-field attenuation restoring a finite terminal regime when that path is causal.

**Reader question:** Which observable outcomes would support or contradict the theory?

**Role:** Translate the theory into falsifiable predictions, intervention logic, and a separated failure taxonomy.

**Topic sentence:** The preceding results imply a falsifiable phase sequence rather than a generic prediction that negative updates are bad.

**Logical moves:**
- Predict the finite Positive-only platform.
- Predict finite stable extrapolation under moderate negative influence.
- Predict support or variance boundary approach as the signed target moves outward.
- Predict persistent drift or non-vanishing residual after equilibrium loss.
- Predict that far removal or capping rescues when the far path is causal, while near removal does not.
- Require separate task, support-boundary, and NaN/Inf reporting.
- Assign controlled identification to C-U1/D-U1 and external validity to Hopper/Countdown.

**Sentence plan:**
- {"anchors": ["Positive-only", "platform"], "instruction": "Predict the finite Positive-only platform.", "role": "prediction_positive_only"}
- {"anchors": ["moderate", "stable extrapolation"], "instruction": "Predict finite stable extrapolation under moderate negative influence.", "role": "prediction_controlled_negative"}
- {"anchors": ["boundary", "outward"], "instruction": "Predict support or variance boundary approach as the signed target moves outward.", "role": "prediction_boundary"}
- {"anchors": ["persistent drift", "residual"], "instruction": "Predict persistent drift or non-vanishing residual after equilibrium loss.", "role": "prediction_runaway"}
- {"anchors": ["far", "near", "rescue"], "instruction": "Predict that far removal or capping rescues when the far path is causal, while near removal does not.", "role": "prediction_intervention"}
- {"anchors": ["task-performance", "support", "NaN/Inf"], "instruction": "Require separate task, support-boundary, and NaN/Inf reporting.", "role": "failure_taxonomy"}
- {"anchors": ["C-U1", "D-U1", "Hopper", "Countdown"], "instruction": "Assign controlled identification to C-U1/D-U1 and external validity to Hopper/Countdown.", "role": "handoff"}

**Evidence use:**
- C-U1 E2--E4
- D-U1 E5--E6
- Hopper E7
- Countdown E8

**Citation refs:**

**Theorem or equation refs:**
- thm:reuse
- thm:equilibrium
- thm:family-runaway
- prop:vanishing

**Appendix bindings:**
- app:proof-reuse
- app:proof-theorem-equilibrium
- app:proof-family-runaway
- app:proof-far-field

**Word budget:** 180--330

**Body:**

The preceding results imply a falsifiable phase sequence rather than a generic prediction that negative updates are bad. Positive-only training should reach a finite platform determined by the observed positive target. Under Theorem~\ref{thm:equilibrium}, moderate negative influence should produce stable extrapolation to a second finite platform when the signed moment remains feasible. Increasing negative mass or moving its moment outward should bring the policy toward a support, variance, or probability boundary; after equilibrium loss, Theorems~\ref{thm:reuse}--\ref{thm:family-runaway} predict persistent drift or a non-vanishing terminal residual rather than a genuine stationary state. The causal prediction is selective: if the far-field component transmits the instability, removing or capping the far path should rescue a finite terminal regime, whereas removing only near-field negatives should not. Global scaling and budget-matched transfers then test whether total magnitude or learner-relative location is the operative mediator. Proposition~\ref{prop:vanishing} predicts that the DRPO envelope removes any finite-order far-field tail, but it does not pre-rank all taper shapes at finite distance. Every experiment must separately report task-performance collapse, support or variance-boundary events, and NaN/Inf numerical failure. C-U1 and D-U1 provide controlled ground-truth tests of these predictions; Hopper and Countdown assess external validity under learned critics and shared sequence parameters. The controlled environments do not substitute for the external tasks, and finite-step external pilots do not establish terminal method rankings.
<!-- MANUSCRIPT:END THEORY-P05 -->

# Distributionally Robust Policy Optimization

<!-- MANUSCRIPT:BEGIN METHOD-P01 -->
## [METHOD-P01] Distributional Reweighting of Signed Actor Updates
Parent-Blueprint-SHA256: `7ca91d925c3bf550f64a482dbdc218810fee7b39067edabf51e734bc5ac2b350`

**Claim:** DRPO controls the signed actor-update distribution by exponentially attenuating learner-relative remote negative updates while leaving positive updates unchanged.

**Reader question:** What is the current paper's self-contained DRPO update?

**Role:** Define the main method as one current formulation, without old-versus-new chronology or an unregistered combined quality-weight objective.

**Topic sentence:** DRPO controls the signed actor-update distribution by exponentially attenuating learner-relative remote negative updates while leaving positive updates unchanged.

**Logical moves:**
- start from the signed actor field defined in the setup
- define the exponential learner-relative weight on negative updates
- state the complete DRPO field used by the main theory and method experiments
- separate this remoteness control from quality-based selection

**Sentence plan:**

**Evidence use:**
- connect to Theorem 1 through the weighted aggregate negative contribution
- refer quality-selection derivation to Appendix~\ref{app:optimistic-dro} without chronology

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

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
Parent-Blueprint-SHA256: `d958b7e28ea6d66e69470fff210e464ff7fb1f00261ac40c7720fb52cb13f67e`

**Claim:** DRPO replaces the same aggregate negative moment that moves the equilibrium, allowing the method to preserve local displacement while reducing boundary-driving far-field mass.

**Reader question:** Which exact theorem object does DRPO modify?

**Role:** Create an unbroken theory--method equation chain.

**Topic sentence:** DRPO replaces the same aggregate negative moment that moves the equilibrium, allowing the method to preserve local displacement while reducing boundary-driving far-field mass.

**Logical moves:**
- uncontrolled negative moment
- weighted negative moment
- controlled signed target
- local retention and far attenuation

**Sentence plan:**

**Evidence use:**
- aggregate-moment diagnostics

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

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
Parent-Blueprint-SHA256: `edc743f9d6e653a3814a0bd7a0de9df35cd9072246bab4ad4f57c557b4f4203d`

**Claim:** An exponential envelope dominates any finite-order score growth, so the weighted negative gradient vanishes in the far field.

**Reader question:** Why use the exponential form rather than an arbitrary taper?

**Role:** Provide the method-level tail guarantee without inventing a utility-decay law.

**Topic sentence:** An exponential envelope dominates any finite-order score growth, so the weighted negative gradient vanishes in the far field.

**Logical moves:**
- finite-order score-growth condition
- exponential times polynomial tends to zero
- no assumption that sample utility decays exponentially

**Sentence plan:**

**Evidence use:**
- analytic proposition
- distance-binned gradient diagnostics

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

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
Parent-Blueprint-SHA256: `a7805f3d0cd142cc384a092c9e75d9b790e76198377cc305096df0df7eaf05d8`

**Claim:** Positive-only, global scaling, hard thresholds, linear tapers, and exponential DRPO isolate which benefits come from retaining local negatives, selecting distance, and controlling total gradient budget.

**Reader question:** How will the method be tested against simpler controls?

**Role:** Define comparisons without predeclaring a winner or conflating quality and distance thresholds.

**Topic sentence:** Positive-only, global scaling, hard thresholds, linear tapers, and exponential DRPO isolate which benefits come from retaining local negatives, selecting distance, and controlling total gradient budget.

**Logical moves:**
- uncontrolled signed baseline
- Positive-only
- global negative scaling
- linear or reciprocal distance taper
- hard distance threshold
- matched raw negative-gradient budgets

**Sentence plan:**

**Evidence use:**
- C-U1 method matrix
- D-U1 analogues
- terminal audit

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

We compare DRPO against an uncontrolled signed baseline, Positive-only learning, non-selective global scaling, registered linear or reciprocal tapers, and a hard threshold on the same remoteness coordinate. A quality-based hard filter is reported separately because it controls a different axis. Where the protocol calls for budget matching, methods receive the same pre-optimizer raw negative-gradient norm and are evaluated with paired seeds. No method ranking is assumed in advance; best and terminal checkpoints, local-negative retention, far-field retention, task performance, boundary events, and numerical failures are all reported.
<!-- MANUSCRIPT:END METHOD-P04 -->

# Experiments

<!-- MANUSCRIPT:BEGIN EXP-P01 -->
## [EXP-P01] Environments and Evidence Roles
Parent-Blueprint-SHA256: `02e0c85f7ab59c5cea314ed457827795be841b2be2a2ec09e51285f28700a0a2`

**Claim:** C-U1 and D-U1 provide controlled identification, while Hopper/D4RL and Countdown provide external validity; no environment substitutes for another's scientific responsibility.

**Reader question:** Why are these environments sufficient and what does each one prove?

**Role:** Introduce environment construction before reporting results so the claims cannot appear manufactured by an opaque simulator.

**Topic sentence:** C-U1 and D-U1 provide controlled identification, while Hopper/D4RL and Countdown provide external validity; no environment substitutes for another's scientific responsibility.

**Logical moves:**
- C-U1 6D context and 2D action
- D-U1 shared semantic categorical actor
- same-distribution held-out contexts
- Hopper learned critic
- Countdown shared parameters
- environment responsibility table

**Sentence plan:**

**Evidence use:**
- Appendix D full specifications

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

We use two primary controlled environments and two external families. C-U1 is a nonlinear state-conditioned Gaussian environment with six-dimensional numerical contexts, two-dimensional actions, a known state-dependent optimum, and independently sampled train and test contexts from the same distribution. It supports matched quality--distance probes, near/far intervention, and mean--support terminal audits. D-U1 is a shared-network categorical environment with unordered semantic actions and a known reward structure; it supports matched quality--rarity probes, common/rare intervention, and probability-boundary audits. We report C-U1 using the precise term held-out-context or unseen-state generalization. Hopper/D4RL adds a learned critic and long-horizon continuous control, while Countdown adds shared Transformer parameters and sequence-level verification. External tasks establish relevance; controlled environments establish source and causality.

\input{tables/environment_roles}
<!-- MANUSCRIPT:END EXP-P01 -->

<!-- MANUSCRIPT:BEGIN EXP-P02 -->
## [EXP-P02] RQ1: External Occurrence
Parent-Blueprint-SHA256: `57cae25e4fad9bd41650525ffb28b9ff4fc0b9278db1d6c8b25481be1bbfe31d`

**Claim:** Hopper and Countdown test whether far-field or rare negative influence becomes disproportionately large before task degradation in realistic policy learning.

**Reader question:** Does the phenomenon occur outside controlled environments?

**Role:** Use external evidence as a reality anchor without treating it as the causal isolation experiment.

**Topic sentence:** Hopper and Countdown test whether far-field or rare negative influence becomes disproportionately large before task degradation in realistic policy learning.

**Logical moves:**
- distance/surprisal bins
- positive/negative imbalance
- temporal ordering
- best and terminal checkpoints
- TBD until formal results close

**Sentence plan:**

**Evidence use:**
- EXT-H-E7-Q2 terminal-audited outputs
- Countdown formal outputs

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

\paragraph{Hopper/D4RL.} We train the registered learned critic, freeze its selected checkpoint, and audit negative actor gradients by current-policy distance. The formal report will include far/near negative-gradient ratios, positive/negative imbalance, policy drift, rollout return, and terminal classification. \textbf{TBD:} insert only terminal-audited formal results from the registered E7 package.

\paragraph{Countdown.} We bin negative tokens or completions by current-policy surprisal and track score magnitude, shared-parameter gradient influence, target-probability change, verifier success, validity, and pass@$k$. \textbf{TBD:} insert only the registered formal result; focused pilots remain provenance rather than final evidence.
<!-- MANUSCRIPT:END EXP-P02 -->

<!-- MANUSCRIPT:BEGIN EXP-P03 -->
## [EXP-P03] RQ2a: Matched Badness--Distance and Badness--Rarity Isolation
Parent-Blueprint-SHA256: `2d9fcf35fee02c4147ce1d2e10f43abd870b6b4d06899d43045624bdd7f31ee5`

**Claim:** When quality, advantage, semantics, count, coefficient, and policy stage are matched, learner-relative distance or rarity independently changes negative influence.

**Reader question:** Are far or rare gradients larger because the samples are farther, or merely because they are worse?

**Role:** Close the paper's decisive rival explanation with a transparent matched protocol.

**Topic sentence:** When quality, advantage, semantics, count, coefficient, and policy stage are matched, learner-relative distance or rarity independently changes negative influence.

**Logical moves:**
- same context and quality coordinate
- same reward and advantage magnitude
- same count and base weight
- only distance or rarity changes
- score and full-parameter gradients
- direction coherence

**Sentence plan:**

**Evidence use:**
- C-U1 E1
- D-U1 matched analogue

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

The source-isolation protocol holds sample badness fixed. In C-U1, negative actions share the same state, reward, advantage magnitude, quality coordinate, count, coefficient, and policy parameters while varying only learner-relative radius. We report output-space score norms, full-parameter per-sample gradients, aggregate gradients, and directional coherence. In D-U1, common and rare actions are matched in semantic role, reward, negative advantage, count, and base coefficient while initial probability or current surprisal changes. A far/near or rare/common gap under these controls identifies policy remoteness as an independent amplifier; it does not imply that distance is the only factor in real data.
<!-- MANUSCRIPT:END EXP-P03 -->

<!-- MANUSCRIPT:BEGIN EXP-P04 -->
## [EXP-P04] RQ2b: Targeted Causal Transmission
Parent-Blueprint-SHA256: `51694a48ee198162512d36b5c4d7eef497b383f34c45d17a55a3947aa3688779`

**Claim:** Near/far and common/rare interventions test whether the anomalous remote component is the pathway that transmits negative influence into drift, boundary events, and task collapse.

**Reader question:** Do large far-field updates actually cause the observed instability?

**Role:** Move from source identification to causal intervention with equal-budget controls.

**Topic sentence:** Near/far and common/rare interventions test whether the anomalous remote component is the pathway that transmits negative influence into drift, boundary events, and task collapse.

**Logical moves:**
- uncontrolled signed baseline
- near/common removal
- far/rare removal
- far/rare cap
- global equal-budget control
- budget transfer
- separate failure events

**Sentence plan:**

**Evidence use:**
- C-U1 E3
- D-U1 causal protocol

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

We next intervene on the location of negative influence. The continuous comparison includes the uncontrolled signed baseline, Near-zero, Far-zero, Far-cap, a non-selective global control matched to the retained raw negative-gradient budget, and a registered budget-transfer control. The categorical analogue suppresses common or rare negatives under matched counts and budgets. We record the onset order of remote influence, parameter drift, support or probability boundaries, and task performance. A selective far/rare rescue together with an ineffective near/common removal supports the remote component as a causal transmission pathway; task-performance collapse, boundary events, and NaN/Inf remain distinct labels.
<!-- MANUSCRIPT:END EXP-P04 -->

<!-- MANUSCRIPT:BEGIN EXP-P05 -->
## [EXP-P05] RQ3: Phase Transition and DRPO Control
Parent-Blueprint-SHA256: `ae0db76266a3eeb35a09a7feeeb40a2b77c7b27359e31745122818f12f01c1b1`

**Claim:** Strength sweeps map the regimes of Theorem 1, while DRPO and matched controls test whether selective far-field attenuation preserves stable extrapolation.

**Reader question:** When does negative feedback become destructive, and can DRPO control that transition?

**Role:** Unify theorem validation, aggregate-term measurement, and method comparison in one result section.

**Topic sentence:** Strength sweeps map the regimes of Theorem 1, while DRPO and matched controls test whether selective far-field attenuation preserves stable extrapolation.

**Logical moves:**
- Positive-only to stable extrapolation to boundary/drift
- continuous and categorical phase maps
- aggregate negative moment proxy
- distance bins
- paired seeds and raw-budget matching
- best and terminal reporting

**Sentence plan:**

**Evidence use:**
- C-U1 E4
- D-U1 E6
- registered taper experiments

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

We sweep the effective negative strength from zero through controlled repulsion and into the unstable regime. The primary test is not a single endpoint score but the phase sequence predicted by Theorem~\ref{thm:equilibrium}: Positive-only, stable extrapolation, boundary approach, and persistent drift or loss of finite equilibrium. We measure a family-specific empirical proxy for the aggregate negative term in~\eqref{eq:weighted-negative-moment}, together with its distance or rarity bins, direction, and relationship to equilibrium displacement. DRPO is compared with Positive-only, global scaling, and registered distance controls under paired seeds and explicit raw-gradient budgets. Results are reported at both selected and terminal checkpoints. \textbf{TBD:} populate the final table only from the current terminal-audited repository artifacts and preserve each result's formal status.

\input{tables/controlled_results}
<!-- MANUSCRIPT:END EXP-P05 -->

<!-- MANUSCRIPT:BEGIN EXP-P06 -->
## [EXP-P06] RQ4: External Task Closure
Parent-Blueprint-SHA256: `af1dc1fb678cd4fc10441c4bad129ea2397012ab837c639f39ffe5aa882aa65c`

**Claim:** External D4RL and Countdown evaluations test whether controlling remote negative influence improves actual task performance without sacrificing the useful negative signal.

**Reader question:** Does the mechanism-targeted control matter on public and sequence tasks?

**Role:** Close the external--controlled--external evidence chain.

**Topic sentence:** External D4RL and Countdown evaluations test whether controlling remote negative influence improves actual task performance without sacrificing the useful negative signal.

**Logical moves:**
- D4RL locomotion datasets
- Countdown model scales and common data bank
- same initialization and selection rules
- best and terminal metrics
- mechanism diagnostics beside performance

**Sentence plan:**

**Evidence use:**
- EXT-H-E7-BENCH-01
- EXT-C-E8-SCALE-01

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

The final evaluation returns to external tasks. On D4RL locomotion, methods share dataset, critic protocol, initialization, seeds, evaluation horizon, and selection rule; we report normalized return, uncertainty, best and terminal checkpoints, and distance-binned negative influence. On Countdown, methods share the SFT initialization, rollout or replay bank, verifier, seeds, and checkpoint-selection protocol; we report greedy success, pass@$k$, validity, terminal degradation, and rare-negative diagnostics. \textbf{TBD:} the main external tables remain unfilled until the corresponding registered formal experiments are terminal-audited and delivered.

\input{tables/external_results}
<!-- MANUSCRIPT:END EXP-P06 -->

# Implications and Conclusion

<!-- MANUSCRIPT:BEGIN DISC-P01 -->
## [DISC-P01] Negative Feedback Is a Resource with a Dynamical Failure Mode
Parent-Blueprint-SHA256: `e5f9c3c091f6ac8a6b861cad596c70ba459da64ba6c0688551df00e3b950267e`

**Claim:** The transferable lesson is to control learner-relative far-field negative influence rather than classify all negative feedback as harmful.

**Reader question:** What should readers remember beyond the specific method?

**Role:** State the positive principle rather than a list of disclaimers.

**Topic sentence:** The transferable lesson is to control learner-relative far-field negative influence rather than classify all negative feedback as harmful.

**Logical moves:**
- negative feedback supports extrapolation
- historical reuse changes its relevance and magnitude
- control object is remote negative influence

**Sentence plan:**

**Evidence use:**
- theory and controlled evidence

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

Negative feedback is a policy-improvement resource with a dynamical failure mode. It provides boundary information and can shift a policy beyond the Positive-only target. The failure arises when historical actions remain active after becoming remote from the current learner, so their optimization influence no longer tracks their local relevance. The transferable design principle is therefore to control learner-relative far-field negative influence rather than delete negative feedback as a class.
<!-- MANUSCRIPT:END DISC-P01 -->

<!-- MANUSCRIPT:BEGIN DISC-P02 -->
## [DISC-P02] Continuous and Categorical Synthesis
Parent-Blueprint-SHA256: `3a419b042a065e2dcc75e6828812198d704e2487a0bb4d7558b32c5a5df01792`

**Claim:** Gaussian and categorical policies share aggregate boundary dynamics while differing in local score growth and failure manifestation.

**Reader question:** What is genuinely unified across policy families?

**Role:** Leave one accurate cross-family synthesis.

**Topic sentence:** Gaussian and categorical policies share aggregate boundary dynamics while differing in local score growth and failure manifestation.

**Logical moves:**
- Gaussian distance-amplified score and support dynamics
- categorical bounded persistent suppression
- shared feasibility-boundary transition

**Sentence plan:**

**Evidence use:**
- family corollaries
- C-U1/D-U1 evidence

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

The continuous and categorical analyses share an aggregate structure but not an identical local law. Gaussian policies can amplify mean scores with standardized distance and couple repulsion to support contraction. Categorical direct-logit scores remain bounded, yet repeated negative updates can persistently suppress probability. In both cases, aggregate positive--negative competition determines whether the policy remains at an interior equilibrium or moves toward a feasibility boundary.
<!-- MANUSCRIPT:END DISC-P02 -->

<!-- MANUSCRIPT:BEGIN DISC-P03 -->
## [DISC-P03] Conclusion
Parent-Blueprint-SHA256: `8f5e0227457a630aef5d72652c8d946ed061f1c98078fe592ab73ebb3d09f660`

**Claim:** DRPO preserves useful negative feedback by attenuating the remote component of the aggregate term that drives equilibrium loss.

**Reader question:** What is the final three-sentence contribution?

**Role:** Close with theory, identification, and method consequence.

**Topic sentence:** DRPO preserves useful negative feedback by attenuating the remote component of the aggregate term that drives equilibrium loss.

**Logical moves:**
- stable extrapolation
- badness--distance isolation
- far-field causal pathway
- DRPO recovery

**Sentence plan:**

**Evidence use:**
- all main claims

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

Negative feedback can create stable policy improvement beyond Positive-only learning. By separating badness from policy remoteness, we identify how historical negative actions become far-field, dominate the aggregate update, and move the policy toward a feasibility boundary where a finite equilibrium can disappear. DRPO controls this same negative term, preserving useful local repulsion while suppressing the destructive far-field tail.
<!-- MANUSCRIPT:END DISC-P03 -->

# Proofs for Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN APP-PROOF-P01 -->
## [APP-PROOF-P01] Proof of Theorem 1
Parent-Blueprint-SHA256: `924605720684059efa25c92011ad5ba6d3dbf3bfb954f0893ee304b9b251400e`

**Claim:** The equilibrium theorem follows from the diffeomorphism between natural and mean parameters in a regular minimal exponential family.

**Reader question:** What are the exact existence, uniqueness, boundary, and stability arguments?

**Role:** Provide the full proof outside the main narrative.

**Topic sentence:** The equilibrium theorem follows from the diffeomorphism between natural and mean parameters in a regular minimal exponential family.

**Logical moves:**
- derive aggregate field
- interior mean-space existence
- boundary divergence
- Jacobian

**Sentence plan:**

**Evidence use:**
- standard exponential-family properties

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

For a regular minimal exponential family, $\nabla\psi$ maps the natural-parameter space diffeomorphically onto the interior of the mean-parameter space. Setting~\eqref{eq:aggregate-field} to zero with $p>q$ gives~\eqref{eq:signed-target}. If $\mathbf m^\star$ is interior, uniqueness follows from strict convexity of $\psi$. The displacement identity follows by subtracting $\mathbf m_+$ from~\eqref{eq:signed-target}. If $\mathbf m^\star$ approaches the boundary, no finite natural parameter realizes it; if it lies outside the closure, no policy in the family realizes the target. Differentiating~\eqref{eq:aggregate-field} gives $J_F=-(p-q)\nabla^2\psi$, which is negative definite at an interior point because the Hessian is the positive-definite covariance of the sufficient statistic. The discrete update is locally stable for step sizes whose transition matrix has spectral radius below one.
<!-- MANUSCRIPT:END APP-PROOF-P01 -->

# Gaussian Mean--Variance Derivations

<!-- MANUSCRIPT:BEGIN APP-GAUSS-P01 -->
## [APP-GAUSS-P01] Corrected Gaussian Mean and Variance Dynamics
Parent-Blueprint-SHA256: `af5747770e6c8fa56f460994354a441ba26fb6dbdde162a60b6c6e7224825f65`

**Claim:** Far-field negative advantages repel the mean and tend to contract variance when standardized distance exceeds one.

**Reader question:** What corrects the earlier mean-and-variance expansion account?

**Role:** Record the exact score signs and joint equilibrium condition.

**Topic sentence:** Far-field negative advantages repel the mean and tend to contract variance when standardized distance exceeds one.

**Logical moves:**
- mean score
- log-standard-deviation score
- four sign/location quadrants
- joint equilibrium

**Sentence plan:**

**Evidence use:**
- analytic Gaussian score

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

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
Parent-Blueprint-SHA256: `34f310e43b558b67b7edf550df83ca18871156f90537b98ae7db52f174dcfecd`

**Claim:** Repeated negative updates produce persistent log-odds decay even though the direct-logit score norm is bounded.

**Reader question:** Why is categorical instability not an unbounded Euclidean gradient explosion?

**Role:** Derive the probability-boundary mechanism accurately.

**Topic sentence:** Repeated negative updates produce persistent log-odds decay even though the direct-logit score norm is bounded.

**Logical moves:**
- softmax score bound
- log-odds update
- probability decay
- shared-parameter caveat

**Sentence plan:**

**Evidence use:**
- categorical algebra

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

For softmax probabilities $p_j$, the score for selected action $y$ is $e_y-p$, whose Euclidean norm is bounded. Under a negative update with fixed local coefficient, the selected logit decreases relative to competing logits, so $z_y-z_j$ falls approximately linearly while $p_y$ can decay exponentially. The failure mode is therefore persistent support suppression toward the simplex boundary. In a shared network, the parameter-space gradient additionally contains the network Jacobian and cross-action interference, which are measured rather than inferred from the direct-logit bound alone.
<!-- MANUSCRIPT:END APP-CAT-P01 -->

# Controlled Environments

<!-- MANUSCRIPT:BEGIN APP-ENV-P01 -->
## [APP-ENV-P01] C-U1 and D-U1 Construction
Parent-Blueprint-SHA256: `51800edeed7e56cc2e89c822d29c28ba6825b2d11e809a230770a5a92d9a66e5`

**Claim:** The controlled environments expose ground truth and permit matched isolation without claiming that construction alone establishes external validity.

**Reader question:** How exactly are the controlled environments generated?

**Role:** Provide enough detail for reproducibility and reviewer inspection.

**Topic sentence:** The controlled environments expose ground truth and permit matched isolation without claiming that construction alone establishes external validity.

**Logical moves:**
- C-U1 states/actions/reward/hidden optimum
- train/test same distribution
- D-U1 semantic actions
- matched probes
- dynamic near/far

**Sentence plan:**

**Evidence use:**
- environment code and manifests

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

C-U1 samples numerical contexts $s\sim\mathcal N(0,I_6)$ independently for training and testing. A fixed generator maps each state to a positive action, hidden optimal action, negative quality coordinate, and reward surface in $\mathbb R^2$. Source-isolation probes replicate reward and advantage across radii while retaining the same state and quality coordinate. Near/far membership for causal interventions is computed relative to the current policy according to the registered metric. D-U1 uses a shared state-conditioned categorical network over unordered semantic actions; quality/semantics and rarity are factorized so that common/rare probes can be matched in reward, advantage, count, and coefficient. Complete constants, seeds, and invariants are taken from the registered experiment configurations.
<!-- MANUSCRIPT:END APP-ENV-P01 -->

# Experimental Protocols and Terminal Audits

<!-- MANUSCRIPT:BEGIN APP-PROT-P01 -->
## [APP-PROT-P01] Stopping, Budget Matching, and Terminal Classification
Parent-Blueprint-SHA256: `43f6ead559cf8a832cca87cdc5a612c2681ff41d571443c7d33819ccbf9e6144`

**Claim:** All dynamic and ranking claims require fixed protocol ownership, paired seeds, explicit budgets, and terminal audits.

**Reader question:** How are false convergence and unfair comparisons prevented?

**Role:** Centralize experiment governance details.

**Topic sentence:** All dynamic and ranking claims require fixed protocol ownership, paired seeds, explicit budgets, and terminal audits.

**Logical moves:**
- maximum horizon
- terminal slopes/residuals
- 2x continuation where registered
- raw pre-optimizer negative-gradient budget
- best and terminal

**Sentence plan:**

**Evidence use:**
- handoff and registry

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

Formal protocols specify maximum horizons, evaluation cadence, paired development and held-out seeds, and a terminal audit before execution. A fixed horizon is not itself called convergence. Depending on the registered experiment, terminal classification uses state slopes, update residuals, boundary checks, and a continuation horizon. Budget-matched comparisons use the pre-optimizer raw negative-gradient $\ell_2$ norm unless another coordinate is explicitly frozen. Adam parameter-update norms are recorded separately. Best-validation and terminal checkpoints are both reported, and task-performance collapse, support or variance boundaries, and NaN/Inf failures remain distinct fields.
<!-- MANUSCRIPT:END APP-PROT-P01 -->

# Additional Results and Failure Taxonomy

<!-- MANUSCRIPT:BEGIN APP-RES-P01 -->
## [APP-RES-P01] Additional Tables, Curves, and Negative Results
Parent-Blueprint-SHA256: `c3260547225981737aa8fd0921485dc30da6bcde2427787e7113d17da2aaaa32`

**Claim:** Appendix results preserve uncertainty, failed runs, and non-ranking outcomes instead of presenting only favorable endpoints.

**Reader question:** What evidence is needed beyond the main figures?

**Role:** Provide placeholders for complete result deposition.

**Topic sentence:** Appendix results preserve uncertainty, failed runs, and non-ranking outcomes instead of presenting only favorable endpoints.

**Logical moves:**
- per-seed tables
- full trajectories
- sensitivity
- negative results
- failure inventory

**Sentence plan:**

**Evidence use:**
- formal artifact packages

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

This appendix will contain per-seed results, full training trajectories, confidence intervals, sensitivity analyses, and terminal classifications for every main comparison. Failed or inconclusive formal runs are indexed rather than silently removed. \textbf{TBD:} populate these materials from the durable formal packages after repository closure; smoke tests, static checks, and focused pilots are not promoted to formal evidence.
<!-- MANUSCRIPT:END APP-RES-P01 -->

# Implementation and Reproducibility

<!-- MANUSCRIPT:BEGIN APP-REPRO-P01 -->
## [APP-REPRO-P01] Code, Data, and Artifact Provenance
Parent-Blueprint-SHA256: `647ca94c4a54d578ba95dbf8b0327c46ec063024a4079f9fe3b4632bc3bca00e`

**Claim:** Every formal claim is bound to code, configuration, seeds, raw outputs, terminal audit, and a source commit.

**Reader question:** How can the study be reproduced and audited?

**Role:** Connect the paper to repository provenance without embedding live status in the outline.

**Topic sentence:** Every formal claim is bound to code, configuration, seeds, raw outputs, terminal audit, and a source commit.

**Logical moves:**
- repository paths
- run manifests
- checksums
- formal package lifecycle
- Overleaf build

**Sentence plan:**

**Evidence use:**
- repository scripts and artifacts

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

The repository records experiment IDs, configurations, seeds, code entry points, result manifests, raw curves, logs, terminal audits, checksums, and source commits. Formal runs progress through registered, running, raw-complete, terminal-audited, packaged, delivered, and applied states. The paper source is generated from the stable-ID manuscript graph, compiled in the imported ICML/Overleaf project, and packaged with a reproducible build command. Exact repository paths and commit hashes will be inserted at submission freeze.
<!-- MANUSCRIPT:END APP-REPRO-P01 -->

# Optimistic Distributional Selection

<!-- MANUSCRIPT:BEGIN APP-DRO-P01 -->
## [APP-DRO-P01] Optimistic-DRO Quality Selection and Its Distinct Role
Parent-Blueprint-SHA256: `3dc2ee394622537a840fb4f2acfafd836231a829504b800b9700f553c8c7aa85`

**Claim:** The Optimistic-DRO subdistribution problem yields a quality-based hard-selection solution; this result is part of the current manuscript but is mathematically distinct from learner-relative remoteness tapering.

**Reader question:** What does the Optimistic-DRO result establish in the current paper, and what does it not establish?

**Role:** Integrate the valid distributional-selection derivation into the current manuscript while keeping quality selection mathematically distinct from learner-relative remoteness tapering.

**Topic sentence:** Optimistic distributional selection and learner-relative remoteness control answer different questions within the same DRPO manuscript.

**Logical moves:**
- state the density-ratio-constrained quality-selection problem
- state its hard-selection solution
- explain the exact scope of that result
- separate it from the remoteness envelope used in the main actor update

**Sentence plan:**

**Evidence use:**
- provide the self-contained optimization and solution statement
- cross-reference the main matched badness--distance experiments

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

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
Parent-Blueprint-SHA256: `8d57969f08bd495d3302e0478e922e7e09e22e43824cdad639ff12419f77ab5b`

**Claim:** The manuscript consistently uses corrected Gaussian scale dynamics, the Jacobian of the actual signed field, held-out-context terminology for C-U1, and separate reporting of task, boundary, and numerical failures.

**Reader question:** Which technical conventions must remain consistent across the main text and appendix?

**Role:** Collect the technical conventions that prevent contradictory statements across theory, experiments, and reporting.

**Topic sentence:** The manuscript consistently uses corrected Gaussian scale dynamics, the Jacobian of the actual signed field, held-out-context terminology for C-U1, and separate reporting of task, boundary, and numerical failures.

**Logical moves:**
- Gaussian mean repulsion with location-dependent scale dynamics
- actual signed-field Jacobian rather than expected Fisher as the stability object
- held-out-context or unseen-state generalization for C-U1
- separate task collapse, policy-family boundary, and NaN/Inf fields
- terminal audit before convergence or long-run language

**Sentence plan:**

**Evidence use:**
- corrected Gaussian derivation
- registered reporting protocol
- terminal-audit governance

**Citation refs:**

**Theorem or equation refs:**

**Appendix bindings:**

**Word budget:**

**Body:**

The manuscript uses the following technical conventions throughout. Far-field negative Gaussian updates combine mean repulsion with location-dependent scale dynamics and commonly support contraction; they do not universally expand both mean and variance. The expected Fisher matrix describes policy geometry but does not by itself determine stability of a fixed off-policy signed field; local stability is evaluated with the Jacobian of the actual aggregate field. C-U1 test states are independently sampled from the same state distribution and are described as held-out-context or unseen-state generalization, not OOD generalization. Task-performance collapse, support or variance boundaries, and NaN/Inf numerical failure are reported separately. Finally, finite-horizon evidence is not relabeled as convergence or long-run validation without the registered terminal audit.
<!-- MANUSCRIPT:END APP-CORR-P01 -->
