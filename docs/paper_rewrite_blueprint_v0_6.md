# DRPO full-paper paragraph blueprint v0.7

Parent: `docs/paper_rewrite_outline_v0_9_2.md`. The first four chapters now carry sentence-unit plans, citation and theorem bindings, appendix links, reviewer objections, and publication word budgets.

This is one DRPO manuscript being rewritten, not an old paper followed by a sequel.

# Abstract

<!-- MANUSCRIPT:BEGIN ABSTRACT-P01 -->
## [ABSTRACT-P01] Paper Summary
Parent-Outline-SHA256: `68037707db36afda0db4ceae7014c3f3241873d74cb7105c5ca1664ff6984739`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END ABSTRACT-P01 -->

# Introduction

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource
Parent-Outline-SHA256: `a7cd4a6d085c2e65bb81e73f740ab9e3e89a96fa2e23d9fab42f9817c761b1dd`

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

**Allowed conclusions:**
- negative feedback can be informative
- Positive-only is a stable reference with a possible imitation ceiling

**Forbidden conclusions:**
- all negative feedback is harmful
- Positive-only is universally inferior

**Reviewer objection:** The opening could sound as though failures are always useful or that Positive-only is an inadequate baseline by definition.

**Objection response:** State complementarity and a conditional limitation, not a universal ranking.

**Word budget:** 125--220

**Transition:** The next paragraph explains why historical reuse changes the character of the same negative signal.
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion
Parent-Outline-SHA256: `c3a88d84e1dab0f4f340a71a79fa4e2d7950d69fe6e3bffb870758ffad9654a2`

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

**Allowed conclusions:**
- historical reuse can change a sample's influence
- Gaussian and categorical manifestations differ

**Forbidden conclusions:**
- categorical per-sample gradients diverge without bound
- every off-policy update necessarily collapses

**Reviewer objection:** The mechanism might be mistaken for ordinary distribution shift or a one-time large update.

**Objection response:** Emphasize the repeated feedback loop and policy-family-specific response laws.

**Word budget:** 125--220

**Transition:** Existing methods control several symptoms of this mismatch, but do not isolate the repeated remoteness feedback loop.
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Existing Controls and the Missing Identification Link
Parent-Outline-SHA256: `4044ce3da95303db92fa096a44487f42b5d5cc0d00963351d613a71723317698`

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

**Allowed conclusions:**
- prior controls have genuine stabilizing value
- remoteness requires separate causal identification

**Forbidden conclusions:**
- prior methods ignore off-policy mismatch
- distance is the only source of instability

**Reviewer objection:** The claimed gap could be a relabeling of clipping, rarity weighting, or conservative offline RL.

**Objection response:** Define the gap as repeated learner-relative dynamics under matched sample quality and then test source and transmission separately.

**Word budget:** 145--250

**Transition:** A theory of repulsive dynamics is needed to connect single-sample remoteness to aggregate equilibria.
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss
Parent-Outline-SHA256: `78fa788bf8b81d2d985c5d619b384f91a0c7c15a8991a81311e1227a7e51f546`

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

**Allowed conclusions:**
- moderate aggregate repulsion can produce a finite extrapolated equilibrium
- policy-family manifestations differ

**Forbidden conclusions:**
- advantage sign alone determines all joint-parameter directions
- all negative-dominant trajectories produce NaN/Inf

**Reviewer objection:** A single equilibrium theorem could hide the distinction between per-sample amplification and aggregate stability.

**Objection response:** Preview the static, repeated-reuse, and aggregate results as separate logical layers.

**Word budget:** 135--235

**Transition:** DRPO is then defined as a control of the far-field contribution in the aggregate negative term.
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term
Parent-Outline-SHA256: `4299d56ae0f3d5e0a0e6412c747d0be8435bb499f81c984abda90b35c179429f`

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

**Allowed conclusions:**
- the exponential envelope has a vanishing-tail guarantee under finite-order score growth
- local negative feedback is retained

**Forbidden conclusions:**
- the exponential taper is universally optimal
- distance control must outperform global scaling

**Reviewer objection:** The exponential envelope could appear to assume that negative-sample utility decays exponentially.

**Objection response:** Ground the choice in domination of finite-order score growth, not in an assumed utility curve.

**Word budget:** 120--215

**Transition:** The experiments separately test occurrence, source, causal transmission, phase behavior, and external task value.
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions
Parent-Outline-SHA256: `bc285bcac56e25abb6e6bebaaf6f0f9c2818c4d4ffcc2af2650bdacb41c13ef7`

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

**Allowed conclusions:**
- controlled experiments identify mechanisms
- external experiments test external validity

**Forbidden conclusions:**
- controlled environments alone establish universal task superiority
- pilot external results are formal closure

**Reviewer objection:** A long experiment list could blur which environment supports which claim.

**Objection response:** Frame the experiments as four questions and explicitly separate controlled identification from external validation.

**Word budget:** 150--270

**Transition:** Related work next locates each component against prior approaches.
<!-- MANUSCRIPT:END INTRO-P06 -->

# Related Work

<!-- MANUSCRIPT:BEGIN RELATED-P01 -->
## [RELATED-P01] Learning from Negative or Suboptimal Behavior
Parent-Outline-SHA256: `9f2c7cfd6a0c7204efb92ad40fb7fca8d456d08d1dafc4e327e1f3924d4c3b03`

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

**Allowed conclusions:**
- prior work establishes both stabilization and information value
- this paper studies the transition across remoteness

**Forbidden conclusions:**
- prior work treats all failures as noise
- this is the first use of negative feedback

**Reviewer objection:** The novelty claim could ignore extensive work on positive-only filtering and negative examples.

**Objection response:** Credit both lines and define the contribution as repeated learner-relative dynamics and matched causal isolation.

**Word budget:** 145--250

**Transition:** The next family controls behavior--learner mismatch and stale or low-probability updates.
<!-- MANUSCRIPT:END RELATED-P01 -->

<!-- MANUSCRIPT:BEGIN RELATED-P02 -->
## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates
Parent-Outline-SHA256: `62bd34389b421554abebf28ed56c2e69e36d2a7466d8fe557c3fe747ad25d004`

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

**Allowed conclusions:**
- mismatch and rarity are established control signals
- repeated negative reuse adds an endogenous feedback loop

**Forbidden conclusions:**
- clipping and importance weighting are ineffective
- staleness is identical to far-field distance

**Reviewer objection:** Far-field control may look like a renamed probability or importance-ratio heuristic.

**Objection response:** Separate the measured coordinate from the dynamic claim that negative reuse increases that coordinate and alters the aggregate equilibrium.

**Word budget:** 145--250

**Transition:** Conservative offline RL addresses a complementary value and support problem.
<!-- MANUSCRIPT:END RELATED-P02 -->

<!-- MANUSCRIPT:BEGIN RELATED-P03 -->
## [RELATED-P03] Robust Offline Policy Learning
Parent-Outline-SHA256: `79d67e60df76ef147429f3d6e7b729c23c9d9bdfd17351ab778fc13a34272066`

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

**Allowed conclusions:**
- DRPO studies a distinct actor-side mechanism
- value-side and actor-side controls can be combined

**Forbidden conclusions:**
- far-field repulsion explains every offline-RL failure
- DRPO replaces conservative value learning

**Reviewer objection:** The paper could overclaim a complete explanation of offline-RL instability.

**Objection response:** Explicitly bound the contribution to signed actor dynamics and describe compatibility with value-side methods.

**Word budget:** 145--255

**Transition:** The setup now formalizes the historical signed actor field and learner-relative remoteness.
<!-- MANUSCRIPT:END RELATED-P03 -->

# Problem Setup

<!-- MANUSCRIPT:BEGIN SETUP-P01 -->
## [SETUP-P01] Signed Actor Update and Influence Factorization
Parent-Outline-SHA256: `0762e3dae95390cb5d771737eddf7edd7ddf23610dcda84d1c432502bec32049`

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

**Allowed conclusions:**
- fixed labels isolate actor-score dynamics
- negative influence has separable severity and geometry factors

**Forbidden conclusions:**
- the setup represents every actor--critic algorithm globally
- critic error is necessary for the mechanism

**Reviewer objection:** Freezing the update distribution and advantage could be mistaken for a claim about all of RL.

**Objection response:** State it as an actor-step mechanism analysis and separate extensions with evolving critics or ratios.

**Word budget:** 190--340

**Transition:** The next setup paragraph defines the common learner-relative remoteness coordinate without equating policy families.
<!-- MANUSCRIPT:END SETUP-P01 -->

<!-- MANUSCRIPT:BEGIN SETUP-P02 -->
## [SETUP-P02] Policy-Relative Far Field
Parent-Outline-SHA256: `91382f2829b087edf495fc99fb07b8a0c5d4aa7efc249590a446005744f1f203`

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

**Allowed conclusions:**
- negative log probability provides a shared learner-relative coordinate
- policy families have different score-response laws

**Forbidden conclusions:**
- Euclidean action distance is universal
- categorical and Gaussian gradients share the same unbounded law

**Reviewer objection:** A common remoteness notation could conceal incompatible continuous and discrete geometries.

**Objection response:** Define the common probabilistic coordinate while deriving each family's response separately.

**Word budget:** 170--300

**Transition:** We first derive the static remoteness--score relation before analyzing repeated reuse.
<!-- MANUSCRIPT:END SETUP-P02 -->

# Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN THEORY-P01 -->
## [THEORY-P01] Learner-Relative Remoteness and Static Score Response
Parent-Outline-SHA256: `a2a151d54e1715c3e8d11b3d280b4b9a20abce3b34a5897233f309a34b9a82d6`

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

**Allowed conclusions:**
- Gaussian mean scores are unbounded in far field
- categorical direct-logit scores remain bounded

**Forbidden conclusions:**
- static remoteness alone proves task collapse
- categorical direct-logit gradients explode without bound

**Reviewer objection:** A cross-family proposition might falsely imply identical growth laws.

**Objection response:** State separate formulas and use only learner-relative ordering as the common abstraction.

**Word budget:** 250--450

**Transition:** The next theorem turns the static relation into a temporal feedback loop under repeated reuse.
<!-- MANUSCRIPT:END THEORY-P01 -->

<!-- MANUSCRIPT:BEGIN THEORY-P02 -->
## [THEORY-P02] Self-Attenuation and Self-Amplification under Reuse
Parent-Outline-SHA256: `b268086022f29b885ce7f7407cbfc838c652af857fcbc7fb6df8036f54999106`

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

**Allowed conclusions:**
- negative reuse monotonically increases remoteness under the assumptions
- positive reuse self-attenuates at a valid step size

**Forbidden conclusions:**
- one negative sample guarantees global divergence
- the theorem covers arbitrary jointly evolving critics without qualification

**Reviewer objection:** The theorem may be read as a global claim about every neural policy parameterization.

**Objection response:** State the convex path and coordinate assumptions, then reserve shared-network effects for empirical measurement.

**Word budget:** 260--470

**Transition:** Aggregate positive attraction can counter this per-sample repulsion, so the next result solves their balance.
<!-- MANUSCRIPT:END THEORY-P02 -->

<!-- MANUSCRIPT:BEGIN THEORY-P03 -->
## [THEORY-P03] Aggregate Attraction--Repulsion Equilibria
Parent-Outline-SHA256: `4091895993a5507e70963a0a549e1f29c53b2261f153e16239a3dcad62bfb56d`

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

**Allowed conclusions:**
- moderate negative feedback can shift a stable equilibrium beyond the positive target
- negative dominance has no stable finite equilibrium

**Forbidden conclusions:**
- extrapolation automatically improves task utility
- p<q necessarily causes NaN/Inf

**Reviewer objection:** The extrapolation identity could be mistaken for a performance theorem.

**Objection response:** Separate geometric equilibrium displacement from task utility and require experiments for the latter.

**Word budget:** 330--620

**Transition:** The same aggregate instability has different terminal signatures in Gaussian and categorical policies.
<!-- MANUSCRIPT:END THEORY-P03 -->

<!-- MANUSCRIPT:BEGIN THEORY-P04 -->
## [THEORY-P04] Policy-Family Manifestations of Negative Dominance
Parent-Outline-SHA256: `340496d576d3f6ac7fbe5746725e49c8450e23a42f51894577cae648143a8bba`

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

**Allowed conclusions:**
- Gaussian fixed-covariance mean runaway is sufficient for unbounded scores
- categorical degeneration can occur with bounded direct-logit scores

**Forbidden conclusions:**
- both Gaussian mean and variance always expand
- categorical boundary degeneration is numerical collapse

**Reviewer objection:** Calling both cases collapse could erase the distinction between gradient amplitude, support, task, and numerical events.

**Objection response:** Use separate theorem statements and preserve the project failure taxonomy.

**Word budget:** 300--540

**Transition:** These family-specific signatures yield a concrete set of experimental predictions.
<!-- MANUSCRIPT:END THEORY-P04 -->

<!-- MANUSCRIPT:BEGIN THEORY-P05 -->
## [THEORY-P05] Observable Regimes and Experimental Predictions
Parent-Outline-SHA256: `34f56aefe69cd8681ffb8204d8d70afaf0463381f1dd36e5dd7922e0b89c6705`

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

**Allowed conclusions:**
- the theory yields falsifiable regime and intervention predictions
- separate environments answer separate claims

**Forbidden conclusions:**
- a finite fixed training horizon proves equilibrium
- controlled evidence substitutes for external validity

**Reviewer objection:** A broad theoretical narrative could become unfalsifiable if every poor outcome is called repulsion.

**Objection response:** Pre-register terminal regimes, targeted interventions, and distinct event classes that can contradict the theory.

**Word budget:** 180--330

**Transition:** The method section now operationalizes selective far-field control and the experiments test each prediction.
<!-- MANUSCRIPT:END THEORY-P05 -->

# Distributionally Robust Policy Optimization

<!-- MANUSCRIPT:BEGIN METHOD-P01 -->
## [METHOD-P01] Distributional Reweighting of Signed Actor Updates
Parent-Outline-SHA256: `b7dce19f0f9bc2a2f03f6cb47662fd1944ae8e895b7559de4471411aa6a391ff`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:** The next block shows that this update modifies the exact aggregate term in Theorem 1.
<!-- MANUSCRIPT:END METHOD-P01 -->

<!-- MANUSCRIPT:BEGIN METHOD-P02 -->
## [METHOD-P02] Direct Bridge to Theorem 1
Parent-Outline-SHA256: `f022cd340d47a60f1308d82b300cc5cbc730c3557d884304853b3754afe6a65b`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END METHOD-P02 -->

<!-- MANUSCRIPT:BEGIN METHOD-P03 -->
## [METHOD-P03] Proposition 2: Vanishing Weighted Far-Field Gradient
Parent-Outline-SHA256: `a28d858a65e1ca1c2d492ae622405d4816d9f956abe0b886a75298c957ba2654`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END METHOD-P03 -->

<!-- MANUSCRIPT:BEGIN METHOD-P04 -->
## [METHOD-P04] Controls and Ablations
Parent-Outline-SHA256: `f972d558397bce6f948c07dd4cf0792b275f3b163d4616ba6477da9a552d3bd5`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END METHOD-P04 -->

# Experiments

<!-- MANUSCRIPT:BEGIN EXP-P01 -->
## [EXP-P01] Environments and Evidence Roles
Parent-Outline-SHA256: `911d8672bb4d9ca9e8238ee481f0f2bd8df8f169327d0a4a635f52803d168fe4`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END EXP-P01 -->

<!-- MANUSCRIPT:BEGIN EXP-P02 -->
## [EXP-P02] RQ1: External Occurrence
Parent-Outline-SHA256: `dbca74317f5d767afe909bab1b74d760984c251c8c882483e25d5c3ae35862d2`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END EXP-P02 -->

<!-- MANUSCRIPT:BEGIN EXP-P03 -->
## [EXP-P03] RQ2a: Matched Badness--Distance and Badness--Rarity Isolation
Parent-Outline-SHA256: `84a1fa0cd982d904b7b61dcc6c2a162e5872c3cda995278294e7f930597791f6`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END EXP-P03 -->

<!-- MANUSCRIPT:BEGIN EXP-P04 -->
## [EXP-P04] RQ2b: Targeted Causal Transmission
Parent-Outline-SHA256: `16bb56f504d2c47f264351b3f0d160ce052d0d75ad2d26ab00c985df098be976`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END EXP-P04 -->

<!-- MANUSCRIPT:BEGIN EXP-P05 -->
## [EXP-P05] RQ3: Phase Transition and DRPO Control
Parent-Outline-SHA256: `61ee39a278ff214d04152a5645d874d6433c0aa9696a3de8dbb0598715df739f`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END EXP-P05 -->

<!-- MANUSCRIPT:BEGIN EXP-P06 -->
## [EXP-P06] RQ4: External Task Closure
Parent-Outline-SHA256: `88580e45d2e808503b1b8cf55ee7cc5fd42e925a28e66e535e9b251d70bab3a5`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END EXP-P06 -->

# Implications and Conclusion

<!-- MANUSCRIPT:BEGIN DISC-P01 -->
## [DISC-P01] Negative Feedback Is a Resource with a Dynamical Failure Mode
Parent-Outline-SHA256: `b58c2837a52ab71c4a814c45e94c781c1c197dacb5dd010cca0e167e726e076d`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END DISC-P01 -->

<!-- MANUSCRIPT:BEGIN DISC-P02 -->
## [DISC-P02] Continuous and Categorical Synthesis
Parent-Outline-SHA256: `bb9bd1e0d1f36fd67d6e878ce88b9f030bc1d02f7e2e7d9293a699b4e8338bb3`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END DISC-P02 -->

<!-- MANUSCRIPT:BEGIN DISC-P03 -->
## [DISC-P03] Conclusion
Parent-Outline-SHA256: `cb639ec0985fdaf4c05845bac64b08449e7761deeaa695381b0bf9c0a83451f1`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END DISC-P03 -->

# Proofs for Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN APP-PROOF-P01 -->
## [APP-PROOF-P01] Proof of Theorem 1
Parent-Outline-SHA256: `6417ceca714290637770be31be209b7131e524b5be2dd163f03b02416415b7ae`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-PROOF-P01 -->

# Gaussian Mean--Variance Derivations

<!-- MANUSCRIPT:BEGIN APP-GAUSS-P01 -->
## [APP-GAUSS-P01] Corrected Gaussian Mean and Variance Dynamics
Parent-Outline-SHA256: `71b0fb4ffc940458ba512b7e03c14a7975b4c8a03f4c1988a9e69eb7bf459ca5`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-GAUSS-P01 -->

# Categorical Boundary Dynamics

<!-- MANUSCRIPT:BEGIN APP-CAT-P01 -->
## [APP-CAT-P01] Categorical Log-Odds and Boundary Behavior
Parent-Outline-SHA256: `58bbfe3cc8dbd055898f1627dbb7163af9c12d135ebdfadbc4916b772f62f94f`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-CAT-P01 -->

# Controlled Environments

<!-- MANUSCRIPT:BEGIN APP-ENV-P01 -->
## [APP-ENV-P01] C-U1 and D-U1 Construction
Parent-Outline-SHA256: `3cc8b9c2f0ea10fc347300a099887aee7ec8da0b7aff0641a1827012f1d3c1a1`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-ENV-P01 -->

# Experimental Protocols and Terminal Audits

<!-- MANUSCRIPT:BEGIN APP-PROT-P01 -->
## [APP-PROT-P01] Stopping, Budget Matching, and Terminal Classification
Parent-Outline-SHA256: `68c8ae0ba9d2362e14ccec56b1e53627629452cdb4c348cb8ad7002c86628a8c`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-PROT-P01 -->

# Additional Results and Failure Taxonomy

<!-- MANUSCRIPT:BEGIN APP-RES-P01 -->
## [APP-RES-P01] Additional Tables, Curves, and Negative Results
Parent-Outline-SHA256: `a017c291cea9bf19231acdbdd619a85cf5eea962cf22ed488e16ef1aa47e3c4f`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-RES-P01 -->

# Implementation and Reproducibility

<!-- MANUSCRIPT:BEGIN APP-REPRO-P01 -->
## [APP-REPRO-P01] Code, Data, and Artifact Provenance
Parent-Outline-SHA256: `42164710012fe7cf262a96a46cd4e45960ab35a5d2727b06dfa51d5806fb7d00`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-REPRO-P01 -->

# Optimistic Distributional Selection

<!-- MANUSCRIPT:BEGIN APP-DRO-P01 -->
## [APP-DRO-P01] Optimistic-DRO Quality Selection and Its Distinct Role
Parent-Outline-SHA256: `fd46430511579c9d4be1f30948b8bc8bb4ea665b8dd20d754fba3c4bb74c73aa`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:** The correction ledger records how this and other retained results differ from superseded claims.
<!-- MANUSCRIPT:END APP-DRO-P01 -->

# Additional Theoretical and Reporting Clarifications

<!-- MANUSCRIPT:BEGIN APP-CORR-P01 -->
## [APP-CORR-P01] Additional Theoretical and Reporting Clarifications
Parent-Outline-SHA256: `d28ea4945802e9a7a538832293499bd862a3e8c4c9b006af182b6ad562f80730`

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

**Allowed conclusions:**

**Forbidden conclusions:**

**Reviewer objection:**

**Objection response:**

**Word budget:**

**Transition:**
<!-- MANUSCRIPT:END APP-CORR-P01 -->
