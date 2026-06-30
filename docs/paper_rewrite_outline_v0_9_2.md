# DRPO paper rewrite v0.9.2: merged review candidate

Status: generated after the writing system was established. This version mechanically merges the user-approved v0.9-review, the useful corrections in repository v0.9, and the confirmed non-buggy refinements in v0.9.1. It is one DRPO manuscript being rewritten. The merge ledger records every accepted and rejected change.

# Abstract

<!-- MANUSCRIPT:BEGIN ABSTRACT-P01 -->
## [ABSTRACT-P01] Paper Summary

**Claim:** Negative feedback is useful for policy improvement, but historical negative actions can become far-field and exert excessive influence; DRPO controls that transition.

**Reader question:** What is the problem, missing link, theory, method, evidence, and implication in one compact sequence?

**Role:** Summarize the full paper without fixed-advantage scope language, split-manuscript framing, or unverified numerical claims.

**Required evidence:**
- Theorem 1
- source-isolation protocols
- targeted interventions
- external tasks marked TBD until formal closure

**Must include:**
- Negative feedback as a resource
- badness--distance isolation
- stable extrapolation to equilibrium loss
- DRPO attenuation of the far-field component
- controlled and external evidence

**Must avoid:**
- split-manuscript framing
- finished external numbers before formal closure
<!-- MANUSCRIPT:END ABSTRACT-P01 -->

# Introduction

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource

**Claim:** Negative feedback is not merely noise: balanced against positive attraction, it can suppress bad modes and shift a policy beyond the Positive-only target.

**Reader question:** Why should policy optimization retain negative feedback at all?

**Role:** Open with the constructive role of failures and establish Positive-only as a stable but limited reference.

**Required evidence:**
- Theorem 1 stable-extrapolation regime
- controlled Positive-only comparison

**Must include:**
- positive attraction reinforces observed successes
- negative feedback suppresses known bad modes
- balanced repulsion can shift the equilibrium beyond observed positive behavior
- central question: preserve benefit without excessive repulsion

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion

**Claim:** Offline logs, replay, stale actors, and asynchronous trajectories continue to reuse negative actions after the current learner has moved away, turning local feedback into persistent far-field repulsion.

**Reader question:** How does useful negative feedback become dangerous in off-policy learning?

**Role:** Introduce the temporal mechanism shared by offline, replay-based, stale-policy, and asynchronous training.

**Required evidence:**
- Gaussian score identity
- categorical log-odds dynamics
- external occurrence diagnostics

**Must include:**
- negative actions can initially be locally informative
- historical reuse persists after policy movement
- Gaussian scores can grow with standardized distance
- categorical scores are bounded but suppression persists
- useful-local to destructive-far-field transition

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Existing Controls and the Missing Identification Link

**Claim:** Existing controls regulate sign, scale, ratio, support, staleness, or data quality, but they neither explain the useful-to-destructive transition nor isolate policy remoteness from sample badness.

**Reader question:** What do existing methods address, what remains unresolved, and why is strict badness--distance isolation necessary?

**Role:** Combine prior-method positioning with the decisive identification control instead of replacing one with the other.

**Required evidence:**
- Related-work synthesis
- C-U1 E1 quality--distance factorization
- D-U1 common/rare matched analogue

**Must include:**
- positive-only, global scaling, clipping, support constraints, low-probability controls, and quality filtering
- their stabilizing value
- unresolved transition from useful local repulsion to destructive far-field influence
- realistic correlation among reward, negative advantage, rarity, and distance
- matched control holding context, semantics, reward, advantage, count, coefficient, and policy stage fixed

**Must avoid:**
- dropping the existing-method gap
- claiming distance is the only factor
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss

**Claim:** Positive attraction and negative repulsion define a phase sequence from the Positive-only target to stable extrapolation, boundary approach, and loss of finite equilibrium.

**Reader question:** What theoretical structure unifies beneficial and harmful negative updates?

**Role:** Preview Theorem 1 and the continuous--categorical distinction without proof details.

**Required evidence:**
- Theorem 1
- Gaussian and categorical corollaries
- E4/E6 phase tests

**Must include:**
- Positive-only target
- moderate negative contribution produces stable extrapolation
- stronger or more outward contribution moves the signed target toward a feasible boundary
- finite equilibrium can disappear
- Gaussian and categorical manifestations differ

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term

**Claim:** DRPO reweights the aggregate negative contribution with a policy-relative exponential envelope, retaining local negative feedback while making the weighted far-field gradient vanish under finite-order score growth.

**Reader question:** How does the method act on the theoretical failure mechanism?

**Role:** Present DRPO as the current paper's unified method, not as a sequel or a revision appended to an older paper.

**Required evidence:**
- method equations
- Proposition 2
- controlled method comparisons

**Must include:**
- distributional reweighting of the empirical actor update
- exponential distance/surprisal weight
- direct modification of the Theorem 1 negative term
- far-field vanishing proposition
- quality selection and remoteness control are distinct axes

**Must avoid:**
- original formulation versus revised formulation
- old paper followed by a new algorithm
- claim that quality hard filtering is mathematically identical to distance tapering
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions

**Claim:** The paper uses an external--controlled--external evidence chain to establish occurrence, identify source and causality, test the phase transition and DRPO control, and close on external task performance.

**Reader question:** How are the claims divided across environments and experiments?

**Role:** End the Introduction with four research questions and four concise contributions.

**Required evidence:**
- environment responsibility table
- registered experiments
- best and terminal metrics

**Must include:**
- RQ1 external occurrence in Hopper and Countdown
- RQ2 matched source isolation plus targeted causal intervention
- RQ3 phase transition, aggregate-term measurement, and controlled method comparison
- RQ4 external task closure
- C-U1/D-U1 controlled roles and Hopper/Countdown external roles
- terminal audit and separate failure events

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P06 -->

# Related Work

<!-- MANUSCRIPT:BEGIN RELATED-P01 -->
## [RELATED-P01] Learning from Negative or Suboptimal Behavior

**Claim:** Prior work establishes that negative or suboptimal data can be useful, but does not characterize the learner-relative transition from useful local repulsion to destructive far-field influence.

**Reader question:** How does the paper differ from positive-only, advantage-weighted, and failure-learning work?

**Role:** Credit prior evidence for negative-feedback value and position the new transition.

**Required evidence:**
- AWR and related actor fitting
- negative-feedback literature

**Must include:**
- positive-only filtering
- advantage-weighted regression
- failure learning and negative reinforcement
- useful information in negative data
- missing dynamics across learner-relative distance

**Must avoid:**
<!-- MANUSCRIPT:END RELATED-P01 -->

<!-- MANUSCRIPT:BEGIN RELATED-P02 -->
## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates

**Claim:** Clipping, importance correction, stale-policy control, and rare-event regulation address mismatch or scale, while this paper connects repeated learner-relative movement to aggregate repulsion and equilibrium loss.

**Reader question:** How does the paper relate to standard off-policy stabilization?

**Role:** Position the far-field mechanism as complementary to ratio, staleness, and probability controls.

**Required evidence:**
- PPO and off-policy references
- Countdown diagnostics

**Must include:**
- PPO-style clipping
- off-policy correction
- stale-policy and asynchronous updates
- low-probability actions or tokens
- aggregate equilibrium consequence

**Must avoid:**
<!-- MANUSCRIPT:END RELATED-P02 -->

<!-- MANUSCRIPT:BEGIN RELATED-P03 -->
## [RELATED-P03] Robust Offline Policy Learning

**Claim:** Offline RL controls value extrapolation, policy support, or data quality; DRPO instead targets the far-field component of signed actor updates while retaining useful local negatives.

**Reader question:** How is DRPO positioned against conservative and behavior-regularized offline RL?

**Role:** Distinguish the actor-dynamics object without dismissing established offline-RL solutions.

**Required evidence:**
- CQL/IQL citations
- external D4RL evaluation

**Must include:**
- CQL and pessimism
- IQL and in-support value learning
- behavior regularization and TD3+BC-style simplicity
- data filtering
- signed actor update as the distinct object

**Must avoid:**
<!-- MANUSCRIPT:END RELATED-P03 -->

# Problem Setup

<!-- MANUSCRIPT:BEGIN SETUP-P01 -->
## [SETUP-P01] Signed Actor Update and Influence Factorization

**Claim:** A negative sample's influence factors into advantage severity and policy-score geometry, while count and directional coherence determine aggregation.

**Reader question:** What is the mathematical object shared by theory, method, and experiments?

**Role:** Define the signed update and isolate badness from geometry.

**Required evidence:**
- policy-gradient identity
- per-sample and aggregate diagnostics

**Must include:**
- signed empirical actor field
- positive and negative advantage parts
- per-sample norm factorization
- aggregation factors

**Must avoid:**
<!-- MANUSCRIPT:END SETUP-P01 -->

<!-- MANUSCRIPT:BEGIN SETUP-P02 -->
## [SETUP-P02] Policy-Relative Far Field

**Claim:** Far field is a dynamic relation between a historical action and the current learner, realized by standardized distance for Gaussian policies and surprisal for categorical policies.

**Reader question:** How is remoteness defined across continuous and categorical policies?

**Role:** Define family-specific remoteness without claiming identical amplification laws.

**Required evidence:**
- C-U1 distance coordinate
- D-U1 rarity coordinate
- Countdown NLL coordinate

**Must include:**
- Gaussian standardized or Mahalanobis distance
- categorical surprisal
- sequence normalized token/completion NLL
- dynamic rather than permanent label

**Must avoid:**
<!-- MANUSCRIPT:END SETUP-P02 -->

# Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN THEORY-P01 -->
## [THEORY-P01] Per-Sample Far-Field Dynamics

**Claim:** Gaussian negative updates can amplify with standardized distance, while categorical negative updates remain bounded in direct-logit norm but persistently suppress probability.

**Reader question:** What happens to one negative action under repeated updates?

**Role:** Establish the family-specific local mechanisms before the aggregate theorem.

**Required evidence:**
- analytic derivations in appendices B and C

**Must include:**
- Gaussian mean score identity
- corrected variance sign depends on standardized location
- categorical direct-logit score bound
- log-odds suppression

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P01 -->

<!-- MANUSCRIPT:BEGIN THEORY-P02 -->
## [THEORY-P02] Aggregate Positive--Negative Competition

**Claim:** For an exponential-family policy, the signed actor field is governed by positive and negative update masses and their sufficient-statistic moments.

**Reader question:** How do per-sample effects combine into a tractable equilibrium model?

**Role:** Define the exact signed objective used by Theorem 1 without paper-wide fixed-advantage disclaimers.

**Required evidence:**
- exponential-family algebra

**Must include:**
- regular minimal exponential family
- positive and negative masses p and q
- moments m+ and m-
- signed field formula

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P02 -->

<!-- MANUSCRIPT:BEGIN THEORY-P03 -->
## [THEORY-P03] Theorem 1: Stable Extrapolation and Loss of Finite Equilibrium

**Claim:** The aggregate field admits a stable finite equilibrium beyond the Positive-only target when the signed moment remains interior, and loses that equilibrium at or beyond the feasible boundary.

**Reader question:** When does negative feedback help, and when does the same force remove a finite equilibrium?

**Role:** Serve as the theoretical hinge connecting useful negative feedback, phase transition, and the method target.

**Required evidence:**
- proof in Appendix A
- phase tests in E4/E6

**Must include:**
- Positive-only limit
- stable extrapolation formula
- boundary approach
- no finite equilibrium
- local stability statement

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P03 -->

<!-- MANUSCRIPT:BEGIN THEORY-P04 -->
## [THEORY-P04] Testable Predictions

**Claim:** The theorem predicts distinct observable regimes and a targeted recovery when the far-field component of the aggregate negative term is attenuated.

**Reader question:** How is Theorem 1 connected to actual experiments?

**Role:** Map each mathematical regime to a pre-specified intervention and terminal observable.

**Required evidence:**
- E4 continuous phase sweep
- E6 categorical phase sweep
- terminal slope/residual
- support and probability boundaries

**Must include:**
- Positive-only platform
- controlled negative stable platform
- boundary event
- persistent drift or non-vanishing residual
- DRPO recovery

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P04 -->

<!-- MANUSCRIPT:BEGIN THEORY-P05 -->
## [THEORY-P05] Gaussian and Categorical Corollaries

**Claim:** Gaussian and categorical policies instantiate the same aggregate boundary principle through different score and feasibility geometries.

**Reader question:** What is shared and what remains family-specific?

**Role:** Prevent an invalid claim that categorical policies exhibit the same unbounded Euclidean gradient explosion as Gaussian policies.

**Required evidence:**
- Appendices B and C

**Must include:**
- Gaussian finite mean/covariance and boundary behavior
- categorical finite logits and simplex boundary
- shared aggregate competition
- different amplification laws

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P05 -->

# Distributionally Robust Policy Optimization

<!-- MANUSCRIPT:BEGIN METHOD-P01 -->
## [METHOD-P01] Distributional Reweighting of Signed Actor Updates

**Claim:** DRPO controls the signed actor-update distribution by exponentially attenuating learner-relative remote negative updates while leaving positive updates unchanged.

**Reader question:** What is the current paper's self-contained DRPO update?

**Role:** Define the main method as one current formulation, without old-versus-new chronology or an unregistered combined quality-weight objective.

**Required evidence:**
- method equation
- ablation family

**Must include:**
- raw signed actor update
- negative exponential distance/surprisal envelope
- positive updates unchanged
- quality and remoteness are distinct analysis axes

**Must avoid:**
- optional quality weights in the main update unless separately registered
- claiming quality filtering and remoteness tapering are identical
<!-- MANUSCRIPT:END METHOD-P01 -->

<!-- MANUSCRIPT:BEGIN METHOD-P02 -->
## [METHOD-P02] Direct Bridge to Theorem 1

**Claim:** DRPO replaces the same aggregate negative moment that moves the equilibrium, allowing the method to preserve local displacement while reducing boundary-driving far-field mass.

**Reader question:** Which exact theorem object does DRPO modify?

**Role:** Create an unbroken theory--method equation chain.

**Required evidence:**
- aggregate-moment diagnostics

**Must include:**
- uncontrolled negative moment
- weighted negative moment
- controlled signed target
- local retention and far attenuation

**Must avoid:**
<!-- MANUSCRIPT:END METHOD-P02 -->

<!-- MANUSCRIPT:BEGIN METHOD-P03 -->
## [METHOD-P03] Proposition 2: Vanishing Weighted Far-Field Gradient

**Claim:** An exponential envelope dominates any finite-order score growth, so the weighted negative gradient vanishes in the far field.

**Reader question:** Why use the exponential form rather than an arbitrary taper?

**Role:** Provide the method-level tail guarantee without inventing a utility-decay law.

**Required evidence:**
- analytic proposition
- distance-binned gradient diagnostics

**Must include:**
- finite-order score-growth condition
- exponential times polynomial tends to zero
- no assumption that sample utility decays exponentially

**Must avoid:**
<!-- MANUSCRIPT:END METHOD-P03 -->

<!-- MANUSCRIPT:BEGIN METHOD-P04 -->
## [METHOD-P04] Controls and Ablations

**Claim:** Positive-only, global scaling, hard thresholds, linear tapers, and exponential DRPO isolate which benefits come from retaining local negatives, selecting distance, and controlling total gradient budget.

**Reader question:** How will the method be tested against simpler controls?

**Role:** Define comparisons without predeclaring a winner or conflating quality and distance thresholds.

**Required evidence:**
- C-U1 method matrix
- D-U1 analogues
- terminal audit

**Must include:**
- uncontrolled signed baseline
- Positive-only
- global negative scaling
- linear or reciprocal distance taper
- hard distance threshold
- matched raw negative-gradient budgets

**Must avoid:**
<!-- MANUSCRIPT:END METHOD-P04 -->

# Experiments

<!-- MANUSCRIPT:BEGIN EXP-P01 -->
## [EXP-P01] Environments and Evidence Roles

**Claim:** C-U1 and D-U1 provide controlled identification, while Hopper/D4RL and Countdown provide external validity; no environment substitutes for another's scientific responsibility.

**Reader question:** Why are these environments sufficient and what does each one prove?

**Role:** Introduce environment construction before reporting results so the claims cannot appear manufactured by an opaque simulator.

**Required evidence:**
- Appendix D full specifications

**Must include:**
- C-U1 6D context and 2D action
- D-U1 shared semantic categorical actor
- same-distribution held-out contexts
- Hopper learned critic
- Countdown shared parameters
- environment responsibility table

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P01 -->

<!-- MANUSCRIPT:BEGIN EXP-P02 -->
## [EXP-P02] RQ1: External Occurrence

**Claim:** Hopper and Countdown test whether far-field or rare negative influence becomes disproportionately large before task degradation in realistic policy learning.

**Reader question:** Does the phenomenon occur outside controlled environments?

**Role:** Use external evidence as a reality anchor without treating it as the causal isolation experiment.

**Required evidence:**
- EXT-H-E7-Q2 terminal-audited outputs
- Countdown formal outputs

**Must include:**
- distance/surprisal bins
- positive/negative imbalance
- temporal ordering
- best and terminal checkpoints
- TBD until formal results close

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P02 -->

<!-- MANUSCRIPT:BEGIN EXP-P03 -->
## [EXP-P03] RQ2a: Matched Badness--Distance and Badness--Rarity Isolation

**Claim:** When quality, advantage, semantics, count, coefficient, and policy stage are matched, learner-relative distance or rarity independently changes negative influence.

**Reader question:** Are far or rare gradients larger because the samples are farther, or merely because they are worse?

**Role:** Close the paper's decisive rival explanation with a transparent matched protocol.

**Required evidence:**
- C-U1 E1
- D-U1 matched analogue

**Must include:**
- same context and quality coordinate
- same reward and advantage magnitude
- same count and base weight
- only distance or rarity changes
- score and full-parameter gradients
- direction coherence

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P03 -->

<!-- MANUSCRIPT:BEGIN EXP-P04 -->
## [EXP-P04] RQ2b: Targeted Causal Transmission

**Claim:** Near/far and common/rare interventions test whether the anomalous remote component is the pathway that transmits negative influence into drift, boundary events, and task collapse.

**Reader question:** Do large far-field updates actually cause the observed instability?

**Role:** Move from source identification to causal intervention with equal-budget controls.

**Required evidence:**
- C-U1 E3
- D-U1 causal protocol

**Must include:**
- uncontrolled signed baseline
- near/common removal
- far/rare removal
- far/rare cap
- global equal-budget control
- budget transfer
- separate failure events

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P04 -->

<!-- MANUSCRIPT:BEGIN EXP-P05 -->
## [EXP-P05] RQ3: Phase Transition and DRPO Control

**Claim:** Strength sweeps map the regimes of Theorem 1, while DRPO and matched controls test whether selective far-field attenuation preserves stable extrapolation.

**Reader question:** When does negative feedback become destructive, and can DRPO control that transition?

**Role:** Unify theorem validation, aggregate-term measurement, and method comparison in one result section.

**Required evidence:**
- C-U1 E4
- D-U1 E6
- registered taper experiments

**Must include:**
- Positive-only to stable extrapolation to boundary/drift
- continuous and categorical phase maps
- aggregate negative moment proxy
- distance bins
- paired seeds and raw-budget matching
- best and terminal reporting

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P05 -->

<!-- MANUSCRIPT:BEGIN EXP-P06 -->
## [EXP-P06] RQ4: External Task Closure

**Claim:** External D4RL and Countdown evaluations test whether controlling remote negative influence improves actual task performance without sacrificing the useful negative signal.

**Reader question:** Does the mechanism-targeted control matter on public and sequence tasks?

**Role:** Close the external--controlled--external evidence chain.

**Required evidence:**
- EXT-H-E7-BENCH-01
- EXT-C-E8-SCALE-01

**Must include:**
- D4RL locomotion datasets
- Countdown model scales and common data bank
- same initialization and selection rules
- best and terminal metrics
- mechanism diagnostics beside performance

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P06 -->

# Implications and Conclusion

<!-- MANUSCRIPT:BEGIN DISC-P01 -->
## [DISC-P01] Negative Feedback Is a Resource with a Dynamical Failure Mode

**Claim:** The transferable lesson is to control learner-relative far-field negative influence rather than classify all negative feedback as harmful.

**Reader question:** What should readers remember beyond the specific method?

**Role:** State the positive principle rather than a list of disclaimers.

**Required evidence:**
- theory and controlled evidence

**Must include:**
- negative feedback supports extrapolation
- historical reuse changes its relevance and magnitude
- control object is remote negative influence

**Must avoid:**
<!-- MANUSCRIPT:END DISC-P01 -->

<!-- MANUSCRIPT:BEGIN DISC-P02 -->
## [DISC-P02] Continuous and Categorical Synthesis

**Claim:** Gaussian and categorical policies share aggregate boundary dynamics while differing in local score growth and failure manifestation.

**Reader question:** What is genuinely unified across policy families?

**Role:** Leave one accurate cross-family synthesis.

**Required evidence:**
- family corollaries
- C-U1/D-U1 evidence

**Must include:**
- Gaussian distance-amplified score and support dynamics
- categorical bounded persistent suppression
- shared feasibility-boundary transition

**Must avoid:**
<!-- MANUSCRIPT:END DISC-P02 -->

<!-- MANUSCRIPT:BEGIN DISC-P03 -->
## [DISC-P03] Conclusion

**Claim:** DRPO preserves useful negative feedback by attenuating the remote component of the aggregate term that drives equilibrium loss.

**Reader question:** What is the final three-sentence contribution?

**Role:** Close with theory, identification, and method consequence.

**Required evidence:**
- all main claims

**Must include:**
- stable extrapolation
- badness--distance isolation
- far-field causal pathway
- DRPO recovery

**Must avoid:**
<!-- MANUSCRIPT:END DISC-P03 -->

# Proofs for Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN APP-PROOF-P01 -->
## [APP-PROOF-P01] Proof of Theorem 1

**Claim:** The equilibrium theorem follows from the diffeomorphism between natural and mean parameters in a regular minimal exponential family.

**Reader question:** What are the exact existence, uniqueness, boundary, and stability arguments?

**Role:** Provide the full proof outside the main narrative.

**Required evidence:**
- standard exponential-family properties

**Must include:**
- derive aggregate field
- interior mean-space existence
- boundary divergence
- Jacobian

**Must avoid:**
<!-- MANUSCRIPT:END APP-PROOF-P01 -->

# Gaussian Mean--Variance Derivations

<!-- MANUSCRIPT:BEGIN APP-GAUSS-P01 -->
## [APP-GAUSS-P01] Corrected Gaussian Mean and Variance Dynamics

**Claim:** Far-field negative advantages repel the mean and tend to contract variance when standardized distance exceeds one.

**Reader question:** What corrects the earlier mean-and-variance expansion account?

**Role:** Record the exact score signs and joint equilibrium condition.

**Required evidence:**
- analytic Gaussian score

**Must include:**
- mean score
- log-standard-deviation score
- four sign/location quadrants
- joint equilibrium

**Must avoid:**
<!-- MANUSCRIPT:END APP-GAUSS-P01 -->

# Categorical Boundary Dynamics

<!-- MANUSCRIPT:BEGIN APP-CAT-P01 -->
## [APP-CAT-P01] Categorical Log-Odds and Boundary Behavior

**Claim:** Repeated negative updates produce persistent log-odds decay even though the direct-logit score norm is bounded.

**Reader question:** Why is categorical instability not an unbounded Euclidean gradient explosion?

**Role:** Derive the probability-boundary mechanism accurately.

**Required evidence:**
- categorical algebra

**Must include:**
- softmax score bound
- log-odds update
- probability decay
- shared-parameter caveat

**Must avoid:**
<!-- MANUSCRIPT:END APP-CAT-P01 -->

# Controlled Environments

<!-- MANUSCRIPT:BEGIN APP-ENV-P01 -->
## [APP-ENV-P01] C-U1 and D-U1 Construction

**Claim:** The controlled environments expose ground truth and permit matched isolation without claiming that construction alone establishes external validity.

**Reader question:** How exactly are the controlled environments generated?

**Role:** Provide enough detail for reproducibility and reviewer inspection.

**Required evidence:**
- environment code and manifests

**Must include:**
- C-U1 states/actions/reward/hidden optimum
- train/test same distribution
- D-U1 semantic actions
- matched probes
- dynamic near/far

**Must avoid:**
<!-- MANUSCRIPT:END APP-ENV-P01 -->

# Experimental Protocols and Terminal Audits

<!-- MANUSCRIPT:BEGIN APP-PROT-P01 -->
## [APP-PROT-P01] Stopping, Budget Matching, and Terminal Classification

**Claim:** All dynamic and ranking claims require fixed protocol ownership, paired seeds, explicit budgets, and terminal audits.

**Reader question:** How are false convergence and unfair comparisons prevented?

**Role:** Centralize experiment governance details.

**Required evidence:**
- handoff and registry

**Must include:**
- maximum horizon
- terminal slopes/residuals
- 2x continuation where registered
- raw pre-optimizer negative-gradient budget
- best and terminal

**Must avoid:**
<!-- MANUSCRIPT:END APP-PROT-P01 -->

# Additional Results and Failure Taxonomy

<!-- MANUSCRIPT:BEGIN APP-RES-P01 -->
## [APP-RES-P01] Additional Tables, Curves, and Negative Results

**Claim:** Appendix results preserve uncertainty, failed runs, and non-ranking outcomes instead of presenting only favorable endpoints.

**Reader question:** What evidence is needed beyond the main figures?

**Role:** Provide placeholders for complete result deposition.

**Required evidence:**
- formal artifact packages

**Must include:**
- per-seed tables
- full trajectories
- sensitivity
- negative results
- failure inventory

**Must avoid:**
<!-- MANUSCRIPT:END APP-RES-P01 -->

# Implementation and Reproducibility

<!-- MANUSCRIPT:BEGIN APP-REPRO-P01 -->
## [APP-REPRO-P01] Code, Data, and Artifact Provenance

**Claim:** Every formal claim is bound to code, configuration, seeds, raw outputs, terminal audit, and a source commit.

**Reader question:** How can the study be reproduced and audited?

**Role:** Connect the paper to repository provenance without embedding live status in the outline.

**Required evidence:**
- repository scripts and artifacts

**Must include:**
- repository paths
- run manifests
- checksums
- formal package lifecycle
- Overleaf build

**Must avoid:**
<!-- MANUSCRIPT:END APP-REPRO-P01 -->

# Optimistic Distributional Selection

<!-- MANUSCRIPT:BEGIN APP-DRO-P01 -->
## [APP-DRO-P01] Optimistic-DRO Quality Selection and Its Distinct Role

**Claim:** The Optimistic-DRO subdistribution problem yields a quality-based hard-selection solution; this result is part of the current manuscript but is mathematically distinct from learner-relative remoteness tapering.

**Reader question:** What does the Optimistic-DRO result establish in the current paper, and what does it not establish?

**Role:** Integrate the valid distributional-selection derivation into the current manuscript while keeping quality selection mathematically distinct from learner-relative remoteness tapering.

**Required evidence:**
- closed-form density-ratio-constrained subdistribution solution
- explicit separation of quality and remoteness axes

**Must include:**
- quality-based uncertainty set
- top-quality hard-selection solution
- current-paper integration
- no implication that hard quality filtering equals exponential distance tapering

**Must avoid:**
- split-manuscript chronology
- claim that quality hard filtering derives exponential remoteness tapering
<!-- MANUSCRIPT:END APP-DRO-P01 -->

# Additional Theoretical and Reporting Clarifications

<!-- MANUSCRIPT:BEGIN APP-CORR-P01 -->
## [APP-CORR-P01] Additional Theoretical and Reporting Clarifications

**Claim:** The manuscript consistently uses corrected Gaussian scale dynamics, the Jacobian of the actual signed field, held-out-context terminology for C-U1, and separate reporting of task, boundary, and numerical failures.

**Reader question:** Which technical conventions must remain consistent across the main text and appendix?

**Role:** Collect the technical conventions that prevent contradictory statements across theory, experiments, and reporting.

**Required evidence:**
- corrected Gaussian derivation
- registered reporting protocol
- terminal-audit governance

**Must include:**
- Gaussian mean repulsion with location-dependent scale dynamics
- actual signed-field Jacobian rather than expected Fisher as the stability object
- held-out-context or unseen-state generalization for C-U1
- separate task collapse, policy-family boundary, and NaN/Inf fields
- terminal audit before convergence or long-run language

**Must avoid:**
- historical chronology
- defensive discussion of manuscript versions
<!-- MANUSCRIPT:END APP-CORR-P01 -->
