# DRPO full-paper paragraph blueprint v0.6

Parent: `docs/paper_rewrite_outline_v0_9_2.md`. Every block is generated from the stable-ID manuscript graph and retains its parent hash.

# Abstract

<!-- MANUSCRIPT:BEGIN ABSTRACT-P01 -->
## [ABSTRACT-P01] Paper Summary
Parent-Outline-SHA256: `68037707db36afda0db4ceae7014c3f3241873d74cb7105c5ca1664ff6984739`

**Claim:** Negative feedback is useful for policy improvement, but historical negative actions can become far-field and exert excessive influence; DRPO controls that transition.

**Topic sentence:** Negative feedback is useful for policy improvement, but historical negative actions can become far-field and exert excessive influence; DRPO controls that transition.

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

**Transition:**
<!-- MANUSCRIPT:END ABSTRACT-P01 -->

# Introduction

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource
Parent-Outline-SHA256: `53d8e25bbff12ba009dea935c2f20e0189e51cf34eab25ca4173d5f9a9d25a06`

**Claim:** Negative feedback is not merely noise: balanced against positive attraction, it can suppress bad modes and shift a policy beyond the Positive-only target.

**Topic sentence:** Negative feedback is not merely noise: balanced against positive attraction, it can suppress bad modes and shift a policy beyond the Positive-only target.

**Logical moves:**
- positive attraction reinforces observed successes
- negative feedback suppresses known bad modes
- balanced repulsion can shift the equilibrium beyond observed positive behavior
- central question: preserve benefit without excessive repulsion

**Evidence use:**
- Theorem 1 stable-extrapolation regime
- controlled Positive-only comparison

**Transition:** Historical reuse explains how initially relevant repulsion becomes excessive.
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion
Parent-Outline-SHA256: `94fd8b0ab24900f356ef12447dbe8002cb138a24a0df1921c8edb0b9c194e14a`

**Claim:** Offline logs, replay, stale actors, and asynchronous trajectories continue to reuse negative actions after the current learner has moved away, turning local feedback into persistent far-field repulsion.

**Topic sentence:** Offline logs, replay, stale actors, and asynchronous trajectories continue to reuse negative actions after the current learner has moved away, turning local feedback into persistent far-field repulsion.

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

**Transition:** The next paragraph identifies the confounding factor that prevents this mechanism from being inferred directly in realistic data.
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Existing Controls and the Missing Identification Link
Parent-Outline-SHA256: `1e6c963b2e682ebec0525cc6e7a3d62eb195a8829baaf93db81159e2f413110f`

**Claim:** Existing controls regulate sign, scale, ratio, support, staleness, or data quality, but they neither explain the useful-to-destructive transition nor isolate policy remoteness from sample badness.

**Topic sentence:** Existing controls regulate sign, scale, ratio, support, staleness, or data quality, but they neither explain the useful-to-destructive transition nor isolate policy remoteness from sample badness.

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

**Transition:** With the confound removed, Repulsive Dynamics can characterize the aggregate transition.
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss
Parent-Outline-SHA256: `454d156a23d8fcedcd51a3bc24dc3b3dc6d37c7bf41b782b29df1a2af01ec6d5`

**Claim:** Positive attraction and negative repulsion define a phase sequence from the Positive-only target to stable extrapolation, boundary approach, and loss of finite equilibrium.

**Topic sentence:** Positive attraction and negative repulsion define a phase sequence from the Positive-only target to stable extrapolation, boundary approach, and loss of finite equilibrium.

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

**Transition:** The method follows by controlling the same aggregate negative term that drives this phase transition.
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term
Parent-Outline-SHA256: `c5a68760f9839fe76381bed515feb1f93cf2e9c3ef3a5cd89a321a27235c1d3e`

**Claim:** DRPO reweights the aggregate negative contribution with a policy-relative exponential envelope, retaining local negative feedback while making the weighted far-field gradient vanish under finite-order score growth.

**Topic sentence:** DRPO reweights the aggregate negative contribution with a policy-relative exponential envelope, retaining local negative feedback while making the weighted far-field gradient vanish under finite-order score growth.

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

**Transition:** The evidence chain tests occurrence, mechanism, transition, control, and external task effect.
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions
Parent-Outline-SHA256: `1878626d7c80e095952b80cb3f2c79421b703f216984b411dfd069fc9a353128`

**Claim:** The paper uses an external--controlled--external evidence chain to establish occurrence, identify source and causality, test the phase transition and DRPO control, and close on external task performance.

**Topic sentence:** The paper uses an external--controlled--external evidence chain to establish occurrence, identify source and causality, test the phase transition and DRPO control, and close on external task performance.

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

**Transition:**
<!-- MANUSCRIPT:END INTRO-P06 -->

# Related Work

<!-- MANUSCRIPT:BEGIN RELATED-P01 -->
## [RELATED-P01] Learning from Negative or Suboptimal Behavior
Parent-Outline-SHA256: `c6d5b54f6152b63685ad6c3b8617a410bad4e890e9562041edbe38bf4d6758aa`

**Claim:** Prior work establishes that negative or suboptimal data can be useful, but does not characterize the learner-relative transition from useful local repulsion to destructive far-field influence.

**Topic sentence:** Prior work establishes that negative or suboptimal data can be useful, but does not characterize the learner-relative transition from useful local repulsion to destructive far-field influence.

**Logical moves:**
- positive-only filtering
- advantage-weighted regression
- failure learning and negative reinforcement
- useful information in negative data
- missing dynamics across learner-relative distance

**Evidence use:**
- AWR and related actor fitting
- negative-feedback literature

**Transition:**
<!-- MANUSCRIPT:END RELATED-P01 -->

<!-- MANUSCRIPT:BEGIN RELATED-P02 -->
## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates
Parent-Outline-SHA256: `4d8204e4b16cfa815a426757d16c3747f945a743abb844d2aebc3e2e8b32ef3b`

**Claim:** Clipping, importance correction, stale-policy control, and rare-event regulation address mismatch or scale, while this paper connects repeated learner-relative movement to aggregate repulsion and equilibrium loss.

**Topic sentence:** Clipping, importance correction, stale-policy control, and rare-event regulation address mismatch or scale, while this paper connects repeated learner-relative movement to aggregate repulsion and equilibrium loss.

**Logical moves:**
- PPO-style clipping
- off-policy correction
- stale-policy and asynchronous updates
- low-probability actions or tokens
- aggregate equilibrium consequence

**Evidence use:**
- PPO and off-policy references
- Countdown diagnostics

**Transition:**
<!-- MANUSCRIPT:END RELATED-P02 -->

<!-- MANUSCRIPT:BEGIN RELATED-P03 -->
## [RELATED-P03] Robust Offline Policy Learning
Parent-Outline-SHA256: `c39ab2c3b3a6d49bfaa5a53a75420c17a7e796e5964ac4c47aaabdc1b425deda`

**Claim:** Offline RL controls value extrapolation, policy support, or data quality; DRPO instead targets the far-field component of signed actor updates while retaining useful local negatives.

**Topic sentence:** Offline RL controls value extrapolation, policy support, or data quality; DRPO instead targets the far-field component of signed actor updates while retaining useful local negatives.

**Logical moves:**
- CQL and pessimism
- IQL and in-support value learning
- behavior regularization and TD3+BC-style simplicity
- data filtering
- signed actor update as the distinct object

**Evidence use:**
- CQL/IQL citations
- external D4RL evaluation

**Transition:**
<!-- MANUSCRIPT:END RELATED-P03 -->

# Problem Setup

<!-- MANUSCRIPT:BEGIN SETUP-P01 -->
## [SETUP-P01] Signed Actor Update and Influence Factorization
Parent-Outline-SHA256: `874cb1551b5858175d1beaf9369e0931a2bc47d63a61c575ba8033b5fede8df4`

**Claim:** A negative sample's influence factors into advantage severity and policy-score geometry, while count and directional coherence determine aggregation.

**Topic sentence:** A negative sample's influence factors into advantage severity and policy-score geometry, while count and directional coherence determine aggregation.

**Logical moves:**
- signed empirical actor field
- positive and negative advantage parts
- per-sample norm factorization
- aggregation factors

**Evidence use:**
- policy-gradient identity
- per-sample and aggregate diagnostics

**Transition:**
<!-- MANUSCRIPT:END SETUP-P01 -->

<!-- MANUSCRIPT:BEGIN SETUP-P02 -->
## [SETUP-P02] Policy-Relative Far Field
Parent-Outline-SHA256: `a8db2d78ce3634e9b55378df0a81bac4956bf2cce49186a2e9e1341eb9d3ce0d`

**Claim:** Far field is a dynamic relation between a historical action and the current learner, realized by standardized distance for Gaussian policies and surprisal for categorical policies.

**Topic sentence:** Far field is a dynamic relation between a historical action and the current learner, realized by standardized distance for Gaussian policies and surprisal for categorical policies.

**Logical moves:**
- Gaussian standardized or Mahalanobis distance
- categorical surprisal
- sequence normalized token/completion NLL
- dynamic rather than permanent label

**Evidence use:**
- C-U1 distance coordinate
- D-U1 rarity coordinate
- Countdown NLL coordinate

**Transition:**
<!-- MANUSCRIPT:END SETUP-P02 -->

# Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN THEORY-P01 -->
## [THEORY-P01] Per-Sample Far-Field Dynamics
Parent-Outline-SHA256: `b4172e57afadce0d4317f875d6c1f5431910f3788d7802d5891890bd5bf7f4b2`

**Claim:** Gaussian negative updates can amplify with standardized distance, while categorical negative updates remain bounded in direct-logit norm but persistently suppress probability.

**Topic sentence:** Gaussian negative updates can amplify with standardized distance, while categorical negative updates remain bounded in direct-logit norm but persistently suppress probability.

**Logical moves:**
- Gaussian mean score identity
- corrected variance sign depends on standardized location
- categorical direct-logit score bound
- log-odds suppression

**Evidence use:**
- analytic derivations in appendices B and C

**Transition:**
<!-- MANUSCRIPT:END THEORY-P01 -->

<!-- MANUSCRIPT:BEGIN THEORY-P02 -->
## [THEORY-P02] Aggregate Positive--Negative Competition
Parent-Outline-SHA256: `ea03a8af375b638d3d94778c21ac98a5626c937af3b1dd11cd6fca6dd72525d5`

**Claim:** For an exponential-family policy, the signed actor field is governed by positive and negative update masses and their sufficient-statistic moments.

**Topic sentence:** For an exponential-family policy, the signed actor field is governed by positive and negative update masses and their sufficient-statistic moments.

**Logical moves:**
- regular minimal exponential family
- positive and negative masses p and q
- moments m+ and m-
- signed field formula

**Evidence use:**
- exponential-family algebra

**Transition:**
<!-- MANUSCRIPT:END THEORY-P02 -->

<!-- MANUSCRIPT:BEGIN THEORY-P03 -->
## [THEORY-P03] Theorem 1: Stable Extrapolation and Loss of Finite Equilibrium
Parent-Outline-SHA256: `dc2e5e14fe2dc9f3057195ad4f559317ee8fab05834f6959a8130468dc2e45ea`

**Claim:** The aggregate field admits a stable finite equilibrium beyond the Positive-only target when the signed moment remains interior, and loses that equilibrium at or beyond the feasible boundary.

**Topic sentence:** The aggregate field admits a stable finite equilibrium beyond the Positive-only target when the signed moment remains interior, and loses that equilibrium at or beyond the feasible boundary.

**Logical moves:**
- Positive-only limit
- stable extrapolation formula
- boundary approach
- no finite equilibrium
- local stability statement

**Evidence use:**
- proof in Appendix A
- phase tests in E4/E6

**Transition:**
<!-- MANUSCRIPT:END THEORY-P03 -->

<!-- MANUSCRIPT:BEGIN THEORY-P04 -->
## [THEORY-P04] Testable Predictions
Parent-Outline-SHA256: `99299d60ece285c70432c38942e701ac94c3d9d646e78f47d0180e237a17574b`

**Claim:** The theorem predicts distinct observable regimes and a targeted recovery when the far-field component of the aggregate negative term is attenuated.

**Topic sentence:** The theorem predicts distinct observable regimes and a targeted recovery when the far-field component of the aggregate negative term is attenuated.

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

**Transition:**
<!-- MANUSCRIPT:END THEORY-P04 -->

<!-- MANUSCRIPT:BEGIN THEORY-P05 -->
## [THEORY-P05] Gaussian and Categorical Corollaries
Parent-Outline-SHA256: `106a108acbf8f270fc068e757842adaab966fed83e41583621e516d511daefe7`

**Claim:** Gaussian and categorical policies instantiate the same aggregate boundary principle through different score and feasibility geometries.

**Topic sentence:** Gaussian and categorical policies instantiate the same aggregate boundary principle through different score and feasibility geometries.

**Logical moves:**
- Gaussian finite mean/covariance and boundary behavior
- categorical finite logits and simplex boundary
- shared aggregate competition
- different amplification laws

**Evidence use:**
- Appendices B and C

**Transition:**
<!-- MANUSCRIPT:END THEORY-P05 -->

# Distributionally Robust Policy Optimization

<!-- MANUSCRIPT:BEGIN METHOD-P01 -->
## [METHOD-P01] Distributional Reweighting of Signed Actor Updates
Parent-Outline-SHA256: `b7dce19f0f9bc2a2f03f6cb47662fd1944ae8e895b7559de4471411aa6a391ff`

**Claim:** DRPO controls the signed actor-update distribution by exponentially attenuating learner-relative remote negative updates while leaving positive updates unchanged.

**Topic sentence:** DRPO controls the signed actor-update distribution by exponentially attenuating learner-relative remote negative updates while leaving positive updates unchanged.

**Logical moves:**
- start from the signed actor field defined in the setup
- define the exponential learner-relative weight on negative updates
- state the complete DRPO field used by the main theory and method experiments
- separate this remoteness control from quality-based selection

**Evidence use:**
- connect to Theorem 1 through the weighted aggregate negative contribution
- refer quality-selection derivation to Appendix~\ref{app:optimistic-dro} without chronology

**Transition:** The next block shows that this update modifies the exact aggregate term in Theorem 1.
<!-- MANUSCRIPT:END METHOD-P01 -->

<!-- MANUSCRIPT:BEGIN METHOD-P02 -->
## [METHOD-P02] Direct Bridge to Theorem 1
Parent-Outline-SHA256: `f022cd340d47a60f1308d82b300cc5cbc730c3557d884304853b3754afe6a65b`

**Claim:** DRPO replaces the same aggregate negative moment that moves the equilibrium, allowing the method to preserve local displacement while reducing boundary-driving far-field mass.

**Topic sentence:** DRPO replaces the same aggregate negative moment that moves the equilibrium, allowing the method to preserve local displacement while reducing boundary-driving far-field mass.

**Logical moves:**
- uncontrolled negative moment
- weighted negative moment
- controlled signed target
- local retention and far attenuation

**Evidence use:**
- aggregate-moment diagnostics

**Transition:**
<!-- MANUSCRIPT:END METHOD-P02 -->

<!-- MANUSCRIPT:BEGIN METHOD-P03 -->
## [METHOD-P03] Proposition 2: Vanishing Weighted Far-Field Gradient
Parent-Outline-SHA256: `a28d858a65e1ca1c2d492ae622405d4816d9f956abe0b886a75298c957ba2654`

**Claim:** An exponential envelope dominates any finite-order score growth, so the weighted negative gradient vanishes in the far field.

**Topic sentence:** An exponential envelope dominates any finite-order score growth, so the weighted negative gradient vanishes in the far field.

**Logical moves:**
- finite-order score-growth condition
- exponential times polynomial tends to zero
- no assumption that sample utility decays exponentially

**Evidence use:**
- analytic proposition
- distance-binned gradient diagnostics

**Transition:**
<!-- MANUSCRIPT:END METHOD-P03 -->

<!-- MANUSCRIPT:BEGIN METHOD-P04 -->
## [METHOD-P04] Controls and Ablations
Parent-Outline-SHA256: `f972d558397bce6f948c07dd4cf0792b275f3b163d4616ba6477da9a552d3bd5`

**Claim:** Positive-only, global scaling, hard thresholds, linear tapers, and exponential DRPO isolate which benefits come from retaining local negatives, selecting distance, and controlling total gradient budget.

**Topic sentence:** Positive-only, global scaling, hard thresholds, linear tapers, and exponential DRPO isolate which benefits come from retaining local negatives, selecting distance, and controlling total gradient budget.

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

**Transition:**
<!-- MANUSCRIPT:END METHOD-P04 -->

# Experiments

<!-- MANUSCRIPT:BEGIN EXP-P01 -->
## [EXP-P01] Environments and Evidence Roles
Parent-Outline-SHA256: `911d8672bb4d9ca9e8238ee481f0f2bd8df8f169327d0a4a635f52803d168fe4`

**Claim:** C-U1 and D-U1 provide controlled identification, while Hopper/D4RL and Countdown provide external validity; no environment substitutes for another's scientific responsibility.

**Topic sentence:** C-U1 and D-U1 provide controlled identification, while Hopper/D4RL and Countdown provide external validity; no environment substitutes for another's scientific responsibility.

**Logical moves:**
- C-U1 6D context and 2D action
- D-U1 shared semantic categorical actor
- same-distribution held-out contexts
- Hopper learned critic
- Countdown shared parameters
- environment responsibility table

**Evidence use:**
- Appendix D full specifications

**Transition:**
<!-- MANUSCRIPT:END EXP-P01 -->

<!-- MANUSCRIPT:BEGIN EXP-P02 -->
## [EXP-P02] RQ1: External Occurrence
Parent-Outline-SHA256: `dbca74317f5d767afe909bab1b74d760984c251c8c882483e25d5c3ae35862d2`

**Claim:** Hopper and Countdown test whether far-field or rare negative influence becomes disproportionately large before task degradation in realistic policy learning.

**Topic sentence:** Hopper and Countdown test whether far-field or rare negative influence becomes disproportionately large before task degradation in realistic policy learning.

**Logical moves:**
- distance/surprisal bins
- positive/negative imbalance
- temporal ordering
- best and terminal checkpoints
- TBD until formal results close

**Evidence use:**
- EXT-H-E7-Q2 terminal-audited outputs
- Countdown formal outputs

**Transition:**
<!-- MANUSCRIPT:END EXP-P02 -->

<!-- MANUSCRIPT:BEGIN EXP-P03 -->
## [EXP-P03] RQ2a: Matched Badness--Distance and Badness--Rarity Isolation
Parent-Outline-SHA256: `84a1fa0cd982d904b7b61dcc6c2a162e5872c3cda995278294e7f930597791f6`

**Claim:** When quality, advantage, semantics, count, coefficient, and policy stage are matched, learner-relative distance or rarity independently changes negative influence.

**Topic sentence:** When quality, advantage, semantics, count, coefficient, and policy stage are matched, learner-relative distance or rarity independently changes negative influence.

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

**Transition:**
<!-- MANUSCRIPT:END EXP-P03 -->

<!-- MANUSCRIPT:BEGIN EXP-P04 -->
## [EXP-P04] RQ2b: Targeted Causal Transmission
Parent-Outline-SHA256: `16bb56f504d2c47f264351b3f0d160ce052d0d75ad2d26ab00c985df098be976`

**Claim:** Near/far and common/rare interventions test whether the anomalous remote component is the pathway that transmits negative influence into drift, boundary events, and task collapse.

**Topic sentence:** Near/far and common/rare interventions test whether the anomalous remote component is the pathway that transmits negative influence into drift, boundary events, and task collapse.

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

**Transition:**
<!-- MANUSCRIPT:END EXP-P04 -->

<!-- MANUSCRIPT:BEGIN EXP-P05 -->
## [EXP-P05] RQ3: Phase Transition and DRPO Control
Parent-Outline-SHA256: `61ee39a278ff214d04152a5645d874d6433c0aa9696a3de8dbb0598715df739f`

**Claim:** Strength sweeps map the regimes of Theorem 1, while DRPO and matched controls test whether selective far-field attenuation preserves stable extrapolation.

**Topic sentence:** Strength sweeps map the regimes of Theorem 1, while DRPO and matched controls test whether selective far-field attenuation preserves stable extrapolation.

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

**Transition:**
<!-- MANUSCRIPT:END EXP-P05 -->

<!-- MANUSCRIPT:BEGIN EXP-P06 -->
## [EXP-P06] RQ4: External Task Closure
Parent-Outline-SHA256: `88580e45d2e808503b1b8cf55ee7cc5fd42e925a28e66e535e9b251d70bab3a5`

**Claim:** External D4RL and Countdown evaluations test whether controlling remote negative influence improves actual task performance without sacrificing the useful negative signal.

**Topic sentence:** External D4RL and Countdown evaluations test whether controlling remote negative influence improves actual task performance without sacrificing the useful negative signal.

**Logical moves:**
- D4RL locomotion datasets
- Countdown model scales and common data bank
- same initialization and selection rules
- best and terminal metrics
- mechanism diagnostics beside performance

**Evidence use:**
- EXT-H-E7-BENCH-01
- EXT-C-E8-SCALE-01

**Transition:**
<!-- MANUSCRIPT:END EXP-P06 -->

# Implications and Conclusion

<!-- MANUSCRIPT:BEGIN DISC-P01 -->
## [DISC-P01] Negative Feedback Is a Resource with a Dynamical Failure Mode
Parent-Outline-SHA256: `b58c2837a52ab71c4a814c45e94c781c1c197dacb5dd010cca0e167e726e076d`

**Claim:** The transferable lesson is to control learner-relative far-field negative influence rather than classify all negative feedback as harmful.

**Topic sentence:** The transferable lesson is to control learner-relative far-field negative influence rather than classify all negative feedback as harmful.

**Logical moves:**
- negative feedback supports extrapolation
- historical reuse changes its relevance and magnitude
- control object is remote negative influence

**Evidence use:**
- theory and controlled evidence

**Transition:**
<!-- MANUSCRIPT:END DISC-P01 -->

<!-- MANUSCRIPT:BEGIN DISC-P02 -->
## [DISC-P02] Continuous and Categorical Synthesis
Parent-Outline-SHA256: `bb9bd1e0d1f36fd67d6e878ce88b9f030bc1d02f7e2e7d9293a699b4e8338bb3`

**Claim:** Gaussian and categorical policies share aggregate boundary dynamics while differing in local score growth and failure manifestation.

**Topic sentence:** Gaussian and categorical policies share aggregate boundary dynamics while differing in local score growth and failure manifestation.

**Logical moves:**
- Gaussian distance-amplified score and support dynamics
- categorical bounded persistent suppression
- shared feasibility-boundary transition

**Evidence use:**
- family corollaries
- C-U1/D-U1 evidence

**Transition:**
<!-- MANUSCRIPT:END DISC-P02 -->

<!-- MANUSCRIPT:BEGIN DISC-P03 -->
## [DISC-P03] Conclusion
Parent-Outline-SHA256: `cb639ec0985fdaf4c05845bac64b08449e7761deeaa695381b0bf9c0a83451f1`

**Claim:** DRPO preserves useful negative feedback by attenuating the remote component of the aggregate term that drives equilibrium loss.

**Topic sentence:** DRPO preserves useful negative feedback by attenuating the remote component of the aggregate term that drives equilibrium loss.

**Logical moves:**
- stable extrapolation
- badness--distance isolation
- far-field causal pathway
- DRPO recovery

**Evidence use:**
- all main claims

**Transition:**
<!-- MANUSCRIPT:END DISC-P03 -->

# Proofs for Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN APP-PROOF-P01 -->
## [APP-PROOF-P01] Proof of Theorem 1
Parent-Outline-SHA256: `6417ceca714290637770be31be209b7131e524b5be2dd163f03b02416415b7ae`

**Claim:** The equilibrium theorem follows from the diffeomorphism between natural and mean parameters in a regular minimal exponential family.

**Topic sentence:** The equilibrium theorem follows from the diffeomorphism between natural and mean parameters in a regular minimal exponential family.

**Logical moves:**
- derive aggregate field
- interior mean-space existence
- boundary divergence
- Jacobian

**Evidence use:**
- standard exponential-family properties

**Transition:**
<!-- MANUSCRIPT:END APP-PROOF-P01 -->

# Gaussian Mean--Variance Derivations

<!-- MANUSCRIPT:BEGIN APP-GAUSS-P01 -->
## [APP-GAUSS-P01] Corrected Gaussian Mean and Variance Dynamics
Parent-Outline-SHA256: `71b0fb4ffc940458ba512b7e03c14a7975b4c8a03f4c1988a9e69eb7bf459ca5`

**Claim:** Far-field negative advantages repel the mean and tend to contract variance when standardized distance exceeds one.

**Topic sentence:** Far-field negative advantages repel the mean and tend to contract variance when standardized distance exceeds one.

**Logical moves:**
- mean score
- log-standard-deviation score
- four sign/location quadrants
- joint equilibrium

**Evidence use:**
- analytic Gaussian score

**Transition:**
<!-- MANUSCRIPT:END APP-GAUSS-P01 -->

# Categorical Boundary Dynamics

<!-- MANUSCRIPT:BEGIN APP-CAT-P01 -->
## [APP-CAT-P01] Categorical Log-Odds and Boundary Behavior
Parent-Outline-SHA256: `58bbfe3cc8dbd055898f1627dbb7163af9c12d135ebdfadbc4916b772f62f94f`

**Claim:** Repeated negative updates produce persistent log-odds decay even though the direct-logit score norm is bounded.

**Topic sentence:** Repeated negative updates produce persistent log-odds decay even though the direct-logit score norm is bounded.

**Logical moves:**
- softmax score bound
- log-odds update
- probability decay
- shared-parameter caveat

**Evidence use:**
- categorical algebra

**Transition:**
<!-- MANUSCRIPT:END APP-CAT-P01 -->

# Controlled Environments

<!-- MANUSCRIPT:BEGIN APP-ENV-P01 -->
## [APP-ENV-P01] C-U1 and D-U1 Construction
Parent-Outline-SHA256: `3cc8b9c2f0ea10fc347300a099887aee7ec8da0b7aff0641a1827012f1d3c1a1`

**Claim:** The controlled environments expose ground truth and permit matched isolation without claiming that construction alone establishes external validity.

**Topic sentence:** The controlled environments expose ground truth and permit matched isolation without claiming that construction alone establishes external validity.

**Logical moves:**
- C-U1 states/actions/reward/hidden optimum
- train/test same distribution
- D-U1 semantic actions
- matched probes
- dynamic near/far

**Evidence use:**
- environment code and manifests

**Transition:**
<!-- MANUSCRIPT:END APP-ENV-P01 -->

# Experimental Protocols and Terminal Audits

<!-- MANUSCRIPT:BEGIN APP-PROT-P01 -->
## [APP-PROT-P01] Stopping, Budget Matching, and Terminal Classification
Parent-Outline-SHA256: `68c8ae0ba9d2362e14ccec56b1e53627629452cdb4c348cb8ad7002c86628a8c`

**Claim:** All dynamic and ranking claims require fixed protocol ownership, paired seeds, explicit budgets, and terminal audits.

**Topic sentence:** All dynamic and ranking claims require fixed protocol ownership, paired seeds, explicit budgets, and terminal audits.

**Logical moves:**
- maximum horizon
- terminal slopes/residuals
- 2x continuation where registered
- raw pre-optimizer negative-gradient budget
- best and terminal

**Evidence use:**
- handoff and registry

**Transition:**
<!-- MANUSCRIPT:END APP-PROT-P01 -->

# Additional Results and Failure Taxonomy

<!-- MANUSCRIPT:BEGIN APP-RES-P01 -->
## [APP-RES-P01] Additional Tables, Curves, and Negative Results
Parent-Outline-SHA256: `a017c291cea9bf19231acdbdd619a85cf5eea962cf22ed488e16ef1aa47e3c4f`

**Claim:** Appendix results preserve uncertainty, failed runs, and non-ranking outcomes instead of presenting only favorable endpoints.

**Topic sentence:** Appendix results preserve uncertainty, failed runs, and non-ranking outcomes instead of presenting only favorable endpoints.

**Logical moves:**
- per-seed tables
- full trajectories
- sensitivity
- negative results
- failure inventory

**Evidence use:**
- formal artifact packages

**Transition:**
<!-- MANUSCRIPT:END APP-RES-P01 -->

# Implementation and Reproducibility

<!-- MANUSCRIPT:BEGIN APP-REPRO-P01 -->
## [APP-REPRO-P01] Code, Data, and Artifact Provenance
Parent-Outline-SHA256: `42164710012fe7cf262a96a46cd4e45960ab35a5d2727b06dfa51d5806fb7d00`

**Claim:** Every formal claim is bound to code, configuration, seeds, raw outputs, terminal audit, and a source commit.

**Topic sentence:** Every formal claim is bound to code, configuration, seeds, raw outputs, terminal audit, and a source commit.

**Logical moves:**
- repository paths
- run manifests
- checksums
- formal package lifecycle
- Overleaf build

**Evidence use:**
- repository scripts and artifacts

**Transition:**
<!-- MANUSCRIPT:END APP-REPRO-P01 -->

# Optimistic Distributional Selection

<!-- MANUSCRIPT:BEGIN APP-DRO-P01 -->
## [APP-DRO-P01] Optimistic-DRO Quality Selection and Its Distinct Role
Parent-Outline-SHA256: `fd46430511579c9d4be1f30948b8bc8bb4ea665b8dd20d754fba3c4bb74c73aa`

**Claim:** The Optimistic-DRO subdistribution problem yields a quality-based hard-selection solution; this result is part of the current manuscript but is mathematically distinct from learner-relative remoteness tapering.

**Topic sentence:** Optimistic distributional selection and learner-relative remoteness control answer different questions within the same DRPO manuscript.

**Logical moves:**
- state the density-ratio-constrained quality-selection problem
- state its hard-selection solution
- explain the exact scope of that result
- separate it from the remoteness envelope used in the main actor update

**Evidence use:**
- provide the self-contained optimization and solution statement
- cross-reference the main matched badness--distance experiments

**Transition:** The correction ledger records how this and other retained results differ from superseded claims.
<!-- MANUSCRIPT:END APP-DRO-P01 -->

# Additional Theoretical and Reporting Clarifications

<!-- MANUSCRIPT:BEGIN APP-CORR-P01 -->
## [APP-CORR-P01] Additional Theoretical and Reporting Clarifications
Parent-Outline-SHA256: `d28ea4945802e9a7a538832293499bd862a3e8c4c9b006af182b6ad562f80730`

**Claim:** The manuscript consistently uses corrected Gaussian scale dynamics, the Jacobian of the actual signed field, held-out-context terminology for C-U1, and separate reporting of task, boundary, and numerical failures.

**Topic sentence:** The manuscript consistently uses corrected Gaussian scale dynamics, the Jacobian of the actual signed field, held-out-context terminology for C-U1, and separate reporting of task, boundary, and numerical failures.

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

**Transition:**
<!-- MANUSCRIPT:END APP-CORR-P01 -->
