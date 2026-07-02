# Publication-quality paragraph blueprint

## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource

- Reader question: Why is negative feedback a necessary policy-improvement signal rather than disposable noise?
- Paragraph claim: Off-policy policy optimization needs both attraction toward successful behavior and suppression of known failures, but the latter must remain dynamically controlled.
- Word budget: 125-220
- Sentence units:
  1. `INTRO-P01-S01` **context** — Open with the role of historical data in offline, replay-based, recommendation, and post-training policy optimization. (anchors: historical data, off-policy)
  2. `INTRO-P01-S02` **positive_role** — Explain positive attraction toward observed successful actions. (anchors: positive updates, successful)
  3. `INTRO-P01-S03` **negative_role** — Explain why negative feedback suppresses known failures and competing modes. (anchors: negative feedback, suppress)
  4. `INTRO-P01-S04` **limitation** — Position Positive-only as stable but potentially limited to the observed positive target. (anchors: Positive-only, empirical positive target)
  5. `INTRO-P01-S05` **question** — State the paper's central problem: retain useful negative information without allowing repulsion to become excessive. (anchors: central question, excessive repulsion)
- Citations: levine2020offline, fujimoto2019off, peng2019advantage, wang2020critic
- Theorem/equation refs: none
- Appendix bindings: none
- Reviewer objection: The opening could sound as though failures are always useful or that Positive-only is an inadequate baseline by definition.
- Response: State complementarity and a conditional limitation, not a universal ranking.
- Transition: The next paragraph explains why historical reuse changes the character of the same negative signal.

## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion

- Reader question: How can locally informative negative feedback become dynamically dangerous?
- Paragraph claim: Repeated reuse makes negative influence learner-relative: as the current policy moves away from a fixed historical action, the same label can become persistent or self-amplifying far-field repulsion.
- Word budget: 125-220
- Sentence units:
  1. `INTRO-P02-S01` **historical_setting** — Describe offline logs, replay buffers, stale actors, and asynchronous trajectories as fixed or delayed update sources. (anchors: offline logs, replay)
  2. `INTRO-P02-S02` **local_value** — Explain that a negative action can initially supply locally relevant boundary information. (anchors: locally informative, boundary)
  3. `INTRO-P02-S03` **reuse_mechanism** — Show that repulsion increases learner-relative distance while the historical label remains active. (anchors: learner-relative, distance)
  4. `INTRO-P02-S04` **gaussian_case** — State that Gaussian mean scores can grow with standardized distance. (anchors: Gaussian, standardized distance)
  5. `INTRO-P02-S05` **categorical_case** — State that categorical direct-logit scores are bounded but repeated suppression persists. (anchors: categorical, bounded, suppression)
  6. `INTRO-P02-S06` **transition** — Name the useful-local to destructive-far-field transition and hand off to prior controls. (anchors: far-field, transition)
- Citations: schulman2017proximal, haarnoja2018soft
- Theorem/equation refs: none
- Appendix bindings: none
- Reviewer objection: The mechanism might be mistaken for ordinary distribution shift or a one-time large update.
- Response: Emphasize the repeated feedback loop and policy-family-specific response laws.
- Transition: Existing methods control several symptoms of this mismatch, but do not isolate the repeated remoteness feedback loop.

## [INTRO-P03] Existing Controls and the Missing Identification Link

- Reader question: What do existing controls solve, and what identification gap remains?
- Paragraph claim: Existing controls stabilize off-policy learning, but the causal role of learner-relative remoteness remains unidentified when quality, advantage magnitude, rarity, and distance are entangled.
- Word budget: 145-250
- Sentence units:
  1. `INTRO-P03-S01` **method_landscape** — Enumerate positive-only, global scaling, clipping, behavior constraints, rarity-aware control, and quality filtering. (anchors: positive-only, clipping, quality filtering)
  2. `INTRO-P03-S02` **acknowledged_value** — Acknowledge that these controls can stabilize learning and should not be dismissed as tricks. (anchors: stabilize, controls)
  3. `INTRO-P03-S03` **unresolved_gap** — Specify the missing explanation of the local-benefit to far-field-harm transition. (anchors: useful local, far-field)
  4. `INTRO-P03-S04` **confounding** — Explain that quality, negative advantage, rarity, and distance are correlated in ordinary logs. (anchors: advantage magnitude, rarity, distance)
  5. `INTRO-P03-S05` **identification** — Describe matched controls that hold context, semantics, reward, coefficient, and policy stage fixed. (anchors: matched, policy stage)
  6. `INTRO-P03-S06` **transition** — Separate source isolation from causal transmission and lead into the theory. (anchors: source identification, intervention protocol)
- Citations: schulman2017proximal, kumar2020conservative, kostrikov2021offline, peng2019advantage
- Theorem/equation refs: none
- Appendix bindings: none
- Reviewer objection: The claimed gap could be a relabeling of clipping, rarity weighting, or conservative offline RL.
- Response: Define the gap as repeated learner-relative dynamics under matched sample quality and then test source and transmission separately.
- Transition: A theory of repulsive dynamics is needed to connect single-sample remoteness to aggregate equilibria.

## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss

- Reader question: What theoretical structure connects beneficial and harmful negative updates?
- Paragraph claim: Repulsive Dynamics unifies stable extrapolation, persistent drift, and instability through the aggregate balance of positive and negative update masses and moments.
- Word budget: 135-235
- Sentence units:
  1. `INTRO-P04-S01` **theory_object** — Introduce the three-layer theoretical decomposition. (anchors: learner-relative remoteness, aggregate)
  2. `INTRO-P04-S02` **stable_regime** — Explain finite stable extrapolation beyond the Positive-only target under positive dominance. (anchors: stable extrapolation, Positive-only)
  3. `INTRO-P04-S03` **boundary_regime** — Explain boundary approach, persistent drift, and loss of finite stability. (anchors: persistent drift, finite)
  4. `INTRO-P04-S04` **family_difference** — Contrast Gaussian unbounded score growth with categorical bounded-gradient boundary degeneration. (anchors: Gaussian, categorical, bounded)
  5. `INTRO-P04-S05` **implication** — State that negative feedback is neither uniformly beneficial nor uniformly harmful. (anchors: neither, beneficial, harmful)
  6. `INTRO-P04-S06` **transition** — Connect the theoretical failure term to the method design. (anchors: method, far-field)
- Citations: none
- Theorem/equation refs: none
- Appendix bindings: app:proof-score-remoteness, app:proof-reuse, app:proof-theorem-equilibrium, app:proof-family-runaway
- Reviewer objection: A single equilibrium theorem could hide the distinction between per-sample amplification and aggregate stability.
- Response: Preview the static, repeated-reuse, and aggregate results as separate logical layers.
- Transition: DRPO is then defined as a control of the far-field contribution in the aggregate negative term.

## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term

- Reader question: How does the method intervene on the mechanism rather than merely discard negative data?
- Paragraph claim: DRPO preserves informative local negative feedback while attenuating the learner-relative far-field tail that drives the unstable aggregate regime.
- Word budget: 120-215
- Sentence units:
  1. `INTRO-P05-S01` **method_target** — Identify the aggregate negative term as the intervention target. (anchors: negative term, actor update)
  2. `INTRO-P05-S02` **local_preservation** — Explain why nearby negative actions retain substantial weight. (anchors: nearby, retain)
  3. `INTRO-P05-S03` **far_attenuation** — Explain nonlinear attenuation with distance or surprisal. (anchors: distance, surprisal, attenuation)
  4. `INTRO-P05-S04` **guarantee** — State the finite-order score-growth tail guarantee. (anchors: finite-order, vanish)
  5. `INTRO-P05-S05` **distinction** — Separate remoteness control from quality-based data selection. (anchors: quality, separate axes)
  6. `INTRO-P05-S06` **transition** — Lead to the layered empirical program without ranking all tapers in advance. (anchors: empirical, ranking)
- Citations: none
- Theorem/equation refs: none
- Appendix bindings: app:proof-far-field
- Reviewer objection: The exponential envelope could appear to assume that negative-sample utility decays exponentially.
- Response: Ground the choice in domination of finite-order score growth, not in an assumed utility curve.
- Transition: The experiments separately test occurrence, source, causal transmission, phase behavior, and external task value.

## [INTRO-P06] Evidence Chain and Contributions

- Reader question: How do the experiments jointly support the paper without conflating controlled mechanisms and external validity?
- Paragraph claim: The evidence chain assigns occurrence, controlled source identification, causal transmission, phase testing, and external validity to distinct experiments and reports terminal outcomes with a separated failure taxonomy.
- Word budget: 150-270
- Sentence units:
  1. `INTRO-P06-S01` **rq1** — State the external occurrence question. (anchors: First, realistic)
  2. `INTRO-P06-S02` **rq2** — State the matched source and targeted causal question. (anchors: Second, matched)
  3. `INTRO-P06-S03` **rq3** — State the phase and control question. (anchors: Third, phase)
  4. `INTRO-P06-S04` **rq4** — State the external task-performance question. (anchors: Fourth, external)
  5. `INTRO-P06-S05` **environment_roles** — Separate C-U1/D-U1 controlled roles from Hopper/Countdown external validity and mention terminal audit. (anchors: C-U1, D-U1, Hopper, Countdown)
  6. `INTRO-P06-S06` **contributions** — Summarize the theory, identification, method, and evidence contributions. (anchors: contribute, theory, causal)
- Citations: none
- Theorem/equation refs: none
- Appendix bindings: none
- Reviewer objection: A long experiment list could blur which environment supports which claim.
- Response: Frame the experiments as four questions and explicitly separate controlled identification from external validation.
- Transition: Related work next locates each component against prior approaches.

## [RELATED-P01] Learning from Negative or Suboptimal Behavior

- Reader question: What is already known about learning from positive, negative, and suboptimal behavior?
- Paragraph claim: Prior work shows both that negative updates can be filtered for stability and that failures can carry useful learning signal; the missing object is how the same negative action changes as it becomes remote from the learner.
- Word budget: 145-250
- Sentence units:
  1. `RELATED-P01-S01` **prior_family** — Introduce quality-aware actor fitting as the broad family. (anchors: quality-aware, actor)
  2. `RELATED-P01-S02` **positive_filtering** — Describe weighting and filtering of low-advantage actions. (anchors: advantage-weighted, filter)
  3. `RELATED-P01-S03` **negative_value** — Explain why failed behavior can still suppress undesirable modes or refine boundaries. (anchors: failed, boundaries)
  4. `RELATED-P01-S04` **established_fact** — Synthesize the shared conclusion that negative feedback is potentially informative but risky. (anchors: informative, risk)
  5. `RELATED-P01-S05` **unresolved_gap** — Identify the missing repeated learner-relative transition. (anchors: repeated, learner-relative)
  6. `RELATED-P01-S06` **positioning** — Position this paper as dynamics and control, not a rejection of prior filtering. (anchors: dynamics, complement)
- Citations: peng2019advantage, kostrikov2021offline, wang2020critic, zhuang2023behavior
- Theorem/equation refs: none
- Appendix bindings: none
- Reviewer objection: The novelty claim could ignore extensive work on positive-only filtering and negative examples.
- Response: Credit both lines and define the contribution as repeated learner-relative dynamics and matched causal isolation.
- Transition: The next family controls behavior--learner mismatch and stale or low-probability updates.

## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates

- Reader question: How does this work differ from established off-policy, stale-policy, and low-probability update controls?
- Paragraph claim: Importance correction, clipping, trust regions, stale-policy controls, and rarity-aware rules regulate mismatch or update scale, but usually treat remoteness as a static signal rather than an endogenous repeated-reuse process.
- Word budget: 145-250
- Sentence units:
  1. `RELATED-P02-S01` **mismatch_controls** — Describe importance correction, clipping, and trust regions. (anchors: importance, clipping, trust)
  2. `RELATED-P02-S02` **stale_updates** — Describe replay-based, stale-policy, and asynchronous reuse. (anchors: replay, stale)
  3. `RELATED-P02-S03` **rarity_controls** — Explain why low probability or surprisal can guide selective attenuation. (anchors: low-probability, surprisal)
  4. `RELATED-P02-S04` **established_fact** — State that these methods establish the importance of learner-relative mismatch. (anchors: learner-relative, mismatch)
  5. `RELATED-P02-S05` **dynamic_gap** — Distinguish static weighting from remoteness increased by the negative update itself. (anchors: endogenous, repeated)
  6. `RELATED-P02-S06` **positioning** — Position the aggregate-equilibrium analysis as complementary. (anchors: aggregate, complement)
- Citations: schulman2017proximal, haarnoja2018soft, levine2020offline, fu2020d4rl
- Theorem/equation refs: none
- Appendix bindings: none
- Reviewer objection: Far-field control may look like a renamed probability or importance-ratio heuristic.
- Response: Separate the measured coordinate from the dynamic claim that negative reuse increases that coordinate and alters the aggregate equilibrium.
- Transition: Conservative offline RL addresses a complementary value and support problem.

## [RELATED-P03] Robust Offline Policy Learning

- Reader question: How is DRPO related to conservative and behavior-regularized offline reinforcement learning?
- Paragraph claim: Conservative value learning, in-support dynamic programming, behavior regularization, and data selection address major offline-RL failures, while DRPO focuses on the signed actor field and selectively preserves local negative information.
- Word budget: 145-255
- Sentence units:
  1. `RELATED-P03-S01` **pessimism** — Describe pessimistic value learning and extrapolation-error control. (anchors: pessimistic, value)
  2. `RELATED-P03-S02` **in_support_learning** — Describe in-support or implicit value learning. (anchors: in-support, implicit)
  3. `RELATED-P03-S03` **behavior_regularization** — Describe behavior regularization and proximal actor constraints. (anchors: behavior, proximal)
  4. `RELATED-P03-S04` **data_selection** — Describe filtering or quality selection as another axis. (anchors: selection, quality)
  5. `RELATED-P03-S05` **distinct_object** — Define the signed actor field as this paper's object. (anchors: signed actor, negative)
  6. `RELATED-P03-S06` **compatibility** — Explain that DRPO can be combined with value-side safeguards and is not a universal offline-RL replacement. (anchors: combined, not replace)
- Citations: kumar2020conservative, kostrikov2021offline, fujimoto2019off, wang2020critic, zhuang2023behavior
- Theorem/equation refs: none
- Appendix bindings: none
- Reviewer objection: The paper could overclaim a complete explanation of offline-RL instability.
- Response: Explicitly bound the contribution to signed actor dynamics and describe compatibility with value-side methods.
- Transition: The setup now formalizes the historical signed actor field and learner-relative remoteness.

## [SETUP-P01] Signed Actor Update and Influence Factorization

- Reader question: What exact update field is analyzed, what is held fixed during an actor step, and how are positive and negative contributions separated?
- Paragraph claim: The analysis studies a historical signed actor field whose per-sample negative influence factors into advantage severity and score geometry, while frequency and directional coherence determine aggregation.
- Word budget: 190-340
- Sentence units:
  1. `SETUP-P01-S01` **historical_distribution** — Define the historical update distribution over state--action pairs. (anchors: historical update distribution, nu)
  2. `SETUP-P01-S02` **actor_field** — Define the signed empirical actor field and discrete update. (anchors: mathbf F, discrete update)
  3. `SETUP-P01-S03` **fixed_step_scope** — State that the update distribution and advantage-like labels are fixed with respect to theta during the actor step. (anchors: treated as fixed, actor step)
  4. `SETUP-P01-S04` **sign_split** — Define positive and negative advantage components. (anchors: A^+, A^-)
  5. `SETUP-P01-S05` **influence_factorization** — Factor negative per-sample magnitude into severity and score geometry. (anchors: severity, score geometry)
  6. `SETUP-P01-S06` **aggregation** — Name frequency and directional coherence as aggregate factors. (anchors: frequency, directional coherence)
  7. `SETUP-P01-S07` **boundary** — Clarify that the field is not assumed to be the exact on-policy policy gradient. (anchors: not assumed, on-policy)
- Citations: sutton1999policy, levine2020offline
- Theorem/equation refs: eq:signed-field, eq:sign-split, eq:influence-factorization
- Appendix bindings: none
- Reviewer objection: Freezing the update distribution and advantage could be mistaken for a claim about all of RL.
- Response: State it as an actor-step mechanism analysis and separate extensions with evolving critics or ratios.
- Transition: The next setup paragraph defines the common learner-relative remoteness coordinate without equating policy families.

## [SETUP-P02] Policy-Relative Far Field

- Reader question: How is far field defined across continuous, categorical, and sequence policies?
- Paragraph claim: Learner-relative remoteness is negative log probability under the current policy; it maps to Mahalanobis distance for Gaussian policies and surprisal for categorical policies without implying identical amplification laws.
- Word budget: 170-300
- Sentence units:
  1. `SETUP-P02-S01` **definition** — Define remoteness as negative log probability under the current policy. (anchors: D_theta, negative log)
  2. `SETUP-P02-S02` **gaussian_mapping** — Map Gaussian remoteness to a covariance constant plus squared Mahalanobis distance. (anchors: Mahalanobis, Gaussian)
  3. `SETUP-P02-S03` **categorical_mapping** — Map categorical remoteness to selected-action surprisal. (anchors: categorical, surprisal)
  4. `SETUP-P02-S04` **sequence_mapping** — Describe normalized token or completion NLL for sequence policies. (anchors: sequence, normalized)
  5. `SETUP-P02-S05` **dynamic_status** — Explain that remoteness changes as theta changes even when data are fixed. (anchors: dynamic, theta)
  6. `SETUP-P02-S06` **non_equivalence** — Define squared score response and warn that its law differs by policy family. (anchors: no universal relation, not assumed)
  7. `SETUP-P02-S07` **transition** — Lead to the static score-remoteness proposition. (anchors: Proposition, score)
- Citations: none
- Theorem/equation refs: eq:remoteness, eq:score-response
- Appendix bindings: none
- Reviewer objection: A common remoteness notation could conceal incompatible continuous and discrete geometries.
- Response: Define the common probabilistic coordinate while deriving each family's response separately.
- Transition: We first derive the static remoteness--score relation before analyzing repeated reuse.

## [THEORY-P01] Learner-Relative Remoteness and Static Score Response

- Reader question: How does learner-relative remoteness determine the strength of a single score contribution before any repeated update is considered?
- Paragraph claim: At a fixed policy, Gaussian mean-score magnitude grows without bound with Mahalanobis remoteness, whereas categorical selected-logit response increases with surprisal but saturates.
- Word budget: 250-450
- Sentence units:
  1. `THEORY-P01-S01` **motivation** — Explain why static score geometry must be isolated before repeated dynamics. (anchors: static relation, fixed learner)
  2. `THEORY-P01-S02` **definitions** — Recall remoteness and score response at a fixed context. (anchors: D_, R_)
  3. `THEORY-P01-S03` **gaussian_statement** — State the Gaussian covariance-eigenvalue bounds and isotropic exact ordering. (anchors: lambda_{\min}, lambda_{\max}, isotropic)
  4. `THEORY-P01-S04` **gaussian_interpretation** — Interpret the Gaussian response as unbounded mean-score growth with remoteness. (anchors: unbounded, Gaussian mean score)
  5. `THEORY-P01-S05` **categorical_statement** — State the categorical selected-logit formula and full-score bound. (anchors: 1-e, bounded)
  6. `THEORY-P01-S06` **categorical_interpretation** — Interpret rarity as stronger but saturating direct-logit suppression. (anchors: saturat, suppression)
  7. `THEORY-P01-S07` **limitation** — State that a static relation alone does not imply global training divergence. (anchors: does not, divergence)
  8. `THEORY-P01-S08` **transition** — Lead to repeated reuse of one fixed sample. (anchors: repeated reuse, next)
- Citations: none
- Theorem/equation refs: prop:score-remoteness, eq:remoteness, eq:score-response
- Appendix bindings: app:proof-score-remoteness
- Reviewer objection: A cross-family proposition might falsely imply identical growth laws.
- Response: State separate formulas and use only learner-relative ordering as the common abstraction.
- Transition: The next theorem turns the static relation into a temporal feedback loop under repeated reuse.

## [THEORY-P02] Self-Attenuation and Self-Amplification under Reuse

- Reader question: How does repeatedly reusing one fixed historical action turn static geometry into a feedback process?
- Paragraph claim: For a convex negative log-likelihood along the update path, repeated negative reuse increases remoteness and cannot decrease score response, while sufficiently small positive reuse decreases both.
- Word budget: 260-470
- Sentence units:
  1. `THEORY-P02-S01` **reuse_setup** — Define one fixed action, negative log likelihood, score response, and repeated update. (anchors: fixed historical action, D_t, R_t)
  2. `THEORY-P02-S02` **negative_statement** — State monotone remoteness increase and non-decreasing response for negative reuse. (anchors: D_{t+1}, negative reuse)
  3. `THEORY-P02-S03` **positive_statement** — State remoteness decrease and non-increasing response for sufficiently small positive reuse. (anchors: positive reuse, L-Lipschitz)
  4. `THEORY-P02-S04` **convexity_scope** — Explain where convexity holds and what coordinate system is intended. (anchors: convexity, natural coordinates)
  5. `THEORY-P02-S05` **gaussian_consequence** — Connect the theorem to unbounded Gaussian mean-score amplification. (anchors: Gaussian, unbounded)
  6. `THEORY-P02-S06` **categorical_consequence** — Connect it to bounded but persistent categorical suppression. (anchors: categorical, persistent)
  7. `THEORY-P02-S07` **limitation** — State that individual-sample amplification does not determine the full aggregate trajectory. (anchors: does not determine, aggregate)
  8. `THEORY-P02-S08` **transition** — Lead to aggregate positive--negative competition. (anchors: aggregate, balance)
- Citations: none
- Theorem/equation refs: thm:reuse, eq:reuse-update
- Appendix bindings: app:proof-reuse
- Reviewer objection: The theorem may be read as a global claim about every neural policy parameterization.
- Response: State the convex path and coordinate assumptions, then reserve shared-network effects for empirical measurement.
- Transition: Aggregate positive attraction can counter this per-sample repulsion, so the next result solves their balance.

## [THEORY-P03] Aggregate Attraction--Repulsion Equilibria

- Reader question: When do positive attraction and negative repulsion aggregate into stable extrapolation, drift, or instability?
- Paragraph claim: In a regular exponential family, positive dominance yields a unique stable finite equilibrium when the signed target is feasible, equality yields persistent drift, and negative dominance admits no stable finite equilibrium.
- Word budget: 330-620
- Sentence units:
  1. `THEORY-P03-S01` **aggregate_setup** — Define the exponential-family policy and positive/negative masses and moments. (anchors: exponential-family, p, q)
  2. `THEORY-P03-S02` **field_derivation** — Derive the signed aggregate field. (anchors: aggregate field, policy-dependent mean parameter)
  3. `THEORY-P03-S03` **positive_regime** — State unique finite equilibrium under positive dominance and feasibility. (anchors: p>q, unique finite)
  4. `THEORY-P03-S04` **extrapolation_identity** — Interpret the displacement beyond the Positive-only target. (anchors: m^star-m_+, beyond)
  5. `THEORY-P03-S05` **boundary_case** — Explain boundary approach and loss of finite realization. (anchors: boundary, no finite)
  6. `THEORY-P03-S06` **critical_regime** — State persistent drift when p=q and moments differ. (anchors: p=q, persistent drift)
  7. `THEORY-P03-S07` **negative_regime** — State instability of finite stationary points when p<q. (anchors: p<q, unstable)
  8. `THEORY-P03-S08` **stability** — Give continuous and discrete local stability conditions. (anchors: Jacobian, step size)
  9. `THEORY-P03-S09` **transition** — Lead to policy-family manifestations under negative dominance. (anchors: policy-family, manifestations)
- Citations: none
- Theorem/equation refs: thm:equilibrium, eq:aggregate-field, eq:signed-target
- Appendix bindings: app:proof-theorem-equilibrium
- Reviewer objection: The extrapolation identity could be mistaken for a performance theorem.
- Response: Separate geometric equilibrium displacement from task utility and require experiments for the latter.
- Transition: The same aggregate instability has different terminal signatures in Gaussian and categorical policies.

## [THEORY-P04] Policy-Family Manifestations of Negative Dominance

- Reader question: What does aggregate instability look like in Gaussian and categorical policies, and which failure labels are justified?
- Paragraph claim: Negative dominance produces unbounded Gaussian mean displacement and far-field scores, but categorical policies approach the simplex boundary while each direct-logit score remains bounded.
- Word budget: 300-540
- Sentence units:
  1. `THEORY-P04-S01` **shared_instability** — Restate negative dominance and the shared absence of a stable finite equilibrium. (anchors: negative dominance, finite stable equilibrium)
  2. `THEORY-P04-S02` **gaussian_form** — Give the Gaussian affine field around the repelling point. (anchors: aggregate mean field, mu^\dagger)
  3. `THEORY-P04-S03` **gaussian_consequence** — Derive unbounded mean displacement and fixed-action score. (anchors: to infinity, score)
  4. `THEORY-P04-S04` **categorical_form** — Describe gauge-fixed categorical logits and convex ascent. (anchors: gauge, logits)
  5. `THEORY-P04-S05` **categorical_consequence** — State escape to the simplex boundary with bounded per-sample score. (anchors: simplex boundary, less than or equal)
  6. `THEORY-P04-S06` **distinction** — Name Gaussian gradient-amplitude runaway versus categorical support degeneration. (anchors: gradient-amplitude, support)
  7. `THEORY-P04-S07` **reporting_boundary** — State that neither result alone proves task collapse or NaN/Inf failure. (anchors: task-performance, NaN/Inf)
  8. `THEORY-P04-S08` **transition** — Lead to empirically testable regimes and separated event reporting. (anchors: predictions, separate)
- Citations: none
- Theorem/equation refs: thm:family-runaway, eq:gaussian-runaway, eq:categorical-bound
- Appendix bindings: app:proof-family-runaway, app:gaussian, app:categorical
- Reviewer objection: Calling both cases collapse could erase the distinction between gradient amplitude, support, task, and numerical events.
- Response: Use separate theorem statements and preserve the project failure taxonomy.
- Transition: These family-specific signatures yield a concrete set of experimental predictions.

## [THEORY-P05] Observable Regimes and Experimental Predictions

- Reader question: Which observable outcomes would support or contradict the theory?
- Paragraph claim: The theory predicts a sequence from a Positive-only platform through stable controlled extrapolation to boundary approach or persistent runaway, with targeted far-field attenuation restoring a finite terminal regime when that path is causal.
- Word budget: 180-330
- Sentence units:
  1. `THEORY-P05-S01` **prediction_positive_only** — Predict the finite Positive-only platform. (anchors: Positive-only, platform)
  2. `THEORY-P05-S02` **prediction_controlled_negative** — Predict finite stable extrapolation under moderate negative influence. (anchors: moderate, stable extrapolation)
  3. `THEORY-P05-S03` **prediction_boundary** — Predict support or variance boundary approach as the signed target moves outward. (anchors: boundary, outward)
  4. `THEORY-P05-S04` **prediction_runaway** — Predict persistent drift or non-vanishing residual after equilibrium loss. (anchors: persistent drift, residual)
  5. `THEORY-P05-S05` **prediction_intervention** — Predict that far removal or capping rescues when the far path is causal, while near removal does not. (anchors: far, near, rescue)
  6. `THEORY-P05-S06` **failure_taxonomy** — Require separate task, support-boundary, and NaN/Inf reporting. (anchors: task-performance, support, NaN/Inf)
  7. `THEORY-P05-S07` **handoff** — Assign controlled identification to C-U1/D-U1 and external validity to Hopper/Countdown. (anchors: C-U1, D-U1, Hopper, Countdown)
- Citations: none
- Theorem/equation refs: thm:reuse, thm:equilibrium, thm:family-runaway, prop:vanishing
- Appendix bindings: app:proof-reuse, app:proof-theorem-equilibrium, app:proof-family-runaway, app:proof-far-field
- Reviewer objection: A broad theoretical narrative could become unfalsifiable if every poor outcome is called repulsion.
- Response: Pre-register terminal regimes, targeted interventions, and distinct event classes that can contradict the theory.
- Transition: The method section now operationalizes selective far-field control and the experiments test each prediction.
