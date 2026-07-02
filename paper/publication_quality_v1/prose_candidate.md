# Publication-quality prose candidate

## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource

Off-policy policy optimization improves a current policy by reusing behavior collected by earlier, stale, or heterogeneous policies. This paradigm is central to offline reinforcement learning, replay-based control, recommender systems, and model post-training because it reduces the need for fresh interaction and exploits experience accumulated across policy iterations~\cite{levine2020offline,fujimoto2019off}. Historical data contain two complementary signals. Positive updates increase the likelihood of observed successful actions, whereas negative feedback suppresses known failures and competing modes~\cite{peng2019advantage,wang2020critic}. The Positive-only endpoint, which removes every negative update, therefore yields a useful stability reference but can restrict the learner to attraction toward the empirical positive target. Balanced repulsion can instead reshape the decision boundary and move the aggregate equilibrium beyond that target. The central question is not whether negative feedback should be kept or discarded wholesale, but how to preserve its policy-improvement value without allowing repeated reuse to turn it into excessive repulsion.

## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion

The difficulty appears when negative actions are historical rather than freshly sampled. Offline logs, replay buffers, stale actors, and asynchronous trajectories can keep presenting the same action after the current learner has changed~\cite{schulman2017proximal,haarnoja2018soft}. A negative sample may initially lie near the policy and provide locally informative boundary information. Its update then pushes the learner away, increasing the action's learner-relative distance or surprisal while the stored negative label remains active. In Gaussian policies, the mean score can grow with standardized distance, so subsequent reuse produces progressively larger repulsion. In categorical policies, the direct-logit score is bounded, yet repeated negative updates continue to lower the selected action's log-odds and can drive its probability toward the support boundary. Historical reuse therefore creates a policy-family-dependent transition from useful local feedback to destructive far-field influence, rather than merely exposing the learner to one unusually large update.

## [INTRO-P03] Existing Controls and the Missing Identification Link

Existing methods already regulate negative or off-policy updates in several principled ways. Positive-only objectives remove negative terms; global coefficients and clipping reduce their scale; trust regions and behavior constraints restrict policy movement; rarity-aware rules target low-probability events; and quality filtering removes selected data~\cite{schulman2017proximal,kumar2020conservative,kostrikov2021offline,peng2019advantage}. These controls can improve stability, but they do not by themselves explain why the same negative action can be useful near the learner and destructive in the far field after repeated reuse. The issue is difficult to identify in ordinary logs because low reward, large negative advantage, rarity, and learner-relative distance are typically correlated. We therefore construct matched controls that use matching context and semantic role, holding negative-advantage magnitude, sample count, base coefficient, and policy stage fixed to isolate policy remoteness. The source-identification protocol tests whether remoteness independently changes gradient scale; a separate intervention protocol tests whether the resulting far-field influence transmits into drift, support-boundary events, and task-performance collapse. This separation motivates an aggregate theory rather than another static weighting heuristic.

## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss

Repulsive Dynamics separates three layers that are often conflated. First, learner-relative remoteness determines the score response of a fixed historical action. Second, repeated reuse creates an asymmetry: positive updates make a successful action more compatible with the learner and attenuate its score, whereas negative updates make a rejected action more remote and preserve or amplify its response. Third, positive and negative contributions aggregate into a signed policy field. When positive mass dominates and the signed target remains inside the feasible mean-parameter space, the policy has a finite stable equilibrium beyond the Positive-only target. At the critical balance the field can produce persistent drift, while negative dominance makes any finite stationary point unstable. The terminal manifestation depends on the policy family: Gaussian mean runaway yields unbounded far-field scores, whereas categorical policies approach the simplex boundary with bounded per-sample logit scores. Negative feedback is therefore neither uniformly beneficial nor uniformly harmful; its value depends on aggregate balance and learner-relative location.

## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term

DRPO acts on the mechanism identified by the theory rather than treating every negative sample as toxic. Starting from the signed empirical actor field, it reweights the aggregate negative term by learner-relative distance or surprisal. Nearby negative actions retain substantial weight and can continue to shape local boundaries or suppress competing bad modes. As an action becomes increasingly remote, the weight decays nonlinearly, preventing the far-field tail from dominating the positive attraction. The exponential envelope is chosen for a precise tail property: under finite-order growth of the unweighted score-times-advantage contribution, the weighted contribution vanishes as remoteness increases. This guarantee does not assume that a sample's utility decays exponentially and does not establish a universal ranking over all tapers. Quality-based selection and learner-relative remoteness control are distinct axes; the experiments evaluate them and global scaling under matched conditions rather than presuming that one control must always win.

## [INTRO-P06] Evidence Chain and Contributions

The empirical program follows the same decomposition as the theory. First, we ask whether far-field or rare negative updates become disproportionately influential in realistic policy learning. Second, using matched quality--distance and quality--rarity controls, we test whether remoteness independently changes negative influence and whether targeted far-field interventions interrupt drift or boundary events. Third, we test the predicted sequence from the Positive-only platform through stable extrapolation to persistent drift or instability, and compare selective attenuation with global and budget-matched controls. Fourth, we evaluate whether the resulting control improves external tasks. C-U1 and D-U1 provide controlled continuous and categorical identification; Hopper/D4RL and Countdown provide external validity and do not replace those controlled mechanisms. Every dynamics claim is terminal-audited, and task-performance collapse, support or variance-boundary events, and NaN/Inf numerical failure are reported separately. We contribute a layered theory of repeated repulsion, matched causal identification of learner-relative remoteness, DRPO control of the far-field negative term, and an evidence chain that keeps mechanism claims distinct from external task claims.

## [RELATED-P01] Learning from Negative or Suboptimal Behavior

Quality-aware policy fitting already provides several ways to use or suppress suboptimal behavior. Advantage-weighted regression, implicit value learning, critic-regularized regression, and proximal behavior objectives emphasize actions estimated to be better while reducing the impact of lower-quality data~\cite{peng2019advantage,kostrikov2021offline,wang2020critic,zhuang2023behavior}. Positive-only and hard-filtering variants take the conservative endpoint and remove selected negative updates altogether. At the same time, failed or suboptimal behavior can remain informative: it can suppress competing bad modes, sharpen a decision boundary, or release probability mass for alternatives represented by the model. The established picture is therefore not that negative feedback is simply noise, but that it combines information value with optimization risk. What remains under-characterized is the temporal relation to the current learner. A fixed negative action can begin as relevant local feedback and become increasingly remote after repeated reuse. We study that transition and its aggregate consequences, complementing rather than dismissing prior weighting and filtering methods.

## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates

Off-policy learning has long controlled mismatch between the behavior distribution and the current policy. Importance correction, PPO-style clipping, and trust regions limit the effect of samples collected by another policy~\cite{schulman2017proximal}. Replay-based algorithms and maximum-entropy off-policy control likewise manage repeated data reuse while the learner evolves~\cite{haarnoja2018soft}. Offline-RL analyses make the same mismatch explicit when no new interaction is available~\cite{levine2020offline,fu2020d4rl}. Low action probability or high surprisal can therefore serve as a useful learner-relative warning signal. These approaches establish that mismatch and rarity matter; our distinction concerns how they arise. We treat remoteness not only as a static property used to choose a weight, but as an endogenous state variable that a negative update increases and subsequent reuse feeds back into the next update. We then connect that per-sample process to aggregate equilibria and targeted causal interventions. This perspective complements clipping and stale-policy control rather than asserting that they are ineffective.

## [RELATED-P03] Robust Offline Policy Learning

Offline reinforcement learning contains several distinct failure channels, and far-field signed actor dynamics is only one of them. Conservative Q-learning controls value extrapolation by assigning pessimistic values to unsupported actions~\cite{kumar2020conservative}. Implicit Q-learning avoids querying out-of-distribution actions during value improvement and then fits an actor to supported high-value behavior~\cite{kostrikov2021offline}. Behavior-constrained and proximal approaches instead keep the learned policy close to the data or a behavior reference~\cite{fujimoto2019off,wang2020critic,zhuang2023behavior}. Data selection adds another axis by changing which transitions participate in learning. DRPO focuses on the signed actor field after an advantage-like signal has been supplied: it asks how negative actor updates change as historical actions become remote, and how to preserve their local information without allowing their far-field contribution to dominate. This actor-side control can be combined with pessimistic critics, in-support value learning, or behavior regularization. It is not presented as a replacement for those safeguards or as a complete explanation of every offline-RL failure.

## [SETUP-P01] Signed Actor Update and Influence Factorization

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

## [SETUP-P02] Policy-Relative Far Field

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

## [THEORY-P01] Learner-Relative Remoteness and Static Score Response

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

## [THEORY-P02] Self-Attenuation and Self-Amplification under Reuse

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

## [THEORY-P03] Aggregate Attraction--Repulsion Equilibria

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

## [THEORY-P04] Policy-Family Manifestations of Negative Dominance

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

## [THEORY-P05] Observable Regimes and Experimental Predictions

The preceding results imply a falsifiable phase sequence rather than a generic prediction that negative updates are bad. Positive-only training should reach a finite platform determined by the observed positive target. Under Theorem~\ref{thm:equilibrium}, moderate negative influence should produce stable extrapolation to a second finite platform when the signed moment remains feasible. Increasing negative mass or moving its moment outward should bring the policy toward a support, variance, or probability boundary; after equilibrium loss, Theorems~\ref{thm:reuse}--\ref{thm:family-runaway} predict persistent drift or a non-vanishing terminal residual rather than a genuine stationary state. The causal prediction is selective: if the far-field component transmits the instability, removing or capping the far path should rescue a finite terminal regime, whereas removing only near-field negatives should not. Global scaling and budget-matched transfers then test whether total magnitude or learner-relative location is the operative mediator. Proposition~\ref{prop:vanishing} predicts that the DRPO envelope removes any finite-order far-field tail, but it does not pre-rank all taper shapes at finite distance. Every experiment must separately report task-performance collapse, support or variance-boundary events, and NaN/Inf numerical failure. C-U1 and D-U1 provide controlled ground-truth tests of these predictions; Hopper and Countdown assess external validity under learned critics and shared sequence parameters. The controlled environments do not substitute for the external tasks, and finite-step external pilots do not establish terminal method rankings.
