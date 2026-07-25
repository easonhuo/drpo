from pathlib import Path
import hashlib
import sys

TARGET = Path(sys.argv[1])
BIB = Path(sys.argv[2]) if len(sys.argv) > 2 else None
text = TARGET.read_text(encoding="utf-8")

repls = []
def add(name, old, new):
    repls.append((name, old, new))

add("E1", r'''A negative update lowers the probability of its
historical action, increasing the action's remoteness and thereby changing the
next policy-score response. Informally, we call historical actions below the
control threshold the \emph{near field} and those beyond it the \emph{far
field}; Section~\ref{sec:drpo} makes this distinction operational through
\(D\leq\tau\) and \(D>\tau\).''', r'''A negative update lowers the probability of its historical action,
increasing the action's remoteness and thereby changing the next
policy-score response (the \emph{near field} \(D\leq\tau\) versus the
\emph{far field} \(D>\tau\); Section~\ref{sec:drpo}).''')

add("E2", r'''This is the curse of repulsion: complying with a negative update can
make its next reuse more remote without making the reused signal self-limiting.''', r'''This is the curse of repulsion: following a negative update makes its next
reuse more remote without making the signal self-limiting.''')

add("E4", r'''Section~\ref{subsec:distance_dependent_utility} states the important
boundary of this design: remoteness alone cannot identify remote samples that
remain aligned with true task improvement.''', r'''Remoteness controls geometric exposure, not directional utility
(Section~\ref{subsec:distance_dependent_utility}).''')

add("E5", r'''The experiments separate existence, identification, transmission, and control.
External diagnostics first test whether the predicted pattern appears in
realistic neural policies; controlled environments then isolate its source,
trace its causal effect on policy behavior, and compare selective tapering
against positive-only, global-scaling, and polynomial-tail alternatives.''', r'''We first verify the pattern externally, then use controlled environments to
isolate its source, test its downstream effect, and compare control rules.''')

add("E6", r'''Our contributions are summarized as follows:
\begin{itemize}
    \item We characterize how repeated negative reuse changes
    policy-relative remoteness and derive aggregate regimes for stable
    extrapolation, persistent drift, and the loss of finite stable
    equilibria.

    \item We derive a control-order hierarchy and a selective exponential
    taper with Gaussian ultimate-boundedness and categorical
    self-limiting-suppression guarantees.

    \item We use matched external diagnostics and controlled interventions
    to separate coefficient magnitude from policy geometry, identify a
    far-field causal pathway, and test the stability--utility trade-off of
    selective tapering.
\end{itemize}''', r'''Our contributions are: \textbf{(1)} we characterize how repeated negative
reuse changes policy-relative remoteness and derive aggregate regimes for
stable extrapolation, persistent drift, and the loss of finite stable
equilibria; \textbf{(2)} we derive a control-order hierarchy and a selective
exponential taper with Gaussian ultimate-boundedness and categorical
self-limiting suppression guarantees; and \textbf{(3)} matched external
diagnostics and controlled interventions separate coefficient magnitude
from policy geometry, identify a far-field causal pathway, and test the
stability--utility trade-off of selective tapering.''')

add("E7-E8", r'''\textbf{Learning from Negative or Suboptimal Behavior.}
A broad line of policy-learning methods weights behavior by estimated
quality, fitting actors toward actions deemed valuable under a learned
critic \citep{peng2019advantage,nair2020awac,kostrikov2021offline,
wang2020critic,zhuang2023behavior}; positive-filtered variants are the
conservative endpoint that removes negative updates altogether. Negative
behavior is nevertheless widely exploited: unlikelihood training
suppresses undesired sequences \citep{welleck2019neural}, preference and
binary-feedback objectives learn from rejected or undesirable responses
\citep{rafailov2023direct,ethayarajh2024kto,duan2024negating}, and
negative-preference optimization documents utility collapse under overly
aggressive suppression \citep{zhang2024negative}. Failed or suboptimal
behavior can also suppress competing bad modes and improve data
efficiency \citep{novati2019remember,fakoor2020p3o,arnal2025asymmetric,
zhu2025negative,song2025good}. Negative feedback is therefore already in
use; what remains unclear is how its influence should evolve as the
learner moves away from it.

\textbf{Off-Policy, Stale, and Learner-Relative Updates.}
Off-policy reuse fixes historical behavior while the learner changes.
Classical remedies control behavior--learner mismatch through importance
corrections, clipping, trust regions, and replay
\citep{precup2000eligibility,munos2016safe,espeholt2018impala,
schulman2015trust,schulman2017proximal,schaul2016prioritized,
haarnoja2018soft}, and offline RL constrains value extrapolation and
policy support
\citep{fujimoto2019off,kumar2019stabilizing,wu2019behavior,
kumar2020conservative,fujimoto2021minimalist,levine2020offline}. In
language-model post-training, fixed preference or offline corpora
support several alignment and offline-RL pipelines
\citep{ouyang2022training,snell2023offline}, and data refresh or
on-/off-policy mixing is proposed to mitigate distribution shift and
staleness \citep{yin2024self,wang2025inco}. Recent asymmetric or tapered
objectives make updates depend on current-policy probability or surprisal
\citep{arnal2025asymmetric,leroux2025tapered}. These works establish that
mismatch and rarity matter; our distinction is how learner-relative
remoteness is created and amplified by repeated negative reuse.''', r'''\textbf{Off-policy historical reuse across policy domains.}
Off-policy policy optimization reuses data generated by earlier policies,
from replay and offline-RL datasets in continuous control to fixed
preference and verifier-labeled corpora in language-model post-training.
Classical approaches correct behavior--learner mismatch or constrain policy
support through importance weighting, clipping, trust regions, and
conservative or behavior-regularized objectives
\citep{precup2000eligibility,munos2016safe,schulman2015trust,
schulman2017proximal,fujimoto2019off,kumar2020conservative,
fujimoto2021minimalist,kostrikov2021offline}. Across both continuous and
discrete policies, our focus is narrower: repeated reuse of signed actor
feedback, especially the negative branch, after the learner has moved away
from the historical action.

\textbf{Controlling negative updates.}
Existing controls span positive filtering, which removes the negative branch
\citep{peng2019advantage,nair2020awac,kostrikov2021offline}; fixed global
down-weighting of negative advantages, as in HPO's hysteretic weight
\citep{sana2026hpo}; baseline-based positive--negative rebalancing in
AsymRE \citep{arnal2025asymmetric}; and probability-aware tapering in TOPR
\citep{leroux2025tapered}. Failed or rejected behavior is also exploited in
unlikelihood and preference-based training
\citep{welleck2019neural,rafailov2023direct,ethayarajh2024kto,
duan2024negating,zhang2024negative,zhu2025negative,song2025good}.
DRPO differs by treating learner-relative remoteness as a dynamical state
created by repeated reuse and by selectively attenuating only the far-field
negative tail.''')

add("E9", r'''To separate favorable and unfavorable feedback, we define
\begin{equation}
\begin{aligned}
    \widehat{A}^{+}(s,a)
    &= \max\{\widehat{A}(s,a),0\}, \\
    \widehat{A}^{-}(s,a)
    &= \max\{-\widehat{A}(s,a),0\}.
\end{aligned}
\end{equation}
The actor field can then be decomposed into positive and negative
components:''', r'''To separate favorable and unfavorable feedback, write
\(\widehat A^{+}=\max\{\widehat A,0\}\) and
\(\widehat A^{-}=\max\{-\widehat A,0\}\), so the actor field decomposes as''')

add("E10", r'''We use \emph{coefficient magnitude} for \(\widehat A^-\), \emph{score
response} for \(\|\nabla\log\pi\|\), and \emph{update magnitude} for their
product. Section~\ref{sec:distance_strength} defines the squared remoteness
gradient as the \emph{reuse loop gain}, while \emph{aggregate force} refers to
the expectation or sum of signed updates. We use the broader term
\emph{influence} only when these distinctions are immaterial.''', r'''We use \emph{coefficient magnitude} for \(\widehat A^-\), \emph{score
response} for \(\|\nabla\log\pi\|\), \emph{update magnitude} for their
product, \emph{reuse loop gain} for the squared remoteness gradient
(Section~\ref{sec:distance_strength}), and \emph{aggregate force} for the
expectation or sum of signed updates.''')

add("E11", r'''This convention matters when comparing conditional sample fields with
population fields: sampling frequency already represented by \(\nu\) is not
applied a second time.''', r'''Because \(p\) and \(q\) are population expectations under \(\nu\), positive
and negative sample frequencies are already included.''')

add("E12", r'''We study how learner-relative remoteness shapes signed policy updates when
historical behavior is reused. We first characterize, in policy-output
coordinates, how the remoteness of a sample determines the strength of its
score contribution. We then show how repeated reuse turns this static
geometry into a dynamic feedback process: positive samples become
self-attenuating as the learner approaches them, whereas negative samples
become self-amplifying as the learner moves away. Finally, we analyze how
these local effects aggregate into finite stable equilibria, persistent
drift, or instability, and derive their specific manifestations in
Gaussian and categorical policies.''', r'''We first characterize how remoteness controls score contribution in
policy-output coordinates. We then show how reuse creates positive
self-attenuation and negative self-amplification, and derive the resulting
aggregate regimes and policy-family manifestations.''')

add("E13", r'''The proof, including the explicit Gaussian eigenvalue bounds and the full
categorical score-norm bound, is given in
Appendix~\ref{app:distance_strength}.

Proposition~\ref{prop:distance_strength} identifies the policy-family
split used below. In Gaussian policies, remoteness
corresponds to squared standardized distance and can produce unbounded
mean-score response. In categorical policies, remoteness is surprisal:
it can grow without bound as probability approaches zero, but the logit
score remains bounded.''', r'''(Proof: Appendix~\ref{app:distance_strength}.) Gaussian remoteness can
produce an unbounded mean-score response, whereas categorical surprisal can
diverge while the logit-score response remains bounded.''')

add("E14", r'''A proof is provided in Appendix~\ref{app:reuse_dynamics}. For the two
policy-output coordinates used in the main theory, the required convexity
condition holds globally: fixed-covariance Gaussian remoteness is
quadratic in the mean, and categorical negative log-likelihood is convex
in logits.''', r'''Both policy coordinates satisfy the theorem's convexity assumption
globally: fixed-covariance Gaussian remoteness is quadratic in the mean,
and categorical negative log-likelihood is convex in logits
(Appendix~\ref{app:reuse_dynamics}).''')

add("E15", r'''Theorem~\ref{thm:reuse_dynamics} separates a one-time distant update from
a historical-reuse feedback loop. Under a negative coefficient, the
learner moves in the direction that increases the historical action's
remoteness; after this movement, repeated reuse preserves or increases
the sample's score response. Combined with
Proposition~\ref{prop:distance_strength}, this yields unbounded
amplification for fixed-covariance Gaussian mean updates unless the
reused action has zero mean score, and bounded but persistent
amplification for categorical logits.''', r'''The theorem separates a one-time distant update from a historical-reuse
feedback loop. Combined with Proposition~\ref{prop:distance_strength}, it
yields unbounded amplification for fixed-covariance Gaussian mean updates
and bounded but persistent amplification for categorical logits.''')

add("E17", r'''The preceding results describe the repeated influence of an individual
historical action, but they do not imply that the aggregate policy
dynamics must diverge. Positive and negative historical samples exert
competing attractive and repulsive forces. We therefore ask whether
their aggregate effect admits a finite stable equilibrium or instead
produces persistent drift or instability.''', r'''The preceding single-sample reuse analysis tracks one fixed historical
action. We now aggregate positive and negative historical samples and
ask whether their competing attraction and repulsion produce a finite
stable equilibrium, persistent drift, or instability.''')

add("E20", r'''Theorem~\ref{thm:aggregate_equilibria} establishes that negative
dominance, \(p<q\), admits no finite stable equilibrium. It does not,
however, determine how this instability manifests or whether
policy-output score norms become unbounded. This depends on the policy
family.''', r'''Negative dominance admits no finite stable equilibrium, but its score
and support manifestations depend on the policy family. We therefore
analyze Gaussian and categorical policies separately.''')

add("E21-equation", r'''\begin{equation}
\begin{aligned}
    \left\|
        \nabla_z\log\pi_z(a)
    \right\|_2^2
    &=
    \left\|
        e_a-\pi_z
    \right\|_2^2 \\
    &\leq
    2
    \left(
        1-\pi_z(a)
    \right)^2
    \leq
    2.
\end{aligned}
\label{eq:categorical_bounded_score}
\end{equation}
Thus, categorical logits can become unbounded and their probabilities
can approach the support boundary, while every individual logit-score
norm remains bounded.''', r'''\begin{equation}
\|\nabla_z\log\pi_z(a)\|_2^2
=
\|e_a-\pi_z\|_2^2
\le 2(1-\pi_z(a))^2
\le 2.
\label{eq:categorical_bounded_score}
\end{equation}''')

add("E21-scope", r'''The proof is provided in
Appendix~\ref{app:policy_family_manifestations}.
Fixed covariance is sufficient for the Gaussian result; learned covariance is
not required. Neither policy-family conclusion alone implies
task-performance collapse or a NaN/Inf numerical failure.''', r'''The proof is provided in
Appendix~\ref{app:policy_family_manifestations}.
Fixed covariance is sufficient for the Gaussian result; learned covariance is
not required.''')

add("E25", r'''\subsection{Distance-Dependent Utility of Negative Feedback}
\label{subsec:distance_dependent_utility}

Bounded loop gain alone does not determine whether a remote update is useful.
Appendix~\ref{app:negative_update_utility} gives an empirical utility
definition and a sufficient shrinkage result, used here only as motivation;
we do not assume a universal monotone utility law in remoteness. When a
historical negative no longer supplies local correction information, the
design target is
\begin{equation}
    \omega(D) I(D)
    \longrightarrow 0
    \qquad
    \text{as }
    D\longrightarrow\infty .
    \label{eq:far_field_residual_vanishes}
\end{equation}
This is a conditional design principle, not a claim that every remote
sample is harmful. If far-field feedback remains aligned with the true
task-improving direction, tapering can remove useful signal.''', r'''\paragraph{Utility boundary.}
\label{subsec:distance_dependent_utility}
Remoteness is a control coordinate rather than a utility label. When
historical negative feedback no longer supplies local correction, DRPO
targets \(\omega(D)I(D)\to0\) as \(D\to\infty\); task direction remains
supplied by the underlying advantage or verifier signal. The corresponding
utility formulation is given in
Appendix~\ref{app:negative_update_utility}.''')

add("E32", r'''\paragraph{Measurement conventions.}
When the policy coordinate is controlled, source claims use policy-score
response. Neural diagnostics instead report the implemented actor-gradient
magnitude only to test whether the policy-level pattern appears in trainable
updates; such gradients also contain the network Jacobian.

\paragraph{Outcome taxonomy.}
We report task-performance collapse, policy-support or variance-boundary
events, and NaN/Inf numerical failure as distinct outcome classes. Their
protocol-specific definitions are in
Appendix~\ref{app:controlled_protocols}.''', r'''\paragraph{Conventions.}
Controlled source-isolation experiments report policy-score response,
whereas neural probes use implemented actor-gradient magnitude only to
test whether the same pattern appears in trainable updates.
Task-performance collapse, policy-support or variance-boundary events,
and NaN/Inf failures are reported as distinct outcome classes;
protocol-specific definitions are in
Appendix~\ref{app:controlled_protocols}.''')

before = text
for name, old, new in repls:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name}: expected old block exactly once, found {count}")
    text = text.replace(old, new, 1)
    if text.count(new) != 1:
        raise SystemExit(f"{name}: replacement count is {text.count(new)}")

for name, old, new in repls:
    if old in text:
        raise SystemExit(f"{name}: old block remains")
    if new not in text:
        raise SystemExit(f"{name}: new block missing")

anchors = [
    r'''\begin{proposition}[Quantitative Gaussian reuse rates]''',
    r'''These conclusions concern policy geometry
and stability; they do not imply that the resulting displacement
improves task utility.''',
    r'''This hierarchy ranks
guarantees, not task performance.''',
    r'''shared-network parameter
gradients additionally contain the network Jacobian and cross-sample
interference.''',
    r'''implemented parameter gradients also contain the network Jacobian.''',
]
for anchor in anchors:
    if anchor not in text:
        raise SystemExit(f"deferred anchor changed or absent: {anchor[:70]}")

intro_cites = r'''\citep{novati2019remember,fakoor2020p3o,peng2019advantage,
schulman2017proximal,fujimoto2019off,kumar2020conservative,
arnal2025asymmetric,leroux2025tapered}'''
if intro_cites not in text:
    raise SystemExit("E3 citation set changed")

TARGET.write_text(text, encoding="utf-8")

if BIB is not None:
    bib = BIB.read_text(encoding="utf-8")
    entry = r'''

@article{sana2026hpo,
  title={HPO: Hysteretic Policy Optimization for Stable and Efficient Training under Sparse-Reward Regime},
  author={Sana, Mohamed and Piovesan, Nicola and De Domenico, Antonio and Ayed, Fadhel and Zhang, Haozhe},
  journal={arXiv preprint arXiv:2605.30201},
  year={2026}
}
'''
    if "@article{sana2026hpo," in bib:
        raise SystemExit("sana2026hpo already exists")
    BIB.write_text(bib.rstrip() + entry, encoding="utf-8")

print(f"target={TARGET} applied={len(repls)}")
print("before_sha256=" + hashlib.sha256(before.encode()).hexdigest())
print("after_sha256=" + hashlib.sha256(text.encode()).hexdigest())
