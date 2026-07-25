from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

mirror_old = r'''\textbf{Learning from Negative or Suboptimal Behavior.}
A broad line of policy-learning methods distinguishes behavior according
to its estimated quality. Advantage-weighted regression, AWAC, implicit
Q-learning, critic-regularized regression, and behavior-proximal
objectives fit actors more strongly toward actions estimated to be
valuable under a learned critic
\citep{peng2019advantage,nair2020awac,kostrikov2021offline,
wang2020critic,zhuang2023behavior}. Positive-filtered or filtering-based
variants represent a conservative endpoint, where selected negative or
low-quality updates are removed rather than reweighted. At the same
time, failed or suboptimal behavior can remain informative: it can
suppress competing bad modes, sharpen local decision boundaries, or
improve data efficiency
\citep{novati2019remember,fakoor2020p3o,arnal2025asymmetric}. These
findings support the view that negative feedback is not merely noise, but
also carries information whose usefulness depends on how it is used.

\textbf{Off-Policy, Stale, and Learner-Relative Updates.}
This issue becomes especially important under off-policy reuse, where
historical behavior remains fixed while the learner continues to change.
Off-policy methods commonly control behavior--learner mismatch through
importance weighting, clipping, trust regions, replay mechanisms, and
behavior regularization
\citep{schulman2017proximal,haarnoja2018soft,levine2020offline,
fujimoto2019off,kumar2020conservative}. Recent asymmetric or tapered
off-policy objectives further make updates depend on current-policy
probability or surprisal
\citep{arnal2025asymmetric,leroux2025tapered}. These works establish
that mismatch and rarity matter; our distinction concerns how
learner-relative remoteness is created and amplified by repeated
negative reuse. This actor-side view is complementary to conservative
offline RL, which primarily addresses value extrapolation and
policy-support mismatch.

\textbf{Tapered Updates and Optimistic Reweighting.}
TOPR uses an asymmetric tapered importance ratio on negative off-policy
updates and provides its own stability analysis
\citep{leroux2025tapered}. Optimistic or best-case distributional
optimization also has a formal precedent in likelihood approximation
and policy optimization
\citep{nguyen2019optimistic,song2020optimistic}. Our construction is
related but distinct: it reweights a finite measure of negative-update
mass according to learner-relative remoteness and may reduce that
measure's total mass. It is therefore an analogue of optimistic
divergence-ball reweighting, not a standard ambiguity set over normalized
probability distributions.'''

main_old = r'''\textbf{Learning from Negative or Suboptimal Behavior.}
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
remoteness is created and amplified by repeated negative reuse.'''

if text.count(mirror_old) != 1:
    raise SystemExit(f"{path}: expected mirror Related Work block exactly once, found {text.count(mirror_old)}")
if main_old in text:
    raise SystemExit(f"{path}: main Related Work block already present")
path.write_text(text.replace(mirror_old, main_old, 1), encoding="utf-8")
print(f"normalized Related Work in {path}")
