# Hopper p/q, advantage refresh and raw IS

Implementation: runnable profile on `dev/e7-rebuttal-pq-is`.
Scientific result status: `not_run`; no 24-run scientific result has been produced.
Experiment: `EXT-H-E7-REBUTTAL-PQ-IS-01`. This is a tracked scientific protocol
and operator roadmap, not a replacement for `docs/handoff.md`.

## Scientific protocol

Use only `hopper-medium-replay-v2`, SHA-256
`e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b`.
The complete matrix is frozen/refreshed advantages × positive_only/signed/drpo/is
× paired actor seeds 200/201/202: 24 branches. Seeds 204–207 are not used.
The shared episode split and MC critic use seed 100. Split episodes 80/10/10;
fit observation and return normalizers only on train and freeze both.
Pretrain the existing 256×256 tanh value network for 100k updates (AdamW,
lr 3e-4, weight decay 1e-4, batch 1024, critic gradient clip **10**), selecting
the best validation checkpoint after the complete budget. Quality/convergence
statistics are report-only. Preserve both selected and final checkpoints.

For each actor seed, clone one random squashed-Gaussian initialization into all
eight branches. No actor warmstart, including Positive-only. Each branch uses
1M updates, AdamW lr 3e-4, weight decay 1e-4, batch 1024, global diagonal
log_std clamped to [-5,2], and actor gradient clip **100**. Clipping is common
optimizer processing; raw IS ratios themselves are never clipped or normalized.
Global gradient norms are reduced in float64 so squaring a large but finite
float32 score gradient does not itself create a numerical failure. Parameters
and stored parameter gradients remain float32; genuine non-finite losses or
gradients are recorded, not replaced by fallback weights.

Define b=1-terminal-timeout (logical OR), delta=r+0.99*b*V(next)-V(state), with
V restored to reward units. Freeze m_A and s_A=max(std(delta_0),1e-8) once on
the full actor train pool; A=(delta-m_A)/s_A. Frozen branches cache A_0.
Refreshed branches compute A with the pre-update critic, detach it, update the
actor, then update the critic on the same batch with a stopped one-step TD
target, expectile 0.5, and the frozen return-target normalizer. Use a fresh
critic optimizer after selecting the pretrained weights. The V-only target
does not depend on sampled actor actions: within a seed, refreshed critic paths
are controlled to be identical across methods. This signal is not oracle A^pi.

Actor loss is -mean(stop_gradient(c)*log_pi_action), always over the full batch.
Coefficients are A+; A; A+ - omega*A-; and exp(log_pi_action-logged_log_mu)*A.
Use the same clipped safe action for atanh and the tanh Jacobian. Preserve all
rows, including boundary actions; record clipping and boundary influence.
The per-row logged denominator is the collection density, not a fitted BC
model. This corrects the action ratio only, not state occupancy or trajectories.
Compute raw ratios in float64; no PPO clip, ratio floor, self-normalization,
fallback weight, or log-ratio clamp. IS ratios and DRPO weights are detached.

For DRPO, D=mean(((atanh(a_safe)-mu)/sigma)^2), and
omega=exp[-0.122667*(sqrt(D)/2)^2]=exp[-0.03066675*D]. Scale is 1,
threshold 0, reference distance 2. This is the explicit new-profile mapping
using the manuscript Hopper-replay coefficient and canonical reference; it is
not a claim of exact reproduction of a historical selected run. Exp has no
nonzero tail floor. These constants are not retuned from this experiment.

## Measurements and interpretation

Fix up to 16384 uniformly sampled train transitions across all branches. Save
their original IDs/hash. Observe step 0, every 5000 updates and the endpoint.
p=mean(A+), q=mean(A-); effective p/q use the actual coefficients. Record
unstandardized TD mass separately. Centering forces initial full-pool p≈q;
frozen raw p/q remain constant. Neither q>p nor a large gradient alone proves
future task collapse. For p=0 serialize a null ratio plus an explicit status.

Save current/initial/previous advantages, bidirectional strict sign flips with
the complete fixed population as denominator, and zero transitions separately.
Near/far are current negative-distance quartiles. Full-parameter positive,
negative, near-negative and far-negative aggregate forces all divide by the
same population N. Report raw and effective norms, allowing vector cancellation.
Near/far probes do not change training sampling.

At steps 0/250k/500k/750k/1M, use 16 independent diagnostic batches of 1024.
At each identical actor/critic checkpoint, compare Signed/DRPO/IS estimators
on the same batches without optimizer updates. Report sample gradient-variance
trace (ddof=1), mean gradient norm, mean batch gradient norm, ESS/N, top-1%
weight mass and boundary influence. Diagnostics use independent RNGs and
autograd.grad, never advance training RNG or change optimizer/parameters/.grad.

Evaluate deterministic policy.action_mean in Gymnasium Hopper-v4, every 50k
updates with 10 paired episodes and at step 0; use 20 at the endpoint. Normalize
with D4RL-v2 references -20.272305 and 3234.3. Primary performance is the equal
mean of steps 750k/800k/850k/900k/950k/1M. Also report terminal, best, drop,
late slope and cross-training-seed sample SD (ddof=1); episode SD is separate.
This is finite-budget external training evidence, not convergence or controlled
causal identification. Report task degradation, support/variance boundaries,
NaN/Inf, incomplete budgets and runtime failures separately. Task-collapse
status is not adjudicated because this cold-start protocol has no locked binary
threshold. Do not import the Q2 warmstart-based collapse rule.

## Local setup and one-command launch

From your existing clone, fetch and check out `dev/e7-rebuttal-pq-is`.
Use a Python >=3.10 environment with a CUDA-compatible PyTorch installation.

```bash
git fetch origin dev/e7-rebuttal-pq-is
git switch --track origin/dev/e7-rebuttal-pq-is
python -m pip install -e '.[dev]'
python -m pip install 'gymnasium[mujoco]'
```

Set `DRPO_HOPPER_MEDIUM_REPLAY` to the actual existing HDF5 path, or pass
`--dataset-path` directly. The launcher also retains the existing standard
server dataset locations. No `mujoco-py` or D4RL import is required.

```bash
python scripts/run_e7_hopper_q2.py --experiment-profile rebuttal_pq_is --plan-only
python scripts/run_e7_hopper_q2.py --experiment-profile rebuttal_pq_is --smoke --device cpu --max-workers 2
python scripts/run_e7_hopper_q2.py --experiment-profile rebuttal_pq_is --device cuda --gpu-ids 0,1,2,3,4,5,6,7 --max-workers 8
```

The smoke command is a separate 8-cell, 10-update engineering check with its own
output directory, never part of the 24 scientific runs. `--plan-only` lists the
matrix/commands and performs no pretraining. The normal run defaults to GPUs
0–7 and eight workers; `--device cpu` permits a CPU run with the same science.
Preparation runs once, then at most one subprocess per GPU slot. A fresh worker
process owns each branch; no fork of an initialized CUDA context is used.

The launcher prints the absolute run directory and supervises the complete
pipeline through the existing hardened guard. The exact clean dev-branch HEAD,
config hashes, dataset SHA, environment and matrix are saved in the manifest.
The default checkpoint storage label is `persistent_local`, for a persistent
training machine. In a disposable container pass
`--large-file-persistence ephemeral` (or `unknown` when persistence is unknown);
the run manifest and checkpoint indexes retain this operational distinction.
Using a clean reviewed development commit does not require a merge into main.
Registry/READY/channel activation is not a launch prerequisite.

## Recovery and outputs

Each branch atomically saves latest actor/critic/optimizer/RNG state every 5000
updates. Preparation saves critic and actor initializations once. Checkpoints
stay in the persistent run directory; the archive contains their indexes.
After interruption use a **new** guarded output directory, referring to the
previous persistent run root:

```bash
python scripts/run_e7_hopper_q2.py --experiment-profile rebuttal_pq_is --resume-from "$PREVIOUS_RUN_ROOT" --device cuda --gpu-ids 0,1,2,3,4,5,6,7 --max-workers 8
```

`PREVIOUS_RUN_ROOT` is the absolute directory printed by the original launcher.
Resume checks source/config/data identity, reuses preparation and completed
cells, and continues only incomplete matching tuples. It never changes the
original run directory. Reuse the same `--smoke` flag when resuming a smoke.

Read `summary/MATRIX.csv` first, including failures. Then inspect
`METHOD_SUMMARY.csv`, `PAIRED_COMPARISONS.csv`, `PQ_TRAJECTORIES.csv`,
`ADVANTAGE_REFRESH.csv`, `GRADIENT_GEOMETRY.csv`, `IS_WEIGHT_STATS.csv`,
`GRADIENT_VARIANCE.csv`, `PER_RUN_SUMMARY.csv` and `TERMINAL_AUDIT.json`.
Five SVG figures are under `figures/`; each cell has training metrics,
diagnostics, rollouts, summary, status/failure and checkpoint index.
Root `terminal_audit.json` and `RUN_COMPLETE.json` describe actual matrix
completion, never claim steady state. Failed rows are not scored as zero and
best checkpoints do not replace missing endpoints or late-window evaluations.

## Roadmap and verification

1. Implement this isolated profile; retain the old Q2 default and artifacts.
2. Run mathematical/data/gradient/RNG/resume and legacy compatibility tests:
   `PYTHONPATH=src python -m pytest -q tests/test_e7_hopper_q2.py tests/test_e7_q2_acceptance_pipeline.py`.
3. Exercise a short real-HDF5 CPU/MuJoCo smoke, plan-only and interrupted resume.
   These are engineering checks only; record hardware limits honestly.
4. On the eight-GPU training machine, run the smoke once with the intended GPU
   IDs, then launch the frozen full command above. Hardware scheduling changes
   do not change seeds, methods, horizons or coefficients.
5. Review all 24 statuses, numerical/boundary/task outcomes and late-window
   coverage; package using the existing guard. Only then interpret the measured
   evidence. This implementation does not modify the manuscript or rebuttal.
