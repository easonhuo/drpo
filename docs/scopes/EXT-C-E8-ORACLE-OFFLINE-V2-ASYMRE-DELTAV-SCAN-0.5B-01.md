# EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-SCAN-0.5B-01

## Development status

- Lifecycle: code-first development pilot; deferred registration with closure required.
- Result status: `pilot / not_run`.
- Scientific role: Countdown external-validity coefficient-response diagnostic.
- This pilot cannot establish convergence, steady state, statistical significance,
  controlled mechanism identification, OOD generalization, or a universal method ranking.

## Authorized scientific delta

The only scientific training change is the AsymRE additive baseline offset applied to
fixed branch-balanced signed rewards:

```text
A = R - delta_v
R_positive = +1
R_negative = -1
positive coefficient = 1 - delta_v
negative-repulsion coefficient = 1 + delta_v
```

The controlled E8 objective assigns equal total mass to the positive and negative
branches, so its empirical prompt-level baseline is exactly zero. No value network,
learned baseline, remoteness taper, bank rebuild, optimizer change, or checkpoint-policy
change is authorized.

## Frozen 16-cell matrix

```text
delta_v = -1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.1
seed offsets = 4000, 5000
8 points x 2 paired seeds = 16 cells
```

The fixed horizon is 1200 optimizer steps with no early stopping. GPU `0-7` run two
cells each, so the full pilot occupies one 16-slot wave after the required smoke gate.

## Held-out evaluation semantics

The file named `val.jsonl` is a structurally disjoint held-out evaluation split. It is
separate from the offline training bank in both canonical structure families and
`(numbers, target)` problem keys, and its rows never enter the training loss.

The paper-facing response curve reports every declared `delta_v` point using the fixed
late window (`800, 900, 1000, 1100, 1200`) and step-1200 terminal metrics. A
validation-selected `best_pass8_adapter` may be retained only as a supplementary local
diagnostic/recovery asset; it must not replace the late-window or terminal curve.

The separately materialized `test.jsonl` is not accessed in this pilot. That file-access
fact does not mean that held-out evaluation is absent: `val.jsonl` carries the
structurally held-out evaluation role for this full-grid response curve.

## Runtime and automatic delivery

Execution must use the canonical E8 RunSpec/lane path. The completed text-first review
package is automatically delivered to:

```text
repository: easonhuo/drpo-results
branch: ingest/e8
export profile: manifest_text_v1
```

Model, LoRA checkpoint, optimizer, and other model-like files remain excluded. A
delivery failure must not retrigger training; after credentials are repaired, only the
canonical manual uploader may be retried.

## Reporting separation

Report separately:

1. task-performance degradation or improvement;
2. valid-expression/structure or support-proxy events;
3. NaN/Inf numerical failure.

Smoke proves liveness only. Fixed 1200 steps are not convergence or steady state.

## Stop conditions

Stop and request a new protocol before changing the `delta_v` list, seeds, bank,
initialization, optimizer, learning rate, horizon, branch normalization, evaluation
window, runtime slots, delivery destination, or artifact exclusions.
