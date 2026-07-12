# GOV-RUNTIME-RESOURCE-AUTOTUNE-01 — Minimal E7/E8 implementation scope

**Approval:** user explicitly authorized implementation on 2026-07-12.  
**Implementation base:** resolve current `main` immediately before branch creation.  
**Scientific experiment impact:** none.  
**Default-policy impact:** none; all new behavior is explicit opt-in.

## Objective

Implement the smallest usable runtime-capacity layer for the current E7 CPU and
E8 GPU sweep runners. The implementation may choose only execution concurrency:

- E7: active subprocess count;
- E8: active GPU device slots, with one process per GPU.

It must account for host RAM and cgroup limits. E8 must also reject unavailable,
busy, or insufficient-VRAM devices. The selected runtime schedule must be written
to a machine-readable artifact and must not change the scientific matrix.

## Authorized files

- `src/drpo/runtime_resource_autotune.py`
- `src/drpo/runtime_resource_adapters.py`
- `scripts/probe_runtime_resources.py`
- `scripts/run_e7_canonical_exp_horizon_joint_auto.py`
- `scripts/run_e7_canonical_exp_horizon_joint_auto_one_click.sh`
- `scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py`
- `docs/runtime_resource_autotuning_v1.md`
- `docs/scopes/GOV-RUNTIME-RESOURCE-AUTOTUNE-01.md`
- `tests/test_runtime_resource_autotune.py`
- `tests/test_runtime_resource_adapters.py`

## Explicitly excluded

- modifying `docs/handoff.md` or `experiments/registry.yaml`;
- modifying any existing E7/E8 scientific config, method, seed, coefficient,
  batch, horizon, stopping rule, evaluation rule, or result status;
- modifying the existing fixed-60 E7 launcher or fixed-eight-GPU E8 runtime;
- modifying the canonical formal execution channel or any closed-stage protected
  file;
- changing the repository-wide default execution policy;
- RunSpec schema or validator integration;
- multi-node, Slurm, Kubernetes, weighted heterogeneous scheduling, same-GPU
  multi-process training, automatic batch changes, or arbitrary online scale-up;
- declaring throughput optimality, scientific convergence, or method ranking;
- using a resource probe as scientific evidence.

## Safety boundary

- Existing fixed launchers remain byte-for-byte unchanged.
- The E7 probe uses a dedicated non-formal seed namespace and an isolated probe
  directory; generated trainer payload is removed after measurement.
- E8 uses the original frozen config in calibration and worker subprocesses. A
  parent-only in-memory view changes only the required active GPU slot count.
- E8 V1 keeps one process per GPU.
- The implementation fails closed when no worker/device can satisfy configured
  CPU, RAM, cgroup, utilization, or VRAM gates.
- No current or interrupted E7/E8 work directory may be reused for a first auto
  launch without an explicit identity audit.

## Acceptance

- targeted unit tests cover cgroup memory limits, CPU-bound and memory-bound
  selection, process-tree RSS probes, GPU visibility/utilization/VRAM rejection,
  host-RAM-limited GPU slots, cache validation, and E8 scientific-config
  preservation;
- Python compilation, Ruff, full pytest, handoff authority verification, formal
  execution channel validation, governance inventory, and governance stage
  validation pass in CI;
- legacy fixed E7/E8 tests continue to pass;
- no real-hardware result is claimed unless the corresponding server/GPU run was
  actually performed.

## Rollback

1. Stop invoking the new `*_auto.py` entrypoints.
2. Continue using the unchanged fixed E7/E8 launchers.
3. Preserve `RUNTIME_SELECTION.json` and probe logs as runtime provenance.
4. Revert the files in the authorized list as one change if the feature must be
   removed.
5. Do not delete scientific history, completed outputs, failed-run evidence, or
   existing fixed-run artifacts.
