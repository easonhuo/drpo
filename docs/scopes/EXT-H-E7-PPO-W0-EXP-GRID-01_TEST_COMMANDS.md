# Test commands

```bash
python -m py_compile \
  src/drpo/e7_ppo_w0_grid_pilot.py \
  src/drpo/e7_ppo_w0_bootstrap.py \
  src/drpo/e7_ppo_w0_aggregate.py \
  src/drpo/e7_ppo_w0_runtime_autotune.py \
  scripts/run_e7_ppo_w0_grid_pilot_auto.py \
  tests/test_e7_ppo_w0_grid_pilot.py \
  tests/test_e7_ppo_w0_runtime_autotune.py \
  tests/test_e7_ppo_w0_runspecs.py \
  tests/test_e7_ppo_w0_failure_audit.py

bash -n \
  scripts/run_e7_ppo_w0_grid_pilot_auto_one_click.sh \
  scripts/run_e7_ppo_w0_grid_pilot_resume_one_click.sh

pytest -q \
  tests/test_e7_ppo_w0_grid_pilot.py \
  tests/test_e7_ppo_w0_runtime_autotune.py \
  tests/test_e7_ppo_w0_runspecs.py \
  tests/test_e7_ppo_w0_failure_audit.py \
  tests/test_e7_canonical_ppo_stability.py \
  tests/test_runtime_resource_autotune.py \
  tests/test_runspec_recovery.py

ruff check \
  src/drpo/e7_ppo_w0_grid_pilot.py \
  src/drpo/e7_ppo_w0_bootstrap.py \
  src/drpo/e7_ppo_w0_aggregate.py \
  src/drpo/e7_ppo_w0_runtime_autotune.py \
  scripts/run_e7_ppo_w0_grid_pilot_auto.py \
  tests/test_e7_ppo_w0_grid_pilot.py \
  tests/test_e7_ppo_w0_runtime_autotune.py \
  tests/test_e7_ppo_w0_runspecs.py \
  tests/test_e7_ppo_w0_failure_audit.py
```

GitHub PR Gate Log is the authoritative repository-wide acceptance channel. Real D4RL liveness remains a separate server-only gate and must not be inferred from CI.
