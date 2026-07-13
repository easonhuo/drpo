# Test commands

```bash
python -m py_compile \
  src/drpo/e7_ppo_w0_grid_pilot.py \
  src/drpo/e7_ppo_w0_bootstrap.py \
  src/drpo/e7_ppo_w0_aggregate.py \
  src/drpo/e7_ppo_w0_runtime_autotune.py \
  scripts/run_e7_ppo_w0_grid_pilot_auto.py \
  tests/test_e7_ppo_w0_grid_pilot.py \
  tests/test_e7_ppo_w0_runtime_autotune.py

bash -n scripts/run_e7_ppo_w0_grid_pilot_auto_one_click.sh

pytest -q \
  tests/test_e7_ppo_w0_grid_pilot.py \
  tests/test_e7_ppo_w0_runtime_autotune.py \
  tests/test_e7_canonical_ppo_stability.py \
  tests/test_runtime_resource_autotune.py

ruff check \
  src/drpo/e7_ppo_w0_grid_pilot.py \
  src/drpo/e7_ppo_w0_bootstrap.py \
  src/drpo/e7_ppo_w0_aggregate.py \
  src/drpo/e7_ppo_w0_runtime_autotune.py \
  scripts/run_e7_ppo_w0_grid_pilot_auto.py \
  tests/test_e7_ppo_w0_grid_pilot.py \
  tests/test_e7_ppo_w0_runtime_autotune.py
```

Only Python compilation, JSON parsing, and shell syntax were executed before the initial push. The remaining commands are required acceptance work and must not be reported as passed until actually executed.
