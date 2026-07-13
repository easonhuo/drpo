# Test commands: EXT-H-E7-W0-HIGHC-ACTOR-01

```bash
python -m pytest -q \
  tests/test_e7_w0_highc_actor.py \
  tests/test_e7_w0_geometry_diagnostics.py \
  tests/test_e7_ppo_w0_grid_pilot.py \
  tests/test_e7_canonical_ppo_injection.py

python -m compileall -q \
  src/drpo/e7_w0_highc_actor.py \
  src/drpo/e7_w0_highc_actor_bootstrap.py \
  src/drpo/e7_w0_highc_actor_aggregate.py \
  src/drpo/e7_w0_geometry_diagnostics.py \
  src/drpo/e7_w0_highc_runtime_autotune.py \
  scripts/run_e7_w0_highc_actor_auto.py

bash -n \
  scripts/run_e7_w0_highc_actor_auto_one_click.sh \
  scripts/run_e7_w0_highc_actor_resume_one_click.sh \
  scripts/run_e7_w0_highc_actor_liveness_one_click.sh

python -m ruff check \
  src/drpo/e7_w0_highc_actor.py \
  src/drpo/e7_w0_highc_actor_bootstrap.py \
  src/drpo/e7_w0_highc_actor_aggregate.py \
  src/drpo/e7_w0_geometry_diagnostics.py \
  src/drpo/e7_w0_highc_runtime_autotune.py \
  scripts/run_e7_w0_highc_actor_auto.py \
  tests/test_e7_w0_highc_actor.py \
  tests/test_e7_w0_geometry_diagnostics.py
```

A real-data liveness gate additionally requires the canonical D4RL contract and datasets on the server:

```bash
bash scripts/run_e7_w0_highc_actor_liveness_one_click.sh
```
