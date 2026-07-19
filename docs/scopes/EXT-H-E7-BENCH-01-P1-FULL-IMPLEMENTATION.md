# EXT-H-E7-BENCH-01 P1 full-run implementation

This stacked development branch starts from the reviewed P1 matrix implementation at `0d47ab23492138b3b4469567b2049d7582157e7c`. It implements the full 198-branch development run, task-balanced aggregation, and terminal audit after the real three-arm Hopper liveness passed. It remains a development hyperparameter screen, not formal scientific evidence or a method-ranking result.

The branch must not change the frozen nine-task matrix, seeds `200,201`, one-million-step horizon, GAE lambda `0.95`, taper lambda `1`, `tau=0`, or common `c` grid. Held-out seeds `204--207` remain untouched. Task-performance collapse, support/variance boundary, rollout failure, and NaN/Inf numerical failure remain separate.

The implementation reuses existing Python files only. The existing one-click script is a lower-level entrypoint and must not self-authorize P1. The only canonical full-run path is the lane-scoped RunSpec executor: a reviewed template is promoted to `runspecs/ready/`, `run_lane.py` claims it, `run_claimed_runspec.py` executes and packages it, and `delivery.auto=true` uploads the text-first result package to `easonhuo/drpo-results` branch `ingest/e7`.

The first full-run RunSpec identity `E7_BENCH_JOINT_GAE_P1_FULL_20260719_01` was blocked at claim-time validation before training because its delivery limits exceeded the results-repository V1 caps. That single-use run ID must not be reused. The replacement RunSpec uses a new run ID and the V1 limits `max_total_size_mb: 30` and `max_file_size_mb: 10`.

The remote review package keeps all top-level identities, branch configurations, branch manifests, completion/failure records, terminal geometry summaries, trainer summaries, aggregate JSON/CSV outputs, and terminal audit. High-volume per-step `geometry_diagnostics.jsonl`, stdout/stderr streams, and trainer logs remain in the server-local run directory but are not copied into the V1 results-repository package. This changes only evidence transport, not training, aggregation, scientific variables, or local result retention.

The checked-in replacement RunSpec remains under `runspecs/templates/` in this implementation PR, so no server lane can claim or launch it. Promotion to READY, execution, merge, and scientific registration closure each require separate explicit approval. Upload failure must produce a partial result-handoff state without deleting the local artifact; no checkpoint, optimizer state, model weight, or adapter may be delivered.
