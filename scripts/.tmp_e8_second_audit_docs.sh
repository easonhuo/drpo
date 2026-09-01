#!/usr/bin/env bash
set -euo pipefail

DOC="docs/experiments/E8_CONFIG_DRIVEN_SWEEP_REVIEW_FIX_PROTOCOL.md"
MARKER="## 2026-09-01 second adversarial audit closure"
if ! grep -Fq "$MARKER" "$DOC"; then
cat >> "$DOC" <<'EOF'

## 2026-09-01 second adversarial audit closure

A second end-to-end audit found additional accepted-but-unconsumable and provenance gaps. These rules are part of the same authority-boundary repair and supersede any weaker interpretation above.

1. A generic cold-start-family experiment ID must be genuinely new with respect to the execution identities known to this runner. In particular it may not reuse the P0, RHO, or DENSE experiment IDs. The three historical cold-start-family IDs remain protected by exact canonical config identity.
2. Redundant configuration fields may not silently diverge from the field that actually drives execution. While the current schema retains them, `sweep.tuning_seed` must equal the transfer Exp seed `sweep.task_transfer_seed_offset`; `sweep.countdown_sentinel_coefficients` must equal the configured Countdown `task_lambda` sequence; and `execution.expected_waves` must equal the nominal wave count derived from the expanded cell matrix and configured capacity.
3. Split sizes are configurable only within the data volume actually supplied by the locked source pipelines. The canonical P0 bank contains exactly 6000 rows per task, so `p0_train_rows + p0_validation_rows + p0_test_rows` must equal 6000 and the training partition must be non-empty. The canonical cold-start Countdown path forbids wrapper subsampling and supplies exactly 6000 training rows and 500 validation rows, so a config that requests different Countdown counts is not consumable by this implementation and must fail preflight rather than fail later in `prepare`.
4. An active task's configured evaluation prompt budget may not exceed its configured validation partition. Otherwise the legacy evaluator would silently slice to the available rows while preflight reported the larger requested budget. Such configurations must fail before launch. Countdown remains subject to its canonical equal greedy/pass-k budget interface.
5. The canonical trainer implements warmup as at least one optimizer step. Therefore a zero `training.warmup_ratio` cannot represent a true zero-warmup experiment through this implementation and must be rejected rather than reported as faithfully consumed.
6. Standalone config preflight must execute before full dependency installation, model download, GPU capacity checks, or other expensive setup on a fresh runtime. A minimal bootstrap environment may install only the lightweight dependencies needed to run the existing preflight; this does not authorize a second validator or duplicated config rules.
7. Formal guarded execution provenance must include every Python path that participates in config interpretation or launch gating, including `src/drpo/e8_experiment_config.py` and `scripts/preflight_e8_multitask_config.py`. Delivery-preflight provenance is not a substitute for a complete raw guarded-run source manifest.
8. Regression tests must cover all findings above. Passing the earlier 63-test suite is historical engineering evidence only and does not close this second-audit set.
EOF
fi

rm -f .github/workflows/e8-second-audit-docs-once.yml scripts/.tmp_e8_second_audit_docs.sh

git config user.name github-actions[bot]
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
git add -A
git diff --cached --check
git commit -m "docs: lock second E8 config authority audit"
git push origin HEAD:dev/e8-config-driven-sweep-01
