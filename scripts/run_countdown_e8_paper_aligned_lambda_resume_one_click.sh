#!/usr/bin/env bash
set -euo pipefail

# Resume is identity checked by the runtime. Completed cells are reused only when
# their code/config/calibration/bank/model identity exactly matches the current run.
exec bash "$(git rev-parse --show-toplevel)/scripts/run_countdown_e8_paper_aligned_lambda_one_click.sh"
