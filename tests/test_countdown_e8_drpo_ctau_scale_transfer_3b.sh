#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 - <<'PY'
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import torch
import yaml

from drpo import countdown_e8_drpo_ctau_scale_transfer_3b as transfer

root = Path.cwd()
config_path = root / "configs/countdown_e8_drpo_ctau_scale_transfer_3b.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

assert transfer.parameter_points(config) == (
    ("A", 1.609437912, 0.125),
    ("B", 1.897119985, 0.25),
    ("C", 2.995732274, 0.125),
    ("D", 4.605170186, 0.75),
)
cells = transfer.build_cells(config)
assert len(cells) == 8
assert len({cell.name for cell in cells}) == 8
assert {cell.seed_offset for cell in cells} == {4000, 5000}
assert {cell.label for cell in cells} == {"A", "B", "C", "D"}
assert config["execution"]["default_gpus"] == [0, 1, 2, 3]
assert config["execution"]["parallel_cells_per_gpu"] == 1
assert config["execution"]["expected_full_waves"] == 2

transfer.set_active_tau(0.25)
seq_lp = torch.tensor([-0.5, -2.5])
actual = transfer.continuous_exp_weights(seq_lp, alpha=1.0, c=2.0)
expected = torch.tensor([1.0, float(torch.exp(torch.tensor(-2.0)))])
assert torch.allclose(actual, expected)
assert actual.requires_grad is False

bad = copy.deepcopy(config)
bad["execution"]["parallel_cells_per_gpu"] = 2
try:
    transfer.validate_grid_config(bad)
except ValueError as error:
    assert "one slot per GPU" in str(error)
else:
    raise AssertionError("two-slot drift was not rejected")

bad = copy.deepcopy(config)
bad["sweep"]["parameter_points"][0]["tau"] = 0.0
try:
    transfer.validate_grid_config(bad)
except ValueError as error:
    assert "points changed" in str(error)
else:
    raise AssertionError("c/tau drift was not rejected")

with tempfile.TemporaryDirectory() as tmp:
    model = Path(tmp) / "Qwen2.5-3B-Instruct"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            }
        ),
        encoding="utf-8",
    )
    identity = transfer._validate_model_identity(model)
    assert identity["expected"] == "Qwen2.5-3B-Instruct"
    assert identity["model_type"] == "qwen2"
PY

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${TMP}/Qwen2.5-3B-Instruct"
printf '{}\n' > "${TMP}/bank.jsonl"
printf '{}\n' > "${TMP}/val.jsonl"

python3 src/drpo/countdown_e8_drpo_ctau_scale_transfer_3b.py _core-plan \
  --model_path "${TMP}/Qwen2.5-3B-Instruct" \
  --bank "${TMP}/bank.jsonl" \
  --val "${TMP}/val.jsonl" \
  --base_config configs/countdown_e8_base_rl_replay_3b.yaml \
  --grid_config configs/countdown_e8_drpo_ctau_scale_transfer_3b.yaml \
  --work_dir "${TMP}/work"

python3 - "${TMP}/work/SWEEP_PLAN.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert plan["experiment_id"] == "EXT-C-E8-DRPO-CTAU-SCALE-TRANSFER-3B-01"
assert plan["parameter_points"] == 4
assert plan["cell_count"] == 8
assert plan["resource_contract"] == {
    "gpu_count": 4,
    "runtime_slots_per_gpu": 1,
    "total_runtime_slots": 4,
    "expected_full_waves": 2,
}
assert {row["label"] for row in plan["cells"]} == {"A", "B", "C", "D"}
assert {float(row["tau"]) for row in plan["cells"]} == {0.125, 0.25, 0.75}
assert {int(row["seed_offset"]) for row in plan["cells"]} == {4000, 5000}
PY

echo "PASS: frozen Qwen2.5-3B DRPO c/tau scale-transfer profile"
