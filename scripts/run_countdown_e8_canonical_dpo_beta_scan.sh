#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODE="${1:-preflight}"
MODEL_PATH="${E8_DPO_MODEL_PATH:-/root/models/Qwen2.5-0.5B-Instruct}"
WORK_DIR="${E8_DPO_WORK_DIR:-/root/experiment_output/e8_canonical_dpo_beta_scan_001}"
BANK="${E8_DPO_BANK:-/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl}"
VAL="${E8_DPO_VAL:-/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl}"
BASE_CONFIG="${E8_DPO_BASE_CONFIG:-configs/countdown_e8_base_rl_replay_0p5b.yaml}"
GRID_CONFIG="configs/countdown_e8_oracle_offline_v2_canonical_dpo_beta_scan_0p5b.yaml"
EXPECTED_COMMIT="${E8_DPO_EXPECTED_COMMIT:-}"
LAUNCHER="scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
if [[ -n "${EXPECTED_COMMIT}" ]] && [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "checkout does not match E8_DPO_EXPECTED_COMMIT" >&2
  exit 2
fi
for required in "${MODEL_PATH}" "${BANK}" "${VAL}" "${BASE_CONFIG}" "${GRID_CONFIG}" "${LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 2
  fi
done

preflight() {
  PYTHONPATH=src python3 - "${GRID_CONFIG}" <<'PY'
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from drpo import countdown_e8_alpha1_highc_scan_common as scan

path = Path(sys.argv[1])
launcher_path = Path("scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py")
spec = importlib.util.spec_from_file_location("_e8_dpo_profile_preflight", launcher_path)
if spec is None or spec.loader is None:
    raise SystemExit("unable to import E8 profile launcher")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher._install_canonical_dpo_profile(path)
config = scan.load_yaml(path)
scan.activate_for_grid_config(path)
scan.validate_grid_config(config)
cells = scan.build_cells(config)
expected_betas = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
assert scan.EXPERIMENT_ID == launcher.CANONICAL_DPO_EXPERIMENT_ID
assert tuple(point[2] for point in scan.parameter_points(config)) == expected_betas
assert len(cells) == 16
assert len({cell.name for cell in cells}) == 16
assert {cell.seed_offset for cell in cells} == {4000, 5000}
assert {cell.method for cell in cells} == {"canonical_dpo"}
assert config["model"]["reference_trainable"] is False
assert config["reference_policy"]["role"] == "exact_frozen_initial_policy"
assert config["preference_data"]["pair_aggregation"] == "mean_within_prompt_then_mean_prompts"
assert config["training"]["steps"] == 1200
assert config["evaluation"]["separate_test_split_access"] is False
print("canonical DPO fixed-profile preflight: PASS")
PY
}

run_smoke() {
  PYTHONPATH=src python3 "${LAUNCHER}" smoke \
    --model_path "${MODEL_PATH}" \
    --work_dir "${WORK_DIR}" \
    --bank "${BANK}" \
    --val "${VAL}" \
    --base_config "${BASE_CONFIG}" \
    --grid_config "${GRID_CONFIG}" \
    --gpus 0,1 \
    --max-devices 2 \
    --allow-dev-unregistered
}

verify_checkpoint_reload() {
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 - "${MODEL_PATH}" "${WORK_DIR}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = Path(sys.argv[1]).resolve()
work_dir = Path(sys.argv[2]).resolve()
smoke_gate_path = work_dir / "SMOKE_GATE.json"
if not smoke_gate_path.is_file():
    raise SystemExit("SMOKE_GATE.json is missing")
smoke_gate = json.loads(smoke_gate_path.read_text(encoding="utf-8"))
if smoke_gate.get("status") != "PASS":
    raise SystemExit("smoke gate did not pass")
summary_path = Path(smoke_gate["summary"])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if float(summary.get("dpo_beta", -1.0)) != 0.1:
    raise SystemExit("liveness checkpoint is not the frozen beta=0.1 cell")
if summary.get("reference_policy", {}).get("trainable") is not False:
    raise SystemExit("summary does not identify a frozen reference policy")
tolerance = float(summary["initial_pair_margin_max_abs_tolerance"])
if float(summary["initial_pair_margin_max_abs"]) > tolerance:
    raise SystemExit("initial policy/reference pair margin exceeded tolerance")
checkpoint = summary_path.parent / "terminal_adapter"
if not checkpoint.is_dir():
    raise SystemExit(f"terminal adapter checkpoint is missing: {checkpoint}")
policy_path = checkpoint
if not (policy_path / "adapter_config.json").is_file():
    policy_path = checkpoint / "default"
reference_path = checkpoint / "reference"
if not (policy_path / "adapter_config.json").is_file():
    raise SystemExit("saved policy adapter_config.json is missing")
if not (reference_path / "adapter_config.json").is_file():
    raise SystemExit("saved reference adapter_config.json is missing")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(
    base,
    policy_path,
    adapter_name="default",
    is_trainable=False,
)
model.load_adapter(reference_path, adapter_name="reference", is_trainable=False)
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("1 + 1", return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}
adapter_checks = {}
model.eval()
with torch.no_grad():
    for adapter_name in ("default", "reference"):
        model.set_adapter(adapter_name)
        logits = model(**inputs, use_cache=False).logits
        finite = bool(torch.isfinite(logits).all())
        norm = float(logits.float().norm().item())
        if not finite or not math.isfinite(norm) or norm <= 0.0:
            raise SystemExit(f"non-finite reload forward for adapter {adapter_name}")
        adapter_checks[adapter_name] = {
            "finite_logits": finite,
            "logit_norm": norm,
        }

payload = {
    "schema_version": 1,
    "status": "PASS",
    "scientific_evidence": False,
    "source_commit": summary["run_identity"]["source"]["commit"],
    "checkpoint": str(checkpoint),
    "policy_adapter_path": str(policy_path),
    "reference_adapter_path": str(reference_path),
    "adapter_checks": adapter_checks,
}
path = work_dir / "CHECKPOINT_RELOAD_GATE.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

require_liveness_gates() {
  PYTHONPATH=src python3 - "${WORK_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("SMOKE_GATE.json", "CHECKPOINT_RELOAD_GATE.json"):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"required liveness gate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise SystemExit(f"required liveness gate failed: {path}")
print("liveness and checkpoint reload gates: PASS")
PY
}

run_full_matrix() {
  if [[ "${E8_DPO_FORMAL_RUN_AUTHORIZED:-0}" != "1" ]]; then
    echo "formal matrix launch requires E8_DPO_FORMAL_RUN_AUTHORIZED=1 after registration approval" >&2
    exit 2
  fi
  require_liveness_gates
  PYTHONPATH=src python3 "${LAUNCHER}" run \
    --model_path "${MODEL_PATH}" \
    --work_dir "${WORK_DIR}" \
    --bank "${BANK}" \
    --val "${VAL}" \
    --base_config "${BASE_CONFIG}" \
    --grid_config "${GRID_CONFIG}" \
    --gpus 0,1 \
    --max-devices 2 \
    --allow-dev-unregistered
}

preflight
case "${MODE}" in
  preflight)
    ;;
  liveness)
    run_smoke
    verify_checkpoint_reload
    ;;
  run)
    run_full_matrix
    ;;
  full)
    run_smoke
    verify_checkpoint_reload
    run_full_matrix
    ;;
  *)
    echo "usage: $0 {preflight|liveness|run|full}" >&2
    exit 2
    ;;
esac
