#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODE="${1:-preflight}"
MODEL_PATH="${E8_WARM_DPO_MODEL_PATH:-/root/models/Qwen2.5-0.5B-Instruct}"
WORK_ROOT="${E8_WARM_DPO_WORK_ROOT:-/root/experiment_output/e8_shared_sft_warmstarted_dpo_001}"
SFT_DIR="${E8_WARM_DPO_SFT_DIR:-${WORK_ROOT}/shared_sft}"
SHARED_ADAPTER="${E8_WARM_DPO_SHARED_ADAPTER:-${SFT_DIR}/epoch_1_adapter}"
DPO_WORK_DIR="${E8_WARM_DPO_DPO_WORK_DIR:-${WORK_ROOT}/dpo}"
BANK="${E8_WARM_DPO_BANK:-/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl}"
VAL="${E8_WARM_DPO_VAL:-/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl}"
BASE_CONFIG="${E8_WARM_DPO_BASE_CONFIG:-configs/countdown_e8_base_rl_replay_0p5b.yaml}"
GRID_CONFIG="configs/countdown_e8_oracle_offline_v2_shared_sft_canonical_dpo_beta_scan_0p5b.yaml"
PIPELINE="scripts/v2_sft.py"
EXPECTED_COMMIT="${E8_WARM_DPO_EXPECTED_COMMIT:-}"
SFT_GPU="${E8_WARM_DPO_SFT_GPU:-0}"
DPO_GPUS="${E8_WARM_DPO_GPUS:-0,1}"
RUNTIME_SLOTS_PER_GPU="${E8_WARM_DPO_RUNTIME_SLOTS_PER_GPU:-2}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
if [[ -n "${EXPECTED_COMMIT}" ]] && [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "checkout does not match E8_WARM_DPO_EXPECTED_COMMIT" >&2
  exit 2
fi
for required in "${MODEL_PATH}" "${BANK}" "${VAL}" "${BASE_CONFIG}" "${GRID_CONFIG}" "${PIPELINE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 2
  fi
done
if [[ "${DPO_GPUS}" != "0,1" ]]; then
  echo "frozen warm-started DPO profile requires E8_WARM_DPO_GPUS=0,1" >&2
  exit 2
fi
if [[ "${RUNTIME_SLOTS_PER_GPU}" != "2" ]]; then
  echo "frozen warm-started DPO profile requires two runtime slots per GPU" >&2
  exit 2
fi

preflight() {
  PYTHONPATH=src python3 "${PIPELINE}" validate --grid-config "${GRID_CONFIG}"
  python3 -m py_compile "${PIPELINE}"
  bash -n "$0"
  echo "shared-SFT warm-started DPO preflight: PASS"
}

run_sft() {
  mkdir -p "${WORK_ROOT}"
  PYTHONPATH=src python3 "${PIPELINE}" run-sft \
    --model-path "${MODEL_PATH}" \
    --train-data "${BANK}" \
    --val-data "${VAL}" \
    --output-dir "${SFT_DIR}" \
    --cuda-visible-devices "${SFT_GPU}"
}

require_sft_gate() {
  PYTHONPATH=src python3 - "${SFT_DIR}" "${SHARED_ADAPTER}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sft_dir = Path(sys.argv[1]).resolve()
adapter = Path(sys.argv[2]).resolve()
gate_path = sft_dir / "SFT_WARMSTART_GATE.json"
if not gate_path.is_file():
    raise SystemExit(f"missing SFT gate: {gate_path}")
gate = json.loads(gate_path.read_text(encoding="utf-8"))
if gate.get("status") != "PASS":
    raise SystemExit("shared SFT gate did not pass")
if int(gate.get("exact_sft_epochs", -1)) != 1:
    raise SystemExit("shared SFT gate is not the frozen one-epoch checkpoint")
if gate.get("adaptive_metric_stopping_used") is not False:
    raise SystemExit("shared SFT gate used adaptive metric stopping")
record = gate.get("shared_adapter", {})
if Path(record.get("path", "")).resolve() != adapter:
    raise SystemExit("shared SFT adapter path does not match the gate")
config = adapter / "adapter_config.json"
weight = adapter / str(record.get("weight_file", ""))
for path in (config, weight):
    if not path.is_file():
        raise SystemExit(f"missing shared SFT adapter file: {path}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha256(config) != record.get("adapter_config_sha256"):
    raise SystemExit("shared SFT adapter config hash changed")
if sha256(weight) != record.get("weight_sha256"):
    raise SystemExit("shared SFT adapter weight hash changed")
print("identity-bound shared SFT adapter gate: PASS")
PY
}

run_plan() {
  require_sft_gate
  PYTHONPATH=src python3 "${PIPELINE}" dpo-runtime \
    --shared-adapter "${SHARED_ADAPTER}" \
    --grid-config "${GRID_CONFIG}" -- \
    plan \
    --model_path "${MODEL_PATH}" \
    --work_dir "${DPO_WORK_DIR}" \
    --bank "${BANK}" \
    --val "${VAL}" \
    --base_config "${BASE_CONFIG}" \
    --grid_config "${GRID_CONFIG}"
}

run_smoke() {
  require_sft_gate
  PYTHONPATH=src python3 "${PIPELINE}" dpo-runtime \
    --shared-adapter "${SHARED_ADAPTER}" \
    --grid-config "${GRID_CONFIG}" -- \
    smoke \
    --model_path "${MODEL_PATH}" \
    --work_dir "${DPO_WORK_DIR}" \
    --bank "${BANK}" \
    --val "${VAL}" \
    --base_config "${BASE_CONFIG}" \
    --grid_config "${GRID_CONFIG}"
}

verify_checkpoint_reload() {
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 - \
    "${MODEL_PATH}" "${DPO_WORK_DIR}" "${SHARED_ADAPTER}" <<'PY'
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
shared_adapter = Path(sys.argv[3]).resolve()
smoke_gate_path = work_dir / "SMOKE_GATE.json"
if not smoke_gate_path.is_file():
    raise SystemExit("SMOKE_GATE.json is missing")
smoke_gate = json.loads(smoke_gate_path.read_text(encoding="utf-8"))
if smoke_gate.get("status") != "PASS":
    raise SystemExit("warm-started DPO smoke gate did not pass")
summary_path = Path(smoke_gate["summary"])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if float(summary.get("dpo_beta", -1.0)) != 0.1:
    raise SystemExit("liveness checkpoint is not the frozen beta=0.1 cell")
if summary.get("model_initialization") != "shared_v2_oracle_sft_epoch1_adapter":
    raise SystemExit("liveness summary does not identify the shared SFT initialization")
identity = summary.get("run_identity", {}).get("shared_sft_warmstart", {})
if Path(identity.get("path", "")).resolve() != shared_adapter:
    raise SystemExit("liveness summary used a different SFT adapter")
if summary.get("reference_policy", {}).get("trainable") is not False:
    raise SystemExit("summary does not identify a frozen reference policy")
tolerance = float(summary["initial_pair_margin_max_abs_tolerance"])
if float(summary["initial_pair_margin_max_abs"]) > tolerance:
    raise SystemExit("initial warm policy/reference pair margin exceeded tolerance")
checkpoint = summary_path.parent / "terminal_adapter"
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
    "experiment_id": summary["experiment_id"],
    "source_commit": summary["run_identity"]["source"]["commit"],
    "shared_sft_adapter": identity,
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
  require_sft_gate
  PYTHONPATH=src python3 - "${DPO_WORK_DIR}" <<'PY'
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
print("SFT, liveness, and checkpoint reload gates: PASS")
PY
}

run_full_matrix() {
  if [[ "${E8_WARM_DPO_FORMAL_RUN_AUTHORIZED:-0}" != "1" ]]; then
    echo "full matrix launch requires E8_WARM_DPO_FORMAL_RUN_AUTHORIZED=1 after registration approval" >&2
    exit 2
  fi
  require_liveness_gates
  PYTHONPATH=src python3 "${PIPELINE}" dpo-runtime \
    --shared-adapter "${SHARED_ADAPTER}" \
    --grid-config "${GRID_CONFIG}" -- \
    run \
    --model_path "${MODEL_PATH}" \
    --work_dir "${DPO_WORK_DIR}" \
    --bank "${BANK}" \
    --val "${VAL}" \
    --base_config "${BASE_CONFIG}" \
    --grid_config "${GRID_CONFIG}" \
    --gpus "${DPO_GPUS}" \
    --runtime-slots-per-gpu "${RUNTIME_SLOTS_PER_GPU}"
}

preflight
case "${MODE}" in
  preflight)
    ;;
  sft)
    run_sft
    require_sft_gate
    run_plan
    ;;
  liveness)
    require_sft_gate
    run_plan
    run_smoke
    verify_checkpoint_reload
    ;;
  run)
    run_full_matrix
    ;;
  full)
    run_sft
    require_sft_gate
    run_plan
    run_smoke
    verify_checkpoint_reload
    run_full_matrix
    ;;
  *)
    echo "usage: $0 {preflight|sft|liveness|run|full}" >&2
    exit 2
    ;;
esac
