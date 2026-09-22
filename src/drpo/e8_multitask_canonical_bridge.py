"""Canonical cold-start compatibility bridge for E8 multitask.

This module owns adaptation/dispatch only. Scientific kernels remain in the
frozen canonical Countdown/paper modules. Dependencies on the composition root
are supplied per call through CanonicalBridgeBindings; this module never imports
e8_multitask_exp_tuning back and keeps no process-global host binding.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from drpo import e8_experiment_config as experiment_config
from drpo.e8_multitask_inputs import (
    TaskInputs,
    _changed_leaf_paths,
    _load_task_adapter_and_instances,
)
from drpo.e8_multitask_p0 import (
    append_jsonl,
    atomic_json,
    model_identity,
    read_jsonl,
    sha256_file,
)
from drpo.e8_multitask_tasks import TaskInstance, stable_hash


class CanonicalBridgeHost(Protocol):
    CANONICAL_COLD_MODULES: Any
    DataLoader: Any
    F: Any
    METHOD_ASYMRE: Any
    METHOD_DPO: Any
    METHOD_GLOBAL: Any
    METHOD_POSITIVE_ONLY: Any
    METHOD_RECIPROCAL_LINEAR: Any
    METHOD_RECIPROCAL_QUADRATIC: Any
    METHOD_TOPR: Any
    TRANSFER_SYSTEM_PROMPT: Any
    _adapter_weight_file: Any
    _canonical_asymre_grid_path: Any
    _canonical_calibration_identity: Any
    _canonical_cold_liveness_cell: Any
    _canonical_paths: Any
    _canonical_topr_grid_path: Any
    _cell_identity: Any
    _coldstart_method: Any
    _coldstart_method_cell: Any
    _is_coldstart: Any
    _is_method_matrix: Any
    _method_liveness_grid: Any
    _method_spec: Any
    _paper_grid_name: Any
    _prepare_cell_output: Any
    _repo_root: Any
    _summarize_evaluations: Any
    _verify_fresh_process_adapter_reload: Any
    audit_canonical_coldstart_sources: Any
    torch: Any
    train_cell: Any


@dataclass(frozen=True)
class CanonicalBridgeBindings:
    host: CanonicalBridgeHost


@dataclass(frozen=True)
class CanonicalBridge:
    _paper_grid_paths_exponential: Callable[..., Any]
    _paper_params_reciprocal: Callable[..., Any]
    _paper_params_exponential: Callable[..., Any]
    _paper_params_asymre: Callable[..., Any]
    _paper_params_topr: Callable[..., Any]
    _canonical_cold_modules: Callable[..., Any]
    _activate_paper_grid_modules: Callable[..., Any]
    _canonical_task_record: Callable[..., Any]
    _paper_grid_for_cell: Callable[..., Any]
    _canonical_environment_evaluator: Callable[..., Any]
    _runtime_bridge_contract: Callable[..., Any]
    _legacy_arena_runtime_bridge: Callable[..., Any]
    _validated_runtime_grid: Callable[..., Any]
    _legacy_paper_runtime_bridge: Callable[..., Any]
    _canonical_baseline_grid_identity: Callable[..., Any]
    _normalized_adapter_config_sequence: Callable[..., Any]
    _dpo_shared_sft_adapter_identity: Callable[..., Any]
    _parameter_sequence_sha256: Callable[..., Any]
    _dpo_prompt_balanced_mean: Callable[..., Any]
    _dpo_quantile: Callable[..., Any]
    _is_nan_inf_numerical_failure: Callable[..., Any]
    _load_verified_canonical_calibration: Callable[..., Any]
    _train_canonical_dpo_transfer_cell: Callable[..., Any]
    _cmd_dpo_liveness: Callable[..., Any]
    _train_canonical_cold_cell: Callable[..., Any]
    _cmd_canonical_cold_liveness: Callable[..., Any]


class _CanonicalBridgeImpl:
    """Call-scoped canonical bridge implementation with explicit host dependencies."""

    def __init__(self, bindings: CanonicalBridgeBindings) -> None:
        host = bindings.host
        self.CANONICAL_COLD_MODULES = host.CANONICAL_COLD_MODULES
        self.DataLoader = host.DataLoader
        self.F = host.F
        self.METHOD_ASYMRE = host.METHOD_ASYMRE
        self.METHOD_DPO = host.METHOD_DPO
        self.METHOD_GLOBAL = host.METHOD_GLOBAL
        self.METHOD_POSITIVE_ONLY = host.METHOD_POSITIVE_ONLY
        self.METHOD_RECIPROCAL_LINEAR = host.METHOD_RECIPROCAL_LINEAR
        self.METHOD_RECIPROCAL_QUADRATIC = host.METHOD_RECIPROCAL_QUADRATIC
        self.METHOD_TOPR = host.METHOD_TOPR
        self.TRANSFER_SYSTEM_PROMPT = host.TRANSFER_SYSTEM_PROMPT
        self._adapter_weight_file = host._adapter_weight_file
        self._canonical_asymre_grid_path = host._canonical_asymre_grid_path
        self._canonical_calibration_identity = host._canonical_calibration_identity
        self._canonical_cold_liveness_cell = host._canonical_cold_liveness_cell
        self._canonical_paths = host._canonical_paths
        self._canonical_topr_grid_path = host._canonical_topr_grid_path
        self._cell_identity = host._cell_identity
        self._coldstart_method = host._coldstart_method
        self._coldstart_method_cell = host._coldstart_method_cell
        self._is_coldstart = host._is_coldstart
        self._is_method_matrix = host._is_method_matrix
        self._method_liveness_grid = host._method_liveness_grid
        self._method_spec = host._method_spec
        self._paper_grid_name = host._paper_grid_name
        self._prepare_cell_output = host._prepare_cell_output
        self._repo_root = host._repo_root
        self._summarize_evaluations = host._summarize_evaluations
        self._verify_fresh_process_adapter_reload = host._verify_fresh_process_adapter_reload
        self.audit_canonical_coldstart_sources = host.audit_canonical_coldstart_sources
        self.torch = host.torch
        self.train_cell = host.train_cell

    def _paper_grid_paths_exponential(
        self,
        config: Mapping[str, Any],
        record: Mapping[str, Any],
        cell: Any,
    ) -> tuple[Path, Path]:
        grid_name = (
            "round1_grid"
            if cell.task != "countdown"
            else self._paper_grid_name(
                0.0 if cell.lambda_value is None else float(cell.lambda_value)
            )
        )
        return Path(str(record[grid_name])), self._canonical_paths(config)[grid_name]

    def _paper_params_reciprocal(self, cell: Any) -> tuple[str, float, float]:
        if cell.method not in {self.METHOD_RECIPROCAL_LINEAR, self.METHOD_RECIPROCAL_QUADRATIC}:
            raise AssertionError(f"Unsupported reciprocal family: {cell.method}")
        if cell.lambda_value is None:
            raise AssertionError("Reciprocal cell has no lambda")
        return cell.method, 1.0, float(cell.lambda_value)

    def _paper_params_exponential(self, cell: Any) -> tuple[str, float, float]:
        alpha = 0.0 if cell.method == self.METHOD_POSITIVE_ONLY else 1.0
        coefficient = (
            0.0
            if cell.method in {self.METHOD_POSITIVE_ONLY, self.METHOD_GLOBAL}
            else float(cell.lambda_value)
        )
        return "exponential", alpha, coefficient

    def _paper_params_asymre(self, cell: Any) -> tuple[str, float, float]:
        if cell.delta_v is None:
            raise AssertionError("AsymRE cell has no delta_v")
        return self.METHOD_ASYMRE, 1.0 + float(cell.delta_v), 0.0

    def _paper_params_topr(self, cell: Any) -> tuple[str, float, float]:
        if cell.beta is None:
            raise AssertionError("Joint Fitted-Reference TOPR cell has no beta")
        return self.METHOD_TOPR, 1.0, float(cell.beta)

    def _canonical_cold_modules(self, config: Mapping[str, Any]) -> dict[str, Any]:
        self.audit_canonical_coldstart_sources(config)
        modules = {
            name: importlib.import_module(module_name)
            for name, module_name in self.CANONICAL_COLD_MODULES.items()
        }
        scan_common = modules["scan_common"]
        scan_runtime = modules["scan_runtime"]
        scan_trainer = modules["scan_trainer"]
        paper_common = modules["paper_common"]
        paper_runtime = modules["paper_runtime"]
        arena = modules["arena"]
        if (
            scan_common.arena is not arena
            or scan_trainer.arena is not arena
            or scan_trainer.continuous_exp_weights is not scan_common.continuous_exp_weights
            or paper_common._base is not scan_common
            or paper_runtime.highc is not paper_common
            or paper_runtime._base_runtime is not scan_runtime
        ):
            raise RuntimeError("Paper cold-start modules do not share one locked implementation graph")
        return modules

    def _activate_paper_grid_modules(self, modules: dict[str, Any], grid_path: Path) -> dict[str, Any]:
        """Bind base trainer imports to the selected paper profile in this cell process."""

        paper_common = modules["paper_common"]
        paper_common.activate_for_grid_config(grid_path)
        modules["scan_trainer"] = importlib.reload(modules["scan_trainer"])
        modules["scan_runtime"] = importlib.reload(modules["scan_runtime"])
        # Reloading the paper adapter after the base modules recreates its wrappers
        # around the just-reloaded, profile-correct trainer.
        modules["paper_runtime"] = importlib.reload(modules["paper_runtime"])
        if (
            modules["scan_trainer"].continuous_exp_weights
            is not modules["scan_common"].continuous_exp_weights
            or modules["paper_runtime"]._base_runtime is not modules["scan_runtime"]
        ):
            raise RuntimeError("Paper grid activation did not bind the selected trainer profile")
        return modules

    def _canonical_task_record(self, split_manifest: Mapping[str, Any], task: str) -> Mapping[str, Any]:
        value = split_manifest["tasks"][task].get("canonical_coldstart")
        if not isinstance(value, Mapping):
            raise TypeError(f"Missing canonical cold-start task record for {task}")
        return value

    def _paper_grid_for_cell(
        self,
        config: Mapping[str, Any],
        record: Mapping[str, Any],
        cell: Any,
    ) -> tuple[Path, Path]:
        paper_runtime = self._method_spec(cell.method).paper_runtime
        if paper_runtime is None:
            raise RuntimeError(
                f"{cell.method} does not use the canonical paper grid runtime"
            )
        return paper_runtime.grid_paths(config, record, cell)

    def _canonical_environment_evaluator(
        self,
        *,
        arena: Any,
        task_adapter: Any,
        instances: Mapping[str, TaskInstance],
        greedy_prompt_rows: int,
        passk_prompt_rows: int,
        sampling_temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Any:
        """Return the old evaluator signature backed only by the selected task verifier."""

        def evaluate_rows(
            model: Any,
            tokenizer: Any,
            rows: Sequence[Mapping[str, Any]],
            batch_size: int,
            max_new_tokens: int,
            pass_k: int,
            seed: int,
            known_structures: set[str] | None = None,
        ) -> dict[str, Any]:
            del known_structures
            if not rows:
                raise RuntimeError("Task validation rows must be non-empty")
            if len(rows) < int(greedy_prompt_rows) or len(rows) < int(passk_prompt_rows):
                raise RuntimeError(
                    "Task validation input is smaller than the frozen Greedy/Pass@k budgets"
                )
            arena.seed_all(seed)
            was_training = bool(model.training)
            greedy_correct: list[float] = []
            greedy_valid: list[float] = []
            pass_success: list[float] = []
            sampled_valid: list[float] = []
            greedy_rows = list(rows[: int(greedy_prompt_rows)])
            sampled_rows = list(rows[: int(passk_prompt_rows)])
            for start in range(0, len(greedy_rows), int(batch_size)):
                chunk = greedy_rows[start : start + int(batch_size)]
                prompts = [str(row["prompt"]) for row in chunk]
                greedy = arena.generate_outputs(
                    model,
                    tokenizer,
                    prompts,
                    int(max_new_tokens),
                    False,
                    1.0,
                    1.0,
                    1,
                )
                for row, greedy_outputs in zip(chunk, greedy, strict=True):
                    instance = instances[str(row["prompt_id"])]
                    greedy_result = task_adapter.verify(instance, greedy_outputs[0])
                    greedy_correct.append(float(greedy_result.correct))
                    greedy_valid.append(float(greedy_result.format_valid))
            for start in range(0, len(sampled_rows), int(batch_size)):
                chunk = sampled_rows[start : start + int(batch_size)]
                prompts = [str(row["prompt"]) for row in chunk]
                sampled = arena.generate_outputs(
                    model,
                    tokenizer,
                    prompts,
                    int(max_new_tokens),
                    int(pass_k) > 1,
                    float(sampling_temperature) if int(pass_k) > 1 else 1.0,
                    float(top_p) if int(pass_k) > 1 else 1.0,
                    int(pass_k),
                )
                for row, sampled_outputs in zip(chunk, sampled, strict=True):
                    instance = instances[str(row["prompt_id"])]
                    sample_results = [
                        task_adapter.verify(instance, completion) for completion in sampled_outputs
                    ]
                    pass_success.append(float(any(result.correct for result in sample_results)))
                    sampled_valid.extend(float(result.format_valid) for result in sample_results)
            metrics = {
                "greedy_success": float(np.mean(greedy_correct)),
                "pass_at_k": float(np.mean(pass_success)),
                "valid_rate": float(np.mean(greedy_valid)),
                "sampled_valid_rate": float(np.mean(sampled_valid)),
                "n_eval": float(len(greedy_rows)),
                "greedy_prompt_rows": float(len(greedy_rows)),
                "passk_prompt_rows": float(len(sampled_rows)),
                "task_verifier_interface": True,
            }
            numeric = [value for value in metrics.values() if isinstance(value, (int, float))]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise RuntimeError("Task verifier evaluation produced a non-finite metric")
            if int(pass_k) == 8:
                evaluate_rows._last_primary_sampled_valid_rate = float(
                    metrics["sampled_valid_rate"]
                )
            if was_training:
                model.train()
            return metrics

        return evaluate_rows

    def _runtime_bridge_contract(self, effective: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "fresh_lora": {
                "rank": int(effective["model"]["lora_rank"]),
                "alpha": int(effective["model"]["lora_alpha"]),
                "dropout": float(effective["model"]["lora_dropout"]),
            },
            "gradient_checkpointing": bool(effective["model"]["gradient_checkpointing"]),
            "optimizer_weight_decay": float(effective["training"]["weight_decay"]),
            "sampling_temperature": float(effective["evaluation"]["sampling_temperature"]),
            "top_p": float(effective["evaluation"]["top_p"]),
        }

    @contextmanager
    def _legacy_arena_runtime_bridge(self, arena: Any, effective: Mapping[str, Any]) -> Any:
        """Temporarily parameterize legacy arena interface literals; never touch loss math."""

        original_lora_config = arena.LoraConfig
        original_load_model = arena.load_model
        original_generate_outputs = arena.generate_outputs
        original_scheduler = getattr(arena, "get_cosine_schedule_with_warmup", None)
        contract = self._runtime_bridge_contract(effective)

        def configured_lora_config(*args: Any, **kwargs: Any) -> Any:
            kwargs["r"] = int(contract["fresh_lora"]["rank"])
            kwargs["lora_alpha"] = int(contract["fresh_lora"]["alpha"])
            kwargs["lora_dropout"] = float(contract["fresh_lora"]["dropout"])
            return original_lora_config(*args, **kwargs)

        def configured_load_model(*args: Any, **kwargs: Any) -> Any:
            values = list(args)
            if len(values) > 5:
                values[5] = bool(contract["gradient_checkpointing"])
            else:
                kwargs["gradient_checkpointing"] = bool(contract["gradient_checkpointing"])
            return original_load_model(*values, **kwargs)

        def configured_generate_outputs(
            model: Any,
            tokenizer: Any,
            prompts: list[str],
            max_new_tokens: int,
            do_sample: bool,
            temperature: float,
            top_p: float,
            num_return_sequences: int = 1,
        ) -> Any:
            if do_sample:
                temperature = float(contract["sampling_temperature"])
                top_p = float(contract["top_p"])
            return original_generate_outputs(
                model,
                tokenizer,
                prompts,
                max_new_tokens,
                do_sample,
                temperature,
                top_p,
                num_return_sequences,
            )

        def configured_scheduler(
            optimizer: Any,
            num_warmup_steps: int,
            num_training_steps: int,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if original_scheduler is None:
                raise RuntimeError("Canonical arena scheduler is unavailable")
            if float(effective["training"]["warmup_ratio"]) == 0.0:
                num_warmup_steps = 0
            return original_scheduler(optimizer, num_warmup_steps, num_training_steps, *args, **kwargs)

        arena.LoraConfig = configured_lora_config
        arena.load_model = configured_load_model
        arena.generate_outputs = configured_generate_outputs
        if original_scheduler is not None:
            arena.get_cosine_schedule_with_warmup = configured_scheduler
        try:
            yield contract
        finally:
            arena.LoraConfig = original_lora_config
            arena.load_model = original_load_model
            arena.generate_outputs = original_generate_outputs
            if original_scheduler is not None:
                arena.get_cosine_schedule_with_warmup = original_scheduler

    def _validated_runtime_grid(
        self,
        candidate: Mapping[str, Any],
        *,
        canonical_grid: Mapping[str, Any],
        effective: Mapping[str, Any],
        strict_validator: Any,
    ) -> None:
        allowed = {"training.steps", "training.eval_every"}
        changed = set(_changed_leaf_paths(canonical_grid, candidate))
        forbidden = sorted(changed - allowed)
        if forbidden:
            raise ValueError(f"Derived runtime grid changed non-runtime fields: {forbidden}")
        training = candidate.get("training", {})
        if int(training.get("steps", -1)) != int(effective["training"]["optimizer_updates"]):
            raise ValueError("Derived runtime grid steps do not match effective runtime")
        if int(training.get("eval_every", -1)) != int(
            effective["training"]["evaluation_every_updates"]
        ):
            raise ValueError("Derived runtime grid eval_every does not match effective runtime")
        strict = copy.deepcopy(dict(candidate))
        strict["training"]["steps"] = canonical_grid["training"]["steps"]
        strict["training"]["eval_every"] = canonical_grid["training"]["eval_every"]
        strict_validator(strict)

    @contextmanager
    def _legacy_paper_runtime_bridge(
        self,
        modules: Mapping[str, Any],
        effective: Mapping[str, Any],
        *,
        grid_path: Path,
        grid_source_path: Path,
    ) -> Any:
        """Bridge configured runtime scalars into byte-locked paper interfaces."""

        scan_trainer = modules["scan_trainer"]
        paper_common = modules["paper_common"]
        canonical_grid = yaml.safe_load(grid_source_path.read_text(encoding="utf-8"))
        if not isinstance(canonical_grid, dict):
            raise TypeError("Canonical paper grid root must be a mapping")
        candidate_grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
        if not isinstance(candidate_grid, dict):
            raise TypeError("Derived paper grid root must be a mapping")

        validator_targets: list[tuple[Any, str, Any]] = []
        seen: set[tuple[int, str]] = set()
        for module_name in ("paper_common", "scan_common", "scan_trainer", "scan_runtime"):
            module = modules[module_name]
            if not hasattr(module, "validate_grid_config"):
                continue
            key = (id(module), "validate_grid_config")
            if key in seen:
                continue
            seen.add(key)
            validator_targets.append((module, "validate_grid_config", module.validate_grid_config))
        strict_validator = paper_common.validate_grid_config

        def configured_validator(value: Mapping[str, Any]) -> None:
            self._validated_runtime_grid(
                value,
                canonical_grid=canonical_grid,
                effective=effective,
                strict_validator=strict_validator,
            )

        optimizer_holder = scan_trainer.torch.optim
        original_adamw = optimizer_holder.AdamW

        def configured_adamw(*args: Any, **kwargs: Any) -> Any:
            kwargs["weight_decay"] = float(effective["training"]["weight_decay"])
            return original_adamw(*args, **kwargs)

        with self._legacy_arena_runtime_bridge(modules["arena"], effective) as contract:
            optimizer_holder.AdamW = configured_adamw
            for module, name, _ in validator_targets:
                setattr(module, name, configured_validator)
            try:
                configured_validator(candidate_grid)
                yield contract
            finally:
                optimizer_holder.AdamW = original_adamw
                for module, name, original in validator_targets:
                    setattr(module, name, original)

    def _canonical_baseline_grid_identity(self, method: str) -> dict[str, Any]:
        """Record actual canonical-grid bytes without introducing a new expected-blob gate."""

        if method == self.METHOD_ASYMRE:
            path = self._canonical_asymre_grid_path()
        elif method == self.METHOD_TOPR:
            path = self._canonical_topr_grid_path()
        else:
            raise ValueError(f"No extra canonical grid identity for method: {method}")
        return {
            "canonical_grid": str(path),
            "canonical_grid_sha256": sha256_file(path),
            "identity_policy": "runtime_actual_sha256_recorded_non_gating",
            "expected_git_blob_gate": False,
        }

    def _normalized_adapter_config_sequence(self, value: Any, *, field: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, Sequence):
            values = tuple(str(item) for item in value)
        else:
            raise TypeError(f"Shared-SFT DPO adapter {field} must be a string sequence or null")
        normalized = tuple(sorted(item.strip() for item in values))
        if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError(f"Shared-SFT DPO adapter {field} contains invalid names")
        return normalized

    def _dpo_shared_sft_adapter_identity(self, config: Mapping[str, Any]) -> dict[str, Any] | None:
        dpo = config["dpo"]
        mode = str(dpo["initialization_mode"])
        if mode != "shared_sft_adapter":
            return None
        contract = dpo["shared_sft_adapter_contract"]
        env_name = str(dpo["shared_sft_adapter_env"])
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"Shared-SFT DPO requires environment variable {env_name}")
        path = Path(value).resolve()
        adapter_config_path = path / "adapter_config.json"
        weight_name = str(contract["adapter_weight_file"])
        recognized_weight_names = {"adapter_model.safetensors", "adapter_model.bin"}
        present_weight_names = {
            name for name in recognized_weight_names if (path / name).is_file()
        }
        if present_weight_names != {weight_name}:
            raise ValueError(
                "Shared-SFT DPO adapter directory must contain exactly the contracted "
                f"recognized weight file: expected={weight_name}, "
                f"present={sorted(present_weight_names)}"
            )
        weight_path = path / weight_name
        provenance_relative = Path(str(contract["provenance_file"]))
        provenance_path = (path / provenance_relative).resolve()
        if path not in provenance_path.parents:
            raise RuntimeError("Shared-SFT DPO provenance escaped the adapter root")
        for required in (adapter_config_path, weight_path, provenance_path):
            if not required.is_file():
                raise FileNotFoundError(f"Shared-SFT DPO adapter is incomplete: {required}")
        observed_hashes = {
            "adapter_config_sha256": sha256_file(adapter_config_path),
            "adapter_weight_sha256": sha256_file(weight_path),
            "provenance_sha256": sha256_file(provenance_path),
        }
        if any(observed_hashes[field] != str(contract[field]) for field in observed_hashes):
            raise ValueError("Shared-SFT DPO adapter identity hash mismatch")

        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        if not isinstance(adapter_config, dict):
            raise TypeError("Shared-SFT DPO adapter_config.json must contain a mapping")
        model_config = config["model"]
        target_modules = self._normalized_adapter_config_sequence(
            adapter_config.get("target_modules"), field="target_modules"
        )
        modules_to_save = self._normalized_adapter_config_sequence(
            adapter_config.get("modules_to_save"), field="modules_to_save"
        )
        expected_target_modules = tuple(sorted(str(value) for value in contract["target_modules"]))
        expected_modules_to_save = tuple(sorted(str(value) for value in contract["modules_to_save"]))
        compatible = (
            str(adapter_config.get("peft_type", "")).upper() == "LORA"
            and int(adapter_config.get("r", -1)) == int(model_config["lora_rank"])
            and int(adapter_config.get("lora_alpha", -1)) == int(model_config["lora_alpha"])
            and math.isclose(
                float(adapter_config.get("lora_dropout", -1.0)),
                float(model_config["lora_dropout"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and str(adapter_config.get("base_model_name_or_path", ""))
            == str(contract["adapter_base_model_name_or_path"])
            and target_modules == expected_target_modules
            and modules_to_save == expected_modules_to_save
            and str(adapter_config.get("bias", "")) == str(contract["bias"])
        )
        if not compatible:
            raise ValueError(
                "Shared-SFT DPO adapter LoRA/base-model parameterization does not match reviewed contract"
            )

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            raise TypeError("Shared-SFT DPO provenance file must contain a mapping")
        provenance_expected = dict(contract["provenance_expected"])
        if any(provenance.get(key) != expected for key, expected in provenance_expected.items()):
            raise ValueError("Shared-SFT DPO provenance does not match reviewed contract")
        if (
            provenance.get("base_model") != model_config["base_model"]
            or provenance.get("base_model_revision") != model_config["revision"]
        ):
            raise ValueError("Shared-SFT DPO provenance base-model/revision mismatch")

        identity = {
            **observed_hashes,
            "adapter_weight_file": weight_name,
            "provenance_file": provenance_relative.as_posix(),
            "adapter_parameterization": {
                "peft_type": "LORA",
                "r": int(adapter_config["r"]),
                "lora_alpha": int(adapter_config["lora_alpha"]),
                "lora_dropout": float(adapter_config["lora_dropout"]),
                "base_model_name_or_path": str(adapter_config["base_model_name_or_path"]),
                "target_modules": list(target_modules),
                "modules_to_save": list(modules_to_save),
                "bias": str(adapter_config["bias"]),
            },
            "provenance_expected": provenance_expected,
            "contract_hash": stable_hash(contract),
        }
        identity["identity_hash"] = stable_hash(identity)
        return {"path": str(path), **identity}

    def _parameter_sequence_sha256(self, parameters: Sequence[Any]) -> str:
        if self.torch is None:
            raise RuntimeError("Torch is required")
        digest = hashlib.sha256()
        if not parameters:
            raise RuntimeError("Cannot hash an empty parameter sequence")
        for index, parameter in enumerate(parameters):
            value = parameter.detach().cpu().contiguous()
            digest.update(str(index).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _dpo_prompt_balanced_mean(
        self,
        values: Any,
        row_index: Any,
        unique_counts: Any,
        *,
        paper_common: Any,
    ) -> Any:
        return paper_common.mean_unique_negative_term(
            values,
            self.torch.ones_like(values),
            row_index,
            unique_counts,
        )

    def _dpo_quantile(self, values: Any, q: float) -> float:
        return float(self.torch.quantile(values.detach().float().cpu(), q).item())

    def _is_nan_inf_numerical_failure(self, value: Any) -> bool:
        return isinstance(value, str) and value.startswith("nonfinite_")

    def _load_verified_canonical_calibration(
        self,
        task: str,
        *,
        split_manifest: Mapping[str, Any],
        base_model_path: str,
        config: Mapping[str, Any],
        output_root: Path,
        error_prefix: str,
    ) -> dict[str, Any]:
        path = output_root / "calibration" / f"{task}.json"
        if not path.is_file():
            raise RuntimeError(f"Run the no-calibration identity gate before {task}")
        calibration = json.loads(path.read_text(encoding="utf-8"))
        expected = self._canonical_calibration_identity(
            task,
            split_manifest=split_manifest,
            base_model_path=base_model_path,
            config=config,
        )
        if (
            calibration.get("identity_hash") != expected["identity_hash"]
            or calibration.get("enabled") is not False
            or not calibration.get("complete")
        ):
            raise RuntimeError(f"{error_prefix} no-calibration identity mismatch for {task}")
        return calibration

    def _train_canonical_dpo_transfer_cell(
        self,
        cell: Any,
        *,
        inputs: TaskInputs,
        split_manifest: Mapping[str, Any],
        base_model_path: str,
        config: Mapping[str, Any],
        output_root: Path,
        force: bool,
        updates_override: int | None = None,
        engineering_liveness: bool = False,
    ) -> dict[str, Any]:
        """Port the reviewed PR #268 DPO semantics onto the frozen multitask task interface."""

        if self.torch is None or self.F is None or self.DataLoader is None:
            raise RuntimeError("Canonical DPO training requires Torch")
        if cell.method != self.METHOD_DPO or cell.beta is None:
            raise ValueError("Canonical DPO transfer trainer received a non-DPO cell")
        configured_initialization = str(config["dpo"]["initialization_mode"])
        if cell.dpo_initialization != configured_initialization:
            raise RuntimeError("DPO cell/config initialization identity mismatch")
        if cell.task == "countdown":
            raise ValueError("Current multitask DPO capability does not execute Countdown cells")
        modules = self._canonical_cold_modules(config)
        arena = modules["arena"]
        paper_common = modules["paper_common"]
        scan_trainer = modules["scan_trainer"]
        record = self._canonical_task_record(split_manifest, cell.task)
        calibration = self._load_verified_canonical_calibration(
            cell.task,
            split_manifest=split_manifest,
            base_model_path=base_model_path,
            config=config,
            output_root=output_root,
            error_prefix="DPO",
        )

        root_name = "liveness" if engineering_liveness else "cells"
        shared_adapter_record = self._dpo_shared_sft_adapter_identity(config)
        shared_adapter = (
            None if shared_adapter_record is None else Path(str(shared_adapter_record["path"]))
        )
        task_sft_adapter = None
        if configured_initialization == "task_positive_warmstart":
            if inputs.reference_adapter is None:
                raise RuntimeError(
                    f"Task-SFT DPO initialization adapter is missing for {cell.task}"
                )
            task_sft_adapter = inputs.reference_adapter.resolve()
        initial_adapter = shared_adapter if shared_adapter is not None else task_sft_adapter
        identity = self._cell_identity(
            cell,
            inputs=inputs,
            split_manifest=split_manifest,
            base_model_path=base_model_path,
            config=config,
            calibration=calibration,
        )
        identity.update(
            {
                "canonical_source_git_blob_shas": dict(
                    config["canonical_coldstart"]["expected_git_blob_shas"]
                ),
                "canonical_dispatch": "e8_multitask_exp_tuning._train_canonical_dpo_transfer_cell",
                "dpo_semantics_source": "historical_PR_268_protected_implementation_cc0ead2be00c89a3c35296b7adc1ddeae8d14759",
                "dpo_initialization_mode": str(config["dpo"]["initialization_mode"]),
                "reference_remoteness_bank_identity_hash": record.get(
                    "reference_remoteness_bank_identity_hash"
                ),
                "canonical_train_sha256": record["train_sha256"],
                "shared_sft_adapter_identity": (
                    None
                    if shared_adapter is None
                    else model_identity(base_model_path, str(shared_adapter))["adapter"]
                ),
                "shared_sft_adapter_provenance": (
                    None
                    if shared_adapter_record is None
                    else {
                        key: value
                        for key, value in shared_adapter_record.items()
                        if key != "path"
                    }
                ),
                "engineering_liveness": engineering_liveness,
                "updates_override": updates_override,
            }
        )
        identity["identity_hash"] = stable_hash(identity)
        cell_root, manifest_path, reusable = self._prepare_cell_output(
            output_root,
            root_name=root_name,
            cell=cell,
            identity=identity,
            force=force,
            mismatch_prefix="Existing DPO cell identity mismatch",
            existing_prefix="DPO cell output exists without reusable identity",
            unsafe_prefix="Refusing unsafe DPO cell removal",
        )
        if reusable is not None:
            return reusable

        bank = Path(str(record["train"]))
        validation = Path(str(record["validation"]))
        base_config_path = Path(str(record["base_config"]))
        base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
        if not isinstance(base_config, dict):
            raise TypeError("DPO paper base config root must be a mapping")
        effective = experiment_config.effective_coldstart_runtime(config, cell.task)
        model_cfg = copy.deepcopy(base_config["model"])
        model_cfg["max_length"] = int(effective["model"]["max_length"])
        model_cfg["max_new_tokens"] = int(effective["model"]["max_new_tokens"])
        train_cfg = base_config["offline_training"]
        eval_cfg = copy.deepcopy(base_config["evaluation"])
        eval_cfg["examples"] = int(effective["evaluation"]["examples"])
        eval_cfg["batch_size"] = int(effective["evaluation"]["batch_size"])
        eval_cfg["pass_ks"] = list(effective["evaluation"]["pass_ks"])
        eval_cfg["seed"] = int(effective["evaluation"]["generation_seed"])
        updates = int(updates_override or effective["training"]["optimizer_updates"])
        eval_every = int(effective["training"]["evaluation_every_updates"])
        log_every = int(train_cfg["log_every"])
        legacy_training_seed_base = int(train_cfg["seed"])
        training_seed_base = (
            experiment_config.coldstart_runtime_seed(config)
            if configured_initialization == "task_positive_warmstart"
            else legacy_training_seed_base
        )
        seed = training_seed_base + int(cell.seed)
        beta = float(cell.beta)
        zero_beta_control = math.isclose(beta, 0.0, rel_tol=0.0, abs_tol=0.0)
        if zero_beta_control and configured_initialization != "task_positive_warmstart":
            raise RuntimeError(
                "DPO beta=0 is reserved for the task-SFT initialization-only control"
            )
        arena.seed_all(seed)

        validation_rows = [] if engineering_liveness else read_jsonl(validation)
        evaluator = None
        if not engineering_liveness:
            task_adapter, instances = _load_task_adapter_and_instances(
                cell.task,
                inputs=inputs,
                validation_rows=validation_rows,
            )
            evaluator = self._canonical_environment_evaluator(
                arena=arena,
                task_adapter=task_adapter,
                instances=instances,
                greedy_prompt_rows=int(config["task_runtime"][cell.task]["greedy_prompt_rows"]),
                passk_prompt_rows=int(config["task_runtime"][cell.task]["passk_prompt_rows"]),
                sampling_temperature=float(effective["evaluation"]["sampling_temperature"]),
                top_p=float(effective["evaluation"]["top_p"]),
            )

        original_clean_expression = arena.clean_expression
        original_system_prompt = arena.SYSTEM_PROMPT
        original_evaluate_rows = arena.evaluate_rows
        try:
            arena.clean_expression = lambda value: str(value)
            arena.SYSTEM_PROMPT = self.TRANSFER_SYSTEM_PROMPT
            if evaluator is not None:
                arena.evaluate_rows = evaluator

            tokenizer = arena.load_tokenizer(str(Path(base_model_path).resolve()))
            iterator = None
            if not zero_beta_control:
                train_rows = arena.read_jsonl(bank)
                dataset = paper_common.ContinuousUniqueBankDataset(
                    train_rows,
                    tokenizer,
                    int(effective["model"]["max_length"]),
                )
                generator = self.torch.Generator().manual_seed(seed)
                loader = self.DataLoader(
                    dataset,
                    batch_size=int(effective["training"]["micro_batch"]),
                    shuffle=True,
                    generator=generator,
                    collate_fn=paper_common.make_continuous_unique_bank_collator(
                        tokenizer.pad_token_id
                    ),
                    num_workers=int(train_cfg["num_workers"]),
                )
                iterator = iter(loader)
            with self._legacy_arena_runtime_bridge(arena, effective):
                model = arena.load_model(
                    str(Path(base_model_path).resolve()),
                    adapter_path=None if initial_adapter is None else str(initial_adapter),
                    trainable_adapter=True,
                    load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
                    dtype=str(effective["model"]["dtype"]),
                    gradient_checkpointing=bool(effective["model"]["gradient_checkpointing"]),
                    parameterization="lora",
                )
            policy_adapter = str(config["dpo"]["policy_adapter"])
            reference_adapter = str(config["dpo"]["reference_adapter"])
            if not hasattr(model, "add_adapter") or not hasattr(model, "set_adapter"):
                raise RuntimeError("Canonical DPO requires PEFT multi-adapter support")
            if policy_adapter not in model.peft_config:
                raise RuntimeError("DPO policy adapter is missing after model initialization")
            if reference_adapter in model.peft_config:
                raise RuntimeError("DPO reference adapter exists before exact initialization copy")
            model.add_adapter(
                reference_adapter,
                copy.deepcopy(model.peft_config[policy_adapter]),
            )
            scan_trainer._copy_adapter_parameters(model, policy_adapter, reference_adapter)
            policy_parameters = scan_trainer._adapter_parameters(model, policy_adapter)
            reference_parameters = scan_trainer._adapter_parameters(model, reference_adapter)

            def activate_reference() -> None:
                model.set_adapter(reference_adapter)
                for parameter in reference_parameters:
                    parameter.requires_grad_(False)

            def activate_policy() -> None:
                model.set_adapter(policy_adapter)
                for parameter in policy_parameters:
                    parameter.requires_grad_(True)
                for parameter in reference_parameters:
                    parameter.requires_grad_(False)

            activate_policy()
            policy_initial_sha256 = self._parameter_sequence_sha256(policy_parameters)
            reference_initial_sha256 = self._parameter_sequence_sha256(reference_parameters)
            if policy_initial_sha256 != reference_initial_sha256:
                raise RuntimeError("DPO policy/reference exact initialization copy failed")
            optimizer = None
            scheduler = None
            device = None
            if not zero_beta_control:
                optimizer = self.torch.optim.AdamW(
                    policy_parameters,
                    lr=float(effective["training"]["learning_rate"]),
                    weight_decay=float(effective["training"]["weight_decay"]),
                )
                warmup_ratio = float(effective["training"]["warmup_ratio"])
                warmup_steps = (
                    0
                    if warmup_ratio == 0.0
                    else max(1, int(updates * warmup_ratio))
                )
                scheduler = arena.get_cosine_schedule_with_warmup(
                    optimizer,
                    warmup_steps,
                    updates,
                )
                device = next(model.parameters()).device
            model.eval()
            training_path = cell_root / "training_metrics.jsonl"
            evaluation_path = cell_root / "evaluation_metrics.jsonl"
            best_dir = cell_root / "supplementary_best_adapter"
            terminal_dir = cell_root / "terminal_adapter"
            last_finite_dir = cell_root / "last_finite_adapter"
            best_pass8 = -math.inf
            optimizer_update_norms: list[float] = []
            initial_pair_margin_max_abs: float | None = None
            tolerance = float(config["dpo"]["initial_pair_margin_max_abs_tolerance"])
            numerical_failure: str | None = None
            stop_reason = (
                "sft_only_no_dpo_update" if zero_beta_control else "max_steps"
            )
            terminal_step = 0
            last_finite_step = 0

            def evaluate(step: int) -> None:
                nonlocal best_pass8
                if evaluator is None:
                    return
                activate_policy()
                model.eval()
                row = scan_trainer._evaluate_validation(
                    model=model,
                    tokenizer=tokenizer,
                    val_rows=validation_rows,
                    known_structures={f"{cell.task}:task_verifier"},
                    model_cfg=model_cfg,
                    eval_cfg=eval_cfg,
                    step=step,
                    pass64_every=200,
                    pass64_enabled=64
                    in {int(value) for value in effective["evaluation"]["auxiliary_pass_ks"]},
                )
                sampled_valid_rate = getattr(evaluator, "_last_primary_sampled_valid_rate", None)
                if sampled_valid_rate is not None:
                    row["val_sampled_valid_rate"] = float(sampled_valid_rate)
                append_jsonl(
                    evaluation_path,
                    {
                        "update": int(row["step"]),
                        "pass8": float(row["val_pass_at_8"]),
                        "greedy_success": float(row["val_greedy"]),
                        "greedy_valid_rate": float(row["val_valid_rate"]),
                        "sampled_valid_rate": row.get("val_sampled_valid_rate"),
                    },
                )
                if float(row["val_pass_at_8"]) > best_pass8:
                    best_pass8 = float(row["val_pass_at_8"])
                    if not zero_beta_control:
                        if best_dir.exists():
                            shutil.rmtree(best_dir)
                        activate_policy()
                        model.save_pretrained(best_dir, safe_serialization=True)
                        tokenizer.save_pretrained(best_dir)

            if not engineering_liveness:
                evaluate(0)
            accumulation = int(effective["training"]["gradient_accumulation"])
            training_updates = 0 if zero_beta_control else updates
            if training_updates:
                if (
                    iterator is None
                    or optimizer is None
                    or scheduler is None
                    or device is None
                ):
                    raise RuntimeError("Positive-beta DPO training runtime was not initialized")
                optimizer.zero_grad(set_to_none=True)
            if zero_beta_control:
                append_jsonl(
                    training_path,
                    {
                        "update": 0,
                        "dpo_beta": beta,
                        "control_role": "sft_only_no_dpo_update",
                        "dpo_pair_loss": None,
                        "raw_gradient_norm_before_clip": None,
                        "optimizer_update_norm": None,
                        "gradient_probe": "not_run_no_dpo_update",
                        "optimizer_step": "not_run_no_dpo_update",
                        "initial_pair_margin_max_abs": None,
                        "initial_pair_margin_probe": (
                            "not_run_exact_policy_reference_state_hashes_match"
                        ),
                        "reference_role": "exact_frozen_initial_policy",
                        "label_smoothing": 0.0,
                        "test_data_used": False,
                    },
                )
            for update in range(1, training_updates + 1):
                loss_total = 0.0
                diagnostic_totals = {
                    "policy_chosen_sum_lp": 0.0,
                    "policy_rejected_sum_lp": 0.0,
                    "reference_chosen_sum_lp": 0.0,
                    "reference_rejected_sum_lp": 0.0,
                    "pair_margin_mean": 0.0,
                    "pair_margin_p10": 0.0,
                    "pair_margin_p50": 0.0,
                    "pair_margin_p90": 0.0,
                    "preference_accuracy": 0.0,
                    "logit_saturation_fraction": 0.0,
                    "unique_negative_count_mean": 0.0,
                    "raw_bank_count_mean": 0.0,
                    "duplicates_removed_mean": 0.0,
                }
                abort_update = False
                for accumulation_index in range(accumulation):
                    try:
                        packed = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        packed = next(iterator)
                    positive_batch = arena.move_to_device(packed["positive"], device)
                    bank_batch = arena.move_to_device(packed["bank"], device)
                    row_index = packed["bank_row_index"].to(device)
                    unique_counts = packed["unique_counts"].to(device)

                    activate_reference()
                    model.eval()
                    with self.torch.no_grad():
                        reference_positive_stats = arena.completion_stats(model, positive_batch)
                        reference_bank_stats = arena.completion_stats(model, bank_batch)
                        reference_chosen = paper_common.full_sequence_log_probability(
                            reference_positive_stats
                        ).detach()
                        reference_rejected = paper_common.full_sequence_log_probability(
                            reference_bank_stats
                        ).detach()

                    activate_policy()
                    model.eval()
                    policy_positive_stats = arena.completion_stats(model, positive_batch)
                    policy_bank_stats = arena.completion_stats(model, bank_batch)
                    policy_chosen = paper_common.full_sequence_log_probability(
                        policy_positive_stats
                    )
                    policy_rejected = paper_common.full_sequence_log_probability(
                        policy_bank_stats
                    )
                    pair_margin = (
                        policy_chosen[row_index]
                        - policy_rejected
                        - reference_chosen[row_index]
                        + reference_rejected
                    )
                    if update == 1 and accumulation_index == 0:
                        initial_pair_margin_max_abs = float(pair_margin.detach().abs().max())
                        if initial_pair_margin_max_abs > tolerance:
                            numerical_failure = "initial_policy_reference_pair_margin_mismatch"
                            stop_reason = numerical_failure
                            abort_update = True
                            break
                    logits = beta * pair_margin
                    pair_losses = self.F.softplus(-logits)
                    loss = self._dpo_prompt_balanced_mean(
                        pair_losses,
                        row_index,
                        unique_counts,
                        paper_common=paper_common,
                    )
                    if not bool(self.torch.isfinite(loss)):
                        numerical_failure = f"nonfinite_loss_at_step_{update}"
                        stop_reason = numerical_failure
                        abort_update = True
                        break
                    (loss / accumulation).backward()
                    loss_total += float(loss.detach().cpu())
                    policy_rejected_mean = self._dpo_prompt_balanced_mean(
                        policy_rejected.detach(),
                        row_index,
                        unique_counts,
                        paper_common=paper_common,
                    )
                    reference_rejected_mean = self._dpo_prompt_balanced_mean(
                        reference_rejected,
                        row_index,
                        unique_counts,
                        paper_common=paper_common,
                    )
                    diagnostics = {
                        "policy_chosen_sum_lp": float(policy_chosen.detach().mean().cpu()),
                        "policy_rejected_sum_lp": float(policy_rejected_mean.detach().cpu()),
                        "reference_chosen_sum_lp": float(reference_chosen.mean().cpu()),
                        "reference_rejected_sum_lp": float(reference_rejected_mean.cpu()),
                        "pair_margin_mean": float(pair_margin.detach().mean().cpu()),
                        "pair_margin_p10": self._dpo_quantile(pair_margin, 0.10),
                        "pair_margin_p50": self._dpo_quantile(pair_margin, 0.50),
                        "pair_margin_p90": self._dpo_quantile(pair_margin, 0.90),
                        "preference_accuracy": float(
                            (pair_margin.detach() > 0.0).float().mean().cpu()
                        ),
                        "logit_saturation_fraction": float(
                            (logits.detach().abs() >= 10.0).float().mean().cpu()
                        ),
                        "unique_negative_count_mean": float(
                            packed["unique_counts"].float().mean().cpu()
                        ),
                        "raw_bank_count_mean": float(
                            packed["raw_bank_counts"].float().mean().cpu()
                        ),
                        "duplicates_removed_mean": float(
                            (packed["raw_bank_counts"] - packed["unique_counts"])
                            .float()
                            .mean()
                            .cpu()
                        ),
                    }
                    for key, value in diagnostics.items():
                        diagnostic_totals[key] += value

                if abort_update:
                    break

                gradient_norm = self.torch.nn.utils.clip_grad_norm_(
                    policy_parameters,
                    float(effective["training"]["max_grad_norm"]),
                )
                if not bool(self.torch.isfinite(gradient_norm)):
                    numerical_failure = f"nonfinite_gradient_at_step_{update}"
                    stop_reason = numerical_failure
                    break
                sample_update_norm = update % log_every == 0 or update == updates
                before = (
                    [parameter.detach().float().cpu().clone() for parameter in policy_parameters]
                    if sample_update_norm
                    else []
                )
                if not arena.optimizer_step_with_last_finite_guard(optimizer, policy_parameters):
                    numerical_failure = f"nonfinite_parameters_at_step_{update}"
                    stop_reason = numerical_failure
                    break
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                terminal_step = update
                last_finite_step = update
                update_norm = (
                    scan_trainer._parameter_update_norm(before, policy_parameters)
                    if sample_update_norm
                    else None
                )
                if update_norm is not None:
                    optimizer_update_norms.append(float(update_norm))
                if update % log_every == 0 or update == updates:
                    append_jsonl(
                        training_path,
                        {
                            "update": update,
                            "dpo_beta": beta,
                            "dpo_pair_loss": loss_total / accumulation,
                            **{
                                key: value / accumulation
                                for key, value in diagnostic_totals.items()
                            },
                            "raw_gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                            "optimizer_update_norm": update_norm,
                            "initial_pair_margin_max_abs": initial_pair_margin_max_abs,
                            "reference_role": "exact_frozen_initial_policy",
                            "label_smoothing": 0.0,
                            "test_data_used": False,
                        },
                    )
                if not engineering_liveness and (update % eval_every == 0 or update == updates):
                    evaluate(update)

            activate_policy()
            terminal_policy_sha256 = self._parameter_sequence_sha256(policy_parameters)
            reference_terminal_sha256 = self._parameter_sequence_sha256(reference_parameters)
            if reference_terminal_sha256 != reference_initial_sha256:
                raise RuntimeError("Frozen DPO reference changed during policy optimization")
            policy_parameters_changed = terminal_policy_sha256 != policy_initial_sha256
            final_adapter_dir = last_finite_dir if numerical_failure else terminal_dir
            model.save_pretrained(final_adapter_dir, safe_serialization=True)
            tokenizer.save_pretrained(final_adapter_dir)
            terminal_identity = model_identity(base_model_path, str(final_adapter_dir))["adapter"]
            if numerical_failure is not None:
                append_jsonl(
                    training_path,
                    {
                        "update": terminal_step,
                        "numerical_failure": numerical_failure,
                        "stop_reason": stop_reason,
                        "last_finite_step": last_finite_step,
                        "reference_role": "exact_frozen_initial_policy",
                        "test_data_used": False,
                    },
                )

            if engineering_liveness:
                summary: dict[str, Any] = {
                    "engineering_liveness": True,
                }
                scientific_status = "not_run"
            else:
                evaluations = read_jsonl(evaluation_path)
                if numerical_failure is not None:
                    summary = {}
                elif zero_beta_control:
                    if len(evaluations) != 1 or int(evaluations[0]["update"]) != 0:
                        raise RuntimeError(
                            "SFT-only beta-zero control must have exactly one step-0 evaluation"
                        )
                    static = evaluations[0]
                    summary = {
                        "late_window_updates": [
                            int(value)
                            for value in config["training"]["late_window_updates"]
                        ],
                        "validation_late_window_pass8_mean": float(static["pass8"]),
                        "validation_late_window_greedy_mean": float(
                            static["greedy_success"]
                        ),
                        "validation_late_window_valid_mean": float(
                            static["greedy_valid_rate"]
                        ),
                        "validation_terminal_pass8": float(static["pass8"]),
                        "validation_terminal_greedy": float(
                            static["greedy_success"]
                        ),
                        "validation_terminal_greedy_valid_rate": float(
                            static["greedy_valid_rate"]
                        ),
                        "validation_terminal_sampled_valid_rate": (
                            None
                            if static.get("sampled_valid_rate") is None
                            else float(static["sampled_valid_rate"])
                        ),
                        "supplementary_best_step": 0,
                        "supplementary_best_pass8": float(static["pass8"]),
                        "supplementary_best_greedy": float(
                            static["greedy_success"]
                        ),
                        "static_control_single_evaluation_step": 0,
                        "static_control_metric_reuse": (
                            "terminal_policy_equals_initial_policy_no_optimizer_updates"
                        ),
                    }
                else:
                    summary = self._summarize_evaluations(evaluations, config)
                scientific_status = "pilot"
            result = {
                **identity,
                **summary,
                "beta": beta,
                "dpo_beta": beta,
                "objective_formula": (
                    "mean_prompt(mean_unique_negative(softplus(-beta*((logpi_chosen-logpi_rejected)-"
                    "(logref_chosen-logref_rejected)))))"
                ),
                "sequence_log_probability": "full_completion_summed_log_probability",
                "chosen_completion": "oracle_completion",
                "rejected_completions": "all_unique_verifier_wrong_completions",
                "pair_aggregation": "mean_unique_negative_within_prompt_then_mean_prompts",
                "label_smoothing": 0.0,
                "reference_role": "exact_frozen_initial_policy",
                "reference_trainable": False,
                "control_role": (
                    "sft_only_no_dpo_update" if zero_beta_control else None
                ),
                "zero_beta_control": zero_beta_control,
                "initial_pair_margin_max_abs": initial_pair_margin_max_abs,
                "initial_pair_margin_probe": (
                    "not_run_exact_policy_reference_state_hashes_match"
                    if zero_beta_control
                    else "measured_on_first_dpo_batch"
                ),
                "initial_pair_margin_max_abs_tolerance": tolerance,
                "policy_initial_state_sha256": policy_initial_sha256,
                "reference_initial_state_sha256": reference_initial_sha256,
                "reference_terminal_state_sha256": reference_terminal_sha256,
                "terminal_trainable_state_sha256": terminal_policy_sha256,
                "initialization_state_sha256": policy_initial_sha256,
                "policy_parameters_changed": policy_parameters_changed,
                "terminal_checkpoint_kind": (
                    "last_finite" if numerical_failure else "terminal"
                ),
                "terminal_adapter": str(final_adapter_dir.resolve()),
                "terminal_adapter_identity": terminal_identity,
                "best_adapter": (
                    str(best_dir.resolve()) if best_dir.is_dir() else str(final_adapter_dir.resolve())
                ),
                "training_metrics": str(training_path.resolve()),
                "evaluation_metrics": (
                    str(evaluation_path.resolve()) if evaluation_path.is_file() else None
                ),
                "canonical_dispatch_verified": True,
                "finite_old_core_updates": (
                    numerical_failure is None and not zero_beta_control
                ),
                "optimizer_update_norm": (
                    None
                    if zero_beta_control
                    else (
                        min(value for value in optimizer_update_norms if math.isfinite(value))
                        if optimizer_update_norms
                        else 0.0
                    )
                ),
                "optimizer_updates": terminal_step,
                "terminal_step": terminal_step,
                "optimizer_updates_requested": training_updates,
                "configured_positive_beta_optimizer_updates": updates,
                "training_seed_base": training_seed_base,
                "dpo_seed_offset": int(cell.seed),
                "training_seed_applied": not zero_beta_control,
                "effective_training_seed": (
                    None if zero_beta_control else seed
                ),
                "last_finite_step": last_finite_step,
                "numerical_failure": numerical_failure,
                "stop_reason": stop_reason,
                "nan_inf_failure": self._is_nan_inf_numerical_failure(numerical_failure),
                "evaluation_status": (
                    "complete" if numerical_failure is None else "incomplete"
                ),
                "test_partition_accessed": False,
                "complete": numerical_failure is None,
                "scientific_status": scientific_status,
            }
            if not engineering_liveness and numerical_failure is None:
                result.update(
                    {
                        "best_step": summary["supplementary_best_step"],
                        "terminal_step": terminal_step,
                        "validation_best_pass8": summary["supplementary_best_pass8"],
                        "validation_terminal_pass8": summary["validation_terminal_pass8"],
                        "validation_best_greedy": summary["supplementary_best_greedy"],
                        "validation_terminal_greedy": summary["validation_terminal_greedy"],
                        "validation_best_greedy_valid_rate": max(
                            float(row["greedy_valid_rate"]) for row in evaluations
                        ),
                        "validation_terminal_greedy_valid_rate": summary[
                            "validation_terminal_greedy_valid_rate"
                        ],
                    }
                )
            if not engineering_liveness:
                summary_path = cell_root / "summary.json"
                atomic_json(summary_path, result)
                result["canonical_summary"] = str(summary_path.resolve())
                result["canonical_summary_sha256"] = sha256_file(summary_path)
                result["canonical_output"] = str(cell_root.resolve())
            atomic_json(manifest_path, result)
            return result
        finally:
            arena.clean_expression = original_clean_expression
            arena.SYSTEM_PROMPT = original_system_prompt
            arena.evaluate_rows = original_evaluate_rows
            if "model" in locals():
                del model
            gc.collect()
            if self.torch is not None and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def _cmd_dpo_liveness(
        self,
        config: Mapping[str, Any],
        config_path: Path,
        output_root: Path,
        *,
        inputs: Mapping[str, TaskInputs],
        splits: Mapping[str, Any],
        base_model_path: str,
        task: str,
        force: bool,
    ) -> dict[str, Any]:
        if task != str(config["dpo"]["liveness_task"]):
            raise RuntimeError("DPO liveness must use the configured transfer-task anchor")
        beta = float(config["dpo"]["liveness_beta"])
        values = experiment_config.task_method_values(config, task, method=self.METHOD_DPO)
        if beta not in values:
            raise RuntimeError("DPO liveness_beta must be one configured beta point")
        cell = self._coldstart_method_cell(
            task,
            self.METHOD_DPO,
            experiment_config.task_transfer_seeds(config)[0],
            "liveness",
            beta,
            lambda_only=False,
            dpo_initialization=str(config["dpo"]["initialization_mode"]),
        )
        result = self.train_cell(
            cell,
            inputs=inputs[task],
            split_manifest=splits,
            base_model_path=base_model_path,
            config=config,
            output_root=output_root,
            force=force,
            updates_override=2,
            engineering_liveness=True,
        )
        reload_result = self._verify_fresh_process_adapter_reload(
            config_path,
            output_root,
            base_model_path=base_model_path,
            adapter_path=Path(result["terminal_adapter"]),
            expected_adapter_identity=result["terminal_adapter_identity"],
            require_base_identity=False,
            label="DPO",
        )
        if not math.isfinite(float(result.get("optimizer_update_norm", 0.0))) or float(
            result.get("optimizer_update_norm", 0.0)
        ) <= 0.0:
            raise RuntimeError("DPO liveness did not perform a finite nonzero optimizer update")
        result.update(
            {
                "reload_gate_passed": True,
                "adapter_weight_changed": True,
                "fresh_process_reload_passed": True,
                "liveness_parent_process_id": os.getpid(),
                "reload_process_id": int(reload_result["process_id"]),
                "terminal_adapter_weight_sha256": sha256_file(self._adapter_weight_file(Path(result["terminal_adapter"]))),
                "evaluation_status": "complete",
            }
        )
        atomic_json(output_root / "liveness" / cell.key / "cell_manifest.json", result)
        return result

    def _train_canonical_cold_cell(
        self,
        cell: Any,
        *,
        inputs: TaskInputs,
        split_manifest: Mapping[str, Any],
        base_model_path: str,
        config: Mapping[str, Any],
        output_root: Path,
        force: bool,
        root_name: str = "cells",
        base_config_override: Path | None = None,
    ) -> dict[str, Any]:
        """Dispatch one cell to the exact paper runtime; this owns no loss math."""

        if not self._is_coldstart(config):
            raise RuntimeError("Canonical cold dispatch is cold-profile only")
        modules = self._canonical_cold_modules(config)
        arena = modules["arena"]
        record = self._canonical_task_record(split_manifest, cell.task)
        calibration = self._load_verified_canonical_calibration(
            cell.task,
            split_manifest=split_manifest,
            base_model_path=base_model_path,
            config=config,
            output_root=output_root,
            error_prefix="Paper",
        )

        bank = Path(str(record["train"]))
        validation = Path(str(record["validation"]))
        base_config_path = base_config_override or Path(str(record["base_config"]))
        method_spec = self._method_spec(cell.method)
        paper_runtime = method_spec.paper_runtime
        if paper_runtime is None:
            raise RuntimeError(
                f"{cell.method} does not use canonical paper cell parameters"
            )
        grid_path, grid_source_path = self._paper_grid_for_cell(config, record, cell)
        modules = self._activate_paper_grid_modules(modules, grid_source_path)
        arena = modules["arena"]
        runtime = modules["paper_runtime"]
        effective_runtime = experiment_config.effective_coldstart_runtime(config, cell.task)
        base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
        if not isinstance(base_config, dict):
            raise TypeError("Paper base config root must be a mapping")
        identity = self._cell_identity(
            cell,
            inputs=inputs,
            split_manifest=split_manifest,
            base_model_path=base_model_path,
            config=config,
            calibration=calibration,
        )
        identity.update(
            {
                "canonical_source_git_blob_shas": dict(
                    config["canonical_coldstart"]["expected_git_blob_shas"]
                ),
                "canonical_dispatch": (
                    "countdown_e8_alpha1_highc_scan_runtime.worker"
                    if cell.task == "countdown"
                    else "countdown_e8_alpha1_c_scan_trainer.train_cell"
                ),
                "paper_formula": paper_runtime.formula,
                "paper_grid_config": str(grid_path.resolve()),
                "paper_grid_config_sha256": sha256_file(grid_path),
                "paper_grid_source": str(grid_source_path.resolve()),
                "paper_grid_source_sha256": sha256_file(grid_source_path),
                "paper_base_config": str(base_config_path.resolve()),
                "paper_base_config_sha256": sha256_file(base_config_path),
                "all_unique_negatives": True,
                "near_far_selection": False,
                "gradient_rms_matching": False,
                "task_runtime_contract": dict(config["task_runtime"][cell.task]),
            }
        )
        if not experiment_config.is_historical_coldstart_config(config):
            identity["effective_runtime"] = effective_runtime
            identity["legacy_runtime_bridge"] = self._runtime_bridge_contract(effective_runtime)
        identity["identity_hash"] = stable_hash(identity)

        cell_root, manifest_path, reusable = self._prepare_cell_output(
            output_root,
            root_name=root_name,
            cell=cell,
            identity=identity,
            force=force,
            mismatch_prefix="Existing paper cell identity mismatch",
            existing_prefix="Cell output exists without a reusable manifest",
            unsafe_prefix="Refusing unsafe paper cell removal",
        )
        if reusable is not None:
            return reusable
        canonical_output = cell_root / "canonical"

        evaluator = None
        if cell.task != "countdown":
            validation_rows = read_jsonl(validation)
            task_adapter, instances = _load_task_adapter_and_instances(
                cell.task,
                inputs=inputs,
                validation_rows=validation_rows,
            )
            evaluator = self._canonical_environment_evaluator(
                arena=arena,
                task_adapter=task_adapter,
                instances=instances,
                greedy_prompt_rows=int(config["task_runtime"][cell.task]["greedy_prompt_rows"]),
                passk_prompt_rows=int(config["task_runtime"][cell.task]["passk_prompt_rows"]),
                sampling_temperature=float(effective_runtime["evaluation"]["sampling_temperature"]),
                top_p=float(effective_runtime["evaluation"]["top_p"]),
            )

        scan_trainer = modules["scan_trainer"]
        paper_common = modules["paper_common"]

        @contextmanager
        def task_interface() -> Any:
            original_evaluate_rows = arena.evaluate_rows
            original_clean_expression = arena.clean_expression
            original_system_prompt = arena.SYSTEM_PROMPT
            original_completion_stats = arena.completion_stats
            original_trainer_evaluate = scan_trainer._evaluate_validation
            try:
                if evaluator is None:

                    def configured_evaluate(**kwargs: Any) -> dict[str, Any]:
                        kwargs["pass64_enabled"] = 64 in {
                            int(value)
                            for value in effective_runtime["evaluation"]["auxiliary_pass_ks"]
                        }
                        return original_trainer_evaluate(**kwargs)

                    scan_trainer._evaluate_validation = configured_evaluate
                if evaluator is not None:
                    arena.evaluate_rows = evaluator
                    arena.SYSTEM_PROMPT = self.TRANSFER_SYSTEM_PROMPT
                    arena.completion_stats = lambda model, batch: {
                        "seq_lp": -arena.sequence_surprisal_only(model, batch),
                        "lengths": (batch["labels"] != -100).sum(dim=1),
                    }
                    # Non-arithmetic task outputs are already canonicalized by the P0
                    # verifier. Arithmetic-only cleanup would corrupt structured outputs.
                    arena.clean_expression = lambda value: str(value)

                    def configured_evaluate(**kwargs: Any) -> dict[str, Any]:
                        kwargs["pass64_enabled"] = 64 in {
                            int(value)
                            for value in effective_runtime["evaluation"]["auxiliary_pass_ks"]
                        }
                        row = original_trainer_evaluate(**kwargs)
                        sampled_valid_rate = getattr(
                            evaluator, "_last_primary_sampled_valid_rate", None
                        )
                        if sampled_valid_rate is None:
                            raise RuntimeError(
                                "Transfer evaluator did not report primary sampled validity"
                            )
                        row["val_sampled_valid_rate"] = float(sampled_valid_rate)
                        return row

                    scan_trainer._evaluate_validation = configured_evaluate
                yield
            finally:
                arena.evaluate_rows = original_evaluate_rows
                arena.clean_expression = original_clean_expression
                arena.SYSTEM_PROMPT = original_system_prompt
                arena.completion_stats = original_completion_stats
                scan_trainer._evaluate_validation = original_trainer_evaluate

        paper_family, alpha, coefficient = paper_runtime.cell_parameters(cell)
        with (
            self._legacy_paper_runtime_bridge(
                modules,
                effective_runtime,
                grid_path=grid_path,
                grid_source_path=grid_source_path,
            ),
            task_interface(),
        ):
            if cell.task == "countdown":
                returncode = runtime.worker(
                    argparse.Namespace(
                        family=paper_family,
                        alpha=alpha,
                        c=coefficient,
                        seed_offset=int(cell.seed),
                        output_dir=str(canonical_output),
                        model_path=str(Path(base_model_path).resolve()),
                        bank=str(bank),
                        val=str(validation),
                        base_config=str(base_config_path),
                        grid_config=str(grid_path),
                    )
                )
                if int(returncode) != 0:
                    raise RuntimeError(f"Paper runtime failed for {cell.key}")
            else:
                paper_cell = paper_common.Cell(
                    alpha=alpha,
                    coefficient=coefficient,
                    seed_offset=int(cell.seed),
                    family=paper_family,
                )
                scan_trainer.train_cell(
                    cell=paper_cell,
                    model_path=Path(base_model_path).resolve(),
                    bank=bank,
                    val=validation,
                    base_config_path=base_config_path,
                    grid_config_path=grid_path,
                    output_dir=canonical_output,
                    repo=self._repo_root(),
                    smoke=False,
                )

        canonical_summary_path = canonical_output / "summary.json"
        canonical_summary = json.loads(canonical_summary_path.read_text(encoding="utf-8"))
        metrics_path = canonical_output / "metrics.csv"
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        evaluations = [
            {
                "update": int(row["step"]),
                "pass8": float(row["val_pass_at_8"]),
                "greedy_success": float(row["val_greedy"]),
                "greedy_valid_rate": float(row["val_valid_rate"]),
                "sampled_valid_rate": (
                    None
                    if row.get("val_sampled_valid_rate") in (None, "")
                    else float(row["val_sampled_valid_rate"])
                ),
            }
            for row in metric_rows
        ]
        numerical_failure = canonical_summary.get("numerical_failure")
        metrics_summary = (
            self._summarize_evaluations(evaluations, config)
            if numerical_failure is None
            else {}
        )
        best_adapter = canonical_output / "best_pass8_adapter"
        terminal_adapter = canonical_output / (
            "last_finite_adapter" if numerical_failure else "terminal_adapter"
        )
        result = {
            **identity,
            **metrics_summary,
            "delta_v": cell.delta_v,
            "beta": cell.beta,
            "canonical_summary": str(canonical_summary_path.resolve()),
            "canonical_summary_sha256": sha256_file(canonical_summary_path),
            "canonical_output": str(canonical_output.resolve()),
            "canonical_training_metrics": str(metrics_path.resolve()),
            "training_metrics": str(metrics_path.resolve()),
            "best_adapter": str(best_adapter.resolve()),
            "terminal_adapter": str(terminal_adapter.resolve()),
            "terminal_adapter_identity": model_identity(base_model_path, str(terminal_adapter))[
                "adapter"
            ],
            "canonical_dispatch_verified": True,
            "countdown_protocol_exact": cell.task == "countdown",
            "task_interface_adapter_only": cell.task != "countdown",
            "reference_remoteness_bank_identity_hash": record.get(
                "reference_remoteness_bank_identity_hash"
            ),
            "static_reference_rank_enters_training_weight": False,
            "current_policy_surprisal_recomputed_each_update": True,
            "adapter_path_argument": None,
            "sft_adapter_path_argument": None,
            "initialization_optimizer_updates": 0,
            "best_step": metrics_summary.get("supplementary_best_step"),
            "terminal_step": canonical_summary.get("terminal_step"),
            "stop_reason": canonical_summary.get("stop_reason"),
            "validation_best_pass8": metrics_summary.get("supplementary_best_pass8"),
            "validation_terminal_pass8": metrics_summary.get("validation_terminal_pass8"),
            "validation_best_greedy": metrics_summary.get("supplementary_best_greedy"),
            "validation_terminal_greedy": metrics_summary.get("validation_terminal_greedy"),
            "validation_best_greedy_valid_rate": (
                max(float(row["greedy_valid_rate"]) for row in evaluations)
                if numerical_failure is None and evaluations
                else None
            ),
            "validation_terminal_greedy_valid_rate": metrics_summary.get(
                "validation_terminal_greedy_valid_rate"
            ),
            "validation_terminal_sampled_valid_rate": metrics_summary.get(
                "validation_terminal_sampled_valid_rate"
            ),
            "optimizer_updates": int(canonical_summary.get("terminal_step") or 0),
            "numerical_failure": numerical_failure,
            "nan_inf_failure": self._is_nan_inf_numerical_failure(numerical_failure),
            "evaluation_status": (
                "complete" if evaluations and numerical_failure is None else "incomplete"
            ),
            "test_partition_accessed": False,
            "complete": bool(evaluations and numerical_failure is None),
            "scientific_status": "pilot",
        }
        atomic_json(manifest_path, result)
        return result

    def _cmd_canonical_cold_liveness(
        self,
        config: Mapping[str, Any],
        config_path: Path,
        output_root: Path,
        *,
        inputs: TaskInputs,
        splits: Mapping[str, Any],
        base_model_path: str,
        force: bool,
        method: str | None = None,
    ) -> dict[str, Any]:
        modules = self._canonical_cold_modules(config)
        record = self._canonical_task_record(splits, "countdown")
        method = method or self._coldstart_method(config)
        paper_runtime = self._method_spec(method).paper_runtime
        if paper_runtime is None:
            raise RuntimeError(f"{method} does not use canonical paper-runtime liveness")
        grid_path = paper_runtime.liveness_grid(config, record)
        grid_path = self._method_liveness_grid(grid_path, method, output_root)
        modules = self._activate_paper_grid_modules(modules, grid_path)
        runtime = modules["paper_runtime"]
        cell = self._canonical_cold_liveness_cell(grid_path)
        smoke_name = (
            f"paper_runtime_smoke_{method}" if self._is_method_matrix(config) else "paper_runtime_smoke"
        )
        smoke_root = output_root / "liveness" / smoke_name
        if force and smoke_root.exists():
            shutil.rmtree(smoke_root)
        returncode = runtime.smoke(
            argparse.Namespace(
                model_path=str(Path(base_model_path).resolve()),
                bank=str(Path(str(record["train"])).resolve()),
                val=str(Path(str(record["validation"])).resolve()),
                base_config=str(Path(str(record["base_config"])).resolve()),
                grid_config=str(grid_path.resolve()),
                work_dir=str(smoke_root.resolve()),
            )
        )
        gate = json.loads((smoke_root / "SMOKE_GATE.json").read_text(encoding="utf-8"))
        if int(returncode) != 0 or gate.get("status") != "PASS":
            raise RuntimeError("Paper-runtime two-update smoke gate failed")
        summary_path = Path(str(gate["summary"]))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        diagnostics_path = Path(str(summary["diagnostic_files"]["training"]))
        diagnostics = read_jsonl(diagnostics_path)
        optimizer_update_norms = [
            float(row["optimizer_update_norm"])
            for row in diagnostics
            if row.get("optimizer_update_norm") is not None
        ]
        if (
            int(summary.get("terminal_step") or -1) != 2
            or summary.get("numerical_failure") is not None
            or not optimizer_update_norms
            or not all(math.isfinite(value) and value > 0.0 for value in optimizer_update_norms)
        ):
            raise RuntimeError("Paper-runtime liveness did not perform two finite optimizer updates")
        canonical_output = summary_path.parent
        terminal_adapter = canonical_output / "terminal_adapter"
        terminal_hash = sha256_file(self._adapter_weight_file(terminal_adapter))

        reload_result = self._verify_fresh_process_adapter_reload(
            config_path,
            output_root,
            base_model_path=base_model_path,
            adapter_path=terminal_adapter,
            expected_adapter_identity=model_identity(base_model_path, str(terminal_adapter))["adapter"],
            require_base_identity=False,
            label="canonical",
        )
        calibration = json.loads(
            (output_root / "calibration" / "countdown.json").read_text(encoding="utf-8")
        )
        result = {
            **self._cell_identity(
                cell,
                inputs=inputs,
                split_manifest=splits,
                base_model_path=base_model_path,
                config=config,
                calibration=calibration,
            ),
            "canonical_dispatch": "countdown_e8_alpha1_highc_scan_runtime.smoke",
            "canonical_dispatch_verified": True,
            "canonical_summary": str(summary_path.resolve()),
            "canonical_summary_sha256": sha256_file(summary_path),
            "terminal_adapter": str(terminal_adapter.resolve()),
            "terminal_adapter_identity": model_identity(base_model_path, str(terminal_adapter))[
                "adapter"
            ],
            "engineering_liveness": True,
            "optimizer_updates": 2,
            "optimizer_update_norm": min(optimizer_update_norms),
            "terminal_adapter_weight_sha256": terminal_hash,
            "adapter_weight_changed": True,
            "finite_old_core_updates": True,
            "reload_gate_passed": True,
            "fresh_process_reload_passed": True,
            "liveness_parent_process_id": os.getpid(),
            "reload_process_id": int(reload_result["process_id"]),
            "nan_inf_failure": False,
            "evaluation_status": "complete",
            "complete": True,
            "scientific_status": "not_run",
        }
        result["identity_hash"] = stable_hash(result)
        atomic_json(
            output_root / "liveness" / cell.key / "cell_manifest.json",
            result,
        )
        return result


def build_bridge(bindings: CanonicalBridgeBindings) -> CanonicalBridge:
    """Build one call-scoped bridge from the explicit implementation object."""
    impl = _CanonicalBridgeImpl(bindings)
    return CanonicalBridge(
        _paper_grid_paths_exponential=impl._paper_grid_paths_exponential,
        _paper_params_reciprocal=impl._paper_params_reciprocal,
        _paper_params_exponential=impl._paper_params_exponential,
        _paper_params_asymre=impl._paper_params_asymre,
        _paper_params_topr=impl._paper_params_topr,
        _canonical_cold_modules=impl._canonical_cold_modules,
        _activate_paper_grid_modules=impl._activate_paper_grid_modules,
        _canonical_task_record=impl._canonical_task_record,
        _paper_grid_for_cell=impl._paper_grid_for_cell,
        _canonical_environment_evaluator=impl._canonical_environment_evaluator,
        _runtime_bridge_contract=impl._runtime_bridge_contract,
        _legacy_arena_runtime_bridge=impl._legacy_arena_runtime_bridge,
        _validated_runtime_grid=impl._validated_runtime_grid,
        _legacy_paper_runtime_bridge=impl._legacy_paper_runtime_bridge,
        _canonical_baseline_grid_identity=impl._canonical_baseline_grid_identity,
        _normalized_adapter_config_sequence=impl._normalized_adapter_config_sequence,
        _dpo_shared_sft_adapter_identity=impl._dpo_shared_sft_adapter_identity,
        _parameter_sequence_sha256=impl._parameter_sequence_sha256,
        _dpo_prompt_balanced_mean=impl._dpo_prompt_balanced_mean,
        _dpo_quantile=impl._dpo_quantile,
        _is_nan_inf_numerical_failure=impl._is_nan_inf_numerical_failure,
        _load_verified_canonical_calibration=impl._load_verified_canonical_calibration,
        _train_canonical_dpo_transfer_cell=impl._train_canonical_dpo_transfer_cell,
        _cmd_dpo_liveness=impl._cmd_dpo_liveness,
        _train_canonical_cold_cell=impl._train_canonical_cold_cell,
        _cmd_canonical_cold_liveness=impl._cmd_canonical_cold_liveness,
    )
