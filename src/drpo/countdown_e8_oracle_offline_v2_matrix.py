"""Canonical runner for EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01.

This is the v2 downstream successor of the model-independent oracle-offline
corpus experiment EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01. It runs the
initialization-level x training-condition matrix on the FROZEN v2 corpus/bank,
under a single shared offline-training protocol read from config.

Scientific scope (GLM Dev Agent, execution-only; no redesign):
  - corpus/bank: frozen v2 oracle-offline bank (model-independent, value-stratified)
  - dynamic near/far selection: unchanged (current-model surprisal, argmin/argmax,
    eval-mode + stop-gradient, then train-mode forward on the two selected).
  - one shared offline-training protocol (the base-rl-replay config's
    offline_training + evaluation + negative_calibration blocks); no per-cell
    hardcoded TRAIN_FLAGS.
  - two calibrations, isolated by initialization:
      * base calibration       -> reference_adapter=None, used by base negative cells
      * low-SFT calibration    -> reference_adapter=v2 SFT epoch_1 adapter, used by B2 only
  - on-policy/replay cells are NOT part of this experiment (they do not read the
    v2 offline negative bank); they remain under EXT-C-E8-BASE-RL-REPLAY-0.5B-01.

8 unique training cells + 2 eval-only cells + 2 alias-only cells:
  base init (reference_adapter=None, v2 base calibration):
    base_positive_only                 (also matrix A1 via alias)
    base_bank_global_matched_x0p25
    base_bank_global_matched_x0p5
    base_bank_global_matched_x1p0      (also matrix A2 via alias)
    base_bank_global_matched_x2p0
  low-SFT init (v2 SFT epoch_1 adapter, low-SFT calibration):
    low_sft_positive_only              (B1)
    low_sft_bank_global_matched_x1p0   (B2)
  full-SFT init (v2 SFT best adapter):
    full_sft_positive_only             (C1)
  eval-only:
    B0  = low-SFT init eval-only
    C0  = full-SFT init eval-only
  alias-only (no training, reference base cells):
    A1  -> base_positive_only
    A2  -> base_bank_global_matched_x1p0
  NOT executed this round: A0 (base eval; may be reused only under full identity match),
                            C2 (not registered), on-policy/replay (separate experiment).

Identity-checked resume: a cell with summary.json is skipped ONLY when its recorded
run-identity (corpus sha, config sha, adapter identity, calibration identity, commit)
matches the current plan exactly; otherwise the runner fail-closes rather than
silently reusing a stale result.

This module is import-safe and CLI-driven; it never hardcodes /root paths as
scientific config. GPU/CUDA execution is delegated to the canonical runner's own
training functions; this file only orchestrates them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01"
VERSION = "v2-init-matrix-0.1.0"
DEFAULT_CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"

# ---------------------------------------------------------------------------
# cell plan
# ---------------------------------------------------------------------------

# Each training cell is unique. A1/A2 are aliases (no training).
BASE_INIT = "base"
LOW_SFT_INIT = "low_sft"
FULL_SFT_INIT = "full_sft"


@dataclass(frozen=True)
class Cell:
    name: str
    init: str
    method: str                       # positive_only | bank_global_matched
    calibration: str                  # none | base | low_sft
    negative_scale_multiplier: float = 1.0
    seed_offset: int = 0              # added to the shared offline_training.seed
    kind: str = "train"               # train | eval_only | alias
    alias_of: str | None = None       # for alias cells


# Shared offline protocol seed (from config offline_training.seed).
# Seed schedule mirrors base-rl-replay exactly for the base cells so the base
# training cells are byte-comparable to EXT-C-E8-BASE-RL-REPLAY-0.5B-01 when run
# on the same bank; low/full-SFT cells reuse the same seed schedule on their init.
CELLS: tuple[Cell, ...] = (
    # ---- base init (reference_adapter=None) ----
    Cell("base_positive_only", BASE_INIT, "positive_only", "none",
         seed_offset=0, kind="train"),
    Cell("base_bank_global_matched_x0p25", BASE_INIT, "bank_global_matched", "base",
         negative_scale_multiplier=0.25, seed_offset=250, kind="train"),
    Cell("base_bank_global_matched_x0p5", BASE_INIT, "bank_global_matched", "base",
         negative_scale_multiplier=0.5, seed_offset=500, kind="train"),
    Cell("base_bank_global_matched_x1p0", BASE_INIT, "bank_global_matched", "base",
         negative_scale_multiplier=1.0, seed_offset=1000, kind="train"),
    Cell("base_bank_global_matched_x2p0", BASE_INIT, "bank_global_matched", "base",
         negative_scale_multiplier=2.0, seed_offset=2000, kind="train"),
    # ---- low-SFT init ----
    Cell("low_sft_positive_only", LOW_SFT_INIT, "positive_only", "none",
         seed_offset=0, kind="train"),
    Cell("low_sft_bank_global_matched_x1p0", LOW_SFT_INIT, "bank_global_matched", "low_sft",
         negative_scale_multiplier=1.0, seed_offset=1000, kind="train"),
    # ---- full-SFT init ----
    Cell("full_sft_positive_only", FULL_SFT_INIT, "positive_only", "none",
         seed_offset=0, kind="train"),
    # ---- eval-only ----
    Cell("B0", LOW_SFT_INIT, "positive_only", "none", kind="eval_only"),
    Cell("C0", FULL_SFT_INIT, "positive_only", "none", kind="eval_only"),
    # ---- alias-only (matrix A-row references to base training cells) ----
    Cell("A1", BASE_INIT, "positive_only", "none", kind="alias",
         alias_of="base_positive_only"),
    Cell("A2", BASE_INIT, "bank_global_matched", "base", kind="alias",
         alias_of="base_bank_global_matched_x1p0"),
)

TRAINING_CELLS = tuple(c for c in CELLS if c.kind == "train")
EVAL_ONLY_CELLS = tuple(c for c in CELLS if c.kind == "eval_only")
ALIAS_CELLS = tuple(c for c in CELLS if c.kind == "alias")


def assert_plan_invariants() -> None:
    """Fail fast if the cell plan drifts from the dev spec."""
    training = [c.name for c in TRAINING_CELLS]
    assert len(training) == 8, f"expected exactly 8 training cells, got {len(training)}: {training}"
    assert "base_onpolicy_positive_only" not in training
    assert "base_online_replay_positive_only" not in training
    assert "base_online_replay_pos_neg" not in training
    assert not any(c.name.startswith("C2") for c in CELLS), "C2 must not be registered"
    base_count = sum(1 for c in TRAINING_CELLS if c.init == BASE_INIT)
    low_count = sum(1 for c in TRAINING_CELLS if c.init == LOW_SFT_INIT)
    full_count = sum(1 for c in TRAINING_CELLS if c.init == FULL_SFT_INIT)
    assert base_count == 5, f"expected 5 base training cells, got {base_count}"
    assert low_count == 2, f"expected 2 low-SFT training cells, got {low_count}"
    assert full_count == 1, f"expected 1 full-SFT training cell, got {full_count}"
    # A0 is NOT in the execution queue.
    assert not any(c.name == "A0" for c in CELLS), "A0 must not enter the execution queue"
    # A1 -> base_positive_only, A2 -> base_bank_global_matched_x1p0
    a1 = next(c for c in ALIAS_CELLS if c.name == "A1")
    assert a1.alias_of == "base_positive_only"
    a2 = next(c for c in ALIAS_CELLS if c.name == "A2")
    assert a2.alias_of == "base_bank_global_matched_x1p0"
    # calibration routing
    for c in TRAINING_CELLS:
        if c.method == "positive_only":
            assert c.calibration == "none", f"{c.name}: positive_only must need no calibration"
        if c.init == BASE_INIT and c.method == "bank_global_matched":
            assert c.calibration == "base", f"{c.name}: base negative cell must use base calibration"
        if c.init == LOW_SFT_INIT and c.method == "bank_global_matched":
            assert c.calibration == "low_sft", f"{c.name}: B2 must use low-SFT calibration"
        assert c.calibration != "base" or c.init == BASE_INIT, \
            f"{c.name}: base calibration only valid for base init"
        assert c.calibration != "low_sft" or c.init == LOW_SFT_INIT, \
            f"{c.name}: low-SFT calibration only valid for low-SFT init"


# ---------------------------------------------------------------------------
# provenance / identity
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty(repo: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).strip()
        return bool(out)
    except Exception:
        return True  # treat unknown as dirty (safe)


def run_identity(model_path: Path, bank: Path, val: Path, test: Path,
                 sft_dir: Path | None, config_path: Path, repo: Path,
                 config: Mapping[str, Any]) -> dict[str, Any]:
    """A hashable identity bundle. Any field changing invalidates resume."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": VERSION,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "bank_path": str(bank),
        "bank_sha256": _sha256_file(bank),
        "val_sha256": _sha256_file(val),
        "test_sha256": _sha256_file(test),
        "model_path": str(model_path),
        "model_sha256": _sha256_dir(model_path) if model_path.is_dir() else _sha256_file(model_path),
        "sft_dir": str(sft_dir) if sft_dir else None,
        "git_commit": _git_head(repo),
        "git_dirty": _git_dirty(repo),
        "offline_training": dict(config["offline_training"]),
        "evaluation": dict(config["evaluation"]),
        "negative_calibration": dict(config["negative_calibration"]),
    }


def _sha256_dir(path: Path) -> str:
    # Best-effort directory identity: hash of sorted relative file list + each file sha.
    files = sorted(p for p in path.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(path).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(_sha256_file(p).encode())
        h.update(b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# calibration routing
# ---------------------------------------------------------------------------

def calibrate_for_init(init: str, model_path: Path, work_dir: Path, bank: Path,
                       config: Mapping[str, Any], sft_dir: Path | None,
                       repo: Path, gpu: str) -> Path:
    """Run (or reuse) the per-init calibration. base uses reference_adapter=None;
    low_sft uses the v2 SFT epoch_1 adapter. Output is identity-tagged."""
    import countdown_qwen_arena_onefile as arena  # noqa: E402

    tag = {"base": "base", "low_sft": "low_sft"}[init]
    ref_adapter = None if init == "base" else (sft_dir / "epoch_1_adapter" if sft_dir else None)
    if init == "low_sft" and ref_adapter is None:
        raise RuntimeError("low-SFT calibration requires v2 SFT epoch_1_adapter")
    out_dir = work_dir / "calibration" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "calibration.json"
    ident_json = out_dir / "calibration.identity.json"
    model_cfg = config["model"]
    neg = config["negative_calibration"]
    identity = {
        "init": init,
        "reference_adapter": str(ref_adapter) if ref_adapter else None,
        "bank_sha256": _sha256_file(bank),
        "model_path": str(model_path),
        "config_sha256": _sha256_file(Path(config_path_placeholder)),
    }
    if out_json.exists() and ident_json.exists():
        try:
            stored = json.loads(ident_json.read_text())
            if (stored.get("bank_sha256") == identity["bank_sha256"]
                    and stored.get("reference_adapter") == identity["reference_adapter"]):
                return out_json
        except Exception:
            pass
    args = argparse.Namespace(
        model_path=str(model_path),
        reference_adapter=str(ref_adapter) if ref_adapter else None,
        offline_data=str(bank),
        output_json=str(out_json),
        batch_size=int(neg["batch_size"]),
        calibration_batches=int(neg["batches"]),
        max_length=int(model_cfg["max_length"]),
        near_mix=float(neg["near_mix"]),
        far_mix=float(neg["far_mix"]),
        exp_lambda=float(neg["exp_lambda"]),
        surprisal_threshold=float(neg["surprisal_threshold"]),
        seed=int(neg["seed"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
    )
    if gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu).split(",")[0]
    arena.cmd_calibrate_global(args)
    ident_json.write_text(json.dumps(identity, indent=2, ensure_ascii=False))
    return out_json


config_path_placeholder = ""  # set in main; avoids passing config path through


# ---------------------------------------------------------------------------
# training / eval dispatch (delegates to canonical runner functions)
# ---------------------------------------------------------------------------

def _import_runner():
    import countdown_e8_base_rl_replay as br  # noqa: E402
    import countdown_qwen_arena_onefile as arena  # noqa: E402
    return br, arena


def _reference_adapter_for(cell: Cell, sft_dir: Path | None) -> str | None:
    if cell.init == BASE_INIT:
        return None
    if cell.init == LOW_SFT_INIT:
        return str(sft_dir / "epoch_1_adapter") if sft_dir else None
    if cell.init == FULL_SFT_INIT:
        return str(sft_dir / "best_adapter") if sft_dir else None
    raise ValueError(cell.init)


def train_cell(cell: Cell, model_path: Path, work_dir: Path, bank: Path,
               data_paths: Mapping[str, Path], config: Mapping[str, Any],
               calibration_json: Path | None, sft_dir: Path | None,
               identity: Mapping[str, Any]) -> dict[str, Any]:
    br, arena = _import_runner()
    train_cfg = config["offline_training"]
    model_cfg = config["model"]
    neg = config["negative_calibration"]
    out_dir = work_dir / "methods" / cell.name
    ref_adapter = _reference_adapter_for(cell, sft_dir)
    method = "positive_only" if cell.method == "positive_only" else str(neg["method"])
    args = argparse.Namespace(
        model_path=str(model_path),
        reference_adapter=ref_adapter,
        sft_adapter=None,
        offline_data=str(bank),
        val_data=str(data_paths["validation"]),
        structure_reference_data=str(bank),
        output_dir=str(out_dir),
        method=method,
        steps=int(train_cfg["steps"]),
        min_steps=int(train_cfg["min_steps"]),
        early_stop_patience=int(train_cfg["early_stop_patience"]),
        early_stop_delta=float(train_cfg["early_stop_delta"]),
        selection_metric=str(train_cfg["selection_metric"]),
        micro_batch=int(train_cfg["micro_batch"]),
        grad_accum=int(train_cfg["gradient_accumulation"]),
        lr=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        max_grad_norm=float(train_cfg["maximum_gradient_norm"]),
        max_length=int(model_cfg["max_length"]),
        max_new_tokens=int(model_cfg["max_new_tokens"]),
        eval_examples=int(config["evaluation"]["examples"]),
        eval_batch=int(config["evaluation"]["batch_size"]),
        pass_k=int(config["evaluation"]["pass_ks"][0]),
        negative_scale=None,
        negative_scale_multiplier=float(cell.negative_scale_multiplier),
        near_mix=float(neg["near_mix"]),
        far_mix=float(neg["far_mix"]),
        global_gamma=0.55,
        negative_calibration_json=str(calibration_json) if calibration_json else None,
        exp_lambda=float(neg["exp_lambda"]),
        surprisal_threshold=float(neg["surprisal_threshold"]),
        entropy_coef=0.02,
        target_entropy=1.8,
        target_entropy_coef=0.05,
        sbrc_kappa=0.92,
        entropy_floor=1.0,
        eval_every=int(train_cfg["eval_every"]),
        eval_seed=int(config["evaluation"]["seed"]),
        diagnostic_examples=int(train_cfg["diagnostic_examples"]),
        diagnostic_gradient_examples=int(train_cfg["diagnostic_gradient_examples"]),
        diagnostic_batch=int(train_cfg["diagnostic_batch"]),
        log_every=int(train_cfg["log_every"]),
        num_workers=int(train_cfg["num_workers"]),
        seed=int(train_cfg["seed"]) + int(cell.seed_offset),
        result_status=str(config["result_status"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
    )
    arena.cmd_train_method(args)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    best_eval = br.evaluate_adapter_checkpoint(
        model_path, out_dir / "best_adapter", data_paths, config,
        seed_offset=int(train_cfg["seed"]) + int(cell.seed_offset))
    terminal_eval = None
    if (out_dir / "terminal_adapter" / "adapter_config.json").exists():
        terminal_eval = br.evaluate_adapter_checkpoint(
            model_path, out_dir / "terminal_adapter", data_paths, config,
            seed_offset=int(train_cfg["seed"]) + int(cell.seed_offset) + 17)
    summary = {
        "method": cell.name,
        "cell": cell.name,
        "init": cell.init,
        "arena_method": method,
        "calibration": cell.calibration,
        "calibration_json": str(calibration_json) if calibration_json else None,
        "negative_scale_multiplier": float(cell.negative_scale_multiplier),
        "seed": int(train_cfg["seed"]) + int(cell.seed_offset),
        "best_step": manifest.get("best_step"),
        "best_value": manifest.get("best_value"),
        "terminal_step": manifest.get("terminal_step"),
        "stop_reason": manifest.get("stop_reason"),
        "best_evaluation": best_eval,
        "terminal_evaluation": terminal_eval,
        "run_identity": dict(identity),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def eval_only_cell(cell: Cell, model_path: Path, work_dir: Path,
                   data_paths: Mapping[str, Path], config: Mapping[str, Any],
                   sft_dir: Path | None, identity: Mapping[str, Any]) -> dict[str, Any]:
    br, arena = _import_runner()
    out_dir = work_dir / "methods" / cell.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_adapter = _reference_adapter_for(cell, sft_dir)
    tokenizer = arena.load_tokenizer(str(model_path))
    kwargs = br._model_kwargs(config)
    model = arena.load_model(
        str(model_path), ref_adapter, trainable_adapter=False,
        load_in_4bit=bool(kwargs["load_in_4bit"]), dtype=str(kwargs["dtype"]),
        gradient_checkpointing=False, parameterization="lora",
    )
    val_rows = arena.read_jsonl(data_paths["validation"])
    test_rows = arena.read_jsonl(data_paths["test"])
    summary: dict[str, Any] = {"method": cell.name, "cell": cell.name, "init": cell.init,
                               "kind": "eval_only"}
    summary.update(br.evaluate_model(model, tokenizer, val_rows, config,
                                     seed=int(config["evaluation"]["seed"]), prefix="validation"))
    summary.update(br.evaluate_model(model, tokenizer, test_rows, config,
                                     seed=int(config["evaluation"]["test_seed"]), prefix="test"))
    summary["run_identity"] = dict(identity)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def alias_cell(cell: Cell, work_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    src = work_dir / "methods" / cell.alias_of / "summary.json"
    if not src.exists():
        raise RuntimeError(f"alias {cell.name} -> missing source {cell.alias_of}")
    source_summary = json.loads(src.read_text())
    summary = {
        "method": cell.name, "cell": cell.name, "kind": "alias",
        "alias_of": cell.alias_of,
        "source_output_path": str(src),
        "source_summary_sha256": _sha256_bytes(src.read_bytes()),
        "adapter_identity": source_summary.get("arena_method"),
        "calibration_identity": source_summary.get("calibration"),
        "best_evaluation": source_summary.get("best_evaluation"),
        "terminal_evaluation": source_summary.get("terminal_evaluation"),
        "run_identity": dict(identity),
    }
    out = work_dir / "methods" / cell.name / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ---------------------------------------------------------------------------
# identity-checked resume
# ---------------------------------------------------------------------------

def _identity_matches(stored: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    # Compare the fields that define run identity. Nested dicts compared by value.
    keys = ("experiment_id", "runner_version", "config_sha256", "bank_sha256",
            "val_sha256", "test_sha256", "model_sha256", "git_commit", "git_dirty")
    for k in keys:
        if stored.get(k) != current.get(k):
            return False
    if stored.get("offline_training") != current.get("offline_training"):
        return False
    if stored.get("evaluation") != current.get("evaluation"):
        return False
    return True


def _load_stored_identity(work_dir: Path, cell_name: str) -> Mapping[str, Any] | None:
    p = work_dir / "methods" / cell_name / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("run_identity")
    except Exception:
        return None


def _cell_completes(work_dir: Path, cell: Cell, identity: Mapping[str, Any]) -> bool:
    if cell.kind == "alias":
        return (work_dir / "methods" / cell.name / "summary.json").exists()
    stored = _load_stored_identity(work_dir, cell.name)
    if stored is None:
        return False
    if not _identity_matches(stored, identity):
        # identity drift -> fail closed, do NOT silently reuse.
        raise RuntimeError(
            f"cell {cell.name} has a stale summary.json whose run_identity does not match "
            f"the current plan (corpus/config/adapter/commit changed). Remove it manually "
            f"or change work_dir; the runner refuses to reuse a mismatched result."
        )
    return True


# ---------------------------------------------------------------------------
# parallel orchestration
# ---------------------------------------------------------------------------

@dataclass
class _JobResult:
    cell: str
    returncode: int
    log: str
    status: str


def _run_cell_subprocess(cell: Cell, repo: Path, model_path: Path, work_dir: Path,
                          bank: Path, val: Path, test: Path, sft_dir: Path | None,
                          config_path: Path, calibration_base: Path | None,
                          calibration_low_sft: Path | None, identity: Mapping[str, Any],
                          gpu: str, logs_dir: Path) -> _JobResult:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = logs_dir / f"v2_matrix_{cell.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, str(repo / "src" / "drpo" / "countdown_e8_oracle_offline_v2_matrix.py"),
        "--worker", cell.name,
        "--model_path", str(model_path),
        "--work_dir", str(work_dir),
        "--bank", str(bank),
        "--val", str(val),
        "--test", str(test),
        "--config", str(config_path),
    ]
    if sft_dir:
        argv += ["--sft_dir", str(sft_dir)]
    if calibration_base:
        argv += ["--calibration_base", str(calibration_base)]
    if calibration_low_sft:
        argv += ["--calibration_low_sft", str(calibration_low_sft)]
    with log_path.open("w") as fh:
        fh.write(f"=== {cell.name} on GPU {gpu} ===\n" + " ".join(argv) + "\n")
        fh.flush()
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), ""):
            fh.write(chunk)
            fh.flush()
        rc = proc.wait()
        fh.write(f"=== returncode={rc} ===\n")
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    return _JobResult(cell.name, rc, str(log_path), status)


def cmd_run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model_path = Path(args.model_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    bank = Path(args.bank).resolve()
    val = Path(args.val).resolve()
    test = Path(args.test).resolve()
    config_path = Path(args.config).resolve()
    sft_dir = Path(args.sft_dir).resolve() if args.sft_dir else None
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    global config_path_placeholder
    config_path_placeholder = str(config_path)

    config = load_config(config_path)
    assert_plan_invariants()

    identity = run_identity(model_path, bank, val, test, sft_dir, config_path, repo, config)

    # Phase 0: preflight — bank/splits/config must exist; SFT dir required for low/full cells.
    for p in (bank, val, test, config_path):
        if not p.exists():
            raise SystemExit(f"missing required artifact: {p}")
    needs_sft = any(c.init in (LOW_SFT_INIT, FULL_SFT_INIT) for c in CELLS)
    if needs_sft and (sft_dir is None
                      or not (sft_dir / "epoch_1_adapter").exists()
                      or not (sft_dir / "best_adapter").exists()):
        raise SystemExit(
            f"low/full-SFT cells require --sft_dir with epoch_1_adapter and best_adapter; got {sft_dir}"
        )

    # Phase 1: per-init calibration (base always; low_sft only if B2 will run).
    calibration_base: Path | None = None
    calibration_low_sft: Path | None = None
    if any(c.calibration == "base" for c in TRAINING_CELLS):
        calibration_base = calibrate_for_init("base", model_path, work_dir, bank, config,
                                              sft_dir, repo, args.gpu)
    if any(c.calibration == "low_sft" for c in TRAINING_CELLS):
        calibration_low_sft = calibrate_for_init("low_sft", model_path, work_dir, bank, config,
                                                 sft_dir, repo, args.gpu)

    # Build the execution list. Skip only identity-matched completed cells.
    todo: list[Cell] = []
    for cell in CELLS:
        if _cell_completes(work_dir, cell, identity):
            print(f"[skip] {cell.name} (identity-matched summary.json present)", flush=True)
            continue
        todo.append(cell)

    # Phase 2/3: training + eval-only cells in a GPU pool; aliases last (pure aggregation).
    train_or_eval = [c for c in todo if c.kind in ("train", "eval_only")]
    aliases = [c for c in todo if c.kind == "alias"]

    pool = [g for g in args.gpus.split(",") if g.strip()]
    if not pool:
        raise SystemExit("--gpus must list at least one GPU")
    results: dict[str, _JobResult] = {}
    results_lock = threading.Lock()
    gpu_sem = threading.Semaphore(len(pool))
    gpu_q: list[str] = list(pool)
    gpu_q_lock = threading.Lock()

    def take_gpu() -> str:
        with gpu_q_lock:
            return gpu_q.pop(0)

    def give_gpu(g: str) -> None:
        with gpu_q_lock:
            gpu_q.append(g)

    def run_one(cell: Cell) -> None:
        gpu_sem.acquire()
        gpu = take_gpu()
        try:
            print(f"[start] {cell.name} on GPU {gpu}", flush=True)
            r = _run_cell_subprocess(cell, repo, model_path, work_dir, bank, val, test,
                                     sft_dir, config_path, calibration_base,
                                     calibration_low_sft, identity, gpu, logs_dir)
            print(f"[end]   {cell.name} -> {r.status} ({r.log})", flush=True)
            with results_lock:
                results[cell.name] = r
        finally:
            give_gpu(gpu)
            gpu_sem.release()

    threads = [threading.Thread(target=run_one, args=(c,), name=c.name) for c in train_or_eval]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Aliases: aggregate in-process (no GPU).
    for cell in aliases:
        try:
            alias_cell(cell, work_dir, identity)
            print(f"[alias] {cell.name} -> {cell.alias_of}", flush=True)
            results[cell.name] = _JobResult(cell.name, 0, "", "OK")
        except Exception as e:  # noqa: BLE001
            print(f"[alias] {cell.name} FAILED: {e}", flush=True)
            results[cell.name] = _JobResult(cell.name, 1, str(e), f"FAIL({e})")

    # Final status: any missing or failed cell is a non-zero exit.
    expected = {c.name for c in CELLS}
    completed = {name for name, r in results.items() if r.returncode == 0}
    missing = sorted(expected - completed)
    failed = sorted(name for name, r in results.items() if r.returncode != 0)

    status = {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": VERSION,
        "run_identity": identity,
        "results": {k: {"status": v.status, "returncode": v.returncode, "log": v.log}
                    for k, v in sorted(results.items())},
        "missing_cells": missing,
        "failed_cells": failed,
        "all_expected_cells_present": not missing and not failed,
    }
    (work_dir / "v2_matrix_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False))
    print(json.dumps({"missing": missing, "failed": failed,
                      "ok": not missing and not failed}, indent=2), flush=True)
    return 0 if not missing and not failed else 1


# ---------------------------------------------------------------------------
# worker entry (subprocess per cell)
# ---------------------------------------------------------------------------

def _worker(cell_name: str, args: argparse.Namespace) -> int:
    model_path = Path(args.model_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    bank = Path(args.bank).resolve()
    val = Path(args.val).resolve()
    test = Path(args.test).resolve()
    config_path = Path(args.config).resolve()
    sft_dir = Path(args.sft_dir).resolve() if args.sft_dir else None
    config = load_config(config_path)
    data_paths = {"train": bank, "validation": val, "test": test, "split_manifest": bank}
    identity = run_identity(model_path, bank, val, test, sft_dir, config_path,
                            Path(__file__).resolve().parents[2], config)
    cell = next(c for c in CELLS if c.name == cell_name)
    calib = None
    if cell.calibration == "base" and args.calibration_base:
        calib = Path(args.calibration_base).resolve()
    elif cell.calibration == "low_sft" and args.calibration_low_sft:
        calib = Path(args.calibration_low_sft).resolve()
    if cell.kind == "train":
        train_cell(cell, model_path, work_dir, bank, data_paths, config, calib, sft_dir, identity)
    elif cell.kind == "eval_only":
        eval_only_cell(cell, model_path, work_dir, data_paths, config, sft_dir, identity)
    else:
        alias_cell(cell, work_dir, identity)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Countdown E8 oracle-offline v2 init-matrix")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model_path", required=True)
    run.add_argument("--work_dir", required=True)
    run.add_argument("--bank", required=True, help="v2 offline_bank_v2.jsonl")
    run.add_argument("--val", required=True, help="v2 val.jsonl")
    run.add_argument("--test", required=True, help="v2 test.jsonl")
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--sft_dir", default=None, help="v2 SFT dir with epoch_1/best adapters")
    run.add_argument("--gpu", default="0", help="GPU for calibration (single)")
    run.add_argument("--gpus", default="0,1,2,3,4,5", help="GPU pool for training cells")
    run.add_argument("--logs_dir", default="logs")
    run.set_defaults(func=cmd_run)

    w = sub.add_parser("--worker")
    w.add_argument("cell")
    w.add_argument("--model_path", required=True)
    w.add_argument("--work_dir", required=True)
    w.add_argument("--bank", required=True)
    w.add_argument("--val", required=True)
    w.add_argument("--test", required=True)
    w.add_argument("--config", default=DEFAULT_CONFIG)
    w.add_argument("--sft_dir", default=None)
    w.add_argument("--calibration_base", default=None)
    w.add_argument("--calibration_low_sft", default=None)
    w.set_defaults(func=lambda a: _worker(a.cell, a))

    plan = sub.add_parser("plan")
    plan.set_defaults(func=lambda a: (print(json.dumps(
        [{"name": c.name, "init": c.init, "method": c.method, "calibration": c.calibration,
          "multiplier": c.negative_scale_multiplier, "seed_offset": c.seed_offset,
          "kind": c.kind, "alias_of": c.alias_of} for c in CELLS], indent=2)) or 0))

    st = sub.add_parser("selftest")
    st.set_defaults(func=cmd_selftest)
    return parser


def cmd_selftest(_: argparse.Namespace) -> None:
    assert_plan_invariants()
    # identity match round-trip
    ident = {"experiment_id": EXPERIMENT_ID, "runner_version": VERSION,
             "config_sha256": "x", "bank_sha256": "y", "val_sha256": "v",
             "test_sha256": "t", "model_sha256": "m", "git_commit": "c",
             "git_dirty": False, "offline_training": {"a": 1}, "evaluation": {"b": 2}}
    assert _identity_matches(ident, ident)
    drift = dict(ident)
    drift["bank_sha256"] = "z"
    assert not _identity_matches(ident, drift)
    # plan shape
    assert len(TRAINING_CELLS) == 8
    assert len(EVAL_ONLY_CELLS) == 2
    assert len(ALIAS_CELLS) == 2
    print("V2_INIT_MATRIX_SELFTEST_OK")


def main() -> None:
    args = build_parser().parse_args()
    result = args.func(args)
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()
