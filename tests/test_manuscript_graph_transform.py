from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "manuscript_graph_transform.py"
SPEC = importlib.util.spec_from_file_location("manuscript_graph_transform", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TransformError = MODULE.TransformError
compile_products = MODULE.compile_products
run = MODULE.run
validate_products = MODULE.validate_products


def _blueprint() -> dict:
    return {
        "task_id": "TEST",
        "nodes": [
            {
                "id": "INTRO-P01",
                "section": "Introduction",
                "order": 1,
                "title": "Disabled",
                "outline_block_sha256": "a" * 64,
                "status": "disabled_with_reason",
                "disabled_reason": "not selected",
            },
            {
                "id": "METHOD-P03",
                "section": "Method",
                "order": 2,
                "title": "Tail guarantee",
                "outline_block_sha256": "b" * 64,
                "status": "enabled",
                "reader_question": "Why exponential?",
                "paragraph_claim": "The tail vanishes under finite-order growth.",
                "sentence_plan": [
                    {"role": "motivation", "instruction": "Motivate the envelope."},
                    {"role": "result", "instruction": "State the result."},
                    {"role": "boundary", "instruction": "Bound the claim."},
                ],
                "evidence_refs": ["PROP-TAIL"],
                "metric_paths": [],
                "figure_refs": [],
                "table_refs": [],
                "allowed_conclusions": ["tail guarantee"],
                "forbidden_conclusions": ["universal superiority"],
                "reviewer_objection": "Arbitrary utility model.",
                "objection_response": "No utility model is assumed.",
                "transition_to_next": "Test the mechanism.",
            },
            {
                "id": "EXP-P04",
                "section": "Experiments",
                "order": 3,
                "title": "Causal intervention",
                "outline_block_sha256": "c" * 64,
                "status": "enabled",
                "reader_question": "Is far-field influence causal?",
                "paragraph_claim": "Far control rescues the controlled environment.",
                "sentence_plan": [
                    {"role": "setup", "instruction": "Define events separately."},
                    {"role": "result", "instruction": "Report exact reward."},
                    {"role": "boundary", "instruction": "Calibrate the claim."},
                ],
                "evidence_refs": ["EVID-E3"],
                "metric_paths": [
                    "methods.baseline.reward",
                    "methods.far.reward",
                    "methods.far.ci",
                ],
                "figure_refs": ["old.pdf"],
                "table_refs": ["results.tex"],
                "allowed_conclusions": ["controlled causal path"],
                "forbidden_conclusions": ["C-U1 demonstrates OOD generalization"],
                "reviewer_objection": "Budget confound.",
                "objection_response": "Use a matched control.",
                "transition_to_next": "Test external validity.",
            },
        ],
    }


def _snapshot() -> dict:
    return {
        "methods": {
            "baseline": {"reward": 0.2},
            "far": {"reward": 0.7, "ci": [0.65, 0.75]},
        }
    }


def _contract() -> dict:
    return {
        "schema_version": 1,
        "task_id": "TEST-DOWNSTREAM",
        "output_root": "out",
        "source": {"blueprint": "blueprint.json", "snapshot": "snapshot.json"},
        "nodes": {
            "METHOD-P03": {
                "sentence_bindings": {
                    "motivation": {
                        "evidence_refs": ["PROP-TAIL"],
                        "metrics": {},
                        "figure_bindings": [],
                        "table_bindings": [],
                        "template": "Use the envelope for tail control.",
                    },
                    "result": {
                        "evidence_refs": ["PROP-TAIL"],
                        "metrics": {},
                        "figure_bindings": [],
                        "table_bindings": [],
                        "template": "Finite-order growth is dominated.",
                    },
                    "boundary": {
                        "evidence_refs": ["PROP-TAIL"],
                        "metrics": {},
                        "figure_bindings": [],
                        "table_bindings": [],
                        "template": "This is not a universal ranking.",
                    },
                },
                "visual_products": [],
            },
            "EXP-P04": {
                "sentence_bindings": {
                    "setup": {
                        "evidence_refs": ["EVID-E3"],
                        "metrics": {},
                        "figure_bindings": ["FIG-E3"],
                        "table_bindings": ["results.tex"],
                        "template": "C-U1 reports task, boundary, and NaN/Inf events separately.",
                    },
                    "result": {
                        "evidence_refs": ["EVID-E3"],
                        "metrics": {
                            "baseline": "methods.baseline.reward",
                            "far": "methods.far.reward",
                            "ci_low": "methods.far.ci.0",
                            "ci_high": "methods.far.ci.1",
                        },
                        "figure_bindings": ["FIG-E3"],
                        "table_bindings": [],
                        "template": "Reward changes from {baseline:.2f} to {far:.2f} (CI {ci_low:.2f}--{ci_high:.2f}).",
                    },
                    "boundary": {
                        "evidence_refs": ["EVID-E3"],
                        "metrics": {},
                        "figure_bindings": ["FIG-E3"],
                        "table_bindings": [],
                        "template": "The conclusion is restricted to the controlled environment.",
                    },
                },
                "visual_products": [
                    {
                        "figure_id": "FIG-E3",
                        "kind": "empirical",
                        "renderer": "bar_panels",
                        "output_files": ["figure.png"],
                        "supports_roles": ["setup", "result", "boundary"],
                        "caption": "C-U1 controlled intervention with separate outcomes.",
                        "panels": [
                            {
                                "panel_id": "A",
                                "question": "What is the reward contrast?",
                                "title": "Reward",
                                "ylabel": "Reward",
                                "series": [
                                    {
                                        "label": "Baseline",
                                        "value_path": "methods.baseline.reward",
                                    },
                                    {
                                        "label": "Far",
                                        "value_path": "methods.far.reward",
                                        "ci_path": "methods.far.ci",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
        },
    }


def _compile(contract: dict | None = None) -> dict:
    return compile_products(
        blueprint=_blueprint(),
        snapshot=_snapshot(),
        contract=contract or _contract(),
        blueprint_sha="1" * 64,
        snapshot_sha="2" * 64,
    )


def test_compile_preserves_graph_and_realizes_sentence_units() -> None:
    products = _compile()
    nodes = products["product_graph"]["nodes"]
    assert [node["id"] for node in nodes] == ["INTRO-P01", "METHOD-P03", "EXP-P04"]
    assert nodes[0]["status"] == "disabled_with_reason"
    assert [unit["sid"] for unit in nodes[2]["sentence_units"]] == [
        "EXP-P04-S01",
        "EXP-P04-S02",
        "EXP-P04-S03",
    ]
    draft = products["prose_packets"]["paragraphs"][1]["deterministic_draft"]
    assert "0.20" in draft
    assert "0.70" in draft
    assert "CI 0.65--0.75" in draft
    figure = products["figure_specs"]["figures"][0]
    assert figure["supports_sentence_ids"] == [
        "EXP-P04-S01",
        "EXP-P04-S02",
        "EXP-P04-S03",
    ]
    assert figure["panels"][0]["series"][1]["ci"] == [0.65, 0.75]


def test_validate_rejects_ood_wording_for_cu1() -> None:
    products = _compile()
    products["prose_packets"]["paragraphs"][1]["deterministic_draft"] += " C-U1 is OOD."
    with pytest.raises(TransformError, match="OOD terminology"):
        validate_products(
            blueprint=_blueprint(),
            snapshot=_snapshot(),
            contract=_contract(),
            products=products,
        )


def test_compile_fails_when_sentence_roles_are_not_one_to_one() -> None:
    contract = _contract()
    del contract["nodes"]["EXP-P04"]["sentence_bindings"]["boundary"]
    with pytest.raises(TransformError, match="must match blueprint roles exactly"):
        _compile(contract)


def test_compile_fails_when_blueprint_metric_is_unassigned() -> None:
    contract = _contract()
    result = contract["nodes"]["EXP-P04"]["sentence_bindings"]["result"]
    del result["metrics"]["baseline"]
    result["template"] = "Far reward is {far:.2f} (CI {ci_low:.2f}--{ci_high:.2f})."
    with pytest.raises(TransformError, match="leaves blueprint metrics unassigned"):
        _compile(contract)


def test_cli_writes_products_and_validation_report(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "blueprint.json").write_text(json.dumps(_blueprint()), encoding="utf-8")
    (tmp_path / "snapshot.json").write_text(json.dumps(_snapshot()), encoding="utf-8")
    contract = _contract()
    (tmp_path / "docs" / "contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")

    args = type(
        "Args",
        (),
        {
            "repo_root": str(tmp_path),
            "contract": "docs/contract.yaml",
            "blueprint": None,
            "snapshot": None,
            "output": None,
            "skip_figures": True,
        },
    )()
    report = run(args)
    assert report["status"] == "PASS"
    assert (tmp_path / "out" / "product_graph.json").is_file()
    assert (tmp_path / "out" / "prose_draft.md").is_file()
    assert (tmp_path / "out" / "figure_specs.json").is_file()
    assert (tmp_path / "out" / "validation_report.json").is_file()


def test_empirical_figure_renderer(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    products = _compile()
    spec = products["figure_specs"]["figures"][0]
    written = MODULE.render_figure(spec, tmp_path)
    assert written
    assert (tmp_path / "figure.png").stat().st_size > 0
