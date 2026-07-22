from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpo.paper_logic_artifacts import validate_mapping  # noqa: E402
from drpo.paper_logic_common import GateError, load_policy  # noqa: E402


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_policy_cannot_drop_required_artifacts(tmp_path: Path) -> None:
    docs = tmp_path / "docs/manuscript"
    docs.mkdir(parents=True)
    (docs / "RL_PAPER_WRITING_GUIDANCE.md").write_text(
        "\n".join(
            [
                "### G02. Frozen baseline",
                "### G03. Review is not rewrite",
                "### G06. One tension",
                "### G07. Precise missing link",
                "### G10. Shared object",
                "### G15. Status separation",
                "### G23. Introduction",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "RL_PAPER_WRITING_PLAYBOOK.md").write_text(
        "\n".join(
            [
                "### Pass 2 — Story and missing link",
                "### Pass 3 — Outline",
                "### Pass 4 — Paragraph blueprint",
                "### Pass 6 — Theory-method-object audit",
                "### Pass 8 — Scientific writing quality",
                "### Pass 9 — Compression and page budget",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    policy = {
        "schema_version": 1,
        "initiative": "PAPER-WRITING-LOGIC-FIRST-01",
        "guidance_path": "docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md",
        "playbook_path": "docs/manuscript/RL_PAPER_WRITING_PLAYBOOK.md",
        "levels": {
            "wording": {
                "guidance_rules": ["G02", "G03", "G23"],
                "playbook_modules": [
                    "### Pass 8 — Scientific writing quality",
                    "### Pass 9 — Compression and page budget",
                ],
                "required_artifacts": ["paragraph_logic", "source_mapping", "candidate"],
            },
            "paragraph": {
                "guidance_rules": ["G02", "G03", "G07", "G15", "G23"],
                "playbook_modules": [
                    "### Pass 3 — Outline",
                    "### Pass 4 — Paragraph blueprint",
                    "### Pass 8 — Scientific writing quality",
                ],
                "required_artifacts": ["paragraph_logic", "source_mapping", "candidate"],
            },
            "section": {
                "guidance_rules": ["G02", "G03", "G06", "G07", "G10", "G15", "G23"],
                "playbook_modules": [
                    "### Pass 2 — Story and missing link",
                    "### Pass 3 — Outline",
                    "### Pass 4 — Paragraph blueprint",
                    "### Pass 6 — Theory-method-object audit",
                ],
                "required_artifacts": [
                    "section_logic",
                    "paragraph_logic",
                    "source_mapping",
                    "candidate",
                ],
            },
        },
    }
    policy_path = docs / "paper_logic_gate_policy.yaml"
    write_yaml(policy_path, policy)
    with pytest.raises(GateError, match="required_artifacts changed"):
        load_policy(tmp_path, policy_path)


def test_source_span_cannot_be_mapped_twice() -> None:
    source = "A unique approved sentence."
    mapping = {
        "schema_version": 1,
        "artifact_type": "source_mapping",
        "status": "complete",
        "source_sha256": "source-sha",
        "operations": [
            {
                "node_id": "INTRO-P01.S01",
                "paragraph_id": "INTRO-P01",
                "action": "KEEP",
                "source_text": source,
                "claim_impact": "none",
            },
            {
                "node_id": "INTRO-P01.S02",
                "paragraph_id": "INTRO-P01",
                "action": "KEEP",
                "source_text": source,
                "claim_impact": "none",
            },
        ],
    }
    with pytest.raises(GateError, match="mapped more than once"):
        validate_mapping(mapping, "source-sha", source)
