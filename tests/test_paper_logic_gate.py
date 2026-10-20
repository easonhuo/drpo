from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "drpo" / "paper_logic_gate.py"


def load_module():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("drpo.paper_logic_gate", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_repo(tmp_path: Path, *, level: str = "paragraph"):
    module = load_module()
    guidance = tmp_path / "docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md"
    playbook = tmp_path / "docs/manuscript/RL_PAPER_WRITING_PLAYBOOK.md"
    guidance.parent.mkdir(parents=True, exist_ok=True)
    guidance.write_text(
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
    playbook.write_text(
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
                "required_artifacts": [
                    "paragraph_logic",
                    "source_mapping",
                    "candidate",
                ],
            },
            "paragraph": {
                "guidance_rules": ["G02", "G03", "G07", "G15", "G23"],
                "playbook_modules": [
                    "### Pass 3 — Outline",
                    "### Pass 4 — Paragraph blueprint",
                    "### Pass 8 — Scientific writing quality",
                ],
                "required_artifacts": [
                    "section_logic",
                    "paragraph_logic",
                    "source_mapping",
                    "candidate",
                ],
            },
            "section": {
                "guidance_rules": [
                    "G02",
                    "G03",
                    "G06",
                    "G07",
                    "G10",
                    "G15",
                    "G23",
                ],
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
    policy_path = tmp_path / "docs/manuscript/paper_logic_gate_policy.yaml"
    write_yaml(policy_path, policy)

    source = tmp_path / "paper/source.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    first = "Against this background, we identify the mechanism."
    second = "The next sentence explains the consequence."
    source.write_text(f"{first}\n\n{second}\n", encoding="utf-8")
    source_sha = sha256(source)

    section_logic = {
        "schema_version": 1,
        "artifact_type": "section_logic_map",
        "status": "approved",
        "source_sha256": source_sha,
        "section_id": "INTRO",
        "central_question": "What changes under repeated reuse?",
        "entry_point": "signed updates",
        "exit_point": "evidence chain",
        "chain": ["signed updates", "reuse", "control"],
        "paragraph_ids": ["INTRO-P01"],
        "approval": {"approved_by": "user", "approved_at": "2026-07-14"},
    }
    section_path = tmp_path / "work/section.yaml"
    write_yaml(section_path, section_logic)

    paragraph_logic = {
        "schema_version": 1,
        "artifact_type": "paragraph_logic_map",
        "status": "approved",
        "source_sha256": source_sha,
        "section_id": "INTRO",
        "paragraphs": [
            {
                "id": "INTRO-P01",
                "responsibility": "identify the mechanism",
                "topic_claim": "repeated reuse changes influence",
                "sentence_nodes": [
                    {
                        "id": "INTRO-P01.S01",
                        "role": "contribution_marker",
                        "instruction": "preserve the approved opening",
                    },
                    {
                        "id": "INTRO-P01.S02",
                        "role": "consequence",
                        "instruction": "state the consequence directly",
                    },
                ],
            }
        ],
        "approval": {"approved_by": "user", "approved_at": "2026-07-14"},
    }
    paragraph_path = tmp_path / "work/paragraph.yaml"
    write_yaml(paragraph_path, paragraph_logic)

    mapping = {
        "schema_version": 1,
        "artifact_type": "source_mapping",
        "status": "complete",
        "source_sha256": source_sha,
        "operations": [
            {
                "node_id": "INTRO-P01.S01",
                "paragraph_id": "INTRO-P01",
                "action": "KEEP",
                "source_text": first,
                "claim_impact": "none",
            },
            {
                "node_id": "INTRO-P01.S02",
                "paragraph_id": "INTRO-P01",
                "action": "REVISE",
                "source_text": second,
                "reason": "make the causal consequence explicit",
                "claim_impact": "none",
            },
        ],
    }
    mapping_path = tmp_path / "work/mapping.yaml"
    write_yaml(mapping_path, mapping)

    candidate = {
        "schema_version": 1,
        "artifact_type": "prose_candidate",
        "status": "draft",
        "source_sha256": source_sha,
        "section_id": "INTRO",
        "paragraphs": [
            {
                "id": "INTRO-P01",
                "sentences": [
                    {"node_id": "INTRO-P01.S01", "text": first},
                    {
                        "node_id": "INTRO-P01.S02",
                        "text": "The next update therefore changes in magnitude.",
                    },
                ],
            }
        ],
    }
    candidate_path = tmp_path / "work/candidate.yaml"
    write_yaml(candidate_path, candidate)

    manifest = {
        "schema_version": 1,
        "initiative": "PAPER-WRITING-LOGIC-FIRST-01",
        "edit_level": level,
        "section_id": "INTRO",
        "target_paragraph_ids": ["INTRO-P01"],
        "source": {"path": "paper/source.md", "sha256": source_sha},
        "paragraph_logic": {
            "path": "work/paragraph.yaml",
            "sha256": sha256(paragraph_path),
        },
        "source_mapping": {
            "path": "work/mapping.yaml",
            "sha256": sha256(mapping_path),
        },
        "candidate": {
            "path": "work/candidate.yaml",
            "sha256": sha256(candidate_path),
        },
        "authorization": {
            "approved_by": "user",
            "approved_at": "2026-07-14",
            "allow_claim_strengthening": False,
        },
    }
    if level in {"paragraph", "section"}:
        manifest["section_logic"] = {
            "path": "work/section.yaml",
            "sha256": sha256(section_path),
        }
    manifest_path = tmp_path / "work/manifest.yaml"
    write_yaml(manifest_path, manifest)
    return module, policy_path, manifest_path


def refresh_ref(manifest_path: Path, key: str, artifact_path: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest[key]["sha256"] = sha256(artifact_path)
    write_yaml(manifest_path, manifest)


def test_wording_level_passes_without_section_map(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path, level="wording")
    result = module.validate_manifest(
        repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
    )
    assert result["status"] == "PASS"
    assert result["selected_guidance_rules"] == ["G02", "G03", "G23"]
    assert result["validated_node_count"] == 2
    assert result["invalidation_scope"] == ["prose:INTRO-P01"]


def test_paragraph_level_requires_section_logic(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path, level="paragraph")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del manifest["section_logic"]
    write_yaml(manifest_path, manifest)
    with pytest.raises(module.PaperLogicGateError, match="section_logic"):
        module.validate_manifest(
            repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
        )


def test_keep_sentence_change_fails_closed(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path)
    candidate_path = tmp_path / "work/candidate.yaml"
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    candidate["paragraphs"][0]["sentences"][0]["text"] = "A rewritten opening."
    write_yaml(candidate_path, candidate)
    refresh_ref(manifest_path, "candidate", candidate_path)
    with pytest.raises(module.PaperLogicGateError, match="frozen sentence changed"):
        module.validate_manifest(
            repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
        )


def test_unauthorized_candidate_node_fails_closed(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path)
    candidate_path = tmp_path / "work/candidate.yaml"
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    candidate["paragraphs"][0]["sentences"].append(
        {"node_id": "INTRO-P01.S99", "text": "An unauthorized new claim."}
    )
    write_yaml(candidate_path, candidate)
    refresh_ref(manifest_path, "candidate", candidate_path)
    with pytest.raises(module.PaperLogicGateError, match="unauthorized"):
        module.validate_manifest(
            repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
        )


def test_stale_paragraph_logic_fails_closed(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path)
    paragraph_path = tmp_path / "work/paragraph.yaml"
    paragraph = yaml.safe_load(paragraph_path.read_text(encoding="utf-8"))
    paragraph["source_sha256"] = "0" * 64
    write_yaml(paragraph_path, paragraph)
    refresh_ref(manifest_path, "paragraph_logic", paragraph_path)
    with pytest.raises(module.PaperLogicGateError, match="stale relative to source"):
        module.validate_manifest(
            repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
        )


def test_claim_strengthening_requires_explicit_authorization(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path)
    mapping_path = tmp_path / "work/mapping.yaml"
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    mapping["operations"][1]["claim_impact"] = "strengthen"
    write_yaml(mapping_path, mapping)
    refresh_ref(manifest_path, "source_mapping", mapping_path)
    with pytest.raises(module.PaperLogicGateError, match="not authorized"):
        module.validate_manifest(
            repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
        )


def test_section_plan_invalidates_only_section_descendants(tmp_path: Path) -> None:
    module, policy_path, _ = prepare_repo(tmp_path)
    policy = module.load_policy(tmp_path, policy_path)
    result = module.plan(
        "section", section_id="INTRO", paragraph_ids=["INTRO-P01"], policy=policy
    )
    assert result["invalidation_scope"] == [
        "paragraph_logic:INTRO",
        "source_mapping:INTRO",
        "prose:INTRO",
    ]
    assert "G06" in result["selected_guidance_rules"]


def test_mapping_must_cover_every_target_sentence_node(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path)
    mapping_path = tmp_path / "work/mapping.yaml"
    candidate_path = tmp_path / "work/candidate.yaml"
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    mapping["operations"] = mapping["operations"][:1]
    candidate["paragraphs"][0]["sentences"] = candidate["paragraphs"][0]["sentences"][:1]
    write_yaml(mapping_path, mapping)
    write_yaml(candidate_path, candidate)
    refresh_ref(manifest_path, "source_mapping", mapping_path)
    refresh_ref(manifest_path, "candidate", candidate_path)
    with pytest.raises(module.PaperLogicGateError, match="cover every approved node"):
        module.validate_manifest(
            repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
        )


def test_add_action_passes_when_node_is_approved(tmp_path: Path) -> None:
    module, policy_path, manifest_path = prepare_repo(tmp_path)
    paragraph_path = tmp_path / "work/paragraph.yaml"
    mapping_path = tmp_path / "work/mapping.yaml"
    candidate_path = tmp_path / "work/candidate.yaml"

    paragraph = yaml.safe_load(paragraph_path.read_text(encoding="utf-8"))
    paragraph["paragraphs"][0]["sentence_nodes"].append(
        {
            "id": "INTRO-P01.S03",
            "role": "boundary",
            "instruction": "add the approved scope boundary",
        }
    )
    write_yaml(paragraph_path, paragraph)

    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    mapping["operations"].append(
        {
            "node_id": "INTRO-P01.S03",
            "paragraph_id": "INTRO-P01",
            "action": "ADD",
            "reason": "approved logical gap",
            "claim_impact": "none",
        }
    )
    write_yaml(mapping_path, mapping)

    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    candidate["paragraphs"][0]["sentences"].append(
        {
            "node_id": "INTRO-P01.S03",
            "text": "This boundary concerns policy geometry rather than task utility.",
        }
    )
    write_yaml(candidate_path, candidate)

    refresh_ref(manifest_path, "paragraph_logic", paragraph_path)
    refresh_ref(manifest_path, "source_mapping", mapping_path)
    refresh_ref(manifest_path, "candidate", candidate_path)
    result = module.validate_manifest(
        repo=tmp_path, manifest_path=manifest_path, policy_path=policy_path
    )
    assert result["status"] == "PASS"
    assert result["validated_node_count"] == 3


def test_repository_policy_references_existing_docs() -> None:
    module = load_module()
    policy = module.load_policy(
        ROOT, ROOT / "docs/manuscript/paper_logic_gate_policy.yaml"
    )
    result = module.plan(
        "wording", section_id="INTRO", paragraph_ids=["INTRO-P01"], policy=policy
    )
    assert result["status"] == "PASS"
    assert result["required_artifacts"] == [
        "paragraph_logic",
        "source_mapping",
        "candidate",
    ]
