from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "445a2e6d129994b2dd48f7c87050206dc705b838"
REVIEW_PATH = ROOT / "docs/manuscript/reviews/PAPER-V091-GUIDANCE-REVIEW.json"
GUIDANCE = ROOT / "docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md"
STRATEGY = ROOT / "docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md"
CORPUS = ROOT / "docs/manuscript/RL_WRITING_CORPUS_NOTES.md"
OUTLINE = ROOT / "docs/paper_rewrite_outline_v0_9_1.md"
BLUEPRINT = ROOT / "docs/paper_rewrite_intro_blueprint_v0_5.md"

BLOCK_RE = re.compile(
    r"<!-- MANUSCRIPT:BEGIN ([A-Z0-9-]+) -->\n(.*?)"
    r"<!-- MANUSCRIPT:END \1 -->",
    re.DOTALL,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review() -> dict:
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def parse_blocks(path: Path) -> list[tuple[str, str, str]]:
    blocks: list[tuple[str, str, str]] = []
    for paragraph_id, payload in BLOCK_RE.findall(path.read_text(encoding="utf-8")):
        heading = next(
            line for line in payload.splitlines() if line.strip().startswith("## ")
        )
        title = heading.split("]", 1)[1].strip()
        blocks.append((paragraph_id, title, payload))
    return blocks


def test_guidance_review_hashes_and_hard_gates_are_current() -> None:
    review = load_review()
    assert review["base_commit"] == BASE_COMMIT
    assert review["verdict"] == "PASS"
    assert review["strategy_path"] == "docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md"

    for artifact in review["artifacts"]:
        path = ROOT / artifact["path"]
        assert sha256(path) == artifact["sha256"]

    gates = {gate["id"]: gate["status"] for gate in review["gates"]}
    assert gates == {f"G{i:02d}": "pass" for i in range(1, 15)}
    assert review["findings"]["blocker"] == []
    assert review["findings"]["major"] == []


def test_stable_guidance_is_separate_from_drpo_strategy() -> None:
    guidance = GUIDANCE.read_text(encoding="utf-8")
    strategy = STRATEGY.read_text(encoding="utf-8")

    assert "stable manuscript quality standard" in guidance
    assert "intentionally slow-moving" in guidance
    assert "DRPO Manuscript Strategy" in strategy

    project_specific_terms = [
        "C-U1",
        "D-U1",
        "Hopper",
        "Countdown",
        "Product manifold",
        "Theorem 1",
        "q\\mathbf m_-",
    ]
    for term in project_specific_terms:
        assert term not in guidance
        assert term in strategy

    assert "do not automatically change this standard" in guidance
    assert "does not automatically change the guidance" in CORPUS.read_text(
        encoding="utf-8"
    ).lower()


def test_drpo_lineage_is_explicit_and_positive() -> None:
    strategy = STRATEGY.read_text(encoding="utf-8")
    outline = OUTLINE.read_text(encoding="utf-8")
    combined = strategy + "\n" + outline

    assert "Breaking the Curse of Repulsion" in combined
    assert "Optimistic Distributionally Robust Policy Optimization" in combined
    assert "arXiv:2602.10430" in combined
    assert "DRPO is retained by design" in strategy
    assert "not a newly named algorithm" in strategy
    assert "lineage commitment" in outline
    assert "SNA2C" in strategy


def test_outline_blueprint_blocks_and_parent_hashes_align() -> None:
    outline_blocks = parse_blocks(OUTLINE)
    blueprint_blocks = parse_blocks(BLUEPRINT)

    assert [(pid, title) for pid, title, _ in outline_blocks] == [
        (pid, title) for pid, title, _ in blueprint_blocks
    ]
    assert len(outline_blocks) == 6

    expected = {
        pid: hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for pid, _, payload in outline_blocks
    }
    for paragraph_id, _, payload in blueprint_blocks:
        parent = re.search(r"Parent-Outline-SHA256: `([0-9a-f]{64})`", payload)
        assert parent is not None
        assert parent.group(1) == expected[paragraph_id]


def test_experiment_story_is_consolidated_to_four_research_questions() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")
    headings = re.findall(r"^## 7\.\d+ RQ(\d) —", outline, flags=re.MULTILINE)
    assert headings == ["1", "2", "3", "4"]
    assert "RQ5" not in outline and "RQ6" not in outline

    rq1 = outline.index("## 7.2 RQ1")
    rq2 = outline.index("## 7.3 RQ2")
    rq3 = outline.index("## 7.4 RQ3")
    rq4 = outline.index("## 7.5 RQ4")
    assert rq1 < rq2 < rq3 < rq4
    assert "external occurrence" in outline.lower()
    assert "controlled explanation" in outline.lower()
    assert "external improvement" in outline.lower()


def test_product_manifold_is_removed_from_main_environment_table() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")
    env_section = outline[
        outline.index("## 7.1 Environments and evidence roles") : outline.index(
            "## 7.2 RQ1"
        )
    ]
    assert "Product" not in env_section.split("### Required environment description")[0]
    assert "Historical Product-manifold provenance" in env_section
    assert "C-U1" in env_section and "D-U1" in env_section
    assert "Hopper/D4RL" in env_section and "Countdown" in env_section


def test_quality_distance_isolation_closes_the_named_rival_explanation() -> None:
    strategy = STRATEGY.read_text(encoding="utf-8")
    outline = OUTLINE.read_text(encoding="utf-8")
    combined = (strategy + "\n" + outline).lower()

    assert "separate badness from distance" in combined
    assert "far-field samples are worse" in combined
    assert "negative-advantage severity" in combined
    assert "sample count" in combined
    assert "base coefficient" in combined
    assert "independent amplifier" in combined
    assert "same-state/same-ray" in combined
    assert "not the only factor" in combined


def test_theory_method_experiment_share_aggregate_negative_term() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")
    compact = " ".join(outline.split())

    assert "q\\mathbf m_-" in compact
    assert "q_\\lambda\\mathbf m_{-,\\lambda}" in compact
    assert "\\widehat{\\mathbf M}_t^-" in compact
    assert "This metric is mandatory" in outline
    assert "raw and weighted norm" in outline
    assert "equilibrium shift" in outline
    assert "terminal drift" in outline


def test_outline_excludes_live_execution_state_but_preserves_evidence_rules() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")

    live_state_tokens = [
        "BUDGET-MATCH-01",
        "TAPER-CONV-01",
        "130–149",
        "140/140",
        "shortlist-freeze",
    ]
    for token in live_state_tokens:
        assert token not in outline

    assert "hand off" not in outline.lower()  # avoid malformed authority wording
    assert "handoff" in outline
    assert "Only formal terminal-audited results" in outline
    assert "paired seeds" in outline
    assert "best and terminal" in outline


def test_fixed_advantage_stays_out_of_theory_and_failure_types_stay_separate() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")
    theory = outline[
        outline.index("# 5. Repulsive Dynamics") : outline.index(
            "# 6. Distributionally Robust"
        )
    ]

    assert "fixed advantage" not in theory.lower()
    assert "freeze the empirical update field" not in theory.lower()
    assert "global convergence" not in theory.lower()
    assert "task collapse" in outline
    assert "support/variance/probability boundary" in outline
    assert "NaN/Inf" in outline
    assert "held-out-context generalization" in outline
    assert "OOD generalization" in outline  # appears only as a prohibited term
