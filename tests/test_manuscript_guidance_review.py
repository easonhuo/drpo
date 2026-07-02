from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "da5488ce13d0d6512b211ba4c68c9cdcd3fa49fc"
REVIEW_PATH = ROOT / "docs/manuscript/reviews/PAPER-PIPELINE-V2-DOMAIN-AGNOSTIC-05.json"
GUIDANCE = ROOT / "docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md"
PLAYBOOK = ROOT / "docs/manuscript/RL_PAPER_WRITING_PLAYBOOK.md"
STRATEGY = ROOT / "docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md"
CORPUS = ROOT / "docs/manuscript/RL_WRITING_CORPUS_NOTES.md"
OUTLINE = ROOT / "docs/paper_rewrite_outline_v0_9_2.md"
BLUEPRINT = ROOT / "docs/paper_rewrite_blueprint_v0_6.md"
PROSE = ROOT / "docs/paper_rewrite_prose_v0_1.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review() -> dict:
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def test_guidance_review_hashes_and_all_gates_are_current() -> None:
    review = load_review()
    assert review["base_commit"] == BASE_COMMIT
    assert review["verdict"] == "PASS"
    for artifact in review["artifacts"]:
        assert sha256(ROOT / artifact["path"]) == artifact["sha256"]
    gates = {gate["id"]: gate["status"] for gate in review["gates"]}
    assert gates == {f"G{i:02d}": "pass" for i in range(1, 43)}
    assert review["findings"]["blocker"] == []
    assert review["findings"]["major"] == []
    architecture = {gate["id"]: gate["status"] for gate in review["architecture_gates"]}
    assert architecture == {f"A{i:02d}": "pass" for i in range(1, 8)}


def test_stable_guidance_playbook_strategy_and_corpus_are_separate() -> None:
    guidance = GUIDANCE.read_text(encoding="utf-8")
    playbook = PLAYBOOK.read_text(encoding="utf-8")
    strategy = STRATEGY.read_text(encoding="utf-8")
    corpus = CORPUS.read_text(encoding="utf-8")
    assert "durable writing and review principles" in guidance
    assert "operational handbook" in playbook
    assert "DRPO Manuscript Strategy" in strategy
    assert "source" in corpus.lower() and "skill" in corpus.lower()
    for term in ["C-U1", "D-U1", "Hopper", "Countdown", "Theorem 1"]:
        assert term not in guidance
        assert term in strategy
    assert "do not automatically change" in guidance.lower()


def test_same_manuscript_rule_and_no_invented_hard_to_exp_continuity() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (STRATEGY, OUTLINE, BLUEPRINT, PROSE)
    ).lower()
    assert "one drpo manuscript" in combined
    assert "old drpo paper" in combined and "sequel" in combined  # appears only as a prohibition
    prohibited_positive_frames = [
        "the revised paper extends the original paper",
        "original formulation and revised formulation",
        "hard filtering naturally derives the exponential",
    ]
    for phrase in prohibited_positive_frames:
        assert phrase not in combined
    assert "quality selection" in combined and "remoteness" in combined


def test_p03_combines_existing_methods_and_matched_isolation() -> None:
    text = PROSE.read_text(encoding="utf-8")
    block = re.search(
        r"<!-- MANUSCRIPT:BEGIN INTRO-P03 -->(.*?)<!-- MANUSCRIPT:END INTRO-P03 -->",
        text,
        flags=re.S,
    )
    assert block is not None
    p03 = block.group(1).lower()
    for term in [
        "positive-only",
        "global coefficients",
        "clipping",
        "behavior constraints",
        "quality filtering",
    ]:
        assert term in p03
    for term in [
        "matching context",
        "negative-advantage magnitude",
        "sample count",
        "base coefficient",
    ]:
        assert term in p03
    assert "useful near the learner and destructive" in p03


def test_guidance_is_detailed_and_playbook_is_operational() -> None:
    guidance = GUIDANCE.read_text(encoding="utf-8")
    playbook = PLAYBOOK.read_text(encoding="utf-8")
    assert len(re.findall(r"^### G\d{2}\.", guidance, flags=re.M)) == 42
    required = [
        "End-to-end workflow",
        "Claim-evidence engineering",
        "Abstract construction",
        "Introduction paragraph recipes",
        "Theory construction",
        "Method construction",
        "Experiment and results construction",
        "Figures and tables",
        "Appendix construction",
        "Multi-reviewer audit",
        "Automated manuscript pipeline",
    ]
    for heading in required:
        assert heading.lower() in playbook.lower()


def test_generated_layers_and_overleaf_outputs_exist() -> None:
    for path in [
        OUTLINE,
        BLUEPRINT,
        PROSE,
        ROOT / "paper/overleaf/main.tex",
        ROOT / "paper/overleaf/main.pdf",
    ]:
        assert path.exists() and path.stat().st_size > 0
    assert (ROOT / "paper/overleaf/appendix/optimistic_dro.tex").exists()
    assert (ROOT / "paper/overleaf/figures/generated/fig1_story.pdf").exists()
    assert (ROOT / "paper/overleaf/figures/generated/fig2_phase_map.pdf").exists()
