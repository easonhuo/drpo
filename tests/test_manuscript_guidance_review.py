from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "docs/manuscript/reviews/PAPER-V09-GUIDANCE-REVIEW.json"
BASE_COMMIT = "84edc2aa0b2f258033ddf2ef9aaf98e7a89a6edd"
OUTLINE = ROOT / "docs/paper_rewrite_outline_v0_9.md"
BLUEPRINT = ROOT / "docs/paper_rewrite_intro_blueprint_v0_4.md"
GUIDANCE = ROOT / "docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md"

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
        heading = next(line for line in payload.splitlines() if line.strip().startswith("## "))
        title = heading.split("]", 1)[1].strip()
        blocks.append((paragraph_id, title, payload))
    return blocks


def test_guidance_review_hashes_and_hard_gates_are_current() -> None:
    review = load_review()
    assert review["base_commit"] == BASE_COMMIT
    assert review["verdict"] == "PASS"

    guidance = ROOT / review["guidance_path"]
    assert sha256(guidance) == review["guidance_sha256"]

    for artifact in review["artifacts"]:
        path = ROOT / artifact["path"]
        assert sha256(path) == artifact["sha256"]

    gates = {gate["id"]: gate["status"] for gate in review["gates"]}
    assert gates == {f"G{i:02d}": "pass" for i in range(1, 15)}
    assert review["findings"]["blocker"] == []
    assert review["findings"]["major"] == []


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


def test_quality_distance_isolation_is_prominent_and_precise() -> None:
    guidance = GUIDANCE.read_text(encoding="utf-8")
    outline = OUTLINE.read_text(encoding="utf-8")
    blueprint = BLUEPRINT.read_text(encoding="utf-8")
    combined = "\n".join((guidance, outline, blueprint)).lower()

    assert "quality–distance" in combined or "quality-distance" in combined
    assert "separating badness from distance" in combined
    assert "distance is an independent" in combined
    assert "same-ray radial probe" in combined
    assert "jacobian-gain decomposition" in combined
    assert "distance is the only cause" in combined  # appears only as a prohibited claim


def test_product_manifold_is_historical_not_primary() -> None:
    guidance = GUIDANCE.read_text(encoding="utf-8")
    outline = OUTLINE.read_text(encoding="utf-8")

    assert "not a third primary paper environment" in guidance
    assert "不作为新版论文的第三个主环境" in outline
    assert "Historical Product-manifold construction" in outline
    assert "C-U1" in outline and "D-U1" in outline
    assert "Hopper" in outline and "Countdown" in outline


def test_fixed_advantage_stays_out_of_theory_scope() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")
    problem_setup = outline[
        outline.index("# 4. Problem Setup") : outline.index("# 5. Repulsive Dynamics")
    ]
    theory = outline[
        outline.index("# 5. Repulsive Dynamics") : outline.index(
            "# 6. Distributionally Robust"
        )
    ]

    assert "fixed advantage" not in problem_setup.lower()
    assert "freeze the empirical update field" not in theory.lower()
    assert "global convergence" not in theory.lower()


def test_theorem_method_experiment_bridge_is_explicit() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")
    compact = " ".join(outline.split())

    assert "q\\mathbf m_-" in compact
    assert "q_\\lambda\\mathbf m_{-,\\lambda}" in compact
    assert "Testable predictions and experiment mapping" in outline
    assert "Positive-only" in outline
    assert "stable extrapolation" in outline.lower()
    assert "boundary" in outline.lower()
    assert "no finite equilibrium" in outline.lower()


def test_experiment_story_uses_external_anchor_controlled_explanation_closure() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")
    anchor = outline.index("## 7.2 RQ1: Does the phenomenon appear in external policy learning?")
    isolation = outline.index("## 7.3 RQ2: Is distance an independent source")
    causal = outline.index("## 7.4 RQ3: Do far-field negative gradients causally")
    external_closure = outline.index("## 7.7 RQ6: Does DRPO improve external tasks?")
    assert anchor < isolation < causal < external_closure
    assert "reality anchor" in outline.lower()
    assert "reality closure" in outline.lower()
    assert "remain `TBD`" in outline


def test_current_experiment_statuses_are_not_overclaimed() -> None:
    outline = OUTLINE.read_text(encoding="utf-8")

    assert "BUDGET-MATCH-01` is completed and finite-step validated" in outline
    assert "TAPER-CONV-01" in outline
    assert "untouched seeds `130–149`" in outline
    assert "Hopper and Countdown remain `TBD`" in outline
    assert "task collapse, boundary event, and NaN/Inf" in outline
    assert "held-out-context/unseen-state" in outline
