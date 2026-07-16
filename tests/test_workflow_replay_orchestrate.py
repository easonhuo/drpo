from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.workflow_replay.orchestrate import (  # noqa: E402
    OrchestrationError,
    ProcessResult,
    run_candidate,
)

READY_SHA = "a" * 40


def response(state: str, **payload) -> ProcessResult:
    return ProcessResult(0, json.dumps({"status": "PASS", "state": state, **payload}))


class FixtureInvoker:
    def __init__(
        self,
        root: Path,
        *,
        registration: bool = True,
        fail_step: str | None = None,
        outside_preparation: bool = False,
    ):
        root.mkdir(parents=True, exist_ok=True)
        self.repo = root / "repo"
        self.repo.mkdir()
        self.spec = root / "spec.yaml"
        self.spec.write_text("schema_version: 1\n", encoding="utf-8")
        self.preparation_root = root / "preparations"
        self.transaction_root = root / "transactions"
        self.preparation_dir = self.preparation_root / "PREP-001"
        self.attempt = self.transaction_root / "PREP-001" / "attempt-0001"
        self.registration = registration
        self.fail_step = fail_step
        self.outside_preparation = outside_preparation
        self.commands = []

    def _prepared_files(self) -> None:
        overlay = (
            self.preparation_dir
            / "repository_overlay"
            / "docs"
            / "integrations"
            / "PREP-001"
        )
        overlay.mkdir(parents=True)
        (overlay / "INTEGRATION_REQUEST.yaml").write_text("request\n", encoding="utf-8")
        (overlay / "REVIEW_DECISION.yaml").write_text("review\n", encoding="utf-8")
        if self.registration:
            inputs = self.preparation_dir / "transaction_inputs"
            inputs.mkdir()
            (inputs / "REGISTRATION_INTENT.yaml").write_text("intent\n", encoding="utf-8")
            (inputs / "REGISTRATION_APPROVAL.yaml").write_text("approval\n", encoding="utf-8")

    def __call__(self, command) -> ProcessResult:
        self.commands.append(command)
        if command.name == self.fail_step:
            return ProcessResult(7, "", "injected failure")
        if command.name == "prepare-inputs":
            self._prepared_files()
            directory = (
                self.preparation_dir.parent.parent / "outside"
                if self.outside_preparation
                else self.preparation_dir
            )
            directory.mkdir(parents=True, exist_ok=True)
            return response(
                "PREPARED_INPUTS",
                preparation_id="PREP-001",
                preparation_dir=str(directory),
            )
        if command.name == "v1-plan":
            request = (
                self.repo
                / "docs"
                / "integrations"
                / "PREP-001"
                / "INTEGRATION_REQUEST.yaml"
            )
            review = request.with_name("REVIEW_DECISION.yaml")
            assert request.read_text(encoding="utf-8") == "request\n"
            assert review.read_text(encoding="utf-8") == "review\n"
            self.attempt.mkdir(parents=True)
            return response("REVIEWED", attempt_dir=str(self.attempt))
        if command.name == "v1-prepare":
            assert not (self.attempt / "REGISTRATION_INTENT.yaml").exists()
            return response("PREPARED")
        if command.name == "v1-normalize":
            if self.registration:
                assert (self.attempt / "REGISTRATION_INTENT.yaml").read_text() == "intent\n"
                assert (self.attempt / "REGISTRATION_APPROVAL.yaml").read_text() == "approval\n"
            return response("NORMALIZED")
        if command.name == "v1-gate":
            return response("REQUIRED_GATES_PASSED")
        if command.name == "v1-finalize":
            return response("READY", ready_commit_sha=READY_SHA)
        raise AssertionError(f"unexpected command: {command.name}")

    def run(self):
        return run_candidate(
            repo_root=self.repo,
            spec_path=self.spec,
            preparation_root=self.preparation_root,
            transaction_root=self.transaction_root,
            python_executable=sys.executable,
            invoke=self,
        )


def test_candidate_composes_existing_stages_and_places_inputs_once(tmp_path: Path) -> None:
    fixture = FixtureInvoker(tmp_path)
    outcome = fixture.run()
    assert outcome.ready_commit_sha == READY_SHA
    assert [command.name for command in outcome.commands] == [
        "prepare-inputs",
        "v1-plan",
        "v1-prepare",
        "v1-normalize",
        "v1-gate",
        "v1-finalize",
    ]
    assert len({command.argv for command in outcome.commands}) == 6
    assert set(outcome.placements) == {
        "repository:docs/integrations/PREP-001/INTEGRATION_REQUEST.yaml",
        "repository:docs/integrations/PREP-001/REVIEW_DECISION.yaml",
        "transaction:REGISTRATION_APPROVAL.yaml",
        "transaction:REGISTRATION_INTENT.yaml",
    }


def test_code_only_candidate_omits_transaction_input_placement(tmp_path: Path) -> None:
    fixture = FixtureInvoker(tmp_path, registration=False)
    outcome = fixture.run()
    assert outcome.ready_commit_sha == READY_SHA
    assert all(not item.startswith("transaction:") for item in outcome.placements)


def test_conflicting_repository_overlay_fails_before_v1_plan(tmp_path: Path) -> None:
    fixture = FixtureInvoker(tmp_path)
    target = fixture.repo / "docs" / "integrations" / "PREP-001"
    target.mkdir(parents=True)
    (target / "INTEGRATION_REQUEST.yaml").write_text("different\n", encoding="utf-8")
    with pytest.raises(OrchestrationError, match="conflicts") as caught:
        fixture.run()
    assert caught.value.step == "repository"
    assert [command.name for command in fixture.commands] == ["prepare-inputs"]


def test_nonzero_child_stops_without_later_component_invocation(tmp_path: Path) -> None:
    fixture = FixtureInvoker(tmp_path, fail_step="v1-plan")
    with pytest.raises(OrchestrationError, match="child exit 7") as caught:
        fixture.run()
    assert caught.value.step == "v1-plan"
    assert [command.name for command in fixture.commands] == ["prepare-inputs", "v1-plan"]


def test_preparation_directory_must_remain_under_declared_root(tmp_path: Path) -> None:
    fixture = FixtureInvoker(tmp_path, outside_preparation=True)
    with pytest.raises(OrchestrationError, match="escapes declared root") as caught:
        fixture.run()
    assert caught.value.step == "preparation_dir"


def test_symlinked_prepared_input_is_rejected(tmp_path: Path) -> None:
    fixture = FixtureInvoker(tmp_path)
    original = fixture._prepared_files

    def prepared_with_symlink() -> None:
        original()
        request = (
            fixture.preparation_dir
            / "repository_overlay"
            / "docs"
            / "integrations"
            / "PREP-001"
            / "INTEGRATION_REQUEST.yaml"
        )
        request.unlink()
        try:
            os.symlink(fixture.spec, request)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

    fixture._prepared_files = prepared_with_symlink
    with pytest.raises(OrchestrationError, match="symlink"):
        fixture.run()
    assert [command.name for command in fixture.commands] == ["prepare-inputs"]


def test_malformed_child_json_fails_closed(tmp_path: Path) -> None:
    fixture = FixtureInvoker(tmp_path)

    def malformed(command):
        fixture.commands.append(command)
        return ProcessResult(0, "not-json")

    with pytest.raises(OrchestrationError, match="not one JSON object"):
        run_candidate(
            repo_root=fixture.repo,
            spec_path=fixture.spec,
            preparation_root=fixture.preparation_root,
            transaction_root=fixture.transaction_root,
            python_executable=sys.executable,
            invoke=malformed,
        )


def test_fixture_orchestration_self_overhead_guardrail(tmp_path: Path) -> None:
    samples = []
    for index in range(20):
        fixture = FixtureInvoker(tmp_path / f"case-{index:02d}")
        started = time.perf_counter_ns()
        fixture.run()
        samples.append((time.perf_counter_ns() - started) / 1_000_000_000)
    assert statistics.median(samples) <= 1.0
