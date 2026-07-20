from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

AUTH_ID = "GOV-STAGE5-PREINTEGRATION-REPORT-HISTORY-BUGFIX-2026-07-20"
AUTHORITY_PATH = Path("scripts/handoff_authority.py")
TEST_PATH = Path("tests/stage5_candidate_integration.py")
AUTH_RECORD = Path(
    "docs/governance_stage_authorizations/"
    "GOV-STAGE5-PREINTEGRATION-REPORT-HISTORY-BUGFIX-2026-07-20.yaml"
)
LEDGER_PATH = Path("docs/governance_pipeline_stage_status.yaml")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


hashes = {
    AUTHORITY_PATH.as_posix(): digest(AUTHORITY_PATH),
    TEST_PATH.as_posix(): digest(TEST_PATH),
}
record = yaml.safe_load(AUTH_RECORD.read_text(encoding="utf-8"))
record["authorized_file_hashes"] = hashes
AUTH_RECORD.write_text(
    yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=120),
    encoding="utf-8",
)
ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))
seen: set[str] = set()
for item in ledger["stages"]["stage_5"]["protected_files"]:
    path = item.get("path")
    if path in hashes:
        item["sha256"] = hashes[path]
        item["authorized_by"] = AUTH_ID
        seen.add(path)
if seen != set(hashes):
    raise SystemExit(f"missing protected paths: {sorted(set(hashes) - seen)}")
LEDGER_PATH.write_text(
    yaml.safe_dump(ledger, sort_keys=False, allow_unicode=True, width=120),
    encoding="utf-8",
)
