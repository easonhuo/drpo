#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('.github/workflows/tmp_e8_terminal_binding_round2.sh')
text = path.read_text(encoding='utf-8')
start = text.index('# The compact terminal-evidence helper fixture needs all deterministic aggregate')
end = text.index("old = '''    audit = {", start)
replacement = '''# The compact terminal-evidence helper fixture needs all deterministic aggregate
# files plus the new scheduler/event hashes and exact audit success fields.
func_start = text.index(
    "def test_completed_workload_terminal_evidence_rejects_stale_hash_chain("
)
aggregate_start = text.index("    aggregate = {", func_start)
scheduler_start = text.index(
    "    p0.atomic_json(\\n        tmp_path / \\\"scheduler\\\" / \\\"dynamic_run.json\\\",",
    aggregate_start,
)
aggregate_fixture = """    aggregate_root = tmp_path / \\\"aggregate\\\"\n    aggregate_root.mkdir(parents=True, exist_ok=True)\n    (aggregate_root / \\\"all_cells.csv\\\").write_text(\\\"cell_key\\\\n\\\", encoding=\\\"utf-8\\\")\n    (aggregate_root / \\\"plot_curve_points.csv\\\").write_text(\\\"cell_key\\\\n\\\", encoding=\\\"utf-8\\\")\n    (aggregate_root / \\\"task_summary.csv\\\").write_text(\\\"task\\\\n\\\", encoding=\\\"utf-8\\\")\n    p0.atomic_json(\n        aggregate_root / \\\"countdown_protocol_diagnostic.json\\\",\n        {\\\"status\\\": \\\"NOT_RUN_ENGINEERING\\\"},\n    )\n    aggregate = {\n        \\\"experiment_id\\\": exp_tuning.experiment_id(config),\n        \\\"source_commit\\\": source_commit,\n        \\\"config_hash\\\": exp_tuning.stable_config_hash(config),\n        \\\"cell_count\\\": expected_cells,\n        \\\"cell_manifest_set_sha256\\\": exp_tuning._cell_manifest_set_sha256(config, tmp_path),\n        \\\"aggregate_artifact_sha256\\\": exp_tuning._baseline_aggregate_artifact_sha256(tmp_path),\n    }\n    p0.atomic_json(aggregate_root / \\\"aggregate_summary.json\\\", aggregate)\n"""
text = text[:aggregate_start] + aggregate_fixture + text[scheduler_start:]

'''
text = text[:start] + replacement + text[end:]

# Ruff SIM102: make the generated finalize guard one combined condition.
nested = '''new = ''' + "'''" + '''    if _is_baseline_matrix(config):
        if not _baseline_terminal_audit_identity_matches(
            config, output_root, audit, aggregate
        ):
            raise RuntimeError(
                \"Baseline terminal audit is stale or inconsistent with current terminal evidence\"
            )
''' + "'''"
combined = '''new = ''' + "'''" + '''    if _is_baseline_matrix(config) and not _baseline_terminal_audit_identity_matches(
        config, output_root, audit, aggregate
    ):
        raise RuntimeError(
            \"Baseline terminal audit is stale or inconsistent with current terminal evidence\"
        )
''' + "'''"
if text.count(nested) != 1:
    raise SystemExit(f'finalize nested guard count={text.count(nested)}')
text = text.replace(nested, combined, 1)

# The preflight script imports the package from src/ without installing it.
text = text.replace(
    'python scripts/preflight_e8_multitask_config.py --config configs/e8_multitask_baseline_matrix_formal.yaml',
    'PYTHONPATH=src python scripts/preflight_e8_multitask_config.py --config configs/e8_multitask_baseline_matrix_formal.yaml',
    1,
)
text = text.replace(
    'git rm .github/workflows/tmp_e8_terminal_binding_round2.sh .github/workflows/tmp_e8_terminal_binding_round2.yml',
    'git rm .github/workflows/tmp_e8_terminal_binding_round2.sh .github/workflows/tmp_e8_terminal_binding_round2_fix.sh .github/workflows/tmp_e8_terminal_binding_round2.yml',
    1,
)
path.write_text(text, encoding='utf-8')
PY

bash .github/workflows/tmp_e8_terminal_binding_round2.sh
