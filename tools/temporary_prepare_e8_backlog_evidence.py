from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

LINEAR_BRANCH = 'origin/dev/e8-linear-scan-round1-result-closure'
TAU_BRANCH = 'origin/dev/e8-tau-curve-result-closure-01'
LINEAR_PREFIX = 'experiments/results/e8_paper_aligned_linear_scan_round1_pilot'
TAU_PREFIX = 'experiments/results/e8_paper_aligned_tau_curve_pilot'


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(['git', *args], text=text)


def copy_tree(ref: str, prefix: str) -> list[str]:
    paths = [
        row for row in git('ls-tree', '-r', '--name-only', ref, '--', prefix).splitlines()
        if row.startswith(prefix + '/')
    ]
    if not paths:
        raise SystemExit(f'no files found at {ref}:{prefix}')
    for rel in paths:
        target = Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git('show', f'{ref}:{rel}', text=False))
    return paths


linear_files = copy_tree(LINEAR_BRANCH, LINEAR_PREFIX)
tau_files = copy_tree(TAU_BRANCH, TAU_PREFIX)

records = [
    {
        'experiment_id': 'EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01',
        'label': '62-cell alpha-by-c continuous EXP grid',
        'cells': 62,
        'source_commit': '9de742532ac8559a0aba1282151e66cc1ce22f9e',
        'source_commit_state': 'dirty_pilot_snapshot_recorded_by_source_package',
        'source_artifact_zip_sha256': '9bc0b3a7623717bd17da29d2478ea4ed52150176e6b2d269fc81d88f9fb1964e',
        'evidence_mode': 'external_source_package_sha256_plus_recovered_compact_audit',
        'terminal_audit': 'PASS',
        'nan_inf_numerical_failures': 0,
        'test_split_used': False,
    },
    {
        'experiment_id': 'EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01',
        'label': '32-cell alpha=1 c scan',
        'cells': 32,
        'source_commit': 'a54dc74b849561c15f6195336fca446ed36f0640',
        'source_commit_state': 'clean_run_commit',
        'source_artifact_zip_sha256': '58522afed3072337138c29752efbf99ca8a4b65fe54a79adf1f7c153354416fb',
        'evidence_mode': 'external_source_package_sha256_plus_recovered_compact_audit',
        'terminal_audit': 'PASS',
        'nan_inf_numerical_failures': 0,
        'test_split_used': False,
    },
    {
        'experiment_id': 'EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01',
        'label': '32-cell alpha=1 high-c scan',
        'cells': 32,
        'source_commit': '929142930a3e2efaa7cafc8e4afe3866600027a5',
        'source_commit_state': 'clean_run_commit',
        'source_artifact_zip_sha256': '73fd7e21b7921e02bb67a0d8ddf4842431a3f6ccd07f80aea7aed2b273c6f53f',
        'evidence_mode': 'external_source_package_sha256_plus_recovered_compact_audit',
        'terminal_audit': 'PASS',
        'nan_inf_numerical_failures': 0,
        'test_split_used': False,
    },
    {
        'experiment_id': 'EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01',
        'label': '32-cell alpha=1 log-c boundary scan',
        'cells': 32,
        'source_commit': '05e8704770bda9a8682cd1031fa8b67bc3b55a41',
        'source_commit_state': 'clean_run_commit',
        'source_artifact_zip_sha256': '4f4237c1261289ca8a6f85850b216f6dd7c107598be1299f6d5f0f6587b941ec',
        'evidence_mode': 'external_source_package_sha256_plus_recovered_compact_audit',
        'terminal_audit': 'PASS',
        'nan_inf_numerical_failures': 0,
        'test_split_used': False,
    },
    {
        'experiment_id': 'EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01',
        'label': '32-cell paper-aligned linear-coordinate scan',
        'cells': 32,
        'run_id': 'E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01',
        'source_commit': 'f957e7f63c376e328e3d677cb143d526f6937c51',
        'source_artifact_zip_sha256': '00d53286a6642998b5563045bbb278876ba38713f98819c679ed4825e49bdd48',
        'results_repository': 'easonhuo/drpo-results',
        'results_branch': 'ingest/e8',
        'results_commit': '69a81c6db3bdd90ed640d54eb5687870d2dba220',
        'result_path': 'runs/e8/E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01',
        'manifest_sha256': '24635fbb634b23450cdfb560fd7b16a2dc0fe4a6d0586f10e1cf385e58bab333',
        'export_profile': 'manifest_text_v1',
        'terminal_audit': 'PASS',
        'nan_inf_numerical_failures': 0,
        'test_split_used': False,
    },
    {
        'experiment_id': 'EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-TAU-CURVE-0.5B-01',
        'label': '32-cell paper-aligned tau response surface',
        'cells': 32,
        'run_id': 'E8_PAPER_ALIGNED_TAU_CURVE_20260717_01',
        'source_commit': 'f9ea5a155ada50e9a4aebbe8ed08e8ffec82d66a',
        'source_commit_resolved': False,
        'source_artifact_zip_sha256': '8487365fc15be097733cad2df3fd235dba7a4cb44d4f5e19e37f40c937ae8982',
        'results_repository': 'easonhuo/drpo-results',
        'results_branch': 'ingest/e8',
        'results_commit': '69a81c6db3bdd90ed640d54eb5687870d2dba220',
        'result_path': 'runs/e8/E8_PAPER_ALIGNED_TAU_CURVE_20260717_01',
        'manifest_sha256': 'e737f94d9bb3f8c4dc08551ac2606a49cea39a639ba8b951d5f99d6d810524a0',
        'export_profile': 'manifest_text_v1',
        'terminal_audit': 'PASS',
        'nan_inf_numerical_failures': 0,
        'test_split_used': False,
    },
]

payload = {
    'schema_version': 1,
    'closure_id': 'EXT-C-E8-COMPLETED-BACKLOG-CLOSURE-2026-07-20',
    'environment': 'Countdown',
    'role': 'external_validity_pilot_closure',
    'experiment_count': len(records),
    'total_cells': sum(row['cells'] for row in records),
    'completed_cells': sum(row['cells'] for row in records),
    'terminal_audits_all_passed': True,
    'nan_inf_numerical_failures': 0,
    'test_split_used': False,
    'records': records,
    'copied_compact_files': sorted(linear_files + tau_files),
    'locked_interpretation': {
        'continuous_exp': 'Global negative pressure is reliably harmful; once tapering is sufficient, a broad usable coefficient band is more defensible than a sharp universal optimum. Historical evaluation RNG contamination and seed-by-hyperparameter variation prevent exact ranking claims.',
        'paper_aligned_linear': 'The best observed coefficient and tested right boundary are near-tied; the descending right branch was not localized in this two-seed pilot.',
        'tau_surface': 'The single-seed response surface is localization evidence only and does not identify a statistically supported optimum.',
    },
    'prohibited_claims': [
        'formal_method_ranking',
        'statistical_significance',
        'convergence_or_steady_state',
        'universal_exponential_superiority',
        'cross_task_or_cross_model_generalization',
        'OOD_generalization',
    ],
}

out_dir = Path('docs/experiments')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'E8_COMPLETED_BACKLOG_PROVENANCE.json').write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)

lines = [
    '# Countdown E8 completed-result backlog closure', '',
    'This record closes six completed validation-only external-validity pilots without rerunning training or strengthening their claims.', '',
    f'- Experiments: **{len(records)}**',
    f'- Completed cells: **{payload["completed_cells"]}/{payload["total_cells"]}**',
    '- Terminal audits: **PASS for all six pilot lines**',
    '- NaN/Inf numerical failures: **0**',
    '- Test split used: **no**', '',
    '## Evidence inventory', '',
]
for row in records:
    lines.append(f'- `{row["experiment_id"]}` — {row["cells"]}/{row["cells"]} cells; source `{row["source_commit"]}`; package `{row["source_artifact_zip_sha256"]}`.')
lines += [
    '', '## Locked interpretation', '',
    'The four historical continuous-EXP scans support a robust qualitative conclusion: uncontrolled Global negative pressure is harmful, while sufficient tapering opens a broad usable coefficient region. They do not support a sharp best coefficient because seed/trajectory variation is large and the historical evaluator changed global RNG state without restoration.', '',
    'The paper-aligned Linear scan localized a near-tied region around the best observed coefficient and the tested right boundary, but did not close the descending branch. The Tau curve is a single-seed response surface and cannot establish an optimum.', '',
    'Task performance, valid-expression/structure diagnostics, and NaN/Inf numerical failure remain separate. Every line remains a pilot: no significance, convergence, steady-state, formal method ranking, OOD claim, or universal exponential-superiority claim is authorized.', '',
    '## Provenance limitation', '',
    'The first four source packages predate automatic `drpo-results` delivery. Their full raw packages remain external and are bound here by the previously audited source commits and SHA-256 values. The Linear and Tau compact evidence is copied from the original result-closure branches and also bound to immutable manifests in `easonhuo/drpo-results`.', '',
]
(out_dir / 'E8_COMPLETED_BACKLOG_CLOSURE.md').write_text('\n'.join(lines), encoding='utf-8')

result_dir = Path('experiments/results/e8_completed_backlog_closure_20260720')
result_dir.mkdir(parents=True, exist_ok=True)
(result_dir / 'RESULT_CLOSURE.json').write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)

# Remove temporary recovery artifacts from this unmerged branch after durable evidence is assembled.
for pattern in (
    '.github/workflows/temporary_e8_*',
    'tools/temporary_*e8*',
    'docs/experiments/E8_HISTORICAL_CLOSURE_*',
    'docs/experiments/E8_COMPLETED_BACKLOG_RECOVERY_AUDIT.md',
    'docs/experiments/E8_COMPLETED_BACKLOG_VERSION_RECOVERY.md',
):
    for candidate in Path('.').glob(pattern):
        if candidate.is_file():
            candidate.unlink()
        elif candidate.is_dir():
            shutil.rmtree(candidate)
