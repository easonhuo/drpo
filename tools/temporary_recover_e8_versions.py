from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

BRANCH = 'origin/dev/e8-four-result-closure-authority-export'
BASE = 'e3718a346e260d0d3666ba55542565b2907c703f'
PREFIXES = ('tools/tmp_e8_patch_small/', 'tools/tmp_e8_patch_chunks/', 'tools/tmp_e8_four_result_closure.patch.gz.b64')


def run(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(['git', *args], text=text)

commits = run('rev-list', '--reverse', f'{BASE}..{BRANCH}').splitlines()
sequence: list[dict[str, object]] = []
for commit in commits:
    subject = run('show', '-s', '--format=%s', commit).strip()
    names = run('diff-tree', '--no-commit-id', '--name-status', '-r', commit).splitlines()
    for row in names:
        parts = row.split('\t')
        status = parts[0]
        path = parts[-1]
        if not path.startswith(PREFIXES):
            continue
        data = b''
        if not status.startswith('D'):
            try:
                data = run('show', f'{commit}:{path}', text=False)
            except subprocess.CalledProcessError:
                pass
        sequence.append({
            'commit': commit,
            'subject': subject,
            'status': status,
            'path': path,
            'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest() if data else None,
            'data': data,
        })

candidates: list[tuple[str, bytes]] = []
# Every changed blob in chronological order; useful when the same filename is reused for successive chunks.
candidates.append(('all_changed_blobs_in_commit_order', b''.join(item['data'].strip() for item in sequence if item['data'])))
# Only subjects explicitly naming a patch chunk.
candidates.append(('chunk_subject_blobs', b''.join(item['data'].strip() for item in sequence if item['data'] and 'chunk' in str(item['subject']).lower())))
# Unique blob identities in chronological order.
seen: set[str] = set()
unique_parts: list[bytes] = []
for item in sequence:
    digest = item['sha256']
    if item['data'] and digest not in seen:
        seen.add(str(digest))
        unique_parts.append(item['data'].strip())
candidates.append(('unique_changed_blobs_in_commit_order', b''.join(unique_parts)))

attempts = []
selected = None
for name, joined in candidates:
    record: dict[str, object] = {'name': name, 'joined_bytes': len(joined)}
    compact = re.sub(rb'\s+', b'', joined)
    record['compact_bytes'] = len(compact)
    try:
        compressed = base64.b64decode(compact, validate=True)
        record['base64'] = 'PASS'
        record['compressed_bytes'] = len(compressed)
        patch = gzip.decompress(compressed)
        record['gzip'] = 'PASS'
        record['patch_bytes'] = len(patch)
        record['patch_sha256'] = hashlib.sha256(patch).hexdigest()
        if selected is None:
            selected = (name, patch)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    attempts.append(record)

serializable = []
for item in sequence:
    serializable.append({key: value for key, value in item.items() if key != 'data'})
payload = {
    'schema_version': 2,
    'base': BASE,
    'branch': BRANCH,
    'commit_count': len(commits),
    'changed_blob_sequence': serializable,
    'attempts': attempts,
    'selected': selected[0] if selected else None,
}
out_dir = Path('docs/experiments')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'E8_HISTORICAL_CLOSURE_VERSION_RECOVERY.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if selected is None:
    raise SystemExit('no chronological changed-blob reconstruction decoded')
name, patch = selected
patch_text = patch.decode('utf-8')
files = re.findall(r'^diff --git a/(.+?) b/(.+?)$', patch_text, flags=re.MULTILINE)
keys = re.compile(r'EXT-C-E8|E8_[A-Z0-9_]+|run_commit|results_commit|manifest_sha256|package_sha256|package_sha|expected_cells|actual_cells|completed_cells|terminal_audit|nan_inf|package_filename|source_package|result_path|artifact_sha|run_id|result_summary', re.IGNORECASE)
references = []
for line in patch_text.splitlines():
    if keys.search(line):
        cleaned = line[1:] if line.startswith(('+', '-', ' ')) else line
        if cleaned not in references:
            references.append(cleaned)
lines = [
    '# E8 completed-backlog version recovery', '',
    'Provenance recovery record only; not a scientific-status upgrade.', '',
    f'- reconstruction: `{name}`',
    f'- patch bytes: `{len(patch)}`',
    f'- patch SHA-256: `{hashlib.sha256(patch).hexdigest()}`',
    f'- changed files: `{len(files)}`', '',
    '## File inventory', '',
]
lines.extend(f'- `{right}`' for _, right in files)
lines += ['', '## Evidence references', '', '```text']
lines.extend(references)
lines += ['```', '']
(out_dir / 'E8_COMPLETED_BACKLOG_VERSION_RECOVERY.md').write_text('\n'.join(lines), encoding='utf-8')
