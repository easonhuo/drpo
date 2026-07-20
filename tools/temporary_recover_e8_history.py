from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

BRANCH = "origin/dev/e8-four-result-closure-authority-export"
BASE = "e3718a346e260d0d3666ba55542565b2907c703f"


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], text=text)


commits = git("rev-list", "--reverse", f"{BASE}..{BRANCH}").splitlines()
versions: dict[str, tuple[str, bytes]] = {}
interesting_prefixes = (
    "tools/tmp_e8_patch_small/",
    "tools/tmp_e8_patch_chunks/",
    "tools/tmp_e8_four_result_closure.patch.gz.b64",
)
for commit in commits:
    paths = git("ls-tree", "-r", "--name-only", commit).splitlines()
    for path in paths:
        if path.startswith(interesting_prefixes):
            try:
                data = git("show", f"{commit}:{path}", text=False)
            except subprocess.CalledProcessError:
                continue
            versions[path] = (commit, data)

attempts: list[dict[str, object]] = []
decoded_patch: bytes | None = None
selected_scheme: str | None = None

for prefix, stem in (
    ("tools/tmp_e8_patch_small/", "chunk"),
    ("tools/tmp_e8_patch_chunks/", "part"),
):
    rows: list[tuple[int, str, str, bytes]] = []
    for path, (commit, data) in versions.items():
        match = re.fullmatch(re.escape(prefix + stem) + r"(\d+)", path)
        if match:
            rows.append((int(match.group(1)), path, commit, data))
    rows.sort()
    indices = [row[0] for row in rows]
    contiguous = indices == list(range(indices[-1] + 1)) if indices else False
    joined = b"".join(row[3].strip() for row in rows)
    result: dict[str, object] = {
        "scheme": prefix + stem,
        "indices": indices,
        "contiguous_from_zero": contiguous,
        "joined_bytes": len(joined),
        "source_commits": {row[1]: row[2] for row in rows},
    }
    try:
        compressed = base64.b64decode(joined, validate=True)
        result["base64"] = "PASS"
        result["compressed_bytes"] = len(compressed)
        patch = gzip.decompress(compressed)
        result["gzip"] = "PASS"
        result["patch_bytes"] = len(patch)
        result["patch_sha256"] = hashlib.sha256(patch).hexdigest()
        if decoded_patch is None:
            decoded_patch = patch
            selected_scheme = prefix + stem
    except Exception as exc:  # diagnostic only
        result["decode_error"] = f"{type(exc).__name__}: {exc}"
    attempts.append(result)

out_dir = Path("docs/experiments")
out_dir.mkdir(parents=True, exist_ok=True)
inspection = {
    "schema_version": 1,
    "branch": BRANCH,
    "base": BASE,
    "commit_count": len(commits),
    "commits": commits,
    "discovered_paths": sorted(versions),
    "attempts": attempts,
    "selected_scheme": selected_scheme,
}
(out_dir / "E8_HISTORICAL_CLOSURE_HISTORY_RECOVERY.json").write_text(
    json.dumps(inspection, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

if decoded_patch is None:
    raise SystemExit("no historical chunk scheme produced a valid gzip patch")

patch_text = decoded_patch.decode("utf-8")
files = re.findall(r"^diff --git a/(.+?) b/(.+?)$", patch_text, flags=re.MULTILINE)
key_pattern = re.compile(
    r"EXT-C-E8|E8_[A-Z0-9_]+|run_commit|results_commit|manifest_sha256|package_sha256|"
    r"package_sha|expected_cells|actual_cells|completed_cells|terminal_audit|nan_inf|"
    r"package_filename|source_package|result_path|artifact_sha|run_id|result_summary",
    flags=re.IGNORECASE,
)
references: list[str] = []
for line in patch_text.splitlines():
    if key_pattern.search(line):
        cleaned = line[1:] if line.startswith(("+", "-", " ")) else line
        if cleaned not in references:
            references.append(cleaned)

markdown = [
    "# E8 completed-backlog history recovery",
    "",
    "This is a provenance recovery record, not a competing research master or a scientific-status upgrade.",
    "",
    f"- historical branch: `{BRANCH}`",
    f"- historical base: `{BASE}`",
    f"- commits inspected: `{len(commits)}`",
    f"- selected chunk scheme: `{selected_scheme}`",
    f"- decoded patch bytes: `{len(decoded_patch)}`",
    f"- decoded patch SHA-256: `{hashlib.sha256(decoded_patch).hexdigest()}`",
    f"- changed-file count in decoded patch: `{len(files)}`",
    "",
    "## Decoded patch file inventory",
    "",
]
markdown.extend(f"- `{right}`" for _, right in files)
markdown += ["", "## Recovered experiment and evidence references", "", "```text"]
markdown.extend(references)
markdown += ["```", ""]
(out_dir / "E8_COMPLETED_BACKLOG_RECOVERY_AUDIT.md").write_text(
    "\n".join(markdown), encoding="utf-8"
)
