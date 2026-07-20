from __future__ import annotations

import base64
import gzip
import hashlib
import re
import subprocess
from pathlib import Path

raw = subprocess.check_output(
    [
        "git",
        "show",
        "origin/dev/e8-four-result-closure-authority-export:tools/tmp_e8_four_result_closure.patch.gz.b64",
    ]
)
text = raw.decode("utf-8", errors="replace")
allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n\t ")
invalid = [(idx, ch, ord(ch)) for idx, ch in enumerate(text) if ch not in allowed]
compact = re.sub(r"\s+", "", text)
report = [
    "# E8 historical closure blob inspection",
    "",
    f"- raw bytes: `{len(raw)}`",
    f"- raw SHA-256: `{hashlib.sha256(raw).hexdigest()}`",
    f"- compact base64 characters: `{len(compact)}`",
    f"- compact length mod 4: `{len(compact) % 4}`",
    f"- invalid character count: `{len(invalid)}`",
    f"- first invalid characters: `{invalid[:20]}`",
    f"- first 120 characters: `{text[:120]}`",
    f"- last 240 characters: `{text[-240:]}`",
]
try:
    decoded = base64.b64decode(compact, validate=True)
    report.extend(
        [
            "- strict decode: `PASS`",
            f"- decoded bytes: `{len(decoded)}`",
            f"- decoded SHA-256: `{hashlib.sha256(decoded).hexdigest()}`",
        ]
    )
    try:
        payload = gzip.decompress(decoded)
        report.extend(
            [
                "- gzip decode: `PASS`",
                f"- patch bytes: `{len(payload)}`",
                f"- patch SHA-256: `{hashlib.sha256(payload).hexdigest()}`",
            ]
        )
    except Exception as exc:  # diagnostic only
        report.append(f"- gzip decode: `FAIL {type(exc).__name__}: {exc}`")
except Exception as exc:  # diagnostic only
    report.append(f"- strict decode: `FAIL {type(exc).__name__}: {exc}`")

out = Path("docs/experiments/E8_HISTORICAL_CLOSURE_BLOB_INSPECTION.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(report) + "\n", encoding="utf-8")
