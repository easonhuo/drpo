#!/usr/bin/env python3
"""Convert v2 oracle-offline corpus -> offline-bank jsonl for cmd_train_method.

The v2 corpus (``countdown_e8_oracle_bank_v2.py``) is model-independent. The
trainer consumes ``prompt``, ``positive``, ``near_negative``, ``far_negative``,
``negative_bank`` and ``pair_matched``. Per-batch bank width must be uniform, so
rows with fewer than 16 unique negatives are padded by cycling exact existing
expressions. The user confirmed this fixed-width protocol on 2026-07-11.

Training-time near/far selection uses the current model's length-normalized
surprisal over the 16 bank entries. It does not read ``base_surprisal``; that
field is therefore emitted as ``null`` rather than as a fake measurement.

Fail-closed input policy:
- zero negatives: raise;
- duplicate expressions in the source row: raise;
- more than 16 unique source negatives: raise;
- 1--15 unique source negatives: preserve the row and cycle exact expressions
  to width 16, marking padded entries and recording the padding count.

Crucially, all source-row integrity checks run before tokenizer/model-stack
initialization. Malformed input therefore fails closed without requiring CUDA,
Transformers, PEFT, or access to the model directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

V2_TRAIN = Path(
    "/root/experiment_output/e8_oracle_bank_v2/data/"
    "oracle_offline_bank_v2_train.jsonl"
)
OUT = Path("/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl")
MANIFEST = Path(
    "/root/experiment_output/e8_oracle_bank_v2/data/"
    "offline_bank_v2.convert_manifest.json"
)
MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
BANK_SIZE = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_and_pad_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]], int]], Counter[int], Counter[int]]:
    """Validate every source row before any tokenizer/model dependency is loaded."""

    prepared: list[tuple[dict[str, Any], list[dict[str, Any]], int]] = []
    padding_hist: Counter[int] = Counter()
    unique_hist: Counter[int] = Counter()

    for row in rows:
        negatives = list(row["negatives"])
        row_id = row.get("row_id")
        if not negatives:
            raise RuntimeError(f"row {row_id} has no negatives (corpus corruption)")

        expressions = [negative["expression"] for negative in negatives]
        unique_expressions = set(expressions)
        unique_hist[len(unique_expressions)] += 1

        if len(unique_expressions) != len(expressions):
            raise RuntimeError(
                f"row {row_id} has duplicate expressions in negatives "
                f"({len(expressions)} entries, {len(unique_expressions)} unique) "
                "-> fail-closed"
            )
        if len(unique_expressions) > BANK_SIZE:
            raise RuntimeError(
                f"row {row_id} has {len(unique_expressions)} unique negatives "
                f"(> {BANK_SIZE}) -> fail-closed (would silently truncate)"
            )

        padding_count = 0
        if len(negatives) < BANK_SIZE:
            source_negatives = list(negatives)
            index = 0
            while len(negatives) < BANK_SIZE:
                negatives.append(
                    {
                        **source_negatives[index % len(source_negatives)],
                        "_padding": True,
                    }
                )
                index += 1
                padding_count += 1

        padding_hist[padding_count] += 1
        prepared.append((row, negatives, padding_count))

    return prepared, padding_hist, unique_hist


def main() -> int:
    corpus_sha = _sha256(V2_TRAIN)
    rows = [json.loads(line) for line in V2_TRAIN.open()]

    # Integrity validation intentionally precedes tokenizer/HF-stack loading.
    prepared, padding_hist, unique_hist = _validate_and_pad_rows(rows)

    sys.path.insert(0, "/root/drpo/src")
    from drpo.countdown_qwen_arena_onefile import load_tokenizer  # noqa: E402

    tokenizer = load_tokenizer(MODEL)

    def token_length(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    out_rows: list[dict[str, Any]] = []
    for row, negatives, padding_count in prepared:
        bank = [
            {
                "expression": negative["expression"],
                "structure": negative.get("structure"),
                "base_surprisal": None,
                "token_length": token_length(negative["expression"]),
                "tree_depth": int(negative.get("tree_depth", 0)),
                "value_error": float(negative.get("value_error", 0.0)),
                "negative_bin": negative.get("negative_bin"),
                "padding": bool(negative.get("_padding", False)),
            }
            for negative in negatives
        ]

        # Required collator fallbacks. Bank methods ignore these static choices and
        # dynamically reselect current-policy near/far from the full bank.
        real_negatives = [
            negative for negative in negatives if not negative.get("_padding", False)
        ]
        by_error = sorted(
            real_negatives,
            key=lambda item: float(item.get("value_error", 0.0)),
        )

        converted = dict(row)
        converted.update(
            {
                "id": row.get("source_prompt_id") or row.get("row_id"),
                "oracle": row["oracle_positive"],
                "positive": row["oracle_positive"],
                "near_negative": by_error[0]["expression"],
                "far_negative": by_error[-1]["expression"],
                "negative_bank": bank,
                "negative_bank_size": BANK_SIZE,
                "negative_bank_padding_count": padding_count,
                "pair_matched": True,
                "bank_source": "oracle_offline_bank_v2",
            }
        )
        out_rows.append(converted)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_sha = _sha256(OUT)
    sizes = {len(row["negative_bank"]) for row in out_rows}
    all_pair_matched = all(row.get("pair_matched") for row in out_rows)
    rows_with_repeated_bank_expressions = sum(
        1
        for row in out_rows
        if len({entry["expression"] for entry in row["negative_bank"]}) != BANK_SIZE
    )
    rows_with_padding = sum(count for padding, count in padding_hist.items() if padding > 0)

    print(f"wrote {len(out_rows)} rows -> {OUT}")
    print(
        "bank_sizes="
        f"{sizes} all_pair_matched={all_pair_matched} "
        "rows_with_repeated_bank_expressions_due_to_padding="
        f"{rows_with_repeated_bank_expressions}"
    )
    print(f"unique-negatives-per-row hist: {dict(sorted(unique_hist.items()))}")
    print(f"padding_count hist: {dict(sorted(padding_hist.items()))}")
    print(f"sample keys: {sorted(out_rows[0].keys())}")

    manifest = {
        "source_corpus": str(V2_TRAIN),
        "source_corpus_sha256": corpus_sha,
        "output_bank": str(OUT),
        "output_bank_sha256": out_sha,
        "bank_size": BANK_SIZE,
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "scale_preserved": len(rows) == len(out_rows) == 6000,
        "padding_policy": "cycling_for_under_16_user_confirmed_2026_07_11",
        "fail_closed_policy": "zero_or_duplicate_expression_or_more_than_16_unique_raise_before_tokenizer_load",
        "duplicate_expression_rows": 0,
        "rows_with_padding": rows_with_padding,
        "padding_count_histogram": dict(sorted(padding_hist.items())),
        "unique_negatives_per_row_histogram": dict(sorted(unique_hist.items())),
        "base_surprisal": "null_by_design_provenance_only_training_reselects_near_far",
        "near_far_selection": "from_non_padded_negatives_by_value_error_collator_fallback_only",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
