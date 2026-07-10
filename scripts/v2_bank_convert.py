#!/usr/bin/env python3
"""Convert v2 oracle-offline corpus -> offline-bank jsonl for cmd_train_method.

The v2 corpus (`countdown_e8_oracle_bank_v2.py`) is model-independent:
  row = {prompt, numbers, target, oracle_positive, oracle_structure, positives[], negatives[{expression, structure, tree_depth, value_error, negative_bin, ...}], ...}

`cmd_train_method`'s OfflineDataset/collator (`countdown_qwen_arena_onefile.py`) reads:
  row["prompt"], row["positive"], row["near_negative"], row["far_negative"],
  row.get("negative_bank", [])  (each entry: {"expression": ...} is all the collator reads),
  row["pair_matched"] must be True for every row (else training refuses),
  and per-batch bank_size must be uniform -> pad every row to 16.

Training-time near/far selection (`current_bank_training_batches`) uses the CURRENT
model's surprisal over the 16 bank entries (dynamic, recomputed each step). It does
NOT read `base_surprisal`; that field is build-time PROVENANCE ONLY, so we emit `null`
(not a fake 0.0 measurement) to make the provenance / measurement distinction
machine-readable. `token_length` is computed with the tokenizer for manifest fidelity.

This is a format conversion only: same v2 negatives, reformatted for the collator.
The scientific variable change (bank composition: model-sampled surprisal-spread ->
v2 value-stratified) was approved by the user.

§6 fail-closed policy (per dev spec):
  - any row whose `negatives` contain a DUPLICATE expression -> raise (real data bug)
  - any row with > 16 unique negatives -> raise (would silently truncate real data)
  - any row with < 16 unique negatives -> preserved at 6000 rows; pad by cycling
    existing negatives to satisfy the collator's uniform bank_size=16 constraint,
    but mark every padded entry with `padding: true` and record `padding_count` on
    the row. User explicitly authorized this relaxation (§6 #3/#4 loosened) so the
    frozen 6000-row corpus scale is NOT changed and the corpus is NOT rebuilt.
  - rows with zero negatives -> raise (corpus corruption).
A sidecar manifest records the corpus SHA-256, padding counts, and the policy.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

V2_TRAIN = Path("/root/experiment_output/e8_oracle_bank_v2/data/oracle_offline_bank_v2_train.jsonl")
OUT = Path("/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl")
MANIFEST = Path("/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.convert_manifest.json")
MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
BANK_SIZE = 16


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    sys.path.insert(0, "/root/drpo/src")
    from drpo.countdown_qwen_arena_onefile import load_tokenizer  # noqa: E402
    tok = load_tokenizer(MODEL)

    def tok_len(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    corpus_sha = _sha256(V2_TRAIN)
    rows = [json.loads(line) for line in V2_TRAIN.open()]
    out_rows = []
    padding_hist = Counter()        # padding_count -> number of rows
    unique_hist = Counter()         # unique-negatives-count -> number of rows
    dup_rows = 0
    for r in rows:
        negs = list(r["negatives"])
        if not negs:
            raise RuntimeError(f"row {r.get('row_id')} has no negatives (corpus corruption)")
        exprs = [n["expression"] for n in negs]
        unique_exprs = set(exprs)
        unique_hist[len(unique_exprs)] += 1
        if len(unique_exprs) != len(exprs):
            dup_rows += 1
            raise RuntimeError(
                f"row {r.get('row_id')} has duplicate expressions in negatives "
                f"({len(exprs)} entries, {len(unique_exprs)} unique) -> §6 fail-closed"
            )
        if len(unique_exprs) > BANK_SIZE:
            raise RuntimeError(
                f"row {r.get('row_id')} has {len(unique_exprs)} unique negatives "
                f"(> {BANK_SIZE}) -> §6 fail-closed (would silently truncate)"
            )
        # pad to BANK_SIZE by cycling existing negatives; mark padded entries.
        padding_count = 0
        if len(negs) < BANK_SIZE:
            base_negs = list(negs)
            i = 0
            while len(negs) < BANK_SIZE:
                negs.append({**base_negs[i % len(base_negs)], "_padding": True})
                i += 1
                padding_count += 1
        padding_hist[padding_count] += 1
        bank = []
        for n in negs:
            bank.append({
                "expression": n["expression"],
                "structure": n.get("structure"),
                "base_surprisal": None,            # provenance only; null = not a measurement
                "token_length": tok_len(n["expression"]),
                "tree_depth": int(n.get("tree_depth", 0)),
                "value_error": float(n.get("value_error", 0.0)),
                "negative_bin": n.get("negative_bin"),
                "padding": bool(n.get("_padding", False)),
            })
        # near_negative / far_negative are required by the collator for ALL rows but are
        # UNUSED by bank methods (bank methods reselect near/far from the bank via the
        # current model). Pick from the non-padded negatives by value_error so the row
        # is well-formed for any method and reflects real (non-cycled) negatives only.
        real_negs = [n for n in negs if not n.get("_padding", False)]
        by_err = sorted(real_negs, key=lambda x: float(x.get("value_error", 0.0)))
        near_neg = by_err[0]["expression"]
        far_neg = by_err[-1]["expression"]
        out = dict(r)  # preserve all v2 fields (numbers, target, oracle_structure, audit, ...)
        out.update({
            "id": r.get("source_prompt_id") or r.get("row_id"),
            "oracle": r["oracle_positive"],      # structure_reference_data reads row["oracle"] fallback
            "positive": r["oracle_positive"],
            "near_negative": near_neg,
            "far_negative": far_neg,
            "negative_bank": bank,
            "negative_bank_size": BANK_SIZE,
            "negative_bank_padding_count": padding_count,
            "pair_matched": True,
            "bank_source": "oracle_offline_bank_v2",
        })
        out_rows.append(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_sha = _sha256(OUT)
    print(f"wrote {len(out_rows)} rows -> {OUT}")
    # sanity: all bank sizes equal, all pair_matched, no duplicate expression within a row's bank
    sizes = {len(r["negative_bank"]) for r in out_rows}
    pm = all(r.get("pair_matched") for r in out_rows)
    intra_dup = sum(1 for r in out_rows
                    if len({e["expression"] for e in r["negative_bank"]}) != BANK_SIZE)
    print(f"bank_sizes={sizes}  all_pair_matched={pm}  rows_with_intra_bank_dup_expr={intra_dup}")
    print(f"unique-negatives-per-row hist: {dict(sorted(unique_hist.items()))}")
    print(f"padding_count hist: {dict(sorted(padding_hist.items()))}")
    print(f"sample keys: {sorted(out_rows[0].keys())}")
    rows_with_padding = sum(n for c, n in padding_hist.items() if c > 0)
    manifest = {
        "source_corpus": str(V2_TRAIN),
        "source_corpus_sha256": corpus_sha,
        "output_bank": str(OUT),
        "output_bank_sha256": out_sha,
        "bank_size": BANK_SIZE,
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "scale_preserved": len(rows) == len(out_rows) == 6000,
        "padding_policy": "cycling_for_under_16_only_user_authorized",
        "fail_closed_policy": "duplicate_expression OR more_than_16_unique -> raise",
        "duplicate_expression_rows": dup_rows,
        "rows_with_padding": rows_with_padding,
        "padding_count_histogram": dict(sorted(padding_hist.items())),
        "unique_negatives_per_row_histogram": dict(sorted(unique_hist.items())),
        "base_surprisal": "null_by_design_provenance_only_training_reselects_near_far",
        "near_far_selection": "from_non_padded_negatives_by_value_error",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
