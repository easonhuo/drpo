#!/usr/bin/env python3
"""Convert v2 oracle-offline corpus -> offline-bank jsonl for cmd_train_method (GLM Dev Agent, one-off).

The v2 corpus (`countdown_e8_oracle_bank_v2.py`) is model-independent:
  row = {prompt, numbers, target, oracle_positive, oracle_structure, positives[], negatives[{expression, structure, tree_depth, value_error, negative_bin, ...}], ...}

`cmd_train_method`'s OfflineDataset/collator (`countdown_qwen_arena_onefile.py`) reads:
  row["prompt"], row["positive"], row["near_negative"], row["far_negative"],
  row.get("negative_bank", [])  (each entry: {"expression": ...} is all the collator reads),
  row["pair_matched"] must be True for every row (else training refuses),
  and per-batch bank_size must be uniform -> pad every row to 16.

Training-time near/far selection (`current_bank_training_batches`) uses the CURRENT
model's surprisal over the 16 bank entries (dynamic, recomputed each step). The bank's
`base_surprisal` field is build-time PROVENANCE ONLY (not read by training selection),
so we fill it with 0.0 placeholder (user choice). token_length is computed with the
tokenizer for manifest fidelity.

This is a format conversion only: same v2 negatives, reformatted for the collator.
The scientific variable change (bank composition: model-sampled surprisal-spread ->
v2 value-stratified) was approved by the user.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V2_TRAIN = Path("/root/experiment_output/e8_oracle_bank_v2/data/oracle_offline_bank_v2_train.jsonl")
OUT = Path("/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl")
MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
BANK_SIZE = 16


def main() -> int:
    sys.path.insert(0, "/root/drpo/src")
    from drpo.countdown_qwen_arena_onefile import load_tokenizer  # noqa: E402
    tok = load_tokenizer(MODEL)

    def tok_len(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    rows = [json.loads(l) for l in V2_TRAIN.open()]
    out_rows = []
    for r in rows:
        negs = list(r["negatives"])
        if not negs:
            raise RuntimeError(f"row {r.get('row_id')} has no negatives")
        # pad to BANK_SIZE by cycling existing negatives (uniform bank_size required by collator)
        if len(negs) < BANK_SIZE:
            i = 0
            while len(negs) < BANK_SIZE:
                negs.append(negs[i % (len(negs) - 1) if len(negs) > 1 else 0])
                i += 1
        elif len(negs) > BANK_SIZE:
            negs = negs[:BANK_SIZE]
        bank = []
        for n in negs:
            bank.append({
                "expression": n["expression"],
                "structure": n.get("structure"),
                "base_surprisal": 0.0,          # provenance placeholder (training uses current-model surprisal)
                "token_length": tok_len(n["expression"]),
                "tree_depth": int(n.get("tree_depth", 0)),
                "value_error": float(n.get("value_error", 0.0)),
                "negative_bin": n.get("negative_bin"),
            })
        # near_negative / far_negative are required by the collator for ALL rows but are
        # UNUSED by bank methods (bank methods reselect near/far from the bank via the
        # current model). Pick by value_error so the row is well-formed for any method.
        by_err = sorted(negs, key=lambda x: float(x.get("value_error", 0.0)))
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
            "pair_matched": True,
            "bank_source": "oracle_offline_bank_v2",
        })
        out_rows.append(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows -> {OUT}")
    # sanity: all bank sizes equal, all pair_matched
    sizes = {len(r["negative_bank"]) for r in out_rows}
    pm = all(r.get("pair_matched") for r in out_rows)
    print(f"bank_sizes={sizes}  all_pair_matched={pm}")
    print(f"sample keys: {sorted(out_rows[0].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())