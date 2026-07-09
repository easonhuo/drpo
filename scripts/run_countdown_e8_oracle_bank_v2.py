#!/usr/bin/env python3
"""One-click entrypoint for EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drpo.countdown_e8_oracle_bank_audit import run_standard_audit, thresholds_from_config  # noqa: E402
from drpo.countdown_e8_oracle_bank_v2 import build_oracle_corpus, load_config  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/countdown_e8_oracle_offline_bank_v2_0p5b.yaml",
        help="Oracle-bank-v2 YAML config.",
    )
    parser.add_argument(
        "--work_dir",
        required=True,
        help="Output directory. Use a new or empty directory for formal/pilot runs.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs already exist.")
    parser.add_argument("--skip_audit", action="store_true", help="Build corpus only; skip standardized REPORT/tables/figures audit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    outputs = build_oracle_corpus(config, Path(args.work_dir), force=args.force)
    print(f"train_corpus={outputs.train_corpus}", flush=True)
    print(f"audit_json={outputs.audit_json}", flush=True)
    print(f"run_manifest={outputs.run_manifest}", flush=True)
    if not args.skip_audit:
        audit_dir = Path(args.work_dir) / "audit"
        summary = run_standard_audit(
            outputs.train_corpus,
            audit_dir,
            thresholds=thresholds_from_config(config),
            make_zip=True,
        )
        print(f"standard_audit_dir={audit_dir}", flush=True)
        print(f"standard_audit_status={summary['status']}", flush=True)
        if summary.get("audit_zip"):
            print(f"standard_audit_zip={summary['audit_zip']}", flush=True)


if __name__ == "__main__":
    main()
