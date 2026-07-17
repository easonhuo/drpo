# DRPO paper reference code

This directory contains the compact paper-facing implementation developed under
`PAPER-CODE-REFERENCE-01`. It is intentionally separate from the historical
research and governance code in the repository.

Current status: Phase 1 shared-kernel implementation. No experiment is marked
reproduced by this scaffold alone.

Run the focused tests from this directory:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The acceptance contract is in
`../docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`.
