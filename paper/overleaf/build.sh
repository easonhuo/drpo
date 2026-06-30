#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BIBTEX_BIN="$(command -v bibtex || true)"
if [[ -z "$BIBTEX_BIN" && -x /usr/bin/bibtex.original ]]; then
  latexmk -e '$bibtex="/usr/bin/bibtex.original %O %B"' -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi
