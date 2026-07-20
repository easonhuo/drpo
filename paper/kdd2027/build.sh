#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 verify_content_lock.py
required_figures=(
  figures/figure1_external_gradient_bottom_labels.pdf
  figures/fig_6_3_1_source_heatmap.pdf
  figures/fig_6_3_2_rescue_plot.pdf
  figures/fig_6_4_1_phase_transition.pdf
  figures/fig_6_4_2_leftfig_bigtext_legend_protocol.pdf
)
for file in "${required_figures[@]}"; do test -s "$file" || { echo "missing figure: $file" >&2; exit 1; }; done
for file in example_paper.bib missing_references.bib; do test -s "$file" || { echo "missing bibliography: $file" >&2; exit 1; }; done
if [[ "${DRPO_SKIP_LATEX_COMPILE:-0}" != "1" ]]; then
  pdflatex --version > TEXLIVE_ENVIRONMENT.txt
  latexmk -C main.tex >/dev/null 2>&1 || true
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi
test -s main.pdf
test -s TEXLIVE_ENVIRONMENT.txt
if ! grep -Fq 'TeX Live 2025' TEXLIVE_ENVIRONMENT.txt; then
  echo 'KDD canonical build requires TeX Live 2025' >&2
  cat TEXLIVE_ENVIRONMENT.txt >&2
  exit 1
fi
if grep -Eqi 'Undefined control sequence|LaTeX Error|Citation .* undefined|Reference .* undefined|There were undefined references|There were undefined citations|multiply defined|Fatal error' main.log; then
  echo 'Unresolved LaTeX diagnostic detected' >&2
  grep -Ein 'Undefined control sequence|LaTeX Error|Citation .* undefined|Reference .* undefined|There were undefined references|There were undefined citations|multiply defined|Fatal error' main.log >&2 || true
  exit 1
fi
if grep -Fq 'Overfull \hbox' main.log; then
  echo 'Overfull horizontal box detected' >&2
  grep -Fn 'Overfull \hbox' main.log >&2 || true
  exit 1
fi
pdfinfo main.pdf > PDFINFO.txt
pdffonts main.pdf > PDFFONTS.txt
pdftotext main.pdf main.txt
for token in 'Yusen Huo' 'Changping Wang' 'Yangru Huang' 'Jun Zhang' 'Jie Jiang' 'Tencent Inc' 'neoxzhang@tencent.com'; do
  if grep -Fqi "$token" main.txt PDFINFO.txt; then echo "anonymous PDF contains forbidden identity token: $token" >&2; exit 1; fi
done
pages="$(awk '/^Pages:/ {print $2}' PDFINFO.txt)"
page_size="$(awk -F': +' '/^Page size:/ {print $2}' PDFINFO.txt)"
[[ -n "$pages" && "$pages" -ge 1 ]] || { echo 'invalid PDF page count' >&2; exit 1; }
[[ "$page_size" == *'612 x 792 pts'* ]] || { echo "unexpected page size: $page_size" >&2; exit 1; }
if awk 'NR>2 && ($5 != "yes" || $6 != "yes") {bad=1} END {exit bad}' PDFFONTS.txt; then :; else echo 'non-embedded or non-subset font detected' >&2; cat PDFFONTS.txt >&2; exit 1; fi
mkdir -p renders
rm -f renders/page-*.png
pdftoppm -png -r 150 main.pdf renders/page >/dev/null 2>&1
render_count="$(find renders -maxdepth 1 -name 'page-*.png' -type f | wc -l | tr -d ' ')"
[[ "$render_count" == "$pages" ]] || { echo "render count mismatch: pages=$pages renders=$render_count" >&2; exit 1; }
{
  echo "source_sha256=6e5ee53daf390f50ee2f2098826c02882a53257cbbe97ccee4a4ace1cde45dae"
  echo "canonical_content_sha256=842bc044055bf9695a23ce26411df6b93859d2582c6eb877ab7751bf8e5d6708"
  echo "pages=$pages"
  echo "page_size=$page_size"
  echo "texlive_version=2025"
  echo "required_figures=${#required_figures[@]}"
  echo "rendered_pages=$render_count"
  echo "pdf_sha256=$(sha256sum main.pdf | awk '{print $1}')"
  echo "content_lock=PASS"
  echo "anonymous_pdf=PASS"
  echo "latex_diagnostics=PASS"
  echo "font_embedding=PASS"
  echo "render_gate=PASS"
} > BUILD_AUDIT.txt
cat BUILD_AUDIT.txt
