#!/usr/bin/env bash
set -euo pipefail

rm -rf repo
git clone --quiet --branch "$HEAD_BRANCH" \
  "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" repo
cd repo

docker run --rm \
  -e SOURCE_DATE_EPOCH \
  -e FORCE_SOURCE_DATE \
  -e TZ \
  -v "$PWD:/work" \
  -w /work \
  debian:trixie-slim \
  bash -lc '
    set -euo pipefail
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      ca-certificates \
      latexmk \
      poppler-utils \
      texlive-bibtex-extra \
      texlive-fonts-recommended \
      texlive-latex-base \
      texlive-latex-extra \
      texlive-latex-recommended \
      texlive-publishers \
      texlive-science >/dev/null
    pdflatex --version | head -n 2
    cd paper/kdd2027
    latexmk -C main.tex >/dev/null 2>&1 || true
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/tmp/latexmk.log 2>&1 || {
      tail -n 100 /tmp/latexmk.log
      exit 1
    }
    tail -n 10 /tmp/latexmk.log
    test -s main.pdf
    echo "PAGES=$(pdfinfo main.pdf | awk '\''/^Pages:/ {print $2}'\'')"
    echo "SIZE=$(stat -c %s main.pdf)"
    echo "SHA256=$(sha256sum main.pdf | awk '\''{print $1}'\'')"
  '

echo "GIT_BLOB=$(git hash-object paper/kdd2027/main.pdf)"

test "$(pdfinfo paper/kdd2027/main.pdf | awk '/^Pages:/ {print $2}')" = '27'
test "$(stat -c '%s' paper/kdd2027/main.pdf)" = '999042'
echo '4c1f576f79c2e6809fd86978515a7104a0f75719ca8361fbe5078dd8f7c6a1fb  paper/kdd2027/main.pdf' | sha256sum --check --strict
test "$(git hash-object paper/kdd2027/main.pdf)" = '50b8f320c21e677d81f8eda464e0d767983c5a8d'

cp paper/kdd2027/main.pdf paper/releases/DRPO_KDD2027_FABLE_VFINAL3_ACCEPTED.pdf
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git add paper/releases/DRPO_KDD2027_FABLE_VFINAL3_ACCEPTED.pdf
if git diff --cached --quiet; then
  exit 0
fi
git commit -m 'PAPER-FABLE-VFINAL3-RELEASE-FIX-01: replace incorrect release PDF'
git push origin "HEAD:${HEAD_BRANCH}"
