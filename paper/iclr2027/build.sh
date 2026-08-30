#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ICLR_DIR="$ROOT/paper/iclr2027"
SOURCE_DIR="$ROOT/paper/overleaf"
BUILD_DIR="$ICLR_DIR/build"
RELEASE_DIR="$ICLR_DIR/release"
OFFICIAL_URL="https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip"

rm -rf "$BUILD_DIR" "$RELEASE_DIR"
mkdir -p "$BUILD_DIR/template_extract" "$RELEASE_DIR"

command -v curl >/dev/null
command -v unzip >/dev/null
command -v latexmk >/dev/null
command -v pdfinfo >/dev/null

printf 'Downloading official ICLR 2027 style archive:\n  %s\n' "$OFFICIAL_URL"
curl -fL --retry 4 --retry-delay 2 --connect-timeout 20 \
  "$OFFICIAL_URL" -o "$BUILD_DIR/iclr-2027-style-files.zip"
sha256sum "$BUILD_DIR/iclr-2027-style-files.zip" > "$BUILD_DIR/OFFICIAL_TEMPLATE_SHA256.txt"
unzip -q "$BUILD_DIR/iclr-2027-style-files.zip" -d "$BUILD_DIR/template_extract"

STYLE_SRC="$(find "$BUILD_DIR/template_extract" -type f -name 'iclr2027_conference.sty' -print -quit)"
BST_SRC="$(find "$BUILD_DIR/template_extract" -type f -name 'iclr2027_conference.bst' -print -quit)"
if [[ -z "$STYLE_SRC" || -z "$BST_SRC" ]]; then
  echo "Official archive did not contain iclr2027_conference.sty/.bst" >&2
  find "$BUILD_DIR/template_extract" -maxdepth 3 -type f -print >&2
  exit 3
fi
cp "$STYLE_SRC" "$BUILD_DIR/iclr2027_conference.sty"
cp "$BST_SRC" "$BUILD_DIR/iclr2027_conference.bst"

bash "$ICLR_DIR/generate_iclr.sh" "$BUILD_DIR"

# Copy manuscript assets byte-for-byte. No figure or bibliography content is edited.
cp "$SOURCE_DIR/example_paper.bib" "$BUILD_DIR/example_paper.bib"
cp "$SOURCE_DIR/missing_references.bib" "$BUILD_DIR/missing_references.bib"
cp -a "$SOURCE_DIR/figures" "$BUILD_DIR/figures"

(
  cd "$SOURCE_DIR"
  find figures -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum example_paper.bib missing_references.bib
) > "$BUILD_DIR/SOURCE_ASSET_SHA256.txt"
(
  cd "$BUILD_DIR"
  find figures -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum example_paper.bib missing_references.bib
) > "$BUILD_DIR/PORT_ASSET_SHA256.txt"

# Paths in the two manifests differ by working directory but the checksums must match.
cut -d' ' -f1 "$BUILD_DIR/SOURCE_ASSET_SHA256.txt" > "$BUILD_DIR/.source_hashes"
cut -d' ' -f1 "$BUILD_DIR/PORT_ASSET_SHA256.txt" > "$BUILD_DIR/.port_hashes"
cmp "$BUILD_DIR/.source_hashes" "$BUILD_DIR/.port_hashes"
rm "$BUILD_DIR/.source_hashes" "$BUILD_DIR/.port_hashes"

cat > "$BUILD_DIR/TEMPLATE_PROVENANCE.txt" <<EOF
ICLR 2027 template provenance
official_url=$OFFICIAL_URL
archive_sha256=$(cut -d' ' -f1 "$BUILD_DIR/OFFICIAL_TEMPLATE_SHA256.txt")
style_sha256=$(sha256sum "$BUILD_DIR/iclr2027_conference.sty" | cut -d' ' -f1)
bst_sha256=$(sha256sum "$BUILD_DIR/iclr2027_conference.bst" | cut -d' ' -f1)
downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

pushd "$BUILD_DIR" >/dev/null
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error -file-line-error main.tex | tee latexmk.stdout.txt
pdfinfo main.pdf > PDFINFO.txt
pdffonts main.pdf > PDFFONTS.txt || true
{
  echo "# LaTeX layout/warning diagnostics"
  grep -E 'Overfull|Underfull|LaTeX Warning:|Package .* Warning:' main.log || true
} > LATEX_DIAGNOSTICS.txt

if grep -Eq 'There were undefined references|Citation .* undefined|Reference .* undefined' main.log; then
  echo "Undefined citation/reference detected" >&2
  exit 4
fi

PAGES="$(awk '/^Pages:/ {print $2}' PDFINFO.txt)"
printf 'compiled_pdf_pages=%s\n' "$PAGES" >> CONTENT_LOCK_REPORT.txt
printf 'figure_and_bibliography_asset_lock=PASS\n' >> CONTENT_LOCK_REPORT.txt
printf 'official_template_download=PASS\n' >> CONTENT_LOCK_REPORT.txt
popd >/dev/null

cp "$BUILD_DIR/main.pdf" "$RELEASE_DIR/main.pdf"
cp "$BUILD_DIR/main.tex" "$RELEASE_DIR/main.tex"
cp "$BUILD_DIR/iclr2027_conference.sty" "$RELEASE_DIR/iclr2027_conference.sty"
cp "$BUILD_DIR/iclr2027_conference.bst" "$RELEASE_DIR/iclr2027_conference.bst"
cp "$BUILD_DIR/example_paper.bib" "$RELEASE_DIR/example_paper.bib"
cp "$BUILD_DIR/missing_references.bib" "$RELEASE_DIR/missing_references.bib"
cp -a "$BUILD_DIR/figures" "$RELEASE_DIR/figures"
cp "$BUILD_DIR/CONTENT_LOCK_REPORT.txt" "$RELEASE_DIR/CONTENT_LOCK_REPORT.txt"
cp "$BUILD_DIR/TEMPLATE_PROVENANCE.txt" "$RELEASE_DIR/TEMPLATE_PROVENANCE.txt"
cp "$BUILD_DIR/OFFICIAL_TEMPLATE_SHA256.txt" "$RELEASE_DIR/OFFICIAL_TEMPLATE_SHA256.txt"
cp "$BUILD_DIR/SOURCE_ASSET_SHA256.txt" "$RELEASE_DIR/SOURCE_ASSET_SHA256.txt"
cp "$BUILD_DIR/PORT_ASSET_SHA256.txt" "$RELEASE_DIR/PORT_ASSET_SHA256.txt"
cp "$BUILD_DIR/PDFINFO.txt" "$RELEASE_DIR/PDFINFO.txt"
cp "$BUILD_DIR/PDFFONTS.txt" "$RELEASE_DIR/PDFFONTS.txt"
cp "$BUILD_DIR/LATEX_DIAGNOSTICS.txt" "$RELEASE_DIR/LATEX_DIAGNOSTICS.txt"
cp "$BUILD_DIR/main.log" "$RELEASE_DIR/main.log"

(
  cd "$RELEASE_DIR"
  zip -qr DRPO_ICLR2027_MECHANICAL_PORT.zip \
    main.pdf main.tex iclr2027_conference.sty iclr2027_conference.bst \
    example_paper.bib missing_references.bib figures \
    CONTENT_LOCK_REPORT.txt TEMPLATE_PROVENANCE.txt OFFICIAL_TEMPLATE_SHA256.txt \
    SOURCE_ASSET_SHA256.txt PORT_ASSET_SHA256.txt PDFINFO.txt PDFFONTS.txt \
    LATEX_DIAGNOSTICS.txt main.log
)

echo "ICLR 2027 mechanical port built at: $RELEASE_DIR/main.pdf"
echo "Uploadable bundle: $RELEASE_DIR/DRPO_ICLR2027_MECHANICAL_PORT.zip"
cat "$RELEASE_DIR/CONTENT_LOCK_REPORT.txt"
