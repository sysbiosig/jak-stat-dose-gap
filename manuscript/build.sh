#!/usr/bin/env bash
# Compile the manuscript. Run from this directory: bash build.sh
# Requires a TeX distribution (MacTeX puts pdflatex in /Library/TeX/texbin).
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"
pdflatex -interaction=nonstopmode main_all.tex >/dev/null
bibtex main_all >/dev/null
pdflatex -interaction=nonstopmode main_all.tex >/dev/null
pdflatex -interaction=nonstopmode main_all.tex >/dev/null
echo "undefined refs/cites: $(grep -ciE 'undefined (reference|citation)' main_all.log || true)"
echo "pages: $(pdfinfo main_all.pdf | awk '/^Pages/{print $2}')"
echo "-> main_all.pdf"
