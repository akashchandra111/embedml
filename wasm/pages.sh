#!/usr/bin/env bash
# Copies an existing browser build into the GitHub Pages folder.
#
#   ./wasm/build.sh          # build first; this script never builds
#   ./wasm/pages.sh          # copy build-web/ into docs/
#
# GitHub Pages serves this repository's docs/ folder. The script does not run
# git, so review and commit the copied files yourself (the steps are printed
# at the end).
#
# ELML_WEB_OUT picks the source (default: build-web) and ELML_PAGES_OUT the
# destination (default: docs).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="${ELML_WEB_OUT:-$ROOT/build-web}"
DEST="${ELML_PAGES_OUT:-$ROOT/docs}"

# Only the page assets are published. build-web/ also holds obj/ and libcui.a,
# which must not end up in docs/.
FILES=(index.html style.css loader.js studio.js elml_studio.wasm)

missing=()
for f in "${FILES[@]}"; do
    [[ -f "$SRC/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} )); then
    echo "wasm/pages.sh: missing ${missing[*]} in $SRC; run wasm/build.sh first" >&2
    exit 1
fi
if [[ ! -d "$SRC/samples" ]]; then
    echo "wasm/pages.sh: missing $SRC/samples; run wasm/build.sh first" >&2
    exit 1
fi

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
    cp "$SRC/$f" "$DEST/$f"
done
rm -rf "$DEST/samples"
cp -R "$SRC/samples" "$DEST/samples"
# Keep GitHub Pages from running Jekyll over the static files.
: > "$DEST/.nojekyll"

printf 'Copied the studio into %s\n\n' "$DEST"
cat <<EOF
Publish it by hand:

  git add docs
  git commit -m "Publish the browser studio"
  git push

Then enable Settings -> Pages -> Deploy from a branch: main, folder /docs.
The studio serves at https://<owner>.github.io/embedml/.

Only the page files and samples/ are copied; build-web/obj/ and libcui.a stay
out of docs/. Re-run ./wasm/build.sh then this script whenever the page or the
wasm module changes.
EOF
