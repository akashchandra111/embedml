#!/usr/bin/env bash
# Download Covertype from UCI if it is not already present.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
url="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
out="$here/covtype.data"

if [[ -s "$out" ]]; then
  echo "using existing $out"
  exit 0
fi

tmp="$out.tmp.$$"
trap 'rm -f "$tmp"' EXIT
curl -fsSL --max-time 300 "$url" | gzip -dc > "$tmp"
mv "$tmp" "$out"
trap - EXIT
echo "downloaded $out ($(wc -l < "$out") rows)"
