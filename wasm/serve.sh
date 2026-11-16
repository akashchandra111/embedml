#!/usr/bin/env bash
# Serves build-web/ so the browser can load the wasm module.
#
#   wasm/serve.sh            http://127.0.0.1:8765
#   PORT=9000 wasm/serve.sh  another port
#
# A tiny Python server is enough. It returns the right MIME type for .wasm on
# current Python versions; the page also works when it does not, because the
# loader falls back to WebAssembly.instantiate(arrayBuffer).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DIR="${ELML_WEB_OUT:-$ROOT/build-web}"
PORT="${PORT:-8765}"

if [[ ! -f "$DIR/elml_studio.wasm" ]]; then
    echo "wasm/serve.sh: $DIR/elml_studio.wasm is missing; run wasm/build.sh first" >&2
    exit 1
fi

echo "embedlml studio: http://127.0.0.1:$PORT/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DIR"
