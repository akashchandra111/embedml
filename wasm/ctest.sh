#!/usr/bin/env bash
# CTest entry point for the browser build: build the wasm module, then run the
# Node smoke test against it.
#
#   wasm/ctest.sh [NODE]
#
# Honours the same variables as build.sh (CC, WASM_LD, LLVM_AR, ELML_WEB_OUT).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NODE="${1:-node}"

"$HERE/build.sh"
exec "$NODE" "$HERE/test.mjs" "${ELML_WEB_OUT:-$HERE/../build-web}"
