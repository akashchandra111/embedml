#!/usr/bin/env bash
# Builds the embedml studio for the browser.
#
#   wasm/build.sh                 build build-web/elml_studio.wasm + the page
#   ELML_WASM_SIMD=0 wasm/build.sh   scalar kernels instead of SIMD128
#   wasm/serve.sh                 serve build-web/ on http://127.0.0.1:8765
#
# Plain clang + wasm-ld, no Emscripten and no downloaded sysroot. CUI's own
# wasm preview shim provides libc; this script reuses it and adds the
# embedml trainer and studio on top.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CUI="$ROOT/third_party/CUI"
OUT="${ELML_WEB_OUT:-$ROOT/build-web}"
OBJ="$OUT/obj"

CC=${CC:-clang}
LD=${WASM_LD:-wasm-ld}
AR=${LLVM_AR:-llvm-ar}
JOBS=${JOBS:-$(nproc)}
SIMD=${ELML_WASM_SIMD:-1}

# Keep scalar and SIMD objects apart: compile() only looks at timestamps, so a
# shared directory would happily link stale -msimd128 objects into a scalar
# build (or the reverse).
if [[ "$SIMD" == "1" ]]; then
    OBJ="$OUT/obj-simd"
else
    OBJ="$OUT/obj-scalar"
fi

if [[ ! -d "$CUI/web/shim/include" ]]; then
    echo "wasm/build.sh: CUI submodule missing; run: git submodule update --init" >&2
    exit 1
fi

mkdir -p "$OBJ" "$OUT/samples"

BASE_FLAGS=(--target=wasm32 -O2 -g0
            -fno-stack-protector -fno-exceptions
            -ffunction-sections -fdata-sections
            -mbulk-memory -mnontrapping-fptoint -msign-ext
            -ffile-prefix-map="$ROOT/"=
            -isystem "$CUI/web/shim/include")

CUI_FLAGS=(-std=c99
           -Wno-unused-function -Wno-unused-variable
           -Wno-unused-but-set-variable -Wno-c11-extensions
           -I "$CUI/include" -I "$CUI/src")
if [[ "$SIMD" == "1" ]]; then
    CUI_FLAGS+=(-msimd128)
fi

ELML_WARNINGS=(-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
               -Wmissing-prototypes -Wconversion -Wsign-conversion
               -Wcast-qual -Wundef -Wvla -Wdouble-promotion -Wformat=2 -Werror)

APP_FLAGS=(-std=c2x "${ELML_WARNINGS[@]}"
           -DELML_WASM=1 -DELML_ARCH_WASM=1
           -I "$ROOT/include" -I "$ROOT/src" -I "$CUI/include")
if [[ "$SIMD" == "1" ]]; then
    APP_FLAGS+=(-msimd128)
fi

LDFLAGS=(--no-entry --gc-sections --strip-all
         -z stack-size=4194304 --initial-memory=16777216 --max-memory=4294967296)

obj() { # source -> object path
    local src=$1
    local rel="${src#"$ROOT"/}"
    rel="${rel//\//_}"
    echo "$OBJ/${rel%.c}.o"
}

compile() { # source flags...
    local src=$1
    shift
    local o
    o="$(obj "$src")"
    if [[ ! -f "$o" || "$src" -nt "$o" || "$HERE/build.sh" -nt "$o" \
          || "$CUI/web/shim/include/_wasm_libc.h" -nt "$o" ]]; then
        "$CC" "${BASE_FLAGS[@]}" "$@" -c "$src" -o "$o"
    fi
}

# ---- CUI library, shim and browser host ------------------------------------
CUI_SRCS=("$CUI"/src/cui_*.c "$CUI/src/render/cui_render_sw.c")
CUI_OBJS=()
for src in "${CUI_SRCS[@]}"; do
    compile "$src" "${CUI_FLAGS[@]}" &
    CUI_OBJS+=("$(obj "$src")")
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
done
compile "$CUI/web/shim/libc.c" "${CUI_FLAGS[@]}" -fno-builtin &
LIBC_OBJ="$(obj "$CUI/web/shim/libc.c")"
compile "$CUI/web/host/web_host.c" "${CUI_FLAGS[@]}" &
HOST_OBJ="$(obj "$CUI/web/host/web_host.c")"
compile "$CUI/web/host/web_app.c" "${CUI_FLAGS[@]}" -Dmain=web_app_main &
APP_HOST_OBJ="$(obj "$CUI/web/host/web_app.c")"
wait

rm -f "$OUT/libcui.a"
"$AR" rcs "$OUT/libcui.a" "${CUI_OBJS[@]}"

# ---- embedml core, trainer and studio ------------------------------------
APP_SRCS=(
    src/core/arena.c
    src/core/error.c
    src/core/registry.c
    src/core/model.c
    src/core/layers/dense.c
    src/core/layers/activation.c
    src/format/crc32.c
    src/format/read.c
    src/format/write.c
    src/kernels/common/dispatch.c
    src/kernels/common/scalar_f32.c
    src/train/csv.c
    src/train/ini.c
    src/train/io.c
    src/train/memfs.c
    src/train/config.c
    src/train/dataset.c
    src/train/net.c
    src/train/threads.c
    src/train/evaluate.c
    src/train/session.c
    src/train/export.c
    src/studio/studio_main.c
)
SIMD_SRC=(src/kernels/wasm/simd128_f32.c)
if [[ "$SIMD" == "1" ]]; then
    APP_SRCS+=("${SIMD_SRC[@]}")
fi

APP_OBJS=()
for src in "${APP_SRCS[@]}"; do
    extra=()
    [[ "$src" == "src/studio/studio_main.c" ]] && extra+=(-Dmain=web_app_main)
    compile "$ROOT/$src" "${APP_FLAGS[@]}" "${extra[@]}" &
    APP_OBJS+=("$(obj "$ROOT/$src")")
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
done
wait

# ---- link ------------------------------------------------------------------
"$LD" "${LDFLAGS[@]}" \
    "${APP_OBJS[@]}" \
    "$HOST_OBJ" "$APP_HOST_OBJ" \
    "$OUT/libcui.a" "$LIBC_OBJ" \
    -o "$OUT/elml_studio.wasm"

# ---- page ------------------------------------------------------------------
cp "$HERE/web/index.html" "$HERE/web/style.css" \
   "$HERE/web/loader.js" "$HERE/web/studio.js" "$OUT/"
cp "$ROOT/examples/iris/iris.ini" "$ROOT/examples/iris/iris.csv" "$OUT/samples/"

printf 'built %s (%s KB, SIMD %s)\n' \
    "$OUT/elml_studio.wasm" \
    "$(( $(stat -c %s "$OUT/elml_studio.wasm") / 1024 ))" \
    "$([[ "$SIMD" == "1" ]] && echo on || echo off)"
printf 'serve it with: %s/serve.sh\n' "$HERE"
printf 'publish to docs/ with: %s/pages.sh\n' "$HERE"
