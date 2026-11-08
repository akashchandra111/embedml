#!/usr/bin/env bash
# End-to-end example:
#   1. train on iris.csv
#   2. evaluate the exported model on the full CSV
#   3. embed the model in a C header and run the bare-metal style demo
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-$here/../../build}"
cc="${CC:-cc}"

if [[ ! -x "$build_dir/elml_train" ]]; then
  echo "run.sh: build the project first (expected $build_dir/elml_train)" >&2
  echo "        cmake -S . -B build && cmake --build build" >&2
  exit 1
fi

cd "$here"

"$build_dir/elml_train" --config iris.ini | tee train_output.txt
"$build_dir/elml_run" iris.elml --csv iris.csv --evaluate | tee run_output.txt
"$build_dir/elml_inspect" iris.elml > inspect_output.txt

accuracy="$(awk '/^accuracy:/ {print $2}' run_output.txt)"
python3 - "$accuracy" <<'PY'
import sys
value = float(sys.argv[1])
if value < 0.90:
    raise SystemExit(f"accuracy gate failed: {value:.4f} < 0.90")
print(f"accuracy gate passed: {value:.4f}")
PY

"$build_dir/elml_run" iris.elml --predict 5.1,3.5,1.4,0.2 | grep -q '^setosa'
"$build_dir/elml_run" iris.elml --predict 5.1,3.5,1.4,0.2 --top 3 --json \
  | grep -q '"task":"classification".*"top":\['
echo "single-row prediction passed"

python3 embed_model.py iris.elml mcu/model_blob.h
"$cc" -std=c23 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I"$here/../../include" -I"$here/mcu" \
  mcu/main.c mcu/host_check.c "$build_dir/libelml_core.a" -lm \
  -o mcu/host_check
./mcu/host_check
