#!/usr/bin/env bash
# Large real-world example:
#   1. download Covertype from UCI (if needed)
#   2. train a 54 -> 128 ReLU -> 64 ReLU -> 7 softmax model
#   3. evaluate all 581,012 rows and gate accuracy
#   4. inspect the model
#   5. run one-shot prediction, a 1000-row persistent REPL loop, and the
#      Python client
#
# This example is deliberately not registered with CTest: it downloads 11 MB
# and trains for roughly 1-2 minutes.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-$here/../../build}"

if [[ ! -x "$build_dir/elml_train" ]]; then
  echo "run.sh: build the project first (expected $build_dir/elml_train)" >&2
  echo "        cmake -S . -B build && cmake --build build" >&2
  exit 1
fi

cd "$here"

./download.sh

start=$(date +%s)
"$build_dir/elml_train" --config covtype.ini | tee train_output.txt
train_seconds=$(( $(date +%s) - start ))

start=$(date +%s)
"$build_dir/elml_run" covtype.elml --csv covtype.data --no-header --evaluate \
  | tee run_output.txt
eval_seconds=$(( $(date +%s) - start ))

"$build_dir/elml_inspect" covtype.elml > inspect_output.txt

accuracy="$(awk '/^accuracy:/ {print $2}' run_output.txt)"
val_accuracy="$(awk '/^validation:/ {print $NF}' train_output.txt)"
python3 - "$accuracy" "$val_accuracy" <<'PY'
import sys

full = float(sys.argv[1])
validation = float(sys.argv[2])
if full < 0.88:
    raise SystemExit(f"full-CSV accuracy gate failed: {full:.4f} < 0.88")
if validation < 0.88:
    raise SystemExit(
        f"validation accuracy gate failed: {validation:.4f} < 0.88"
    )
print(f"accuracy gates passed: full={full:.4f} validation={validation:.4f}")
PY

grep -q '\[54\]' inspect_output.txt
grep -q '\[7\]' inspect_output.txt
echo "inspect shape check passed"
echo "train: ${train_seconds}s, evaluate: ${eval_seconds}s"

row="$(awk -F, 'NR == 1 {
  for (i = 1; i <= 54; i++) printf "%s%s", $i, (i < 54 ? "," : "")
}' covtype.data)"

"$build_dir/elml_run" covtype.elml --predict "$row" --top 3 --json \
  > predict.json
grep -q '"task":"classification"' predict.json
grep -q '"top":\[' predict.json
echo "one-shot prediction passed"

head -1000 covtype.data | cut -d, -f1-54 > repl_input.csv
start=$(date +%s%N)
"$build_dir/elml_run" covtype.elml --repl --json < repl_input.csv \
  > repl.jsonl
repl_ms=$(( ($(date +%s%N) - start) / 1000000 ))
lines="$(wc -l < repl.jsonl)"
if [[ "$lines" -ne 1000 ]]; then
  echo "REPL answered $lines of 1000 rows" >&2
  exit 1
fi
echo "REPL answered 1000 rows in ${repl_ms} ms"

start=$(date +%s%N)
python3 ../repl/repl.py covtype.elml --runner "$build_dir/elml_run" --top 1 \
  "$row" > python_output.txt
python_ms=$(( ($(date +%s%N) - start) / 1000000 ))
grep -q '^[1-7] ' python_output.txt
echo "Python client passed in ${python_ms} ms"
