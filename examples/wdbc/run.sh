#!/usr/bin/env bash
# End-to-end example on a real dataset:
#   1. train on wdbc.csv
#   2. evaluate the exported model on the full CSV
#   3. inspect the model
#   4. predict one benign and one malignant row, then reuse the model in a
#      REPL session and through the Python client
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-$here/../../build}"

if [[ ! -x "$build_dir/elml_train" ]]; then
  echo "run.sh: build the project first (expected $build_dir/elml_train)" >&2
  echo "        cmake -S . -B build && cmake --build build" >&2
  exit 1
fi

cd "$here"

"$build_dir/elml_train" --config wdbc.ini | tee train_output.txt
"$build_dir/elml_run" wdbc.elml --csv wdbc.csv --evaluate | tee run_output.txt
"$build_dir/elml_inspect" wdbc.elml > inspect_output.txt

accuracy="$(awk '/^accuracy:/ {print $2}' run_output.txt)"
val_accuracy="$(awk '/^validation:/ {print $NF}' train_output.txt)"
python3 - "$accuracy" "$val_accuracy" <<'PY'
import sys

full = float(sys.argv[1])
validation = float(sys.argv[2])
if full < 0.95:
    raise SystemExit(f"full-CSV accuracy gate failed: {full:.4f} < 0.95")
if validation < 0.95:
    raise SystemExit(
        f"validation accuracy gate failed: {validation:.4f} < 0.95"
    )
print(f"accuracy gates passed: full={full:.4f} validation={validation:.4f}")
PY

row_for() {
  awk -F, -v want="$1" 'NR > 1 && $2 == want {
    for (i = 3; i <= NF; i++) printf "%s%s", $i, (i < NF ? "," : "\n")
    exit
  }' wdbc.csv
}

malignant_row="$(row_for M)"
benign_row="$(row_for B)"

[[ "$("$build_dir/elml_run" wdbc.elml --predict "$malignant_row")" == M\ * ]]
[[ "$("$build_dir/elml_run" wdbc.elml --predict "$benign_row")" == B\ * ]]
echo "single-row predictions passed"

printf '%s\n%s\nquit\n' "$malignant_row" "$benign_row" \
  | "$build_dir/elml_run" wdbc.elml --repl --json > repl.jsonl 2> /dev/null
grep -q '"prediction":"M"' repl.jsonl
grep -q '"prediction":"B"' repl.jsonl
echo "REPL session passed"

python3 ../repl/repl.py wdbc.elml --runner "$build_dir/elml_run" --top 2 \
  "$malignant_row" "$benign_row" > python_output.txt
grep -q '^M  1\.' python_output.txt
grep -q '^B  0\.9' python_output.txt
echo "Python client passed"
