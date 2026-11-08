#!/usr/bin/env bash
# End-to-end regression test for the CLI tools.
set -euo pipefail

train="$1"
run="$2"
work="$3"

rm -rf "$work"
mkdir -p "$work"
cd "$work"

python3 - <<'PY'
import random
random.seed(11)
with open('data.csv', 'w') as out:
    out.write('x1,x2,y\n')
    for _ in range(240):
        a = random.uniform(-2.0, 2.0)
        b = random.uniform(-2.0, 2.0)
        y = 1.5 * a - 2.0 * b + 0.75 + random.uniform(-0.02, 0.02)
        out.write(f'{a:.6f},{b:.6f},{y:.6f}\n')
PY

cat > config.ini <<'EOF'
[data]
csv = data.csv
inputs = x1,x2
outputs = y
split = 0.75
seed = 3

[model]
layers = 8
activations = relu,linear
loss = mse

[train]
epochs = 400
batch = 16
optimizer = adam
lr = 0.01
early_stop = 40

[export]
out = model.elml
EOF

"$train" --config config.ini --epochs 200 > train.log
"$run" model.elml --csv data.csv --evaluate --out predictions.csv > run.log

grep -q '^mse:' run.log
mse="$(awk '/^mse:/ {print $2}' run.log)"
python3 - "$mse" <<'PY'
import sys
value = float(sys.argv[1])
if value > 0.01:
    raise SystemExit(f"regression gate failed: mse {value:.6f} > 0.01")
print(f"regression gate passed: mse {value:.6f}")
PY

head -n 1 predictions.csv | grep -q 'pred_0'
lines="$(wc -l < predictions.csv)"
if [[ "$lines" -ne 241 ]]; then
  echo "expected 241 lines in predictions.csv, got $lines" >&2
  exit 1
fi

"$run" model.elml --csv data.csv --limit 7 --out limited.csv > /dev/null
lines="$(wc -l < limited.csv)"
if [[ "$lines" -ne 8 ]]; then
  echo "expected 8 lines in limited.csv, got $lines" >&2
  exit 1
fi

"$run" model.elml --predict 1.0,-1.0 > predict.log
grep -q '^y = ' predict.log
"$run" model.elml --predict 1.0,-1.0 --json | grep -q '"task":"regression"'
"$run" model.elml --predict 1.0,-1.0 --json | grep -q '"name":"y"'
printf '1.0,-1.0\n' | "$run" model.elml --stdin --json | grep -q '"name":"y"'
if "$run" model.elml --predict 1.0 > /dev/null 2>&1; then
  echo "elml_run accepted the wrong number of prediction values" >&2
  exit 1
fi
if "$run" model.elml --predict 1.0,-1.0 --csv data.csv > /dev/null 2>&1; then
  echo "elml_run accepted --predict together with --csv" >&2
  exit 1
fi

printf '1.0,-1.0\n0.5,0.5\nquit\n' \
  | "$run" model.elml --repl --json > repl.jsonl 2> /dev/null
lines="$(wc -l < repl.jsonl)"
if [[ "$lines" -ne 2 ]]; then
  echo "expected 2 REPL responses, got $lines" >&2
  exit 1
fi
if [[ "$(grep -c '"task":"regression"' repl.jsonl)" -ne 2 ]]; then
  echo "REPL did not return one JSON object per row" >&2
  exit 1
fi
printf 'bad\n1.0,-1.0\nexit\n' \
  | "$run" model.elml --repl --json > repl_bad.jsonl 2> /dev/null
if [[ "$(wc -l < repl_bad.jsonl)" -ne 1 ]]; then
  echo "a malformed REPL row terminated the session" >&2
  exit 1
fi
printf '1.0,-1.0\nquit\n' \
  | "$run" model.elml --repl --json --prompt > repl_prompt.jsonl 2> prompt.log
grep -q '^> ' prompt.log
if "$run" model.elml --repl --csv data.csv > /dev/null 2>&1; then
  echo "elml_run accepted --repl together with --csv" >&2
  exit 1
fi

source_dir="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$source_dir/examples/repl/repl.py" model.elml --runner "$run" \
  "1.0,-1.0" | grep -q '^y = '

"$train" --help > /dev/null
"$run" --help > /dev/null

if "$train" --config missing.ini > /dev/null 2>&1; then
  echo "elml_train accepted a missing config" >&2
  exit 1
fi
if "$run" missing.elml --csv data.csv > /dev/null 2>&1; then
  echo "elml_run accepted a missing model" >&2
  exit 1
fi
if "$run" model.elml --csv data.csv --evaluate --outputs missing > /dev/null 2>&1; then
  echo "elml_run accepted an unknown output column" >&2
  exit 1
fi

echo "cli regression test passed"
