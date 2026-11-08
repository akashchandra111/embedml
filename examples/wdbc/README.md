# WDBC: a real end-to-end dataset

This example trains on the **Breast Cancer Wisconsin (Diagnostic)** dataset and
checks the whole host pipeline: training, export, inspection, CSV evaluation,
one-shot prediction, the persistent REPL, and the Python client.

* Source: UCI Machine Learning Repository, *Breast Cancer Wisconsin
  (Diagnostic)*, <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>.
* 569 rows, 30 real-valued features, binary diagnosis (`B` benign, `M`
  malignant).
* `wdbc.csv` contains an `id` column (unused), the `diagnosis` target, and the
  30 features.

## Model

```
30 -> dense 24 -> ReLU -> dense 2 -> softmax
loss = cce, optimizer = adam, lr = 0.01, l2 = 0.0001
split = 0.8, seed = 42, early stop = 40
```

Training stops around epoch 57 with best epoch 17:

```
train:      loss 0.017784  accuracy 0.9934
validation: loss 0.015303  accuracy 1.0000
```

Evaluating the exported model on all 569 rows gives **0.9947 (566/569)**. The
three misses are training rows whose probabilities sit just below the decision
boundary; validation accuracy is 1.0000.

## Run it

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./examples/wdbc/run.sh
```

The script:

1. trains and exports `wdbc.elml`;
2. evaluates the full CSV (`--evaluate`) and gates accuracy at `>= 0.95` for
   both full-CSV and validation accuracy;
3. writes `inspect_output.txt`;
4. predicts one malignant and one benign row with `--predict`;
5. loads the model once and answers both rows through `--repl --json`;
6. runs the same two questions through `examples/repl/repl.py`.

It accepts a build directory as its first argument:

```sh
./examples/wdbc/run.sh "$PWD/build-simd"
```

## Manual run

```sh
cd examples/wdbc
../../build/elml_train --config wdbc.ini
../../build/elml_run wdbc.elml --csv wdbc.csv --evaluate
../../build/elml_inspect wdbc.elml
```

One row of features, in the order listed by `columns.inputs`:

```sh
row="$(awk -F, 'NR == 2 {for (i = 3; i <= NF; i++) printf "%s%s", $i, (i < NF ? "," : "")}' wdbc.csv)"
../../build/elml_run wdbc.elml --predict "$row" --top 2 --json
```

With the shipped model this prints the malignant class at probability 1.0.
Feed a line of features per query in a persistent loop:

```sh
../../build/elml_run wdbc.elml --repl --json
```

or from Python:

```sh
python3 ../repl/repl.py wdbc.elml
```

## Files

| File | Purpose |
| --- | --- |
| `wdbc.csv` | dataset with a header row |
| `wdbc.ini` | model, training, split, and export configuration |
| `run.sh` | end-to-end training + validation script (registered with CTest) |
| `wdbc.elml` | generated model (ignored by git) |
| `train_output.txt`, `run_output.txt`, `inspect_output.txt` | generated logs |
| `repl.jsonl`, `python_output.txt` | generated REPL/Python transcripts |

`wdbc.elml` is only 4.4 KB and needs 256 bytes of scratch for batch-1 inference,
so it is small enough to embed in an MCU like the iris model. The model file
records the 30 input column names and the `B,M` labels.
