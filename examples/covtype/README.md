# Covertype: large real-world validation

A larger counterpart to the iris and WDBC examples, using the **Forest
Covertype** dataset from UCI:

* Source: <https://archive.ics.uci.edu/ml/datasets/Covertype>
* 581,012 rows, 54 numeric features, 7 cover-type classes.
* The raw file has no header, so `covtype.ini` selects columns by 1-based
  index.
* The dataset is downloaded by `download.sh` and is **not** committed.

## Model

```
54 -> dense 128 -> ReLU -> dense 64 -> ReLU -> dense 7 -> softmax
loss = cce, optimizer = adam, lr = 0.002, l2 = 0.0001
split = 0.8, seed = 42, early stop = 5
```

Measured on this machine (x86_64, SIMD build, single thread):

| Metric | Value |
| --- | --- |
| Training | 28 epochs, ~95 s |
| Best epoch | 23 |
| Validation accuracy | 0.8979 |
| Full 581,012-row accuracy | 0.9015 (523,791 correct) |
| Model size | 64,200 bytes |
| Batch-1 scratch | 1,024 bytes |
| Full evaluation | ~3 s |
| 10,000 REPL predictions | ~61 ms (~164k predictions/s) |

The scalar and strict-FP builds produce the same accuracy; a single row on the
decision boundary can flip because of floating-point ordering.

## Run it

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./examples/covtype/run.sh
```

The script:

1. downloads and decompresses `covtype.data` if needed;
2. trains and exports `covtype.elml`;
3. evaluates all 581,012 rows and gates full-CSV and validation accuracy at
   `>= 0.88`;
4. checks the inspected input/output shapes;
5. runs one-shot prediction on the first row;
6. sends 1,000 rows through the persistent `--repl --json` loop and verifies
   one response per row;
7. runs the Python client against the same loaded model.

It accepts a build directory as its first argument:

```sh
./examples/covtype/run.sh "$PWD/build-simd"
```

This example is **not registered with CTest**: it needs a network download and
takes roughly 1-2 minutes, while the regular suite stays under a second.

Because the file has no header, evaluation must use `--no-header`; otherwise
the runner treats the first data row as a header and numeric column names such
as `1` can accidentally match feature values.

## Files

| File | Purpose |
| --- | --- |
| `covtype.ini` | model, training, split, and export configuration |
| `download.sh` | idempotent UCI download and decompression |
| `run.sh` | training, evaluation, REPL, and Python validation |
| `covtype.data` | generated dataset (ignored by git) |
| `covtype.elml` | generated model (ignored by git) |
| `train_output.txt`, `run_output.txt`, `inspect_output.txt` | generated logs |
| `predict.json`, `repl_input.csv`, `repl.jsonl`, `python_output.txt` | generated validation artifacts |
