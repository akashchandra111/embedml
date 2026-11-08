# Iris example: end-to-end training and MCU-style inference

This example trains a small classifier on the classic iris dataset, exports a
portable model file, evaluates it from the command line, embeds the model in a C
header, and runs a bare-metal style inference demo. It is the quickest way to
verify that the whole toolchain works.

The model is:

```
4 inputs -> dense 8 -> ReLU -> dense 3 -> softmax
loss: categorical cross-entropy, optimizer: Adam
```

## 1. Build the tools

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces:

* `build/elml_train` — host trainer
* `build/elml_run` — inference/evaluation runner
* `build/elml_inspect` — model printer
* `build/libelml_core.a` — the no-malloc inference library

## 2. Run everything

```sh
./examples/iris/run.sh
```

If your build directory is not `build`, pass it explicitly:

```sh
./examples/iris/run.sh /path/to/build
```

The script:

1. trains using `iris.ini`;
2. evaluates the exported `iris.elml` on all 150 rows;
3. writes an inspection report;
4. fails unless full-dataset accuracy is at least 0.90;
5. converts the model into `mcu/model_blob.h`;
6. compiles and runs `mcu/main.c` with `mcu/host_check.c`;
7. fails unless the three known samples are classified correctly.

Expected tail of the output (exact loss values may vary slightly by compiler and
floating-point mode):

```
train: loss 0.038075 accuracy 0.9917
validation: loss 0.123232 accuracy 0.9333
wrote iris.elml
rows: 150
accuracy: 0.9800 (147/150)
accuracy gate passed: 0.9800
embedded 960 bytes into mcu/model_blob.h
setosa     -> 0 (1.000 0.000 0.000)
versicolor -> 1 (0.004 0.977 0.019)
virginica  -> 2 (0.000 0.004 0.996)
```

`run.sh` writes these generated files into this directory:

| File | Contents |
| --- | --- |
| `iris.elml` | exported model |
| `train_output.txt` | full training log |
| `run_output.txt` | evaluation summary |
| `inspect_output.txt` | model inspection report |
| `mcu/model_blob.h` | model bytes as a C array |
| `mcu/host_check` | compiled host checker |

They are ignored by git.

## 3. Train manually

```sh
cd examples/iris
../../build/elml_train --config iris.ini
```

`iris.ini`:

```ini
[data]
csv = iris.csv
inputs = sepal_length,sepal_width,petal_length,petal_width
outputs = species
has_header = true
split = 0.8
seed = 42

[model]
layers = 8
activations = relu,softmax
loss = cce

[train]
epochs = 300
batch = 8
optimizer = adam
lr = 0.01
l2 = 0.0001
early_stop = 50
threads = 1

[export]
out = iris.elml
```

Key points:

* `inputs` and `outputs` accept header names or 1-based column indices.
* `layers` lists hidden layer sizes; the output layer is implicit.
* `activations` must contain one entry per dense layer, including the output:
  here `relu,softmax`.
* `loss = cce` turns the task into classification; `mse`/`mae` are regression.
* CLI flags override INI values, for example
  `--epochs 500 --lr 0.005`.
* Normalization is computed on the training split only and stored in the model.

Equivalent command without an INI file:

```sh
../../build/elml_train \
  --csv iris.csv \
  --inputs sepal_length,sepal_width,petal_length,petal_width \
  --outputs species \
  --layers 8 --act relu,softmax --loss cce \
  --epochs 300 --batch 8 --lr 0.01 --l2 0.0001 \
  --export iris.elml
```

## 4. Evaluate and predict

Evaluate on a labelled CSV:

```sh
../../build/elml_run iris.elml --csv iris.csv --evaluate
```

Output:

```
rows: 150
accuracy: 0.9800 (147/150)
```

The runner reads `columns.inputs` and `columns.outputs` from the model metadata,
so you do not need to repeat the column names. You may override them:

```sh
../../build/elml_run iris.elml --csv iris.csv --evaluate \
  --inputs sepal_length,sepal_width,petal_length,petal_width \
  --outputs species
```

Write predictions with the original columns plus a `prediction` column and one
probability column per class:

```sh
../../build/elml_run iris.elml --csv iris.csv --out predictions.csv
head -n 3 predictions.csv
```

```
sepal_length,sepal_width,petal_length,petal_width,species,prediction,prob_setosa,prob_versicolor,prob_virginica
5.1,3.5,1.4,0.2,setosa,setosa,1.000000,0.000000,0.000000
4.9,3.0,1.4,0.2,setosa,setosa,1.000000,0.000000,0.000000
```

Use `--limit N` to process only the first N data rows, `--no-header` for
headerless CSV, and `--delimiter ';'` for other separators (`\t` is accepted for
tabs).

For a single, ad-hoc question, pass the raw feature values directly:

```sh
../../build/elml_run iris.elml --predict 5.1,3.5,1.4,0.2
```

```
setosa 0.999771
```

Show the three best classes, emit JSON, or read the row from stdin:

```sh
../../build/elml_run iris.elml --predict 5.1,3.5,1.4,0.2 --top 3
../../build/elml_run iris.elml --predict 5.1,3.5,1.4,0.2 --json
printf '5.1,3.5,1.4,0.2\n' | ../../build/elml_run iris.elml --stdin
```

Keep the model loaded and answer many questions:

```sh
printf '5.1,3.5,1.4,0.2\n6.2,2.9,4.3,1.3\nquit\n' \
  | ../../build/elml_run iris.elml --repl --json
```

Or use the Python client, which spawns the REPL once and reuses it:

```sh
python3 ../repl/repl.py iris.elml "5.1,3.5,1.4,0.2"
```

## 5. Inspect the model

```sh
../../build/elml_inspect iris.elml
```

The report shows the format version, dtype, size, chunk count, input/output
shapes, every layer with its shape and parameter bytes, normalization modes,
metadata, required scratch bytes, and whether the model is runnable in this
build. `--json` prints the same information as JSON.

A healthy model reports four layers:

```
0  dense    [4] -> [8]  params=192B
1  relu     [8] -> [8]
2  dense    [8] -> [3]  params=192B
3  softmax  [3] -> [3]
```

## 6. Embed the model for an MCU build

Convert the binary to a C header:

```sh
python3 embed_model.py iris.elml mcu/model_blob.h
```

The generated header contains:

```c
static const uint8_t elml_example_model[] __attribute__((aligned(64))) = { ... };
static const size_t elml_example_model_len = 960u;
```

The alignment matters: `elml_model_init` requires the blob to be 64-byte
aligned. A `const` array in flash is normally fine; if your linker places it
oddly, add a section/alignment directive.

## 7. Integrate into your own firmware

The inference core needs only:

* the `include/elml` headers,
* `libelml_core.a` (or the individual core/format/kernel sources),
* `-lm` on platforms that require it.

Minimal integration:

```c
#include <stddef.h>
#include <stdint.h>
#include "elml/elml.h"
#include "model_blob.h"

/* 16-byte aligned, sized from elml_scratch_bytes(); 1 KiB is plenty here. */
static uint8_t g_scratch[1024] __attribute__((aligned(ELML_TENSOR_ALIGN)));

int predict(const float input[4], float output[3])
{
  elml_model model;
  elml_status status =
      elml_model_init(&model, elml_example_model, elml_example_model_len,
                      ELML_DTYPE_F32);
  if (status != ELML_OK) {
    return (int)status;
  }
  if (elml_scratch_bytes(&model, 1u) > sizeof g_scratch) {
    return (int)ELML_ERR_SCRATCH_TOO_SMALL;
  }
  return (int)elml_forward(&model, g_scratch, sizeof g_scratch, input, output);
}
```

Then pick the class with the largest output value. Because the model stores
input normalization, feed raw sensor/CSV values; do not standardize them in
firmware.

### Memory and timing notes

* `elml_model_init` copies descriptors into `elml_model` but does not copy the
  blob or the weights. The blob must stay alive for the model's lifetime.
* `elml_model` itself is a few kilobytes because it contains
  `ELML_MAX_LAYERS` descriptor and layer arrays. Reduce `ELML_MAX_LAYERS` at
  compile time for a smaller footprint, as long as it stays large enough for
  the model (iris needs 4).
* Scratch is two ping-pong activation buffers of the largest activation size,
  rounded up to 16 bytes. Iris needs 64 bytes; the demo reserves 1 KiB.
* Inference performs no allocation, no file I/O, and no printing.
* `elml_forward` is the only function in the hot path. `elml_model_init` can run
  once at boot.

### Checking the firmware build on the host

`mcu/host_check.c` compiles the same `mcu/main.c` used for the target and runs it
on the host. This catches integration mistakes without hardware:

```sh
cc -std=c23 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I../../include -Imcu \
  mcu/main.c mcu/host_check.c ../../build/libelml_core.a -lm \
  -o mcu/host_check
./mcu/host_check
```

For a cross-compiler, replace `cc` with your target compiler and link the core
library built for that target. Strip `mcu/host_check.c` from the firmware image;
it exists only for verification.

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `cannot load model: invalid tensor shape` | The file is corrupt or was edited; retrain. |
| `cannot load model: misaligned` | The blob is not 64-byte aligned in memory. |
| `cannot load model: dtype mismatch` | You passed a different `expected_dtype` than the file. |
| `inference failed: scratch too small` | Allocate at least `elml_scratch_bytes()` bytes. |
| `inference failed: misaligned` | Scratch is not 16-byte aligned. |
| `cannot resolve input columns` | Pass `--inputs` or check the CSV header. |
| Accuracy gate fails | Retrain with more epochs, a different seed, or a larger hidden layer; the dataset split is deterministic for a given seed. |

## 9. What this example does not cover

* int8 quantization: the format reserves a `QNT` chunk, but the quantizer and
  int8 runtime are not part of this milestone.
* Convolutional layers and non-sequential topologies.
* SIMD and multi-threaded training: `threads` currently clamps to 1.

See `docs/FORMAT.md` and `docs/CODE_GUIDE.md` for the full format and code map.
