# EmbedLML

EmbedLML is a small, portable neural-network library and toolchain for embedded
targets. You train a model on a host machine from CSV data, export a single
versioned binary file, and run inference on an MCU from a read-only blob in
flash with no `malloc`, no `stdio`, and no floating-point file I/O.

The project is deliberately split into two worlds:

* **Host tools** (`elml_train`, `elml_inspect`, the writer, CSV/INI/config
  code) may allocate and use the C standard library.
* **Inference core** (`libelml_core`) is allocation-free. Every temporary lives
  in a caller-supplied scratch buffer. Model bytes are read in place.

## Status

This is the v1 milestone. Implemented:

* f32 dense layers with fused ReLU, leaky ReLU, tanh, sigmoid, and linear
  activations; softmax output layers.
* Sequential topologies.
* Model format v1.0 with mandatory chunk CRCs, a whole-file CRC, metadata, and
  input/output normalization.
* Configuration from CLI flags and INI files; CLI flags override INI values.
* Regression (`mse`, `mae`) and classification (`cce`) training with Adam or
  SGD, L2 weight decay, early stopping, and best-weight restore.
* A no-malloc inference API, plus an inspect tool and a complete iris example
  with a bare-metal style demo.
* SIMD kernels for x86_64 (AVX2/FMA, runtime-dispatched), arm64 (NEON), and
  RISC-V (RVV), with a portable scalar fallback for MCUs.
* One-shot prediction and a persistent `--repl`/Python loop that loads the
  model once and reuses it for every question.
* An optional CUI desktop studio for editing training parameters and watching
  loss and accuracy update live.

Not yet implemented, and explicitly rejected rather than silently ignored:

* int8 training/quantization (`QNT` chunk is reserved; the runtime rejects it).
* `CONV1D` (layer id reserved).
* Multi-threaded training.
* Non-sequential topologies.

## Repository layout

```
include/elml/        public headers (C11-compatible)
src/core/            inference core: arena, registry, model, layers
src/format/          on-disk parser, writer, CRC32, inspect helpers
src/kernels/         common dispatch/scalar plus x86_64, arm64, and riscv kernels
src/train/           host-only trainer: CSV, INI, config, dataset, net, session, export
src/studio/          optional CUI desktop frontend for the trainer
src/run/             elml_run command-line inference/evaluation tool
tools/               elml_inspect
examples/iris/       end-to-end example, including an MCU-style demo
examples/wdbc/       real 30-feature dataset: train, evaluate, REPL, Python
examples/covtype/    optional 581k-row, 7-class dataset with download script
examples/repl/       persistent model loop and Python client
tests/               unit and end-to-end tests
docs/                format, code guide, coding style
```

## Build

Requirements: a C23 compiler, CMake 3.25 or newer, and `make` or Ninja.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `ELML_TRAIN` | `ON` | Build the host trainer, runner, and writer |
| `ELML_BUILD_TESTS` | `ON` | Build and register tests |
| `ELML_BUILD_EXAMPLES` | `ON` | Register the iris/WDBC examples with CTest |
| `ELML_SIMD` | `ON` | Build the kernel for the target architecture (x86_64 AVX2/FMA, arm64 NEON, RISC-V RVV) |
| `ELML_STRICT_FP` | `OFF` | Force the scalar f32 path |
| `ELML_BUILD_STUDIO` | `OFF` | Build the CUI desktop training studio |
| `ELML_CUI_SOURCE_DIR` | empty | Use a local CUI checkout instead of downloading it |
| `ELML_MAX_LAYERS` | `64` | Maximum layers per model |
| `ELML_MAX_RANK` | `4` | Maximum tensor rank |

For an MCU build, configure with `-DELML_TRAIN=OFF -DELML_BUILD_TESTS=OFF
-DELML_BUILD_EXAMPLES=OFF` and link only `libelml_core`. Add
`-DELML_SIMD=OFF` to use only the portable scalar kernels and let the vendor
compiler apply its own unrolling/vectorization. If you use a vendor toolchain
that does not accept C23, the public headers fall back to C11 through
`include/elml/compat.h`; compile the library sources with a C23 compiler or port
the few C23 spellings used internally.

### SIMD layout

Architecture-specific code lives under `src/kernels/<arch>/`:

| Target | File | Acceleration |
| --- | --- | --- |
| x86_64 | `x86_64/simd_f32.c` | AVX2 + FMA, runtime-dispatched (no AVX2 requirement for the rest of the binary) |
| arm64 | `arm64/neon_f32.c` | NEON (baseline on AArch64) |
| RISC-V | `riscv/vector_f32.c` | RVV 1.0, built when the toolchain defines `__riscv_vector` |
| any | `common/scalar_f32.c` | Portable scalar fallback and reference |

`kernels/common/dispatch.c` selects the implementation. Training and inference
call the same dense forward/backward/input-gradient kernels, so the SIMD code
accelerates both paths. The x86_64 forward and backward kernels process four
output rows per input load; dimensions not divisible by the vector width use a
scalar tail. `test_kernels` checks every SIMD kernel against the scalar
reference for vector and non-vector sizes.

## Quick start

Train and evaluate on the included iris dataset:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./examples/iris/run.sh
```

The example trains a 4 -> 8 ReLU -> 3 softmax classifier, checks accuracy, then
embeds the model into a C header and runs a bare-metal style inference demo. See
`examples/iris/README.md` for the full walkthrough.

For a real dataset, the WDBC example trains a 30 -> 24 ReLU -> 2 softmax
classifier and exercises evaluation, one-shot prediction, the persistent REPL,
and the Python client:

```sh
./examples/wdbc/run.sh
```

It usually reaches 1.0000 validation accuracy and about 0.995 on the full CSV.
See `examples/wdbc/README.md` for details.

For scale, `examples/covtype/` downloads the 581,012-row, 7-class Forest
Covertype dataset and trains a 54 -> 128 -> 64 -> 7 model. It is not part of the
fast CTest suite because it downloads data and trains for about 1-2 minutes:

```sh
./examples/covtype/run.sh
```

Manual training:

```sh
cd examples/iris
../../build/elml_train --config iris.ini
../../build/elml_run iris.elml --csv iris.csv --evaluate
```

Train without an INI file:

```sh
build/elml_train --csv data.csv --inputs x1,x2 --outputs y \
  --layers 8 --act relu,linear --loss mse \
  --epochs 200 --batch 16 --lr 0.01 --export model.elml
```

## Training studio (CUI)

`elml_studio` is an optional desktop frontend built on
[CUI](https://github.com/yappologistic/CUI). It edits the same parameters as
`elml_train`, runs training on a background thread, and draws loss and accuracy
after every epoch. Loading an `.ini` fills the form and resolves its relative
data and export paths next to the configuration file.

```sh
cmake -S . -B build-studio -DELML_BUILD_STUDIO=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-studio -j
./build-studio/elml_studio
```

The first configure downloads CUI at a pinned commit. To build against an
existing checkout instead (offline development, or to modify CUI):

```sh
cmake -S . -B build-studio -DELML_BUILD_STUDIO=ON \
  -DELML_CUI_SOURCE_DIR=/path/to/CUI
```

On Linux, install the X11 development headers. Native file dialogs use
`zenity` or `kdialog` when present; without them you can always type a path.
The studio is host-only: it does not change `libelml_core`, the default build,
or any existing CLI behavior.

The **Data** section reads the first CSV row and turns inputs and outputs into
searchable multi-select lists. A column can belong to only one side: choosing
it as an input clears it from the outputs, and vice versa. The same overlap
check runs before training. Headerless files show `Column 1`, `Column 2`, and
so on and pass 1-based indices to the trainer. Use **Rescan CSV columns** after
editing the file header by hand.

Useful flags:

```sh
./build-studio/elml_studio --config examples/wdbc/wdbc.ini
./build-studio/elml_studio --autostart   # start as soon as the window opens
```

Press **Stop and export best** at any time. The best monitored epoch is
restored and written to the configured `.elml` path; the same export happens
automatically when training finishes or early-stops.

## Single-row prediction

`elml_run` can also answer one ad-hoc question without a CSV:

```sh
# classification: label + probability
build/elml_run examples/iris/iris.elml --predict 5.1,3.5,1.4,0.2
# setosa 0.999771

# top-3 classes
build/elml_run examples/iris/iris.elml --predict 5.1,3.5,1.4,0.2 --top 3

# regression: one "name = value" line per output
build/elml_run model.elml --predict 1.0,-1.0
# y = 4.2379

# machine-readable, or read the row from stdin
build/elml_run model.elml --predict 1.0,-1.0 --json
printf '5.1,3.5,1.4,0.2\n' | build/elml_run examples/iris/iris.elml --stdin --json
```

The number of values must equal the model input dimension. `--predict` cannot
be combined with `--csv` or `--evaluate`.

## Reusing a loaded model (REPL / Python)

The model and scratch buffers can be reused indefinitely; `elml_forward` does
not modify the model. For an MCU, allocate the blob, scratch, input, and output
once at startup, then call `elml_forward` for every question:

```c
elml_model_init(&model, blob, blob_len, ELML_DTYPE_F32);
size_t scratch_len = elml_scratch_bytes(&model, 1u);
/* static 16-byte aligned scratch[scratch_len] on the MCU */

for (;;) {
  read_features(x); /* UART / sensor / Python request */
  elml_forward(&model, scratch, scratch_len, x, y);
  send_prediction(y);
}
```

On the host, `elml_run --repl` keeps everything loaded and speaks a one-line
protocol:

```sh
printf '5.1,3.5,1.4,0.2\n6.2,2.9,4.3,1.3\nquit\n' \
  | build/elml_run examples/iris/iris.elml --repl --json
```

One comma-separated row in, one JSON object out. Bad rows are reported and do
not stop the loop. A ready-made Python client is in
[`examples/repl/repl.py`](examples/repl/repl.py):

```sh
python3 examples/repl/repl.py examples/iris/iris.elml "5.1,3.5,1.4,0.2"
```

## Inference in an application

```c
#include <stdlib.h>
#include "elml/elml.h"

elml_model model;
if (elml_model_init(&model, blob, blob_len, ELML_DTYPE_F32) != ELML_OK) {
  /* reject the model */
}

size_t need = elml_scratch_bytes(&model, 1u);
void *scratch = malloc(need); /* or a static, 16-byte aligned buffer */
float y[3];
if (elml_forward(&model, scratch, need, x, y) != ELML_OK) {
  /* handle */
}
```

`elml_forward` applies the normalization stored in the model, so pass raw CSV
feature values. Regression outputs are denormalized automatically.

Two alignment requirements are part of the contract:

* The model blob must be aligned to `ELML_PARAM_ALIGN` (64 bytes).
* The scratch buffer must be aligned to `ELML_TENSOR_ALIGN` (16 bytes).

`elml_model_init` validates both and fails with `ELML_ERR_MISALIGNED` rather
than producing wrong results.

## Error handling

Every public function returns `elml_status` (or a pointer/size). Negative values
are errors; `elml_error_string` returns a stable human-readable name. The loader
is strict: unknown required chunks, duplicate chunks, bad CRCs, truncated files,
unsupported versions, wrong dtypes, and malformed shapes all fail the load.

## Documentation

* `docs/FORMAT.md` — the binary file format, byte by byte.
* `docs/CODE_GUIDE.md` — every file and every function in the project.
* `docs/CODING_STYLE.md` — conventions used by the codebase.
* `examples/iris/README.md` — complete end-to-end run guide.
* `examples/wdbc/README.md` — real-dataset training, evaluation, and REPL guide.
* `examples/covtype/README.md` — large 581k-row Covertype walkthrough.

## License

MIT. See `LICENSE`.
