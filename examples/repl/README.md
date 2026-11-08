# Repeated prediction / Python REPL

The model is loaded once, then reused for every question. On an MCU the blob
can live in flash and the scratch buffers are allocated once at startup:

```c
elml_model model;
elml_model_init(&model, model_blob, model_blob_len, ELML_DTYPE_F32);

size_t scratch_len = elml_scratch_bytes(&model, 1u);
static uint8_t scratch[/* size at build time, 16-byte aligned */];

float x[INPUT_COUNT];
float y[OUTPUT_COUNT];

for (;;) {
  read_features(x);                          /* CSV line, sensor, UART, ... */
  elml_forward(&model, scratch, scratch_len, x, y);
  send_prediction(y);
}
```

`elml_forward` does not modify the model, so any number of calls can use the
same `elml_model`, scratch buffer, input buffer, and output buffer.

## Host REPL

`elml_run --repl` keeps the model and buffers alive and reads one CSV row per
line until EOF or `quit`/`exit`:

```sh
../../build/elml_run ../../examples/iris/iris.elml --repl --json
```

Protocol:

* input: one comma-separated feature row per line;
* output: one JSON object per line;
* blank lines are ignored;
* invalid rows report an error on stderr and do **not** terminate the loop;
* `quit` or `exit` ends the session.

Use `--top K` to include the K best classes. `--prompt` writes a `> ` prompt to
stderr, so stdout stays machine-readable.

## Python client

The example client spawns the C REPL once and reuses it:

```sh
python3 examples/repl/repl.py examples/iris/iris.elml
# features> 5.1,3.5,1.4,0.2
# setosa  0.999771
```

Batch mode:

```sh
python3 examples/repl/repl.py examples/iris/iris.elml \
  "5.1,3.5,1.4,0.2" "6.2,2.9,4.3,1.3"
```

Set `--runner` or `ELML_RUN` if `build/elml_run` is not in the default place.

For an MCU link, keep the same protocol but replace the subprocess pipes with
the serial/UART transport: write one feature line, read one JSON (or compact
binary) response. Nothing in the model needs to be reloaded between questions.
