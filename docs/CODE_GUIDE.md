# Code guide

This guide describes every source file and every function in EmbedLML. It is
organized by directory and follows the dependency direction: public API, then
inference core, format, kernels, host trainer, runner, tools, tests, and
examples.

Conventions used below:

* **Public** functions are declared in `include/elml/` and are part of the ABI.
* **Internal** functions are either `static` or declared in a header under
  `src/`.
* A *status* return is `elml_status`; `ELML_OK` is zero and errors are negative.
* "Scratch" always means the caller-owned buffer passed to `elml_forward`.

---

## 1. Build files

### `CMakeLists.txt`

Defines the project, options, warning set, and targets.

* `elml_core` (static library): inference only, no `malloc`, links `m` publicly.
  Sources are `arena.c`, `error.c`, `registry.c`, `model.c`, `dense.c`,
  `activation.c`, `crc32.c`, `read.c`, `kernels/common/dispatch.c`,
  `kernels/common/scalar_f32.c`, plus one architecture kernel file selected by
  `ELML_SIMD` and `CMAKE_SYSTEM_PROCESSOR`.
* `elml_train_lib` (static library, host only): writer plus trainer support.
  Sources are `write.c`, `csv.c`, `ini.c`, `io.c`, `config.c`, `dataset.c`,
  `net.c`, `session.c`, `export.c`.
* `elml_inspect`: parses and prints a model without running it.
* `elml_train`: the training executable.
* `elml_run`: the inference/evaluation executable.
* `elml_studio` (optional, `ELML_BUILD_STUDIO=ON`): CUI desktop frontend that
  links `elml_train_lib`, `Threads::Threads`, and `cui::cui`.
* `ELML_MAX_LAYERS` and `ELML_MAX_RANK` are public compile definitions so the
  public structs and the parser agree.
* Tests and the iris example are only registered when `ELML_TRAIN` is on.

### `tests/CMakeLists.txt`

* `elml_add_test(name)` builds one `tests/<name>.c` executable, links
  `elml_train_lib`, adds the test include directories, applies the strict warning
  set, registers it with CTest, and sets a timeout.
* Registers the C unit tests and the shell test `test_cli` (passing the built
  `elml_train` and `elml_run` paths plus a scratch directory).

### `.gitignore`

Ignores build trees, object/archive files, generated example model artifacts,
and test scratch directories.

---

## 2. Public headers (`include/elml/`)

### `compat.h`

C23-to-C11 spelling shim. When the compiler is not C23 and not C++, it defines
`nullptr` as `NULL`, `static_assert` as `_Static_assert`, `alignas` as
`_Alignas`, `alignof` as `_Alignof`, and `constexpr` as `const`. Also defines
`ELML_ARRAY_COUNT(a)`. No feature logic lives here.

### `types.h`

Core value types and limits.

* `ELML_MAX_LAYERS` (default 64) and `ELML_MAX_RANK` (default 4), overridable by
  the build.
* `ELML_PARAM_ALIGN` = 64 (model blob and parameter alignment).
* `ELML_TENSOR_ALIGN` = 16 (scratch alignment).
* `elml_status`: every error code used by the library.
* `elml_dtype`: `F32`, `INT8`, reserved `F16`.
* `elml_shape`: rank plus up to `ELML_MAX_RANK` dimensions.
* `elml_quant`: int8 scale and zero point.
* `elml_tensor`: data pointer, dtype, shape, quantizer for kernel calls.
* Compile-time assertions validate the configurable limits.

### `arena.h`

Declares the caller-owned monotonic allocator used to derive scratch layouts:

* `elml_arena_init`, `elml_arena_mark`, `elml_arena_alloc`, `elml_arena_reset`,
  `elml_align_up`.
* The core itself does not allocate through the arena during inference; the
  arena is available for applications that want to carve scratch or metadata
  storage.

### `format.h`

On-disk constants: format version, header and chunk sizes, magic bytes, FourCC
helpers and chunk ids (`CONF`, `PARM`, `QNT`, `NORM`, `META`, `TOPO`, `CKPT`),
chunk flags, and file flags. See `docs/FORMAT.md`.

### `registry.h`

Layer registry types.

* `elml_layer_id`: permanent ids for dense, reserved conv1d, and six activation
  types.
* `elml_layer_flags`: trainable, fused, has-bias, and the runtime-only skipped
  flag.
* `elml_activation`: activation selector used by the trainer and kernels.
* `elml_layer_desc`: parsed descriptor (type, version, flags, parameter range,
  shapes, extra bytes).
* `elml_layer`: bound layer (vtable, descriptor, parameter pointers, fused
  activation, leaky alpha, flags).
* `elml_layer_vt`: `prepare` and `forward` function pointers plus id, version,
  and name.
* `elml_layer_find` returns a built-in vtable or `NULL`.
* `elml_registry_add` registers a host-provided vtable before model loading. On
  static-registry builds it can be compiled to return
  `ELML_ERR_UNSUPPORTED_LAYER_TYPE`.

### `model.h`

The inference API.

* `elml_norm_mode`: none, standard, min-max.
* `elml_norm`: parsed normalization modes, counts, and parameter views.
* `elml_meta`: raw metadata view.
* `elml_model`: bound model: blob, shapes, descriptors, bound layers, PARM/QNT
  views, normalization, metadata, and computed scratch size.
* `elml_model_init`: validates and binds a model; checks dtype and alignment.
* `elml_model_dtype`: reads only the header to discover the dtype.
* `elml_scratch_bytes`: required scratch bytes.
* `elml_forward`, `elml_forward_batch`: f32 inference.
* `elml_forward_q`: int8 entry point (reserved; currently returns
  `ELML_ERR_UNSUPPORTED_DTYPE` for int8 models).
* `elml_meta_get`, `elml_label_count`, `elml_label_get`.
* `elml_error_string`, `elml_version_string`.

### `elml.h`

Umbrella header that includes all public headers in dependency order.

---

## 3. Inference core (`src/core/`)

### `arena.c`

Implements the monotonic arena.

* `elml_align_up(value, alignment)`: rounds up to a power-of-two alignment.
  Returns 0 for a non-power-of-two or on overflow.
* `elml_arena_init(arena, base, size)`: resets the arena over caller storage.
  Null arguments produce a zero-sized arena.
* `elml_arena_mark(arena)`: current used byte count (0 for null).
* `elml_arena_alloc(arena, bytes, alignment)`: allocates aligned storage from
  the arena or returns `NULL` without consuming space on failure. Zero-byte
  requests return the base pointer.
* `elml_arena_reset(arena, mark)`: rewinds to a previous mark; an out-of-range
  mark resets to zero.

### `error.c`

* `elml_error_string(status)`: maps every `elml_status` to a stable
  human-readable name; unknown values return `"unknown error"`.
* `elml_version_string()`: returns `"EmbedLML <major>.<minor>.<patch>"`.

### `registry.c`

Maintains the table of built-in layer vtables. In the default build
`ELML_STATIC_REGISTRY` is not defined, so a fixed table is consulted and
`elml_registry_add` is unavailable.

* `elml_registry_add(vt)`: host build: validates the vtable (non-null id, name,
  prepare, forward; no duplicate id) and registers it. In a static-registry
  build this overload returns `ELML_ERR_UNSUPPORTED_LAYER_TYPE`.
* `elml_layer_find(id)`: linear search over registered vtables; returns `NULL`
  if the type is unknown.

### `model.c`

Binds a parsed model and runs inference.

* `elml_shape_count(shape)`: product of dimensions using checked multiplication;
  returns 0 for a null/zero-rank shape or overflow.
* `elml_shape_equal(a, b)`: rank and dimension equality.
* `elml_activation_is_fusable(id)`: true for relu, leaky relu, tanh, sigmoid,
  softmax, and linear.
* `elml_resolve_layers(model)`: for each descriptor, looks up the vtable, checks
  the layer version, binds `params` to `PARM + param_off` (and validates 64-byte
  alignment), runs `prepare`, then fuses an activation into a preceding dense
  layer when shapes match and marks the activation `SKIPPED`.
* `elml_compute_scratch(model)`: finds the largest activation (or input),
  rounds it to `ELML_TENSOR_ALIGN`, and records `scratch_bytes` as two
  ping-pong buffers.
* `elml_model_init(model, blob, len, expected_dtype)`: validates arguments,
  rejects unsupported expected dtypes, requires 64-byte blob alignment, zeroes
  the model, parses the format, enforces dtype equality, copies descriptors and
  normalization views, resolves layers, and computes scratch. Rejects int8 with
  `ELML_ERR_UNSUPPORTED_DTYPE`.
* `elml_model_dtype(blob, len, dtype)`: thin wrapper over `elml_format_probe`.
* `elml_scratch_bytes(model, batch)`: returns the computed scratch size. The
  batch argument is accepted for API stability; batch inference reuses one
  sample-sized workspace.
* `elml_normalize_value(value, mode, params)`: applies standard or min-max input
  normalization; a zero stddev leaves the value unchanged, and a zero min-max
  range yields 0.
* `elml_denormalize_value(value, mode, params)`: inverse of the above.
* `elml_forward(model, scratch, len, input, output)`: validates dtype,
  scratch alignment and size, normalizes the input into buffer A, walks the
  layer chain with two aligned ping-pong buffers, skips fused activations,
  denormalizes the result into `output`. Returns `ELML_ERR_INTERNAL` if a bound
  layer has no forward function.
* `elml_forward_batch(model, scratch, len, input, output, batch)`: loops over
  samples calling `elml_forward`. Does not enlarge scratch.
* `elml_forward_q(...)`: returns `ELML_ERR_INVALID_ARG` for a null model,
  `ELML_ERR_DTYPE_MISMATCH` for a non-int8 model, and
  `ELML_ERR_UNSUPPORTED_DTYPE` otherwise (int8 runtime not implemented).
* `elml_meta_find(meta, key, value, value_len)`: walks TLV records and returns a
  view of the matching value or `ELML_ERR_NOT_FOUND`.
* `elml_meta_get(model, key, buf, cap, out_len)`: copies a metadata value as a
  NUL-terminated string; returns `ELML_ERR_BUFFER_TOO_SMALL` when needed.
* `elml_label_count(model)`: counts comma-separated entries in the `labels`
  metadata value; returns 0 if absent.
* `elml_label_get(model, index, buf, cap, out_len)`: extracts one label by
  index, copying it as a NUL-terminated string.

### `layers/dense.c`

Dense layer implementation.

* `elml_dense_prepare(layer, desc, dtype)`: requires f32, rank-1 in/out, non-zero
  dimensions, a valid parameter offset, and enough `param_bytes` for
  `align_up(in*out*4, 64) + align_up(out*4, 64)`. Binds `weights` at the block
  start and `biases` after the padded weight block; clears the fused activation.
* `elml_dense_forward(layer, in, out)`: validates f32 tensors, non-null data,
  parameter pointers, rank-1 shapes, and shape agreement, calls
  `elml_kernel_dense_f32`, then applies the fused activation (when present) with
  `elml_kernel_activation_f32`.
* `elml_dense_layer_vt`: vtable (`id = DENSE`, version 1, name `"dense"`).

### `layers/activation.c`

Activation layer implementation.

* `g_act_kinds`: table mapping layer id to `elml_activation` and a display name.
* `elml_act_kind_find(id)`: looks up an activation kind.
* `elml_activation_prepare(layer, desc, dtype)`: requires f32, rank-1 shapes,
  equal non-zero in/out dimensions, and a known activation id. Sets
  `fused_act`, defaults leaky alpha to 0.01, and reads an optional f32 alpha
  from `desc->extra`.
* `elml_activation_forward(layer, in, out)`: validates f32, non-null data, and
  matching shape, then calls `elml_kernel_activation_f32`.
* `ELML_DEFINE_ACT_VT` and the six `elml_act_*_layer_vt` objects: one vtable per
  activation id.

---

## 4. Format (`src/format/`)

### `crc32.h` / `crc32.c`

* `elml_crc32(data, len, seed)`: reflected IEEE CRC32 with polynomial
  `0xEDB88320`, no final XOR. Used for chunk payloads and the chunk region.

### `endian.h`

Inline little-endian helpers. No struct is ever mapped onto the byte stream.

* `elml_le16`, `elml_le32`, `elml_le64`: load little-endian integers.
* `elml_le_f32`: load a little-endian f32 (bit copy through `memcpy`).
* `elml_store_le16`, `elml_store_le32`, `elml_store_le64`, `elml_store_le_f32`:
  store values.

### `format_internal.h`

Non-public parser views shared by the model binder and the inspect tool.

* `elml_chunk_view`: type, version, flags, CRC, payload pointer and length.
* `elml_format`: header fields, input/output shapes, descriptors, raw views of
  each known chunk, parsed normalization fields, chunk count, and file length.
* `elml_format_parse`: full strict parse.
* `elml_format_probe`: header-only parse.
* `elml_chunk_next`: iterate chunks.

### `read.c`

Strict, allocation-free parser.

* `elml_region_ok(base, need, limit)`: overflow-safe bounds check.
* `elml_parse_header(blob, len, out)`: validates length, magic, major version,
  header size, dtype range, and `total_bytes == len`; fills header fields.
* `elml_format_probe(blob, len, dtype)`: header-only validation for tools that
  only need the dtype.
* `elml_chunk_next(blob, len, offset, out)`: reads one chunk header, validates
  the payload region, computes the next 8-byte-aligned offset, and fills the
  chunk view. Returns `ELML_ERR_NOT_FOUND` at end of file.
* `elml_shape_equal(a, b)`: local shape comparison.
* `elml_parse_conf(format, out)`: parses `CONF`, validates layer count, ranks,
  dimensions, parameter ranges, sequential shape chaining, and model boundary
  shapes. Fills descriptors and extra-byte views.
* `elml_parse_norm(format, out)`: parses the 8-byte `NORM` header, validates
  modes and counts against model dimensions, and binds parameter views.
* `elml_format_parse(blob, len, out)`: zeroes the view, parses the header,
  verifies the whole-region CRC, walks chunks (rejecting duplicates and unknown
  required chunks), enforces required `CONF`/`PARM` and int8 `QNT`, then parses
  `CONF` and `NORM`.

### `write.h` / `write.c`

Host-only writer. May allocate.

* `elml_buf`: growable byte buffer (`data`, `len`, `cap`).
* `elml_write_layer`: per-layer export input (dimensions, activation, leaky
  alpha, weights, biases).
* `elml_write_norm`: normalization modes and interleaved parameter arrays.
* `elml_write_meta`: key/value pair.
* `elml_write_spec`: full model description.
* `elml_buf_reserve(buf, extra)`: grows the buffer geometrically with checked
  arithmetic.
* `elml_buf_append(buf, data, len)`: appends bytes.
* `elml_buf_zero(buf, len)`: appends zero bytes.
* `elml_buf_pad(buf, alignment)`: pads to an absolute alignment.
* `elml_buf_free(buf)`: releases the buffer and zeroes it.
* `elml_layer_id_for_activation(act)`: maps an activation to its permanent layer
  id (defaults to linear).
* `elml_validate_spec(spec)`: validates dimensions, non-null layers/weights/
  biases, strict sequential chaining, output dimension, layer-count limit, and
  normalization modes.
* `elml_buf_pad_to(buf, base, alignment)`: pads relative to a base offset (used
  for per-layer 64-byte strides).
* `elml_write_parm(spec, parm, offsets, sizes)`: writes 48 bytes of leading
  padding (so the first block is at absolute 128), then each layer's weights
  padded to 64 bytes and biases padded to 64 bytes; records relative offsets and
  sizes.
* `elml_write_conf(spec, offsets, sizes, conf)`: writes the chain header, model
  input/output dimensions, and two descriptors per network layer (dense then
  activation). Activation layers are shape-preserving and carry a leaky-alpha
  `extra` block when needed.
* `elml_buf_zero_floats(buf, count)`: appends `2 * count` zero f32 values for
  absent normalization parameters.
* `elml_write_norm_chunk(spec, norm)`: writes the `NORM` payload; omitted when
  both modes are none. Always writes parameter slots, zero-filled for a mode
  that is none.
* `elml_write_meta_entry(meta, key, value)`: writes one TLV record.
* `elml_write_meta_chunk(spec, meta)`: always writes the `producer` key, then
  every caller-provided key.
* `elml_append_chunk(file, type, flags, payload)`: pads to 8 bytes, writes a
  chunk header with version 1 and payload CRC, appends the payload, pads again.
* `elml_model_write(spec, out)`: validates the spec, builds `PARM`, `CONF`,
  `NORM`, and `META`, writes the 64-byte header and chunks in the order
  `PARM`, `CONF`, `NORM?`, `META?`, fills the header fields and CRCs, and
  returns the completed buffer. On failure the output buffer is freed.

---

## 5. Kernels (`src/kernels/`)

Kernels are split into a portable common layer and per-architecture folders.
Only `common/` is compiled on every target; the architecture folders are added
by CMake when `ELML_SIMD=ON` (the default) and the target matches.

```
kernels/
  kernels.h              common interface + scalar reference declarations
  common/dispatch.c      runtime/compile-time dispatch to the best kernel
  common/scalar_f32.c    portable reference implementation
  x86_64/x86_64.h        x86_64 kernel declarations and feature probe
  x86_64/simd_f32.c      AVX2 + FMA kernels
  arm64/arm64.h          arm64 kernel declarations
  arm64/neon_f32.c       NEON kernels
  riscv/riscv.h          RISC-V kernel declarations
  riscv/vector_f32.c     RVV 1.0 kernels
```

### `kernels.h`

Declares the only entry points the rest of the library may call:

* `elml_kernel_dense_f32(weights, biases, input, output, in_dim, out_dim)`:
  `output[o] = biases[o] + dot(weights[o], input)`, no activation.
* `elml_kernel_activation_f32(input, output, count, act, alpha)`: elementwise
  activation, including softmax.
* `elml_kernel_dense_backward_f32(delta, input, grad_w, grad_b, in_dim,
  out_dim)`: accumulates `grad_w[o][i] += delta[o] * input[i]` and
  `grad_b[o] += delta[o]`.
* `elml_kernel_dense_input_grad_f32(weights, delta, grad_input, in_dim,
  out_dim)`: computes `grad_input[i] = sum_o weights[o][i] * delta[o]`.
* `elml_scalar_*`: the same three dense operations as portable references, used
  by the dispatcher fallback and by the kernel equivalence test.

The inference path and the trainer both call these functions, so one SIMD
implementation accelerates training and inference at the same time.

### `common/scalar_f32.c`

Portable C used by every target and by MCUs that select the generic path:

* `elml_scalar_activate(x, act, alpha)`: scalar activation. Softmax is a no-op
  here because it needs the whole vector.
* `elml_scalar_softmax(values, count)`: max-subtracted, numerically stable
  softmax in place.
* `elml_kernel_activation_f32(...)`: copies then applies softmax, or applies a
  scalar activation elementwise.
* `elml_scalar_dense_f32(...)`, `elml_scalar_dense_backward_f32(...)`,
  `elml_scalar_dense_input_grad_f32(...)`: reference implementations.

The scalar loops are ordinary C so a vendor MCU compiler can apply its own
unrolling, auto-vectorization, or DSP extensions. Configure with
`-DELML_SIMD=OFF` to guarantee this path is the only one compiled.

### `common/dispatch.c`

Selects an implementation for each call:

* x86_64: AVX2 + FMA when the CPU reports both features at run time, otherwise
  scalar. The probe is cached in a relaxed atomic.
* arm64: NEON (baseline on AArch64).
* RISC-V: RVV when the translation unit is built with vector support
  (`__riscv_vector`), otherwise scalar.
* `ELML_STRICT_FP` or `ELML_SIMD=OFF`: always scalar.

### `x86_64/simd_f32.c`

AVX2/FMA kernels built with a function-level target attribute, so the rest of
the library (and the baseline binary) does not require AVX2:

* dense forward processes four output rows per input load and uses FMA
  accumulation over 8-wide vectors.
* dense backward updates four weight rows per input load.
* input-gradient accumulation handles four output rows per pass over the input
  gradient vector.
* scalar tails handle dimensions that are not multiples of 8.

### `arm64/neon_f32.c`

NEON equivalents of the three dense kernels, using 4-wide `vfmaq_f32`
accumulation and scalar tails.

### `riscv/vector_f32.c`

RVV 1.0 equivalents using `vsetvl`, `vle32`/`vse32`, `vfmacc`, and a
`vfredusum` reduction. The file is only built when vector support is enabled in
the toolchain.


---

## 6. Trainer (`src/train/`)

The trainer is host-only and may allocate.

### `io.h` / `io.c`

* `elml_read_file(path, data, len, alignment)`: reads an entire file. A
  non-zero alignment must be a power of two; the returned buffer is aligned and
  sized to a multiple of the alignment when needed. Returns `ELML_ERR_TRUNCATED`
  for an empty file.
* `elml_write_file(path, data, len)`: writes the entire buffer, creating or
  truncating the file; reports `ELML_ERR_NOT_FOUND` if the file cannot be
  opened.

### `csv.h` / `csv.c`

* `elml_csv`: parsed table: owned storage, cell pointer array, row/column counts,
  and `data_start` (0 without a header, 1 with one).
* `elml_is_space(c)`: ASCII whitespace test.
* `elml_csv_builder` (private): grows the storage and cell arrays.
* `elml_builder_add_offset`, `elml_builder_add_row`, `elml_builder_put`: manage
  storage, row boundaries, and byte appends.
* `elml_parse_quoted(text, len, pos, delimiter, out_cell, builder)`: parses an
  RFC-4180-style quoted field, including doubled quotes, embedded delimiters,
  and embedded newlines.
* `elml_parse_unquoted(...)`: parses a plain field up to a delimiter, newline,
  or end of input.
* `elml_line_is_comment(text, len, pos)`: true when a line begins (after
  whitespace) with `#` or `;`.
* `elml_skip_blank_or_comment(...)`: advances past blank lines and comment
  lines.
* `elml_csv_parse(text, len, delimiter, has_header, out)`: parses a buffer into
  rows and columns, strips a UTF-8 BOM, handles `\n` and `\r\n`, and rejects
  inconsistent row widths.
* `elml_csv_load(path, delimiter, has_header, out)`: reads a file and parses it.
* `elml_csv_read_header(path, delimiter, out)`: reads only the first logical
  record, skipping BOMs, blank lines, and comments. The result has `rows == 1`
  and `data_start == 0`, so column `i` is available as
  `elml_csv_cell(&header, 0u, i)`. Quoted fields may span physical lines. The
  studio uses this to populate its column pickers without reading the full CSV.
* `elml_csv_free(csv)`: releases storage; safe on null or repeated calls.
* `elml_csv_cell(csv, row, col)`: returns a cell or `NULL` when out of range.

### `ini.h` / `ini.c`

* `elml_ini_entry`: section, key, and value strings.
* `elml_ini`: growable array of entries.
* `elml_ini_space(c)`: whitespace test.
* `elml_dup_range(start, end)`: duplicates a substring.
* `elml_trim(start, end)`: trims leading and trailing whitespace in place.
* `elml_ini_add(...)`: appends an entry, replacing a previous value with the
  same section and key (last value wins).
* `elml_ini_skip_line(...)`: skips blank lines and `;`/`#` comments.
* `elml_ini_parse(text, len, out)`: parses `[section]` headers and `key = value`
  lines, supports quoted values, and rejects malformed syntax.
* `elml_ini_load(path, out)`: reads and parses a file.
* `elml_ini_free(ini)`: releases all strings.
* `elml_ini_get(ini, section, key)`: returns the last matching value or `NULL`;
  matching is case-sensitive.

### `config.h` / `config.c`

Configuration model and parser.

* `ELML_MAX_HIDDEN_LAYERS` = 31 and `ELML_MAX_MODEL_LAYERS` = 32, chosen so the
  writer's two-descriptors-per-layer chain fits `ELML_MAX_LAYERS` (64).
* `elml_loss`: `MSE`, `MAE`, `CCE`.
* `elml_optimizer`: `ADAM`, `SGD`.
* `elml_missing`: `ERROR`, `ZERO`, `MEAN`.
* `elml_config`: all data/model/train/export settings, fixed-size string
  buffers, hidden-layer array, activation array, and classification flag.
* `elml_copy_string(dst, cap, src)`: bounded string copy.
* `elml_parse_u32`, `elml_parse_u64`, `elml_parse_double`, `elml_parse_bool`,
  `elml_parse_delimiter`, `elml_parse_loss`, `elml_parse_optimizer`,
  `elml_parse_missing`, `elml_parse_activation`: strict scalar parsers.
* `elml_parse_u32_list(text, values, max, count)`: parses a comma-separated
  integer list.
* `elml_parse_activation_list(text, values, max, count)`: parses activation
  names.
* `elml_config_init(config)`: sets defaults (header on, `,`, missing=error,
  split 0.8, seed 42, MSE, Adam, 100 epochs, batch 16, lr 0.001, early stop 20,
  threads 1).
* `elml_key_known(section, key)`: rejects unknown INI sections and keys; `l2`,
  `early_stop`, and `threads` are also accepted under `[train]`.
* `elml_apply_ini(config, ini)`: applies all recognized keys.
* `elml_require_value(argc, argv, index, value)`: fetches the next CLI argument.
* `elml_apply_cli(config, argc, argv)`: parses CLI flags and overrides INI
  values. Returns `ELML_ERR_NOT_FOUND` for `--help`.
* `elml_finalize_config(config)`: validates required fields, derives the
  classification flag from the loss, enforces the activation count
  (`hidden + 1`), requires softmax output for CCE and forbids softmax for
  regression, validates ranges, and clamps `threads` to 1 with a warning.
* `elml_config_load(config, argc, argv)`: initializes defaults, loads the INI
  file if `--config` is present, applies CLI overrides, and finalizes.
* `elml_loss_name(loss)`, `elml_activation_name(activation)`: stable strings.
* `elml_config_print_help()`: prints the CLI reference.

### `dataset.h` / `dataset.c`

CSV-to-tensor conversion, splitting, and normalization.

* `elml_dataset`: rows, dimensions, flat `x`/`y` arrays, per-row class index
  for classification, label strings, and label count.
* `elml_norm_stats`: per-feature mean/std for inputs and outputs, with
  `has_input`/`has_output` flags.
* `elml_parse_float(text, out)`: strict finite f32 parse.
* `elml_strdup_local(text)`: local string duplication.
* `elml_find_column(csv, name)`: case-insensitive header lookup.
* `elml_columns_resolve(list, csv, indices, count)`: resolves a comma-separated
  list of header names or 1-based indices. The empty list is an error, names
  must exist, indices are bounds-checked, and duplicates are rejected. Header
  names are tried first, so a headerless file must be opened with
  `has_header = false` / `--no-header`; otherwise a numeric entry such as `1`
  can accidentally match a value in the first data row.
* `elml_indices_contain(indices, count, value)`: duplicate check.
* `elml_cmp_strings(a, b)`: `qsort` comparator for labels.
* `elml_labels_add(labels, count, capacity, value)`: adds a unique label,
  rejecting commas and more than 255 classes.
* `elml_copy_labels(source, destination)`: deep-copies labels.
* `elml_alloc_dataset(dataset, rows, input_dim, output_dim)`: allocates `x` and
  `y` with checked sizes.
* `elml_compute_input_means(...)`: computes per-column means over non-missing
  cells for the `missing = mean` policy.
* `elml_dataset_build(csv, config, out)`: resolves columns, reads rows,
  handles missing values, builds one-hot targets and class indices for
  classification, sorts labels, and rejects single-class datasets and
  input/output column overlap.
* `elml_rng_next(state)`: SplitMix64 step; deterministic across platforms.
* `elml_shuffle_indices(indices, count, state)`: Fisher-Yates shuffle.
* `elml_copy_rows(source, destination, rows, source_rows)`: copies selected
  rows into a new dataset, including labels and class indices.
* `elml_dataset_split(full, train_fraction, seed, train, val)`: shuffled split
  with deterministic ordering; requires a fraction in `(0, 1]`.
* `elml_dataset_compute_norm(train, normalize_output, out)`: computes input
  mean/std and optionally output mean/std, replacing zero stddev with 1.
* `elml_dataset_apply_norm(dataset, stats)`: standardizes inputs and, when
  present, outputs in place.
* `elml_norm_stats_free(stats)`, `elml_dataset_free(dataset)`: release all
  owned buffers.

### `net.h` / `net.c`

Host training network with forward/backward passes and optimizer state.

* `elml_net_layer`: dimensions, activation, weights, biases, gradients, Adam
  moments, pre-activation/activation/delta buffers, and best-weight snapshots.
* `elml_net`: layer array, loss, optimizer hyperparameters, Adam step counters,
  best-state flag, and input copy.
* `elml_activate_value(act, value)`: scalar forward activation.
* `elml_apply_softmax(z, out, count)`: stable softmax.
* `elml_activation_derivative(layer, index)`: derivative from stored `z`/`a`;
  linear/softmax return 1 because the softmax gradient is handled by CCE.
* `elml_alloc_floats(count)`: zeroed `calloc` wrapper.
* `elml_layer_alloc(layer)`: allocates weights, biases, gradients, activations,
  deltas, best copies, and Adam state; all optimizer state is always allocated.
* `elml_layer_free(layer)`: frees all buffers and zeroes the struct.
* `elml_uniform(state)`: uniform `[0, 1)` from the RNG.
* `elml_net_init_weights(net, seed)`: He-style initialization for ReLU and
  Xavier-style otherwise, with zero biases.
* `elml_net_create(...)`: validates arguments and activation/loss pairing,
  allocates every layer, and initializes weights.
* `elml_net_free(net)`: frees all layers and the input copy.
* `elml_net_forward(net, input, output)`: caches the input and all layer
  pre-activations/activations; optional output copy.
* `elml_net_argmax(net)`: index of the largest output activation (argmax over
  the last layer).
* `elml_net_zero_grad(net)`: clears accumulated gradients.
* `elml_net_backward(net, target, class_index)`: computes output deltas for CCE
  or MSE/MAE, accumulates weight/bias gradients, and propagates deltas
  backwards through each layer.
* `elml_net_accumulate(net, input, target, class_index)`: forward plus backward
  for one sample.
* `elml_net_update_parameter(...)`: applies either Adam or SGD to one parameter,
  with optional L2 decay.
* `elml_net_update(net, batch_size)`: scales gradients by batch size, advances
  Adam bias-correction powers, and updates every weight and bias.
* `elml_net_evaluate(net, dataset, correct)`: mean loss over a dataset (CCE uses
  mean negative log-likelihood; regression uses MSE or MAE), and fills the
  correct-prediction count for classification.
* `elml_net_save_best(net)`, `elml_net_restore_best(net)`: snapshot and restore
  the weights/biases with the best monitored loss.

### `export.h` / `export.c`

Converts training state into a `elml_write_spec` and writes the file.

* `elml_interleave_norm(out, mean, stddev, count)`: interleaves `(mean, std)`
  pairs matching the `NORM` payload.
* `elml_join_labels(dataset)`: joins sorted labels with commas into one
  metadata string.
* `elml_build_norm(net, norm, out, input_params, output_params)`: allocates and
  fills normalization parameters and modes.
* `elml_export_model(net, norm, dataset, config, path)`: builds layer
  descriptors, normalization, and metadata (`labels`, `columns.inputs`,
  `columns.outputs`, `task`, `loss`), calls `elml_model_write`, writes the
  buffer to `path`, and frees all temporary state.

### `session.h` / `session.c`

Reusable host-only training session shared by the CLI and the studio.

* `elml_train_metrics`: per-epoch progress record (epoch, total, best epoch,
  rows, monitored loss/accuracy, early-stop and stop flags).
* `elml_train_progress_fn`: callback invoked once per completed epoch.
* `elml_training`: owns the config, datasets, normalization, network, shuffle
  order, best-weight bookkeeping, final metrics, error text, and an atomic stop
  flag.
* `elml_training_init(training, config)`: loads the CSV(s), builds and splits
  the datasets, computes normalization, creates the network, and allocates the
  shuffle order. On failure it releases partial state and zeroes the struct.
* `elml_training_run(training, progress, user)`: trains epoch by epoch, checks
  the atomic stop flag at batch boundaries, monitors validation loss, snapshots
  and restores the best weights, applies early stopping, and fills the final
  metrics.
* `elml_training_request_stop(training)`: safely requests a stop from another
  thread; the loop exits at the next batch boundary.
* `elml_training_export(training)`: exports the restored best weights.
* `elml_training_destroy(training)`: releases resources in reverse allocation
  order: training order, network, normalization, validation set, then training
  set. Safe on partially initialized sessions.

### `train_main.c`

Command-line training executable.

* `elml_wants_help(argc, argv)`: detects `--help`/`-h`.
* `elml_shuffle_order(order, count, state)`: Fisher-Yates shuffle over row
  indices.
* `elml_same_labels(a, b)`: compares train/validation label sets.
* `elml_dataset_mae(net, dataset, norm)`: computes mean absolute error in
  original output units by undoing output normalization per feature.
* `elml_print_metrics(name, net, dataset, norm)`: runs evaluation and prints
  accuracy (classification) or loss/MAE (regression).
* `elml_report_progress(metrics, user)`: prints periodic epoch progress.
* `main`: loads the configuration, initializes an `elml_training` session,
  runs it with the progress callback, prints final metrics, exports the model,
  and destroys the session. Every failure path reports a readable error.

---

## 7. Studio (`src/studio/`)

The studio is optional, host-only, and built only when
`ELML_BUILD_STUDIO=ON`. It links the same `elml_train_lib` as the CLI.

### `studio_main.c`

CUI desktop frontend with a parameter sidebar and a live results panel.

* `studio_snapshot` / `studio_view`: shared run state copied from the worker
  under a mutex once per frame; the view owns the chart arrays handed to CUI.
* `studio_state`: UI parameters, training session pointer, worker thread,
  mutex, chart history, column-picker state, and theme.
* `studio_lock` / `studio_unlock` / `studio_is_running`: protect the snapshot
  and stop state shared with the training thread.
* `studio_defaults`: sets initial parameter values and loads
  `examples/iris/iris.ini` when available.
* `studio_apply_config` / `studio_load_config`: reuse `elml_config_load` and
  copy the parsed configuration into the editable fields. Relative paths are
  resolved next to the `.ini` file.
* `studio_columns_*`: read the CSV header with `elml_csv_read_header`, build
  display labels and 1-based index values, apply the configured inputs/outputs,
  and keep the two multi-selects exclusive. Selecting a column on one side
  clears it on the other. `studio_columns_validate` ensures the header is
  loaded and rejects empty, overlapping, or numerically ambiguous selections
  before training starts.
* `studio_column_picker`: renders two searchable `cui_multi_select` controls
  when columns are available, or fallback text fields while the CSV cannot be
  read.
* `studio_build_config`: synthesizes the equivalent CLI argv and calls
  `elml_config_load`, so validation and finalization are identical to the CLI.
* `studio_progress`: mutex-protected epoch callback; appends loss/accuracy
  history for the charts.
* `studio_worker`: owns the `elml_training` session, runs the loop, exports the
  best weights, and publishes final status/error text.
* `studio_start` / `studio_stop` / `studio_shutdown`: create/join the worker
  thread and request a stop through `elml_training_request_stop`.
* `studio_sidebar` / `studio_main` / `studio_frame`: draw the parameter panel,
  KPI cards, progress bar, charts, run details, export status, and error alerts.
* `main`: parses `--config FILE`, `--autostart`, and `--help`, creates the CUI
  app, runs frames, and releases the column state and worker on exit.

The worker never calls CUI. Only the UI thread touches widgets and column
state; the training thread only updates the mutex-protected snapshot and its
atomic stop flag.

---

## 8. Runner (`src/run/`)

### `run_main.c`

Standalone inference/evaluation tool. It links both the core and the host
libraries (for CSV and column resolution).

* `elml_run_options`: model/CSV/output paths, prediction text, column lists,
  header/delimiter flags, evaluate/quiet/json/stdin/repl/prompt flags, row
  limit, and top-K.
* `elml_run_usage()`: prints the CLI reference.
* `elml_parse_delimiter(text, out)`: accepts a single character or `\t`.
* `elml_parse_limit(text, out)`: parses a `uint32_t` row limit.
* `elml_parse_top(text, out)`: parses a positive `--top` class count.
* `elml_parse_cell(text, out)`: strict finite f32 parse.
* `elml_run_parse(options, argc, argv)`: parses arguments, resolves
  `--predict -` to stdin, and rejects conflicting modes. Returns
  `ELML_ERR_NOT_FOUND` for `--help`.
* `elml_meta_string(model, key, buffer, capacity, found)`: reads a metadata
  string, treating an absent key as "not found" rather than an error.
* `elml_write_field(file, text, delimiter)`: writes a CSV field, quoting when
  it contains the delimiter, quotes, CR, or LF.
* `elml_target_label(model, index, buffer, capacity)`: returns a model label or
  a generated `class_N` fallback.
* `elml_model_is_classification(model)`: reads the `task` metadata, falling
  back to "labels exist ⇒ classification" for older models.
* `elml_meta_token(list, index, out, capacity)`: copies the index-th
  comma-separated token of a metadata list, trimming spaces.
* `elml_output_name(model, index, buffer, capacity)`: returns the index-th
  `columns.outputs` name or a generated `output_N` fallback.
* `elml_json_write_string(file, text)`: writes a JSON-escaped string.
* `elml_parse_feature_row(text, values, dim)`: parses exactly `dim`
  comma-separated finite floats, rejecting missing, extra, or malformed values.
* `elml_line_reader` / `elml_line_read` / `elml_line_reader_free`: reusable
  dynamically grown stdin line reader (no fixed length limit).
* `elml_predict_buffers` / `elml_predict_buffers_alloc` /
  `elml_predict_buffers_free`: allocate scratch, input, prediction, and top-K
  order once for reuse across many questions.
* `elml_print_prediction(model, options, prediction, order)`: prints the
  classification label/probability or top-K list, or regression
  `name = value` lines, in human or JSON form.
* `elml_run_predict(model, options)`: one-shot inference path. Validates the
  input count, allocates scratch and output buffers, runs `elml_forward`, then
  prints the result. `--json` emits a single JSON line.
* `elml_run_repl(model, options)`: loads nothing after startup; loops over
  stdin lines with the same buffers, ignores blank lines, continues after a
  malformed row, accepts `quit`/`exit`, and flushes stdout after every answer.
* `main`:
  1. parses options and loads the model with 64-byte alignment;
  2. rejects unsupported dtypes and a metadata label count that does not match
     the output dimension;
  3. takes the one-shot `--predict`/`--stdin` or looping `--repl` path when
     requested;
  4. loads the CSV and resolves input columns, preferring `--inputs`, then
     `columns.inputs`, then the first N columns;
  5. requires output columns for `--evaluate`, preferring `--outputs`, then
     `columns.outputs`, and validates the regression output count;
  6. allocates aligned scratch and per-sample buffers;
  7. optionally writes an augmented prediction CSV (original columns plus
     `prediction` and per-class probabilities, or `pred_N`);
  8. runs `elml_forward` per row, optionally computes accuracy or
     MSE/RMSE/MAE, and prints a summary unless `--quiet`.

---

## 9. Tools (`tools/`)

### `inspect_main.c`

* `usage(out)`: prints usage.
* `main`: parses `--model`, `--json`, and `--help`, reads the file, calls
  `elml_format_parse`, then `elml_inspect_print`. Unknown layer types are shown
  rather than causing an early failure because inspection uses the parser views
  directly.

### `src/format/inspect.c` / `inspect.h`

* `elml_dtype_name(dtype)`, `elml_layer_type_name(id)`: display names.
* `elml_print_shape(out, shape)`: prints `[d0,d1,...]`.
* `elml_json_string(out, data, len)`: prints a JSON-escaped string.
* `elml_print_header_text`, `elml_print_layers_text`, `elml_print_norm_text`,
  `elml_print_meta_text`, `elml_print_scratch_text`: human-readable sections.
* `elml_inspect_json(...)`: machine-readable output.
* `elml_inspect_print(format, blob, out, json)`: dispatches to text or JSON.
  Text output includes header fields, layers, normalization, metadata, and
  computed scratch/runnability.

---

## 10. Tests (`tests/`)

### `test.h`

Minimal assertion macros used by all C tests: `CHECK`, `CHECK_EQ_INT`,
`CHECK_CLOSE`, `CHECK_STR`, and `TEST_END`. `TEST_END` prints a summary and
returns 1 on any failure. No external test framework is required.

### `test_arena.c`

Exercises `elml_align_up` (including non-power-of-two and overflow cases),
null arenas, alignment guarantees, mark/reset behavior, out-of-range resets,
zero-byte allocations, and allocation failure without consuming space.

### `test_registry.c`

Defines a custom vtable, verifies every built-in layer is present, verifies
reserved/unknown ids are absent, tests registration validation (null, duplicate,
missing callbacks, reserved id), and checks names and versions.

### `test_format.c`

Builds a model with the writer, parses it, verifies header/chunk fields,
parameter alignment, normalization views, labels/metadata access, dtype probing,
misalignment rejection, scratch sizing, forward output, deliberate CRC
corruption, bad magic, unsupported versions, truncation, and writer validation
errors. It also writes a 30 -> 24 -> 2 model to cover non-power-of-two parameter
strides, the case that previously produced a false out-of-memory error.

### `test_forward.c`

Writes a single softmax layer, verifies scratch sizing, exact softmax output,
batch inference, scratch-too-small and misalignment errors, null arguments, and
the int8 entry-point status codes.

### `test_csv.c`

Tests quoted fields, escaped quotes, embedded delimiters/newlines, comments, a
UTF-8 BOM, semicolon delimiter, inconsistent rows, unterminated quotes, invalid
delimiters, headerless files, file loading, and the first-record header reader
(including BOM/comment skipping and a missing file).

### `test_ini.c`

Tests sections, comments, quoted and single-quoted values, duplicate keys,
trimming, parse errors, and file loading.

### `test_config.c`

Exercises INI defaults, CLI overrides, lists, activation defaults, classification
derivation, unknown sections, invalid loss/activation pairings, ranges, missing
required keys, help/bogus/missing-file handling, and a full CLI-only
configuration.

### `test_dataset.c`

Covers column resolution by name and 1-based index, duplicate/missing columns,
input/output overlap, regression tensor layout, deterministic splitting, 100%
and invalid splits, normalization math, classification labels and one-hot
targets, comma labels, single-class rejection, missing-value policies, and
free-null safety.

### `test_net.c`

Trains XOR to convergence, verifies deterministic training with the same seed,
tests best-weight save/restore, trains a small softmax classifier, checks
evaluation loss/accuracy and empty datasets, and exercises every invalid
`elml_net_create` argument and activation/loss mismatch.

### `test_io.c`

Tests read/write round trips, alignment guarantees, empty files, missing files,
null arguments, unwritable paths, and non-power-of-two alignment rejection.

### `test_misc.c`

Tests every error string, the version string, chunk iteration and end-of-file
behavior, metadata and label lookups on a model without metadata, batch argument
validation, dtype probing (including reserved f16 and representable int8), and
header truncation/bad magic.

### `test_cli.sh`

End-to-end CLI regression test. Generates a synthetic linear dataset, trains via
INI plus a CLI override, evaluates and writes predictions, checks MSE against a
gate, checks prediction CSV shape and `--limit`, checks `--help` and missing
inputs, and verifies that an unknown output column is rejected.

### `examples/iris` and `examples/wdbc` (registered in CTest)

Run each example's `run.sh` with the build directory and require its own
accuracy/host-check gates to pass.

---

## 11. Examples (`examples/`)

### Iris (`examples/iris/`)

### `iris.csv`

The classic 150-row iris dataset with named columns
`sepal_length,sepal_width,petal_length,petal_width,species`.

### `iris.ini`

Training configuration: 4 inputs, species output, 80/20 split, one hidden layer
of 8 ReLU units, softmax output, CCE loss, Adam, 300 epochs, batch 8, learning
rate 0.01, L2 0.0001, early stop 50, export `iris.elml`.

### `run.sh`

Complete pipeline: check the build, train, evaluate on the full CSV, inspect the
model, parse the accuracy and enforce `>= 0.90`, embed the model into a C header,
compile the MCU-style demo plus host checker, and run it. Accepts an optional
build directory as the first argument.

### `embed_model.py`

Converts a `.elml` file into `mcu/model_blob.h`:

* emits a 64-byte-aligned `static const uint8_t` array;
* emits `elml_example_model_len`;
* uses compiler-specific alignment attributes for GCC/Clang and MSVC.

### `mcu/main.c`

Bare-metal style inference demo. Uses a static 16-byte-aligned scratch buffer,
initializes the model from the generated blob, checks scratch size, runs
`elml_forward`, and computes the argmax. No `malloc`, `stdio`, or file I/O.
`elml_example_label_count()` exposes metadata to the host checker.

### `mcu/host_check.c`

Host-side harness for the demo. Runs three known iris samples through
`elml_example_predict`, prints probabilities, and fails if any label does not
match. This is how the example proves that the embedded blob and the inference
core agree.

### Generated files

`iris.elml`, `train_output.txt`, `run_output.txt`, `inspect_output.txt`,
`mcu/model_blob.h`, and `mcu/host_check` are produced by `run.sh` and ignored by
git.

### Repl (`examples/repl/`)

* `README.md`: explains the persistent model pattern for MCU firmware, the
  one-line `elml_run --repl` protocol, and the Python transport.
* `repl.py`: `Runner` spawns `elml_run --repl --json` once, `predict(row)`
  writes one feature line and parses one JSON line, `close()` sends `quit`, and
  `main` runs either the supplied rows or an interactive loop. The same
  request/response protocol can be carried over a UART/serial link.

### WDBC (`examples/wdbc/`)

A real 569-row, 30-feature binary classification dataset (Breast Cancer
Wisconsin Diagnostic, UCI). Unlike iris, its parameter byte sizes are not powers
of two, which is what exposed the writer padding bug fixed in `write.c`.

* `wdbc.csv`: header plus `id`, `diagnosis`, and 30 feature columns.
* `wdbc.ini`: 30 inputs, 24 ReLU hidden units, 2 softmax outputs, CCE, Adam,
  80/20 split, seed 42.
* `run.sh`: trains, evaluates the full CSV, inspects, gates full and validation
  accuracy at 0.95, predicts one benign and one malignant row, runs a two-row
  `--repl` session, and runs the Python client. Takes an optional build
  directory.
* Generated `wdbc.elml`, logs, and REPL transcripts are ignored by git.

The model is about 4.4 KB with 256 bytes of batch-1 scratch. Typical results:
validation accuracy 1.0000 and full-CSV accuracy 0.9947.

### Covertype (`examples/covtype/`)

Optional large-data validation using the UCI Forest Covertype dataset:
581,012 rows, 54 numeric features, 7 classes, no header row.

* `download.sh`: idempotently downloads and decompresses `covtype.data`
  (about 72 MB, ignored by git).
* `covtype.ini`: numeric columns `1..54`, output `55`, `has_header = false`,
  hidden layers `128,64`, CCE, Adam, 80/20 split.
* `run.sh`: trains, evaluates all rows with `--no-header`, gates validation and
  full accuracy at 0.88, inspects the shape, predicts one row, sends 1,000 rows
  through the persistent REPL, and runs the Python client. Takes an optional
  build directory.
* Not registered with CTest because it needs a network download and takes about
  1-2 minutes.

Typical results: 28 epochs (~100 s), validation accuracy 0.8979, full-CSV
accuracy 0.9015, 64,200-byte model, 1,024-byte scratch, ~9 ms for 1,000 REPL
predictions.

---

## 12. Control-flow walkthroughs

### Training

```
main (train_main.c)
  elml_config_load
    elml_config_init
    elml_ini_load / elml_ini_parse
    elml_apply_ini
    elml_apply_cli
    elml_finalize_config
  elml_training_init
    elml_csv_load -> elml_csv_parse
    elml_dataset_build -> elml_columns_resolve, elml_labels_add
    elml_dataset_split -> elml_rng_next, elml_shuffle_indices, elml_copy_rows
    elml_dataset_compute_norm / elml_dataset_apply_norm
    elml_net_create -> elml_layer_alloc, elml_net_init_weights
  elml_training_run
    per epoch: elml_shuffle_order, elml_net_accumulate, elml_net_update
    elml_net_evaluate, elml_net_save_best / elml_net_restore_best
  elml_training_export
    elml_export_model
      elml_build_norm, elml_join_labels
      elml_model_write -> elml_write_parm, elml_write_conf,
                          elml_write_norm_chunk, elml_write_meta_chunk,
                          elml_append_chunk
      elml_write_file
  elml_training_destroy (reverse allocation order)
```

The studio follows the same path. `studio_build_config` synthesizes CLI
arguments and calls `elml_config_load`; `studio_worker` owns the session behind
the shared `elml_train` library.

### Loading and inference

```
elml_model_init
  pointer and alignment checks
  elml_format_parse
    elml_parse_header
    whole-region CRC
    elml_chunk_next (per chunk) + per-chunk CRC
    elml_parse_conf
    elml_parse_norm
  elml_resolve_layers
    elml_layer_find, vt->prepare
    fuse activation into preceding dense layer
  elml_compute_scratch

elml_forward
  validate dtype, scratch alignment and size
  normalize input into buffer A
  for each bound layer:
    skip ELML_LAYER_FLAG_SKIPPED
    vt->forward -> scalar f32 kernel
    swap ping-pong buffers
  denormalize into output
```

### Inspection

```
elml_inspect (inspect_main.c)
  read file
  elml_format_parse
  elml_inspect_print
    text: header, layers, normalization, metadata, scratch
    json: machine-readable equivalent
```

---

## 13. Where to make common changes

| Change | Files |
| --- | --- |
| New activation | `registry.h` (permanent id), `kernels/common/scalar_f32.c`, `layers/activation.c`, `registry.c` table, writer mapping, config parser, tests |
| New layer type | `registry.h`, new `src/core/layers/*.c`, `registry.c`, `layer_vt.h`, writer/parser descriptors, format docs |
| New CSV option | `config.h/.c`, `session.c`, `train_main.c`, `dataset.c`, tests |
| New studio control | `src/studio/studio_main.c`, `studio_build_config`, `README.md` |
| Change column picker | `csv.c` (`elml_csv_read_header`), `studio_main.c` (`studio_columns_*`), `test_csv.c` |
| New metadata key | `export.c`, `model.c` accessors if needed, `run_main.c`, format docs |
| Format change | `format.h`, `read.c`, `write.c`, `docs/FORMAT.md`, `test_format.c` |
| SIMD kernel | `kernels/kernels.h`, arch file under `kernels/<arch>/`, `kernels/common/dispatch.c`, CMake, `test_kernels.c` |
