# Coding style

The codebase is C23, built with `-Werror` and a deliberately strict warning set.
Consistency matters more than personal preference; follow the existing code.

## Language and compatibility

* Implementation sources are compiled as C23. Public headers avoid C23-only
  syntax so C11 consumers can include them.
* `include/elml/compat.h` supplies `nullptr`, `static_assert`, `alignas`, and
  `alignof` for pre-C23 compilers. Do not add feature logic there; it is a
  spelling shim only.
* Use `nullptr`, `bool`/`true`/`false`, and `static_assert` in internal code.
* Do not use C++ features, VLAs, or compiler-specific extensions outside the
  small portability shims.

## Formatting

* 2 spaces per indent level. Never tabs.
* 80 columns where practical; long conditions may wrap with continuation lines
  indented by 4 spaces.
* Opening brace on its own line for functions; same line for control flow and
  initializers.
* Space after keywords (`if (`, `for (`, `while (`), no space between a function
  name and `(`.
* Pointer star binds to the name: `float *value`, `const char *name`.
* One declaration per statement where it aids readability; grouped declarations
  are acceptable for closely related locals.
* `case` labels align with the `switch` body; use braces for a case that
  declares variables.

```c
static elml_status elml_do_thing(elml_model *model, const float *input)
{
  if (model == nullptr || input == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  for (uint32_t i = 0u; i < model->layer_count; ++i) {
    ...
  }
  return ELML_OK;
}
```

## Naming

| Kind | Convention | Example |
| --- | --- | --- |
| Public function | `elml_` prefix, lower snake case | `elml_forward` |
| Internal function | `static`, `elml_` prefix | `elml_parse_conf` |
| Type | `elml_` prefix, lower snake case | `elml_layer_desc` |
| Enum constant | `ELML_` prefix, upper snake case | `ELML_ERR_BAD_SHAPE` |
| Macro / constant | `ELML_` prefix, upper snake case | `ELML_MAX_LAYERS` |
| File-scope variable | `g_` prefix, lower snake case | `g_act_kinds` |
| Local variable | lower snake case | `payload_len` |

Keep names descriptive. Prefer `weight_bytes` to `n`. Avoid abbreviations that
are not already established in the format or API.

Every non-`static` function declared in a header has a matching declaration; the
build uses `-Wmissing-prototypes` to enforce it. `static` helpers are preferred
over exposing new symbols.

## Types and integers

* Include `<stdint.h>` and use fixed-width types on the wire and in the parser:
  `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `int32_t`.
* Use `size_t` for object sizes and counts, and `uint32_t` where the format
  stores a 32-bit field.
* Do not mix signed and unsigned carelessly. The build uses
  `-Wsign-conversion` and `-Wconversion`; cast explicitly after a bounds check.
* Use `u` suffixes for unsigned constants (`0u`, `1u`, `UINT32_MAX`) and `f` for
  float literals (`0.0f`, `1e-8f`).
* Use `size_t` arithmetic for allocation sizes and route multiplications
  through `ckd_mul` when an attacker-controlled file or CSV can influence the
  operands.

## Error handling

* Public functions return `elml_status`. `ELML_OK` is zero; errors are negative.
* Validate every pointer and every externally supplied length before use.
* Never leave an output partially initialized on failure. Either zero it first
  or document which fields are valid.
* Use `goto cleanup` in functions that allocate several resources; keep the
  cleanup labels in reverse acquisition order.
* Return the most specific status available. Do not collapse every failure to
  `ELML_ERR_INTERNAL`.
* Do not print from library code. Only command-line tools write to stdout or
  stderr.

## Memory

* `libelml_core` must never call `malloc`, `calloc`, `realloc`, or `free`.
  Inference temporaries come from the caller's scratch buffer.
* Host-only code (`src/train`, `src/format/write.c`, tools) may allocate.
  Check every allocation and free on all paths.
* Prefer `calloc` for structs and arrays whose fields are pointers, so a failed
  partial initialization is safe to free.
* Treat model blobs as read-only. Never cast away `const` from a blob.
* Do not assume struct layout for on-disk data. Use the explicit little-endian
  helpers in `src/format/endian.h`.

## Headers and includes

* Every header has an include guard named `ELML_<NAME>_H`.
* Public headers wrap declarations in `extern "C"` for C++ consumers.
* Include order: the matching header first, then standard headers, then project
  headers, alphabetically within each group.
* Include what you use. Do not rely on transitive includes.
* Keep internal headers under `src/` and out of the installed include tree.

## Kernels and numerics

* The reference kernel in `src/kernels/scalar_f32.c` is the source of truth. Any
  SIMD or multi-threaded implementation must produce results within the
  documented tolerance and must be selected behind a runtime or compile-time
  dispatcher, never by changing the reference.
* Use `float` (`f32`) for training and inference math; use `double` only for
  host-side reporting and configuration parsing.
* Prefer numerically stable forms: max-subtracted softmax, branch-stable
  sigmoid, epsilon guards on division and logarithms.
* Determinism matters. Training uses a seeded RNG and stable iteration order;
  do not introduce unordered iteration or uninitialized reads.

## Tests

* Every new behavior needs a test. Put unit tests in `tests/test_*.c` using
  `tests/test.h`, and end-to-end tests in `tests/test_cli.sh` or
  `examples/*/run.sh`.
* Test both the happy path and the error path: null pointers, truncated input,
  misalignment, bad CRC, invalid configuration, buffer too small.
* Tests must be deterministic and must not depend on the build directory.
* Register new C tests with `elml_add_test` in `tests/CMakeLists.txt`; shell
  tests use `add_test` with explicit arguments.

## Documentation

* Document the contract, not the implementation: ownership, alignment, units,
  valid ranges, and failure modes.
* Keep `docs/FORMAT.md`, `docs/CODE_GUIDE.md`, and the public headers in sync
  with behavior. A format change requires all three.
* Comments explain *why*; the code explains *what*. Avoid commenting every
  line. Use a short file-level comment for non-obvious modules.

## Commits and generated files

* Do not commit build output, generated example artifacts, or editor state.
  See `.gitignore`.
* Keep changes focused. A format or API change should include its documentation
  and tests in the same change.
