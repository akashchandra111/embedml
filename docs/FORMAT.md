# EmbedLML model format v1.0

This document specifies the binary model file consumed by `libelml_core`. It is
the normative reference for the writer in `src/format/write.c` and the parser in
`src/format/read.c`.

## Design rules

* **Little-endian.** Every multi-byte integer is stored least-significant byte
  first. Floating-point values are IEEE-754 binary32, also little-endian.
* **No structs on the wire.** Fields are written and read one at a time through
  explicit helpers. No padding, packing, or compiler layout affects the file.
* **Versioned.** `major` changes when the meaning of existing bytes changes.
  `minor` changes for backwards-compatible additions. A v1.0 reader accepts
  chunk payload versions newer than 1 only when the chunk is not marked
  `REQUIRED`.
* **One dtype per file.** A file is entirely f32 or entirely int8. v1.0 tools
  produce f32 only; int8 is reserved and rejected by the runtime.
* **Checksummed.** Every chunk has a CRC32, and the whole chunk region has a
  second CRC32.
* **Aligned parameters.** Parameter blocks start at 64-byte absolute file
  offsets so an MCU can map them in place.

All integers below are unsigned unless stated otherwise.

## File layout

```
+---------------------------+
| header (64 bytes)         |
+---------------------------+
| chunk 0 (16 + payload)    |  PARM (required, always first)
+---------------------------+
| padding to 8-byte boundary|
+---------------------------+
| chunk 1                   |  CONF (required)
+---------------------------+
| ...                       |  NORM, META, future chunks
+---------------------------+
```

Chunks are independent and may be reordered by a future writer, with one
exception: `PARM` must start at absolute offset 64 because its parameter offsets
are computed from that position. Readers locate chunks by scanning, not by
position.

## Header (64 bytes)

| Offset | Size | Field | Value / meaning |
| --- | --- | --- | --- |
| 0 | 4 | magic | `45 4C 4D 4C` = `"ELML"` |
| 4 | 2 | major | `1` |
| 6 | 2 | minor | `0` |
| 8 | 2 | header_bytes | `64` |
| 10 | 1 | dtype | `0` = f32, `1` = int8, `2` = reserved (f16) |
| 11 | 1 | file_flags | see below |
| 12 | 4 | chunk_count | number of chunks that follow |
| 16 | 8 | total_bytes | size of the whole file, including the header |
| 20 | 4 | reserved | must be zero in v1.0 |
| 24 | 4 | crc32_chunks | CRC32 over bytes `[64, total_bytes)` |
| 28 | 36 | reserved | must be zero in v1.0 |

`file_flags`:

| Bit | Mask | Meaning |
| --- | --- | --- |
| 0 | `0x01` | little-endian (always set) |
| 1 | `0x02` | sequential topology (always set in v1.0) |
| 2 | `0x04` | a `NORM` chunk is present |
| 3 | `0x08` | a `META` chunk is present |
| 4-7 | | reserved, zero |

CRC32 is the reflected IEEE 802.3 polynomial `0xEDB88320`, initial value `0`,
no final XOR. It is the same function used for chunks; see
`src/format/crc32.c`.

## Chunk header (16 bytes)

| Offset | Size | Field | Meaning |
| --- | --- | --- | --- |
| 0 | 4 | type | FourCC, little-endian (see below) |
| 4 | 2 | version | payload version, `1` in v1.0 |
| 6 | 2 | flags | `0x0001` required, `0x0002` critical |
| 8 | 4 | payload_len | bytes of payload |
| 12 | 4 | crc32 | CRC32 over the payload |

The payload begins immediately at chunk offset + 16. After the payload the file
is padded with zero bytes to the next 8-byte boundary. Padding is not included
in `payload_len` or the chunk CRC.

FourCC values (ASCII in file order):

| FourCC | Constant | Required | Purpose |
| --- | --- | --- | --- |
| `CONF` | `ELML_CHUNK_CONF` | yes | topology and layer descriptors |
| `PARM` | `ELML_CHUNK_PARM` | yes | weights and biases |
| `QNT ` | `ELML_CHUNK_QNT` | int8 only | quantization parameters (reserved) |
| `NORM` | `ELML_CHUNK_NORM` | no | input/output normalization |
| `META` | `ELML_CHUNK_META` | no | UTF-8 key/value metadata |
| `TOPO` | `ELML_CHUNK_TOPO` | reserved | future non-sequential topology |
| `CKPT` | `ELML_CHUNK_CKPT` | reserved | future training checkpoint |

Parser rules:

* `CONF` and `PARM` must each appear exactly once.
* Each optional chunk type may appear at most once.
* Unknown chunks are skipped unless `REQUIRED` is set, in which case loading
  fails with `ELML_ERR_UNKNOWN_CHUNK_REQUIRED`.
* `TOPO` and `CKPT` are recognized and skipped in v1.0.
* For an int8 file, `QNT` is mandatory.

## `CONF` payload

```
u32 layer_count            // number of layers in the chain
u8  input_rank             // 1 in v1.0
u8  output_rank            // 1 in v1.0
u16 reserved               // 0
u32 input_dim[input_rank]
u32 output_dim[output_rank]

repeat layer_count times:
  u16 type_id
  u16 type_version         // 1
  u32 flags
  u32 param_off            // relative to PARM payload, UINT32_MAX if none
  u32 param_bytes
  u8  in_rank
  u8  out_rank
  u32 in_dim[in_rank]
  u32 out_dim[out_rank]
  u32 extra_bytes
  u8  extra[extra_bytes]
```

The chain in v1.0 is strictly sequential: layer 0's `in` equals the model input,
each layer's `in` equals the previous layer's `out`, and the last layer's `out`
equals the model output. The parser rejects any mismatch.

The trainer emits a dense layer followed by an activation layer for every
configured network layer. A 4 -> 8 -> 3 classifier therefore contains four
layers:

```
dense [4]->[8], relu [8]->[8], dense [8]->[3], softmax [3]->[3]
```

Layer ids and versions are permanent. A reader that does not know a type id
fails with `ELML_ERR_UNSUPPORTED_LAYER_TYPE` rather than guessing.

| Id | Type | Payload |
| --- | --- | --- |
| `0x0001` | DENSE | weights then biases in `PARM` |
| `0x0010` | ReLU | none |
| `0x0011` | leaky ReLU | optional `f32 alpha` in `extra` |
| `0x0012` | tanh | none |
| `0x0013` | sigmoid | none |
| `0x0014` | softmax | none |
| `0x0015` | linear | none |
| `0x0002` | CONV1D | reserved |

Layer flags:

| Mask | Meaning |
| --- | --- |
| `0x0001` | trainable |
| `0x0002` | fused with previous layer |
| `0x0004` | has bias |
| `0x0100` | runtime-only: folded into the preceding layer |

The writer sets `TRAINABLE | HAS_BIAS` on dense layers and `0` on activation
layers. The runtime fuses an activation into a preceding dense layer when the
shapes match; the activation descriptor remains in `CONF` but is skipped during
the forward pass.

## `PARM` payload

`PARM` is required and must be the first chunk, at absolute offset 64. Its
payload starts at absolute offset 80. The first 48 bytes are zero padding so the
first parameter block begins at absolute offset 128:

```
PARM payload offset 0    : 48 zero bytes
PARM payload offset 48   : layer 0 parameters   (absolute 128)
PARM payload offset ...  : layer 1 parameters
```

Each dense layer's parameters are laid out as:

```
weights: out_dim * in_dim f32, row-major (row = output neuron)
         padded with zeros to a multiple of 64 bytes
biases:  out_dim f32
         padded with zeros to a multiple of 64 bytes
```

The padding is measured from the start of the layer's block, and every layer
block is a multiple of 64 bytes. Consequently every weights pointer and every
biases pointer is 64-byte aligned in absolute file coordinates, provided the
file itself is loaded at a 64-byte-aligned address. `desc.param_off` points at
the start of the layer block, relative to the `PARM` payload; `desc.param_bytes`
counts weights, weight padding, biases, and bias padding.

Activation layers have `param_off = UINT32_MAX` and `param_bytes = 0`.

## `NORM` payload

`NORM` is optional. When both modes are `NONE`, the writer omits the chunk.
When present, the payload is:

```
u8  input_mode             // 0 none, 1 standard, 2 min-max
u8  output_mode
u16 reserved
u32 input_count
f32 input_params[2 * input_count]    // always present, zero-filled if mode == 0
u32 output_count
f32 output_params[2 * output_count]  // always present, zero-filled if mode == 0
```

For `standard`, the pair is `(mean, stddev)`:

```
input  : x' = (x - mean) / stddev
output : y  = y' * stddev + mean
```

For `min-max`, the pair is `(min, max)` and values map to `[-1, 1]`:

```
input  : x' = (x - min) / (max - min) * 2 - 1
output : y  = (y' + 1) / 2 * (max - min) + min
```

`input_count` must equal the model input dimension and `output_count` must equal
the model output dimension. The runtime applies input normalization before layer
0 and output denormalization after the last layer, so callers pass raw feature
values to `elml_forward`.

The trainer computes statistics on the training split only. Output
normalization is applied for regression and skipped for classification.

## `META` payload

`META` is optional. It is a sequence of TLV records:

```
u16 key_len
u16 value_len
u8  key[key_len]
u8  value[value_len]
```

Keys and values are UTF-8 byte strings and are not NUL-terminated in the file.
The writer always emits `producer = "EmbedLML 0.1.0"` and may emit:

| Key | Value |
| --- | --- |
| `producer` | library name and version |
| `columns.inputs` | comma-separated input column names |
| `columns.outputs` | comma-separated output column names |
| `labels` | comma-separated class labels, sorted |
| `task` | `regression` or `classification` |
| `loss` | `mse`, `mae`, or `cce` |

`elml_run` uses `columns.inputs` and `columns.outputs` as defaults when the user
does not pass `--inputs` / `--outputs`. `elml_label_count` and `elml_label_get`
read the `labels` key. Unknown keys are preserved and ignored.

Class labels must not contain commas and there may be at most 255 classes,
because labels are stored as one comma-separated value and class indices are
`uint8_t`.

## `QNT` payload (reserved)

int8 files require a `QNT` chunk. Its layout is not finalized in v1.0 and the
current runtime rejects int8 models with `ELML_ERR_UNSUPPORTED_DTYPE` after dtype
validation. Writers do not emit `QNT`.

## Validation summary

The loader rejects a file when:

* the magic is wrong (`ELML_ERR_BAD_MAGIC`);
* the major version is not 1 (`ELML_ERR_UNSUPPORTED_MAJOR`);
* `header_bytes` is not 64 (`ELML_ERR_UNSUPPORTED_MINOR`);
* the dtype is unknown or not the requested dtype
  (`ELML_ERR_UNSUPPORTED_DTYPE`, `ELML_ERR_DTYPE_MISMATCH`);
* `total_bytes` does not match the supplied buffer length, a chunk header or
  payload runs past the buffer, or trailing bytes remain
  (`ELML_ERR_TRUNCATED`);
* either CRC does not match (`ELML_ERR_CRC`);
* a required chunk is missing or duplicated, a payload is malformed, or an int8
  file lacks `QNT` (`ELML_ERR_BAD_PARAM`);
* a shape is zero, ranked too high, not sequential, or inconsistent with the
  PARM block (`ELML_ERR_BAD_SHAPE`);
* a layer type is not linked into the build
  (`ELML_ERR_UNSUPPORTED_LAYER_TYPE`);
* the model blob is not 64-byte aligned, or a parameter block is not
  (`ELML_ERR_MISALIGNED`).

## Portability note

The file is endian-independent, but the current inference core reads f32
parameter and normalization values through host `float` pointers. It therefore
assumes an IEEE-754 little-endian host, which covers common MCU cores
(Cortex-M, RISC-V little-endian, Xtensa). Supporting a big-endian host would
require byte-swapping parameter views at load time; that is a future change, not
a change to this file format.
