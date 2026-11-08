/*
 * Core value types shared by the inference API.
 *
 * The public headers intentionally avoid C23-only syntax so that legacy MCU
 * toolchains (C11) can consume the library without a custom preprocessor pass.
 * The implementation itself is built as C23.
 */
#ifndef ELML_TYPES_H
#define ELML_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "elml/compat.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ELML_MAX_LAYERS
#  define ELML_MAX_LAYERS 64
#endif

#ifndef ELML_MAX_RANK
#  define ELML_MAX_RANK 4
#endif

/* Parameter blobs in a model file start on 64-byte absolute offsets. */
#define ELML_PARAM_ALIGN 64u
/* Scratch buffers are aligned for the widest SIMD access we plan to use. */
#define ELML_TENSOR_ALIGN 16u

typedef enum elml_status {
  ELML_OK = 0,
  ELML_ERR_INVALID_ARG = -1,
  ELML_ERR_OUT_OF_MEMORY = -2,
  ELML_ERR_TRUNCATED = -3,
  ELML_ERR_BAD_MAGIC = -4,
  ELML_ERR_UNSUPPORTED_MAJOR = -5,
  ELML_ERR_UNSUPPORTED_MINOR = -6,
  ELML_ERR_DTYPE_MISMATCH = -7,
  ELML_ERR_UNSUPPORTED_DTYPE = -8,
  ELML_ERR_UNSUPPORTED_LAYER_TYPE = -9,
  ELML_ERR_BAD_SHAPE = -10,
  ELML_ERR_BAD_PARAM = -11,
  ELML_ERR_CRC = -12,
  ELML_ERR_MISALIGNED = -13,
  ELML_ERR_SCRATCH_TOO_SMALL = -14,
  ELML_ERR_UNKNOWN_CHUNK_REQUIRED = -15,
  ELML_ERR_NOT_FOUND = -16,
  ELML_ERR_BUFFER_TOO_SMALL = -17,
  ELML_ERR_INTERNAL = -18
} elml_status;

typedef enum elml_dtype {
  ELML_DTYPE_F32 = 0,
  ELML_DTYPE_INT8 = 1,
  ELML_DTYPE_F16 = 2 /* reserved, never produced by this version */
} elml_dtype;

typedef struct elml_shape {
  uint8_t rank;
  uint32_t dim[ELML_MAX_RANK];
} elml_shape;

typedef struct elml_quant {
  float scale;
  int32_t zero_point;
} elml_quant;

typedef struct elml_tensor {
  void *data;
  elml_dtype dtype;
  elml_shape shape;
  elml_quant q; /* meaningful for ELML_DTYPE_INT8 only */
} elml_tensor;

static_assert(ELML_MAX_LAYERS > 0, "ELML_MAX_LAYERS must be positive");
static_assert(ELML_MAX_RANK > 0 && ELML_MAX_RANK <= 8,
              "ELML_MAX_RANK must be in [1, 8]");

#ifdef __cplusplus
}
#endif

#endif /* ELML_TYPES_H */
