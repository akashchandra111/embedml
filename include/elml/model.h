/*
 * Model loading and inference API.
 *
 * All pointers reference the caller-owned model blob; nothing is copied and
 * nothing is allocated. The blob must stay alive for the lifetime of the model
 * and must be aligned to ELML_PARAM_ALIGN for zero-copy parameter access.
 */
#ifndef ELML_MODEL_H
#define ELML_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "elml/compat.h"
#include "elml/registry.h"
#include "elml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum elml_norm_mode {
  ELML_NORM_NONE = 0,
  ELML_NORM_STANDARD = 1,
  ELML_NORM_MINMAX = 2
} elml_norm_mode;

typedef struct elml_norm {
  elml_norm_mode input_mode;
  elml_norm_mode output_mode;
  uint32_t input_count;
  uint32_t output_count;
  const float *input_params;  /* 2 * input_count: mean/std or min/max */
  const float *output_params; /* 2 * output_count */
} elml_norm;

typedef struct elml_meta {
  const uint8_t *data;
  uint32_t len;
} elml_meta;

typedef struct elml_model {
  const uint8_t *blob;
  size_t blob_len;
  elml_dtype dtype;
  elml_shape input;
  elml_shape output;
  uint16_t layer_count;
  uint16_t file_flags;

  elml_layer_desc descs[ELML_MAX_LAYERS];
  elml_layer layers[ELML_MAX_LAYERS];

  const uint8_t *parm;
  uint32_t parm_len;
  const uint8_t *qnt;
  uint32_t qnt_len;

  elml_norm norm;
  elml_meta meta;

  size_t max_activation_bytes;
  size_t scratch_bytes;
} elml_model;

/*
 * Validate and bind a model.
 *
 * expected_dtype is the precision the caller compiled for. A mismatch is an
 * error (ELML_ERR_DTYPE_MISMATCH); this API never converts silently.
 */
elml_status elml_model_init(elml_model *model, const void *blob,
                            size_t blob_len, elml_dtype expected_dtype);

/* Inspect the header without initializing a model. */
elml_status elml_model_dtype(const void *blob, size_t blob_len,
                             elml_dtype *dtype);

/* Scratch bytes required for a forward pass. Never underestimates. */
size_t elml_scratch_bytes(const elml_model *model, uint32_t batch);

/*
 * Run one sample. Input values are raw feature values; normalization stored in
 * the model is applied internally. Output is denormalized for regressions.
 */
elml_status elml_forward(const elml_model *model, void *scratch,
                         size_t scratch_len, const float *input, float *output);

elml_status elml_forward_batch(const elml_model *model, void *scratch,
                               size_t scratch_len, const float *input,
                               float *output, uint32_t batch);

/*
 * Fully quantized boundary for MCU builds: no float in the hot loop. The
 * caller supplies the quantized input and receives the quantized output plus
 * the scale needed to interpret it.
 */
elml_status elml_forward_q(const elml_model *model, void *scratch,
                           size_t scratch_len, const int8_t *input,
                           int8_t *output, elml_quant *output_quant);

/* UTF-8 metadata lookup. Returns ELML_ERR_NOT_FOUND when the key is absent. */
elml_status elml_meta_get(const elml_model *model, const char *key, char *buf,
                          size_t cap, size_t *out_len);

uint32_t elml_label_count(const elml_model *model);
elml_status elml_label_get(const elml_model *model, uint32_t index, char *buf,
                           size_t cap, size_t *out_len);

const char *elml_error_string(elml_status status);
const char *elml_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* ELML_MODEL_H */
