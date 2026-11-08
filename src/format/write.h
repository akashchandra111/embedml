/*
 * Host-only model writer. The inference core never writes files, so this lives
 * outside libelml_core and may allocate.
 */
#ifndef ELML_WRITE_H
#define ELML_WRITE_H

#include <stddef.h>
#include <stdint.h>

#include "elml/registry.h"
#include "elml/types.h"

typedef struct elml_buf {
  uint8_t *data;
  size_t len;
  size_t cap;
} elml_buf;

typedef struct elml_write_layer {
  uint32_t in_dim;
  uint32_t out_dim;
  elml_activation act;
  float leaky_alpha;
  const float *weights; /* out_dim * in_dim, row-major */
  const float *biases;  /* out_dim */
} elml_write_layer;

typedef struct elml_write_norm {
  uint8_t input_mode;         /* elml_norm_mode, 0 = none */
  uint8_t output_mode;
  const float *input_params;  /* 2 * input_dim */
  const float *output_params; /* 2 * output_dim */
} elml_write_norm;

typedef struct elml_write_meta {
  const char *key;
  const char *value;
} elml_write_meta;

typedef struct elml_write_spec {
  uint32_t input_dim;
  uint32_t output_dim;
  uint16_t layer_count;
  const elml_write_layer *layers;
  elml_write_norm norm;
  const elml_write_meta *meta;
  uint32_t meta_count;
} elml_write_spec;

elml_status elml_model_write(const elml_write_spec *spec, elml_buf *out);

void elml_buf_free(elml_buf *buf);

#endif /* ELML_WRITE_H */
