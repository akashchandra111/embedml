/*
 * Non-normative parser views. The public model API binds these views into
 * elml_model; the inspect tool uses them directly so it can describe files that
 * contain layer types this build does not implement.
 */
#ifndef ELML_FORMAT_INTERNAL_H
#define ELML_FORMAT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "elml/format.h"
#include "elml/registry.h"
#include "elml/types.h"

typedef struct elml_chunk_view {
  uint32_t type;
  uint16_t version;
  uint16_t flags;
  uint32_t crc;
  const uint8_t *data;
  uint32_t len;
} elml_chunk_view;

typedef struct elml_format {
  elml_dtype dtype;
  uint16_t major;
  uint16_t minor;
  uint8_t file_flags;
  elml_shape input;
  elml_shape output;
  uint16_t layer_count;
  elml_layer_desc descs[ELML_MAX_LAYERS];

  const uint8_t *conf;
  uint32_t conf_len;
  const uint8_t *parm;
  uint32_t parm_len;
  const uint8_t *qnt;
  uint32_t qnt_len;
  const uint8_t *norm;
  uint32_t norm_len;
  const uint8_t *meta;
  uint32_t meta_len;

  uint8_t norm_input_mode;
  uint8_t norm_output_mode;
  uint32_t norm_input_count;
  uint32_t norm_output_count;
  const float *norm_input_params;
  const float *norm_output_params;

  uint32_t chunk_count;
  size_t file_len;
} elml_format;

/* Parse and fully validate the container, without resolving layer types. */
elml_status elml_format_parse(const void *blob, size_t blob_len,
                              elml_format *out);

/* Header-only probe. */
elml_status elml_format_probe(const void *blob, size_t blob_len,
                              elml_dtype *dtype);

/* Iterate chunk headers. Returns ELML_ERR_NOT_FOUND at end of file. */
elml_status elml_chunk_next(const uint8_t *blob, size_t blob_len, size_t *offset,
                            elml_chunk_view *out);

#endif /* ELML_FORMAT_INTERNAL_H */
