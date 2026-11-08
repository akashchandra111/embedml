#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "elml/format.h"
#include "elml/model.h"
#include "elml/types.h"
#include "format/format_internal.h"
#include "format/write.h"
#include "test.h"

static uint8_t *align_copy(const uint8_t *data, size_t len)
{
  uint8_t *copy = (uint8_t *)aligned_alloc(ELML_PARAM_ALIGN,
                                          (len + ELML_PARAM_ALIGN - 1u) &
                                              ~(size_t)(ELML_PARAM_ALIGN - 1u));
  if (copy != NULL) {
    memcpy(copy, data, len);
  }
  return copy;
}

int main(void)
{
  static const elml_status statuses[] = {
      ELML_OK,
      ELML_ERR_INVALID_ARG,
      ELML_ERR_OUT_OF_MEMORY,
      ELML_ERR_TRUNCATED,
      ELML_ERR_BAD_MAGIC,
      ELML_ERR_UNSUPPORTED_MAJOR,
      ELML_ERR_UNSUPPORTED_MINOR,
      ELML_ERR_DTYPE_MISMATCH,
      ELML_ERR_UNSUPPORTED_DTYPE,
      ELML_ERR_UNSUPPORTED_LAYER_TYPE,
      ELML_ERR_BAD_SHAPE,
      ELML_ERR_BAD_PARAM,
      ELML_ERR_CRC,
      ELML_ERR_MISALIGNED,
      ELML_ERR_SCRATCH_TOO_SMALL,
      ELML_ERR_UNKNOWN_CHUNK_REQUIRED,
      ELML_ERR_NOT_FOUND,
      ELML_ERR_BUFFER_TOO_SMALL,
      ELML_ERR_INTERNAL,
  };
  for (size_t i = 0u; i < sizeof statuses / sizeof statuses[0]; ++i) {
    const char *text = elml_error_string(statuses[i]);
    CHECK(text != NULL);
    CHECK(strlen(text) > 0u);
  }
  CHECK_STR(elml_error_string((elml_status)-999), "unknown error");
  CHECK(strstr(elml_version_string(), "EmbedLML") != NULL);

  static const float weights[2] = {1.0f, 2.0f};
  static const float biases[1] = {0.0f};
  const elml_write_layer layer = {
      .in_dim = 2u,
      .out_dim = 1u,
      .act = ELML_ACT_LINEAR,
      .weights = weights,
      .biases = biases,
  };
  const elml_write_spec spec = {
      .input_dim = 2u,
      .output_dim = 1u,
      .layer_count = 1u,
      .layers = &layer,
  };
  elml_buf buffer;
  CHECK_EQ_INT(elml_model_write(&spec, &buffer), ELML_OK);
  uint8_t *blob = align_copy(buffer.data, buffer.len);
  CHECK(blob != NULL);
  const size_t blob_len = buffer.len;
  elml_buf_free(&buffer);
  if (blob == NULL) {
    TEST_END();
  }

  size_t offset = ELML_HEADER_BYTES;
  uint32_t chunks = 0u;
  elml_chunk_view chunk;
  while (elml_chunk_next(blob, blob_len, &offset, &chunk) == ELML_OK) {
    chunks++;
    CHECK(chunk.len > 0u);
    CHECK(chunk.data != NULL);
  }
  CHECK_EQ_INT(chunks, 3u);
  CHECK_EQ_INT(elml_chunk_next(blob, blob_len, &offset, &chunk),
               ELML_ERR_NOT_FOUND);
  CHECK_EQ_INT(elml_chunk_next(NULL, blob_len, &offset, &chunk),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_chunk_next(blob, blob_len, NULL, &chunk),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_chunk_next(blob, blob_len, &offset, NULL),
               ELML_ERR_INVALID_ARG);
  size_t bad_offset = blob_len;
  CHECK_EQ_INT(elml_chunk_next(blob, blob_len, &bad_offset, &chunk),
               ELML_ERR_NOT_FOUND);

  elml_model model;
  CHECK_EQ_INT(elml_model_init(&model, blob, blob_len, ELML_DTYPE_F32),
               ELML_OK);
  CHECK_EQ_INT(elml_label_count(&model), 0u);
  char label[8];
  size_t label_len = 0u;
  CHECK_EQ_INT(elml_label_get(&model, 0u, label, sizeof label, &label_len),
               ELML_ERR_NOT_FOUND);
  CHECK_EQ_INT(elml_meta_get(&model, "labels", label, sizeof label,
                             &label_len),
               ELML_ERR_NOT_FOUND);
  CHECK_EQ_INT(elml_meta_get(NULL, "labels", label, sizeof label, &label_len),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_meta_get(&model, NULL, label, sizeof label, &label_len),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_meta_get(&model, "labels", label, 0u, &label_len),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_meta_get(&model, "labels", label, sizeof label, NULL),
               ELML_ERR_INVALID_ARG);

  float input[2] = {1.0f, 2.0f};
  float output[1] = {0.0f};
  CHECK_EQ_INT(elml_forward_batch(&model, NULL, 0u, input, output, 1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_forward_batch(&model, (void *)input, 64u, NULL, output,
                                  1u),
               ELML_ERR_INVALID_ARG);

  CHECK_EQ_INT(elml_model_init(&model, NULL, blob_len, ELML_DTYPE_F32),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_model_init(&model, blob, 0u, ELML_DTYPE_F32),
               ELML_ERR_INVALID_ARG);

  elml_dtype dtype = ELML_DTYPE_F32;
  CHECK_EQ_INT(elml_model_dtype(NULL, blob_len, &dtype),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_model_dtype(blob, blob_len, NULL), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_model_dtype(blob, 4u, &dtype), ELML_ERR_TRUNCATED);

  uint8_t header[ELML_HEADER_BYTES] = {0};
  header[0] = ELML_MAGIC_0;
  header[1] = ELML_MAGIC_1;
  header[2] = ELML_MAGIC_2;
  header[3] = ELML_MAGIC_3;
  header[4] = 1u;
  header[8] = ELML_HEADER_BYTES;
  header[10] = 2u; /* f16 is reserved and unsupported */
  CHECK_EQ_INT(elml_model_dtype(header, sizeof header, &dtype),
               ELML_ERR_UNSUPPORTED_DTYPE);
  header[10] = 1u; /* int8 is representable in the format */
  CHECK_EQ_INT(elml_model_dtype(header, sizeof header, &dtype), ELML_OK);
  CHECK_EQ_INT(dtype, ELML_DTYPE_INT8);
  header[0] = 'X';
  CHECK_EQ_INT(elml_model_dtype(header, sizeof header, &dtype),
               ELML_ERR_BAD_MAGIC);

  free(blob);
  TEST_END();
}
