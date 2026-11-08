#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "elml/arena.h"
#include "elml/format.h"
#include "elml/model.h"
#include "elml/types.h"
#include "format/format_internal.h"
#include "format/write.h"
#include "test.h"

static const float g_w0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
static const float g_b0[2] = {0.5f, -0.5f};
static const float g_w1[4] = {1.0f, 0.0f, 0.0f, 1.0f};
static const float g_b1[2] = {0.0f, 0.0f};

static const elml_write_layer g_layers[2] = {
    {2u, 2u, ELML_ACT_RELU, 0.01f, g_w0, g_b0},
    {2u, 2u, ELML_ACT_LINEAR, 0.0f, g_w1, g_b1},
};

/* 30x24 layers make parameter sizes that are not powers of two. */
static float g_wide_w0[30u * 24u];
static float g_wide_b0[24u];
static float g_wide_w1[24u * 2u];
static float g_wide_b1[2u];

static elml_write_spec make_spec(void)
{
  static const float input_norm[4] = {0.0f, 1.0f, 0.0f, 1.0f};
  static const float output_norm[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  static const elml_write_meta meta[2] = {
      {"labels", "cat,dog"},
      {"columns.inputs", "x1,x2"},
  };
  elml_write_spec spec = {
      .input_dim = 2u,
      .output_dim = 2u,
      .layer_count = 2u,
      .layers = g_layers,
      .norm = {.input_mode = 1u,
               .output_mode = 1u,
               .input_params = input_norm,
               .output_params = output_norm},
      .meta = meta,
      .meta_count = 2u,
  };
  return spec;
}

static uint8_t *align_copy(const uint8_t *data, size_t len, size_t *out_len)
{
  const size_t capacity = elml_align_up(len, ELML_PARAM_ALIGN);
  uint8_t *copy = (uint8_t *)aligned_alloc(ELML_PARAM_ALIGN, capacity);
  if (copy == NULL) {
    return NULL;
  }
  memcpy(copy, data, len);
  *out_len = len;
  return copy;
}

int main(void)
{
  elml_write_spec spec = make_spec();
  elml_buf buffer;
  CHECK_EQ_INT(elml_model_write(&spec, &buffer), ELML_OK);
  CHECK(buffer.len > ELML_HEADER_BYTES);
  CHECK_EQ_INT(buffer.data[0], 'E');
  CHECK_EQ_INT(buffer.data[1], 'L');
  CHECK_EQ_INT(buffer.data[2], 'M');
  CHECK_EQ_INT(buffer.data[3], 'L');

  size_t blob_len = 0u;
  uint8_t *blob = align_copy(buffer.data, buffer.len, &blob_len);
  CHECK(blob != NULL);
  if (blob == NULL) {
    elml_buf_free(&buffer);
    TEST_END();
  }
  /* The blob owns its own copy; validation tests below reuse ``buffer``. */
  elml_buf_free(&buffer);

  elml_format format;
  CHECK_EQ_INT(elml_format_parse(blob, blob_len, &format), ELML_OK);
  CHECK_EQ_INT(format.dtype, ELML_DTYPE_F32);
  CHECK_EQ_INT(format.major, 1u);
  CHECK_EQ_INT(format.chunk_count, 4u);
  CHECK((format.file_flags & ELML_FILE_FLAG_HAS_NORM) != 0u);
  CHECK((format.file_flags & ELML_FILE_FLAG_HAS_META) != 0u);
  CHECK_EQ_INT(format.input.dim[0], 2u);
  CHECK_EQ_INT(format.output.dim[0], 2u);
  CHECK_EQ_INT(format.layer_count, 4u);
  CHECK_EQ_INT(format.descs[0].type_id, ELML_LAYER_DENSE);
  CHECK_EQ_INT(format.descs[1].type_id, ELML_LAYER_ACT_RELU);
  CHECK_EQ_INT(format.descs[2].type_id, ELML_LAYER_DENSE);
  CHECK_EQ_INT(format.descs[3].type_id, ELML_LAYER_ACT_LINEAR);
  CHECK_EQ_INT(format.descs[0].in.dim[0], 2u);
  CHECK_EQ_INT(format.descs[0].out.dim[0], 2u);
  CHECK_EQ_INT(format.descs[0].param_off % ELML_PARAM_ALIGN,
               ELML_PARAM_ALIGN -
                   ((ELML_HEADER_BYTES + ELML_CHUNK_HEADER_BYTES) %
                    ELML_PARAM_ALIGN));
  CHECK_EQ_INT(((uintptr_t)(format.parm + format.descs[0].param_off)) %
                   ELML_PARAM_ALIGN,
               0u);
  CHECK_EQ_INT(((uintptr_t)(format.parm + format.descs[2].param_off)) %
                   ELML_PARAM_ALIGN,
               0u);
  CHECK_EQ_INT(format.norm_input_mode, 1u);
  CHECK_EQ_INT(format.norm_output_mode, 1u);
  CHECK_EQ_INT(format.norm_input_count, 2u);
  CHECK_EQ_INT(format.norm_output_count, 2u);
  CHECK_CLOSE(format.norm_input_params[0], 0.0, 1e-6);
  CHECK_CLOSE(format.norm_input_params[1], 1.0, 1e-6);
  CHECK_CLOSE(format.norm_output_params[2], 3.0, 1e-6);

  elml_model model;
  CHECK_EQ_INT(elml_model_init(&model, blob, blob_len, ELML_DTYPE_F32),
               ELML_OK);
  CHECK_EQ_INT(model.layer_count, 4u);
  CHECK_EQ_INT(elml_label_count(&model), 2u);
  char label[16];
  size_t label_len = 0u;
  CHECK_EQ_INT(elml_label_get(&model, 0u, label, sizeof label, &label_len),
               ELML_OK);
  CHECK_STR(label, "cat");
  CHECK_EQ_INT(label_len, 3u);
  CHECK_EQ_INT(elml_label_get(&model, 2u, label, sizeof label, &label_len),
               ELML_ERR_NOT_FOUND);
  CHECK_EQ_INT(elml_label_get(&model, 1u, label, 2u, &label_len),
               ELML_ERR_BUFFER_TOO_SMALL);
  char value[32];
  CHECK_EQ_INT(elml_meta_get(&model, "columns.inputs", value, sizeof value,
                             &label_len),
               ELML_OK);
  CHECK_STR(value, "x1,x2");
  CHECK_EQ_INT(elml_meta_get(&model, "missing", value, sizeof value,
                             &label_len),
               ELML_ERR_NOT_FOUND);
  CHECK_EQ_INT(elml_meta_get(&model, "labels", value, 3u, &label_len),
               ELML_ERR_BUFFER_TOO_SMALL);

  elml_dtype dtype = ELML_DTYPE_F16;
  CHECK_EQ_INT(elml_model_dtype(blob, blob_len, &dtype), ELML_OK);
  CHECK_EQ_INT(dtype, ELML_DTYPE_F32);
  CHECK_EQ_INT(elml_model_init(&model, blob, blob_len, ELML_DTYPE_INT8),
               ELML_ERR_DTYPE_MISMATCH);
  CHECK_EQ_INT(elml_model_init(&model, blob, blob_len, ELML_DTYPE_F16),
               ELML_ERR_UNSUPPORTED_DTYPE);
  CHECK_EQ_INT(elml_model_init(&model, blob + 1, blob_len - 1u,
                               ELML_DTYPE_F32),
               ELML_ERR_MISALIGNED);
  CHECK_EQ_INT(elml_model_init(NULL, blob, blob_len, ELML_DTYPE_F32),
               ELML_ERR_INVALID_ARG);
  /* Failed init calls above zero the model, so bind it again. */
  CHECK_EQ_INT(elml_model_init(&model, blob, blob_len, ELML_DTYPE_F32),
               ELML_OK);

  uint8_t *scratch = (uint8_t *)aligned_alloc(ELML_TENSOR_ALIGN,
                                              elml_scratch_bytes(&model, 1u));
  CHECK(scratch != NULL);
  if (scratch != NULL) {
    float input[2] = {3.0f, 5.0f};
    float output[2] = {0.0f, 0.0f};
    CHECK_EQ_INT(elml_forward(&model, scratch, elml_scratch_bytes(&model, 1u),
                              input, output),
                 ELML_OK);
    /* Hidden: relu([13.5,28.5]); identity output; denormalized. */
    CHECK_CLOSE(output[0], 1.0 + 2.0 * 13.5, 1e-5);
    CHECK_CLOSE(output[1], 3.0 + 4.0 * 28.5, 1e-5);
    free(scratch);
  }

  uint8_t *corrupt = (uint8_t *)malloc(blob_len);
  CHECK(corrupt != NULL);
  if (corrupt != NULL) {
    memcpy(corrupt, blob, blob_len);
    corrupt[ELML_HEADER_BYTES + ELML_CHUNK_HEADER_BYTES + 3u] ^= 0x01u;
    CHECK_EQ_INT(elml_format_parse(corrupt, blob_len, &format), ELML_ERR_CRC);

    memcpy(corrupt, blob, blob_len);
    corrupt[0] = 'X';
    CHECK_EQ_INT(elml_format_parse(corrupt, blob_len, &format),
                 ELML_ERR_BAD_MAGIC);

    memcpy(corrupt, blob, blob_len);
    corrupt[4] = 2u;
    corrupt[5] = 0u;
    CHECK_EQ_INT(elml_format_parse(corrupt, blob_len, &format),
                 ELML_ERR_UNSUPPORTED_MAJOR);

    memcpy(corrupt, blob, blob_len);
    corrupt[8] = 32u;
    corrupt[9] = 0u;
    CHECK_EQ_INT(elml_format_parse(corrupt, blob_len, &format),
                 ELML_ERR_UNSUPPORTED_MINOR);

    CHECK_EQ_INT(elml_format_parse(blob, blob_len - 1u, &format),
                 ELML_ERR_TRUNCATED);
    CHECK_EQ_INT(elml_format_parse(NULL, blob_len, &format),
                 ELML_ERR_INVALID_ARG);
    free(corrupt);
  }

  elml_write_spec bad = spec;
  bad.layers = NULL;
  CHECK_EQ_INT(elml_model_write(&bad, &buffer), ELML_ERR_INVALID_ARG);
  bad = spec;
  bad.layer_count = 0u;
  CHECK_EQ_INT(elml_model_write(&bad, &buffer), ELML_ERR_INVALID_ARG);
  bad = spec;
  bad.layers = g_layers;
  elml_write_layer broken[2] = {g_layers[0], g_layers[1]};
  broken[1].in_dim = 3u;
  bad.layers = broken;
  CHECK_EQ_INT(elml_model_write(&bad, &buffer), ELML_ERR_BAD_SHAPE);
  bad = spec;
  elml_write_layer null_weights[2] = {g_layers[0], g_layers[1]};
  null_weights[0].weights = NULL;
  bad.layers = null_weights;
  CHECK_EQ_INT(elml_model_write(&bad, &buffer), ELML_ERR_BAD_SHAPE);
  bad = spec;
  bad.norm.input_mode = 9u;
  CHECK_EQ_INT(elml_model_write(&bad, &buffer), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_model_write(NULL, &buffer), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_model_write(&spec, NULL), ELML_ERR_INVALID_ARG);

  /* Regression: non-power-of-two parameter strides used to be passed as an
     alignment, which produced a false out-of-memory error on export. */
  {
    elml_write_layer wide[2] = {
        {30u, 24u, ELML_ACT_RELU, 0.0f, g_wide_w0, g_wide_b0},
        {24u, 2u, ELML_ACT_LINEAR, 0.0f, g_wide_w1, g_wide_b1},
    };
    elml_write_spec wide_spec = {
        .input_dim = 30u,
        .output_dim = 2u,
        .layer_count = 2u,
        .layers = wide,
    };
    elml_buf wide_buf;
    CHECK_EQ_INT(elml_model_write(&wide_spec, &wide_buf), ELML_OK);
    CHECK(wide_buf.len > ELML_HEADER_BYTES);

    uint8_t *wide_blob = align_copy(wide_buf.data, wide_buf.len, &blob_len);
    CHECK(wide_blob != NULL);
    if (wide_blob != NULL) {
      CHECK_EQ_INT(elml_format_parse(wide_blob, blob_len, &format), ELML_OK);
      CHECK_EQ_INT(format.layer_count, 4u);
      CHECK_EQ_INT(((uintptr_t)(format.parm + format.descs[0].param_off)) %
                       ELML_PARAM_ALIGN,
                   0u);
      CHECK_EQ_INT(((uintptr_t)(format.parm + format.descs[2].param_off)) %
                       ELML_PARAM_ALIGN,
                   0u);
      free(wide_blob);
    }
    elml_buf_free(&wide_buf);
  }

  elml_buf_free(&buffer);
  elml_buf_free(NULL);
  free(blob);
  TEST_END();
}
