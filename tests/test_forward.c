#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "elml/arena.h"
#include "elml/model.h"
#include "elml/types.h"
#include "format/write.h"
#include "test.h"

static uint8_t *align_copy(const uint8_t *data, size_t len)
{
  const size_t capacity = elml_align_up(len, ELML_PARAM_ALIGN);
  uint8_t *copy = (uint8_t *)aligned_alloc(ELML_PARAM_ALIGN, capacity);
  if (copy != NULL) {
    memcpy(copy, data, len);
  }
  return copy;
}

static int load_softmax_model(elml_model *model, uint8_t **blob_out)
{
  static const float weights[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  static const float biases[2] = {0.0f, 0.0f};
  const elml_write_layer layer = {
      .in_dim = 2u,
      .out_dim = 2u,
      .act = ELML_ACT_SOFTMAX,
      .leaky_alpha = 0.0f,
      .weights = weights,
      .biases = biases,
  };
  const elml_write_spec spec = {
      .input_dim = 2u,
      .output_dim = 2u,
      .layer_count = 1u,
      .layers = &layer,
  };

  elml_buf buffer;
  if (elml_model_write(&spec, &buffer) != ELML_OK) {
    return 1;
  }
  const size_t length = buffer.len;
  uint8_t *blob = align_copy(buffer.data, length);
  elml_buf_free(&buffer);
  if (blob == NULL) {
    return 1;
  }
  if (elml_model_init(model, blob, length, ELML_DTYPE_F32) != ELML_OK) {
    free(blob);
    return 1;
  }
  *blob_out = blob;
  return 0;
}

int main(void)
{
  elml_model model;
  uint8_t *blob = NULL;
  CHECK_EQ_INT(load_softmax_model(&model, &blob), 0);
  if (blob == NULL) {
    TEST_END();
  }

  const size_t scratch_size = elml_scratch_bytes(&model, 1u);
  CHECK(scratch_size >= 2u * 2u * sizeof(float));
  uint8_t *scratch = (uint8_t *)aligned_alloc(ELML_TENSOR_ALIGN, scratch_size);
  CHECK(scratch != NULL);
  if (scratch == NULL) {
    free(blob);
    TEST_END();
  }

  float input[2] = {0.0f, 1.0f};
  float output[2] = {0.0f, 0.0f};
  CHECK_EQ_INT(elml_forward(&model, scratch, scratch_size, input, output),
               ELML_OK);
  CHECK_CLOSE(output[0], 0.26894143, 1e-5);
  CHECK_CLOSE(output[1], 0.73105858, 1e-5);
  CHECK_CLOSE(output[0] + output[1], 1.0, 1e-6);

  /* The same model and scratch must be reusable for every question. */
  float swapped[2] = {1.0f, 0.0f};
  CHECK_EQ_INT(elml_forward(&model, scratch, scratch_size, swapped, output),
               ELML_OK);
  CHECK_CLOSE(output[0], 0.73105858, 1e-5);
  CHECK_CLOSE(output[1], 0.26894143, 1e-5);

  float batched[4] = {0.0f, 1.0f, 1.0f, 0.0f};
  float batched_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  CHECK_EQ_INT(elml_forward_batch(&model, scratch, scratch_size, batched,
                                  batched_out, 2u),
               ELML_OK);
  CHECK_CLOSE(batched_out[0], 0.26894143, 1e-5);
  CHECK_CLOSE(batched_out[3], 0.26894143, 1e-5);

  CHECK_EQ_INT(elml_forward(&model, scratch, scratch_size - 1u, input, output),
               ELML_ERR_SCRATCH_TOO_SMALL);
  CHECK_EQ_INT(elml_forward(&model, scratch + 1, scratch_size, input, output),
               ELML_ERR_MISALIGNED);
  CHECK_EQ_INT(elml_forward(&model, scratch, scratch_size, NULL, output),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_forward(&model, scratch, scratch_size, input, NULL),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_forward(NULL, scratch, scratch_size, input, output),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_forward_batch(&model, scratch, scratch_size, batched,
                                  batched_out, 0u),
               ELML_OK);

  int8_t q_in[2] = {0, 0};
  int8_t q_out[2] = {0, 0};
  elml_quant quant = {0.0f, 0};
  CHECK_EQ_INT(elml_forward_q(&model, scratch, scratch_size, q_in, q_out,
                              &quant),
               ELML_ERR_DTYPE_MISMATCH);
  CHECK_EQ_INT(elml_forward_q(NULL, scratch, scratch_size, q_in, q_out, &quant),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_scratch_bytes(NULL, 1u), 0u);

  free(scratch);
  free(blob);
  TEST_END();
}
