/*
 * Bare-metal style inference demo.
 *
 * This file deliberately avoids malloc, stdio and floating-point file I/O.
 * The model blob is generated into model_blob.h by embed_model.py and lives in
 * flash. Scratch space is a static, 16-byte aligned array owned by the
 * application, exactly as required by elml_forward.
 */
#include <stddef.h>
#include <stdint.h>

#include "elml/model.h"
#include "elml/types.h"
#include "model_blob.h"

#define ELML_EXAMPLE_SCRATCH_BYTES 1024u

static uint8_t g_scratch[ELML_EXAMPLE_SCRATCH_BYTES]
    __attribute__((aligned(ELML_TENSOR_ALIGN)));

/*
 * Run one sample. Returns 0 on success, a negative elml_status value on
 * failure, and writes the predicted class index to *label.
 */
int elml_example_predict(const float *input, float *output, int *label)
{
  elml_model model;
  elml_status status =
      elml_model_init(&model, elml_example_model, elml_example_model_len,
                      ELML_DTYPE_F32);
  if (status != ELML_OK) {
    return (int)status;
  }
  if (elml_scratch_bytes(&model, 1u) > ELML_EXAMPLE_SCRATCH_BYTES) {
    return (int)ELML_ERR_SCRATCH_TOO_SMALL;
  }

  status = elml_forward(&model, g_scratch, ELML_EXAMPLE_SCRATCH_BYTES, input,
                        output);
  if (status != ELML_OK) {
    return (int)status;
  }

  if (label != NULL) {
    uint32_t best = 0u;
    const uint32_t count = model.output.dim[0];
    for (uint32_t i = 1u; i < count; ++i) {
      if (output[i] > output[best]) {
        best = i;
      }
    }
    *label = (int)best;
  }
  return 0;
}

/* Expose the metadata needed by the host-side checker without using stdio. */
int elml_example_label_count(void)
{
  elml_model model;
  if (elml_model_init(&model, elml_example_model, elml_example_model_len,
                      ELML_DTYPE_F32) != ELML_OK) {
    return 0;
  }
  return (int)elml_label_count(&model);
}
