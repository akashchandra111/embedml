/*
 * Extensible layer registry.
 *
 * A layer type is identified by a permanent 16-bit id. The id is stored in the
 * model file and is never reused. Implementations are selected through a small
 * vtable so new layer types do not require changes to the model runner.
 */
#ifndef ELML_REGISTRY_H
#define ELML_REGISTRY_H

#include <stdint.h>

#include "elml/compat.h"
#include "elml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum elml_layer_id {
  ELML_LAYER_DENSE = 0x0001,
  ELML_LAYER_CONV1D = 0x0002, /* reserved for a future version */

  ELML_LAYER_ACT_RELU = 0x0010,
  ELML_LAYER_ACT_LEAKY_RELU = 0x0011,
  ELML_LAYER_ACT_TANH = 0x0012,
  ELML_LAYER_ACT_SIGMOID = 0x0013,
  ELML_LAYER_ACT_SOFTMAX = 0x0014,
  ELML_LAYER_ACT_LINEAR = 0x0015
} elml_layer_id;

typedef enum elml_layer_flags {
  ELML_LAYER_FLAG_TRAINABLE = 0x0001,
  ELML_LAYER_FLAG_FUSED_PREV = 0x0002,
  ELML_LAYER_FLAG_HAS_BIAS = 0x0004,
  /* Runtime-only flag: the layer was folded into its predecessor. */
  ELML_LAYER_FLAG_SKIPPED = 0x0100
} elml_layer_flags;

typedef enum elml_activation {
  ELML_ACT_NONE = 0,
  ELML_ACT_RELU,
  ELML_ACT_LEAKY_RELU,
  ELML_ACT_TANH,
  ELML_ACT_SIGMOID,
  ELML_ACT_SOFTMAX,
  ELML_ACT_LINEAR
} elml_activation;

typedef struct elml_layer_desc {
  elml_layer_id type_id;
  uint16_t type_version;
  uint32_t flags;
  uint32_t param_off;   /* relative to the PARM payload, UINT32_MAX if absent */
  uint32_t param_bytes;
  elml_shape in;
  elml_shape out;
  const uint8_t *extra;
  uint32_t extra_bytes;
} elml_layer_desc;

typedef struct elml_layer {
  const struct elml_layer_vt *vt;
  const elml_layer_desc *desc;
  const void *params; /* PARM payload + desc->param_off */
  const void *weights;
  const void *biases;
  elml_activation fused_act;
  float leaky_alpha;
  uint32_t flags;
} elml_layer;

typedef struct elml_layer_vt {
  elml_layer_id id;
  uint16_t version;
  const char *name;
  elml_status (*prepare)(elml_layer *layer, const elml_layer_desc *desc,
                         elml_dtype dtype);
  elml_status (*forward)(const elml_layer *layer, const elml_tensor *in,
                         elml_tensor *out);
} elml_layer_vt;

/* Returns NULL when the type is not linked into this build. */
const elml_layer_vt *elml_layer_find(elml_layer_id id);

/*
 * Host builds may register additional layer types before loading a model.
 * Static-registry MCU builds compile this call to ELML_ERR_UNSUPPORTED_LAYER_TYPE.
 */
elml_status elml_registry_add(const elml_layer_vt *vt);

#ifdef __cplusplus
}
#endif

#endif /* ELML_REGISTRY_H */
