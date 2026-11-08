#include <stddef.h>
#include <stdint.h>

#include "core/layer_vt.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "format/endian.h"
#include "kernels/kernels.h"

typedef struct elml_act_kind {
  elml_layer_id id;
  elml_activation act;
  const char *name;
} elml_act_kind;

static const elml_act_kind g_act_kinds[] = {
    {ELML_LAYER_ACT_RELU, ELML_ACT_RELU, "relu"},
    {ELML_LAYER_ACT_LEAKY_RELU, ELML_ACT_LEAKY_RELU, "leaky_relu"},
    {ELML_LAYER_ACT_TANH, ELML_ACT_TANH, "tanh"},
    {ELML_LAYER_ACT_SIGMOID, ELML_ACT_SIGMOID, "sigmoid"},
    {ELML_LAYER_ACT_SOFTMAX, ELML_ACT_SOFTMAX, "softmax"},
    {ELML_LAYER_ACT_LINEAR, ELML_ACT_LINEAR, "linear"},
};

static const elml_act_kind *elml_act_kind_find(elml_layer_id id)
{
  for (size_t i = 0u; i < ELML_ARRAY_COUNT(g_act_kinds); ++i) {
    if (g_act_kinds[i].id == id) {
      return &g_act_kinds[i];
    }
  }
  return nullptr;
}

static elml_status elml_activation_prepare(elml_layer *layer,
                                           const elml_layer_desc *desc,
                                           elml_dtype dtype)
{
  if (layer == nullptr || desc == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (dtype != ELML_DTYPE_F32) {
    return ELML_ERR_UNSUPPORTED_DTYPE;
  }
  if (desc->in.rank != 1u || desc->out.rank != 1u) {
    return ELML_ERR_BAD_SHAPE;
  }
  if (desc->in.dim[0] == 0u || desc->in.dim[0] != desc->out.dim[0]) {
    return ELML_ERR_BAD_SHAPE;
  }

  const elml_act_kind *kind = elml_act_kind_find(desc->type_id);
  if (kind == nullptr) {
    return ELML_ERR_UNSUPPORTED_LAYER_TYPE;
  }

  layer->fused_act = kind->act;
  layer->leaky_alpha = 0.01f;
  if (kind->act == ELML_ACT_LEAKY_RELU && desc->extra != nullptr &&
      desc->extra_bytes >= sizeof(float)) {
    layer->leaky_alpha = elml_le_f32(desc->extra);
  }

  return ELML_OK;
}

static elml_status elml_activation_forward(const elml_layer *layer,
                                           const elml_tensor *in,
                                           elml_tensor *out)
{
  if (layer == nullptr || in == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (in->dtype != ELML_DTYPE_F32 || out->dtype != ELML_DTYPE_F32) {
    return ELML_ERR_UNSUPPORTED_DTYPE;
  }
  if (in->data == nullptr || out->data == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (in->shape.rank != 1u || out->shape.rank != 1u ||
      in->shape.dim[0] != out->shape.dim[0]) {
    return ELML_ERR_BAD_SHAPE;
  }

  elml_kernel_activation_f32((const float *)in->data, (float *)out->data,
                             in->shape.dim[0], layer->fused_act,
                             layer->leaky_alpha);
  return ELML_OK;
}

#define ELML_DEFINE_ACT_VT(symbol, layer_id_value, display_name) \
  const elml_layer_vt symbol = {                                 \
      .id = (layer_id_value),                                    \
      .version = 1u,                                             \
      .name = (display_name),                                    \
      .prepare = elml_activation_prepare,                        \
      .forward = elml_activation_forward,                        \
  }

ELML_DEFINE_ACT_VT(elml_act_relu_layer_vt, ELML_LAYER_ACT_RELU, "relu");
ELML_DEFINE_ACT_VT(elml_act_leaky_relu_layer_vt, ELML_LAYER_ACT_LEAKY_RELU,
                   "leaky_relu");
ELML_DEFINE_ACT_VT(elml_act_tanh_layer_vt, ELML_LAYER_ACT_TANH, "tanh");
ELML_DEFINE_ACT_VT(elml_act_sigmoid_layer_vt, ELML_LAYER_ACT_SIGMOID,
                   "sigmoid");
ELML_DEFINE_ACT_VT(elml_act_softmax_layer_vt, ELML_LAYER_ACT_SOFTMAX,
                   "softmax");
ELML_DEFINE_ACT_VT(elml_act_linear_layer_vt, ELML_LAYER_ACT_LINEAR, "linear");
