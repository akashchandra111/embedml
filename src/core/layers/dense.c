#include <stddef.h>
#include <stdint.h>

#include "ckdint_compat.h"
#include "core/layer_vt.h"
#include "elml/arena.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "kernels/kernels.h"

static elml_status elml_dense_prepare(elml_layer *layer,
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
  if (desc->in.dim[0] == 0u || desc->out.dim[0] == 0u) {
    return ELML_ERR_BAD_SHAPE;
  }
  if (desc->param_off == UINT32_MAX) {
    return ELML_ERR_BAD_PARAM;
  }

  const size_t in_dim = desc->in.dim[0];
  const size_t out_dim = desc->out.dim[0];

  size_t weight_bytes;
  if (ckd_mul(&weight_bytes, in_dim, out_dim)) {
    return ELML_ERR_BAD_SHAPE;
  }
  if (ckd_mul(&weight_bytes, weight_bytes, sizeof(float))) {
    return ELML_ERR_BAD_SHAPE;
  }

  const size_t weight_stride = elml_align_up(weight_bytes, ELML_PARAM_ALIGN);
  if (weight_stride == 0u) {
    return ELML_ERR_BAD_PARAM;
  }

  size_t bias_bytes;
  if (ckd_mul(&bias_bytes, out_dim, sizeof(float))) {
    return ELML_ERR_BAD_SHAPE;
  }
  const size_t bias_stride = elml_align_up(bias_bytes, ELML_PARAM_ALIGN);
  if (bias_stride == 0u) {
    return ELML_ERR_BAD_PARAM;
  }

  size_t required;
  if (ckd_add(&required, weight_stride, bias_stride)) {
    return ELML_ERR_BAD_PARAM;
  }
  if ((size_t)desc->param_bytes < required) {
    return ELML_ERR_BAD_PARAM;
  }

  layer->weights = layer->params;
  layer->biases = (const uint8_t *)layer->params + weight_stride;
  layer->fused_act = ELML_ACT_NONE;
  return ELML_OK;
}

static elml_status elml_dense_forward(const elml_layer *layer,
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
  if (layer->weights == nullptr || layer->biases == nullptr) {
    return ELML_ERR_BAD_PARAM;
  }

  const elml_layer_desc *desc = layer->desc;
  if (desc->in.rank != 1u || desc->out.rank != 1u) {
    return ELML_ERR_BAD_SHAPE;
  }
  if (in->shape.dim[0] != desc->in.dim[0] ||
      out->shape.dim[0] != desc->out.dim[0]) {
    return ELML_ERR_BAD_SHAPE;
  }

  elml_kernel_dense_f32((const float *)layer->weights,
                        (const float *)layer->biases, (const float *)in->data,
                        (float *)out->data, desc->in.dim[0], desc->out.dim[0]);
  if (layer->fused_act != ELML_ACT_NONE) {
    elml_kernel_activation_f32((const float *)out->data, (float *)out->data,
                               desc->out.dim[0], layer->fused_act,
                               layer->leaky_alpha);
  }
  return ELML_OK;
}

const elml_layer_vt elml_dense_layer_vt = {
    .id = ELML_LAYER_DENSE,
    .version = 1u,
    .name = "dense",
    .prepare = elml_dense_prepare,
    .forward = elml_dense_forward,
};
