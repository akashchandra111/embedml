#include "elml/model.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ckdint_compat.h"
#include "elml/arena.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "format/format_internal.h"

static size_t elml_shape_count(const elml_shape *shape)
{
  if (shape == nullptr || shape->rank == 0u) {
    return 0u;
  }
  size_t count = 1u;
  for (uint8_t i = 0u; i < shape->rank; ++i) {
    size_t next;
    if (ckd_mul(&next, count, (size_t)shape->dim[i])) {
      return 0u;
    }
    count = next;
  }
  return count;
}

static bool elml_shape_equal(const elml_shape *a, const elml_shape *b)
{
  if (a->rank != b->rank) {
    return false;
  }
  for (uint8_t i = 0u; i < a->rank; ++i) {
    if (a->dim[i] != b->dim[i]) {
      return false;
    }
  }
  return true;
}

static bool elml_activation_is_fusable(elml_layer_id id)
{
  return id == ELML_LAYER_ACT_RELU || id == ELML_LAYER_ACT_LEAKY_RELU ||
         id == ELML_LAYER_ACT_TANH || id == ELML_LAYER_ACT_SIGMOID ||
         id == ELML_LAYER_ACT_LINEAR;
}

static elml_status elml_resolve_layers(elml_model *model)
{
  for (uint16_t i = 0u; i < model->layer_count; ++i) {
    elml_layer_desc *desc = &model->descs[i];
    elml_layer *layer = &model->layers[i];

    const elml_layer_vt *vt = elml_layer_find(desc->type_id);
    if (vt == nullptr) {
      return ELML_ERR_UNSUPPORTED_LAYER_TYPE;
    }
    if (desc->type_version != vt->version) {
      return ELML_ERR_UNSUPPORTED_MINOR;
    }

    layer->vt = vt;
    layer->desc = desc;
    layer->params = nullptr;
    if (desc->param_off != UINT32_MAX) {
      layer->params = model->parm + desc->param_off;
      if (((uintptr_t)layer->params % ELML_PARAM_ALIGN) != 0u) {
        return ELML_ERR_MISALIGNED;
      }
    }
    layer->flags = desc->flags;

    const elml_status status = vt->prepare(layer, desc, model->dtype);
    if (status != ELML_OK) {
      return status;
    }
  }

  for (uint16_t i = 0u; i + 1u < model->layer_count; ++i) {
    if (model->descs[i].type_id != ELML_LAYER_DENSE ||
        !elml_activation_is_fusable(model->descs[i + 1u].type_id)) {
      continue;
    }
    if (!elml_shape_equal(&model->descs[i].out, &model->descs[i + 1u].in) ||
        !elml_shape_equal(&model->descs[i + 1u].in,
                          &model->descs[i + 1u].out)) {
      return ELML_ERR_BAD_SHAPE;
    }
    model->layers[i].fused_act = model->layers[i + 1u].fused_act;
    model->layers[i].leaky_alpha = model->layers[i + 1u].leaky_alpha;
    model->layers[i + 1u].flags |= ELML_LAYER_FLAG_SKIPPED;
  }

  return ELML_OK;
}

static elml_status elml_compute_scratch(elml_model *model)
{
  size_t max_elems = elml_shape_count(&model->input);
  if (max_elems == 0u) {
    return ELML_ERR_BAD_SHAPE;
  }

  for (uint16_t i = 0u; i < model->layer_count; ++i) {
    const size_t elems = elml_shape_count(&model->descs[i].out);
    if (elems == 0u) {
      return ELML_ERR_BAD_SHAPE;
    }
    if (elems > max_elems) {
      max_elems = elems;
    }
  }

  size_t max_bytes;
  if (ckd_mul(&max_bytes, max_elems, sizeof(float))) {
    return ELML_ERR_BAD_SHAPE;
  }
  const size_t stride = elml_align_up(max_bytes, ELML_TENSOR_ALIGN);
  if (stride == 0u) {
    return ELML_ERR_BAD_SHAPE;
  }
  size_t scratch;
  if (ckd_mul(&scratch, stride, 2u)) {
    return ELML_ERR_BAD_SHAPE;
  }
  model->max_activation_bytes = max_bytes;
  model->scratch_bytes = scratch;
  return ELML_OK;
}

elml_status elml_model_init(elml_model *model, const void *blob,
                            size_t blob_len, elml_dtype expected_dtype)
{
  if (model == nullptr || blob == nullptr || blob_len == 0u) {
    return ELML_ERR_INVALID_ARG;
  }
  if (expected_dtype != ELML_DTYPE_F32 && expected_dtype != ELML_DTYPE_INT8) {
    return ELML_ERR_UNSUPPORTED_DTYPE;
  }
  if (((uintptr_t)blob % ELML_PARAM_ALIGN) != 0u) {
    return ELML_ERR_MISALIGNED;
  }

  memset(model, 0, sizeof *model);

  elml_format format;
  elml_status status = elml_format_parse(blob, blob_len, &format);
  if (status != ELML_OK) {
    return status;
  }
  if (format.dtype != expected_dtype) {
    return ELML_ERR_DTYPE_MISMATCH;
  }
  if (format.dtype != ELML_DTYPE_F32) {
    /* The int8 runtime is not part of this build yet. */
    return ELML_ERR_UNSUPPORTED_DTYPE;
  }

  model->blob = (const uint8_t *)blob;
  model->blob_len = blob_len;
  model->dtype = format.dtype;
  model->input = format.input;
  model->output = format.output;
  model->layer_count = format.layer_count;
  model->file_flags = format.file_flags;
  model->parm = format.parm;
  model->parm_len = format.parm_len;
  model->qnt = format.qnt;
  model->qnt_len = format.qnt_len;
  model->meta.data = format.meta;
  model->meta.len = format.meta_len;

  memcpy(model->descs, format.descs,
         (size_t)format.layer_count * sizeof format.descs[0]);

  model->norm.input_mode = (elml_norm_mode)format.norm_input_mode;
  model->norm.output_mode = (elml_norm_mode)format.norm_output_mode;
  model->norm.input_count = format.norm_input_count;
  model->norm.output_count = format.norm_output_count;
  model->norm.input_params = format.norm_input_params;
  model->norm.output_params = format.norm_output_params;

  status = elml_resolve_layers(model);
  if (status != ELML_OK) {
    return status;
  }

  return elml_compute_scratch(model);
}

elml_status elml_model_dtype(const void *blob, size_t blob_len,
                             elml_dtype *dtype)
{
  return elml_format_probe(blob, blob_len, dtype);
}

size_t elml_scratch_bytes(const elml_model *model, uint32_t batch)
{
  (void)batch; /* forward_batch reuses one sample-sized workspace */
  return (model == nullptr) ? 0u : model->scratch_bytes;
}

static float elml_normalize_value(float value, elml_norm_mode mode,
                                  const float *params)
{
  switch (mode) {
  case ELML_NORM_STANDARD: {
    const float stddev = params[1];
    return (stddev == 0.0f) ? value : ((value - params[0]) / stddev);
  }
  case ELML_NORM_MINMAX: {
    const float range = params[1] - params[0];
    return (range == 0.0f) ? 0.0f
                           : (((value - params[0]) / range) * 2.0f - 1.0f);
  }
  case ELML_NORM_NONE:
  default:
    return value;
  }
}

static float elml_denormalize_value(float value, elml_norm_mode mode,
                                    const float *params)
{
  switch (mode) {
  case ELML_NORM_STANDARD:
    return value * params[1] + params[0];
  case ELML_NORM_MINMAX:
    return ((value + 1.0f) * 0.5f) * (params[1] - params[0]) + params[0];
  case ELML_NORM_NONE:
  default:
    return value;
  }
}

elml_status elml_forward(const elml_model *model, void *scratch,
                         size_t scratch_len, const float *input, float *output)
{
  if (model == nullptr || scratch == nullptr || input == nullptr ||
      output == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (model->dtype != ELML_DTYPE_F32) {
    return ELML_ERR_UNSUPPORTED_DTYPE;
  }
  if (((uintptr_t)scratch % ELML_TENSOR_ALIGN) != 0u) {
    return ELML_ERR_MISALIGNED;
  }
  if (scratch_len < model->scratch_bytes) {
    return ELML_ERR_SCRATCH_TOO_SMALL;
  }

  const size_t input_count = elml_shape_count(&model->input);
  const size_t output_count = elml_shape_count(&model->output);
  if (input_count == 0u || output_count == 0u) {
    return ELML_ERR_BAD_SHAPE;
  }

  const size_t stride =
      elml_align_up(model->max_activation_bytes, ELML_TENSOR_ALIGN);
  if (stride == 0u || stride * 2u > scratch_len) {
    return ELML_ERR_SCRATCH_TOO_SMALL;
  }

  uint8_t *buffer_a = (uint8_t *)scratch;
  uint8_t *buffer_b = buffer_a + stride;

  if (model->norm.input_mode == ELML_NORM_NONE ||
      model->norm.input_params == nullptr) {
    for (size_t i = 0u; i < input_count; ++i) {
      ((float *)buffer_a)[i] = input[i];
    }
  } else {
    for (size_t i = 0u; i < input_count; ++i) {
      ((float *)buffer_a)[i] = elml_normalize_value(
          input[i], model->norm.input_mode, model->norm.input_params + 2u * i);
    }
  }

  void *current = buffer_a;
  void *next = buffer_b;

  for (uint16_t i = 0u; i < model->layer_count; ++i) {
    const elml_layer *layer = &model->layers[i];
    if ((layer->flags & ELML_LAYER_FLAG_SKIPPED) != 0u) {
      continue;
    }
    if (layer->vt == nullptr || layer->vt->forward == nullptr) {
      return ELML_ERR_INTERNAL;
    }

    elml_tensor in_tensor = {
        .data = current,
        .dtype = ELML_DTYPE_F32,
        .shape = layer->desc->in,
        .q = {.scale = 0.0f, .zero_point = 0},
    };
    elml_tensor out_tensor = {
        .data = next,
        .dtype = ELML_DTYPE_F32,
        .shape = layer->desc->out,
        .q = {.scale = 0.0f, .zero_point = 0},
    };

    const elml_status status = layer->vt->forward(layer, &in_tensor, &out_tensor);
    if (status != ELML_OK) {
      return status;
    }

    void *swap = current;
    current = next;
    next = swap;
  }

  const float *result = (const float *)current;
  if (model->norm.output_mode == ELML_NORM_NONE ||
      model->norm.output_params == nullptr) {
    for (size_t i = 0u; i < output_count; ++i) {
      output[i] = result[i];
    }
  } else {
    for (size_t i = 0u; i < output_count; ++i) {
      output[i] = elml_denormalize_value(
          result[i], model->norm.output_mode,
          model->norm.output_params + 2u * i);
    }
  }

  return ELML_OK;
}

elml_status elml_forward_batch(const elml_model *model, void *scratch,
                               size_t scratch_len, const float *input,
                               float *output, uint32_t batch)
{
  if (model == nullptr || scratch == nullptr || input == nullptr ||
      output == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  const size_t input_count = elml_shape_count(&model->input);
  const size_t output_count = elml_shape_count(&model->output);
  if (input_count == 0u || output_count == 0u) {
    return ELML_ERR_BAD_SHAPE;
  }

  for (uint32_t sample = 0u; sample < batch; ++sample) {
    const elml_status status =
        elml_forward(model, scratch, scratch_len,
                     input + (size_t)sample * input_count,
                     output + (size_t)sample * output_count);
    if (status != ELML_OK) {
      return status;
    }
  }
  return ELML_OK;
}

elml_status elml_forward_q(const elml_model *model, void *scratch,
                           size_t scratch_len, const int8_t *input,
                           int8_t *output, elml_quant *output_quant)
{
  (void)scratch;
  (void)scratch_len;
  (void)input;
  (void)output;
  (void)output_quant;
  if (model == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (model->dtype != ELML_DTYPE_INT8) {
    return ELML_ERR_DTYPE_MISMATCH;
  }
  return ELML_ERR_UNSUPPORTED_DTYPE;
}

static elml_status elml_meta_find(const elml_meta *meta, const char *key,
                                  const uint8_t **value, uint32_t *value_len)
{
  if (meta == nullptr || key == nullptr || value == nullptr ||
      value_len == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (meta->data == nullptr || meta->len == 0u) {
    return ELML_ERR_NOT_FOUND;
  }

  const size_t key_len = strlen(key);
  size_t offset = 0u;

  while (offset < (size_t)meta->len) {
    if ((size_t)meta->len - offset < 4u) {
      return ELML_ERR_TRUNCATED;
    }
    const uint16_t entry_key_len =
        (uint16_t)((uint16_t)meta->data[offset] |
                   ((uint16_t)meta->data[offset + 1u] << 8));
    const uint16_t entry_value_len =
        (uint16_t)((uint16_t)meta->data[offset + 2u] |
                   ((uint16_t)meta->data[offset + 3u] << 8));
    offset += 4u;

    if ((size_t)meta->len - offset < (size_t)entry_key_len + entry_value_len) {
      return ELML_ERR_TRUNCATED;
    }
    if ((size_t)entry_key_len == key_len &&
        memcmp(meta->data + offset, key, key_len) == 0) {
      *value = meta->data + offset + entry_key_len;
      *value_len = entry_value_len;
      return ELML_OK;
    }
    offset += (size_t)entry_key_len + entry_value_len;
  }

  return ELML_ERR_NOT_FOUND;
}

elml_status elml_meta_get(const elml_model *model, const char *key, char *buf,
                          size_t cap, size_t *out_len)
{
  if (model == nullptr || key == nullptr || buf == nullptr || cap == 0u ||
      out_len == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  const uint8_t *value = nullptr;
  uint32_t value_len = 0u;
  elml_status status = elml_meta_find(&model->meta, key, &value, &value_len);
  if (status != ELML_OK) {
    return status;
  }
  *out_len = (size_t)value_len;
  if ((size_t)value_len + 1u > cap) {
    return ELML_ERR_BUFFER_TOO_SMALL;
  }
  memcpy(buf, value, (size_t)value_len);
  buf[value_len] = '\0';
  return ELML_OK;
}

uint32_t elml_label_count(const elml_model *model)
{
  if (model == nullptr) {
    return 0u;
  }
  const uint8_t *value = nullptr;
  uint32_t value_len = 0u;
  if (elml_meta_find(&model->meta, "labels", &value, &value_len) != ELML_OK ||
      value_len == 0u) {
    return 0u;
  }

  uint32_t count = 1u;
  for (uint32_t i = 0u; i < value_len; ++i) {
    if (value[i] == (uint8_t)',') {
      count++;
    }
  }
  return count;
}

elml_status elml_label_get(const elml_model *model, uint32_t index, char *buf,
                           size_t cap, size_t *out_len)
{
  if (model == nullptr || buf == nullptr || cap == 0u || out_len == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  const uint8_t *value = nullptr;
  uint32_t value_len = 0u;
  elml_status status =
      elml_meta_find(&model->meta, "labels", &value, &value_len);
  if (status != ELML_OK) {
    return status;
  }
  if (value_len == 0u) {
    return ELML_ERR_NOT_FOUND;
  }

  uint32_t current = 0u;
  uint32_t start = 0u;
  for (uint32_t i = 0u; i <= value_len; ++i) {
    if (i == value_len || value[i] == (uint8_t)',') {
      if (current == index) {
        const size_t len = (size_t)(i - start);
        *out_len = len;
        if (len + 1u > cap) {
          return ELML_ERR_BUFFER_TOO_SMALL;
        }
        memcpy(buf, value + start, len);
        buf[len] = '\0';
        return ELML_OK;
      }
      current++;
      start = i + 1u;
    }
  }

  return ELML_ERR_NOT_FOUND;
}
