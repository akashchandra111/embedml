#include "format/write.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ckdint_compat.h"
#include "elml/arena.h"
#include "elml/format.h"
#include "elml/model.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "format/crc32.h"
#include "format/endian.h"

#define ELML_WRITE_ALIGN 8u
#define ELML_WRITE_PARAM_ALIGN ELML_PARAM_ALIGN

static elml_status elml_buf_reserve(elml_buf *buf, size_t extra)
{
  if (buf == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  size_t needed;
  if (ckd_add(&needed, buf->len, extra)) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  if (needed <= buf->cap) {
    return ELML_OK;
  }

  size_t capacity = (buf->cap == 0u) ? 1024u : buf->cap;
  while (capacity < needed) {
    size_t next;
    if (ckd_mul(&next, capacity, 2u)) {
      capacity = needed;
      break;
    }
    capacity = next;
  }

  uint8_t *data = (uint8_t *)realloc(buf->data, capacity);
  if (data == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  buf->data = data;
  buf->cap = capacity;
  return ELML_OK;
}

static elml_status elml_buf_append(elml_buf *buf, const void *data, size_t len)
{
  elml_status status = elml_buf_reserve(buf, len);
  if (status != ELML_OK) {
    return status;
  }
  if (len > 0u) {
    memcpy(buf->data + buf->len, data, len);
  }
  buf->len += len;
  return ELML_OK;
}

static elml_status elml_buf_zero(elml_buf *buf, size_t len)
{
  elml_status status = elml_buf_reserve(buf, len);
  if (status != ELML_OK) {
    return status;
  }
  memset(buf->data + buf->len, 0, len);
  buf->len += len;
  return ELML_OK;
}

static elml_status elml_buf_pad(elml_buf *buf, size_t alignment)
{
  const size_t aligned = elml_align_up(buf->len, alignment);
  if (aligned == 0u && buf->len != 0u) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  return elml_buf_zero(buf, aligned - buf->len);
}

void elml_buf_free(elml_buf *buf)
{
  if (buf == nullptr) {
    return;
  }
  free(buf->data);
  buf->data = nullptr;
  buf->len = 0u;
  buf->cap = 0u;
}

static uint16_t elml_layer_id_for_activation(elml_activation act)
{
  switch (act) {
  case ELML_ACT_RELU:
    return (uint16_t)ELML_LAYER_ACT_RELU;
  case ELML_ACT_LEAKY_RELU:
    return (uint16_t)ELML_LAYER_ACT_LEAKY_RELU;
  case ELML_ACT_TANH:
    return (uint16_t)ELML_LAYER_ACT_TANH;
  case ELML_ACT_SIGMOID:
    return (uint16_t)ELML_LAYER_ACT_SIGMOID;
  case ELML_ACT_SOFTMAX:
    return (uint16_t)ELML_LAYER_ACT_SOFTMAX;
  case ELML_ACT_LINEAR:
  case ELML_ACT_NONE:
  default:
    return (uint16_t)ELML_LAYER_ACT_LINEAR;
  }
}

static elml_status elml_validate_spec(const elml_write_spec *spec)
{
  if (spec == nullptr || spec->layers == nullptr || spec->layer_count == 0u ||
      spec->input_dim == 0u || spec->output_dim == 0u) {
    return ELML_ERR_INVALID_ARG;
  }
  if ((size_t)spec->layer_count * 2u > ELML_MAX_LAYERS) {
    return ELML_ERR_BAD_SHAPE;
  }

  uint32_t previous_out = spec->input_dim;
  for (uint16_t i = 0u; i < spec->layer_count; ++i) {
    const elml_write_layer *layer = &spec->layers[i];
    if (layer->in_dim == 0u || layer->out_dim == 0u ||
        layer->in_dim != previous_out || layer->weights == nullptr ||
        layer->biases == nullptr) {
      return ELML_ERR_BAD_SHAPE;
    }
    previous_out = layer->out_dim;
  }
  if (previous_out != spec->output_dim) {
    return ELML_ERR_BAD_SHAPE;
  }

  if (spec->norm.input_mode != 0u &&
      (spec->norm.input_params == nullptr ||
       spec->norm.input_mode > (uint8_t)ELML_NORM_MINMAX)) {
    return ELML_ERR_INVALID_ARG;
  }
  if (spec->norm.output_mode != 0u &&
      (spec->norm.output_params == nullptr ||
       spec->norm.output_mode > (uint8_t)ELML_NORM_MINMAX)) {
    return ELML_ERR_INVALID_ARG;
  }
  return ELML_OK;
}

static elml_status elml_buf_pad_to(elml_buf *buf, size_t base,
                                    size_t alignment)
{
  const size_t target = elml_align_up(base, alignment);
  if (target == 0u && base != 0u) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  return elml_buf_zero(buf, target - base);
}

static elml_status elml_write_parm(const elml_write_spec *spec, elml_buf *parm,
                                   uint32_t *offsets, uint32_t *sizes)
{
  const size_t absolute_payload = ELML_HEADER_BYTES + ELML_CHUNK_HEADER_BYTES;
  const size_t leading =
      elml_align_up(absolute_payload, ELML_WRITE_PARAM_ALIGN) -
      absolute_payload;
  elml_status status = elml_buf_zero(parm, leading);
  if (status != ELML_OK) {
    return status;
  }

  for (uint16_t i = 0u; i < spec->layer_count; ++i) {
    const elml_write_layer *layer = &spec->layers[i];
    const size_t weight_bytes =
        (size_t)layer->out_dim * (size_t)layer->in_dim * sizeof(float);
    const size_t bias_bytes = (size_t)layer->out_dim * sizeof(float);

    if (parm->len > UINT32_MAX) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    const size_t layer_start = parm->len;
    offsets[i] = (uint32_t)layer_start;

    status = elml_buf_append(parm, layer->weights, weight_bytes);
    if (status != ELML_OK) {
      return status;
    }
    status = elml_buf_pad_to(parm, parm->len - layer_start,
                             ELML_WRITE_PARAM_ALIGN);
    if (status != ELML_OK) {
      return status;
    }

    const size_t bias_start = parm->len;
    status = elml_buf_append(parm, layer->biases, bias_bytes);
    if (status != ELML_OK) {
      return status;
    }
    status = elml_buf_pad_to(parm, parm->len - bias_start,
                             ELML_WRITE_PARAM_ALIGN);
    if (status != ELML_OK) {
      return status;
    }

    const size_t total = parm->len - layer_start;
    if (total > UINT32_MAX) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    sizes[i] = (uint32_t)total;
  }

  return ELML_OK;
}

static elml_status elml_write_conf(const elml_write_spec *spec,
                                   const uint32_t *offsets,
                                   const uint32_t *sizes, elml_buf *conf)
{
  const uint32_t chain_count = (uint32_t)spec->layer_count * 2u;

  uint8_t header[8] = {0};
  elml_store_le32(header, chain_count);
  header[4] = 1u; /* input rank */
  header[5] = 1u; /* output rank */
  elml_status status = elml_buf_append(conf, header, sizeof header);
  if (status != ELML_OK) {
    return status;
  }

  uint8_t dim[4];
  elml_store_le32(dim, spec->input_dim);
  status = elml_buf_append(conf, dim, sizeof dim);
  if (status != ELML_OK) {
    return status;
  }
  elml_store_le32(dim, spec->output_dim);
  status = elml_buf_append(conf, dim, sizeof dim);
  if (status != ELML_OK) {
    return status;
  }

  for (uint16_t i = 0u; i < spec->layer_count; ++i) {
    const elml_write_layer *layer = &spec->layers[i];

    uint8_t dense[16] = {0};
    elml_store_le16(dense, (uint16_t)ELML_LAYER_DENSE);
    elml_store_le16(dense + 2, 1u);
    elml_store_le32(dense + 4,
                    ELML_LAYER_FLAG_TRAINABLE | ELML_LAYER_FLAG_HAS_BIAS);
    elml_store_le32(dense + 8, offsets[i]);
    elml_store_le32(dense + 12, sizes[i]);
    status = elml_buf_append(conf, dense, sizeof dense);
    if (status != ELML_OK) {
      return status;
    }

    const uint8_t ranks[2] = {1u, 1u};
    status = elml_buf_append(conf, ranks, sizeof ranks);
    if (status != ELML_OK) {
      return status;
    }
    uint8_t in_dim[4];
    uint8_t out_dim[4];
    elml_store_le32(in_dim, layer->in_dim);
    elml_store_le32(out_dim, layer->out_dim);
    status = elml_buf_append(conf, in_dim, sizeof in_dim);
    if (status != ELML_OK) {
      return status;
    }
    status = elml_buf_append(conf, out_dim, sizeof out_dim);
    if (status != ELML_OK) {
      return status;
    }
    const uint8_t no_extra[4] = {0};
    status = elml_buf_append(conf, no_extra, sizeof no_extra);
    if (status != ELML_OK) {
      return status;
    }

    uint8_t act[16] = {0};
    elml_store_le16(act, elml_layer_id_for_activation(layer->act));
    elml_store_le16(act + 2, 1u);
    elml_store_le32(act + 4, 0u);
    elml_store_le32(act + 8, UINT32_MAX);
    elml_store_le32(act + 12, 0u);
    status = elml_buf_append(conf, act, sizeof act);
    if (status != ELML_OK) {
      return status;
    }
    status = elml_buf_append(conf, ranks, sizeof ranks);
    if (status != ELML_OK) {
      return status;
    }
    /* An activation layer is shape-preserving: in == out. */
    status = elml_buf_append(conf, out_dim, sizeof out_dim);
    if (status != ELML_OK) {
      return status;
    }
    status = elml_buf_append(conf, out_dim, sizeof out_dim);
    if (status != ELML_OK) {
      return status;
    }

    if (layer->act == ELML_ACT_LEAKY_RELU) {
      const uint8_t extra_len[4] = {4u, 0u, 0u, 0u};
      status = elml_buf_append(conf, extra_len, sizeof extra_len);
      if (status != ELML_OK) {
        return status;
      }
      uint8_t alpha[4];
      elml_store_le_f32(alpha, layer->leaky_alpha);
      status = elml_buf_append(conf, alpha, sizeof alpha);
      if (status != ELML_OK) {
        return status;
      }
    } else {
      const uint8_t no_extra_act[4] = {0};
      status = elml_buf_append(conf, no_extra_act, sizeof no_extra_act);
      if (status != ELML_OK) {
        return status;
      }
    }
  }

  return ELML_OK;
}

static elml_status elml_buf_zero_floats(elml_buf *buf, uint32_t count)
{
  size_t bytes;
  if (ckd_mul(&bytes, (size_t)count, 2u * sizeof(float))) {
    return ELML_ERR_BAD_SHAPE;
  }
  return elml_buf_zero(buf, bytes);
}

static elml_status elml_write_norm_chunk(const elml_write_spec *spec, elml_buf *norm)
{
  if (spec->norm.input_mode == 0u && spec->norm.output_mode == 0u) {
    return ELML_OK;
  }

  uint8_t header[8] = {0};
  header[0] = spec->norm.input_mode;
  header[1] = spec->norm.output_mode;
  elml_store_le32(header + 4, spec->input_dim);
  elml_status status = elml_buf_append(norm, header, sizeof header);
  if (status != ELML_OK) {
    return status;
  }
  if (spec->norm.input_mode != 0u) {
    size_t bytes;
    if (ckd_mul(&bytes, (size_t)spec->input_dim, 2u * sizeof(float))) {
      return ELML_ERR_BAD_SHAPE;
    }
    status = elml_buf_append(norm, spec->norm.input_params, bytes);
  } else {
    status = elml_buf_zero_floats(norm, spec->input_dim);
  }
  if (status != ELML_OK) {
    return status;
  }

  uint8_t output_count[4];
  elml_store_le32(output_count, spec->output_dim);
  status = elml_buf_append(norm, output_count, sizeof output_count);
  if (status != ELML_OK) {
    return status;
  }
  if (spec->norm.output_mode != 0u) {
    size_t bytes;
    if (ckd_mul(&bytes, (size_t)spec->output_dim, 2u * sizeof(float))) {
      return ELML_ERR_BAD_SHAPE;
    }
    return elml_buf_append(norm, spec->norm.output_params, bytes);
  }
  return elml_buf_zero_floats(norm, spec->output_dim);
}

static elml_status elml_write_meta_entry(elml_buf *meta, const char *key,
                                         const char *value)
{
  const size_t key_len = strlen(key);
  const size_t value_len = strlen(value);
  if (key_len > UINT16_MAX || value_len > UINT16_MAX) {
    return ELML_ERR_INVALID_ARG;
  }

  uint8_t header[4];
  elml_store_le16(header, (uint16_t)key_len);
  elml_store_le16(header + 2, (uint16_t)value_len);
  elml_status status = elml_buf_append(meta, header, sizeof header);
  if (status != ELML_OK) {
    return status;
  }
  status = elml_buf_append(meta, key, key_len);
  if (status != ELML_OK) {
    return status;
  }
  return elml_buf_append(meta, value, value_len);
}

static elml_status elml_write_meta_chunk(const elml_write_spec *spec, elml_buf *meta)
{
  elml_status status =
      elml_write_meta_entry(meta, "producer", "EmbedLML 0.1.0");
  if (status != ELML_OK) {
    return status;
  }
  for (uint32_t i = 0u; i < spec->meta_count; ++i) {
    if (spec->meta == nullptr || spec->meta[i].key == nullptr ||
        spec->meta[i].value == nullptr) {
      return ELML_ERR_INVALID_ARG;
    }
    status = elml_write_meta_entry(meta, spec->meta[i].key,
                                   spec->meta[i].value);
    if (status != ELML_OK) {
      return status;
    }
  }
  return ELML_OK;
}

static elml_status elml_append_chunk(elml_buf *file, uint32_t type,
                                     uint16_t flags, const elml_buf *payload)
{
  elml_status status = elml_buf_pad(file, ELML_WRITE_ALIGN);
  if (status != ELML_OK) {
    return status;
  }
  if (payload->len > UINT32_MAX) {
    return ELML_ERR_OUT_OF_MEMORY;
  }

  uint8_t header[ELML_CHUNK_HEADER_BYTES] = {0};
  elml_store_le32(header, type);
  elml_store_le16(header + 4, 1u);
  elml_store_le16(header + 6, flags);
  elml_store_le32(header + 8, (uint32_t)payload->len);
  elml_store_le32(header + 12, elml_crc32(payload->data, payload->len, 0u));

  status = elml_buf_append(file, header, sizeof header);
  if (status != ELML_OK) {
    return status;
  }
  status = elml_buf_append(file, payload->data, payload->len);
  if (status != ELML_OK) {
    return status;
  }
  return elml_buf_pad(file, ELML_WRITE_ALIGN);
}

elml_status elml_model_write(const elml_write_spec *spec, elml_buf *out)
{
  if (out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  out->data = nullptr;
  out->len = 0u;
  out->cap = 0u;

  elml_status status = elml_validate_spec(spec);
  if (status != ELML_OK) {
    return status;
  }

  elml_buf parm = {0};
  elml_buf conf = {0};
  elml_buf norm = {0};
  elml_buf meta = {0};
  uint32_t offsets[ELML_MAX_LAYERS] = {0};
  uint32_t sizes[ELML_MAX_LAYERS] = {0};

  status = elml_write_parm(spec, &parm, offsets, sizes);
  if (status != ELML_OK) {
    goto cleanup;
  }
  status = elml_write_conf(spec, offsets, sizes, &conf);
  if (status != ELML_OK) {
    goto cleanup;
  }
  status = elml_write_norm_chunk(spec, &norm);
  if (status != ELML_OK) {
    goto cleanup;
  }
  status = elml_write_meta_chunk(spec, &meta);
  if (status != ELML_OK) {
    goto cleanup;
  }

  status = elml_buf_zero(out, ELML_HEADER_BYTES);
  if (status != ELML_OK) {
    goto cleanup;
  }

  status = elml_append_chunk(out, ELML_CHUNK_PARM,
                             ELML_CHUNK_FLAG_REQUIRED | ELML_CHUNK_FLAG_CRITICAL,
                             &parm);
  if (status != ELML_OK) {
    goto cleanup;
  }
  status = elml_append_chunk(out, ELML_CHUNK_CONF,
                             ELML_CHUNK_FLAG_REQUIRED | ELML_CHUNK_FLAG_CRITICAL,
                             &conf);
  if (status != ELML_OK) {
    goto cleanup;
  }
  if (norm.len > 0u) {
    status = elml_append_chunk(out, ELML_CHUNK_NORM, 0u, &norm);
    if (status != ELML_OK) {
      goto cleanup;
    }
  }
  if (meta.len > 0u) {
    status = elml_append_chunk(out, ELML_CHUNK_META, 0u, &meta);
    if (status != ELML_OK) {
      goto cleanup;
    }
  }

  uint8_t *header = out->data;
  header[0] = ELML_MAGIC_0;
  header[1] = ELML_MAGIC_1;
  header[2] = ELML_MAGIC_2;
  header[3] = ELML_MAGIC_3;
  elml_store_le16(header + 4, ELML_FORMAT_MAJOR);
  elml_store_le16(header + 6, ELML_FORMAT_MINOR);
  elml_store_le16(header + 8, ELML_HEADER_BYTES);
  header[10] = (uint8_t)ELML_DTYPE_F32;
  uint8_t file_flags =
      ELML_FILE_FLAG_LITTLE_ENDIAN | ELML_FILE_FLAG_SEQUENTIAL;
  if (norm.len > 0u) {
    file_flags |= ELML_FILE_FLAG_HAS_NORM;
  }
  if (meta.len > 0u) {
    file_flags |= ELML_FILE_FLAG_HAS_META;
  }
  header[11] = file_flags;
  uint32_t chunk_count = 2u; /* PARM + CONF */
  if (norm.len > 0u) {
    chunk_count++;
  }
  if (meta.len > 0u) {
    chunk_count++;
  }
  elml_store_le32(header + 12, chunk_count);
  elml_store_le64(header + 16, (uint64_t)out->len);
  elml_store_le32(header + 24,
                  elml_crc32(out->data + ELML_HEADER_BYTES,
                             out->len - ELML_HEADER_BYTES, 0u));

cleanup:
  elml_buf_free(&parm);
  elml_buf_free(&conf);
  elml_buf_free(&norm);
  elml_buf_free(&meta);
  if (status != ELML_OK) {
    elml_buf_free(out);
  }
  return status;
}
