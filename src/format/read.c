#include "format/format_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ckdint_compat.h"
#include "elml/arena.h"
#include "elml/format.h"
#include "elml/model.h"
#include "elml/types.h"
#include "format/crc32.h"
#include "format/endian.h"

static bool elml_region_ok(size_t base, size_t need, size_t limit)
{
  return base <= limit && need <= (limit - base);
}

static elml_status elml_parse_header(const uint8_t *blob, size_t blob_len,
                                     elml_format *out)
{
  if (blob_len < ELML_HEADER_BYTES) {
    return ELML_ERR_TRUNCATED;
  }
  if (blob[0] != ELML_MAGIC_0 || blob[1] != ELML_MAGIC_1 ||
      blob[2] != ELML_MAGIC_2 || blob[3] != ELML_MAGIC_3) {
    return ELML_ERR_BAD_MAGIC;
  }

  const uint16_t major = elml_le16(blob + 4);
  const uint16_t minor = elml_le16(blob + 6);
  const uint16_t header_bytes = elml_le16(blob + 8);
  const uint8_t dtype = blob[10];
  const uint8_t flags = blob[11];
  const uint64_t total_bytes = elml_le64(blob + 16);

  if (major != ELML_FORMAT_MAJOR) {
    return ELML_ERR_UNSUPPORTED_MAJOR;
  }
  if (header_bytes != ELML_HEADER_BYTES) {
    return ELML_ERR_UNSUPPORTED_MINOR;
  }
  if (dtype > (uint8_t)ELML_DTYPE_INT8) {
    return ELML_ERR_UNSUPPORTED_DTYPE;
  }
  if (total_bytes != (uint64_t)blob_len) {
    return ELML_ERR_TRUNCATED;
  }

  out->dtype = (elml_dtype)dtype;
  out->major = major;
  out->minor = minor;
  out->file_flags = flags;
  out->file_len = blob_len;
  return ELML_OK;
}

elml_status elml_format_probe(const void *blob, size_t blob_len,
                              elml_dtype *dtype)
{
  if (blob == nullptr || dtype == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (blob_len < ELML_HEADER_BYTES) {
    return ELML_ERR_TRUNCATED;
  }
  const uint8_t *bytes = (const uint8_t *)blob;
  if (bytes[0] != ELML_MAGIC_0 || bytes[1] != ELML_MAGIC_1 ||
      bytes[2] != ELML_MAGIC_2 || bytes[3] != ELML_MAGIC_3) {
    return ELML_ERR_BAD_MAGIC;
  }
  if (elml_le16(bytes + 4) != ELML_FORMAT_MAJOR) {
    return ELML_ERR_UNSUPPORTED_MAJOR;
  }
  if (elml_le16(bytes + 8) != ELML_HEADER_BYTES) {
    return ELML_ERR_UNSUPPORTED_MINOR;
  }
  const uint8_t value = bytes[10];
  if (value > (uint8_t)ELML_DTYPE_INT8) {
    return ELML_ERR_UNSUPPORTED_DTYPE;
  }
  *dtype = (elml_dtype)value;
  return ELML_OK;
}

elml_status elml_chunk_next(const uint8_t *blob, size_t blob_len, size_t *offset,
                            elml_chunk_view *out)
{
  if (blob == nullptr || offset == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (*offset >= blob_len) {
    return ELML_ERR_NOT_FOUND;
  }
  if (!elml_region_ok(*offset, ELML_CHUNK_HEADER_BYTES, blob_len)) {
    return ELML_ERR_TRUNCATED;
  }

  const uint8_t *header = blob + *offset;
  const uint32_t payload_len = elml_le32(header + 8);
  const size_t payload_start = *offset + ELML_CHUNK_HEADER_BYTES;

  if (!elml_region_ok(payload_start, (size_t)payload_len, blob_len)) {
    return ELML_ERR_TRUNCATED;
  }

  const size_t payload_end = payload_start + (size_t)payload_len;
  const size_t next = elml_align_up(payload_end, 8u);
  if (next == 0u || next > blob_len) {
    return ELML_ERR_TRUNCATED;
  }

  out->type = elml_le32(header);
  out->version = elml_le16(header + 4);
  out->flags = elml_le16(header + 6);
  out->crc = elml_le32(header + 12);
  out->data = header + ELML_CHUNK_HEADER_BYTES;
  out->len = payload_len;
  *offset = next;
  return ELML_OK;
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

static elml_status elml_parse_conf(const elml_format *format, elml_format *out)
{
  const uint8_t *cursor = format->conf;
  const uint8_t *end = format->conf + format->conf_len;

  if ((size_t)(end - cursor) < 12u) {
    return ELML_ERR_TRUNCATED;
  }

  const uint32_t layer_count = elml_le32(cursor);
  cursor += 4;
  const uint8_t input_rank = *cursor++;
  const uint8_t output_rank = *cursor++;
  cursor += 2; /* reserved */

  if (layer_count == 0u || layer_count > ELML_MAX_LAYERS) {
    return ELML_ERR_BAD_SHAPE;
  }
  if (input_rank == 0u || input_rank > ELML_MAX_RANK || output_rank == 0u ||
      output_rank > ELML_MAX_RANK) {
    return ELML_ERR_BAD_SHAPE;
  }

  out->input.rank = input_rank;
  out->output.rank = output_rank;
  for (uint8_t i = 0u; i < input_rank; ++i) {
    if ((size_t)(end - cursor) < sizeof(uint32_t)) {
      return ELML_ERR_TRUNCATED;
    }
    out->input.dim[i] = elml_le32(cursor);
    cursor += sizeof(uint32_t);
    if (out->input.dim[i] == 0u) {
      return ELML_ERR_BAD_SHAPE;
    }
  }
  for (uint8_t i = input_rank; i < ELML_MAX_RANK; ++i) {
    out->input.dim[i] = 1u;
  }
  for (uint8_t i = 0u; i < output_rank; ++i) {
    if ((size_t)(end - cursor) < sizeof(uint32_t)) {
      return ELML_ERR_TRUNCATED;
    }
    out->output.dim[i] = elml_le32(cursor);
    cursor += sizeof(uint32_t);
    if (out->output.dim[i] == 0u) {
      return ELML_ERR_BAD_SHAPE;
    }
  }
  for (uint8_t i = output_rank; i < ELML_MAX_RANK; ++i) {
    out->output.dim[i] = 1u;
  }

  out->layer_count = (uint16_t)layer_count;

  for (uint16_t index = 0u; index < out->layer_count; ++index) {
    if ((size_t)(end - cursor) < 16u) {
      return ELML_ERR_TRUNCATED;
    }

    elml_layer_desc *desc = &out->descs[index];
    desc->type_id = (elml_layer_id)elml_le16(cursor);
    desc->type_version = elml_le16(cursor + 2);
    desc->flags = elml_le32(cursor + 4);
    desc->param_off = elml_le32(cursor + 8);
    desc->param_bytes = elml_le32(cursor + 12);
    cursor += 16;

    if ((size_t)(end - cursor) < 2u) {
      return ELML_ERR_TRUNCATED;
    }
    const uint8_t rank_in = *cursor++;
    const uint8_t rank_out = *cursor++;
    if (rank_in == 0u || rank_in > ELML_MAX_RANK || rank_out == 0u ||
        rank_out > ELML_MAX_RANK) {
      return ELML_ERR_BAD_SHAPE;
    }

    desc->in.rank = rank_in;
    desc->out.rank = rank_out;
    for (uint8_t i = 0u; i < rank_in; ++i) {
      if ((size_t)(end - cursor) < sizeof(uint32_t)) {
        return ELML_ERR_TRUNCATED;
      }
      desc->in.dim[i] = elml_le32(cursor);
      cursor += sizeof(uint32_t);
      if (desc->in.dim[i] == 0u) {
        return ELML_ERR_BAD_SHAPE;
      }
    }
    for (uint8_t i = rank_in; i < ELML_MAX_RANK; ++i) {
      desc->in.dim[i] = 1u;
    }
    for (uint8_t i = 0u; i < rank_out; ++i) {
      if ((size_t)(end - cursor) < sizeof(uint32_t)) {
        return ELML_ERR_TRUNCATED;
      }
      desc->out.dim[i] = elml_le32(cursor);
      cursor += sizeof(uint32_t);
      if (desc->out.dim[i] == 0u) {
        return ELML_ERR_BAD_SHAPE;
      }
    }
    for (uint8_t i = rank_out; i < ELML_MAX_RANK; ++i) {
      desc->out.dim[i] = 1u;
    }

    if ((size_t)(end - cursor) < sizeof(uint32_t)) {
      return ELML_ERR_TRUNCATED;
    }
    const uint32_t extra_bytes = elml_le32(cursor);
    cursor += sizeof(uint32_t);
    if (!elml_region_ok((size_t)(cursor - format->conf), (size_t)extra_bytes,
                        format->conf_len)) {
      return ELML_ERR_TRUNCATED;
    }
    desc->extra = cursor;
    desc->extra_bytes = extra_bytes;
    cursor += extra_bytes;

    if (desc->param_off != UINT32_MAX) {
      if (desc->param_off > format->parm_len ||
          desc->param_bytes > format->parm_len - desc->param_off) {
        return ELML_ERR_BAD_PARAM;
      }
    }
  }

  for (uint16_t i = 1u; i < out->layer_count; ++i) {
    if (!elml_shape_equal(&out->descs[i - 1u].out, &out->descs[i].in)) {
      return ELML_ERR_BAD_SHAPE;
    }
  }
  if (!elml_shape_equal(&out->descs[0].in, &out->input) ||
      !elml_shape_equal(&out->descs[out->layer_count - 1u].out, &out->output)) {
    return ELML_ERR_BAD_SHAPE;
  }

  return ELML_OK;
}

static elml_status elml_parse_norm(const elml_format *format, elml_format *out)
{
  const uint8_t *cursor = format->norm;
  const uint8_t *end = format->norm + format->norm_len;

  if ((size_t)(end - cursor) < 12u) {
    return ELML_ERR_TRUNCATED;
  }

  const uint8_t input_mode = *cursor++;
  const uint8_t output_mode = *cursor++;
  cursor += 2; /* reserved */

  if (input_mode > (uint8_t)ELML_NORM_MINMAX ||
      output_mode > (uint8_t)ELML_NORM_MINMAX) {
    return ELML_ERR_BAD_PARAM;
  }

  const uint32_t input_count = elml_le32(cursor);
  cursor += 4;
  if (input_count != out->input.dim[0]) {
    return ELML_ERR_BAD_SHAPE;
  }

  size_t params_bytes;
  if (ckd_mul(&params_bytes, (size_t)input_count, 2u * sizeof(float))) {
    return ELML_ERR_BAD_PARAM;
  }
  if (!elml_region_ok((size_t)(cursor - format->norm), params_bytes,
                      format->norm_len)) {
    return ELML_ERR_TRUNCATED;
  }
  const float *input_params = (const float *)cursor;
  cursor += params_bytes;

  if ((size_t)(end - cursor) < sizeof(uint32_t)) {
    return ELML_ERR_TRUNCATED;
  }
  const uint32_t output_count = elml_le32(cursor);
  cursor += 4;
  if (output_count != out->output.dim[0]) {
    return ELML_ERR_BAD_SHAPE;
  }

  if (ckd_mul(&params_bytes, (size_t)output_count, 2u * sizeof(float))) {
    return ELML_ERR_BAD_PARAM;
  }
  if (!elml_region_ok((size_t)(cursor - format->norm), params_bytes,
                      format->norm_len)) {
    return ELML_ERR_TRUNCATED;
  }
  const float *output_params = (const float *)cursor;

  out->norm_input_mode = input_mode;
  out->norm_output_mode = output_mode;
  out->norm_input_count = input_count;
  out->norm_output_count = output_count;
  out->norm_input_params = input_params;
  out->norm_output_params = output_params;
  return ELML_OK;
}

elml_status elml_format_parse(const void *blob, size_t blob_len,
                              elml_format *out)
{
  if (blob == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  memset(out, 0, sizeof *out);

  const uint8_t *bytes = (const uint8_t *)blob;
  elml_status status = elml_parse_header(bytes, blob_len, out);
  if (status != ELML_OK) {
    return status;
  }

  if (elml_crc32(bytes + ELML_HEADER_BYTES, blob_len - ELML_HEADER_BYTES, 0u) !=
      elml_le32(bytes + 24)) {
    return ELML_ERR_CRC;
  }

  size_t offset = ELML_HEADER_BYTES;
  uint32_t chunks = 0u;
  bool have_conf = false;
  bool have_parm = false;

  while (offset < blob_len) {
    elml_chunk_view chunk;
    status = elml_chunk_next(bytes, blob_len, &offset, &chunk);
    if (status == ELML_ERR_NOT_FOUND) {
      break;
    }
    if (status != ELML_OK) {
      return status;
    }
    if (elml_crc32(chunk.data, chunk.len, 0u) != chunk.crc) {
      return ELML_ERR_CRC;
    }
    if (chunk.version > 1u &&
        (chunk.flags & ELML_CHUNK_FLAG_REQUIRED) != 0u) {
      return ELML_ERR_UNSUPPORTED_MINOR;
    }

    chunks++;
    switch (chunk.type) {
    case ELML_CHUNK_CONF:
      if (have_conf) {
        return ELML_ERR_BAD_PARAM;
      }
      have_conf = true;
      out->conf = chunk.data;
      out->conf_len = chunk.len;
      break;
    case ELML_CHUNK_PARM:
      if (have_parm) {
        return ELML_ERR_BAD_PARAM;
      }
      have_parm = true;
      out->parm = chunk.data;
      out->parm_len = chunk.len;
      break;
    case ELML_CHUNK_QNT:
      if (out->qnt != nullptr) {
        return ELML_ERR_BAD_PARAM;
      }
      out->qnt = chunk.data;
      out->qnt_len = chunk.len;
      break;
    case ELML_CHUNK_NORM:
      if (out->norm != nullptr) {
        return ELML_ERR_BAD_PARAM;
      }
      out->norm = chunk.data;
      out->norm_len = chunk.len;
      break;
    case ELML_CHUNK_META:
      if (out->meta != nullptr) {
        return ELML_ERR_BAD_PARAM;
      }
      out->meta = chunk.data;
      out->meta_len = chunk.len;
      break;
    case ELML_CHUNK_TOPO:
    case ELML_CHUNK_CKPT:
      break;
    default:
      if ((chunk.flags & ELML_CHUNK_FLAG_REQUIRED) != 0u) {
        return ELML_ERR_UNKNOWN_CHUNK_REQUIRED;
      }
      break;
    }
  }

  if (offset != blob_len) {
    return ELML_ERR_TRUNCATED;
  }
  if (!have_conf || !have_parm) {
    return ELML_ERR_BAD_PARAM;
  }
  if (out->dtype == ELML_DTYPE_INT8 && out->qnt == nullptr) {
    return ELML_ERR_BAD_PARAM;
  }

  out->chunk_count = chunks;

  status = elml_parse_conf(out, out);
  if (status != ELML_OK) {
    return status;
  }

  if (out->norm != nullptr) {
    status = elml_parse_norm(out, out);
    if (status != ELML_OK) {
      return status;
    }
  }

  return ELML_OK;
}
