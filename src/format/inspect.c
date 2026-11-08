#include "format/inspect.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "elml/format.h"
#include "elml/model.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "format/format_internal.h"

static const char *elml_dtype_name(elml_dtype dtype)
{
  switch (dtype) {
  case ELML_DTYPE_F32:
    return "f32";
  case ELML_DTYPE_INT8:
    return "int8";
  case ELML_DTYPE_F16:
    return "f16";
  }
  return "unknown";
}

static const char *elml_layer_type_name(elml_layer_id id)
{
  const elml_layer_vt *vt = elml_layer_find(id);
  if (vt != nullptr) {
    return vt->name;
  }
  switch (id) {
  case ELML_LAYER_DENSE:
    return "dense (not linked)";
  case ELML_LAYER_CONV1D:
    return "conv1d (not linked)";
  case ELML_LAYER_ACT_RELU:
    return "relu (not linked)";
  case ELML_LAYER_ACT_LEAKY_RELU:
    return "leaky_relu (not linked)";
  case ELML_LAYER_ACT_TANH:
    return "tanh (not linked)";
  case ELML_LAYER_ACT_SIGMOID:
    return "sigmoid (not linked)";
  case ELML_LAYER_ACT_SOFTMAX:
    return "softmax (not linked)";
  case ELML_LAYER_ACT_LINEAR:
    return "linear (not linked)";
  default:
    return "unknown";
  }
}

static void elml_print_shape(FILE *out, const elml_shape *shape)
{
  fputc('[', out);
  for (uint8_t i = 0u; i < shape->rank; ++i) {
    if (i != 0u) {
      fputc(',', out);
    }
    fprintf(out, "%" PRIu32, shape->dim[i]);
  }
  fputc(']', out);
}

static void elml_json_string(FILE *out, const uint8_t *data, uint32_t len)
{
  fputc('"', out);
  for (uint32_t i = 0u; i < len; ++i) {
    const uint8_t c = data[i];
    switch (c) {
    case '"':
      fputs("\\\"", out);
      break;
    case '\\':
      fputs("\\\\", out);
      break;
    case '\n':
      fputs("\\n", out);
      break;
    case '\r':
      fputs("\\r", out);
      break;
    case '\t':
      fputs("\\t", out);
      break;
    default:
      if (c < 0x20u) {
        fprintf(out, "\\u%04x", (unsigned)c);
      } else {
        fputc((int)c, out);
      }
      break;
    }
  }
  fputc('"', out);
}

static void elml_print_header_text(const elml_format *format, FILE *out)
{
  fprintf(out, "EmbedLML model\n");
  fprintf(out, "  format       : %u.%u\n", (unsigned)format->major,
          (unsigned)format->minor);
  fprintf(out, "  dtype        : %s\n", elml_dtype_name(format->dtype));
  fprintf(out, "  bytes        : %zu\n", format->file_len);
  fprintf(out, "  chunks       : %" PRIu32 "\n", format->chunk_count);
  fprintf(out, "  file flags   : 0x%02x\n", (unsigned)format->file_flags);
  fprintf(out, "  input        : ");
  elml_print_shape(out, &format->input);
  fputc('\n', out);
  fprintf(out, "  output       : ");
  elml_print_shape(out, &format->output);
  fputc('\n', out);
}

static void elml_print_layers_text(const elml_format *format, FILE *out)
{
  fprintf(out, "\nLayers (%u)\n", (unsigned)format->layer_count);
  for (uint16_t i = 0u; i < format->layer_count; ++i) {
    const elml_layer_desc *desc = &format->descs[i];
    fprintf(out, "  %2u  %-18s id=0x%04x v%u  ", (unsigned)i,
            elml_layer_type_name(desc->type_id), (unsigned)desc->type_id,
            (unsigned)desc->type_version);
    elml_print_shape(out, &desc->in);
    fputs(" -> ", out);
    elml_print_shape(out, &desc->out);
    if (desc->param_bytes > 0u) {
      fprintf(out, "  params=%" PRIu32 "B", desc->param_bytes);
    }
    fputc('\n', out);
  }
}

static void elml_print_norm_text(const elml_format *format, FILE *out)
{
  if (format->norm == nullptr) {
    fprintf(out, "\nNormalization: none\n");
    return;
  }
  fprintf(out, "\nNormalization\n");
  fprintf(out, "  input mode   : %u\n", (unsigned)format->norm_input_mode);
  fprintf(out, "  output mode  : %u\n", (unsigned)format->norm_output_mode);
  fprintf(out, "  input count  : %" PRIu32 "\n", format->norm_input_count);
  fprintf(out, "  output count : %" PRIu32 "\n", format->norm_output_count);
}

static void elml_print_meta_text(const elml_format *format, FILE *out)
{
  fprintf(out, "\nMetadata\n");
  if (format->meta == nullptr || format->meta_len == 0u) {
    fprintf(out, "  (none)\n");
    return;
  }

  size_t offset = 0u;
  while (offset + 4u <= (size_t)format->meta_len) {
    const uint16_t key_len =
        (uint16_t)((uint16_t)format->meta[offset] |
                   ((uint16_t)format->meta[offset + 1u] << 8));
    const uint16_t value_len =
        (uint16_t)((uint16_t)format->meta[offset + 2u] |
                   ((uint16_t)format->meta[offset + 3u] << 8));
    offset += 4u;
    if (offset + key_len + value_len > (size_t)format->meta_len) {
      fprintf(out, "  (truncated)\n");
      return;
    }
    fprintf(out, "  %.*s = %.*s\n", (int)key_len,
            (const char *)(format->meta + offset), (int)value_len,
            (const char *)(format->meta + offset + key_len));
    offset += (size_t)key_len + value_len;
  }
}

static void elml_print_scratch_text(const uint8_t *blob, size_t len,
                                    FILE *out)
{
  elml_model model;
  const elml_status status = elml_model_init(&model, blob, len, ELML_DTYPE_F32);
  if (status == ELML_OK) {
    fprintf(out, "\nRuntime\n");
    fprintf(out, "  scratch      : %zu bytes (batch 1)\n",
            elml_scratch_bytes(&model, 1u));
    fprintf(out, "  runnable     : yes\n");
  } else {
    fprintf(out, "\nRuntime\n");
    fprintf(out, "  runnable     : no (%s)\n", elml_error_string(status));
  }
}

static void elml_inspect_json(const elml_format *format, const uint8_t *blob,
                              FILE *out)
{
  fprintf(out, "{\"format\":{\"major\":%u,\"minor\":%u,\"dtype\":\"%s\"",
          (unsigned)format->major, (unsigned)format->minor,
          elml_dtype_name(format->dtype));
  fprintf(out, ",\"bytes\":%zu,\"chunks\":%" PRIu32 "},\"input\":[",
          format->file_len, format->chunk_count);
  for (uint8_t i = 0u; i < format->input.rank; ++i) {
    fprintf(out, "%s%" PRIu32, (i == 0u) ? "" : ",",
            format->input.dim[i]);
  }
  fputs("],\"output\":[", out);
  for (uint8_t i = 0u; i < format->output.rank; ++i) {
    fprintf(out, "%s%" PRIu32, (i == 0u) ? "" : ",",
            format->output.dim[i]);
  }
  fputs("],\"layers\":[", out);
  for (uint16_t i = 0u; i < format->layer_count; ++i) {
    const elml_layer_desc *desc = &format->descs[i];
    fprintf(out, "%s{\"index\":%u,\"type\":\"%s\",\"type_id\":%u",
            (i == 0u) ? "" : ",", (unsigned)i,
            elml_layer_type_name(desc->type_id), (unsigned)desc->type_id);
    fprintf(out, ",\"params\":%" PRIu32 "}", desc->param_bytes);
  }
  fputs("],\"norm\":", out);
  if (format->norm == nullptr) {
    fputs("null", out);
  } else {
    fprintf(out, "{\"input_mode\":%u,\"output_mode\":%u}",
            (unsigned)format->norm_input_mode,
            (unsigned)format->norm_output_mode);
  }

  fputs(",\"meta\":{", out);
  bool first = true;
  size_t offset = 0u;
  while (format->meta != nullptr && offset + 4u <= (size_t)format->meta_len) {
    const uint16_t key_len =
        (uint16_t)((uint16_t)format->meta[offset] |
                   ((uint16_t)format->meta[offset + 1u] << 8));
    const uint16_t value_len =
        (uint16_t)((uint16_t)format->meta[offset + 2u] |
                   ((uint16_t)format->meta[offset + 3u] << 8));
    offset += 4u;
    if (offset + key_len + value_len > (size_t)format->meta_len) {
      break;
    }
    if (!first) {
      fputc(',', out);
    }
    first = false;
    elml_json_string(out, format->meta + offset, key_len);
    fputc(':', out);
    elml_json_string(out, format->meta + offset + key_len, value_len);
    offset += (size_t)key_len + value_len;
  }
  fputs("}", out);

  elml_model model;
  const elml_status status = elml_model_init(&model, blob, format->file_len,
                                             ELML_DTYPE_F32);
  if (status == ELML_OK) {
    fprintf(out, ",\"scratch\":%zu,\"runnable\":true}\n",
            elml_scratch_bytes(&model, 1u));
  } else {
    fprintf(out, ",\"scratch\":null,\"runnable\":false,\"reason\":\"%s\"}\n",
            elml_error_string(status));
  }
}

void elml_inspect_print(const elml_format *format, const uint8_t *blob,
                        FILE *out, bool json)
{
  if (format == nullptr || blob == nullptr || out == nullptr) {
    return;
  }
  if (json) {
    elml_inspect_json(format, blob, out);
    return;
  }

  elml_print_header_text(format, out);
  elml_print_layers_text(format, out);
  elml_print_norm_text(format, out);
  elml_print_meta_text(format, out);
  elml_print_scratch_text(blob, format->file_len, out);
}
