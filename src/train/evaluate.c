#include "train/evaluate.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/arena.h"
#include "train/dataset.h"
#include "train/io.h"

[[gnu::format(printf, 3, 4)]]
static elml_status eval_fail(elml_eval_result *result, elml_status status,
                             const char *format, ...)
{
  if (result != nullptr) {
    va_list args;
    va_start(args, format);
    vsnprintf(result->message, sizeof result->message, format, args);
    va_end(args);
  }
  return status;
}

static elml_status eval_parse_cell(const char *text, float *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  char *end = nullptr;
  const float value = strtof(text, &end);
  if (end == text || *end != '\0' || !isfinite(value)) {
    return ELML_ERR_INVALID_ARG;
  }
  *out = value;
  return ELML_OK;
}

bool elml_eval_is_classification(const elml_model *model)
{
  if (model == nullptr) {
    return false;
  }
  char task[32];
  size_t len = 0u;
  if (elml_meta_get(model, "task", task, sizeof task, &len) == ELML_OK) {
    return strcmp(task, "classification") == 0;
  }
  return elml_label_count(model) > 0u;
}

const char *elml_eval_class_label(const elml_model *model, uint32_t index,
                                  char *buffer, size_t capacity)
{
  size_t len = 0u;
  if (model != nullptr &&
      elml_label_get(model, index, buffer, capacity, &len) == ELML_OK) {
    return buffer;
  }
  snprintf(buffer, capacity, "class_%u", index);
  return buffer;
}

static bool eval_meta_token(const char *list, uint32_t index, char *out,
                            size_t capacity)
{
  if (list == nullptr || out == nullptr || capacity == 0u) {
    return false;
  }

  uint32_t current = 0u;
  const char *start = list;
  for (;;) {
    const char *end = strchr(start, ',');
    if (end == nullptr) {
      end = start + strlen(start);
    }
    if (current == index) {
      while (start < end && (*start == ' ' || *start == '\t')) {
        start++;
      }
      while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
        end--;
      }
      const size_t len = (size_t)(end - start);
      if (len + 1u > capacity) {
        return false;
      }
      memcpy(out, start, len);
      out[len] = '\0';
      return true;
    }
    if (*end == '\0') {
      return false;
    }
    start = end + 1;
    current++;
  }
}

const char *elml_eval_output_name(const elml_model *model, uint32_t index,
                                  char *buffer, size_t capacity)
{
  char list[ELML_CONFIG_LIST_MAX];
  size_t len = 0u;
  if (model != nullptr &&
      elml_meta_get(model, "columns.outputs", list, sizeof list, &len) ==
          ELML_OK &&
      eval_meta_token(list, index, buffer, capacity)) {
    return buffer;
  }
  snprintf(buffer, capacity, "output_%u", index);
  return buffer;
}

const float *elml_eval_values(const elml_eval_result *result, uint32_t row)
{
  if (result == nullptr || result->values == nullptr ||
      result->output_dim == 0u || row >= result->row_count) {
    return nullptr;
  }
  return &result->values[(size_t)row * (size_t)result->output_dim];
}

const float *elml_eval_targets(const elml_eval_result *result, uint32_t row)
{
  if (result == nullptr || result->targets == nullptr ||
      result->output_dim == 0u || row >= result->row_count) {
    return nullptr;
  }
  return &result->targets[(size_t)row * (size_t)result->output_dim];
}

static void eval_column_names(const elml_csv *csv, const uint32_t *columns,
                              uint32_t count, char *out, size_t capacity)
{
  if (out == nullptr || capacity == 0u) {
    return;
  }
  out[0] = '\0';
  size_t used = 0u;
  for (uint32_t i = 0u; i < count; ++i) {
    const char *name =
        (csv->data_start > 0u) ? elml_csv_cell(csv, 0u, columns[i]) : nullptr;
    char fallback[32];
    if (name == nullptr || *name == '\0') {
      snprintf(fallback, sizeof fallback, "column_%u", columns[i] + 1u);
      name = fallback;
    }
    const int written = snprintf(out + used, capacity - used, "%s%s",
                                 used > 0u ? "," : "", name);
    if (written < 0 || (size_t)written >= capacity - used) {
      out[capacity - 1u] = '\0';
      return;
    }
    used += (size_t)written;
  }
}

static elml_status eval_resolve_inputs(const elml_eval_options *options,
                                       elml_eval_result *result,
                                       uint32_t input_dim)
{
  char meta[ELML_CONFIG_LIST_MAX];
  const char *text = options->inputs;
  if (text == nullptr || *text == '\0') {
    size_t len = 0u;
    if (elml_meta_get(&result->model, "columns.inputs", meta, sizeof meta,
                      &len) == ELML_OK) {
      text = meta;
    }
  }

  if (text != nullptr && *text != '\0') {
    const elml_status status =
        elml_columns_resolve(text, &result->csv, &result->input_columns,
                             &result->input_count);
    if (status != ELML_OK) {
      return eval_fail(result, status,
                       "cannot resolve input columns \"%s\": %s", text,
                       elml_error_string(status));
    }
    snprintf(result->inputs_text, sizeof result->inputs_text, "%s", text);
  } else {
    result->input_columns =
        (uint32_t *)malloc((size_t)input_dim * sizeof *result->input_columns);
    if (result->input_columns == nullptr) {
      return eval_fail(result, ELML_ERR_OUT_OF_MEMORY, "out of memory");
    }
    for (uint32_t i = 0u; i < input_dim; ++i) {
      result->input_columns[i] = i;
    }
    result->input_count = input_dim;
    eval_column_names(&result->csv, result->input_columns, input_dim,
                      result->inputs_text, sizeof result->inputs_text);
  }

  if (result->input_count != input_dim) {
    return eval_fail(result, ELML_ERR_INVALID_ARG,
                     "model expects %u inputs but the selection has %u",
                     input_dim, result->input_count);
  }
  if (result->csv.cols < input_dim) {
    return eval_fail(result, ELML_ERR_INVALID_ARG,
                     "CSV has %u columns; the model needs %u inputs",
                     result->csv.cols, input_dim);
  }
  return ELML_OK;
}

static elml_status eval_resolve_targets(const elml_eval_options *options,
                                        elml_eval_result *result,
                                        uint32_t output_dim)
{
  char meta[ELML_CONFIG_LIST_MAX];
  const char *text = options->targets;
  if (text == nullptr || *text == '\0') {
    size_t len = 0u;
    if (elml_meta_get(&result->model, "columns.outputs", meta, sizeof meta,
                      &len) == ELML_OK) {
      text = meta;
    }
  }
  if (text == nullptr || *text == '\0') {
    return eval_fail(result, ELML_ERR_INVALID_ARG,
                     "scoring needs target columns: pass them or evaluate a "
                     "model exported with column metadata");
  }

  const elml_status status =
      elml_columns_resolve(text, &result->csv, &result->target_columns,
                           &result->target_count);
  if (status != ELML_OK) {
    return eval_fail(result, status,
                     "cannot resolve target columns \"%s\": %s", text,
                     elml_error_string(status));
  }
  snprintf(result->targets_text, sizeof result->targets_text, "%s", text);

  if (!result->classification && result->target_count != output_dim) {
    return eval_fail(result, ELML_ERR_INVALID_ARG,
                     "model expects %u outputs but %u target columns were "
                     "given",
                     output_dim, result->target_count);
  }
  if (result->classification && result->target_count != 1u) {
    return eval_fail(result, ELML_ERR_INVALID_ARG,
                     "classification scoring needs exactly one target column");
  }
  result->scored = true;
  return ELML_OK;
}

static void eval_note_skip(elml_eval_result *result, const char *what,
                           uint32_t line)
{
  if (result->skipped == 0u) {
    snprintf(result->skipped_note, sizeof result->skipped_note,
             "row %u: %s", line, what);
  }
}

static elml_status eval_rows(const elml_eval_options *options,
                             elml_eval_result *result)
{
  if (result->csv.rows < result->csv.data_start) {
    return eval_fail(result, ELML_ERR_INVALID_ARG, "CSV has no data rows");
  }
  const uint32_t data_rows = result->csv.rows - result->csv.data_start;
  const uint32_t limit =
      (options->limit > 0u && options->limit < data_rows) ? options->limit
                                                         : data_rows;
  result->data_rows = data_rows;
  result->limit = limit;
  if (limit == 0u) {
    return ELML_OK;
  }

  const uint32_t input_dim = result->model.input.dim[0];
  const uint32_t output_dim = result->output_dim;
  const uint32_t label_count = elml_label_count(&result->model);
  const size_t scratch_bytes =
      elml_align_up(elml_scratch_bytes(&result->model, 1u), ELML_TENSOR_ALIGN);

  void *scratch = elml_aligned_alloc(ELML_TENSOR_ALIGN, scratch_bytes);
  float *input = (float *)malloc((size_t)input_dim * sizeof *input);
  float *prediction = (float *)malloc((size_t)output_dim * sizeof *prediction);
  float *target_buffer = nullptr;
  if (result->scored && !result->classification) {
    target_buffer = (float *)malloc((size_t)output_dim * sizeof *target_buffer);
  }
  result->values = (float *)malloc((size_t)limit * (size_t)output_dim *
                                   sizeof *result->values);
  if (result->scored) {
    result->targets = (float *)malloc((size_t)limit * (size_t)output_dim *
                                      sizeof *result->targets);
  }
  result->rows = (elml_eval_row *)calloc(limit, sizeof *result->rows);
  if (scratch == nullptr || input == nullptr || prediction == nullptr ||
      result->values == nullptr || result->rows == nullptr ||
      (result->scored && result->targets == nullptr) ||
      (result->scored && !result->classification && target_buffer == nullptr)) {
    elml_aligned_free(scratch);
    free(input);
    free(prediction);
    free(target_buffer);
    return eval_fail(result, ELML_ERR_OUT_OF_MEMORY,
                     "out of memory for %u rows", limit);
  }

  double loss_sum = 0.0;
  double abs_sum = 0.0;
  uint64_t loss_count = 0u;
  elml_status status = ELML_OK;

  for (uint32_t i = 0u; i < limit; ++i) {
    const uint32_t csv_row = result->csv.data_start + i;
    const uint32_t line = csv_row + 1u;

    bool bad_input = false;
    for (uint32_t f = 0u; f < input_dim; ++f) {
      if (eval_parse_cell(elml_csv_cell(&result->csv, csv_row,
                                        result->input_columns[f]),
                          &input[f]) != ELML_OK) {
        bad_input = true;
        break;
      }
    }
    if (bad_input) {
      if (!options->skip_bad_rows) {
        status = eval_fail(result, ELML_ERR_INVALID_ARG,
                           "bad input value at row %u", line);
        break;
      }
      eval_note_skip(result, "bad input value", line);
      result->skipped++;
      continue;
    }

    status = elml_forward(&result->model, scratch, scratch_bytes, input,
                          prediction);
    if (status != ELML_OK) {
      (void)eval_fail(result, status, "inference failed at row %u: %s", line,
                      elml_error_string(status));
      break;
    }

    bool bad_target = false;
    if (result->scored && !result->classification) {
      for (uint32_t o = 0u; o < output_dim; ++o) {
        if (eval_parse_cell(elml_csv_cell(&result->csv, csv_row,
                                          result->target_columns[o]),
                            &target_buffer[o]) != ELML_OK) {
          bad_target = true;
          break;
        }
      }
      if (bad_target) {
        if (!options->skip_bad_rows) {
          status = eval_fail(result, ELML_ERR_INVALID_ARG,
                             "bad target value at row %u", line);
          break;
        }
        eval_note_skip(result, "bad target value", line);
        result->skipped++;
        continue;
      }
    }

    elml_eval_row *row = &result->rows[result->row_count];
    row->line = line;
    row->expected = -1;
    row->correct = false;

    float *values =
        &result->values[(size_t)result->row_count * (size_t)output_dim];
    memcpy(values, prediction, (size_t)output_dim * sizeof *values);

    if (result->classification) {
      uint32_t best = 0u;
      for (uint32_t o = 1u; o < output_dim; ++o) {
        if (prediction[o] > prediction[best]) {
          best = o;
        }
      }
      row->predicted = best;
      row->confidence = prediction[best];

      if (result->scored) {
        result->scored_rows++;
        const char *target = elml_csv_cell(&result->csv, csv_row,
                                           result->target_columns[0]);
        char label[128];
        size_t len = 0u;
        if (target != nullptr) {
          for (uint32_t l = 0u; l < label_count; ++l) {
            if (elml_label_get(&result->model, l, label, sizeof label, &len) ==
                    ELML_OK &&
                strcmp(label, target) == 0) {
              row->expected = (int32_t)l;
              if (l == best) {
                row->correct = true;
                result->correct++;
              }
              break;
            }
          }
        }
      }
    } else {
      row->predicted = 0u;
      row->confidence = prediction[0];
      if (result->scored) {
        result->scored_rows++;
        float *targets =
            &result->targets[(size_t)result->row_count * (size_t)output_dim];
        for (uint32_t o = 0u; o < output_dim; ++o) {
          targets[o] = target_buffer[o];
          const double diff = (double)prediction[o] - (double)targets[o];
          loss_sum += diff * diff;
          abs_sum += fabs(diff);
          loss_count++;
        }
      }
    }

    result->row_count++;
  }

  if (status == ELML_OK && result->scored && !result->classification &&
      loss_count > 0u) {
    result->mse = loss_sum / (double)loss_count;
    result->mae = abs_sum / (double)loss_count;
  }

  elml_aligned_free(scratch);
  free(input);
  free(prediction);
  free(target_buffer);
  return status;
}

elml_status elml_eval_run(const elml_eval_options *options,
                          elml_eval_result *result)
{
  if (result == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  memset(result, 0, sizeof *result);
  if (options == nullptr || options->model_path == nullptr ||
      options->csv_path == nullptr || options->model_path[0] == '\0' ||
      options->csv_path[0] == '\0') {
    return eval_fail(result, ELML_ERR_INVALID_ARG,
                     "choose a model and a CSV file to evaluate");
  }

  result->delimiter = (options->delimiter != '\0') ? options->delimiter : ',';
  result->has_header = options->has_header;

  elml_status status = elml_read_file(options->model_path, &result->blob,
                                      &result->blob_len, ELML_PARAM_ALIGN);
  if (status != ELML_OK) {
    return eval_fail(result, status, "cannot read %s: %s", options->model_path,
                     elml_error_string(status));
  }

  status = elml_model_init(&result->model, result->blob, result->blob_len,
                           ELML_DTYPE_F32);
  if (status != ELML_OK) {
    return eval_fail(result, status, "cannot load %s: %s", options->model_path,
                     elml_error_string(status));
  }

  const uint32_t input_dim = result->model.input.dim[0];
  const uint32_t output_dim = result->model.output.dim[0];
  const uint32_t label_count = elml_label_count(&result->model);
  result->output_dim = output_dim;
  result->classification = elml_eval_is_classification(&result->model);

  if (input_dim == 0u || output_dim == 0u) {
    return eval_fail(result, ELML_ERR_BAD_SHAPE,
                     "%s has an empty input or output shape",
                     options->model_path);
  }
  if (label_count > 0u && label_count != output_dim) {
    return eval_fail(result, ELML_ERR_BAD_SHAPE,
                     "%s lists %u labels for %u outputs", options->model_path,
                     label_count, output_dim);
  }

  status = elml_csv_load(options->csv_path, result->delimiter,
                         result->has_header, &result->csv);
  if (status != ELML_OK) {
    return eval_fail(result, status, "cannot read %s: %s", options->csv_path,
                     elml_error_string(status));
  }

  status = eval_resolve_inputs(options, result, input_dim);
  if (status != ELML_OK) {
    return status;
  }
  if (options->score) {
    status = eval_resolve_targets(options, result, output_dim);
    if (status != ELML_OK) {
      return status;
    }
  }
  return eval_rows(options, result);
}

void elml_eval_free(elml_eval_result *result)
{
  if (result == nullptr) {
    return;
  }
  /* Reverse allocation order: rows, targets, values, columns, csv, blob. */
  free(result->rows);
  free(result->targets);
  free(result->values);
  free(result->target_columns);
  free(result->input_columns);
  elml_csv_free(&result->csv);
  elml_aligned_free(result->blob);
  memset(result, 0, sizeof *result);
}

/*
 * The prediction file is assembled in memory and then handed to
 * elml_write_file(). That keeps the writer identical on the host and in the
 * browser, where stdio has no file system behind it.
 */
typedef struct eval_sink {
  char *data;
  size_t len;
  size_t cap;
  bool failed;
} eval_sink;

static bool eval_sink_reserve(eval_sink *sink, size_t extra)
{
  if (sink->failed) {
    return false;
  }
  if (extra > SIZE_MAX - sink->len) {
    sink->failed = true;
    return false;
  }
  const size_t needed = sink->len + extra;
  if (needed <= sink->cap) {
    return true;
  }
  size_t next = (sink->cap == 0u) ? 4096u : sink->cap;
  while (next < needed) {
    if (next > SIZE_MAX / 2u) {
      next = needed;
      break;
    }
    next *= 2u;
  }
  char *grown = (char *)realloc(sink->data, next);
  if (grown == nullptr) {
    sink->failed = true;
    return false;
  }
  sink->data = grown;
  sink->cap = next;
  return true;
}

static void eval_sink_putc(eval_sink *sink, char c)
{
  if (eval_sink_reserve(sink, 1u)) {
    sink->data[sink->len++] = c;
  }
}

static void eval_sink_puts(eval_sink *sink, const char *text)
{
  const size_t length = strlen(text);
  if (eval_sink_reserve(sink, length)) {
    memcpy(sink->data + sink->len, text, length);
    sink->len += length;
  }
}

static void eval_write_field(eval_sink *sink, const char *text, char delimiter)
{
  bool needs_quotes = false;
  for (const char *cursor = text; *cursor != '\0'; ++cursor) {
    if (*cursor == delimiter || *cursor == '"' || *cursor == '\n' ||
        *cursor == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) {
    eval_sink_puts(sink, text);
    return;
  }
  eval_sink_putc(sink, '"');
  for (const char *cursor = text; *cursor != '\0'; ++cursor) {
    if (*cursor == '"') {
      eval_sink_putc(sink, '"');
    }
    eval_sink_putc(sink, *cursor);
  }
  eval_sink_putc(sink, '"');
}

elml_status elml_eval_write_csv(const elml_eval_result *result,
                                const char *path)
{
  if (result == nullptr || path == nullptr || result->rows == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  eval_sink sink = {0};

  const char delimiter = result->delimiter;
  const uint32_t output_dim = result->output_dim;
  const uint32_t label_count = elml_label_count(&result->model);

  if (result->has_header) {
    for (uint32_t col = 0u; col < result->csv.cols; ++col) {
      if (col > 0u) {
        eval_sink_putc(&sink, delimiter);
      }
      const char *name = elml_csv_cell(&result->csv, 0u, col);
      eval_write_field(&sink, (name != nullptr) ? name : "", delimiter);
    }
    char label[128];
    if (label_count > 0u) {
      eval_sink_putc(&sink, delimiter);
      eval_write_field(&sink, "prediction", delimiter);
      for (uint32_t i = 0u; i < label_count; ++i) {
        char column[160];
        snprintf(column, sizeof column, "prob_%s",
                 elml_eval_class_label(&result->model, i, label, sizeof label));
        eval_sink_putc(&sink, delimiter);
        eval_write_field(&sink, column, delimiter);
      }
    } else {
      for (uint32_t o = 0u; o < output_dim; ++o) {
        char column[64];
        snprintf(column, sizeof column, "pred_%u", o);
        eval_sink_putc(&sink, delimiter);
        eval_write_field(&sink, column, delimiter);
      }
    }
    eval_sink_putc(&sink, '\n');
  }

  for (uint32_t r = 0u; r < result->row_count; ++r) {
    const elml_eval_row *row = &result->rows[r];
    const uint32_t csv_row = row->line - 1u;
    for (uint32_t col = 0u; col < result->csv.cols; ++col) {
      if (col > 0u) {
        eval_sink_putc(&sink, delimiter);
      }
      const char *cell = elml_csv_cell(&result->csv, csv_row, col);
      eval_write_field(&sink, (cell != nullptr) ? cell : "", delimiter);
    }

    const float *values = elml_eval_values(result, r);
    if (label_count > 0u) {
      char label[128];
      eval_sink_putc(&sink, delimiter);
      eval_write_field(&sink,
                       elml_eval_class_label(&result->model, row->predicted,
                                             label, sizeof label),
                       delimiter);
      for (uint32_t i = 0u; i < label_count; ++i) {
        char value[64];
        snprintf(value, sizeof value, "%.6f", (double)values[i]);
        eval_sink_putc(&sink, delimiter);
        eval_write_field(&sink, value, delimiter);
      }
    } else {
      for (uint32_t o = 0u; o < output_dim; ++o) {
        char value[64];
        snprintf(value, sizeof value, "%.6f", (double)values[o]);
        eval_sink_putc(&sink, delimiter);
        eval_write_field(&sink, value, delimiter);
      }
    }
    eval_sink_putc(&sink, '\n');
  }

  if (sink.failed) {
    free(sink.data);
    return ELML_ERR_OUT_OF_MEMORY;
  }
  const elml_status status = elml_write_file(path, sink.data, sink.len);
  free(sink.data);
  return status;
}
