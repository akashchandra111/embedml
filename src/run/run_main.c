#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/arena.h"
#include "elml/model.h"
#include "elml/types.h"
#include "train/csv.h"
#include "train/dataset.h"
#include "train/io.h"

typedef struct elml_run_options {
  const char *model_path;
  const char *csv_path;
  const char *output_path;
  const char *predict_text;
  char inputs[ELML_CONFIG_LIST_MAX];
  char outputs[ELML_CONFIG_LIST_MAX];
  bool has_inputs;
  bool has_outputs;
  bool has_header;
  char delimiter;
  bool evaluate;
  bool quiet;
  bool use_stdin;
  bool json;
  bool repl;
  bool prompt;
  uint32_t limit;
  uint32_t top_k;
} elml_run_options;

static void elml_run_usage(void)
{
  printf("usage: elml_run MODEL (--csv FILE | --predict VALUES | --stdin | --repl) [options]\n"
         "\n"
         "  --csv FILE         evaluate a CSV file\n"
         "  --predict V1,V2    predict one row of comma-separated values\n"
         "  --stdin            read the prediction row from stdin\n"
         "  --repl             read rows from stdin until EOF (load model once)\n"
         "  --prompt           print a prompt to stderr in --repl mode\n"
         "  --top K            show K best classes (default 1)\n"
         "  --json             emit one-line JSON (prediction modes)\n"
         "  --inputs A,B       input columns (default: model metadata)\n"
         "  --outputs Y        evaluation target columns (default: metadata)\n"
         "  --no-header        CSV has no header row\n"
         "  --delimiter C      field delimiter (default ',')\n"
         "  --evaluate         report accuracy (classification) or error\n"
         "  --out FILE         write predictions to a CSV file\n"
         "  --limit N          process at most N data rows\n"
         "  --quiet            suppress the CSV summary\n");
}

static elml_status elml_parse_delimiter(const char *text, char *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  if (text[1] == '\0') {
    *out = text[0];
    return ELML_OK;
  }
  if (strcmp(text, "\\t") == 0) {
    *out = '\t';
    return ELML_OK;
  }
  return ELML_ERR_INVALID_ARG;
}

static elml_status elml_parse_limit(const char *text, uint32_t *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  char *end = nullptr;
  const unsigned long value = strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value > UINT32_MAX) {
    return ELML_ERR_INVALID_ARG;
  }
  *out = (uint32_t)value;
  return ELML_OK;
}

static elml_status elml_parse_top(const char *text, uint32_t *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  char *end = nullptr;
  const unsigned long value = strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value == 0ul || value > UINT32_MAX) {
    return ELML_ERR_INVALID_ARG;
  }
  *out = (uint32_t)value;
  return ELML_OK;
}

static elml_status elml_parse_cell(const char *text, float *out)
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

static elml_status elml_run_parse(elml_run_options *options, int argc,
                                  char **argv)
{
  memset(options, 0, sizeof *options);
  options->has_header = true;
  options->delimiter = ',';

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      return ELML_ERR_NOT_FOUND;
    }
    if (strcmp(arg, "--no-header") == 0) {
      options->has_header = false;
      continue;
    }
    if (strcmp(arg, "--has-header") == 0) {
      options->has_header = true;
      continue;
    }
    if (strcmp(arg, "--evaluate") == 0) {
      options->evaluate = true;
      continue;
    }
    if (strcmp(arg, "--quiet") == 0) {
      options->quiet = true;
      continue;
    }
    if (strcmp(arg, "--stdin") == 0) {
      options->use_stdin = true;
      continue;
    }
    if (strcmp(arg, "--json") == 0) {
      options->json = true;
      continue;
    }
    if (strcmp(arg, "--repl") == 0) {
      options->repl = true;
      continue;
    }
    if (strcmp(arg, "--prompt") == 0) {
      options->prompt = true;
      continue;
    }

    if (arg[0] != '-') {
      if (options->model_path != nullptr) {
        return ELML_ERR_INVALID_ARG;
      }
      options->model_path = arg;
      continue;
    }

    if (i + 1 >= argc) {
      return ELML_ERR_INVALID_ARG;
    }
    const char *value = argv[++i];

    if (strcmp(arg, "--csv") == 0) {
      options->csv_path = value;
    } else if (strcmp(arg, "--predict") == 0) {
      options->predict_text = value;
    } else if (strcmp(arg, "--out") == 0) {
      options->output_path = value;
    } else if (strcmp(arg, "--inputs") == 0) {
      const size_t len = strlen(value);
      if (len + 1u > sizeof options->inputs) {
        return ELML_ERR_BUFFER_TOO_SMALL;
      }
      memcpy(options->inputs, value, len + 1u);
      options->has_inputs = true;
    } else if (strcmp(arg, "--outputs") == 0) {
      const size_t len = strlen(value);
      if (len + 1u > sizeof options->outputs) {
        return ELML_ERR_BUFFER_TOO_SMALL;
      }
      memcpy(options->outputs, value, len + 1u);
      options->has_outputs = true;
    } else if (strcmp(arg, "--delimiter") == 0) {
      const elml_status status = elml_parse_delimiter(value, &options->delimiter);
      if (status != ELML_OK) {
        return status;
      }
    } else if (strcmp(arg, "--limit") == 0) {
      const elml_status status = elml_parse_limit(value, &options->limit);
      if (status != ELML_OK) {
        return status;
      }
    } else if (strcmp(arg, "--top") == 0) {
      const elml_status status = elml_parse_top(value, &options->top_k);
      if (status != ELML_OK) {
        return status;
      }
    } else {
      return ELML_ERR_INVALID_ARG;
    }
  }

  if (options->model_path == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  if (options->predict_text != nullptr &&
      strcmp(options->predict_text, "-") == 0) {
    options->predict_text = nullptr;
    options->use_stdin = true;
  }

  if (options->repl) {
    if (options->csv_path != nullptr || options->predict_text != nullptr ||
        options->use_stdin || options->evaluate ||
        options->output_path != nullptr || options->has_inputs ||
        options->has_outputs || options->limit != 0u) {
      return ELML_ERR_INVALID_ARG;
    }
    if (options->top_k == 0u) {
      options->top_k = 1u;
    }
    return ELML_OK;
  }

  if (options->prompt) {
    return ELML_ERR_INVALID_ARG;
  }

  const bool predict_mode =
      options->predict_text != nullptr || options->use_stdin;
  if (predict_mode) {
    if (options->csv_path != nullptr || options->evaluate ||
        options->output_path != nullptr || options->has_inputs ||
        options->has_outputs || options->limit != 0u ||
        (options->predict_text != nullptr && options->use_stdin)) {
      return ELML_ERR_INVALID_ARG;
    }
    if (options->top_k == 0u) {
      options->top_k = 1u;
    }
  } else {
    if (options->top_k != 0u || options->json || options->csv_path == nullptr) {
      return ELML_ERR_INVALID_ARG;
    }
  }
  return ELML_OK;
}

static elml_status elml_meta_string(const elml_model *model, const char *key,
                                    char *buffer, size_t capacity,
                                    bool *found)
{
  *found = false;
  size_t len = 0u;
  const elml_status status =
      elml_meta_get(model, key, buffer, capacity, &len);
  if (status == ELML_OK) {
    *found = true;
    return ELML_OK;
  }
  if (status == ELML_ERR_NOT_FOUND) {
    return ELML_OK;
  }
  return status;
}

static void elml_write_field(FILE *file, const char *text, char delimiter)
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
    fputs(text, file);
    return;
  }
  fputc('"', file);
  for (const char *cursor = text; *cursor != '\0'; ++cursor) {
    if (*cursor == '"') {
      fputc('"', file);
    }
    fputc(*cursor, file);
  }
  fputc('"', file);
}

static const char *elml_target_label(const elml_model *model, uint32_t index,
                                     char *buffer, size_t capacity)
{
  size_t len = 0u;
  if (elml_label_get(model, index, buffer, capacity, &len) == ELML_OK) {
    return buffer;
  }
  snprintf(buffer, capacity, "class_%u", index);
  return buffer;
}

static bool elml_model_is_classification(const elml_model *model)
{
  char task[32];
  size_t len = 0u;
  if (elml_meta_get(model, "task", task, sizeof task, &len) == ELML_OK) {
    return strcmp(task, "classification") == 0;
  }
  return elml_label_count(model) > 0u;
}

static bool elml_meta_token(const char *list, uint32_t index, char *out,
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

static const char *elml_output_name(const elml_model *model, uint32_t index,
                                    char *buffer, size_t capacity)
{
  char list[ELML_CONFIG_LIST_MAX];
  size_t len = 0u;
  if (elml_meta_get(model, "columns.outputs", list, sizeof list, &len) ==
          ELML_OK &&
      elml_meta_token(list, index, buffer, capacity)) {
    return buffer;
  }
  snprintf(buffer, capacity, "output_%u", index);
  return buffer;
}

static void elml_json_write_string(FILE *file, const char *text)
{
  fputc('"', file);
  for (const unsigned char *cursor = (const unsigned char *)text;
       *cursor != 0u; ++cursor) {
    const unsigned char value = *cursor;
    switch (value) {
    case '"':
      fputs("\\\"", file);
      break;
    case '\\':
      fputs("\\\\", file);
      break;
    case '\n':
      fputs("\\n", file);
      break;
    case '\r':
      fputs("\\r", file);
      break;
    case '\t':
      fputs("\\t", file);
      break;
    default:
      if (value < 0x20u) {
        fprintf(file, "\\u%04x", (unsigned)value);
      } else {
        fputc((int)value, file);
      }
      break;
    }
  }
  fputc('"', file);
}

static elml_status elml_parse_feature_row(const char *text, float *values,
                                          uint32_t dim)
{
  if (text == nullptr || values == nullptr || dim == 0u) {
    return ELML_ERR_INVALID_ARG;
  }

  const char *cursor = text;
  uint32_t index = 0u;
  for (;;) {
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r') {
      cursor++;
    }
    if (*cursor == '\0') {
      break;
    }
    if (index == dim) {
      return ELML_ERR_INVALID_ARG;
    }

    char *end = nullptr;
    const float value = strtof(cursor, &end);
    if (end == cursor || !isfinite(value)) {
      return ELML_ERR_INVALID_ARG;
    }
    values[index] = value;
    index++;

    while (*end == ' ' || *end == '\t' || *end == '\r') {
      end++;
    }
    if (*end == ',') {
      cursor = end + 1;
      if (*cursor == '\0') {
        return ELML_ERR_INVALID_ARG;
      }
    } else if (*end == '\0') {
      cursor = end;
    } else {
      return ELML_ERR_INVALID_ARG;
    }
  }

  return (index == dim) ? ELML_OK : ELML_ERR_INVALID_ARG;
}

typedef struct elml_line_reader {
  char *data;
  size_t capacity;
} elml_line_reader;

static elml_status elml_line_read(elml_line_reader *reader)
{
  if (reader == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (reader->data == nullptr || reader->capacity == 0u) {
    const size_t initial = 256u;
    char *data = (char *)malloc(initial);
    if (data == nullptr) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    reader->data = data;
    reader->capacity = initial;
  }

  size_t length = 0u;
  int value;
  while ((value = fgetc(stdin)) != EOF && value != '\n') {
    if (length + 1u >= reader->capacity) {
      const size_t next_capacity = reader->capacity * 2u;
      if (next_capacity <= reader->capacity) {
        return ELML_ERR_OUT_OF_MEMORY;
      }
      char *grown = (char *)realloc(reader->data, next_capacity);
      if (grown == nullptr) {
        return ELML_ERR_OUT_OF_MEMORY;
      }
      reader->data = grown;
      reader->capacity = next_capacity;
    }
    reader->data[length] = (char)value;
    length++;
  }

  if (value == EOF && length == 0u) {
    return ELML_ERR_NOT_FOUND;
  }
  if (length > 0u && reader->data[length - 1u] == '\r') {
    length--;
  }
  reader->data[length] = '\0';
  return ELML_OK;
}

static void elml_line_reader_free(elml_line_reader *reader)
{
  if (reader != nullptr) {
    free(reader->data);
    reader->data = nullptr;
    reader->capacity = 0u;
  }
}

typedef struct elml_predict_buffers {
  void *scratch;
  size_t scratch_len;
  float *input;
  float *prediction;
  uint32_t *order;
} elml_predict_buffers;

static void elml_predict_buffers_free(elml_predict_buffers *buffers)
{
  if (buffers != nullptr) {
    free(buffers->scratch);
    free(buffers->input);
    free(buffers->prediction);
    free(buffers->order);
    memset(buffers, 0, sizeof *buffers);
  }
}

static elml_status elml_predict_buffers_alloc(const elml_model *model,
                                              bool classification,
                                              elml_predict_buffers *buffers)
{
  if (model == nullptr || buffers == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  memset(buffers, 0, sizeof *buffers);

  const uint32_t input_dim = model->input.dim[0];
  const uint32_t output_dim = model->output.dim[0];
  if (input_dim == 0u || output_dim == 0u) {
    return ELML_ERR_BAD_SHAPE;
  }

  const size_t scratch_bytes =
      elml_align_up(elml_scratch_bytes(model, 1u), ELML_TENSOR_ALIGN);
  buffers->scratch_len = scratch_bytes;
  buffers->scratch = aligned_alloc(ELML_TENSOR_ALIGN, scratch_bytes);
  buffers->input = (float *)malloc((size_t)input_dim * sizeof(float));
  buffers->prediction = (float *)malloc((size_t)output_dim * sizeof(float));
  if (classification) {
    buffers->order =
        (uint32_t *)malloc((size_t)output_dim * sizeof(uint32_t));
  }

  if (buffers->scratch == nullptr || buffers->input == nullptr ||
      buffers->prediction == nullptr ||
      (classification && buffers->order == nullptr)) {
    elml_predict_buffers_free(buffers);
    return ELML_ERR_OUT_OF_MEMORY;
  }
  return ELML_OK;
}

static void elml_print_prediction(const elml_model *model,
                                  const elml_run_options *options,
                                  const float *prediction, uint32_t *order)
{
  const uint32_t output_dim = model->output.dim[0];
  const bool classification = elml_model_is_classification(model);
  if (!classification) {
    if (options->json) {
      printf("{\"task\":\"regression\",\"outputs\":[");
      for (uint32_t o = 0u; o < output_dim; ++o) {
        char name[128];
        if (o > 0u) {
          fputc(',', stdout);
        }
        printf("{\"name\":");
        elml_json_write_string(stdout,
                               elml_output_name(model, o, name, sizeof name));
        printf(",\"value\":%.6g}", (double)prediction[o]);
      }
      printf("]}\n");
    } else {
      for (uint32_t o = 0u; o < output_dim; ++o) {
        char name[128];
        printf("%s = %.6g\n",
               elml_output_name(model, o, name, sizeof name),
               (double)prediction[o]);
      }
    }
    return;
  }

  uint32_t top_k = options->top_k;
  if (top_k == 0u || top_k > output_dim) {
    top_k = (output_dim == 0u) ? 0u : output_dim;
    if (options->top_k == 0u && output_dim > 0u) {
      top_k = 1u;
    }
  }
  for (uint32_t i = 0u; i < output_dim; ++i) {
    order[i] = i;
  }
  for (uint32_t i = 0u; i < top_k; ++i) {
    uint32_t best = i;
    for (uint32_t j = i + 1u; j < output_dim; ++j) {
      if (prediction[order[j]] > prediction[order[best]]) {
        best = j;
      }
    }
    const uint32_t swap = order[i];
    order[i] = order[best];
    order[best] = swap;
  }

  if (options->json) {
    char label[128];
    printf("{\"task\":\"classification\",\"prediction\":");
    elml_json_write_string(
        stdout, elml_target_label(model, order[0], label, sizeof label));
    printf(",\"index\":%u,\"probability\":%.6f,\"top\":[", order[0],
           (double)prediction[order[0]]);
    for (uint32_t i = 0u; i < top_k; ++i) {
      if (i > 0u) {
        fputc(',', stdout);
      }
      printf("{\"label\":");
      elml_json_write_string(
          stdout, elml_target_label(model, order[i], label, sizeof label));
      printf(",\"index\":%u,\"probability\":%.6f}", order[i],
             (double)prediction[order[i]]);
    }
    printf("]}\n");
  } else {
    for (uint32_t i = 0u; i < top_k; ++i) {
      char label[128];
      const char *name =
          elml_target_label(model, order[i], label, sizeof label);
      if (top_k == 1u) {
        printf("%s %.6f\n", name, (double)prediction[order[i]]);
      } else {
        printf("%u. %s %.6f\n", i + 1u, name, (double)prediction[order[i]]);
      }
    }
  }
}

static int elml_run_predict(const elml_model *model,
                            const elml_run_options *options)
{
  const bool classification = elml_model_is_classification(model);

  char *stdin_row = nullptr;
  const char *row = options->predict_text;
  if (options->use_stdin) {
    elml_line_reader reader = {0};
    const elml_status status = elml_line_read(&reader);
    if (status != ELML_OK) {
      fprintf(stderr, "elml_run: cannot read a row from stdin: %s\n",
              elml_error_string(status));
      elml_line_reader_free(&reader);
      return 1;
    }
    stdin_row = reader.data;
    row = stdin_row;
  }

  elml_predict_buffers buffers;
  elml_status status =
      elml_predict_buffers_alloc(model, classification, &buffers);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: cannot allocate prediction buffers: %s\n",
            elml_error_string(status));
    free(stdin_row);
    return 1;
  }

  status = elml_parse_feature_row(row, buffers.input, model->input.dim[0]);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: expected %u comma-separated input values\n",
            model->input.dim[0]);
    free(stdin_row);
    elml_predict_buffers_free(&buffers);
    return 1;
  }

  status = elml_forward(model, buffers.scratch, buffers.scratch_len,
                        buffers.input, buffers.prediction);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: inference failed: %s\n",
            elml_error_string(status));
    free(stdin_row);
    elml_predict_buffers_free(&buffers);
    return 1;
  }

  elml_print_prediction(model, options, buffers.prediction, buffers.order);
  free(stdin_row);
  elml_predict_buffers_free(&buffers);
  return 0;
}

static int elml_run_repl(const elml_model *model,
                         const elml_run_options *options)
{
  const bool classification = elml_model_is_classification(model);
  elml_predict_buffers buffers;
  elml_status status =
      elml_predict_buffers_alloc(model, classification, &buffers);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: cannot allocate prediction buffers: %s\n",
            elml_error_string(status));
    return 1;
  }

  elml_line_reader reader = {0};
  int exit_code = 0;
  for (;;) {
    if (options->prompt) {
      fputs("> ", stderr);
      fflush(stderr);
    }

    status = elml_line_read(&reader);
    if (status == ELML_ERR_NOT_FOUND) {
      break;
    }
    if (status != ELML_OK) {
      fprintf(stderr, "elml_run: cannot read a row: %s\n",
              elml_error_string(status));
      exit_code = 1;
      break;
    }

    char *line = reader.data;
    while (*line == ' ' || *line == '\t') {
      line++;
    }
    size_t length = strlen(line);
    while (length > 0u &&
           (line[length - 1u] == ' ' || line[length - 1u] == '\t' ||
            line[length - 1u] == '\r')) {
      length--;
    }
    line[length] = '\0';

    if (length == 0u) {
      continue;
    }
    if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0 ||
        strcmp(line, ":quit") == 0 || strcmp(line, ":exit") == 0) {
      break;
    }

    status = elml_parse_feature_row(line, buffers.input, model->input.dim[0]);
    if (status != ELML_OK) {
      fprintf(stderr, "elml_run: expected %u comma-separated input values\n",
              model->input.dim[0]);
      continue;
    }

    status = elml_forward(model, buffers.scratch, buffers.scratch_len,
                          buffers.input, buffers.prediction);
    if (status != ELML_OK) {
      fprintf(stderr, "elml_run: inference failed: %s\n",
              elml_error_string(status));
      continue;
    }

    elml_print_prediction(model, options, buffers.prediction, buffers.order);
    fflush(stdout);
  }

  elml_line_reader_free(&reader);
  elml_predict_buffers_free(&buffers);
  return exit_code;
}

int main(int argc, char **argv)
{
  elml_run_options options;
  elml_status status = elml_run_parse(&options, argc, argv);
  if (status == ELML_ERR_NOT_FOUND) {
    elml_run_usage();
    return 0;
  }
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: invalid arguments\n");
    elml_run_usage();
    return 1;
  }

  uint8_t *blob = nullptr;
  size_t blob_len = 0u;
  status = elml_read_file(options.model_path, &blob, &blob_len,
                          ELML_PARAM_ALIGN);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: cannot read %s: %s\n", options.model_path,
            elml_error_string(status));
    return 1;
  }

  elml_model model;
  status = elml_model_init(&model, blob, blob_len, ELML_DTYPE_F32);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: cannot load model: %s\n",
            elml_error_string(status));
    free(blob);
    return 1;
  }

  const uint32_t input_dim = model.input.dim[0];
  const uint32_t output_dim = model.output.dim[0];
  const uint32_t label_count = elml_label_count(&model);
  if (label_count > 0u && label_count != output_dim) {
    fprintf(stderr,
            "elml_run: model metadata lists %u labels for %u outputs\n",
            label_count, output_dim);
    free(blob);
    return 1;
  }

  if (options.repl) {
    const int result = elml_run_repl(&model, &options);
    free(blob);
    return result;
  }

  if (options.predict_text != nullptr || options.use_stdin) {
    const int result = elml_run_predict(&model, &options);
    free(blob);
    return result;
  }

  elml_csv csv;
  status = elml_csv_load(options.csv_path, options.delimiter,
                         options.has_header, &csv);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: cannot read %s: %s\n", options.csv_path,
            elml_error_string(status));
    free(blob);
    return 1;
  }

  char metadata[ELML_CONFIG_LIST_MAX];
  bool have_meta = false;

  if (!options.has_inputs) {
    status = elml_meta_string(&model, "columns.inputs", metadata,
                              sizeof metadata, &have_meta);
    if (status == ELML_OK && have_meta) {
      const size_t len = strlen(metadata);
      if (len + 1u > sizeof options.inputs) {
        status = ELML_ERR_BUFFER_TOO_SMALL;
      } else {
        memcpy(options.inputs, metadata, len + 1u);
        options.has_inputs = true;
      }
    }
  }
  if (!options.has_outputs) {
    status = elml_meta_string(&model, "columns.outputs", metadata,
                              sizeof metadata, &have_meta);
    if (status == ELML_OK && have_meta) {
      const size_t len = strlen(metadata);
      if (len + 1u > sizeof options.outputs) {
        status = ELML_ERR_BUFFER_TOO_SMALL;
      } else {
        memcpy(options.outputs, metadata, len + 1u);
        options.has_outputs = true;
      }
    }
  }

  uint32_t *input_columns = nullptr;
  uint32_t input_count = 0u;
  if (options.has_inputs) {
    status = elml_columns_resolve(options.inputs, &csv, &input_columns,
                                  &input_count);
  } else {
    input_columns =
        (uint32_t *)malloc((size_t)input_dim * sizeof(uint32_t));
    if (input_columns == nullptr) {
      status = ELML_ERR_OUT_OF_MEMORY;
    } else {
      input_count = input_dim;
      for (uint32_t i = 0u; i < input_dim; ++i) {
        input_columns[i] = i;
      }
    }
  }
  if (status == ELML_OK && input_count != input_dim) {
    fprintf(stderr, "elml_run: model expects %u inputs, %u columns given\n",
            input_dim, input_count);
    status = ELML_ERR_INVALID_ARG;
  }
  if (status == ELML_OK && csv.cols < input_dim) {
    status = ELML_ERR_INVALID_ARG;
  }
  if (status != ELML_OK) {
    fprintf(stderr, "elml_run: cannot resolve input columns: %s\n",
            elml_error_string(status));
    free(input_columns);
    elml_csv_free(&csv);
    free(blob);
    return 1;
  }

  uint32_t *output_columns = nullptr;
  uint32_t output_count = 0u;
  if (options.evaluate) {
    if (!options.has_outputs) {
      fprintf(stderr, "elml_run: --evaluate needs --outputs or model metadata\n");
      free(input_columns);
      elml_csv_free(&csv);
      free(blob);
      return 1;
    }
    status = elml_columns_resolve(options.outputs, &csv, &output_columns,
                                  &output_count);
    if (status != ELML_OK) {
      fprintf(stderr, "elml_run: cannot resolve output columns: %s\n",
              elml_error_string(status));
      free(input_columns);
      elml_csv_free(&csv);
      free(blob);
      return 1;
    }
  }

  const size_t scratch_bytes = elml_scratch_bytes(&model, 1u);
  const size_t aligned_scratch = elml_align_up(scratch_bytes, ELML_TENSOR_ALIGN);
  void *scratch = aligned_alloc(ELML_TENSOR_ALIGN, aligned_scratch);
  if (scratch == nullptr) {
    fprintf(stderr, "elml_run: out of memory\n");
    free(output_columns);
    free(input_columns);
    elml_csv_free(&csv);
    free(blob);
    return 1;
  }

  float *input = (float *)malloc((size_t)input_dim * sizeof(float));
  float *prediction = (float *)malloc((size_t)output_dim * sizeof(float));
  if (input == nullptr || prediction == nullptr) {
    fprintf(stderr, "elml_run: out of memory\n");
    free(input);
    free(prediction);
    free(scratch);
    free(output_columns);
    free(input_columns);
    elml_csv_free(&csv);
    free(blob);
    return 1;
  }

  FILE *output_file = nullptr;
  if (options.output_path != nullptr) {
    output_file = fopen(options.output_path, "wb");
    if (output_file == nullptr) {
      fprintf(stderr, "elml_run: cannot write %s\n", options.output_path);
      free(input);
      free(prediction);
      free(scratch);
      free(output_columns);
      free(input_columns);
      elml_csv_free(&csv);
      free(blob);
      return 1;
    }
    if (options.has_header) {
      for (uint32_t col = 0u; col < csv.cols; ++col) {
        if (col > 0u) {
          fputc(options.delimiter, output_file);
        }
        const char *name = elml_csv_cell(&csv, 0u, col);
        elml_write_field(output_file, (name != nullptr) ? name : "",
                         options.delimiter);
      }
      if (label_count > 0u) {
        fputc(options.delimiter, output_file);
        elml_write_field(output_file, "prediction", options.delimiter);
        char label[128];
        for (uint32_t i = 0u; i < label_count; ++i) {
          char column[160];
          snprintf(column, sizeof column, "prob_%s",
                   elml_target_label(&model, i, label, sizeof label));
          fputc(options.delimiter, output_file);
          elml_write_field(output_file, column, options.delimiter);
        }
      } else {
        for (uint32_t o = 0u; o < output_dim; ++o) {
          char column[64];
          snprintf(column, sizeof column, "pred_%u", o);
          fputc(options.delimiter, output_file);
          elml_write_field(output_file, column, options.delimiter);
        }
      }
      fputc('\n', output_file);
    }
  }

  const uint32_t data_rows = csv.rows - csv.data_start;
  const uint32_t row_limit =
      (options.limit > 0u && options.limit < data_rows) ? options.limit
                                                        : data_rows;

  double loss_sum = 0.0;
  double abs_sum = 0.0;
  uint64_t loss_count = 0u;
  uint64_t correct = 0u;
  uint64_t counted = 0u;

  for (uint32_t row = 0u; row < row_limit; ++row) {
    const uint32_t csv_row = row + csv.data_start;

    for (uint32_t i = 0u; i < input_dim; ++i) {
      const char *cell = elml_csv_cell(&csv, csv_row, input_columns[i]);
      if (elml_parse_cell(cell, &input[i]) != ELML_OK) {
        fprintf(stderr, "elml_run: bad input value at row %u\n", csv_row + 1u);
        free(input);
        free(prediction);
        free(scratch);
        free(output_columns);
        free(input_columns);
        if (output_file != nullptr) {
          fclose(output_file);
        }
        elml_csv_free(&csv);
        free(blob);
        return 1;
      }
    }

    status = elml_forward(&model, scratch, aligned_scratch, input, prediction);
    if (status != ELML_OK) {
      fprintf(stderr, "elml_run: inference failed: %s\n",
              elml_error_string(status));
      free(input);
      free(prediction);
      free(scratch);
      free(output_columns);
      free(input_columns);
      if (output_file != nullptr) {
        fclose(output_file);
      }
      elml_csv_free(&csv);
      free(blob);
      return 1;
    }

    if (options.evaluate) {
      counted++;
      if (label_count > 0u) {
        const char *target =
            elml_csv_cell(&csv, csv_row, output_columns[0]);
        uint32_t best = 0u;
        for (uint32_t i = 1u; i < output_dim; ++i) {
          if (prediction[i] > prediction[best]) {
            best = i;
          }
        }
        char label[128];
        size_t len = 0u;
        if (target != nullptr) {
          for (uint32_t i = 0u; i < label_count; ++i) {
            if (elml_label_get(&model, i, label, sizeof label, &len) ==
                    ELML_OK &&
                strcmp(label, target) == 0) {
              if (i == best) {
                correct++;
              }
              break;
            }
          }
        }
      } else {
        if (output_count != output_dim) {
          fprintf(stderr,
                  "elml_run: model expects %u outputs, %u columns given\n",
                  output_dim, output_count);
          free(input);
          free(prediction);
          free(scratch);
          free(output_columns);
          free(input_columns);
          if (output_file != nullptr) {
            fclose(output_file);
          }
          elml_csv_free(&csv);
          free(blob);
          return 1;
        }
        for (uint32_t o = 0u; o < output_dim; ++o) {
          float target = 0.0f;
          if (elml_parse_cell(elml_csv_cell(&csv, csv_row, output_columns[o]),
                              &target) != ELML_OK) {
            fprintf(stderr, "elml_run: bad target value at row %u\n",
                    csv_row + 1u);
            free(input);
            free(prediction);
            free(scratch);
            free(output_columns);
            free(input_columns);
            if (output_file != nullptr) {
              fclose(output_file);
            }
            elml_csv_free(&csv);
            free(blob);
            return 1;
          }
          const double diff = (double)prediction[o] - (double)target;
          loss_sum += diff * diff;
          abs_sum += fabs(diff);
          loss_count++;
        }
      }
    }

    if (output_file != nullptr) {
      for (uint32_t col = 0u; col < csv.cols; ++col) {
        if (col > 0u) {
          fputc(options.delimiter, output_file);
        }
        const char *cell = elml_csv_cell(&csv, csv_row, col);
        elml_write_field(output_file, (cell != nullptr) ? cell : "",
                         options.delimiter);
      }
      if (label_count > 0u) {
        uint32_t best = 0u;
        for (uint32_t i = 1u; i < output_dim; ++i) {
          if (prediction[i] > prediction[best]) {
            best = i;
          }
        }
        char label[128];
        fputc(options.delimiter, output_file);
        elml_write_field(output_file,
                         elml_target_label(&model, best, label, sizeof label),
                         options.delimiter);
        for (uint32_t i = 0u; i < label_count; ++i) {
          char value[64];
          snprintf(value, sizeof value, "%.6f", (double)prediction[i]);
          fputc(options.delimiter, output_file);
          elml_write_field(output_file, value, options.delimiter);
        }
      } else {
        for (uint32_t o = 0u; o < output_dim; ++o) {
          char value[64];
          snprintf(value, sizeof value, "%.6f", (double)prediction[o]);
          fputc(options.delimiter, output_file);
          elml_write_field(output_file, value, options.delimiter);
        }
      }
      fputc('\n', output_file);
    }
  }

  if (output_file != nullptr) {
    fclose(output_file);
  }

  if (!options.quiet) {
    printf("rows: %u\n", row_limit);
    if (options.evaluate) {
      if (label_count > 0u) {
        const double accuracy =
            (counted == 0u) ? 0.0 : ((double)correct / (double)counted);
        printf("accuracy: %.4f (%llu/%llu)\n", accuracy,
               (unsigned long long)correct, (unsigned long long)counted);
      } else {
        const double mse =
            (loss_count == 0u) ? 0.0 : (loss_sum / (double)loss_count);
        const double mae =
            (loss_count == 0u) ? 0.0 : (abs_sum / (double)loss_count);
        printf("mse: %.6f\n", mse);
        printf("rmse: %.6f\n", sqrt(mse));
        printf("mae: %.6f\n", mae);
      }
    }
  }

  free(input);
  free(prediction);
  free(scratch);
  free(output_columns);
  free(input_columns);
  elml_csv_free(&csv);
  free(blob);
  return 0;
}
