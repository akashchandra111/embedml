#include "train/dataset.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ckdint_compat.h"
#include "elml/types.h"
#include "train/config.h"
#include "train/csv.h"

#define ELML_MAX_COLUMNS 1024u

static bool elml_parse_float(const char *text, float *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const float value = strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
    return false;
  }
  *out = value;
  return true;
}

static char *elml_strdup_local(const char *text)
{
  const size_t len = strlen(text);
  char *copy = (char *)malloc(len + 1u);
  if (copy == nullptr) {
    return nullptr;
  }
  memcpy(copy, text, len + 1u);
  return copy;
}

static int elml_find_column(const elml_csv *csv, const char *name)
{
  if (csv == nullptr || name == nullptr || *name == '\0') {
    return -1;
  }

  if (csv->data_start == 1u) {
    for (uint32_t col = 0u; col < csv->cols; ++col) {
      const char *header = elml_csv_cell(csv, 0u, col);
      if (header != nullptr && strcmp(header, name) == 0) {
        return (int)col;
      }
    }
  }

  /* Fall back to a 1-based index, useful for headerless files. */
  errno = 0;
  char *end = nullptr;
  const unsigned long index = strtoul(name, &end, 10);
  if (errno == 0 && end != name && *end == '\0' && index >= 1u &&
      index <= csv->cols) {
    return (int)(index - 1u);
  }
  return -1;
}

elml_status elml_columns_resolve(const char *list, const elml_csv *csv,
                                 uint32_t **indices, uint32_t *count)
{
  if (list == nullptr || csv == nullptr || indices == nullptr ||
      count == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  *indices = nullptr;
  *count = 0u;

  uint32_t capacity = 8u;
  uint32_t *result = (uint32_t *)malloc((size_t)capacity * sizeof(uint32_t));
  if (result == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }

  const char *cursor = list;
  elml_status status = ELML_OK;

  while (*cursor != '\0') {
    while (*cursor == ' ' || *cursor == '\t') {
      cursor++;
    }
    if (*cursor == '\0') {
      break;
    }

    const char *end = cursor;
    while (*end != '\0' && *end != ',') {
      end++;
    }
    const char *trimmed = end;
    while (trimmed > cursor && (trimmed[-1] == ' ' || trimmed[-1] == '\t')) {
      trimmed--;
    }
    const size_t len = (size_t)(trimmed - cursor);
    if (len == 0u || len >= 128u) {
      status = ELML_ERR_INVALID_ARG;
      break;
    }

    char name[128];
    memcpy(name, cursor, len);
    name[len] = '\0';

    const int column = elml_find_column(csv, name);
    if (column < 0) {
      status = ELML_ERR_NOT_FOUND;
      break;
    }

    if (*count == capacity) {
      const uint32_t next_capacity = capacity * 2u;
      if (next_capacity < capacity) {
        status = ELML_ERR_OUT_OF_MEMORY;
        break;
      }
      void *next =
          realloc(result, (size_t)next_capacity * sizeof(uint32_t));
      if (next == nullptr) {
        status = ELML_ERR_OUT_OF_MEMORY;
        break;
      }
      result = (uint32_t *)next;
      capacity = next_capacity;
    }
    result[*count] = (uint32_t)column;
    (*count)++;

    cursor = (*end == ',') ? end + 1 : end;
  }

  if (status != ELML_OK || *count == 0u) {
    free(result);
    *indices = nullptr;
    *count = 0u;
    return (status == ELML_OK) ? ELML_ERR_INVALID_ARG : status;
  }

  *indices = result;
  return ELML_OK;
}

static bool elml_indices_contain(const uint32_t *indices, uint32_t count,
                                 uint32_t value)
{
  for (uint32_t i = 0u; i < count; ++i) {
    if (indices[i] == value) {
      return true;
    }
  }
  return false;
}

static int elml_cmp_strings(const void *left, const void *right)
{
  const char *const *a = (const char *const *)left;
  const char *const *b = (const char *const *)right;
  return strcmp(*a, *b);
}

static elml_status elml_labels_add(char ***labels, uint32_t *count,
                                   uint32_t *capacity, const char *value)
{
  /* Labels are stored as a comma-separated metadata string, and the class
   * index is a uint8_t, so both delimiters and >255 classes are rejected. */
  if (strchr(value, ',') != nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0u; i < *count; ++i) {
    if (strcmp((*labels)[i], value) == 0) {
      return ELML_OK;
    }
  }
  if (*count >= 255u) {
    return ELML_ERR_INVALID_ARG;
  }

  if (*count == *capacity) {
    uint32_t next_capacity = (*capacity == 0u) ? 8u : *capacity * 2u;
    if (next_capacity < *capacity) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    void *next = realloc(*labels, (size_t)next_capacity * sizeof(char *));
    if (next == nullptr) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    *labels = (char **)next;
    *capacity = next_capacity;
  }

  char *copy = elml_strdup_local(value);
  if (copy == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  (*labels)[*count] = copy;
  (*count)++;
  return ELML_OK;
}

static elml_status elml_copy_labels(const elml_dataset *source,
                                    elml_dataset *destination)
{
  destination->label_count = 0u;
  destination->labels = nullptr;
  if (source->label_count == 0u) {
    return ELML_OK;
  }

  char **labels =
      (char **)malloc((size_t)source->label_count * sizeof(char *));
  if (labels == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  for (uint32_t i = 0u; i < source->label_count; ++i) {
    const size_t len = strlen(source->labels[i]);
    labels[i] = (char *)malloc(len + 1u);
    if (labels[i] == nullptr) {
      for (uint32_t j = 0u; j < i; ++j) {
        free(labels[j]);
      }
      free(labels);
      return ELML_ERR_OUT_OF_MEMORY;
    }
    memcpy(labels[i], source->labels[i], len + 1u);
  }
  destination->labels = labels;
  destination->label_count = source->label_count;
  return ELML_OK;
}

static elml_status elml_alloc_dataset(elml_dataset *dataset, uint32_t rows,
                                      uint32_t input_dim, uint32_t output_dim)
{
  memset(dataset, 0, sizeof *dataset);
  dataset->rows = rows;
  dataset->input_dim = input_dim;
  dataset->output_dim = output_dim;
  if (rows == 0u || input_dim == 0u || output_dim == 0u) {
    return ELML_ERR_INVALID_ARG;
  }

  size_t x_count;
  size_t y_count;
  if (ckd_mul(&x_count, (size_t)rows, (size_t)input_dim) ||
      ckd_mul(&y_count, (size_t)rows, (size_t)output_dim)) {
    return ELML_ERR_OUT_OF_MEMORY;
  }

  dataset->x = (float *)malloc(x_count * sizeof(float));
  dataset->y = (float *)malloc(y_count * sizeof(float));
  if (dataset->x == nullptr || dataset->y == nullptr) {
    free(dataset->x);
    free(dataset->y);
    dataset->x = nullptr;
    dataset->y = nullptr;
    return ELML_ERR_OUT_OF_MEMORY;
  }
  return ELML_OK;
}

static elml_status elml_compute_input_means(const elml_csv *csv,
                                            const uint32_t *input_columns,
                                            uint32_t input_dim,
                                            double *means)
{
  const uint32_t rows = csv->rows - csv->data_start;
  double *sums = (double *)calloc(input_dim, sizeof(double));
  uint32_t *counts = (uint32_t *)calloc(input_dim, sizeof(uint32_t));
  if (sums == nullptr || counts == nullptr) {
    free(sums);
    free(counts);
    return ELML_ERR_OUT_OF_MEMORY;
  }

  elml_status status = ELML_OK;
  for (uint32_t row = 0u; row < rows; ++row) {
    for (uint32_t col = 0u; col < input_dim; ++col) {
      const char *cell =
          elml_csv_cell(csv, row + csv->data_start, input_columns[col]);
      if (cell == nullptr || *cell == '\0') {
        continue;
      }
      float value;
      if (!elml_parse_float(cell, &value)) {
        status = ELML_ERR_INVALID_ARG;
        goto done;
      }
      sums[col] += (double)value;
      counts[col]++;
    }
  }

  for (uint32_t col = 0u; col < input_dim; ++col) {
    means[col] = (counts[col] == 0u) ? 0.0 : (sums[col] / (double)counts[col]);
  }

done:
  free(sums);
  free(counts);
  return status;
}

elml_status elml_dataset_build(const elml_csv *csv, const elml_config *config,
                               elml_dataset *out)
{
  if (csv == nullptr || config == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof *out);

  if (csv->rows <= csv->data_start) {
    return ELML_ERR_TRUNCATED;
  }

  uint32_t *input_columns = nullptr;
  uint32_t *output_columns = nullptr;
  uint32_t input_dim = 0u;
  uint32_t output_count = 0u;

  elml_status status =
      elml_columns_resolve(config->inputs, csv, &input_columns, &input_dim);
  if (status != ELML_OK) {
    return status;
  }
  status = elml_columns_resolve(config->outputs, csv, &output_columns,
                                &output_count);
  if (status != ELML_OK) {
    free(input_columns);
    return status;
  }

  for (uint32_t i = 0u; i < output_count; ++i) {
    if (elml_indices_contain(input_columns, input_dim, output_columns[i])) {
      status = ELML_ERR_INVALID_ARG;
      goto cleanup;
    }
  }
  if (config->classification && output_count != 1u) {
    status = ELML_ERR_INVALID_ARG;
    goto cleanup;
  }

  const uint32_t rows = csv->rows - csv->data_start;
  double *input_means = (double *)malloc((size_t)input_dim * sizeof(double));
  if (input_means == nullptr) {
    status = ELML_ERR_OUT_OF_MEMORY;
    goto cleanup;
  }

  status = elml_compute_input_means(csv, input_columns, input_dim, input_means);
  if (status != ELML_OK) {
    free(input_means);
    goto cleanup;
  }

  char **labels = nullptr;
  uint32_t label_count = 0u;
  uint32_t label_capacity = 0u;

  if (config->classification) {
    for (uint32_t row = 0u; row < rows; ++row) {
      const char *cell =
          elml_csv_cell(csv, row + csv->data_start, output_columns[0]);
      if (cell == nullptr || *cell == '\0') {
        status = ELML_ERR_INVALID_ARG;
        goto cleanup_means;
      }
      status = elml_labels_add(&labels, &label_count, &label_capacity, cell);
      if (status != ELML_OK) {
        goto cleanup_means;
      }
    }
    if (label_count < 2u) {
      status = ELML_ERR_INVALID_ARG;
      goto cleanup_means;
    }
    qsort(labels, label_count, sizeof(char *), elml_cmp_strings);
  }

  const uint32_t output_dim = config->classification ? label_count : output_count;
  status = elml_alloc_dataset(out, rows, input_dim, output_dim);
  if (status != ELML_OK) {
    goto cleanup_means;
  }

  if (config->classification) {
    out->class_index = (uint8_t *)malloc((size_t)rows);
    if (out->class_index == nullptr) {
      status = ELML_ERR_OUT_OF_MEMORY;
      goto cleanup_dataset;
    }
  }

  for (uint32_t row = 0u; row < rows; ++row) {
    for (uint32_t col = 0u; col < input_dim; ++col) {
      const char *cell =
          elml_csv_cell(csv, row + csv->data_start, input_columns[col]);
      float value = 0.0f;
      if (cell == nullptr || *cell == '\0') {
        if (config->missing == ELML_MISSING_ERROR) {
          status = ELML_ERR_INVALID_ARG;
          goto cleanup_dataset;
        }
        value = (config->missing == ELML_MISSING_MEAN)
                    ? (float)input_means[col]
                    : 0.0f;
      } else if (!elml_parse_float(cell, &value)) {
        status = ELML_ERR_INVALID_ARG;
        goto cleanup_dataset;
      }
      out->x[(size_t)row * input_dim + col] = value;
    }

    if (config->classification) {
      const char *cell =
          elml_csv_cell(csv, row + csv->data_start, output_columns[0]);
      uint32_t index = 0u;
      while (index < label_count && strcmp(labels[index], cell) != 0) {
        index++;
      }
      if (index == label_count) {
        status = ELML_ERR_INVALID_ARG;
        goto cleanup_dataset;
      }
      out->class_index[row] = (uint8_t)index;
      for (uint32_t col = 0u; col < output_dim; ++col) {
        out->y[(size_t)row * output_dim + col] =
            (col == index) ? 1.0f : 0.0f;
      }
    } else {
      for (uint32_t col = 0u; col < output_dim; ++col) {
        const char *cell =
            elml_csv_cell(csv, row + csv->data_start, output_columns[col]);
        float value;
        if (cell == nullptr || *cell == '\0') {
          if (config->missing == ELML_MISSING_ERROR) {
            status = ELML_ERR_INVALID_ARG;
            goto cleanup_dataset;
          }
          value = 0.0f;
        } else if (!elml_parse_float(cell, &value)) {
          status = ELML_ERR_INVALID_ARG;
          goto cleanup_dataset;
        }
        out->y[(size_t)row * output_dim + col] = value;
      }
    }
  }

  out->labels = labels;
  out->label_count = label_count;
  labels = nullptr;
  free(input_means);
  free(input_columns);
  free(output_columns);
  return ELML_OK;

cleanup_dataset:
  elml_dataset_free(out);
cleanup_means:
  free(input_means);
  for (uint32_t i = 0u; i < label_count; ++i) {
    free(labels[i]);
  }
  free(labels);
cleanup:
  free(input_columns);
  free(output_columns);
  return status;
}

uint64_t elml_rng_next(uint64_t *state)
{
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 2685821657736338717ull;
}

static void elml_shuffle_indices(uint32_t *indices, uint32_t count,
                                 uint64_t *state)
{
  for (uint32_t i = count; i > 1u; --i) {
    const uint32_t j = (uint32_t)(elml_rng_next(state) % (uint64_t)i);
    const uint32_t swap = indices[i - 1u];
    indices[i - 1u] = indices[j];
    indices[j] = swap;
  }
}

static elml_status elml_copy_rows(const elml_dataset *source,
                                  const uint32_t *indices, uint32_t rows,
                                  elml_dataset *destination)
{
  const elml_status status =
      elml_alloc_dataset(destination, rows, source->input_dim,
                         source->output_dim);
  if (status != ELML_OK) {
    return status;
  }

  for (uint32_t row = 0u; row < rows; ++row) {
    const uint32_t source_row = indices[row];
    memcpy(&destination->x[(size_t)row * source->input_dim],
           &source->x[(size_t)source_row * source->input_dim],
           (size_t)source->input_dim * sizeof(float));
    memcpy(&destination->y[(size_t)row * source->output_dim],
           &source->y[(size_t)source_row * source->output_dim],
           (size_t)source->output_dim * sizeof(float));
    if (source->class_index != nullptr) {
      if (destination->class_index == nullptr) {
        destination->class_index = (uint8_t *)malloc((size_t)rows);
        if (destination->class_index == nullptr) {
          elml_dataset_free(destination);
          return ELML_ERR_OUT_OF_MEMORY;
        }
      }
      destination->class_index[row] = source->class_index[source_row];
    }
  }

  return elml_copy_labels(source, destination);
}

elml_status elml_dataset_split(const elml_dataset *full, double train_fraction,
                               uint64_t seed, elml_dataset *train,
                               elml_dataset *val)
{
  if (full == nullptr || train == nullptr || val == nullptr || full->rows == 0u ||
      train_fraction <= 0.0 || train_fraction > 1.0) {
    return ELML_ERR_INVALID_ARG;
  }
  memset(train, 0, sizeof *train);
  memset(val, 0, sizeof *val);

  uint32_t *indices =
      (uint32_t *)malloc((size_t)full->rows * sizeof(uint32_t));
  if (indices == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  for (uint32_t i = 0u; i < full->rows; ++i) {
    indices[i] = i;
  }
  uint64_t state = seed ^ 0x9E3779B97F4A7C15ull;
  elml_shuffle_indices(indices, full->rows, &state);

  uint32_t train_rows =
      (uint32_t)((double)full->rows * train_fraction + 0.5);
  if (train_rows == 0u) {
    train_rows = 1u;
  }
  if (train_rows >= full->rows) {
    train_rows = full->rows;
  }
  const uint32_t val_rows = full->rows - train_rows;
  if (val_rows == 0u && train_fraction < 1.0) {
    free(indices);
    return ELML_ERR_INVALID_ARG;
  }

  elml_status status = elml_copy_rows(full, indices, train_rows, train);
  if (status != ELML_OK) {
    free(indices);
    return status;
  }

  if (val_rows > 0u) {
    status = elml_copy_rows(full, indices + train_rows, val_rows, val);
    if (status != ELML_OK) {
      elml_dataset_free(train);
      free(indices);
      return status;
    }
  } else {
    memset(val, 0, sizeof *val);
  }

  free(indices);
  return ELML_OK;
}

elml_status elml_dataset_compute_norm(const elml_dataset *train,
                                      bool normalize_output,
                                      elml_norm_stats *out)
{
  if (train == nullptr || out == nullptr || train->rows == 0u) {
    return ELML_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof *out);

  out->input_mean =
      (float *)calloc(train->input_dim, sizeof(float));
  out->input_std = (float *)calloc(train->input_dim, sizeof(float));
  if (out->input_mean == nullptr || out->input_std == nullptr) {
    elml_norm_stats_free(out);
    return ELML_ERR_OUT_OF_MEMORY;
  }

  for (uint32_t col = 0u; col < train->input_dim; ++col) {
    double sum = 0.0;
    for (uint32_t row = 0u; row < train->rows; ++row) {
      sum += (double)train->x[(size_t)row * train->input_dim + col];
    }
    const double mean = sum / (double)train->rows;
    double variance = 0.0;
    for (uint32_t row = 0u; row < train->rows; ++row) {
      const double diff =
          (double)train->x[(size_t)row * train->input_dim + col] - mean;
      variance += diff * diff;
    }
    variance /= (double)train->rows;
    const double stddev = sqrt(variance);
    out->input_mean[col] = (float)mean;
    out->input_std[col] = (stddev < 1e-6) ? 1.0f : (float)stddev;
  }
  out->has_input = true;

  if (normalize_output) {
    out->output_mean =
        (float *)calloc(train->output_dim, sizeof(float));
    out->output_std = (float *)calloc(train->output_dim, sizeof(float));
    if (out->output_mean == nullptr || out->output_std == nullptr) {
      elml_norm_stats_free(out);
      return ELML_ERR_OUT_OF_MEMORY;
    }
    for (uint32_t col = 0u; col < train->output_dim; ++col) {
      double sum = 0.0;
      for (uint32_t row = 0u; row < train->rows; ++row) {
        sum += (double)train->y[(size_t)row * train->output_dim + col];
      }
      const double mean = sum / (double)train->rows;
      double variance = 0.0;
      for (uint32_t row = 0u; row < train->rows; ++row) {
        const double diff =
            (double)train->y[(size_t)row * train->output_dim + col] - mean;
        variance += diff * diff;
      }
      variance /= (double)train->rows;
      const double stddev = sqrt(variance);
      out->output_mean[col] = (float)mean;
      out->output_std[col] = (stddev < 1e-6) ? 1.0f : (float)stddev;
    }
    out->has_output = true;
  }

  return ELML_OK;
}

void elml_dataset_apply_norm(elml_dataset *dataset,
                             const elml_norm_stats *stats)
{
  if (dataset == nullptr || stats == nullptr || !stats->has_input) {
    return;
  }
  for (uint32_t row = 0u; row < dataset->rows; ++row) {
    for (uint32_t col = 0u; col < dataset->input_dim; ++col) {
      const float stddev = stats->input_std[col];
      dataset->x[(size_t)row * dataset->input_dim + col] =
          (stddev == 0.0f)
              ? dataset->x[(size_t)row * dataset->input_dim + col]
              : ((dataset->x[(size_t)row * dataset->input_dim + col] -
                  stats->input_mean[col]) /
                 stddev);
    }
  }
  if (stats->has_output) {
    for (uint32_t row = 0u; row < dataset->rows; ++row) {
      for (uint32_t col = 0u; col < dataset->output_dim; ++col) {
        const float stddev = stats->output_std[col];
        dataset->y[(size_t)row * dataset->output_dim + col] =
            (stddev == 0.0f)
                ? dataset->y[(size_t)row * dataset->output_dim + col]
                : ((dataset->y[(size_t)row * dataset->output_dim + col] -
                    stats->output_mean[col]) /
                   stddev);
      }
    }
  }
}

void elml_norm_stats_free(elml_norm_stats *stats)
{
  if (stats == nullptr) {
    return;
  }
  free(stats->input_mean);
  free(stats->input_std);
  free(stats->output_mean);
  free(stats->output_std);
  memset(stats, 0, sizeof *stats);
}

void elml_dataset_free(elml_dataset *dataset)
{
  if (dataset == nullptr) {
    return;
  }
  free(dataset->x);
  free(dataset->y);
  free(dataset->class_index);
  for (uint32_t i = 0u; i < dataset->label_count; ++i) {
    free(dataset->labels[i]);
  }
  free(dataset->labels);
  memset(dataset, 0, sizeof *dataset);
}
