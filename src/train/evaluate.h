/*
 * Host-only evaluation of a saved model over a CSV file.
 *
 * `elml_run` and the studio's Evaluate tab share this module so per-row
 * predictions, column resolution, and the accuracy or error summary stay
 * identical in both. Like the training session, it is host code and is not part
 * of the allocation-free inference API.
 */
#ifndef ELML_EVALUATE_H
#define ELML_EVALUATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/model.h"
#include "elml/types.h"
#include "train/config.h"
#include "train/csv.h"

typedef struct elml_eval_options {
  const char *model_path;
  const char *csv_path;
  char delimiter;       /* field delimiter; '\0' means ',' */
  bool has_header;
  const char *inputs;   /* names or 1-based indices; NULL/"" = metadata */
  const char *targets;  /* target columns; NULL/"" uses model metadata */
  bool score;           /* resolve targets and compute accuracy or error */
  bool skip_bad_rows;   /* skip rows that do not parse instead of failing */
  uint32_t limit;       /* evaluate at most this many data rows; 0 means all */
} elml_eval_options;

typedef struct elml_eval_row {
  uint32_t line;      /* 1-based CSV record, header included */
  uint32_t predicted; /* top class for classification, otherwise 0 */
  int32_t expected;   /* matched target class, or -1 when unknown */
  float confidence;   /* probability of the top class, or the first output */
  bool correct;
} elml_eval_row;

typedef struct elml_eval_result {
  uint8_t *blob;
  size_t blob_len;
  elml_model model;
  elml_csv csv;

  uint32_t *input_columns;
  uint32_t input_count;
  uint32_t *target_columns;
  uint32_t target_count;

  float *values;  /* row_count * output_dim predictions */
  float *targets; /* row_count * output_dim parsed targets, when scored */
  elml_eval_row *rows;
  uint32_t row_count;
  uint32_t output_dim;
  uint32_t data_rows; /* data rows available before the limit */
  uint32_t limit;

  uint64_t skipped;
  bool has_header;
  char delimiter;
  bool classification;
  bool scored;
  uint64_t scored_rows;
  uint64_t correct;
  double mse;
  double mae;

  char inputs_text[ELML_CONFIG_LIST_MAX];
  char targets_text[ELML_CONFIG_LIST_MAX];
  char skipped_note[192];
  char message[512];
} elml_eval_result;

/*
 * Load the model, read the CSV, resolve the columns, and predict every row.
 * On failure the message field explains why; the caller still owns the result
 * and must release it with elml_eval_free().
 */
elml_status elml_eval_run(const elml_eval_options *options,
                          elml_eval_result *result);
void elml_eval_free(elml_eval_result *result);

/* Model metadata helpers shared with the CLI's single-row output. */
bool elml_eval_is_classification(const elml_model *model);
const char *elml_eval_class_label(const elml_model *model, uint32_t index,
                                  char *buffer, size_t capacity);
const char *elml_eval_output_name(const elml_model *model, uint32_t index,
                                  char *buffer, size_t capacity);

/* Row accessors; NULL when the row or the block is absent. */
const float *elml_eval_values(const elml_eval_result *result, uint32_t row);
const float *elml_eval_targets(const elml_eval_result *result, uint32_t row);

/* Write the source columns plus the predictions as a CSV file. */
elml_status elml_eval_write_csv(const elml_eval_result *result,
                                const char *path);

#endif /* ELML_EVALUATE_H */
