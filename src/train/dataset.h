#ifndef ELML_DATASET_H
#define ELML_DATASET_H

#include <stdbool.h>
#include <stdint.h>

#include "elml/types.h"
#include "train/config.h"
#include "train/csv.h"

typedef struct elml_dataset {
  uint32_t rows;
  uint32_t input_dim;
  uint32_t output_dim;
  float *x;           /* rows * input_dim */
  float *y;           /* rows * output_dim */
  uint8_t *class_index; /* classification: rows, else NULL */
  char **labels;      /* classification: output_dim strings */
  uint32_t label_count;
} elml_dataset;

typedef struct elml_norm_stats {
  bool has_input;
  float *input_mean;
  float *input_std;
  bool has_output;
  float *output_mean;
  float *output_std;
} elml_norm_stats;

elml_status elml_dataset_build(const elml_csv *csv, const elml_config *config,
                               elml_dataset *out);

/* Resolve a comma-separated list of header names or 1-based column indices. */
elml_status elml_columns_resolve(const char *list, const elml_csv *csv,
                                 uint32_t **indices, uint32_t *count);

elml_status elml_dataset_split(const elml_dataset *full, double train_fraction,
                               uint64_t seed, elml_dataset *train,
                               elml_dataset *val);

elml_status elml_dataset_compute_norm(const elml_dataset *train,
                                      bool normalize_output,
                                      elml_norm_stats *out);
void elml_dataset_apply_norm(elml_dataset *dataset,
                             const elml_norm_stats *stats);
void elml_norm_stats_free(elml_norm_stats *stats);

void elml_dataset_free(elml_dataset *dataset);

uint64_t elml_rng_next(uint64_t *state);

#endif /* ELML_DATASET_H */
