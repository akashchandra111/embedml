#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "elml/types.h"
#include "test.h"
#include "train/config.h"
#include "train/csv.h"
#include "train/dataset.h"

static void configure(elml_config *config, const char *inputs,
                      const char *outputs, bool classification)
{
  elml_config_init(config);
  snprintf(config->inputs, sizeof config->inputs, "%s", inputs);
  snprintf(config->outputs, sizeof config->outputs, "%s", outputs);
  config->classification = classification;
}

static void test_regression(void)
{
  const char *text = "x1,x2,y\n1,2,3\n2,4,6\n3,6,9\n4,8,12\n";
  elml_csv csv;
  CHECK_EQ_INT(elml_csv_parse(text, strlen(text), ',', true, &csv), ELML_OK);

  elml_config config;
  configure(&config, "x1,x2", "y", false);
  elml_dataset dataset;
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset), ELML_OK);
  CHECK_EQ_INT(dataset.rows, 4u);
  CHECK_EQ_INT(dataset.input_dim, 2u);
  CHECK_EQ_INT(dataset.output_dim, 1u);
  CHECK(dataset.class_index == NULL);
  CHECK_CLOSE(dataset.x[0], 1.0, 1e-6);
  CHECK_CLOSE(dataset.x[1], 2.0, 1e-6);
  CHECK_CLOSE(dataset.x[7], 8.0, 1e-6);
  CHECK_CLOSE(dataset.y[3], 12.0, 1e-6);

  uint32_t *indices = NULL;
  uint32_t count = 0u;
  CHECK_EQ_INT(elml_columns_resolve("x1,2", &csv, &indices, &count), ELML_OK);
  CHECK_EQ_INT(count, 2u);
  CHECK_EQ_INT(indices[0], 0u);
  CHECK_EQ_INT(indices[1], 1u);
  free(indices);
  CHECK_EQ_INT(elml_columns_resolve("missing", &csv, &indices, &count),
               ELML_ERR_NOT_FOUND);
  CHECK_EQ_INT(elml_columns_resolve("", &csv, &indices, &count),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_columns_resolve("x1", NULL, &indices, &count),
               ELML_ERR_INVALID_ARG);

  elml_dataset_free(&dataset);
  configure(&config, "x1", "x1", false);
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset),
               ELML_ERR_INVALID_ARG);

  elml_dataset_free(&dataset);
  elml_dataset_free(NULL);
  elml_csv_free(&csv);
}

static void test_split_and_norm(void)
{
  const char *text = "x,y\n1,2\n3,4\n5,6\n7,8\n";
  elml_csv csv;
  CHECK_EQ_INT(elml_csv_parse(text, strlen(text), ',', true, &csv), ELML_OK);
  elml_config config;
  configure(&config, "x", "y", false);

  elml_dataset full;
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &full), ELML_OK);
  elml_dataset train;
  elml_dataset val;
  CHECK_EQ_INT(elml_dataset_split(&full, 0.5, 7u, &train, &val), ELML_OK);
  CHECK_EQ_INT(train.rows, 2u);
  CHECK_EQ_INT(val.rows, 2u);
  CHECK_EQ_INT(train.input_dim, 1u);

  elml_dataset train2;
  elml_dataset val2;
  CHECK_EQ_INT(elml_dataset_split(&full, 0.5, 7u, &train2, &val2), ELML_OK);
  CHECK_EQ_INT(train2.rows, train.rows);
  CHECK(memcmp(train.x, train2.x, (size_t)train.rows * sizeof(float)) == 0);
  CHECK(memcmp(val.x, val2.x, (size_t)val.rows * sizeof(float)) == 0);
  elml_dataset_free(&train2);
  elml_dataset_free(&val2);

  elml_dataset all;
  elml_dataset none;
  CHECK_EQ_INT(elml_dataset_split(&full, 1.0, 7u, &all, &none), ELML_OK);
  CHECK_EQ_INT(all.rows, 4u);
  CHECK_EQ_INT(none.rows, 0u);
  elml_dataset_free(&all);
  elml_dataset_free(&none);

  CHECK_EQ_INT(elml_dataset_split(&full, 0.0, 7u, &train2, &val2),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_dataset_split(NULL, 0.5, 7u, &train2, &val2),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_dataset_split(&full, 0.5, 7u, NULL, &val2),
               ELML_ERR_INVALID_ARG);

  elml_norm_stats norm;
  CHECK_EQ_INT(elml_dataset_compute_norm(&train, true, &norm), ELML_OK);
  CHECK(norm.has_input);
  CHECK(norm.has_output);
  elml_dataset_apply_norm(&train, &norm);
  for (uint32_t row = 0u; row < train.rows; ++row) {
    CHECK_CLOSE(train.x[row], (row == 0u) ? -1.0 : 1.0, 1e-5);
    CHECK_CLOSE(train.y[row], (row == 0u) ? -1.0 : 1.0, 1e-5);
  }
  elml_norm_stats_free(&norm);
  elml_norm_stats_free(NULL);

  CHECK_EQ_INT(elml_dataset_compute_norm(NULL, false, &norm),
               ELML_ERR_INVALID_ARG);

  elml_dataset_free(&full);
  elml_dataset_free(&train);
  elml_dataset_free(&val);
  elml_csv_free(&csv);
}

static void test_classification(void)
{
  const char *text =
      "a,b,label\n1,2,cat\n2,1,dog\n3,3,cat\n4,4,dog\n5,5,cat\n";
  elml_csv csv;
  CHECK_EQ_INT(elml_csv_parse(text, strlen(text), ',', true, &csv), ELML_OK);
  elml_config config;
  configure(&config, "a,b", "label", true);

  elml_dataset dataset;
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset), ELML_OK);
  CHECK_EQ_INT(dataset.rows, 5u);
  CHECK_EQ_INT(dataset.output_dim, 2u);
  CHECK_EQ_INT(dataset.label_count, 2u);
  CHECK_STR(dataset.labels[0], "cat");
  CHECK_STR(dataset.labels[1], "dog");
  CHECK_EQ_INT(dataset.class_index[0], 0u);
  CHECK_EQ_INT(dataset.class_index[1], 1u);
  CHECK_CLOSE(dataset.y[0], 1.0, 1e-6);
  CHECK_CLOSE(dataset.y[1], 0.0, 1e-6);
  CHECK_CLOSE(dataset.y[2], 0.0, 1e-6);
  CHECK_CLOSE(dataset.y[3], 1.0, 1e-6);

  elml_dataset_free(&dataset);
  configure(&config, "a", "a,label", true);
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset),
               ELML_ERR_INVALID_ARG);
  configure(&config, "a", "label", false);
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset), ELML_ERR_INVALID_ARG);

  elml_dataset_free(&dataset);
  elml_csv_free(&csv);

  const char *single = "x,label\n1,cat\n2,cat\n";
  CHECK_EQ_INT(elml_csv_parse(single, strlen(single), ',', true, &csv),
               ELML_OK);
  configure(&config, "x", "label", true);
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset),
               ELML_ERR_INVALID_ARG);
  elml_csv_free(&csv);

  /* A comma inside a label would corrupt the comma-separated metadata field. */
  const char *comma = "x,label\n1,\"cat,dog\"\n2,bird\n";
  CHECK_EQ_INT(elml_csv_parse(comma, strlen(comma), ',', true, &csv),
               ELML_OK);
  configure(&config, "x", "label", true);
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset),
               ELML_ERR_INVALID_ARG);
  elml_csv_free(&csv);
}

static void test_missing(void)
{
  const char *text = "a,b,y\n1,,3\n3,5,9\n";
  elml_csv csv;
  CHECK_EQ_INT(elml_csv_parse(text, strlen(text), ',', true, &csv), ELML_OK);
  elml_config config;
  configure(&config, "a,b", "y", false);

  elml_dataset dataset;
  config.missing = ELML_MISSING_ERROR;
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset),
               ELML_ERR_INVALID_ARG);

  config.missing = ELML_MISSING_ZERO;
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset), ELML_OK);
  CHECK_CLOSE(dataset.x[1], 0.0, 1e-6);
  elml_dataset_free(&dataset);

  config.missing = ELML_MISSING_MEAN;
  CHECK_EQ_INT(elml_dataset_build(&csv, &config, &dataset), ELML_OK);
  CHECK_CLOSE(dataset.x[1], 5.0, 1e-6);
  elml_dataset_free(&dataset);

  CHECK_EQ_INT(elml_dataset_build(NULL, &config, &dataset),
               ELML_ERR_INVALID_ARG);
  elml_csv_free(&csv);
}

int main(void)
{
  test_regression();
  test_split_and_norm();
  test_classification();
  test_missing();
  TEST_END();
}
