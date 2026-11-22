#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "elml/arena.h"
#include "elml/model.h"
#include "elml/types.h"
#include "format/write.h"
#include "test.h"
#include "train/evaluate.h"
#include "train/io.h"

static int make_temp(char *path)
{
  return mkstemp(path);
}

static bool write_text(const char *path, const char *text)
{
  return elml_write_file(path, text, strlen(text)) == ELML_OK;
}

static bool write_model(const char *path, const elml_write_layer *layers,
                        uint32_t layer_count, uint32_t input_dim,
                        uint32_t output_dim, const elml_write_meta *meta,
                        uint32_t meta_count)
{
  const elml_write_spec spec = {
      .input_dim = input_dim,
      .output_dim = output_dim,
      .layer_count = (uint16_t)layer_count,
      .layers = layers,
      .meta = meta,
      .meta_count = meta_count,
  };
  elml_buf buffer = {0};
  if (elml_model_write(&spec, &buffer) != ELML_OK) {
    return false;
  }
  const bool ok = elml_write_file(path, buffer.data, buffer.len) == ELML_OK;
  elml_buf_free(&buffer);
  return ok;
}

static bool contains(const uint8_t *haystack, size_t hay_len,
                     const char *needle)
{
  const size_t needle_len = strlen(needle);
  if (needle_len == 0u || hay_len < needle_len) {
    return false;
  }
  for (size_t i = 0; i + needle_len <= hay_len; ++i) {
    if (memcmp(haystack + i, needle, needle_len) == 0) {
      return true;
    }
  }
  return false;
}

static elml_eval_options options(const char *model_path, const char *csv_path)
{
  elml_eval_options opts = {
      .model_path = model_path,
      .csv_path = csv_path,
      .delimiter = ',',
      .has_header = true,
      .score = true,
  };
  return opts;
}

int main(void)
{
  char class_model[] = "elml_eval_classmodel_XXXXXX";
  char class_csv[] = "elml_eval_classdata_XXXXXX";
  char bad_csv[] = "elml_eval_baddata_XXXXXX";
  char skip_csv[] = "elml_eval_skipdata_XXXXXX";
  char plain_model[] = "elml_eval_plainmodel_XXXXXX";
  char reg_model[] = "elml_eval_regmodel_XXXXXX";
  char reg_csv[] = "elml_eval_regdata_XXXXXX";
  char out_csv[] = "elml_eval_outdata_XXXXXX";

  const int fds[] = {
      make_temp(class_model),
      make_temp(class_csv),
      make_temp(bad_csv),
      make_temp(skip_csv),
      make_temp(plain_model),
      make_temp(reg_model),
      make_temp(reg_csv),
      make_temp(out_csv),
  };
  bool created = true;
  for (size_t i = 0; i < sizeof fds / sizeof fds[0]; ++i) {
    created = created && fds[i] >= 0;
    if (fds[i] >= 0) {
      close(fds[i]);
    }
  }
  CHECK(created);
  if (!created) {
    TEST_END();
  }

  /* A two-input identity layer whose argmax is the larger input. */
  static const float identity[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  static const float no_bias[2] = {0.0f, 0.0f};
  static const elml_write_layer class_layers[1] = {
      {2u, 2u, ELML_ACT_LINEAR, 0.0f, identity, no_bias},
  };
  static const elml_write_meta class_meta[4] = {
      {"labels", "left,right"},
      {"task", "classification"},
      {"columns.inputs", "x1,x2"},
      {"columns.outputs", "species"},
  };
  if (created) {
    CHECK(write_model(class_model, class_layers, 1u, 2u, 2u, class_meta, 4u));
    CHECK(write_text(class_csv,
                     "x1,x2,species\n"
                     "1.0,0.0,left\n"
                     "0.0,1.0,right\n"
                     "2.0,1.0,right\n"));
    CHECK(write_text(bad_csv,
                     "x1,x2,species\n"
                     "1.0,0.0,left\n"
                     "bad,1.0,right\n"
                     "2.0,1.0,left\n"));
    CHECK(write_text(skip_csv,
                     "x1,x2,species\n"
                     "1.0,0.0,left\n"
                     "bad,1.0,right\n"
                     "2.0,1.0,left\n"));
  }

  elml_eval_result result;
  elml_eval_options opts = options(class_model, class_csv);

  /* Argument validation. */
  CHECK_EQ_INT(elml_eval_run(NULL, &result), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_eval_run(&opts, NULL), ELML_ERR_INVALID_ARG);
  const elml_eval_options empty = {0};
  CHECK_EQ_INT(elml_eval_run(&empty, &result), ELML_ERR_INVALID_ARG);
  CHECK(result.message[0] != '\0');
  opts.model_path = "does-not-exist.elml";
  CHECK(elml_eval_run(&opts, &result) != ELML_OK);
  CHECK(strstr(result.message, "cannot read") != NULL);
  elml_eval_free(&result);
  opts = options(class_model, class_csv);

  /* Metadata fallback: inputs and targets are read from the model. */
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_OK);
  CHECK(result.classification);
  CHECK(result.scored);
  CHECK_EQ_INT(result.row_count, 3u);
  CHECK_EQ_INT(result.data_rows, 3u);
  CHECK_EQ_INT(result.scored_rows, 3u);
  CHECK_EQ_INT(result.correct, 2u);
  CHECK_EQ_INT(result.output_dim, 2u);
  CHECK_STR(result.inputs_text, "x1,x2");
  CHECK_STR(result.targets_text, "species");
  CHECK_EQ_INT(result.rows[0].predicted, 0u);
  CHECK_EQ_INT(result.rows[0].expected, 0);
  CHECK(result.rows[0].correct);
  CHECK_EQ_INT(result.rows[1].predicted, 1u);
  CHECK(result.rows[1].correct);
  CHECK_EQ_INT(result.rows[2].predicted, 0u);
  CHECK_EQ_INT(result.rows[2].expected, 1);
  CHECK(!result.rows[2].correct);
  CHECK_EQ_INT(result.rows[2].line, 4u);
  CHECK(elml_eval_values(&result, 2u) != NULL);
  CHECK(elml_eval_values(&result, 3u) == NULL);
  CHECK(elml_eval_is_classification(&result.model));

  char label[32];
  CHECK_STR(elml_eval_class_label(&result.model, 0u, label, sizeof label),
            "left");
  CHECK_STR(elml_eval_class_label(&result.model, 7u, label, sizeof label),
            "class_7");
  CHECK_STR(elml_eval_output_name(&result.model, 0u, label, sizeof label),
            "species");
  CHECK_STR(elml_eval_output_name(&result.model, 3u, label, sizeof label),
            "output_3");
  elml_eval_free(&result);

  /* Limit and explicit column selection (names and 1-based indices). */
  opts = options(class_model, class_csv);
  opts.limit = 2u;
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_OK);
  CHECK_EQ_INT(result.row_count, 2u);
  CHECK_EQ_INT(result.data_rows, 3u);
  CHECK_EQ_INT(result.limit, 2u);
  elml_eval_free(&result);

  opts = options(class_model, class_csv);
  opts.inputs = "x2,x1";
  opts.targets = "species";
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_OK);
  CHECK_EQ_INT(result.rows[0].predicted, 1u);
  CHECK_EQ_INT(result.rows[1].predicted, 0u);
  CHECK_EQ_INT(result.correct, 1u);
  elml_eval_free(&result);

  opts = options(class_model, class_csv);
  opts.inputs = "1,2";
  opts.targets = "3";
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_OK);
  CHECK_EQ_INT(result.correct, 2u);
  elml_eval_free(&result);

  /* Classification scoring needs exactly one target column. */
  opts = options(class_model, class_csv);
  opts.targets = "x1,x2";
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_ERR_INVALID_ARG);
  CHECK(strstr(result.message, "exactly one target") != NULL);
  elml_eval_free(&result);

  /* Bad input values fail unless skip_bad_rows is set. */
  opts = options(class_model, bad_csv);
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_ERR_INVALID_ARG);
  CHECK(strstr(result.message, "bad input value") != NULL);
  elml_eval_free(&result);

  opts = options(class_model, skip_csv);
  opts.skip_bad_rows = true;
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_OK);
  CHECK_EQ_INT(result.row_count, 2u);
  CHECK_EQ_INT(result.skipped, 1u);
  CHECK(strstr(result.skipped_note, "row 3") != NULL);
  CHECK_EQ_INT(result.correct, 2u);

  /* Prediction CSV: source columns plus label and probabilities. */
  CHECK_EQ_INT(elml_eval_write_csv(&result, out_csv), ELML_OK);
  uint8_t *written = NULL;
  size_t written_len = 0u;
  CHECK_EQ_INT(elml_read_file(out_csv, &written, &written_len, 0u), ELML_OK);
  CHECK(written != NULL);
  if (written != NULL) {
    CHECK(contains(written, written_len, "prediction"));
    CHECK(contains(written, written_len, "prob_left"));
    CHECK(contains(written, written_len, "prob_right"));
    CHECK(contains(written, written_len, "left"));
    free(written);
  }
  CHECK_EQ_INT(elml_eval_write_csv(NULL, out_csv), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_eval_write_csv(&result, NULL), ELML_ERR_INVALID_ARG);
  elml_eval_free(&result);

  /* Without metadata, scoring has no target columns to resolve. */
  if (created) {
    CHECK(write_model(plain_model, class_layers, 1u, 2u, 2u, NULL, 0u));
  }
  opts = options(plain_model, class_csv);
  opts.score = true;
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_ERR_INVALID_ARG);
  CHECK(strstr(result.message, "target columns") != NULL);
  elml_eval_free(&result);

  /* Regression: predictions 3 and 5 against 3.5 and 4.5. */
  static const float reg_weights[1] = {2.0f};
  static const float reg_bias[1] = {1.0f};
  static const elml_write_layer reg_layers[1] = {
      {1u, 1u, ELML_ACT_LINEAR, 0.0f, reg_weights, reg_bias},
  };
  static const elml_write_meta reg_meta[3] = {
      {"task", "regression"},
      {"columns.inputs", "x"},
      {"columns.outputs", "y"},
  };
  if (created) {
    CHECK(write_model(reg_model, reg_layers, 1u, 1u, 1u, reg_meta, 3u));
    CHECK(write_text(reg_csv, "x,y\n1.0,3.5\n2.0,4.5\n"));
  }
  opts = options(reg_model, reg_csv);
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_OK);
  CHECK(!result.classification);
  CHECK(result.scored);
  CHECK_EQ_INT(result.row_count, 2u);
  CHECK_EQ_INT(result.correct, 0u);
  CHECK_CLOSE(result.mse, 0.25, 1e-9);
  CHECK_CLOSE(result.mae, 0.5, 1e-9);
  CHECK_CLOSE(result.rows[0].confidence, 3.0, 1e-6);
  CHECK(elml_eval_targets(&result, 0u) != NULL);
  if (elml_eval_targets(&result, 0u) != NULL) {
    CHECK_CLOSE(elml_eval_targets(&result, 0u)[0], 3.5, 1e-6);
  }
  CHECK(elml_eval_targets(&result, 2u) == NULL);
  elml_eval_free(&result);

  opts = options(reg_model, reg_csv);
  opts.inputs = "1";
  opts.targets = "2";
  opts.limit = 1u;
  CHECK_EQ_INT(elml_eval_run(&opts, &result), ELML_OK);
  CHECK_EQ_INT(result.row_count, 1u);
  CHECK_EQ_INT(result.data_rows, 2u);
  CHECK_CLOSE(result.mse, 0.25, 1e-9);
  elml_eval_free(&result);

  if (created) {
    remove(class_model);
    remove(class_csv);
    remove(bad_csv);
    remove(skip_csv);
    remove(plain_model);
    remove(reg_model);
    remove(reg_csv);
    remove(out_csv);
  }
  TEST_END();
}
