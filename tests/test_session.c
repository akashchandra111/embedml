#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "elml/arena.h"
#include "elml/model.h"
#include "elml/types.h"
#include "test.h"
#include "train/config.h"
#include "train/io.h"
#include "train/session.h"

#define SESSION_ROWS 150u

static bool write_rows(const char *path, uint32_t rows)
{
  FILE *file = fopen(path, "w");
  if (file == NULL) {
    return false;
  }
  if (fputs("x1,x2,y\n", file) < 0) {
    fclose(file);
    return false;
  }
  for (uint32_t i = 0u; i < rows; ++i) {
    const double t = (rows > 1u) ? (double)i / (double)(rows - 1u) : 0.0;
    const double x1 = -2.0 + 4.0 * t;
    const double x2 = 1.0 - 2.0 * t;
    const double y = 1.5 * x1 - 2.0 * x2 + 0.75;
    if (fprintf(file, "%.6f,%.6f,%.6f\n", x1, x2, y) < 0) {
      fclose(file);
      return false;
    }
  }
  return fclose(file) == 0;
}

static elml_config make_config(const char *csv_path, const char *export_path)
{
  elml_config config;
  elml_config_init(&config);
  snprintf(config.csv_path, sizeof config.csv_path, "%s", csv_path);
  snprintf(config.inputs, sizeof config.inputs, "%s", "x1,x2");
  snprintf(config.outputs, sizeof config.outputs, "%s", "y");
  snprintf(config.export_path, sizeof config.export_path, "%s", export_path);
  config.hidden[0] = 8u;
  config.hidden_count = 1u;
  config.activations[0] = ELML_ACT_RELU;
  config.activations[1] = ELML_ACT_LINEAR;
  config.activation_count = 2u;
  config.loss = ELML_LOSS_MSE;
  config.epochs = 150u;
  config.batch = 16u;
  config.learning_rate = 0.01;
  config.l2 = 0.0;
  config.early_stop = 0u;
  config.split = 0.75;
  config.seed = 5u;
  return config;
}

static int g_progress_calls;
static uint32_t g_progress_last_epoch;

static void progress_cb(const elml_train_metrics *metrics, void *user)
{
  (void)user;
  g_progress_calls++;
  g_progress_last_epoch = metrics->epoch;
  CHECK(metrics->epoch > 0u);
}

static void test_full_run(const char *csv_path, const char *export_path)
{
  elml_config config = make_config(csv_path, export_path);
  elml_training training;
  CHECK_EQ_INT(elml_training_init(&training, &config), ELML_OK);
  CHECK(training.prepared);
  CHECK(training.order != NULL);
  CHECK_EQ_INT(training.train.input_dim, 2u);
  CHECK_EQ_INT(training.train.output_dim, 1u);
  CHECK_EQ_INT(training.train.rows + training.val.rows, SESSION_ROWS);
  CHECK(training.val.rows > 0u);
  CHECK_EQ_INT(training.net.layer_count, 2u);
  CHECK_EQ_INT(training.epochs_done, 0u);

  elml_train_step step;
  CHECK_EQ_INT(elml_training_step_epoch(&training, &step), ELML_OK);
  CHECK(step.epoch_completed);
  CHECK(!step.done);
  CHECK_EQ_INT(step.metrics.epoch, 1u);
  CHECK_EQ_INT(step.metrics.epochs, 150u);
  CHECK(step.metrics.loss >= 0.0f);
  CHECK(!step.metrics.has_accuracy);
  CHECK(step.metrics.train_rows + step.metrics.val_rows == SESSION_ROWS);

  uint32_t completed = step.epoch_completed ? 1u : 0u;
  for (int guard = 0; guard < 1000 && !step.done; ++guard) {
    CHECK_EQ_INT(elml_training_step_epoch(&training, &step), ELML_OK);
    if (step.epoch_completed) {
      completed++;
    }
  }
  CHECK(step.done);
  CHECK_EQ_INT(completed, 150u);
  CHECK_EQ_INT(training.epochs_done, 150u);
  CHECK(training.finalized);
  CHECK(training.ran);
  CHECK(!training.early_stopped);
  CHECK(!training.stopped);
  CHECK(training.best_epoch >= 1u && training.best_epoch <= 150u);
  CHECK(isfinite((double)training.final_train_loss));
  CHECK(isfinite((double)training.final_val_loss));
  CHECK(!training.has_final_accuracy);

  /* A finished session keeps reporting done and does not run more epochs. */
  memset(&step, 0, sizeof step);
  CHECK_EQ_INT(elml_training_step_epoch(&training, &step), ELML_OK);
  CHECK(step.done);
  CHECK(!step.epoch_completed);
  CHECK_EQ_INT(training.epochs_done, 150u);

  CHECK_EQ_INT(elml_training_export(&training), ELML_OK);
  CHECK(training.exported);

  uint8_t *blob = NULL;
  size_t blob_len = 0u;
  CHECK_EQ_INT(elml_read_file(export_path, &blob, &blob_len, ELML_PARAM_ALIGN),
               ELML_OK);
  CHECK(blob != NULL);
  CHECK(blob_len > 0u);
  if (blob != NULL) {
    elml_model model;
    CHECK_EQ_INT(elml_model_init(&model, blob, blob_len, ELML_DTYPE_F32),
                 ELML_OK);
    CHECK_EQ_INT(model.input.dim[0], 2u);
    CHECK_EQ_INT(model.output.dim[0], 1u);
    const size_t scratch_len = elml_scratch_bytes(&model, 1u);
    uint8_t *scratch =
        (uint8_t *)aligned_alloc(ELML_TENSOR_ALIGN, scratch_len);
    CHECK(scratch != NULL);
    if (scratch != NULL) {
      const float input[2] = {0.5f, -0.5f};
      float output[1] = {0.0f};
      CHECK_EQ_INT(elml_forward(&model, scratch, scratch_len, input, output),
                   ELML_OK);
      CHECK(isfinite((double)output[0]));
      free(scratch);
    }
    elml_aligned_free(blob);
  }
  elml_training_destroy(&training);
  elml_training_destroy(&training);
}

static void test_stop_before_first_epoch(const char *csv_path,
                                         const char *export_path)
{
  elml_config config = make_config(csv_path, export_path);
  elml_training training;
  CHECK_EQ_INT(elml_training_init(&training, &config), ELML_OK);

  /* Export is only valid after at least one completed epoch. */
  CHECK_EQ_INT(elml_training_export(&training), ELML_ERR_INVALID_ARG);

  elml_training_request_stop(&training);
  elml_train_step step;
  CHECK_EQ_INT(elml_training_step_epoch(&training, &step), ELML_OK);
  CHECK(step.done);
  CHECK(!step.epoch_completed);
  CHECK(training.stopped);
  CHECK(!training.ran);
  CHECK_EQ_INT(training.epochs_done, 0u);
  CHECK_EQ_INT(elml_training_export(&training), ELML_ERR_INVALID_ARG);
  elml_training_destroy(&training);
}

static void test_early_stop(const char *csv_path, const char *export_path)
{
  elml_config config = make_config(csv_path, export_path);
  config.epochs = 100u;
  config.early_stop = 2u;
  /* A learning rate this small leaves the weights unchanged at f32 precision,
   * so the loss stops improving immediately and patience expires
   * deterministically at epoch three. */
  config.learning_rate = 1e-12;

  elml_training training;
  CHECK_EQ_INT(elml_training_init(&training, &config), ELML_OK);
  elml_train_step step;
  for (int guard = 0; guard < 20; ++guard) {
    CHECK_EQ_INT(elml_training_step_epoch(&training, &step), ELML_OK);
    if (step.done) {
      break;
    }
  }
  CHECK(step.done);
  CHECK(training.early_stopped);
  CHECK(training.epochs_done < config.epochs);
  CHECK_EQ_INT(training.epochs_done, 3u);
  CHECK_EQ_INT(training.best_epoch, 1u);
  elml_training_destroy(&training);
}

static void test_run_callback(const char *csv_path, const char *export_path)
{
  elml_config config = make_config(csv_path, export_path);
  config.epochs = 10u;
  g_progress_calls = 0;
  g_progress_last_epoch = 0u;

  elml_training training;
  CHECK_EQ_INT(elml_training_init(&training, &config), ELML_OK);
  CHECK_EQ_INT(elml_training_run(&training, progress_cb, NULL), ELML_OK);
  CHECK_EQ_INT(g_progress_calls, 10);
  CHECK_EQ_INT(g_progress_last_epoch, 10u);
  CHECK_EQ_INT(training.epochs_done, 10u);
  elml_training_destroy(&training);
}

static void test_threaded_run(const char *csv_path, const char *export_path)
{
  elml_config config = make_config(csv_path, export_path);
  config.epochs = 40u;
  config.threads = 4u;

  elml_training training;
  CHECK_EQ_INT(elml_training_init(&training, &config), ELML_OK);
#if defined(ELML_HAVE_THREADS)
  CHECK_EQ_INT(training.worker_count, 4u);
#else
  CHECK_EQ_INT(training.worker_count, 1u);
#endif
  CHECK_EQ_INT(training.config.threads, training.worker_count);
  CHECK_EQ_INT(elml_training_run(&training, NULL, NULL), ELML_OK);
  CHECK_EQ_INT(training.epochs_done, 40u);
  CHECK(training.ran);
  CHECK(isfinite((double)training.final_train_loss));
  CHECK(isfinite((double)training.final_val_loss));
  elml_training_destroy(&training);

  /* A stop request before the first batch must still finalize cleanly with
   * the worker pool alive. */
  config.epochs = 1000u;
  CHECK_EQ_INT(elml_training_init(&training, &config), ELML_OK);
  elml_training_request_stop(&training);
  elml_train_step step;
  CHECK_EQ_INT(elml_training_step_epoch(&training, &step), ELML_OK);
  CHECK(step.done);
  CHECK(!step.epoch_completed);
  CHECK(training.stopped);
  elml_training_destroy(&training);
}

static void test_validation_file(const char *csv_path, const char *val_path,
                                 const char *export_path)
{
  elml_config config = make_config(csv_path, export_path);
  config.epochs = 5u;
  snprintf(config.val_csv_path, sizeof config.val_csv_path, "%s", val_path);

  elml_training training;
  CHECK_EQ_INT(elml_training_init(&training, &config), ELML_OK);
  CHECK_EQ_INT(training.train.rows, SESSION_ROWS);
  CHECK_EQ_INT(training.val.rows, SESSION_ROWS);
  CHECK_EQ_INT(elml_training_run(&training, NULL, NULL), ELML_OK);
  CHECK(training.has_final_accuracy == false);
  elml_training_destroy(&training);
}

int main(void)
{
  char csv_path[] = "elml_session_data_XXXXXX";
  char val_path[] = "elml_session_val_XXXXXX";
  char export_path[] = "elml_session_model_XXXXXX";

  const int csv_fd = mkstemp(csv_path);
  const int val_fd = mkstemp(val_path);
  const int export_fd = mkstemp(export_path);
  CHECK(csv_fd >= 0 && val_fd >= 0 && export_fd >= 0);
  if (csv_fd >= 0) {
    close(csv_fd);
  }
  if (val_fd >= 0) {
    close(val_fd);
  }
  if (export_fd >= 0) {
    close(export_fd);
  }

  const bool files_ok = csv_fd >= 0 && val_fd >= 0 && export_fd >= 0 &&
                        write_rows(csv_path, SESSION_ROWS) &&
                        write_rows(val_path, SESSION_ROWS);
  CHECK(files_ok);

  if (files_ok) {
    test_full_run(csv_path, export_path);
    test_stop_before_first_epoch(csv_path, export_path);
    test_early_stop(csv_path, export_path);
    test_run_callback(csv_path, export_path);
    test_validation_file(csv_path, val_path, export_path);
    test_threaded_run(csv_path, export_path);
  }

  /* Missing data reports a message and leaves the session unusable. */
  elml_config missing = make_config("does-not-exist.csv", export_path);
  elml_training training;
  CHECK(elml_training_init(&training, &missing) != ELML_OK);
  CHECK(training.error[0] != '\0');
  CHECK(!training.prepared);

  elml_train_step step;
  CHECK_EQ_INT(elml_training_init(NULL, &missing), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_training_init(&training, NULL), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_training_step_epoch(NULL, &step), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_training_step_epoch(&training, NULL),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_training_run(NULL, NULL, NULL), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_training_export(NULL), ELML_ERR_INVALID_ARG);
  elml_training_request_stop(NULL);
  elml_training_destroy(NULL);

  if (files_ok) {
    remove(csv_path);
    remove(val_path);
    remove(export_path);
  }
  TEST_END();
}
