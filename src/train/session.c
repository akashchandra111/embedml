#include "train/session.h"

#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/model.h"
#include "train/csv.h"
#include "train/export.h"

static void elml_training_error(elml_training *training, const char *fmt, ...)
{
  if (training == nullptr || fmt == nullptr) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  vsnprintf(training->error, sizeof training->error, fmt, args);
  va_end(args);
}

static void elml_shuffle_order(uint32_t *order, uint32_t count, uint64_t *state)
{
  for (uint32_t i = count; i > 1u; --i) {
    const uint32_t j = (uint32_t)(elml_rng_next(state) % (uint64_t)i);
    const uint32_t swap = order[i - 1u];
    order[i - 1u] = order[j];
    order[j] = swap;
  }
}

static bool elml_same_labels(const elml_dataset *a, const elml_dataset *b)
{
  if (a->label_count != b->label_count) {
    return false;
  }
  for (uint32_t i = 0u; i < a->label_count; ++i) {
    if (strcmp(a->labels[i], b->labels[i]) != 0) {
      return false;
    }
  }
  return true;
}

void elml_training_destroy(elml_training *training)
{
  if (training == nullptr) {
    return;
  }
  /* Reverse allocation order: order, net, norm, validation, training. */
  free(training->order);
  training->order = nullptr;
  elml_net_free(&training->net);
  elml_norm_stats_free(&training->norm);
  elml_dataset_free(&training->val);
  elml_dataset_free(&training->train);
  memset(training, 0, sizeof *training);
}

elml_status elml_training_init(elml_training *training,
                               const elml_config *config)
{
  if (training == nullptr || config == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }

  memset(training, 0, sizeof *training);
  atomic_init(&training->stop_requested, false);
  training->config = *config;
  training->best_loss = FLT_MAX;
  training->rng_state = config->seed ^ 0xA5A5A5A55A5A5A5Aull;

  elml_status status = ELML_OK;
  elml_csv csv;
  elml_dataset full;
  memset(&full, 0, sizeof full);

  status = elml_csv_load(config->csv_path, config->delimiter,
                         config->has_header, &csv);
  if (status != ELML_OK) {
    elml_training_error(training, "cannot read %s: %s", config->csv_path,
                        elml_error_string(status));
    goto fail;
  }
  status = elml_dataset_build(&csv, config, &full);
  elml_csv_free(&csv);
  if (status != ELML_OK) {
    elml_training_error(training, "cannot build dataset: %s",
                        elml_error_string(status));
    goto fail;
  }

  if (config->val_csv_path[0] != '\0') {
    elml_csv val_csv;
    status = elml_csv_load(config->val_csv_path, config->delimiter,
                           config->has_header, &val_csv);
    if (status != ELML_OK) {
      elml_training_error(training, "cannot read %s: %s",
                          config->val_csv_path, elml_error_string(status));
      goto fail;
    }
    status = elml_dataset_build(&val_csv, config, &training->val);
    elml_csv_free(&val_csv);
    if (status != ELML_OK) {
      elml_training_error(training, "cannot build validation dataset: %s",
                          elml_error_string(status));
      goto fail;
    }
    training->train = full;
    memset(&full, 0, sizeof full);
  } else {
    status = elml_dataset_split(&full, config->split, config->seed,
                                &training->train, &training->val);
    elml_dataset_free(&full);
    if (status != ELML_OK) {
      elml_training_error(training, "cannot split dataset: %s",
                          elml_error_string(status));
      goto fail;
    }
  }

  if (training->val.rows > 0u) {
    if (training->train.input_dim != training->val.input_dim ||
        training->train.output_dim != training->val.output_dim) {
      status = ELML_ERR_INVALID_ARG;
      elml_training_error(training,
                          "train and validation dimensions differ");
      goto fail;
    }
    if (config->classification &&
        !elml_same_labels(&training->train, &training->val)) {
      status = ELML_ERR_INVALID_ARG;
      elml_training_error(training, "train and validation labels differ");
      goto fail;
    }
  }

  status = elml_dataset_compute_norm(&training->train,
                                     !config->classification, &training->norm);
  if (status != ELML_OK) {
    elml_training_error(training, "cannot compute normalization: %s",
                        elml_error_string(status));
    goto fail;
  }
  elml_dataset_apply_norm(&training->train, &training->norm);
  elml_dataset_apply_norm(&training->val, &training->norm);

  status = elml_net_create(
      &training->net, config->hidden, config->hidden_count, config->activations,
      config->activation_count, training->train.input_dim,
      training->train.output_dim, config->loss, config->optimizer,
      (float)config->learning_rate, (float)config->l2, config->seed);
  if (status != ELML_OK) {
    elml_training_error(training, "cannot create network: %s",
                        elml_error_string(status));
    goto fail;
  }

  training->order =
      (uint32_t *)malloc((size_t)training->train.rows * sizeof(uint32_t));
  if (training->order == nullptr) {
    status = ELML_ERR_OUT_OF_MEMORY;
    elml_training_error(training, "out of memory");
    goto fail;
  }
  for (uint32_t i = 0u; i < training->train.rows; ++i) {
    training->order[i] = i;
  }

  training->prepared = true;
  return ELML_OK;

fail:
  {
    char saved[sizeof training->error];
    snprintf(saved, sizeof saved, "%s", training->error);
    elml_dataset_free(&full);
    elml_training_destroy(training);
    memcpy(training->error, saved, sizeof training->error);
  }
  return status;
}

elml_status elml_training_run(elml_training *training,
                              elml_train_progress_fn progress, void *user)
{
  if (training == nullptr || !training->prepared) {
    return ELML_ERR_INVALID_ARG;
  }

  const elml_config *config = &training->config;
  const elml_dataset *monitor =
      (training->val.rows > 0u) ? &training->val : &training->train;
  const bool has_accuracy = (monitor->class_index != nullptr);
  elml_status status = ELML_OK;

  for (uint32_t epoch = training->epochs_done + 1u; epoch <= config->epochs;
       ++epoch) {
    elml_shuffle_order(training->order, training->train.rows,
                       &training->rng_state);

    bool aborted = false;
    for (uint32_t start = 0u; start < training->train.rows;
         start += config->batch) {
      if (atomic_load(&training->stop_requested)) {
        aborted = true;
        break;
      }

      uint32_t count = config->batch;
      if (count > training->train.rows - start) {
        count = training->train.rows - start;
      }
      elml_net_zero_grad(&training->net);
      for (uint32_t k = 0u; k < count; ++k) {
        if (atomic_load(&training->stop_requested)) {
          aborted = true;
          break;
        }
        const uint32_t row = training->order[start + k];
        const float *input =
            &training->train.x[(size_t)row * training->train.input_dim];
        const float *target =
            &training->train.y[(size_t)row * training->train.output_dim];
        const uint32_t class_index =
            (training->train.class_index != nullptr)
                ? training->train.class_index[row]
                : 0u;
        status = elml_net_accumulate(&training->net, input, target,
                                     class_index);
        if (status != ELML_OK) {
          elml_training_error(training, "training failed: %s",
                              elml_error_string(status));
          return status;
        }
      }
      if (aborted) {
        break;
      }
      elml_net_update(&training->net, count);
    }
    if (aborted) {
      training->stopped = true;
      break;
    }

    uint32_t correct = 0u;
    const float monitor_loss =
        elml_net_evaluate(&training->net, monitor, &correct);
    bool improved = false;
    if (monitor_loss < training->best_loss - 1e-6f) {
      training->best_loss = monitor_loss;
      training->best_epoch = epoch;
      training->stale = 0u;
      elml_net_save_best(&training->net);
      improved = true;
    } else {
      training->stale++;
    }

    training->epochs_done = epoch;

    elml_train_metrics metrics = {
        .epoch = epoch,
        .epochs = config->epochs,
        .best_epoch = training->best_epoch,
        .train_rows = training->train.rows,
        .val_rows = training->val.rows,
        .loss = monitor_loss,
        .accuracy =
            (has_accuracy && monitor->rows > 0u)
                ? (float)((double)correct / (double)monitor->rows)
                : 0.0f,
        .has_accuracy = has_accuracy,
        .improved = improved,
        .early_stopped = false,
        .stopped = false,
    };

    if (config->early_stop > 0u && training->stale >= config->early_stop) {
      training->early_stopped = true;
      metrics.early_stopped = true;
    }
    if (progress != nullptr) {
      progress(&metrics, user);
    }
    if (metrics.early_stopped) {
      break;
    }
  }

  if (atomic_load(&training->stop_requested)) {
    training->stopped = true;
  }
  elml_net_restore_best(&training->net);
  training->ran = training->epochs_done > 0u;

  if (training->ran) {
    uint32_t correct = 0u;
    training->final_train_loss =
        elml_net_evaluate(&training->net, &training->train, &correct);
    training->final_train_accuracy =
        (training->train.rows > 0u)
            ? (float)((double)correct / (double)training->train.rows)
            : 0.0f;
    training->has_final_accuracy =
        (training->train.class_index != nullptr);
    if (training->val.rows > 0u) {
      correct = 0u;
      training->final_val_loss =
          elml_net_evaluate(&training->net, &training->val, &correct);
      training->final_val_accuracy =
          (float)((double)correct / (double)training->val.rows);
    }
  }

  return ELML_OK;
}

void elml_training_request_stop(elml_training *training)
{
  if (training != nullptr) {
    atomic_store(&training->stop_requested, true);
  }
}

elml_status elml_training_export(elml_training *training)
{
  if (training == nullptr || !training->prepared || !training->ran) {
    return ELML_ERR_INVALID_ARG;
  }
  elml_status status =
      elml_export_model(&training->net, &training->norm, &training->train,
                        &training->config, training->config.export_path);
  if (status == ELML_OK) {
    training->exported = true;
  } else {
    elml_training_error(training, "cannot export %s: %s",
                        training->config.export_path,
                        elml_error_string(status));
  }
  return status;
}
