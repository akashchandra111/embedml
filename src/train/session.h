/*
 * Reusable host-only training session.
 *
 * The CLI trainer and the CUI studio share this module so both run exactly the
 * same loop: load CSV, split, normalize, create the net, train epoch by epoch,
 * restore the best weights, and export a portable model.
 *
 * This header is internal to the host trainer. It is not part of the
 * allocation-free inference API.
 */
#ifndef ELML_SESSION_H
#define ELML_SESSION_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/types.h"
#include "train/config.h"
#include "train/dataset.h"
#include "train/net.h"

typedef struct elml_train_metrics {
  uint32_t epoch;      /* epoch that just finished */
  uint32_t epochs;     /* requested total */
  uint32_t best_epoch;
  uint32_t train_rows;
  uint32_t val_rows;
  float loss;          /* loss on the monitored split (validation if present) */
  float accuracy;      /* 0..1 when has_accuracy */
  bool has_accuracy;
  bool improved;       /* this epoch produced a new best loss */
  bool early_stopped;  /* patience ran out on this epoch */
  bool stopped;        /* stop was requested */
} elml_train_metrics;

/* Called once per completed epoch. Keep it short; the studio uses it to update
 * its shared snapshot under a mutex. */
typedef void (*elml_train_progress_fn)(const elml_train_metrics *metrics,
                                       void *user);

typedef struct elml_training {
  elml_config config;

  elml_dataset train;
  elml_dataset val;
  elml_norm_stats norm;
  elml_net net;
  uint32_t *order;

  uint64_t rng_state;
  float best_loss;
  uint32_t best_epoch;
  uint32_t stale;
  uint32_t epochs_done;

  /* Final metrics after restoring the best weights. */
  float final_train_loss;
  float final_train_accuracy;
  float final_val_loss;
  float final_val_accuracy;
  bool has_final_accuracy;

  bool prepared;
  bool ran;
  bool stopped;
  bool early_stopped;
  bool exported;

  atomic_bool stop_requested;
  char error[256];
} elml_training;

/*
 * Prepare a session: read the CSV(s), build and split the dataset, compute
 * normalization, create the net, and allocate the shuffle order.
 *
 * On success the caller must eventually call elml_training_destroy().
 * On failure everything allocated so far is released and the struct is zeroed.
 */
elml_status elml_training_init(elml_training *training,
                               const elml_config *config);

/*
 * Run epochs until config.epochs, early stopping, or a stop request. Can be
 * called again to continue a stopped session. Restores the best weights before
 * returning. The progress callback may be NULL.
 */
elml_status elml_training_run(elml_training *training,
                              elml_train_progress_fn progress, void *user);

/* Ask a running session to stop at the next batch boundary. Thread-safe. */
void elml_training_request_stop(elml_training *training);

/* Export the best model. Requires a completed run. */
elml_status elml_training_export(elml_training *training);

/*
 * Release everything in reverse allocation order:
 * order, net, norm, validation set, training set.
 */
void elml_training_destroy(elml_training *training);

#endif /* ELML_SESSION_H */
