#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/model.h"
#include "elml/types.h"
#include "train/config.h"
#include "train/dataset.h"
#include "train/net.h"
#include "train/session.h"

static bool elml_wants_help(int argc, char **argv)
{
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      return true;
    }
  }
  return false;
}

static double elml_dataset_mae(elml_net *net, const elml_dataset *dataset,
                               const elml_norm_stats *norm)
{
  if (dataset->rows == 0u) {
    return 0.0;
  }
  const elml_net_layer *last = &net->layers[net->layer_count - 1u];
  double total = 0.0;
  for (uint32_t row = 0u; row < dataset->rows; ++row) {
    if (elml_net_forward(net, &dataset->x[(size_t)row * dataset->input_dim],
                         nullptr) != ELML_OK) {
      return 0.0;
    }
    for (uint32_t o = 0u; o < dataset->output_dim; ++o) {
      const float scale =
          (norm != nullptr && norm->has_output) ? norm->output_std[o] : 1.0f;
      total += (double)fabsf((last->a[o] -
                              dataset->y[(size_t)row * dataset->output_dim + o]) *
                             scale);
    }
  }
  return total / ((double)dataset->rows * (double)dataset->output_dim);
}

static void elml_print_metrics(const char *name, elml_net *net,
                               const elml_dataset *dataset,
                               const elml_norm_stats *norm)
{
  uint32_t correct = 0u;
  const float loss = elml_net_evaluate(net, dataset, &correct);

  if (net->loss == ELML_LOSS_CCE) {
    const double accuracy =
        (dataset->rows == 0u) ? 0.0
                              : ((double)correct / (double)dataset->rows);
    printf("%s: loss %.6f accuracy %.4f\n", name, (double)loss, accuracy);
    return;
  }

  const double mae = elml_dataset_mae(net, dataset, norm);
  printf("%s: loss %.6f mae %.6f\n", name, (double)loss, mae);
}

static void elml_print_progress(const elml_train_metrics *metrics, void *user)
{
  (void)user;

  const bool print_this =
      (metrics->epochs <= 50u) || (metrics->epoch % 10u == 0u) ||
      (metrics->epoch == metrics->epochs) || metrics->improved;
  if (print_this) {
    if (metrics->has_accuracy) {
      printf("epoch %4u: val_loss %.6f accuracy %.4f\n", metrics->epoch,
             (double)metrics->loss, (double)metrics->accuracy);
    } else {
      printf("epoch %4u: val_loss %.6f\n", metrics->epoch,
             (double)metrics->loss);
    }
    fflush(stdout);
  }
  if (metrics->early_stopped) {
    printf("early stop at epoch %u (best epoch %u)\n", metrics->epoch,
           metrics->best_epoch);
  }
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    elml_config_print_help();
    return 1;
  }
  if (elml_wants_help(argc, argv)) {
    elml_config_print_help();
    return 0;
  }

  elml_config config;
  elml_status status = elml_config_load(&config, argc, argv);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_train: configuration error: %s\n",
            elml_error_string(status));
    return 1;
  }

  elml_training training;
  status = elml_training_init(&training, &config);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_train: %s\n",
            (training.error[0] != '\0') ? training.error
                                        : elml_error_string(status));
    elml_training_destroy(&training);
    return 1;
  }

  printf("embedlml: %u train / %u validation rows, %u -> %u\n",
         training.train.rows, training.val.rows, training.train.input_dim,
         training.train.output_dim);
  printf("embedlml: layers");
  for (uint32_t i = 0u; i < training.net.layer_count; ++i) {
    printf(" %u", training.net.layers[i].out_dim);
  }
  printf(" (%s, %s)\n", elml_loss_name(config.loss),
         (config.optimizer == ELML_OPTIMIZER_ADAM) ? "adam" : "sgd");

  status = elml_training_run(&training, elml_print_progress, nullptr);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_train: %s\n",
            (training.error[0] != '\0') ? training.error
                                        : elml_error_string(status));
    elml_training_destroy(&training);
    return 1;
  }

  elml_print_metrics("train", &training.net, &training.train,
                     &training.norm);
  if (training.val.rows > 0u) {
    elml_print_metrics("validation", &training.net, &training.val,
                       &training.norm);
  }

  status = elml_training_export(&training);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_train: %s\n",
            (training.error[0] != '\0') ? training.error
                                        : elml_error_string(status));
    elml_training_destroy(&training);
    return 1;
  }

  printf("wrote %s\n", config.export_path);

  elml_training_destroy(&training);
  return 0;
}
