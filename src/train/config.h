#ifndef ELML_CONFIG_H
#define ELML_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/registry.h"
#include "elml/types.h"

#define ELML_CONFIG_PATH_MAX 1024u
#define ELML_CONFIG_LIST_MAX 512u
#define ELML_MAX_HIDDEN_LAYERS 31u
#define ELML_MAX_MODEL_LAYERS (ELML_MAX_HIDDEN_LAYERS + 1u)

typedef enum elml_loss {
  ELML_LOSS_MSE = 0,
  ELML_LOSS_MAE = 1,
  ELML_LOSS_CCE = 2
} elml_loss;

typedef enum elml_optimizer {
  ELML_OPTIMIZER_ADAM = 0,
  ELML_OPTIMIZER_SGD = 1
} elml_optimizer;

typedef enum elml_missing {
  ELML_MISSING_ERROR = 0,
  ELML_MISSING_ZERO = 1,
  ELML_MISSING_MEAN = 2
} elml_missing;

typedef struct elml_config {
  char csv_path[ELML_CONFIG_PATH_MAX];
  char val_csv_path[ELML_CONFIG_PATH_MAX];
  char inputs[ELML_CONFIG_LIST_MAX];
  char outputs[ELML_CONFIG_LIST_MAX];
  bool has_header;
  char delimiter;
  elml_missing missing;
  double split;
  uint64_t seed;

  uint32_t hidden[ELML_MAX_HIDDEN_LAYERS];
  uint32_t hidden_count;
  elml_activation activations[ELML_MAX_MODEL_LAYERS];
  uint32_t activation_count;

  elml_loss loss;
  bool classification;
  elml_optimizer optimizer;

  uint32_t epochs;
  uint32_t batch;
  double learning_rate;
  double l2;
  uint32_t early_stop;
  uint32_t threads;

  char export_path[ELML_CONFIG_PATH_MAX];
} elml_config;

void elml_config_init(elml_config *config);

elml_status elml_config_load(elml_config *config, int argc, char **argv);

void elml_config_print_help(void);

const char *elml_loss_name(elml_loss loss);
const char *elml_activation_name(elml_activation activation);

#endif /* ELML_CONFIG_H */
