#include "train/config.h"

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/registry.h"
#include "elml/types.h"
#include "train/ini.h"

static elml_status elml_copy_string(char *dst, size_t cap, const char *src)
{
  if (dst == nullptr || src == nullptr || cap == 0u) {
    return ELML_ERR_INVALID_ARG;
  }
  const size_t len = strlen(src);
  if (len + 1u > cap) {
    return ELML_ERR_BUFFER_TOO_SMALL;
  }
  memcpy(dst, src, len + 1u);
  return ELML_OK;
}

static elml_status elml_parse_u32(const char *text, uint32_t *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long value = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
    return ELML_ERR_INVALID_ARG;
  }
  *out = (uint32_t)value;
  return ELML_OK;
}

static elml_status elml_parse_u64(const char *text, uint64_t *out)
{
  if (text == nullptr || out == nullptr || *text == '\0' || *text == '-') {
    return ELML_ERR_INVALID_ARG;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  *out = (uint64_t)value;
  return ELML_OK;
}

static elml_status elml_parse_double(const char *text, double *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  errno = 0;
  char *end = nullptr;
  const double value = strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  *out = value;
  return ELML_OK;
}

static elml_status elml_parse_bool(const char *text, bool *out)
{
  if (text == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (strcmp(text, "true") == 0 || strcmp(text, "yes") == 0 ||
      strcmp(text, "1") == 0 || strcmp(text, "on") == 0) {
    *out = true;
    return ELML_OK;
  }
  if (strcmp(text, "false") == 0 || strcmp(text, "no") == 0 ||
      strcmp(text, "0") == 0 || strcmp(text, "off") == 0) {
    *out = false;
    return ELML_OK;
  }
  return ELML_ERR_INVALID_ARG;
}

static elml_status elml_parse_delimiter(const char *text, char *out)
{
  if (text == nullptr || out == nullptr || *text == '\0') {
    return ELML_ERR_INVALID_ARG;
  }
  if (text[1] == '\0') {
    *out = text[0];
    return ELML_OK;
  }
  if (strcmp(text, "\\t") == 0) {
    *out = '\t';
    return ELML_OK;
  }
  return ELML_ERR_INVALID_ARG;
}

static elml_status elml_parse_loss(const char *text, elml_loss *out)
{
  if (text == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (strcmp(text, "mse") == 0) {
    *out = ELML_LOSS_MSE;
    return ELML_OK;
  }
  if (strcmp(text, "mae") == 0) {
    *out = ELML_LOSS_MAE;
    return ELML_OK;
  }
  if (strcmp(text, "cce") == 0 || strcmp(text, "cross_entropy") == 0 ||
      strcmp(text, "cross-entropy") == 0) {
    *out = ELML_LOSS_CCE;
    return ELML_OK;
  }
  return ELML_ERR_INVALID_ARG;
}

static elml_status elml_parse_optimizer(const char *text, elml_optimizer *out)
{
  if (text == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (strcmp(text, "adam") == 0) {
    *out = ELML_OPTIMIZER_ADAM;
    return ELML_OK;
  }
  if (strcmp(text, "sgd") == 0) {
    *out = ELML_OPTIMIZER_SGD;
    return ELML_OK;
  }
  return ELML_ERR_INVALID_ARG;
}

static elml_status elml_parse_missing(const char *text, elml_missing *out)
{
  if (text == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (strcmp(text, "error") == 0) {
    *out = ELML_MISSING_ERROR;
    return ELML_OK;
  }
  if (strcmp(text, "zero") == 0) {
    *out = ELML_MISSING_ZERO;
    return ELML_OK;
  }
  if (strcmp(text, "mean") == 0) {
    *out = ELML_MISSING_MEAN;
    return ELML_OK;
  }
  return ELML_ERR_INVALID_ARG;
}

static elml_status elml_parse_activation(const char *text,
                                         elml_activation *out)
{
  if (text == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (strcmp(text, "relu") == 0) {
    *out = ELML_ACT_RELU;
    return ELML_OK;
  }
  if (strcmp(text, "leaky_relu") == 0 || strcmp(text, "leaky-relu") == 0) {
    *out = ELML_ACT_LEAKY_RELU;
    return ELML_OK;
  }
  if (strcmp(text, "tanh") == 0) {
    *out = ELML_ACT_TANH;
    return ELML_OK;
  }
  if (strcmp(text, "sigmoid") == 0) {
    *out = ELML_ACT_SIGMOID;
    return ELML_OK;
  }
  if (strcmp(text, "softmax") == 0) {
    *out = ELML_ACT_SOFTMAX;
    return ELML_OK;
  }
  if (strcmp(text, "linear") == 0 || strcmp(text, "none") == 0) {
    *out = ELML_ACT_LINEAR;
    return ELML_OK;
  }
  return ELML_ERR_INVALID_ARG;
}

static elml_status elml_parse_u32_list(const char *text, uint32_t *values,
                                       uint32_t capacity, uint32_t *count)
{
  if (text == nullptr || values == nullptr || count == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  *count = 0u;
  const char *cursor = text;

  while (*cursor != '\0') {
    while (*cursor == ' ' || *cursor == '\t') {
      cursor++;
    }
    if (*cursor == '\0') {
      break;
    }
    if (*count >= capacity) {
      return ELML_ERR_INVALID_ARG;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long value = strtoul(cursor, &end, 10);
    if (errno != 0 || end == cursor || value > UINT32_MAX) {
      return ELML_ERR_INVALID_ARG;
    }
    values[*count] = (uint32_t)value;
    (*count)++;
    cursor = end;
    while (*cursor == ' ' || *cursor == '\t') {
      cursor++;
    }
    if (*cursor == ',') {
      cursor++;
    } else if (*cursor != '\0') {
      return ELML_ERR_INVALID_ARG;
    }
  }

  return (*count == 0u) ? ELML_ERR_INVALID_ARG : ELML_OK;
}

static elml_status elml_parse_activation_list(const char *text,
                                              elml_activation *values,
                                              uint32_t capacity,
                                              uint32_t *count)
{
  if (text == nullptr || values == nullptr || count == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  *count = 0u;
  const char *cursor = text;

  while (*cursor != '\0') {
    while (*cursor == ' ' || *cursor == '\t') {
      cursor++;
    }
    if (*cursor == '\0') {
      break;
    }
    if (*count >= capacity) {
      return ELML_ERR_INVALID_ARG;
    }

    const char *token_end = cursor;
    while (*token_end != '\0' && *token_end != ',') {
      token_end++;
    }

    const char *trimmed_end = token_end;
    while (trimmed_end > cursor &&
           (trimmed_end[-1] == ' ' || trimmed_end[-1] == '\t')) {
      trimmed_end--;
    }

    const size_t len = (size_t)(trimmed_end - cursor);
    if (len == 0u || len >= 32u) {
      return ELML_ERR_INVALID_ARG;
    }
    char name[32];
    memcpy(name, cursor, len);
    name[len] = '\0';

    const elml_status status = elml_parse_activation(name, &values[*count]);
    if (status != ELML_OK) {
      return status;
    }
    (*count)++;

    cursor = (*token_end == ',') ? token_end + 1 : token_end;
  }

  return (*count == 0u) ? ELML_ERR_INVALID_ARG : ELML_OK;
}

void elml_config_init(elml_config *config)
{
  if (config == nullptr) {
    return;
  }
  memset(config, 0, sizeof *config);
  config->has_header = true;
  config->delimiter = ',';
  config->missing = ELML_MISSING_ERROR;
  config->split = 0.8;
  config->seed = 42u;
  config->loss = ELML_LOSS_MSE;
  config->classification = false;
  config->optimizer = ELML_OPTIMIZER_ADAM;
  config->epochs = 100u;
  config->batch = 16u;
  config->learning_rate = 0.001;
  config->l2 = 0.0;
  config->early_stop = 20u;
  config->threads = 1u;
}

static bool elml_key_known(const char *section, const char *key)
{
  static const char *const data_keys[] = {
      "csv",       "val_csv",  "inputs",  "outputs", "has_header",
      "delimiter", "missing",  "split",   "seed",
  };
  static const char *const model_keys[] = {"layers", "activations", "loss"};
  static const char *const train_keys[] = {
      "epochs", "batch", "optimizer", "lr", "l2", "early_stop", "threads",
  };
  static const char *const export_keys[] = {"out"};

  const char *const *keys = nullptr;
  size_t count = 0u;
  if (strcmp(section, "data") == 0) {
    keys = data_keys;
    count = ELML_ARRAY_COUNT(data_keys);
  } else if (strcmp(section, "model") == 0) {
    keys = model_keys;
    count = ELML_ARRAY_COUNT(model_keys);
  } else if (strcmp(section, "train") == 0) {
    keys = train_keys;
    count = ELML_ARRAY_COUNT(train_keys);
  } else if (strcmp(section, "export") == 0) {
    keys = export_keys;
    count = ELML_ARRAY_COUNT(export_keys);
  } else {
    return false;
  }

  for (size_t i = 0u; i < count; ++i) {
    if (strcmp(keys[i], key) == 0) {
      return true;
    }
  }
  return false;
}

static elml_status elml_apply_ini(elml_config *config, const elml_ini *ini)
{
  for (size_t i = 0u; i < ini->count; ++i) {
    const elml_ini_entry *entry = &ini->entries[i];
    if (!elml_key_known(entry->section, entry->key)) {
      fprintf(stderr, "config: unknown key [%s] %s\n", entry->section,
              entry->key);
      return ELML_ERR_INVALID_ARG;
    }
  }

  const char *value = nullptr;
  elml_status status = ELML_OK;

  if ((value = elml_ini_get(ini, "data", "csv")) != nullptr) {
    status = elml_copy_string(config->csv_path, sizeof config->csv_path, value);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "val_csv")) != nullptr) {
    status = elml_copy_string(config->val_csv_path, sizeof config->val_csv_path,
                              value);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "inputs")) != nullptr) {
    status = elml_copy_string(config->inputs, sizeof config->inputs, value);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "outputs")) != nullptr) {
    status = elml_copy_string(config->outputs, sizeof config->outputs, value);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "has_header")) != nullptr) {
    status = elml_parse_bool(value, &config->has_header);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "delimiter")) != nullptr) {
    status = elml_parse_delimiter(value, &config->delimiter);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "missing")) != nullptr) {
    status = elml_parse_missing(value, &config->missing);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "split")) != nullptr) {
    status = elml_parse_double(value, &config->split);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "data", "seed")) != nullptr) {
    status = elml_parse_u64(value, &config->seed);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "model", "layers")) != nullptr) {
    status = elml_parse_u32_list(value, config->hidden,
                                 ELML_MAX_HIDDEN_LAYERS, &config->hidden_count);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "model", "activations")) != nullptr) {
    status = elml_parse_activation_list(
        value, config->activations, ELML_MAX_MODEL_LAYERS,
        &config->activation_count);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "model", "loss")) != nullptr) {
    status = elml_parse_loss(value, &config->loss);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "train", "epochs")) != nullptr) {
    status = elml_parse_u32(value, &config->epochs);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "train", "batch")) != nullptr) {
    status = elml_parse_u32(value, &config->batch);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "train", "optimizer")) != nullptr) {
    status = elml_parse_optimizer(value, &config->optimizer);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "train", "lr")) != nullptr) {
    status = elml_parse_double(value, &config->learning_rate);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "train", "l2")) != nullptr) {
    status = elml_parse_double(value, &config->l2);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "train", "early_stop")) != nullptr) {
    status = elml_parse_u32(value, &config->early_stop);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "train", "threads")) != nullptr) {
    status = elml_parse_u32(value, &config->threads);
  }
  if (status == ELML_OK &&
      (value = elml_ini_get(ini, "export", "out")) != nullptr) {
    status = elml_copy_string(config->export_path,
                              sizeof config->export_path, value);
  }

  if (status != ELML_OK) {
    fprintf(stderr, "config: invalid value in %s\n", "INI file");
  }
  return status;
}

static elml_status elml_require_value(int argc, char **argv, int *index,
                                      const char **value)
{
  if (*index + 1 >= argc) {
    fprintf(stderr, "config: %s requires a value\n", argv[*index]);
    return ELML_ERR_INVALID_ARG;
  }
  *value = argv[++(*index)];
  return ELML_OK;
}

static elml_status elml_apply_cli(elml_config *config, int argc, char **argv)
{
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    const char *value = nullptr;

    if (strcmp(arg, "--config") == 0) {
      i++; /* value was consumed by the pre-scan */
      continue;
    }
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      return ELML_ERR_NOT_FOUND;
    }
    if (strcmp(arg, "--no-header") == 0) {
      config->has_header = false;
      continue;
    }
    if (strcmp(arg, "--header") == 0) {
      config->has_header = true;
      continue;
    }

    elml_status status = elml_require_value(argc, argv, &i, &value);
    if (status != ELML_OK) {
      return status;
    }

    if (strcmp(arg, "--csv") == 0) {
      status = elml_copy_string(config->csv_path, sizeof config->csv_path,
                                value);
    } else if (strcmp(arg, "--val-csv") == 0) {
      status = elml_copy_string(config->val_csv_path,
                                sizeof config->val_csv_path, value);
    } else if (strcmp(arg, "--inputs") == 0) {
      status = elml_copy_string(config->inputs, sizeof config->inputs, value);
    } else if (strcmp(arg, "--outputs") == 0) {
      status = elml_copy_string(config->outputs, sizeof config->outputs, value);
    } else if (strcmp(arg, "--layers") == 0) {
      status = elml_parse_u32_list(value, config->hidden,
                                   ELML_MAX_HIDDEN_LAYERS,
                                   &config->hidden_count);
    } else if (strcmp(arg, "--act") == 0 ||
               strcmp(arg, "--activations") == 0) {
      status = elml_parse_activation_list(value, config->activations,
                                          ELML_MAX_MODEL_LAYERS,
                                          &config->activation_count);
    } else if (strcmp(arg, "--loss") == 0) {
      status = elml_parse_loss(value, &config->loss);
    } else if (strcmp(arg, "--epochs") == 0) {
      status = elml_parse_u32(value, &config->epochs);
    } else if (strcmp(arg, "--batch") == 0) {
      status = elml_parse_u32(value, &config->batch);
    } else if (strcmp(arg, "--opt") == 0 ||
               strcmp(arg, "--optimizer") == 0) {
      status = elml_parse_optimizer(value, &config->optimizer);
    } else if (strcmp(arg, "--lr") == 0) {
      status = elml_parse_double(value, &config->learning_rate);
    } else if (strcmp(arg, "--l2") == 0) {
      status = elml_parse_double(value, &config->l2);
    } else if (strcmp(arg, "--seed") == 0) {
      status = elml_parse_u64(value, &config->seed);
    } else if (strcmp(arg, "--val") == 0) {
      status = elml_parse_double(value, &config->split);
    } else if (strcmp(arg, "--threads") == 0) {
      status = elml_parse_u32(value, &config->threads);
    } else if (strcmp(arg, "--early-stop") == 0) {
      status = elml_parse_u32(value, &config->early_stop);
    } else if (strcmp(arg, "--export") == 0) {
      status = elml_copy_string(config->export_path,
                                sizeof config->export_path, value);
    } else if (strcmp(arg, "--delimiter") == 0) {
      status = elml_parse_delimiter(value, &config->delimiter);
    } else if (strcmp(arg, "--missing") == 0) {
      status = elml_parse_missing(value, &config->missing);
    } else {
      fprintf(stderr, "config: unknown option %s\n", arg);
      return ELML_ERR_INVALID_ARG;
    }

    if (status != ELML_OK) {
      fprintf(stderr, "config: invalid value for %s\n", arg);
      return status;
    }
  }
  return ELML_OK;
}

static elml_status elml_finalize_config(elml_config *config)
{
  if (config->csv_path[0] == '\0' || config->inputs[0] == '\0' ||
      config->outputs[0] == '\0' || config->export_path[0] == '\0') {
    fprintf(stderr,
            "config: csv, inputs, outputs and export.out are required\n");
    return ELML_ERR_INVALID_ARG;
  }
  if (config->hidden_count == 0u) {
    fprintf(stderr, "config: model.layers is required\n");
    return ELML_ERR_INVALID_ARG;
  }
  if (config->hidden_count > ELML_MAX_HIDDEN_LAYERS) {
    return ELML_ERR_INVALID_ARG;
  }

  if (config->activation_count == 0u) {
    for (uint32_t i = 0u; i < config->hidden_count; ++i) {
      config->activations[i] = ELML_ACT_RELU;
    }
    config->activations[config->hidden_count] =
        (config->loss == ELML_LOSS_CCE) ? ELML_ACT_SOFTMAX : ELML_ACT_LINEAR;
    config->activation_count = config->hidden_count + 1u;
  } else if (config->activation_count == config->hidden_count) {
    config->activations[config->hidden_count] =
        (config->loss == ELML_LOSS_CCE) ? ELML_ACT_SOFTMAX : ELML_ACT_LINEAR;
    config->activation_count++;
  } else if (config->activation_count != config->hidden_count + 1u) {
    fprintf(stderr, "config: activations must have %u or %u entries\n",
            (unsigned)config->hidden_count,
            (unsigned)(config->hidden_count + 1u));
    return ELML_ERR_INVALID_ARG;
  }

  config->classification = (config->loss == ELML_LOSS_CCE);
  const elml_activation output_act =
      config->activations[config->activation_count - 1u];
  if (config->classification && output_act != ELML_ACT_SOFTMAX) {
    fprintf(stderr, "config: cce requires a softmax output activation\n");
    return ELML_ERR_INVALID_ARG;
  }
  if (!config->classification && output_act != ELML_ACT_LINEAR) {
    fprintf(stderr, "config: regression requires a linear output activation\n");
    return ELML_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0u; i + 1u < config->activation_count; ++i) {
    if (config->activations[i] == ELML_ACT_SOFTMAX) {
      fprintf(stderr, "config: softmax is only allowed on the output layer\n");
      return ELML_ERR_INVALID_ARG;
    }
  }

  if (config->split < 0.0 || config->split > 1.0) {
    fprintf(stderr, "config: val split must be in [0, 1]\n");
    return ELML_ERR_INVALID_ARG;
  }
  if (config->batch == 0u || config->epochs == 0u ||
      config->learning_rate <= 0.0) {
    fprintf(stderr, "config: batch, epochs and lr must be positive\n");
    return ELML_ERR_INVALID_ARG;
  }
  if (config->threads == 0u) {
    config->threads = 1u;
  }
  if (config->threads > 1u) {
    fprintf(stderr,
            "config: threads > 1 is reserved for a later milestone; using 1\n");
    config->threads = 1u;
  }
  if (config->l2 < 0.0) {
    fprintf(stderr, "config: l2 must be non-negative\n");
    return ELML_ERR_INVALID_ARG;
  }
  return ELML_OK;
}

elml_status elml_config_load(elml_config *config, int argc, char **argv)
{
  if (config == nullptr || argc < 1 || argv == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  elml_config_init(config);

  const char *config_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--config") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "config: --config requires a value\n");
        return ELML_ERR_INVALID_ARG;
      }
      config_path = argv[i + 1];
    }
  }

  if (config_path != nullptr) {
    elml_ini ini;
    elml_status status = elml_ini_load(config_path, &ini);
    if (status != ELML_OK) {
      fprintf(stderr, "config: cannot read %s\n", config_path);
      return status;
    }
    status = elml_apply_ini(config, &ini);
    elml_ini_free(&ini);
    if (status != ELML_OK) {
      return status;
    }
  }

  elml_status status = elml_apply_cli(config, argc, argv);
  if (status != ELML_OK) {
    return status;
  }
  return elml_finalize_config(config);
}

const char *elml_loss_name(elml_loss loss)
{
  switch (loss) {
  case ELML_LOSS_MSE:
    return "mse";
  case ELML_LOSS_MAE:
    return "mae";
  case ELML_LOSS_CCE:
    return "cce";
  }
  return "unknown";
}

const char *elml_activation_name(elml_activation activation)
{
  switch (activation) {
  case ELML_ACT_NONE:
    return "none";
  case ELML_ACT_RELU:
    return "relu";
  case ELML_ACT_LEAKY_RELU:
    return "leaky_relu";
  case ELML_ACT_TANH:
    return "tanh";
  case ELML_ACT_SIGMOID:
    return "sigmoid";
  case ELML_ACT_SOFTMAX:
    return "softmax";
  case ELML_ACT_LINEAR:
    return "linear";
  }
  return "unknown";
}

void elml_config_print_help(void)
{
  printf("usage: elml_train --config FILE [overrides]\n"
         "       elml_train --csv FILE --inputs A,B --outputs Y ...\n"
         "\n"
         "data:\n"
         "  --csv FILE            training CSV\n"
         "  --val-csv FILE        optional validation CSV\n"
         "  --inputs A,B          input column names (or 1-based indices)\n"
         "  --outputs Y           target column name(s)\n"
         "  --no-header           CSV has no header row\n"
         "  --delimiter C         field delimiter (default ',')\n"
         "  --missing MODE        error | zero | mean (default error)\n"
         "  --val FRACTION        validation split (default 0.8 train)\n"
         "  --seed N              shuffling and init seed\n"
         "\n"
         "model:\n"
         "  --layers N,N          hidden layer sizes\n"
         "  --act A,A,...         activations including the output layer\n"
         "  --loss mse|mae|cce    loss and task selection\n"
         "\n"
         "training:\n"
         "  --epochs N            default 100\n"
         "  --batch N             default 16\n"
         "  --opt adam|sgd        default adam\n"
         "  --lr R                default 0.001\n"
         "  --l2 R                L2 weight decay\n"
         "  --early-stop N        default 20 epochs\n"
         "  --threads N           worker count (1 in this milestone)\n"
         "\n"
         "export:\n"
         "  --export FILE         output .elml path\n");
}
