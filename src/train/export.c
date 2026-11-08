#include "train/export.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "elml/model.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "format/write.h"
#include "train/config.h"
#include "train/dataset.h"
#include "train/io.h"
#include "train/net.h"

#define ELML_LEAKY_ALPHA 0.01f

static elml_status elml_interleave_norm(float *out, const float *mean,
                                        const float *stddev, uint32_t count)
{
  for (uint32_t i = 0u; i < count; ++i) {
    out[2u * i] = mean[i];
    out[2u * i + 1u] = stddev[i];
  }
  return ELML_OK;
}

static char *elml_join_labels(const elml_dataset *dataset)
{
  if (dataset == nullptr || dataset->label_count == 0u) {
    return nullptr;
  }

  size_t total = 1u;
  for (uint32_t i = 0u; i < dataset->label_count; ++i) {
    total += strlen(dataset->labels[i]);
    if (i + 1u < dataset->label_count) {
      total++; /* comma */
    }
  }

  char *joined = (char *)malloc(total);
  if (joined == nullptr) {
    return nullptr;
  }
  size_t offset = 0u;
  for (uint32_t i = 0u; i < dataset->label_count; ++i) {
    const size_t len = strlen(dataset->labels[i]);
    memcpy(joined + offset, dataset->labels[i], len);
    offset += len;
    if (i + 1u < dataset->label_count) {
      joined[offset] = ',';
      offset++;
    }
  }
  joined[offset] = '\0';
  return joined;
}

static elml_status elml_build_norm(const elml_net *net,
                                   const elml_norm_stats *norm,
                                   elml_write_norm *out, float **input_params,
                                   float **output_params)
{
  memset(out, 0, sizeof *out);
  *input_params = nullptr;
  *output_params = nullptr;

  if (norm == nullptr) {
    return ELML_OK;
  }

  if (norm->has_input) {
    const size_t count = (size_t)net->input_dim;
    *input_params = (float *)malloc(count * 2u * sizeof(float));
    if (*input_params == nullptr) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    elml_interleave_norm(*input_params, norm->input_mean, norm->input_std,
                         net->input_dim);
    out->input_mode = (uint8_t)ELML_NORM_STANDARD;
    out->input_params = *input_params;
  }

  if (norm->has_output) {
    const size_t count = (size_t)net->output_dim;
    *output_params = (float *)malloc(count * 2u * sizeof(float));
    if (*output_params == nullptr) {
      free(*input_params);
      *input_params = nullptr;
      return ELML_ERR_OUT_OF_MEMORY;
    }
    elml_interleave_norm(*output_params, norm->output_mean, norm->output_std,
                         net->output_dim);
    out->output_mode = (uint8_t)ELML_NORM_STANDARD;
    out->output_params = *output_params;
  }

  return ELML_OK;
}

elml_status elml_export_model(const elml_net *net,
                              const elml_norm_stats *norm,
                              const elml_dataset *dataset,
                              const elml_config *config, const char *path)
{
  if (net == nullptr || config == nullptr || path == nullptr ||
      net->layer_count == 0u) {
    return ELML_ERR_INVALID_ARG;
  }

  elml_status status = ELML_OK;
  elml_write_layer *layers = (elml_write_layer *)calloc(
      net->layer_count, sizeof(elml_write_layer));
  if (layers == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }

  for (uint32_t i = 0u; i < net->layer_count; ++i) {
    const elml_net_layer *layer = &net->layers[i];
    layers[i].in_dim = layer->in_dim;
    layers[i].out_dim = layer->out_dim;
    layers[i].act = layer->act;
    layers[i].leaky_alpha =
        (layer->act == ELML_ACT_LEAKY_RELU) ? ELML_LEAKY_ALPHA : 0.0f;
    layers[i].weights = layer->weights;
    layers[i].biases = layer->biases;
  }

  elml_write_norm write_norm;
  float *input_params = nullptr;
  float *output_params = nullptr;
  status = elml_build_norm(net, norm, &write_norm, &input_params,
                           &output_params);
  if (status != ELML_OK) {
    free(layers);
    return status;
  }

  char *labels = elml_join_labels(dataset);
  const bool has_labels = labels != nullptr;

  elml_write_meta meta[6];
  uint32_t meta_count = 0u;
  if (has_labels) {
    meta[meta_count].key = "labels";
    meta[meta_count].value = labels;
    meta_count++;
  }
  if (config->inputs[0] != '\0') {
    meta[meta_count].key = "columns.inputs";
    meta[meta_count].value = config->inputs;
    meta_count++;
  }
  if (config->outputs[0] != '\0') {
    meta[meta_count].key = "columns.outputs";
    meta[meta_count].value = config->outputs;
    meta_count++;
  }
  meta[meta_count].key = "task";
  meta[meta_count].value =
      (net->loss == ELML_LOSS_CCE) ? "classification" : "regression";
  meta_count++;
  meta[meta_count].key = "loss";
  meta[meta_count].value = elml_loss_name(net->loss);
  meta_count++;

  elml_write_spec spec = {
      .input_dim = net->input_dim,
      .output_dim = net->output_dim,
      .layer_count = (uint16_t)net->layer_count,
      .layers = layers,
      .norm = write_norm,
      .meta = meta,
      .meta_count = meta_count,
  };

  elml_buf buffer = {0};
  status = elml_model_write(&spec, &buffer);
  if (status == ELML_OK) {
    status = elml_write_file(path, buffer.data, buffer.len);
  }

  elml_buf_free(&buffer);
  free(labels);
  free(input_params);
  free(output_params);
  free(layers);
  return status;
}
