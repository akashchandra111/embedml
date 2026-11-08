#include "train/net.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ckdint_compat.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "kernels/kernels.h"
#include "train/config.h"
#include "train/dataset.h"

#define ELML_LEAKY_ALPHA 0.01f

static float elml_activation_derivative(const elml_net_layer *layer,
                                        uint32_t index)
{
  switch (layer->act) {
  case ELML_ACT_RELU:
    return (layer->z[index] > 0.0f) ? 1.0f : 0.0f;
  case ELML_ACT_LEAKY_RELU:
    return (layer->z[index] > 0.0f) ? 1.0f : ELML_LEAKY_ALPHA;
  case ELML_ACT_TANH: {
    const float value = layer->a[index];
    return 1.0f - value * value;
  }
  case ELML_ACT_SIGMOID: {
    const float value = layer->a[index];
    return value * (1.0f - value);
  }
  case ELML_ACT_LINEAR:
  case ELML_ACT_NONE:
  case ELML_ACT_SOFTMAX:
  default:
    return 1.0f;
  }
}

static float *elml_alloc_floats(size_t count)
{
  if (count == 0u) {
    return nullptr;
  }
  return (float *)calloc(count, sizeof(float));
}

static elml_status elml_layer_alloc(elml_net_layer *layer)
{
  size_t weight_count;
  size_t bias_count = (size_t)layer->out_dim;
  if (ckd_mul(&weight_count, (size_t)layer->in_dim,
              (size_t)layer->out_dim)) {
    return ELML_ERR_OUT_OF_MEMORY;
  }

  layer->weights = elml_alloc_floats(weight_count);
  layer->biases = elml_alloc_floats(bias_count);
  layer->grad_w = elml_alloc_floats(weight_count);
  layer->grad_b = elml_alloc_floats(bias_count);
  layer->z = elml_alloc_floats(bias_count);
  layer->a = elml_alloc_floats(bias_count);
  layer->delta = elml_alloc_floats(bias_count);
  layer->best_weights = elml_alloc_floats(weight_count);
  layer->best_biases = elml_alloc_floats(bias_count);
  /* Optimizer state is always allocated so the update loop stays branch-free
   * with respect to pointer validity. */
  layer->adam_mw = elml_alloc_floats(weight_count);
  layer->adam_vw = elml_alloc_floats(weight_count);
  layer->adam_mb = elml_alloc_floats(bias_count);
  layer->adam_vb = elml_alloc_floats(bias_count);

  if (layer->weights == nullptr || layer->biases == nullptr ||
      layer->grad_w == nullptr || layer->grad_b == nullptr ||
      layer->z == nullptr || layer->a == nullptr || layer->delta == nullptr ||
      layer->best_weights == nullptr || layer->best_biases == nullptr ||
      layer->adam_mw == nullptr || layer->adam_vw == nullptr ||
      layer->adam_mb == nullptr || layer->adam_vb == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  return ELML_OK;
}

static void elml_layer_free(elml_net_layer *layer)
{
  free(layer->weights);
  free(layer->biases);
  free(layer->grad_w);
  free(layer->grad_b);
  free(layer->adam_mw);
  free(layer->adam_vw);
  free(layer->adam_mb);
  free(layer->adam_vb);
  free(layer->z);
  free(layer->a);
  free(layer->delta);
  free(layer->best_weights);
  free(layer->best_biases);
  memset(layer, 0, sizeof *layer);
}

static float elml_uniform(uint64_t *state)
{
  const uint32_t bits = (uint32_t)(elml_rng_next(state) >> 32);
  return (float)bits / 4294967296.0f;
}

static elml_status elml_net_init_weights(elml_net *net, uint64_t seed)
{
  uint64_t state = seed ^ 0xD1B54A32D192ED03ull;
  if (state == 0u) {
    state = 0x9E3779B97F4A7C15ull;
  }

  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    elml_net_layer *layer = &net->layers[l];
    const size_t weight_count =
        (size_t)layer->in_dim * (size_t)layer->out_dim;

    float limit;
    if (layer->act == ELML_ACT_RELU || layer->act == ELML_ACT_LEAKY_RELU) {
      limit = sqrtf(6.0f / (float)layer->in_dim);
    } else {
      limit = sqrtf(6.0f / (float)(layer->in_dim + layer->out_dim));
    }

    for (size_t i = 0u; i < weight_count; ++i) {
      layer->weights[i] = (elml_uniform(&state) * 2.0f - 1.0f) * limit;
    }
    for (uint32_t i = 0u; i < layer->out_dim; ++i) {
      layer->biases[i] = 0.0f;
    }
  }
  return ELML_OK;
}

elml_status elml_net_create(elml_net *net, const uint32_t *hidden,
                            uint32_t hidden_count,
                            const elml_activation *activations,
                            uint32_t activation_count, uint32_t input_dim,
                            uint32_t output_dim, elml_loss loss,
                            elml_optimizer optimizer, float learning_rate,
                            float l2, uint64_t seed)
{
  if (net == nullptr || hidden == nullptr || activations == nullptr ||
      hidden_count == 0u || hidden_count > ELML_MAX_HIDDEN_LAYERS ||
      activation_count != hidden_count + 1u || input_dim == 0u ||
      output_dim == 0u || learning_rate <= 0.0f || l2 < 0.0f) {
    return ELML_ERR_INVALID_ARG;
  }

  const elml_activation output_act = activations[hidden_count];
  if (loss == ELML_LOSS_CCE) {
    if (output_act != ELML_ACT_SOFTMAX) {
      return ELML_ERR_INVALID_ARG;
    }
  } else if (output_act == ELML_ACT_SOFTMAX) {
    return ELML_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0u; i < hidden_count; ++i) {
    if (activations[i] == ELML_ACT_SOFTMAX) {
      return ELML_ERR_INVALID_ARG;
    }
  }

  memset(net, 0, sizeof *net);
  net->input_dim = input_dim;
  net->output_dim = output_dim;
  net->layer_count = hidden_count + 1u;
  net->loss = loss;
  net->optimizer = optimizer;
  net->learning_rate = learning_rate;
  net->l2 = l2;
  net->beta1 = 0.9f;
  net->beta2 = 0.999f;
  net->epsilon = 1e-8f;
  net->beta1_pow = 1.0f;
  net->beta2_pow = 1.0f;

  net->input_copy = elml_alloc_floats((size_t)input_dim);
  if (net->input_copy == nullptr) {
    elml_net_free(net);
    return ELML_ERR_OUT_OF_MEMORY;
  }

  uint32_t in_dim = input_dim;
  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    const uint32_t out_dim =
        (l + 1u == net->layer_count) ? output_dim : hidden[l];
    elml_net_layer *layer = &net->layers[l];
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->act = activations[l];

    const elml_status status = elml_layer_alloc(layer);
    if (status != ELML_OK) {
      elml_net_free(net);
      return status;
    }
    in_dim = out_dim;
  }

  return elml_net_init_weights(net, seed);
}

void elml_net_free(elml_net *net)
{
  if (net == nullptr) {
    return;
  }
  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    elml_layer_free(&net->layers[l]);
  }
  free(net->input_copy);
  memset(net, 0, sizeof *net);
}

elml_status elml_net_forward(elml_net *net, const float *input, float *output)
{
  if (net == nullptr || input == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  memcpy(net->input_copy, input, (size_t)net->input_dim * sizeof(float));

  const float *current = net->input_copy;
  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    elml_net_layer *layer = &net->layers[l];
    elml_kernel_dense_f32(layer->weights, layer->biases, current, layer->z,
                          layer->in_dim, layer->out_dim);
    elml_kernel_activation_f32(layer->z, layer->a, layer->out_dim, layer->act,
                               ELML_LEAKY_ALPHA);
    current = layer->a;
  }

  if (output != nullptr) {
    memcpy(output, current, (size_t)net->output_dim * sizeof(float));
  }
  return ELML_OK;
}

uint32_t elml_net_argmax(const elml_net *net)
{
  if (net == nullptr || net->layer_count == 0u) {
    return 0u;
  }
  const elml_net_layer *layer = &net->layers[net->layer_count - 1u];
  uint32_t best = 0u;
  for (uint32_t i = 1u; i < layer->out_dim; ++i) {
    if (layer->a[i] > layer->a[best]) {
      best = i;
    }
  }
  return best;
}

void elml_net_zero_grad(elml_net *net)
{
  if (net == nullptr) {
    return;
  }
  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    elml_net_layer *layer = &net->layers[l];
    const size_t weight_count = (size_t)layer->in_dim * layer->out_dim;
    memset(layer->grad_w, 0, weight_count * sizeof(float));
    memset(layer->grad_b, 0, (size_t)layer->out_dim * sizeof(float));
  }
}

static void elml_net_backward(elml_net *net, const float *target,
                              uint32_t class_index)
{
  elml_net_layer *last = &net->layers[net->layer_count - 1u];

  if (net->loss == ELML_LOSS_CCE) {
    for (uint32_t o = 0u; o < last->out_dim; ++o) {
      const float expected = (o == class_index) ? 1.0f : 0.0f;
      last->delta[o] = last->a[o] - expected;
    }
  } else {
    const float scale =
        (net->loss == ELML_LOSS_MSE) ? (2.0f / (float)last->out_dim)
                                     : (1.0f / (float)last->out_dim);
    for (uint32_t o = 0u; o < last->out_dim; ++o) {
      const float diff = last->a[o] - target[o];
      float derivative;
      if (net->loss == ELML_LOSS_MAE) {
        derivative = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
      } else {
        derivative = diff;
      }
      last->delta[o] =
          derivative * elml_activation_derivative(last, o) * scale;
    }
  }

  for (uint32_t l = net->layer_count; l > 0u; --l) {
    const uint32_t index = l - 1u;
    elml_net_layer *layer = &net->layers[index];
    const float *input =
        (index == 0u) ? net->input_copy : net->layers[index - 1u].a;

    elml_kernel_dense_backward_f32(layer->delta, input, layer->grad_w,
                                   layer->grad_b, layer->in_dim,
                                   layer->out_dim);

    if (index > 0u) {
      elml_net_layer *previous = &net->layers[index - 1u];
      elml_kernel_dense_input_grad_f32(layer->weights, layer->delta,
                                       previous->delta, layer->in_dim,
                                       layer->out_dim);
      for (uint32_t i = 0u; i < previous->out_dim; ++i) {
        previous->delta[i] *= elml_activation_derivative(previous, i);
      }
    }
  }
}

elml_status elml_net_accumulate(elml_net *net, const float *input,
                                const float *target, uint32_t class_index)
{
  if (net == nullptr || input == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (net->loss != ELML_LOSS_CCE && target == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (net->loss == ELML_LOSS_CCE &&
      class_index >= net->layers[net->layer_count - 1u].out_dim) {
    return ELML_ERR_INVALID_ARG;
  }

  const elml_status status = elml_net_forward(net, input, nullptr);
  if (status != ELML_OK) {
    return status;
  }
  elml_net_backward(net, target, class_index);
  return ELML_OK;
}

static void elml_net_update_parameter(float *parameter, float *gradient,
                                      float *moment, float *velocity,
                                      float scale, float learning_rate,
                                      float l2, bool decay,
                                      elml_optimizer optimizer,
                                      float beta1, float beta2, float epsilon,
                                      float beta1_pow, float beta2_pow)
{
  float value = gradient[0] * scale;
  if (decay) {
    value += l2 * parameter[0];
  }

  if (optimizer == ELML_OPTIMIZER_ADAM) {
    moment[0] = beta1 * moment[0] + (1.0f - beta1) * value;
    velocity[0] = beta2 * velocity[0] + (1.0f - beta2) * value * value;
    const float m_hat = moment[0] / (1.0f - beta1_pow);
    const float v_hat = velocity[0] / (1.0f - beta2_pow);
    parameter[0] -=
        learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
  } else {
    parameter[0] -= learning_rate * value;
  }
}

void elml_net_update(elml_net *net, uint32_t batch_size)
{
  if (net == nullptr || batch_size == 0u) {
    return;
  }
  const float scale = 1.0f / (float)batch_size;
  net->step++;
  net->beta1_pow *= net->beta1;
  net->beta2_pow *= net->beta2;

  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    elml_net_layer *layer = &net->layers[l];
    const size_t weight_count = (size_t)layer->in_dim * layer->out_dim;
    for (size_t i = 0u; i < weight_count; ++i) {
      elml_net_update_parameter(&layer->weights[i], &layer->grad_w[i],
                                &layer->adam_mw[i], &layer->adam_vw[i], scale,
                                net->learning_rate, net->l2, true,
                                net->optimizer, net->beta1, net->beta2,
                                net->epsilon, net->beta1_pow, net->beta2_pow);
    }
    for (uint32_t i = 0u; i < layer->out_dim; ++i) {
      elml_net_update_parameter(&layer->biases[i], &layer->grad_b[i],
                                &layer->adam_mb[i], &layer->adam_vb[i], scale,
                                net->learning_rate, net->l2, false,
                                net->optimizer, net->beta1, net->beta2,
                                net->epsilon, net->beta1_pow, net->beta2_pow);
    }
  }
}

float elml_net_evaluate(elml_net *net, const elml_dataset *dataset,
                        uint32_t *correct)
{
  if (correct != nullptr) {
    *correct = 0u;
  }
  if (net == nullptr || dataset == nullptr || dataset->rows == 0u) {
    return 0.0f;
  }

  const elml_net_layer *last = &net->layers[net->layer_count - 1u];
  float total = 0.0f;
  uint32_t hits = 0u;

  for (uint32_t row = 0u; row < dataset->rows; ++row) {
    const float *input =
        &dataset->x[(size_t)row * dataset->input_dim];
    if (elml_net_forward(net, input, nullptr) != ELML_OK) {
      break;
    }

    if (net->loss == ELML_LOSS_CCE) {
      uint32_t target = 0u;
      if (dataset->class_index != nullptr) {
        target = dataset->class_index[row];
      } else {
        const float *expected =
            &dataset->y[(size_t)row * dataset->output_dim];
        for (uint32_t i = 1u; i < dataset->output_dim; ++i) {
          if (expected[i] > expected[target]) {
            target = i;
          }
        }
      }
      const float probability = last->a[target];
      total += -logf(probability + 1e-7f);
      if (elml_net_argmax(net) == target) {
        hits++;
      }
    } else {
      const float *expected =
          &dataset->y[(size_t)row * dataset->output_dim];
      for (uint32_t o = 0u; o < dataset->output_dim; ++o) {
        const float diff = last->a[o] - expected[o];
        total += (net->loss == ELML_LOSS_MAE) ? fabsf(diff) : diff * diff;
      }
    }
  }

  float denominator = (float)dataset->rows;
  if (net->loss != ELML_LOSS_CCE) {
    denominator *= (float)dataset->output_dim;
  }
  if (correct != nullptr) {
    *correct = hits;
  }
  return total / denominator;
}

void elml_net_save_best(elml_net *net)
{
  if (net == nullptr) {
    return;
  }
  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    const elml_net_layer *layer = &net->layers[l];
    const size_t weight_count = (size_t)layer->in_dim * layer->out_dim;
    memcpy(layer->best_weights, layer->weights,
           weight_count * sizeof(float));
    memcpy(layer->best_biases, layer->biases,
           (size_t)layer->out_dim * sizeof(float));
  }
  net->has_best = true;
}

void elml_net_restore_best(elml_net *net)
{
  if (net == nullptr || !net->has_best) {
    return;
  }
  for (uint32_t l = 0u; l < net->layer_count; ++l) {
    elml_net_layer *layer = &net->layers[l];
    const size_t weight_count = (size_t)layer->in_dim * layer->out_dim;
    memcpy(layer->weights, layer->best_weights,
           weight_count * sizeof(float));
    memcpy(layer->biases, layer->best_biases,
           (size_t)layer->out_dim * sizeof(float));
  }
}
