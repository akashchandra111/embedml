#include "kernels/kernels.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/registry.h"

static float elml_scalar_activate(float x, elml_activation act, float alpha)
{
  switch (act) {
  case ELML_ACT_RELU:
    return (x > 0.0f) ? x : 0.0f;
  case ELML_ACT_LEAKY_RELU:
    return (x > 0.0f) ? x : (alpha * x);
  case ELML_ACT_TANH:
    return tanhf(x);
  case ELML_ACT_SIGMOID:
    return 1.0f / (1.0f + expf(-x));
  case ELML_ACT_LINEAR:
  case ELML_ACT_NONE:
  case ELML_ACT_SOFTMAX: /* handled by the caller */
    return x;
  }
  return x;
}

static void elml_scalar_softmax(float *values, uint32_t count)
{
  if (count == 0u) {
    return;
  }

  float maximum = values[0];
  for (uint32_t i = 1u; i < count; ++i) {
    if (values[i] > maximum) {
      maximum = values[i];
    }
  }

  float sum = 0.0f;
  for (uint32_t i = 0u; i < count; ++i) {
    const float e = expf(values[i] - maximum);
    values[i] = e;
    sum += e;
  }

  if (sum > 0.0f) {
    const float inv = 1.0f / sum;
    for (uint32_t i = 0u; i < count; ++i) {
      values[i] *= inv;
    }
  }
}

void elml_kernel_activation_f32(const float *input, float *output,
                                uint32_t count, elml_activation act,
                                float leaky_alpha)
{
  if (act == ELML_ACT_SOFTMAX) {
    for (uint32_t i = 0u; i < count; ++i) {
      output[i] = input[i];
    }
    elml_scalar_softmax(output, count);
    return;
  }

  for (uint32_t i = 0u; i < count; ++i) {
    output[i] = elml_scalar_activate(input[i], act, leaky_alpha);
  }
}

void elml_scalar_dense_f32(const float *weights, const float *biases,
                           const float *input, float *output, uint32_t in_dim,
                           uint32_t out_dim)
{
  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    float sum = biases[o];
    for (uint32_t i = 0u; i < in_dim; ++i) {
      sum += row[i] * input[i];
    }
    output[o] = sum;
  }
}

void elml_scalar_dense_backward_f32(const float *delta, const float *input,
                                    float *grad_w, float *grad_b,
                                    uint32_t in_dim, uint32_t out_dim)
{
  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float d = delta[o];
    float *row = grad_w + ((size_t)o * (size_t)in_dim);
    for (uint32_t i = 0u; i < in_dim; ++i) {
      row[i] += d * input[i];
    }
    grad_b[o] += d;
  }
}

void elml_scalar_dense_input_grad_f32(const float *weights, const float *delta,
                                      float *grad_input, uint32_t in_dim,
                                      uint32_t out_dim)
{
  for (uint32_t i = 0u; i < in_dim; ++i) {
    grad_input[i] = 0.0f;
  }
  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float d = delta[o];
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    for (uint32_t i = 0u; i < in_dim; ++i) {
      grad_input[i] += row[i] * d;
    }
  }
}
