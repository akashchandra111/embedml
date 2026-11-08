#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kernels/kernels.h"
#include "test.h"

static uint32_t g_rng = 0x12345678u;

static uint32_t elml_test_rand(void)
{
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

static float elml_test_value(void)
{
  return ((float)(elml_test_rand() % 2001u) / 1000.0f) - 1.0f;
}

static void elml_fill(float *values, size_t count)
{
  for (size_t i = 0u; i < count; ++i) {
    values[i] = elml_test_value();
  }
}

static int elml_close(float actual, float expected)
{
  const float tolerance = 1e-4f;
  float difference = actual - expected;
  if (difference < 0.0f) {
    difference = -difference;
  }
  const float scale = (expected < 0.0f) ? -expected : expected;
  return difference <= tolerance * (1.0f + scale);
}

static void elml_check_dense(uint32_t in_dim, uint32_t out_dim)
{
  const size_t weight_count = (size_t)in_dim * (size_t)out_dim;
  float *weights = (float *)malloc(weight_count * sizeof(float));
  float *input = (float *)malloc((size_t)in_dim * sizeof(float));
  float *biases = (float *)malloc((size_t)out_dim * sizeof(float));
  float *expected = (float *)malloc((size_t)out_dim * sizeof(float));
  float *actual = (float *)malloc((size_t)out_dim * sizeof(float));
  CHECK(weights != NULL && input != NULL && biases != NULL &&
        expected != NULL && actual != NULL);
  if (weights == NULL || input == NULL || biases == NULL ||
      expected == NULL || actual == NULL) {
    free(weights);
    free(input);
    free(biases);
    free(expected);
    free(actual);
    return;
  }

  elml_fill(weights, weight_count);
  elml_fill(input, in_dim);
  elml_fill(biases, out_dim);

  elml_scalar_dense_f32(weights, biases, input, expected, in_dim, out_dim);
  elml_kernel_dense_f32(weights, biases, input, actual, in_dim, out_dim);
  for (uint32_t o = 0u; o < out_dim; ++o) {
    CHECK(elml_close(actual[o], expected[o]));
  }

  free(weights);
  free(input);
  free(biases);
  free(expected);
  free(actual);
}

static void elml_check_backward(uint32_t in_dim, uint32_t out_dim)
{
  const size_t weight_count = (size_t)in_dim * (size_t)out_dim;
  float *input = (float *)malloc((size_t)in_dim * sizeof(float));
  float *delta = (float *)malloc((size_t)out_dim * sizeof(float));
  float *grad_expected = (float *)malloc(weight_count * sizeof(float));
  float *grad_actual = (float *)malloc(weight_count * sizeof(float));
  float *bias_expected = (float *)malloc((size_t)out_dim * sizeof(float));
  float *bias_actual = (float *)malloc((size_t)out_dim * sizeof(float));
  CHECK(input != NULL && delta != NULL && grad_expected != NULL &&
        grad_actual != NULL && bias_expected != NULL && bias_actual != NULL);
  if (input == NULL || delta == NULL || grad_expected == NULL ||
      grad_actual == NULL || bias_expected == NULL || bias_actual == NULL) {
    free(input);
    free(delta);
    free(grad_expected);
    free(grad_actual);
    free(bias_expected);
    free(bias_actual);
    return;
  }

  elml_fill(input, in_dim);
  elml_fill(delta, out_dim);
  elml_fill(grad_expected, weight_count);
  memcpy(grad_actual, grad_expected, weight_count * sizeof(float));
  elml_fill(bias_expected, out_dim);
  memcpy(bias_actual, bias_expected, (size_t)out_dim * sizeof(float));

  elml_scalar_dense_backward_f32(delta, input, grad_expected, bias_expected,
                                 in_dim, out_dim);
  elml_kernel_dense_backward_f32(delta, input, grad_actual, bias_actual,
                                 in_dim, out_dim);
  for (size_t i = 0u; i < weight_count; ++i) {
    CHECK(elml_close(grad_actual[i], grad_expected[i]));
  }
  for (uint32_t o = 0u; o < out_dim; ++o) {
    CHECK(elml_close(bias_actual[o], bias_expected[o]));
  }

  free(input);
  free(delta);
  free(grad_expected);
  free(grad_actual);
  free(bias_expected);
  free(bias_actual);
}

static void elml_check_input_grad(uint32_t in_dim, uint32_t out_dim)
{
  const size_t weight_count = (size_t)in_dim * (size_t)out_dim;
  float *weights = (float *)malloc(weight_count * sizeof(float));
  float *delta = (float *)malloc((size_t)out_dim * sizeof(float));
  float *expected = (float *)malloc((size_t)in_dim * sizeof(float));
  float *actual = (float *)malloc((size_t)in_dim * sizeof(float));
  CHECK(weights != NULL && delta != NULL && expected != NULL &&
        actual != NULL);
  if (weights == NULL || delta == NULL || expected == NULL ||
      actual == NULL) {
    free(weights);
    free(delta);
    free(expected);
    free(actual);
    return;
  }

  elml_fill(weights, weight_count);
  elml_fill(delta, out_dim);
  elml_fill(expected, in_dim);
  memcpy(actual, expected, (size_t)in_dim * sizeof(float));

  elml_scalar_dense_input_grad_f32(weights, delta, expected, in_dim, out_dim);
  elml_kernel_dense_input_grad_f32(weights, delta, actual, in_dim, out_dim);
  for (uint32_t i = 0u; i < in_dim; ++i) {
    CHECK(elml_close(actual[i], expected[i]));
  }

  free(weights);
  free(delta);
  free(expected);
  free(actual);
}

int main(void)
{
  static const uint32_t sizes[] = {1u,  2u,  3u,  4u,  5u,  7u,  8u,
                                   9u,  15u, 16u, 17u, 31u, 32u, 33u};
  static const uint32_t out_sizes[] = {1u, 2u, 3u, 8u, 9u};

  for (size_t s = 0u; s < sizeof sizes / sizeof sizes[0]; ++s) {
    for (size_t o = 0u; o < sizeof out_sizes / sizeof out_sizes[0]; ++o) {
      elml_check_dense(sizes[s], out_sizes[o]);
      elml_check_backward(sizes[s], out_sizes[o]);
      elml_check_input_grad(sizes[s], out_sizes[o]);
    }
  }

  TEST_END();
}
