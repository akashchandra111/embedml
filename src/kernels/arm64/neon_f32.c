#if defined(__aarch64__) && defined(__ARM_NEON)

#include "kernels/arm64/arm64.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

void elml_arm64_dense_f32(const float *weights, const float *biases,
                          const float *input, float *output, uint32_t in_dim,
                          uint32_t out_dim)
{
  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    uint32_t i = 0u;

    for (; in_dim - i >= 4u; i += 4u) {
      accumulator = vfmaq_f32(accumulator, vld1q_f32(row + i),
                              vld1q_f32(input + i));
    }

    float sum = vaddvq_f32(accumulator) + biases[o];
    for (; i < in_dim; ++i) {
      sum += row[i] * input[i];
    }
    output[o] = sum;
  }
}

void elml_arm64_dense_backward_f32(const float *delta, const float *input,
                                   float *grad_w, float *grad_b,
                                   uint32_t in_dim, uint32_t out_dim)
{
  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float d = delta[o];
    float *row = grad_w + ((size_t)o * (size_t)in_dim);
    const float32x4_t vd = vdupq_n_f32(d);
    uint32_t i = 0u;

    for (; in_dim - i >= 4u; i += 4u) {
      const float32x4_t g = vld1q_f32(row + i);
      const float32x4_t x = vld1q_f32(input + i);
      vst1q_f32(row + i, vfmaq_f32(g, vd, x));
    }
    for (; i < in_dim; ++i) {
      row[i] += d * input[i];
    }
    grad_b[o] += d;
  }
}

void elml_arm64_dense_input_grad_f32(const float *weights, const float *delta,
                                     float *grad_input, uint32_t in_dim,
                                     uint32_t out_dim)
{
  const float32x4_t zero = vdupq_n_f32(0.0f);
  uint32_t i = 0u;
  for (; in_dim - i >= 4u; i += 4u) {
    vst1q_f32(grad_input + i, zero);
  }
  for (; i < in_dim; ++i) {
    grad_input[i] = 0.0f;
  }

  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    const float32x4_t vd = vdupq_n_f32(delta[o]);
    i = 0u;

    for (; in_dim - i >= 4u; i += 4u) {
      const float32x4_t g = vld1q_f32(grad_input + i);
      const float32x4_t w = vld1q_f32(row + i);
      vst1q_f32(grad_input + i, vfmaq_f32(g, vd, w));
    }
    for (; i < in_dim; ++i) {
      grad_input[i] += row[i] * delta[o];
    }
  }
}

#endif /* aarch64 && NEON */
