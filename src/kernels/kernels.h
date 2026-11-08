#ifndef ELML_KERNELS_H
#define ELML_KERNELS_H

#include <stdint.h>

#include "elml/registry.h"

/*
 * Common kernel interface used by the inference core and the host trainer.
 *
 * The functions below are the only entry points the rest of the library may
 * call. They dispatch to the best implementation available for the target at
 * run time (x86_64) or at compile time (arm64, riscv vector). Architecture
 * specific code lives under x86_64/, arm64/ and riscv/; the scalar reference
 * and the dispatcher live under common/.
 */

/* output[o] = biases[o] + dot(weights[o], input) */
void elml_kernel_dense_f32(const float *weights, const float *biases,
                           const float *input, float *output, uint32_t in_dim,
                           uint32_t out_dim);

/* Elementwise activation, including softmax when act is ELML_ACT_SOFTMAX. */
void elml_kernel_activation_f32(const float *input, float *output,
                                uint32_t count, elml_activation act,
                                float leaky_alpha);

/* grad_w[o][i] += delta[o] * input[i]; grad_b[o] += delta[o]. */
void elml_kernel_dense_backward_f32(const float *delta, const float *input,
                                    float *grad_w, float *grad_b,
                                    uint32_t in_dim, uint32_t out_dim);

/* grad_input[i] = sum_o weights[o][i] * delta[o]. */
void elml_kernel_dense_input_grad_f32(const float *weights, const float *delta,
                                      float *grad_input, uint32_t in_dim,
                                      uint32_t out_dim);

/*
 * Scalar reference implementations. Architecture kernels must produce the
 * same results up to normal floating-point reassociation. These symbols are
 * internal to the library and are also used by the dispatcher as the fallback.
 */
void elml_scalar_dense_f32(const float *weights, const float *biases,
                           const float *input, float *output, uint32_t in_dim,
                           uint32_t out_dim);

void elml_scalar_dense_backward_f32(const float *delta, const float *input,
                                    float *grad_w, float *grad_b,
                                    uint32_t in_dim, uint32_t out_dim);

void elml_scalar_dense_input_grad_f32(const float *weights, const float *delta,
                                      float *grad_input, uint32_t in_dim,
                                      uint32_t out_dim);

#endif /* ELML_KERNELS_H */
