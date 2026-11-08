#ifndef ELML_KERNELS_ARM64_H
#define ELML_KERNELS_ARM64_H

#include <stdint.h>

void elml_arm64_dense_f32(const float *weights, const float *biases,
                          const float *input, float *output, uint32_t in_dim,
                          uint32_t out_dim);

void elml_arm64_dense_backward_f32(const float *delta, const float *input,
                                   float *grad_w, float *grad_b,
                                   uint32_t in_dim, uint32_t out_dim);

void elml_arm64_dense_input_grad_f32(const float *weights, const float *delta,
                                     float *grad_input, uint32_t in_dim,
                                     uint32_t out_dim);

#endif /* ELML_KERNELS_ARM64_H */
