#ifndef ELML_KERNELS_X86_64_H
#define ELML_KERNELS_X86_64_H

#include <stdbool.h>
#include <stdint.h>

/* True when the CPU supports every instruction used by the AVX2 kernels. */
bool elml_x86_simd_available(void);

void elml_x86_dense_f32(const float *weights, const float *biases,
                        const float *input, float *output, uint32_t in_dim,
                        uint32_t out_dim);

void elml_x86_dense_backward_f32(const float *delta, const float *input,
                                 float *grad_w, float *grad_b, uint32_t in_dim,
                                 uint32_t out_dim);

void elml_x86_dense_input_grad_f32(const float *weights, const float *delta,
                                   float *grad_input, uint32_t in_dim,
                                   uint32_t out_dim);

#endif /* ELML_KERNELS_X86_64_H */
