#ifndef ELML_KERNELS_WASM_H
#define ELML_KERNELS_WASM_H

#include <stdint.h>

/*
 * WebAssembly SIMD128 kernels. Compiled when the target has the simd128
 * feature enabled (`-msimd128`, macro __wasm_simd128__). There is no runtime
 * probe: a module that contains these instructions either validates in the
 * engine or it does not, so the build decides once.
 */
void elml_wasm_dense_f32(const float *weights, const float *biases,
                         const float *input, float *output, uint32_t in_dim,
                         uint32_t out_dim);

void elml_wasm_dense_backward_f32(const float *delta, const float *input,
                                  float *grad_w, float *grad_b, uint32_t in_dim,
                                  uint32_t out_dim);

void elml_wasm_dense_input_grad_f32(const float *weights, const float *delta,
                                    float *grad_input, uint32_t in_dim,
                                    uint32_t out_dim);

#endif /* ELML_KERNELS_WASM_H */
