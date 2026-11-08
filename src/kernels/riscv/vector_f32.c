#if defined(__riscv_vector)

#include "kernels/riscv/riscv.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void elml_riscv_dense_f32(const float *weights, const float *biases,
                          const float *input, float *output, uint32_t in_dim,
                          uint32_t out_dim)
{
  const size_t max_vl = __riscv_vsetvlmax_e32m1();

  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    vfloat32m1_t accumulator = __riscv_vfmv_v_f_f32m1(0.0f, max_vl);
    uint32_t i = 0u;

    while (i < in_dim) {
      const size_t vl = __riscv_vsetvl_e32m1((size_t)(in_dim - i));
      const vfloat32m1_t w = __riscv_vle32_v_f32m1(row + i, vl);
      const vfloat32m1_t x = __riscv_vle32_v_f32m1(input + i, vl);
      accumulator = __riscv_vfmacc_vv_f32m1(accumulator, w, x, vl);
      i += (uint32_t)vl;
    }

    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, max_vl);
    const vfloat32m1_t total =
        __riscv_vfredusum_vs_f32m1_f32m1(accumulator, zero, max_vl);
    output[o] = __riscv_vfmv_f_s_f32m1_f32(total) + biases[o];
  }
}

void elml_riscv_dense_backward_f32(const float *delta, const float *input,
                                   float *grad_w, float *grad_b,
                                   uint32_t in_dim, uint32_t out_dim)
{
  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float d = delta[o];
    float *row = grad_w + ((size_t)o * (size_t)in_dim);
    uint32_t i = 0u;

    while (i < in_dim) {
      const size_t vl = __riscv_vsetvl_e32m1((size_t)(in_dim - i));
      const vfloat32m1_t vd = __riscv_vfmv_v_f_f32m1(d, vl);
      const vfloat32m1_t g = __riscv_vle32_v_f32m1(row + i, vl);
      const vfloat32m1_t x = __riscv_vle32_v_f32m1(input + i, vl);
      const vfloat32m1_t updated = __riscv_vfmacc_vv_f32m1(g, vd, x, vl);
      __riscv_vse32_v_f32m1(row + i, updated, vl);
      i += (uint32_t)vl;
    }
    grad_b[o] += d;
  }
}

void elml_riscv_dense_input_grad_f32(const float *weights, const float *delta,
                                     float *grad_input, uint32_t in_dim,
                                     uint32_t out_dim)
{
  uint32_t i = 0u;
  while (i < in_dim) {
    const size_t vl = __riscv_vsetvl_e32m1((size_t)(in_dim - i));
    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    __riscv_vse32_v_f32m1(grad_input + i, zero, vl);
    i += (uint32_t)vl;
  }

  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    i = 0u;

    while (i < in_dim) {
      const size_t vl = __riscv_vsetvl_e32m1((size_t)(in_dim - i));
      const vfloat32m1_t vd = __riscv_vfmv_v_f_f32m1(delta[o], vl);
      const vfloat32m1_t g = __riscv_vle32_v_f32m1(grad_input + i, vl);
      const vfloat32m1_t w = __riscv_vle32_v_f32m1(row + i, vl);
      const vfloat32m1_t updated = __riscv_vfmacc_vv_f32m1(g, vd, w, vl);
      __riscv_vse32_v_f32m1(grad_input + i, updated, vl);
      i += (uint32_t)vl;
    }
  }
}

#endif /* __riscv_vector */
