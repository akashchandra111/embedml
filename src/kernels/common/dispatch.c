#include "kernels/kernels.h"

#include <stdint.h>

#if defined(ELML_ARCH_X86_64)
#include "kernels/x86_64/x86_64.h"
#elif defined(ELML_ARCH_ARM64)
#include "kernels/arm64/arm64.h"
#elif defined(ELML_ARCH_RISCV)
#include "kernels/riscv/riscv.h"
#endif

void elml_kernel_dense_f32(const float *weights, const float *biases,
                           const float *input, float *output, uint32_t in_dim,
                           uint32_t out_dim)
{
#if defined(ELML_ARCH_X86_64) && !defined(ELML_STRICT_FP)
  if (elml_x86_simd_available()) {
    elml_x86_dense_f32(weights, biases, input, output, in_dim, out_dim);
    return;
  }
#elif defined(ELML_ARCH_ARM64) && defined(__ARM_NEON) && !defined(ELML_STRICT_FP)
  elml_arm64_dense_f32(weights, biases, input, output, in_dim, out_dim);
  return;
#elif defined(ELML_ARCH_RISCV) && defined(__riscv_vector) && !defined(ELML_STRICT_FP)
  elml_riscv_dense_f32(weights, biases, input, output, in_dim, out_dim);
  return;
#endif
  elml_scalar_dense_f32(weights, biases, input, output, in_dim, out_dim);
}

void elml_kernel_dense_backward_f32(const float *delta, const float *input,
                                    float *grad_w, float *grad_b,
                                    uint32_t in_dim, uint32_t out_dim)
{
#if defined(ELML_ARCH_X86_64) && !defined(ELML_STRICT_FP)
  if (elml_x86_simd_available()) {
    elml_x86_dense_backward_f32(delta, input, grad_w, grad_b, in_dim, out_dim);
    return;
  }
#elif defined(ELML_ARCH_ARM64) && defined(__ARM_NEON) && !defined(ELML_STRICT_FP)
  elml_arm64_dense_backward_f32(delta, input, grad_w, grad_b, in_dim, out_dim);
  return;
#elif defined(ELML_ARCH_RISCV) && defined(__riscv_vector) && !defined(ELML_STRICT_FP)
  elml_riscv_dense_backward_f32(delta, input, grad_w, grad_b, in_dim, out_dim);
  return;
#endif
  elml_scalar_dense_backward_f32(delta, input, grad_w, grad_b, in_dim, out_dim);
}

void elml_kernel_dense_input_grad_f32(const float *weights, const float *delta,
                                      float *grad_input, uint32_t in_dim,
                                      uint32_t out_dim)
{
#if defined(ELML_ARCH_X86_64) && !defined(ELML_STRICT_FP)
  if (elml_x86_simd_available()) {
    elml_x86_dense_input_grad_f32(weights, delta, grad_input, in_dim, out_dim);
    return;
  }
#elif defined(ELML_ARCH_ARM64) && defined(__ARM_NEON) && !defined(ELML_STRICT_FP)
  elml_arm64_dense_input_grad_f32(weights, delta, grad_input, in_dim, out_dim);
  return;
#elif defined(ELML_ARCH_RISCV) && defined(__riscv_vector) && !defined(ELML_STRICT_FP)
  elml_riscv_dense_input_grad_f32(weights, delta, grad_input, in_dim, out_dim);
  return;
#endif
  elml_scalar_dense_input_grad_f32(weights, delta, grad_input, in_dim, out_dim);
}
