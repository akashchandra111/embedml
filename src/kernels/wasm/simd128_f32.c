#if defined(__wasm_simd128__)

#include "kernels/wasm/wasm.h"

#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

/*
 * SIMD128 kernels for WebAssembly.
 *
 * Wasm has no fused multiply-add in the base SIMD proposal, so the dense
 * kernels use one multiply and one add per vector. That still evaluates four
 * floats per instruction and is a large win over the scalar reference on every
 * current engine. The scalar tail keeps short layers exact.
 */

static float elml_wasm_hsum(v128_t value)
{
  /* [a0 a1 a2 a3] -> [a0+a2 a1+a3 ...] -> lane 0 holds the total. */
  v128_t sum = wasm_f32x4_add(
      value, wasm_i32x4_shuffle(value, value, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2));
  return wasm_f32x4_extract_lane(sum, 0);
}

static void elml_wasm_dense_row(const float *row, float bias,
                                const float *input, float *output,
                                uint32_t in_dim)
{
  v128_t accumulator = wasm_f32x4_splat(0.0f);
  uint32_t i = 0u;
  for (; in_dim - i >= 4u; i += 4u) {
    const v128_t w = wasm_v128_load(row + i);
    const v128_t x = wasm_v128_load(input + i);
    accumulator = wasm_f32x4_add(accumulator, wasm_f32x4_mul(w, x));
  }
  float sum = elml_wasm_hsum(accumulator) + bias;
  for (; i < in_dim; ++i) {
    sum += row[i] * input[i];
  }
  *output = sum;
}

void elml_wasm_dense_f32(const float *weights, const float *biases,
                         const float *input, float *output, uint32_t in_dim,
                         uint32_t out_dim)
{
  uint32_t o = 0u;

  /* Two rows share each input load, which halves the input traffic. */
  for (; out_dim - o >= 2u; o += 2u) {
    const float *row0 = weights + ((size_t)(o + 0u) * (size_t)in_dim);
    const float *row1 = weights + ((size_t)(o + 1u) * (size_t)in_dim);
    v128_t acc0 = wasm_f32x4_splat(0.0f);
    v128_t acc1 = wasm_f32x4_splat(0.0f);
    uint32_t i = 0u;
    for (; in_dim - i >= 4u; i += 4u) {
      const v128_t x = wasm_v128_load(input + i);
      acc0 = wasm_f32x4_add(
          acc0, wasm_f32x4_mul(wasm_v128_load(row0 + i), x));
      acc1 = wasm_f32x4_add(
          acc1, wasm_f32x4_mul(wasm_v128_load(row1 + i), x));
    }
    float sum0 = elml_wasm_hsum(acc0) + biases[o + 0u];
    float sum1 = elml_wasm_hsum(acc1) + biases[o + 1u];
    for (; i < in_dim; ++i) {
      const float x = input[i];
      sum0 += row0[i] * x;
      sum1 += row1[i] * x;
    }
    output[o + 0u] = sum0;
    output[o + 1u] = sum1;
  }

  for (; o < out_dim; ++o) {
    elml_wasm_dense_row(weights + ((size_t)o * (size_t)in_dim), biases[o],
                        input, output + o, in_dim);
  }
}

void elml_wasm_dense_backward_f32(const float *delta, const float *input,
                                  float *grad_w, float *grad_b, uint32_t in_dim,
                                  uint32_t out_dim)
{
  for (uint32_t o = 0u; o < out_dim; ++o) {
    const float d = delta[o];
    float *row = grad_w + ((size_t)o * (size_t)in_dim);
    const v128_t vd = wasm_f32x4_splat(d);
    uint32_t i = 0u;
    for (; in_dim - i >= 4u; i += 4u) {
      const v128_t g = wasm_v128_load(row + i);
      const v128_t x = wasm_v128_load(input + i);
      wasm_v128_store(row + i,
                      wasm_f32x4_add(g, wasm_f32x4_mul(vd, x)));
    }
    for (; i < in_dim; ++i) {
      row[i] += d * input[i];
    }
    grad_b[o] += d;
  }
}

void elml_wasm_dense_input_grad_f32(const float *weights, const float *delta,
                                    float *grad_input, uint32_t in_dim,
                                    uint32_t out_dim)
{
  uint32_t i = 0u;
  const v128_t zero = wasm_f32x4_splat(0.0f);
  for (; in_dim - i >= 4u; i += 4u) {
    wasm_v128_store(grad_input + i, zero);
  }
  for (; i < in_dim; ++i) {
    grad_input[i] = 0.0f;
  }

  for (uint32_t o = 0u; o < out_dim; ++o) {
    const v128_t vd = wasm_f32x4_splat(delta[o]);
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    i = 0u;
    for (; in_dim - i >= 4u; i += 4u) {
      const v128_t g = wasm_v128_load(grad_input + i);
      const v128_t w = wasm_v128_load(row + i);
      wasm_v128_store(grad_input + i,
                      wasm_f32x4_add(g, wasm_f32x4_mul(w, vd)));
    }
    for (; i < in_dim; ++i) {
      grad_input[i] += row[i] * delta[o];
    }
  }
}

#endif /* __wasm_simd128__ */
