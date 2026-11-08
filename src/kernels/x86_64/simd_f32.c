#if defined(__x86_64__) || defined(_M_X64)

#include "kernels/x86_64/x86_64.h"

#include <immintrin.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ELML_X86_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define ELML_X86_TARGET_AVX2
#endif

/*
 * Cached feature probe. The check runs once; subsequent dispatch decisions are
 * a relaxed atomic load, which compiles to a plain load on x86.
 */
static atomic_int g_x86_simd_state = -1;

bool elml_x86_simd_available(void)
{
  int state = atomic_load_explicit(&g_x86_simd_state, memory_order_relaxed);
  if (state < 0) {
#if defined(__GNUC__) || defined(__clang__)
    state = (__builtin_cpu_supports("avx2") != 0 &&
             __builtin_cpu_supports("fma") != 0)
                ? 1
                : 0;
#else
    state = 0;
#endif
    atomic_store_explicit(&g_x86_simd_state, state, memory_order_relaxed);
  }
  return state != 0;
}

ELML_X86_TARGET_AVX2
static float elml_x86_hsum256(__m256 value)
{
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
  return _mm_cvtss_f32(sum);
}

ELML_X86_TARGET_AVX2
static void elml_x86_dense_row(const float *row, float bias,
                               const float *input, float *output,
                               uint32_t in_dim)
{
  __m256 accumulator = _mm256_setzero_ps();
  uint32_t i = 0u;
  for (; in_dim - i >= 8u; i += 8u) {
    accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                  _mm256_loadu_ps(input + i), accumulator);
  }
  float sum = elml_x86_hsum256(accumulator) + bias;
  for (; i < in_dim; ++i) {
    sum += row[i] * input[i];
  }
  *output = sum;
}

ELML_X86_TARGET_AVX2
void elml_x86_dense_f32(const float *weights, const float *biases,
                        const float *input, float *output, uint32_t in_dim,
                        uint32_t out_dim)
{
  uint32_t o = 0u;

  /* Four rows share each input load, which is the main bandwidth win. */
  for (; out_dim - o >= 4u; o += 4u) {
    const float *r0 = weights + ((size_t)(o + 0u) * (size_t)in_dim);
    const float *r1 = weights + ((size_t)(o + 1u) * (size_t)in_dim);
    const float *r2 = weights + ((size_t)(o + 2u) * (size_t)in_dim);
    const float *r3 = weights + ((size_t)(o + 3u) * (size_t)in_dim);
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    uint32_t i = 0u;

    for (; in_dim - i >= 8u; i += 8u) {
      const __m256 x = _mm256_loadu_ps(input + i);
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), x, a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), x, a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), x, a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), x, a3);
    }

    float s0 = elml_x86_hsum256(a0) + biases[o + 0u];
    float s1 = elml_x86_hsum256(a1) + biases[o + 1u];
    float s2 = elml_x86_hsum256(a2) + biases[o + 2u];
    float s3 = elml_x86_hsum256(a3) + biases[o + 3u];
    for (; i < in_dim; ++i) {
      const float x = input[i];
      s0 += r0[i] * x;
      s1 += r1[i] * x;
      s2 += r2[i] * x;
      s3 += r3[i] * x;
    }
    output[o + 0u] = s0;
    output[o + 1u] = s1;
    output[o + 2u] = s2;
    output[o + 3u] = s3;
  }

  for (; o < out_dim; ++o) {
    elml_x86_dense_row(weights + ((size_t)o * (size_t)in_dim), biases[o],
                       input, output + o, in_dim);
  }
}

ELML_X86_TARGET_AVX2
static void elml_x86_backward_row(const float *input, float d, float *row,
                                  float *grad_b, uint32_t in_dim)
{
  const __m256 vd = _mm256_set1_ps(d);
  uint32_t i = 0u;
  for (; in_dim - i >= 8u; i += 8u) {
    const __m256 g = _mm256_loadu_ps(row + i);
    const __m256 x = _mm256_loadu_ps(input + i);
    _mm256_storeu_ps(row + i, _mm256_fmadd_ps(vd, x, g));
  }
  for (; i < in_dim; ++i) {
    row[i] += d * input[i];
  }
  *grad_b += d;
}

ELML_X86_TARGET_AVX2
void elml_x86_dense_backward_f32(const float *delta, const float *input,
                                 float *grad_w, float *grad_b, uint32_t in_dim,
                                 uint32_t out_dim)
{
  uint32_t o = 0u;

  for (; out_dim - o >= 4u; o += 4u) {
    float *g0 = grad_w + ((size_t)(o + 0u) * (size_t)in_dim);
    float *g1 = grad_w + ((size_t)(o + 1u) * (size_t)in_dim);
    float *g2 = grad_w + ((size_t)(o + 2u) * (size_t)in_dim);
    float *g3 = grad_w + ((size_t)(o + 3u) * (size_t)in_dim);
    const __m256 d0 = _mm256_set1_ps(delta[o + 0u]);
    const __m256 d1 = _mm256_set1_ps(delta[o + 1u]);
    const __m256 d2 = _mm256_set1_ps(delta[o + 2u]);
    const __m256 d3 = _mm256_set1_ps(delta[o + 3u]);
    uint32_t i = 0u;

    for (; in_dim - i >= 8u; i += 8u) {
      const __m256 x = _mm256_loadu_ps(input + i);
      _mm256_storeu_ps(g0 + i,
                       _mm256_fmadd_ps(d0, x, _mm256_loadu_ps(g0 + i)));
      _mm256_storeu_ps(g1 + i,
                       _mm256_fmadd_ps(d1, x, _mm256_loadu_ps(g1 + i)));
      _mm256_storeu_ps(g2 + i,
                       _mm256_fmadd_ps(d2, x, _mm256_loadu_ps(g2 + i)));
      _mm256_storeu_ps(g3 + i,
                       _mm256_fmadd_ps(d3, x, _mm256_loadu_ps(g3 + i)));
    }

    for (; i < in_dim; ++i) {
      const float x = input[i];
      g0[i] += delta[o + 0u] * x;
      g1[i] += delta[o + 1u] * x;
      g2[i] += delta[o + 2u] * x;
      g3[i] += delta[o + 3u] * x;
    }
    grad_b[o + 0u] += delta[o + 0u];
    grad_b[o + 1u] += delta[o + 1u];
    grad_b[o + 2u] += delta[o + 2u];
    grad_b[o + 3u] += delta[o + 3u];
  }

  for (; o < out_dim; ++o) {
    elml_x86_backward_row(input, delta[o],
                          grad_w + ((size_t)o * (size_t)in_dim),
                          grad_b + o, in_dim);
  }
}

ELML_X86_TARGET_AVX2
void elml_x86_dense_input_grad_f32(const float *weights, const float *delta,
                                   float *grad_input, uint32_t in_dim,
                                   uint32_t out_dim)
{
  const __m256 zero = _mm256_setzero_ps();
  uint32_t i = 0u;
  for (; in_dim - i >= 8u; i += 8u) {
    _mm256_storeu_ps(grad_input + i, zero);
  }
  for (; i < in_dim; ++i) {
    grad_input[i] = 0.0f;
  }

  uint32_t o = 0u;
  for (; out_dim - o >= 4u; o += 4u) {
    const float *r0 = weights + ((size_t)(o + 0u) * (size_t)in_dim);
    const float *r1 = weights + ((size_t)(o + 1u) * (size_t)in_dim);
    const float *r2 = weights + ((size_t)(o + 2u) * (size_t)in_dim);
    const float *r3 = weights + ((size_t)(o + 3u) * (size_t)in_dim);
    const __m256 d0 = _mm256_set1_ps(delta[o + 0u]);
    const __m256 d1 = _mm256_set1_ps(delta[o + 1u]);
    const __m256 d2 = _mm256_set1_ps(delta[o + 2u]);
    const __m256 d3 = _mm256_set1_ps(delta[o + 3u]);
    i = 0u;

    for (; in_dim - i >= 8u; i += 8u) {
      __m256 g = _mm256_loadu_ps(grad_input + i);
      g = _mm256_fmadd_ps(d0, _mm256_loadu_ps(r0 + i), g);
      g = _mm256_fmadd_ps(d1, _mm256_loadu_ps(r1 + i), g);
      g = _mm256_fmadd_ps(d2, _mm256_loadu_ps(r2 + i), g);
      g = _mm256_fmadd_ps(d3, _mm256_loadu_ps(r3 + i), g);
      _mm256_storeu_ps(grad_input + i, g);
    }

    for (; i < in_dim; ++i) {
      grad_input[i] += r0[i] * delta[o + 0u] + r1[i] * delta[o + 1u] +
                       r2[i] * delta[o + 2u] + r3[i] * delta[o + 3u];
    }
  }

  for (; o < out_dim; ++o) {
    const float *row = weights + ((size_t)o * (size_t)in_dim);
    const __m256 vd = _mm256_set1_ps(delta[o]);
    i = 0u;
    for (; in_dim - i >= 8u; i += 8u) {
      const __m256 g = _mm256_loadu_ps(grad_input + i);
      const __m256 w = _mm256_loadu_ps(row + i);
      _mm256_storeu_ps(grad_input + i, _mm256_fmadd_ps(vd, w, g));
    }
    for (; i < in_dim; ++i) {
      grad_input[i] += row[i] * delta[o];
    }
  }
}

#endif /* x86_64 */
