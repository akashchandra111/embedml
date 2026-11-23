#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "elml/registry.h"
#include "elml/types.h"
#include "test.h"
#include "train/net.h"
#include "train/threads.h"

#if defined(ELML_HAVE_THREADS)
typedef struct pool_probe {
  atomic_uint calls;
  atomic_ullong seen;
} pool_probe;

static void pool_task(void *opaque, uint32_t worker)
{
  pool_probe *probe = (pool_probe *)opaque;
  (void)atomic_fetch_add(&probe->calls, 1u);
  (void)atomic_fetch_or(&probe->seen, 1ull << worker);
}
#endif

static void count_task(void *opaque, uint32_t worker)
{
  uint32_t *calls = (uint32_t *)opaque;
  if (worker == 0u) {
    (*calls)++;
  }
}

static void test_pool(void)
{
  uint32_t calls = 0u;
  CHECK_EQ_INT(elml_thread_pool_workers(NULL), 1u);
  CHECK(elml_thread_pool_create(1u) == NULL);
  elml_thread_pool_run(NULL, count_task, &calls);
  CHECK_EQ_INT(calls, 1u);

#if defined(ELML_HAVE_THREADS)
  elml_thread_pool *pool = elml_thread_pool_create(4u);
  CHECK(pool != NULL);
  if (pool == NULL) {
    return;
  }
  CHECK_EQ_INT(elml_thread_pool_workers(pool), 4u);

  pool_probe probe;
  atomic_init(&probe.calls, 0u);
  atomic_init(&probe.seen, 0ull);
  elml_thread_pool_run(pool, pool_task, &probe);
  CHECK_EQ_INT(atomic_load(&probe.calls), 4u);
  CHECK_EQ_INT(atomic_load(&probe.seen), 0xfull);

  /* The pool is reusable and resets the per-run counters. */
  elml_thread_pool_run(pool, pool_task, &probe);
  CHECK_EQ_INT(atomic_load(&probe.calls), 8u);
  CHECK_EQ_INT(atomic_load(&probe.seen), 0xfull);
  elml_thread_pool_destroy(pool);
#endif
}

static void test_worker_reduce(void)
{
  const uint32_t hidden[1] = {4u};
  const elml_activation activations[2] = {ELML_ACT_RELU, ELML_ACT_LINEAR};
  elml_net net;
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, activations, 2u, 3u, 2u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.01f, 0.0f,
                               7u),
               ELML_OK);

  elml_net_worker worker;
  CHECK_EQ_INT(elml_net_worker_init(&worker, &net), ELML_OK);

  const float input[3] = {0.25f, -0.5f, 0.75f};
  const float target[2] = {0.1f, -0.2f};

  elml_net_zero_grad(&net);
  CHECK_EQ_INT(elml_net_accumulate(&net, input, target, 0u), ELML_OK);

  size_t weight_total = 0u;
  size_t bias_total = 0u;
  for (uint32_t l = 0u; l < net.layer_count; ++l) {
    weight_total += (size_t)net.layers[l].in_dim * net.layers[l].out_dim;
    bias_total += (size_t)net.layers[l].out_dim;
  }
  float *base_w = (float *)malloc(weight_total * sizeof(float));
  float *base_b = (float *)malloc(bias_total * sizeof(float));
  CHECK(base_w != NULL && base_b != NULL);
  if (base_w == NULL || base_b == NULL) {
    free(base_w);
    free(base_b);
    elml_net_worker_free(&worker);
    elml_net_free(&net);
    return;
  }

  size_t wi = 0u;
  size_t bi = 0u;
  for (uint32_t l = 0u; l < net.layer_count; ++l) {
    const size_t wc = (size_t)net.layers[l].in_dim * net.layers[l].out_dim;
    const size_t bc = (size_t)net.layers[l].out_dim;
    memcpy(&base_w[wi], net.layers[l].grad_w, wc * sizeof(float));
    memcpy(&base_b[bi], net.layers[l].grad_b, bc * sizeof(float));
    wi += wc;
    bi += bc;
  }

  /* The worker must reproduce the same single-sample gradient, and reducing
   * it must add to (not replace) the caller's accumulation. */
  CHECK_EQ_INT(elml_net_accumulate(&worker.net, input, target, 0u), ELML_OK);
  elml_net_worker_reduce(&worker, &net);

  wi = 0u;
  bi = 0u;
  for (uint32_t l = 0u; l < net.layer_count; ++l) {
    const size_t wc = (size_t)net.layers[l].in_dim * net.layers[l].out_dim;
    const size_t bc = (size_t)net.layers[l].out_dim;
    for (size_t i = 0u; i < wc; ++i) {
      CHECK_CLOSE(net.layers[l].grad_w[i], 2.0 * (double)base_w[wi + i],
                  1e-6);
    }
    for (size_t i = 0u; i < bc; ++i) {
      CHECK_CLOSE(net.layers[l].grad_b[i], 2.0 * (double)base_b[bi + i],
                  1e-6);
    }
    wi += wc;
    bi += bc;
  }

  free(base_w);
  free(base_b);
  elml_net_worker_free(&worker);
  elml_net_free(&net);
}

int main(void)
{
  test_pool();
  test_worker_reduce();
  TEST_END();
}
