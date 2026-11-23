#include "train/threads.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(ELML_HAVE_THREADS) && \
    (defined(__unix__) || defined(__APPLE__))
#define ELML_USE_PTHREADS 1
#include <pthread.h>
#else
#define ELML_USE_PTHREADS 0
#endif

#if ELML_USE_PTHREADS
typedef struct elml_worker_arg {
  struct elml_thread_pool *pool;
  uint32_t index;
} elml_worker_arg;
#endif

struct elml_thread_pool {
#if ELML_USE_PTHREADS
  pthread_t *threads;
  elml_worker_arg *args;
  uint32_t thread_count;
  pthread_mutex_t mutex;
  pthread_cond_t work_ready;
  pthread_cond_t work_done;
  pthread_cond_t ready;
  elml_parallel_fn fn;
  void *arg;
  uint64_t generation;
  uint32_t pending;
  uint32_t ready_count;
  bool shutdown;
#endif
  uint32_t workers;
};

#if ELML_USE_PTHREADS
static void *elml_thread_main(void *opaque)
{
  elml_worker_arg *wa = (elml_worker_arg *)opaque;
  elml_thread_pool *pool = wa->pool;
  const uint32_t index = wa->index;
  uint64_t last = 0u;

  pthread_mutex_lock(&pool->mutex);
  pool->ready_count++;
  pthread_cond_signal(&pool->ready);
  pthread_mutex_unlock(&pool->mutex);

  for (;;) {
    pthread_mutex_lock(&pool->mutex);
    while (!pool->shutdown && pool->generation == last) {
      pthread_cond_wait(&pool->work_ready, &pool->mutex);
    }
    if (pool->shutdown) {
      pthread_mutex_unlock(&pool->mutex);
      break;
    }
    last = pool->generation;
    const elml_parallel_fn fn = pool->fn;
    void *const arg = pool->arg;
    pthread_mutex_unlock(&pool->mutex);

    if (fn != nullptr) {
      fn(arg, index);
    }

    pthread_mutex_lock(&pool->mutex);
    pool->pending--;
    if (pool->pending == 0u) {
      pthread_cond_signal(&pool->work_done);
    }
    pthread_mutex_unlock(&pool->mutex);
  }
  return nullptr;
}
#endif

elml_thread_pool *elml_thread_pool_create(uint32_t workers)
{
#if ELML_USE_PTHREADS
  if (workers <= 1u) {
    return nullptr;
  }

  elml_thread_pool *pool = (elml_thread_pool *)calloc(1u, sizeof *pool);
  if (pool == nullptr) {
    return nullptr;
  }
  pool->workers = workers;
  pool->thread_count = workers - 1u;
  pool->threads = (pthread_t *)calloc((size_t)pool->thread_count,
                                      sizeof *pool->threads);
  pool->args = (elml_worker_arg *)calloc((size_t)pool->thread_count,
                                         sizeof *pool->args);
  if (pool->threads == nullptr || pool->args == nullptr) {
    free(pool->args);
    free(pool->threads);
    free(pool);
    return nullptr;
  }
  if (pthread_mutex_init(&pool->mutex, nullptr) != 0) {
    free(pool->args);
    free(pool->threads);
    free(pool);
    return nullptr;
  }
  if (pthread_cond_init(&pool->work_ready, nullptr) != 0) {
    pthread_mutex_destroy(&pool->mutex);
    free(pool->args);
    free(pool->threads);
    free(pool);
    return nullptr;
  }
  if (pthread_cond_init(&pool->work_done, nullptr) != 0) {
    pthread_cond_destroy(&pool->work_ready);
    pthread_mutex_destroy(&pool->mutex);
    free(pool->args);
    free(pool->threads);
    free(pool);
    return nullptr;
  }
  if (pthread_cond_init(&pool->ready, nullptr) != 0) {
    pthread_cond_destroy(&pool->work_done);
    pthread_cond_destroy(&pool->work_ready);
    pthread_mutex_destroy(&pool->mutex);
    free(pool->args);
    free(pool->threads);
    free(pool);
    return nullptr;
  }

  uint32_t created = 0u;
  for (uint32_t i = 0u; i < pool->thread_count; ++i) {
    pool->args[i].pool = pool;
    pool->args[i].index = i + 1u;
    if (pthread_create(&pool->threads[i], nullptr, elml_thread_main,
                       &pool->args[i]) != 0) {
      break;
    }
    created++;
  }

  if (created != pool->thread_count) {
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
    for (uint32_t i = 0u; i < created; ++i) {
      (void)pthread_join(pool->threads[i], nullptr);
    }
    pthread_cond_destroy(&pool->ready);
    pthread_cond_destroy(&pool->work_done);
    pthread_cond_destroy(&pool->work_ready);
    pthread_mutex_destroy(&pool->mutex);
    free(pool->args);
    free(pool->threads);
    free(pool);
    return nullptr;
  }

  /* Do not return until every worker is parked; otherwise a run() call could
   * advance the generation before a late worker reads its starting value. */
  pthread_mutex_lock(&pool->mutex);
  while (pool->ready_count != pool->thread_count) {
    pthread_cond_wait(&pool->ready, &pool->mutex);
  }
  pthread_mutex_unlock(&pool->mutex);

  return pool;
#else
  (void)workers;
  return nullptr;
#endif
}

void elml_thread_pool_destroy(elml_thread_pool *pool)
{
  if (pool == nullptr) {
    return;
  }
#if ELML_USE_PTHREADS
  pthread_mutex_lock(&pool->mutex);
  pool->shutdown = true;
  pool->generation++;
  pthread_cond_broadcast(&pool->work_ready);
  pthread_mutex_unlock(&pool->mutex);

  for (uint32_t i = 0u; i < pool->thread_count; ++i) {
    (void)pthread_join(pool->threads[i], nullptr);
  }

  pthread_cond_destroy(&pool->ready);
  pthread_cond_destroy(&pool->work_done);
  pthread_cond_destroy(&pool->work_ready);
  pthread_mutex_destroy(&pool->mutex);
  free(pool->args);
  free(pool->threads);
#endif
  free(pool);
}

void elml_thread_pool_run(elml_thread_pool *pool, elml_parallel_fn fn,
                          void *arg)
{
  if (fn == nullptr) {
    return;
  }
#if ELML_USE_PTHREADS
  if (pool != nullptr) {
    pthread_mutex_lock(&pool->mutex);
    pool->fn = fn;
    pool->arg = arg;
    pool->pending = pool->thread_count;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);

    fn(arg, 0u);

    pthread_mutex_lock(&pool->mutex);
    while (pool->pending != 0u) {
      pthread_cond_wait(&pool->work_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
    return;
  }
#else
  (void)pool;
#endif
  fn(arg, 0u);
}

uint32_t elml_thread_pool_workers(const elml_thread_pool *pool)
{
  return (pool != nullptr) ? pool->workers : 1u;
}
