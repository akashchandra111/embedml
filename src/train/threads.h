/*
 * Minimal worker pool for host training.
 *
 * This is an optimization only: when threading is unavailable (WebAssembly,
 * MCU builds) elml_thread_pool_create() returns NULL and the caller keeps its
 * serial path. The pool is intentionally small: one persistent set of worker
 * threads and a single fork/join per call, so the per-batch cost stays low.
 */
#ifndef ELML_THREADS_H
#define ELML_THREADS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* worker is in [0, workers); index 0 is the calling thread. */
typedef void (*elml_parallel_fn)(void *arg, uint32_t worker);

typedef struct elml_thread_pool elml_thread_pool;

/*
 * Create a pool of `workers` participants including the caller. Returns NULL
 * when `workers` <= 1, when the target has no thread support, or when a
 * worker could not be created. The caller must fall back to serial execution
 * in that case.
 */
elml_thread_pool *elml_thread_pool_create(uint32_t workers);

void elml_thread_pool_destroy(elml_thread_pool *pool);

/*
 * Run fn(arg, 0) on the caller and fn(arg, 1..workers-1) on the pool threads,
 * then wait for all of them. The pool is reusable; calls are serialized.
 */
void elml_thread_pool_run(elml_thread_pool *pool, elml_parallel_fn fn,
                          void *arg);

/* Number of participants, or 1 when `pool` is NULL. */
uint32_t elml_thread_pool_workers(const elml_thread_pool *pool);

#ifdef __cplusplus
}
#endif

#endif /* ELML_THREADS_H */
