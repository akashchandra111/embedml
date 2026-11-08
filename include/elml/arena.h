/*
 * Caller-owned monotonic arena.
 *
 * The inference library never calls malloc. Every temporary lives in a buffer
 * supplied by the application (stack, static storage, or a linker section).
 */
#ifndef ELML_ARENA_H
#define ELML_ARENA_H

#include <stddef.h>
#include <stdint.h>

#include "elml/compat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct elml_arena {
  uint8_t *base;
  size_t size;
  size_t used;
} elml_arena;

void elml_arena_init(elml_arena *arena, void *base, size_t size);
size_t elml_arena_mark(const elml_arena *arena);
void *elml_arena_alloc(elml_arena *arena, size_t bytes, size_t alignment);
void elml_arena_reset(elml_arena *arena, size_t mark);
size_t elml_align_up(size_t value, size_t alignment);

#ifdef __cplusplus
}
#endif

#endif /* ELML_ARENA_H */
