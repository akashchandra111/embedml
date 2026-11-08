#include "elml/arena.h"

#include <stdint.h>

#include "ckdint_compat.h"
#include "elml/types.h"

size_t elml_align_up(size_t value, size_t alignment)
{
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return 0u;
  }
  const size_t mask = alignment - 1u;
  if (value > SIZE_MAX - mask) {
    return 0u;
  }
  return (value + mask) & ~mask;
}

void elml_arena_init(elml_arena *arena, void *base, size_t size)
{
  if (arena == nullptr) {
    return;
  }
  arena->base = (uint8_t *)base;
  arena->size = (base == nullptr) ? 0u : size;
  arena->used = 0u;
}

size_t elml_arena_mark(const elml_arena *arena)
{
  return (arena == nullptr) ? 0u : arena->used;
}

void *elml_arena_alloc(elml_arena *arena, size_t bytes, size_t alignment)
{
  if (arena == nullptr || arena->base == nullptr) {
    return nullptr;
  }
  if (alignment < alignof(max_align_t)) {
    alignment = alignof(max_align_t);
  }

  const size_t offset = elml_align_up(arena->used, alignment);
  if (offset == 0u && arena->used != 0u) {
    return nullptr;
  }

  size_t end;
  if (ckd_add(&end, offset, bytes)) {
    return nullptr;
  }
  if (end > arena->size) {
    return nullptr;
  }

  arena->used = end;
  return arena->base + offset;
}

void elml_arena_reset(elml_arena *arena, size_t mark)
{
  if (arena == nullptr) {
    return;
  }
  arena->used = (mark <= arena->size) ? mark : 0u;
}
