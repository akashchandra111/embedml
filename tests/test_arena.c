#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/arena.h"
#include "test.h"

int main(void)
{
  CHECK_EQ_INT(elml_align_up(0u, 64u), 0u);
  CHECK_EQ_INT(elml_align_up(1u, 64u), 64u);
  CHECK_EQ_INT(elml_align_up(64u, 64u), 64u);
  CHECK_EQ_INT(elml_align_up(65u, 64u), 128u);
  CHECK_EQ_INT(elml_align_up(8u, 1u), 8u);
  CHECK_EQ_INT(elml_align_up(8u, 3u), 0u);
  CHECK_EQ_INT(elml_align_up(8u, 0u), 0u);

  elml_arena null_arena;
  elml_arena_init(NULL, (void *)1, 10u);
  CHECK_EQ_INT(elml_arena_mark(NULL), 0u);
  elml_arena_init(&null_arena, NULL, 100u);
  CHECK_EQ_INT(null_arena.size, 0u);
  CHECK(elml_arena_alloc(&null_arena, 1u, 8u) == NULL);
  elml_arena_reset(NULL, 0u);

  alignas(64) uint8_t storage[256];
  elml_arena arena;
  elml_arena_init(&arena, storage, sizeof storage);
  CHECK_EQ_INT(elml_arena_mark(&arena), 0u);

  void *first = elml_arena_alloc(&arena, 3u, 1u);
  CHECK(first != NULL);
  CHECK_EQ_INT((uintptr_t)first % alignof(max_align_t), 0u);
  const size_t mark = elml_arena_mark(&arena);
  CHECK(mark >= 3u);

  void *second = elml_arena_alloc(&arena, 16u, 32u);
  CHECK(second != NULL);
  if (second != NULL) {
    CHECK_EQ_INT((uintptr_t)second % 32u, 0u);
    CHECK((uintptr_t)second > (uintptr_t)first);
  }

  elml_arena_reset(&arena, mark);
  CHECK_EQ_INT(elml_arena_mark(&arena), mark);
  void *reused = elml_arena_alloc(&arena, 1u, 1u);
  CHECK(reused == (void *)(storage + elml_align_up(mark, alignof(max_align_t))));

  elml_arena_reset(&arena, 0u);
  CHECK_EQ_INT(elml_arena_mark(&arena), 0u);

  void *too_big = elml_arena_alloc(&arena, sizeof storage + 1u, 1u);
  CHECK(too_big == NULL);
  CHECK_EQ_INT(elml_arena_mark(&arena), 0u);

  elml_arena_reset(&arena, sizeof storage + 100u);
  CHECK_EQ_INT(elml_arena_mark(&arena), 0u);

  void *zero = elml_arena_alloc(&arena, 0u, 1u);
  CHECK(zero == storage);

  TEST_END();
}
