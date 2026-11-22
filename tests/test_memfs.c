#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "test.h"
#include "train/memfs.h"

static bool same_bytes(const uint8_t *data, size_t len, const void *expected,
                       size_t expected_len)
{
  return len == expected_len &&
         (len == 0u || (data != NULL && memcmp(data, expected, len) == 0));
}

static void test_basic(void)
{
  elml_memfs_clear();
  CHECK_EQ_INT(elml_memfs_count(), 0);

  const char first[] = "hello";
  CHECK(elml_memfs_put("notes.txt", first, sizeof first - 1u));

  const uint8_t *data = NULL;
  size_t len = 0u;
  CHECK(elml_memfs_get("notes.txt", &data, &len));
  CHECK(same_bytes(data, len, first, sizeof first - 1u));

  CHECK(elml_memfs_get("notes.txt", NULL, NULL));
  CHECK(elml_memfs_get("missing.txt", &data, &len) == false);

  CHECK_EQ_INT(elml_memfs_count(), 1);
  CHECK_STR(elml_memfs_name(0), "notes.txt");
  CHECK_EQ_INT(elml_memfs_size(0), sizeof first - 1u);
  CHECK(elml_memfs_name(1) == NULL);
  CHECK_EQ_INT(elml_memfs_size(1), 0u);
  CHECK(elml_memfs_name(-1) == NULL);
  CHECK_EQ_INT(elml_memfs_size(-1), 0u);

  CHECK(elml_memfs_remove("notes.txt"));
  CHECK(elml_memfs_remove("notes.txt") == false);
  CHECK_EQ_INT(elml_memfs_count(), 0);

  elml_memfs_clear();
}

static void test_paths(void)
{
  elml_memfs_clear();
  const char payload[] = "abc";
  CHECK(elml_memfs_put("a.csv", payload, sizeof payload - 1u));

  const uint8_t *data = NULL;
  size_t len = 0u;
  CHECK(elml_memfs_get("./a.csv", &data, &len));
  CHECK(same_bytes(data, len, payload, sizeof payload - 1u));
  CHECK(elml_memfs_get("/a.csv", &data, &len));
  CHECK(same_bytes(data, len, payload, sizeof payload - 1u));

  CHECK(elml_memfs_put("dir//b.csv", payload, sizeof payload - 1u));
  CHECK(elml_memfs_get("dir/b.csv", &data, &len));
  CHECK(same_bytes(data, len, payload, sizeof payload - 1u));
  CHECK(elml_memfs_get("dir/./b.csv", &data, &len));
  CHECK(same_bytes(data, len, payload, sizeof payload - 1u));
  CHECK_EQ_INT(elml_memfs_count(), 2);
  elml_memfs_clear();
}

static void test_overwrite_and_empty(void)
{
  elml_memfs_clear();
  const char one[] = "one";
  const char two[] = "second value";
  CHECK(elml_memfs_put("x", one, sizeof one - 1u));
  CHECK(elml_memfs_put("x", two, sizeof two - 1u));

  const uint8_t *data = NULL;
  size_t len = 0u;
  CHECK(elml_memfs_get("x", &data, &len));
  CHECK(same_bytes(data, len, two, sizeof two - 1u));
  CHECK_EQ_INT(elml_memfs_count(), 1);
  CHECK_EQ_INT(elml_memfs_size(0), sizeof two - 1u);

  CHECK(elml_memfs_put("empty", NULL, 0u));
  data = (const uint8_t *)0x1;
  len = 99u;
  CHECK(elml_memfs_get("empty", &data, &len));
  CHECK(data == NULL);
  CHECK_EQ_INT(len, 0u);
  elml_memfs_clear();
}

static void test_invalid(void)
{
  elml_memfs_clear();
  const char byte = 'z';
  CHECK(elml_memfs_put(NULL, &byte, 1u) == false);
  CHECK(elml_memfs_put("", &byte, 1u) == false);
  CHECK(elml_memfs_put("/", &byte, 1u) == false);
  CHECK(elml_memfs_put("./", &byte, 1u) == false);
  CHECK(elml_memfs_put("x", NULL, 1u) == false);
  CHECK_EQ_INT(elml_memfs_count(), 0);
  elml_memfs_clear();
}

static void test_capacity(void)
{
  elml_memfs_clear();
  char name[32];
  const char byte = 'c';
  for (int i = 0; i < ELML_MEMFS_MAX_FILES; ++i) {
    snprintf(name, sizeof name, "file%02d.bin", i);
    CHECK(elml_memfs_put(name, &byte, 1u));
  }
  CHECK_EQ_INT(elml_memfs_count(), ELML_MEMFS_MAX_FILES);
  CHECK(elml_memfs_put("overflow.bin", &byte, 1u) == false);
  CHECK_EQ_INT(elml_memfs_count(), ELML_MEMFS_MAX_FILES);

  CHECK(elml_memfs_remove("file07.bin"));
  CHECK(elml_memfs_put("replacement.bin", &byte, 1u));
  CHECK_EQ_INT(elml_memfs_count(), ELML_MEMFS_MAX_FILES);
  CHECK(elml_memfs_get("replacement.bin", NULL, NULL));
  elml_memfs_clear();
  CHECK_EQ_INT(elml_memfs_count(), 0);
}

int main(void)
{
  test_basic();
  test_paths();
  test_overwrite_and_empty();
  test_invalid();
  test_capacity();
  TEST_END();
}
