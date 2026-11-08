#define _POSIX_C_SOURCE 200809L

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "elml/arena.h"
#include "elml/types.h"
#include "test.h"
#include "train/io.h"

int main(void)
{
  char path[] = "elml_io_XXXXXX";
  const int fd = mkstemp(path);
  CHECK(fd >= 0);
  if (fd >= 0) {
    close(fd);
    const uint8_t data[5] = {1u, 2u, 3u, 4u, 5u};
    CHECK_EQ_INT(elml_write_file(path, data, sizeof data), ELML_OK);

    uint8_t *read = NULL;
    size_t len = 0u;
    CHECK_EQ_INT(elml_read_file(path, &read, &len, 64u), ELML_OK);
    CHECK_EQ_INT(len, 5u);
    CHECK(read != NULL);
    if (read != NULL) {
      CHECK_EQ_INT((uintptr_t)read % 64u, 0u);
      CHECK(memcmp(read, data, sizeof data) == 0);
    }
    free(read);

    CHECK_EQ_INT(elml_read_file(path, &read, &len, 0u), ELML_OK);
    CHECK(read != NULL);
    free(read);

    CHECK_EQ_INT(elml_write_file(path, data, 0u), ELML_OK);
    CHECK_EQ_INT(elml_read_file(path, &read, &len, 0u), ELML_ERR_TRUNCATED);
    remove(path);
  }

  uint8_t *read = NULL;
  size_t len = 0u;
  CHECK_EQ_INT(elml_read_file("missing.bin", &read, &len, 0u),
               ELML_ERR_NOT_FOUND);
  CHECK_EQ_INT(elml_read_file(path, &read, &len, 3u), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_read_file(NULL, &read, &len, 0u), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_read_file("missing.bin", NULL, &len, 0u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_read_file("missing.bin", &read, NULL, 0u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_write_file(NULL, NULL, 0u), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_write_file("missing.bin", NULL, 1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_write_file("no-such-dir/file.bin", "x", 1u),
               ELML_ERR_NOT_FOUND);

  TEST_END();
}
