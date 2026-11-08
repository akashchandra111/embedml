#include "train/io.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "elml/arena.h"
#include "elml/types.h"

elml_status elml_read_file(const char *path, uint8_t **data, size_t *len,
                           size_t alignment)
{
  if (path == nullptr || data == nullptr || len == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (alignment != 0u && (alignment & (alignment - 1u)) != 0u) {
    return ELML_ERR_INVALID_ARG;
  }
  *data = nullptr;
  *len = 0u;

  FILE *file = fopen(path, "rb");
  if (file == nullptr) {
    return ELML_ERR_NOT_FOUND;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return ELML_ERR_INTERNAL;
  }
  const long length = ftell(file);
  if (length < 0) {
    fclose(file);
    return ELML_ERR_INTERNAL;
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return ELML_ERR_INTERNAL;
  }
  if (length == 0) {
    fclose(file);
    return ELML_ERR_TRUNCATED;
  }

  const size_t size = (size_t)length;
  uint8_t *buffer;
  if (alignment == 0u) {
    buffer = (uint8_t *)malloc(size);
  } else {
    const size_t capacity = elml_align_up(size, alignment);
    if (capacity == 0u) {
      fclose(file);
      return ELML_ERR_OUT_OF_MEMORY;
    }
    buffer = (uint8_t *)aligned_alloc(alignment, capacity);
  }
  if (buffer == nullptr) {
    fclose(file);
    return ELML_ERR_OUT_OF_MEMORY;
  }

  const size_t read = fread(buffer, 1u, size, file);
  fclose(file);
  if (read != size) {
    free(buffer);
    return ELML_ERR_TRUNCATED;
  }

  *data = buffer;
  *len = size;
  return ELML_OK;
}

elml_status elml_write_file(const char *path, const void *data, size_t len)
{
  if (path == nullptr || (data == nullptr && len > 0u)) {
    return ELML_ERR_INVALID_ARG;
  }

  FILE *file = fopen(path, "wb");
  if (file == nullptr) {
    return ELML_ERR_NOT_FOUND;
  }
  size_t written = 0u;
  if (len > 0u) {
    written = fwrite(data, 1u, len, file);
  }
  const int close_status = fclose(file);
  if (written != len || close_status != 0) {
    return ELML_ERR_INTERNAL;
  }
  return ELML_OK;
}
