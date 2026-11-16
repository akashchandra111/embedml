#include "train/io.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/arena.h"
#include "elml/types.h"
#include "train/memfs.h"

void *elml_aligned_alloc(size_t alignment, size_t size)
{
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return nullptr;
  }
  const size_t header = sizeof(void *);
  if (size > SIZE_MAX - header - alignment) {
    return nullptr;
  }
  void *raw = malloc(size + header + alignment);
  if (raw == nullptr) {
    return nullptr;
  }
  const uintptr_t base = (uintptr_t)raw + (uintptr_t)header;
  const uintptr_t aligned =
      (base + (uintptr_t)alignment - 1u) & ~((uintptr_t)alignment - 1u);
  void *result = (void *)aligned;
  ((void **)result)[-1] = raw;
  return result;
}

void elml_aligned_free(void *ptr)
{
  if (ptr == nullptr) {
    return;
  }
  free(((void **)ptr)[-1]);
}

#if defined(ELML_WASM)

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

  const uint8_t *source = nullptr;
  size_t size = 0u;
  if (!elml_memfs_get(path, &source, &size)) {
    return ELML_ERR_NOT_FOUND;
  }
  if (size == 0u) {
    return ELML_ERR_TRUNCATED;
  }

  uint8_t *buffer;
  if (alignment == 0u) {
    buffer = (uint8_t *)malloc(size);
  } else {
    const size_t capacity = elml_align_up(size, alignment);
    if (capacity == 0u) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    buffer = (uint8_t *)elml_aligned_alloc(alignment, capacity);
  }
  if (buffer == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  memcpy(buffer, source, size);

  *data = buffer;
  *len = size;
  return ELML_OK;
}

elml_status elml_write_file(const char *path, const void *data, size_t len)
{
  if (path == nullptr || (data == nullptr && len > 0u)) {
    return ELML_ERR_INVALID_ARG;
  }
  if (!elml_memfs_put(path, data, len)) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  return ELML_OK;
}

#else /* host: ordinary stdio */

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
    buffer = (uint8_t *)elml_aligned_alloc(alignment, capacity);
  }
  if (buffer == nullptr) {
    fclose(file);
    return ELML_ERR_OUT_OF_MEMORY;
  }

  const size_t read = fread(buffer, 1u, size, file);
  fclose(file);
  if (read != size) {
    if (alignment == 0u) {
      free(buffer);
    } else {
      elml_aligned_free(buffer);
    }
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

#endif /* ELML_WASM */
