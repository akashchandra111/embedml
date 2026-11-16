#ifndef ELML_IO_H
#define ELML_IO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/types.h"

/*
 * Aligned temporary storage.
 *
 * The inference API requires 64-byte model blobs and 16-byte scratch buffers.
 * Both the host tools and the wasm build use these helpers so the trainer does
 * not depend on C11 aligned_alloc (missing from the browser libc and from some
 * MCU toolchains). Free the returned pointer with elml_aligned_free().
 */
void *elml_aligned_alloc(size_t alignment, size_t size);
void elml_aligned_free(void *ptr);

/*
 * On the host these read and write real files. In the wasm build they resolve
 * against train/memfs.h instead, which is filled by the page's file picker.
 * `alignment` must be 0 or a power of two.
 */
elml_status elml_read_file(const char *path, uint8_t **data, size_t *len,
                           size_t alignment);

elml_status elml_write_file(const char *path, const void *data, size_t len);

#endif /* ELML_IO_H */
