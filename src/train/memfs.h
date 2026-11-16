/*
 * Tiny in-memory file table used by the WebAssembly build.
 *
 * The browser has no POSIX file system and CUI's wasm libc returns NULL from
 * fopen. The page uploads a file as bytes, registers it here under a logical
 * path, and every trainer call that takes a path (config, CSV, model, export)
 * resolves against this table instead.
 *
 * Paths are flat: a leading "/" and "./" segments are ignored, and "a//b" is
 * the same key as "a/b". That is enough for the studio, whose files all come
 * from a browser file picker and live side by side.
 */
#ifndef ELML_MEMFS_H
#define ELML_MEMFS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ELML_MEMFS_MAX_FILES 32

void elml_memfs_clear(void);

/* Copy `len` bytes into a new or existing entry. Returns false when the table
 * is full or the allocation fails. */
bool elml_memfs_put(const char *path, const void *data, size_t len);

/* Point at the stored bytes. The pointer stays valid until the entry is
 * replaced or the table is cleared. */
bool elml_memfs_get(const char *path, const uint8_t **data, size_t *len);

bool elml_memfs_remove(const char *path);

int elml_memfs_count(void);
const char *elml_memfs_name(int index);
size_t elml_memfs_size(int index);

#ifdef __cplusplus
}
#endif

#endif /* ELML_MEMFS_H */
