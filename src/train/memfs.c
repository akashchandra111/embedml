#include "train/memfs.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ELML_MEMFS_PATH_MAX 192

typedef struct elml_memfs_entry {
  char name[ELML_MEMFS_PATH_MAX];
  uint8_t *data;
  size_t len;
  bool used;
} elml_memfs_entry;

static elml_memfs_entry elml_memfs_entries[ELML_MEMFS_MAX_FILES];

/* Collapse "//", drop a leading "/" and any "./" segments. The result is the
 * key used for both puts and gets, so the studio can hand us "iris.csv",
 * "./iris.csv" or "/iris.csv" and always hit the same entry. */
static void elml_memfs_normalize(const char *path, char *out, size_t cap)
{
  if (out == nullptr || cap == 0u) {
    return;
  }
  out[0] = '\0';
  if (path == nullptr) {
    return;
  }

  size_t used = 0u;
  size_t i = 0u;
  while (path[i] == '/') {
    ++i;
  }
  while (path[i] != '\0') {
    if (path[i] == '.' && path[i + 1u] == '/') {
      i += 2u;
      continue;
    }
    if (path[i] == '/') {
      /* Avoid a leading or doubled separator. */
      if (used > 0u && used + 1u < cap && out[used - 1u] != '/') {
        out[used++] = '/';
      }
      ++i;
      while (path[i] == '/') {
        ++i;
      }
      continue;
    }
    if (used + 1u < cap) {
      out[used++] = path[i];
    }
    ++i;
  }
  if (used > 0u && out[used - 1u] == '/') {
    --used;
  }
  out[used] = '\0';
}

static int elml_memfs_find(const char *name)
{
  for (int i = 0; i < ELML_MEMFS_MAX_FILES; ++i) {
    if (elml_memfs_entries[i].used &&
        strcmp(elml_memfs_entries[i].name, name) == 0) {
      return i;
    }
  }
  return -1;
}

void elml_memfs_clear(void)
{
  for (int i = 0; i < ELML_MEMFS_MAX_FILES; ++i) {
    free(elml_memfs_entries[i].data);
    memset(&elml_memfs_entries[i], 0, sizeof elml_memfs_entries[i]);
  }
}

bool elml_memfs_put(const char *path, const void *data, size_t len)
{
  char name[ELML_MEMFS_PATH_MAX];
  elml_memfs_normalize(path, name, sizeof name);
  if (name[0] == '\0' || (data == nullptr && len > 0u)) {
    return false;
  }

  int index = elml_memfs_find(name);
  if (index < 0) {
    for (int i = 0; i < ELML_MEMFS_MAX_FILES; ++i) {
      if (!elml_memfs_entries[i].used) {
        index = i;
        break;
      }
    }
  }
  if (index < 0) {
    return false;
  }

  uint8_t *copy = nullptr;
  if (len > 0u) {
    copy = (uint8_t *)malloc(len);
    if (copy == nullptr) {
      return false;
    }
    memcpy(copy, data, len);
  }

  elml_memfs_entry *entry = &elml_memfs_entries[index];
  free(entry->data);
  memcpy(entry->name, name, strlen(name) + 1u);
  entry->data = copy;
  entry->len = len;
  entry->used = true;
  return true;
}

bool elml_memfs_get(const char *path, const uint8_t **data, size_t *len)
{
  char name[ELML_MEMFS_PATH_MAX];
  elml_memfs_normalize(path, name, sizeof name);
  const int index = elml_memfs_find(name);
  if (index < 0) {
    return false;
  }
  if (data != nullptr) {
    *data = elml_memfs_entries[index].data;
  }
  if (len != nullptr) {
    *len = elml_memfs_entries[index].len;
  }
  return true;
}

bool elml_memfs_remove(const char *path)
{
  char name[ELML_MEMFS_PATH_MAX];
  elml_memfs_normalize(path, name, sizeof name);
  const int index = elml_memfs_find(name);
  if (index < 0) {
    return false;
  }
  free(elml_memfs_entries[index].data);
  memset(&elml_memfs_entries[index], 0, sizeof elml_memfs_entries[index]);
  return true;
}

int elml_memfs_count(void)
{
  int count = 0;
  for (int i = 0; i < ELML_MEMFS_MAX_FILES; ++i) {
    if (elml_memfs_entries[i].used) {
      ++count;
    }
  }
  return count;
}

const char *elml_memfs_name(int index)
{
  int seen = 0;
  for (int i = 0; i < ELML_MEMFS_MAX_FILES; ++i) {
    if (!elml_memfs_entries[i].used) {
      continue;
    }
    if (seen == index) {
      return elml_memfs_entries[i].name;
    }
    ++seen;
  }
  return nullptr;
}

size_t elml_memfs_size(int index)
{
  int seen = 0;
  for (int i = 0; i < ELML_MEMFS_MAX_FILES; ++i) {
    if (!elml_memfs_entries[i].used) {
      continue;
    }
    if (seen == index) {
      return elml_memfs_entries[i].len;
    }
    ++seen;
  }
  return 0u;
}
