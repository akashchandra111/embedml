#include "train/ini.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ckdint_compat.h"
#include "elml/types.h"
#include "train/io.h"

static bool elml_ini_space(char c)
{
  return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

static char *elml_dup_range(const char *start, const char *end)
{
  const size_t len = (size_t)(end - start);
  char *copy = (char *)malloc(len + 1u);
  if (copy == nullptr) {
    return nullptr;
  }
  memcpy(copy, start, len);
  copy[len] = '\0';
  return copy;
}

static void elml_trim(const char **start, const char **end)
{
  while (*start < *end && elml_ini_space(**start)) {
    (*start)++;
  }
  while (*end > *start && elml_ini_space((*end)[-1])) {
    (*end)--;
  }
}

static elml_status elml_ini_add(elml_ini *ini, const char *section_start,
                                const char *section_end, const char *key_start,
                                const char *key_end, const char *value_start,
                                const char *value_end)
{
  if (ini->count == ini->capacity) {
    size_t next_capacity = (ini->capacity == 0u) ? 16u : ini->capacity * 2u;
    size_t bytes;
    if (ckd_mul(&bytes, next_capacity, sizeof(elml_ini_entry))) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    void *next = realloc(ini->entries, bytes);
    if (next == nullptr) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    ini->entries = (elml_ini_entry *)next;
    ini->capacity = next_capacity;
  }

  elml_ini_entry *entry = &ini->entries[ini->count];
  entry->section = elml_dup_range(section_start, section_end);
  entry->key = elml_dup_range(key_start, key_end);
  entry->value = elml_dup_range(value_start, value_end);
  if (entry->section == nullptr || entry->key == nullptr ||
      entry->value == nullptr) {
    free(entry->section);
    free(entry->key);
    free(entry->value);
    return ELML_ERR_OUT_OF_MEMORY;
  }
  ini->count++;
  return ELML_OK;
}

static bool elml_ini_skip_line(const char *text, size_t len, size_t *pos)
{
  while (*pos < len) {
    size_t look = *pos;
    while (look < len && elml_ini_space(text[look])) {
      look++;
    }
    if (look >= len) {
      *pos = len;
      return false;
    }
    if (text[look] == '\n') {
      *pos = look + 1u;
      continue;
    }
    if (text[look] == '#' || text[look] == ';') {
      while (*pos < len && text[*pos] != '\n') {
        (*pos)++;
      }
      if (*pos < len) {
        (*pos)++;
      }
      continue;
    }
    *pos = look;
    return true;
  }
  return false;
}

elml_status elml_ini_parse(const char *text, size_t len, elml_ini *out)
{
  if (text == nullptr || out == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof *out);

  size_t pos = 0u;
  const char *section_start = text;
  const char *section_end = text;

  while (elml_ini_skip_line(text, len, &pos)) {
    size_t line_end = pos;
    while (line_end < len && text[line_end] != '\n') {
      line_end++;
    }

    const char *start = text + pos;
    const char *end = text + line_end;
    elml_trim(&start, &end);

    if (start < end && *start == '[') {
      const char *close = (const char *)memchr(start, ']', (size_t)(end - start));
      if (close == nullptr || close + 1 != end) {
        return ELML_ERR_INVALID_ARG;
      }
      section_start = start + 1;
      section_end = close;
      elml_trim(&section_start, &section_end);
      if (section_start == section_end) {
        return ELML_ERR_INVALID_ARG;
      }
    } else {
      const char *equals =
          (const char *)memchr(start, '=', (size_t)(end - start));
      if (equals == nullptr) {
        return ELML_ERR_INVALID_ARG;
      }
      const char *key_start = start;
      const char *key_end = equals;
      const char *value_start = equals + 1;
      const char *value_end = end;
      elml_trim(&key_start, &key_end);
      elml_trim(&value_start, &value_end);
      if (key_start == key_end) {
        return ELML_ERR_INVALID_ARG;
      }

      if (value_end > value_start + 1u &&
          ((*value_start == '"' && value_end[-1] == '"') ||
           (*value_start == '\'' && value_end[-1] == '\''))) {
        value_start++;
        value_end--;
      }

      const elml_status status =
          elml_ini_add(out, section_start, section_end, key_start, key_end,
                       value_start, value_end);
      if (status != ELML_OK) {
        return status;
      }
    }

    pos = line_end;
    if (pos < len && text[pos] == '\n') {
      pos++;
    }
  }

  return ELML_OK;
}

elml_status elml_ini_load(const char *path, elml_ini *out)
{
  uint8_t *data = nullptr;
  size_t len = 0u;
  const elml_status status = elml_read_file(path, &data, &len, 0u);
  if (status != ELML_OK) {
    return status;
  }
  const elml_status parse_status =
      elml_ini_parse((const char *)data, len, out);
  free(data);
  return parse_status;
}

void elml_ini_free(elml_ini *ini)
{
  if (ini == nullptr) {
    return;
  }
  for (size_t i = 0u; i < ini->count; ++i) {
    free(ini->entries[i].section);
    free(ini->entries[i].key);
    free(ini->entries[i].value);
  }
  free(ini->entries);
  ini->entries = nullptr;
  ini->count = 0u;
  ini->capacity = 0u;
}

const char *elml_ini_get(const elml_ini *ini, const char *section,
                         const char *key)
{
  if (ini == nullptr || section == nullptr || key == nullptr) {
    return nullptr;
  }
  for (size_t i = ini->count; i > 0u; --i) {
    const elml_ini_entry *entry = &ini->entries[i - 1u];
    if (strcmp(entry->section, section) == 0 && strcmp(entry->key, key) == 0) {
      return entry->value;
    }
  }
  return nullptr;
}
