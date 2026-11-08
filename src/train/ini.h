#ifndef ELML_INI_H
#define ELML_INI_H

#include <stddef.h>

#include "elml/types.h"

typedef struct elml_ini_entry {
  char *section;
  char *key;
  char *value;
} elml_ini_entry;

typedef struct elml_ini {
  elml_ini_entry *entries;
  size_t count;
  size_t capacity;
} elml_ini;

elml_status elml_ini_parse(const char *text, size_t len, elml_ini *out);
elml_status elml_ini_load(const char *path, elml_ini *out);
void elml_ini_free(elml_ini *ini);

/* Returns the last value for section/key, or NULL. Case-sensitive. */
const char *elml_ini_get(const elml_ini *ini, const char *section,
                         const char *key);

#endif /* ELML_INI_H */
