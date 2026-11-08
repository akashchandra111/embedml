#ifndef ELML_IO_H
#define ELML_IO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/types.h"

elml_status elml_read_file(const char *path, uint8_t **data, size_t *len,
                           size_t alignment);

elml_status elml_write_file(const char *path, const void *data, size_t len);

#endif /* ELML_IO_H */
