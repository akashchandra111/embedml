#ifndef ELML_INSPECT_H
#define ELML_INSPECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "format/format_internal.h"

void elml_inspect_print(const elml_format *format, const uint8_t *blob,
                        FILE *out, bool json);

#endif /* ELML_INSPECT_H */
