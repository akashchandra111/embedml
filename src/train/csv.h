#ifndef ELML_CSV_H
#define ELML_CSV_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "elml/types.h"

typedef struct elml_csv {
  char *storage;
  char **cells;
  uint32_t rows;
  uint32_t cols;
  uint32_t data_start; /* 0 without a header, 1 with one */
} elml_csv;

elml_status elml_csv_parse(const char *text, size_t len, char delimiter,
                           bool has_header, elml_csv *out);
elml_status elml_csv_load(const char *path, char delimiter, bool has_header,
                          elml_csv *out);
/* Read only the first CSV record. The result has rows == 1 and data_start ==
 * 0, so elml_csv_cell(&header, 0u, col) gives the header name (or the first
 * data row for headerless files). BOMs, blank lines and # comments are
 * skipped. This is used by the studio to list selectable columns without
 * loading the whole file. */
elml_status elml_csv_read_header(const char *path, char delimiter,
                                 elml_csv *out);
void elml_csv_free(elml_csv *csv);

const char *elml_csv_cell(const elml_csv *csv, uint32_t row, uint32_t col);

#endif /* ELML_CSV_H */
