#include "train/csv.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckdint_compat.h"
#include "elml/types.h"
#include "train/io.h"

static bool elml_is_space(char c)
{
  return c == ' ' || c == '\t' || c == '\v' || c == '\f';
}

typedef struct elml_csv_builder {
  uint8_t *offsets; /* size_t entries, stored as bytes for realloc ergonomics */
  size_t offset_count;
  size_t offset_cap;
  uint32_t *row_cols;
  size_t row_count;
  size_t row_cap;
  char *storage;
  size_t storage_len;
  size_t storage_cap;
} elml_csv_builder;

static elml_status elml_builder_add_offset(elml_csv_builder *builder,
                                           size_t offset)
{
  if (builder->offset_count == builder->offset_cap) {
    size_t next_cap = (builder->offset_cap == 0u) ? 64u : builder->offset_cap * 2u;
    size_t bytes;
    if (ckd_mul(&bytes, next_cap, sizeof(size_t))) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    void *next = realloc(builder->offsets, bytes);
    if (next == nullptr) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    builder->offsets = (uint8_t *)next;
    builder->offset_cap = next_cap;
  }
  size_t *offsets = (size_t *)(void *)builder->offsets;
  offsets[builder->offset_count] = offset;
  builder->offset_count++;
  return ELML_OK;
}

static elml_status elml_builder_add_row(elml_csv_builder *builder,
                                        uint32_t cols)
{
  if (builder->row_count == builder->row_cap) {
    size_t next_cap = (builder->row_cap == 0u) ? 32u : builder->row_cap * 2u;
    size_t bytes;
    if (ckd_mul(&bytes, next_cap, sizeof(uint32_t))) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    void *next = realloc(builder->row_cols, bytes);
    if (next == nullptr) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    builder->row_cols = (uint32_t *)next;
    builder->row_cap = next_cap;
  }
  builder->row_cols[builder->row_count] = cols;
  builder->row_count++;
  return ELML_OK;
}

static elml_status elml_builder_put(elml_csv_builder *builder, char c)
{
  if (builder->storage_len == builder->storage_cap) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  builder->storage[builder->storage_len++] = c;
  return ELML_OK;
}

static elml_status elml_parse_quoted(const char *text, size_t len, size_t *pos,
                                     char delimiter, elml_csv_builder *builder,
                                     size_t *field)
{
  *field = builder->storage_len;
  (*pos)++; /* opening quote */

  while (*pos < len) {
    const char c = text[*pos];
    (*pos)++;
    if (c == '"') {
      if (*pos < len && text[*pos] == '"') {
        (*pos)++;
        const elml_status status = elml_builder_put(builder, '"');
        if (status != ELML_OK) {
          return status;
        }
        continue;
      }
      /* Closing quote: allow trailing spaces, then delimiter or newline. */
      while (*pos < len && elml_is_space(text[*pos])) {
        (*pos)++;
      }
      if (*pos < len && text[*pos] != delimiter && text[*pos] != '\n' &&
          text[*pos] != '\r') {
        return ELML_ERR_INVALID_ARG;
      }
      return elml_builder_put(builder, '\0');
    }
    const elml_status status = elml_builder_put(builder, c);
    if (status != ELML_OK) {
      return status;
    }
  }

  return ELML_ERR_TRUNCATED; /* unterminated quote */
}

static elml_status elml_parse_unquoted(const char *text, size_t len, size_t *pos,
                                       char delimiter, elml_csv_builder *builder,
                                       size_t *field)
{
  const size_t start = *pos;
  while (*pos < len && text[*pos] != delimiter && text[*pos] != '\n' &&
         text[*pos] != '\r') {
    (*pos)++;
  }

  size_t begin = start;
  size_t end = *pos;
  while (begin < end && elml_is_space(text[begin])) {
    begin++;
  }
  while (end > begin && elml_is_space(text[end - 1u])) {
    end--;
  }

  *field = builder->storage_len;
  for (size_t i = begin; i < end; ++i) {
    const elml_status status = elml_builder_put(builder, text[i]);
    if (status != ELML_OK) {
      return status;
    }
  }
  return elml_builder_put(builder, '\0');
}

static bool elml_line_is_comment(const char *text, size_t len, size_t pos)
{
  while (pos < len && elml_is_space(text[pos])) {
    pos++;
  }
  return pos < len && text[pos] == '#';
}

static bool elml_skip_blank_or_comment(const char *text, size_t len,
                                       size_t *pos)
{
  while (*pos < len) {
    if (elml_line_is_comment(text, len, *pos)) {
      while (*pos < len && text[*pos] != '\n') {
        (*pos)++;
      }
      if (*pos < len) {
        (*pos)++;
      }
      continue;
    }

    size_t look = *pos;
    while (look < len && elml_is_space(text[look])) {
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
    if (text[look] == '\r') {
      *pos = look + 1u;
      if (*pos < len && text[*pos] == '\n') {
        (*pos)++;
      }
      continue;
    }
    *pos = look;
    return true;
  }
  return false;
}

static elml_status elml_csv_parse_impl(const char *text, size_t len,
                                       char delimiter, bool has_header,
                                       bool allow_single_row, elml_csv *out)
{
  if (text == nullptr || out == nullptr || delimiter == '\0' ||
      delimiter == '"' || delimiter == '\n' || delimiter == '\r') {
    return ELML_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof *out);

  size_t start = 0u;
  if (len >= 3u && (uint8_t)text[0] == 0xEFu && (uint8_t)text[1] == 0xBBu &&
      (uint8_t)text[2] == 0xBFu) {
    start = 3u;
  }

  size_t capacity;
  if (ckd_add(&capacity, len, 2u)) {
    return ELML_ERR_OUT_OF_MEMORY;
  }

  elml_csv_builder builder = {
      .storage = (char *)malloc(capacity),
      .storage_cap = capacity,
  };
  if (builder.storage == nullptr) {
    return ELML_ERR_OUT_OF_MEMORY;
  }

  elml_status status = ELML_OK;
  size_t pos = start;
  uint32_t expected_cols = 0u;
  bool have_expected = false;

  while (elml_skip_blank_or_comment(text, len, &pos)) {
    uint32_t cols = 0u;
    bool end_row = false;

    while (!end_row) {
      size_t field = 0u;
      if (pos < len && text[pos] == '"') {
        status = elml_parse_quoted(text, len, &pos, delimiter, &builder, &field);
      } else {
        status = elml_parse_unquoted(text, len, &pos, delimiter, &builder,
                                     &field);
      }
      if (status != ELML_OK) {
        goto fail;
      }
      status = elml_builder_add_offset(&builder, field);
      if (status != ELML_OK) {
        goto fail;
      }
      cols++;

      if (pos < len && text[pos] == delimiter) {
        pos++;
        continue;
      }
      if (pos < len && text[pos] == '\r') {
        pos++;
      }
      if (pos < len && text[pos] == '\n') {
        pos++;
      }
      end_row = true;
    }

    if (!have_expected) {
      expected_cols = cols;
      have_expected = true;
    } else if (cols != expected_cols) {
      status = ELML_ERR_INVALID_ARG;
      goto fail;
    }
    status = elml_builder_add_row(&builder, cols);
    if (status != ELML_OK) {
      goto fail;
    }
  }

  if (builder.row_count == 0u || expected_cols == 0u) {
    status = ELML_ERR_TRUNCATED;
    goto fail;
  }
  if (has_header && !allow_single_row && builder.row_count < 2u) {
    status = ELML_ERR_TRUNCATED;
    goto fail;
  }

  size_t cell_count;
  if (ckd_mul(&cell_count, builder.row_count, (size_t)expected_cols)) {
    status = ELML_ERR_OUT_OF_MEMORY;
    goto fail;
  }
  char **cells = (char **)calloc(cell_count, sizeof(char *));
  if (cells == nullptr) {
    status = ELML_ERR_OUT_OF_MEMORY;
    goto fail;
  }

  const size_t *offsets = (const size_t *)(const void *)builder.offsets;
  for (size_t row = 0u; row < builder.row_count; ++row) {
    for (uint32_t col = 0u; col < expected_cols; ++col) {
      const size_t index = row * (size_t)expected_cols + col;
      cells[index] = builder.storage + offsets[index];
    }
  }

  out->storage = builder.storage;
  out->cells = cells;
  out->rows = (uint32_t)builder.row_count;
  out->cols = expected_cols;
  out->data_start = has_header ? 1u : 0u;

  free(builder.offsets);
  free(builder.row_cols);
  return ELML_OK;

fail:
  free(builder.storage);
  free(builder.offsets);
  free(builder.row_cols);
  return status;
}

elml_status elml_csv_parse(const char *text, size_t len, char delimiter,
                           bool has_header, elml_csv *out)
{
  return elml_csv_parse_impl(text, len, delimiter, has_header, false, out);
}

static elml_status elml_csv_buffer_put(char **buffer, size_t *length,
                                       size_t *capacity, char c)
{
  if (*length == SIZE_MAX) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  const size_t needed = *length + 1u;
  if (needed > *capacity) {
    size_t next = (*capacity == 0u) ? 128u : *capacity;
    while (next < needed) {
      if (next > SIZE_MAX / 2u) {
        next = needed;
        break;
      }
      next *= 2u;
    }
    char *grown = (char *)realloc(*buffer, next);
    if (grown == nullptr) {
      return ELML_ERR_OUT_OF_MEMORY;
    }
    *buffer = grown;
    *capacity = next;
  }
  (*buffer)[*length] = c;
  *length = needed;
  return ELML_OK;
}

static bool elml_csv_record_has_open_quote(const char *text, size_t len,
                                           size_t pos)
{
  bool in_quotes = false;
  for (size_t i = pos; i < len; ++i) {
    if (text[i] != '"') {
      continue;
    }
    if (in_quotes && i + 1u < len && text[i + 1u] == '"') {
      ++i;
      continue;
    }
    in_quotes = !in_quotes;
  }
  return in_quotes;
}

/* Read one physical line, appending it to the record buffer. */
static bool elml_csv_read_line(FILE *file, char **buffer, size_t *length,
                               size_t *capacity, elml_status *status)
{
  bool got = false;
  int c;
  while ((c = fgetc(file)) != EOF) {
    got = true;
    *status = elml_csv_buffer_put(buffer, length, capacity, (char)c);
    if (*status != ELML_OK) {
      return false;
    }
    if (c == '\n') {
      break;
    }
  }
  return got;
}

elml_status elml_csv_read_header(const char *path, char delimiter,
                                 elml_csv *out)
{
  if (path == nullptr || out == nullptr || delimiter == '\0' ||
      delimiter == '"' || delimiter == '\n' || delimiter == '\r') {
    return ELML_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof *out);

  FILE *file = fopen(path, "rb");
  if (file == nullptr) {
    return ELML_ERR_NOT_FOUND;
  }

  /* Consume an optional UTF-8 BOM. */
  const int first = fgetc(file);
  const int second = fgetc(file);
  const int third = fgetc(file);
  if (first == EOF) {
    fclose(file);
    return ELML_ERR_TRUNCATED;
  }
  if (!((uint8_t)first == 0xEFu && second != EOF && third != EOF &&
        (uint8_t)second == 0xBBu && (uint8_t)third == 0xBFu)) {
    if (third != EOF) {
      (void)ungetc(third, file);
    }
    if (second != EOF) {
      (void)ungetc(second, file);
    }
    (void)ungetc(first, file);
  }

  char *record = nullptr;
  size_t length = 0u;
  size_t capacity = 0u;
  elml_status status = ELML_OK;

  for (;;) {
    length = 0u;
    status = ELML_OK;
    if (!elml_csv_read_line(file, &record, &length, &capacity, &status)) {
      if (status == ELML_OK) {
        status = ELML_ERR_TRUNCATED;
      }
      goto done;
    }

    size_t pos = 0u;
    if (!elml_skip_blank_or_comment(record, length, &pos)) {
      continue; /* blank line or comment */
    }

    while (elml_csv_record_has_open_quote(record, length, pos)) {
      status = ELML_OK;
      if (!elml_csv_read_line(file, &record, &length, &capacity, &status)) {
        if (status == ELML_OK) {
          status = ELML_ERR_TRUNCATED;
        }
        goto done;
      }
    }
    break;
  }

  status = elml_csv_parse_impl(record, length, delimiter, false, true, out);

done:
  free(record);
  fclose(file);
  return status;
}

elml_status elml_csv_load(const char *path, char delimiter, bool has_header,
                          elml_csv *out)
{
  uint8_t *data = nullptr;
  size_t len = 0u;
  const elml_status status = elml_read_file(path, &data, &len, 0u);
  if (status != ELML_OK) {
    return status;
  }
  const elml_status parse_status =
      elml_csv_parse((const char *)data, len, delimiter, has_header, out);
  free(data);
  return parse_status;
}

void elml_csv_free(elml_csv *csv)
{
  if (csv == nullptr) {
    return;
  }
  free(csv->storage);
  free(csv->cells);
  csv->storage = nullptr;
  csv->cells = nullptr;
  csv->rows = 0u;
  csv->cols = 0u;
  csv->data_start = 0u;
}

const char *elml_csv_cell(const elml_csv *csv, uint32_t row, uint32_t col)
{
  if (csv == nullptr || row >= csv->rows || col >= csv->cols) {
    return nullptr;
  }
  return csv->cells[(size_t)row * csv->cols + col];
}
