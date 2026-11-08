#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/model.h"
#include "elml/types.h"
#include "format/format_internal.h"
#include "format/inspect.h"

static void usage(FILE *out)
{
  fprintf(out,
          "usage: elml_inspect --model FILE [--json]\n"
          "\n"
          "Parses a model without running it. Unknown layer types are shown\n"
          "but do not prevent inspection.\n");
}

int main(int argc, char **argv)
{
  const char *path = nullptr;
  bool json = false;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      path = argv[++i];
    } else if (strcmp(argv[i], "--json") == 0) {
      json = true;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      usage(stdout);
      return 0;
    } else if (path == nullptr) {
      path = argv[i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  if (path == nullptr) {
    usage(stderr);
    return 2;
  }

  FILE *file = fopen(path, "rb");
  if (file == nullptr) {
    fprintf(stderr, "elml_inspect: cannot open %s\n", path);
    return 1;
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    fprintf(stderr, "elml_inspect: cannot seek %s\n", path);
    fclose(file);
    return 1;
  }
  const long length = ftell(file);
  if (length <= 0) {
    fprintf(stderr, "elml_inspect: empty file %s\n", path);
    fclose(file);
    return 1;
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    fprintf(stderr, "elml_inspect: cannot rewind %s\n", path);
    fclose(file);
    return 1;
  }

  const size_t size = (size_t)length;
  const size_t capacity = (size + ELML_PARAM_ALIGN - 1u) &
                          ~(size_t)(ELML_PARAM_ALIGN - 1u);
  uint8_t *blob = (uint8_t *)aligned_alloc(ELML_PARAM_ALIGN, capacity);
  if (blob == nullptr) {
    fprintf(stderr, "elml_inspect: out of memory\n");
    fclose(file);
    return 1;
  }

  const size_t read = fread(blob, 1u, size, file);
  fclose(file);
  if (read != size) {
    fprintf(stderr, "elml_inspect: short read\n");
    free(blob);
    return 1;
  }

  elml_format format;
  const elml_status status = elml_format_parse(blob, size, &format);
  if (status != ELML_OK) {
    fprintf(stderr, "elml_inspect: %s\n", elml_error_string(status));
    free(blob);
    return 1;
  }

  elml_inspect_print(&format, blob, stdout, json);
  free(blob);
  return 0;
}
