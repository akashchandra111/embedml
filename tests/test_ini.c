#define _POSIX_C_SOURCE 200809L

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "elml/types.h"
#include "test.h"
#include "train/ini.h"

static void test_parse(void)
{
  const char *text =
      "; comment\n"
      "# another comment\n"
      "root = 1\n"
      "[data]\n"
      "csv = iris.csv\n"
      "quoted = \"hello world\"\n"
      "single = 'value'\n"
      "[model]\n"
      "layers = 8,4\n"
      "layers = 8\n"
      "spaces   =    padded value   \n"
      "[data]\n"
      "csv = override.csv\n";

  elml_ini ini;
  CHECK_EQ_INT(elml_ini_parse(text, strlen(text), &ini), ELML_OK);
  CHECK_EQ_INT(ini.count, 8u);
  CHECK_STR(elml_ini_get(&ini, "", "root"), "1");
  CHECK_STR(elml_ini_get(&ini, "data", "csv"), "override.csv");
  CHECK_STR(elml_ini_get(&ini, "data", "quoted"), "hello world");
  CHECK_STR(elml_ini_get(&ini, "data", "single"), "value");
  CHECK_STR(elml_ini_get(&ini, "model", "layers"), "8");
  CHECK_STR(elml_ini_get(&ini, "model", "spaces"), "padded value");
  CHECK(elml_ini_get(&ini, "model", "missing") == NULL);
  CHECK(elml_ini_get(&ini, "missing", "csv") == NULL);
  CHECK(elml_ini_get(NULL, "data", "csv") == NULL);
  CHECK(elml_ini_get(&ini, NULL, "csv") == NULL);
  CHECK(elml_ini_get(&ini, "data", NULL) == NULL);
  elml_ini_free(&ini);
  elml_ini_free(&ini);
  elml_ini_free(NULL);
}

static void test_errors(void)
{
  elml_ini ini;
  CHECK_EQ_INT(elml_ini_parse(NULL, 0u, &ini), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_ini_parse("[data\ncsv = x\n", strlen("[data\ncsv = x\n"), &ini),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_ini_parse("[]\n", strlen("[]\n"), &ini), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_ini_parse("[data]\nnovalue\n", strlen("[data]\nnovalue\n"), &ini),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_ini_parse("[data]\n = value\n", strlen("[data]\n = value\n"), &ini),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_ini_parse("[data]\ncsv = x\n", strlen("[data]\ncsv = x\n"), &ini), ELML_OK);
  elml_ini_free(&ini);
}

static void test_load(void)
{
  char path[] = "elml_ini_XXXXXX";
  const int fd = mkstemp(path);
  CHECK(fd >= 0);
  if (fd < 0) {
    return;
  }
  FILE *file = fdopen(fd, "wb");
  CHECK(file != NULL);
  if (file != NULL) {
    fputs("[data]\ncsv = test.csv\n", file);
    fclose(file);
  }

  elml_ini ini;
  CHECK_EQ_INT(elml_ini_load(path, &ini), ELML_OK);
  CHECK_STR(elml_ini_get(&ini, "data", "csv"), "test.csv");
  elml_ini_free(&ini);
  remove(path);

  CHECK_EQ_INT(elml_ini_load("does-not-exist.ini", &ini), ELML_ERR_NOT_FOUND);
}

int main(void)
{
  test_parse();
  test_errors();
  test_load();
  TEST_END();
}
