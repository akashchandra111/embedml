#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "elml/types.h"
#include "test.h"
#include "train/csv.h"

static void test_quoted(void)
{
  const char *text =
      "# leading comment\n"
      "\n"
      "name,note,value\r\n"
      "\"a,b\",\"he said \"\"hi\"\"\",1\r\n"
      "plain,\" spaced \",2\n";
  elml_csv csv;
  CHECK_EQ_INT(elml_csv_parse(text, strlen(text), ',', true, &csv), ELML_OK);
  CHECK_EQ_INT(csv.rows, 3u);
  CHECK_EQ_INT(csv.cols, 3u);
  CHECK_EQ_INT(csv.data_start, 1u);
  CHECK_STR(elml_csv_cell(&csv, 0u, 0u), "name");
  CHECK_STR(elml_csv_cell(&csv, 1u, 0u), "a,b");
  CHECK_STR(elml_csv_cell(&csv, 1u, 1u), "he said \"hi\"");
  CHECK_STR(elml_csv_cell(&csv, 1u, 2u), "1");
  CHECK_STR(elml_csv_cell(&csv, 2u, 1u), " spaced ");
  CHECK(elml_csv_cell(&csv, 3u, 0u) == NULL);
  CHECK(elml_csv_cell(&csv, 0u, 3u) == NULL);
  CHECK(elml_csv_cell(NULL, 0u, 0u) == NULL);
  elml_csv_free(&csv);
  elml_csv_free(&csv);
  elml_csv_free(NULL);
}

static void test_bom_and_semicolon(void)
{
  const char *text = "\xEF" "\xBB" "\xBF" "a;b\n1;\"x;y\"\n";
  elml_csv csv;
  CHECK_EQ_INT(elml_csv_parse(text, strlen(text), ';', true, &csv), ELML_OK);
  CHECK_STR(elml_csv_cell(&csv, 0u, 0u), "a");
  CHECK_STR(elml_csv_cell(&csv, 0u, 1u), "b");
  CHECK_STR(elml_csv_cell(&csv, 1u, 0u), "1");
  CHECK_STR(elml_csv_cell(&csv, 1u, 1u), "x;y");
  elml_csv_free(&csv);
}

static void test_errors(void)
{
  elml_csv csv;

  CHECK_EQ_INT(elml_csv_parse("", 0u, ',', false, &csv), ELML_ERR_TRUNCATED);
  CHECK_EQ_INT(elml_csv_parse(NULL, 1u, ',', false, &csv),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_csv_parse("a,b\n", 4u, ',', true, &csv),
               ELML_ERR_TRUNCATED);
  CHECK_EQ_INT(elml_csv_parse("a,b\n1\n", 6u, ',', false, &csv),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_csv_parse("a,b\n\"x\n", 7u, ',', false, &csv),
               ELML_ERR_TRUNCATED);
  CHECK_EQ_INT(elml_csv_parse("a,b\n1,2\n", 8u, '"', false, &csv),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_csv_parse("a,b\n1,2\n", 8u, '\n', false, &csv),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_csv_parse("a,b\n1,2\n", 8u, '\0', false, &csv),
               ELML_ERR_INVALID_ARG);

  const char *no_header = "1,2\n3,4\n";
  CHECK_EQ_INT(elml_csv_parse(no_header, strlen(no_header), ',', false, &csv),
               ELML_OK);
  CHECK_EQ_INT(csv.data_start, 0u);
  CHECK_STR(elml_csv_cell(&csv, 1u, 1u), "4");
  elml_csv_free(&csv);
}

static void test_load(void)
{
  char path[] = "elml_csv_XXXXXX";
  const int fd = mkstemp(path);
  CHECK(fd >= 0);
  if (fd < 0) {
    return;
  }
  FILE *file = fdopen(fd, "wb");
  CHECK(file != NULL);
  if (file != NULL) {
    fputs("x,y\n1,2\n", file);
    fclose(file);
  }

  elml_csv csv;
  CHECK_EQ_INT(elml_csv_load(path, ',', true, &csv), ELML_OK);
  CHECK_STR(elml_csv_cell(&csv, 1u, 1u), "2");
  elml_csv_free(&csv);
  remove(path);

  CHECK_EQ_INT(elml_csv_load("does-not-exist.csv", ',', true, &csv),
               ELML_ERR_NOT_FOUND);
}

static void test_read_header(void)
{
  char path[] = "elml_csv_header_XXXXXX";
  const int fd = mkstemp(path);
  CHECK(fd >= 0);
  if (fd < 0) {
    return;
  }
  FILE *file = fdopen(fd, "wb");
  CHECK(file != NULL);
  if (file != NULL) {
    fputs("\xEF" "\xBB" "\xBF" "# generated file\n"
          "\n"
          "\"name\",note,value\n"
          "1,2,3\n",
          file);
    fclose(file);
  }

  elml_csv header;
  CHECK_EQ_INT(elml_csv_read_header(path, ',', &header), ELML_OK);
  CHECK_EQ_INT(header.rows, 1u);
  CHECK_EQ_INT(header.cols, 3u);
  CHECK_EQ_INT(header.data_start, 0u);
  CHECK_STR(elml_csv_cell(&header, 0u, 0u), "name");
  CHECK_STR(elml_csv_cell(&header, 0u, 1u), "note");
  elml_csv_free(&header);
  remove(path);

  char quoted_path[] = "elml_csv_headerq_XXXXXX";
  const int quoted_fd = mkstemp(quoted_path);
  CHECK(quoted_fd >= 0);
  if (quoted_fd >= 0) {
    FILE *quoted = fdopen(quoted_fd, "wb");
    CHECK(quoted != NULL);
    if (quoted != NULL) {
      fputs("# a comment with a \" quote\n"
            "a,\"multi\nline\",c\n"
            "1,2,3\n",
            quoted);
      fclose(quoted);
    }
    CHECK_EQ_INT(elml_csv_read_header(quoted_path, ',', &header), ELML_OK);
    CHECK_EQ_INT(header.cols, 3u);
    CHECK_STR(elml_csv_cell(&header, 0u, 0u), "a");
    CHECK_STR(elml_csv_cell(&header, 0u, 1u), "multi\nline");
    CHECK_STR(elml_csv_cell(&header, 0u, 2u), "c");
    elml_csv_free(&header);
    remove(quoted_path);
  }

  CHECK_EQ_INT(elml_csv_read_header("does-not-exist.csv", ',', &header),
               ELML_ERR_NOT_FOUND);
}

int main(void)
{
  test_quoted();
  test_bom_and_semicolon();
  test_errors();
  test_load();
  test_read_header();
  TEST_END();
}
