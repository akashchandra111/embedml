#ifndef ELML_TEST_H
#define ELML_TEST_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int elml_test_checks;
static int elml_test_failures;

#define CHECK(condition)                                                     \
  do {                                                                       \
    elml_test_checks++;                                                      \
    if (!(condition)) {                                                      \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,       \
              #condition);                                                   \
      elml_test_failures++;                                                  \
    }                                                                        \
  } while (0)

#define CHECK_EQ_INT(actual, expected)                                       \
  do {                                                                       \
    const long long elml_a = (long long)(actual);                            \
    const long long elml_b = (long long)(expected);                          \
    elml_test_checks++;                                                      \
    if (elml_a != elml_b) {                                                  \
      fprintf(stderr, "%s:%d: %s == %lld, expected %lld\n", __FILE__,        \
              __LINE__, #actual, elml_a, elml_b);                            \
      elml_test_failures++;                                                  \
    }                                                                        \
  } while (0)

#define CHECK_CLOSE(actual, expected, tolerance)                             \
  do {                                                                       \
    const double elml_a = (double)(actual);                                  \
    const double elml_b = (double)(expected);                                \
    const double elml_t = (double)(tolerance);                               \
    elml_test_checks++;                                                      \
    if (!(fabs(elml_a - elml_b) <= elml_t)) {                                \
      fprintf(stderr, "%s:%d: %s == %.9f, expected %.9f (tol %.9f)\n",       \
              __FILE__, __LINE__, #actual, elml_a, elml_b, elml_t);          \
      elml_test_failures++;                                                  \
    }                                                                        \
  } while (0)

#define CHECK_STR(actual, expected)                                          \
  do {                                                                       \
    const char *elml_a = (actual);                                           \
    const char *elml_b = (expected);                                         \
    elml_test_checks++;                                                      \
    if (elml_a == NULL || elml_b == NULL || strcmp(elml_a, elml_b) != 0) {   \
      fprintf(stderr, "%s:%d: %s == \"%s\", expected \"%s\"\n", __FILE__,    \
              __LINE__, #actual, (elml_a != NULL) ? elml_a : "(null)",       \
              (elml_b != NULL) ? elml_b : "(null)");                         \
      elml_test_failures++;                                                  \
    }                                                                        \
  } while (0)

#define TEST_END()                                                           \
  do {                                                                       \
    if (elml_test_failures != 0) {                                           \
      fprintf(stderr, "%d/%d checks failed\n", elml_test_failures,           \
              elml_test_checks);                                             \
      return 1;                                                              \
    }                                                                        \
    printf("%d checks passed\n", elml_test_checks);                          \
    return 0;                                                                \
  } while (0)

#endif /* ELML_TEST_H */
