/* Host-side test harness for the bare-metal example. */
#include <stddef.h>
#include <stdio.h>

#include "elml/types.h"

int elml_example_predict(const float *input, float *output, int *label);
int elml_example_label_count(void);

typedef struct sample {
  float input[4];
  int label;
  const char *name;
} sample;

int main(void)
{
  static const sample samples[] = {
      {{5.1f, 3.5f, 1.4f, 0.2f}, 0, "setosa"},
      {{5.9f, 3.0f, 4.2f, 1.5f}, 1, "versicolor"},
      {{6.7f, 3.0f, 5.2f, 2.3f}, 2, "virginica"},
  };

  const int labels = elml_example_label_count();
  if (labels != 3) {
    fprintf(stderr, "host_check: expected 3 labels, got %d\n", labels);
    return 1;
  }

  int failures = 0;
  for (size_t i = 0u; i < sizeof samples / sizeof samples[0]; ++i) {
    float output[3] = {0.0f, 0.0f, 0.0f};
    int label = -1;
    const int status =
        elml_example_predict(samples[i].input, output, &label);
    if (status != 0) {
      fprintf(stderr, "host_check: sample %zu failed with %d\n", i, status);
      failures++;
      continue;
    }
    printf("%-10s -> %d (%.3f %.3f %.3f)\n", samples[i].name, label,
           (double)output[0], (double)output[1], (double)output[2]);
    if (label != samples[i].label) {
      fprintf(stderr, "host_check: sample %zu expected %d\n", i,
              samples[i].label);
      failures++;
    }
  }

  return (failures == 0) ? 0 : 1;
}
