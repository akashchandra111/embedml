#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "elml/arena.h"
#include "elml/model.h"
#include "elml/types.h"
#include "test.h"
#include "train/config.h"
#include "train/dataset.h"
#include "train/export.h"
#include "train/io.h"
#include "train/net.h"

static int make_temp_path(char *path, size_t capacity, const char *name)
{
  const int written = snprintf(path, capacity, "elml_export_%s_XXXXXX", name);
  if (written < 0 || (size_t)written >= capacity) {
    return -1;
  }
  return mkstemp(path);
}

static elml_net make_net(void)
{
  const uint32_t hidden[1] = {3u};
  const elml_activation acts[2] = {ELML_ACT_RELU, ELML_ACT_SOFTMAX};
  elml_net net;
  memset(&net, 0, sizeof net);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, acts, 2u, 2u, 2u,
                               ELML_LOSS_CCE, ELML_OPTIMIZER_ADAM, 0.01f, 0.0f,
                               123u),
               ELML_OK);
  return net;
}

static elml_dataset make_dataset(char **labels)
{
  elml_dataset dataset;
  memset(&dataset, 0, sizeof dataset);
  dataset.rows = 3u;
  dataset.input_dim = 2u;
  dataset.output_dim = 2u;
  if (labels != NULL) {
    dataset.labels = labels;
    dataset.label_count = 2u;
  }
  return dataset;
}

/*
 * Load a model into caller-owned storage. elml_model keeps pointers into its
 * own fields (layers reference descs), so returning one by value would leave
 * dangling pointers into the dead stack frame.
 */
static bool load_model(const char *path, elml_model *model, uint8_t **blob_out)
{
  uint8_t *blob = NULL;
  size_t blob_len = 0u;
  memset(model, 0, sizeof *model);
  *blob_out = NULL;
  if (elml_read_file(path, &blob, &blob_len, ELML_PARAM_ALIGN) != ELML_OK) {
    return false;
  }
  if (elml_model_init(model, blob, blob_len, ELML_DTYPE_F32) != ELML_OK) {
    elml_aligned_free(blob);
    return false;
  }
  *blob_out = blob;
  return true;
}

int main(void)
{
  char path[] = "elml_export_plain_XXXXXX";
  const int fd = make_temp_path(path, sizeof path, "plain");
  CHECK(fd >= 0);
  if (fd >= 0) {
    close(fd);

    elml_net net = make_net();
    static char cat_label[] = "cat";
    static char dog_label[] = "dog";
    char *label_list[2] = {cat_label, dog_label};
    elml_dataset dataset = make_dataset(label_list);

    elml_config config;
    elml_config_init(&config);
    snprintf(config.inputs, sizeof config.inputs, "%s", "x1,x2");
    snprintf(config.outputs, sizeof config.outputs, "%s", "species");
    snprintf(config.export_path, sizeof config.export_path, "%s", path);

    /* Argument validation. */
    CHECK_EQ_INT(elml_export_model(NULL, NULL, &dataset, &config, path),
                 ELML_ERR_INVALID_ARG);
    CHECK_EQ_INT(elml_export_model(&net, NULL, &dataset, NULL, path),
                 ELML_ERR_INVALID_ARG);
    CHECK_EQ_INT(elml_export_model(&net, NULL, &dataset, &config, NULL),
                 ELML_ERR_INVALID_ARG);
    elml_net empty;
    memset(&empty, 0, sizeof empty);
    CHECK_EQ_INT(elml_export_model(&empty, NULL, &dataset, &config, path),
                 ELML_ERR_INVALID_ARG);

    CHECK_EQ_INT(elml_export_model(&net, NULL, &dataset, &config, path),
                 ELML_OK);

    uint8_t *blob = NULL;
    elml_model model;
    CHECK(load_model(path, &model, &blob));
    CHECK(blob != NULL);
    CHECK_EQ_INT(model.layer_count, 4u);
    CHECK_EQ_INT(model.input.dim[0], 2u);
    CHECK_EQ_INT(model.output.dim[0], 2u);
    CHECK_EQ_INT(model.norm.input_mode, ELML_NORM_NONE);
    CHECK_EQ_INT(model.norm.output_mode, ELML_NORM_NONE);
    CHECK_EQ_INT(elml_label_count(&model), 2u);

    char text[128];
    size_t len = 0u;
    CHECK_EQ_INT(elml_label_get(&model, 0u, text, sizeof text, &len), ELML_OK);
    CHECK_STR(text, "cat");
    CHECK_EQ_INT(elml_meta_get(&model, "task", text, sizeof text, &len),
                 ELML_OK);
    CHECK_STR(text, "classification");
    CHECK_EQ_INT(elml_meta_get(&model, "producer", text, sizeof text, &len),
                 ELML_OK);
    CHECK_STR(text, "EmbedML " ELML_VERSION_STRING);
    CHECK_EQ_INT(elml_meta_get(&model, "loss", text, sizeof text, &len),
                 ELML_OK);
    CHECK_STR(text, "cce");
    CHECK_EQ_INT(
        elml_meta_get(&model, "columns.inputs", text, sizeof text, &len),
        ELML_OK);
    CHECK_STR(text, "x1,x2");
    CHECK_EQ_INT(
        elml_meta_get(&model, "columns.outputs", text, sizeof text, &len),
        ELML_OK);
    CHECK_STR(text, "species");

    float input[2] = {0.5f, -0.25f};
    float output[2] = {0.0f, 0.0f};
    const size_t scratch_len = elml_scratch_bytes(&model, 1u);
    uint8_t *scratch = (uint8_t *)aligned_alloc(ELML_TENSOR_ALIGN, scratch_len);
    CHECK(scratch != NULL);
    if (scratch != NULL) {
      CHECK_EQ_INT(elml_forward(&model, scratch, scratch_len, input, output),
                   ELML_OK);
      CHECK_CLOSE((double)output[0] + (double)output[1], 1.0, 1e-5);
      free(scratch);
    }
    elml_aligned_free(blob);
    elml_net_free(&net);
  }

  char norm_path[] = "elml_export_norm_XXXXXX";
  const int norm_fd = make_temp_path(norm_path, sizeof norm_path, "norm");
  CHECK(norm_fd >= 0);
  if (norm_fd >= 0) {
    close(norm_fd);

    elml_net net = make_net();
    elml_dataset dataset = make_dataset(NULL);
    elml_config config;
    elml_config_init(&config);

    float in_mean[2] = {1.0f, 2.0f};
    float in_std[2] = {3.0f, 4.0f};
    float out_mean[2] = {5.0f, 6.0f};
    float out_std[2] = {7.0f, 8.0f};
    elml_norm_stats stats = {
        .has_input = true,
        .input_mean = in_mean,
        .input_std = in_std,
        .has_output = true,
        .output_mean = out_mean,
        .output_std = out_std,
    };

    CHECK_EQ_INT(elml_export_model(&net, &stats, &dataset, &config, norm_path),
                 ELML_OK);

    uint8_t *blob = NULL;
    elml_model model;
    CHECK(load_model(norm_path, &model, &blob));
    CHECK(blob != NULL);
    CHECK_EQ_INT(model.norm.input_mode, ELML_NORM_STANDARD);
    CHECK_EQ_INT(model.norm.output_mode, ELML_NORM_STANDARD);
    CHECK_EQ_INT(model.norm.input_count, 2u);
    CHECK_EQ_INT(model.norm.output_count, 2u);
    CHECK(model.norm.input_params != NULL);
    CHECK(model.norm.output_params != NULL);
    if (model.norm.input_params != NULL) {
      CHECK_CLOSE(model.norm.input_params[0], 1.0, 0.0);
      CHECK_CLOSE(model.norm.input_params[1], 3.0, 0.0);
      CHECK_CLOSE(model.norm.input_params[2], 2.0, 0.0);
      CHECK_CLOSE(model.norm.input_params[3], 4.0, 0.0);
    }
    if (model.norm.output_params != NULL) {
      CHECK_CLOSE(model.norm.output_params[0], 5.0, 0.0);
      CHECK_CLOSE(model.norm.output_params[1], 7.0, 0.0);
      CHECK_CLOSE(model.norm.output_params[2], 6.0, 0.0);
      CHECK_CLOSE(model.norm.output_params[3], 8.0, 0.0);
    }
    CHECK_EQ_INT(elml_label_count(&model), 0u);

    /* No column metadata in the config means the keys are absent. */
    char text[32];
    size_t len = 0u;
    CHECK_EQ_INT(elml_meta_get(&model, "labels", text, sizeof text, &len),
                 ELML_ERR_NOT_FOUND);
    CHECK_EQ_INT(
        elml_meta_get(&model, "columns.inputs", text, sizeof text, &len),
        ELML_ERR_NOT_FOUND);

    elml_aligned_free(blob);
    elml_net_free(&net);
  }

  CHECK_EQ_INT(remove(path), 0);
  CHECK_EQ_INT(remove(norm_path), 0);
  TEST_END();
}
