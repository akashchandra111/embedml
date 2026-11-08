#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "elml/types.h"
#include "test.h"
#include "train/config.h"

static bool make_file(char *path, size_t capacity, const char *content)
{
  if (snprintf(path, capacity, "elml_cfg_XXXXXX") >= (int)capacity) {
    return false;
  }
  const int fd = mkstemp(path);
  if (fd < 0) {
    return false;
  }
  FILE *file = fdopen(fd, "wb");
  if (file == NULL) {
    close(fd);
    return false;
  }
  const bool ok = fputs(content, file) >= 0;
  return (fclose(file) == 0) && ok;
}

static elml_status load(char **argv, int argc, elml_config *config)
{
  return elml_config_load(config, argc, argv);
}

int main(void)
{
  char ini_path[64];
  const char *ini =
      "[data]\n"
      "csv = data.csv\n"
      "inputs = x1,x2\n"
      "outputs = y\n"
      "split = 0.7\n"
      "has_header = true\n"
      "[model]\n"
      "layers = 4,8\n"
      "activations = relu,tanh,linear\n"
      "loss = mse\n"
      "[train]\n"
      "epochs = 5\n"
      "batch = 2\n"
      "lr = 0.01\n"
      "threads = 4\n"
      "[export]\n"
      "out = model.elml\n";
  CHECK(make_file(ini_path, sizeof ini_path, ini));

  {
    char *argv[] = {"elml_train", "--config", ini_path, "--epochs", "9",
                    "--no-header", "--threads", "2"};
    elml_config config;
    CHECK_EQ_INT(load(argv, 8, &config), ELML_OK);
    CHECK_STR(config.csv_path, "data.csv");
    CHECK_STR(config.inputs, "x1,x2");
    CHECK_STR(config.outputs, "y");
    CHECK_STR(config.export_path, "model.elml");
    CHECK_EQ_INT(config.hidden_count, 2u);
    CHECK_EQ_INT(config.hidden[0], 4u);
    CHECK_EQ_INT(config.hidden[1], 8u);
    CHECK_EQ_INT(config.activation_count, 3u);
    CHECK_EQ_INT(config.activations[0], ELML_ACT_RELU);
    CHECK_EQ_INT(config.activations[1], ELML_ACT_TANH);
    CHECK_EQ_INT(config.activations[2], ELML_ACT_LINEAR);
    CHECK_EQ_INT(config.loss, ELML_LOSS_MSE);
    CHECK_EQ_INT(config.epochs, 9u);
    CHECK_EQ_INT(config.batch, 2u);
    CHECK_EQ_INT(config.threads, 1u);
    CHECK(!config.has_header);
    CHECK_CLOSE(config.split, 0.7, 1e-9);
    CHECK_CLOSE(config.learning_rate, 0.01, 1e-9);
  }

  {
    const char *cc = "[data]\ncsv = x.csv\ninputs = x\noutputs = y\n"
                     "[model]\nlayers = 3\nloss = cce\n"
                     "[export]\nout = y.elml\n";
    char cpath[64];
    CHECK(make_file(cpath, sizeof cpath, cc));
    char *argv[] = {"elml_train", "--config", cpath};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_OK);
    CHECK_EQ_INT(config.activation_count, 2u);
    CHECK_EQ_INT(config.activations[0], ELML_ACT_RELU);
    CHECK_EQ_INT(config.activations[1], ELML_ACT_SOFTMAX);
    CHECK(config.classification);
    remove(cpath);
  }

  {
    const char *bad = "[unknown]\nkey = 1\n";
    char path[64];
    CHECK(make_file(path, sizeof path, bad));
    char *argv[] = {"elml_train", "--config", path};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_INVALID_ARG);
    remove(path);
  }

  {
    const char *bad = "[data]\ncsv = x.csv\ninputs = x\noutputs = y\n"
                      "[model]\nlayers = 3\nactivations = relu,linear\n"
                      "loss = cce\n[export]\nout = y.elml\n";
    char path[64];
    CHECK(make_file(path, sizeof path, bad));
    char *argv[] = {"elml_train", "--config", path};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_INVALID_ARG);
    remove(path);
  }

  {
    const char *bad = "[data]\ncsv = x.csv\ninputs = x\noutputs = y\n"
                      "[model]\nlayers = 3\nactivations = softmax,linear\n"
                      "loss = mse\n[export]\nout = y.elml\n";
    char path[64];
    CHECK(make_file(path, sizeof path, bad));
    char *argv[] = {"elml_train", "--config", path};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_INVALID_ARG);
    remove(path);
  }

  {
    const char *bad = "[data]\ncsv = x.csv\ninputs = x\noutputs = y\n"
                      "[model]\nlayers = 4,8\nactivations = relu\n"
                      "[export]\nout = y.elml\n";
    char path[64];
    CHECK(make_file(path, sizeof path, bad));
    char *argv[] = {"elml_train", "--config", path};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_INVALID_ARG);
    remove(path);
  }

  {
    const char *bad = "[data]\ncsv = x.csv\ninputs = x\noutputs = y\n"
                      "split = 1.5\n[model]\nlayers = 3\n"
                      "[export]\nout = y.elml\n";
    char path[64];
    CHECK(make_file(path, sizeof path, bad));
    char *argv[] = {"elml_train", "--config", path};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_INVALID_ARG);
    remove(path);
  }

  {
    const char *bad = "[data]\ncsv = x.csv\ninputs = x\noutputs = y\n"
                      "[model]\nlayers = 3\n[train]\nl2 = -1\n"
                      "[export]\nout = y.elml\n";
    char path[64];
    CHECK(make_file(path, sizeof path, bad));
    char *argv[] = {"elml_train", "--config", path};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_INVALID_ARG);
    remove(path);
  }

  {
    const char *bad = "[data]\ncsv = x.csv\ninputs = x\noutputs = y\n"
                      "[model]\nlayers = 3\n";
    char path[64];
    CHECK(make_file(path, sizeof path, bad));
    char *argv[] = {"elml_train", "--config", path};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_INVALID_ARG);
    remove(path);
  }

  {
    char *argv[] = {"elml_train", "--help"};
    elml_config config;
    CHECK_EQ_INT(load(argv, 2, &config), ELML_ERR_NOT_FOUND);
  }
  {
    char *argv[] = {"elml_train", "--bogus"};
    elml_config config;
    CHECK_EQ_INT(load(argv, 2, &config), ELML_ERR_INVALID_ARG);
  }
  {
    char *argv[] = {"elml_train", "--config", "does-not-exist.ini"};
    elml_config config;
    CHECK_EQ_INT(load(argv, 3, &config), ELML_ERR_NOT_FOUND);
  }
  {
    char *argv[] = {"elml_train", "--csv", "x.csv", "--inputs", "x",
                    "--outputs", "y", "--layers", "3", "--act", "relu,softmax",
                    "--loss", "cce", "--export", "y.elml"};
    elml_config config;
    CHECK_EQ_INT(load(argv, 15, &config), ELML_OK);
    CHECK_EQ_INT(config.loss, ELML_LOSS_CCE);
    CHECK(config.classification);
    CHECK_EQ_INT(config.activations[1], ELML_ACT_SOFTMAX);
  }

  remove(ini_path);
  TEST_END();
}
