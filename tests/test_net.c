#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "elml/registry.h"
#include "elml/types.h"
#include "test.h"
#include "train/config.h"
#include "train/dataset.h"
#include "train/net.h"

static const float g_xor_x[4][2] = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f},
};
static const float g_xor_y[4] = {0.0f, 1.0f, 1.0f, 0.0f};

static void train_xor(elml_net *net, uint32_t epochs)
{
  for (uint32_t epoch = 0u; epoch < epochs; ++epoch) {
    elml_net_zero_grad(net);
    for (uint32_t i = 0u; i < 4u; ++i) {
      CHECK_EQ_INT(elml_net_accumulate(net, g_xor_x[i], &g_xor_y[i], 0u),
                   ELML_OK);
    }
    elml_net_update(net, 4u);
  }
}

static void test_xor(void)
{
  const uint32_t hidden[1] = {4u};
  const elml_activation acts[2] = {ELML_ACT_TANH, ELML_ACT_SIGMOID};
  elml_net net;
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, acts, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1234u),
               ELML_OK);
  train_xor(&net, 2000u);

  for (uint32_t i = 0u; i < 4u; ++i) {
    CHECK_EQ_INT(elml_net_forward(&net, g_xor_x[i], NULL), ELML_OK);
    const float value = net.layers[net.layer_count - 1u].a[0];
    if (g_xor_y[i] > 0.5f) {
      CHECK(value > 0.9f);
    } else {
      CHECK(value < 0.1f);
    }
  }

  elml_net_save_best(&net);
  const float saved = net.layers[0].weights[0];
  net.layers[0].weights[0] = 123.0f;
  elml_net_restore_best(&net);
  CHECK_CLOSE(net.layers[0].weights[0], saved, 0.0);
  elml_net_restore_best(&net);

  elml_net second;
  CHECK_EQ_INT(elml_net_create(&second, hidden, 1u, acts, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1234u),
               ELML_OK);
  train_xor(&second, 2000u);
  for (uint32_t l = 0u; l < net.layer_count; ++l) {
    const size_t weights =
        (size_t)net.layers[l].in_dim * net.layers[l].out_dim;
    CHECK(memcmp(net.layers[l].weights, second.layers[l].weights,
                 weights * sizeof(float)) == 0);
    CHECK(memcmp(net.layers[l].biases, second.layers[l].biases,
                 (size_t)net.layers[l].out_dim * sizeof(float)) == 0);
  }
  elml_net_free(&second);
  elml_net_free(&net);
}

static void test_classification(void)
{
  const uint32_t hidden[1] = {4u};
  const elml_activation acts[2] = {ELML_ACT_RELU, ELML_ACT_SOFTMAX};
  elml_net net;
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, acts, 2u, 2u, 2u,
                               ELML_LOSS_CCE, ELML_OPTIMIZER_ADAM, 0.05f, 0.0f,
                               7u),
               ELML_OK);

  float x[4][2] = {{-1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}};
  float y[4][2] = {{1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}};
  uint8_t classes[4] = {0u, 0u, 1u, 1u};

  for (uint32_t epoch = 0u; epoch < 500u; ++epoch) {
    elml_net_zero_grad(&net);
    for (uint32_t i = 0u; i < 4u; ++i) {
      CHECK_EQ_INT(elml_net_accumulate(&net, x[i], y[i], classes[i]),
                   ELML_OK);
    }
    elml_net_update(&net, 4u);
  }

  elml_dataset dataset = {
      .rows = 4u,
      .input_dim = 2u,
      .output_dim = 2u,
      .x = &x[0][0],
      .y = &y[0][0],
      .class_index = classes,
  };
  uint32_t correct = 0u;
  const float loss = elml_net_evaluate(&net, &dataset, &correct);
  CHECK(loss < 0.2f);
  CHECK_EQ_INT(correct, 4u);
  CHECK_EQ_INT(elml_net_argmax(&net), 1u);

  elml_dataset empty = {0};
  CHECK_CLOSE(elml_net_evaluate(&net, &empty, &correct), 0.0, 0.0);
  CHECK_EQ_INT(correct, 0u);

  elml_net_free(&net);
}

static void test_invalid(void)
{
  const uint32_t hidden[1] = {4u};
  const elml_activation softmax_hidden[2] = {ELML_ACT_SOFTMAX,
                                             ELML_ACT_LINEAR};
  const elml_activation cce_linear[2] = {ELML_ACT_RELU, ELML_ACT_LINEAR};
  const elml_activation mse_softmax[2] = {ELML_ACT_RELU, ELML_ACT_SOFTMAX};
  const elml_activation good[2] = {ELML_ACT_RELU, ELML_ACT_LINEAR};
  elml_net net;

  CHECK_EQ_INT(elml_net_create(NULL, hidden, 1u, good, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, NULL, 1u, good, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 0u, good, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, good, 1u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, softmax_hidden, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, cce_linear, 2u, 2u, 2u,
                               ELML_LOSS_CCE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, mse_softmax, 2u, 2u, 2u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, good, 2u, 0u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, good, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.0f, 0.0f,
                               1u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, good, 2u, 2u, 1u,
                               ELML_LOSS_MSE, ELML_OPTIMIZER_ADAM, 0.1f, -1.0f,
                               1u),
               ELML_ERR_INVALID_ARG);

  const elml_activation cce_softmax[2] = {ELML_ACT_RELU, ELML_ACT_SOFTMAX};
  CHECK_EQ_INT(elml_net_create(&net, hidden, 1u, cce_softmax, 2u, 2u, 2u,
                               ELML_LOSS_CCE, ELML_OPTIMIZER_SGD, 0.1f, 0.0f,
                               1u),
               ELML_OK);
  CHECK_EQ_INT(elml_net_accumulate(&net, g_xor_x[0], NULL, 2u),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_accumulate(&net, g_xor_x[0], NULL, 1u), ELML_OK);
  elml_net_update(&net, 1u);
  elml_net_update(&net, 0u);
  elml_net_zero_grad(NULL);
  elml_net_update(NULL, 1u);
  CHECK_EQ_INT(elml_net_forward(NULL, g_xor_x[0], NULL),
               ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_forward(&net, NULL, NULL), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_net_argmax(NULL), 0u);
  elml_net_free(NULL);
  elml_net_free(&net);
}

int main(void)
{
  test_xor();
  test_classification();
  test_invalid();
  TEST_END();
}
