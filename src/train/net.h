/*
 * Host-only training network.
 *
 * The training net is deliberately separate from the inference core: it owns
 * mutable buffers, gradients and optimizer state, while the inference core is
 * allocation-free and read-only. Weights are exported through elml_model_write
 * and then consumed by elml_model_init / elml_forward.
 */
#ifndef ELML_NET_H
#define ELML_NET_H

#include <stdbool.h>
#include <stdint.h>

#include "elml/registry.h"
#include "elml/types.h"
#include "train/config.h"
#include "train/dataset.h"

typedef struct elml_net_layer {
  uint32_t in_dim;
  uint32_t out_dim;
  elml_activation act;

  float *weights; /* out_dim * in_dim, row-major */
  float *biases;  /* out_dim */

  float *grad_w;
  float *grad_b;

  float *adam_mw;
  float *adam_vw;
  float *adam_mb;
  float *adam_vb;

  float *z;     /* pre-activation, out_dim */
  float *a;     /* activation, out_dim */
  float *delta; /* backprop delta, out_dim */

  float *best_weights;
  float *best_biases;
} elml_net_layer;

typedef struct elml_net {
  uint32_t input_dim;
  uint32_t output_dim;
  uint32_t layer_count;
  elml_net_layer layers[ELML_MAX_MODEL_LAYERS];

  elml_loss loss;
  elml_optimizer optimizer;
  float learning_rate;
  float l2;

  float beta1;
  float beta2;
  float epsilon;
  float beta1_pow;
  float beta2_pow;
  uint64_t step;

  bool has_best;
  float *input_copy;
} elml_net;

elml_status elml_net_create(elml_net *net, const uint32_t *hidden,
                            uint32_t hidden_count,
                            const elml_activation *activations,
                            uint32_t activation_count, uint32_t input_dim,
                            uint32_t output_dim, elml_loss loss,
                            elml_optimizer optimizer, float learning_rate,
                            float l2, uint64_t seed);

void elml_net_free(elml_net *net);

/* Forward pass; stores every pre-activation and activation for backprop. */
elml_status elml_net_forward(elml_net *net, const float *input, float *output);

/* Classification prediction: argmax over the output layer. */
uint32_t elml_net_argmax(const elml_net *net);

void elml_net_zero_grad(elml_net *net);

/*
 * Runs forward + backward for one sample and accumulates gradients. The caller
 * must call elml_net_update once per mini-batch afterwards.
 */
elml_status elml_net_accumulate(elml_net *net, const float *input,
                                const float *target, uint32_t class_index);

void elml_net_update(elml_net *net, uint32_t batch_size);

/*
 * Average loss over a dataset. For classification the returned loss is the
 * mean cross-entropy and *correct receives the number of correct predictions.
 * For regression *correct is set to 0.
 */
float elml_net_evaluate(elml_net *net, const elml_dataset *dataset,
                        uint32_t *correct);

void elml_net_save_best(elml_net *net);
void elml_net_restore_best(elml_net *net);

#endif /* ELML_NET_H */
