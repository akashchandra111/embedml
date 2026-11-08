#include "elml/registry.h"

#include <stddef.h>

#include "core/layer_vt.h"
#include "elml/types.h"

static const elml_layer_vt *const g_builtin_layers[] = {
    &elml_dense_layer_vt,        &elml_act_relu_layer_vt,
    &elml_act_leaky_relu_layer_vt, &elml_act_tanh_layer_vt,
    &elml_act_sigmoid_layer_vt,  &elml_act_softmax_layer_vt,
    &elml_act_linear_layer_vt,
};

#ifndef ELML_LAYERS_STATIC
#  define ELML_MAX_DYNAMIC_LAYERS 16
static const elml_layer_vt *g_dynamic_layers[ELML_MAX_DYNAMIC_LAYERS];
static size_t g_dynamic_count;

elml_status elml_registry_add(const elml_layer_vt *vt)
{
  if (vt == nullptr || vt->prepare == nullptr || vt->forward == nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (vt->id == ELML_LAYER_DENSE) {
    return ELML_ERR_INVALID_ARG;
  }
  if (elml_layer_find(vt->id) != nullptr) {
    return ELML_ERR_INVALID_ARG;
  }
  if (g_dynamic_count >= ELML_MAX_DYNAMIC_LAYERS) {
    return ELML_ERR_OUT_OF_MEMORY;
  }
  g_dynamic_layers[g_dynamic_count] = vt;
  g_dynamic_count++;
  return ELML_OK;
}
#else
elml_status elml_registry_add(const elml_layer_vt *vt)
{
  (void)vt;
  return ELML_ERR_UNSUPPORTED_LAYER_TYPE;
}
#endif

const elml_layer_vt *elml_layer_find(elml_layer_id id)
{
  for (size_t i = 0u; i < ELML_ARRAY_COUNT(g_builtin_layers); ++i) {
    if (g_builtin_layers[i]->id == id) {
      return g_builtin_layers[i];
    }
  }
#ifndef ELML_LAYERS_STATIC
  for (size_t i = 0u; i < g_dynamic_count; ++i) {
    if (g_dynamic_layers[i]->id == id) {
      return g_dynamic_layers[i];
    }
  }
#endif
  return nullptr;
}
