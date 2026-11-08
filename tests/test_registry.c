#include <stddef.h>
#include <stdint.h>

#include "elml/registry.h"
#include "elml/types.h"
#include "test.h"

static elml_status custom_prepare(elml_layer *layer,
                                  const elml_layer_desc *desc,
                                  elml_dtype dtype)
{
  (void)layer;
  (void)desc;
  (void)dtype;
  return ELML_OK;
}

static elml_status custom_forward(const elml_layer *layer,
                                  const elml_tensor *in, elml_tensor *out)
{
  (void)layer;
  (void)in;
  (void)out;
  return ELML_OK;
}

static const elml_layer_vt custom_vt = {
    .id = (elml_layer_id)0x0200,
    .version = 1u,
    .name = "custom",
    .prepare = custom_prepare,
    .forward = custom_forward,
};

int main(void)
{
  CHECK(elml_layer_find(ELML_LAYER_DENSE) != NULL);
  CHECK(elml_layer_find(ELML_LAYER_ACT_RELU) != NULL);
  CHECK(elml_layer_find(ELML_LAYER_ACT_LEAKY_RELU) != NULL);
  CHECK(elml_layer_find(ELML_LAYER_ACT_TANH) != NULL);
  CHECK(elml_layer_find(ELML_LAYER_ACT_SIGMOID) != NULL);
  CHECK(elml_layer_find(ELML_LAYER_ACT_SOFTMAX) != NULL);
  CHECK(elml_layer_find(ELML_LAYER_ACT_LINEAR) != NULL);
  CHECK(elml_layer_find(ELML_LAYER_CONV1D) == NULL);
  CHECK(elml_layer_find((elml_layer_id)0x7fff) == NULL);

  CHECK_EQ_INT(elml_registry_add(NULL), ELML_ERR_INVALID_ARG);
  CHECK_EQ_INT(elml_registry_add(&custom_vt), ELML_OK);
  CHECK(elml_layer_find(custom_vt.id) == &custom_vt);
  CHECK_EQ_INT(elml_registry_add(&custom_vt), ELML_ERR_INVALID_ARG);

  elml_layer_vt bad = custom_vt;
  bad.id = (elml_layer_id)0x0201;
  bad.prepare = NULL;
  CHECK_EQ_INT(elml_registry_add(&bad), ELML_ERR_INVALID_ARG);

  bad = custom_vt;
  bad.id = (elml_layer_id)0x0202;
  bad.forward = NULL;
  CHECK_EQ_INT(elml_registry_add(&bad), ELML_ERR_INVALID_ARG);

  bad = custom_vt;
  bad.id = ELML_LAYER_DENSE;
  CHECK_EQ_INT(elml_registry_add(&bad), ELML_ERR_INVALID_ARG);

  CHECK_STR(elml_layer_find(ELML_LAYER_DENSE)->name, "dense");
  CHECK_STR(elml_layer_find(ELML_LAYER_ACT_SOFTMAX)->name, "softmax");
  CHECK_EQ_INT(elml_layer_find(ELML_LAYER_ACT_RELU)->version, 1u);

  TEST_END();
}
