/*
 * Host-side model export. This is the only place where training state is
 * converted into the portable on-disk format.
 */
#ifndef ELML_EXPORT_H
#define ELML_EXPORT_H

#include <stdint.h>

#include "elml/types.h"
#include "train/config.h"
#include "train/dataset.h"
#include "train/net.h"

elml_status elml_export_model(const elml_net *net,
                              const elml_norm_stats *norm,
                              const elml_dataset *dataset,
                              const elml_config *config, const char *path);

#endif /* ELML_EXPORT_H */
