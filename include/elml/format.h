/*
 * On-disk format constants. The layout itself is specified in docs/FORMAT.md.
 */
#ifndef ELML_FORMAT_H
#define ELML_FORMAT_H

#include <stdint.h>

#include "elml/compat.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ELML_FORMAT_MAJOR 1u
#define ELML_FORMAT_MINOR 0u
#define ELML_HEADER_BYTES 64u
#define ELML_CHUNK_HEADER_BYTES 16u
#define ELML_MAX_CHUNKS 32u

#define ELML_MAGIC_0 ((uint8_t)'E')
#define ELML_MAGIC_1 ((uint8_t)'L')
#define ELML_MAGIC_2 ((uint8_t)'M')
#define ELML_MAGIC_3 ((uint8_t)'L')

#define ELML_FOURCC(a, b, c, d)                                      \
  ((uint32_t)(uint8_t)(a) | ((uint32_t)(uint8_t)(b) << 8) |          \
   ((uint32_t)(uint8_t)(c) << 16) | ((uint32_t)(uint8_t)(d) << 24))

#define ELML_CHUNK_CONF ELML_FOURCC('C', 'O', 'N', 'F')
#define ELML_CHUNK_PARM ELML_FOURCC('P', 'A', 'R', 'M')
#define ELML_CHUNK_QNT ELML_FOURCC('Q', 'N', 'T', ' ')
#define ELML_CHUNK_NORM ELML_FOURCC('N', 'O', 'R', 'M')
#define ELML_CHUNK_META ELML_FOURCC('M', 'E', 'T', 'A')
#define ELML_CHUNK_TOPO ELML_FOURCC('T', 'O', 'P', 'O')
#define ELML_CHUNK_CKPT ELML_FOURCC('C', 'K', 'P', 'T')

#define ELML_CHUNK_FLAG_REQUIRED 0x0001u
#define ELML_CHUNK_FLAG_CRITICAL 0x0002u

#define ELML_FILE_FLAG_LITTLE_ENDIAN 0x01u
#define ELML_FILE_FLAG_SEQUENTIAL 0x02u
#define ELML_FILE_FLAG_HAS_NORM 0x04u
#define ELML_FILE_FLAG_HAS_META 0x08u

#ifdef __cplusplus
}
#endif

#endif /* ELML_FORMAT_H */
