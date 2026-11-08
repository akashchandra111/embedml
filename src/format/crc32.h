#ifndef ELML_CRC32_H
#define ELML_CRC32_H

#include <stddef.h>
#include <stdint.h>

/* IEEE 802.3 CRC-32 (reflected, polynomial 0xEDB88320). */
uint32_t elml_crc32(const void *data, size_t len, uint32_t seed);

#endif /* ELML_CRC32_H */
