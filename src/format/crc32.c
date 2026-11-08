#include "format/crc32.h"

#include <stddef.h>
#include <stdint.h>

uint32_t elml_crc32(const void *data, size_t len, uint32_t seed)
{
  const uint8_t *bytes = (const uint8_t *)data;
  uint32_t crc = seed ^ 0xFFFFFFFFu;

  for (size_t i = 0u; i < len; ++i) {
    crc ^= bytes[i];
    for (unsigned bit = 0u; bit < 8u; ++bit) {
      const uint32_t mask = (uint32_t)(0u - (crc & 1u));
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }

  return crc ^ 0xFFFFFFFFu;
}
