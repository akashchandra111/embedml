/*
 * Explicit little-endian encoding helpers.
 *
 * Files are always little-endian. No struct is ever cast onto the byte stream.
 */
#ifndef ELML_ENDIAN_H
#define ELML_ENDIAN_H

#include <stdint.h>
#include <string.h>

static inline uint16_t elml_le16(const uint8_t *p)
{
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint32_t elml_le32(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static inline uint64_t elml_le64(const uint8_t *p)
{
  return (uint64_t)elml_le32(p) | ((uint64_t)elml_le32(p + 4) << 32);
}

static inline float elml_le_f32(const uint8_t *p)
{
  uint32_t bits = elml_le32(p);
  float value;
  memcpy(&value, &bits, sizeof value);
  return value;
}

static inline void elml_store_le16(uint8_t *p, uint16_t value)
{
  p[0] = (uint8_t)(value & 0xFFu);
  p[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static inline void elml_store_le32(uint8_t *p, uint32_t value)
{
  p[0] = (uint8_t)(value & 0xFFu);
  p[1] = (uint8_t)((value >> 8) & 0xFFu);
  p[2] = (uint8_t)((value >> 16) & 0xFFu);
  p[3] = (uint8_t)((value >> 24) & 0xFFu);
}

static inline void elml_store_le64(uint8_t *p, uint64_t value)
{
  elml_store_le32(p, (uint32_t)(value & 0xFFFFFFFFu));
  elml_store_le32(p + 4, (uint32_t)(value >> 32));
}

static inline void elml_store_le_f32(uint8_t *p, float value)
{
  uint32_t bits;
  memcpy(&bits, &value, sizeof bits);
  elml_store_le32(p, bits);
}

#endif /* ELML_ENDIAN_H */
