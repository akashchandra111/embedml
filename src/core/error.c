#include "elml/model.h"

#include "elml/types.h"

const char *elml_error_string(elml_status status)
{
  switch (status) {
  case ELML_OK:
    return "ok";
  case ELML_ERR_INVALID_ARG:
    return "invalid argument";
  case ELML_ERR_OUT_OF_MEMORY:
    return "out of memory";
  case ELML_ERR_TRUNCATED:
    return "truncated file";
  case ELML_ERR_BAD_MAGIC:
    return "bad magic";
  case ELML_ERR_UNSUPPORTED_MAJOR:
    return "unsupported format major version";
  case ELML_ERR_UNSUPPORTED_MINOR:
    return "unsupported format minor version";
  case ELML_ERR_DTYPE_MISMATCH:
    return "model dtype does not match the requested dtype";
  case ELML_ERR_UNSUPPORTED_DTYPE:
    return "dtype is not supported by this build";
  case ELML_ERR_UNSUPPORTED_LAYER_TYPE:
    return "model uses a layer type that is not linked into this build";
  case ELML_ERR_BAD_SHAPE:
    return "invalid tensor shape";
  case ELML_ERR_BAD_PARAM:
    return "invalid layer parameters";
  case ELML_ERR_CRC:
    return "checksum mismatch";
  case ELML_ERR_MISALIGNED:
    return "model buffer is not correctly aligned";
  case ELML_ERR_SCRATCH_TOO_SMALL:
    return "scratch buffer is too small";
  case ELML_ERR_UNKNOWN_CHUNK_REQUIRED:
    return "unsupported required chunk";
  case ELML_ERR_NOT_FOUND:
    return "not found";
  case ELML_ERR_BUFFER_TOO_SMALL:
    return "destination buffer is too small";
  case ELML_ERR_INTERNAL:
    return "internal error";
  default:
    return "unknown error";
  }
}

const char *elml_version_string(void)
{
  return "EmbedLML 0.1.0 (format 1.0)";
}
