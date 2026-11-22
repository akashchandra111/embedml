/*
 * Library version. These numbers are the single source of truth: CMake parses
 * them to set project(embedml VERSION ...), and the writer metadata plus
 * elml_version_string() stringify them, so the values cannot drift.
 */
#ifndef ELML_VERSION_H
#define ELML_VERSION_H

#define ELML_VERSION_MAJOR 0
#define ELML_VERSION_MINOR 1
#define ELML_VERSION_PATCH 0

#define ELML_VERSION_STRINGIFY_(value) #value
#define ELML_VERSION_STRINGIFY(value) ELML_VERSION_STRINGIFY_(value)
#define ELML_VERSION_STRING                                     \
  ELML_VERSION_STRINGIFY(ELML_VERSION_MAJOR) "."                \
  ELML_VERSION_STRINGIFY(ELML_VERSION_MINOR) "."                \
  ELML_VERSION_STRINGIFY(ELML_VERSION_PATCH)

#endif /* ELML_VERSION_H */
