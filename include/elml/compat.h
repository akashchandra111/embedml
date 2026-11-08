/*
 * C23 -> C11 compatibility shim.
 *
 * EmbedLML targets C23. Vendor MCU toolchains occasionally lag behind, so the
 * public headers remain usable with C11 by providing the handful of spellings
 * that the implementation relies on. Do not add feature logic here.
 */
#ifndef ELML_COMPAT_H
#define ELML_COMPAT_H

#include <stddef.h>

#if !defined(__cplusplus) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L)
#  ifndef nullptr
#    define nullptr NULL
#  endif
#  ifndef static_assert
#    define static_assert _Static_assert
#  endif
#  ifndef alignas
#    define alignas _Alignas
#  endif
#  ifndef alignof
#    define alignof _Alignof
#  endif
#  ifndef constexpr
#    define constexpr const
#  endif
#endif

#define ELML_ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

#endif /* ELML_COMPAT_H */
