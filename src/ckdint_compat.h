/*
 * Overflow-checked integer helpers.
 *
 * C23 provides <stdckdint.h>; older toolchains get equivalent builtins. Every
 * size calculation in the parser goes through these macros.
 */
#ifndef ELML_CKDINT_COMPAT_H
#define ELML_CKDINT_COMPAT_H

#if defined(__has_include)
#  if __has_include(<stdckdint.h>)
#    include <stdckdint.h>
#    define ELML_HAVE_STDCKDINT 1
#  endif
#endif

#ifndef ELML_HAVE_STDCKDINT
#  define ckd_add(result, a, b) __builtin_add_overflow((a), (b), (result))
#  define ckd_sub(result, a, b) __builtin_sub_overflow((a), (b), (result))
#  define ckd_mul(result, a, b) __builtin_mul_overflow((a), (b), (result))
#endif

#endif /* ELML_CKDINT_COMPAT_H */
