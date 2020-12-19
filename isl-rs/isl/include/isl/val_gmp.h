#ifndef ISL_VAL_GMP_H
#define ISL_VAL_GMP_H

#include <gmp.h>
#include <isl/val.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_val *isl_val_int_from_gmp(__isl_keep isl_ctx *ctx, __isl_keep mpz_t z);
__isl_give isl_val *isl_val_from_gmp(__isl_keep isl_ctx *ctx,
	__isl_keep const mpz_t n, __isl_keep const mpz_t d);
int isl_val_get_num_gmp(__isl_keep isl_val *v, __isl_keep mpz_t z);
int isl_val_get_den_gmp(__isl_keep isl_val *v, __isl_keep mpz_t z);

#if defined(__cplusplus)
}
#endif

#endif
