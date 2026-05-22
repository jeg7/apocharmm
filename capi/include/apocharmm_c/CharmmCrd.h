// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CHARMM_CRD_H__
#define __APOCHARMM_C_CHARMM_CRD_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_charmm_crd apo_charmm_crd;

APOCHARMM_C_API apo_status apo_charmm_crd_create(apo_charmm_crd **out,
                                                 const char *path);

APOCHARMM_C_API void apo_charmm_crd_destroy(apo_charmm_crd *crd);

APOCHARMM_C_API apo_status
apo_charmm_crd_get_num_atoms(size_t *num_atoms, const apo_charmm_crd *crd);

APOCHARMM_C_API apo_status apo_charmm_crd_get_coordinates(
    double *xyz, const size_t xyz_len, const apo_charmm_crd *crd);

#ifdef __cplusplus
}
#endif

#endif
