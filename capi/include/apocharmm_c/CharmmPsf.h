// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CHARMM_PSF_H__
#define __APOCHARMM_C_CHARMM_PSF_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_charmm_psf apo_charmm_psf;

APOCHARMM_C_API apo_status apo_charmm_psf_create(apo_charmm_psf **out,
                                                 const char *path);

APOCHARMM_C_API void apo_charmm_psf_destroy(apo_charmm_psf *psf);

APOCHARMM_C_API apo_status
apo_charmm_psf_get_num_atoms(size_t *num_atoms, const apo_charmm_psf *psf);

#ifdef __cplusplus
}
#endif

#endif
