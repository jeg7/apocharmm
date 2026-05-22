// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_FORCE_MANAGER_H__
#define __APOCHARMM_C_FORCE_MANAGER_H__

#include "apocharmm_c/CharmmParameters.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_force_manager apo_force_manager;

APOCHARMM_C_API apo_status
apo_force_manager_create(apo_force_manager **out, const apo_charmm_psf *psf,
                         const apo_charmm_parameters *parameters);

APOCHARMM_C_API void
apo_force_manager_destroy(apo_force_manager *force_manager);

APOCHARMM_C_API apo_status apo_force_manager_set_box_dimensions(
    apo_force_manager *force_manager, const double x, const double y,
    const double z);

APOCHARMM_C_API apo_status apo_force_manager_get_box_dimensions(
    double *xyz, const size_t xyz_len, const apo_force_manager *force_manager);

#ifdef __cplusplus
}
#endif

#endif
