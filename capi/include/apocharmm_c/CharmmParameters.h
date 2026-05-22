// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CHARMM_PARAMETERS_H__
#define __APOCHARMM_C_CHARMM_PARAMETERS_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_charmm_parameters apo_charmm_parameters;

APOCHARMM_C_API apo_status
apo_charmm_parameters_create(apo_charmm_parameters **out, const char *path);

APOCHARMM_C_API void
apo_charmm_parameters_destroy(apo_charmm_parameters *parameters);

#ifdef __cplusplus
}
#endif

#endif
