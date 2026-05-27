// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CHARMM_CONTEXT_H__
#define __APOCHARMM_C_CHARMM_CONTEXT_H__

#include "apocharmm_c/CharmmCrd.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/Status.h"

#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_charmm_context apo_charmm_context;

APOCHARMM_C_API apo_status apo_charmm_context_create(
    apo_charmm_context **out, const apo_force_manager *force_manager);

APOCHARMM_C_API void apo_charmm_context_destroy(apo_charmm_context *context);

APOCHARMM_C_API apo_status apo_charmm_context_set_coordinates(
    apo_charmm_context *context, const apo_charmm_crd *crd);

APOCHARMM_C_API apo_status apo_charmm_context_set_random_seed_for_velocities(
    apo_charmm_context *context, const uint64_t seed);

APOCHARMM_C_API
apo_status apo_charmm_context_use_holonomic_constraints(
    apo_charmm_context *context, const bool useHolonomicConstraints);

APOCHARMM_C_API apo_status apo_charmm_context_assign_velocities_at_temperature(
    apo_charmm_context *context, const double temperature);

APOCHARMM_C_API apo_status apo_charmm_context_compute_temperature(
    double *temperature, apo_charmm_context *context);

#ifdef __cplusplus
}
#endif

#endif
