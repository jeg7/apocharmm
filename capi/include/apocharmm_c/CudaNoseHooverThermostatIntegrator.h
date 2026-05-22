// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CUDA_NOSE_HOOVER_THERMOSTAT_INTEGRATOR_H__
#define __APOCHARMM_C_CUDA_NOSE_HOOVER_THERMOSTAT_INTEGRATOR_H__

#include "apocharmm_c/CharmmContext.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#ifdef __cplusplus
extern "C" {
#endif

// cnht --> cuda_nose_hoover_thermostat

typedef struct apo_cnht_integrator apo_cnht_integrator;

APOCHARMM_C_API apo_status apo_cnht_integrator_create(apo_cnht_integrator **out,
                                                      const double time_step);

APOCHARMM_C_API void
apo_cnht_integrator_destroy(apo_cnht_integrator *integrator);

/////////////////
//// Setters ////
////////////////////////////////////////////////////////////////////////
APOCHARMM_C_API apo_status apo_cnht_integrator_set_time_step(
    apo_cnht_integrator *integrator, const double time_step);
APOCHARMM_C_API apo_status apo_cnht_integrator_set_charmm_context(
    apo_cnht_integrator *integrator, const apo_charmm_context *context);
APOCHARMM_C_API apo_status apo_cnht_integrator_set_reference_temperature(
    apo_cnht_integrator *integrator, const double temperature);
APOCHARMM_C_API apo_status apo_cnht_integrator_set_nose_hoover_piston_mass(
    apo_cnht_integrator *integrator, const double mass);
APOCHARMM_C_API apo_status apo_cnht_integrator_use_old_temperature(
    apo_cnht_integrator *integrator, const bool flag);
APOCHARMM_C_API apo_status
apo_cnht_integrator_reset_average_temperature(apo_cnht_integrator *integrator);
////////////////////////////////////////////////////////////////////////

/////////////////
//// Getters ////
////////////////////////////////////////////////////////////////////////
APOCHARMM_C_API apo_status apo_cnht_integrator_get_reference_temperature(
    double *temperature, const apo_cnht_integrator *integrator);
APOCHARMM_C_API apo_status apo_cnht_integrator_get_average_temperature(
    double *temperature, const apo_cnht_integrator *integrator);
APOCHARMM_C_API apo_status apo_cnht_integrator_get_instantaneous_temperature(
    double *temperature, const apo_cnht_integrator *integrator);
////////////////////////////////////////////////////////////////////////

///////////////////
//// Functions ////
////////////////////////////////////////////////////////////////////////
APOCHARMM_C_API apo_status apo_cnht_integrator_propagate(
    apo_cnht_integrator *integrator, const int num_steps);
// APOCHARMM_C_API apo_status apo_cnht_integrator_subscribe(
//     apo_cnht_integrator *integrator, const apo_dcd_subscriber *dcd);
////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif
