// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CUDA_NOSE_HOOVER_INTEGRATOR_H__
#define __APOCHARMM_C_CUDA_NOSE_HOOVER_INTEGRATOR_H__

#include "apocharmm_c/CharmmContext.h"
#include "apocharmm_c/CudaIntegrator.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/Subscriber.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_cuda_nose_hoover_integrator apo_cuda_nose_hoover_integrator;

APOCHARMM_C_API apo_status apo_cuda_nose_hoover_integrator_create(
    apo_cuda_nose_hoover_integrator **out, const double time_step);

APOCHARMM_C_API void apo_cuda_nose_hoover_integrator_destroy(
    apo_cuda_nose_hoover_integrator *integrator);

APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_set_reference_temperature(
    apo_cuda_nose_hoover_integrator *integrator, const double temperature);

APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass(
    apo_cuda_nose_hoover_integrator *integrator, const double mass);

APOCHARMM_C_API apo_status apo_cuda_nose_hoover_integrator_use_old_temperature(
    apo_cuda_nose_hoover_integrator *integrator, const bool flag);

APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_reset_average_temperature(
    apo_cuda_nose_hoover_integrator *integrator);

APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_get_reference_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator);

APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_get_average_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator);

APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_get_instantaneous_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator);

APOCHARMM_C_API apo_status apo_cuda_nose_hoover_integrator_as_cuda_integrator(
    apo_cuda_integrator **out, apo_cuda_nose_hoover_integrator *integrator);

#ifdef __cplusplus
}
#endif

#endif
