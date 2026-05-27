// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CUDA_INTEGRATOR_H__
#define __APOCHARMM_C_CUDA_INTEGRATOR_H__

#include "apocharmm_c/CharmmContext.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/Subscriber.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_cuda_integrator apo_cuda_integrator;

APOCHARMM_C_API apo_status apo_cuda_integrator_set_time_step(
    apo_cuda_integrator *integrator, const double time_step);

APOCHARMM_C_API apo_status apo_cuda_integrator_set_charmm_context(
    apo_cuda_integrator *integrator, apo_charmm_context *context);

APOCHARMM_C_API apo_status apo_cuda_integrator_subscribe(
    apo_cuda_integrator *integrator, apo_subscriber *subscriber);

APOCHARMM_C_API apo_status apo_cuda_integrator_propagate(
    apo_cuda_integrator *integrator, const int num_steps);

APOCHARMM_C_API apo_status apo_cuda_integrator_initialize_from_restart_file(
    apo_cuda_integrator *integrator, const char *path);

#ifdef __cplusplus
}
#endif

#endif
