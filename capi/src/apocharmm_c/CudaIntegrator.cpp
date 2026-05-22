// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CudaIntegrator.h"
#include "apocharmm_c/detail/CharmmContextHandle.h"
#include "apocharmm_c/detail/CudaIntegratorHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/SubscriberHandle.h"
#include "apocharmm_c/detail/Validation.h"

extern "C" apo_status
apo_cuda_integrator_set_time_step(apo_cuda_integrator *integrator,
                                  const double time_step) {
  const char *function_name = "apo_cuda_integrator_set_time_step";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_integrator>(
                integrator, function_name, "CudaIntegrator"));

        integrator->object->setTimeStep(time_step);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_integrator_set_charmm_context(apo_cuda_integrator *integrator,
                                       apo_charmm_context *context) {
  const char *function_name = "apo_cuda_integrator_set_charmm_context";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_integrator>(
                integrator, function_name, "CudaIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        integrator->context = context->object;
        integrator->object->setCharmmContext(context->object);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_integrator_subscribe(apo_cuda_integrator *integrator,
                              apo_subscriber *subscriber) {
  const char *function_name = "apo_cuda_integrator_subscribe";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_integrator>(
                integrator, function_name, "CudaIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_subscriber>(
                subscriber, function_name, "Subscriber"));

        integrator->object->subscribe(subscriber->object);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_integrator_propagate(apo_cuda_integrator *integrator,
                              const int num_steps) {
  const char *function_name = "apo_cuda_integrator_propagate";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_integrator>(
                integrator, function_name, "CudaIntegrator"));

        integrator->object->propagate(num_steps);

        return APO_STATUS_OK;
      },
      function_name);
}
