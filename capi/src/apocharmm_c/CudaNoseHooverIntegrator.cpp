// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CudaNoseHooverIntegrator.h"
#include "apocharmm_c/detail/CharmmContextHandle.h"
#include "apocharmm_c/detail/CudaNoseHooverIntegratorHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/SubscriberHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <memory>

extern "C" apo_status
apo_cuda_nose_hoover_integrator_create(apo_cuda_nose_hoover_integrator **out,
                                       const double time_step) {
  const char *function_name = "apo_cuda_nose_hoover_integrator_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<
                apo_cuda_nose_hoover_integrator>(out, function_name, "out"));

        std::unique_ptr<apo_cuda_nose_hoover_integrator> handle(
            new apo_cuda_nose_hoover_integrator());
        handle->object = std::make_shared<CudaNoseHooverIntegrator>(time_step);
        handle->base.object = handle->object;

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_cuda_nose_hoover_integrator_destroy(
    apo_cuda_nose_hoover_integrator *integrator) {
  delete integrator;
  return;
}

/////////////////
//// Setters ////
////////////////////////////////////////////////////////////////////////
extern "C" apo_status apo_cuda_nose_hoover_integrator_set_reference_temperature(
    apo_cuda_nose_hoover_integrator *integrator, const double temperature) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_set_reference_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        integrator->object->setReferenceTemperature(temperature);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass(
    apo_cuda_nose_hoover_integrator *integrator, const double mass) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        integrator->object->setNoseHooverPistonMass(mass);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_cuda_nose_hoover_integrator_use_old_temperature(
    apo_cuda_nose_hoover_integrator *integrator, const bool flag) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_use_old_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        integrator->object->useOldTemperature(flag);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_cuda_nose_hoover_integrator_reset_average_temperature(
    apo_cuda_nose_hoover_integrator *integrator) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_reset_average_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        integrator->object->resetAverageTemperature();

        return APO_STATUS_OK;
      },
      function_name);
}
////////////////////////////////////////////////////////////////////////

/////////////////
//// Getters ////
////////////////////////////////////////////////////////////////////////
extern "C" apo_status apo_cuda_nose_hoover_integrator_get_reference_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_get_reference_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            temperature, function_name, "temperature"));

        *temperature = integrator->object->getReferenceTemperature();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_cuda_nose_hoover_integrator_get_average_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_get_average_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            temperature, function_name, "temperature"));

        integrator->object->getAverageTemperature().transferToHost();

        if (integrator->object->usingOldTemperature())
          *temperature = integrator->object->getAverageTemperature()[0];
        else
          *temperature = integrator->object->getAverageTemperature()[1];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_nose_hoover_integrator_get_instantaneous_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_get_instantaneous_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            temperature, function_name, "temperature"));

        *temperature = integrator->object->getInstantaneousTemperature();

        return APO_STATUS_OK;
      },
      function_name);
}
////////////////////////////////////////////////////////////////////////

///////////////////
//// Functions ////
////////////////////////////////////////////////////////////////////////
extern "C" apo_status apo_cuda_nose_hoover_integrator_as_cuda_integrator(
    apo_cuda_integrator **out, apo_cuda_nose_hoover_integrator *integrator) {
  const char *function_name =
      "apo_cuda_nose_hoover_integrator_as_cuda_integrator";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_cuda_integrator>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_cuda_nose_hoover_integrator>(
                integrator, function_name, "CudaNoseHooverIntegrator"));

        *out = &integrator->base;

        return APO_STATUS_OK;
      },
      function_name);
}
////////////////////////////////////////////////////////////////////////
