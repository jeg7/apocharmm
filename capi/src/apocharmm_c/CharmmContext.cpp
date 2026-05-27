// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CharmmContext.h"
#include "apocharmm_c/detail/CharmmContextHandle.h"
#include "apocharmm_c/detail/CharmmCrdHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <memory>

extern "C" apo_status
apo_charmm_context_create(apo_charmm_context **out,
                          const apo_force_manager *force_manager) {
  const char *function_name = "apo_charmm_context_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_charmm_context>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        std::unique_ptr<apo_charmm_context> handle(new apo_charmm_context());
        handle->force_manager = force_manager->object;
        handle->object = std::make_shared<CharmmContext>(handle->force_manager);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_charmm_context_destroy(apo_charmm_context *context) {
  delete context;
  return;
}

extern "C" apo_status
apo_charmm_context_set_coordinates(apo_charmm_context *context,
                                   const apo_charmm_crd *crd) {
  const char *function_name = "apo_charmm_context_set_coordinates";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_crd>(
                crd, function_name, "CharmmCrd"));

        context->object->setCoordinates(crd->object);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_random_seed_for_velocities(apo_charmm_context *context,
                                                  const uint64_t seed) {
  const char *function_name =
      "apo_charmm_context_set_random_seed_for_velocities";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setRandomSeedForVelocities(seed);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_use_holonomic_constraints(
    apo_charmm_context *context, const bool useHolonomicConstraints) {
  const char *function_name = "apo_charmm_context_use_holonomic_constraints";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->useHolonomicConstraints(useHolonomicConstraints);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_assign_velocities_at_temperature(apo_charmm_context *context,
                                                    const double temperature) {
  const char *function_name =
      "apo_charmm_context_assign_velocities_at_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->assignVelocitiesAtTemperature(
            static_cast<float>(temperature));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_compute_temperature(double *temperature,
                                       apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_compute_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            temperature, function_name, "temperature"));

        const float t = context->object->computeTemperature();

        *temperature = static_cast<double>(t);

        return APO_STATUS_OK;
      },
      function_name);
}
