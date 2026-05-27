// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/detail/CharmmParametersHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <memory>
#include <vector>

extern "C" apo_status
apo_force_manager_create(apo_force_manager **out, const apo_charmm_psf *psf,
                         const apo_charmm_parameters *parameters) {
  const char *function_name = "apo_force_manager_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_force_manager>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf handle"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_parameters>(
                parameters, function_name, "CharmmParameters handle"));

        std::unique_ptr<apo_force_manager> handle(new apo_force_manager());
        handle->psf = psf->object;
        handle->parameters = parameters->object;
        handle->object =
            std::make_shared<ForceManager>(handle->psf, handle->parameters);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_force_manager_destroy(apo_force_manager *force_manager) {
  delete force_manager;
  return;
}

extern "C" apo_status
apo_force_manager_set_box_dimensions(apo_force_manager *force_manager,
                                     const double x, const double y,
                                     const double z) {
  const char *function_name = "apo_force_manager_set_box_dimensions";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager handle"));

        const std::vector<double> box_dimensions = {x, y, z};

        force_manager->object->setBoxDimensions(box_dimensions);

        return APO_STATUS_OK;
      },
      function_name);
}
