// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CharmmParameters.h"
#include "apocharmm_c/detail/CharmmParametersHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include "CharmmParameters.h"

#include <memory>
#include <string>

extern "C" apo_status apo_charmm_parameters_create(apo_charmm_parameters **out,
                                                   const char *path) {
  const char *function_name = "apo_charmm_parameters_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_charmm_parameters>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_c_string(path, function_name, "path"));

        std::unique_ptr<apo_charmm_parameters> handle(
            new apo_charmm_parameters());
        handle->object = std::make_shared<CharmmParameters>(std::string(path));

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_parameters_create_from_files(apo_charmm_parameters **out,
                                        const char *const *paths,
                                        const size_t num_paths) {
  const char *function_name = "apo_charmm_parameters_create_from_files";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_charmm_parameters>(
                out, function_name, "out"));

        std::vector<std::string> fnames;
        for (size_t i = 0; i < num_paths; i++) {
          APOCHARMM_C_RETURN_IF_ERROR(
              apocharmm_c::require_c_string(paths[i], function_name, "path"));
          fnames.push_back(std::string(paths[i]));
        }

        std::unique_ptr<apo_charmm_parameters> handle(
            new apo_charmm_parameters());
        handle->object = std::make_shared<CharmmParameters>(fnames);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void
apo_charmm_parameters_destroy(apo_charmm_parameters *parameters) {
  delete parameters;
}
