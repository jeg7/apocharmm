// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include "CharmmPSF.h"

#include <memory>
#include <string>

extern "C" apo_status apo_charmm_psf_create(apo_charmm_psf **out,
                                            const char *path) {
  const char *function_name = "apo_charmm_psf_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_charmm_psf>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_c_string(path, function_name, "PSF path"));

        std::unique_ptr<apo_charmm_psf> handle(new apo_charmm_psf());
        handle->object = std::make_shared<CharmmPSF>(std::string(path));

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_charmm_psf_destroy(apo_charmm_psf *psf) { delete psf; }

extern "C" apo_status apo_charmm_psf_get_num_atoms(size_t *num_atoms,
                                                   const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_num_atoms";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_atoms, function_name, "num_atoms"));

        *num_atoms = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const int n = psf->object->getNumAtoms();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_psf_get_num_atoms: CharmmPSF returned a negative "
              "atom count");
        }

        *num_atoms = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}
