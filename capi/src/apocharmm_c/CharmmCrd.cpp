// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CharmmCrd.h"
#include "apocharmm_c/detail/CharmmCrdHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include "CharmmCrd.h"

#include <memory>
#include <string>

extern "C" apo_status apo_charmm_crd_create(apo_charmm_crd **out,
                                            const char *path) {
  const char *function_name = "apo_charmm_crd_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_charmm_crd>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_c_string(path, function_name, "path"));

        std::unique_ptr<apo_charmm_crd> handle(new apo_charmm_crd());
        handle->object = std::make_shared<CharmmCrd>(std::string(path));

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_charmm_crd_destroy(apo_charmm_crd *crd) { delete crd; }

extern "C" apo_status apo_charmm_crd_get_num_atoms(size_t *num_atoms,
                                                   const apo_charmm_crd *crd) {
  const char *function_name = "apo_charmm_crd_get_num_atoms";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_atoms, function_name, "num_atoms"));

        *num_atoms = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_crd>(
                crd, function_name, "CharmmCrd"));

        const int n = crd->object->getNumAtoms();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_crd_get_num_atoms: CharmmCrd returned a negative "
              "atom count");
        }

        *num_atoms = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_crd_get_coordinates(double *xyz, const size_t xyz_len,
                               const apo_charmm_crd *crd) {
  const char *function_name = "apo_charmm_crd_get_coordinates";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_crd>(
                crd, function_name, "CharmmCrd"));

        const std::vector<double4> &coordinates =
            crd->object->getCoordinatesD();
        const size_t num_atoms = coordinates.size();
        const size_t required_len = 3 * num_atoms;

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            xyz, xyz_len, required_len, function_name,
            "Coordinate output buffer"));

        for (size_t i = 0; i < num_atoms; i++) {
          xyz[i * 3 + 0] = coordinates[i].x;
          xyz[i * 3 + 1] = coordinates[i].y;
          xyz[i * 3 + 2] = coordinates[i].z;
        }

        return APO_STATUS_OK;
      },
      function_name);
}
