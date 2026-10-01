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
#include "apocharmm_c/detail/EnumConversion.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <memory>
#include <string>
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
  const char *function_name = "apo_force_manager_destroy";
  apocharmm_c::guard_destroy(
      [force_manager](void) -> void {
        delete force_manager;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status
apo_force_manager_set_box_dimensions(apo_force_manager *force_manager,
                                     const double *box_dimensions,
                                     const size_t box_dimensions_len) {
  const char *function_name = "apo_force_manager_set_box_dimensions";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            box_dimensions, function_name, "box_dimensions"));

        if (box_dimensions_len != 3) {
          return apocharmm_c::invalid_argument(
              function_name, "box_dimensions must contain exactly 3 elements");
        }

        const std::vector<double> cpp_box_dimensions = {
            box_dimensions[0], box_dimensions[1], box_dimensions[2]};

        force_manager->object->setBoxDimensions(cpp_box_dimensions);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_set_kappa(apo_force_manager *force_manager,
                            const double kappa) {
  const char *function_name = "apo_force_manager_set_kappa";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        force_manager->object->setKappa(static_cast<float>(kappa));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_set_cutoff(apo_force_manager *force_manager,
                             const double cutoff) {
  const char *function_name = "apo_force_manager_set_cutoff";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        force_manager->object->setCutoff(static_cast<float>(cutoff));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_set_ctonnb(apo_force_manager *force_manager,
                             const double ctonnb) {
  const char *function_name = "apo_force_manager_set_ctonnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        force_manager->object->setCtonnb(static_cast<float>(ctonnb));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_set_ctofnb(apo_force_manager *force_manager,
                             const double ctofnb) {
  const char *function_name = "apo_force_manager_set_ctofnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        force_manager->object->setCtofnb(static_cast<float>(ctofnb));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_set_fft_grid(apo_force_manager *force_manager,
                               const int *grid, const size_t grid_len) {
  const char *function_name = "apo_force_manager_set_fft_grid";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<int>(grid, function_name, "grid"));

        if (grid_len != 3) {
          return apocharmm_c::invalid_argument(
              function_name, "grid must contain exactly 3 elements");
        }

        force_manager->object->setFFTGrid(grid[0], grid[1], grid[2]);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_set_pme_spline_order(apo_force_manager *force_manager,
                                       const int order) {
  const char *function_name = "apo_force_manager_set_pme_spline_order";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        force_manager->object->setPmeSplineOrder(order);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_force_manager_set_periodic_boundary_condition(
    apo_force_manager *force_manager, const apo_pbc pbc) {
  const char *function_name =
      "apo_force_manager_set_periodic_boundary_condition";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        PBC cpp_pbc = PBC::UNSET;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_pbc(&cpp_pbc, pbc, function_name));

        force_manager->object->setPeriodicBoundaryCondition(cpp_pbc);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_set_vdw_type(apo_force_manager *force_manager,
                               const int vdw_type) {
  const char *function_name = "apo_force_manager_set_vdw_type";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        force_manager->object->setVdwType(vdw_type);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_force_manager_set_print_energy_decomposition(
    apo_force_manager *force_manager, const bool flag) {
  const char *function_name =
      "apo_force_manager_set_print_energy_decomposition";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        force_manager->object->setPrintEnergyDecomposition(flag);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_num_atoms(int *num_atoms,
                                const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_num_atoms";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            num_atoms, function_name, "num_atoms"));

        *num_atoms = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        const std::shared_ptr<CharmmPSF> psf = force_manager->object->getPsf();

        if (psf == nullptr) {
          return apocharmm_c::set_last_error(APO_STATUS_NOT_INITIALIZED,
                                             function_name,
                                             "CharmmPSF is not set");
        }

        *num_atoms = psf->getNumAtoms();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_is_initialized(bool *flag,
                                 const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_is_initialized";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<bool>(flag, function_name, "flag"));

        *flag = force_manager->object->isInitialized();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_box_dimensions(double *box_dimensions,
                                     const size_t box_dimensions_len,
                                     const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_box_dimensions";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        const std::vector<double> box_dims =
            force_manager->object->getBoxDimensions();

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            box_dimensions, box_dimensions_len, 3, function_name,
            "Box dimension buffer"));

        for (size_t i = 0; i < 3; i++)
          box_dimensions[i] = box_dims[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_kappa(double *kappa,
                            const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_kappa";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            kappa, function_name, "kappa"));

        *kappa = static_cast<double>(force_manager->object->getKappa());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_cutoff(double *cutoff,
                             const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_cutoff";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            cutoff, function_name, "cutoff"));

        *cutoff = static_cast<double>(force_manager->object->getCutoff());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_ctonnb(double *ctonnb,
                             const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_ctonnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            ctonnb, function_name, "ctonnb"));

        *ctonnb = static_cast<double>(force_manager->object->getCtonnb());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_ctofnb(double *ctofnb,
                             const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_ctofnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            ctofnb, function_name, "ctofnb"));

        *ctofnb = static_cast<double>(force_manager->object->getCtofnb());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_fft_grid(int *grid, const size_t grid_len,
                               const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_fft_grid";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        const std::vector<int> fft_grid = force_manager->object->getFFTGrid();

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<int>(
            grid, grid_len, 3, function_name, "FFT grid buffer"));

        for (size_t i = 0; i < 3; i++)
          grid[i] = fft_grid[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_pme_spline_order(int *order,
                                       const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_pme_spline_order";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<int>(order, function_name, "order"));

        *order = force_manager->object->getPmeSplineOrder();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_force_manager_get_periodic_boundary_condition(
    apo_pbc *pbc, const apo_force_manager *force_manager) {
  const char *function_name =
      "apo_force_manager_get_periodic_boundary_condition";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<apo_pbc>(pbc, function_name, "pbc"));

        return apocharmm_c::from_pbc(
            pbc, force_manager->object->getPeriodicBoundaryCondition(),
            function_name);
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_get_vdw_type(int *vdw_type,
                               const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_get_vdw_type";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            vdw_type, function_name, "vdw_type"));

        *vdw_type = force_manager->object->getVdwType();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_is_composite(bool *flag,
                               const apo_force_manager *force_manager) {
  const char *function_name = "apo_force_manager_is_composite";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<bool>(flag, function_name, "flag"));

        *flag = force_manager->object->isComposite();

        return APO_STATUS_OK;
      },
      function_name);
}
