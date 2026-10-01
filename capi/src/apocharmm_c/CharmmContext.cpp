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

extern "C" apo_status apo_charmm_context_create_from_psf_parameters(
    apo_charmm_context **out, const apo_charmm_psf *psf,
    const apo_charmm_parameters *parameters) {
  const char *function_name = "apo_charmm_context_create_from_psf_parameters";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_charmm_context>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_parameters>(
                parameters, function_name, "CharmmParameters"));

        std::unique_ptr<apo_charmm_context> handle(new apo_charmm_context());
        handle->psf = psf->object;
        handle->parameters = parameters->object;
        handle->object =
            std::make_shared<CharmmContext>(handle->psf, handle->parameters);
        handle->force_manager = handle->object->getForceManager();

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_charmm_context_destroy(apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_destroy";
  apocharmm_c::guard_destroy(
      [context](void) -> void {
        delete context;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status
apo_charmm_context_set_prm(apo_charmm_context *context,
                           apo_charmm_parameters *parameters) {
  const char *function_name = "apo_charmm_context_set_prm";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_parameters>(
                parameters, function_name, "CharmmParameters"));

        context->object->setPrm(parameters->object);
        context->parameters = context->object->getPrm();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_psf(apo_charmm_context *context,
                                                 apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_context_set_psf";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        context->object->setPsf(psf->object);
        context->psf = context->object->getPsf();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_force_manager(apo_charmm_context *context,
                                     apo_force_manager *force_manager) {
  const char *function_name = "apo_charmm_context_set_force_manager";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        context->object->setForceManager(force_manager->object);
        context->force_manager = context->object->getForceManager();
        context->psf = context->object->getPsf();
        context->parameters = context->object->getPrm();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_coordinates_charges(
    apo_charmm_context *context, const double *xyzq, const size_t xyzq_len) {
  const char *function_name = "apo_charmm_context_set_coordinates_charges";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<double>(xyzq, function_name, "xyzq"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_flat_array_length(
            xyzq_len, 4, "xyzq", function_name));

        const std::vector<double> cpp_xyzq(xyzq, xyzq + xyzq_len);
        context->object->setCoordinatesCharges(cpp_xyzq);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_coordinates_from_array(
    apo_charmm_context *context, const double *xyz, const size_t xyz_len) {
  const char *function_name = "apo_charmm_context_set_coordinates_from_array";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<double>(xyz, function_name, "xyz"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_flat_array_length(
            xyz_len, 3, "xyz", function_name));

        const std::vector<double> cpp_xyz(xyz, xyz + xyz_len);
        context->object->setCoordinates(cpp_xyz);

        return APO_STATUS_OK;
      },
      function_name);
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
apo_charmm_context_set_charges(apo_charmm_context *context,
                               const double *charges,
                               const size_t charges_len) {
  const char *function_name = "apo_charmm_context_set_charges";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            charges, function_name, "charges"));

        const std::vector<double> cpp_charges(charges, charges + charges_len);
        context->object->setCharges(cpp_charges);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_velocities_inverse_masses(
    apo_charmm_context *context, const double *xyzm, const size_t xyzm_len) {
  const char *function_name =
      "apo_charmm_context_set_velocities_inverse_masses";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<double>(xyzm, function_name, "xyzm"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_flat_array_length(
            xyzm_len, 4, "xyzm", function_name));

        const std::vector<double> cpp_xyzm(xyzm, xyzm + xyzm_len);
        context->object->setVelocitiesInverseMasses(cpp_xyzm);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_velocities(apo_charmm_context *context,
                                  const double *xyz, const size_t xyz_len) {
  const char *function_name = "apo_charmm_context_set_velocities";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<double>(xyz, function_name, "xyz"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_flat_array_length(
            xyz_len, 3, "xyz", function_name));

        const std::vector<double> cpp_xyz(xyz, xyz + xyz_len);
        context->object->setVelocities(cpp_xyz);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_velocities_from_charmm_velocity_file(
    apo_charmm_context *context, const char *path) {
  const char *function_name =
      "apo_charmm_context_set_velocities_from_charmm_velocity_file";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_c_string(
            path, function_name, "CHARMM velocity file"));

        context->object->setVelocitiesFromCHARMMVelocityFile(std::string(path));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_masses(apo_charmm_context *context,
                                                    const double *masses,
                                                    const size_t masses_len) {
  const char *function_name = "apo_charmm_context_set_masses";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            masses, function_name, "masses"));

        const std::vector<double> cpp_masses(masses, masses + masses_len);
        context->object->setMasses(cpp_masses);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_temperature(apo_charmm_context *context,
                                   const double temperature) {
  const char *function_name = "apo_charmm_context_set_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setTemperature(temperature);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_periodic_boundary_condition(apo_charmm_context *context,
                                                   const apo_pbc pbc) {
  const char *function_name =
      "apo_charmm_context_set_periodic_boundary_condition";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        PBC cpp_pbc = PBC::UNSET;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_pbc(&cpp_pbc, pbc, function_name));

        context->object->setPeriodicBoundaryCondition(cpp_pbc);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_box_dimensions(apo_charmm_context *context,
                                      const double *box_dimensions,
                                      const size_t box_dimensions_len) {
  const char *function_name = "apo_charmm_context_set_box_dimensions";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            box_dimensions, function_name, "box_dimensions"));

        if (box_dimensions_len != 3) {
          return apocharmm_c::invalid_argument(
              function_name, "box_dimensions must contain exactly 3 elements");
        }

        const std::vector<double> cpp_box_dimensions = {
            box_dimensions[0], box_dimensions[1], box_dimensions[2]};

        context->object->setBoxDimensions(cpp_box_dimensions);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_random_seed(apo_charmm_context *context,
                                   const uint64_t seed) {
  const char *function_name = "apo_charmm_context_set_random_seed";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setRandomSeed(seed);

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

extern "C" apo_status apo_charmm_context_set_kappa(apo_charmm_context *context,
                                                   const double kappa) {
  const char *function_name = "apo_charmm_context_set_kappa";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setKappa(static_cast<float>(kappa));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_cutoff(apo_charmm_context *context,
                                                    const double cutoff) {
  const char *function_name = "apo_charmm_context_set_cutoff";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setCutoff(static_cast<float>(cutoff));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_ctonnb(apo_charmm_context *context,
                                                    const double ctonnb) {
  const char *function_name = "apo_charmm_context_set_ctonnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setCtonnb(static_cast<float>(ctonnb));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_set_ctofnb(apo_charmm_context *context,
                                                    const double ctofnb) {
  const char *function_name = "apo_charmm_context_set_ctofnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setCtofnb(static_cast<float>(ctofnb));

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_fft_grid(apo_charmm_context *context, const int *grid,
                                const size_t grid_len) {
  const char *function_name = "apo_charmm_context_set_fft_grid";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<int>(grid, function_name, "grid"));

        if (grid_len != 3) {
          return apocharmm_c::invalid_argument(
              function_name, "grid must contain exactly 3 elements");
        }

        context->object->setFFTGrid(grid[0], grid[1], grid[2]);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_pme_spline_order(apo_charmm_context *context,
                                        const int order) {
  const char *function_name = "apo_charmm_context_set_pme_spline_order";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setPmeSplineOrder(order);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_set_vdw_type(apo_charmm_context *context,
                                const int vdw_type) {
  const char *function_name = "apo_charmm_context_set_vdw_type";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->setVdwType(vdw_type);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_num_atoms(int *num_atoms,
                                 const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_num_atoms";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            num_atoms, function_name, "num_atoms"));

        *num_atoms = context->object->getNumAtoms();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_get_num_degrees_of_freedom(
    int *ndegf, const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_num_degrees_of_freedom";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<int>(ndegf, function_name, "ndegf"));

        *ndegf = context->object->getNumDegreesOfFreedom();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_coordinates_charges(double *xyzq, const size_t xyzq_len,
                                           const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_coordinates_charges";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        CudaContainer<double4> &coordinatesCharges =
            context->object->getCoordinatesChargesDP();
        const size_t num_atoms = coordinatesCharges.size();
        const size_t req_len = 4 * num_atoms;

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            xyzq, xyzq_len, req_len, function_name,
            "Coordinate and charge output buffer"));

        coordinatesCharges.transferToHost();

        for (size_t i = 0; i < num_atoms; i++) {
          xyzq[i * 4 + 0] = coordinatesCharges[i].x;
          xyzq[i * 4 + 1] = coordinatesCharges[i].y;
          xyzq[i * 4 + 2] = coordinatesCharges[i].z;
          xyzq[i * 4 + 3] = coordinatesCharges[i].w;
        }

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_velocity_mass(double *xyzm, const size_t xyzm_len,
                                     const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_velocity_mass";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        CudaContainer<double4> &velMass =
            context->object->getVelocitiesInverseMasses();
        const size_t num_atoms = velMass.size();
        const size_t req_len = 4 * num_atoms;

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            xyzm, xyzm_len, req_len, function_name,
            "Velocity and mass output buffer"));

        velMass.transferToHost();

        for (size_t i = 0; i < num_atoms; i++) {
          xyzm[i * 4 + 0] = velMass[i].x;
          xyzm[i * 4 + 1] = velMass[i].y;
          xyzm[i * 4 + 2] = velMass[i].z;
          xyzm[i * 4 + 3] = velMass[i].w;
        }

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_get_periodic_boundary_condition(
    apo_pbc *pbc, const apo_charmm_context *context) {
  const char *function_name =
      "apo_charmm_context_get_periodic_boundary_condition";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<apo_pbc>(pbc, function_name, "pbc"));

        return apocharmm_c::from_pbc(
            pbc, context->object->getPeriodicBoundaryCondition(),
            function_name);
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_box_dimensions(double *box_dimensions,
                                      const size_t box_dimensions_len,
                                      const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_box_dimensions";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        std::vector<double> box_dims = context->object->getBoxDimensions();

        if (box_dims.size() != 3) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_context_get_box_dimensions: CharmmContext did not "
              "return exactly 3 box dimensions");
        }

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
apo_charmm_context_get_random_seed(uint64_t *seed,
                                   const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_random_seed";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<uint64_t>(
            seed, function_name, "seed"));

        *seed = context->object->getRandomSeed();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_volume(double *volume,
                              const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_volume";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            volume, function_name, "volume"));

        *volume = context->object->getVolume();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_kappa(double *kappa, const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_kappa";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            kappa, function_name, "kappa"));

        *kappa = static_cast<double>(context->object->getKappa());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_cutoff(double *cutoff,
                              const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_cutoff";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            cutoff, function_name, "cutoff"));

        *cutoff = static_cast<double>(context->object->getCutoff());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_ctonnb(double *ctonnb,
                              const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_ctonnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            ctonnb, function_name, "ctonnb"));

        *ctonnb = static_cast<double>(context->object->getCtonnb());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_ctofnb(double *ctofnb,
                              const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_ctofnb";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            ctofnb, function_name, "ctofnb"));

        *ctofnb = static_cast<double>(context->object->getCtofnb());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_fft_grid(int *grid, const size_t grid_len,
                                const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_fft_grid";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        std::vector<int> fft_grid = context->object->getFFTGrid();

        if (fft_grid.size() != 3) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_context_get_fft_grid: CharmmContext did not return "
              "exactly 3 FFT grid dimensions");
        }

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<int>(
            grid, grid_len, 3, function_name, "FFT grid buffer"));

        for (size_t i = 0; i < 3; i++)
          grid[i] = fft_grid[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_pme_spline_order(int *order,
                                        const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_pme_spline_order";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<int>(order, function_name, "order"));

        *order = context->object->getPmeSplineOrder();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_vdw_type(int *vdw_type,
                                const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_vdw_type";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            vdw_type, function_name, "vdw_type"));

        *vdw_type = context->object->getVdwType();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_context_get_force_manager(apo_force_manager **out,
                                     const apo_charmm_context *context) {
  const char *function_name = "apo_charmm_context_get_force_manager";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_force_manager>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        std::shared_ptr<ForceManager> fm = context->object->getForceManager();

        if (fm == nullptr) {
          return apocharmm_c::set_last_error(APO_STATUS_NOT_INITIALIZED,
                                             function_name,
                                             "ForceManager is not set");
        }

        std::unique_ptr<apo_force_manager> handle(new apo_force_manager());
        handle->object = fm;
        handle->psf = fm->getPsf();
        handle->parameters = fm->getPrm();

        *out = handle.release();

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

        context->object->assignVelocitiesAtTemperature(temperature);

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

        *temperature = context->object->computeTemperature();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_context_calculate_potential_energy(
    apo_charmm_context *context, const bool reset, const bool print) {
  const char *function_name = "apo_charmm_context_calculate_potential_energy";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_context>(
                context, function_name, "CharmmContext"));

        context->object->calculatePotentialEnergy(reset, print);

        return APO_STATUS_OK;
      },
      function_name);
}
