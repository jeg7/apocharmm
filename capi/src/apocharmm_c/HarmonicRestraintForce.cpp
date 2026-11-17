// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/HarmonicRestraintForce.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/HarmonicRestraintForceHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

extern "C" apo_status
apo_harmonic_restraint_force_create(apo_harmonic_restraint_force **out,
                                    const apo_charmm_psf *psf) {
  const char *function_name = "apo_harmonic_restraint_force_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_harmonic_restraint_force>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        std::unique_ptr<apo_harmonic_restraint_force> handle(
            new apo_harmonic_restraint_force());

        handle->object =
            std::make_shared<HarmonicRestraintForce<long long int, float>>(
                psf->object);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void
apo_harmonic_restraint_force_destroy(apo_harmonic_restraint_force *restraint) {
  const char *function_name = "apo_harmonic_restraint_force_destroy";
  apocharmm_c::guard_destroy(
      [restraint](void) -> void {
        delete restraint;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status apo_harmonic_restraint_force_set_selection(
    apo_harmonic_restraint_force *restraint,
    const apo_atom_selection *selection) {
  const char *function_name = "apo_harmonic_restraint_force_set_selection";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_selection>(
                selection, function_name, "AtomSelection"));

        restraint->object->setSelection(*selection->object);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_harmonic_restraint_force_set_force_constant(
    apo_harmonic_restraint_force *restraint, const double force_constant) {
  const char *function_name = "apo_harmonic_restraint_force_set_force_constant";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        restraint->object->setForceConstant(force_constant);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_harmonic_restraint_force_set_force_constants(
    apo_harmonic_restraint_force *restraint, const double *force_constants,
    const size_t force_constants_len) {
  const char *function_name =
      "apo_harmonic_restraint_force_set_force_constants";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            force_constants, function_name, "force_constants"));

        std::vector<double> cpp_force_constants(force_constants_len);
        for (size_t i = 0; i < force_constants_len; i++)
          cpp_force_constants[i] = force_constants[i];

        restraint->object->setForceConstants(cpp_force_constants);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_harmonic_restraint_force_set_reference_coordinates(
    apo_harmonic_restraint_force *restraint,
    const double *reference_coordinates,
    const size_t reference_coordinates_len) {
  const char *function_name =
      "apo_harmonic_restraint_force_set_reference_coordinates";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            reference_coordinates, function_name, "reference_coordinates"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_flat_array_length(
            reference_coordinates_len, 3, "reference_coordinates",
            function_name));

        const size_t num_coordiantes = reference_coordinates_len / 3;

        std::vector<std::vector<double>> cpp_reference_coordinates(
            num_coordiantes, std::vector<double>(3));
        for (size_t i = 0; i < num_coordiantes; i++) {
          for (size_t j = 0; j < 3; j++)
            cpp_reference_coordinates[i][j] = reference_coordinates[i * 3 + j];
        }

        restraint->object->setReferenceCoordinates(cpp_reference_coordinates);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_harmonic_restraint_force_set_masses(apo_harmonic_restraint_force *restraint,
                                        const double *masses,
                                        const size_t masses_len) {
  const char *function_name = "apo_harmonic_restraint_force_set_masses";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            masses, function_name, "masses"));

        std::vector<double> cpp_masses(masses_len);
        for (size_t i = 0; i < masses_len; i++)
          cpp_masses[i] = masses[i];

        restraint->object->setMasses(cpp_masses);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_harmonic_restraint_force_set_exponent(
    apo_harmonic_restraint_force *restraint, const int exponent) {
  const char *function_name = "apo_harmonic_restraint_force_set_exponent";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        restraint->object->setExponent(exponent);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_harmonic_restraint_force_set_scale_factors(
    apo_harmonic_restraint_force *restraint, const double *scale_factors,
    const size_t scale_factors_len) {
  const char *function_name = "apo_harmonic_restraint_force_set_scale_factors";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            scale_factors, function_name, "scale_factors"));

        std::vector<double> cpp_scale_factors(scale_factors_len);
        for (size_t i = 0; i < scale_factors_len; i++)
          cpp_scale_factors[i] = scale_factors[i];

        restraint->object->setScaleFactors(cpp_scale_factors);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_harmonic_restraint_force_set_box_dimensions(
    apo_harmonic_restraint_force *restraint, const double *box_dimensions,
    const size_t box_dimensions_len) {
  const char *function_name = "apo_harmonic_restraint_force_set_box_dimensions";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            box_dimensions, function_name, "box_dimensions"));

        std::vector<double> cpp_box_dimensions(box_dimensions_len);
        for (size_t i = 0; i < box_dimensions_len; i++)
          cpp_box_dimensions[i] = box_dimensions[i];

        restraint->object->setBoxDimensions(cpp_box_dimensions);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_force_manager_subscribe_harmonic_restraint_force(
    apo_force_manager *force_manager, apo_harmonic_restraint_force *restraint,
    const char *force_tag) {
  const char *function_name =
      "apo_force_manager_subscribe_harmonic_restraint_force";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<char>(
            force_tag, function_name, "force_tag"));

        force_manager->object->subscribe(
            restraint->object, std::string(force_tag),
            restraint->object->getStream(), restraint->object->getForce(),
            restraint->object->getEnergyVirial());

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_force_manager_unsubscribe_harmonic_restraint_force(
    apo_force_manager *force_manager, apo_harmonic_restraint_force *restraint) {
  const char *function_name =
      "apo_force_manager_unsubscribe_harmonic_restraint_force";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_harmonic_restraint_force>(
                restraint, function_name, "HarmonicRestraintForce"));

        force_manager->object->unsubscribe(restraint->object);

        return APO_STATUS_OK;
      },
      function_name);
}
