// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/HarmonicCenterOfMassRestraintForce.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/HarmonicCenterOfMassRestraintForceHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

extern "C" apo_status apo_harmonic_center_of_mass_restraint_force_create(
    apo_harmonic_center_of_mass_restraint_force **out,
    const apo_charmm_psf *psf) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<
                apo_harmonic_center_of_mass_restraint_force>(out, function_name,
                                                             "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        std::unique_ptr<apo_harmonic_center_of_mass_restraint_force> handle(
            new apo_harmonic_center_of_mass_restraint_force);

        handle->object = std::make_shared<
            HarmonicCenterOfMassRestraintForce<long long int, float>>(
            psf->object);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_harmonic_center_of_mass_restraint_force_destroy(
    apo_harmonic_center_of_mass_restraint_force *restraint) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_destroy";
  apocharmm_c::guard_destroy(
      [restraint](void) -> void {
        delete restraint;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status apo_harmonic_center_of_mass_restraint_force_set_selection(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const apo_atom_selection *selection) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_set_selection";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_selection>(
                selection, function_name, "AtomSelection"));

        restraint->object->setSelection(*selection->object);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_harmonic_center_of_mass_restraint_force_set_force_constant(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double force_constant) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_set_force_constant";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

        restraint->object->setForceConstant(force_constant);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_harmonic_center_of_mass_restraint_force_set_reference_position(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double *reference_position, const size_t reference_position_len,
    const int *reference_mask, const size_t reference_mask_len) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_set_reference_position";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            reference_position, function_name, "reference_position"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            reference_mask, function_name, "reference_mask"));

        std::vector<double> ref_pos(reference_position_len);
        for (size_t i = 0; i < reference_position_len; i++)
          ref_pos[i] = reference_position[i];

        std::vector<int> ref_mask(reference_mask_len);
        for (size_t i = 0; i < reference_mask_len; i++)
          ref_mask[i] = reference_mask[i];

        restraint->object->setReferencePosition(ref_pos, ref_mask);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_harmonic_center_of_mass_restraint_force_set_reference_distance(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double reference_distance) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_set_reference_distance";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

        restraint->object->setReferenceDistance(reference_distance);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_harmonic_center_of_mass_restraint_force_set_masses(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double *masses, const size_t masses_len) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_set_masses";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

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

extern "C" apo_status
apo_harmonic_center_of_mass_restraint_force_set_mass_weighting(
    apo_harmonic_center_of_mass_restraint_force *restraint, const bool flag) {
  const char *function_name =
      "apo_harmonic_center_of_mass_restraint_force_set_mass_weighting";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

        restraint->object->setMassWeighting(flag);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
    apo_force_manager *force_manager,
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const char *force_tag) {
  const char *function_name =
      "apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

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

extern "C" apo_status
apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force(
    apo_force_manager *force_manager,
    apo_harmonic_center_of_mass_restraint_force *restraint) {
  const char *function_name =
      "apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<
                apo_harmonic_center_of_mass_restraint_force>(
                restraint, function_name,
                "HarmonicCenterOfMassRestraintForce"));

        force_manager->object->unsubscribe(restraint->object);

        return APO_STATUS_OK;
      },
      function_name);
}
