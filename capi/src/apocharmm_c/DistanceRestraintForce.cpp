// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/DistanceRestraintForce.h"
#include "apocharmm_c/detail/DistanceRestraintForceHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

extern "C" apo_status
apo_distance_restraint_force_create(apo_distance_restraint_force **out,
                                    const int num_atoms) {
  const char *function_name = "apo_distance_restraint_force_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_distance_restraint_force>(
                out, function_name, "out"));

        std::unique_ptr<apo_distance_restraint_force> handle(
            new apo_distance_restraint_force());

        handle->object =
            std::make_shared<DistanceRestraintForce<long long int, float>>(
                num_atoms);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void
apo_distance_restraint_force_destroy(apo_distance_restraint_force *restraint) {
  const char *function_name = "apo_distance_restraint_force_destroy";
  apocharmm_c::guard_destroy(
      [restraint](void) -> void {
        delete restraint;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status
apo_distance_restraint_force_set_scale(apo_distance_restraint_force *restraint,
                                       const double scale) {
  const char *function_name = "apo_distance_restraint_force_set_scale";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_distance_restraint_force>(
                restraint, function_name, "DistanceRestraintForce"));

        restraint->object->setScale(scale);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_distance_restraint_force_add_restraint(
    apo_distance_restraint_force *restraint, const int *first_atom_indices,
    const size_t first_atom_indices_len, const int *second_atom_indices,
    const size_t second_atom_indices_len, const double *coefficients,
    const size_t coefficients_len, const double force_constant,
    const double reference_value, const int distance_exponent,
    const int energy_exponent,
    const apo_distance_restraint_condition condition) {
  const char *function_name = "apo_distance_restraint_force_add_restraint";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_distance_restraint_force>(
                restraint, function_name, "DistanceRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            first_atom_indices, function_name, "first_atom_indices"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            second_atom_indices, function_name, "second_atom_indices"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            coefficients, function_name, "coefficients"));

        if (first_atom_indices_len != second_atom_indices_len) {
          return apocharmm_c::invalid_argument(
              function_name, "first_atom_indices and second_atom_indices must "
                             "have matching lengths");
        }

        std::vector<std::array<int, 2>> cpp_atom_pairs(first_atom_indices_len);
        for (size_t i = 0; i < first_atom_indices_len; i++)
          cpp_atom_pairs[i] = {first_atom_indices[i], second_atom_indices[i]};

        std::vector<double> cpp_coefficients(coefficients_len);
        for (size_t i = 0; i < coefficients_len; i++)
          cpp_coefficients[i] = coefficients[i];

        DistanceRestraintCondition cpp_condition =
            DistanceRestraintCondition::NONE;
        switch (condition) {
        case APO_DISTANCE_RESTRAINT_CONDITION_NONE:
          cpp_condition = DistanceRestraintCondition::NONE;
          break;
        case APO_DISTANCE_RESTRAINT_CONDITION_POSITIVE:
          cpp_condition = DistanceRestraintCondition::POSITIVE;
          break;
        case APO_DISTANCE_RESTRAINT_CONDITION_NEGATIVE:
          cpp_condition = DistanceRestraintCondition::NEGATIVE;
          break;
        default:
          return apocharmm_c::invalid_argument(
              function_name, "condition is not a declared "
                             "apo_distance_restraint_condition value");
        }

        restraint->object->addRestraint(
            cpp_atom_pairs, cpp_coefficients, force_constant, reference_value,
            distance_exponent, energy_exponent, cpp_condition);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_distance_restraint_force_reset(apo_distance_restraint_force *restraint) {
  const char *function_name = "apo_distance_restraint_force_reset";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_distance_restraint_force>(
                restraint, function_name, "DistanceRestraintForce"));

        restraint->object->reset();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_force_manager_subscribe_distance_restraint_force(
    apo_force_manager *force_manager, apo_distance_restraint_force *restraint,
    const char *force_tag) {
  const char *function_name =
      "apo_force_manager_subscribe_distance_restraint_force";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_distance_restraint_force>(
                restraint, function_name, "DistanceRestraintForce"));

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

extern "C" apo_status apo_force_manager_unsubscribe_distance_restraint_force(
    apo_force_manager *force_manager, apo_distance_restraint_force *restraint) {
  const char *function_name =
      "apo_force_manager_unsubscribe_distance_restraint_force";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_force_manager>(
                force_manager, function_name, "ForceManager"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_distance_restraint_force>(
                restraint, function_name, "DistanceRestraintForce"));

        force_manager->object->unsubscribe(restraint->object);

        return APO_STATUS_OK;
      },
      function_name);
}
