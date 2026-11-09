// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/GeometricRestraintForce.h"

#include "apocharmm_c/detail/AtomReferenceHandle.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/EnumConversion.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/GeometricRestraintForceHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

inline static apo_status copy_cartesian_vector(std::array<double, 3> *out,
                                               const double *values,
                                               const size_t values_len,
                                               const char *function_name,
                                               const char *argument_name) {
  APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
      values, function_name, argument_name));

  if (values_len != 3) {
    return apocharmm_c::invalid_argument(
        function_name,
        std::string(argument_name) + " must contain exactly 3 elements");
  }

  *out = {values[0], values[1], values[2]};

  return APO_STATUS_OK;
}

extern "C" apo_status
apo_geometric_restraint_force_create(apo_geometric_restraint_force **out,
                                     const apo_charmm_psf *psf) {
  const char *function_name = "apo_geometric_restraint_force_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_geometric_restraint_force>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        std::unique_ptr<apo_geometric_restraint_force> handle(
            new apo_geometric_restraint_force());

        handle->object =
            std::make_shared<GeometricRestraintForce<long long int, float>>(
                psf->object);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_geometric_restraint_force_destroy(
    apo_geometric_restraint_force *restraint) {
  const char *function_name = "apo_geometric_restraint_force_destroy";

  apocharmm_c::guard_destroy(
      [restraint](void) -> void {
        delete restraint;
        return;
      },
      function_name);

  return;
}

extern "C" apo_status apo_geometric_restraint_force_add_plane_from_selection(
    apo_geometric_restraint_force *restraint,
    const apo_atom_selection *selection, const double *origin,
    const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode) {
  const char *function_name =
      "apo_geometric_restraint_force_add_plane_from_selection";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_selection>(
                selection, function_name, "AtomSelection"));

        std::array<double, 3> cpp_origin = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(copy_cartesian_vector(
            &cpp_origin, origin, origin_len, function_name, "origin"));

        std::array<double, 3> cpp_direction = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(
            copy_cartesian_vector(&cpp_direction, direction, direction_len,
                                  function_name, "direction"));

        GeometricRestraintForce<long long int,
                                float>::Activation cpp_activation =
            GeometricRestraintForce<long long int, float>::Activation::INSIDE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_activation(
                &cpp_activation, activation, function_name));

        GeometricRestraintForce<long long int, float>::SelectionMode
            cpp_selection_mode =
                GeometricRestraintForce<long long int,
                                        float>::SelectionMode::ATOMWISE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_selection_mode(
                &cpp_selection_mode, selection_mode, function_name));

        restraint->object->addPlane(*selection->object, cpp_origin,
                                    cpp_direction, force_constant, offset,
                                    cpp_activation, cpp_selection_mode);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_geometric_restraint_force_add_plane_from_atom_reference(
    apo_geometric_restraint_force *restraint, const apo_atom_reference *atom,
    const double *origin, const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode) {
  const char *function_name =
      "apo_geometric_restraint_force_add_plane_from_atom_reference";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                atom, function_name, "AtomReference"));

        std::array<double, 3> cpp_origin = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(copy_cartesian_vector(
            &cpp_origin, origin, origin_len, function_name, "origin"));

        std::array<double, 3> cpp_direction = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(
            copy_cartesian_vector(&cpp_direction, direction, direction_len,
                                  function_name, "direction"));

        GeometricRestraintForce<long long int,
                                float>::Activation cpp_activation =
            GeometricRestraintForce<long long int, float>::Activation::INSIDE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_activation(
                &cpp_activation, activation, function_name));

        GeometricRestraintForce<long long int, float>::SelectionMode
            cpp_selection_mode =
                GeometricRestraintForce<long long int,
                                        float>::SelectionMode::ATOMWISE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_selection_mode(
                &cpp_selection_mode, selection_mode, function_name));

        restraint->object->addPlane(*atom->object, cpp_origin, cpp_direction,
                                    force_constant, offset, cpp_activation,
                                    cpp_selection_mode);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_geometric_restraint_force_add_cylinder_from_selection(
    apo_geometric_restraint_force *restraint,
    const apo_atom_selection *selection, const double *origin,
    const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode) {
  const char *function_name =
      "apo_geometric_restraint_force_add_cylinder_from_selection";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_selection>(
                selection, function_name, "AtomSelection"));

        std::array<double, 3> cpp_origin = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(copy_cartesian_vector(
            &cpp_origin, origin, origin_len, function_name, "origin"));

        std::array<double, 3> cpp_direction = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(
            copy_cartesian_vector(&cpp_direction, direction, direction_len,
                                  function_name, "direction"));

        GeometricRestraintForce<long long int,
                                float>::Activation cpp_activation =
            GeometricRestraintForce<long long int, float>::Activation::INSIDE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_activation(
                &cpp_activation, activation, function_name));

        GeometricRestraintForce<long long int, float>::SelectionMode
            cpp_selection_mode =
                GeometricRestraintForce<long long int,
                                        float>::SelectionMode::ATOMWISE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_selection_mode(
                &cpp_selection_mode, selection_mode, function_name));

        restraint->object->addCylinder(*selection->object, cpp_origin,
                                       cpp_direction, force_constant, offset,
                                       cpp_activation, cpp_selection_mode);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_geometric_restraint_force_add_cylinder_from_atom_reference(
    apo_geometric_restraint_force *restraint, const apo_atom_reference *atom,
    const double *origin, const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode) {
  const char *function_name =
      "apo_geometric_restraint_force_add_cylinder_from_atom_reference";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                atom, function_name, "AtomReference"));

        std::array<double, 3> cpp_origin = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(copy_cartesian_vector(
            &cpp_origin, origin, origin_len, function_name, "origin"));

        std::array<double, 3> cpp_direction = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(
            copy_cartesian_vector(&cpp_direction, direction, direction_len,
                                  function_name, "direction"));

        GeometricRestraintForce<long long int,
                                float>::Activation cpp_activation =
            GeometricRestraintForce<long long int, float>::Activation::INSIDE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_activation(
                &cpp_activation, activation, function_name));

        GeometricRestraintForce<long long int, float>::SelectionMode
            cpp_selection_mode =
                GeometricRestraintForce<long long int,
                                        float>::SelectionMode::ATOMWISE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_selection_mode(
                &cpp_selection_mode, selection_mode, function_name));

        restraint->object->addCylinder(*atom->object, cpp_origin, cpp_direction,
                                       force_constant, offset, cpp_activation,
                                       cpp_selection_mode);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_geometric_restraint_force_add_sphere_from_selection(
    apo_geometric_restraint_force *restraint,
    const apo_atom_selection *selection, const double *center,
    const size_t center_len, const double force_constant, const double offset,
    const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode) {
  const char *function_name =
      "apo_geometric_restraint_force_add_sphere_from_selection";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_selection>(
                selection, function_name, "AtomSelection"));

        std::array<double, 3> cpp_center = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(copy_cartesian_vector(
            &cpp_center, center, center_len, function_name, "center"));

        GeometricRestraintForce<long long int,
                                float>::Activation cpp_activation =
            GeometricRestraintForce<long long int, float>::Activation::INSIDE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_activation(
                &cpp_activation, activation, function_name));

        GeometricRestraintForce<long long int, float>::SelectionMode
            cpp_selection_mode =
                GeometricRestraintForce<long long int,
                                        float>::SelectionMode::ATOMWISE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_selection_mode(
                &cpp_selection_mode, selection_mode, function_name));

        restraint->object->addSphere(*selection->object, cpp_center,
                                     force_constant, offset, cpp_activation,
                                     cpp_selection_mode);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_geometric_restraint_force_add_sphere_from_atom_reference(
    apo_geometric_restraint_force *restraint, const apo_atom_reference *atom,
    const double *center, const size_t center_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode) {
  const char *function_name =
      "apo_geometric_restraint_force_add_sphere_from_atom_reference";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                atom, function_name, "AtomReference"));

        std::array<double, 3> cpp_center = {0.0, 0.0, 0.0};
        APOCHARMM_C_RETURN_IF_ERROR(copy_cartesian_vector(
            &cpp_center, center, center_len, function_name, "center"));

        GeometricRestraintForce<long long int,
                                float>::Activation cpp_activation =
            GeometricRestraintForce<long long int, float>::Activation::INSIDE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_activation(
                &cpp_activation, activation, function_name));

        GeometricRestraintForce<long long int, float>::SelectionMode
            cpp_selection_mode =
                GeometricRestraintForce<long long int,
                                        float>::SelectionMode::ATOMWISE;
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::to_geometric_restraint_selection_mode(
                &cpp_selection_mode, selection_mode, function_name));

        restraint->object->addSphere(*atom->object, cpp_center, force_constant,
                                     offset, cpp_activation,
                                     cpp_selection_mode);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_geometric_restraint_force_set_masses(
    apo_geometric_restraint_force *restraint, const double *masses,
    const size_t masses_len) {
  const char *function_name = "apo_geometric_restraint_force_set_masses";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

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
apo_geometric_restraint_force_reset(apo_geometric_restraint_force *restraint) {
  const char *function_name = "apo_geometric_restraint_force_reset";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_geometric_restraint_force>(
                restraint, function_name, "GeometricRestraintForce"));

        restraint->object->reset();

        return APO_STATUS_OK;
      },
      function_name);
}
