// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/AtomReference.h"
#include "apocharmm_c/detail/AtomReferenceHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include "AtomReference.h"

#include <memory>

extern "C" apo_status apo_atom_reference_create(apo_atom_reference **out,
                                                const apo_charmm_psf *psf,
                                                const int atom_index) {
  const char *function_name = "apo_atom_reference_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_atom_reference>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        std::unique_ptr<apo_atom_reference> handle(new apo_atom_reference());
        handle->object =
            std::make_unique<AtomReference>(psf->object, atom_index);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_atom_reference_destroy(apo_atom_reference *reference) {
  const char *function_name = "apo_atom_reference_destroy";

  apocharmm_c::guard_destroy(
      [reference](void) -> void {
        delete reference;
        return;
      },
      function_name);

  return;
}

extern "C" apo_status
apo_atom_reference_get_atom_index(int *atom_index,
                                  const apo_atom_reference *reference) {
  const char *function_name = "apo_atom_reference_get_atom_index";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            atom_index, function_name, "atom_index"));

        *atom_index = -1;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                reference, function_name, "AtomReference"));

        *atom_index = reference->object->getAtomIndex();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_atom_reference_has_same_topology(bool *has_same_topology,
                                     const apo_atom_reference *reference,
                                     const apo_atom_reference *other) {
  const char *function_name = "apo_atom_reference_has_same_topology";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<bool>(
            has_same_topology, function_name, "has_same_topology"));

        *has_same_topology = false;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                reference, function_name, "AtomReference"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                other, function_name, "Other AtomReference"));

        *has_same_topology = reference->object->hasSameTopology(*other->object);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_atom_reference_equals(bool *equals, const apo_atom_reference *reference,
                          const apo_atom_reference *other) {
  const char *function_name = "apo_atom_reference_equals";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<bool>(
            equals, function_name, "equals"));

        *equals = false;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                reference, function_name, "AtomReference"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_reference>(
                other, function_name, "Other AtomReference"));

        *equals = *reference->object == *other->object;

        return APO_STATUS_OK;
      },
      function_name);
}
