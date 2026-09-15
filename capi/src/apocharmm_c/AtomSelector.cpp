// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/AtomSelector.h"
#include "apocharmm_c/detail/AtomReferenceHandle.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/AtomSelectorHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include "AtomReference.h"
#include "AtomSelection.h"
#include "AtomSelector.h"

#include <memory>
#include <string_view>

extern "C" apo_status apo_atom_selector_create(apo_atom_selector **out,
                                               const apo_charmm_psf *psf) {
  const char *function_name = "apo_atom_selector_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_atom_selector>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        std::unique_ptr<apo_atom_selector> handle(new apo_atom_selector());
        handle->object = std::make_shared<AtomSelector>(psf->object);

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_atom_selector_destroy(apo_atom_selector *selector) {
  const char *function_name = "apo_atom_selector_destroy";
  apocharmm_c::guard_destroy(
      [selector](void) -> void {
        delete selector;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status
apo_atom_selector_select(apo_atom_selection **out,
                         const apo_atom_selector *selector,
                         const char *selection_string) {
  const char *function_name = "apo_atom_selector_select";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_atom_selection>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_selector>(
                selector, function_name, "AtomSelector"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_c_string(
            selection_string, function_name, "selection_string"));

        std::unique_ptr<apo_atom_selection> handle(new apo_atom_selection());
        handle->object = std::make_shared<AtomSelection>(
            selector->object->select(std::string_view(selection_string)));

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_atom_selector_select_atom(apo_atom_reference **out,
                              const apo_atom_selector *selector,
                              const char *selection_string) {
  const char *function_name = "apo_atom_selector_select_atom";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_atom_reference>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_atom_selector>(
                selector, function_name, "AtomSelector"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_c_string(
            selection_string, function_name, "selection_string"));

        std::unique_ptr<apo_atom_reference> handle(new apo_atom_reference());
        handle->object = std::make_unique<AtomReference>(
            selector->object->selectAtom(std::string_view(selection_string)));

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}
