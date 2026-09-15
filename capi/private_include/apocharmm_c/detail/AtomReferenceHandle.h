// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_ATOM_REFERENCE_HANDLE_H__
#define __APOCHARMM_C_DETAIL_ATOM_REFERENCE_HANDLE_H__

#include "apocharmm_c/AtomReference.h"

#include "AtomReference.h"

#include <memory>

/**
 * @brief Defines the private owning state behind an atom-reference handle.
 *
 * Successful C construction allocates this wrapper and stores unique ownership
 * of one native @ref AtomReference. The native value retains shared const PSF
 * ownership, so destroying the source public PSF or selector handle does not
 * invalidate this wrapper. Public C callers see only the opaque declaration and
 * release the wrapper through @ref apo_atom_reference_destroy.
 */
struct apo_atom_reference {
  /** @brief Owns the native topology-aware atom value. */
  std::unique_ptr<AtomReference> object = nullptr;
};

#endif
