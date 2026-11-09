// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_GEOMETRIC_RESTRAINT_FORCE_HANDLE_H__
#define __APOCHARMM_C_DETAIL_GEOMETRIC_RESTRAINT_FORCE_HANDLE_H__

#include "apocharmm_c/GeometricRestraintForce.h"

#include "GeometricRestraintForce.h"

#include <memory>

/**
 * @brief Owns one shared native geometric-restraint reference.
 *
 * The public C handle exclusively owns this wrapper object. Its `object`
 * member retains shared ownership of the supported
 * `GeometricRestraintForce<long long int, float>` specialization, matching the
 * other C ABI force handles. Construction borrows the source PSF handle only
 * for the native constructor call, so the wrapper stores no PSF owner.
 *
 * A non-NULL public handle is valid only while this wrapper remains allocated
 * and `object` is non-NULL. Public C ABI entry points must validate both
 * conditions before dereferencing the native object. The wrapper provides no
 * host-thread synchronization.
 *
 * This private representation is not part of the stable C ABI.
 */
struct apo_geometric_restraint_force {
  std::shared_ptr<GeometricRestraintForce<long long int, float>> object =
      nullptr;
};

#endif
