// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_DISTANCE_RESTRAINT_FORCE_HANDLE_H__
#define __APOCHARMM_C_DETAIL_DISTANCE_RESTRAINT_FORCE_HANDLE_H__

#include "apocharmm_c/DistanceRestraintForce.h"

#include "DistanceRestraintForce.h"

#include <memory>

/**
 * @brief Owns one shared native distance-restraint force reference.
 *
 * The public C handle exclusively owns this wrapper object. Its `object` member
 * shares the native `DistanceRestraintForce<long long int, float>` instance
 * with any subscribed `ForceManager`. Deleting the C handle releases only this
 * shared reference; manager subscription can keep the native restraint, CUDA
 * stream holder, force storage, and energy-virial storage alive.
 *
 * A non-NULL public handle is valid only while this wrapper remains allocated
 * and `object` is non-NULL. Public C ABI entry points must validate both
 * conditions before dereferencing the native object. The wrapper provides no
 * host-thread synchronization.
 *
 * This private representation is not part of the stable C ABI.
 */
struct apo_distance_restraint_force {
  std::shared_ptr<DistanceRestraintForce<long long int, float>> object =
      nullptr;
};

#endif
