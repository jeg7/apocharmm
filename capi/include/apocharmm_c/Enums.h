// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

/**
 * @file
 * @brief Declares public enumeration types used by the apoCHARMM C ABI.
 */

#ifndef __APOCHARMM_C_ENUMS_H__
#define __APOCHARMM_C_ENUMS_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Identifies the periodic boundary condition used by the C ABI.
 *
 * The values map one-to-one to the native `PBC` enumeration.
 */
typedef enum apo_pbc {
  /**
   * Represents an unconfigured, non-runnable periodic-boundary state.
   *
   * This value does not select an aperiodic system and is rejected by the
   * public periodic-boundary setters.
   */
  APO_PBC_UNSET = 0,
  /** Selects conventional three-dimensional translational periodicity. */
  APO_PBC_P1 = 1,
  /** Selects the P2_1 screw-symmetry periodic boundary condition. */
  APO_PBC_P21 = 2
} apo_pbc;

/**
 * @brief Selects the C ABI crystal symmetry for Langevin-piston propagation.
 *
 * The values map directly to the native @ref CRYSTAL enum. The piston setter
 * accepts `APO_CRYSTAL_TYPE_CUBIC`, `APO_CRYSTAL_TYPE_TETRAGONAL`, and
 * `APO_CRYSTAL_TYPE_ORTHORHOMBIC`. `APO_CRYSTAL_TYPE_NONE` represents the
 * initial unconfigured state and is rejected as a setter input.
 *
 * @see apo_cuda_langevin_piston_integrator_set_crystal_type
 */
typedef enum apo_crystal_type {
  /** Represents no configured piston degree of freedom. */
  APO_CRYSTAL_TYPE_NONE = 0,
  /** Selects one isotropic X/Y/Z piston degree of freedom. */
  APO_CRYSTAL_TYPE_CUBIC = 1,
  /** Selects coupled X/Y and independent Z piston degrees of freedom. */
  APO_CRYSTAL_TYPE_TETRAGONAL = 2,
  /** Selects independent X, Y, and Z piston degrees of freedom. */
  APO_CRYSTAL_TYPE_ORTHORHOMBIC = 3,
} apo_crystal_type;

#ifdef __cplusplus
}
#endif

#endif
