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
 * @brief Selects the C ABI crystal type owned by a CharmmContext.
 *
 * The values map directly to the native @ref CRYSTAL enum. The context setter
 * accepts `APO_CRYSTAL_TYPE_CUBIC`, `APO_CRYSTAL_TYPE_TETRAGONAL`, and
 * `APO_CRYSTAL_TYPE_ORTHORHOMBIC`. `APO_CRYSTAL_TYPE_UNSET` represents the
 * initial unusable crystal configuration. It does not select an aperiodic
 * system and is rejected as a setter input. Compatible type changes on the
 * context are allowed between completed propagation calls; an attached piston
 * integrator must then be reinitialized before its state is used.
 *
 * @see apo_charmm_context_set_crystal_type
 */
typedef enum apo_crystal_type {
  /** Represents an unusable crystal configuration, not an aperiodic system. */
  APO_CRYSTAL_TYPE_UNSET = 0,
  /** Selects one isotropic X/Y/Z piston degree of freedom. */
  APO_CRYSTAL_TYPE_CUBIC = 1,
  /** Selects coupled X/Y and independent Z piston degrees of freedom. */
  APO_CRYSTAL_TYPE_TETRAGONAL = 2,
  /** Selects independent X, Y, and Z piston degrees of freedom. */
  APO_CRYSTAL_TYPE_ORTHORHOMBIC = 3,
} apo_crystal_type;

/**
 * @brief Selects the activation predicate of one geometric restraint term.
 *
 * The explicit values equal the native `GeometricRestraintForce::Activation`
 * enumerators. Every term-addition function rejects any other value before
 * native conversion. For reaction coordinate `q` and offset `D`, an active
 * evaluation uses `Delta = q - D`, and equality contributes zero. A plane uses
 * its signed distance for `SYMMETRIC` and its absolute slab distance for
 * `INSIDE` and `OUTSIDE`. CHARMM defaults to `INSIDE`; the C ABI requires an
 * explicit value.
 */
typedef enum apo_geometric_restraint_activation {
  /** Applies the harmonic law on both sides of the offset. */
  APO_GEOMETRIC_RESTRAINT_ACTIVATION_SYMMETRIC = 0,
  /** Activates when the reaction coordinate is greater than the offset. */
  APO_GEOMETRIC_RESTRAINT_ACTIVATION_INSIDE = 1,
  /** Activates when the reaction coordinate is less than the offset. */
  APO_GEOMETRIC_RESTRAINT_ACTIVATION_OUTSIDE = -1
} apo_geometric_restraint_activation;

/**
 * @brief Selects independent atom evaluations or one mass-weighted center.
 *
 * The explicit values equal the native
 * `GeometricRestraintForce::SelectionMode` enumerators. Every term-addition
 * function rejects any other value before native conversion.
 */
typedef enum apo_geometric_restraint_selection_mode {
  /** Evaluates every selected atom independently and ignores masses. */
  APO_GEOMETRIC_RESTRAINT_SELECTION_MODE_ATOMWISE = 0,
  /**
   * Evaluates once at the signed mass-weighted center of the selected atoms
   * and distributes the center gradient by signed mass fraction.
   */
  APO_GEOMETRIC_RESTRAINT_SELECTION_MODE_RCM = 1
} apo_geometric_restraint_selection_mode;

#ifdef __cplusplus
}
#endif

#endif
