// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

/**
 * @file
 * @brief Declares apoCHARMM crystal identifiers.
 */

#pragma once

/**
 * @brief Identifies the orthogonal crystal symmetry owned by CharmmContext.
 *
 * @ref CharmmContext accepts `CUBIC`, `TETRAGONAL`, and `ORTHORHOMBIC`;
 * `UNSET` represents an unconfigured, unusable crystal; it does not select an
 * aperiodic system. The type may change between completed propagation calls
 * when the current box is compatible with the new selection. Box lengths do
 * not determine or change the type. An attached piston must be explicitly
 * initialized after a type change before its state is used again.
 * Langevin-piston degrees of freedom and restart crystal labels use this same
 * context value.
 *
 * @see charmm_context
 * @see cuda_integrators
 */
enum class CRYSTAL {
  /**
   * Represents an invalid, non-runnable crystal configuration.
   * Does not select an aperiodic system.
   */
  UNSET = 0,
  /** Couples X, Y, and Z to one isotropic piston degree of freedom. */
  CUBIC = 1,
  /** Couples X and Y and assigns Z a second piston degree of freedom. */
  TETRAGONAL = 2,
  /** Assigns independent piston degrees of freedom to X, Y, and Z. */
  ORTHORHOMBIC = 3
};
