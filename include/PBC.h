// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

/**
 * @file
 * @brief Declares apoCHARMM periodic-boundary and crystal identifiers.
 */

#pragma once

/**
 * @brief Identifies a periodic-boundary configuration value.
 *
 * `UNSET` is an invalid, non-runnable sentinel. It does not select an aperiodic
 * system. The public periodic-boundary setters accept only `P1` and `P21`. `P1`
 * is the default imported from a newly constructed @ref ForceManager.
 */
enum class PBC {
  /** Represents an unconfigured, non-runnable periodic-boundary state. */
  UNSET = 0,
  /** Selects conventional three-dimensional translational periodicity. */
  P1 = 1,
  /** Selects the P2_1 screw-symmetry periodic boundary condition. */
  P21 = 2
};

/**
 * @brief Selects the crystal symmetry used by the Langevin-piston integrator.
 *
 * The selected symmetry determines the number and ordering of active piston
 * degrees of freedom. @ref CudaLangevinPistonIntegrator accepts only `CUBIC`,
 * `TETRAGONAL`, and `ORTHORHOMBIC`; `NONE` represents an unconfigured state.
 *
 * @see cuda_integrators
 */
enum class CRYSTAL {
  /** Represents an unconfigured crystal with no piston degree of freedom. */
  NONE,
  /** Couples X, Y, and Z to one isotropic piston degree of freedom. */
  CUBIC,
  /** Couples X and Y and assigns Z a second piston degree of freedom. */
  TETRAGONAL,
  /** Assigns independent piston degrees of freedom to X, Y, and Z. */
  ORTHORHOMBIC
};
