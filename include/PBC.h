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
 * @brief Declares apoCHARMM periodic-boundary identifiers.
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
