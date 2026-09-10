// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "Force.h"

#include <array>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

/**
 * @brief Selects the one-sided activation mode for a distance-restraint term.
 */
enum class DistanceRestraintCondition : int {
  /** Applies no one-sided activation gate. */
  NONE = 0,
  /** Selects the positive-side activation mode. */
  POSITIVE = 1,
  /** Selects the negative-side activation mode. */
  NEGATIVE = -1
};

/**
 * @brief Computes one or more flattened distance-restraint terms.
 *
 * For term `t`, the associated pairs occupy the half-open flattened range
 * `[m_TermPairOffsets[t], m_TermPairOffsets[t + 1])`. For pair coefficient
 * `c_tj`, raw primary-coordinate distance `r_tj`, distance exponent `IVAL_t`,
 * reference value `RVAL_t`, force constant `KVAL_t`, energy exponent `EVAL_t`,
 * and global scale `SCALE`, an active term uses:
 *
 * @code
 * D_t = sum_j(c_tj * r_tj^IVAL_t) - RVAL_t
 * U_t = SCALE * (KVAL_t / EVAL_t) * D_t^EVAL_t
 * @endcode
 *
 * One object may own multiple terms. `addRestraint()` appends one term and its
 * pair data to flattened host mirrors. The corresponding device mirrors are
 * synchronized lazily before calculation.
 *
 * The controlling characterization requires raw primary-coordinate pair
 * distances for the characterized nonperiodic and orthorhombic P1 cases.
 * Stored box dimensions satisfy the `ForceManager` lifecycle and do not imply
 * minimum-image replacement of those pair distances.
 *
 * @tparam AT Device force-accumulator representation. The supported public
 * specializations use apoCHARMM's fixed-point force representation.
 * @tparam CT Arithmetic representation used for pair-gradient conversion to
 * the device force accumulator.
 *
 * @warning The complete accepted ranges of the distance and energy exponents
 * remain unresolved outside the characterized values.
 * @warning Nonorthorhombic, non-P1, and other uncharacterized periodic or image
 * configurations remain unsupported and must not be inferred.
 * @warning Every evaluated exact zero-distance pair is rejected before
 * exponentiation or gradient formation.
 * @warning The class provides no internal host-thread synchronization. Callers
 * must serialize configuration, clearing, calculation, and destruction.
 *
 * @see ForceManager
 */
template <typename AT, typename CT> class DistanceRestraintForce {
public:
  /**
   * @brief Reports that this restraint contributes virial state.
   */
  static constexpr bool contributesVirial = true;

public:
  /**
   * @brief Prevents construction without an explicit atom count.
   */
  DistanceRestraintForce(void) = delete;

  /**
   * @brief Constructs a distance-restraint force for a fixed atom count.
   *
   * The object initially owns no restraint terms. The confirmed initial global
   * scale is one. The atom count remains fixed for the lifetime of the object.
   *
   * @param[in] numAtoms Dimensionless number of atoms represented by coordinate
   * and force arrays.
   */
  DistanceRestraintForce(const int numAtoms);

  /**
   * @brief Releases owned CUDA and output resources.
   */
  ~DistanceRestraintForce(void) noexcept;

public:
  /**
   * @brief Sets the global scale applied to every active restraint term.
   *
   * The characterized behavior accepts positive, zero, and negative finite
   * scale values. The scale is host scalar state and does not alter stored term
   * definitions.
   *
   * @param[in] scale Global multiplicative scale.
   */
  void setScale(const double scale);

  /**
   * @brief Appends one restraint term and its flattened pair definitions.
   *
   * The pair and coefficient vectors are borrowed for this call and are copied
   * into owned host mirrors. Native atom indices use apoCHARMM's zero-based
   * indexing convention.
   *
   * The distance exponent remains a required argument because the following
   * energy exponent has no confirmed default. The activation condition defaults
   * to `NONE`, corresponding to the confirmed absence of a one-sided selector.
   *
   * @param[in] atomPairs Atom-index pairs belonging to this term.
   * @param[in] coefficients Algebraic coefficient corresponding to each pair.
   * @param[in] forceConstant Nonzero term force constant.
   * @param[in] referenceValue Term reference value.
   * @param[in] distanceExponent Integer exponent applied to each pair distance.
   * @param[in] energyExponent Integer exponent applied to the term deviation.
   * @param[in] condition One-sided activation mode.
   */
  void addRestraint(const std::vector<std::array<int, 2>> &atomPairs,
                    const std::vector<double> &coefficients,
                    const double forceConstant, const double referenceValue,
                    const int distanceExponent, const int energyExponent,
                    const DistanceRestraintCondition condition =
                        DistanceRestraintCondition::NONE);

  /**
   * @brief Removes every restraint term and restores the global scale to one.
   *
   * Output force, energy, and virial storage are not substituted for the term
   * reset operation; use `clear()` to clear accumulated output.
   */
  void reset(void);

  /**
   * @brief Stores the current three box dimensions.
   *
   * The dimensions are retained for `ForceManager` compatibility. They must not
   * cause the characterized raw primary-coordinate pair distances to be
   * replaced by minimum-image distances.
   *
   * @param[in] boxDimensions Three box lengths in `[x, y, z]` order.
   */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

  /**
   * @brief Initializes the restraint through the `ForceManager` interface.
   *
   * @param[in] numAtoms Atom count supplied by the manager.
   * @param[in] boxDimensions Three box lengths in `[x, y, z]` order.
   */
  void initialize(const int numAtoms, const std::vector<double> &boxDimensions);

  /**
   * @brief Enqueues clearing of accumulated force, energy, and virial output.
   *
   * This operation does not remove stored restraint terms and does not restore
   * the global scale.
   */
  void clear(void);

  /**
   * @brief Enqueues distance-restraint force and optional energy and virial
   * accumulation.
   *
   * Dirty flattened parameter mirrors are synchronized before their device data
   * are consumed. Exact zero-distance pairs terminate device evaluation before
   * exponentiation or gradient formation. A zero global scale returns before
   * geometry is read.
   *
   * @param[in] xyzq Borrowed CUDA-device coordinate-charge array.
   * @param[in] calcEnergy Whether to accumulate the restraint energy.
   * @param[in] calcVirial Whether to accumulate the restraint virial.
   */
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial);

  /**
   * @brief Returns shared access to the privately owned CUDA stream holder.
   */
  std::shared_ptr<cudaStream_t> getStream(void);

  /**
   * @brief Returns shared ownership of the atom-sized device force storage.
   */
  std::shared_ptr<Force<AT>> getForce(void);

  /**
   * @brief Returns shared ownership of the energy and virial storage.
   */
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void);

private:
  /**
   * @brief Synchronizes dirty flattened host parameter mirrors to device
   * storage.
   */
  void syncDeviceData(void);

  /**
   * @brief Releases the privately owned CUDA stream without propagating cleanup
   * failures.
   */
  void dealloc(void) noexcept;

private:
  /** Fixed dimensionless atom count established during construction. */
  int m_NumAtoms;

  /** Global factor applied to every active restraint term. */
  double m_Scale;

  /** Stored X box dimension. */
  double m_BoxDimX;
  /** Stored Y box dimension. */
  double m_BoxDimY;
  /** Stored Z box dimension. */
  double m_BoxDimZ;

  /** True when flattened host mirrors are newer than their device mirrors. */
  bool m_DeviceDataDirty;

  /**
   * Flattened host/device pair offsets. For `N` terms, element `t` begins term
   * `t`, and element `t + 1` ends its half-open pair range.
   */
  CudaContainer<int> m_TermPairOffsets;

  /** Flattened host/device first atom index for each pair. */
  CudaContainer<int> m_FirstAtomIndices;

  /** Flattened host/device second atom index for each pair. */
  CudaContainer<int> m_SecondAtomIndices;

  /** Flattened host/device algebraic coefficient for each pair. */
  CudaContainer<double> m_PairCoefficients;

  /** Host/device force constant for each restraint term. */
  CudaContainer<double> m_ForceConstants;

  /** Host/device reference value for each restraint term. */
  CudaContainer<double> m_ReferenceValues;

  /** Host/device distance exponent for each restraint term. */
  CudaContainer<int> m_DistanceExponents;

  /** Host/device energy exponent for each restraint term. */
  CudaContainer<int> m_EnergyExponents;

  /**
   * Host/device activation modes stored as the explicit integer representation
   * of `DistanceRestraintCondition`.
   */
  CudaContainer<int> m_ActivationConditions;

  /** Shared owner of distance-restraint energy and virial state. */
  std::shared_ptr<CudaEnergyVirial> m_EnergyVirial;

  /** Shared owner of atom-sized device force storage. */
  std::shared_ptr<Force<AT>> m_Forces;

  /** Shared host holder for the privately owned CUDA stream handle. */
  std::shared_ptr<cudaStream_t> m_Stream;
};
