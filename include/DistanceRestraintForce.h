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

#include "AtomReference.h"
#include "CharmmPSF.h"
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
 * One object may own multiple terms. `addRestraint()` accepts topology-aware
 * endpoint pairs, extracts each endpoint's zero-based atom index, and appends
 * one term and its pair data to flattened host mirrors. The corresponding
 * device mirrors are synchronized lazily before calculation.
 *
 * Construction copies only the atom count of a CharmmPSF. The force remains
 * topology-agnostic: it does not retain that PSF or AtomReference objects,
 * compare source topologies, or require the two endpoints of a pair to
 * originate from the same topology. Only the extracted integer indices are
 * stored.
 *
 * Like CHARMM, the force uses raw primary-coordinate pair distances for
 * nonperiodic systems and orthorhombic P1 boxes. Stored box dimensions exist
 * for the `ForceManager` lifecycle and never cause minimum-image replacement of
 * those pair distances.
 *
 * @tparam AT Device force-accumulator representation. The supported public
 * specializations use apoCHARMM's fixed-point force representation.
 * @tparam CT Representation of the force arithmetic, including the per-term
 * reduction and each term's energy value, which is converted to double before
 * it is added to the energy storage, so the sum over terms is double. The
 * library explicitly instantiates `float` and `double`.
 *
 * @warning Only exponent values whose CHARMM behavior has been verified are
 * accepted, although CHARMM itself accepts a wider range.
 * @warning Nonorthorhombic, non-P1, and other periodic or image configurations
 * whose CHARMM behavior has not been verified are unsupported.
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
   * @brief Prevents construction without a PSF atom count.
   *
   * @see CharmmPSF
   */
  DistanceRestraintForce(void) = delete;

  /**
   * @brief Constructs a distance-restraint force from a PSF atom count.
   *
   * Only the positive atom count is copied; PSF masses are not read. The PSF
   * and its shared owner are borrowed only for this call and are not
   * retained. The object initially owns no restraint terms, and its global
   * scale starts at one, as in CHARMM. The atom count remains fixed for the
   * lifetime of the object.
   *
   * @param[in] psf Non-null PSF with an initialized positive atom count, the
   * number of atoms represented by coordinate and force arrays.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the PSF atom count is negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `psf` is null or its atom count is zero.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if CUDA
   * allocation or stream creation fails.
   * @throws std::bad_alloc If host storage, shared ownership state, or an error
   * diagnostic cannot be allocated.
   * @throws std::length_error If requested storage or an error diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post `getForce()->size() == psf->getNumAtoms()`.
   */
  explicit DistanceRestraintForce(const std::shared_ptr<const CharmmPSF> &psf);

  /**
   * @brief Prevents copying CUDA stream and output ownership state.
   */
  DistanceRestraintForce(const DistanceRestraintForce &other) = delete;

  /**
   * @brief Prevents moving a force whose object identity and CUDA resources may
   * be retained by a force manager.
   */
  DistanceRestraintForce(DistanceRestraintForce &&other) = delete;

  /**
   * @brief Prevents copy assignment of CUDA stream and output ownership state.
   */
  DistanceRestraintForce &
  operator=(const DistanceRestraintForce &other) = delete;

  /**
   * @brief Prevents move assignment of CUDA stream and output ownership state.
   */
  DistanceRestraintForce &operator=(DistanceRestraintForce &&other) = delete;

  /**
   * @brief Releases owned CUDA and output resources.
   */
  ~DistanceRestraintForce(void) noexcept;

public:
  /**
   * @brief Sets the global scale applied to every active restraint term.
   *
   * As in CHARMM, positive, zero, and negative finite scale values are
   * accepted. The scale is host scalar state and does not alter stored term
   * definitions.
   *
   * @param[in] scale Global multiplicative scale.
   */
  void setScale(const double scale);

  /**
   * @brief Appends one restraint term from topology-aware atom references.
   *
   * `atomReferencePairs` and `coefficients` are parallel vectors and must have
   * matching lengths. Both vectors are borrowed only for this call. The
   * coefficient values and each endpoint's zero-based atom index are copied
   * into force-owned host storage.
   *
   * Topology provenance is deliberately ignored. This method does not call
   * `AtomReference::hasSameTopology()`, does not require pair endpoints to
   * originate from the same topology, and retains neither the references nor
   * their source CharmmPSF objects after returning. The force stores only the
   * values returned by `AtomReference::getAtomIndex()`.
   *
   * Each extracted index must be in the force-local range `[0, N)`, where `N`
   * is the atom count copied at construction, even when the reference is valid
   * for a larger source topology. The two extracted indices in one pair must be
   * distinct. References from different topologies with distinct valid
   * extracted indices are accepted; references from different topologies with
   * the same extracted index are rejected.
   *
   * Pair order and coefficient correspondence are preserved exactly. Repeated
   * pairs remain accepted.
   *
   * Both exponents are required arguments; apoCHARMM does not reproduce
   * CHARMM's parser defaults for them. The activation condition defaults to
   * `NONE`, which matches a CHARMM definition with neither `POSITIVE` nor
   * `NEGATIVE`.
   *
   * @param[in] atomReferencePairs Atom-reference endpoint pairs belonging to
   * this term.
   * @param[in] coefficients Algebraic coefficient corresponding to each pair.
   * @param[in] forceConstant Nonzero term force constant.
   * @param[in] referenceValue Term reference value.
   * @param[in] distanceExponent Integer exponent applied to each pair distance.
   * @param[in] energyExponent Integer exponent applied to the term deviation.
   * @param[in] condition One-sided activation mode.
   */
  void addRestraint(
      const std::vector<std::array<AtomReference, 2>> &atomReferencePairs,
      const std::vector<double> &coefficients, const double forceConstant,
      const double referenceValue, const int distanceExponent,
      const int energyExponent,
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
   * The dimensions are retained for `ForceManager` compatibility. They never
   * cause the raw primary-coordinate pair distances, which CHARMM also uses, to
   * be replaced by minimum-image distances.
   *
   * @param[in] boxDimensions Three box lengths in `[x, y, z]` order.
   */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

  /**
   * @brief Initializes the restraint through the `ForceManager` interface.
   *
   * The method checks the atom count, stores the box, and synchronizes dirty
   * flattened parameter mirrors to device storage.
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

  /** Stored box dimensions in `[x, y, z]` order. */
  std::array<double, 3> m_BoxDimensions;

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
