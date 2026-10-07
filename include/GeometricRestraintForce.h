// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "AtomReference.h"
#include "AtomSelection.h"
#include "CharmmPSF.h"
#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "Force.h"
#include "PBC.h"

#include <array>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <vector>

/**
 * @brief Owns atomwise harmonic MMFP plane terms for a fixed atom count.
 *
 * Construction snapshots the complete PSF mass vector and atom count; no PSF
 * owner or pointer is retained. Term addition copies selected zero-based atom
 * indices. Separate terms may overlap or be identical.
 *
 * For normalized direction n and raw Cartesian displacement r - origin, let
 * s = dot(n, r - origin). SYMMETRIC uses q = s. INSIDE and OUTSIDE use q = |s|,
 * with derivative +n at s == 0. INSIDE contributes when q > offset; OUTSIDE
 * contributes when q < offset. Equality contributes zero. An active atom has
 * E = K * (q - offset)^2 / 2 and physical force -K * (q - offset) * grad(q).
 * Box dimensions do not cause minimum-image replacement of the displacement.
 *
 * Geometry, potential evaluation, and ordered accumulation use double
 * precision. CT selects the final Cartesian-gradient conversion precision,
 * consistently with DistanceRestraintForce. The coordinate input remains
 * float4 for both specializations. getForce() stores energetic gradients, not
 * physical forces, in the repository's signed fixed-point representation.
 *
 * Only addPlane() is exposed: neither other geometries nor RCM can be added.
 * The flattened term/membership storage is independent of plane geometry.
 *
 * @tparam AT Must be long long int, as required by ForceView.
 * @tparam CT Must be float or double.
 * @warning Calls on one object must be externally serialized. Borrowed device
 * coordinates must remain valid until the owned stream has finished using them.
 */
template <typename AT, typename CT> class GeometricRestraintForce {
  static_assert(std::is_same<AT, long long int>::value,
                "GeometricRestraintForce requires fixed-point accumulation");
  static_assert(std::is_same<CT, float>::value ||
                    std::is_same<CT, double>::value,
                "GeometricRestraintForce supports float or double arithmetic");

public:
  /** Geometry identifiers; only plane addition is currently exposed. */
  enum class Geometry : int { PLANE = 0, CYLINDER = 1, SPHERE = 2 };
  /** Selects the signed symmetric law or a one-sided slab-distance gate. */
  enum class Activation : int { SYMMETRIC = 0, INSIDE = 1, OUTSIDE = -1 };
  /** Selection identifiers; only atomwise addition is currently exposed. */
  enum class SelectionMode : int { ATOMWISE = 0, RCM = 1 };
  /** Identifies the implemented harmonic potential. */
  enum class PotentialType : int { HARMONIC = 0 };

  // JEG261007: The CHARMM-compatible virial path is deferred. Do not include
  // these gradients in ForceManager's virial construction at this intermediate
  // stage.
  static constexpr bool contributesVirial = false;

public:
  /** Prevents construction without a PSF mass and atom-count snapshot. */
  GeometricRestraintForce(void) = delete;

  /**
   * @brief Copies a non-null PSF's positive atom count and complete masses.
   *
   * The shared pointer is borrowed only for this call. Finite signed and zero
   * masses are accepted. A default/uninitialized PSF, zero atoms, a mass-count
   * mismatch, and nonfinite masses are rejected before resource allocation.
   */
  explicit GeometricRestraintForce(const std::shared_ptr<const CharmmPSF> &psf);

  /** Prevents copying stream and force-output ownership. */
  GeometricRestraintForce(const GeometricRestraintForce &other) = delete;

  /** Preserves the object identity held by subscribed ForceViews. */
  GeometricRestraintForce(GeometricRestraintForce &&other) = delete;

  /**
   * Releases the stream and owned storage without propagating cleanup errors.
   */
  ~GeometricRestraintForce(void) noexcept;

public:
  /** Prevents copy assignment of CUDA-owning state. */
  GeometricRestraintForce &
  operator=(const GeometricRestraintForce &other) = delete;

  /** Prevents move assignment of a potentially subscribed force. */
  GeometricRestraintForce &operator=(GeometricRestraintForce &&other) = delete;

public:
  /**
   * @brief Adds one atomwise harmonic plane term from a nonempty selection.
   *
   * The selection must represent getNumAtoms() atoms. Origin is in angstroms;
   * direction is dimensionless; forceConstant is in kcal/mol/angstrom^2; offset
   * is the characterized DROFF in angstroms. All inputs must be finite. Signed
   * and zero force constants and offsets are accepted. Direction is normalized
   * in double precision and must have a positive finite Euclidean norm.
   *
   * Inputs are borrowed for the call and copied into owned host storage. No
   * selection or topology is retained. Allocation/validation failure leaves
   * the installed terms unchanged. Device mirrors are updated before use.
   *
   * @note The typed direction envelope uses std::hypot, not CHARMM's lexical
   * parser or its underflow/overflow acceptance paths.
   */
  void addPlane(const AtomSelection &selection,
                const std::array<double, 3> &origin,
                const std::array<double, 3> &direction,
                const double forceConstant, const double offset = 0.0,
                const Activation activation = Activation::INSIDE);

  /**
   * @brief Adds the same plane term for one force-local atom index.
   *
   * Only atom.getAtomIndex() is validated, against [0, getNumAtoms()). The
   * source topology is not compared, copied, or retained. Other parameter and
   * synchronization semantics are identical to the AtomSelection overload.
   */
  void addPlane(const AtomReference &atom, const std::array<double, 3> &origin,
                const std::array<double, 3> &direction,
                const double forceConstant, const double offset = 0.0,
                const Activation activation = Activation::INSIDE);

public:
  /** Returns the fixed, positive atom count copied during construction. */
  int getNumAtoms(void) const noexcept;

  /** Returns the number of installed terms, including duplicates. */
  std::size_t getNumTerms(void) const noexcept;

  /** Returns a borrowed, read-only reference to the complete mass snapshot. */
  const std::vector<double> &getMasses(void) const noexcept;

  /** Returns shared access to the holder of the privately owned CUDA stream. */
  std::shared_ptr<cudaStream_t> getStream(void);

  /** Returns shared ownership of the fixed-point energetic-gradient storage. */
  std::shared_ptr<Force<AT>> getForce(void);

  /** Returns shared ownership of the energy and cleared virial storage. */
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void);

public:
  /** Removes all terms without changing masses, box dimensions, or output. */
  void reset(void);

  /** Stores three finite positive box lengths without imaging coordinates. */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

  /** Validates the fixed atom count, stores the box, and synchronizes terms. */
  void initialize(const int numAtoms, const std::vector<double> &boxDimensions);

  /** Clears output only; terms and the mass snapshot are preserved. */
  void clear(void);

  /**
   * @brief Accumulates gradients and optionally energy for all installed terms.
   *
   * Term gradients are summed per atom in insertion order before conversion to
   * fixed point. Nonfinite arithmetic, unrepresentable converted gradients,
   * and signed-integer addition overflow raise ApoCharmmError (Runtime).
   * Numeric validation occurs on the owned stream and synchronizes that stream
   * before committing output. Numeric rejection leaves previous output intact;
   * CUDA failures do not provide that guarantee. Successful output commitment
   * is enqueued on the same stream. calcEnergy == false preserves energy.
   *
   * calcVirial is accepted for ForceView compatibility but adds no virial.
   * An empty term collection returns without reading xyzq; otherwise xyzq
   * must be non-null and address at least getNumAtoms() device float4 records.
   */
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial);

  /** Reports P1-only compatibility through ForceView's optional operation. */
  bool supportsPBC(const PBC pbc) const noexcept;

private:
  // JEG261007: Shared by the two types public overloads. Raw indices are not a
  // public API.
  void appendPlane(const std::vector<int> &atomIndices,
                   const std::array<double, 3> &origin,
                   const std::array<double, 3> &direction,
                   const double forceConstant, const double offset,
                   const Activation activation);

  void syncDeviceData(void);

private:
  int m_NumAtoms;
  std::array<double, 3> m_BoxDimensions;
  bool m_DeviceDataDirty;
  CudaContainer<double> m_Masses;

  // JEG261002: Terms own half-open membership ranges, independent of
  // geometry/potential
  CudaContainer<std::size_t> m_TermOffsets;
  CudaContainer<int> m_AtomIndices;

  // JEG261002: x -> Geometry
  //            y -> Activation
  //            z -> SelectionMode
  //            w -> PotentialType
  CudaContainer<int4> m_TermTypes;
  CudaContainer<double3> m_Origins;
  CudaContainer<double3> m_Directions;
  CudaContainer<double> m_Offsets;
  CudaContainer<double> m_ForceConstants;

  // JEG261002: Derived inverse membership lists preserve term insertion order
  // per atom
  CudaContainer<std::size_t> m_AtomOffsets;
  CudaContainer<std::size_t> m_AtomMemberships;

  // JEG261002: Private evaluation and checked-commit workspace (never exposed
  // as output)
  CudaContainer<double3> m_MembershipGradients;
  CudaContainer<double> m_TermEnergies;
  CudaContainer<longlong3> m_PendingForces;
  CudaContainer<double> m_PendingEnergy;
  CudaContainer<int> m_EvaluationError;

  std::shared_ptr<CudaEnergyVirial> m_EnergyVirial;
  std::shared_ptr<Force<AT>> m_Forces;
  std::shared_ptr<cudaStream_t> m_Stream;
};
