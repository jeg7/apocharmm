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
 * @brief Owns atomwise and RCM harmonic MMFP terms for a fixed atom count.
 *
 * Construction snapshots the complete PSF mass vector and atom count; no PSF
 * owner or pointer is retained. Term addition copies selected zero-based atom
 * indices. Separate terms may overlap or be identical.
 *
 * For normalized direction n and raw Cartesian displacement u = r - origin,
 * a plane uses s = dot(n, u). SYMMETRIC uses q = s. INSIDE and OUTSIDE use
 * q = |s|, with derivative +n at s == 0. A cylinder uses p = u - dot(n, u) * n,
 * q = |p|, and grad(q) = p / q. A sphere uses q = |u| and grad(q) = u / q.
 * Cylinder and sphere terms contribute zero energy and gradient when the
 * computed squared radial distance is <= 1e-10 angstrom^2, before activation
 * or deviation is evaluated. This includes the exact axis or center.
 *
 * Otherwise, INSIDE contributes when q > offset; OUTSIDE contributes when
 * q < offset; SYMMETRIC has no one-sided gate. Equality contributes zero.
 * An active atom has E = K * (q - offset)^2 / 2 and physical force
 * -K * (q - offset) * grad(q). Box dimensions do not cause minimum-image
 * replacement of any displacement.
 *
 * Geometry, potential evaluation, and ordered accumulation use double
 * precision. CT selects the final Cartesian-gradient conversion precision,
 * consistently with DistanceRestraintForce. The coordinate input remains
 * float4 for both specializations. getForce() stores energetic gradients, not
 * physical forces, in the repository's signed fixed-point representation.
 *
 * Plane, cylinder, and sphere terms coexist in one flattened collection and
 * contribute additively, including overlapping selections and duplicate terms.
 * Each term records SelectionMode::ATOMWISE or SelectionMode::RCM. Existing
 * calls default to ATOMWISE. RCM evaluates the geometry once at the signed
 * mass-weighted center R = sum(m_i * r_i) / sum(m_i), then distributes its
 * energetic gradient using m_i / sum(m_i). Coordinates and center-reference
 * displacements are raw Cartesian values: no periodic unwrapping, anchor, or
 * minimum-image operation is applied. AtomSelection yields ascending indices;
 * expression/insertion order does not define an RCM anchor.
 *
 * One owned active mass vector serves every RCM term. setMasses() copies and
 * transactionally replaces it; atomwise terms ignore masses. Finite signed and
 * zero individual masses are accepted. Every RCM term requires a finite,
 * nonzero signed total and finite individual mass/total ratios. No positivity
 * restriction or small-total cutoff is imposed. Selected totals are summed in
 * ascending atom-index order in double precision. The runtime  also checks
 * center products/reductions and geometry results for finite arithmetic.
 *
 * RCM products and partial sums use double precision, but input coordinates
 * remain float4. Near cancellation, their coordinate quantization can change
 * the center substantially; CT=double does not recover decimal coordinates lost
 * at input. The existing fixed-point output range still applies. Raw RCM
 * centers and center gradients are retained as evaluation workspace.
 *
 * CHARMM adds overlapping term gradients at each atom. We therefore sum those
 * gradients before forming the external tensor W_ab = sum_i r_i,a * g_i,b in
 * kcal/mol. Components are stored as XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ in
 * Virial_t::virmat. Raw absolute input coordinates and uncoverted double
 * gradients are used. The tensor need not be symmetric or invariant under
 * common translation of atoms and restraint origins. CHARMM distributes RCM
 * gradients to the selected atoms, so their individual moments enter this sum.
 * CHARMM uses raw Cartesian coordinates here, so no image correction is added.
 * This tensor is defined from coordinates and gradients; it does not specify
 * CHARMM's internal pressure accumulator.
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
  /** Identifies the geometry of an atomwise or RCM harmonic term. */
  enum class Geometry : int { PLANE = 0, CYLINDER = 1, SPHERE = 2 };
  /** Selects the symmetric law or a one-sided slab-distance gate. */
  enum class Activation : int { SYMMETRIC = 0, INSIDE = 1, OUTSIDE = -1 };
  /** Selects independent atoms or one signed mass-weighted center. */
  enum class SelectionMode : int { ATOMWISE = 0, RCM = 1 };
  /** Identifies the implemented harmonic potential. */
  enum class PotentialType : int { HARMONIC = 0 };

  /** Includes the complete external tensor in manager aggregation. */
  static constexpr bool contributesVirial = true;

  /**
   * calcForce supplies the complete tensor; no manager reconstruction is
   * needed .
   */
  static constexpr bool computesVirial = true;

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
   * @brief Adds one harmonic plane term from a nonempty selection.
   *
   * The selection must represent getNumAtoms() atoms. Origin is in angstroms;
   * direction is dimensionless; forceConstant is in kcal/mol/angstrom^2; offset
   * is CHARMM's DROFF in angstroms. All inputs must be finite. Signed
   * and zero force constants and offsets are accepted. Direction is normalized
   * in double precision and must have a positive finite Euclidean norm.
   *
   * Inputs are borrowed for the call and copied into owned host storage. No
   * selection or topology is retained. Allocation/validation failure leaves
   * the installed terms unchanged. Device mirrors are updated before use.
   * selectionMode explicitly chooses atomwise or RCM evaluation. RCM addition
   * validates the selected signed total and shares under the active masses;
   * this also applies when forceConstant is zero. An empty selection is
   * rejected for either mode. AtomReference provides the one-atom form.
   *
   * @note The typed direction envelope uses std::hypot, not CHARMM's lexical
   * parser or its underflow/overflow acceptance paths.
   */
  void addPlane(const AtomSelection &selection,
                const std::array<double, 3> &origin,
                const std::array<double, 3> &direction,
                const double forceConstant, const double offset = 0.0,
                const Activation activation = Activation::INSIDE,
                const SelectionMode selectionMode = SelectionMode::ATOMWISE);

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
                const Activation activation = Activation::INSIDE,
                const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds one harmonic infinite-cylinder term.
   *
   * The Cartesian axis passes through origin with the supplied direction.
   * Direction is normalized using the same finite std::hypot envelope as
   * addPlane(); positive scaling or reversal leaves the cylinder unchanged.
   * Translation of origin along the axis leaves the geometry unchanged. The
   * radial coordinate is the distance to the axis. A computed squared distance
   * <= 1e-10 A^2 suppresses both energy and gradient before activation,
   * including on-axis and near-axis positions.
   *
   * Selection validation, units, signed parameter acceptance, ownership,
   * failure guarantees, and synchronization are identical to addPlane(). Raw
   * Cartesian displacement is used; P1 box lengths do not image the axis.
   */
  void addCylinder(const AtomSelection &selection,
                   const std::array<double, 3> &origin,
                   const std::array<double, 3> &direction,
                   const double forceConstant, const double offset = 0.0,
                   const Activation activation = Activation::INSIDE,
                   const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds the same cylinder term for one force-local atom index.
   *
   * AtomReference validation and ownership are identical to addPlane(). Other
   * semantics are identical to the AtomSelection overload of addCylinder().
   */
  void addCylinder(const AtomReference &atom,
                   const std::array<double, 3> &origin,
                   const std::array<double, 3> &direction,
                   const double forceConstant, const double offset = 0.0,
                   const Activation activation = Activation::INSIDE,
                   const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds one harmonic sphere term with a Cartesian center.
   *
   * The radial coordinate is the distance to center. A computed squared
   * distance <= 1e-10 angstrom^2 suppresses both energy and gradient before
   * activation, including the exact center and nearby positions.
   *
   * Center components must be finite and are in angstroms. Selection
   * validation, force-constant and offset units, signed parameter acceptance,
   * ownership, failure guarantees, and synchronization are identical to
   * addPlane(). Raw Cartesian displacement is used; P1 box lengths do not
   * image the center. There is no direction parameter.
   */
  void addSphere(const AtomSelection &selection,
                 const std::array<double, 3> &center,
                 const double forceConstant, const double offset = 0.0,
                 const Activation activation = Activation::INSIDE,
                 const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds the same sphere term for one force-local atom index.
   *
   * AtomReference validation and ownership are identical to addPlane(). Other
   * semantics are identical to the AtomSelection overload of addSphere().
   */
  void addSphere(const AtomReference &atom, const std::array<double, 3> &center,
                 const double forceConstant, const double offset = 0.0,
                 const Activation activation = Activation::INSIDE,
                 const SelectionMode selectionMode = SelectionMode::ATOMWISE);

public:
  /**
   * @brief Copies one replacement mass vector for all current and future RCM
   * terms.
   *
   * The length must equal getNumAtoms(); every entry must be finite. Each
   * installed RCM term must have a finite nonzero signed selected total and
   * finite mass/total ratios. Signed and zero individual entries, negative
   * totals, and tiny nonzero totals are accepted within this finite envelope.
   * Atomwise selections impose no selected-total restriction.
   *
   * Validates the complete proposal and all RCM selections before allocating
   * replacement host/device storage. Both coherent replacements are prepared
   * before a nonthrowing commt. Any precommit failure preserves active host and
   * device masses, terms, and subsequent calculations. No caller storage is
   * retained. The next calculation uses these masses, including after
   * initialization or subscription; reset() and clear() preserve them. Borrowed
   * element pointers/iterators obtained from getMasses() must be reacquired
   * after a successful replacement.
   */
  void setMasses(const std::vector<double> &masses);

public:
  /** Returns the fixed, positive atom count copied during construction. */
  int getNumAtoms(void) const noexcept;

  /** Returns the number of installed terms, including duplicates. */
  std::size_t getNumTerms(void) const noexcept;

  /** Returns a borrowed, read-only reference to the active mass vector. */
  const std::vector<double> &getMasses(void) const noexcept;

  /** Returns shared access to the holder of the privately owned CUDA stream. */
  std::shared_ptr<cudaStream_t> getStream(void);

  /** Returns shared ownership of the fixed-point energetic-gradient storage. */
  std::shared_ptr<Force<AT>> getForce(void);

  /** Returns shared ownership of the energy and complete external tensor. */
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void);

public:
  /** Removes all terms without changing masses, box dimensions, or output. */
  void reset(void);

  /** Stores three finite positive box lengths without imaging coordinates. */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

  /** Validates the fixed atom count, stores the box, and synchronizes terms. */
  void initialize(const int numAtoms, const std::vector<double> &boxDimensions);

  /** Clears all force, energy, and virial output; terms and masses persist. */
  void clear(void);

  /**
   * @brief Clears forces and only the requested energy and virial outputs.
   *
   * ForceManager uses this overload before evaluation. A false flag preserves
   * the corresponding previous value, which must be treated as stale. Both
   * overloads enqueue clearing on the owned stream and preserve all terms.
   */
  void clear(const bool calcEnergy, const bool calcVirial);

  /**
   * @brief Accumulates gradients, optional energy, and optional external
   * tensor.
   *
   * Term gradients are summed per atom in insertion order before conversion to
   * fixed point. Nonfinite arithmetic, unrepresentable converted gradients,
   * and signed-integer addition overflow raise ApoCharmmError (Runtime).
   * Numeric validation occurs on the owned stream and synchronizes that stream
   * before committing output. Numeric rejection leaves previous output intact;
   * CUDA failures do not provide that guarantee. Successful output commitment
   * is enqueued on the same stream. calcEnergy controls only energy
   * accumulation; calcVirial controls only tensor accumulation. Gradients are
   * evaluated in all four flag combinations. A false flag preserves the
   * corresponding previous output, which is stale for the current coordinates.
   * clear() zeros all output; reset() only removes terms. Repeated calls
   * without clearing add only the newly evaluated gradients, energy, and
   * requested tensor contribution. Nonfinite tensor products or accumulation
   * reject the whole calculation.
   */
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial);

  /** Reports P1-only compatibility through ForceView's optional operation. */
  bool supportsPBC(const PBC pbc) const noexcept;

private:
  // JEG261007: Shared selection-count validation and flattening for all three
  // geometries
  std::vector<int> getSelectionIndices(const AtomSelection &selection) const;

  // JEG261007: Shared by all six typed public overloads. Raw indices are not a
  // public API.
  void appendTerm(const std::vector<int> &atomIndices, const Geometry geometry,
                  const std::array<double, 3> &origin,
                  const std::array<double, 3> &direction,
                  const double forceConstant, const double offset,
                  const Activation activation,
                  const SelectionMode selectionMode);

  // JEG261019: Shared by construction and replacement - validates every atom's
  // mass.
  void validateMasses(const std::vector<double> &masses) const;

  // JEG261019: Shared by term addition and replacement - zero total has no
  // valid center.
  double validateRcmMasses(const std::vector<double> &masses,
                           const std::vector<int> &indices,
                           const std::size_t begin,
                           const std::size_t end) const;

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

  // JEG261019: Derived signed totals; atomwise entries are zero and never
  // consumed
  CudaContainer<double> m_SelectedMassTotals;

  // JEG261019: Dynamically sized RCM reduction and center state. Chunk ranges
  // index the flattened membership list. No atom-count ceiling is imposed per
  // term.
  CudaContainer<std::size_t> m_RcmChunkOffsets;
  CudaContainer<std::size_t> m_RcmChunkBegins;
  CudaContainer<std::size_t> m_RcmChunkEnds;
  CudaContainer<double3> m_RcmPartials;
  CudaContainer<double3> m_RcmCenters;
  CudaContainer<double3> m_RcmCenterGradients;

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
  CudaContainer<double> m_VirialPartials;
  CudaContainer<double> m_PendingVirial;
  CudaContainer<int> m_EvaluationError;

  std::shared_ptr<CudaEnergyVirial> m_EnergyVirial;
  std::shared_ptr<Force<AT>> m_Forces;
  std::shared_ptr<cudaStream_t> m_Stream;
};
