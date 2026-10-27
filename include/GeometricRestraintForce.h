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
 * @brief Computes atomwise and center-of-mass harmonic geometric restraints.
 *
 * One object owns a flattened collection of plane, cylinder, and sphere
 * terms. For normalized direction `n`, position `r`, reference `c`, and
 * displacement `u`, the geometry coordinates are:
 *
 * @code
 * u = r - c
 * s = dot(n, u)
 * q_plane = s                    for SYMMETRIC
 * q_plane = abs(s)               for INSIDE or OUTSIDE
 * p = u - dot(n, u) * n
 * q_cylinder = norm(p)
 * q_sphere = norm(u)
 * @endcode
 *
 * A symmetric plane has `grad(q) = n`. A one-sided plane has
 * `grad(q) = sign(s) * n`, with `+n` at exactly zero. Cylinder and sphere
 * gradients are `p / q` and `u / q`, respectively. A computed squared radial
 * distance at or below `1e-10` angstrom squared suppresses the entire radial
 * contribution before activation, including the exact axis or center.
 *
 * For force constant `K` and offset `D`, each active evaluation uses:
 *
 * @code
 * Delta = q - D
 * U_mmfp = (K / 2) * Delta^2
 * gradient = K * Delta * grad(q)
 * physical_force = -gradient
 * @endcode
 *
 * `INSIDE` is active for `Delta > 0`; `OUTSIDE` is active for `Delta < 0`;
 * `SYMMETRIC` has no one-sided gate. Equality contributes zero. Finite
 * positive, zero, and negative force constants and offsets are accepted.
 * To preserve the characterized arithmetic, energy is evaluated as
 * `(0.5 * K) * (Delta * Delta)`. A computed zero energy suppresses the
 * gradient even when energy output is disabled.
 *
 * `ATOMWISE` evaluates each selected atom independently. `RCM` evaluates
 * once at a signed mass-weighted center and distributes its gradient:
 *
 * @code
 * M = sum over selected atoms i of m_i
 * R = (sum over selected atoms i of m_i * r_i) / M
 * gradient_i = (m_i / M) * gradient_R
 * @endcode
 *
 * One owned mass vector serves all RCM terms. Signed and zero individual
 * masses are accepted; each nonempty RCM selection requires a finite,
 * nonzero total and finite mass fractions. Atomwise terms ignore masses.
 * Selected totals use ascending atom-index order in double precision.
 * Construction copies the PSF atom count and masses; term addition copies
 * selected indices. No PSF, AtomSelection, or AtomReference owner is retained.
 * A valid empty selection adds no term. Separate terms may overlap or repeat.
 *
 * Only orthorhombic `PBC::P1` is supported through ForceManager. Atom
 * positions, RCM construction, and reference displacements use raw Cartesian
 * coordinates without unwrapping or minimum-image replacement. Stored box
 * lengths serve the manager lifecycle and do not alter this geometry.
 * Positions and offsets are in angstroms, masses normally in atomic mass
 * units, and `K` in kilocalories per mole per angstrom squared.
 *
 * Gradients are summed at each atom in term insertion order before conversion
 * to fixed point. Energy uses the named term `mmfp`. The complete external
 * tensor uses raw absolute coordinates and unconverted double gradients:
 *
 * @code
 * W_ab = sum over atoms i of r_i,a * gradient_i,b
 * @endcode
 *
 * Energy and tensor components are in kilocalories per mole. Tensor order is
 * `XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ`; the tensor need not be symmetric or
 * translation invariant. This defines the external coordinate-gradient
 * tensor, not CHARMM's internal pressure accumulator. ForceManager aggregates
 * it once and skips reconstruction because `computesVirial` is true.
 *
 * Terms, selections, and reduction storage grow dynamically. Additions and
 * `reset()` update host state; dirty device mirrors synchronize before use.
 * `setMasses()` prepares coherent replacements before committing. Subscription
 * retains shared ownership without freezing terms or masses. These operations
 * remain available after initialization and subscription. `reset()` removes
 * terms while preserving masses, box lengths, and output; `clear()` clears
 * output while preserving the configured terms and masses.
 *
 * @tparam AT Device force-accumulator representation. Only `long long int`
 * is supported, using apoCHARMM's fixed-point scale of `2^40`.
 * @tparam CT Final Cartesian-gradient conversion representation, either
 * `float` or `double`. Geometry, energy, and RCM arithmetic use double
 * precision; both specializations read current coordinates from `float4`.
 *
 * @warning The class provides no internal host-thread synchronization. Callers
 * must serialize configuration, clearing, calculation, destruction, and all
 * operations on a subscribing manager that access this force.
 * @warning Coordinate quantization and fixed-point range limit the supported
 * numerical domain. `CT = double` does not recover coordinates lost at input.
 * Directions use a positive finite `std::hypot` norm; this typed envelope
 * does not emulate CHARMM's text-parser underflow or overflow paths.
 * @warning `calcForce()` synchronizes for numerical validation before queuing
 * its output commit. CUDA graph capture is unsupported. Keep the CUDA context
 * and borrowed device buffers alive until queued work completes, and release
 * owned resources before resetting their device.
 *
 * @see AtomReference
 * @see AtomSelection
 * @see ForceManager
 * @see PBC
 */
template <typename AT, typename CT> class GeometricRestraintForce {
  static_assert(std::is_same<AT, long long int>::value,
                "GeometricRestraintForce requires fixed-point accumulation");
  static_assert(std::is_same<CT, float>::value ||
                    std::is_same<CT, double>::value,
                "GeometricRestraintForce supports float or double arithmetic");

public:
  /**
   * @brief Selects the geometry of an atomwise or RCM restraint term.
   */
  enum class Geometry : int {
    /** Uses a plane through the reference with the configured normal. */
    PLANE = 0,
    /** Uses an infinite cylinder with the configured reference and axis. */
    CYLINDER = 1,
    /** Uses radial distance from the configured center. */
    SPHERE = 2
  };

  /**
   * @brief Selects the activation predicate applied to the geometry coordinate.
   *
   * A plane uses signed distance for `SYMMETRIC` and absolute distance for
   * one-sided modes. Cylinder and sphere coordinates are nonnegative radii. The
   * radial singular-suppression rule takes precedence over activation.
   */
  enum class Activation : int {
    /** Applies the harmonic law on both sides of the requested offset. */
    SYMMETRIC = 0,
    /** Activates when the geometry coordinate is greater than the offset. */
    INSIDE = 1,
    /** Activates when the geometry coordinate is less than the offset. */
    OUTSIDE = -1
  };

  /**
   * @brief Selects independent atom evaluations or one mass-weighted center.
   */
  enum class SelectionMode : int {
    /** Evaluates each selected atom independently without mass weighting. */
    ATOMWISE = 0,
    /** Evaluates the signed center and distributes its gradient by mass. */
    RCM = 1
  };

  /**
   * @brief Identifies the implemented geometric-restraint potential.
   */
  enum class PotentialType : int {
    /** Uses energy `K * (q - offset)^2 / 2` for an active evaluation. */
    HARMONIC = 0
  };

  /**
   * @brief Reports that this restraint contributes the external virial tensor.
   *
   * ForceManager requests the tensor when global virial calculation is enabled
   * and includes the stored nine-component contribution in its aggregate.
   */
  static constexpr bool contributesVirial = true;

  /**
   * @brief Reports that force evaluation supplies the complete virial tensor.
   *
   * ForceView uses this trait to skip manager-side force conversion and
   * coordinate-gradient reconstruction. The tensor is aggregated once.
   */
  static constexpr bool computesVirial = true;

public:
  /**
   * @brief Prevents construction without a PSF atom-count and mass snapshot.
   *
   * @see CharmmPSF
   */
  GeometricRestraintForce(void) = delete;

  /**
   * @brief Constructs an empty geometric restraint from a PSF snapshot.
   *
   * The positive atom count and complete mass vector are copied. The PSF and
   * its shared owner are borrowed only for this call and are not retained.
   * Finite positive, zero, and negative individual masses are accepted.
   *
   * Construction creates a private CUDA stream, a fixed-point force array, and
   * a `CudaEnergyVirial` containing the named term `mmfp`. Force, energy, and
   * virial outputs are cleared and the private stream is synchronized before
   * return. Stored box dimensions are initially zero.
   *
   * @param[in] psf Non-null PSF with an initialized positive atom count and
   * exactly one finite mass, in atomic mass units, per atom. The atom count
   * must fit the existing fixed-point force-storage representation.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the PSF atom count is negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `psf` is null, the atom count is zero or exceeds the force-storage
   * representation limit, or a mass is NaN or infinite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the PSF
   * mass-vector length differs from its atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if allocation,
   * transfer, stream creation, output clearing, or synchronization fails.
   * @throws std::bad_alloc If host storage, shared ownership state, or an error
   * diagnostic cannot be allocated.
   * @throws std::length_error If requested storage or an error diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post `getNumAtoms() == psf->getNumAtoms()` and
   * `getForce()->size() == getNumAtoms()`.
   * @post `getNumTerms() == 0`, and force, energy, and virial outputs are zero.
   * @see setMasses
   * @see initialize
   */
  explicit GeometricRestraintForce(const std::shared_ptr<const CharmmPSF> &psf);

  /**
   * @brief Prevents copying a restraint that owns CUDA resources.
   *
   * @param[in] other Source restraint. Copy construction is not provided.
   */
  GeometricRestraintForce(const GeometricRestraintForce &other) = delete;

  /**
   * @brief Prevents moving an object referenced by subscribed force views.
   *
   * @param[in] other Source restraint. Move construction is not provided.
   * @see ForceView
   */
  GeometricRestraintForce(GeometricRestraintForce &&other) = delete;

  /**
   * @brief Destroys the private CUDA stream and releases owned resources.
   *
   * CUDA stream destruction is attempted without propagating cleanup failure. A
   * shared holder returned by `getStream()` may outlive this object, but its
   * stored stream value is set to `nullptr` during destruction.
   *
   * Force and energy objects returned by `getForce()` and `getEnergyVirial()`
   * remain allocated while another shared owner exists. No restraint remains to
   * calculate into them after destruction.
   *
   * @post The private CUDA stream is no longer valid.
   * @warning Do not overlap destruction with host calls or external CUDA work
   * that accesses this restraint's storage.
   * @see getStream
   * @see ForceManager::subscribe
   */
  ~GeometricRestraintForce(void) noexcept;

public:
  /**
   * @brief Prevents copy assignment of CUDA resource ownership.
   *
   * @param[in] other Source restraint. Copy assignment is not provided.
   */
  GeometricRestraintForce &
  operator=(const GeometricRestraintForce &other) = delete;

  /**
   * @brief Prevents move assignment of a potentially subscribed restraint.
   *
   * @param[in] other Source restraint. Move assignment is not provided.
   * @see ForceView
   */
  GeometricRestraintForce &operator=(GeometricRestraintForce &&other) = delete;

public:
  /**
   * @brief Adds a harmonic plane term for a copied atom selection.
   *
   * The plane passes through `origin` with a normalized `direction`. SYMMETRIC
   * uses signed distance; INSIDE and OUTSIDE use absolute slab distance.
   * Direction normalization uses double-precision `std::hypot`.
   *
   * Selected indices and parameters are copied into owned host storage. No
   * selection, topology, or caller storage is retained. This method does not
   * transfer or synchronize CUDA data; device mirrors are updated before the
   * next evaluation. Terms may be added after initialization or subscription
   * without replacing output owners or resubscribing.
   *
   * @param[in] selection Selection representing exactly `getNumAtoms()` atoms.
   * A valid empty selection is an accepted no-op after parameter validation; it
   * adds no term and does not require an RCM total-mass calculation.
   * @param[in] origin Three finite coordinates in `[x, y, z]` order and
   * angstrom units.
   * @param[in] direction Three finite dimensionless components with a positive
   * finite Euclidean norm. The normalized direction is copied.
   * @param[in] forceConstant Finite harmonic coefficient in kilocalories per
   * mole per square angstrom. Positive, zero, and negative values are accepted.
   * @param[in] offset Finite CHARMM DROFF value in angstroms. Positive, zero,
   * and negative values are accepted; the default is zero.
   * @param[in] activation SYMMETRIC, INSIDE, or OUTSIDE; the default is INSIDE.
   * @param[in] selectionMode ATOMWISE or RCM; the default is ATOMWISE. A
   * nonempty RCM selection requires a finite nonzero signed total mass and
   * finite mass/total ratios, including when `forceConstant` is zero.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the selection atom count differs, a parameter or enum value is invalid, the
   * direction norm is invalid, membership count would overflow, or an RCM mass
   * requirement is violated.
   * @throws std::bad_alloc If copied term, index, or diagnostic storage cannot
   * be allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success with a nonempty selection, one term is appended in
   * insertion order. Existing terms and output values are preserved.
   * @post Validation or allocation failure leaves installed terms unchanged.
   * @see addCylinder
   * @see addSphere
   * @see setMasses
   */
  void addPlane(const AtomSelection &selection,
                const std::array<double, 3> &origin,
                const std::array<double, 3> &direction,
                const double forceConstant, const double offset = 0.0,
                const Activation activation = Activation::INSIDE,
                const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds a harmonic plane term for one force-local atom index.
   *
   * The AtomReference topology is not compared or retained. Only its index is
   * validated against this restraint. Geometry, signed-parameter acceptance,
   * host copying, and deferred device-update behavior match the selection
   * overload. This method may be used after initialization or subscription.
   *
   * @param[in] atom Borrowed reference with an index in `[0, getNumAtoms())`.
   * @param[in] origin Three finite coordinates in `[x, y, z]` order and
   * angstrom units.
   * @param[in] direction Three finite dimensionless components with a positive
   * finite Euclidean norm; the normalized direction is copied.
   * @param[in] forceConstant Finite signed harmonic coefficient in kilocalories
   * per mole per square angstrom.
   * @param[in] offset Finite signed CHARMM DROFF value in angstroms; default
   * zero.
   * @param[in] activation SYMMETRIC, INSIDE, or OUTSIDE; default INSIDE.
   * @param[in] selectionMode ATOMWISE or RCM; default ATOMWISE. For RCM, the
   * selected atom must have a finite nonzero active mass.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the atom index, parameter, enum value, direction norm, membership count, or
   * RCM mass is invalid.
   * @throws std::bad_alloc If copied term, index, or diagnostic storage cannot
   * be allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, one term is appended; existing terms and outputs remain.
   * @post Validation or allocation failure leaves installed terms unchanged.
   * @see AtomReference
   * @see addPlane
   */
  void addPlane(const AtomReference &atom, const std::array<double, 3> &origin,
                const std::array<double, 3> &direction,
                const double forceConstant, const double offset = 0.0,
                const Activation activation = Activation::INSIDE,
                const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds a harmonic infinite-cylinder term for a copied selection.
   *
   * The axis passes through `origin` along the normalized `direction`. The
   * reaction coordinate is perpendicular distance to that axis. A computed
   * squared radial distance at or below `1e-10` square angstroms suppresses
   * energy and gradient before activation, including on the axis. Raw Cartesian
   * displacements are used without periodic imaging.
   *
   * Indices and parameters are copied into owned host storage. No caller
   * storage is retained and no CUDA transfer or synchronization occurs. Device
   * mirrors are updated before use. Adding a term after initialization or
   * subscription does not require resubscription.
   *
   * @param[in] selection Selection representing exactly `getNumAtoms()` atoms.
   * A valid empty selection is a no-op after parameter validation and adds
   * neither a term nor an RCM center.
   * @param[in] origin Three finite axis-reference coordinates in `[x, y, z]`
   * order and angstrom units.
   * @param[in] direction Three finite dimensionless axis components with a
   * positive finite Euclidean norm, normalized using double-precision
   * `std::hypot`. Positive scaling or reversal leaves the geometry unchanged.
   * @param[in] forceConstant Finite signed harmonic coefficient in kilocalories
   * per mole per square angstrom.
   * @param[in] offset Finite signed CHARMM DROFF value in angstroms; default
   * zero.
   * @param[in] activation SYMMETRIC, INSIDE, or OUTSIDE; default INSIDE.
   * @param[in] selectionMode ATOMWISE or RCM; default ATOMWISE. A nonempty RCM
   * selection requires a finite nonzero signed total and finite mass/total
   * ratios, even when `forceConstant` is zero.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the selection atom count, parameter, enum value, direction norm, membership
   * count, or RCM mass requirement is invalid.
   * @throws std::bad_alloc If copied term, index, or diagnostic storage cannot
   * be allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success with a nonempty selection, one term is appended; existing
   * terms and outputs remain. Validation or allocation failure preserves terms.
   * @see addPlane
   * @see addSphere
   * @see setMasses
   */
  void addCylinder(const AtomSelection &selection,
                   const std::array<double, 3> &origin,
                   const std::array<double, 3> &direction,
                   const double forceConstant, const double offset = 0.0,
                   const Activation activation = Activation::INSIDE,
                   const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds a harmonic infinite-cylinder term for one atom index.
   *
   * Only the force-local index is used; the AtomReference topology is not
   * compared or retained. Geometry, singular suppression, host copying, and
   * deferred device updates match the selection overload. The operation is
   * valid after initialization or subscription.
   *
   * @param[in] atom Borrowed reference with an index in `[0, getNumAtoms())`.
   * @param[in] origin Three finite axis-reference coordinates in `[x, y, z]`
   * order and angstrom units.
   * @param[in] direction Three finite dimensionless axis components with a
   * positive finite Euclidean norm; the normalized direction is copied.
   * @param[in] forceConstant Finite signed harmonic coefficient in kilocalories
   * per mole per square angstrom.
   * @param[in] offset Finite signed CHARMM DROFF value in angstroms; default
   * zero.
   * @param[in] activation SYMMETRIC, INSIDE, or OUTSIDE; default INSIDE.
   * @param[in] selectionMode ATOMWISE or RCM; default ATOMWISE. For RCM, the
   * selected atom must have a finite nonzero active mass.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the atom index, parameter, enum value, direction norm, membership count, or
   * RCM mass is invalid.
   * @throws std::bad_alloc If copied term, index, or diagnostic storage cannot
   * be allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, one term is appended; existing terms and outputs remain.
   * @post Validation or allocation failure leaves installed terms unchanged.
   * @see AtomReference
   * @see addCylinder
   */
  void addCylinder(const AtomReference &atom,
                   const std::array<double, 3> &origin,
                   const std::array<double, 3> &direction,
                   const double forceConstant, const double offset = 0.0,
                   const Activation activation = Activation::INSIDE,
                   const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds a harmonic sphere term for a copied atom selection.
   *
   * The reaction coordinate is the raw Cartesian distance to `center`, with no
   * periodic imaging. A computed squared distance at or below `1e-10` square
   * angstroms suppresses energy and gradient before activation, including at
   * the exact center.
   *
   * Indices and parameters are copied into owned host storage. No caller
   * storage is retained and no CUDA transfer or synchronization occurs. Device
   * mirrors are updated before use. Adding a term after initialization or
   * subscription does not require resubscription.
   *
   * @param[in] selection Selection representing exactly `getNumAtoms()` atoms.
   * A valid empty selection is a no-op after parameter validation and adds
   * neither a term nor an RCM center.
   * @param[in] center Three finite center coordinates in `[x, y, z]` order and
   * angstrom units.
   * @param[in] forceConstant Finite signed harmonic coefficient in kilocalories
   * per mole per square angstrom.
   * @param[in] offset Finite signed CHARMM DROFF value in angstroms; default
   * zero.
   * @param[in] activation SYMMETRIC, INSIDE, or OUTSIDE; default INSIDE.
   * @param[in] selectionMode ATOMWISE or RCM; default ATOMWISE. A nonempty RCM
   * selection requires a finite nonzero signed total and finite mass/total
   * ratios, even when `forceConstant` is zero.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the selection atom count, parameter, enum value, membership count, or RCM
   * mass requirement is invalid.
   * @throws std::bad_alloc If copied term, index, or diagnostic storage cannot
   * be allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success with a nonempty selection, one term is appended; existing
   * terms and outputs remain. Validation or allocation failure preserves terms.
   * @see addPlane
   * @see addCylinder
   * @see setMasses
   */
  void addSphere(const AtomSelection &selection,
                 const std::array<double, 3> &center,
                 const double forceConstant, const double offset = 0.0,
                 const Activation activation = Activation::INSIDE,
                 const SelectionMode selectionMode = SelectionMode::ATOMWISE);

  /**
   * @brief Adds a harmonic sphere term for one force-local atom index.
   *
   * Only the index is used; the AtomReference topology is not compared or
   * retained. Geometry, singular suppression, host copying, and deferred device
   * updates match the selection overload. The operation is valid after
   * initialization or subscription.
   *
   * @param[in] atom Borrowed reference with an index in `[0, getNumAtoms())`.
   * @param[in] center Three finite center coordinates in `[x, y, z]` order and
   * angstrom units.
   * @param[in] forceConstant Finite signed harmonic coefficient in kilocalories
   * per mole per square angstrom.
   * @param[in] offset Finite signed CHARMM DROFF value in angstroms; default
   * zero.
   * @param[in] activation SYMMETRIC, INSIDE, or OUTSIDE; default INSIDE.
   * @param[in] selectionMode ATOMWISE or RCM; default ATOMWISE. For RCM, the
   * selected atom must have a finite nonzero active mass.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the atom index, parameter, enum value, membership count, or RCM mass is
   * invalid.
   * @throws std::bad_alloc If copied term, index, or diagnostic storage cannot
   * be allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, one term is appended; existing terms and outputs remain.
   * @post Validation or allocation failure leaves installed terms unchanged.
   * @see AtomReference
   * @see addSphere
   */
  void addSphere(const AtomReference &atom, const std::array<double, 3> &center,
                 const double forceConstant, const double offset = 0.0,
                 const Activation activation = Activation::INSIDE,
                 const SelectionMode selectionMode = SelectionMode::ATOMWISE);

public:
  /**
   * @brief Replaces the owned mass vector for all current and future RCM terms.
   *
   * The entire vector and every installed RCM selection are validated before
   * replacement storage is allocated. Every RCM total must be finite and
   * nonzero, and each selected mass/total ratio must be finite. Signed and zero
   * individual masses, negative totals, and tiny nonzero totals are accepted
   * within that envelope. Atomwise terms ignore masses.
   *
   * The private stream is synchronized before replacement. Independent host and
   * device copies of masses and derived totals are prepared with synchronous
   * transfers before a nonthrowing commit. Caller storage is not retained.
   * Initialization or subscription does not restrict this operation, and
   * `reset()` and `clear()` preserve the resulting masses.
   *
   * @param[in] masses Exactly `getNumAtoms()` finite values in atom-index order
   * and atomic mass units, borrowed for the duration of the call.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the length differs, any mass is nonfinite, or an installed RCM total or
   * mass fraction is invalid.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if stream
   * synchronization, replacement allocation, transfer, or device
   * synchronization fails.
   * @throws std::bad_alloc If replacement host or diagnostic storage cannot be
   * allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, all RCM terms use the copied masses on their next
   * evaluation; installed terms and output owners are unchanged.
   * @post Failure before commit preserves active host/device masses and totals.
   * @warning Reacquire element pointers and iterators borrowed through
   * `getMasses()` after successful replacement.
   * @see getMasses
   * @see SelectionMode
   */
  void setMasses(const std::vector<double> &masses);

public:
  /**
   * @brief Returns the fixed atom count established during construction.
   *
   * @return Positive dimensionless number of atom entries in the mass and force
   * arrays. No CUDA transfer or synchronization occurs.
   */
  int getNumAtoms(void) const noexcept;

  /**
   * @brief Returns the number of currently installed geometric terms.
   *
   * @return Dimensionless term count, including duplicates and overlapping
   * selections. Accepted empty selections add no term. No CUDA transfer or
   * synchronization occurs.
   * @see reset
   */
  std::size_t getNumTerms(void) const noexcept;

  /**
   * @brief Returns read-only access to the owned active mass vector.
   *
   * @return Borrowed host vector containing `getNumAtoms()` masses in
   * atom-index order and atomic mass units. No transfer or synchronization
   * occurs.
   * @warning The reference does not extend the restraint's lifetime. Successful
   * `setMasses()` replaces element storage and invalidates borrowed element
   * pointers and iterators.
   * @see setMasses
   */
  const std::vector<double> &getMasses(void) const noexcept;

  /**
   * @brief Returns shared access to the private CUDA stream holder.
   *
   * The shared pointer owns the host scalar containing the handle. This
   * restraint remains responsible for destroying the underlying CUDA stream and
   * sets the scalar to `nullptr` during destruction.
   *
   * @return A copied shared owner of the stream holder. The underlying stream
   * is valid only while the restraint remains alive. No synchronization occurs.
   * @warning Do not replace the handle or destroy the stream through the
   * holder.
   * @see ForceManager::subscribe
   */
  std::shared_ptr<cudaStream_t> getStream(void);

  /**
   * @brief Returns shared ownership of fixed-point device gradient storage.
   *
   * The array contains `getNumAtoms()` vectors in structure-of-arrays layout.
   * Multiply raw `long long int` components by `INV_FORCE_SCALE` to recover
   * potential gradients in kilocalories per mole per angstrom. Physical forces
   * have the opposite sign.
   *
   * @return A copied shared owner of the restraint's `Force<AT>` object. No
   * transfer or synchronization occurs. Storage may outlive this restraint
   * while another shared owner remains.
   * @warning Order accesses after pending work on `getStream()` and do not
   * resize or concurrently modify this storage while the restraint or manager
   * is using it.
   * @see calcForce
   * @see ForceManager::subscribe
   */
  std::shared_ptr<Force<AT>> getForce(void);

  /**
   * @brief Returns shared ownership of energy and complete virial storage.
   *
   * The object contains the named energy term `mmfp` and the complete external
   * tensor `W_ab = sum_i r_i,a * g_i,b`, both in kilocalories per mole. Tensor
   * components are ordered XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ. The manager must
   * not add another coordinate-gradient reconstruction for this force.
   *
   * @return A copied shared owner of the restraint's `CudaEnergyVirial` object.
   * No transfer or synchronization occurs. Storage may outlive this restraint
   * while another shared owner remains.
   * @warning Copy device values to the host and synchronize before reading
   * them. A disabled calculation flag leaves its previous output stale.
   * @see calcForce
   * @see computesVirial
   */
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void);

public:
  /**
   * @brief Removes all geometric terms while preserving masses and output.
   *
   * Host term storage is emptied and device mirrors are marked for rebuilding
   * before use. No CUDA work or synchronization is performed. The box, active
   * masses, stream, force storage, and energy-virial owner remain unchanged.
   * This method is valid while subscribed and does not unsubscribe the force.
   *
   * @throws std::bad_alloc If the replacement empty offset vector cannot be
   * allocated.
   *
   * @post `getNumTerms() == 0`; subsequent evaluations add no contribution
   * until new terms are installed.
   * @post An allocation failure leaves installed terms unchanged.
   * @note Existing output is not cleared and earlier queued output work is not
   * waited for. Use `clear()` when zeroed output is required.
   * @see clear
   * @see setMasses
   */
  void reset(void);

  /**
   * @brief Stores orthorhombic box lengths for ForceManager compatibility.
   *
   * The three values are copied into host scalar state after validation. They
   * do not wrap coordinates, unwrap RCM selections, image displacements, or
   * rescale restraint references. No CUDA transfer or synchronization occurs.
   *
   * @param[in] boxDimensions Exactly three finite positive lengths in
   * `[x, y, z]` order and angstrom units, borrowed for this call.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the vector length is not three or any length is non-positive, NaN, or
   * infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the stored box is replaced. Validation failure preserves
   * the previous box and all terms and outputs.
   * @see initialize
   * @see supportsPBC
   */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

  /**
   * @brief Validates manager dimensions and prepares current device terms.
   *
   * This ForceManager hook checks the fixed atom count, stores the supplied
   * box, and synchronizes term mirrors when they are dirty. Rebuilding waits
   * for the private stream and uses synchronous host-to-device transfers. No
   * separate initialized flag is set and output owners are not replaced.
   *
   * @param[in] numAtoms Dimensionless manager atom count; it must equal
   * `getNumAtoms()`.
   * @param[in] boxDimensions Exactly three finite positive orthorhombic box
   * lengths in `[x, y, z]` order and angstrom units.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the atom count differs or the box contract is violated.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if device
   * allocation, transfer, or synchronization fails.
   * @throws std::bad_alloc If derived host storage or an error diagnostic
   * cannot be allocated.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, device terms match current host terms and the supplied
   * box is stored. Terms, masses, and output values are preserved.
   * @warning Preparation is not fully transactional: the box is stored before
   * device rebuilding, and allocation or transfer failure can leave mirrors
   * partly rebuilt. They remain dirty and must rebuild successfully before use.
   * @see setBoxDimensions
   * @see calcForce
   */
  void initialize(const int numAtoms, const std::vector<double> &boxDimensions);

  /**
   * @brief Enqueues clearing of all force, energy, and virial outputs.
   *
   * This overload delegates to `clear(true, true)`. Clears use the private
   * stream and return without synchronizing. Terms, masses, and box values are
   * preserved.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if an output
   * clear reports an immediate CUDA failure.
   *
   * @post After the private stream completes the queued clears, all force,
   * energy, and virial outputs are zero.
   * @warning A CUDA failure can leave some clears queued and others incomplete.
   * @see reset
   * @see getStream
   */
  void clear(void);

  /**
   * @brief Clears gradients and the requested energy and virial outputs.
   *
   * ForceManager uses this overload before calculation. Clears are enqueued on
   * the private stream without synchronizing before return. Terms, masses, and
   * box values are preserved.
   *
   * @param[in] calcEnergy Whether to zero energy storage. A false value
   * preserves the previous energy, which may be stale for new coordinates.
   * @param[in] calcVirial Whether to zero virial storage. A false value
   * preserves the previous tensor, which may be stale for new coordinates.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if an output
   * clear reports an immediate CUDA failure.
   *
   * @post After the private stream completes the queued clears, gradients and
   * the requested outputs are zero; disabled outputs retain their values.
   * @warning A CUDA failure can leave some clears queued and others incomplete.
   * @see calcForce
   * @see reset
   */
  void clear(const bool calcEnergy, const bool calcVirial);

  /**
   * @brief Accumulates native gradients and optional energy and full virial.
   *
   * Dirty term mirrors are rebuilt before evaluation. Term gradients are summed
   * at each atom in insertion order, converted through CT, and added to the
   * fixed-point output. Physical force is the negative of that stored gradient.
   * The tensor uses raw positions and the new unconverted double gradient sum,
   * including mass-distributed RCM gradients.
   *
   * Evaluation and output-range checks are staged on the private stream. The
   * method synchronizes that stream to validate the result, then enqueues the
   * output commit. Energy and virial are accumulated only when their flags are
   * true; disabled outputs retain previous values and can be stale.
   *
   * @param[in] xyzq Borrowed CUDA-device pointer to at least `getNumAtoms()`
   * contiguous `float4` records in `[x, y, z, charge]` order. Coordinates use
   * angstroms; the charge component is ignored. With installed terms the
   * pointer must be non-null and remain valid until the private stream has
   * finished using it.
   * @param[in] calcEnergy Whether to add the evaluated energy to the device
   * `mmfp` term. The energy-zero gradient gate and finite-arithmetic checks
   * still apply when this flag is false.
   * @param[in] calcVirial Whether to add all nine external virial components in
   * `W_ab = sum_i r_i,a * g_i,b` order.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * terms exist and `xyzq` is null.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if a term is
   * unsupported, evaluated arithmetic is nonfinite, gradient conversion is
   * outside the fixed-point range, or accumulation would overflow.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if allocation,
   * transfer, a kernel launch, or synchronization fails.
   * @throws std::bad_alloc If rebuilding host storage or constructing an error
   * diagnostic cannot allocate storage.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @pre With installed terms, `xyzq` is device-accessible, correctly sized,
   * ready for this stream to read, and not subject to concurrent writes.
   * @pre Order `clear()` before this call when fresh output is required.
   * @post A numerical rejection leaves the pre-existing public outputs intact.
   * @post Successful return with terms means output commitment was enqueued;
   * the final commit may still be executing on the private stream.
   * @note With no terms, the method returns without validating `xyzq`, clearing
   * outputs, or synchronizing. It adds no contribution.
   * @warning CUDA failures do not provide an output rollback guarantee.
   * Complete the private stream before consuming committed outputs on another
   * stream.
   * @see clear
   * @see getForce
   * @see getEnergyVirial
   */
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial);

  /**
   * @brief Reports the P1-only compatibility of geometric restraints.
   *
   * ForceView uses this query during manager subscription and PBC mutation. The
   * query does not change state or perform CUDA work.
   *
   * @param[in] pbc Periodic-boundary value to inspect.
   * @return `true` only for `PBC::P1`; `false` for `PBC::UNSET`, `PBC::P21`,
   * and unnamed enum values. UNSET is an invalid sentinel, not an aperiodic
   * configuration.
   * @see ForceView::supportsPBC
   * @see ForceManager::setPeriodicBoundaryCondition
   */
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
