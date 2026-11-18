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

#include "AtomSelection.h"
#include "CharmmPSF.h"
#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "Force.h"
#include "cuda_utils.h"

#include <array>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

/**
 * @brief Computes a CHARMM CONS HMCM harmonic restraint on a group center.
 *
 * The selected atoms define either a uniformly weighted geometric center or a
 * mass-weighted center. As in CHARMM, the center is built from the raw
 * Cartesian coordinates, without unwrapping. Let `w_i` be one when mass
 * weighting is disabled, or, when it is enabled, the stored mass, which
 * construction copies from a CharmmPSF and `setMasses()` replaces:
 *
 * @code
 * W = sum over selected atoms i of w_i
 * C = (1 / W) * sum over selected atoms i of w_i * r_i
 * @endcode
 *
 * A three-element mask selects the Cartesian components of the raw
 * center-to-reference displacement `d = C - R`; inactive components are zero,
 * so the center is restrained to a point, a line, or a plane. For force
 * constant `k` and reference distance `r0`:
 *
 * @code
 * s = norm(d)
 * U_hmcm = k * (s - r0)^2
 * gradient_i = w_i * (d * 2 * k / W * (s - r0) / s)
 * @endcode
 *
 * Both are evaluated in CHARMM's operation order. A zero `s` gives a zero
 * gradient; CHARMM also gives zero there when `r0` is zero, but divides by
 * zero otherwise. The force array receives `gradient_i`, which apoCHARMM's
 * force aggregation subtracts to obtain the physical restoring force. Energy
 * is accumulated in the named term `hmcm`. As in CHARMM, force constants,
 * reference distances, and masses may have either sign, provided the selected
 * total weight is finite and nonzero. The restraint contributes no virial, and
 * the stored box lengths do not affect evaluation.
 *
 * Configuration is split between host scalar state and host/device arrays.
 * Selection or weight changes rebuild the host selected-index and
 * selected-weight arrays; `initialize()` and `calcForce()` transfer them and
 * resize device storage before use. `calcForce()` enqueues a three-kernel
 * pipeline on a private CUDA stream and does not synchronize before returning.
 *
 * @tparam AT Device force-accumulator representation. The library explicitly
 * instantiates `long long int`, using apoCHARMM's fixed-point force scale.
 * @tparam CT Representation of the force arithmetic, including the center
 * reductions and the energy value, which is converted to double before it is
 * added to the energy storage. The library explicitly instantiates `float` and
 * `double`.
 *
 * @warning The class provides no internal host-thread synchronization. Callers
 * must serialize configuration, clearing, force evaluation, and destruction.
 * @warning Do not copy or copy-assign this class. Its compiler-generated copy
 * operations share stream and output owners without defining an independent
 * CUDA-resource lifecycle.
 * @warning Call `clear()` before the first calculation and before every
 * calculation that requires fresh rather than accumulated output.
 * @warning `CharmmContext` recenters every bond-connected group whose center
 * leaves the half box. A selection spanning several groups, or a reference
 * point outside the centered cell, can therefore move the raw center by a box
 * length, as CHARMM's image centering can.
 *
 * @see harmonic_center_of_mass_restraint_force
 * @see AtomSelection
 * @see ForceManager
 */
template <typename AT, typename CT> class HarmonicCenterOfMassRestraintForce {
private:
  using CT4 = typename CudaVectorType<CT, 4>::type;

public:
  /**
   * @brief Reports that this restraint does not contribute to virial
   * aggregation.
   *
   * CHARMM evaluates CONS HMCM after it forms its internal virial. Likewise,
   * ForceManager passes `calcVirial == false` to this force and excludes it
   * from the aggregate virial.
   */
  static constexpr bool contributesVirial = false;

  /**
   * @brief Names the CHARMM energy term of this restraint.
   *
   * Like CHARMM's `ETERM(HMCM)`, the energy is reported as `hmcm` and printed
   * in the `RESTR>` column that CHARMM fills with `CQRT + PATH + HMCM`.
   */
  static constexpr const char *energyTerm = "hmcm";

public:
  /**
   * @brief Prevents construction without a PSF atom-count and mass snapshot.
   *
   * @see CharmmPSF
   */
  HarmonicCenterOfMassRestraintForce(void) = delete;

  /**
   * @brief Constructs a restraint from a PSF atom-count and mass snapshot.
   *
   * The positive atom count and complete mass vector are copied. The PSF and
   * its shared owner are borrowed only for this call and are not retained.
   * Every atom is initially selected. The force constant and reference
   * distance are zero, mass weighting is disabled, the stored masses are the
   * copied PSF masses, the reference position is `(0, 0, 0)`, every Cartesian
   * mask component is active, and box dimensions are unset at zero. As with
   * CHARMM's WEIG or MASS keyword, enabling mass weighting then uses the PSF
   * masses.
   *
   * Construction copies the masses into host storage, builds the host
   * selected-index and selected-weight arrays, which the first `initialize()`
   * or `calcForce()` transfers, and allocates partial-sum and restraint-state
   * storage. It also creates the `hmcm` energy term, a `Force<AT>` sized for
   * the atom count, and a private CUDA stream. Output force and energy storage
   * must be cleared before use.
   *
   * @param[in] psf Non-null PSF with an initialized positive atom count and
   * exactly one finite mass, in atomic mass units, per atom. As in CHARMM, a
   * mass may have either sign. The count remains fixed for the lifetime of
   * the restraint.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the PSF atom count is negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `psf` is null, its atom count is zero, or a mass is NaN or infinite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the PSF
   * mass-vector length differs from its atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if CUDA
   * allocation, transfer, synchronization, force allocation, or stream
   * creation fails.
   * @throws std::bad_alloc If host storage, shared ownership state, or an error
   * diagnostic cannot be allocated.
   * @throws std::length_error If requested storage or an error diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post `getForce()->size() == psf->getNumAtoms()`.
   * @post Every atom is selected with uniform weight one, and later changes to
   * the PSF masses do not affect the restraint.
   * @see setMassWeighting
   * @see setMasses
   */
  explicit HarmonicCenterOfMassRestraintForce(
      const std::shared_ptr<const CharmmPSF> &psf);

  /**
   * @brief Destroys the private CUDA stream and releases owned resources.
   *
   * CUDA stream destruction is attempted without propagating cleanup failure.
   * A shared stream holder previously returned by `getStream()` can outlive
   * this object, but its stored stream value is reset during destruction.
   * Shared force and energy objects can also outlive this restraint while
   * another owner exists, but no restraint remains to update them.
   *
   * @post The private CUDA stream is no longer valid.
   */
  ~HarmonicCenterOfMassRestraintForce(void) noexcept;

public:
  /**
   * @brief Sets the nonempty atom selection used to define the group center.
   *
   * The selection is copied. Its selected indices are materialized in
   * ascending order, and selected weights are rebuilt from either one or the
   * configured masses. The next `initialize()` or `calcForce()` copies both
   * arrays to device memory, resizes partial-sum storage to one element per
   * 256-thread calculation block, and clears internal state.
   *
   * @param[in] selection Selection with exactly the restraint's fixed atom
   * count and at least one selected atom. The object is borrowed for the call,
   * copied, and not retained by reference.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the atom count differs, no atom is selected, or mass weighting is active
   * and the selected masses do not have a finite nonzero total with finite
   * fractions.
   * @throws std::bad_alloc If selection, index, weight, or diagnostic storage
   * cannot be allocated.
   * @throws std::length_error If copied storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the restraint owns an independent selection copy and
   * the selected host arrays describe exactly that selection.
   * @warning Failure after selection assignment begins can leave the copied
   * selection or selected count newer than the host index and weight arrays.
   * @warning The next transfer can invalidate device pointers previously
   * borrowed from the selected arrays or partial-sum storage.
   */
  void setSelection(const AtomSelection &selection);

  /**
   * @brief Sets the scalar harmonic force constant.
   *
   * The value is retained in host scalar state and passed by value to later
   * kernel launches. This method performs no CUDA transfer or synchronization.
   *
   * @param[in] forceConstant Finite force constant in kilocalories per mole per
   * square angstrom. As in CHARMM, a negative value is accepted.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `forceConstant` is NaN or infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, subsequent evaluations use `forceConstant`.
   * @note Zero disables force evaluation.
   */
  void setForceConstant(const double forceConstant);

  /**
   * @brief Sets a Cartesian reference position with all components active.
   *
   * This overload delegates to the masked overload with `{1, 1, 1}`. The
   * values are copied into host scalar state; no caller storage is retained and
   * no CUDA transfer occurs.
   *
   * @param[in] referencePosition Exactly three finite Cartesian coordinates in
   * `[x, y, z]` order and angstrom units.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the vector length is not three or any component is NaN or infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, every Cartesian component participates in the restraint.
   */
  void setReferencePosition(const std::vector<double> &referencePosition);

  /**
   * @brief Sets the Cartesian reference position and active-component mask.
   *
   * Active components use the raw center-to-reference displacement. Inactive
   * components contribute neither energy nor gradient. The position and mask
   * are copied into host scalar state; no CUDA transfer occurs.
   *
   * @param[in] referencePosition Exactly three finite coordinates in
   * `[x, y, z]` order and angstrom units.
   * @param[in] referenceMask Exactly three dimensionless values in
   * `[x, y, z]` order. Each value must be zero or one, and at least one value
   * must be one.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if either vector length is not three,
   * a position is NaN or infinite, a mask value is not zero or one, or every
   * mask value is zero.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, subsequent evaluations use independent scalar copies of
   * the supplied position and mask.
   */
  void setReferencePosition(const std::vector<double> &referencePosition,
                            const std::vector<int> &referenceMask);

  /**
   * @brief Sets the target radial distance from the reference position.
   *
   * The energy is `k * (s - referenceDistance)^2` for the masked displacement
   * magnitude `s`. The scalar is stored on the host and no CUDA transfer
   * occurs.
   *
   * @param[in] referenceDistance Finite target distance in angstroms. As in
   * CHARMM, a negative value is accepted.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `referenceDistance` is NaN or infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, subsequent evaluations use `referenceDistance`.
   * @note For a nonzero target and an exactly zero current displacement, the
   * implementation assigns a zero gradient at the nondifferentiable point,
   * where CHARMM divides by zero.
   */
  void setReferenceDistance(const double referenceDistance);

  /**
   * @brief Sets per-atom masses and enables mass weighting.
   *
   * The full atom-order array is copied into owned host storage. The
   * selected weights are then rebuilt from the copied masses; the next
   * `initialize()` or `calcForce()` transfers them to device memory. Only
   * relative selected masses affect the normalized center; multiplying every
   * selected mass by the same positive factor has no effect.
   *
   * @param[in] masses Exactly one finite value per atom, in atom order and
   * normally atomic mass units. As in CHARMM, a value may have either sign.
   * The vector is borrowed for the call, copied, and not retained by
   * reference. Selected masses must have a finite nonzero total and finite
   * fractions.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the vector length differs from the atom count, a mass is NaN or infinite,
   * or the selected masses do not have a finite nonzero total with finite
   * fractions.
   * @throws std::bad_alloc If host or diagnostic storage cannot be allocated.
   * @throws std::length_error If copied storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, mass weighting is enabled and the selected host weights
   * correspond to `masses`.
   * @warning Failure after mass assignment begins can leave new masses or a new
   * weighting mode paired with older selected host arrays.
   * @warning The next transfer can invalidate previously borrowed selected
   * array or partial-sum device pointers.
   */
  void setMasses(const std::vector<double> &masses);

  /**
   * @brief Enables mass weighting or restores uniform weighting.
   *
   * The selected-weight array is rebuilt in host memory; the next
   * `initialize()` or `calcForce()` copies it to device memory. `true` uses the
   * stored per-atom masses, which are the PSF masses copied at construction
   * until `setMasses()` replaces them; `false` assigns every selected atom
   * weight one. The stored mass array is retained when weighting is disabled.
   *
   * @param[in] useMassWeighting `true` to use configured masses or `false` to
   * use uniform dimensionless weights.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * weighting is enabled while the selected masses do not have a finite nonzero
   * total with finite fractions.
   * @throws std::bad_alloc If selected-index, selected-weight, or diagnostic
   * storage cannot be allocated.
   * @throws std::length_error If rebuilt storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the selected host weights match the requested mode.
   * @warning Failure after the mode flag changes can leave the mode and
   * selected host arrays inconsistent.
   * @warning The next transfer can invalidate previously borrowed selected
   * array or partial-sum device pointers.
   */
  void setMassWeighting(const bool useMassWeighting);

  /**
   * @brief Sets the orthorhombic periodic box lengths.
   *
   * The three values are copied into host scalar state for the manager
   * lifecycle. As in CHARMM, evaluation uses raw coordinates, so the box does
   * not affect the restraint. No CUDA transfer or synchronization occurs, and
   * changing the box does not rescale the reference position.
   *
   * @param[in] boxDimensions Exactly three finite positive lengths in
   * `[x, y, z]` order and angstrom units.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the vector length is not three or
   * any length is non-positive, NaN, or infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the stored box equals `boxDimensions`.
   * @warning Triclinic box vectors and angles are not represented by this API.
   */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

public:
  /**
   * @brief Returns shared ownership of energy and virial storage.
   *
   * No transfer or synchronization occurs. The object contains the named
   * energy term `hmcm`. This force never writes the object's virial, which
   * remains zero.
   *
   * @return A copied shared owner of the restraint's `CudaEnergyVirial` object.
   * The object can outlive this restraint, but no restraint remains to update
   * it after destruction.
   */
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void);

  /**
   * @brief Returns shared ownership of device force storage.
   *
   * The force array contains `m_NumAtoms` vectors in structure-of-arrays
   * component layout. For the public fixed-point specialization, multiply raw
   * `long long int` values by `INV_FORCE_SCALE` to recover potential-gradient
   * values in kilocalories per mole per angstrom.
   *
   * @return A copied shared owner of the restraint's `Force<AT>` object. No
   * transfer or synchronization occurs.
   */
  std::shared_ptr<Force<AT>> getForce(void);

  /**
   * @brief Returns shared access to the private CUDA stream holder.
   *
   * The returned shared pointer owns only the host scalar containing the stream
   * handle. This restraint remains responsible for destroying the underlying
   * CUDA stream and resets the scalar during destruction.
   *
   * @return A copied shared owner of the stream holder. The CUDA stream remains
   * valid only while the restraint remains alive.
   * @warning Do not destroy the CUDA stream through the returned holder.
   */
  std::shared_ptr<cudaStream_t> getStream(void);

public:
  /**
   * @brief Initializes the restraint for an atom count and periodic box.
   *
   * This is the `ForceManager` initialization hook. It verifies that the
   * manager atom count matches construction, delegates to `setBoxDimensions()`,
   * and transfers changed selected arrays to the device. It does not allocate a
   * new force array, rebuild the selection, clear outputs, or set a separate
   * initialized flag.
   *
   * @param[in] numAtoms Dimensionless atom count supplied by the manager. It
   * must equal the fixed construction count.
   * @param[in] boxDimensions Exactly three finite positive orthorhombic box
   * lengths in `[x, y, z]` order and angstrom units.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the atom count differs or the box
   * contract is violated.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if transferring
   * the selected arrays or resizing reduction storage fails.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the stored box equals `boxDimensions`, and the device
   * selected arrays and reduction storage match the host arrays.
   */
  void initialize(const int numAtoms, const std::vector<double> &boxDimensions);

  /**
   * @brief Enqueues clearing of force, energy, and virial outputs.
   *
   * Both clears are issued on the private CUDA stream. The method checks
   * immediate CUDA failures but does not synchronize before returning.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if output
   * clearing or an immediate CUDA operation fails.
   *
   * @post After the private stream reaches the queued clears, force, energy,
   * and virial storage is zero.
   */
  void clear(void);

  /**
   * @brief Enqueues center calculation, state calculation, and force updates.
   *
   * Changed selected arrays are first copied to the device on the private
   * stream, which is then synchronized. The method then reads `m_NumAtoms`
   * device `float4` coordinate-charge records, computes the selected center
   * from raw coordinates, computes the energy and the CHARMM gradient factors,
   * and distributes them to the selected atoms in proportion to their weights.
   * All three kernels use the private stream and return after immediate launch
   * checks.
   *
   * Force output is always accumulated. Energy is accumulated only when
   * `calcEnergy` is true; a disabled energy retains its previous value and can
   * therefore be stale.
   *
   * @param[in] xyzq Borrowed non-NULL CUDA-device pointer to at least
   * `m_NumAtoms` contiguous `float4` records in `[x, y, z, charge]` order.
   * Coordinates use angstroms and the charge component is ignored. The pointer
   * must remain valid until the private stream completes all queued work.
   * @param[in] calcEnergy Whether to add the current restraint energy to the
   * device `hmcm` term.
   * @param[in] calcVirial Accepted for the force-manager interface but ignored
   * because this restraint contributes no virial.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if transferring
   * changed selected arrays fails or any kernel launch reports an immediate
   * CUDA failure.
   *
   * @pre `xyzq` is device-accessible, correctly sized, and does not overlap an
   * incompatible concurrent write.
   * @pre `clear()` has been ordered before this call when fresh output is
   * required.
   * @post Successful return means all requested kernels were enqueued; it does
   * not mean device execution has completed.
   * @note If the force constant is zero, the method returns without
   * transferring data, reading `xyzq`, or modifying outputs.
   * @warning A later asynchronous CUDA failure is reported only by a subsequent
   * CUDA synchronization or CUDA API operation.
   * @warning A launch failure after earlier kernels were enqueued can leave a
   * partially updated calculation once the stream completes.
   */
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial);

private:
  /**
   * @brief Rebuilds the selected indices and weights in host memory.
   *
   * The method materializes ascending selected atom indices, chooses mass or
   * uniform weights, verifies a finite nonzero selected total with finite
   * fractions, stores the indices and weights in the host mirrors, and marks
   * them for transfer.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the selected total weight is not finite and nonzero, or a selected weight
   * fraction is not finite.
   * @throws std::bad_alloc If temporary or host storage cannot be allocated.
   * @throws std::length_error If temporary or host storage exceeds an
   * implementation-defined limit.
   *
   * @post On success, the selected host arrays match `m_Selection` and
   * `m_UseMassWeighting`.
   * @warning The method is sequential rather than transactional.
   */
  void updateSelectedAtoms(void);

  /**
   * @brief Transfers dirty selected arrays and resizes reduction storage.
   *
   * When the selected arrays have changed, the method copies the selected
   * indices and weights to device memory on the private stream, resizes the
   * block partial sums, clears the device copies of the partial and
   * restraint-state records on the same stream, and then synchronizes that
   * stream. Resizing can invalidate previously borrowed device pointers through
   * reallocation.
   */
  void syncDeviceData(void);

private:
  /** Fixed dimensionless atom count established during construction. */
  int m_NumAtoms;
  /** Current selected count; updated before selected-array replacement. */
  int m_NumSelected;

  /** Harmonic coefficient in kilocalories per mole per square angstrom. */
  double m_ForceConstant;
  /** Target center distance in angstroms. */
  double m_ReferenceDistance;
  /** Selects configured masses or uniform weights. */
  bool m_UseMassWeighting;

  /** Owned host/device selected indices in ascending atom order. */
  CudaContainer<int> m_AtomIndices;
  /** Owned host/device selected weights corresponding to `m_AtomIndices`. */
  CudaContainer<double> m_AtomWeights;
  /** Owned host per-atom masses in atom order. */
  std::vector<double> m_Masses;

  /** One device reduction record per 256-thread selected-atom block. */
  CudaContainer<CT4> m_PartialSums;

  /**
   * One device state record whose components x, y, and z store the per-axis
   * gradient factors `d * 2 * k / W * (s - r0) / s`.
   */
  CudaContainer<CT4> m_RestraintState;

  /** Host-only orthorhombic box lengths in angstroms, in `[x, y, z]` order. */
  std::array<double, 3> m_BoxDimensions;

  /** True when the selected host arrays are newer than their device mirrors. */
  bool m_DeviceDataDirty;

  /** Host-only reference position in angstroms, in `[x, y, z]` order. */
  std::array<double, 3> m_ReferencePosition;
  /** Dimensionless `[x, y, z]` activation flags, each zero or one. */
  std::array<int, 3> m_ReferenceMask;

  /** Owned host selection used when rebuilding selected arrays. */
  AtomSelection m_Selection;

  /** Shared owner of the `hmcm` energy storage. */
  std::shared_ptr<CudaEnergyVirial> m_EnergyVirial;
  /** Shared owner of device structure-of-arrays force storage. */
  std::shared_ptr<Force<AT>> m_Forces;
  /** Shared host holder for the privately owned CUDA stream handle. */
  std::shared_ptr<cudaStream_t> m_Stream;
};
