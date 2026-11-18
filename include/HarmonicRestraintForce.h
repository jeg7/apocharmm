// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "AtomSelection.h"
#include "CharmmPSF.h"
#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "Force.h"

#include <array>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

/**
 * @brief Computes CHARMM CONS HARM ABSOLUTE positional restraints for atoms.
 *
 * For atom `i`, the restraint stores an absolute reference position
 * `(x0_i, y0_i, z0_i)`, a force constant `k_i`, and a mass or weight `m_i`.
 * One exponent `n` and three scale factors `(sx, sy, sz)` apply to every atom,
 * like CHARMM's EXPOnent and XSCAle, YSCAle, and ZSCAle for one restraint set.
 * With `C_i = k_i * m_i` and the raw displacement
 *
 * @code
 * dx_i = x_i - x0_i,  dy_i = y_i - y0_i,  dz_i = z_i - z0_i
 * S_i = sx * dx_i * dx_i + sy * dy_i * dy_i + sz * dz_i * dz_i
 * @endcode
 *
 * each atom with nonzero `C_i` contributes `E_i` and the gradient
 * `(dx_i * D_i * sx, dy_i * D_i * sy, dz_i * D_i * sz)`, where
 *
 * @code
 * n == 2:   E_i = C_i * S_i              D_i = 2 * C_i
 * n == 4:   E_i = C_i * S_i * S_i        D_i = C_i * 4 * S_i
 * n == 3:   E_i = C_i * S_i * sqrt(S_i)  D_i = C_i * 3 * sqrt(S_i)
 * n == 1:   E_i = C_i * sqrt(S_i)        D_i = C_i / sqrt(S_i), or zero
 *                                        when S_i <= 1e-12
 * n <= 0:   E_i = 0                      D_i = 0
 * other n:  P_i = C_i * S_i^((n - 4) / 2)
 *           E_i = P_i * S_i * S_i        D_i = P_i * n * S_i
 * @endcode
 *
 * These are CHARMM's branches and operation order; as in CHARMM, there is no
 * factor of one half. apoCHARMM integration code subtracts this stored
 * potential gradient, so the corresponding physical restoring force has the
 * opposite sign. Contributions are accumulated atomically into the force and
 * optional `harm` energy buffers.
 *
 * Selection is implemented by setting the force constant of every unselected
 * atom to zero. Current coordinates are read from a borrowed device `float4`
 * array; its `w` charge component is ignored. Reference coordinates and
 * current coordinates are compared directly without minimum-image wrapping.
 * As in CHARMM, per-atom force constants and masses may have either sign.
 *
 * Construction copies the atom count of a CharmmPSF and allocates per-atom
 * host/device parameter mirrors, one CUDA stream, one fixed-point force array,
 * and one energy-virial object. Parameter setters copy caller data into the
 * host mirrors only; `initialize()` and `calcForce()` transfer changed mirrors
 * to the device before they are used. Force calculation and output clearing
 * enqueue work on the private stream without synchronizing it before return.
 *
 * @tparam AT Device force-accumulator representation. The library explicitly
 * instantiates only `long long int`, using apoCHARMM's fixed-point force scale.
 * @tparam CT Representation of the force arithmetic, including each atom's
 * energy value, which is converted to double before it is summed; the energy
 * sums and the stored energy are double. The library explicitly instantiates
 * `float` and `double`.
 *
 * @warning The class provides no internal host-thread synchronization. Do not
 * overlap configuration, force calculation, clearing, or destruction.
 * @warning Do not copy or copy-assign this class. The current
 * compiler-generated operations combine copied parameter containers with
 * shared stream, force, and energy owners and do not define a safe independent
 * object lifecycle.
 * @warning Call `clear()` before the first calculation and before any
 * calculation that requires fresh rather than accumulated output.
 *
 * @see harmonic_restraint_force
 * @see AtomSelection
 * @see ForceManager
 */
template <typename AT, typename CT> class HarmonicRestraintForce {
public:
  /**
   * @brief Reports that this restraint does not contribute to virial
   * aggregation.
   *
   * CHARMM evaluates absolute CONS HARM restraints after it forms its internal
   * virial, so ForceManager excludes this force from the aggregate virial.
   */
  static constexpr bool contributesVirial = false;

  /**
   * @brief Names the CHARMM energy term of this restraint.
   *
   * Like CHARMM's CONS HARM energy, `ETERM(CHARM)`, the energy is reported as
   * `harm` and printed in the `HARMonic` column of the `CONSTR>` line.
   */
  static constexpr const char *energyTerm = "harm";

public:
  /**
   * @brief Prevents construction without a PSF atom count.
   *
   * @see CharmmPSF
   */
  HarmonicRestraintForce(void) = delete;

  /**
   * @brief Constructs a harmonic restraint from a PSF atom count.
   *
   * Only the positive atom count is copied; PSF masses are not read. The PSF
   * and its shared owner are borrowed only for this call and are not retained.
   * All atoms are initially selected. Force constants and reference coordinates
   * are initialized to zero, per-atom weights are initialized to one, the
   * exponent is two, every scale factor is one, and the stored box dimensions
   * are initialized to zero. The unit weights match CHARMM CONS HARM without
   * the MASS keyword; `setMasses()` supplies masses. Host and device parameter
   * mirrors are coherent when construction succeeds.
   *
   * The constructor also creates a private CUDA stream, allocates a
   * `Force<AT>` with one active vector per atom, and creates a
   * `CudaEnergyVirial` containing the named energy term `harm`. Output force
   * and energy storage must be cleared before its first use.
   *
   * @param[in] psf Non-null PSF with an initialized positive atom count. The
   * count remains fixed for the lifetime of the restraint.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the PSF atom count is negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `psf` is null or its atom count is zero.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if CUDA
   * allocation, parameter initialization, force allocation, transfer,
   * synchronization, or stream creation fails.
   * @throws std::bad_alloc If host storage, shared ownership state, or an
   * error diagnostic cannot be allocated.
   * @throws std::length_error If requested host storage or an error diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post `getForce()->size() == psf->getNumAtoms()`.
   * @post Every atom is selected, every force constant is zero, every
   * reference position is `(0, 0, 0)`, every weight is one, the exponent is
   * two, and every scale factor is one.
   * @see setMasses
   */
  explicit HarmonicRestraintForce(const std::shared_ptr<const CharmmPSF> &psf);

  /**
   * @brief Destroys the private CUDA stream and releases owned resources.
   *
   * The underlying CUDA stream is destroyed without propagating cleanup
   * failures. A `std::shared_ptr<cudaStream_t>` previously returned by
   * `getStream()` may keep the host-side stream holder alive, but the stored
   * CUDA stream value is set to `nullptr` during destruction.
   *
   * Shared force and energy objects returned by `getForce()` and
   * `getEnergyVirial()` may remain allocated while another shared owner
   * exists. They no longer have an owning restraint capable of calculating
   * into them.
   *
   * @post The private CUDA stream is no longer valid.
   */
  ~HarmonicRestraintForce(void) noexcept;

public:
  /**
   * @brief Sets the atoms that may receive harmonic restraint constants.
   *
   * The selection is copied and is not retained. Every atom that is
   * unselected by the new value has its stored force constant set to zero.
   * Selecting an atom does not restore a force constant that was zeroed by an
   * earlier selection change; call `setForceConstant()` or
   * `setForceConstants()` after expanding a selection.
   *
   * The updated force constants are stored in the host mirror; the next
   * `initialize()` or `calcForce()` transfers them to the device.
   *
   * @param[in] selection Selection with exactly the restraint's fixed atom
   * count. The object is borrowed for the call and remains unchanged.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the represented atom count
   * differs from the restraint atom count.
   * @throws std::bad_alloc If selection copying or diagnostic construction
   * cannot allocate storage.
   * @throws std::length_error If copied selection storage or a diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post On success, the restraint owns an independent copy of `selection`
   * and every unselected atom has a zero force constant.
   * @warning The operation is not transactional after selection copying
   * begins. An allocation failure can leave the copied selection and the
   * host constants representing different stages.
   */
  void setSelection(const AtomSelection &selection);

  /**
   * @brief Sets one force constant for every currently selected atom.
   *
   * The method first sets every per-atom force constant to zero and then
   * assigns `forceConstant` to the selected atoms. Unselected atoms therefore
   * remain disabled.
   *
   * With weights supplied in atomic mass units, `forceConstant` has units of
   * kilocalories per mole per square angstrom per atomic mass unit. When every
   * weight is one, its numerical effect is the ordinary per-atom harmonic
   * constant in kilocalories per mole per square angstrom.
   *
   * @param[in] forceConstant Finite, non-negative force constant applied to
   * every selected atom. CHARMM replaces a negative FORCE value with zero and
   * warns; this method rejects it instead.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `forceConstant` is negative, NaN, or infinite.
   * @throws std::bad_alloc If selected-index storage or an error diagnostic
   * cannot be allocated.
   * @throws std::length_error If selected-index storage or a diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post On success, selected atoms have `forceConstant` and all other atoms
   * have zero force constants.
   * @note The constants are stored in the host mirror; the next `initialize()`
   * or `calcForce()` transfers them to the device.
   * @warning A failure after the zero fill can leave every host constant zero.
   */
  void setForceConstant(const double forceConstant);

  /**
   * @brief Sets one force constant per atom.
   *
   * The vector is indexed in atom order and must contain exactly one value for
   * every atom. Only values belonging to currently selected atoms are
   * validated and used. Values at unselected indices are ignored, and the
   * corresponding stored constants are set to zero.
   *
   * With weights supplied in atomic mass units, each selected value has units
   * of kilocalories per mole per square angstrom per atomic mass unit.
   *
   * @param[in] forceConstants Contiguous host vector in atom order. Its length
   * must equal the fixed atom count. Every selected value must be finite; as
   * with CHARMM's WEIGhting option, it may have either sign. The vector is
   * borrowed for the call and is not retained.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the vector length differs from the atom count or a selected value is NaN or
   * infinite.
   * @throws std::bad_alloc If selected-index storage or diagnostic
   * construction cannot allocate storage.
   * @throws std::length_error If selected-index storage or a diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post On success, every selected atom has the corresponding input value
   * and every unselected atom has a zero force constant.
   * @post A validation failure leaves the stored constants unchanged.
   * @note The constants are stored in the host mirror; the next `initialize()`
   * or `calcForce()` transfers them to the device.
   */
  void setForceConstants(const std::vector<double> &forceConstants);

  /**
   * @brief Sets one Cartesian reference position per atom.
   *
   * The positions are copied into the host reference mirror in atom order;
   * the next `initialize()` or `calcForce()` transfers the array to the
   * device. Masses are stored separately and are not changed.
   *
   * Reference positions are absolute. They are not wrapped, imaged, or
   * rescaled when box dimensions change.
   *
   * @param[in] referenceCoordinates Contiguous host vector containing exactly
   * one finite `double3` position per atom, in angstroms. The vector is
   * borrowed for the call and is not retained.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the vector length differs from the atom count or any Cartesian component is
   * NaN or infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the host reference positions equal the supplied
   * values, and the masses are unchanged.
   * @warning Validation and host mutation are interleaved by atom. If a later
   * coordinate is invalid, an earlier host prefix may already contain new
   * values, which reach the device with the next transfer.
   */
  void
  setReferenceCoordinates(const std::vector<double3> &referenceCoordinates);

  /**
   * @brief Sets reference positions from nested Cartesian rows.
   *
   * The outer vector must contain one row per atom. Every row must contain
   * exactly three values in `[x, y, z]` order and angstrom units. The method
   * copies the rows into a temporary `double3` vector and delegates to the
   * primary reference-coordinate setter.
   *
   * @param[in] referenceCoordinates Host rows in atom order. The outer vector
   * must have the fixed atom count, and every row must have exactly three
   * finite values. No caller storage is retained.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the row count differs from the atom count, a row does not have exactly
   * three values, or a Cartesian value is NaN or infinite.
   * @throws std::bad_alloc If the temporary coordinate vector or an error
   * diagnostic cannot be allocated.
   * @throws std::length_error If temporary storage or an error diagnostic
   * exceeds an implementation-defined limit.
   *
   * @post On success, the result is identical to calling the `double3`
   * overload with the converted rows.
   * @post An outer-size or row-size validation failure leaves restraint state
   * unchanged.
   * @warning A non-finite value detected by the delegated primary setter can
   * leave an earlier host prefix updated.
   */
  void setReferenceCoordinates(
      const std::vector<std::vector<double>> &referenceCoordinates);

  /**
   * @brief Sets the per-atom masses or restraint weights.
   *
   * The values are copied into the host mass mirror in atom order; the next
   * `initialize()` or `calcForce()` transfers the array to the device.
   * Reference positions are stored separately and are not changed.
   *
   * A zero value is accepted and suppresses the harmonic energy and
   * force-array contribution for that atom, even when the atom is selected and
   * has a nonzero force constant. As with CHARMM's MASS option, a value may
   * have either sign.
   *
   * @param[in] masses Contiguous host vector containing exactly one finite
   * value per atom, normally in atomic mass units. The vector is borrowed for
   * the call and is not retained.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the vector length differs from the atom count or any value is NaN or
   * infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the host masses equal `masses`, and the reference
   * positions are unchanged.
   * @warning Validation and host mutation are interleaved by atom. If a later
   * mass is invalid, an earlier host prefix may already contain new values,
   * which reach the device with the next transfer.
   */
  void setMasses(const std::vector<double> &masses);

  /**
   * @brief Sets the exponent applied to every atom.
   *
   * The value selects one of the energy branches in the class description and
   * takes effect at the next calculation. As with CHARMM's EXPOnent option,
   * every integer is accepted: two gives the harmonic form, and zero or a
   * negative value makes every contribution zero.
   *
   * @param[in] exponent Dimensionless integer exponent.
   *
   * @post The next calculation uses `exponent`.
   */
  void setExponent(const int exponent);

  /**
   * @brief Sets the Cartesian scale factors applied to every atom.
   *
   * The factors multiply the squared X, Y, and Z displacements in `S_i` and the
   * corresponding gradient components, like CHARMM's XSCAle, YSCAle, and ZSCAle
   * options. As in CHARMM, a factor may be zero or negative; a zero factor
   * removes that axis from the restraint.
   *
   * @param[in] scaleFactors Host vector containing exactly three finite,
   * dimensionless factors in `[x, y, z]` order. The vector is borrowed for the
   * call and is not retained.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the vector does not contain exactly three values or any value is NaN or
   * infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the next calculation uses `scaleFactors`.
   * @post A validation failure leaves the stored factors unchanged.
   */
  void setScaleFactors(const std::vector<double> &scaleFactors);

public:
  /**
   * @brief Returns shared access to the private CUDA stream holder.
   *
   * The returned `std::shared_ptr` is a copied shared owner of the host scalar
   * containing the CUDA stream handle. No stream synchronization is performed.
   *
   * The restraint remains responsible for destroying the underlying CUDA
   * stream. Its destructor writes `nullptr` into the shared scalar before
   * releasing its own shared owner.
   *
   * @return A copied shared owner of the stream holder. The pointer is
   * non-null for a successfully constructed live restraint.
   * @warning Retaining the returned `std::shared_ptr` does not extend the
   * lifetime of the underlying CUDA stream beyond restraint destruction.
   */
  std::shared_ptr<cudaStream_t> getStream(void);

public:
  /**
   * @brief Validates manager state and stores current box dimensions.
   *
   * This method is the initialization hook used by `ForceManager`. It verifies
   * that the manager atom count equals the immutable construction count,
   * delegates to `setBoxDimensions()`, and transfers changed parameter mirrors
   * to the device.
   *
   * The call does not allocate new per-atom storage, clear outputs, mark a
   * separate initialized state, or alter reference coordinates. Direct native
   * force calculation is not gated by a successful prior call, although
   * manager-driven use calls this hook as part of subscription initialization.
   *
   * @param[in] numAtoms Dimensionless atom count supplied by the manager. It
   * must equal the construction count.
   * @param[in] boxDimensions Three finite, positive box lengths in
   * `[x, y, z]` order and angstrom units.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `numAtoms` differs from the fixed atom count or `boxDimensions` violates
   * the box contract.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if transferring
   * changed parameter mirrors fails.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the stored box dimensions equal `boxDimensions`, and the
   * device parameter mirrors match the host mirrors.
   */
  void initialize(const int numAtoms, const std::vector<double> &boxDimensions);

  /**
   * @brief Enqueues clearing of the force, energy, and virial output storage.
   *
   * Device clearing is enqueued on the restraint's private CUDA stream. The
   * method does not synchronize that stream before return. The host buffer
   * owned by `CudaEnergyVirial` is zeroed during the call.
   *
   * Call this method before the first calculation and before every calculation
   * that requires fresh output, because `calcForce()` atomically adds rather
   * than replaces force and energy contributions.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if output
   * allocation, clearing, or immediate CUDA launch checking fails.
   * @throws std::bad_alloc If output or diagnostic storage cannot be
   * allocated.
   * @throws std::length_error If output or diagnostic storage exceeds an
   * implementation-defined limit.
   *
   * @post After all previously ordered work on `getStream()` completes, the
   * restraint force array and energy-virial device buffer are zero.
   * @warning Clearing the energy buffer and force buffer is sequential rather
   * than transactional. A failure can leave only one output object cleared.
   */
  void clear(void);

  /**
   * @brief Enqueues harmonic force-array and optional energy accumulation.
   *
   * One logical input record is read for each atom. Components `x`, `y`, and
   * `z` are Cartesian coordinates in angstroms; `w` is ignored. Atoms whose
   * product `k_i * m_i` is zero are skipped, as in CHARMM. Force-array
   * contributions are always accumulated with the configured exponent and
   * scale factors. The `harm` energy is accumulated only when `calcEnergy` is
   * `true`.
   *
   * Changed parameter mirrors are first copied to the device on the private
   * stream, which is then synchronized. The method then launches one of two
   * compile-time kernel variants on the private stream and performs only the
   * immediate kernel-launch check. It does not synchronize the stream after the
   * launch. Asynchronous execution errors can surface from a later CUDA
   * synchronization or dependent operation.
   *
   * @param[in] xyzq Borrowed CUDA-device pointer to at least the fixed atom
   * count of contiguous `float4` records. The pointer may not be null and must
   * remain valid until the enqueued kernel completes.
   * @param[in] calcEnergy `true` to atomically add the harmonic energy to the
   * named `harm` device term; `false` to leave that energy term unchanged.
   * @param[in] calcVirial Accepted for the force-manager interface but ignored
   * because this restraint does not currently calculate a virial contribution.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if transferring
   * changed parameter mirrors fails or CUDA reports an immediate kernel-launch
   * failure.
   * @throws std::bad_alloc If construction of a CUDA failure diagnostic cannot
   * allocate storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined limit.
   *
   * @pre `xyzq` addresses at least the fixed atom count of live device records.
   * @pre Call `clear()` first when fresh rather than accumulated output is
   * required.
   * @post On successful immediate launch checking, calculation is enqueued on
   * `getStream()` but need not be complete.
   * @warning This performance-critical method does not validate `xyzq`, output
   * storage, parameter sizes, or stream state.
   */
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial);

  /**
   * @brief Stores the current orthogonal box lengths.
   *
   * The vector must contain `[x, y, z]` box lengths in angstroms. The values
   * are copied into host state for the manager lifecycle; no CUDA transfer or
   * synchronization occurs.
   *
   * The current force kernel does not read these dimensions, apply
   * minimum-image displacement, or rescale stored reference positions.
   *
   * @param[in] boxDimensions Host vector containing exactly three finite,
   * strictly positive lengths in `[x, y, z]` order and angstrom units. The
   * vector is borrowed for the call and is not retained.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the vector does not contain
   * exactly three values or any value is non-positive, NaN, or infinite.
   * @throws std::bad_alloc If an error diagnostic cannot be allocated.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, the stored box equals `boxDimensions`.
   * @post Input validation failure leaves the stored box unchanged.
   */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

  /**
   * @brief Returns shared ownership of the device force-array storage.
   *
   * The force object contains the fixed atom count of vectors in
   * structure-of-arrays form: X components begin at offset zero, Y components
   * begin at `stride()`, and Z components begin at `2 * stride()`.
   *
   * For the supported `AT = long long int` instantiations, values use
   * apoCHARMM's fixed-point scale of `2^40`. Divide each stored integer by
   * `2^40` to recover the force-array quantity in kilocalories per mole per
   * angstrom.
   *
   * No transfer, clearing, or stream synchronization is performed.
   *
   * @return A copied shared owner of the force object. The pointer is non-null
   * for a successfully constructed restraint and can outlive the restraint.
   * @warning A retained force object does not retain a valid calculation
   * stream or a restraint capable of updating it.
   * @warning Contents are additive and can be stale or uninitialized unless
   * the caller orders `clear()`, `calcForce()`, and stream synchronization
   * correctly.
   */
  std::shared_ptr<Force<AT>> getForce(void);

  /**
   * @brief Returns shared ownership of the energy-virial output object.
   *
   * The object contains the named scalar energy term `harm`, in kilocalories
   * per mole. The restraint does not populate a virial contribution. Device
   * energy is accumulated by `calcForce()` only when energy calculation is
   * requested.
   *
   * No device-to-host transfer or stream synchronization is performed.
   * `CudaEnergyVirial::getEnergy("harm")` reads the host mirror and therefore
   * remains stale until the caller orders an appropriate `copyToHost()` and
   * synchronization, or until a manager operation performs them.
   *
   * @return A copied shared owner of the energy-virial object. The pointer is
   * non-null for a successfully constructed restraint and can outlive the
   * restraint.
   * @warning A retained object does not retain a valid calculation stream or a
   * restraint capable of updating it.
   */
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void);

private:
  /**
   * @brief Synchronizes dirty host parameter mirrors to device storage.
   */
  void syncDeviceData(void);

private:
  int m_NumAtoms;

  CudaContainer<double> m_ForceConstants;

  CudaContainer<double3> m_ReferenceCoordinates;

  CudaContainer<double> m_Masses;

  std::array<double, 3> m_BoxDimensions;

  bool m_DeviceDataDirty;

  int m_Exponent;

  std::array<double, 3> m_ScaleFactors;

  AtomSelection m_Selection;

  std::shared_ptr<CudaEnergyVirial> m_EnergyVirial;

  std::shared_ptr<Force<AT>> m_Forces;

  std::shared_ptr<cudaStream_t> m_Stream;
};
