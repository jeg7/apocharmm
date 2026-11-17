// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

/**
 * @file
 * @brief Declares the C ABI for harmonic center-of-mass restraints.
 */
#ifndef __APOCHARMM_C_HARMONIC_CENTER_OF_MASS_RESTRAINT_FORCE_H__
#define __APOCHARMM_C_HARMONIC_CENTER_OF_MASS_RESTRAINT_FORCE_H__

#include "apocharmm_c/AtomSelection.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/Status.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an owned harmonic center-of-mass restraint in the C ABI.
 *
 * A successful @ref apo_harmonic_center_of_mass_restraint_force_create call
 * returns one newly owned handle. Release it with
 * @ref apo_harmonic_center_of_mass_restraint_force_destroy. Configuration
 * functions borrow a live handle for one call and copy all supplied selections
 * and arrays.
 *
 * A successful manager subscription retains an independent native shared owner
 * of the restraint, CUDA stream holder, force array, and energy-virial object.
 * Destroying this C handle therefore does not remove an existing subscription.
 * Unsubscribe before destruction when later explicit unsubscription is needed.
 *
 * The C ABI exposes configuration and manager subscription, not direct kernel
 * invocation or output access. Use `ForceManager` or `CharmmContext` to drive
 * force, energy, and virial calculation.
 *
 * A non-NULL pointer must designate a live handle created by apoCHARMM. Passing
 * a dangling, already-destroyed, or foreign pointer has undefined behavior.
 * The handle and native object provide no internal synchronization; callers
 * must serialize mutation, manager operations, calculation, and destruction.
 *
 * Every status-returning function clears the calling thread's previous
 * diagnostic at entry. Success leaves it empty. On failure, call
 * @ref apo_last_error on the same thread. Its borrowed pointer remains valid
 * only until another diagnostic-changing C ABI call on that thread.
 *
 * @see harmonic_center_of_mass_restraint_force
 */
typedef struct apo_harmonic_center_of_mass_restraint_force
    apo_harmonic_center_of_mass_restraint_force;

/**
 * @brief Creates an owned harmonic center-of-mass restraint handle from a PSF
 * atom-count and mass snapshot.
 *
 * The native object uses fixed-point `long long int` force storage,
 * single-precision arithmetic for forces and the restraint energy, and
 * double-precision energy accumulation. All atoms begin selected,
 * weighting is uniform, the stored masses are copied from the PSF, the force
 * constant and reference distance are zero, the reference position is
 * `(0, 0, 0)`, all Cartesian mask components are active, and box dimensions
 * are unset. The native constructor copies the PSF atom count and complete
 * mass vector; finite masses of either sign are accepted, and enabling mass
 * weighting then uses them.
 *
 * @param[out] out Non-NULL pointer to the caller's handle slot. The function
 * stores `NULL` in `*out` before validating `psf` and stores a newly owned
 * handle on success.
 * @param[in] psf Borrowed live PSF handle with an initialized positive atom
 * count and exactly one finite mass per atom. The handle is borrowed only for
 * this call; neither it nor its native PSF is retained.
 * @retval APO_STATUS_OK `*out` contains a newly owned handle that must be
 * released with
 * @ref apo_harmonic_center_of_mass_restraint_force_destroy.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` or `psf` is `NULL`, the PSF handle
 * contains no native object, the atom count is zero, or a mass is NaN or
 * infinite.
 * @retval APO_STATUS_NOT_INITIALIZED The native PSF atom count is negative.
 * @retval APO_STATUS_CUDA_ERROR Native CUDA allocation, transfer,
 * synchronization, force allocation, or stream creation failed.
 * @retval APO_STATUS_RUNTIME_ERROR The PSF mass count differs from its atom
 * count, or handle allocation, host allocation, diagnostic construction, or
 * another standard or nonstandard C++ operation failed.
 *
 * @post On every failure after a valid `out` pointer is accepted,
 * `*out == NULL`.
 * @post On success, destroying the PSF handle or mutating its native PSF does
 * not change the restraint's atom count or masses.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves nonempty text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_harmonic_center_of_mass_restraint_force_create(
    apo_harmonic_center_of_mass_restraint_force **out,
    const apo_charmm_psf *psf);

/**
 * @brief Destroys an owned harmonic center-of-mass restraint handle.
 *
 * Deleting the handle releases its native shared owner. A force manager that
 * already subscribed the restraint retains an independent owner, so the native
 * restraint can remain active after this call.
 *
 * @param[in] restraint Owned handle to release. `NULL` is accepted and is a
 * no-op. A non-NULL pointer is invalid after this call returns.
 *
 * @post No C++ exception escapes the C ABI boundary.
 * @note A normally returning destruction preserves the calling thread's
 * existing @ref apo_last_error diagnostic. An internally caught destruction
 * failure can replace that diagnostic but cannot be returned by this void API.
 * @warning Do not destroy the same handle twice or use it after destruction.
 * @warning Unsubscribe before destroying the handle when later explicit
 * unsubscription is required.
 */
APOCHARMM_C_API void apo_harmonic_center_of_mass_restraint_force_destroy(
    apo_harmonic_center_of_mass_restraint_force *restraint);

/**
 * @brief Copies the nonempty atom selection used to define the group center.
 *
 * Selected indices and either uniform or mass-based selected weights are
 * rebuilt in host storage; the next initialization or calculation copies them
 * to the device. No C handle or caller selection storage is retained.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] selection Borrowed live selection handle with the same atom count
 * as `restraint` and at least one selected atom. The native selection is
 * copied.
 * @retval APO_STATUS_OK The selection and selected host arrays were updated.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, atom counts differ, the selection is empty, or
 * mass weighting is active and the selected masses have a zero or non-finite
 * total or a non-finite weight fraction.
 * @retval APO_STATUS_RUNTIME_ERROR Copying selection or temporary storage,
 * constructing a diagnostic, or another standard or nonstandard C++ operation
 * failed.
 *
 * @post On success, the restraint owns an independent selection copy.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Failure after native mutation begins can leave the copied selection
 * or selected count newer than the host index and weight arrays.
 */
APOCHARMM_C_API apo_status
apo_harmonic_center_of_mass_restraint_force_set_selection(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const apo_atom_selection *selection);

/**
 * @brief Sets the scalar harmonic force constant.
 *
 * The value is retained in native host scalar state. This operation performs no
 * CUDA transfer or synchronization.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] force_constant Finite coefficient in kilocalories per mole per
 * square angstrom. As in CHARMM, it may have either sign. Zero disables
 * evaluation.
 * @retval APO_STATUS_OK The scalar force constant was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL`, the handle
 * contains no native object, or `force_constant` is NaN or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Diagnostic construction or another standard
 * or nonstandard C++ operation failed.
 *
 * @post On validation failure, the prior force constant remains unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_harmonic_center_of_mass_restraint_force_set_force_constant(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double force_constant);

/**
 * @brief Copies the Cartesian reference position and component mask.
 *
 * As in CHARMM, active components use the raw center-to-reference
 * displacement without minimum imaging; inactive components contribute
 * neither energy nor gradient. Position and mask values are copied
 * into native host scalar state. No caller buffer is retained and no CUDA
 * transfer occurs.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] reference_position Non-NULL pointer to
 * `reference_position_len` contiguous `double` values in `[x, y, z]` order and
 * angstrom units. The array is borrowed for the call and copied.
 * @param[in] reference_position_len Dimensionless number of scalar elements in
 * `reference_position`. It must equal three.
 * @param[in] reference_mask Non-NULL pointer to `reference_mask_len` contiguous
 * C `int` values in `[x, y, z]` order. Every value must be zero or one.
 * @param[in] reference_mask_len Dimensionless number of scalar elements in
 * `reference_mask`. It must equal three, and at least one element must be one.
 * @retval APO_STATUS_OK The position and mask were copied.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint`, `reference_position`, or
 * `reference_mask` is `NULL`; the handle contains no native object; either
 * length is not three; a position is NaN or infinite; a mask value is not zero
 * or one; or all mask values are zero.
 * @retval APO_STATUS_RUNTIME_ERROR Allocating temporary vectors, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On success, the restraint retains no caller pointer.
 * @post On validation failure, the prior position and mask remain unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_harmonic_center_of_mass_restraint_force_set_reference_position(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double *reference_position, const size_t reference_position_len,
    const int *reference_mask, const size_t reference_mask_len);

/**
 * @brief Sets the target distance from the reference position.
 *
 * The potential is `k * (r - reference_distance)^2`, so zero gives
 * `k * r^2`. As in CHARMM, a negative value is accepted. The value is copied
 * into native host scalar state and no CUDA transfer occurs.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] reference_distance Finite target distance in angstroms.
 * @retval APO_STATUS_OK The reference distance was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL`, the handle
 * contains no native object, or `reference_distance` is NaN or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Diagnostic construction or another standard
 * or nonstandard C++ operation failed.
 *
 * @post On validation failure, the prior reference distance remains unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_harmonic_center_of_mass_restraint_force_set_reference_distance(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double reference_distance);

/**
 * @brief Copies per-atom masses and enables mass weighting.
 *
 * Only relative selected masses affect the normalized center. The complete
 * array is copied to owned host storage and the selected weights are rebuilt;
 * the next initialization or calculation copies the weights to the device. No
 * caller buffer is retained.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] masses Non-NULL pointer to `masses_len` contiguous `double` values
 * in atom order, normally in atomic mass units. Every value must be finite;
 * as in CHARMM, it may have either sign.
 * @param[in] masses_len Dimensionless number of values in `masses`. It must
 * equal the restraint atom count, and the selected values must have a finite
 * nonzero total and finite weight fractions.
 * @retval APO_STATUS_OK Masses were copied, mass weighting was enabled, and
 * selected host weights were rebuilt.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or `masses` is `NULL`, the
 * handle contains no native object, the length differs from the atom count, a
 * mass is NaN or infinite, or the selected masses have a zero or non-finite
 * total or a non-finite weight fraction.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the C array, allocating temporary
 * storage, constructing a diagnostic, or another standard or nonstandard C++
 * operation failed.
 *
 * @post On success, no caller pointer is retained and mass weighting is active.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Failure after native mutation begins can leave new masses or an
 * enabled weighting mode paired with older selected host arrays.
 */
APOCHARMM_C_API apo_status
apo_harmonic_center_of_mass_restraint_force_set_masses(
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const double *masses, const size_t masses_len);

/**
 * @brief Enables mass weighting or restores uniform selected weights.
 *
 * Enabling uses the stored masses: the PSF masses copied at creation until
 * @ref apo_harmonic_center_of_mass_restraint_force_set_masses replaces them.
 * Disabling assigns every selected atom dimensionless weight one without
 * discarding the stored mass array. Selected weights are rebuilt in host
 * storage; the next initialization or calculation copies them to the device.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] flag `true` to use configured masses or `false` to use uniform
 * selected weights.
 * @retval APO_STATUS_OK The selected weighting mode and host weights were
 * updated.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL`, the handle
 * contains no native object, or `flag` is true and the selected masses have a
 * zero or non-finite total or a non-finite weight fraction.
 * @retval APO_STATUS_RUNTIME_ERROR Allocating temporary storage, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Failure after the native mode flag changes can leave the mode and
 * selected host arrays inconsistent.
 */
APOCHARMM_C_API apo_status
apo_harmonic_center_of_mass_restraint_force_set_mass_weighting(
    apo_harmonic_center_of_mass_restraint_force *restraint, const bool flag);

/**
 * @brief Subscribes a harmonic center-of-mass restraint to a force manager.
 *
 * On success, the manager retains native shared ownership of the restraint,
 * CUDA stream holder, force array, and energy-virial object. The two C handles
 * remain owned by their callers.
 *
 * If the manager is initialized, subscription first verifies the restraint
 * atom count and installs the manager's orthorhombic box. Otherwise, manager
 * initialization performs that hook later. Duplicate tags are permitted, but
 * the same native restraint object may not be subscribed twice to one manager.
 *
 * @param[in,out] force_manager Borrowed live manager handle. The handle is not
 * retained.
 * @param[in] restraint Borrowed live restraint handle. The manager retains the
 * underlying native object on success but does not retain this C handle.
 * @param[in] force_tag Borrowed non-NULL, nonempty, null-terminated byte
 * string. Native code copies the bytes into `std::string` and retains no caller
 * pointer.
 * @retval APO_STATUS_OK The manager retained the restraint and its resources.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, `force_tag` is `NULL` or empty, the restraint is
 * already subscribed, or immediate initialization detects an atom-count
 * mismatch.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the tag, growing manager
 * subscription storage, constructing a diagnostic, or another standard or
 * nonstandard C++ operation failed.
 *
 * @post On success, destroying the restraint C handle alone does not remove the
 * subscribed native object.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Manager subscription state is held in parallel vectors. Allocation
 * failure during vector growth is not guaranteed to leave every vector at the
 * same prior length.
 */
APOCHARMM_C_API apo_status
apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
    apo_force_manager *force_manager,
    apo_harmonic_center_of_mass_restraint_force *restraint,
    const char *force_tag);

/**
 * @brief Unsubscribes a harmonic center-of-mass restraint by object identity.
 *
 * The manager removes the first matching native object from all parallel
 * subscription vectors and releases its shared references. It does not clear,
 * deinitialize, or destroy the restraint while another owner exists.
 *
 * @param[in,out] force_manager Borrowed live manager handle. The handle is not
 * retained.
 * @param[in] restraint Borrowed live restraint handle identifying the native
 * object to remove. The handle itself is not retained or destroyed.
 * @retval APO_STATUS_OK The matching subscription was removed.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, or the restraint is not subscribed to the manager.
 * @retval APO_STATUS_RUNTIME_ERROR Diagnostic construction or another standard
 * or nonstandard C++ operation failed.
 *
 * @post On success, the restraint no longer participates in manager box
 * propagation, clearing, force evaluation, force aggregation, or energy
 * aggregation.
 * @post On lookup or validation failure, manager subscription state is
 * unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force(
    apo_force_manager *force_manager,
    apo_harmonic_center_of_mass_restraint_force *restraint);

#ifdef __cplusplus
}
#endif

#endif
