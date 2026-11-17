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
 * @brief Declares the C ABI for absolute harmonic restraint configuration.
 */

#ifndef __APOCHARMM_C_HARMONIC_RESTRAINT_FORCE_H__
#define __APOCHARMM_C_HARMONIC_RESTRAINT_FORCE_H__

#include "apocharmm_c/AtomSelection.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an owned absolute harmonic restraint in the C ABI.
 *
 * A successful @ref apo_harmonic_restraint_force_create call returns one newly
 * owned handle. The caller releases that handle with
 * @ref apo_harmonic_restraint_force_destroy. Configuration functions borrow a
 * live handle for one call and copy all supplied selections and arrays.
 *
 * A successful force-manager subscription retains an independent native shared
 * owner of the restraint, stream, force array, and energy object. Destroying
 * this C handle does not unsubscribe that retained native object. Unsubscribe
 * before destruction when the caller will need to remove the force explicitly.
 *
 * The C ABI exposes configuration and subscription but no direct force-kernel
 * invocation or output access. Use `ForceManager` or `CharmmContext` to drive
 * calculation.
 *
 * A non-NULL pointer must designate a live handle created by apoCHARMM.
 * Passing a dangling, already-destroyed, or foreign pointer has undefined
 * behavior. The handle and associated native object provide no internal
 * synchronization; callers must serialize mutation, subscription, calculation,
 * unsubscription, and destruction.
 *
 * Every status-returning function clears the calling thread's previous
 * diagnostic at entry. Success leaves it empty. On failure, call
 * @ref apo_last_error immediately on the same thread; its borrowed pointer is
 * valid only until another diagnostic-changing C ABI call on that thread.
 *
 * @see harmonic_restraint_force
 */
typedef struct apo_harmonic_restraint_force apo_harmonic_restraint_force;

/**
 * @brief Creates an owned harmonic-restraint handle from a PSF atom count.
 *
 * The native constructor copies only the PSF atom count; PSF masses are not
 * read. The created native restraint uses fixed-point `long long int` force
 * storage, single-precision arithmetic for forces and per-atom energies, and
 * double-precision energy sums. All atoms begin selected, force constants
 * and reference positions begin at zero, per-atom weights begin at one, the
 * exponent is two, every scale factor is one, and stored box dimensions
 * begin at zero.
 *
 * @param[out] out Non-NULL pointer to the caller's handle slot. The function
 * stores `NULL` in `*out` before validating `psf` and stores a newly owned
 * handle on success.
 * @param[in] psf Borrowed live PSF handle with an initialized positive atom
 * count. The handle is borrowed only for this call; neither it nor its native
 * PSF is retained.
 * @retval APO_STATUS_OK `*out` contains a newly owned handle that must be
 * released with @ref apo_harmonic_restraint_force_destroy.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` or `psf` is `NULL`, the PSF handle
 * contains no native object, or the atom count is zero.
 * @retval APO_STATUS_NOT_INITIALIZED The native PSF atom count is negative.
 * @retval APO_STATUS_CUDA_ERROR Native CUDA allocation, initialization,
 * transfer, synchronization, force allocation, or stream creation failed.
 * @retval APO_STATUS_RUNTIME_ERROR Handle allocation, shared ownership,
 * host-storage allocation, or another standard or nonstandard C++ operation
 * failed.
 *
 * @post On every failure after a valid `out` pointer is accepted,
 * `*out == NULL`.
 * @post On success, destroying the PSF handle or mutating its native PSF does
 * not change the restraint's atom count.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves nonempty text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_create(
    apo_harmonic_restraint_force **out, const apo_charmm_psf *psf);

/**
 * @brief Destroys an owned harmonic-restraint handle.
 *
 * Deleting the handle releases its native shared owner. A force manager that
 * already subscribed the restraint retains an independent owner, so the
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
APOCHARMM_C_API void
apo_harmonic_restraint_force_destroy(apo_harmonic_restraint_force *restraint);

/**
 * @brief Copies the atom selection used by a harmonic restraint.
 *
 * Every newly unselected atom has its stored force constant set to zero.
 * Newly selected atoms do not recover constants that an earlier selection
 * change zeroed.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained by this call.
 * @param[in] selection Borrowed live atom-selection handle with the same atom
 * count as `restraint`. The native selection is copied and no ownership is
 * transferred or retained.
 * @retval APO_STATUS_OK The selection was copied and the host constants were
 * updated; the next initialization or calculation transfers them.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, or the two atom counts differ.
 * @retval APO_STATUS_RUNTIME_ERROR Copying selection storage, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On success, the restraint owns an independent selection copy.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Failure after native mutation begins can leave selection state and
 * host force constants representing different stages.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_set_selection(
    apo_harmonic_restraint_force *restraint,
    const apo_atom_selection *selection);

/**
 * @brief Sets one force constant for every selected atom.
 *
 * All unselected atoms receive zero. With masses in atomic mass units,
 * `force_constant` has units of kilocalories per mole per square angstrom per
 * atomic mass unit.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] force_constant Finite, non-negative scalar applied to every
 * selected atom.
 * @retval APO_STATUS_OK Host constants were updated; the next initialization
 * or calculation transfers them.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL`, the handle
 * contains no native object, or `force_constant` is negative, NaN, or
 * infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Selected-index allocation, diagnostic
 * construction, or another standard or nonstandard C++ operation failed.
 *
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Failure after native mutation begins can leave all host constants
 * zero.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_set_force_constant(
    apo_harmonic_restraint_force *restraint, const double force_constant);

/**
 * @brief Copies one force constant per atom.
 *
 * The input is indexed in atom order. Only values at selected indices are
 * validated and used; unselected input values are ignored and the corresponding
 * stored constants are zero. As with CHARMM's WEIGhting option, a selected
 * value may have either sign.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] force_constants Non-NULL pointer to
 * `force_constants_len` contiguous `double` values. The array is borrowed for
 * the call and copied.
 * @param[in] force_constants_len Dimensionless number of elements in
 * `force_constants`. It must equal the restraint atom count.
 * @retval APO_STATUS_OK The selected constants were copied; the next
 * initialization or calculation transfers them.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or `force_constants` is
 * `NULL`, the handle contains no native object, the length differs from the
 * atom count, or a selected value is NaN or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the C array, allocating selected
 * indices, constructing a diagnostic, or another standard or nonstandard C++
 * operation failed.
 *
 * @post On success, selected atoms contain their corresponding input values
 * and unselected atoms contain zero.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_set_force_constants(
    apo_harmonic_restraint_force *restraint, const double *force_constants,
    const size_t force_constants_len);

/**
 * @brief Copies absolute Cartesian reference coordinates.
 *
 * The flat array contains one `[x, y, z]` record per atom in row-major atom
 * order. Coordinates are absolute angstrom values and are not minimum-image
 * wrapped.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] reference_coordinates Non-NULL pointer to
 * `reference_coordinates_len` contiguous `double` values. The input is
 * borrowed for the call and copied.
 * @param[in] reference_coordinates_len Dimensionless scalar-element count. It
 * must be nonzero, an exact multiple of three, and exactly
 * `3 * atom_count`.
 * @retval APO_STATUS_OK Reference coordinates were copied; the next
 * initialization or calculation transfers them.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or
 * `reference_coordinates` is `NULL`, the handle contains no native object,
 * the length is zero or not a multiple of three, the derived atom count is
 * wrong, or a coordinate is NaN or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Allocating the nested native copy,
 * constructing a diagnostic, or another standard or nonstandard C++ operation
 * failed.
 *
 * @post On success, the restraint has copied every row and retains no caller
 * pointer.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Native finite-value validation and host mutation are interleaved.
 * A later invalid coordinate can leave an earlier host prefix changed.
 */
APOCHARMM_C_API apo_status
apo_harmonic_restraint_force_set_reference_coordinates(
    apo_harmonic_restraint_force *restraint,
    const double *reference_coordinates,
    const size_t reference_coordinates_len);

/**
 * @brief Copies one mass or weight per atom.
 *
 * Values normally use atomic mass units. As with CHARMM's MASS option, a value
 * may have either sign. A zero value is accepted and suppresses that atom's
 * harmonic contribution.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] masses Non-NULL pointer to `masses_len` contiguous `double`
 * values in atom order. The array is borrowed for the call and copied.
 * @param[in] masses_len Dimensionless number of elements in `masses`. It must
 * equal the restraint atom count.
 * @retval APO_STATUS_OK Masses were copied; the next initialization or
 * calculation transfers them.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or `masses` is `NULL`, the
 * handle contains no native object, the length differs from the atom count, or
 * a mass is NaN or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the C array, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On success, the restraint retains no caller pointer and the
 * reference positions are unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Native validation and host mutation are interleaved. A later
 * invalid mass can leave an earlier host prefix changed.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_set_masses(
    apo_harmonic_restraint_force *restraint, const double *masses,
    const size_t masses_len);

/**
 * @brief Sets the exponent applied to every atom.
 *
 * Two gives the harmonic form, and zero or a negative value makes every
 * contribution zero. As with CHARMM's EXPOnent option, every integer is
 * accepted.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] exponent Dimensionless integer exponent.
 * @retval APO_STATUS_OK The exponent was stored.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL` or the handle
 * contains no native object.
 * @retval APO_STATUS_RUNTIME_ERROR Diagnostic construction or another
 * standard or nonstandard C++ operation failed.
 *
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_set_exponent(
    apo_harmonic_restraint_force *restraint, const int exponent);

/**
 * @brief Copies the Cartesian scale factors applied to every atom.
 *
 * The factors multiply the squared X, Y, and Z displacements and the
 * corresponding gradient components, like CHARMM's XSCAle, YSCAle, and ZSCAle
 * options. A factor may be zero or negative.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] scale_factors Non-NULL pointer to three contiguous dimensionless
 * `double` factors in `[x, y, z]` order. The array is borrowed for the call
 * and copied.
 * @param[in] scale_factors_len Dimensionless number of elements in
 * `scale_factors`. It must equal three.
 * @retval APO_STATUS_OK The scale factors were stored.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or `scale_factors` is
 * `NULL`, the handle contains no native object, the length is not three, or a
 * factor is NaN or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the C array, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On success, no caller pointer is retained.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_set_scale_factors(
    apo_harmonic_restraint_force *restraint, const double *scale_factors,
    const size_t scale_factors_len);

/**
 * @brief Copies three orthogonal box lengths.
 *
 * The current restraint kernel stores but does not read these dimensions. The
 * values do not apply minimum-image displacement or rescale reference
 * positions.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] box_dimensions Non-NULL pointer to three contiguous `double`
 * lengths in `[x, y, z]` order and angstrom units. The array is borrowed for
 * the call and copied.
 * @param[in] box_dimensions_len Dimensionless number of elements in
 * `box_dimensions`. It must equal three.
 * @retval APO_STATUS_OK The box dimensions were stored in host memory.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or `box_dimensions` is
 * `NULL`, the handle contains no native object, the length is not three, or a
 * length is non-positive, NaN, or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the C array, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On success, no caller pointer is retained.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_harmonic_restraint_force_set_box_dimensions(
    apo_harmonic_restraint_force *restraint, const double *box_dimensions,
    const size_t box_dimensions_len);

/**
 * @brief Subscribes a harmonic restraint to a force manager.
 *
 * On success, the manager retains native shared ownership of the restraint,
 * its private stream holder, its force array, and its energy-virial object. The
 * two C handles remain owned by their callers.
 *
 * If the manager is already initialized, subscription first initializes the
 * restraint with the manager atom count and box dimensions. Otherwise, the
 * manager initializes the retained restraint during its later initialization.
 * Duplicate force tags are permitted, but the same native restraint object
 * may not be subscribed twice to one manager.
 *
 * @param[in,out] force_manager Borrowed live manager handle. The handle is not
 * retained.
 * @param[in] restraint Borrowed live restraint handle. The manager retains the
 * underlying native object on success but does not retain this C handle.
 * @param[in] force_tag Borrowed non-NULL, nonempty, null-terminated tag. The
 * text is copied and no caller pointer is retained.
 * @retval APO_STATUS_OK The manager retained the restraint and its resources.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, `force_tag` is `NULL` or empty, the restraint is
 * already subscribed, or initialization detects an atom-count, box, or other
 * invalid-argument mismatch.
 * @retval APO_STATUS_CUDA_ERROR Immediate restraint initialization or manager
 * resource setup encounters a CUDA failure.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the tag, growing manager
 * subscription storage, constructing a diagnostic, or another standard or
 * nonstandard C++ operation failed.
 *
 * @post On success, destroying the restraint C handle alone does not remove
 * the subscribed native object.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 * @warning Manager subscription storage is maintained in parallel vectors.
 * An allocation failure during vector growth is not guaranteed to preserve a
 * completely unchanged manager subscription state.
 */
APOCHARMM_C_API apo_status apo_force_manager_subscribe_harmonic_restraint_force(
    apo_force_manager *force_manager, apo_harmonic_restraint_force *restraint,
    const char *force_tag);

/**
 * @brief Unsubscribes a harmonic restraint from a force manager.
 *
 * The manager releases its retained native owners of the restraint, stream
 * holder, force array, and energy-virial object. Other C, C++, or Python owners
 * remain valid. The operation does not clear or otherwise reinitialize the
 * restraint.
 *
 * @param[in,out] force_manager Borrowed live manager handle. The handle is not
 * retained.
 * @param[in] restraint Borrowed live restraint handle identifying the exact
 * native object to remove. The handle is not retained.
 * @retval APO_STATUS_OK The manager released the matching subscription.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, or the restraint is not subscribed to the
 * manager.
 * @retval APO_STATUS_RUNTIME_ERROR Diagnostic construction or another
 * unexpected standard or nonstandard C++ operation failed.
 *
 * @post On success, the manager will not schedule later calculations for this
 * restraint.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_force_manager_unsubscribe_harmonic_restraint_force(
    apo_force_manager *force_manager, apo_harmonic_restraint_force *restraint);

#ifdef __cplusplus
}
#endif

#endif
