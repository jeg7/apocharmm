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
 * @brief Declares the C ABI for MMFP geometric restraint configuration.
 */

#ifndef __APOCHARMM_C_GEOMETRIC_RESTRAINT_FORCE_H__
#define __APOCHARMM_C_GEOMETRIC_RESTRAINT_FORCE_H__

#include "apocharmm_c/AtomReference.h"
#include "apocharmm_c/AtomSelection.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Enums.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an owned MMFP geometric restraint in the C ABI.
 *
 * A successful @ref apo_geometric_restraint_force_create call returns one
 * newly owned handle. Release it with
 * @ref apo_geometric_restraint_force_destroy. The handle retains shared
 * ownership of one native `GeometricRestraintForce<long long int, float>`,
 * which uses fixed-point `long long int` force storage and single-precision
 * final gradient conversion. Geometry, energy, and RCM arithmetic use double
 * precision.
 *
 * Construction borrows a live @ref apo_charmm_psf handle only for the call.
 * The native constructor copies the PSF atom count and complete mass vector.
 * Neither the public PSF handle nor its native PSF is retained, so destroying
 * the PSF handle or mutating its native PSF after construction does not change
 * the restraint. Use @ref apo_geometric_restraint_force_set_masses to replace
 * the copied masses.
 *
 * Every term-addition function borrows exactly one typed atom handle: either
 * an @ref apo_atom_selection or an @ref apo_atom_reference. No entry point
 * accepts raw atom indices, index arrays, or untyped atom pointers. Native
 * code copies the selected indices or the single referenced index together
 * with every term parameter. No selection, reference, topology, or caller
 * array is retained, so borrowed handles may be destroyed as soon as the call
 * returns. A selection must represent the restraint's atom count. A reference
 * contributes only its zero-based index, which must lie in the restraint's
 * force-local atom range; its topology is neither compared nor retained.
 *
 * Every term is harmonic. Positions, origins, centers, and offsets use
 * angstroms; force constants use kilocalories per mole per square angstrom;
 * masses normally use atomic mass units. Separate terms may overlap or repeat
 * and contribute additively in insertion order. See
 * @ref geometric_restraint_force for the geometry, activation, RCM, and virial
 * definitions.
 *
 * A successful force-manager subscription retains independent native shared
 * ownership of the restraint, CUDA stream holder, force array, and
 * energy-virial object. Destroying this C handle therefore does not remove an
 * existing subscription. Unsubscribe before destruction when later explicit
 * unsubscription is needed. Term addition, mass replacement, and reset remain
 * available while subscribed and take effect at the next manager evaluation
 * without resubscription.
 *
 * This header exposes construction, term addition, force-wide mass
 * replacement, reset, and force-manager subscription. It does not expose
 * kernel invocation or output access; use `ForceManager` or `CharmmContext` to
 * drive calculation.
 *
 * A non-NULL pointer must designate a live handle created by apoCHARMM.
 * Passing a dangling, already-destroyed, or foreign pointer has undefined
 * behavior. The handle and native object provide no internal synchronization.
 * Callers must serialize every call that uses the same restraint handle,
 * including subscription, unsubscription, destruction, and manager
 * calculations that evaluate the restraint, and must not destroy or mutate a
 * borrowed PSF, selection, or reference while a call borrows it.
 *
 * Every status-returning function clears the calling thread's previous
 * diagnostic at entry. Success leaves it empty. On failure, call
 * @ref apo_last_error immediately on the same thread. Its borrowed pointer is
 * valid only until another diagnostic-changing C ABI call on that thread.
 *
 * @see geometric_restraint_force
 */
typedef struct apo_geometric_restraint_force apo_geometric_restraint_force;

/**
 * @brief Creates an owned geometric restraint from a PSF snapshot.
 *
 * The native constructor copies the PSF atom count and complete mass vector,
 * creates a private CUDA stream, a fixed-point force array, and an
 * energy-virial object containing the named term `mmfp`, and clears every
 * output. It uses synchronous device transfers and synchronizes the private
 * stream before returning, so it can wait for other work on the device. The
 * new restraint has no terms. Finite positive, zero, and negative masses are
 * accepted.
 *
 * @param[out] out Non-NULL pointer to the caller's handle slot. The function
 * stores `NULL` in `*out` before validating `psf` and stores a newly owned,
 * non-NULL handle only on success.
 * @param[in] psf Borrowed live PSF handle with an initialized positive atom
 * count and exactly one finite mass per atom. The handle is borrowed only for
 * this call; neither it nor its native PSF is retained.
 * @retval APO_STATUS_OK `*out` contains a newly owned handle that must be
 * released with @ref apo_geometric_restraint_force_destroy.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` or `psf` is `NULL`, the PSF handle
 * contains no native object, the atom count is zero or exceeds the fixed-point
 * force-storage representation limit, or a mass is NaN or infinite.
 * @retval APO_STATUS_NOT_INITIALIZED The native PSF atom count is negative.
 * @retval APO_STATUS_CUDA_ERROR Stream creation, allocation, transfer, output
 * clearing, or synchronization failed.
 * @retval APO_STATUS_RUNTIME_ERROR The PSF mass count differs from its atom
 * count, handle or host allocation failed, or another standard or nonstandard
 * C++ operation failed.
 *
 * @post On every failure after a valid `out` pointer is accepted,
 * `*out == NULL`.
 * @post On success, destroying the PSF handle or mutating its native PSF does
 * not change the restraint's atom count or masses.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves nonempty text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_geometric_restraint_force_create(
    apo_geometric_restraint_force **out, const apo_charmm_psf *psf);

/**
 * @brief Destroys an owned geometric restraint handle.
 *
 * Deleting the handle releases its shared native owner. A force manager that
 * already subscribed the restraint retains an independent owner of the
 * restraint, its CUDA stream holder, its force array, and its energy-virial
 * object, so the restraint can remain active after this call. Without another
 * owner, destruction also releases the native restraint, its private CUDA
 * stream, and its output storage.
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
apo_geometric_restraint_force_destroy(apo_geometric_restraint_force *restraint);

/**
 * @brief Appends one harmonic plane term for a borrowed atom selection.
 *
 * The plane passes through `origin`. Native code normalizes `direction` with
 * double-precision `std::hypot` to obtain the normal `n`. `SYMMETRIC` uses the
 * signed distance `dot(n, r - origin)`; `INSIDE` and `OUTSIDE` use its
 * absolute value, with gradient `+n` at exactly zero distance.
 *
 * The selection handle is borrowed only for this call. Native code copies the
 * selected indices and every parameter into owned host storage; no handle,
 * selection, or caller array is retained. Device mirrors are rebuilt before
 * the next evaluation, and this call performs no CUDA work. A valid empty
 * selection is accepted after parameter validation and adds no term.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @param[in] selection Borrowed live selection handle representing exactly the
 * restraint's atom count.
 * @param[in] origin Non-NULL pointer to `origin_len` finite coordinates in
 * `[x, y, z]` order and angstrom units. The values are copied.
 * @param[in] origin_len Dimensionless number of elements in `origin`. It must
 * equal three.
 * @param[in] direction Non-NULL pointer to `direction_len` finite
 * dimensionless components with a positive finite Euclidean norm. The values
 * are copied.
 * @param[in] direction_len Dimensionless number of elements in `direction`. It
 * must equal three.
 * @param[in] force_constant Finite signed `K` in kilocalories per mole per
 * square angstrom. Zero is accepted.
 * @param[in] offset Finite signed CHARMM `DROFF` value in angstroms.
 * @param[in] activation Declared @ref apo_geometric_restraint_activation value.
 * @param[in] selection_mode Declared
 * @ref apo_geometric_restraint_selection_mode value. A nonempty RCM selection
 * requires a finite nonzero signed total of its active masses and finite mass
 * fractions, even when `force_constant` is zero.
 * @retval APO_STATUS_OK One term was appended, or a valid empty selection was
 * accepted without adding a term.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle, native object, or array
 * pointer is `NULL`; `origin_len` or `direction_len` is not three;
 * `activation` or `selection_mode` is not declared; the selection atom count
 * differs from the restraint atom count; an origin, direction, force-constant,
 * or offset value is NaN or infinite; the direction norm is not positive and
 * finite; the membership count would overflow; or an RCM mass requirement is
 * violated.
 * @retval APO_STATUS_RUNTIME_ERROR Copying indices, growing term storage,
 * constructing a diagnostic, or another standard or nonstandard C++ operation
 * failed.
 *
 * @post On failure, previously installed terms are unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_geometric_restraint_force_add_plane_from_selection(
    apo_geometric_restraint_force *restraint,
    const apo_atom_selection *selection, const double *origin,
    const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode);

/**
 * @brief Appends one harmonic plane term for a borrowed atom reference.
 *
 * The plane geometry matches
 * @ref apo_geometric_restraint_force_add_plane_from_selection. Only the
 * reference's zero-based atom index is used. It is validated against the
 * restraint's force-local atom range; the reference topology is neither
 * compared nor retained. The reference handle is borrowed only for this call,
 * every parameter is copied, and this call performs no CUDA work.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @param[in] atom Borrowed live atom-reference handle whose index lies in
 * `[0, N)`, where `N` is the restraint atom count.
 * @param[in] origin Non-NULL pointer to `origin_len` finite coordinates in
 * `[x, y, z]` order and angstrom units. The values are copied.
 * @param[in] origin_len Dimensionless number of elements in `origin`. It must
 * equal three.
 * @param[in] direction Non-NULL pointer to `direction_len` finite
 * dimensionless components with a positive finite Euclidean norm. The values
 * are copied.
 * @param[in] direction_len Dimensionless number of elements in `direction`. It
 * must equal three.
 * @param[in] force_constant Finite signed `K` in kilocalories per mole per
 * square angstrom. Zero is accepted.
 * @param[in] offset Finite signed CHARMM `DROFF` value in angstroms.
 * @param[in] activation Declared @ref apo_geometric_restraint_activation value.
 * @param[in] selection_mode Declared
 * @ref apo_geometric_restraint_selection_mode value. RCM requires a finite
 * nonzero active mass for the referenced atom.
 * @retval APO_STATUS_OK One term was appended.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle, native object, or array
 * pointer is `NULL`; `origin_len` or `direction_len` is not three;
 * `activation` or `selection_mode` is not declared; the atom index lies
 * outside the restraint's atom range; an origin, direction, force-constant, or
 * offset value is NaN or infinite; the direction norm is not positive and
 * finite; the membership count would overflow; or the RCM mass requirement is
 * violated.
 * @retval APO_STATUS_RUNTIME_ERROR Growing term storage, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On failure, previously installed terms are unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_geometric_restraint_force_add_plane_from_atom_reference(
    apo_geometric_restraint_force *restraint, const apo_atom_reference *atom,
    const double *origin, const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode);

/**
 * @brief Appends one harmonic cylinder term for a borrowed atom selection.
 *
 * The infinite axis passes through `origin` along `direction`, which native
 * code normalizes with double-precision `std::hypot`. The reaction coordinate
 * is the raw Cartesian perpendicular distance to that axis; reversing or
 * positively scaling `direction` leaves the geometry unchanged. A computed
 * squared radial distance at or below `1e-10` square angstroms suppresses the
 * whole evaluation before activation.
 *
 * The selection handle is borrowed only for this call. Native code copies the
 * selected indices and every parameter into owned host storage; no handle,
 * selection, or caller array is retained. Device mirrors are rebuilt before
 * the next evaluation, and this call performs no CUDA work. A valid empty
 * selection is accepted after parameter validation and adds no term.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @param[in] selection Borrowed live selection handle representing exactly the
 * restraint's atom count.
 * @param[in] origin Non-NULL pointer to `origin_len` finite axis-reference
 * coordinates in `[x, y, z]` order and angstrom units. The values are copied.
 * @param[in] origin_len Dimensionless number of elements in `origin`. It must
 * equal three.
 * @param[in] direction Non-NULL pointer to `direction_len` finite
 * dimensionless axis components with a positive finite Euclidean norm. The
 * values are copied.
 * @param[in] direction_len Dimensionless number of elements in `direction`. It
 * must equal three.
 * @param[in] force_constant Finite signed `K` in kilocalories per mole per
 * square angstrom. Zero is accepted.
 * @param[in] offset Finite signed CHARMM `DROFF` value in angstroms.
 * @param[in] activation Declared @ref apo_geometric_restraint_activation value.
 * @param[in] selection_mode Declared
 * @ref apo_geometric_restraint_selection_mode value. A nonempty RCM selection
 * requires a finite nonzero signed total of its active masses and finite mass
 * fractions, even when `force_constant` is zero.
 * @retval APO_STATUS_OK One term was appended, or a valid empty selection was
 * accepted without adding a term.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle, native object, or array
 * pointer is `NULL`; `origin_len` or `direction_len` is not three;
 * `activation` or `selection_mode` is not declared; the selection atom count
 * differs from the restraint atom count; an origin, direction, force-constant,
 * or offset value is NaN or infinite; the direction norm is not positive and
 * finite; the membership count would overflow; or an RCM mass requirement is
 * violated.
 * @retval APO_STATUS_RUNTIME_ERROR Copying indices, growing term storage,
 * constructing a diagnostic, or another standard or nonstandard C++ operation
 * failed.
 *
 * @post On failure, previously installed terms are unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_geometric_restraint_force_add_cylinder_from_selection(
    apo_geometric_restraint_force *restraint,
    const apo_atom_selection *selection, const double *origin,
    const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode);

/**
 * @brief Appends one harmonic cylinder term for a borrowed atom reference.
 *
 * The cylinder geometry matches
 * @ref apo_geometric_restraint_force_add_cylinder_from_selection. Only the
 * reference's zero-based atom index is used. It is validated against the
 * restraint's force-local atom range; the reference topology is neither
 * compared nor retained. The reference handle is borrowed only for this call,
 * every parameter is copied, and this call performs no CUDA work.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @param[in] atom Borrowed live atom-reference handle whose index lies in
 * `[0, N)`, where `N` is the restraint atom count.
 * @param[in] origin Non-NULL pointer to `origin_len` finite axis-reference
 * coordinates in `[x, y, z]` order and angstrom units. The values are copied.
 * @param[in] origin_len Dimensionless number of elements in `origin`. It must
 * equal three.
 * @param[in] direction Non-NULL pointer to `direction_len` finite
 * dimensionless axis components with a positive finite Euclidean norm. The
 * values are copied.
 * @param[in] direction_len Dimensionless number of elements in `direction`. It
 * must equal three.
 * @param[in] force_constant Finite signed `K` in kilocalories per mole per
 * square angstrom. Zero is accepted.
 * @param[in] offset Finite signed CHARMM `DROFF` value in angstroms.
 * @param[in] activation Declared @ref apo_geometric_restraint_activation value.
 * @param[in] selection_mode Declared
 * @ref apo_geometric_restraint_selection_mode value. RCM requires a finite
 * nonzero active mass for the referenced atom.
 * @retval APO_STATUS_OK One term was appended.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle, native object, or array
 * pointer is `NULL`; `origin_len` or `direction_len` is not three;
 * `activation` or `selection_mode` is not declared; the atom index lies
 * outside the restraint's atom range; an origin, direction, force-constant, or
 * offset value is NaN or infinite; the direction norm is not positive and
 * finite; the membership count would overflow; or the RCM mass requirement is
 * violated.
 * @retval APO_STATUS_RUNTIME_ERROR Growing term storage, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On failure, previously installed terms are unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_geometric_restraint_force_add_cylinder_from_atom_reference(
    apo_geometric_restraint_force *restraint, const apo_atom_reference *atom,
    const double *origin, const size_t origin_len, const double *direction,
    const size_t direction_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode);

/**
 * @brief Appends one harmonic sphere term for a borrowed atom selection.
 *
 * The reaction coordinate is the raw Cartesian distance to `center`, without
 * periodic imaging. A computed squared distance at or below `1e-10` square
 * angstroms suppresses the whole evaluation before activation.
 *
 * The selection handle is borrowed only for this call. Native code copies the
 * selected indices and every parameter into owned host storage; no handle,
 * selection, or caller array is retained. Device mirrors are rebuilt before
 * the next evaluation, and this call performs no CUDA work. A valid empty
 * selection is accepted after parameter validation and adds no term.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @param[in] selection Borrowed live selection handle representing exactly the
 * restraint's atom count.
 * @param[in] center Non-NULL pointer to `center_len` finite coordinates in
 * `[x, y, z]` order and angstrom units. The values are copied.
 * @param[in] center_len Dimensionless number of elements in `center`. It must
 * equal three.
 * @param[in] force_constant Finite signed `K` in kilocalories per mole per
 * square angstrom. Zero is accepted.
 * @param[in] offset Finite signed CHARMM `DROFF` value in angstroms.
 * @param[in] activation Declared @ref apo_geometric_restraint_activation value.
 * @param[in] selection_mode Declared
 * @ref apo_geometric_restraint_selection_mode value. A nonempty RCM selection
 * requires a finite nonzero signed total of its active masses and finite mass
 * fractions, even when `force_constant` is zero.
 * @retval APO_STATUS_OK One term was appended, or a valid empty selection was
 * accepted without adding a term.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle, native object, or array
 * pointer is `NULL`; `center_len` is not three; `activation` or
 * `selection_mode` is not declared; the selection atom count differs from the
 * restraint atom count; a center, force-constant, or offset value is NaN or
 * infinite; the membership count would overflow; or an RCM mass requirement is
 * violated.
 * @retval APO_STATUS_RUNTIME_ERROR Copying indices, growing term storage,
 * constructing a diagnostic, or another standard or nonstandard C++ operation
 * failed.
 *
 * @post On failure, previously installed terms are unchanged.
 * @note Native diagnostics name a nonfinite center component as an origin
 * component.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_geometric_restraint_force_add_sphere_from_selection(
    apo_geometric_restraint_force *restraint,
    const apo_atom_selection *selection, const double *center,
    const size_t center_len, const double force_constant, const double offset,
    const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode);

/**
 * @brief Appends one harmonic sphere term for a borrowed atom reference.
 *
 * The sphere geometry matches
 * @ref apo_geometric_restraint_force_add_sphere_from_selection. Only the
 * reference's zero-based atom index is used. It is validated against the
 * restraint's force-local atom range; the reference topology is neither
 * compared nor retained. The reference handle is borrowed only for this call,
 * every parameter is copied, and this call performs no CUDA work.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @param[in] atom Borrowed live atom-reference handle whose index lies in
 * `[0, N)`, where `N` is the restraint atom count.
 * @param[in] center Non-NULL pointer to `center_len` finite coordinates in
 * `[x, y, z]` order and angstrom units. The values are copied.
 * @param[in] center_len Dimensionless number of elements in `center`. It must
 * equal three.
 * @param[in] force_constant Finite signed `K` in kilocalories per mole per
 * square angstrom. Zero is accepted.
 * @param[in] offset Finite signed CHARMM `DROFF` value in angstroms.
 * @param[in] activation Declared @ref apo_geometric_restraint_activation value.
 * @param[in] selection_mode Declared
 * @ref apo_geometric_restraint_selection_mode value. RCM requires a finite
 * nonzero active mass for the referenced atom.
 * @retval APO_STATUS_OK One term was appended.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle, native object, or array
 * pointer is `NULL`; `center_len` is not three; `activation` or
 * `selection_mode` is not declared; the atom index lies outside the
 * restraint's atom range; a center, force-constant, or offset value is NaN or
 * infinite; the membership count would overflow; or the RCM mass requirement
 * is violated.
 * @retval APO_STATUS_RUNTIME_ERROR Growing term storage, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On failure, previously installed terms are unchanged.
 * @note Native diagnostics name a nonfinite center component as an origin
 * component.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_geometric_restraint_force_add_sphere_from_atom_reference(
    apo_geometric_restraint_force *restraint, const apo_atom_reference *atom,
    const double *center, const size_t center_len, const double force_constant,
    const double offset, const apo_geometric_restraint_activation activation,
    const apo_geometric_restraint_selection_mode selection_mode);

/**
 * @brief Replaces the force-wide mass vector used by every RCM term.
 *
 * Native code validates the complete vector and every installed RCM selection
 * before preparing independent host and device replacement storage, then
 * commits without throwing. Finite positive, zero, and negative individual
 * masses, negative totals, and tiny nonzero totals are accepted when every
 * installed RCM total is finite and nonzero and every selected mass fraction
 * is finite. Atomwise terms ignore masses. The masses apply to current and
 * later RCM terms and are preserved by
 * @ref apo_geometric_restraint_force_reset.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @param[in] masses Non-NULL pointer to `masses_len` readable finite values in
 * atom-index order, normally in atomic mass units. All `masses_len` values are
 * copied before native validation; the pointer is not retained.
 * @param[in] masses_len Dimensionless number of elements to copy from
 * `masses`. Native validation requires it to equal the restraint atom count.
 * @retval APO_STATUS_OK The active masses were replaced.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or `masses` is `NULL`, the
 * handle contains no native object, `masses_len` differs from the atom count,
 * a mass is NaN or infinite, or an installed RCM total or mass fraction would
 * become invalid.
 * @retval APO_STATUS_CUDA_ERROR Stream synchronization, replacement
 * allocation, transfer, or device synchronization failed.
 * @retval APO_STATUS_RUNTIME_ERROR Allocating the temporary copy of
 * `masses_len` values, allocating replacement host storage, constructing a
 * diagnostic, or another standard or nonstandard C++ operation failed.
 *
 * @post On failure, the previously active masses and derived RCM totals remain
 * in effect, and installed terms are unchanged.
 * @note The function synchronizes the restraint's private CUDA stream and
 * copies the replacement storage with synchronous device transfers, so it can
 * wait for other work on the device.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_geometric_restraint_force_set_masses(
    apo_geometric_restraint_force *restraint, const double *masses,
    const size_t masses_len);

/**
 * @brief Removes every geometric term while preserving masses and output.
 *
 * Host term storage is emptied, and device mirrors are rebuilt before the next
 * evaluation. The active masses, including masses installed by
 * @ref apo_geometric_restraint_force_set_masses, the stored box, the private
 * CUDA stream, and existing force, energy, and virial values are preserved.
 * Reset does not clear output and performs no CUDA work. New terms may be
 * added immediately, and resetting a restraint without terms is accepted.
 *
 * @param[in,out] restraint Borrowed live restraint handle.
 * @retval APO_STATUS_OK Every term was removed.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL` or the handle
 * contains no native object.
 * @retval APO_STATUS_RUNTIME_ERROR Allocating the replacement empty term
 * storage, constructing a diagnostic, or another standard or nonstandard C++
 * operation failed.
 *
 * @post On success, the restraint owns no terms. On failure, installed terms
 * are unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_geometric_restraint_force_reset(apo_geometric_restraint_force *restraint);

/**
 * @brief Subscribes a geometric restraint to a force manager.
 *
 * On success, the manager retains native shared ownership of the restraint,
 * its CUDA stream holder, its force array, and its energy-virial object. The
 * two C handles remain owned by their callers.
 *
 * The manager accepts this restraint only under `APO_PBC_P1`. Before changing
 * subscription state, native code requires the manager's current periodic
 * boundary condition to be `APO_PBC_P1` or `APO_PBC_P21` and supported by
 * every existing subscription, then rejects the restraint unless the condition
 * is `APO_PBC_P1`. While the restraint is subscribed,
 * @ref apo_force_manager_set_periodic_boundary_condition rejects `APO_PBC_P21`
 * without changing the manager.
 *
 * If the manager is initialized, subscription first initializes the restraint
 * with the manager atom count and box dimensions and synchronizes pending term
 * data to the device. Otherwise, manager initialization performs that hook
 * later. Duplicate tags are permitted, but the same native restraint may not
 * be subscribed twice to one manager.
 *
 * @param[in,out] force_manager Borrowed live manager handle. The handle is not
 * retained.
 * @param[in] restraint Borrowed live restraint handle. The manager retains the
 * underlying native object on success but does not retain this C handle.
 * @param[in] force_tag Borrowed non-NULL, nonempty, null-terminated byte
 * string used as the manager's diagnostic and energy-print label. Native code
 * copies the bytes and retains no caller pointer.
 * @retval APO_STATUS_OK The manager retained the restraint and its resources.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, `force_tag` is `NULL` or empty, the restraint is
 * already subscribed, the manager's periodic boundary condition is not
 * `APO_PBC_P1`, or immediate initialization rejects the manager atom count or
 * box.
 * @retval APO_STATUS_CUDA_ERROR Immediate initialization failed while
 * synchronizing, allocating, or transferring restraint term data.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the tag, growing manager
 * subscription storage, constructing a diagnostic, or another standard or
 * nonstandard C++ operation failed.
 *
 * @post On success, destroying the restraint C handle alone does not remove the
 * subscribed native object.
 * @post On failure, manager subscription state is unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_force_manager_subscribe_geometric_restraint_force(
    apo_force_manager *force_manager, apo_geometric_restraint_force *restraint,
    const char *force_tag);

/**
 * @brief Unsubscribes a geometric restraint by object identity.
 *
 * The manager removes the subscription whose retained native object is the
 * restraint owned by `restraint` and releases its shared references to the
 * restraint, CUDA stream holder, force array, and energy-virial object. It
 * does not reset, clear, deinitialize, or destroy the restraint. Terms,
 * masses, and existing outputs are preserved, and the restraint may be
 * subscribed again.
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
 * propagation, clearing, force evaluation, force aggregation, energy
 * aggregation, or virial aggregation, and it no longer constrains the
 * manager's periodic boundary condition.
 * @post On lookup or validation failure, manager subscription state is
 * unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_force_manager_unsubscribe_geometric_restraint_force(
    apo_force_manager *force_manager, apo_geometric_restraint_force *restraint);

#ifdef __cplusplus
}
#endif

#endif
