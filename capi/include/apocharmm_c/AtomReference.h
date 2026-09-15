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
 * @brief Declares the C ABI topology-aware atom-reference interface.
 */

#ifndef __APOCHARMM_C_ATOM_REFERENCE_H__
#define __APOCHARMM_C_ATOM_REFERENCE_H__

#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents one owned topology-aware atom identity in the C ABI.
 *
 * Each handle owns one native AtomReference. The native value retains shared
 * const ownership of one CharmmPSF object and one validated zero-based atom
 * index. The public PSF handle supplied to @ref apo_atom_reference_create is
 * borrowed for that call; destroying it afterward does not invalidate the
 * reference. A reference returned by @ref apo_atom_selector_select_atom is
 * likewise independent of the selector and public PSF handle lifetimes.
 *
 * Topology identity is native CharmmPSF object identity. Two separately parsed
 * PSFs are different topologies even when they came from the same path or have
 * identical contents. Equality requires both identical topology and identical
 * atom index.
 *
 * Atom indices are dimensionless and zero-based. A reference contains no
 * coordinate, force, energy, CUDA allocation, device-transfer, stream, or
 * synchronization state.
 *
 * A non-NULL pointer must designate a live handle created by apoCHARMM. Passing
 * a dangling, already-destroyed, or foreign pointer has undefined behavior.
 * The implementation provides no internal locking. Concurrent read-only
 * queries require every involved handle and its shared PSF to remain alive and
 * immutable; do not overlap destruction with another call on the same handle.
 *
 * Status-returning functions clear the calling thread's previous diagnostic at
 * entry. On failure, call @ref apo_last_error immediately on the same thread;
 * its borrowed pointer remains valid only until another diagnostic-changing C
 * ABI call on that thread.
 *
 * @see atom_selection
 */
typedef struct apo_atom_reference apo_atom_reference;

/**
 * @brief Creates a reference to one atom in one PSF.
 *
 * @param[out] out Non-NULL address receiving a newly owned atom-reference
 * handle. The function stores `NULL` before validating `psf` or `atom_index`
 * and leaves `*out == NULL` on every failure path. Release a successful result
 * with @ref apo_atom_reference_destroy.
 * @param[in] psf Borrowed live PSF handle. The public handle is not retained,
 * but its native shared ownership is copied into the new AtomReference. The
 * pointer may not be `NULL` and must contain a native PSF object.
 * @param[in] atom_index Dimensionless zero-based atom index. The value must be
 * in `[0, atom_count)`.
 * @retval APO_STATUS_OK A new owned atom-reference handle was stored in
 * `*out`.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, `psf` is `NULL`, the PSF
 * handle contains no native object, or `atom_index` is outside the native atom
 * range.
 * @retval APO_STATUS_NOT_INITIALIZED The native PSF atom count is negative.
 * @retval APO_STATUS_RUNTIME_ERROR Handle or native-object allocation failed,
 * or another unexpected standard or nonstandard C++ exception crossed the
 * boundary.
 *
 * @post On success, `*out` owns one reference that keeps the native PSF alive.
 * @post On failure with a valid `out` pointer, `*out == NULL`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_atom_reference_create(apo_atom_reference **out,
                                                     const apo_charmm_psf *psf,
                                                     const int atom_index);

/**
 * @brief Destroys an owned atom-reference handle.
 *
 * @param[in] reference Owned handle to release. `NULL` is accepted and is a
 * no-op. A non-NULL pointer is invalid after this call returns.
 *
 * @post No C++ exception escapes the C ABI boundary.
 * @note A normal destruction preserves the calling thread's existing
 * @ref apo_last_error diagnostic. An internal destruction failure cannot be
 * returned by this void API and may replace that diagnostic.
 * @warning The caller must not destroy the same handle twice or use it after
 * destruction.
 */
APOCHARMM_C_API void apo_atom_reference_destroy(apo_atom_reference *reference);

/**
 * @brief Returns the stored zero-based atom index.
 *
 * @param[out] atom_index Non-NULL output pointer. When this pointer is valid,
 * the function stores `-1` before validating `reference`, then stores the
 * dimensionless zero-based index on success. Valid atom indices are
 * nonnegative, so `-1` is an invalid sentinel.
 * @param[in] reference Borrowed live atom-reference handle. The pointer may not
 * be `NULL` and is not retained.
 * @retval APO_STATUS_OK The stored index was written to `*atom_index`.
 * @retval APO_STATUS_INVALID_ARGUMENT `atom_index` is `NULL`, `reference` is
 * `NULL`, or the handle contains no native AtomReference object.
 * @retval APO_STATUS_RUNTIME_ERROR Another unexpected standard or nonstandard
 * C++ exception crossed the query boundary.
 *
 * @post On failure after a valid `atom_index` pointer is accepted,
 * `*atom_index == -1`.
 * @warning Callers must check the returned status before using `*atom_index`;
 * the sentinel does not indicate success or replace status handling.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_atom_reference_get_atom_index(
    int *atom_index, const apo_atom_reference *reference);

/**
 * @brief Tests whether two references retain the same native PSF object.
 *
 * This is a pointer-identity comparison of the retained native CharmmPSF
 * objects. Atom indices do not affect the result.
 *
 * @param[out] has_same_topology Non-NULL output pointer. When this pointer is
 * valid, the function stores `false` before validating either handle, then
 * stores the topology-identity result on success.
 * @param[in] reference Borrowed live first atom-reference handle. The pointer
 * may not be `NULL` and is not retained.
 * @param[in] other Borrowed live second atom-reference handle. The pointer may
 * not be `NULL` and is not retained.
 * @retval APO_STATUS_OK The topology comparison was written to
 * `*has_same_topology`.
 * @retval APO_STATUS_INVALID_ARGUMENT `has_same_topology` is `NULL`, either
 * handle is `NULL`, or either handle contains no native AtomReference object.
 * @retval APO_STATUS_RUNTIME_ERROR Another unexpected standard or nonstandard
 * C++ exception crossed the query boundary.
 *
 * @post On failure after a valid output pointer is accepted,
 * `*has_same_topology == false`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_atom_reference_has_same_topology(
    bool *has_same_topology, const apo_atom_reference *reference,
    const apo_atom_reference *other);

/**
 * @brief Tests AtomReference value equality.
 *
 * Equality is true only when both references retain the same native CharmmPSF
 * object and store the same zero-based atom index.
 *
 * @param[out] equals Non-NULL output pointer. When this pointer is valid, the
 * function stores `false` before validating either handle, then stores the
 * equality result on success.
 * @param[in] reference Borrowed live first atom-reference handle. The pointer
 * may not be `NULL` and is not retained.
 * @param[in] other Borrowed live second atom-reference handle. The pointer may
 * not be `NULL` and is not retained.
 * @retval APO_STATUS_OK The equality result was written to `*equals`.
 * @retval APO_STATUS_INVALID_ARGUMENT `equals` is `NULL`, either handle is
 * `NULL`, or either handle contains no native AtomReference object.
 * @retval APO_STATUS_RUNTIME_ERROR Another unexpected standard or nonstandard
 * C++ exception crossed the query boundary.
 *
 * @post On failure after a valid output pointer is accepted,
 * `*equals == false`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_atom_reference_equals(bool *equals, const apo_atom_reference *reference,
                          const apo_atom_reference *other);

#ifdef __cplusplus
}
#endif

#endif
