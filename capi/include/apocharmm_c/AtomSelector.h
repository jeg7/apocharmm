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
 * @brief Declares the C ABI atom-selection parser interface.
 */

#ifndef __APOCHARMM_C_ATOM_SELECTOR_H__
#define __APOCHARMM_C_ATOM_SELECTOR_H__

#include "apocharmm_c/AtomReference.h"
#include "apocharmm_c/AtomSelection.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an owned atom-selection evaluator in the C ABI.
 *
 * A selector shares ownership of the native PSF supplied at construction. The
 * PSF handle is borrowed by @ref apo_atom_selector_create and may be destroyed
 * after creation; the selector keeps the native topology alive independently.
 * The topology is shared rather than cloned, so mutation through another native
 * owner can affect later selections.
 *
 * Selection calls read host-resident PSF metadata. The set-valued
 * @ref apo_atom_selector_select operation produces an independently owned
 * @ref apo_atom_selection containing zero or more atoms. The exact-one
 * @ref apo_atom_selector_select_atom operation requires exactly one selected
 * atom and produces an independently owned @ref apo_atom_reference that keeps
 * the native PSF alive. No CUDA allocation, transfer, stream work, or
 * synchronization is performed by the selector.
 *
 * A non-NULL pointer must designate a live handle created by apoCHARMM. Passing
 * a dangling, already-destroyed, or foreign pointer has undefined behavior.
 * The implementation provides no internal locking. Do not overlap destruction
 * with another call, and keep the shared PSF immutable during selection.
 *
 * Status-returning functions clear the calling thread's previous diagnostic at
 * entry. On failure, call @ref apo_last_error immediately on the same thread;
 * its borrowed pointer remains valid only until another diagnostic-changing C
 * ABI call on that thread.
 *
 * @see atom_selection
 */
typedef struct apo_atom_selector apo_atom_selector;

/**
 * @brief Creates an atom selector for a PSF.
 *
 * @param[out] out Non-NULL address receiving a newly owned selector handle. The
 * function stores `NULL` before validating `psf` and leaves `*out == NULL` on
 * every failure path. Release a successful result with
 * @ref apo_atom_selector_destroy.
 * @param[in] psf Borrowed live PSF handle. The public handle is not retained,
 * but its native shared object is retained by the new selector. The pointer may
 * not be `NULL` and must contain an initialized native PSF.
 * @retval APO_STATUS_OK A new owned selector was stored in `*out`.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, `psf` is `NULL`, or the
 * PSF handle contains no native object.
 * @retval APO_STATUS_NOT_INITIALIZED The native PSF atom count is negative.
 * @retval APO_STATUS_RUNTIME_ERROR Allocating the handle, selector, shared
 * ownership, or a diagnostic failed, or another unexpected standard or
 * nonstandard C++ exception crossed the boundary.
 *
 * @post On success, `*out` owns one selector that keeps the native PSF alive.
 * @post On failure with a valid `out` pointer, `*out == NULL`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_atom_selector_create(apo_atom_selector **out,
                                                    const apo_charmm_psf *psf);

/**
 * @brief Destroys an owned atom-selector handle.
 *
 * @param[in] selector Owned handle to release. `NULL` is accepted and is a
 * no-op. A non-NULL pointer is invalid after this call returns.
 *
 * @post No C++ exception escapes the C ABI boundary.
 * @note A normal destruction preserves the calling thread's existing
 * @ref apo_last_error diagnostic. An internal destruction failure cannot be
 * returned by this void API and may replace that diagnostic.
 * @warning The caller must not destroy the same handle twice or use it after
 * destruction.
 */
APOCHARMM_C_API void apo_atom_selector_destroy(apo_atom_selector *selector);

/**
 * @brief Evaluates one atom-selection expression.
 *
 * `selection_string` is parsed as a null-terminated byte string. Keywords and
 * dotted operators use ASCII spellings and case-insensitive matching. The C
 * interface cannot represent an embedded null byte; bytes after the first null
 * are not part of the expression. See @ref atom_selection for fields,
 * wildcards, ranges, precedence, and expansion operators.
 *
 * @param[out] out Non-NULL address receiving a newly owned selection handle.
 * The function stores `NULL` before validating other arguments and leaves
 * `*out == NULL` on every failure path. Release a successful result with
 * @ref apo_atom_selection_destroy.
 * @param[in] selector Borrowed live selector handle. The pointer may not be
 * `NULL` and is not retained.
 * @param[in] selection_string Borrowed nonempty, null-terminated expression.
 * The bytes are copied during tokenization and are not retained.
 * @retval APO_STATUS_OK A new owned selection was stored in `*out`.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, `selector` is `NULL`,
 * the selector handle contains no native object, `selection_string` is `NULL`
 * or empty, the expression has a lexical, syntax, operator, range, or
 * parenthesis error, or a stored PSF bonded-neighbor index is out of range.
 * @retval APO_STATUS_RUNTIME_ERROR The shared PSF violates parser residue,
 * group, or bonded-connectivity invariants; an internal parser invariant
 * failed; allocation failed; or another unexpected standard or nonstandard
 * C++ exception crossed the boundary.
 *
 * @pre Per-atom PSF metadata arrays remain consistent with the PSF atom count.
 * @post On success, `*out` owns an immutable selection independent of the
 * selector and PSF lifetimes.
 * @post On failure with a valid `out` pointer, `*out == NULL`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_atom_selector_select(
    apo_atom_selection **out, const apo_atom_selector *selector,
    const char *selection_string);

/**
 * @brief Evaluates one expression that must select exactly one atom.
 *
 * The same native tokenizer, parser, fields, wildcards, ranges, precedence,
 * and expansion operators used by @ref apo_atom_selector_select are applied.
 * Native AtomSelector code creates a temporary set-valued AtomSelection,
 * validates that its cardinality is exactly one, and then returns the
 * topology-aware atom identity. The temporary selection is destroyed before
 * this function returns.
 *
 * @param[out] out Non-NULL address receiving a newly owned atom-reference
 * handle. The function stores `NULL` before validating other arguments and
 * leaves `*out == NULL` on every failure path. Release a successful result with
 * @ref apo_atom_reference_destroy.
 * @param[in] selector Borrowed live selector handle. The pointer may not be
 * `NULL` and is not retained.
 * @param[in] selection_string Borrowed nonempty, null-terminated expression.
 * The bytes are copied during tokenization and are not retained.
 * @retval APO_STATUS_OK A new owned atom reference was stored in `*out`.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, `selector` is `NULL`,
 * the selector handle contains no native object, `selection_string` is `NULL`
 * or empty, the expression is invalid, a stored PSF bonded-neighbor index is
 * out of range, or the result contains zero or more than one atom.
 * @retval APO_STATUS_RUNTIME_ERROR The shared PSF violates parser residue,
 * group, or bonded-connectivity invariants; an internal parser invariant
 * failed; allocation failed; or another unexpected standard or nonstandard
 * C++ exception crossed the boundary.
 *
 * @pre Per-atom PSF metadata arrays remain consistent with the PSF atom count.
 * @post On success, `*out` owns one immutable topology-aware atom reference
 * that remains valid after the selector and public PSF handles are destroyed.
 * @post On failure with a valid `out` pointer, `*out == NULL`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_atom_selector_select_atom(
    apo_atom_reference **out, const apo_atom_selector *selector,
    const char *selection_string);

#ifdef __cplusplus
}
#endif

#endif
