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
 * @brief Declares the C ABI for distance restraint configuration.
 */

#ifndef __APOCHARMM_C_DISTANCE_RESTRAINT_FORCE_H__
#define __APOCHARMM_C_DISTANCE_RESTRAINT_FORCE_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an owned distance-restraint force in the C ABI.
 *
 * A successful @ref apo_distance_restraint_force_create call returns one newly
 * owned handle. Release it with @ref apo_distance_restraint_force_destroy.
 * Configuration functions borrow a live handle for one call. Functions that
 * accept arrays copy all array elements and retain no caller pointer.
 *
 * One native object may own multiple restraint terms. For term `t`, pair `j`
 * has coefficient `c_tj`, raw primary-coordinate distance `r_tj`, distance
 * exponent `IVAL_t`, reference value `RVAL_t`, force constant `KVAL_t`, and
 * energy exponent `EVAL_t`. An active term uses:
 *
 * @code
 * D_t = sum_j(c_tj * r_tj^IVAL_t) - RVAL_t
 * U_t = SCALE * (KVAL_t / EVAL_t) * D_t^EVAL_t
 * @endcode
 *
 * Stored terms accumulate additively. The supported native specialization uses
 * raw primary-coordinate pair distances for the characterized nonperiodic and
 * orthorhombic P1 behavior. A stored manager box does not replace those
 * distances with minimum-image distances. Uncharacterized exponent values are
 * rejected. Behavior for nonorthorhombic, non-P1, and other uncharacterized
 * periodic configurations is not claimed.
 *
 * Every evaluated exact zero-distance pair is rejected before exponentiation
 * or gradient formation. A zero global scale returns zero contribution before
 * pair geometry is read.
 *
 * A successful force-manager subscription retains independent native shared
 * ownership of the restraint, CUDA stream holder, force array, and
 * energy-virial object. Destroying this C handle therefore does not remove an
 * existing subscription. Unsubscribe before destruction when later explicit
 * unsubscription is needed.
 *
 * The C ABI exposes configuration and manager subscription, not direct kernel
 * invocation or output access. Use `ForceManager` or `CharmmContext` to drive
 * force, energy, and virial calculation.
 *
 * A non-NULL pointer must designate a live handle created by apoCHARMM. Passing
 * a dangling, already-destroyed, or foreign pointer has undefined behavior.
 * The handle and native object provide no internal synchronization; callers
 * must serialize mutation, manager operations, calculation, unsubscription,
 * and destruction.
 *
 * Every status-returning function clears the calling thread's previous
 * diagnostic at entry. Success leaves it empty. On failure, call
 * @ref apo_last_error immediately on the same thread. Its borrowed pointer is
 * valid only until another diagnostic-changing C ABI call on that thread.
 *
 * @see DistanceRestraintForce
 */
typedef struct apo_distance_restraint_force apo_distance_restraint_force;

/**
 * @brief Selects the one-sided activation mode for one restraint term.
 *
 * The explicit values map one-to-one to the native
 * `DistanceRestraintCondition` enumeration.
 */
typedef enum apo_distance_restraint_condition {
  /** Applies no one-sided activation gate. */
  APO_DISTANCE_RESTRAINT_CONDITION_NONE = 0,
  /** Activates the term exactly when its deviation satisfies `D >= 0`. */
  APO_DISTANCE_RESTRAINT_CONDITION_POSITIVE = 1,
  /** Activates the term exactly when its deviation satisfies `D <= 0`. */
  APO_DISTANCE_RESTRAINT_CONDITION_NEGATIVE = -1
} apo_distance_restraint_condition;

/**
 * @brief Creates an owned distance-restraint handle.
 *
 * The native object uses fixed-point `long long int` force storage and
 * single-precision pair-gradient conversion. It has the requested fixed atom
 * count, no restraint terms, global scale one, unset box dimensions, an
 * atom-sized force array, one energy-virial object, and one private CUDA
 * stream.
 *
 * @param[out] out Non-NULL pointer to the caller's handle slot. The function
 * stores `NULL` in `*out` before native construction and stores a newly owned
 * handle on success.
 * @param[in] num_atoms Dimensionless atom count. The value must be greater than
 * zero and is already represented as C `int` by the function signature.
 * @retval APO_STATUS_OK `*out` contains a newly owned handle that must be
 * released with @ref apo_distance_restraint_force_destroy.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL` or `num_atoms` is not
 * positive.
 * @retval APO_STATUS_CUDA_ERROR Native CUDA-backed output allocation, force
 * allocation, or stream creation failed.
 * @retval APO_STATUS_RUNTIME_ERROR Handle allocation, native shared ownership,
 * host-storage allocation, diagnostic construction, or another standard or
 * nonstandard C++ operation failed.
 *
 * @post On every failure after a valid `out` pointer is accepted,
 * `*out == NULL`.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves nonempty text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_distance_restraint_force_create(
    apo_distance_restraint_force **out, const int num_atoms);

/**
 * @brief Destroys an owned distance-restraint handle.
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
APOCHARMM_C_API void
apo_distance_restraint_force_destroy(apo_distance_restraint_force *restraint);

/**
 * @brief Sets the global scale applied to every active restraint term.
 *
 * Positive, zero, and negative finite values are accepted. The scale is native
 * host scalar state and does not alter stored term definitions. Appending a
 * term preserves the active scale.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] scale Finite global multiplicative scale.
 * @retval APO_STATUS_OK The scale was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL`, the handle
 * contains no native object, or `scale` is NaN or infinite.
 * @retval APO_STATUS_RUNTIME_ERROR Diagnostic construction or another standard
 * or nonstandard C++ operation failed.
 *
 * @post On validation failure, the prior scale remains unchanged.
 * @note Requiring a finite value is the current apoCHARMM API policy. External
 * CHARMM magnitude limits remain unresolved outside the characterized cases.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_distance_restraint_force_set_scale(
    apo_distance_restraint_force *restraint, const double scale);

/**
 * @brief Appends one distance-restraint term from parallel pair arrays.
 *
 * Element `j` of `first_atom_indices`, `second_atom_indices`, and
 * `coefficients` defines one atom pair and its algebraic coefficient. The
 * arrays are borrowed for this call, copied into native storage, and never
 * retained. Atom indices use apoCHARMM's zero-based convention.
 *
 * For each pair, the two atom indices must be distinct and within the fixed
 * atom count. Every coefficient must be finite and nonzero. Repeated pairs are
 * accepted and accumulate algebraically. The force constant must be finite and
 * nonzero; positive and negative values are accepted. The reference value must
 * be finite.
 *
 * The deliberately supported typed distance-exponent set is
 * `{-2, -1, 0, 1, 2, 6, 7}`. The deliberately supported typed energy-exponent
 * set is `{1, 2, 3, 4}`. Complete native compatibility outside these
 * characterized sets is unresolved, so other values are rejected rather than
 * inferred. The activation condition must be one of the three declared
 * enumeration values.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @param[in] first_atom_indices Non-NULL pointer to `first_atom_indices_len`
 * contiguous first atom indices.
 * @param[in] first_atom_indices_len Dimensionless number of elements in
 * `first_atom_indices`. It must be nonzero.
 * @param[in] second_atom_indices Non-NULL pointer to `second_atom_indices_len`
 * contiguous second atom indices.
 * @param[in] second_atom_indices_len Dimensionless number of elements in
 * `second_atom_indices`. It must equal `first_atom_indices_len`.
 * @param[in] coefficients Non-NULL pointer to `coefficients_len` contiguous
 * algebraic pair coefficients.
 * @param[in] coefficients_len Dimensionless number of elements in
 * `coefficients`.
 * @param[in] force_constant Finite nonzero term force constant.
 * @param[in] reference_value Finite term reference value.
 * @param[in] distance_exponent Integer exponent applied to every pair distance.
 * @param[in] energy_exponent Integer exponent applied to the term deviation.
 * @param[in] condition One-sided activation mode.
 * @retval APO_STATUS_OK The term and independent copies of all pair data were
 * appended.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` or any array pointer is
 * `NULL`; the handle contains no native object; the atom-index array lengths
 * differ; appending would exceed a native `int` count limit; an atom index is
 * out of range; a pair contains the same atom twice; a coefficient is zero,
 * NaN, or infinite; `force_constant` is zero, NaN, or infinite;
 * `reference_value` is NaN or infinite; an exponent is outside its supported
 * set; or `condition` is not a declared enumeration value.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the C arrays, allocating or growing
 * native host storage, diagnostic construction, or another standard or
 * nonstandard C++ operation failed.
 *
 * @post On success, the new term follows all previously stored terms, no caller
 * pointer is retained, and the active global scale is unchanged.
 * @post On validation or allocation failure, previously stored definitions and
 * the global scale remain unchanged.
 * @note Finite-value validation and restriction to the characterized exponent
 * sets are current apoCHARMM API policies. They do not claim unrestricted
 * external CHARMM compatibility for unresolved values.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_distance_restraint_force_add_restraint(
    apo_distance_restraint_force *restraint, const int *first_atom_indices,
    const size_t first_atom_indices_len, const int *second_atom_indices,
    const size_t second_atom_indices_len, const double *coefficients,
    const size_t coefficients_len, const double force_constant,
    const double reference_value, const int distance_exponent,
    const int energy_exponent,
    const apo_distance_restraint_condition condition);

/**
 * @brief Removes every restraint term and restores the global scale to one.
 *
 * Reset changes stored configuration only. It does not clear force, energy, or
 * virial output accumulated by an earlier calculation.
 *
 * @param[in,out] restraint Borrowed live restraint handle. The handle is not
 * retained.
 * @retval APO_STATUS_OK Every term was removed and the scale was restored.
 * @retval APO_STATUS_INVALID_ARGUMENT `restraint` is `NULL` or the handle
 * contains no native object.
 * @retval APO_STATUS_RUNTIME_ERROR Reset-storage allocation, diagnostic
 * construction, or another standard or nonstandard C++ operation failed.
 *
 * @post On success, the object owns no terms and has global scale one. New
 * terms may be appended normally.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_distance_restraint_force_reset(apo_distance_restraint_force *restraint);

/**
 * @brief Subscribes a distance-restraint force to a force manager.
 *
 * On success, the manager retains native shared ownership of the restraint,
 * CUDA stream holder, force array, and energy-virial object. The two C handles
 * remain owned by their callers.
 *
 * If the manager is initialized, subscription first verifies the restraint's
 * fixed atom count and installs the manager's orthorhombic box dimensions.
 * Otherwise, manager initialization performs that hook later. Duplicate tags
 * are permitted, but the same native restraint object may not be subscribed
 * twice to one manager.
 *
 * @param[in,out] force_manager Borrowed live manager handle. The handle is not
 * retained.
 * @param[in] restraint Borrowed live restraint handle. The manager retains the
 * underlying native object on success but does not retain this C handle.
 * @param[in] force_tag Borrowed non-NULL, nonempty, null-terminated byte
 * string. Native code copies the bytes and retains no caller pointer.
 * @retval APO_STATUS_OK The manager retained the restraint and its resources.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle is `NULL`, either handle
 * contains no native object, `force_tag` is `NULL` or  empty, the restraint is
 * already subscribed, or immediate initialization detects an atom-count or box
 * mismatch.
 * @retval APO_STATUS_RUNTIME_ERROR Copying the tag, growing manager
 * subscription storage, diagnostic construction, or another standard or
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
APOCHARMM_C_API apo_status apo_force_manager_subscribe_distance_restraint_force(
    apo_force_manager *force_manager, apo_distance_restraint_force *restraint,
    const char *force_tag);

/**
 * @brief Unsubscribes a distance-restraint force by object identity.
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
 * propagation, clearing, force evaluation, force aggregation, energy
 * aggregation, or virial aggregation.
 * @post On lookup or validation failure, manager subscription state is
 * unchanged.
 * @note The function clears the calling thread's previous diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_force_manager_unsubscribe_distance_restraint_force(
    apo_force_manager *force_manager, apo_distance_restraint_force *restraint);

#ifdef __cplusplus
}
#endif

#endif
