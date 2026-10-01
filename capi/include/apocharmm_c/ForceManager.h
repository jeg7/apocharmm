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
 * @brief Declares the C ABI force-manager interface.
 */

#ifndef __APOCHARMM_C_FORCE_MANAGER_H__
#define __APOCHARMM_C_FORCE_MANAGER_H__

#include "apocharmm_c/CharmmParameters.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Enums.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an opaque owning ForceManager handle.
 *
 * A handle created by @ref apo_force_manager_create owns a native manager and
 * retains native shared ownership of the PSF and parameter set used to create
 * it. The original PSF and parameter C handles may therefore be destroyed
 * without invalidating the manager.
 *
 * Release an owned handle exactly once with
 * @ref apo_force_manager_destroy. Inputs named `force_manager` in other
 * functions are borrowed for the duration of the call.
 *
 * Status-returning functions clear the calling thread's previous native
 * diagnostic on entry. A successful call leaves that diagnostic empty. A
 * failed call leaves nonempty thread-local text available through
 * @ref apo_last_error until the next guarded C ABI call on the same thread or
 * until thread exit. Handles provide no internal synchronization for
 * concurrent host mutation or force-manager use.
 */
typedef struct apo_force_manager apo_force_manager;

/**
 * @brief Creates an owning ForceManager handle.
 *
 * The function retains the native PSF and parameter objects and constructs an
 * uninitialized native manager. No force backends are initialized by this
 * call.
 *
 * @param[out] out Non-null output slot. It is set to `NULL` before collaborator
 * validation and receives a newly owned handle on success.
 * @param[in] psf Borrowed non-null live PSF handle.
 * @param[in] parameters Borrowed non-null live parameter handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if `out`, `psf`, or `parameters` is
 * `NULL`, or if either collaborator handle contains no native object.
 * @return `APO_STATUS_CUDA_ERROR` if native manager construction fails while
 * allocating or initializing CUDA-backed virial storage.
 * @return `APO_STATUS_RUNTIME_ERROR` if handle, manager, or native shared-state
 * allocation fails or another uncategorized standard exception is caught.
 *
 * @post On success, `*out` is non-null and owned by the caller.
 * @post On failure after validating `out`, `*out` remains `NULL`.
 * @post Success clears the calling thread's diagnostic; failure leaves a
 * nonempty diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_force_manager_create(apo_force_manager **out, const apo_charmm_psf *psf,
                         const apo_charmm_parameters *parameters);

/**
 * @brief Destroys an owning ForceManager handle.
 *
 * @param[in] force_manager Owned handle to destroy, or `NULL`. Passing `NULL`
 * is accepted and has no effect.
 *
 * @post Native destruction is contained within a non-throwing C ABI boundary.
 * @post A normal call preserves the calling thread's previous
 * @ref apo_last_error diagnostic rather than clearing it.
 *
 * @note Do not use the handle after this call.
 */
APOCHARMM_C_API void
apo_force_manager_destroy(apo_force_manager *force_manager);

/**
 * @brief Sets three orthorhombic box dimensions.
 *
 * The values are copied before return. Existing native and subscribed force
 * objects receive the update when present.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] box_dimensions Non-null buffer of exactly three `double` values in
 * `[x, y, z]` order, in angstroms.
 * @param[in] box_dimensions_len Number of elements in `box_dimensions`; it must
 * be exactly `3`.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle or buffer is invalid, the
 * length is not `3`, or any value is non-finite or not positive.
 * @return `APO_STATUS_CUDA_ERROR` if an existing native or subscribed force
 * reports a categorized CUDA failure while receiving the box.
 * @return `APO_STATUS_RUNTIME_ERROR` if temporary native vector allocation or
 * another uncategorized native operation fails.
 *
 * @post On success, the manager stores the copied dimensions.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 *
 * @warning A downstream force failure can occur after the manager and earlier
 * force objects have accepted the new box.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_box_dimensions(
    apo_force_manager *force_manager, const double *box_dimensions,
    const size_t box_dimensions_len);

/**
 * @brief Sets the Ewald splitting parameter.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] kappa Finite non-negative value in inverse angstroms. The C
 * `double` is narrowed to native `float` before validation and storage.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or the
 * narrowed value is non-finite or negative.
 * @return `APO_STATUS_RUNTIME_ERROR` if native diagnostic construction or
 * another uncategorized standard operation fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_kappa(
    apo_force_manager *force_manager, const double kappa);

/**
 * @brief Sets the direct-space pair-list cutoff.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] cutoff Finite positive distance in angstroms. The C `double` is
 * narrowed to native `float`.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or the
 * narrowed cutoff is non-finite or not positive.
 * @return `APO_STATUS_RUNTIME_ERROR` if native diagnostic construction or
 * another uncategorized standard operation fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 * @note Native initialization later requires the cutoff not to exceed half the
 * X box length.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_cutoff(
    apo_force_manager *force_manager, const double cutoff);

/**
 * @brief Sets the outer switching distance exposed as `ctonnb`.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] ctonnb Finite positive distance in angstroms. The C `double` is
 * narrowed to native `float`.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or the
 * narrowed value is non-finite or not positive.
 * @return `APO_STATUS_RUNTIME_ERROR` if native diagnostic construction or
 * another uncategorized standard operation fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 * @note The current native backend uses this as the distance at which switching
 * reaches zero and does not validate its ordering relative to `ctofnb`.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_ctonnb(
    apo_force_manager *force_manager, const double ctonnb);

/**
 * @brief Sets the inner switching distance exposed as `ctofnb`.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] ctofnb Finite positive distance in angstroms. The C `double` is
 * narrowed to native `float`.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or the
 * narrowed value is non-finite or not positive.
 * @return `APO_STATUS_RUNTIME_ERROR` if native diagnostic construction or
 * another uncategorized standard operation fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 * @note The current native backend uses this as the distance at which switching
 * begins and does not validate its ordering relative to `ctonnb`.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_ctofnb(
    apo_force_manager *force_manager, const double ctofnb);

/**
 * @brief Sets the three PME FFT grid dimensions.
 *
 * The input values are copied before return.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] grid Non-null buffer of exactly three positive `int` values in
 * `[x, y, z]` axis order.
 * @param[in] grid_len Number of elements in `grid`; it must be exactly `3`.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle or buffer is invalid, the
 * length is not `3`, or any grid value is not positive.
 * @return `APO_STATUS_RUNTIME_ERROR` if native diagnostic construction or
 * another uncategorized standard operation fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_fft_grid(
    apo_force_manager *force_manager, const int *grid, const size_t grid_len);

/**
 * @brief Sets the PME interpolation spline order.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] order Positive dimensionless spline order.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `order` is
 * not positive.
 * @return `APO_STATUS_RUNTIME_ERROR` if native diagnostic construction or
 * another uncategorized standard operation fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_pme_spline_order(
    apo_force_manager *force_manager, const int order);

/**
 * @brief Sets the periodic boundary condition.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] pbc Either `APO_PBC_P1` or `APO_PBC_P21`. `APO_PBC_UNSET` is an
 * invalid sentinel and does not select an aperiodic system.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid, `pbc` is
 * `APO_PBC_UNSET`, or `pbc` is not a delcared @ref apo_pbc value.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_periodic_boundary_condition(
    apo_force_manager *force_manager, const apo_pbc pbc);

/**
 * @brief Sets the native van der Waals model code.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] vdw_type Integer model code in `[1, 6]`.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `vdw_type`
 * is outside `[1, 6]`.
 * @return `APO_STATUS_RUNTIME_ERROR` if native diagnostic construction or
 * another uncategorized standard operation fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_vdw_type(
    apo_force_manager *force_manager, const int vdw_type);

/**
 * @brief Enables or disables native energy-decomposition printing.
 *
 * @param[in] force_manager Borrowed non-null live manager handle.
 * @param[in] flag Whether energy-producing native force calculations should
 * print their decomposition to standard output.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_set_print_energy_decomposition(
    apo_force_manager *force_manager, const bool flag);

/**
 * @brief Returns the PSF atom count.
 *
 * @param[out] num_atoms Non-null output pointer. It is set to `0` before the
 * manager handle is validated, then replaced with the PSF atom count on
 * success.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if `num_atoms` is `NULL` or the handle
 * is invalid.
 * @return `APO_STATUS_NOT_INITIALIZED` if the native manager has no PSF.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On success, `*num_atoms` contains the current native count.
 * @post On failure after validating `num_atoms`, `*num_atoms` is `0`.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_num_atoms(
    int *num_atoms, const apo_force_manager *force_manager);

/**
 * @brief Reports whether native initialization completed.
 *
 * @param[out] flag Non-null output pointer receiving the native initialized
 * flag on success.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `flag` is
 * `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, the previous contents of `*flag` remain unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_is_initialized(
    bool *flag, const apo_force_manager *force_manager);

/**
 * @brief Copies the three stored box dimensions.
 *
 * @param[out] box_dimensions Non-null output buffer with capacity for at least
 * three `double` values. The first three elements receive `[x, y, z]` lengths
 * in angstroms. Elements beyond index `2` are untouched.
 * @param[in] box_dimensions_len Capacity of `box_dimensions`; it must be at
 * least `3`.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle or output buffer is
 * invalid or `box_dimensions_len` is less than `3`.
 * @return `APO_STATUS_RUNTIME_ERROR` if copying the native box vector or
 * another uncategorized standard operation fails.
 *
 * @post On failure, output-buffer contents remain unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_box_dimensions(
    double *box_dimensions, const size_t box_dimensions_len,
    const apo_force_manager *force_manager);

/**
 * @brief Returns the stored Ewald splitting parameter.
 *
 * @param[out] kappa Non-null output pointer receiving the value in inverse
 * angstroms as a `double`.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `kappa` is
 * `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*kappa` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_kappa(
    double *kappa, const apo_force_manager *force_manager);

/**
 * @brief Returns the stored pair-list cutoff.
 *
 * @param[out] cutoff Non-null output pointer receiving the value in angstroms
 * as a `double`.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `cutoff`
 * is `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*cutoff` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_cutoff(
    double *cutoff, const apo_force_manager *force_manager);

/**
 * @brief Returns the outer switching distance exposed as `ctonnb`.
 *
 * @param[out] ctonnb Non-null output pointer receiving the value in angstroms.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `ctonnb`
 * is `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*ctonnb` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_ctonnb(
    double *ctonnb, const apo_force_manager *force_manager);

/**
 * @brief Returns the inner switching distance exposed as `ctofnb`.
 *
 * @param[out] ctofnb Non-null output pointer receiving the value in angstroms.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `ctofnb`
 * is `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*ctofnb` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_ctofnb(
    double *ctofnb, const apo_force_manager *force_manager);

/**
 * @brief Copies the three stored PME FFT dimensions.
 *
 * @param[out] grid Non-null output buffer with capacity for at least three
 * `int` values. The first three elements receive `[nfftx, nffty, nfftz]`;
 * trailing elements are untouched.
 * @param[in] grid_len Capacity of `grid`; it must be at least `3`.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle or output buffer is
 * invalid or `grid_len` is less than `3`.
 * @return `APO_STATUS_RUNTIME_ERROR` if native vector allocation or another
 * uncategorized standard operation fails.
 *
 * @post On failure, output-buffer contents remain unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_fft_grid(
    int *grid, const size_t grid_len, const apo_force_manager *force_manager);

/**
 * @brief Returns the PME interpolation spline order.
 *
 * @param[out] order Non-null output pointer receiving the dimensionless order.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `order` is
 * `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*order` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_pme_spline_order(
    int *order, const apo_force_manager *force_manager);

/**
 * @brief Returns the stored periodic boundary condition.
 *
 * @param[out] pbc Non-null output pointer receiving a declared @ref apo_pbc
 * value.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle or output pointer is
 * invalid, or if the native manager contains an unknown `PBC` value.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*pbc` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_periodic_boundary_condition(
    apo_pbc *pbc, const apo_force_manager *force_manager);

/**
 * @brief Returns the native van der Waals model code.
 *
 * @param[out] vdw_type Non-null output pointer receiving the stored integer
 * code.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `vdw_type`
 * is `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*vdw_type` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_get_vdw_type(
    int *vdw_type, const apo_force_manager *force_manager);

/**
 * @brief Reports whether the native manager is composite.
 *
 * @param[out] flag Non-null output pointer receiving the native result.
 * @param[in] force_manager Borrowed non-null live manager handle.
 *
 * @return `APO_STATUS_OK` on success.
 * @return `APO_STATUS_INVALID_ARGUMENT` if the handle is invalid or `flag` is
 * `NULL`.
 * @return `APO_STATUS_RUNTIME_ERROR` if an uncategorized standard operation
 * fails.
 *
 * @post On failure, `*flag` remains unchanged.
 * @post Success clears the thread-local diagnostic; failure leaves a nonempty
 * diagnostic available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_force_manager_is_composite(
    bool *flag, const apo_force_manager *force_manager);

#ifdef __cplusplus
}
#endif

#endif
