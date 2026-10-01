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
 * @brief Declares the C ABI molecular-context interface.
 */

#ifndef __APOCHARMM_C_CHARMM_CONTEXT_H__
#define __APOCHARMM_C_CHARMM_CONTEXT_H__

#include "apocharmm_c/CharmmCrd.h"
#include "apocharmm_c/CharmmParameters.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>
#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque owning handle for a native CharmmContext.
 *
 * A non-NULL handle owns one shared native context and retains shared native
 * references to collaborators needed to preserve its lifetime. Input context
 * handles accepted by status-returning functions are borrowed and must contain
 * a live native object.
 *
 * Every status-returning function clears the calling thread's previous
 * diagnostic at entry. `APO_STATUS_OK` therefore leaves no stale diagnostic.
 * On failure, `apo_last_error()` returns a borrowed pointer to the current
 * thread's diagnostic until the next C ABI call on that thread or thread exit.
 *
 * Native `ApoCharmmErrorCode` values map to their corresponding `apo_status`
 * values. Legacy `std::invalid_argument` maps to
 * `APO_STATUS_INVALID_ARGUMENT`; other standard or unknown exceptions map to
 * `APO_STATUS_RUNTIME_ERROR`. No exception escapes a status-returning C ABI
 * entry point.
 *
 * Context mutation is not transactional. A failure from native or CUDA work
 * can leave earlier context-side changes observable. The handle and its native
 * object provide no internal thread synchronization.
 */
typedef struct apo_charmm_context apo_charmm_context;

/**
 * @brief Creates a CharmmContext from a ForceManager.
 *
 * The input handle is borrowed. The new context and its C handle retain shared
 * native ownership of the force manager and any imported collaborators.
 *
 * @param[out] out Non-NULL pointer to the caller's handle slot. The slot is set
 * to `NULL` before input validation and receives a newly owned handle on
 * success.
 * @param[in] force_manager Borrowed non-NULL force-manager handle containing a
 * live native object. The C handle itself is not retained.
 *
 * @retval APO_STATUS_OK The context was created and `*out` owns it.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is NULL, `force_manager` is NULL,
 * its native object is NULL, or imported native configuration is invalid.
 * @retval APO_STATUS_CUDA_ERROR GPU setup or triggered native initialization
 * failed in CUDA.
 * @retval APO_STATUS_RUNTIME_ERROR Handle allocation or another uncategorized
 * native construction operation failed.
 *
 * @post On any failure after `out` validation, `*out` remains NULL.
 * @see apo_charmm_context_destroy
 */
APOCHARMM_C_API apo_status apo_charmm_context_create(
    apo_charmm_context **out, const apo_force_manager *force_manager);

/**
 * @brief Creates a CharmmContext from a PSF and CHARMM parameters.
 *
 * The input handles are borrowed. Their native objects are retained through
 * shared ownership by the returned context handle.
 *
 * @param[out] out Non-NULL pointer to the caller's handle slot. The slot is set
 * to `NULL` before input validation and receives a newly owned handle on
 * success.
 * @param[in] psf Borrowed non-NULL PSF handle containing a live native object.
 * @param[in] parameters Borrowed non-NULL parameter handle containing a live
 * native object.
 *
 * @retval APO_STATUS_OK The context was created and `*out` owns it.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is NULL, either input handle or
 * native object is NULL, or imported native state is invalid.
 * @retval APO_STATUS_CUDA_ERROR GPU setup, per-atom storage, or triggered
 * native initialization failed in CUDA.
 * @retval APO_STATUS_RUNTIME_ERROR Handle allocation, topology or parameter
 * preparation, or another uncategorized native operation failed.
 *
 * @post On any failure after `out` validation, `*out` remains NULL.
 * @see apo_charmm_context_destroy
 */
APOCHARMM_C_API apo_status apo_charmm_context_create_from_psf_parameters(
    apo_charmm_context **out, const apo_charmm_psf *psf,
    const apo_charmm_parameters *parameters);

/**
 * @brief Destroys an owned CharmmContext handle.
 *
 * @param[in] context Owned handle to destroy. `NULL` is accepted and has no
 * effect.
 *
 * The function is non-throwing across the C ABI. Successful destruction does
 * not clear or replace an existing thread-local diagnostic because destruction
 * uses the dedicated non-throwing cleanup guard.
 */
APOCHARMM_C_API void apo_charmm_context_destroy(apo_charmm_context *context);

/**
 * @brief Sets the context's CHARMM parameter set.
 *
 * The input handle is borrowed. On success, the context handle retains shared
 * native ownership of the parameter object, so the caller may destroy the
 * original parameter handle.
 *
 * @param[in,out] context Borrowed non-NULL context handle to modify.
 * @param[in] parameters Borrowed non-NULL parameter handle containing a live
 * native object.
 *
 * @retval APO_STATUS_OK The parameter object was retained.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or native object is NULL, or
 * completed force configuration is invalid.
 * @retval APO_STATUS_CUDA_ERROR Triggered force-manager initialization failed
 * in CUDA.
 * @retval APO_STATUS_RUNTIME_ERROR Parameter or force initialization failed
 * for another runtime condition.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_prm(
    apo_charmm_context *context, apo_charmm_parameters *parameters);

/**
 * @brief Sets the context's PSF and imports atom charges and masses.
 *
 * The input handle is borrowed. On success, the context handle retains shared
 * native ownership of the PSF object.
 *
 * @param[in,out] context Borrowed non-NULL context handle to modify.
 * @param[in] psf Borrowed non-NULL PSF handle containing a live native object.
 *
 * @retval APO_STATUS_OK The PSF was retained and its atom data was imported.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or native object is NULL, or the
 * PSF atom count conflicts with the context's established atom count.
 * @retval APO_STATUS_CUDA_ERROR Per-atom resizing, transfer, or triggered force
 * initialization failed in CUDA.
 * @retval APO_STATUS_RUNTIME_ERROR PSF or force initialization failed for
 * another runtime condition.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_set_psf(apo_charmm_context *context, apo_charmm_psf *psf);

/**
 * @brief Sets the context's ForceManager.
 *
 * The input handle is borrowed. On success, the context handle retains shared
 * native ownership of the manager and any PSF or parameters imported from it.
 *
 * @param[in,out] context Borrowed non-NULL context handle to modify.
 * @param[in] force_manager Borrowed non-NULL force-manager handle containing a
 * live native object.
 *
 * @retval APO_STATUS_OK The manager was retained and state was reconciled.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or native object is NULL, or
 * reconciled force configuration is invalid.
 * @retval APO_STATUS_CUDA_ERROR State synchronization or initialization failed
 * in CUDA.
 * @retval APO_STATUS_RUNTIME_ERROR Force initialization failed for another
 * runtime condition.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_force_manager(
    apo_charmm_context *context, apo_force_manager *force_manager);

/**
 * @brief Sets coordinates and charges from a flat host buffer.
 *
 * The buffer contains `N` consecutive `[x, y, z, charge]` records. Coordinates
 * use angstroms and charges use elementary-charge units. The values are copied;
 * the caller retains ownership of the buffer.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] xyzq Borrowed contiguous `double` buffer. It must be non-NULL.
 * @param[in] xyzq_len Number of scalar `double` elements. It must be nonzero,
 * divisible by four, and exactly `4 * N`.
 *
 * @retval APO_STATUS_OK Coordinates and charges were copied and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or pointer is NULL, the length
 * is zero or not divisible by four, or the record count differs from the
 * context or initialized force-manager atom count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR Transfer, image centering, synchronization, or
 * neighbor-list rebuilding failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_coordinates_charges(
    apo_charmm_context *context, const double *xyzq, const size_t xyzq_len);

/**
 * @brief Sets coordinates from a flat host buffer.
 *
 * The buffer contains `N` consecutive `[x, y, z]` records in angstroms.
 * Existing charges are preserved. The values are copied; the caller retains
 * ownership of the buffer.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] xyz Borrowed contiguous `double` buffer. It must be non-NULL.
 * @param[in] xyz_len Number of scalar elements. It must be nonzero, divisible
 * by three, and exactly `3 * N`.
 *
 * @retval APO_STATUS_OK Coordinates were copied and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or pointer is NULL, the length
 * is zero or not divisible by three, or the record count differs from the atom
 * count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR Transfer, image centering, synchronization, or
 * neighbor-list rebuilding failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_coordinates_from_array(
    apo_charmm_context *context, const double *xyz, const size_t xyz_len);

/**
 * @brief Sets coordinates from a CharmmCrd handle.
 *
 * Coordinates are copied from the input object's double-precision storage.
 * The input handle and native object are not retained. Existing charges are
 * preserved.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] crd Borrowed non-NULL coordinate handle containing a live native
 * object.
 *
 * @retval APO_STATUS_OK Coordinates were copied and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or native object is NULL, or
 * the coordinate count differs from the atom count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR Transfer, image centering, synchronization, or
 * neighbor-list rebuilding failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_coordinates(
    apo_charmm_context *context, const apo_charmm_crd *crd);

/**
 * @brief Sets per-atom charges from a host buffer.
 *
 * Charges use elementary-charge units. Existing coordinates are preserved.
 * The values are copied; the caller retains ownership of the buffer.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] charges Borrowed contiguous `double` buffer with one element per
 * atom. It must be non-NULL.
 * @param[in] charges_len Number of charge elements. It must equal the context
 * atom count.
 *
 * @retval APO_STATUS_OK Charges were copied and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or pointer is NULL, or
 * `charges_len` differs from the atom count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR Either host-to-device transfer failed.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_set_charges(apo_charmm_context *context,
                               const double *charges, const size_t charges_len);

/**
 * @brief Sets velocities and inverse masses from a flat host buffer.
 *
 * The buffer contains `N` consecutive `[vx, vy, vz, inverse_mass]` records.
 * Velocity uses angstroms per AKMA time unit and inverse mass uses reciprocal
 * atomic mass units. Values are copied.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] xyzm Borrowed contiguous `double` buffer. It must be non-NULL.
 * @param[in] xyzm_len Number of scalar elements. It must be nonzero, divisible
 * by four, and exactly `4 * N`.
 *
 * @retval APO_STATUS_OK Values were copied and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or pointer is NULL, the length
 * is zero or not divisible by four, or the record count differs from the atom
 * count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR The host-to-device transfer failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_velocities_inverse_masses(
    apo_charmm_context *context, const double *xyzm, const size_t xyzm_len);

/**
 * @brief Sets velocities while preserving inverse masses.
 *
 * The buffer contains `N` consecutive `[vx, vy, vz]` records in angstroms per
 * AKMA time unit. Values are copied.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] xyz Borrowed contiguous `double` buffer. It must be non-NULL.
 * @param[in] xyz_len Number of scalar elements. It must be nonzero, divisible
 * by three, and exactly `3 * N`.
 *
 * @retval APO_STATUS_OK Velocities were copied and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or pointer is NULL, the length
 * is zero or not divisible by three, or the record count differs from the atom
 * count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR The host-to-device transfer failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_velocities(
    apo_charmm_context *context, const double *xyz, const size_t xyz_len);

/**
 * @brief Loads velocities from a CHARMM velocity file.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] path Borrowed non-NULL, nonempty, null-terminated path. The string
 * is copied into native storage for the duration of the call and is not
 * retained.
 *
 * @retval APO_STATUS_OK All velocity records were parsed and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle is NULL, `path` is NULL or
 * empty, or the file atom count differs from the context atom count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_RUNTIME_ERROR The file cannot be opened or its atom count
 * or a velocity record cannot be read or parsed.
 * @retval APO_STATUS_CUDA_ERROR The final host-to-device transfer failed.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_set_velocities_from_charmm_velocity_file(
    apo_charmm_context *context, const char *path);

/**
 * @brief Sets masses and stores their reciprocals.
 *
 * Each mass uses atomic mass units. The context stores `1.0 / mass` with the
 * corresponding velocity record. Values are copied.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] masses Borrowed contiguous `double` buffer with one mass per atom.
 * It must be non-NULL.
 * @param[in] masses_len Number of masses. It must equal the context atom count.
 *
 * @retval APO_STATUS_OK Reciprocal masses were stored and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or pointer is NULL, or
 * `masses_len` differs from the atom count.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR The host-to-device transfer failed.
 *
 * @warning The current native implementation does not reject zero, negative,
 * infinite, or NaN mass values before division.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_masses(
    apo_charmm_context *context, const double *masses, const size_t masses_len);

/**
 * @brief Sets the stored target temperature.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] temperature Temperature in kelvin. It must be finite and
 * non-negative.
 *
 * @retval APO_STATUS_OK The stored temperature was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The context handle or native object is
 * NULL, or `temperature` is non-finite or negative.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_temperature(
    apo_charmm_context *context, const double temperature);

/**
 * @brief Sets the periodic boundary condition.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] pbc Either `APO_PBC_P1` or `APO_PBC_P21`. `APO_PBC_UNSET` is an
 * invalid sentinel and does not select an aperiodic system.
 *
 * @retval APO_STATUS_OK The boundary condition was stored and propagated.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or native object is NULL, `pbc`
 * is `APO_PBC_UNSET`, `pbc` is not a declared value, or completed force
 * configuration is invalid.
 * @retval APO_STATUS_CUDA_ERROR Triggered initialization, image centering,
 * synchronization, or neighbor-list rebuilding failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_periodic_boundary_condition(
    apo_charmm_context *context, const apo_pbc pbc);

/**
 * @brief Sets orthorhombic box dimensions.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] box_dimensions Borrowed contiguous three-element `double` buffer
 * containing X, Y, Z lengths in angstroms. It must be non-NULL.
 * @param[in] box_dimensions_len Number of elements. It must equal three.
 *
 * @retval APO_STATUS_OK The dimensions were copied and propagated.
 * @retval APO_STATUS_INVALID_ARGUMENT A handle or buffer is NULL, the length
 * is not three, a dimension is not positive, an attached manager rejects a
 * non-finite dimension, or the cutoff exceeds half the X length.
 * @retval APO_STATUS_CUDA_ERROR Triggered initialization, image centering,
 * synchronization, or neighbor-list rebuilding failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_box_dimensions(
    apo_charmm_context *context, const double *box_dimensions,
    const size_t box_dimensions_len);

/**
 * @brief Sets the random seed used for generated velocities.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] seed Unsigned 64-bit dimensionless seed.
 *
 * @retval APO_STATUS_OK The seed was stored.
 * @retval APO_STATUS_INVALID_ARGUMENT The context handle or native object is
 * NULL.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_random_seed(
    apo_charmm_context *context, const uint64_t seed);

/**
 * @brief Selects constrained degree-of-freedom accounting.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] useHolonomicConstraints `true` to subtract water and SHAKE
 * constraints from the reported degree-of-freedom count.
 *
 * @retval APO_STATUS_OK The flag and degree-of-freedom count were updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The context handle or native object is
 * NULL.
 * @retval APO_STATUS_NOT_INITIALIZED The native context has no PSF or no force
 * manager.
 */
APOCHARMM_C_API
apo_status apo_charmm_context_use_holonomic_constraints(
    apo_charmm_context *context, const bool useHolonomicConstraints);

/**
 * @brief Sets the Ewald splitting parameter.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] kappa Finite, non-negative value in inverse angstroms. The value
 * is converted to native `float`.
 *
 * @retval APO_STATUS_OK The stored value was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid or the converted
 * value is non-finite or negative.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 *
 * @warning Configure this value before force-manager initialization.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_set_kappa(apo_charmm_context *context, const double kappa);

/**
 * @brief Sets the direct-space cutoff.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] cutoff Finite, positive distance in angstroms. The value is
 * converted to native `float`.
 *
 * @retval APO_STATUS_OK The stored cutoff was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid or the converted
 * cutoff is non-finite or not positive.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 *
 * @warning Configure this value before force-manager initialization.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_set_cutoff(apo_charmm_context *context, const double cutoff);

/**
 * @brief Sets the nonbonded distance exposed as `ctonnb`.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] ctonnb Finite, positive distance in angstroms. The value is
 * converted to native `float`.
 *
 * @retval APO_STATUS_OK The stored value was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid or the converted
 * value is non-finite or not positive.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 *
 * @warning The current backend forwards this value as `roff`. Configure it
 * before force-manager initialization.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_set_ctonnb(apo_charmm_context *context, const double ctonnb);

/**
 * @brief Sets the nonbonded distance exposed as `ctofnb`.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] ctofnb Finite, positive distance in angstroms. The value is
 * converted to native `float`.
 *
 * @retval APO_STATUS_OK The stored value was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid or the converted
 * value is non-finite or not positive.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 *
 * @warning The current backend forwards this value as `ron`. Configure it
 * before force-manager initialization.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_set_ctofnb(apo_charmm_context *context, const double ctofnb);

/**
 * @brief Sets the three-dimensional PME FFT grid.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] grid Borrowed contiguous array containing X, Y, Z grid sizes. It
 * must be non-NULL.
 * @param[in] grid_len Number of integer elements. It must equal three.
 *
 * @retval APO_STATUS_OK The three positive dimensions were stored.
 * @retval APO_STATUS_INVALID_ARGUMENT The context or buffer is invalid, the
 * length is not three, or any dimension is not positive.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 *
 * @warning Configure the grid before force-manager initialization.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_fft_grid(
    apo_charmm_context *context, const int *grid, const size_t grid_len);

/**
 * @brief Sets the PME interpolation spline order.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] order Positive dimensionless spline order.
 *
 * @retval APO_STATUS_OK The order was stored.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid or `order` is not
 * positive.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 *
 * @warning Configure the order before force-manager initialization.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_pme_spline_order(
    apo_charmm_context *context, const int order);

/**
 * @brief Sets the van der Waals model identifier.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] vdw_type Native model code in the inclusive range `[1, 6]`.
 *
 * @retval APO_STATUS_OK The model identifier was stored.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid or `vdw_type` is
 * outside `[1, 6]`.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 *
 * @warning Configure the model before force-manager initialization.
 */
APOCHARMM_C_API apo_status apo_charmm_context_set_vdw_type(
    apo_charmm_context *context, const int vdw_type);

/**
 * @brief Returns the context atom count.
 *
 * @param[out] num_atoms Non-NULL output pointer. On success it receives the
 * dimensionless atom count.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*num_atoms` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer, handle, or native
 * object is NULL.
 *
 * @post On failure, `*num_atoms` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_num_atoms(
    int *num_atoms, const apo_charmm_context *context);

/**
 * @brief Returns the current degree-of-freedom count.
 *
 * @param[out] ndegf Non-NULL output pointer. On success it receives the
 * dimensionless count.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*ndegf` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer, handle, or native
 * object is NULL.
 *
 * @post On failure, `*ndegf` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_num_degrees_of_freedom(
    int *ndegf, const apo_charmm_context *context);

/**
 * @brief Copies double-precision coordinates and charges to a host buffer.
 *
 * The output is `N` consecutive `[x, y, z, charge]` records. Coordinates use
 * angstroms and charges use elementary-charge units. Device storage is
 * transferred to the context's host mirror before copying.
 *
 * @param[out] xyzq Caller-owned output buffer. It may be NULL only when the
 * required output length is zero.
 * @param[in] xyzq_len Number of available `double` elements. It must be at
 * least `4 * N`.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK Exactly `4 * N` elements were written.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid, the buffer is
 * NULL when output is required, or `xyzq_len` is too small.
 * @retval APO_STATUS_CUDA_ERROR The device-to-host transfer failed.
 *
 * @post On a validation or transfer failure, the caller's output buffer is
 * unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_coordinates_charges(
    double *xyzq, const size_t xyzq_len, const apo_charmm_context *context);

/**
 * @brief Copies velocities and inverse masses to a host buffer.
 *
 * Despite the function name, the fourth value is inverse mass, not mass. The
 * output is `N` consecutive `[vx, vy, vz, inverse_mass]` records. Velocity uses
 * angstroms per AKMA time unit and inverse mass uses reciprocal atomic mass
 * units.
 *
 * @param[out] xyzm Caller-owned output buffer. It may be NULL only when the
 * required output length is zero.
 * @param[in] xyzm_len Number of available `double` elements. It must be at
 * least `4 * N`.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK Exactly `4 * N` elements were written.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid, the buffer is
 * NULL when output is required, or `xyzm_len` is too small.
 * @retval APO_STATUS_CUDA_ERROR The device-to-host transfer failed.
 *
 * @post On a validation or transfer failure, the caller's output buffer is
 * unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_velocity_mass(
    double *xyzm, const size_t xyzm_len, const apo_charmm_context *context);

/**
 * @brief Returns the periodic boundary condition.
 *
 * @param[out] pbc Non-NULL output pointer receiving the mapped `apo_pbc` value.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*pbc` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer, handle, or native
 * object is NULL, or the native PBC value is not representable by `apo_pbc`.
 *
 * @post On failure, `*pbc` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_periodic_boundary_condition(
    apo_pbc *pbc, const apo_charmm_context *context);

/**
 * @brief Copies the three box dimensions.
 *
 * @param[out] box_dimensions Caller-owned output buffer for X, Y, Z lengths in
 * angstroms. It must be non-NULL.
 * @param[in] box_dimensions_len Number of available `double` elements. It must
 * be at least three.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK Three values were written.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid, the output
 * buffer is NULL, or its capacity is less than three.
 * @retval APO_STATUS_RUNTIME_ERROR The native context returns a vector whose
 * size is not exactly three.
 *
 * @post On failure, the output buffer is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_box_dimensions(
    double *box_dimensions, const size_t box_dimensions_len,
    const apo_charmm_context *context);

/**
 * @brief Returns the random seed.
 *
 * @param[out] seed Non-NULL output pointer receiving the unsigned 64-bit seed.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*seed` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer, handle, or native
 * object is NULL.
 *
 * @post On failure, `*seed` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_random_seed(
    uint64_t *seed, const apo_charmm_context *context);

/**
 * @brief Returns the orthorhombic box volume.
 *
 * @param[out] volume Non-NULL output pointer receiving cubic angstroms.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*volume` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer, handle, or native
 * object is NULL.
 * @retval APO_STATUS_NOT_INITIALIZED Valid positive box dimensions have not
 * been set.
 *
 * @post On failure, `*volume` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_volume(
    double *volume, const apo_charmm_context *context);

/**
 * @brief Returns the stored Ewald splitting parameter.
 *
 * @param[out] kappa Non-NULL output pointer receiving inverse angstroms.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*kappa` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer or context is invalid.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 * @post On failure, `*kappa` is unchanged.
 */
APOCHARMM_C_API apo_status
apo_charmm_context_get_kappa(double *kappa, const apo_charmm_context *context);

/**
 * @brief Returns the stored direct-space cutoff.
 *
 * @param[out] cutoff Non-NULL output pointer receiving angstroms.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*cutoff` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer or context is invalid.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 * @post On failure, `*cutoff` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_cutoff(
    double *cutoff, const apo_charmm_context *context);

/**
 * @brief Returns the distance exposed as `ctonnb`.
 *
 * @param[out] ctonnb Non-NULL output pointer receiving angstroms.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*ctonnb` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer or context is invalid.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 * @post On failure, `*ctonnb` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_ctonnb(
    double *ctonnb, const apo_charmm_context *context);

/**
 * @brief Returns the distance exposed as `ctofnb`.
 *
 * @param[out] ctofnb Non-NULL output pointer receiving angstroms.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*ctofnb` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer or context is invalid.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 * @post On failure, `*ctofnb` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_ctofnb(
    double *ctofnb, const apo_charmm_context *context);

/**
 * @brief Copies the stored three-dimensional PME FFT grid.
 *
 * @param[out] grid Caller-owned integer buffer in X, Y, Z order. It must be
 * non-NULL.
 * @param[in] grid_len Number of available integer elements. It must be at least
 * three.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK Three grid dimensions were written.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid, `grid` is NULL,
 * or `grid_len` is less than three.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 * @retval APO_STATUS_RUNTIME_ERROR The native manager returns a grid whose size
 * is not exactly three.
 * @post On failure, the output buffer is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_fft_grid(
    int *grid, const size_t grid_len, const apo_charmm_context *context);

/**
 * @brief Returns the stored PME interpolation spline order.
 *
 * @param[out] order Non-NULL output pointer receiving the dimensionless order.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*order` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer or context is invalid.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 * @post On failure, `*order` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_pme_spline_order(
    int *order, const apo_charmm_context *context);

/**
 * @brief Returns the stored van der Waals model identifier.
 *
 * @param[out] vdw_type Non-NULL output pointer receiving the native model code.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*vdw_type` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer or context is invalid.
 * @retval APO_STATUS_NOT_INITIALIZED No force manager is attached.
 * @post On failure, `*vdw_type` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_vdw_type(
    int *vdw_type, const apo_charmm_context *context);

/**
 * @brief Creates an owned C handle for the context's ForceManager.
 *
 * The returned handle shares the existing native manager; it does not clone
 * force state. Destroying either handle does not invalidate the other native
 * owner.
 *
 * @param[out] out Non-NULL pointer to the caller's force-manager handle slot.
 * The slot is set to NULL before context validation.
 * @param[in] context Borrowed non-NULL context handle.
 *
 * @retval APO_STATUS_OK `*out` owns a newly allocated C handle sharing the
 * native force manager.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is NULL or the context handle or
 * native object is NULL.
 * @retval APO_STATUS_NOT_INITIALIZED The context has no force manager.
 * @retval APO_STATUS_RUNTIME_ERROR Allocation of the new C handle failed.
 *
 * @post On any failure after `out` validation, `*out` remains NULL.
 * @see apo_force_manager_destroy
 */
APOCHARMM_C_API apo_status apo_charmm_context_get_force_manager(
    apo_force_manager **out, const apo_charmm_context *context);

/**
 * @brief Assigns sampled velocities at a temperature.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] temperature Sampling temperature in kelvin. It must be finite and
 * non-negative.
 *
 * @retval APO_STATUS_OK Velocities were generated and transferred.
 * @retval APO_STATUS_INVALID_ARGUMENT The context is invalid or `temperature`
 * is non-finite or negative.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR The velocity transfer failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_assign_velocities_at_temperature(
    apo_charmm_context *context, const double temperature);

/**
 * @brief Computes the instantaneous kinetic temperature.
 *
 * @param[out] temperature Non-NULL output pointer receiving kelvin.
 * @param[in,out] context Borrowed non-NULL context handle. Kinetic-energy
 * bookkeeping and host mirrors can be updated.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT An output pointer or context is invalid.
 * @retval APO_STATUS_NOT_INITIALIZED Atom or velocity storage is inconsistent,
 * or the degree-of-freedom count is not positive.
 * @retval APO_STATUS_CUDA_ERROR Kinetic-energy clearing, launch,
 * synchronization, or transfer failed.
 *
 * @post On failure, `*temperature` is unchanged.
 */
APOCHARMM_C_API apo_status apo_charmm_context_compute_temperature(
    double *temperature, apo_charmm_context *context);

/**
 * @brief Computes forces, potential energy, and virial.
 *
 * @param[in,out] context Borrowed non-NULL context handle.
 * @param[in] reset Requests force reset behavior. The current native
 * `ForceManager` rejects `true`.
 * @param[in] print `true` to write a CHARMM-style energy table to standard
 * output.
 *
 * @retval APO_STATUS_OK The force, energy, and virial evaluation completed.
 * @retval APO_STATUS_INVALID_ARGUMENT The context handle or native object is
 * NULL.
 * @retval APO_STATUS_NOT_INITIALIZED The force manager is missing or not
 * initialized.
 * @retval APO_STATUS_NOT_IMPLEMENTED `reset` is true or printing is requested
 * for a composite force manager.
 * @retval APO_STATUS_CUDA_ERROR Force calculation, transfer, or synchronization
 * failed in CUDA.
 * @retval APO_STATUS_RUNTIME_ERROR Energy-table formatting or another
 * uncategorized native operation failed.
 */
APOCHARMM_C_API apo_status apo_charmm_context_calculate_potential_energy(
    apo_charmm_context *context, const bool reset, const bool print);

#ifdef __cplusplus
}
#endif

#endif
