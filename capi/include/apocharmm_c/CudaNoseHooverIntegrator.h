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
 * @brief Declares the C ABI Nose-Hoover integrator interface.
 */

#ifndef __APOCHARMM_C_CUDA_NOSE_HOOVER_INTEGRATOR_H__
#define __APOCHARMM_C_CUDA_NOSE_HOOVER_INTEGRATOR_H__

#include "apocharmm_c/CudaIntegrator.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Owns a native Nose-Hoover CUDA integrator.
 *
 * Create the handle with @ref apo_cuda_nose_hoover_integrator_create and
 * release it with @ref apo_cuda_nose_hoover_integrator_destroy. The embedded
 * base view returned by
 * @ref apo_cuda_nose_hoover_integrator_as_cuda_integrator is borrowed and
 * remains valid only until this concrete handle is destroyed.
 *
 * Status calls clear the current thread's diagnostic at entry. On failure,
 * call @ref apo_last_error immediately on the same thread.
 *
 * @anchor c_apo_cuda_nose_hoover_integrator
 * @see cuda_integrators
 */
typedef struct apo_cuda_nose_hoover_integrator apo_cuda_nose_hoover_integrator;

/**
 * @brief Creates a Nose-Hoover integrator.
 *
 * @param[out] out Non-`NULL` output slot. The function sets `*out` to `NULL`
 * before later construction.
 * @param[in] time_step Finite positive time step in picoseconds.
 *
 * @retval APO_STATUS_OK A newly owned handle was written to `*out`.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, or `time_step` is
 * non-finite or not positive.
 * @retval APO_STATUS_CUDA_ERROR CUDA stream or fixed-state allocation failed.
 * @retval APO_STATUS_RUNTIME_ERROR Host allocation or another unexpected C++
 * exception occurred.
 *
 * @post On every failure after output-slot validation, `*out` is `NULL`.
 */
APOCHARMM_C_API apo_status apo_cuda_nose_hoover_integrator_create(
    apo_cuda_nose_hoover_integrator **out, const double time_step);

/**
 * @brief Destroys a Nose-Hoover integrator handle.
 *
 * @param[in] integrator Owned handle to release, or `NULL`.
 *
 * The function is non-throwing. Successful destruction does not clear an
 * existing thread-local diagnostic.
 */
APOCHARMM_C_API void apo_cuda_nose_hoover_integrator_destroy(
    apo_cuda_nose_hoover_integrator *integrator);

/**
 * @brief Sets the Nose-Hoover reference temperature.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] temperature Finite non-negative temperature in kelvin.
 *
 * @retval APO_STATUS_OK The temperature was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`, or
 * `temperature` is non-finite or negative.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_set_reference_temperature(
    apo_cuda_nose_hoover_integrator *integrator, const double temperature);

/**
 * @brief Sets the scalar Nose-Hoover coupling mass.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] mass Finite non-negative native coupling-mass value. Its exact
 * dimensional convention is not established by the repository.
 *
 * @retval APO_STATUS_OK The scalar was copied into native storage.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`, or
 * `mass` is non-finite or negative.
 * @retval APO_STATUS_CUDA_ERROR The scalar could not be copied to device
 * storage.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @warning Zero is accepted, but active propagation divides by this value.
 */
APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass(
    apo_cuda_nose_hoover_integrator *integrator, const double mass);

/**
 * @brief Selects the native old-temperature estimator.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] flag `true` selects element 1 for native thermostat feedback and
 * instantaneous temperature; `false` selects element 0.
 *
 * @retval APO_STATUS_OK The selector was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status apo_cuda_nose_hoover_integrator_use_old_temperature(
    apo_cuda_nose_hoover_integrator *integrator, const bool flag);

/**
 * @brief Resets both running-temperature values and their sample count.
 *
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK The averages were reset.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The device array could not be reset.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_reset_average_temperature(
    apo_cuda_nose_hoover_integrator *integrator);

/**
 * @brief Returns the Nose-Hoover reference temperature.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin on success.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output
 * pointer is `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure, a valid output location retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_get_reference_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator);

/**
 * @brief Returns the scalar Nose-Hoover coupling mass.
 *
 * @param[out] mass Non-`NULL` output receiving the native scalar on success.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*mass` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output
 * pointer is `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure, a valid output location retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_get_nose_hoover_piston_mass(
    double *mass, const apo_cuda_nose_hoover_integrator *integrator);

/**
 * @brief Returns one running-temperature estimator.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin on success.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output
 * pointer is `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The two-element average container could not be
 * transferred to the host.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure, a valid output location retains its previous contents.
 * @warning Current C ABI behavior returns element 0 when old-temperature mode
 * is enabled and element 1 when it is disabled. This is opposite the native
 * instantaneous-temperature selector and the documented array convention.
 */
APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_get_average_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator);

/**
 * @brief Returns the native selected instantaneous temperature.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin on success.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output
 * pointer is `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED No context is attached.
 * @retval APO_STATUS_CUDA_ERROR Kinetic energy could not be transferred.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure, a valid output location retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_nose_hoover_integrator_get_instantaneous_temperature(
    double *temperature, const apo_cuda_nose_hoover_integrator *integrator);

/**
 * @brief Returns the borrowed embedded base-integrator view.
 *
 * @param[out] out Non-`NULL` output slot set to `NULL` before later validation.
 * @param[in] integrator Borrowed live concrete handle.
 *
 * @retval APO_STATUS_OK `*out` points to the embedded borrowed base view.
 * @retval APO_STATUS_INVALID_ARGUMENT `out`, `integrator`, or its native object
 * is `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure after output-slot validation, `*out` is `NULL`.
 * @warning Do not destroy the returned view or retain it beyond destruction of
 * `integrator`.
 */
APOCHARMM_C_API apo_status apo_cuda_nose_hoover_integrator_as_cuda_integrator(
    apo_cuda_integrator **out, apo_cuda_nose_hoover_integrator *integrator);

#ifdef __cplusplus
}
#endif

#endif
