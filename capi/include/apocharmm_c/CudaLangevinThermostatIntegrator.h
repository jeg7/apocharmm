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
 * @brief Declares the C ABI Langevin-thermostat integrator interface.
 */

#ifndef __APOCHARMM_C_CUDA_LANGEVIN_THERMOSTAT_INTEGRATOR_H__
#define __APOCHARMM_C_CUDA_LANGEVIN_THERMOSTAT_INTEGRATOR_H__

#include "apocharmm_c/CudaIntegrator.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Owns a native Langevin thermostat CUDA integrator.
 *
 * Destroy the handle with
 * @ref apo_cuda_langevin_thermostat_integrator_destroy. Its embedded base view
 * is borrowed and valid only while this concrete handle remains alive.
 *
 * Status calls clear the current thread's diagnostic at entry. On failure,
 * call @ref apo_last_error immediately on the same thread.
 *
 * @anchor c_apo_cuda_langevin_thermostat_integrator
 * @see cuda_integrators
 */
typedef struct apo_cuda_langevin_thermostat_integrator
    apo_cuda_langevin_thermostat_integrator;

/**
 * @brief Creates a Langevin thermostat integrator.
 *
 * @param[out] out Non-`NULL` output slot set to `NULL` before construction.
 * @param[in] time_step Finite positive time step in picoseconds.
 *
 * @retval APO_STATUS_OK A newly owned handle was written to `*out`.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, or `time_step` is
 * non-finite or not positive.
 * @retval APO_STATUS_CUDA_ERROR CUDA stream or fixed-state allocation failed.
 * @retval APO_STATUS_RUNTIME_ERROR Host allocation, random-source
 * initialization, or another unexpected C++ exception failed.
 *
 * @post On failure after output-slot validation, `*out` is `NULL`.
 */
APOCHARMM_C_API apo_status apo_cuda_langevin_thermostat_integrator_create(
    apo_cuda_langevin_thermostat_integrator **out, const double time_step);

/**
 * @brief Destroys a Langevin thermostat handle.
 *
 * @param[in] integrator Owned handle to release, or `NULL`.
 *
 * The function is non-throwing. Successful destruction preserves an existing
 * thread-local diagnostic.
 */
APOCHARMM_C_API void apo_cuda_langevin_thermostat_integrator_destroy(
    apo_cuda_langevin_thermostat_integrator *integrator);

/**
 * @brief Sets the reference temperature.
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
apo_cuda_langevin_thermostat_integrator_set_reference_temperature(
    apo_cuda_langevin_thermostat_integrator *integrator,
    const double temperature);

/**
 * @brief Sets the thermostat friction coefficient.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] friction Finite non-negative friction in inverse picoseconds.
 *
 * @retval APO_STATUS_OK The value and derived per-step coefficient were
 * updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`, or
 * `friction` is non-finite or negative.
 * @retval APO_STATUS_CUDA_ERROR Context-attached reinitialization required CUDA
 * work that failed.
 * @retval APO_STATUS_RUNTIME_ERROR Host allocation or another unexpected C++
 * exception occurred.
 *
 * @note When a context is attached and no propagation request has been made,
 * the setter immediately reinitializes concrete state.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_set_thermostat_friction(
    apo_cuda_langevin_thermostat_integrator *integrator, const double friction);

/**
 * @brief Sets the thermostat RNG seed.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] seed Dimensionless 64-bit seed.
 *
 * @retval APO_STATUS_OK The seed was stored and attached-context states were
 * regenerated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_CUDA_ERROR RNG allocation or initialization failed.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post The recorded RNG sequence position is preserved.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_set_thermostat_rng_seed(
    apo_cuda_langevin_thermostat_integrator *integrator, const uint64_t seed);

/**
 * @brief Resets both running-temperature values and their sample count.
 *
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK The averages were reset.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The device average array could not be reset.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_reset_average_temperature(
    apo_cuda_langevin_thermostat_integrator *integrator);

/**
 * @brief Returns the reference temperature.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin on success.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_get_reference_temperature(
    double *temperature,
    const apo_cuda_langevin_thermostat_integrator *integrator);

/**
 * @brief Returns the thermostat friction coefficient.
 *
 * @param[out] friction Non-`NULL` output receiving inverse picoseconds.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*friction` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_get_thermostat_friction(
    double *friction,
    const apo_cuda_langevin_thermostat_integrator *integrator);

/**
 * @brief Returns the thermostat RNG seed.
 *
 * @param[out] seed Non-`NULL` output receiving the dimensionless 64-bit seed.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*seed` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_get_thermostat_rng_seed(
    uint64_t *seed, const apo_cuda_langevin_thermostat_integrator *integrator);

/**
 * @brief Returns the three-point running-average temperature.
 *
 * @param[out] temperature Non-`NULL` output receiving element 0 in kelvin.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The average array could not be transferred.
 * @retval APO_STATUS_RUNTIME_ERROR The native average array does not contain
 * exactly two elements, or another unexpected exception occurred.
 *
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_get_average_temperature(
    double *temperature,
    const apo_cuda_langevin_thermostat_integrator *integrator);

/**
 * @brief Returns the instantaneous three-point temperature.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin on success.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED No context is attached.
 * @retval APO_STATUS_CUDA_ERROR Kinetic energy could not be transferred.
 * @retval APO_STATUS_RUNTIME_ERROR The kinetic-energy array does not contain
 * exactly two elements, or another unexpected exception occurred.
 *
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_get_instantaneous_temperature(
    double *temperature,
    const apo_cuda_langevin_thermostat_integrator *integrator);

/**
 * @brief Returns the borrowed embedded base-integrator view.
 *
 * @param[out] out Non-`NULL` output slot set to `NULL` before validation.
 * @param[in] integrator Borrowed live concrete handle.
 *
 * @retval APO_STATUS_OK `*out` points to the embedded base view.
 * @retval APO_STATUS_INVALID_ARGUMENT `out`, `integrator`, or its native object
 * is `NULL`.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @post On failure after output-slot validation, `*out` is `NULL`.
 * @warning Do not destroy the returned view or use it after `integrator` is
 * destroyed.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_thermostat_integrator_as_cuda_integrator(
    apo_cuda_integrator **out,
    apo_cuda_langevin_thermostat_integrator *integrator);

#ifdef __cplusplus
}
#endif

#endif
