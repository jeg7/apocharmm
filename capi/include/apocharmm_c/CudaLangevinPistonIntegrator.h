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
 * @brief Declares the C ABI Langevin-piston integrator interface.
 */

#ifndef __APOCHARMM_C_CUDA_LANGEVIN_PISTON_INTEGRATOR_H__
#define __APOCHARMM_C_CUDA_LANGEVIN_PISTON_INTEGRATOR_H__

#include "apocharmm_c/CudaIntegrator.h"
#include "apocharmm_c/Enums.h"
#include "apocharmm_c/Export.h"
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
 * @brief Owns a native Langevin-piston CUDA integrator.
 *
 * Destroy the handle with @ref apo_cuda_langevin_piston_integrator_destroy. Its
 * embedded @ref apo_cuda_integrator view is borrowed and valid only while this
 * concrete handle remains alive.
 *
 * Configure the context's crystal type before attachment. Compatible changes
 * are allowed between completed propagation calls. After a type change, call
 * @ref apo_cuda_integrator_initialize, then configure custom new-layout masses
 * if needed. Friction and scalar settings are retained; piston/Nose-Hoover
 * dynamic history and phase averages are reset. Until initialization succeeds,
 * piston state getters/setters, propagation, and restart operations return
 * `APO_STATUS_NOT_INITIALIZED`. The crystal getter remains readable and
 * forwards the context's selection. Reinitializing an unchanged type preserves
 * masses and friction but rebuilds integration history and RNG state; the
 * completed-step counter is retained.
 *
 * Status calls clear the current thread's diagnostic at entry. On failure,
 * call @ref apo_last_error immediately on the same thread.
 *
 * @anchor c_apo_cuda_langevin_piston_integrator
 * @see cuda_integrators
 */
typedef struct apo_cuda_langevin_piston_integrator
    apo_cuda_langevin_piston_integrator;

/**
 * @brief Creates a Langevin-piston integrator.
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
APOCHARMM_C_API apo_status apo_cuda_langevin_piston_integrator_create(
    apo_cuda_langevin_piston_integrator **out, const double time_step);

/**
 * @brief Destroys a Langevin-piston handle.
 *
 * @param[in] integrator Owned handle to release, or `NULL`.
 *
 * The function is non-throwing. Successful destruction preserves an existing
 * thread-local diagnostic.
 */
APOCHARMM_C_API void apo_cuda_langevin_piston_integrator_destroy(
    apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Enables or disables Nose-Hoover temperature coupling.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] flag `true` enables Nose-Hoover coupling.
 *
 * @retval APO_STATUS_OK The flag was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_use_nose_hoover_thermostat(
    apo_cuda_langevin_piston_integrator *integrator, const bool flag);

/**
 * @brief Sets the reference temperature.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] temperature Finite non-negative temperature in kelvin.
 *
 * @retval APO_STATUS_OK The temperature and any active stochastic factors were
 * updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`, or
 * `temperature` is non-finite or negative.
 * @retval APO_STATUS_NOT_INITIALIZED Selected-crystal piston state is
 * inconsistent while factors are recomputed, or a context crystal change
 * requires reinitialization.
 * @retval APO_STATUS_CUDA_ERROR Updated factors could not be transferred.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_set_reference_temperature(
    apo_cuda_langevin_piston_integrator *integrator, const double temperature);

/**
 * @brief Sets the scalar Nose-Hoover coupling mass.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] mass Finite non-negative native coupling-mass value.
 *
 * @retval APO_STATUS_OK The scalar was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`, or
 * `mass` is non-finite or negative.
 * @retval APO_STATUS_CUDA_ERROR The scalar could not be copied to device
 * storage.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @warning Zero is accepted, but propagation divides by this value while
 * Nose-Hoover coupling is enabled.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_set_nose_hoover_piston_mass(
    apo_cuda_langevin_piston_integrator *integrator, const double mass);

/**
 * @brief Selects the native old-temperature estimator.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] flag `true` selects element 1 for native feedback and
 * instantaneous temperature.
 *
 * @retval APO_STATUS_OK The selector was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_use_old_temperature(
    apo_cuda_langevin_piston_integrator *integrator, const bool flag);

/**
 * @brief Sets the row-major reference pressure tensor.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] pressure_tensor Borrowed non-`NULL` contiguous values copied
 * during the call.
 * @param[in] pressure_tensor_len Number of available `double` values; it must
 * equal nine.
 *
 * @retval APO_STATUS_OK Nine finite atmosphere values were copied.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or buffer is
 * `NULL`; the length is not nine; or a value is non-finite.
 * @retval APO_STATUS_CUDA_ERROR The tensor could not be transferred to device
 * storage.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR Host allocation or another unexpected C++
 * exception occurred.
 *
 * @post On failure, the previous reference tensor is unchanged unless a later
 * transfer fails after native host-side assignment.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_set_reference_pressure(
    apo_cuda_langevin_piston_integrator *integrator,
    const double *pressure_tensor, const size_t pressure_tensor_len);

/**
 * @brief Enables or disables the constant-surface-tension branch.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] flag `true` enables the branch.
 *
 * @retval APO_STATUS_OK The flag was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 *
 * @warning The current API exposes no target-surface-tension setter and the
 * target container is not initialized after allocation. Do not enable this
 * branch until that implementation defect is corrected.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_set_constant_surface_tension(
    apo_cuda_langevin_piston_integrator *integrator, const bool flag);

/**
 * @brief Sets crystal-sized Langevin-piston masses.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] mass Borrowed non-`NULL` contiguous values copied during the call.
 * @param[in] mass_len Number of values. It must be one, two, or three at the C
 * layer and must exactly match the selected crystal degree-of-freedom count.
 *
 * @retval APO_STATUS_OK Masses, inverse masses, and forcing factors were
 * updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or buffer is
 * `NULL`; `mass_len` is outside one through three; the length does not match
 * the crystal; or a value is non-finite or negative.
 * @retval APO_STATUS_NOT_INITIALIZED No configured context has been attached,
 * or the context crystal type changed and requires reinitialization.
 * @retval APO_STATUS_CUDA_ERROR Device updates failed.
 * @retval APO_STATUS_RUNTIME_ERROR Host allocation or another unexpected C++
 * exception occurred.
 *
 * @note The exact public dimensional convention is not established by the
 * repository. A zero value stores zero inverse mass for that degree of freedom.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
    apo_cuda_langevin_piston_integrator *integrator, const double *mass,
    const size_t mass_len);

/**
 * @brief Sets the piston RNG seed.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] seed Dimensionless 64-bit seed.
 *
 * @retval APO_STATUS_OK The seed was stored and selected-crystal RNG states
 * were regenerated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_CUDA_ERROR RNG allocation or initialization failed.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_set_langevin_piston_friction_seed(
    apo_cuda_langevin_piston_integrator *integrator, const uint64_t seed);

/**
 * @brief Sets the Langevin-piston friction coefficient.
 *
 * @param[in] integrator Borrowed live handle.
 * @param[in] friction Finite non-negative value in inverse picoseconds.
 *
 * @retval APO_STATUS_OK Damping and random-force factors were updated.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`, or
 * `friction` is non-finite or negative.
 * @retval APO_STATUS_NOT_INITIALIZED No configured context has been attached,
 * or the context crystal type changed and requires reinitialization.
 * @retval APO_STATUS_CUDA_ERROR Random-force factors could not be transferred.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_set_langevin_piston_friction(
    apo_cuda_langevin_piston_integrator *integrator, const double friction);

/**
 * @brief Resets temperature and pressure running averages.
 *
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK All average state was reset.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`.
 * @retval APO_STATUS_CUDA_ERROR A device array could not be reset.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 */
APOCHARMM_C_API apo_status apo_cuda_langevin_piston_integrator_reset_averages(
    apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns the reference temperature.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_reference_temperature(
    double *temperature, const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns the scalar Nose-Hoover coupling mass.
 *
 * @param[out] mass Non-`NULL` output receiving the native scalar.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*mass` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The scalar container could not be transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The native container does not contain
 * exactly one element, or another unexpected exception occurred.
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_nose_hoover_piston_mass(
    double *mass, const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns one running-temperature estimator.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The average container could not be transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The average container does not contain two
 * elements, or another unexpected exception occurred.
 * @post On failure, a valid output retains its previous contents.
 * @warning Current C ABI behavior returns element 0 when old-temperature mode
 * is enabled and element 1 when it is disabled, opposite the native selector.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_average_temperature(
    double *temperature, const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Copies the reference pressure tensor.
 *
 * @param[out] pressure_tensor Non-`NULL` buffer receiving nine row-major
 * atmosphere values on success.
 * @param[in] pressure_tensor_len Number of available `double` elements; it must
 * be at least nine.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK Nine values were written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`,
 * the buffer is `NULL`, or its capacity is smaller than nine.
 * @retval APO_STATUS_CUDA_ERROR The native tensor could not be transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The native tensor does not contain nine
 * elements, or another unexpected exception occurred.
 * @post On failure, the caller buffer retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_reference_pressure_tensor(
    double *pressure_tensor, const size_t pressure_tensor_len,
    const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns the attached context's crystal type.
 *
 * @param[out] crystal_type Non-`NULL` output receiving the C enum value.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*crystal_type` was written, including
 * `APO_CRYSTAL_TYPE_UNSET` before context attachment. The current context
 * selection is returned even while piston state awaits reinitialization.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`, or the native enum cannot be represented.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status apo_cuda_langevin_piston_integrator_get_crystal_type(
    apo_crystal_type *crystal_type,
    const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Copies the Langevin-piston mass array.
 *
 * @param[out] num_mass Non-`NULL` output receiving the required element count.
 * @param[out] mass Buffer receiving the mass values. It may be `NULL` only when
 * the required count is zero.
 * @param[in] mass_len Number of available `double` elements in `mass`.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*num_mass` and all required mass values were written.
 * @retval APO_STATUS_INVALID_ARGUMENT `num_mass` is `NULL`; the handle or
 * native object is `NULL`; or a nonempty output buffer is `NULL` or too small.
 * @retval APO_STATUS_CUDA_ERROR A nonempty native mass array could not be
 * transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The native mass count exceeds three, or
 * another unexpected exception occurred.
 *
 * @post If `num_mass` is valid, it is set to zero before handle validation and
 * later set to the required count before buffer validation. A later failure can
 * therefore leave zero or the required count in `*num_mass`.
 * @post On failure, the `mass` buffer retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_langevin_piston_mass(
    size_t *num_mass, double *mass, const size_t mass_len,
    const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Copies the instantaneous pressure tensor.
 *
 * @param[out] pressure_tensor Non-`NULL` buffer receiving nine row-major
 * atmosphere values.
 * @param[in] pressure_tensor_len Buffer capacity in `double` elements; it must
 * be at least nine.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK Nine values were written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`,
 * the buffer is `NULL`, or capacity is smaller than nine.
 * @retval APO_STATUS_CUDA_ERROR The tensor could not be transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The native tensor does not contain nine
 * elements, or another unexpected exception occurred.
 * @post On failure, the buffer retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_tensor(
    double *pressure_tensor, const size_t pressure_tensor_len,
    const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns the instantaneous scalar pressure.
 *
 * @param[out] pressure_scalar Non-`NULL` output receiving atmospheres.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*pressure_scalar` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The scalar container could not be transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The native container does not contain one
 * element, or another unexpected exception occurred.
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_scalar(
    double *pressure_scalar,
    const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Copies the running-average pressure tensor.
 *
 * @param[out] pressure_tensor Non-`NULL` buffer receiving nine row-major
 * atmosphere values.
 * @param[in] pressure_tensor_len Buffer capacity in `double` elements; it must
 * be at least nine.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK Nine values were written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`,
 * the buffer is `NULL`, or capacity is smaller than nine.
 * @retval APO_STATUS_CUDA_ERROR The tensor could not be transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The native tensor does not contain nine
 * elements, or another unexpected exception occurred.
 * @post On failure, the buffer retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_average_pressure_tensor(
    double *pressure_tensor, const size_t pressure_tensor_len,
    const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns the running-average scalar pressure.
 *
 * @param[out] pressure_scalar Non-`NULL` output receiving atmospheres.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*pressure_scalar` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_CUDA_ERROR The scalar container could not be transferred.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR The native container does not contain one
 * element, or another unexpected exception occurred.
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_average_pressure_scalar(
    double *pressure_scalar,
    const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns the piston RNG seed.
 *
 * @param[out] seed Non-`NULL` output receiving the dimensionless 64-bit seed.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*seed` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED The context crystal type changed and
 * piston state requires reinitialization.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_langevin_piston_friction_seed(
    uint64_t *seed, const apo_cuda_langevin_piston_integrator *integrator);

/**
 * @brief Returns the native selected instantaneous temperature.
 *
 * @param[out] temperature Non-`NULL` output receiving kelvin.
 * @param[in] integrator Borrowed live handle.
 *
 * @retval APO_STATUS_OK `*temperature` was written.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle, native object, or output is
 * `NULL`.
 * @retval APO_STATUS_NOT_INITIALIZED No context is attached, or a context
 * crystal change requires reinitialization.
 * @retval APO_STATUS_CUDA_ERROR Kinetic energy could not be transferred.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception occurred.
 * @post On failure, a valid output retains its previous contents.
 */
APOCHARMM_C_API apo_status
apo_cuda_langevin_piston_integrator_get_instantaneous_temperature(
    double *temperature, const apo_cuda_langevin_piston_integrator *integrator);

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
apo_cuda_langevin_piston_integrator_as_cuda_integrator(
    apo_cuda_integrator **out, apo_cuda_langevin_piston_integrator *integrator);

#ifdef __cplusplus
}
#endif

#endif
