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
 * @brief Declares the C ABI base CUDA-integrator interface.
 */

#ifndef __APOCHARMM_C_CUDA_INTEGRATOR_H__
#define __APOCHARMM_C_CUDA_INTEGRATOR_H__

#include "apocharmm_c/CharmmContext.h"
#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/Subscriber.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents a borrowed polymorphic CUDA-integrator view.
 *
 * A concrete integrator handle embeds this view. Obtain it with the concrete
 * `_as_cuda_integrator` function and keep the concrete owner alive for the
 * entire period of use. This type has no destroy function and must not be freed
 * by the caller.
 *
 * Functions in this header borrow the view for one call. Context attachment
 * retains the shared native context. Subscription retains the shared native
 * subscriber. The C pointers themselves are not retained.
 *
 * Status-returning calls clear the calling thread's previous diagnostic at
 * entry. On failure, call @ref apo_last_error immediately on the same thread.
 * Its returned pointer is borrowed and is invalidated by the next
 * diagnostic-changing C ABI call on that thread.
 *
 * The view provides no internal synchronization. Do not overlap propagation,
 * restart loading, mutation, subscription, unsubscription, or destruction
 * involving the same native integrator or context from multiple threads.
 *
 * @anchor c_apo_cuda_integrator
 * @see cuda_integrators
 * @see subscriber
 */
typedef struct apo_cuda_integrator apo_cuda_integrator;

/**
 * @brief Sets the public integrator time step.
 *
 * @param[in] integrator Borrowed live base view. The pointer may not be `NULL`.
 * @param[in] time_step Finite positive time step in picoseconds.
 *
 * @retval APO_STATUS_OK The time-step scalar was updated.
 * @retval APO_STATUS_INVALID_ARGUMENT `integrator` or its native object is
 * `NULL`, or `time_step` is non-finite or not positive.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception crossed the C
 * ABI boundary.
 *
 * @post On failure, the previous time-step scalar is unchanged.
 * @warning Existing constraint and concrete-integrator derived state is not
 * reinitialized.
 * @note On failure, diagnostic text is available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_cuda_integrator_set_time_step(
    apo_cuda_integrator *integrator, const double time_step);

/**
 * @brief Attaches a context and initializes concrete integrator state.
 *
 * @param[in] integrator Borrowed live base view. The pointer may not be `NULL`.
 * @param[in] context Borrowed live context handle. The C handle is not
 * retained; the shared native context is retained on success.
 *
 * @retval APO_STATUS_OK The context was retained and concrete initialization
 * completed.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle or native object is `NULL`,
 * or a context was already attached.
 * @retval APO_STATUS_NOT_INITIALIZED The context atom count or a concrete
 * initialization prerequisite, such as piston crystal type, is absent.
 * @retval APO_STATUS_NOT_IMPLEMENTED The direct base initialization hook was
 * reached.
 * @retval APO_STATUS_CUDA_ERROR CUDA allocation, setup, transfer, launch, or
 * synchronization failed.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected exception or explicitly
 * categorized concrete runtime failure occurred.
 *
 * @post On success, the source C context handle may be destroyed without
 * invalidating the integrator's native context.
 * @post On failure, the C base view does not retain the context. Native
 * per-atom storage resized before the failure can remain resized.
 * @note On failure, diagnostic text is available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_cuda_integrator_set_charmm_context(
    apo_cuda_integrator *integrator, apo_charmm_context *context);

/**
 * @brief Reinitializes concrete state using the already attached context.
 *
 * @param[in] integrator Borrowed live base view. The pointer may not be `NULL`.
 *
 * @retval APO_STATUS_OK Concrete initialization completed.
 * @retval APO_STATUS_INVALID_ARGUMENT The handle or native object is `NULL`, or
 * the configured context does not satisfy concrete validation.
 * @retval APO_STATUS_NOT_INITIALIZED No context is attached or a concrete
 * initialization prerequisite, such as a configured crystal type, is absent.
 * @retval APO_STATUS_NOT_IMPLEMENTED The direct base hook was reached.
 * @retval APO_STATUS_CUDA_ERROR CUDA initialization failed.
 * @retval APO_STATUS_RUNTIME_ERROR Another native failure occurred.
 *
 * @note Call between completed propagation calls after changing the context's
 * crystal type. This rebuilds concrete integration history and RNG state
 * without resetting the completed-step counter. Friction, RNG seed/sequence,
 * reference temperature/pressure, Nose-Hoover mass, and atomic velocities are
 * retained. When the piston layout changes, piston storage is rebuilt with
 * default masses and piston/Nose-Hoover dynamic history and phase averages are
 * reset; configure custom new-layout masses afterward if needed. Repeating
 * initialization with an unchanged type preserves the selected masses and
 * friction.
 * @warning Initialization is not transactional. Serialize it with propagation
 * and other access to the same integrator/context. Do not use this operation to
 * resume an exact restart; use
 * @ref apo_cuda_integrator_initialize_from_restart_file for that purpose.
 * @note On failure, diagnostic text is available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status
apo_cuda_integrator_initialize(apo_cuda_integrator *integrator);

/**
 * @brief Attaches one subscriber and caches its current reporting interval.
 *
 * @param[in] integrator Borrowed live base view. The pointer may not be `NULL`.
 * @param[in] subscriber Borrowed live subscriber base view. The native
 * subscriber is retained on success; preserve its concrete C owner when the
 * view will later be used for unsubscription.
 *
 * @retval APO_STATUS_OK The native subscriber and scheduling interval were
 * retained.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle or native object is `NULL`,
 * the subscriber is already attached, or subscriber attachment rejects the
 * context or integrator.
 * @retval APO_STATUS_NOT_INITIALIZED The native integrator cannot obtain the
 * shared self-reference required by subscriber attachment.
 * @retval APO_STATUS_RUNTIME_ERROR Host allocation failed or another unexpected
 * C++ exception crossed the boundary.
 *
 * @warning Attachment is not transactional. A failure after installing a
 * backlink or appending one scheduling array can leave partial native state.
 * @note On failure, diagnostic text is available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_cuda_integrator_subscribe(
    apo_cuda_integrator *integrator, apo_subscriber *subscriber);

/**
 * @brief Removes one subscriber from the scheduling arrays.
 *
 * @param[in] integrator Borrowed live base view. The pointer may not be `NULL`.
 * @param[in] subscriber Borrowed live subscriber view identifying the native
 * subscriber to remove.
 *
 * @retval APO_STATUS_OK The subscriber and cached interval were removed.
 * @retval APO_STATUS_INVALID_ARGUMENT Either handle or native object is `NULL`,
 * or the subscriber is not currently attached.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected C++ exception crossed the
 * boundary.
 *
 * @post The removed subscriber retains its previously installed context and
 * integrator backlinks.
 * @note On failure, diagnostic text is available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_cuda_integrator_unsubscribe(
    apo_cuda_integrator *integrator, apo_subscriber *subscriber);

/**
 * @brief Propagates dynamics and invokes scheduled subscribers.
 *
 * @param[in] integrator Borrowed live base view. The pointer may not be `NULL`.
 * @param[in] num_steps Positive number of dynamics steps.
 *
 * @retval APO_STATUS_OK Every requested step and matching callback completed.
 * @retval APO_STATUS_INVALID_ARGUMENT `integrator` or its native object is
 * `NULL`, `num_steps` is not positive, or a native participant rejects an
 * argument.
 * @retval APO_STATUS_NOT_INITIALIZED Context, force-manager, concrete
 * integrator, or subscriber state is not ready.
 * @retval APO_STATUS_NOT_IMPLEMENTED A concrete operation or subscriber reports
 * an unsupported operation.
 * @retval APO_STATUS_CUDA_ERROR A kernel, CUDA allocation, transfer, force
 * calculation, or synchronization failed. @retval APO_STATUS_RUNTIME_ERROR A
 * restart/output operation, NaN check, allocation, or other runtime operation
 * failed.
 *
 * @warning Propagation is not transactional. The completed-step counter
 * advances after each successful concrete step. State and output remain
 * observable after a later failure; the failing step may partly modify state.
 * @note On failure, diagnostic text is available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_cuda_integrator_propagate(
    apo_cuda_integrator *integrator, const int num_steps);

/**
 * @brief Initializes concrete integrator state from a restart file.
 *
 * @param[in] integrator Borrowed live base view. The pointer may not be `NULL`.
 * @param[in] path Borrowed non-`NULL`, null-terminated path. The string is
 * converted to a native file-system path for the call and is not retained. An
 * empty string reaches native file-open handling.
 *
 * @retval APO_STATUS_OK Compatible state was restored.
 * @retval APO_STATUS_INVALID_ARGUMENT `integrator`, its native object, or
 * `path` is `NULL`, or parsed state is incompatible with configured state.
 * @retval APO_STATUS_NOT_INITIALIZED No context is attached or required
 * concrete state is absent.
 * @retval APO_STATUS_NOT_IMPLEMENTED The direct base restart hook was reached,
 * or the file uses an unsupported dynamics type, crystal label, or extension.
 * @retval APO_STATUS_CUDA_ERROR Restored state could not be allocated,
 * transferred, or synchronized.
 * @retval APO_STATUS_RUNTIME_ERROR The file could not be opened or parsed, or
 * another runtime failure occurred.
 *
 * @warning Restart loading is not transactional. Earlier parsed or transferred
 * state can remain observable after failure.
 * @note On failure, diagnostic text is available through @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_cuda_integrator_initialize_from_restart_file(
    apo_cuda_integrator *integrator, const char *path);

#ifdef __cplusplus
}
#endif

#endif
