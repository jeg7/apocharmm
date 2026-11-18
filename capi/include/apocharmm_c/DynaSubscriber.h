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
 * @brief Declares the C ABI DYNA energy-output subscriber interface.
 */

#ifndef __APOCHARMM_C_DYNA_SUBSCRIBER_H__
#define __APOCHARMM_C_DYNA_SUBSCRIBER_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/Subscriber.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an owned CHARMM DYNA energy writer in the C ABI.
 *
 * A successful constructor returns a newly owned handle whose text output file
 * has already been created or truncated. Release it exactly once with
 * @ref apo_dyna_subscriber_destroy. The handle can provide a borrowed
 * @ref apo_subscriber view for frequency changes and integrator attachment.
 *
 * The native writer prints CHARMM's `DYNA` energy blocks: the header and a
 * step-0 block when a run starts, then one block after every reporting interval
 * of the run. Updates are requested indirectly by subscribing the base view to
 * an integrator and propagating dynamics.
 *
 * Do not overlap destruction, subscription, propagation, or frequency changes
 * involving the same handle from multiple threads.
 *
 * @see DynaSubscriber
 * @see subscriber
 */
typedef struct apo_dyna_subscriber apo_dyna_subscriber;

/**
 * @brief Creates a DYNA writer with the default 10-step interval.
 *
 * Ten steps is CHARMM's default `NPRINT`.
 *
 * @param[out] out Non-`NULL` output pointer. The function stores `NULL` before
 * validating `path` or constructing the writer, then stores a newly owned
 * handle on success.
 * @param[in] path Non-`NULL`, null-terminated filesystem path borrowed for the
 * call and copied by the native subscriber. The string must be nonempty. When
 * the resulting native path has a nonempty parent path, that parent path must
 * exist.
 *
 * @retval APO_STATUS_OK A newly owned handle was written to `*out` and the file
 * was created or truncated.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, `path` is `NULL`, the
 * path is empty, or its checked parent path does not exist.
 * @retval APO_STATUS_RUNTIME_ERROR The file could not be opened, native handle
 * or object allocation failed, or another unexpected C++ exception crossed the
 * boundary.
 *
 * @post On failure after a valid `out` pointer is accepted, `*out == NULL`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_dyna_subscriber_create(apo_dyna_subscriber **out,
                                                      const char *path);

/**
 * @brief Creates a DYNA writer with an explicit reporting interval.
 *
 * @param[out] out Non-`NULL` output pointer. The function stores `NULL` before
 * validating inputs or constructing the writer, then stores a newly owned
 * handle on success.
 * @param[in] path Non-`NULL`, null-terminated filesystem path borrowed for the
 * call and copied by the native subscriber. The string must be nonempty. When
 * the resulting native path has a nonempty parent path, that parent path must
 * exist.
 * @param[in] report_frequency Positive, dimensionless number of propagated
 * steps between DYNA blocks.
 *
 * @retval APO_STATUS_OK A newly owned handle was written to `*out` and the file
 * was created or truncated.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, `path` is `NULL`, the
 * path is empty, its checked parent path does not exist, or
 * `report_frequency` is not positive.
 * @retval APO_STATUS_RUNTIME_ERROR The file could not be opened, native handle
 * or object allocation failed, or another unexpected C++ exception crossed the
 * boundary.
 *
 * @post On failure after a valid `out` pointer is accepted, `*out == NULL`.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_dyna_subscriber_create_with_report_frequency(
    apo_dyna_subscriber **out, const char *path, const int report_frequency);

/**
 * @brief Destroys an owned DYNA-subscriber handle.
 *
 * @param[in] subscriber Owned handle to release. `NULL` is accepted and is a
 * no-op. A non-`NULL` pointer and every borrowed base view obtained from it are
 * invalid after this call returns.
 *
 * @post No C++ exception escapes the C ABI boundary.
 * @note Normal destruction preserves the calling thread's existing
 * @ref apo_last_error diagnostic. An internal destruction failure cannot be
 * returned by this void API and may replace that diagnostic.
 * @warning Destroy the handle exactly once. Native integrator ownership of the
 * C++ subscriber does not keep this C handle or its embedded base view alive.
 */
APOCHARMM_C_API void
apo_dyna_subscriber_destroy(apo_dyna_subscriber *subscriber);

/**
 * @brief Returns the borrowed base-subscriber view of a DYNA handle.
 *
 * @param[out] out Non-`NULL` output pointer. The function stores `NULL` before
 * validating `subscriber`, then stores a borrowed pointer on success.
 * @param[in] subscriber Borrowed live DYNA handle. The pointer may not be
 * `NULL` and is not retained by this call.
 *
 * @retval APO_STATUS_OK A borrowed base view was written to `*out`.
 * @retval APO_STATUS_INVALID_ARGUMENT `out` is `NULL`, `subscriber` is `NULL`,
 * or the handle contains no native DYNA object.
 * @retval APO_STATUS_RUNTIME_ERROR An unexpected standard or nonstandard C++
 * exception crossed the C ABI boundary.
 *
 * @post On failure after a valid `out` pointer is accepted, `*out == NULL`.
 * @note The returned pointer is valid only while `subscriber` remains alive. It
 * must not be destroyed separately.
 * @note The function clears the previous thread-local diagnostic at entry.
 * Success leaves it empty; failure leaves text available through
 * @ref apo_last_error.
 */
APOCHARMM_C_API apo_status apo_dyna_subscriber_as_subscriber(
    apo_subscriber **out, apo_dyna_subscriber *subscriber);

#ifdef __cplusplus
}
#endif

#endif
