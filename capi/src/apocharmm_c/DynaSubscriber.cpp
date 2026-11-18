// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/DynaSubscriber.h"
#include "apocharmm_c/detail/DynaSubscriberHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include <filesystem>
#include <memory>

extern "C" apo_status apo_dyna_subscriber_create(apo_dyna_subscriber **out,
                                                 const char *path) {
  const char *function_name = "apo_dyna_subscriber_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_dyna_subscriber>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<char>(path, function_name, "path"));

        std::unique_ptr<apo_dyna_subscriber> handle(new apo_dyna_subscriber());
        handle->object =
            std::make_shared<DynaSubscriber>(std::filesystem::path(path));
        handle->base.object = handle->object;

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_dyna_subscriber_create_with_report_frequency(
    apo_dyna_subscriber **out, const char *path, const int report_frequency) {
  const char *function_name =
      "apo_dyna_subscriber_create_with_report_frequency";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_dyna_subscriber>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<char>(path, function_name, "path"));

        std::unique_ptr<apo_dyna_subscriber> handle(new apo_dyna_subscriber());
        handle->object = std::make_shared<DynaSubscriber>(
            std::filesystem::path(path), report_frequency);
        handle->base.object = handle->object;

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_dyna_subscriber_destroy(apo_dyna_subscriber *subscriber) {
  const char *function_name = "apo_dyna_subscriber_destroy";
  apocharmm_c::guard_destroy(
      [subscriber](void) -> void {
        delete subscriber;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status
apo_dyna_subscriber_as_subscriber(apo_subscriber **out,
                                  apo_dyna_subscriber *subscriber) {
  const char *function_name = "apo_dyna_subscriber_as_subscriber";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_subscriber>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_dyna_subscriber>(
                subscriber, function_name, "DynaSubscriber"));

        *out = &subscriber->base;

        return APO_STATUS_OK;
      },
      function_name);
}
