// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/DcdSubscriber.h"
#include "apocharmm_c/detail/DcdSubscriberHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include <memory>

extern "C" apo_status apo_dcd_subscriber_create(apo_dcd_subscriber **out,
                                                const char *file_name) {
  const char *function_name = "apo_dcd_subscriber_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_dcd_subscriber>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_c_string(
            file_name, function_name, "file_name"));

        std::unique_ptr<apo_dcd_subscriber> handle(new apo_dcd_subscriber());
        handle->object = std::make_shared<DcdSubscriber>(file_name);
        handle->base.object = handle->object;

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_dcd_subscriber_create_with_report_frequency(apo_dcd_subscriber **out,
                                                const char *file_name,
                                                const int report_frequency) {
  const char *function_name = "apo_dcd_subscriber_create_with_report_frequency";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_dcd_subscriber>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_c_string(
            file_name, function_name, "file_name"));

        std::unique_ptr<apo_dcd_subscriber> handle(new apo_dcd_subscriber());
        handle->object =
            std::make_shared<DcdSubscriber>(file_name, report_frequency);
        handle->base.object = handle->object;

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_dcd_subscriber_destroy(apo_dcd_subscriber *subscriber) {
  delete subscriber;
  return;
}

extern "C" apo_status
apo_dcd_subscriber_as_subscriber(apo_subscriber **out,
                                 apo_dcd_subscriber *subscriber) {
  const char *function_name = "apo_dcd_subscriber_as_subscriber";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_subscriber>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_dcd_subscriber>(
                subscriber, function_name, "DcdSubscriber"));

        *out = &subscriber->base;

        return APO_STATUS_OK;
      },
      function_name);
}
