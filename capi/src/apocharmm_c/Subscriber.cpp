// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/Subscriber.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/SubscriberHandle.h"
#include "apocharmm_c/detail/Validation.h"

extern "C" apo_status
apo_subscriber_set_report_frequency(apo_subscriber *subscriber,
                                    const int report_frequency) {
  const char *function_name = "apo_subscriber_set_report_frequency";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_subscriber>(
                subscriber, function_name, "Subscriber"));

        subscriber->object->setReportFrequency(report_frequency);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_subscriber_get_report_frequency(int *report_frequency,
                                    const apo_subscriber *subscriber) {
  const char *function_name = "apo_subscriber_get_report_frequency";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_subscriber>(
                subscriber, function_name, "Subscriber"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<int>(
            report_frequency, function_name, "report_frequency"));

        *report_frequency = subscriber->object->getReportFrequency();

        return APO_STATUS_OK;
      },
      function_name);
}
