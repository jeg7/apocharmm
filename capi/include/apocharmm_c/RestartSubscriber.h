// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_RESTART_SUBSCRIBER_H__
#define __APOCHARMM_C_RESTART_SUBSCRIBER_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/Subscriber.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_restart_subscriber apo_restart_subscriber;

APOCHARMM_C_API apo_status apo_restart_subscriber_create(
    apo_restart_subscriber **out, const char *file_name);

APOCHARMM_C_API apo_status apo_restart_subscriber_create_with_report_frequency(
    apo_restart_subscriber **out, const char *file_name,
    const int report_frequency);

APOCHARMM_C_API void
apo_restart_subscriber_destroy(apo_restart_subscriber *subscriber);

// JEG260522: Returns a borrowed base-subscriber view.
// The returned pointer is valud only while "subscriber" is alive.
APOCHARMM_C_API apo_status apo_restart_subscriber_as_subscriber(
    apo_subscriber **out, apo_restart_subscriber *subscriber);

#ifdef __cplusplus
}
#endif

#endif
