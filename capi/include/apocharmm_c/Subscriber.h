// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_SUBSCRIBER_H__
#define __APOCHARMM_C_SUBSCRIBER_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_subscriber apo_subscriber;

APOCHARMM_C_API apo_status apo_subscriber_set_report_frequency(
    apo_subscriber *subscriber, const int report_frequency);

APOCHARMM_C_API apo_status apo_subscriber_get_report_frequency(
    int *report_frequency, const apo_subscriber *subscriber);

#ifdef __cplusplus
}
#endif

#endif
