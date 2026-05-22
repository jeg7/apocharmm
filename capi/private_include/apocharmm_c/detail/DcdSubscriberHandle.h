// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_DCD_SUBSCRIBER_HANDLE_H__
#define __APOCHARMM_C_DETAIL_DCD_SUBSCRIBER_HANDLE_H__

#include "apocharmm_c/DcdSubscriber.h"
#include "apocharmm_c/detail/SubscriberHandle.h"

#include "DcdSubscriber.h"

#include <memory>

struct apo_dcd_subscriber {
  std::shared_ptr<DcdSubscriber> object = nullptr;
  apo_subscriber base;
};

#endif
