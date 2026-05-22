// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_CHARMM_CONTEXT_HANDLE_H__
#define __APOCHARMM_C_DETAIL_CHARMM_CONTEXT_HANDLE_H__

#include "apocharmm_c/CharmmContext.h"

#include "CharmmContext.h"
#include "ForceManager.h"

#include <memory>

struct apo_charmm_context {
  std::shared_ptr<ForceManager> force_manager = nullptr;
  std::shared_ptr<CharmmContext> object = nullptr;
};

#endif
