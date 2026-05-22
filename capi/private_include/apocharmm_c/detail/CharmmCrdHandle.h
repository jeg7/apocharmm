// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_CHARMM_CRD_HANDLE_H__
#define __APOCHARMM_C_DETAIL_CHARMM_CRD_HANDLE_H__

#include "apocharmm_c/CharmmCrd.h"

#include "CharmmCrd.h"

#include <memory>

struct apo_charmm_crd {
  std::shared_ptr<CharmmCrd> object = nullptr;
};

#endif
