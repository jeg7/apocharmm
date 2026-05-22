// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_CHARMM_PARAMETERS_H__
#define __APOCHARMM_C_DETAIL_CHARMM_PARAMETERS_H__

#include "apocharmm_c/CharmmParameters.h"

#include "CharmmParameters.h"

#include <memory>

struct apo_charmm_parameters {
  std::shared_ptr<CharmmParameters> object = nullptr;
};

#endif
