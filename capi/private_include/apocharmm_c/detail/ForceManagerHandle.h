// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_FORCE_MANAGER_H__
#define __APOCHARMM_C_DETAIL_FORCE_MANAGER_H__

#include "apocharmm_c/ForceManager.h"

#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "ForceManager.h"

#include <memory>

struct apo_force_manager {
  std::shared_ptr<CharmmPSF> psf = nullptr;
  std::shared_ptr<CharmmParameters> parameters = nullptr;
  std::shared_ptr<ForceManager> object = nullptr;
};

#endif
