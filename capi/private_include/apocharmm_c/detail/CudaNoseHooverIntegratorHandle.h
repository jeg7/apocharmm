// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_CUDA_NOSE_HOOVER_INTEGRATOR_HANDLE_H__
#define __APOCHARMM_C_DETAIL_CUDA_NOSE_HOOVER_INTEGRATOR_HANDLE_H__

#include "apocharmm_c/CudaNoseHooverIntegrator.h"
#include "apocharmm_c/detail/CudaIntegratorHandle.h"

#include "CudaNoseHooverIntegrator.h"

#include <memory>

struct apo_cuda_nose_hoover_integrator {
  std::shared_ptr<CudaNoseHooverIntegrator> object = nullptr;
  apo_cuda_integrator base;
};

#endif
