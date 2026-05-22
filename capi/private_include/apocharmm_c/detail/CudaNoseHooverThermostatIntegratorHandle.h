// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_CUDA_NOSE_HOOVER_THERMOSTAT_INTEGRATOR_HANDLE_H__
#define __APOCHARMM_C_DETAIL_CUDA_NOSE_HOOVER_THERMOSTAT_INTEGRATOR_HANDLE_H__

#include "apocharmm_c/CudaNoseHooverThermostatIntegrator.h"

#include "CharmmContext.h"
#include "CudaNoseHooverThermostatIntegrator.h"

#include <memory>

// cnht --> cuda_nose_hoover_thermostat

struct apo_cnht_integrator {
  std::shared_ptr<CharmmContext> context = nullptr;
  std::shared_ptr<CudaNoseHooverThermostatIntegrator> object = nullptr;
};

#endif
