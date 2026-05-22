// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_STATUS_H__
#define __APOCHARMM_C_STATUS_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef enum apo_status {
  APO_STATUS_OK = 0,
  APO_STATUS_INVALID_ARGUMENT = 1,
  APO_STATUS_RUNTIME_ERROR = 2,
  APO_STATUS_CUDA_ERROR = 3,
  APO_STATUS_NOT_INITIALIZED = 4,
  APO_STATUS_NOT_IMPLEMENTED = 5,
} apo_status;

#ifdef __cplusplus
}
#endif

#endif
