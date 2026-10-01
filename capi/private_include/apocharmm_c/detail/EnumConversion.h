// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_ENUM_CONVERSION_H__
#define __APOCHARMM_C_DETAIL_ENUM_CONVERSION_H__

#include "PBC.h"
#include "apocharmm_c/Enums.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/Validation.h"

namespace apocharmm_c {

inline apo_status to_pbc(PBC *out, const apo_pbc pbc,
                         const char *function_name) {
  APOCHARMM_C_RETURN_IF_ERROR(require_pointer<PBC>(out, function_name, "out"));

  switch (pbc) {
  case APO_PBC_UNSET:
    *out = PBC::UNSET;
    return APO_STATUS_OK;
  case APO_PBC_P1:
    *out = PBC::P1;
    return APO_STATUS_OK;
  case APO_PBC_P21:
    *out = PBC::P21;
    return APO_STATUS_OK;
  default:
    return invalid_argument(function_name,
                            "invalid periodic boundary condition");
  }
}

inline apo_status from_pbc(apo_pbc *out, const PBC pbc,
                           const char *function_name) {
  APOCHARMM_C_RETURN_IF_ERROR(
      require_pointer<apo_pbc>(out, function_name, "out"));

  switch (pbc) {
  case PBC::UNSET:
    *out = APO_PBC_UNSET;
    return APO_STATUS_OK;
  case PBC::P1:
    *out = APO_PBC_P1;
    return APO_STATUS_OK;
  case PBC::P21:
    *out = APO_PBC_P21;
    return APO_STATUS_OK;
  default:
    return invalid_argument(function_name, "unknown C++ PBC value");
  }
}

inline apo_status to_crystal_type(CRYSTAL *out,
                                  const apo_crystal_type crystal_type,
                                  const char *function_name) {
  APOCHARMM_C_RETURN_IF_ERROR(
      require_pointer<CRYSTAL>(out, function_name, "out"));

  switch (crystal_type) {
  case APO_CRYSTAL_TYPE_NONE:
    *out = CRYSTAL::NONE;
    return APO_STATUS_OK;
  case APO_CRYSTAL_TYPE_CUBIC:
    *out = CRYSTAL::CUBIC;
    return APO_STATUS_OK;
  case APO_CRYSTAL_TYPE_TETRAGONAL:
    *out = CRYSTAL::TETRAGONAL;
    return APO_STATUS_OK;
  case APO_CRYSTAL_TYPE_ORTHORHOMBIC:
    *out = CRYSTAL::ORTHORHOMBIC;
    return APO_STATUS_OK;
  default:
    return invalid_argument(function_name, "invalid crystal type");
  }
}

inline apo_status from_crystal_type(apo_crystal_type *out,
                                    const CRYSTAL crystal_type,
                                    const char *function_name) {
  APOCHARMM_C_RETURN_IF_ERROR(
      require_pointer<apo_crystal_type>(out, function_name, "out"));

  switch (crystal_type) {
  case CRYSTAL::NONE:
    *out = APO_CRYSTAL_TYPE_NONE;
    return APO_STATUS_OK;
  case CRYSTAL::CUBIC:
    *out = APO_CRYSTAL_TYPE_CUBIC;
    return APO_STATUS_OK;
  case CRYSTAL::TETRAGONAL:
    *out = APO_CRYSTAL_TYPE_TETRAGONAL;
    return APO_STATUS_OK;
  case CRYSTAL::ORTHORHOMBIC:
    *out = APO_CRYSTAL_TYPE_ORTHORHOMBIC;
    return APO_STATUS_OK;
  default:
    return invalid_argument(function_name, "unknown C++ crystal type value");
  }
}

} // namespace apocharmm_c

#endif
