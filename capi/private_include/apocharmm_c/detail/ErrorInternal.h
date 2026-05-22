// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_ERROR_INTERNAL_H__
#define __APOCHARMM_C_ERROR_INTERNAL_H__

#include "apocharmm_c/Status.h"

#include <exception>
#include <string>

namespace apocharmm_c {

void clear_last_error(void) noexcept;

apo_status set_last_error(apo_status status, const char *message) noexcept;

apo_status set_last_error(apo_status status,
                          const std::string &message) noexcept;

template <typename Function>
apo_status guard(Function &&function, const char *function_name) noexcept {
  clear_last_error();

  try {
    return function();
  } catch (const std::exception &e) {
    return set_last_error(APO_STATUS_RUNTIME_ERROR, e.what());
  } catch (...) {
    return set_last_error(APO_STATUS_RUNTIME_ERROR,
                          std::string(function_name) +
                              ": Unknown C++ exception");
  }
}

} // namespace apocharmm_c

#endif
