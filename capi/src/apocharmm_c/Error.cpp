// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/Error.h"
#include "apocharmm_c/detail/ErrorInternal.h"

#include <string>

// Anonymous namespace to prevent usage outside of this file
namespace {

thread_local std::string g_last_error;

}

extern "C" const char *apo_last_error(void) { return g_last_error.c_str(); }

void apocharmm_c::clear_last_error(void) noexcept {
  g_last_error.clear();
  return;
}

apo_status apocharmm_c::set_last_error(apo_status status,
                                       const char *message) noexcept {
  try {
    g_last_error = (message != nullptr) ? message : "";
  } catch (...) {
    // Last-error storage is best-effort. Do not throw from the C ABI error
    // path.
  }
  return status;
}

apo_status apocharmm_c::set_last_error(apo_status status,
                                       const std::string &message) noexcept {
  try {
    g_last_error = message;
  } catch (...) {
    // Last-error storage is best-effort. Do not throw from the C ABI error
    // path.
  }
  return status;
}
