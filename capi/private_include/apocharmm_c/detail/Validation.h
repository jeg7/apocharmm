// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_DETAIL_VALIDATION_H__
#define __APOCHARMM_C_DETAIL_VALIDATION_H__

#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/ErrorInternal.h"

#include <cmath>
#include <cstddef>
#include <string>

namespace apocharmm_c {

inline apo_status invalid_argument(const char *function_name,
                                   const std::string &message) {
  return set_last_error(APO_STATUS_INVALID_ARGUMENT,
                        std::string(function_name) + ":" + message);
}

template <typename T>
apo_status prepare_output_pointer(T **out, const char *function_name,
                                  const char *argument_name) {
  if (out == nullptr) {
    return invalid_argument(function_name,
                            std::string(argument_name) + " pointer is NULL");
  }

  *out = nullptr;

  return APO_STATUS_OK;
}

inline apo_status require_c_string(const char *value, const char *function_name,
                                   const char *argument_name) {
  if ((value != nullptr) && (value[0] != '\0'))
    return APO_STATUS_OK;

  return invalid_argument(function_name,
                          std::string(argument_name) + " is NULL or empty");
}

template <typename T>
apo_status require_pointer(const T *pointer, const char *function_name,
                           const char *argument_name) {
  if (pointer != nullptr)
    return APO_STATUS_OK;

  return invalid_argument(function_name,
                          std::string(argument_name) + " is NULL");
}

// JEG260522: This utility function requires that every wrapper handle struct
// have the same member name.
//
// std::shared_ptr<ObjectType> object;
//
// Worth standardizing now.
template <typename T>
apo_status require_handle_object(const T *handle, const char *function_name,
                                 const char *handle_name) {
  apo_status status = require_pointer<T>(handle, function_name, handle_name);

  if (status != APO_STATUS_OK)
    return status;

  if (handle->object != nullptr)
    return APO_STATUS_OK;

  return invalid_argument(function_name,
                          std::string(handle_name) + " object is NULLx");
}

template <typename T>
apo_status
require_output_buffer(const T *buffer, const std::size_t provided_len,
                      const std::size_t required_len, const char *function_name,
                      const char *buffer_name) {
  if (required_len == 0)
    return APO_STATUS_OK;

  if (buffer == nullptr) {
    return invalid_argument(function_name,
                            std::string(buffer_name) + " is NULL");
  }

  if (provided_len < required_len) {
    return invalid_argument(function_name,
                            std::string(buffer_name) + " is too small");
  }

  return APO_STATUS_OK;
}

} // namespace apocharmm_c

#define APOCHARMM_C_RETURN_IF_ERROR(expression)                                \
  do {                                                                         \
    const apo_status apocharmm_c_status__ = (expression);                      \
    if (apocharmm_c_status__ != APO_STATUS_OK)                                 \
      return apocharmm_c_status__;                                             \
  } while (false);

#endif
