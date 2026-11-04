// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "ApoCharmmError.h"
#include "CudaContainer.h"
#include "DeviceVector.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/Status.h"
#include "catch.hpp"
#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <vector_types.h>

namespace apo_test {

[[nodiscard]] inline std::filesystem::path GetDataDir(void) {
  return std::filesystem::path{APOCHARMM_TEST_DATA_DIR};
}

[[nodiscard]] inline std::filesystem::path GetTopparDir(void) {
  return std::filesystem::path{APOCHARMM_TOPPAR_DIR};
}

inline void WriteTextFile(const std::filesystem::path &filePath,
                          const std::string &contents) {
  std::ofstream fout(filePath);
  REQUIRE(fout.good());

  fout << contents;
  fout.close();

  REQUIRE(fout.good());

  return;
}

[[nodiscard]] inline std::string
ReadTextFile(const std::filesystem::path &filePath) {
  std::ifstream fin(filePath);
  REQUIRE(fin.good());

  const std::string contents((std::istreambuf_iterator<char>(fin)),
                             std::istreambuf_iterator<char>());

  REQUIRE(!fin.bad());

  return contents;
}

inline void RemoveIfExists(const std::filesystem::path &filePath) {
  static_cast<void>(std::filesystem::remove(filePath));
  return;
}

inline std::vector<int> MakeRange(const int first, const int last) {
  std::vector<int> values;
  values.reserve(static_cast<std::size_t>(std::max(last - first, 0)));

  for (int value = first; value < last; value++)
    values.push_back(value);

  return values;
}

inline double Deg2Rad(const double deg) {
  return deg * std::acos(-1.0) / 180.0;
}

inline void CheckFiniteScalar(const double value) {
  CHECK(std::isfinite(value));
  return;
}

inline void CheckFiniteVector(const std::vector<double> &values) {
  for (const double value : values)
    CheckFiniteScalar(value);
  return;
}

inline void CheckFiniteNonnegativeScalar(const double value) {
  CHECK(std::isfinite(value));
  CHECK(value >= 0.0);
  return;
}

inline void CheckFiniteNonnegativeVector(const std::vector<double> &values) {
  for (const double value : values)
    CheckFiniteNonnegativeScalar(value);
  return;
}

inline void CheckFiniteTemperature(const double temperature,
                                   const double upperBound = 1.0e8) {
  CheckFiniteNonnegativeScalar(temperature);
  CHECK(temperature < upperBound);
  return;
}

template <typename T>
std::vector<T> CopyToHost(const DeviceVector<T> &deviceVector,
                          const std::size_t count) {
  REQUIRE(count <= deviceVector.size());

  std::vector<T> hostVector(count);

  if (!hostVector.empty()) {
    cudaCheck(cudaMemcpy(static_cast<void *>(hostVector.data()),
                         static_cast<const void *>(deviceVector.data()),
                         count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  return hostVector;
}

template <typename T>
std::vector<T> CopyToHost(const DeviceVector<T> &deviceVector) {
  return CopyToHost<T>(deviceVector, deviceVector.size());
}

template <typename T> std::vector<T> CopyToHost(CudaContainer<T> container) {
  container.transferToHost();
  return container.getHostArray();
}

template <typename T>
void CopyToDevice(DeviceVector<T> &deviceVector,
                  const std::vector<T> &hostVector) {
  REQUIRE(deviceVector.size() >= hostVector.size());

  if (!hostVector.empty()) {
    cudaCheck(cudaMemcpy(static_cast<void *>(deviceVector.data()),
                         static_cast<const void *>(hostVector.data()),
                         hostVector.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  }

  return;
}

template <typename T>
void CopyToDevice(CudaContainer<T> &container,
                  const std::vector<T> &hostVector) {
  REQUIRE(container.size() >= hostVector.size());

  container.getHostArray() = hostVector;
  container.transferToDevice();

  return;
}

template <typename T> T CopyScalarToHost(CudaContainer<T> container) {
  REQUIRE(container.size() == 1);

  container.transferToHost();

  return container[0];
}

template <typename T>
void CheckHostAndDeviceEqual(const CudaContainer<T> &container,
                             const std::vector<T> &expected) {
  CHECK(container.getHostArray().size() == expected.size());
  CHECK(container.getDeviceArray().size() == expected.size());
  CHECK(container.getHostArray() == expected);
  CHECK(CopyToHost(container.getDeviceArray()) == expected);
  return;
}

template <typename T>
void CheckVectorsEqual(const std::string &label, const std::vector<T> &observed,
                       const std::vector<T> &expected) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double observedValue = static_cast<double>(observed[i]);
    const double expectedValue = static_cast<double>(expected[i]);
    const double difference = std::abs(observedValue - expectedValue);

    INFO("index:     " << i);
    INFO("observed:  " << std::setprecision(17) << observedValue);
    INFO("expected:  " << std::setprecision(17) << expectedValue);
    INFO("abs diff:  " << std::setprecision(17) << difference);

    CHECK(observedValue == expectedValue);
  }

  return;
}

template <>
inline void CheckVectorsEqual(const std::string &label,
                              const std::vector<int2> &observed,
                              const std::vector<int2> &expected) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double2 observedValue = make_double2(
        static_cast<double>(observed[i].x), static_cast<double>(observed[i].y));
    const double2 expectedValue = make_double2(
        static_cast<double>(expected[i].x), static_cast<double>(expected[i].y));
    const double2 difference =
        make_double2(std::abs(observedValue.x - expectedValue.x),
                     std::abs(observedValue.y - expectedValue.y));

    INFO("index:     " << i);
    INFO("observed:  {" << std::setprecision(17) << observedValue.x << ", "
                        << std::setprecision(17) << observedValue.y << "}");
    INFO("expected:  {" << std::setprecision(17) << expectedValue.x << ", "
                        << std::setprecision(17) << expectedValue.y << "}");
    INFO("abs diff:  {" << std::setprecision(17) << difference.x << ", "
                        << std::setprecision(17) << difference.y << "}");

    CHECK(observedValue.x == expectedValue.x);
    CHECK(observedValue.y == expectedValue.y);
  }

  return;
}

template <>
inline void CheckVectorsEqual(const std::string &label,
                              const std::vector<int4> &observed,
                              const std::vector<int4> &expected) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double4 observedValue = make_double4(
        static_cast<double>(observed[i].x), static_cast<double>(observed[i].y),
        static_cast<double>(observed[i].z), static_cast<double>(observed[i].w));
    const double4 expectedValue = make_double4(
        static_cast<double>(expected[i].x), static_cast<double>(expected[i].y),
        static_cast<double>(expected[i].z), static_cast<double>(expected[i].w));
    const double4 difference =
        make_double4(std::abs(observedValue.x - expectedValue.x),
                     std::abs(observedValue.y - expectedValue.y),
                     std::abs(observedValue.z - expectedValue.z),
                     std::abs(observedValue.w - expectedValue.w));

    INFO("index:     " << i);
    INFO("observed:  {" << std::setprecision(17) << observedValue.x << ", "
                        << std::setprecision(17) << observedValue.y << ", "
                        << std::setprecision(17) << observedValue.z << ", "
                        << std::setprecision(17) << observedValue.w << "}");
    INFO("expected:  {" << std::setprecision(17) << expectedValue.x << ", "
                        << std::setprecision(17) << expectedValue.y << ", "
                        << std::setprecision(17) << expectedValue.z << ", "
                        << std::setprecision(17) << expectedValue.w << "}");
    INFO("abs diff:  {" << std::setprecision(17) << difference.x << ", "
                        << std::setprecision(17) << difference.y << ", "
                        << std::setprecision(17) << difference.z << ", "
                        << std::setprecision(17) << difference.w << "}");

    CHECK(observedValue.x == expectedValue.x);
    CHECK(observedValue.y == expectedValue.y);
    CHECK(observedValue.z == expectedValue.z);
    CHECK(observedValue.w == expectedValue.w);
  }

  return;
}

template <>
inline void CheckVectorsEqual(const std::string &label,
                              const std::vector<double4> &observed,
                              const std::vector<double4> &expected) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double4 observedValue = observed[i];
    const double4 expectedValue = expected[i];
    const double4 difference =
        make_double4(std::abs(observedValue.x - expectedValue.x),
                     std::abs(observedValue.y - expectedValue.y),
                     std::abs(observedValue.z - expectedValue.z),
                     std::abs(observedValue.w - expectedValue.w));

    INFO("index:     " << i);
    INFO("observed:  {" << std::setprecision(17) << observedValue.x << ", "
                        << std::setprecision(17) << observedValue.y << ", "
                        << std::setprecision(17) << observedValue.z << ", "
                        << std::setprecision(17) << observedValue.w << "}");
    INFO("expected:  {" << std::setprecision(17) << expectedValue.x << ", "
                        << std::setprecision(17) << expectedValue.y << ", "
                        << std::setprecision(17) << expectedValue.z << ", "
                        << std::setprecision(17) << expectedValue.w << "}");
    INFO("abs diff:  {" << std::setprecision(17) << difference.x << ", "
                        << std::setprecision(17) << difference.y << ", "
                        << std::setprecision(17) << difference.z << ", "
                        << std::setprecision(17) << difference.w << "}");

    CHECK(observedValue.x == expectedValue.x);
    CHECK(observedValue.y == expectedValue.y);
    CHECK(observedValue.z == expectedValue.z);
    CHECK(observedValue.w == expectedValue.w);
  }

  return;
}

template <typename T>
void CheckVectorsEqual(const std::vector<T> &observed,
                       const std::vector<T> &expected) {
  CheckVectorsEqual<T>("unnamed vector", observed, expected);
  return;
}

template <typename T>
void CheckVectorsClose(const std::string &label, const std::vector<T> &observed,
                       const std::vector<T> &expected, const double tolerance) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double observedValue = static_cast<double>(observed[i]);
    const double expectedValue = static_cast<double>(expected[i]);
    const double difference = std::abs(observedValue - expectedValue);

    INFO("index:     " << i);
    INFO("observed:  " << std::setprecision(17) << observedValue);
    INFO("expected:  " << std::setprecision(17) << expectedValue);
    INFO("abs diff:  " << std::setprecision(17) << difference);
    INFO("tolerance: " << std::setprecision(17) << tolerance);

    CHECK(observedValue == Approx(expectedValue).margin(tolerance));
  }

  return;
}

template <>
inline void
CheckVectorsClose(const std::string &label, const std::vector<float3> &observed,
                  const std::vector<float3> &expected, const double tolerance) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double3 observedValue = make_double3(
        static_cast<double>(observed[i].x), static_cast<double>(observed[i].y),
        static_cast<double>(observed[i].z));
    const double3 expectedValue = make_double3(
        static_cast<double>(expected[i].x), static_cast<double>(expected[i].y),
        static_cast<double>(expected[i].z));
    const double3 difference =
        make_double3(std::abs(observedValue.x - expectedValue.x),
                     std::abs(observedValue.y - expectedValue.y),
                     std::abs(observedValue.z - expectedValue.z));

    INFO("index:     " << i);
    INFO("observed:  {" << std::setprecision(17) << observedValue.x << ", "
                        << std::setprecision(17) << observedValue.y << ", "
                        << std::setprecision(17) << observedValue.z << "}");
    INFO("expected:  {" << std::setprecision(17) << expectedValue.x << ", "
                        << std::setprecision(17) << expectedValue.y << ", "
                        << std::setprecision(17) << expectedValue.z << "}");
    INFO("abs diff:  {" << std::setprecision(17) << difference.x << ", "
                        << std::setprecision(17) << difference.y << ", "
                        << std::setprecision(17) << difference.z << "}");
    INFO("tolerance: " << std::setprecision(17) << tolerance);

    CHECK(observedValue.x == Approx(expectedValue.x).margin(tolerance));
    CHECK(observedValue.y == Approx(expectedValue.y).margin(tolerance));
    CHECK(observedValue.z == Approx(expectedValue.z).margin(tolerance));
  }

  return;
}

template <>
inline void CheckVectorsClose(const std::string &label,
                              const std::vector<double3> &observed,
                              const std::vector<double3> &expected,
                              const double tolerance) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double3 observedValue = observed[i];
    const double3 expectedValue = expected[i];
    const double3 difference =
        make_double3(std::abs(observedValue.x - expectedValue.x),
                     std::abs(observedValue.y - expectedValue.y),
                     std::abs(observedValue.z - expectedValue.z));

    INFO("index:     " << i);
    INFO("observed:  {" << std::setprecision(17) << observedValue.x << ", "
                        << std::setprecision(17) << observedValue.y << ", "
                        << std::setprecision(17) << observedValue.z << "}");
    INFO("expected:  {" << std::setprecision(17) << expectedValue.x << ", "
                        << std::setprecision(17) << expectedValue.y << ", "
                        << std::setprecision(17) << expectedValue.z << "}");
    INFO("abs diff:  {" << std::setprecision(17) << difference.x << ", "
                        << std::setprecision(17) << difference.y << ", "
                        << std::setprecision(17) << difference.z << "}");
    INFO("tolerance: " << std::setprecision(17) << tolerance);

    CHECK(observedValue.x == Approx(expectedValue.x).margin(tolerance));
    CHECK(observedValue.y == Approx(expectedValue.y).margin(tolerance));
    CHECK(observedValue.z == Approx(expectedValue.z).margin(tolerance));
  }

  return;
}

template <>
inline void
CheckVectorsClose(const std::string &label, const std::vector<float4> &observed,
                  const std::vector<float4> &expected, const double tolerance) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double4 observedValue = make_double4(
        static_cast<double>(observed[i].x), static_cast<double>(observed[i].y),
        static_cast<double>(observed[i].z), static_cast<double>(observed[i].w));
    const double4 expectedValue = make_double4(
        static_cast<double>(expected[i].x), static_cast<double>(expected[i].y),
        static_cast<double>(expected[i].z), static_cast<double>(expected[i].w));
    const double4 difference =
        make_double4(std::abs(observedValue.x - expectedValue.x),
                     std::abs(observedValue.y - expectedValue.y),
                     std::abs(observedValue.z - expectedValue.z),
                     std::abs(observedValue.w - expectedValue.w));

    INFO("index:     " << i);
    INFO("observed:  {" << std::setprecision(17) << observedValue.x << ", "
                        << std::setprecision(17) << observedValue.y << ", "
                        << std::setprecision(17) << observedValue.z << ", "
                        << std::setprecision(17) << observedValue.w << "}");
    INFO("expected:  {" << std::setprecision(17) << expectedValue.x << ", "
                        << std::setprecision(17) << expectedValue.y << ", "
                        << std::setprecision(17) << expectedValue.z << ", "
                        << std::setprecision(17) << expectedValue.w << "}");
    INFO("abs diff:  {" << std::setprecision(17) << difference.x << ", "
                        << std::setprecision(17) << difference.y << ", "
                        << std::setprecision(17) << difference.z << ", "
                        << std::setprecision(17) << difference.w << "}");
    INFO("tolerance: " << std::setprecision(17) << tolerance);

    CHECK(observedValue.x == Approx(expectedValue.x).margin(tolerance));
    CHECK(observedValue.y == Approx(expectedValue.y).margin(tolerance));
    CHECK(observedValue.z == Approx(expectedValue.z).margin(tolerance));
    CHECK(observedValue.w == Approx(expectedValue.w).margin(tolerance));
  }

  return;
}

template <>
inline void CheckVectorsClose(const std::string &label,
                              const std::vector<double4> &observed,
                              const std::vector<double4> &expected,
                              const double tolerance) {
  INFO("vector label: " << label);
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    const double4 observedValue = observed[i];
    const double4 expectedValue = expected[i];
    const double4 difference =
        make_double4(std::abs(observedValue.x - expectedValue.x),
                     std::abs(observedValue.y - expectedValue.y),
                     std::abs(observedValue.z - expectedValue.z),
                     std::abs(observedValue.w - expectedValue.w));

    INFO("index:     " << i);
    INFO("observed:  {" << std::setprecision(17) << observedValue.x << ", "
                        << std::setprecision(17) << observedValue.y << ", "
                        << std::setprecision(17) << observedValue.z << ", "
                        << std::setprecision(17) << observedValue.w << "}");
    INFO("expected:  {" << std::setprecision(17) << expectedValue.x << ", "
                        << std::setprecision(17) << expectedValue.y << ", "
                        << std::setprecision(17) << expectedValue.z << ", "
                        << std::setprecision(17) << expectedValue.w << "}");
    INFO("abs diff:  {" << std::setprecision(17) << difference.x << ", "
                        << std::setprecision(17) << difference.y << ", "
                        << std::setprecision(17) << difference.z << ", "
                        << std::setprecision(17) << difference.w << "}");
    INFO("tolerance: " << std::setprecision(17) << tolerance);

    CHECK(observedValue.x == Approx(expectedValue.x).margin(tolerance));
    CHECK(observedValue.y == Approx(expectedValue.y).margin(tolerance));
    CHECK(observedValue.z == Approx(expectedValue.z).margin(tolerance));
    CHECK(observedValue.w == Approx(expectedValue.w).margin(tolerance));
  }

  return;
}

template <typename T>
void CheckVectorsClose(const std::vector<T> &observed,
                       const std::vector<T> &expected, const double tolerance) {
  CheckVectorsClose<T>("unnamed vector", observed, expected, tolerance);
  return;
}

template <typename T>
void CheckVectorsClose2D(const std::vector<std::vector<T>> &observed,
                         const std::vector<std::vector<T>> &expected,
                         const double tolerance) {
  REQUIRE(observed.size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); i++) {
    CAPTURE(i);
    REQUIRE(observed[i].size() == expected[i].size());
    for (std::size_t j = 0; j < expected[i].size(); j++) {
      CHECK(static_cast<double>(observed[i][j]) ==
            Approx(static_cast<double>(expected[i][j])).margin(tolerance));
    }
  }

  return;
}

template <typename Function>
void CheckApoCharmmError(Function action, const ApoCharmmErrorCode expectedCode,
                         const std::string_view expectedMessage) {
  const std::string_view expectedCodeName =
      GetApoCharmmErrorCodeName(expectedCode);

  INFO("Expected ApoCharmmError code: " << expectedCodeName);
  INFO("Expected ApoCharmmError message: " << expectedMessage);

  try {
    action();
  } catch (const ApoCharmmError &error) {
    INFO("Observed ApoCharmmError code: "
         << GetApoCharmmErrorCodeName(error.getCode()));
    INFO("Observed ApoCharmmError message: " << error.getMessage());
    INFO("Observed ApoCharmmError source: " << error.getSourceFile() << ':'
                                            << error.getSourceLine());
    INFO("Observed ApoCharmmError function: " << error.getSourceFunction());

    CHECK(error.getCode() == expectedCode);
    CHECK(error.getMessage() == expectedMessage);
    CHECK(error.getMessage().find("ERROR:") == std::string_view::npos);
    CHECK((error.getMessage().empty() || error.getMessage().back() != '\n'));

    return;
  } catch (const std::exception &error) {
    FAIL(
        "Expected ApoCharmmError, but caught another std::exception\n  what(): "
        << error.what());
    return;
  }

  FAIL_CHECK("Expected ApoCharmmError, but no exception was thrown");

  return;
}

void CheckStatusAndDiagnostic(const apo_status status,
                              const apo_status expectedStatus,
                              const std::string_view expectedDiagnostic) {
  const std::string diagnostic(apo_last_error());

  INFO("Expected apo_status: " << static_cast<int>(expectedStatus));
  INFO("Observed apo_status: " << static_cast<int>(status));
  INFO("Expected C API diagnostic:\n" << expectedDiagnostic);
  INFO("Observed C API diagnostic:\n" << diagnostic);

  CHECK(status == expectedStatus);
  CHECK(diagnostic == expectedDiagnostic);

  return;
}

void CheckNativeError(const apo_status status, const apo_status expectedStatus,
                      const std::string_view expectedCodeName,
                      const std::string_view functionName,
                      const std::string_view expectedMessage,
                      const std::string_view expectedSourceFile,
                      const std::string_view expectedSourceFunction) {
  const std::string diagnostic(apo_last_error());

  const std::string expectedPrefix = std::string(functionName) + ": ";
  const std::string expectedError = "apoCHARMM error [" +
                                    std::string(expectedCodeName) +
                                    "]: " + std::string(expectedMessage);
  const std::string expectedSource =
      "  source: " + std::string(expectedSourceFile) + ':';
  const std::string expectedFunction =
      "  function: " + std::string(expectedSourceFunction);

  INFO("Expected apo_status: " << static_cast<int>(expectedStatus));
  INFO("Observed apo_status: " << static_cast<int>(status));
  INFO("Expected diagnostic prefix:\n" << expectedPrefix);
  INFO("Expected native error text:\n" << expectedError);
  INFO("Expected native source text:\n" << expectedSource);
  INFO("Expected native function text:\n" << expectedFunction);
  INFO("Observed C API diagnostic:\n" << diagnostic);

  CHECK(status == expectedStatus);
  REQUIRE(diagnostic.empty() == false);

  CHECK(diagnostic.compare(0, expectedPrefix.size(), expectedPrefix) == 0);
  CHECK(diagnostic.find(expectedError) != std::string::npos);
  CHECK(diagnostic.find(expectedSource) != std::string::npos);
  CHECK(diagnostic.find(expectedFunction) != std::string::npos);
  CHECK(diagnostic.find("ERROR:") == std::string::npos);
  CHECK(diagnostic.back() != '\n');

  return;
}

} // namespace apo_test
