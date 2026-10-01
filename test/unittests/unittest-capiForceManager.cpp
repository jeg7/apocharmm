// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/CharmmParametersHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/EnumConversion.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "catch.hpp"

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr double TOLERANCE = 1.0e-6;

struct ForceManagerDeleter {
  void operator()(apo_force_manager *forceManager) const noexcept {
    apo_force_manager_destroy(forceManager);
    return;
  }
};

using ForceManagerHandle =
    std::unique_ptr<apo_force_manager, ForceManagerDeleter>;

struct ForceManagerInputs {
  std::shared_ptr<CharmmPSF> psf;
  std::shared_ptr<CharmmParameters> parameters;
  apo_charmm_psf psfHandle;
  apo_charmm_parameters parametersHandle;

  ForceManagerInputs()
      : psf(std::make_shared<CharmmPSF>(apo_test::GetDataDir() /
                                        "nacl_pair.psf")),
        parameters(std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                      "toppar_water_ions.str")),
        psfHandle(), parametersHandle() {
    psfHandle.object = psf;
    parametersHandle.object = parameters;
  }
};

ForceManagerHandle MakeForceManager(ForceManagerInputs &inputs) {
  apo_force_manager *rawForceManager = nullptr;
  REQUIRE(apo_force_manager_create(&rawForceManager, &inputs.psfHandle,
                                   &inputs.parametersHandle) == APO_STATUS_OK);
  REQUIRE(rawForceManager != nullptr);
  return ForceManagerHandle(rawForceManager);
}

ForceManagerHandle MakeDefaultForceManager(void) {
  ForceManagerHandle handle(new apo_force_manager());
  handle->object = std::make_shared<ForceManager>();
  return handle;
}

void CheckStatusAndDiagnostic(const apo_status status,
                              const apo_status expectedStatus,
                              const std::string_view expectedDiagnostic) {
  CHECK(status == expectedStatus);
  CHECK(std::string(apo_last_error()) == expectedDiagnostic);
  return;
}

void CheckNativeError(const apo_status status, const apo_status expectedStatus,
                      const std::string_view functionName,
                      const std::string_view codeName,
                      const std::string_view expectedMessage,
                      const std::string_view sourceFunction) {
  CHECK(status == expectedStatus);

  const std::string diagnostic(apo_last_error());
  REQUIRE(diagnostic.empty() == false);

  const std::string expectedPrefix = std::string(functionName) + ": ";
  const std::string expectedError = "apoCHARMM error [" +
                                    std::string(codeName) +
                                    "]: " + std::string(expectedMessage);

  CHECK(diagnostic.compare(0, expectedPrefix.size(), expectedPrefix) == 0);
  CHECK(diagnostic.find(expectedError) != std::string::npos);
  CHECK(diagnostic.find("  source: src/ForceManager.cu:") != std::string::npos);
  CHECK(diagnostic.find("  function: " + std::string(sourceFunction)) !=
        std::string::npos);
  CHECK(diagnostic.find("ERROR:") == std::string::npos);
  CHECK(diagnostic.back() != '\n');

  return;
}

template <typename Function>
void CheckForceManagerHandleValidation(const std::string_view functionName,
                                       Function function) {
  CheckStatusAndDiagnostic(function(nullptr), APO_STATUS_INVALID_ARGUMENT,
                           std::string(functionName) +
                               ": ForceManager is NULL");

  apo_force_manager emptyForceManager;
  CheckStatusAndDiagnostic(
      function(&emptyForceManager), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": ForceManager object is NULL");

  return;
}

} // namespace

TEST_CASE("CapiForceManagerSettersAndGetters") {
  ForceManagerInputs inputs;
  ForceManagerHandle forceManager = MakeForceManager(inputs);

  int numAtoms = 0;
  CHECK(apo_force_manager_get_num_atoms(&numAtoms, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(numAtoms == 2);

  bool flag = true;
  CHECK(apo_force_manager_is_initialized(&flag, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(flag == false);

  double boxDimensions[3] = {0.0, 0.0, 0.0};
  CHECK(apo_force_manager_get_box_dimensions(
            boxDimensions, 3, forceManager.get()) == APO_STATUS_OK);
  CHECK(boxDimensions[0] == Approx(-9999.9999).margin(TOLERANCE));
  CHECK(boxDimensions[1] == Approx(-9999.9999).margin(TOLERANCE));
  CHECK(boxDimensions[2] == Approx(-9999.9999).margin(TOLERANCE));

  double value = 0.0;
  CHECK(apo_force_manager_get_kappa(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(0.34).margin(TOLERANCE));
  CHECK(apo_force_manager_get_cutoff(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(14.0).margin(TOLERANCE));
  CHECK(apo_force_manager_get_ctonnb(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(12.0).margin(TOLERANCE));
  CHECK(apo_force_manager_get_ctofnb(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(10.0).margin(TOLERANCE));

  int grid[3] = {0, 0, 0};
  CHECK(apo_force_manager_get_fft_grid(grid, 3, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(grid[0] == -1);
  CHECK(grid[1] == -1);
  CHECK(grid[2] == -1);

  int order = 0;
  CHECK(apo_force_manager_get_pme_spline_order(&order, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(order == 4);

  apo_pbc pbc = APO_PBC_UNSET;
  CHECK(apo_force_manager_get_periodic_boundary_condition(
            &pbc, forceManager.get()) == APO_STATUS_OK);
  CHECK(pbc == APO_PBC_P1);

  int vdwType = 0;
  CHECK(apo_force_manager_get_vdw_type(&vdwType, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(vdwType == 3);

  flag = true;
  CHECK(apo_force_manager_is_composite(&flag, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(flag == false);

  const double newBoxDimensions[3] = {20.0, 21.0, 22.0};
  const int newGrid[3] = {32, 34, 36};

  CHECK(apo_force_manager_set_box_dimensions(
            forceManager.get(), newBoxDimensions, 3) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_kappa(forceManager.get(), 0.45) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_cutoff(forceManager.get(), 9.0) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_ctonnb(forceManager.get(), 8.5) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_ctofnb(forceManager.get(), 7.5) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_fft_grid(forceManager.get(), newGrid, 3) ==
        APO_STATUS_OK);
  CHECK(apo_force_manager_set_pme_spline_order(forceManager.get(), 6) ==
        APO_STATUS_OK);
  CHECK(apo_force_manager_set_periodic_boundary_condition(
            forceManager.get(), APO_PBC_P21) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_vdw_type(forceManager.get(), 6) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_print_energy_decomposition(
            forceManager.get(), true) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_print_energy_decomposition(
            forceManager.get(), false) == APO_STATUS_OK);

  CHECK(apo_force_manager_get_box_dimensions(
            boxDimensions, 3, forceManager.get()) == APO_STATUS_OK);
  CHECK(boxDimensions[0] == Approx(20.0).margin(TOLERANCE));
  CHECK(boxDimensions[1] == Approx(21.0).margin(TOLERANCE));
  CHECK(boxDimensions[2] == Approx(22.0).margin(TOLERANCE));

  CHECK(apo_force_manager_get_kappa(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(0.45).margin(TOLERANCE));
  CHECK(apo_force_manager_get_cutoff(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(9.0).margin(TOLERANCE));
  CHECK(apo_force_manager_get_ctonnb(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(8.5).margin(TOLERANCE));
  CHECK(apo_force_manager_get_ctofnb(&value, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(value == Approx(7.5).margin(TOLERANCE));

  CHECK(apo_force_manager_get_fft_grid(grid, 3, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(grid[0] == 32);
  CHECK(grid[1] == 34);
  CHECK(grid[2] == 36);

  CHECK(apo_force_manager_get_pme_spline_order(&order, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(order == 6);

  CHECK(apo_force_manager_get_periodic_boundary_condition(
            &pbc, forceManager.get()) == APO_STATUS_OK);
  CHECK(pbc == APO_PBC_P21);

  CHECK(apo_force_manager_get_vdw_type(&vdwType, forceManager.get()) ==
        APO_STATUS_OK);
  CHECK(vdwType == 6);

  CHECK(std::string(apo_last_error()).empty() == true);
}

TEST_CASE("CapiForceManagerValidatesConstructionHandles") {
  ForceManagerInputs inputs;

  CheckStatusAndDiagnostic(apo_force_manager_create(nullptr, &inputs.psfHandle,
                                                    &inputs.parametersHandle),
                           APO_STATUS_INVALID_ARGUMENT,
                           "apo_force_manager_create: out pointer is NULL");

  apo_force_manager sentinel;
  apo_force_manager *output = &sentinel;

  CheckStatusAndDiagnostic(
      apo_force_manager_create(&output, nullptr, &inputs.parametersHandle),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_create: CharmmPsf handle is NULL");
  CHECK(output == nullptr);

  apo_charmm_psf emptyPsfHandle;
  output = &sentinel;
  CheckStatusAndDiagnostic(
      apo_force_manager_create(&output, &emptyPsfHandle,
                               &inputs.parametersHandle),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_create: CharmmPsf handle object is NULL");
  CHECK(output == nullptr);

  output = &sentinel;
  CheckStatusAndDiagnostic(
      apo_force_manager_create(&output, &inputs.psfHandle, nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_create: CharmmParameters handle is NULL");
  CHECK(output == nullptr);

  apo_charmm_parameters emptyParametersHandle;
  output = &sentinel;
  CheckStatusAndDiagnostic(
      apo_force_manager_create(&output, &inputs.psfHandle,
                               &emptyParametersHandle),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_create: CharmmParameters handle object is NULL");
  CHECK(output == nullptr);
}

TEST_CASE("CapiForceManagerValidatesEveryFunctionHandle") {
  const double boxDimensions[3] = {20.0, 20.0, 20.0};
  const int inputGrid[3] = {32, 32, 32};
  int numAtoms = 0;
  bool flag = false;
  double outputBox[3] = {0.0, 0.0, 0.0};
  double scalar = 0.0;
  int outputGrid[3] = {0, 0, 0};
  int integer = 0;
  apo_pbc pbc = APO_PBC_UNSET;

  CheckForceManagerHandleValidation(
      "apo_force_manager_set_box_dimensions",
      [&boxDimensions](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_box_dimensions(forceManager, boxDimensions,
                                                    3);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_kappa",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_kappa(forceManager, 0.34);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_cutoff",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_cutoff(forceManager, 14.0);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_ctonnb",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_ctonnb(forceManager, 12.0);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_ctofnb",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_ctofnb(forceManager, 10.0);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_fft_grid",
      [&inputGrid](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_fft_grid(forceManager, inputGrid, 3);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_pme_spline_order",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_pme_spline_order(forceManager, 4);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_periodic_boundary_condition",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_periodic_boundary_condition(forceManager,
                                                                 APO_PBC_P1);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_vdw_type",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_vdw_type(forceManager, 3);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_set_print_energy_decomposition",
      [](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_set_print_energy_decomposition(forceManager,
                                                                false);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_num_atoms",
      [&numAtoms](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_num_atoms(&numAtoms, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_is_initialized",
      [&flag](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_is_initialized(&flag, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_box_dimensions",
      [&outputBox](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_box_dimensions(outputBox, 3, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_kappa",
      [&scalar](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_kappa(&scalar, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_cutoff",
      [&scalar](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_cutoff(&scalar, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_ctonnb",
      [&scalar](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_ctonnb(&scalar, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_ctofnb",
      [&scalar](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_ctofnb(&scalar, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_fft_grid",
      [&outputGrid](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_fft_grid(outputGrid, 3, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_pme_spline_order",
      [&integer](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_pme_spline_order(&integer, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_periodic_boundary_condition",
      [&pbc](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_periodic_boundary_condition(&pbc,
                                                                 forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_get_vdw_type",
      [&integer](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_get_vdw_type(&integer, forceManager);
      });
  CheckForceManagerHandleValidation(
      "apo_force_manager_is_composite",
      [&flag](apo_force_manager *forceManager) -> apo_status {
        return apo_force_manager_is_composite(&flag, forceManager);
      });
}

TEST_CASE("CapiForceManagerValidatesPointersBuffersAndEnums") {
  ForceManagerInputs inputs;
  ForceManagerHandle forceManager = MakeForceManager(inputs);

  const double boxDimensions[3] = {20.0, 20.0, 20.0};
  CheckStatusAndDiagnostic(
      apo_force_manager_set_box_dimensions(forceManager.get(), nullptr, 3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_set_box_dimensions: box_dimensions is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_set_box_dimensions(forceManager.get(), boxDimensions,
                                           2),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_set_box_dimensions: box_dimensions must contain "
      "exactly 3 elements");

  const int grid[3] = {32, 32, 32};
  CheckStatusAndDiagnostic(
      apo_force_manager_set_fft_grid(forceManager.get(), nullptr, 3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_set_fft_grid: grid is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_set_fft_grid(forceManager.get(), grid, 2),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_set_fft_grid: grid must contain exactly 3 elements");

  CheckStatusAndDiagnostic(
      apo_force_manager_set_periodic_boundary_condition(
          forceManager.get(), static_cast<apo_pbc>(99)),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_set_periodic_boundary_condition: invalid periodic "
      "boundary condition");

  CheckStatusAndDiagnostic(
      apo_force_manager_get_num_atoms(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_num_atoms: num_atoms is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_is_initialized(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_is_initialized: flag is NULL");

  double outputBox[3] = {0.0, 0.0, 0.0};
  CheckStatusAndDiagnostic(
      apo_force_manager_get_box_dimensions(nullptr, 3, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_box_dimensions: Box dimension buffer is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_get_box_dimensions(outputBox, 2, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_box_dimensions: Box dimension buffer is too "
      "small");

  CheckStatusAndDiagnostic(
      apo_force_manager_get_kappa(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_kappa: kappa is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_get_cutoff(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_cutoff: cutoff is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_get_ctonnb(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_ctonnb: ctonnb is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_get_ctofnb(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_ctofnb: ctofnb is NULL");

  int outputGrid[3] = {0, 0, 0};
  CheckStatusAndDiagnostic(
      apo_force_manager_get_fft_grid(nullptr, 3, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_fft_grid: FFT grid buffer is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_get_fft_grid(outputGrid, 2, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_fft_grid: FFT grid buffer is too small");

  CheckStatusAndDiagnostic(
      apo_force_manager_get_pme_spline_order(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_pme_spline_order: order is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_get_periodic_boundary_condition(nullptr,
                                                        forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_periodic_boundary_condition: pbc is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_get_vdw_type(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_get_vdw_type: vdw_type is NULL");
  CheckStatusAndDiagnostic(
      apo_force_manager_is_composite(nullptr, forceManager.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_is_composite: flag is NULL");
}

TEST_CASE("CapiForceManagerMapsNativeValidationErrors") {
  ForceManagerInputs inputs;
  ForceManagerHandle forceManager = MakeForceManager(inputs);
  apo_status status = APO_STATUS_OK;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double nonfiniteBox[3] = {20.0, nan, 20.0};
  CHECK_NOTHROW((status = apo_force_manager_set_box_dimensions(
                     forceManager.get(), nonfiniteBox, 3)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_box_dimensions", "InvalidArgument",
                   "Box dimension at index 1 must be finite; observed " +
                       std::to_string(nan),
                   "checkBoxDimensions");

  const double zeroBox[3] = {20.0, 0.0, 20.0};
  CHECK_NOTHROW((status = apo_force_manager_set_box_dimensions(
                     forceManager.get(), zeroBox, 3)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_box_dimensions", "InvalidArgument",
                   "Box dimension at index 1 must be positive; observed "
                   "0.000000",
                   "checkBoxDimensions");

  const float floatNan = std::numeric_limits<float>::quiet_NaN();

  CHECK_NOTHROW((status = apo_force_manager_set_kappa(
                     forceManager.get(), static_cast<double>(floatNan))));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_kappa", "InvalidArgument",
                   "Kappa must be finite; observed " + std::to_string(floatNan),
                   "setKappa");

  CHECK_NOTHROW(
      (status = apo_force_manager_set_kappa(forceManager.get(), -1.0)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_kappa", "InvalidArgument",
                   "Kappa must be non-negative; observed -1.000000",
                   "setKappa");

  CHECK_NOTHROW((status = apo_force_manager_set_cutoff(
                     forceManager.get(), static_cast<double>(floatNan))));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_cutoff", "InvalidArgument",
                   "Cutoff must be finite; observed " +
                       std::to_string(floatNan),
                   "setCutoff");

  CHECK_NOTHROW(
      (status = apo_force_manager_set_cutoff(forceManager.get(), 0.0)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_cutoff", "InvalidArgument",
                   "Cutoff must be positive; observed 0.000000", "setCutoff");

  CHECK_NOTHROW((status = apo_force_manager_set_ctonnb(
                     forceManager.get(), static_cast<double>(floatNan))));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_ctonnb", "InvalidArgument",
                   "Ctonnb must be finite; observed " +
                       std::to_string(floatNan),
                   "setCtonnb");

  CHECK_NOTHROW(
      (status = apo_force_manager_set_ctonnb(forceManager.get(), 0.0)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_ctonnb", "InvalidArgument",
                   "Ctonnb must be positive; observed 0.000000", "setCtonnb");

  CHECK_NOTHROW((status = apo_force_manager_set_ctofnb(
                     forceManager.get(), static_cast<double>(floatNan))));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_ctofnb", "InvalidArgument",
                   "Ctofnb must be finite; observed " +
                       std::to_string(floatNan),
                   "setCtofnb");

  CHECK_NOTHROW(
      (status = apo_force_manager_set_ctofnb(forceManager.get(), 0.0)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_ctofnb", "InvalidArgument",
                   "Ctofnb must be positive; observed 0.000000", "setCtofnb");

  CHECK_NOTHROW(
      (status = apo_force_manager_set_pme_spline_order(forceManager.get(), 0)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_pme_spline_order", "InvalidArgument",
                   "PME spline order must be positive; observed 0",
                   "setPmeSplineOrder");

  CHECK_NOTHROW((status = apo_force_manager_set_periodic_boundary_condition(
                     forceManager.get(), APO_PBC_UNSET)));
  CheckNativeError(
      status, APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_set_periodic_boundary_condition", "InvalidArgument",
      "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 0",
      "checkPBCCompatibility");

  CHECK_NOTHROW(
      (status = apo_force_manager_set_vdw_type(forceManager.get(), 0)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_vdw_type", "InvalidArgument",
                   "Van der Waals type must be in [1, 6]; observed 0",
                   "setVdwType");

  CHECK_NOTHROW(
      (status = apo_force_manager_set_vdw_type(forceManager.get(), 7)));
  CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                   "apo_force_manager_set_vdw_type", "InvalidArgument",
                   "Van der Waals type must be in [1, 6]; observed 7",
                   "setVdwType");
}

TEST_CASE("CapiForceManagerMapsMissingPsf") {
  ForceManagerHandle forceManager = MakeDefaultForceManager();
  int numAtoms = 17;
  apo_status status = APO_STATUS_OK;

  CHECK_NOTHROW((
      status = apo_force_manager_get_num_atoms(&numAtoms, forceManager.get())));
  CHECK(numAtoms == 0);
  CheckStatusAndDiagnostic(
      status, APO_STATUS_NOT_INITIALIZED,
      "apo_force_manager_get_num_atoms: CharmmPSF is not set");

  auto uninitializedPsf = std::make_shared<CharmmPSF>();
  forceManager->object->setPsf(uninitializedPsf);
  numAtoms = 17;

  CHECK_NOTHROW((
      status = apo_force_manager_get_num_atoms(&numAtoms, forceManager.get())));
  CHECK(status == APO_STATUS_OK);
  CHECK(numAtoms == -1);
  CHECK(std::string(apo_last_error()).empty() == true);
}

TEST_CASE("CapiPbcconversionRejectsUnknownNativeValue") {
  apo_pbc pbc = APO_PBC_UNSET;
  apo_status status = APO_STATUS_OK;

  CHECK_NOTHROW((status = apocharmm_c::from_pbc(
                     &pbc, static_cast<PBC>(99),
                     "apo_force_manager_get_periodic_boundary_condition")));

  CHECK(pbc == APO_PBC_UNSET);
  CheckStatusAndDiagnostic(status, APO_STATUS_INVALID_ARGUMENT,
                           "apo_force_manager_get_periodic_boundary_condition: "
                           "unknown C++ PBC value");
}

TEST_CASE("CapiForceManagerClearsStaleErrorsAndDestroyPreservesDiagnostic") {
  ForceManagerInputs inputs;
  ForceManagerHandle forceManager = MakeForceManager(inputs);

  CHECK(apo_force_manager_set_kappa(forceManager.get(), -1.0) ==
        APO_STATUS_INVALID_ARGUMENT);
  CHECK(std::string(apo_last_error()).empty() == false);

  CHECK(apo_force_manager_set_kappa(forceManager.get(), 0.34) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);

  CHECK(apo_force_manager_set_cutoff(forceManager.get(), 0.0) ==
        APO_STATUS_INVALID_ARGUMENT);

  const std::string cutoffDiagnostic(apo_last_error());
  REQUIRE(cutoffDiagnostic.empty() == false);

  apo_force_manager *rawForceManager = forceManager.release();
  CHECK_NOTHROW(apo_force_manager_destroy(rawForceManager));
  CHECK(std::string(apo_last_error()) == cutoffDiagnostic);

  CHECK_NOTHROW(apo_force_manager_destroy(nullptr));
  CHECK(std::string(apo_last_error()) == cutoffDiagnostic);
}
