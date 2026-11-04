// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "Crystal.h"
#include "apo_test_helpers.h"
#include "apocharmm_c/CudaLangevinPistonIntegrator.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/CharmmContextHandle.h"
#include "apocharmm_c/detail/CudaLangevinPistonIntegratorHandle.h"
#include "catch.hpp"

#include <limits>
#include <memory>
#include <string>

namespace {

constexpr double TIME_STEP = 0.001;

apo_cuda_langevin_piston_integrator *CreateIntegrator(void) {
  apo_cuda_langevin_piston_integrator *integrator = nullptr;

  const apo_status status =
      apo_cuda_langevin_piston_integrator_create(&integrator, TIME_STEP);

  REQUIRE(status == APO_STATUS_OK);
  REQUIRE(integrator != nullptr);

  return integrator;
}

std::shared_ptr<CharmmContext>
AttachCubicContext(apo_cuda_langevin_piston_integrator *integrator) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setCrystalType(CRYSTAL::CUBIC);
  ctx->setBoxDimensions({50.0, 50.0, 50.0});
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->assignVelocitiesAtTemperature(300.0);

  integrator->object->setCharmmContext(ctx);

  return ctx;
}

} // namespace

TEST_CASE("CapiCudaLangevinPistonConstructionAndDestruction") {
  SECTION("NullOutputPointer") {
    apo_status status = APO_STATUS_OK;

    CHECK_NOTHROW(status = apo_cuda_langevin_piston_integrator_create(
                      nullptr, TIME_STEP));

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_cuda_langevin_piston_integrator_create: out pointer is NULL");
  }

  SECTION("ZeroTimeStep") {
    apo_cuda_langevin_piston_integrator *integrator = nullptr;
    apo_status status = APO_STATUS_OK;

    CHECK_NOTHROW(
        status = apo_cuda_langevin_piston_integrator_create(&integrator, 0.0));

    CHECK(integrator == nullptr);

    apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                               "InvalidArgument",
                               "apo_cuda_langevin_piston_integrator_create",
                               "Time step must be positive; observed 0.000000",
                               "src/CudaIntegrator.cu", "setTimeStep");
  }

  SECTION("InfiniteTimeStep") {
    const double infinity = std::numeric_limits<double>::infinity();

    apo_cuda_langevin_piston_integrator *integrator = nullptr;
    apo_status status = APO_STATUS_OK;

    CHECK_NOTHROW(status = apo_cuda_langevin_piston_integrator_create(
                      &integrator, infinity));

    CHECK(integrator == nullptr);

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_cuda_langevin_piston_integrator_create",
        "Time step must be finite; observed " + std::to_string(infinity),
        "src/CudaIntegrator.cu", "setTimeStep");
  }

  SECTION("DestroyIsNoThrow") {
    apo_cuda_langevin_piston_integrator *integrator = CreateIntegrator();

    CHECK_NOTHROW(apo_cuda_langevin_piston_integrator_destroy(integrator));
    CHECK_NOTHROW(apo_cuda_langevin_piston_integrator_destroy(nullptr));
  }
}

TEST_CASE("CapiCudaLangevinPistonValidation") {
  const double infinity = std::numeric_limits<double>::infinity();

  apo_cuda_langevin_piston_integrator *integrator = CreateIntegrator();

  apo_status status = APO_STATUS_OK;

  CHECK_NOTHROW(status =
                    apo_cuda_langevin_piston_integrator_set_reference_pressure(
                        integrator, nullptr, 9));

  apo_test::CheckStatusAndDiagnostic(
      status, APO_STATUS_INVALID_ARGUMENT,
      "apo_cuda_langevin_piston_integrator_set_reference_pressure: "
      "pressure_tensor is NULL");

  double pressure[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  CHECK_NOTHROW(status =
                    apo_cuda_langevin_piston_integrator_set_reference_pressure(
                        integrator, pressure, 8));

  apo_test::CheckStatusAndDiagnostic(
      status, APO_STATUS_INVALID_ARGUMENT,
      "apo_cuda_langevin_piston_integrator_set_reference_pressure: "
      "pressure_tensor must contain exactly 9 elements");

  pressure[4] = infinity;

  CHECK_NOTHROW(status =
                    apo_cuda_langevin_piston_integrator_set_reference_pressure(
                        integrator, pressure, 9));

  apo_test::CheckNativeError(
      status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
      "apo_cuda_langevin_piston_integrator_set_reference_pressure",
      "Reference pressure tensor values must be finite; non-finite value at "
      "index 4",
      "src/CudaLangevinPistonIntegrator.cu", "setReferencePressure");

  double mass[2] = {1.0, 2.0};

  CHECK_NOTHROW(
      status = apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
          integrator, mass, 1));

  apo_test::CheckNativeError(
      status, APO_STATUS_NOT_INITIALIZED, "NotInitialized",
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_mass",
      "Attach a CharmmContext with a crystal type before setting the Langevin "
      "piston mass",
      "src/CudaLangevinPistonIntegrator.cu", "setLangevinPistonMass");

  double temperature = 0.0;

  CHECK_NOTHROW(
      status =
          apo_cuda_langevin_piston_integrator_get_instantaneous_temperature(
              &temperature, integrator));

  apo_test::CheckNativeError(
      status, APO_STATUS_NOT_INITIALIZED, "NotInitialized",
      "apo_cuda_langevin_piston_integrator_get_instantaneous_temperature",
      "CharmmContext must be set before computing instantaneous temperature",
      "src/CudaLangevinPistonIntegrator.cu", "getInstantaneousTemperature");

  AttachCubicContext(integrator);
  apo_crystal_type selected = APO_CRYSTAL_TYPE_UNSET;
  REQUIRE(apo_cuda_langevin_piston_integrator_get_crystal_type(
              &selected, integrator) == APO_STATUS_OK);
  CHECK(selected == APO_CRYSTAL_TYPE_CUBIC);

  CHECK_NOTHROW(
      status = apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
          integrator, mass, 2));

  apo_test::CheckNativeError(
      status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_mass",
      "Langevin piston mass must contain 1 element(s); observed 2",
      "src/CudaLangevinPistonIntegrator.cu", "setLangevinPistonMass");

  mass[0] = -1.0;

  CHECK_NOTHROW(
      status = apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
          integrator, mass, 1));

  apo_test::CheckNativeError(
      status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_mass",
      "Langevin piston mass values must be non-negative; observed "
      "-1.000000 at index 0",
      "src/CudaLangevinPistonIntegrator.cu", "setLangevinPistonMass");

  CHECK_NOTHROW(
      status = apo_cuda_langevin_piston_integrator_set_langevin_piston_friction(
          integrator, -1.0));

  apo_test::CheckNativeError(
      status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_friction",
      "Langevin piston friction must be non-negative; observed -1.000000",
      "src/CudaLangevinPistonIntegrator.cu", "setLangevinPistonFriction");

  CHECK_NOTHROW(apo_cuda_langevin_piston_integrator_destroy(integrator));
}

TEST_CASE("CapiCudaLangevinPistonGetterInvariants") {
  apo_cuda_langevin_piston_integrator *integrator = CreateIntegrator();

  apo_status status = APO_STATUS_OK;

  SECTION("NoseHooverPistonMass") {
    integrator->object->getNoseHooverPistonMass().resize(0);
    double value = 0.0;

    CHECK_NOTHROW(
        status =
            apo_cuda_langevin_piston_integrator_get_nose_hoover_piston_mass(
                &value, integrator));

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_RUNTIME_ERROR,
        "apo_cuda_langevin_piston_integrator_get_nose_hoover_piston_mass: "
        "Nose-Hoover piston mass does not contain exactly 1 element");
  }

  SECTION("AverageTemperature") {
    integrator->object->getAverageTemperature().resize(1);
    double value = 0.0;

    CHECK_NOTHROW(
        status = apo_cuda_langevin_piston_integrator_get_average_temperature(
            &value, integrator));

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_RUNTIME_ERROR,
        "apo_cuda_langevin_piston_integrator_get_average_temperature: "
        "Average temperature does not contain exactly 2 elements");
  }

  SECTION("InstantaneousPressureScalar") {
    integrator->object->getInstantaneousPressureScalar().resize(0);
    double value = 0.0;

    CHECK_NOTHROW(
        status =
            apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_scalar(
                &value, integrator));

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_RUNTIME_ERROR,
        "apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_scalar:"
        " "
        "Instantaneous pressure scalar does not contain exactly 1 element");
  }

  SECTION("AveragePressureScalar") {
    integrator->object->getAveragePressureScalar().resize(0);
    double value = 0.0;

    CHECK_NOTHROW(
        status =
            apo_cuda_langevin_piston_integrator_get_average_pressure_scalar(
                &value, integrator));

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_RUNTIME_ERROR,
        "apo_cuda_langevin_piston_integrator_get_average_pressure_scalar: "
        "Average pressure scalar does not contain exactly 1 element");
  }

  CHECK_NOTHROW(apo_cuda_langevin_piston_integrator_destroy(integrator));
}

TEST_CASE("CapiCudaLangevinPistonCrystalPhaseChange") {
  apo_cuda_langevin_piston_integrator *integrator = CreateIntegrator();
  apo_cuda_integrator *base = nullptr;
  REQUIRE(apo_cuda_langevin_piston_integrator_as_cuda_integrator(
              &base, integrator) == APO_STATUS_OK);
  REQUIRE(base != nullptr);

  apo_cuda_integrator empty;
  CHECK(apo_cuda_integrator_initialize(nullptr) == APO_STATUS_INVALID_ARGUMENT);
  CHECK(apo_cuda_integrator_initialize(&empty) == APO_STATUS_INVALID_ARGUMENT);
  apo_test::CheckNativeError(apo_cuda_integrator_initialize(base),
                             APO_STATUS_NOT_INITIALIZED, "NotInitialized",
                             "apo_cuda_integrator_initialize",
                             "CharmmContext must be set before initialization",
                             "src/CudaIntegrator.cu", "initialize");

  apo_charmm_context context;
  context.object = AttachCubicContext(integrator);

  const double cubicMass[1] = {5.0};
  REQUIRE(apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
              integrator, cubicMass, 1) == APO_STATUS_OK);
  REQUIRE(apo_cuda_langevin_piston_integrator_set_langevin_piston_friction(
              integrator, 5.0) == APO_STATUS_OK);
  REQUIRE(apo_cuda_integrator_propagate(base, 1) == APO_STATUS_OK);
  const auto completed = integrator->object->getTotNumSteps();

  // An unchanged selection keeps the current piston layout usable.
  REQUIRE(apo_charmm_context_set_crystal_type(
              &context, APO_CRYSTAL_TYPE_CUBIC) == APO_STATUS_OK);
  size_t numMass = 0;
  double mass[3] = {0.0, 0.0, 0.0};
  REQUIRE(apo_cuda_langevin_piston_integrator_get_langevin_piston_mass(
              &numMass, mass, 3, integrator) == APO_STATUS_OK);
  CHECK(numMass == 1);
  CHECK(mass[0] == 5.0);

  REQUIRE(apo_charmm_context_set_crystal_type(
              &context, APO_CRYSTAL_TYPE_ORTHORHOMBIC) == APO_STATUS_OK);

  apo_crystal_type selected = APO_CRYSTAL_TYPE_UNSET;
  REQUIRE(apo_cuda_langevin_piston_integrator_get_crystal_type(
              &selected, integrator) == APO_STATUS_OK);
  CHECK(selected == APO_CRYSTAL_TYPE_ORTHORHOMBIC);

  const std::string stale =
      "Langevin piston state does not match the context crystal type; "
      "reinitialize the integrator";
  const auto checkStale = [&](apo_status status, const char *function) {
    apo_test::CheckNativeError(
        status, APO_STATUS_NOT_INITIALIZED, "NotInitialized", function, stale,
        "src/CudaLangevinPistonIntegrator.cu", "requireCurrentState");
  };
  mass[0] = 123.0;
  checkStale(apo_cuda_langevin_piston_integrator_get_langevin_piston_mass(
                 &numMass, mass, 3, integrator),
             "apo_cuda_langevin_piston_integrator_get_langevin_piston_mass");
  CHECK(numMass == 0);
  CHECK(mass[0] == 123.0);

  const double orthMass[3] = {5.0, 6.0, 7.0};
  checkStale(apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
                 integrator, orthMass, 3),
             "apo_cuda_langevin_piston_integrator_set_langevin_piston_mass");
  checkStale(
      apo_cuda_langevin_piston_integrator_set_langevin_piston_friction(
          integrator, 6.0),
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_friction");
  checkStale(apo_cuda_integrator_propagate(base, 1),
             "apo_cuda_integrator_propagate");
  checkStale(apo_cuda_integrator_initialize_from_restart_file(
                 base, "unused_while_piston_state_is_stale.rst"),
             "apo_cuda_integrator_initialize_from_restart_file");
  CHECK(integrator->object->getTotNumSteps() == completed);

  REQUIRE(apo_cuda_integrator_initialize(base) == APO_STATUS_OK);
  CHECK(integrator->object->getTotNumSteps() == completed);
  REQUIRE(apo_cuda_langevin_piston_integrator_get_langevin_piston_mass(
              &numMass, mass, 3, integrator) == APO_STATUS_OK);
  CHECK(numMass == 3);
  REQUIRE(apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
              integrator, orthMass, 3) == APO_STATUS_OK);
  REQUIRE(apo_cuda_integrator_initialize(base) == APO_STATUS_OK);
  REQUIRE(apo_cuda_langevin_piston_integrator_get_langevin_piston_mass(
              &numMass, mass, 3, integrator) == APO_STATUS_OK);
  REQUIRE(numMass == 3);
  for (size_t i = 0; i < numMass; ++i)
    CHECK(mass[i] == orthMass[i]);
  CHECK(integrator->object->getTotNumSteps() == completed);
  REQUIRE(apo_cuda_integrator_propagate(base, 1) == APO_STATUS_OK);
  CHECK(integrator->object->getTotNumSteps() == completed + 1);
  CHECK_NOTHROW(apo_cuda_langevin_piston_integrator_destroy(integrator));
}
