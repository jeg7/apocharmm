// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "ApoCharmmError.h"
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "Crystal.h"
#include "CudaIntegrator.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaNoseHooverIntegrator.h"
#include "DcdSubscriber.h"
#include "ForceManager.h"
#include "RestartIO.h"
#include "RestartSubscriber.h"
#include "apo_test_helpers.h"
#include "catch.hpp"
#include "str_utils.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int NUM_STEPS = 3;
constexpr int REPORT_FREQUENCY = NUM_STEPS;
constexpr int RANDOM_SEED = 314159;
constexpr double TIME_STEP = 0.001;
constexpr double REFERENCE_TEMPERATURE = 300.0;
constexpr double NOSE_HOOVER_PISTON_MASS = 300.0;
constexpr double LANGEVIN_THERMOSTAT_FRICTION = 1.0;
constexpr double LANGEVIN_PISTON_MASS = 5.0;
constexpr double LANGEVIN_PISTON_FRICTION = 5.0;
constexpr std::uint64_t LANGEVIN_RNG_SEED = 987654321ULL;
constexpr double TOLERANCE = 0.0;

const std::vector<double> BOX_DIMENSIONS = {50.0, 50.0, 50.0};
const std::vector<double> REFERENCE_PRESSURE_TENSOR = {1.0, 0.0, 0.0, 0.0, 1.0,
                                                       0.0, 0.0, 0.0, 1.0};

std::shared_ptr<CharmmContext>
CreateContext(const bool assignVelocities,
              const CRYSTAL crystalType = CRYSTAL::CUBIC) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  if (crystalType != CRYSTAL::UNSET)
    ctx->setCrystalType(crystalType);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);

  if (assignVelocities == true) {
    ctx->setRandomSeed(RANDOM_SEED);
    ctx->assignVelocitiesAtTemperature(REFERENCE_TEMPERATURE);
  }

  return ctx;
}

std::shared_ptr<CudaNoseHooverIntegrator>
CreateNoseHooverIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaNoseHooverIntegrator>(TIME_STEP);
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator->useOldTemperature(false);
  integrator->setCharmmContext(ctx);
  return integrator;
}

std::shared_ptr<CudaLangevinThermostatIntegrator>
CreateLangevinThermostatIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setThermostatFriction(LANGEVIN_THERMOSTAT_FRICTION);
  integrator->setThermostatRngSeed(LANGEVIN_RNG_SEED);
  integrator->setRngSequencePos(0ULL);
  integrator->setCharmmContext(ctx);
  return integrator;
}

std::shared_ptr<CudaLangevinPistonIntegrator>
CreateLangevinPistonIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator->useNoseHooverThermostat(true);
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator->useOldTemperature(false);
  integrator->setReferencePressure(REFERENCE_PRESSURE_TENSOR);
  integrator->setConstantSurfaceTension(false);
  integrator->setLangevinPistonFrictionSeed(LANGEVIN_RNG_SEED);
  integrator->setRngSequencePos(0ULL);
  integrator->setCharmmContext(ctx);
  integrator->setLangevinPistonMass(std::vector<double>(
      integrator->getLangevinPistonMass().size(), LANGEVIN_PISTON_MASS));
  integrator->setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION);
  return integrator;
}

void WriteRestartFile(const std::filesystem::path &filePath,
                      const std::shared_ptr<CharmmContext> &ctx,
                      const std::shared_ptr<CudaIntegrator> &integrator) {
  RestartSubscriber rst(filePath, REPORT_FREQUENCY);
  rst.setCharmmContext(ctx);
  rst.setIntegrator(integrator);
  rst.update();
  return;
}

void CheckRestartFileContainsRequiredSections(
    const std::filesystem::path &filePath) {
  const std::string contents = apo_test::ReadTextFile(filePath);

  REQUIRE(contents.size() >= 36);
  CHECK(contents.compare(0, 4, "REST") == 0);
  CHECK(contents.compare(32, 3, "APO") == 0);
  CHECK(contents.find("!CRYSTAL PARAMETERS") != std::string::npos);
  CHECK(
      contents.find("!NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL") !=
      std::string::npos);
  CHECK(contents.find("!ENERGIES and STATISTICS") != std::string::npos);
  CHECK(contents.find("!XOLD, YOLD, ZOLD") != std::string::npos);
  CHECK(contents.find("!VX, VY, VZ") != std::string::npos);
  CHECK(contents.find("!X, Y, Z") != std::string::npos);

  return;
}

void CheckCharmmContextStateMatches(
    const std::shared_ptr<CharmmContext> &ctx1,
    const std::shared_ptr<CharmmContext> &ctx2) {
  apo_test::CheckVectorsClose<double4>(
      "Coordinates/charges",
      apo_test::CopyToHost<double4>(ctx1->getCoordinatesChargesDP()),
      apo_test::CopyToHost<double4>(ctx2->getCoordinatesChargesDP()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double4>(
      "Velocities/masses",
      apo_test::CopyToHost<double4>(ctx1->getVelocitiesInverseMasses()),
      apo_test::CopyToHost<double4>(ctx2->getVelocitiesInverseMasses()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>("Box dimensions",
                                      ctx1->getBoxDimensions(),
                                      ctx2->getBoxDimensions(), TOLERANCE);
  return;
}

void CheckBaseIntegratorStateMatches(CudaIntegrator &integrator1,
                                     CudaIntegrator &integrator2) {
  CHECK(integrator1.getTotNumSteps() == integrator2.getTotNumSteps());
  CHECK(integrator1.getTotNumSteps() ==
        static_cast<unsigned long long int>(NUM_STEPS + 1));
  apo_test::CheckVectorsClose<double4>(
      "Coordinates delta",
      apo_test::CopyToHost<double4>(integrator1.getCoordsDelta()),
      apo_test::CopyToHost<double4>(integrator2.getCoordsDelta()), TOLERANCE);
  apo_test::CheckVectorsClose<double4>(
      "Coordinate delta previous",
      apo_test::CopyToHost<double4>(integrator1.getCoordsDeltaPrevious()),
      apo_test::CopyToHost<double4>(integrator2.getCoordsDeltaPrevious()),
      TOLERANCE);
  return;
}

void CheckNoseHooverStateMatches(CudaNoseHooverIntegrator &integrator1,
                                 CudaNoseHooverIntegrator &integrator2) {
  CheckBaseIntegratorStateMatches(integrator1, integrator2);

  CHECK(integrator1.getReferenceTemperature() ==
        Approx(integrator2.getReferenceTemperature()).margin(TOLERANCE));
  CHECK(integrator1.usingOldTemperature() == integrator2.usingOldTemperature());

  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston velocity",
      apo_test::CopyToHost<double>(integrator1.getNoseHooverPistonVelocity()),
      apo_test::CopyToHost<double>(integrator2.getNoseHooverPistonVelocity()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston velocity previous",
      apo_test::CopyToHost<double>(
          integrator1.getNoseHooverPistonVelocityPrevious()),
      apo_test::CopyToHost<double>(
          integrator2.getNoseHooverPistonVelocityPrevious()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston force",
      apo_test::CopyToHost<double>(integrator1.getNoseHooverPistonForce()),
      apo_test::CopyToHost<double>(integrator2.getNoseHooverPistonForce()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston force previous",
      apo_test::CopyToHost<double>(
          integrator1.getNoseHooverPistonForcePrevious()),
      apo_test::CopyToHost<double>(
          integrator2.getNoseHooverPistonForcePrevious()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Kinetic energy",
      apo_test::CopyToHost<double>(integrator1.getKineticEnergy()),
      apo_test::CopyToHost<double>(integrator2.getKineticEnergy()), TOLERANCE);

  return;
}

void CheckLangevinThermostatStateMatches(
    CudaLangevinThermostatIntegrator &integrator1,
    CudaLangevinThermostatIntegrator &integrator2) {
  CheckBaseIntegratorStateMatches(integrator1, integrator2);

  CHECK(integrator1.getReferenceTemperature() ==
        Approx(integrator2.getReferenceTemperature()).margin(TOLERANCE));
  CHECK(integrator1.getThermostatFriction() ==
        Approx(integrator2.getThermostatFriction()).margin(TOLERANCE));
  CHECK(integrator1.getThermostatRngSeed() ==
        integrator2.getThermostatRngSeed());
  CHECK(integrator1.getRngSequencePos() == integrator2.getRngSequencePos());

  apo_test::CheckVectorsClose<double>(
      "Kinetic energy",
      apo_test::CopyToHost<double>(integrator1.getKineticEnergy()),
      apo_test::CopyToHost<double>(integrator2.getKineticEnergy()), TOLERANCE);

  return;
}

void CheckLangevinPistonStateMatches(
    CudaLangevinPistonIntegrator &integrator1,
    CudaLangevinPistonIntegrator &integrator2) {
  CheckBaseIntegratorStateMatches(integrator1, integrator2);

  CHECK(integrator1.getReferenceTemperature() ==
        Approx(integrator2.getReferenceTemperature()).margin(TOLERANCE));
  CHECK(integrator1.usingOldTemperature() == integrator2.usingOldTemperature());
  CHECK(integrator1.getCrystalType() == integrator2.getCrystalType());
  CHECK(integrator1.getLangevinPistonFrictionSeed() ==
        integrator2.getLangevinPistonFrictionSeed());
  CHECK(integrator1.getRngSequencePos() == integrator2.getRngSequencePos());

  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston velocity",
      apo_test::CopyToHost<double>(integrator1.getNoseHooverPistonVelocity()),
      apo_test::CopyToHost<double>(integrator2.getNoseHooverPistonVelocity()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston velocity previous",
      apo_test::CopyToHost<double>(
          integrator1.getNoseHooverPistonVelocityPrevious()),
      apo_test::CopyToHost<double>(
          integrator2.getNoseHooverPistonVelocityPrevious()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston force",
      apo_test::CopyToHost<double>(integrator1.getNoseHooverPistonForce()),
      apo_test::CopyToHost<double>(integrator2.getNoseHooverPistonForce()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Nose-Hoover piston force previous",
      apo_test::CopyToHost<double>(
          integrator1.getNoseHooverPistonForcePrevious()),
      apo_test::CopyToHost<double>(
          integrator2.getNoseHooverPistonForcePrevious()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Kinetic energy",
      apo_test::CopyToHost<double>(integrator1.getKineticEnergy()),
      apo_test::CopyToHost<double>(integrator2.getKineticEnergy()), TOLERANCE);

  apo_test::CheckVectorsClose<double>(
      "Langevin piston on step position",
      apo_test::CopyToHost<double>(
          integrator1.getLangevinPistonOnStepPosition()),
      apo_test::CopyToHost<double>(
          integrator2.getLangevinPistonOnStepPosition()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Langevin piston half step position",
      apo_test::CopyToHost<double>(
          integrator1.getLangevinPistonHalfStepPosition()),
      apo_test::CopyToHost<double>(
          integrator2.getLangevinPistonHalfStepPosition()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Langevin piston on step velocity",
      apo_test::CopyToHost<double>(
          integrator1.getLangevinPistonOnStepVelocity()),
      apo_test::CopyToHost<double>(
          integrator2.getLangevinPistonOnStepVelocity()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Langevin piston half step velocity",
      apo_test::CopyToHost<double>(
          integrator1.getLangevinPistonHalfStepVelocity()),
      apo_test::CopyToHost<double>(
          integrator2.getLangevinPistonHalfStepVelocity()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Langevin piston delta position",
      apo_test::CopyToHost<double>(
          integrator1.getLangevinPistonDeltaPosition()),
      apo_test::CopyToHost<double>(
          integrator2.getLangevinPistonDeltaPosition()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Langevin piston delta position previous",
      apo_test::CopyToHost<double>(
          integrator1.getLangevinPistonDeltaPositionPrevious()),
      apo_test::CopyToHost<double>(
          integrator2.getLangevinPistonDeltaPositionPrevious()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Instantaneous pressure tensor",
      apo_test::CopyToHost<double>(
          integrator1.getInstantaneousPressureTensor()),
      apo_test::CopyToHost<double>(
          integrator2.getInstantaneousPressureTensor()),
      TOLERANCE);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator1.getInstantaneousPressureScalar()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   integrator2.getInstantaneousPressureScalar()))
            .margin(TOLERANCE));

  return;
}

} // namespace

TEST_CASE("RestartSubscriberConstructionAndReportFrequency") {
  SECTION("DefaultConstruction") {
    RestartSubscriber rst;

    CHECK(rst.getFilePath().empty() == true);
    CHECK(rst.getReportFrequency() == 1000);
  }

  SECTION("FileConstructor") {
    const std::filesystem::path filePath =
        "tmpRestartSubscriberConstructor.rst";
    apo_test::RemoveIfExists(filePath);

    RestartSubscriber rst(filePath);

    CHECK(rst.getFilePath() == filePath);
    CHECK(rst.getReportFrequency() == 1000);

    std::ifstream fin(filePath);
    CHECK(fin.good());

    apo_test::RemoveIfExists(filePath);
  }

  SECTION("FileAndReportFrequencyConstructor") {
    const std::filesystem::path filePath =
        "tmpRestartSubscriberReportFrequency.rst";
    apo_test::RemoveIfExists(filePath);

    RestartSubscriber rst(filePath, REPORT_FREQUENCY);

    CHECK(rst.getFilePath() == filePath);
    CHECK(rst.getReportFrequency() == REPORT_FREQUENCY);

    rst.setReportFrequency(1);
    CHECK(rst.getReportFrequency() == 1);

    apo_test::RemoveIfExists(filePath);
  }

  SECTION("RejectsMissingOutputDirectory") {
    apo_test::CheckApoCharmmError(
        []() {
          RestartSubscriber("missing_restart_subscriber_dir/output.rst");
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Output directory does not exist: missing_restart_subscriber_dir");
  }
}

TEST_CASE("RestartSubscriberUpdateValidatesRequiredState") {
  SECTION("RequiresOutputFile") {
    auto ctx = CreateContext(true);
    auto integrator = CreateNoseHooverIntegrator(ctx);

    RestartSubscriber rst;
    rst.setCharmmContext(ctx);
    rst.setIntegrator(integrator);

    apo_test::CheckApoCharmmError(
        [&]() { rst.update(); }, ApoCharmmErrorCode::NotInitialized,
        "RestartSubscriber requires an output file before update");
  }

  SECTION("RequiresCharmmContext") {
    const std::string fileName = "tmpRestartSubscriberMissingContext.rst";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext(true);
    auto integrator = CreateNoseHooverIntegrator(ctx);

    RestartSubscriber rst(fileName, REPORT_FREQUENCY);
    rst.setIntegrator(integrator);

    apo_test::CheckApoCharmmError(
        [&]() { rst.update(); }, ApoCharmmErrorCode::NotInitialized,
        "RestartSubscriber requires a CharmmContext before update");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresIntegrator") {
    const std::string fileName = "tmpRestartSubscriberMissingIntegrator.rst";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext(true);

    RestartSubscriber rst(fileName, REPORT_FREQUENCY);
    rst.setCharmmContext(ctx);

    apo_test::CheckApoCharmmError(
        [&]() { rst.update(); }, ApoCharmmErrorCode::NotInitialized,
        "RestartSubscriber requires an integrator before update");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresIntegratorCharmmContext") {
    const std::string fileName =
        "tmpRestartSubscriberMissingIntegratorContext.rst";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext(true);
    auto integrator = std::make_shared<CudaNoseHooverIntegrator>(TIME_STEP);

    RestartSubscriber rst(fileName, REPORT_FREQUENCY);
    rst.setCharmmContext(ctx);
    rst.setIntegrator(integrator);

    apo_test::CheckApoCharmmError(
        [&]() { rst.update(); }, ApoCharmmErrorCode::NotInitialized,
        "RestartSubscriber integrator has no CharmmContext");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresMatchingCharmmContext") {
    const std::string fileName =
        "tmpRestartSubscriberMismatchedIntegratorContext.rst";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext(true);
    auto integratorContext = CreateContext(true);
    auto integrator = CreateNoseHooverIntegrator(integratorContext);

    RestartSubscriber rst(fileName, REPORT_FREQUENCY);
    rst.setCharmmContext(ctx);
    rst.setIntegrator(integrator);

    apo_test::CheckApoCharmmError([&]() { rst.update(); },
                                  ApoCharmmErrorCode::Runtime,
                                  "RestartSubscriber CharmmContext does not "
                                  "match the integrator CharmmContext");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresBoxDimensions") {
    const std::string fileName = "tmpRestartSubscriberMissingBoxDimensions.rst";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext(true);
    auto integrator = CreateNoseHooverIntegrator(ctx);
    ctx->getBoxDimensions().clear();

    RestartSubscriber rst(fileName, REPORT_FREQUENCY);
    rst.setCharmmContext(ctx);
    rst.setIntegrator(integrator);

    apo_test::CheckApoCharmmError(
        [&]() { rst.update(); }, ApoCharmmErrorCode::NotInitialized,
        "Crystal box requires exactly 3 finite positive dimensions");

    apo_test::RemoveIfExists(fileName);
  }
}

TEST_CASE("RestartSubscriberUpdateReportsOutputOpenFailure") {
  auto ctx = CreateContext(true);
  auto integrator = CreateNoseHooverIntegrator(ctx);

  RestartSubscriber rst;
  rst.setReportFrequency(REPORT_FREQUENCY);
  rst.setFilePath(".");
  rst.setCharmmContext(ctx);
  rst.setIntegrator(integrator);

  apo_test::CheckApoCharmmError(
      [&]() { rst.update(); }, ApoCharmmErrorCode::Runtime,
      "Failed to open subscriber output file for writing: .");
}

#if defined(__linux__)
TEST_CASE("RestartSubscriberUpdateReportsOutputWriteFailure") {
  constexpr const char *FILE_NAME = "/dev/full";

  auto ctx = CreateContext(true);
  auto integrator = CreateNoseHooverIntegrator(ctx);

  RestartSubscriber rst(FILE_NAME, REPORT_FREQUENCY);
  rst.setCharmmContext(ctx);
  rst.setIntegrator(integrator);

  apo_test::CheckApoCharmmError([&]() { rst.update(); },
                                ApoCharmmErrorCode::Runtime,
                                "Failed to write restart file: /dev/full");
}
#endif

TEST_CASE("RestartSubscriberRejectsUnsupportedIntegrator") {
  const std::string fileName = "tmpUnsupportedRestartSubscriber.rst";
  apo_test::RemoveIfExists(fileName);

  auto ctx = CreateContext(true);
  auto integrator = std::make_shared<CudaIntegrator>(TIME_STEP);

  RestartSubscriber rst(fileName, REPORT_FREQUENCY);
  rst.setCharmmContext(ctx);
  rst.setIntegrator(integrator);

  apo_test::CheckApoCharmmError(
      [&]() { rst.update(); }, ApoCharmmErrorCode::NotImplemented,
      "RestartSubscriber supports only CudaNoseHooverIntegrator, "
      "CudaLangevinPistonIntegrator, and CudaLangevinThermostatIntegrator");

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("RestartSubscriberRejectsDuplicateContextOrIntegrator") {
  const std::string fileName = "tmpDuplicateRestartSubscriber.rst";
  apo_test::RemoveIfExists(fileName);

  auto ctx = CreateContext(true);
  auto integrator = CreateNoseHooverIntegrator(ctx);

  RestartSubscriber rst(fileName, REPORT_FREQUENCY);

  rst.setCharmmContext(ctx);
  apo_test::CheckApoCharmmError([&]() { rst.setCharmmContext(ctx); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Subscriber already has a CharmmContext");

  rst.setIntegrator(integrator);
  apo_test::CheckApoCharmmError([&]() { rst.setIntegrator(integrator); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Subscriber already has an Integrator");

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("RestartSubscriberNoseHooverTrajectoryEquivalence") {
  const std::string fileName = "tmpRestartNoseHoover.rst";
  apo_test::RemoveIfExists(fileName);

  auto ctx1 = CreateContext(true);
  auto ctx2 = CreateContext(true);
  auto ctx3 = CreateContext(false);

  auto integrator1 = CreateNoseHooverIntegrator(ctx1);
  auto integrator2 = CreateNoseHooverIntegrator(ctx2);
  auto integrator3 = CreateNoseHooverIntegrator(ctx3);

  integrator1->propagate(NUM_STEPS + 1);

  integrator2->propagate(NUM_STEPS);
  WriteRestartFile(fileName, ctx2, integrator2);
  CheckRestartFileContainsRequiredSections(fileName);

  integrator3->initializeFromRestartFile(fileName);

  CheckCharmmContextStateMatches(ctx2, ctx3);

  CHECK(integrator3->getTotNumSteps() ==
        static_cast<unsigned long long int>(NUM_STEPS));
  CHECK(integrator3->getNumSteps() == NUM_STEPS);
  CHECK(integrator3->getCurrentPropagatedStep() == NUM_STEPS);

  // The CHARMM !X, Y, Z restart section stores the previous coordinate
  // displacement. The current displacement is not serialized.
  apo_test::CheckVectorsClose<double4>(
      "Loaded coordinates delta previous",
      apo_test::CopyToHost<double4>(integrator2->getCoordsDeltaPrevious()),
      apo_test::CopyToHost<double4>(integrator3->getCoordsDeltaPrevious()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Loaded Nose-Hoover piston velocity",
      apo_test::CopyToHost<double>(integrator2->getNoseHooverPistonVelocity()),
      apo_test::CopyToHost<double>(integrator3->getNoseHooverPistonVelocity()),
      TOLERANCE);
  apo_test::CheckVectorsClose<double>(
      "Loaded Nose-Hoover piston force",
      apo_test::CopyToHost<double>(integrator2->getNoseHooverPistonForce()),
      apo_test::CopyToHost<double>(integrator3->getNoseHooverPistonForce()),
      TOLERANCE);

  integrator2->propagate(1);
  integrator3->propagate(1);

  CheckCharmmContextStateMatches(ctx1, ctx2);
  CheckCharmmContextStateMatches(ctx1, ctx3);
  CheckCharmmContextStateMatches(ctx2, ctx3);

  CheckNoseHooverStateMatches(*integrator1, *integrator2);
  CheckNoseHooverStateMatches(*integrator1, *integrator3);
  CheckNoseHooverStateMatches(*integrator2, *integrator3);

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("RestartSubscriberWritesCurrentCHARMMStructure") {
  const std::filesystem::path path = "tmpRestartCHARMMStructure.rst";
  const std::filesystem::path dcdPath = "tmpRestartCHARMMStructure.dcd";
  auto ctx = CreateContext(true);
  auto integrator = CreateLangevinThermostatIntegrator(ctx);
  integrator->setThermostatRngSeed(std::numeric_limits<std::uint64_t>::max());
  auto dcd = std::make_shared<DcdSubscriber>(dcdPath, 7);
  integrator->subscribe(dcd);
  integrator->propagate(NUM_STEPS);
  WriteRestartFile(path, ctx, integrator);
  const std::string contents = apo_test::ReadTextFile(path);
  const auto lineAfter = [&contents](const std::string &marker) {
    const std::size_t markerPosition = contents.find(marker);
    REQUIRE(markerPosition != std::string::npos);
    const std::size_t first = contents.find('\n', markerPosition) + 1;
    const std::size_t last = contents.find('\n', first);
    REQUIRE(last != std::string::npos);
    return contents.substr(first, last - first);
  };
  REQUIRE(contents.substr(0, 4) == "REST");
  CHECK(std::stoi(contents.substr(4, 6)) == 52);
  CHECK(std::stoi(contents.substr(10, 6)) == 1);
  CHECK(contents.substr(24, 4) == "    ");
  CHECK(contents.substr(28, 4) == "    "); // DHDGB field stays blank
  CHECK(contents.substr(32, 3) == "APO");  // marker past READYN's fields
  CHECK(contents.substr(35, 1) == "\n");
  const std::string metadata = lineAfter(" !NATOM,NPRIV,NSTEP");
  REQUIRE(metadata.size() == 162);
  CHECK(std::stoi(metadata.substr(0, 12)) == ctx->getNumAtoms());
  CHECK(std::stoull(metadata.substr(12, 12)) ==
        static_cast<unsigned long long int>(NUM_STEPS));
  CHECK(std::stoi(metadata.substr(36, 12)) == 7);
  CHECK(std::stoi(metadata.substr(106, 12)) == 0);
  CHECK(std::stoll(metadata.substr(118, 22)) == 0);
  CHECK(std::stoll(metadata.substr(140, 22)) == 0);
  CHECK(metadata.substr(84, 22).find('D') != std::string::npos);
  const std::string propertyFlags = lineAfter(" !ENERGIES and STATISTICS");
  REQUIRE(propertyFlags.size() == 96);
  const std::string termFlags = lineAfter(propertyFlags);
  REQUIRE(termFlags.size() == 192);
  const auto snapshot = apo::ReadRestartFile(path, ctx->getNumAtoms());
  CHECK(snapshot.hasNativeRng);
  CHECK(snapshot.rngSeed == std::numeric_limits<std::uint64_t>::max());
  CHECK(snapshot.rngState == integrator->getRngStates());
  CHECK(snapshot.requestedSteps == NUM_STEPS);
  CHECK(snapshot.heatingSteps == 1);
  CHECK(snapshot.degreesOfFreedom == ctx->getNumDegreesOfFreedom());
  CHECK(snapshot.averageTemperature ==
        Approx(ctx->computeTemperature()).margin(1.0e-12));
  // Every non-RNG record is retained, including fields apoCHARMM does not use.
  CHECK(snapshot.version == 52);
  CHECK(snapshot.dynamicsType == 1);
  CHECK(snapshot.crystalLabel == "CUBI");
  CHECK(snapshot.titles.size() == 2);
  CHECK((snapshot.xtlabc ==
         std::array<double, 6>{50.0, 0.0, 50.0, 0.0, 0.0, 50.0}));
  CHECK((snapshot.uc1a == std::array<double, 6>{}));
  CHECK((snapshot.uc2b == std::array<double, 6>{}));
  CHECK((snapshot.cellGradients == std::array<double, 4>{}));
  CHECK(snapshot.numAtoms == ctx->getNumAtoms());
  CHECK(snapshot.coordinateSaveFrequency == 7);
  CHECK(snapshot.velocitySaveFrequency == 0);
  CHECK(snapshot.seed == 1.0);
  CHECK(snapshot.lambdaSaveFrequency == 0);
  CHECK(snapshot.propertyFlags == std::string(96, 'T'));
  CHECK(snapshot.termFlags == std::string(192, 'T'));
  CHECK(snapshot.averagingSteps == 0);
  CHECK(snapshot.fitA == 0.0);
  CHECK(snapshot.fitP == 0.0);
  REQUIRE(snapshot.propertyHistory.size() == 96);
  CHECK(snapshot.propertyHistory[0][0] == 0.0);
  CHECK(snapshot.propertyHistory[21][0] == -999.0); // PEPR carries FMARK
  CHECK(snapshot.propertyHistory[32][0] == -999.0); // XTLKP2 carries FMARK
  CHECK(snapshot.propertyAverages.size() == 96);
  CHECK(snapshot.termHistory.size() == 192);
  CHECK(snapshot.termAverages.size() == 192);
  CHECK(snapshot.pressureHistory.size() == 50);
  CHECK(snapshot.pressureAverages.size() == 50);
  integrator->unsubscribe(dcd);
  dcd.reset();
  apo_test::RemoveIfExists(path);
  apo_test::RemoveIfExists(dcdPath);
}

TEST_CASE("RestartSubscriberForeignRNGKeepsConfiguredSeed") {
  const std::filesystem::path path = "tmpRestartForeignRNG.rst";
  auto sourceContext = CreateContext(true);
  auto source = CreateLangevinThermostatIntegrator(sourceContext);
  source->propagate(NUM_STEPS);
  WriteRestartFile(path, sourceContext, source);
  std::string contents = apo_test::ReadTextFile(path);
  const std::size_t footer = contents.find("\n !APOCHARMM RNG STATE\n");
  REQUIRE(footer != std::string::npos);
  contents.erase(footer);
  // A CHARMM-written header has no marker at offsets 32-34.
  REQUIRE(contents.substr(32, 3) == "APO");
  contents.replace(32, 3, "   ");
  apo_test::WriteTextFile(path, contents);
  auto destinationContext = CreateContext(false);
  auto destination = CreateLangevinThermostatIntegrator(destinationContext);
  constexpr std::uint64_t configuredSeed = 123456789ULL;
  destination->setThermostatRngSeed(configuredSeed);
  destination->initializeFromRestartFile(path);
  CHECK(destination->getThermostatRngSeed() == configuredSeed);
  CHECK(destination->getRngSequencePos() == 0);
  CHECK(destination->getTotNumSteps() ==
        static_cast<unsigned long long int>(NUM_STEPS));
  CheckCharmmContextStateMatches(sourceContext, destinationContext);
  CHECK(destination->getAverageWindowSize() == 0);
  apo_test::RemoveIfExists(path);
}

TEST_CASE("ForeignRestartWithJhstrtZeroRegeneratesDisplacementHistory") {
  const std::filesystem::path path = "tmpRestartForeignHeating.rst";
  auto sourceContext = CreateContext(true);
  auto source = CreateLangevinThermostatIntegrator(sourceContext);
  // A step-0 snapshot records JHSTRT=0. Without the marker and the footer it
  // is the file CHARMM writes at the end of an equilibration run whose last
  // step rescaled velocities.
  WriteRestartFile(path, sourceContext, source);
  std::string contents = apo_test::ReadTextFile(path);
  const std::size_t footer = contents.find("\n !APOCHARMM RNG STATE\n");
  REQUIRE(footer != std::string::npos);
  contents.erase(footer);
  REQUIRE(contents.substr(32, 3) == "APO");
  contents.replace(32, 3, "   ");
  // Poison the !X, Y, Z block: only CHARMM's two-step start (IGVOPT=2), which
  // regenerates the previous displacement from velocities and forces, ignores
  // it.
  const std::size_t deltas = contents.find(" !X, Y, Z\n");
  REQUIRE(deltas != std::string::npos);
  contents.erase(contents.find('\n', deltas) + 1);
  const std::string poison = apo::format_fortran_double(1.0e3, 22, 15);
  for (int atom = 0; atom < sourceContext->getNumAtoms(); ++atom)
    contents += poison + poison + poison + '\n';
  apo_test::WriteTextFile(path, contents);
  const auto snapshot =
      apo::ReadRestartFile(path, sourceContext->getNumAtoms());
  REQUIRE(snapshot.heatingSteps == 0);
  REQUIRE(snapshot.nativeKind.empty());
  REQUIRE(snapshot.deltas[0][0] == 1.0e3);

  auto destinationContext = CreateContext(false);
  auto destination = CreateLangevinThermostatIntegrator(destinationContext);
  destination->initializeFromRestartFile(path);
  CHECK(destination->getTotNumSteps() == 0ULL);
  CheckCharmmContextStateMatches(sourceContext, destinationContext);
  // Both integrators regenerate the displacement from the same coordinates
  // and velocities, so the poisoned block must not be visible.
  apo_test::CheckVectorsClose<double4>(
      "Regenerated coordinate delta previous",
      apo_test::CopyToHost<double4>(destination->getCoordsDeltaPrevious()),
      apo_test::CopyToHost<double4>(source->getCoordsDeltaPrevious()),
      TOLERANCE);
  apo_test::RemoveIfExists(path);
}

TEST_CASE("RestartSubscriberRejectsInvalidFilesBeforeStateMutation") {
  const std::filesystem::path path = "tmpRestartInvalidState.rst";
  auto sourceContext = CreateContext(true);
  auto source = CreateLangevinThermostatIntegrator(sourceContext);
  source->propagate(NUM_STEPS);
  WriteRestartFile(path, sourceContext, source);
  std::string contents = apo_test::ReadTextFile(path);
  const std::string fileContext = "restart file \"" + path.string() + "\"";
  ApoCharmmErrorCode expectedCode = ApoCharmmErrorCode::Runtime;
  std::string expectedMessage;
  SECTION("UnsupportedDynamics") {
    contents.replace(10, 6, "    -1");
    expectedCode = ApoCharmmErrorCode::NotImplemented;
    expectedMessage =
        "Only ordinary leapfrog (LDYNA=1) restarts are supported in " +
        fileContext;
  }
  SECTION("MissingNativeFooter") {
    const std::size_t footer = contents.find("\n !APOCHARMM RNG STATE\n");
    REQUIRE(footer != std::string::npos);
    contents.erase(footer);
    expectedMessage =
        "APOCHARMM restart is missing its RNG footer in " + fileContext;
  }
  SECTION("TruncatedCoordinateField") {
    const std::size_t section = contents.find(" !XOLD, YOLD, ZOLD\n");
    REQUIRE(section != std::string::npos);
    const std::size_t first = contents.find('\n', section) + 1;
    const std::size_t last = contents.find('\n', first);
    REQUIRE(last != std::string::npos);
    contents.erase(first + 21, last - first - 21);
    expectedMessage = "Field \"XOLD,YOLD,ZOLD\" is truncated in " + fileContext;
  }
  apo_test::WriteTextFile(path, contents);
  auto destinationContext = CreateContext(true);
  auto destination = CreateLangevinThermostatIntegrator(destinationContext);
  destination->propagate(1);
  const auto coordinates = apo_test::CopyToHost<double4>(
      destinationContext->getCoordinatesChargesDP());
  const auto velocities = apo_test::CopyToHost<double4>(
      destinationContext->getVelocitiesInverseMasses());
  const auto box = destinationContext->getBoxDimensions();
  const auto rng = destination->getRngStates();
  const auto seed = destination->getThermostatRngSeed();
  const auto steps = destination->getTotNumSteps();
  apo_test::CheckApoCharmmError(
      [&]() { destination->initializeFromRestartFile(path); }, expectedCode,
      expectedMessage);
  apo_test::CheckVectorsEqual<double4>(
      "Coordinates/charges",
      apo_test::CopyToHost<double4>(
          destinationContext->getCoordinatesChargesDP()),
      coordinates);
  apo_test::CheckVectorsEqual<double4>(
      "Velocities/inverse masses",
      apo_test::CopyToHost<double4>(
          destinationContext->getVelocitiesInverseMasses()),
      velocities);
  CHECK(destinationContext->getBoxDimensions() == box);
  CHECK(destination->getRngStates() == rng);
  CHECK(destination->getThermostatRngSeed() == seed);
  CHECK(destination->getTotNumSteps() == steps);
  apo_test::RemoveIfExists(path);
}

namespace {

std::shared_ptr<CudaIntegrator> CreateCrystalRestartTestIntegrator(
    const std::shared_ptr<CharmmContext> &context, const int kind) {
  if (kind == 0)
    return CreateNoseHooverIntegrator(context);
  if (kind == 1)
    return CreateLangevinThermostatIntegrator(context);
  return CreateLangevinPistonIntegrator(context);
}

} // namespace

TEST_CASE("RestartSubscriberRequiresExplicitCrystalBeforeReplacingFile") {
  const std::filesystem::path path = "tmpRestartRequiresCrystal.rst";
  auto context = CreateContext(true, CRYSTAL::UNSET);
  auto integrator = CreateLangevinThermostatIntegrator(context);
  RestartSubscriber writer(path, REPORT_FREQUENCY);
  writer.setCharmmContext(context);
  writer.setIntegrator(integrator);
  // Construction itself can truncate the path; this checks update's validation
  // boundary and does not claim that construction preserves an existing file.
  const std::string previous = "previous usable restart marker\n";
  apo_test::WriteTextFile(path, previous);
  apo_test::CheckApoCharmmError(
      [&]() { writer.update(); }, ApoCharmmErrorCode::NotInitialized,
      "Crystal type must be configured before this operation");
  CHECK(apo_test::ReadTextFile(path) == previous);
  CHECK(context->getCrystalType() == CRYSTAL::UNSET);
  apo_test::RemoveIfExists(path);
}

TEST_CASE("RestartSubscriberPreservesEqualEdgedOrthorhombicIdentity") {
  for (const int kind : {0, 1, 2}) {
    INFO("integrator kind=" << kind);
    const std::filesystem::path path =
        "tmpRestartExplicitOrth" + std::to_string(kind) + ".rst";
    auto sourceContext = CreateContext(true, CRYSTAL::ORTHORHOMBIC);
    auto source = CreateCrystalRestartTestIntegrator(sourceContext, kind);
    // Save the initialized equal-edge cell before any anisotropic piston step.
    WriteRestartFile(path, sourceContext, source);
    const auto snapshot =
        apo::ReadRestartFile(path, sourceContext->getNumAtoms());
    CHECK(snapshot.crystal == CRYSTAL::ORTHORHOMBIC);
    CHECK(snapshot.box[0] == snapshot.box[1]);
    CHECK(snapshot.box[1] == snapshot.box[2]);
    std::ifstream input(path);
    std::string header;
    REQUIRE(static_cast<bool>(std::getline(input, header)));
    REQUIRE(header.size() >= 22);
    CHECK(header.substr(18, 4) == "ORTH");
    input.close();

    auto destinationContext = CreateContext(false, CRYSTAL::ORTHORHOMBIC);
    auto destination =
        CreateCrystalRestartTestIntegrator(destinationContext, kind);
    destination->initializeFromRestartFile(path);
    CHECK(destinationContext->getCrystalType() == CRYSTAL::ORTHORHOMBIC);
    CheckCharmmContextStateMatches(sourceContext, destinationContext);
    CHECK(destination->getTotNumSteps() == source->getTotNumSteps());
    CHECK(destination->getNumSteps() == source->getNumSteps());
    if (kind == 2) {
      const auto piston =
          std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(destination);
      REQUIRE(piston != nullptr);
      CHECK(piston->getCrystalType() == CRYSTAL::ORTHORHOMBIC);
      CHECK(piston->getLangevinPistonDeltaPosition().size() == 3);
    }
    apo_test::RemoveIfExists(path);
  }
}

TEST_CASE("RestartCrystalMismatchIsRejectedBeforeStateMutation") {
  for (const int kind : {0, 1, 2}) {
    INFO("integrator kind=" << kind);
    const std::filesystem::path path =
        "tmpRestartCrystalMismatch" + std::to_string(kind) + ".rst";
    auto sourceContext = CreateContext(true, CRYSTAL::ORTHORHOMBIC);
    auto source = CreateCrystalRestartTestIntegrator(sourceContext, kind);
    WriteRestartFile(path, sourceContext, source);

    auto context = CreateContext(true, CRYSTAL::CUBIC);
    auto destination = CreateCrystalRestartTestIntegrator(context, kind);
    destination->propagate(1);
    const auto coordinates =
        apo_test::CopyToHost<double4>(context->getCoordinatesChargesDP());
    const auto velocities =
        apo_test::CopyToHost<double4>(context->getVelocitiesInverseMasses());
    const auto deltas =
        apo_test::CopyToHost<double4>(destination->getCoordsDeltaPrevious());
    const auto box = context->getBoxDimensions();
    const auto completed = destination->getTotNumSteps();
    const auto requested = destination->getNumSteps();
    const auto thermostat =
        std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(
            destination);
    const auto piston =
        std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(destination);
    const std::string rng = thermostat ? thermostat->getRngStates()
                            : piston   ? piston->getRngStates()
                                       : std::string();
    apo_test::CheckApoCharmmError(
        [&]() { destination->initializeFromRestartFile(path); },
        ApoCharmmErrorCode::InvalidArgument,
        "Restart crystal type does not match the configured CharmmContext");
    apo_test::CheckVectorsEqual<double4>(
        "Coordinates/charges",
        apo_test::CopyToHost<double4>(context->getCoordinatesChargesDP()),
        coordinates);
    apo_test::CheckVectorsEqual<double4>(
        "Velocities/inverse masses",
        apo_test::CopyToHost<double4>(context->getVelocitiesInverseMasses()),
        velocities);
    apo_test::CheckVectorsEqual<double4>(
        "Coordinate delta previous",
        apo_test::CopyToHost<double4>(destination->getCoordsDeltaPrevious()),
        deltas);
    CHECK(context->getBoxDimensions() == box);
    CHECK(context->getCrystalType() == CRYSTAL::CUBIC);
    CHECK(destination->getTotNumSteps() == completed);
    CHECK(destination->getNumSteps() == requested);
    if (thermostat)
      CHECK(thermostat->getRngStates() == rng);
    if (piston)
      CHECK(piston->getRngStates() == rng);
    apo_test::RemoveIfExists(path);
  }
}

TEST_CASE("FixedCellRestartReaderRequiresExplicitCrystal") {
  for (const int kind : {0, 1}) {
    INFO("integrator kind=" << kind);
    const std::filesystem::path path =
        "tmpRestartUnsetCrystal" + std::to_string(kind) + ".rst";
    auto sourceContext = CreateContext(true, CRYSTAL::CUBIC);
    auto source = CreateCrystalRestartTestIntegrator(sourceContext, kind);
    WriteRestartFile(path, sourceContext, source);
    auto context = CreateContext(true, CRYSTAL::UNSET);
    auto destination = CreateCrystalRestartTestIntegrator(context, kind);
    const auto completed = destination->getTotNumSteps();
    const auto requested = destination->getNumSteps();
    apo_test::CheckApoCharmmError(
        [&]() { destination->initializeFromRestartFile(path); },
        ApoCharmmErrorCode::NotInitialized,
        "Crystal type must be configured before this operation");
    CHECK(context->getCrystalType() == CRYSTAL::UNSET);
    CHECK(destination->getTotNumSteps() == completed);
    CHECK(destination->getNumSteps() == requested);
    apo_test::RemoveIfExists(path);
  }
}

TEST_CASE("RestartSubscriberLangevinThermostatTrajectoryEquivalence") {
  const std::string fileName = "tmpRestartLangevinThermostat.rst";
  apo_test::RemoveIfExists(fileName);

  auto ctx1 = CreateContext(true);
  auto ctx2 = CreateContext(true);
  auto ctx3 = CreateContext(false);

  auto integrator1 = CreateLangevinThermostatIntegrator(ctx1);
  auto integrator2 = CreateLangevinThermostatIntegrator(ctx2);
  auto integrator3 = CreateLangevinThermostatIntegrator(ctx3);

  integrator1->propagate(NUM_STEPS + 1);

  integrator2->propagate(NUM_STEPS);
  WriteRestartFile(fileName, ctx2, integrator2);
  CheckRestartFileContainsRequiredSections(fileName);

  integrator3->initializeFromRestartFile(fileName);

  CheckCharmmContextStateMatches(ctx2, ctx3);

  CHECK(integrator3->getTotNumSteps() ==
        static_cast<unsigned long long int>(NUM_STEPS));
  CHECK(integrator3->getNumSteps() == NUM_STEPS);
  CHECK(integrator3->getCurrentPropagatedStep() == NUM_STEPS);

  // The CHARMM !X, Y, Z restart section stores the previous coordinate
  // displacement. The current displacement is not serialized.
  apo_test::CheckVectorsClose<double4>(
      "Loaded coordinates delta previous",
      apo_test::CopyToHost<double4>(integrator2->getCoordsDeltaPrevious()),
      apo_test::CopyToHost<double4>(integrator3->getCoordsDeltaPrevious()),
      TOLERANCE);

  CHECK(integrator2->getThermostatRngSeed() ==
        integrator3->getThermostatRngSeed());
  CHECK(integrator2->getRngSequencePos() == integrator3->getRngSequencePos());
  CHECK(integrator2->getRngStates() == integrator3->getRngStates());

  integrator2->propagate(1);
  integrator3->propagate(1);

  CheckCharmmContextStateMatches(ctx1, ctx2);
  CheckCharmmContextStateMatches(ctx1, ctx3);
  CheckCharmmContextStateMatches(ctx2, ctx3);

  CheckLangevinThermostatStateMatches(*integrator1, *integrator2);
  CheckLangevinThermostatStateMatches(*integrator1, *integrator3);
  CheckLangevinThermostatStateMatches(*integrator2, *integrator3);

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("RestartSubscriberLangevinPistonTrajectoryEquivalence") {
  const std::string fileName = "tmpRestartLangevinPiston.rst";
  apo_test::RemoveIfExists(fileName);

  auto ctx1 = CreateContext(true);
  auto ctx2 = CreateContext(true);
  auto ctx3 = CreateContext(false);

  auto integrator1 = CreateLangevinPistonIntegrator(ctx1);
  auto integrator2 = CreateLangevinPistonIntegrator(ctx2);
  auto integrator3 = CreateLangevinPistonIntegrator(ctx3);

  integrator1->propagate(NUM_STEPS + 1);

  integrator2->propagate(NUM_STEPS);
  WriteRestartFile(fileName, ctx2, integrator2);
  CheckRestartFileContainsRequiredSections(fileName);
  integrator2->propagate(1);

  integrator3->initializeFromRestartFile(fileName);
  integrator3->propagate(1);

  CheckCharmmContextStateMatches(ctx1, ctx2);
  CheckCharmmContextStateMatches(ctx1, ctx3);
  CheckCharmmContextStateMatches(ctx2, ctx3);

  CheckLangevinPistonStateMatches(*integrator1, *integrator2);
  CheckLangevinPistonStateMatches(*integrator1, *integrator3);
  CheckLangevinPistonStateMatches(*integrator2, *integrator3);

  apo_test::RemoveIfExists(fileName);
}
