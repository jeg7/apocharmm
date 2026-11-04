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
#include "CudaNoseHooverIntegrator.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <filesystem>

namespace {

constexpr int RANDOM_SEED = 314159;
constexpr int NUM_STEPS = 5;
constexpr double TIME_STEP = 0.001;
constexpr double TEMPERATURE = 300.0;
constexpr double REFERENCE_TEMPERATURE = 315.0;
constexpr double NOSE_HOOVER_PISTON_MASS = 500.0;
constexpr double NOSE_HOOVER_PISTON_VELOCITY = 0.125;
constexpr double NOSE_HOOVER_PISTON_VELOCITY_PREVIOUS = -0.25;
constexpr double NOSE_HOOVER_PISTON_FORCE = 0.5;
constexpr double NOSE_HOOVER_PISTON_FORCE_PREVIOUS = -0.75;
constexpr int MAX_PREDICTOR_CORRECTOR_ITERATIONS = 4;
constexpr double DETERMINISTIC_TOLERANCE = 0.0;
constexpr double FINITE_TEMPERATURE_UPPER_BOUND = 1.0e8;

const std::vector<double> BOX_DIMENSIONS = {50.0, 50.0, 50.0};

void CheckIntegratorStateMatches(CudaNoseHooverIntegrator &left,
                                 CudaNoseHooverIntegrator &right,
                                 const double tolerance) {
  CHECK(left.getReferenceTemperature() ==
        Approx(right.getReferenceTemperature()).margin(tolerance));
  CHECK(left.usingOldTemperature() == right.usingOldTemperature());
  CHECK(left.getMaxPredictorCorrectorIterations() ==
        right.getMaxPredictorCorrectorIterations());

  CHECK(apo_test::CopyScalarToHost<double>(left.getNoseHooverPistonMass()) ==
        Approx(
            apo_test::CopyScalarToHost<double>(right.getNoseHooverPistonMass()))
            .margin(tolerance));
  CHECK(
      apo_test::CopyScalarToHost<double>(left.getNoseHooverPistonVelocity()) ==
      Approx(apo_test::CopyScalarToHost<double>(
                 right.getNoseHooverPistonVelocity()))
          .margin(tolerance));
  CHECK(apo_test::CopyScalarToHost<double>(
            left.getNoseHooverPistonVelocityPrevious()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right.getNoseHooverPistonVelocityPrevious()))
            .margin(tolerance));
  CHECK(apo_test::CopyScalarToHost<double>(left.getNoseHooverPistonForce()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right.getNoseHooverPistonForce()))
            .margin(tolerance));
  CHECK(apo_test::CopyScalarToHost<double>(
            left.getNoseHooverPistonForcePrevious()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right.getNoseHooverPistonForcePrevious()))
            .margin(tolerance));

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left.getKineticEnergy()),
      apo_test::CopyToHost<double>(right.getKineticEnergy()), tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left.getAverageTemperature()),
      apo_test::CopyToHost<double>(right.getAverageTemperature()), tolerance);

  return;
}

} // namespace

TEST_CASE("CudaNoseHooverIntegratorConstructorDefaults") {
  CudaNoseHooverIntegrator integrator(TIME_STEP);

  CHECK(integrator.getTimeStep() == Approx(TIME_STEP));
  CHECK(integrator.getReferenceTemperature() == Approx(300.0));
  CHECK(integrator.usingOldTemperature() == false);
  CHECK(integrator.getMaxPredictorCorrectorIterations() == 3);

  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonMass()) == Approx(-9999.9999));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonVelocity()) == Approx(0.0));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonVelocityPrevious()) == Approx(0.0));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonForce()) == Approx(0.0));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonForcePrevious()) == Approx(0.0));

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator.getKineticEnergy()),
      std::vector<double>{0.0, 0.0}, 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator.getAverageTemperature()),
      std::vector<double>{0.0, 0.0}, 0.0);
}

TEST_CASE("CudaNoseHooverIntegratorSettersAndReset") {
  CudaNoseHooverIntegrator integrator(TIME_STEP);

  integrator.setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator.setNoseHooverPistonVelocity(NOSE_HOOVER_PISTON_VELOCITY);
  integrator.setNoseHooverPistonVelocityPrevious(
      NOSE_HOOVER_PISTON_VELOCITY_PREVIOUS);
  integrator.setNoseHooverPistonForce(NOSE_HOOVER_PISTON_FORCE);
  integrator.setNoseHooverPistonForcePrevious(
      NOSE_HOOVER_PISTON_FORCE_PREVIOUS);
  integrator.setMaxPredictorCorrectorIterations(
      MAX_PREDICTOR_CORRECTOR_ITERATIONS);
  integrator.useOldTemperature(true);

  CHECK(integrator.getReferenceTemperature() == Approx(REFERENCE_TEMPERATURE));
  CHECK(integrator.usingOldTemperature() == true);
  CHECK(integrator.getMaxPredictorCorrectorIterations() ==
        MAX_PREDICTOR_CORRECTOR_ITERATIONS);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonMass()) ==
        Approx(NOSE_HOOVER_PISTON_MASS));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonVelocity()) ==
        Approx(NOSE_HOOVER_PISTON_VELOCITY));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonVelocityPrevious()) ==
        Approx(NOSE_HOOVER_PISTON_VELOCITY_PREVIOUS));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonForce()) ==
        Approx(NOSE_HOOVER_PISTON_FORCE));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonForcePrevious()) ==
        Approx(NOSE_HOOVER_PISTON_FORCE_PREVIOUS));

  integrator.getAverageTemperature().setToValue(42.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator.getAverageTemperature()),
      std::vector<double>{42.0, 42.0}, 0.0);

  integrator.resetAverageTemperature();
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator.getAverageTemperature()),
      std::vector<double>{0.0, 0.0}, 0.0);
}

TEST_CASE("CudaNoseHooverIntegratorContextInitialization") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->setRandomSeed(RANDOM_SEED);
  ctx->assignVelocitiesAtTemperature(TEMPERATURE);

  CudaNoseHooverIntegrator integrator(TIME_STEP);
  integrator.setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator.setMaxPredictorCorrectorIterations(
      MAX_PREDICTOR_CORRECTOR_ITERATIONS);
  integrator.useOldTemperature(false);

  integrator.setCharmmContext(ctx);

  CHECK(integrator.getCharmmContext() == ctx);
  CHECK(integrator.getCoordsDelta().size() == 2);
  CHECK(integrator.getCoordsDeltaPrevious().size() == 2);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonMass()) ==
        Approx(NOSE_HOOVER_PISTON_MASS));
  apo_test::CheckFiniteTemperature(ctx->computeTemperature());

  apo_test::CheckApoCharmmError(
      [&]() { integrator.setCharmmContext(ctx); },
      ApoCharmmErrorCode::InvalidArgument,
      "A CharmmContext object was already set for this CudaIntegrator.");
}

TEST_CASE("CudaNoseHooverIntegratorAutoPistonMass") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->setRandomSeed(RANDOM_SEED);
  ctx->assignVelocitiesAtTemperature(TEMPERATURE);

  CudaNoseHooverIntegrator integrator(TIME_STEP);
  integrator.setCharmmContext(ctx);

  const double expectedMass = 0.2 * (22.9898 + 35.45);

  CHECK(apo_test::CopyScalarToHost<double>(
            integrator.getNoseHooverPistonMass()) ==
        Approx(expectedMass).margin(1.0e-4));
}

TEST_CASE("CudaNoseHooverIntegratorShortPropagation") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->setRandomSeed(RANDOM_SEED);
  ctx->assignVelocitiesAtTemperature(TEMPERATURE);

  CudaNoseHooverIntegrator integrator(TIME_STEP);
  integrator.setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator.setMaxPredictorCorrectorIterations(
      MAX_PREDICTOR_CORRECTOR_ITERATIONS);
  integrator.useOldTemperature(false);
  integrator.setCharmmContext(ctx);
  integrator.resetAverageTemperature();

  const std::vector<double4> initialCoordinates =
      apo_test::CopyToHost<double4>(ctx->getCoordinatesChargesDP());

  integrator.propagate(NUM_STEPS);

  const std::vector<double4> finalCoordinates =
      apo_test::CopyToHost<double4>(ctx->getCoordinatesChargesDP());
  const std::vector<double> averageTemperature =
      apo_test::CopyToHost<double>(integrator.getAverageTemperature());
  const std::vector<double> kineticEnergy =
      apo_test::CopyToHost<double>(integrator.getKineticEnergy());

  CHECK(integrator.getCurrentPropagatedStep() == NUM_STEPS);
  CHECK(integrator.getTotNumSteps() ==
        static_cast<unsigned long long>(NUM_STEPS));
  CHECK(integrator.getNumSteps() == NUM_STEPS);

  apo_test::CheckFiniteTemperature(ctx->computeTemperature());
  apo_test::CheckFiniteTemperature(integrator.getInstantaneousTemperature());

  REQUIRE(averageTemperature.size() == 2);
  apo_test::CheckFiniteTemperature(averageTemperature[0]);
  apo_test::CheckFiniteTemperature(averageTemperature[1]);

  REQUIRE(kineticEnergy.size() == 2);
  apo_test::CheckFiniteNonnegativeScalar(kineticEnergy[0]);
  apo_test::CheckFiniteNonnegativeScalar(kineticEnergy[1]);

  REQUIRE(finalCoordinates.size() == initialCoordinates.size());
  CHECK((finalCoordinates[0].x != initialCoordinates[0].x ||
         finalCoordinates[0].y != initialCoordinates[0].y ||
         finalCoordinates[0].z != initialCoordinates[0].z));
}

TEST_CASE("CudaNoseHooverIntegratorDeterministicTrajectory") {
  auto prm1 = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                 "toppar_water_ions.str");
  auto psf1 =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd1 =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx1 = std::make_shared<CharmmContext>(psf1, prm1);
  ctx1->setBoxDimensions(BOX_DIMENSIONS);
  ctx1->setCoordinates(crd1);
  ctx1->useHolonomicConstraints(false);
  ctx1->setRandomSeed(RANDOM_SEED);
  ctx1->assignVelocitiesAtTemperature(TEMPERATURE);

  auto prm2 = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                 "toppar_water_ions.str");
  auto psf2 =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd2 =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx2 = std::make_shared<CharmmContext>(psf2, prm2);
  ctx2->setBoxDimensions(BOX_DIMENSIONS);
  ctx2->setCoordinates(crd2);
  ctx2->useHolonomicConstraints(false);
  ctx2->setRandomSeed(RANDOM_SEED);
  ctx2->assignVelocitiesAtTemperature(TEMPERATURE);

  apo_test::CheckVectorsClose<double4>(
      apo_test::CopyToHost<double4>(ctx1->getCoordinatesChargesDP()),
      apo_test::CopyToHost<double4>(ctx2->getCoordinatesChargesDP()),
      DETERMINISTIC_TOLERANCE);
  apo_test::CheckVectorsClose<double4>(
      apo_test::CopyToHost<double4>(ctx1->getVelocitiesInverseMasses()),
      apo_test::CopyToHost<double4>(ctx2->getVelocitiesInverseMasses()),
      DETERMINISTIC_TOLERANCE);

  CudaNoseHooverIntegrator integrator1(TIME_STEP);
  integrator1.setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator1.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator1.setMaxPredictorCorrectorIterations(
      MAX_PREDICTOR_CORRECTOR_ITERATIONS);
  integrator1.useOldTemperature(false);
  integrator1.setCharmmContext(ctx1);

  CudaNoseHooverIntegrator integrator2(TIME_STEP);
  integrator2.setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator2.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator2.setMaxPredictorCorrectorIterations(
      MAX_PREDICTOR_CORRECTOR_ITERATIONS);
  integrator2.useOldTemperature(false);
  integrator2.setCharmmContext(ctx2);

  CheckIntegratorStateMatches(integrator1, integrator2,
                              DETERMINISTIC_TOLERANCE);

  const std::vector<double4> initialCoordinates1 =
      apo_test::CopyToHost<double4>(ctx1->getCoordinatesChargesDP());
  const std::vector<double4> initialCoordinates2 =
      apo_test::CopyToHost<double4>(ctx2->getCoordinatesChargesDP());

  integrator1.propagate(NUM_STEPS);
  integrator2.propagate(NUM_STEPS);

  const std::vector<double4> finalCoordinates1 =
      apo_test::CopyToHost<double4>(ctx1->getCoordinatesChargesDP());
  const std::vector<double4> finalCoordinates2 =
      apo_test::CopyToHost<double4>(ctx2->getCoordinatesChargesDP());

  apo_test::CheckVectorsClose<double4>(finalCoordinates1, finalCoordinates2,
                                       DETERMINISTIC_TOLERANCE);
  apo_test::CheckVectorsClose<double4>(
      apo_test::CopyToHost<double4>(ctx1->getVelocitiesInverseMasses()),
      apo_test::CopyToHost<double4>(ctx2->getVelocitiesInverseMasses()),
      DETERMINISTIC_TOLERANCE);
  CheckIntegratorStateMatches(integrator1, integrator2,
                              DETERMINISTIC_TOLERANCE);

  CHECK(integrator1.getCurrentPropagatedStep() == NUM_STEPS);
  CHECK(integrator2.getCurrentPropagatedStep() == NUM_STEPS);
  CHECK(integrator1.getTotNumSteps() ==
        static_cast<unsigned long long>(NUM_STEPS));
  CHECK(integrator2.getTotNumSteps() ==
        static_cast<unsigned long long>(NUM_STEPS));

  CHECK((finalCoordinates1[0].x != initialCoordinates1[0].x ||
         finalCoordinates1[0].y != initialCoordinates1[0].y ||
         finalCoordinates1[0].z != initialCoordinates1[0].z));
  CHECK((finalCoordinates2[0].x != initialCoordinates2[0].x ||
         finalCoordinates2[0].y != initialCoordinates2[0].y ||
         finalCoordinates2[0].z != initialCoordinates2[0].z));
}

TEST_CASE("CudaNoseHooverIntegratorRestartValidation") {
  const std::filesystem::path missingFilePath = "missing.rst";

  CudaNoseHooverIntegrator integrator(TIME_STEP);

  bool caughtMissingContext = false;

  try {
    integrator.initializeFromRestartFile(missingFilePath);
  } catch (const ApoCharmmError &error) {
    caughtMissingContext = true;

    CHECK(error.getCode() == ApoCharmmErrorCode::NotInitialized);
    CHECK(error.getMessage() ==
          "CharmmContext must be set before initializing from a restart file");
  }

  CHECK(caughtMissingContext == true);

  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCrystalType(CRYSTAL::CUBIC);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->setRandomSeed(RANDOM_SEED);
  ctx->assignVelocitiesAtTemperature(TEMPERATURE);

  integrator.setCharmmContext(ctx);

  bool caughtMissingFile = false;

  try {
    integrator.initializeFromRestartFile(missingFilePath);
  } catch (const ApoCharmmError &error) {
    caughtMissingFile = true;

    CHECK(error.getCode() == ApoCharmmErrorCode::Runtime);
    CHECK(error.getMessage() ==
          "Could not open restart file: " + missingFilePath.string());
  }

  CHECK(caughtMissingFile == true);

  const std::filesystem::path truncatedFilePath =
      "cudaNoseHooverIntegrator-truncated.rst";
  std::string truncatedHeader(28, ' ');
  truncatedHeader.replace(0, 4, "REST");
  truncatedHeader.replace(4, 6, "    52");
  truncatedHeader.replace(10, 6, "     1");
  truncatedHeader.replace(18, 4, "CUBI");

  apo_test::RemoveIfExists(truncatedFilePath);
  apo_test::WriteTextFile(truncatedFilePath,
                          truncatedHeader +
                              "\n\n       0 !NTITLE followed by title\n"
                              "\n !CRYSTAL PARAMETERS\n");

  apo_test::CheckApoCharmmError(
      [&]() { integrator.initializeFromRestartFile(truncatedFilePath); },
      ApoCharmmErrorCode::Runtime,
      "Unexpected end of file while reading XTLABC in restart "
      "file \"" +
          truncatedFilePath.string() + "\"");

  apo_test::RemoveIfExists(truncatedFilePath);
}
