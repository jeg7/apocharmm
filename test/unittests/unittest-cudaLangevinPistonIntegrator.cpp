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
#include "CudaLangevinPistonIntegrator.h"
#include "CurandStateString.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int RANDOM_SEED = 314159;
constexpr int NUM_STEPS = 3;
constexpr double TIME_STEP = 0.001;
constexpr double TEMPERATURE = 300.0;
constexpr double REFERENCE_TEMPERATURE = 315.0;
constexpr double NOSE_HOOVER_PISTON_MASS = 500.0;
constexpr double LANGEVIN_PISTON_MASS = 5.0;
constexpr double LANGEVIN_PISTON_FRICTION = 5.0;
constexpr std::uint64_t LANGEVIN_PISTON_SEED = 987654321ULL;
constexpr int MAX_PREDICTOR_CORRECTOR_ITERATIONS = 4;
constexpr double DETERMINISTIC_TOLERANCE = 0.0;
constexpr double FLOAT_TOLERANCE = 1.0e-10;

const std::vector<double> BOX_DIMENSIONS = {50.0, 50.0, 50.0};
const std::vector<double> REFERENCE_PRESSURE_TENSOR = {1.0, 0.0, 0.0, 0.0, 1.0,
                                                       0.0, 0.0, 0.0, 1.0};

class PistonLifecycleProbe : public CudaLangevinPistonIntegrator {
public:
  using CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator;
  using CudaLangevinPistonIntegrator::m_LangevinPistonDegreesOfFreedom;
  using CudaLangevinPistonIntegrator::m_Pgamma;
  using CudaLangevinPistonIntegrator::m_ReferenceTemperature;
  using CudaLangevinPistonIntegrator::m_RngSequencePos;
  using CudaLangevinPistonIntegrator::m_Seed;
};

std::shared_ptr<CharmmContext> CreatePistonContext(const CRYSTAL crystalType) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCrystalType(crystalType);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->setRandomSeed(RANDOM_SEED);
  ctx->assignVelocitiesAtTemperature(TEMPERATURE);

  return ctx;
}

void ConfigureIntegrator(
    std::shared_ptr<CudaLangevinPistonIntegrator> integrator) {
  integrator->useNoseHooverThermostat(true);
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator->setMaxPredictorCorrectorIterations(
      MAX_PREDICTOR_CORRECTOR_ITERATIONS);
  integrator->useOldTemperature(false);

  integrator->setReferencePressure(REFERENCE_PRESSURE_TENSOR);
  integrator->setConstantSurfaceTension(false);
  integrator->setLangevinPistonMass({LANGEVIN_PISTON_MASS});
  integrator->setLangevinPistonFrictionSeed(LANGEVIN_PISTON_SEED);
  integrator->setRngSequencePos(0ULL);
  integrator->setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION);

  return;
}

void CheckIntegratorStateMatches(
    std::shared_ptr<CudaLangevinPistonIntegrator> left,
    std::shared_ptr<CudaLangevinPistonIntegrator> right,
    const double tolerance) {
  CHECK(left->getReferenceTemperature() ==
        Approx(right->getReferenceTemperature()).margin(tolerance));
  CHECK(left->usingOldTemperature() == right->usingOldTemperature());
  CHECK(left->getMaxPredictorCorrectorIterations() ==
        right->getMaxPredictorCorrectorIterations());
  CHECK(left->getCrystalType() == right->getCrystalType());
  CHECK(left->getLangevinPistonFrictionSeed() ==
        right->getLangevinPistonFrictionSeed());
  CHECK(left->getRngSequencePos() == right->getRngSequencePos());

  CHECK(apo_test::CopyScalarToHost<double>(left->getNoseHooverPistonMass()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right->getNoseHooverPistonMass()))
            .margin(tolerance));
  CHECK(
      apo_test::CopyScalarToHost<double>(left->getNoseHooverPistonVelocity()) ==
      Approx(apo_test::CopyScalarToHost<double>(
                 right->getNoseHooverPistonVelocity()))
          .margin(tolerance));
  CHECK(apo_test::CopyScalarToHost<double>(
            left->getNoseHooverPistonVelocityPrevious()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right->getNoseHooverPistonVelocityPrevious()))
            .margin(tolerance));
  CHECK(apo_test::CopyScalarToHost<double>(left->getNoseHooverPistonForce()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right->getNoseHooverPistonForce()))
            .margin(tolerance));
  CHECK(apo_test::CopyScalarToHost<double>(
            left->getNoseHooverPistonForcePrevious()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right->getNoseHooverPistonForcePrevious()))
            .margin(tolerance));

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getReferencePressureTensor()),
      apo_test::CopyToHost<double>(right->getReferencePressureTensor()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getLangevinPistonMass()),
      apo_test::CopyToHost<double>(right->getLangevinPistonMass()), tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getLangevinPistonOnStepPosition()),
      apo_test::CopyToHost<double>(right->getLangevinPistonOnStepPosition()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getLangevinPistonHalfStepPosition()),
      apo_test::CopyToHost<double>(right->getLangevinPistonHalfStepPosition()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getLangevinPistonOnStepVelocity()),
      apo_test::CopyToHost<double>(right->getLangevinPistonOnStepVelocity()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getLangevinPistonHalfStepVelocity()),
      apo_test::CopyToHost<double>(right->getLangevinPistonHalfStepVelocity()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getLangevinPistonDeltaPosition()),
      apo_test::CopyToHost<double>(right->getLangevinPistonDeltaPosition()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(
          left->getLangevinPistonDeltaPositionPrevious()),
      apo_test::CopyToHost<double>(
          right->getLangevinPistonDeltaPositionPrevious()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(
          left->getLangevinPistonDeltaPositionPredicted()),
      apo_test::CopyToHost<double>(
          right->getLangevinPistonDeltaPositionPredicted()),
      tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getLangevinPistonDeltaPressure()),
      apo_test::CopyToHost<double>(right->getLangevinPistonDeltaPressure()),
      tolerance);

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getKineticEnergy()),
      apo_test::CopyToHost<double>(right->getKineticEnergy()), tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getAverageTemperature()),
      apo_test::CopyToHost<double>(right->getAverageTemperature()), tolerance);

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getInstantaneousPressureTensor()),
      apo_test::CopyToHost<double>(right->getInstantaneousPressureTensor()),
      tolerance);
  CHECK(apo_test::CopyScalarToHost<double>(
            left->getInstantaneousPressureScalar()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right->getInstantaneousPressureScalar()))
            .margin(tolerance));
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getAveragePressureTensor()),
      apo_test::CopyToHost<double>(right->getAveragePressureTensor()),
      tolerance);
  CHECK(apo_test::CopyScalarToHost<double>(left->getAveragePressureScalar()) ==
        Approx(apo_test::CopyScalarToHost<double>(
                   right->getAveragePressureScalar()))
            .margin(tolerance));

  return;
}

} // namespace

TEST_CASE("CudaLangevinPistonIntegratorConstructorDefaults") {
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);

  CHECK(integrator->getTimeStep() == Approx(TIME_STEP));
  CHECK(integrator->getReferenceTemperature() == Approx(300.0));
  CHECK(integrator->usingOldTemperature() == false);
  CHECK(integrator->getMaxPredictorCorrectorIterations() == 3);
  CHECK(integrator->getCrystalType() == CRYSTAL::UNSET);
  CHECK(integrator->getRngSequencePos() == 0ULL);

  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonMass()) == Approx(-9999.9999));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonVelocity()) == Approx(0.0));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonVelocityPrevious()) == Approx(0.0));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonForce()) == Approx(0.0));
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonForcePrevious()) == Approx(0.0));

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost(integrator->getReferencePressureTensor()),
      REFERENCE_PRESSURE_TENSOR, 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost(integrator->getKineticEnergy()),
      std::vector<double>{0.0, 0.0}, 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost(integrator->getAverageTemperature()),
      std::vector<double>{0.0, 0.0}, 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost(integrator->getAveragePressureTensor()),
      std::vector<double>(9, 0.0), 0.0);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getAveragePressureScalar()) == Approx(0.0).margin(0.0));

  CHECK(integrator->getLangevinPistonMass().size() == 0);
}

TEST_CASE("CudaLangevinPistonIntegratorSettersAndCrystalType") {
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);

  SECTION("RejectsPressureAndPistonSetterBeforeCrystalType") {
    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonMass({1.0}); },
        ApoCharmmErrorCode::NotInitialized,
        "Attach a CharmmContext with a crystal type before setting the "
        "Langevin piston mass");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonFriction(1.0); },
        ApoCharmmErrorCode::NotInitialized,
        "Attach a CharmmContext with a crystal type before setting the "
        "Langevin piston friction");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setReferencePressure({1.0, 0.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference pressure tensor must contain exactly 9 elements; observed "
        "2");
  }

  SECTION("CubicCrystal") {
    integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
    integrator->setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
    integrator->setNoseHooverPistonVelocity(0.125);
    integrator->setNoseHooverPistonVelocityPrevious(-0.25);
    integrator->setNoseHooverPistonForce(0.5);
    integrator->setNoseHooverPistonForcePrevious(-0.75);
    integrator->setMaxPredictorCorrectorIterations(
        MAX_PREDICTOR_CORRECTOR_ITERATIONS);
    integrator->useOldTemperature(true);
    integrator->setReferencePressure(REFERENCE_PRESSURE_TENSOR);
    integrator->setConstantSurfaceTension(false);
    integrator->setCharmmContext(CreatePistonContext(CRYSTAL::CUBIC));
    integrator->setLangevinPistonMass({LANGEVIN_PISTON_MASS});
    integrator->setLangevinPistonFrictionSeed(LANGEVIN_PISTON_SEED);
    integrator->setRngSequencePos(4ULL);
    integrator->setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION);

    CHECK(integrator->getReferenceTemperature() ==
          Approx(REFERENCE_TEMPERATURE));
    CHECK(integrator->usingOldTemperature() == true);
    CHECK(integrator->getMaxPredictorCorrectorIterations() ==
          MAX_PREDICTOR_CORRECTOR_ITERATIONS);
    CHECK(integrator->getCrystalType() == CRYSTAL::CUBIC);
    CHECK(integrator->getLangevinPistonFrictionSeed() == LANGEVIN_PISTON_SEED);
    CHECK(integrator->getRngSequencePos() == 4ULL);

    CHECK(apo_test::CopyScalarToHost<double>(
              integrator->getNoseHooverPistonMass()) ==
          Approx(NOSE_HOOVER_PISTON_MASS));
    apo_test::CheckVectorsClose<double>(
        apo_test::CopyToHost<double>(integrator->getReferencePressureTensor()),
        REFERENCE_PRESSURE_TENSOR, 0.0);
    apo_test::CheckVectorsClose<double>(
        apo_test::CopyToHost<double>(integrator->getLangevinPistonMass()),
        std::vector<double>{LANGEVIN_PISTON_MASS}, 0.0);

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonMass({1.0, 2.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Langevin piston mass must contain 1 element(s); observed 2");
  }

  SECTION("TetragonalCrystal") {
    integrator->setCharmmContext(CreatePistonContext(CRYSTAL::TETRAGONAL));
    integrator->setLangevinPistonMass({1.0, 2.0});

    CHECK(integrator->getCrystalType() == CRYSTAL::TETRAGONAL);
    apo_test::CheckVectorsClose<double>(
        apo_test::CopyToHost<double>(integrator->getLangevinPistonMass()),
        std::vector<double>{1.0, 2.0}, 0.0);

    CHECK(integrator->getLangevinPistonOnStepPosition().size() == 2);
    CHECK(integrator->getLangevinPistonDeltaPressure().size() == 2);
  }

  SECTION("OrthorhombicCrystal") {
    integrator->setCharmmContext(CreatePistonContext(CRYSTAL::ORTHORHOMBIC));
    integrator->setLangevinPistonMass({1.0, 2.0, 3.0});

    CHECK(integrator->getCrystalType() == CRYSTAL::ORTHORHOMBIC);
    apo_test::CheckVectorsClose<double>(
        apo_test::CopyToHost<double>(integrator->getLangevinPistonMass()),
        std::vector<double>{1.0, 2.0, 3.0}, 0.0);

    CHECK(integrator->getLangevinPistonOnStepPosition().size() == 3);
    CHECK(integrator->getLangevinPistonDeltaPressure().size() == 3);
  }
}

TEST_CASE("CudaLangevinPistonIntegratorRejectsInvalidValues") {
  const double infinity = std::numeric_limits<double>::infinity();
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);

  SECTION("ScalarSetters") {
    apo_test::CheckApoCharmmError(
        [&]() { integrator->setReferenceTemperature(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference temperature must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setReferenceTemperature(-1.0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference temperature must be non-negative; observed -1.000000");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setNoseHooverPistonMass(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Nose-Hoover piston mass must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setNoseHooverPistonMass(-1.0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Nose-Hoover piston mass must be non-negative; observed -1.000000");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setNoseHooverPistonVelocity(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Nose-Hoover piston velocity must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setNoseHooverPistonVelocityPrevious(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Previous Nose-Hoover piston velocity must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setNoseHooverPistonForce(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Nose-Hoover piston force must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setNoseHooverPistonForcePrevious(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Previous Nose-Hoover piston force must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setMaxPredictorCorrectorIterations(0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Maximum predictor-corrector iterations must be positive; observed 0");
  }

  SECTION("PressureTensor") {
    apo_test::CheckApoCharmmError(
        [&]() { integrator->setReferencePressure({1.0, 0.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference pressure tensor must contain exactly 9 elements; observed "
        "2");

    std::vector<double> pressure(9, 0.0);
    pressure[4] = infinity;

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setReferencePressure(pressure); },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference pressure tensor values must be finite; non-finite value at "
        "index 4");
  }

  SECTION("PistonMassAndFriction") {
    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonMass({1.0}); },
        ApoCharmmErrorCode::NotInitialized,
        "Attach a CharmmContext with a crystal type before setting the "
        "Langevin piston mass");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonFriction(1.0); },
        ApoCharmmErrorCode::NotInitialized,
        "Attach a CharmmContext with a crystal type before setting the "
        "Langevin piston friction");

    integrator->setCharmmContext(CreatePistonContext(CRYSTAL::CUBIC));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonMass({1.0, 2.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Langevin piston mass must contain 1 element(s); observed 2");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonMass({infinity}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Langevin piston mass values must be finite; non-finite value at index "
        "0");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonMass({-1.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Langevin piston mass values must be non-negative; observed -1.000000 "
        "at index 0");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonFriction(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Langevin piston friction must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setLangevinPistonFriction(-1.0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Langevin piston friction must be non-negative; observed -1.000000");
  }

  SECTION("RngStates") {
    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates("INVALID"); },
        ApoCharmmErrorCode::NotInitialized,
        "Attach a CharmmContext with a crystal type before setting RNG states");

    apo_test::CheckApoCharmmError(
        [&]() { static_cast<void>(integrator->getRngStates()); },
        ApoCharmmErrorCode::NotInitialized,
        "Attach a CharmmContext with a crystal type before retrieving RNG "
        "states");

    integrator->setCharmmContext(CreatePistonContext(CRYSTAL::CUBIC));

    CHECK_NOTHROW(integrator->getRngStates());

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates("INVALID"); },
        ApoCharmmErrorCode::InvalidArgument,
        "Curand state string has invalid prefix");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates("APO_PHILOX_V1:0"); },
        ApoCharmmErrorCode::InvalidArgument,
        "Curand state string is missing sequence position");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates("APO_PHILOX_V1:BAD:1:1:"); },
        ApoCharmmErrorCode::InvalidArgument,
        "Curand state string has invalid sequence position");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates("APO_PHILOX_V1:0:BAD:1:"); },
        ApoCharmmErrorCode::InvalidArgument,
        "Curand state string has invalid state count");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates("APO_PHILOX_V1:0:1:BAD:"); },
        ApoCharmmErrorCode::InvalidArgument,
        "Curand state string has invalid state size");

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates("APO_PHILOX_V1:0:0:1:"); },
        ApoCharmmErrorCode::Runtime,
        "Curand state string was written with a different curand state size");

    const std::string emptyHex =
        "APO_PHILOX_V1:0:1:" +
        std::to_string(sizeof(curandStatePhilox4_32_10_t)) + ":";

    apo_test::CheckApoCharmmError([&]() { integrator->setRngStates(emptyHex); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Curand state string has invalid hex length");

    const std::vector<curandStatePhilox4_32_10_t> oneState(1);
    std::string invalidHex = apo::curand_states_to_string(0ULL, oneState);
    REQUIRE(invalidHex.empty() == false);
    invalidHex.back() = 'G';

    unsigned long long int parsedPosition = 42ULL;
    std::vector<curandStatePhilox4_32_10_t> parsedStates(2);

    apo_test::CheckApoCharmmError(
        [&]() {
          apo::curand_states_from_string(parsedPosition, parsedStates,
                                         invalidHex);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Invalid hex digit in curand state string");

    CHECK(parsedPosition == 42ULL);
    CHECK(parsedStates.size() == 2);

    const std::vector<curandStatePhilox4_32_10_t> twoStates(2);
    const std::string wrongCount =
        apo::curand_states_to_string(0ULL, twoStates);

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates(wrongCount); },
        ApoCharmmErrorCode::InvalidArgument,
        "RNG state count must match Langevin piston degrees of freedom; "
        "expected 1, observed 2");

    const std::size_t stateSize = sizeof(curandStatePhilox4_32_10_t);
    const std::size_t maximumSize = std::numeric_limits<std::size_t>::max();
    const std::size_t overflowingCount = maximumSize / stateSize + 1;

    const std::string byteCountOverflow =
        "APO_PHILOX_V1:0:" + std::to_string(overflowingCount) + ":" +
        std::to_string(stateSize) + ":";

    apo_test::CheckApoCharmmError(
        [&]() { integrator->setRngStates(byteCountOverflow); },
        ApoCharmmErrorCode::InvalidArgument,
        "Curand state string byte count is out of range");
  }

  SECTION("ContextDependentGetters") {
    apo_test::CheckApoCharmmError(
        [&]() { static_cast<void>(integrator->getInstantaneousTemperature()); },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmContext must be set before computing instantaneous temperature");

    apo_test::CheckApoCharmmError(
        [&]() {
          static_cast<void>(integrator->getInstantaneousSurfaceTension());
        },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmContext must be set before computing instantaneous surface "
        "tension");
  }
}

TEST_CASE("CudaLangevinPistonIntegratorResetAverages") {
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);

  integrator->getAverageTemperature().set(42.0);
  integrator->getAveragePressureTensor().set(2.0);
  integrator->getAveragePressureScalar().set(3.0);

  integrator->resetAverages();

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator->getAverageTemperature()),
      std::vector<double>{0.0, 0.0}, 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator->getAveragePressureTensor()),
      std::vector<double>(9, 0.0), 0.0);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getAveragePressureScalar()) == Approx(0.0).margin(0.0));
}

TEST_CASE("CudaLangevinPistonIntegratorRequireCrystalTypeBeforeContext") {
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

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);

  apo_test::CheckApoCharmmError(
      [&]() { integrator->setCharmmContext(ctx); },
      ApoCharmmErrorCode::NotInitialized,
      "Crystal type must be configured before this operation");

  CHECK(integrator->getCharmmContext() == nullptr);
}

TEST_CASE("CudaLangevinPistonIntegratorContextInitialization") {
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

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator->setCharmmContext(ctx);
  ConfigureIntegrator(integrator);

  CHECK(integrator->getCharmmContext() == ctx);
  CHECK(integrator->getCoordsDelta().size() == 2);
  CHECK(integrator->getCoordsDeltaPrevious().size() == 2);
  CHECK(integrator->getLangevinPistonMass().size() == 1);
  CHECK(integrator->getRngSequencePos() == 0ULL);

  apo_test::CheckFiniteTemperature(ctx->computeTemperature());

  apo_test::CheckApoCharmmError(
      [&]() { integrator->setCharmmContext(ctx); },
      ApoCharmmErrorCode::InvalidArgument,
      "A CharmmContext object was already set for this CudaIntegrator.");
}

TEST_CASE("CudaLangevinPistonIntegratorAutoPistonMass") {
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

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator->setLangevinPistonFrictionSeed(LANGEVIN_PISTON_SEED);
  integrator->setRngSequencePos(0ULL);
  integrator->setCharmmContext(ctx);
  integrator->setLangevinPistonFriction(0.0);

  constexpr double expectedMass = 0.02 * (22.9898 + 35.45);

  CHECK(
      apo_test::CopyScalarToHost<double>(integrator->getLangevinPistonMass()) ==
      Approx(expectedMass).margin(1.0e-4));
}

TEST_CASE("CudaLangevinPistonIntegratorShortPropagation") {
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

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator->setCharmmContext(ctx);
  ConfigureIntegrator(integrator);
  integrator->resetAverages();

  const std::vector<double4> initialCoordinates =
      apo_test::CopyToHost<double4>(ctx->getCoordinatesChargesDP());
  const std::vector<double> initialBoxDimensions = ctx->getBoxDimensions();

  integrator->propagate(NUM_STEPS);

  const std::vector<double4> finalCoordinates =
      apo_test::CopyToHost<double4>(ctx->getCoordinatesChargesDP());
  const std::vector<double> finalBoxDimensions = ctx->getBoxDimensions();

  CHECK(integrator->getCurrentPropagatedStep() == NUM_STEPS);
  CHECK(integrator->getTotNumSteps() ==
        static_cast<unsigned long long int>(NUM_STEPS));
  CHECK(integrator->getNumSteps() == NUM_STEPS);
  CHECK(
      integrator->getRngSequencePos() ==
      static_cast<unsigned long long int>(MAX_PREDICTOR_CORRECTOR_ITERATIONS) *
          NUM_STEPS);

  apo_test::CheckFiniteTemperature(ctx->computeTemperature());
  apo_test::CheckFiniteTemperature(integrator->getInstantaneousTemperature());

  apo_test::CheckFiniteNonnegativeVector(
      apo_test::CopyToHost<double>(integrator->getKineticEnergy()));
  apo_test::CheckFiniteNonnegativeVector(
      apo_test::CopyToHost<double>(integrator->getAverageTemperature()));
  apo_test::CheckFiniteVector(apo_test::CopyToHost<double>(
      integrator->getInstantaneousPressureTensor()));
  apo_test::CheckFiniteScalar(apo_test::CopyScalarToHost<double>(
      integrator->getInstantaneousPressureScalar()));
  apo_test::CheckFiniteVector(
      apo_test::CopyToHost<double>(integrator->getAveragePressureTensor()));
  apo_test::CheckFiniteScalar(apo_test::CopyScalarToHost<double>(
      integrator->getAveragePressureScalar()));

  REQUIRE(finalCoordinates.size() == initialCoordinates.size());
  CHECK(((finalCoordinates[0].x != initialCoordinates[0].x) ||
         (finalCoordinates[0].y != initialCoordinates[0].y) ||
         (finalCoordinates[0].z != initialCoordinates[0].z)));

  REQUIRE(finalBoxDimensions.size() == initialBoxDimensions.size());
  CHECK(std::isfinite(finalBoxDimensions[0]));
  CHECK(std::isfinite(finalBoxDimensions[1]));
  CHECK(std::isfinite(finalBoxDimensions[2]));
}

TEST_CASE("CudaLangevinPistonIntegratorDeterministicTrajectory") {
  auto prm1 = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                 "toppar_water_ions.str");
  auto prm2 = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                 "toppar_water_ions.str");

  auto psf1 =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto psf2 =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  auto crd1 =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");
  auto crd2 =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx1 = std::make_shared<CharmmContext>(psf1, prm1);
  ctx1->setBoxDimensions(BOX_DIMENSIONS);
  ctx1->setCrystalType(CRYSTAL::CUBIC);
  ctx1->setCoordinates(crd1);
  ctx1->useHolonomicConstraints(false);
  ctx1->setRandomSeed(RANDOM_SEED);
  ctx1->assignVelocitiesAtTemperature(TEMPERATURE);

  auto ctx2 = std::make_shared<CharmmContext>(psf2, prm2);
  ctx2->setBoxDimensions(BOX_DIMENSIONS);
  ctx2->setCrystalType(CRYSTAL::CUBIC);
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

  auto integrator1 = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator1->setCharmmContext(ctx1);
  ConfigureIntegrator(integrator1);

  auto integrator2 = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator2->setCharmmContext(ctx2);
  ConfigureIntegrator(integrator2);

  CheckIntegratorStateMatches(integrator1, integrator2,
                              DETERMINISTIC_TOLERANCE);

  const std::vector<double4> initialCoordinates1 =
      apo_test::CopyToHost<double4>(ctx1->getCoordinatesChargesDP());
  const std::vector<double4> initialCoordinates2 =
      apo_test::CopyToHost<double4>(ctx2->getCoordinatesChargesDP());

  integrator1->propagate(NUM_STEPS);
  integrator2->propagate(NUM_STEPS);

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

  CHECK(integrator1->getCurrentPropagatedStep() == NUM_STEPS);
  CHECK(integrator2->getCurrentPropagatedStep() == NUM_STEPS);
  CHECK(
      integrator1->getRngSequencePos() ==
      static_cast<unsigned long long int>(MAX_PREDICTOR_CORRECTOR_ITERATIONS) *
          NUM_STEPS);
  CHECK(
      integrator2->getRngSequencePos() ==
      static_cast<unsigned long long int>(MAX_PREDICTOR_CORRECTOR_ITERATIONS) *
          NUM_STEPS);

  CHECK(((finalCoordinates1[0].x != initialCoordinates1[0].x) ||
         (finalCoordinates1[0].y != initialCoordinates1[0].y) ||
         (finalCoordinates1[0].z != initialCoordinates1[0].z)));
  CHECK(((finalCoordinates2[0].x != initialCoordinates2[0].x) ||
         (finalCoordinates2[0].y != initialCoordinates2[0].y) ||
         (finalCoordinates2[0].z != initialCoordinates2[0].z)));
}

TEST_CASE("CudaLangevinPistonIntegratorUsesContextCrystalAndPreservesMass") {
  auto ctx = CreatePistonContext(CRYSTAL::ORTHORHOMBIC);
  auto integrator = std::make_shared<PistonLifecycleProbe>(TIME_STEP);
  integrator->setCharmmContext(ctx);
  CHECK(integrator->getCrystalType() == ctx->getCrystalType());
  CHECK(integrator->getCrystalType() == CRYSTAL::ORTHORHOMBIC);
  REQUIRE(integrator->getLangevinPistonMass().size() == 3);
  integrator->setLangevinPistonMass({1.0, 2.0, 3.0});
  integrator->setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION);
  integrator->initialize();
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator->getLangevinPistonMass()),
      std::vector<double>{1.0, 2.0, 3.0}, 0.0);
  CHECK(integrator->m_Pgamma == LANGEVIN_PISTON_FRICTION);
}

TEST_CASE("CudaLangevinPistonIntegratorReinitializesChangedCrystalPhase") {
  auto ctx = CreatePistonContext(CRYSTAL::CUBIC);
  auto integrator = std::make_shared<PistonLifecycleProbe>(TIME_STEP);
  integrator->setCharmmContext(ctx);
  ConfigureIntegrator(integrator);
  integrator->propagate(2);
  integrator->setNoseHooverPistonVelocity(2.0);
  integrator->setNoseHooverPistonVelocityPrevious(3.0);
  integrator->setNoseHooverPistonForce(4.0);
  integrator->setNoseHooverPistonForcePrevious(5.0);
  const auto velocities =
      apo_test::CopyToHost<double4>(ctx->getVelocitiesInverseMasses());
  const auto completedSteps = integrator->getTotNumSteps();
  const auto sequence = integrator->getRngSequencePos();
  const auto seed = integrator->getLangevinPistonFrictionSeed();
  const std::string oldRng = integrator->getRngStates();
  const CudaIntegrator &base = *integrator;
  const CudaLangevinPistonIntegrator &constant = *integrator;

  ctx->setCrystalType(CRYSTAL::ORTHORHOMBIC);
  CHECK(integrator->getCrystalType() == CRYSTAL::ORTHORHOMBIC);
  const auto stale = [&](auto operation) {
    apo_test::CheckApoCharmmError(
        operation, ApoCharmmErrorCode::NotInitialized,
        "Langevin piston state does not match the context crystal type; "
        "reinitialize the integrator");
  };
  stale([&]() { integrator->setLangevinPistonMass({1.0, 2.0, 3.0}); });
  stale([&]() { integrator->setLangevinPistonFriction(9.0); });
  stale([&]() { integrator->setReferenceTemperature(400.0); });
  stale([&]() { integrator->setNoseHooverPistonVelocity(8.0); });
  stale([&]() { integrator->setLangevinPistonFrictionSeed(12); });
  stale([&]() { integrator->setRngSequencePos(99); });
  stale([&]() { integrator->setRngStates(oldRng); });
  stale([&]() { static_cast<void>(integrator->getRngStates()); });
  stale([&]() { static_cast<void>(integrator->getLangevinPistonMass()); });
  stale(
      [&]() { static_cast<void>(constant.getLangevinPistonOnStepVelocity()); });
  stale([&]() {
    static_cast<void>(integrator->getLangevinPistonDeltaPosition());
  });
  stale([&]() { static_cast<void>(base.getCoordsDeltaPrevious()); });
  stale([&]() { static_cast<void>(integrator->getCoordsDelta()); });
  stale([&]() { integrator->initializeFromRestartFile("missing.rst"); });
  stale([&]() { integrator->propagateOneStep(); });
  stale([&]() { integrator->propagate(1); });
  CHECK(integrator->m_Pgamma == LANGEVIN_PISTON_FRICTION);
  CHECK(integrator->m_ReferenceTemperature == REFERENCE_TEMPERATURE);
  CHECK(integrator->m_Seed == seed);
  CHECK(integrator->m_RngSequencePos == sequence);
  CHECK(integrator->getTotNumSteps() == completedSteps);

  integrator->initialize();
  CHECK(integrator->m_LangevinPistonDegreesOfFreedom == 3);
  REQUIRE(integrator->getLangevinPistonMass().size() == 3);
  double expectedMass = 0.0;
  for (const auto &velocity : velocities)
    expectedMass += 0.02 / velocity.w;
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator->getLangevinPistonMass()),
      std::vector<double>(3, expectedMass), FLOAT_TOLERANCE);
  apo_test::CheckVectorsClose<double4>(
      apo_test::CopyToHost<double4>(ctx->getVelocitiesInverseMasses()),
      velocities, 0.0);
  CHECK(integrator->getTotNumSteps() == completedSteps);
  CHECK(integrator->getRngSequencePos() == sequence);
  CHECK(integrator->getLangevinPistonFrictionSeed() == seed);
  CHECK(integrator->m_Pgamma == LANGEVIN_PISTON_FRICTION);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonMass()) == NOSE_HOOVER_PISTON_MASS);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonVelocity()) == 0.0);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonVelocityPrevious()) == 0.0);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonForce()) == 0.0);
  CHECK(apo_test::CopyScalarToHost<double>(
            integrator->getNoseHooverPistonForcePrevious()) == 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(
          integrator->getLangevinPistonDeltaPosition()),
      std::vector<double>(3, 0.0), 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator->getAveragePressureTensor()),
      std::vector<double>(9, 0.0), 0.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator->getAverageTemperature()),
      std::vector<double>(2, 0.0), 0.0);
  unsigned long long parsedSequence = 0;
  std::vector<curandStatePhilox4_32_10_t> parsedStates;
  apo::curand_states_from_string(parsedSequence, parsedStates,
                                 integrator->getRngStates());
  CHECK(parsedSequence == sequence);
  CHECK(parsedStates.size() == 3);
  integrator->setLangevinPistonMass({1.0, 2.0, 3.0});
  CHECK_NOTHROW(integrator->propagate(1));
  CHECK(integrator->getTotNumSteps() == completedSteps + 1);
}

TEST_CASE("CudaLangevinPistonIntegratorRejectsMutatedCrystalBox") {
  auto ctx = CreatePistonContext(CRYSTAL::CUBIC);
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator->setCharmmContext(ctx);
  ctx->getBoxDimensions()[1] = 51.0;
  apo_test::CheckApoCharmmError(
      [&]() { integrator->initialize(); }, ApoCharmmErrorCode::InvalidArgument,
      "Box dimensions are incompatible with the crystal type");
  apo_test::CheckApoCharmmError(
      [&]() { integrator->propagateOneStep(); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimensions are incompatible with the crystal type");
}

TEST_CASE("CudaLangevinPistonIntegratorRestartValidation") {
  const std::filesystem::path missingFilePath = "missing.rst";

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);

  apo_test::CheckApoCharmmError(
      [&]() { integrator->initializeFromRestartFile(missingFilePath); },
      ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before initializing from a restart file");

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

  integrator->setCharmmContext(ctx);
  ConfigureIntegrator(integrator);

  apo_test::CheckApoCharmmError(
      [&]() { integrator->initializeFromRestartFile(missingFilePath); },
      ApoCharmmErrorCode::Runtime,
      "Could not open restart file: " + missingFilePath.string());

  const std::filesystem::path truncatedFilePath =
      "cudaLangevinPistonIntegrator-truncated.rst";
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
      [&]() { integrator->initializeFromRestartFile(truncatedFilePath); },
      ApoCharmmErrorCode::Runtime,
      "Unexpected end of file while reading XTLABC in restart "
      "file \"" +
          truncatedFilePath.string() + "\"");

  apo_test::RemoveIfExists(truncatedFilePath);

  CHECK(integrator->getCrystalType() == CRYSTAL::CUBIC);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator->getLangevinPistonMass()),
      std::vector<double>{LANGEVIN_PISTON_MASS}, 0.0);
}
