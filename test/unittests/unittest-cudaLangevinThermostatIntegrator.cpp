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
#include "CudaLangevinThermostatIntegrator.h"
#include "CurandStateString.h"
#include "ForceManager.h"
#include "RestartIO.h"
#include "apo_test_helpers.h"
#include "catch.hpp"
#include "str_utils.h"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

constexpr int RANDOM_SEED = 314159;
constexpr int NUM_STEPS = 5;
constexpr double TIME_STEP = 0.001;
constexpr double TEMPERATURE = 300.0;
constexpr double REFERENCE_TEMPERATURE = 315;
constexpr double THERMOSTAT_FRICTION = 1.0;
constexpr std::uint64_t THERMOSTAT_SEED = 987654321ULL;
constexpr unsigned long long int RNG_SEQUENCE_POS = 8ULL;
constexpr double DETERMINISTIC_TOLERANCE = 0.0;
constexpr double TOLERANCE = 1.0e-10;
constexpr double FINITE_TEMPERATURE_UPPER_BOUND = 1.e8;

const std::vector<double> BOX_DIMENSIONS = {50.0, 50.0, 50.0};

class TestCudaLangevinThermostatIntegrator
    : public CudaLangevinThermostatIntegrator {
public:
  using CudaLangevinThermostatIntegrator::CudaLangevinThermostatIntegrator;
  using CudaLangevinThermostatIntegrator::dealloc;
};

std::string MakeRestartPrefix(const bool isApoRestart, const int natom,
                              const int ndegf,
                              const std::string &rngStateString) {
  std::ostringstream output;
  const std::string zero = apo::format_fortran_double(0.0, 22, 15);
  // apoCHARMM output carries "APO" in columns 32-34; CHARMM output does not.
  output << "REST" << std::setw(6) << 52 << std::setw(6) << 1
         << (isApoRestart ? "  CUBI          APO\n\n" : "  CUBI      \n\n");
  output << std::setw(8) << 1 << " !NTITLE followed by title\n";
  output << (isApoRestart ? "* APOCHARMM RESTART FILE\n"
                          : "* CHARMM RESTART PARSER TEST\n");
  output << "\n !CRYSTAL PARAMETERS\n";
  output << apo::format_fortran_double(50.0, 22, 15) << zero
         << apo::format_fortran_double(50.0, 22, 15) << '\n';
  output << zero << zero << apo::format_fortran_double(50.0, 22, 15) << '\n';
  // HDOT, PNH/PNHV/PNHF, four cell-statistic arrays and first three GRADs.
  for (int i = 0; i < 12; ++i)
    output << zero << zero << zero << '\n';
  output << zero << '\n'; // GRAD2B
  output << "\n !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n";
  output << std::setw(12) << natom << std::setw(12) << 1 << std::setw(12) << 1
         << std::setw(12) << 0 << std::setw(12) << 0 << std::setw(12) << 1
         << std::setw(12) << ndegf << apo::format_fortran_double(1.0, 22, 15)
         << std::setw(12) << 0 << std::setw(22) << 0 << std::setw(22) << 0
         << '\n';
  output << "\n !ENERGIES and STATISTICS\n"
         << std::string(96, 'T') << '\n'
         << std::string(192, 'T') << '\n';
  output << std::setw(8) << 0 << zero << zero << zero << '\n';
  for (const int count : {96, 192, 50}) {
    for (int i = 0; i < count; ++i)
      output << zero << zero << zero << '\n';
    for (int i = 0; i < count; ++i)
      output << zero << zero << '\n';
  }
  for (const char *section :
       {" !XOLD, YOLD, ZOLD", " !VX, VY, VZ", " !X, Y, Z"}) {
    output << '\n' << section << '\n';
    for (int i = 0; i < natom; ++i)
      output << zero << zero << zero << '\n';
  }
  if (isApoRestart) {
    output << "\n !APOCHARMM RNG STATE\n"
           << "1 LANGEVIN_THERMOSTAT PHILOX4_32_10\n"
           << THERMOSTAT_SEED << '\n'
           << rngStateString << '\n';
  }
  return output.str();
}

void ConfigureIntegrator(
    std::shared_ptr<CudaLangevinThermostatIntegrator> integrator) {
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setThermostatFriction(THERMOSTAT_FRICTION);
  integrator->setThermostatRngSeed(THERMOSTAT_SEED);
  integrator->setRngSequencePos(0ULL);
  return;
}

void CheckIntegratorStateMatches(
    std::shared_ptr<CudaLangevinThermostatIntegrator> left,
    std::shared_ptr<CudaLangevinThermostatIntegrator> right,
    const double tolerance) {
  CHECK(left->getReferenceTemperature() ==
        Approx(right->getReferenceTemperature()).margin(tolerance));
  CHECK(left->getThermostatFriction() ==
        Approx(right->getThermostatFriction()).margin(tolerance));
  CHECK(left->getThermostatRngSeed() == right->getThermostatRngSeed());
  CHECK(left->getRngSequencePos() == right->getRngSequencePos());
  CHECK(left->getAverageWindowSize() == right->getAverageWindowSize());

  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getKineticEnergy()),
      apo_test::CopyToHost<double>(right->getKineticEnergy()), tolerance);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(left->getAverageTemperature()),
      apo_test::CopyToHost<double>(right->getAverageTemperature()), tolerance);

  return;
}

} // namespace

TEST_CASE("CudaLangevinThermostatIntegratorConstructorDefaults") {
  TestCudaLangevinThermostatIntegrator integrator(TIME_STEP);

  CHECK(integrator.getTimeStep() == Approx(TIME_STEP));
  CHECK(integrator.getReferenceTemperature() == Approx(300.0));
  CHECK(integrator.getThermostatFriction() == Approx(0.0));
  CHECK(integrator.getRngSequencePos() == 0ULL);
  CHECK(integrator.getAverageWindowSize() == 0);
  CHECK(integrator.getKineticEnergy().size() == 2);
  CHECK(integrator.getAverageTemperature().size() == 2);
}

TEST_CASE("CudaLangevinThermostatIntegratorValidation") {
  const double infinity = std::numeric_limits<double>::infinity();

  CudaLangevinThermostatIntegrator integrator(TIME_STEP);

  apo_test::CheckApoCharmmError(
      [&]() { integrator.setReferenceTemperature(infinity); },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference temperature must be finite; observed " +
          std::to_string(infinity));

  apo_test::CheckApoCharmmError(
      [&]() { integrator.setReferenceTemperature(-1.0); },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference temperature must be non-negative; observed -1.000000");

  apo_test::CheckApoCharmmError(
      [&]() { integrator.setThermostatFriction(infinity); },
      ApoCharmmErrorCode::InvalidArgument,
      "Thermostat friction must be finite; observed " +
          std::to_string(infinity));

  apo_test::CheckApoCharmmError(
      [&]() { integrator.setThermostatFriction(-1.0); },
      ApoCharmmErrorCode::InvalidArgument,
      "Thermostat friction must be non-negative; observed -1.000000");

  apo_test::CheckApoCharmmError(
      [&]() { integrator.setRngStates("not parsed without a context"); },
      ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before RNG states");

  apo_test::CheckApoCharmmError(
      [&]() { static_cast<void>(integrator.getRngStates()); },
      ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before retrieving RNG states");

  apo_test::CheckApoCharmmError(
      [&]() { static_cast<void>(integrator.getInstantaneousTemperature()); },
      ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before computing instantaneous temperature");

  CHECK(integrator.getReferenceTemperature() == Approx(300.0));
  CHECK(integrator.getThermostatFriction() == Approx(0.0));
}

TEST_CASE("CudaLangevinThermostatIntegratorSettersAndReset") {
  CudaLangevinThermostatIntegrator integrator(TIME_STEP);

  integrator.setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator.setThermostatFriction(THERMOSTAT_FRICTION);
  integrator.setThermostatRngSeed(THERMOSTAT_SEED);
  integrator.setRngSequencePos(RNG_SEQUENCE_POS);

  CHECK(integrator.getReferenceTemperature() == Approx(REFERENCE_TEMPERATURE));
  CHECK(integrator.getThermostatFriction() == Approx(THERMOSTAT_FRICTION));
  CHECK(integrator.getThermostatRngSeed() == THERMOSTAT_SEED);
  CHECK(integrator.getRngSequencePos() == RNG_SEQUENCE_POS);

  integrator.getAverageTemperature().set(42.0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator.getAverageTemperature()),
      std::vector<double>{42.0, 42.0}, 0.0);

  integrator.resetAverageTemperature();

  CHECK(integrator.getAverageWindowSize() == 0);
  apo_test::CheckVectorsClose<double>(
      apo_test::CopyToHost<double>(integrator.getAverageTemperature()),
      std::vector<double>{0.0, 0.0}, 0.0);
}

TEST_CASE("CudaLangevinThermostatIntegratorContextInitialization") {
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

  auto integrator =
      std::make_shared<TestCudaLangevinThermostatIntegrator>(TIME_STEP);

  ConfigureIntegrator(integrator);

  integrator->setCharmmContext(ctx);

  CHECK(integrator->getCharmmContext() == ctx);
  CHECK(integrator->getCoordsDelta().size() == 2);
  CHECK(integrator->getCoordsDeltaPrevious().size() == 2);
  CHECK(integrator->getRngSequencePos() == 0ULL);
  apo_test::CheckFiniteTemperature(ctx->computeTemperature());

  const std::string rngStateString = integrator->getRngStates();
  CHECK_NOTHROW(integrator->setRngStates(rngStateString));

  const std::vector<curandStatePhilox4_32_10_t> emptyRngStates;
  const std::string emptyRngStateString =
      apo::curand_states_to_string(0ULL, emptyRngStates);

  apo_test::CheckApoCharmmError(
      [&]() { integrator->setRngStates(emptyRngStateString); },
      ApoCharmmErrorCode::InvalidArgument,
      "RNG state count must match number of atoms; expected 2, observed 0");

  CHECK_NOTHROW(integrator->dealloc());

  apo_test::CheckApoCharmmError(
      [&]() { static_cast<void>(integrator->getRngStates()); },
      ApoCharmmErrorCode::NotInitialized, "RNG states are not initialized");

  CHECK_NOTHROW(integrator->setThermostatRngSeed(THERMOSTAT_SEED));

  integrator->getKineticEnergy().resize(0);

  apo_test::CheckApoCharmmError(
      [&]() { static_cast<void>(integrator->getInstantaneousTemperature()); },
      ApoCharmmErrorCode::Runtime,
      "Kinetic energy does not contain exactly 2 elements");

  integrator->getKineticEnergy().resize(2);
  integrator->getKineticEnergy().set(0.0);

  apo_test::CheckApoCharmmError(
      [&]() { integrator->setCharmmContext(ctx); },
      ApoCharmmErrorCode::InvalidArgument,
      "A CharmmContext object was already set for this CudaIntegrator.");
}

TEST_CASE(
    "CudaLangevinThermostatIntegratorFrictionReinitializesBeforePropagation") {
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

  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setThermostatRngSeed(THERMOSTAT_SEED);
  integrator->setCharmmContext(ctx);

  CHECK_NOTHROW(integrator->setThermostatFriction(THERMOSTAT_FRICTION));
  CHECK(integrator->getThermostatFriction() == Approx(THERMOSTAT_FRICTION));
  CHECK(integrator->getTotNumSteps() == 0ULL);
}

TEST_CASE("CudaLangevinThermostatIntegratorShortPropagation") {
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

  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  ConfigureIntegrator(integrator);
  integrator->setCharmmContext(ctx);
  integrator->resetAverageTemperature();

  const std::vector<double4> initialCoordinates =
      apo_test::CopyToHost<double4>(ctx->getCoordinatesChargesDP());

  integrator->propagate(NUM_STEPS);

  const std::vector<double4> finalCoordinates =
      apo_test::CopyToHost<double4>(ctx->getCoordinatesChargesDP());
  const std::vector<double> averageTemperature =
      apo_test::CopyToHost<double>(integrator->getAverageTemperature());
  const std::vector<double> kineticEnergy =
      apo_test::CopyToHost<double>(integrator->getKineticEnergy());

  CHECK(integrator->getCurrentPropagatedStep() == NUM_STEPS);
  CHECK(integrator->getTotNumSteps() ==
        static_cast<unsigned long long int>(NUM_STEPS));
  CHECK(integrator->getNumSteps() == NUM_STEPS);
  CHECK(integrator->getRngSequencePos() == 4ULL * NUM_STEPS);
  CHECK(integrator->getAverageWindowSize() == NUM_STEPS);

  apo_test::CheckFiniteTemperature(ctx->computeTemperature());
  apo_test::CheckFiniteTemperature(integrator->getInstantaneousTemperature());

  REQUIRE(averageTemperature.size() == 2);
  apo_test::CheckFiniteTemperature(averageTemperature[0]);
  apo_test::CheckFiniteTemperature(averageTemperature[1]);

  REQUIRE(kineticEnergy.size() == 2);
  apo_test::CheckFiniteNonnegativeScalar(kineticEnergy[0]);
  apo_test::CheckFiniteNonnegativeScalar(kineticEnergy[1]);

  REQUIRE(finalCoordinates.size() == initialCoordinates.size());
  CHECK(((finalCoordinates[0].x != initialCoordinates[0].x) ||
         (finalCoordinates[0].y != initialCoordinates[0].y) ||
         (finalCoordinates[0].z != initialCoordinates[0].z)));
}

TEST_CASE("CudaLangevinThermostatIntegratorDeterministicTrajectory") {
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
  ctx1->setCoordinates(crd1);
  ctx1->useHolonomicConstraints(false);
  ctx1->setRandomSeed(RANDOM_SEED);
  ctx1->assignVelocitiesAtTemperature(TEMPERATURE);

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

  auto integrator1 =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  ConfigureIntegrator(integrator1);
  integrator1->setCharmmContext(ctx1);

  auto integrator2 =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  ConfigureIntegrator(integrator2);
  integrator2->setCharmmContext(ctx2);

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
  CHECK(integrator1->getRngSequencePos() == 4ULL * NUM_STEPS);
  CHECK(integrator2->getRngSequencePos() == 4ULL * NUM_STEPS);

  CHECK(((finalCoordinates1[0].x != initialCoordinates1[0].x) ||
         (finalCoordinates1[0].y != initialCoordinates1[0].y) ||
         (finalCoordinates1[0].z != initialCoordinates1[0].z)));
  CHECK(((finalCoordinates2[0].x != initialCoordinates2[0].x) ||
         (finalCoordinates2[0].y != initialCoordinates2[0].y) ||
         (finalCoordinates2[0].z != initialCoordinates2[0].z)));
}

TEST_CASE("CudaLangevinThermostatIntegratorRestartValidation") {
  const std::filesystem::path missingFilePath = "missing.rst";

  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);

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

  apo_test::CheckApoCharmmError(
      [&]() { integrator->initializeFromRestartFile(missingFilePath); },
      ApoCharmmErrorCode::Runtime,
      "Could not open restart file: " + missingFilePath.string());

  const int numAtoms = ctx->getNumAtoms();
  const int ndegf = ctx->getNumDegreesOfFreedom();

  const std::filesystem::path missingRngStatePath =
      "cuda_langevin_thermostat_missing_rng_state.rst";
  apo_test::RemoveIfExists(missingRngStatePath);
  apo_test::WriteTextFile(missingRngStatePath,
                          MakeRestartPrefix(true, numAtoms, ndegf, ""));

  apo_test::CheckApoCharmmError(
      [&]() { integrator->initializeFromRestartFile(missingRngStatePath); },
      ApoCharmmErrorCode::Runtime,
      "Missing or inconsistent APOCHARMM RNG state in restart file \"" +
          missingRngStatePath.string() + "\"");

  apo_test::RemoveIfExists(missingRngStatePath);

  const std::filesystem::path natomMismatchPath =
      "cuda_langevin_thermostat_natom_mismatch.rst";
  apo_test::RemoveIfExists(natomMismatchPath);
  apo_test::WriteTextFile(natomMismatchPath,
                          MakeRestartPrefix(false, numAtoms + 1, ndegf, ""));

  apo_test::CheckApoCharmmError(
      [&]() { integrator->initializeFromRestartFile(natomMismatchPath); },
      ApoCharmmErrorCode::InvalidArgument,
      "NATOM mismatch in restart file \"" + natomMismatchPath.string() + "\"");

  apo_test::RemoveIfExists(natomMismatchPath);

  const std::vector<curandStatePhilox4_32_10_t> emptyRngStates;
  const std::string emptyRngStateString =
      apo::curand_states_to_string(0ULL, emptyRngStates);
  const std::filesystem::path rngCountMismatchPath =
      "cuda_langevin_thermostat_rng_count_mismatch.rst";

  apo_test::RemoveIfExists(rngCountMismatchPath);
  apo_test::WriteTextFile(
      rngCountMismatchPath,
      MakeRestartPrefix(true, numAtoms, ndegf, emptyRngStateString));

  apo_test::CheckApoCharmmError(
      [&]() { integrator->initializeFromRestartFile(rngCountMismatchPath); },
      ApoCharmmErrorCode::InvalidArgument,
      "Restart RNG state count must match the atom count");

  apo_test::RemoveIfExists(rngCountMismatchPath);
}
