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
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "DistanceRestraintForce.h"
#include "Force.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "catch.hpp"
#include "cuda_utils.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <vector_types.h>

namespace {

using ResdForce = DistanceRestraintForce<long long int, float>;

constexpr int NUM_ATOMS = 3;
constexpr double EXACT_GRADIENT_TOLERANCE = 0.0;
constexpr double EXACT_ENERGY_TOLERANCE = 0.0;
constexpr double EXACT_VIRIAL_TOLERANCE = 0.0;
constexpr double ANALYTICAL_GRADIENT_TOLERANCE = 2.0e-7;
constexpr double ANALYTICAL_ENERGY_TOLERANCE = 1.0e-10;
constexpr double ANALYTICAL_VIRIAL_TOLERANCE = 1.0e-8;
constexpr double CHARMM_GRADIENT_TOLERANCE = 1.0e-8;
constexpr double CHARMM_MULTIPAIR_GRADIENT_TOLERANCE = 1.0e-5;
constexpr double CHARMM_ENERGY_TOLERANCE = 1.0e-8;
constexpr double CHARMM_VIRIAL_TOLERANCE = 1.0e-7;
constexpr double CHARMM_LARGE_VIRIAL_TOLERANCE = 1.0e-5;
constexpr double FINITE_DIFFERENCE_STEP = 1.0 / 1024.0;
constexpr double FINITE_DIFFERENCE_ABS_TOLERANCE = 2.0e-5;
constexpr double FINITE_DIFFERENCE_REL_TOLERANCE = 2.0e-5;
constexpr double FINITE_STRAIN_TOLERANCE = 1.0e-7;
constexpr double INV_FORCE_SCALE_TEST = 1.0 / static_cast<double>(1LL << 40);

const std::vector<double> BOX_DIMENSIONS = {40.0, 41.0, 42.0};
const std::vector<std::array<int, 2>> SINGLE_PAIR = {{0, 1}};
const std::vector<double> UNIT_COEFFICIENT = {1.0};
const std::vector<float4> STANDARD_COORDINATES = {
    make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(2.0f, 0.0f, 0.0f, 0.0f),
    make_float4(5.0f, 0.0f, 0.0f, 0.0f)};
const std::array<double, 9> ZERO_VIRIAL = {0.0, 0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0};

constexpr int INTEGRATION_NUM_ATOMS = 2;
constexpr std::uint64_t INTEGRATION_RANDOM_SEED = 314159ULL;
constexpr int INTEGRATION_PROPAGATION_STEPS = 2;
constexpr double INTEGRATION_TEMPERATURE = 300.0;
constexpr double INTEGRATION_TIME_STEP = 0.001;
constexpr double INTEGRATION_THERMOSTAT_FRICTION = 0.0;
constexpr double INTEGRATION_FORCE_TOLERANCE = 1.0e-8;
constexpr double INTEGRATION_ENERGY_TOLERANCE = 1.0e-8;

const std::vector<double> INTEGRATION_BOX_DIMENSIONS = {30.0, 32.0, 34.0};
const std::vector<double> UPDATED_INTEGRATION_BOX_DIMENSIONS = {40.0, 41.0,
                                                                42.0};
const std::vector<double3> INTEGRATION_COORDINATES = {
    make_double3(0.0, 0.0, 0.0), make_double3(1.0, 2.0, 3.0)};

struct ResdIntegrationSystem {
  std::shared_ptr<CharmmContext> context;
  std::shared_ptr<ForceManager> forceManager;
};

struct ResdManagerOutput {
  std::vector<double3> forceValues;
  double potentialEnergy;
};

std::vector<double3> MakeExpectedXGradients(const double firstAtomGradient,
                                            const double secondAtomGradient) {
  return {make_double3(firstAtomGradient, 0.0, 0.0),
          make_double3(secondAtomGradient, 0.0, 0.0),
          make_double3(0.0, 0.0, 0.0)};
}

void AddStandardRestraint(ResdForce &restraint,
                          const double forceConstant = 2.0,
                          const double referenceValue = 1.0) {
  restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, forceConstant,
                         referenceValue, 1, 2);
  return;
}

void ClearAndCalculate(ResdForce &restraint, CudaContainer<float4> &xyzq,
                       const bool calcEnergy = true,
                       const bool calcVirial = true) {
  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), calcEnergy, calcVirial);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  return;
}

std::vector<double3> CopyResdGradients(ResdForce &restraint) {
  const std::shared_ptr<Force<long long int>> force = restraint.getForce();
  std::vector<long long int> gx(force->size());
  std::vector<long long int> gy(force->size());
  std::vector<long long int> gz(force->size());

  force->getXYZ(gx.data(), gy.data(), gz.data());

  std::vector<double3> observedGradients(force->size());
  for (std::size_t i = 0; i < observedGradients.size(); i++) {
    observedGradients[i].x = static_cast<double>(gx[i]) * INV_FORCE_SCALE_TEST;
    observedGradients[i].y = static_cast<double>(gy[i]) * INV_FORCE_SCALE_TEST;
    observedGradients[i].z = static_cast<double>(gz[i]) * INV_FORCE_SCALE_TEST;
  }

  return observedGradients;
}

double GetResdEnergy(ResdForce &restraint) {
  const std::shared_ptr<cudaStream_t> stream = restraint.getStream();
  cudaCheck(cudaStreamSynchronize(*stream));
  restraint.getEnergyVirial()->copyToHost(*stream);
  cudaCheck(cudaStreamSynchronize(*stream));
  return restraint.getEnergyVirial()->getEnergy("resd");
}

std::array<double, 9> CopyStoredVirial(ResdForce &restraint) {
  const std::shared_ptr<cudaStream_t> stream = restraint.getStream();
  cudaCheck(cudaStreamSynchronize(*stream));
  restraint.getEnergyVirial()->copyToHost(*stream);
  cudaCheck(cudaStreamSynchronize(*stream));

  std::array<double, 9> virial{};
  restraint.getEnergyVirial()->getVirial(virial.data());
  return virial;
}

std::array<double, 9>
CalculateManagerStyleVirial(ResdForce &restraint, CudaContainer<float4> &xyzq,
                            const std::vector<double> &boxDimensions) {
  REQUIRE(boxDimensions.size() == 3);

  const std::shared_ptr<cudaStream_t> stream = restraint.getStream();
  const std::shared_ptr<Force<long long int>> fixedGradients =
      restraint.getForce();
  Force<double> doubleGradients(fixedGradients->size());

  fixedGradients->convert<double>(doubleGradients, *stream);
  restraint.getEnergyVirial()->calcVirial(
      fixedGradients->size(), xyzq.getDeviceArray().data(), boxDimensions[0],
      boxDimensions[1], boxDimensions[2], doubleGradients.stride(),
      doubleGradients.xyz(), *stream);
  cudaCheck(cudaStreamSynchronize(*stream));

  return CopyStoredVirial(restraint);
}

void CheckVirial(const std::string &label,
                 const std::array<double, 9> &observed,
                 const std::array<double, 9> &expected,
                 const double tolerance) {
  apo_test::CheckVectorsClose<double>(
      label, std::vector<double>(observed.begin(), observed.end()),
      std::vector<double>(expected.begin(), expected.end()), tolerance);
  return;
}

void CheckShiftedForcesAreZero(ResdForce &restraint, const std::string &label,
                               const double tolerance) {
  const std::shared_ptr<cudaStream_t> stream = restraint.getStream();
  restraint.getEnergyVirial()->copyToHost(*stream);
  cudaCheck(cudaStreamSynchronize(*stream));

  std::vector<double> observedShiftedForces(27 * 3, 0.0);
  restraint.getEnergyVirial()->getSforce(observedShiftedForces.data());
  const std::vector<double> expectedShiftedForces(27 * 3, 0.0);
  apo_test::CheckVectorsClose<double>(label, observedShiftedForces,
                                      expectedShiftedForces, tolerance);
  return;
}

void CheckZeroTotalGradient(const std::string &label,
                            const std::vector<double3> &gradients,
                            const double tolerance) {
  double3 sum = make_double3(0.0, 0.0, 0.0);
  for (const double3 gradient : gradients) {
    sum.x += gradient.x;
    sum.y += gradient.y;
    sum.z += gradient.z;
  }

  apo_test::CheckVectorsClose<double3>(
      label, {sum}, {make_double3(0.0, 0.0, 0.0)}, tolerance);
  return;
}

void CheckResdOutput(
    ResdForce &restraint, const std::string &label, const double expectedEnergy,
    const std::vector<double3> &expectedGradients,
    const double energyTolerance = EXACT_ENERGY_TOLERANCE,
    const double gradientTolerance = EXACT_GRADIENT_TOLERANCE) {
  CHECK(GetResdEnergy(restraint) ==
        Approx(expectedEnergy).margin(energyTolerance));
  apo_test::CheckVectorsClose<double3>(label + " gradients",
                                       CopyResdGradients(restraint),
                                       expectedGradients, gradientTolerance);
  return;
}

double EvaluateEnergy(ResdForce &restraint,
                      const std::vector<float4> &coordinates) {
  CudaContainer<float4> xyzq = coordinates;
  ClearAndCalculate(restraint, xyzq, true, false);
  return GetResdEnergy(restraint);
}

float GetCoordinateComponent(const float4 &coordinate, const int component) {
  REQUIRE(((component >= 0) && (component < 3)));
  if (component == 0)
    return coordinate.x;
  if (component == 1)
    return coordinate.y;
  return coordinate.z;
}

void SetCoordinateComponent(float4 &coordinate, const int component,
                            const float value) {
  REQUIRE(((component >= 0) && (component < 3)));
  if (component == 0)
    coordinate.x = value;
  else if (component == 1)
    coordinate.y = value;
  else
    coordinate.z = value;
  return;
}

void CheckCentralFiniteDifference(
    ResdForce &restraint, const std::string &label,
    const std::vector<float4> &coordinates, const double expectedEnergy,
    const std::vector<double3> &expectedGradients) {
  REQUIRE(coordinates.size() == expectedGradients.size());

  CudaContainer<float4> centerXyzq = coordinates;
  ClearAndCalculate(restraint, centerXyzq, true, false);
  CheckResdOutput(restraint, label, expectedEnergy, expectedGradients,
                  ANALYTICAL_ENERGY_TOLERANCE, ANALYTICAL_GRADIENT_TOLERANCE);

  for (std::size_t atomIndex = 0; atomIndex < coordinates.size(); atomIndex++) {
    for (int component = 0; component < 3; component++) {
      std::vector<float4> plusCoordinates = coordinates;
      std::vector<float4> minusCoordinates = coordinates;
      const float center =
          GetCoordinateComponent(coordinates[atomIndex], component);
      const float plus = center + static_cast<float>(FINITE_DIFFERENCE_STEP);
      const float minus = center - static_cast<float>(FINITE_DIFFERENCE_STEP);
      SetCoordinateComponent(plusCoordinates[atomIndex], component, plus);
      SetCoordinateComponent(minusCoordinates[atomIndex], component, minus);

      const double plusEnergy = EvaluateEnergy(restraint, plusCoordinates);
      const double minusEnergy = EvaluateEnergy(restraint, minusCoordinates);
      const double denominator =
          static_cast<double>(plus) - static_cast<double>(minus);
      const double numericalGradient = (plusEnergy - minusEnergy) / denominator;
      const double expectedGradient =
          (component == 0)
              ? expectedGradients[atomIndex].x
              : ((component == 1) ? expectedGradients[atomIndex].y
                                  : expectedGradients[atomIndex].z);

      INFO("finite-difference label: " << label);
      INFO("atom index: " << atomIndex);
      INFO("component: " << component);
      CHECK(numericalGradient == Approx(expectedGradient)
                                     .margin(FINITE_DIFFERENCE_ABS_TOLERANCE)
                                     .epsilon(FINITE_DIFFERENCE_REL_TOLERANCE));
    }
  }

  return;
}

ResdIntegrationSystem MakeUninitializedResdIntegrationSystem(void) {
  auto parameters = std::make_shared<CharmmParameters>(
      apo_test::GetTopparDir() / "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto context = std::make_shared<CharmmContext>(psf, parameters);
  std::shared_ptr<ForceManager> forceManager = context->getForceManager();

  REQUIRE(context->getNumAtoms() == INTEGRATION_NUM_ATOMS);
  REQUIRE(forceManager != nullptr);
  REQUIRE(forceManager->isInitialized() == false);

  return {context, forceManager};
}

void InitializeResdIntegrationSystem(ResdIntegrationSystem &system) {
  REQUIRE_NOTHROW(system.context->setBoxDimensions(INTEGRATION_BOX_DIMENSIONS));
  REQUIRE(system.forceManager->isInitialized() == true);

  REQUIRE_NOTHROW(system.context->setCoordinates(INTEGRATION_COORDINATES));
  REQUIRE_NOTHROW(system.context->useHolonomicConstraints(false));

  REQUIRE(system.forceManager->getPsf() != nullptr);
  REQUIRE(system.forceManager->getPsf()->getNumAtoms() ==
          INTEGRATION_NUM_ATOMS);
  apo_test::CheckVectorsClose<double>("RESD integration context box",
                                      system.context->getBoxDimensions(),
                                      INTEGRATION_BOX_DIMENSIONS, 0.0);
  apo_test::CheckVectorsClose<double>("RESD integration ForceManager box",
                                      system.forceManager->getBoxDimensions(),
                                      INTEGRATION_BOX_DIMENSIONS, 0.0);

  return;
}

std::shared_ptr<ResdForce>
MakeStandardIntegrationRestraint(const int numAtoms) {
  auto restraint = std::make_shared<ResdForce>(numAtoms);
  AddStandardRestraint(*restraint, 2.0, 0.0);
  return restraint;
}

void SubscribeIntegrationRestraint(
    const std::shared_ptr<ForceManager> &forceManager,
    const std::shared_ptr<ResdForce> &restraint, const std::string &forceTag) {
  forceManager->subscribe(restraint, forceTag, restraint->getStream(),
                          restraint->getForce(), restraint->getEnergyVirial());
  return;
}

ResdManagerOutput
CalculateResdManagerOutput(const ResdIntegrationSystem &system) {
  system.context->calculateForces(false, true, false);

  const std::shared_ptr<Force<double>> force = system.forceManager->getForces();
  const std::size_t numAtoms = static_cast<std::size_t>(force->size());
  std::vector<double> forceX(numAtoms);
  std::vector<double> forceY(numAtoms);
  std::vector<double> forceZ(numAtoms);
  force->getXYZ(forceX.data(), forceY.data(), forceZ.data());

  std::vector<double3> forceValues(numAtoms);
  for (std::size_t i = 0; i < numAtoms; i++)
    forceValues[i] = make_double3(forceX[i], forceY[i], forceZ[i]);

  CudaContainer<double> &potentialEnergy =
      system.forceManager->getPotentialEnergy();
  potentialEnergy.transferToHost();
  REQUIRE(potentialEnergy.size() == 1);

  return {forceValues, potentialEnergy[0]};
}

std::vector<double3> MakeAntisymmetricTwoAtomContribution(
    const double firstAtomX, const double firstAtomY, const double firstAtomZ) {
  return {make_double3(firstAtomX, firstAtomY, firstAtomZ),
          make_double3(-firstAtomX, -firstAtomY, -firstAtomZ)};
}

void CheckResdManagerContribution(
    const std::string &label, const ResdManagerOutput &baseline,
    const ResdManagerOutput &observed, const double expectedEnergy,
    const std::vector<double3> &expectedForceValues) {
  REQUIRE(observed.forceValues.size() == baseline.forceValues.size());
  REQUIRE(observed.forceValues.size() == expectedForceValues.size());

  std::vector<double3> forceDifference(observed.forceValues.size());
  for (std::size_t i = 0; i < forceDifference.size(); i++) {
    forceDifference[i] =
        make_double3(observed.forceValues[i].x - baseline.forceValues[i].x,
                     observed.forceValues[i].y - baseline.forceValues[i].y,
                     observed.forceValues[i].z - baseline.forceValues[i].z);
  }

  INFO("RESD integration contribution: " << label);
  CHECK((observed.potentialEnergy - baseline.potentialEnergy) ==
        Approx(expectedEnergy).margin(INTEGRATION_ENERGY_TOLERANCE));
  apo_test::CheckVectorsClose<double3>(label + " total-force contribution",
                                       forceDifference, expectedForceValues,
                                       INTEGRATION_FORCE_TOLERANCE);

  return;
}

} // namespace

TEST_CASE("DistanceRestraintForceConstructorAndMetadata") {
  CHECK(std::is_default_constructible<ResdForce>::value == false);

  apo_test::CheckApoCharmmError([]() -> void { (void)ResdForce(0); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Atom count must be positive; observed 0");

  apo_test::CheckApoCharmmError([]() -> void { (void)ResdForce(-1); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Atom count must be positive; observed -1");

  ResdForce restraint(NUM_ATOMS);

  CHECK(decltype(restraint)::contributesVirial == true);

  REQUIRE(restraint.getStream() != nullptr);
  CHECK(*restraint.getStream() != nullptr);

  REQUIRE(restraint.getForce() != nullptr);
  CHECK(restraint.getForce()->xyz() != nullptr);
  CHECK(restraint.getForce()->size() == NUM_ATOMS);
  CHECK(restraint.getForce()->stride() >= NUM_ATOMS);

  REQUIRE(restraint.getEnergyVirial() != nullptr);
  CHECK(restraint.getEnergyVirial()->getEnergyPointer("resd") != nullptr);
}

TEST_CASE("DistanceRestraintForceZeroTermsProduceZeroOutput") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);

  CheckResdOutput(restraint, "RESD zero terms", 0.0,
                  MakeExpectedXGradients(0.0, 0.0));
}

TEST_CASE("DistanceRestraintForceDefaultScaleIsOne") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);

  CheckResdOutput(restraint, "RESD default SCALE", 1.0,
                  MakeExpectedXGradients(-2.0, 2.0));
}

TEST_CASE("DistanceRestraintForceRejectsEmptyPairList") {
  ResdForce restraint(NUM_ATOMS);

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.addRestraint({}, {}, 2.0, 1.0, 1, 2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "A distance-restraint term must contain at least one atom pair");
}

TEST_CASE("DistanceRestraintForceRejectsPairCoefficientCountMismatch") {
  ResdForce restraint(NUM_ATOMS);

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.addRestraint(SINGLE_PAIR, {}, 2.0, 1.0, 1, 2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Pair and coefficient counts must match; observed 1 pairs and 0 "
      "coefficients");
}

TEST_CASE("DistanceRestraintForceValidatesAtomIndices") {
  ResdForce restraint(NUM_ATOMS);

  SECTION("NegativeFirstAtomIndex") {
    const std::vector<std::array<int, 2>> atomPairs = {{-1, 1}};
    apo_test::CheckApoCharmmError(
        [&restraint, &atomPairs]() -> void {
          restraint.addRestraint(atomPairs, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "First atom index at pair index 0 is out of range; expected [0, 3), "
        "observed -1");
  }

  SECTION("NegativeSecondAtomIndex") {
    const std::vector<std::array<int, 2>> atomPairs = {{0, -1}};
    apo_test::CheckApoCharmmError(
        [&restraint, &atomPairs]() -> void {
          restraint.addRestraint(atomPairs, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Second atom index at pair index 0 is out of range; expected [0, 3), "
        "observed -1");
  }

  SECTION("FirstAtomIndexEqualsAtomCount") {
    const std::vector<std::array<int, 2>> atomPairs = {{NUM_ATOMS, 1}};
    apo_test::CheckApoCharmmError(
        [&restraint, &atomPairs]() -> void {
          restraint.addRestraint(atomPairs, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "First atom index at pair index 0 is out of range; expected [0, 3), "
        "observed 3");
  }

  SECTION("SecondAtomIndexEqualsAtomCount") {
    const std::vector<std::array<int, 2>> atomPairs = {{0, NUM_ATOMS}};
    apo_test::CheckApoCharmmError(
        [&restraint, &atomPairs]() -> void {
          restraint.addRestraint(atomPairs, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Second atom index at pair index 0 is out of range; expected [0, 3), "
        "observed 3");
  }
}

TEST_CASE("DistanceRestraintForceRejectsNonfiniteCoefficient") {
  ResdForce restraint(NUM_ATOMS);
  const double infinity = std::numeric_limits<double>::infinity();
  const std::vector<double> coefficients = {infinity};

  apo_test::CheckApoCharmmError(
      [&restraint, &coefficients]() -> void {
        restraint.addRestraint(SINGLE_PAIR, coefficients, 2.0, 1.0, 1, 2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Coefficient at pair index 0 must be finite; observed " +
          std::to_string(infinity));
}

TEST_CASE("DistanceRestraintForceRejectsNonfiniteForceConstant") {
  ResdForce restraint(NUM_ATOMS);
  const double infinity = std::numeric_limits<double>::infinity();

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, infinity, 1.0, 1,
                               2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force constant must be finite; observed " + std::to_string(infinity));
}

TEST_CASE("DistanceRestraintForceImplementsForceConstantSignRules") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  SECTION("PositiveNonzeroForceConstantIsAccepted") {
    CHECK_NOTHROW(
        restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2));
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD positive KVAL", 1.0,
                    MakeExpectedXGradients(-2.0, 2.0));
  }

  SECTION("NegativeNonzeroForceConstantIsAccepted") {
    CHECK_NOTHROW(
        restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, -2.0, 1.0, 1, 2));
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD negative KVAL", -1.0,
                    MakeExpectedXGradients(2.0, -2.0));
  }

  SECTION("ZeroForceConstantIsRejected") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 0.0, 1.0, 1, 2);
        },
        ApoCharmmErrorCode::InvalidArgument, "Force constant must be nonzero");
  }
}

TEST_CASE("DistanceRestraintForceRejectsNonfiniteReferenceValue") {
  ResdForce restraint(NUM_ATOMS);
  const double infinity = std::numeric_limits<double>::infinity();

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, infinity, 1,
                               2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference value must be finite; observed " + std::to_string(infinity));
}

TEST_CASE("DistanceRestraintForceRejectsUnsupportedDistanceExponent") {
  ResdForce restraint(NUM_ATOMS);

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 3, 2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Distance exponent must be one of the characterized values {-2, -1, 0, "
      "1, 2, 6, 7}; observed 3");
}

TEST_CASE("DistanceRestraintForceRejectsInvalidEnergyExponent") {
  ResdForce restraint(NUM_ATOMS);

  SECTION("NegativeEnergyExponent") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1,
                                 -1);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Energy exponent must be one of the characterized values {1, 2, 3, "
        "4}; observed -1");
  }

  SECTION("ZeroEnergyExponent") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 0);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Energy exponent must be one of the characterized values {1, 2, 3, "
        "4}; observed 0");
  }
}

TEST_CASE("DistanceRestraintForceRejectsInvalidActivationCondition") {
  ResdForce restraint(NUM_ATOMS);
  const DistanceRestraintCondition invalidCondition =
      static_cast<DistanceRestraintCondition>(2);

  apo_test::CheckApoCharmmError(
      [&restraint, invalidCondition]() -> void {
        restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2,
                               invalidCondition);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Activation condition must be NONE, POSITIVE, or NEGATIVE; observed 2");
}

TEST_CASE("DistanceRestraintForceRejectsIdenticalAtomIndices") {
  ResdForce restraint(NUM_ATOMS);
  const std::vector<std::array<int, 2>> identicalPair = {{1, 1}};

  apo_test::CheckApoCharmmError(
      [&restraint, &identicalPair]() -> void {
        restraint.addRestraint(identicalPair, UNIT_COEFFICIENT, 2.0, -1.0, 1,
                               2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Atom indices at pair index 0 must be distinct; observed (1, 1)");
}

TEST_CASE("DistanceRestraintForceValidatesBoxDimensions") {
  ResdForce restraint(NUM_ATOMS);
  const double infinity = std::numeric_limits<double>::infinity();

  SECTION("WrongBoxDimensionCount") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void { restraint.setBoxDimensions({10.0, 10.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Box-dimension array size mismatch; expected 3, observed 2");
  }

  SECTION("NonfiniteBoxDimension") {
    apo_test::CheckApoCharmmError(
        [&restraint, infinity]() -> void {
          restraint.setBoxDimensions({10.0, infinity, 10.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index 1 must be finite; observed " +
            std::to_string(infinity));
  }

  SECTION("ZeroBoxDimension") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setBoxDimensions({10.0, 0.0, 10.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index 1 must be positive; observed " +
            std::to_string(0.0));
  }

  SECTION("NegativeBoxDimension") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setBoxDimensions({10.0, -1.0, 10.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index 1 must be positive; observed " +
            std::to_string(-1.0));
  }

  SECTION("PositiveFiniteBoxDimensionsAreAccepted") {
    CHECK_NOTHROW(restraint.setBoxDimensions(BOX_DIMENSIONS));
  }
}

TEST_CASE("DistanceRestraintForceRejectsInitializeAtomCountMismatch") {
  ResdForce restraint(NUM_ATOMS);

  CHECK_NOTHROW(restraint.initialize(NUM_ATOMS, BOX_DIMENSIONS));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.initialize(NUM_ATOMS + 1, BOX_DIMENSIONS);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Initialization atom count mismatch; expected 3, observed 4");
}

TEST_CASE("DistanceRestraintForceValidatesScale") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;
  const double infinity = std::numeric_limits<double>::infinity();

  SECTION("NonfiniteScaleIsRejected") {
    apo_test::CheckApoCharmmError(
        [&restraint, infinity]() -> void { restraint.setScale(infinity); },
        ApoCharmmErrorCode::InvalidArgument,
        "Scale must be finite; observed " + std::to_string(infinity));
  }

  SECTION("PositiveScaleIsAccepted") {
    AddStandardRestraint(restraint);
    CHECK_NOTHROW(restraint.setScale(3.0));
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD positive SCALE", 3.0,
                    MakeExpectedXGradients(-6.0, 6.0));
  }

  SECTION("ZeroScaleIsAccepted") {
    AddStandardRestraint(restraint);
    CHECK_NOTHROW(restraint.setScale(0.0));
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD zero SCALE", 0.0,
                    MakeExpectedXGradients(0.0, 0.0));
  }

  SECTION("NegativeScaleIsAccepted") {
    AddStandardRestraint(restraint);
    CHECK_NOTHROW(restraint.setScale(-2.0));
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD negative SCALE", -2.0,
                    MakeExpectedXGradients(4.0, -4.0));
  }
}

TEST_CASE("DistanceRestraintForceResetRemovesAllTerms") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD before reset", 1.0,
                  MakeExpectedXGradients(-2.0, 2.0));

  restraint.reset();
  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD after reset", 0.0,
                  MakeExpectedXGradients(0.0, 0.0));
}

TEST_CASE("DistanceRestraintForceResetRestoresScaleToOne") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  restraint.setScale(3.0);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD scaled before reset", 3.0,
                  MakeExpectedXGradients(-6.0, 6.0));

  restraint.reset();
  AddStandardRestraint(restraint, 4.0);
  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD new term after scale reset", 2.0,
                  MakeExpectedXGradients(-4.0, 4.0));
}

TEST_CASE("DistanceRestraintForceCanAddNewTermsAfterReset") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD old term before reset", 1.0,
                  MakeExpectedXGradients(-2.0, 2.0));

  restraint.reset();
  AddStandardRestraint(restraint, 4.0);
  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD replacement term after reset", 2.0,
                  MakeExpectedXGradients(-4.0, 4.0));
}

TEST_CASE("DistanceRestraintForceClearPreservesDefinitions") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD before clear", 1.0,
                  MakeExpectedXGradients(-2.0, 2.0));

  restraint.clear();
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  CheckResdOutput(restraint, "RESD immediately after clear", 0.0,
                  MakeExpectedXGradients(0.0, 0.0));

  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  CheckResdOutput(restraint, "RESD recalculated after clear", 1.0,
                  MakeExpectedXGradients(-2.0, 2.0));
}

TEST_CASE(
    "DistanceRestraintForceRepeatedClearCalculateCyclesHaveNoStaleOutput") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  for (int cycle = 0; cycle < 3; cycle++) {
    CAPTURE(cycle);
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD repeated cycle", 1.0,
                    MakeExpectedXGradients(-2.0, 2.0));
  }
}

TEST_CASE(
    "DistanceRestraintForceSynchronizesDefinitionsAddedAfterCalculation") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  restraint.setScale(3.0);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD before second term", 3.0,
                  MakeExpectedXGradients(-6.0, 6.0));

  restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 4.0, 0.0, 1, 2);
  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD after second term", 27.0,
                  MakeExpectedXGradients(-30.0, 30.0));
}

TEST_CASE("DistanceRestraintForceFailedAddPreservesPriorDefinitions") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD before failed add", 1.0,
                  MakeExpectedXGradients(-2.0, 2.0));

  const std::vector<std::array<int, 2>> atomPairs = {{0, 1}, {1, 2}};
  const std::vector<double> coefficients = {1.0, 0.0};
  apo_test::CheckApoCharmmError(
      [&restraint, &atomPairs, &coefficients]() -> void {
        restraint.addRestraint(atomPairs, coefficients, 4.0, 0.0, 1, 2);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Coefficient at pair index 1 must be nonzero");

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD after failed add", 1.0,
                  MakeExpectedXGradients(-2.0, 2.0));
}

TEST_CASE("DistanceRestraintForceCalculatesOneQuadraticPair") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  const double distance = 2.0;
  const double deviation = 1.0 * distance - 1.0;
  const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double firstAtomGradient =
      derivative * 1.0 * 1.0 * (0.0 - 2.0) / distance;
  const double secondAtomGradient = -firstAtomGradient;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(
      restraint, "RESD one quadratic pair", expectedEnergy,
      MakeExpectedXGradients(firstAtomGradient, secondAtomGradient));
}

TEST_CASE("DistanceRestraintForceQuadraticPairAtReferenceValueIsStationary") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 2.0, 1, 2);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  const double distance = 2.0;
  const double deviation = 1.0 * distance - 2.0;
  const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double firstAtomGradient =
      derivative * 1.0 * 1.0 * (0.0 - 2.0) / distance;
  const double secondAtomGradient = -firstAtomGradient;

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(
      restraint, "RESD pair exactly at RVAL", expectedEnergy,
      MakeExpectedXGradients(firstAtomGradient, secondAtomGradient));
}

TEST_CASE("DistanceRestraintForceAppliesPairCoefficientsAlgebraically") {
  SECTION("CoefficientTwo") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, {2.0}, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double distance = 2.0;
    const double deviation = 2.0 * distance - 0.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient =
        derivative * 2.0 * 1.0 * (0.0 - 2.0) / distance;

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD coefficient 2", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
  }

  SECTION("CoefficientNegativeOne") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, {-1.0}, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double distance = 2.0;
    const double deviation = -1.0 * distance - 0.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient =
        derivative * -1.0 * 1.0 * (0.0 - 2.0) / distance;

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD coefficient -1", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
  }

  SECTION("CoefficientOneHalf") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, {0.5}, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double distance = 2.0;
    const double deviation = 0.5 * distance - 0.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient =
        derivative * 0.5 * 1.0 * (0.0 - 2.0) / distance;

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD coefficient 0.5", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
  }

  SECTION("CoefficientZeroIsRejected") {
    ResdForce restraint(NUM_ATOMS);
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.addRestraint(SINGLE_PAIR, {0.0}, 2.0, 0.0, 1, 2);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Coefficient at pair index 0 must be nonzero");
  }
}

TEST_CASE("DistanceRestraintForceCalculatesSignedTwoPairReactionCoordinate") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  const std::vector<std::array<int, 2>> atomPairs = {{0, 1}, {1, 2}};
  const std::vector<double> coefficients = {1.0, -1.0};
  restraint.addRestraint(atomPairs, coefficients, 2.0, -2.0, 1, 2);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  const double r01 = 2.0;
  const double r12 = 3.0;
  const double deviation = r01 - r12 - (-2.0);
  const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double pair01FirstGradient = derivative * 1.0 * (0.0 - 2.0) / r01;
  const double pair01SecondGradient = -pair01FirstGradient;
  const double pair12FirstGradient = derivative * -1.0 * (2.0 - 5.0) / r12;
  const double pair12SecondGradient = -pair12FirstGradient;
  const std::vector<double3> expectedGradients = {
      make_double3(pair01FirstGradient, 0.0, 0.0),
      make_double3(pair01SecondGradient + pair12FirstGradient, 0.0, 0.0),
      make_double3(pair12SecondGradient, 0.0, 0.0)};

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD r01-r12-RVAL", expectedEnergy,
                  expectedGradients);
  CheckZeroTotalGradient("RESD r01-r12-RVAL total gradient",
                         CopyResdGradients(restraint),
                         EXACT_GRADIENT_TOLERANCE);
}

TEST_CASE("DistanceRestraintForceAccumulatesSeveralPairsSharingOneAtom") {
  constexpr int numAtoms = 5;
  ResdForce restraint(numAtoms);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  const std::vector<std::array<int, 2>> atomPairs = {{0, 1}, {0, 2}, {0, 3}};
  restraint.addRestraint(atomPairs, {1.0, 1.0, 1.0}, 2.0, 0.0, 1, 2);
  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(1.0f, 0.0f, 0.0f, 0.0f),
      make_float4(2.0f, 0.0f, 0.0f, 0.0f), make_float4(3.0f, 0.0f, 0.0f, 0.0f),
      make_float4(9.0f, 4.0f, -7.0f, 0.0f)};

  const double deviation = 1.0 + 2.0 + 3.0 - 0.0;
  const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double eachSecondAtomGradient = derivative;
  const double sharedAtomGradient = -3.0 * derivative;
  const std::vector<double3> expectedGradients = {
      make_double3(sharedAtomGradient, 0.0, 0.0),
      make_double3(eachSecondAtomGradient, 0.0, 0.0),
      make_double3(eachSecondAtomGradient, 0.0, 0.0),
      make_double3(eachSecondAtomGradient, 0.0, 0.0),
      make_double3(0.0, 0.0, 0.0)};

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD three pairs sharing atom zero",
                  expectedEnergy, expectedGradients);
  const std::vector<double3> observedGradients = CopyResdGradients(restraint);
  CheckZeroTotalGradient("RESD shared-atom total gradient", observedGradients,
                         EXACT_GRADIENT_TOLERANCE);
  CHECK(observedGradients[4].x == 0.0);
  CHECK(observedGradients[4].y == 0.0);
  CHECK(observedGradients[4].z == 0.0);
}

TEST_CASE("DistanceRestraintForceAccumulatesMultipleTermsInOneObject") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2);
  restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 4.0, 0.0, 1, 2);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  const double firstDeviation = 2.0 - 1.0;
  const double secondDeviation = 2.0 - 0.0;
  const double firstEnergy = (2.0 / 2.0) * firstDeviation * firstDeviation;
  const double secondEnergy = (4.0 / 2.0) * secondDeviation * secondDeviation;
  const double firstDerivative = 2.0 * firstDeviation;
  const double secondDerivative = 4.0 * secondDeviation;
  const double firstAtomGradient = -(firstDerivative + secondDerivative);

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(
      restraint, "RESD two terms in one object", firstEnergy + secondEnergy,
      MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
}

TEST_CASE("DistanceRestraintForceAccumulatesMultipleTermsSharingAtoms") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  restraint.addRestraint({{0, 1}}, {1.0}, 2.0, 1.0, 1, 2);
  restraint.addRestraint({{1, 2}}, {1.0}, 4.0, 1.0, 1, 2);
  CudaContainer<float4> xyzq = STANDARD_COORDINATES;

  const double firstDeviation = 2.0 - 1.0;
  const double secondDeviation = 3.0 - 1.0;
  const double firstEnergy = (2.0 / 2.0) * firstDeviation * firstDeviation;
  const double secondEnergy = (4.0 / 2.0) * secondDeviation * secondDeviation;
  const double firstDerivative = 2.0 * firstDeviation;
  const double secondDerivative = 4.0 * secondDeviation;
  const std::vector<double3> expectedGradients = {
      make_double3(-firstDerivative, 0.0, 0.0),
      make_double3(firstDerivative - secondDerivative, 0.0, 0.0),
      make_double3(secondDerivative, 0.0, 0.0)};

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD terms sharing atom one",
                  firstEnergy + secondEnergy, expectedGradients);
  CheckZeroTotalGradient("RESD shared-term total gradient",
                         CopyResdGradients(restraint),
                         EXACT_GRADIENT_TOLERANCE);
}

TEST_CASE("DistanceRestraintForceCalculatesCharacterizedDistanceExponents") {
  SECTION("IVALOne") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double distancePower = 2.0;
    const double deviation = distancePower - 0.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient = derivative * (0.0 - 2.0) / 2.0;

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD IVAL=1", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
  }

  SECTION("IVALTwo") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 2, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double distance = 2.0;
    const double distancePower = distance * distance;
    const double deviation = distancePower - 0.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient = derivative * 2.0 * (0.0 - 2.0);

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD IVAL=2", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
  }

  SECTION("IVALSix") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 6, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double distance = 2.0;
    const double distanceSquared = distance * distance;
    const double distanceFourth = distanceSquared * distanceSquared;
    const double distanceSixth = distanceFourth * distanceSquared;
    const double deviation = distanceSixth - 0.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient =
        derivative * 6.0 * distanceFourth * (0.0 - 2.0);

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD IVAL=6", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
  }
}

TEST_CASE("DistanceRestraintForceCalculatesCharacterizedEnergyExponents") {
  SECTION("EVALOne") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 1);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 1.0;
    const double expectedEnergy = (2.0 / 1.0) * deviation;
    const double derivative = 2.0;
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD EVAL=1", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }

  SECTION("EVALTwo") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 1.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD EVAL=2", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }

  SECTION("EVALThree") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 3);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 1.0;
    const double expectedEnergy =
        (2.0 / 3.0) * deviation * deviation * deviation;
    const double derivative = 2.0 * deviation * deviation;
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD EVAL=3", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative),
                    ANALYTICAL_ENERGY_TOLERANCE, EXACT_GRADIENT_TOLERANCE);
  }

  SECTION("EVALFour") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 4);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 1.0;
    const double expectedEnergy =
        (2.0 / 4.0) * deviation * deviation * deviation * deviation;
    const double derivative = 2.0 * deviation * deviation * deviation;
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD EVAL=4", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }
}

TEST_CASE("DistanceRestraintForceUsesSignedPowersForNegativeDeviation") {
  SECTION("OddEVALProducesNegativeEnergy") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 3.0, 1, 3);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 3.0;
    const double expectedEnergy =
        (2.0 / 3.0) * deviation * deviation * deviation;
    const double derivative = 2.0 * deviation * deviation;
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD negative D with odd EVAL", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative),
                    ANALYTICAL_ENERGY_TOLERANCE, EXACT_GRADIENT_TOLERANCE);
  }

  SECTION("EvenEVALProducesPositiveEnergyAndNegativeDerivative") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 3.0, 1, 4);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 3.0;
    const double expectedEnergy =
        (2.0 / 4.0) * deviation * deviation * deviation * deviation;
    const double derivative = 2.0 * deviation * deviation * deviation;
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD negative D with even EVAL", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }
}

TEST_CASE("DistanceRestraintForceImplementsPositiveActivation") {
  SECTION("InactiveForNegativeDeviation") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 3.0, 1, 2,
                           DistanceRestraintCondition::POSITIVE);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 3.0;
    REQUIRE(deviation < 0.0);
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD POSITIVE inactive", 0.0,
                    MakeExpectedXGradients(0.0, 0.0));
  }

  SECTION("ActiveAtBoundary") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 2.0, 1, 1,
                           DistanceRestraintCondition::POSITIVE);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 2.0;
    const double expectedEnergy = (2.0 / 1.0) * deviation;
    const double derivative = 2.0;
    REQUIRE(deviation == 0.0);
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD POSITIVE boundary", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }

  SECTION("ActiveForPositiveDeviation") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2,
                           DistanceRestraintCondition::POSITIVE);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 1.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    REQUIRE(deviation > 0.0);
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD POSITIVE active", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }
}

TEST_CASE("DistanceRestraintForceImplementsNegativeActivation") {
  SECTION("ActiveForNegativeDeviation") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 3.0, 1, 2,
                           DistanceRestraintCondition::NEGATIVE);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 3.0;
    const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
    const double derivative = 2.0 * deviation;
    REQUIRE(deviation < 0.0);
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD NEGATIVE active", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }

  SECTION("ActiveAtBoundary") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 2.0, 1, 1,
                           DistanceRestraintCondition::NEGATIVE);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 2.0;
    const double expectedEnergy = (2.0 / 1.0) * deviation;
    const double derivative = 2.0;
    REQUIRE(deviation == 0.0);
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD NEGATIVE boundary", expectedEnergy,
                    MakeExpectedXGradients(-derivative, derivative));
  }

  SECTION("InactiveForPositiveDeviation") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 1.0, 1, 2,
                           DistanceRestraintCondition::NEGATIVE);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 1.0;
    REQUIRE(deviation > 0.0);
    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD NEGATIVE inactive", 0.0,
                    MakeExpectedXGradients(0.0, 0.0));
  }
}

TEST_CASE("DistanceRestraintForceAppliesRequestedScale") {
  const std::array<double, 4> scales = {0.0, 0.5, 1.0, 2.0};

  for (const double scale : scales) {
    CAPTURE(scale);
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    AddStandardRestraint(restraint);
    restraint.setScale(scale);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    const double deviation = 2.0 - 1.0;
    const double unscaledEnergy = (2.0 / 2.0) * deviation * deviation;
    const double unscaledDerivative = 2.0 * deviation;
    const double expectedEnergy = scale * unscaledEnergy;
    const double firstAtomGradient = -scale * unscaledDerivative;

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD SCALE", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
  }
}

TEST_CASE("DistanceRestraintForceIsTranslationInvariant") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint, 2.0, 0.0);

  CudaContainer<float4> originXyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(1.0f, 2.0f, 3.0f, 0.0f),
      make_float4(9.0f, -4.0f, 7.0f, 0.0f)};
  CudaContainer<float4> translatedXyzq =
      std::vector<float4>{make_float4(7.0f, -11.0f, 13.0f, 0.0f),
                          make_float4(8.0f, -9.0f, 16.0f, 0.0f),
                          make_float4(16.0f, -15.0f, 20.0f, 0.0f)};

  const double dx = 0.0 - 1.0;
  const double dy = 0.0 - 2.0;
  const double dz = 0.0 - 3.0;
  const double distanceSquared = dx * dx + dy * dy + dz * dz;
  const double distance = std::sqrt(distanceSquared);
  const double deviation = distance - 0.0;
  const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double pairGradientCoefficient = derivative / distance;
  const double3 firstAtomGradient =
      make_double3(pairGradientCoefficient * dx, pairGradientCoefficient * dy,
                   pairGradientCoefficient * dz);
  const double3 secondAtomGradient = make_double3(
      -firstAtomGradient.x, -firstAtomGradient.y, -firstAtomGradient.z);
  const std::vector<double3> expectedGradients = {
      firstAtomGradient, secondAtomGradient, make_double3(0.0, 0.0, 0.0)};
  const std::array<double, 9> expectedVirial = {
      -dx * firstAtomGradient.x, -dx * firstAtomGradient.y,
      -dx * firstAtomGradient.z, -dy * firstAtomGradient.x,
      -dy * firstAtomGradient.y, -dy * firstAtomGradient.z,
      -dz * firstAtomGradient.x, -dz * firstAtomGradient.y,
      -dz * firstAtomGradient.z};

  ClearAndCalculate(restraint, originXyzq);
  CheckResdOutput(restraint, "RESD origin coordinates", expectedEnergy,
                  expectedGradients, ANALYTICAL_ENERGY_TOLERANCE,
                  ANALYTICAL_GRADIENT_TOLERANCE);
  const std::array<double, 9> originVirial =
      CalculateManagerStyleVirial(restraint, originXyzq, BOX_DIMENSIONS);
  CheckVirial("RESD origin virial", originVirial, expectedVirial,
              ANALYTICAL_VIRIAL_TOLERANCE);

  ClearAndCalculate(restraint, translatedXyzq);
  CheckResdOutput(restraint, "RESD translated coordinates", expectedEnergy,
                  expectedGradients, ANALYTICAL_ENERGY_TOLERANCE,
                  ANALYTICAL_GRADIENT_TOLERANCE);
  const std::array<double, 9> translatedVirial =
      CalculateManagerStyleVirial(restraint, translatedXyzq, BOX_DIMENSIONS);
  CheckVirial("RESD translated virial", translatedVirial, expectedVirial,
              ANALYTICAL_VIRIAL_TOLERANCE);
  CheckVirial("RESD translation-invariant virial", translatedVirial,
              originVirial, ANALYTICAL_VIRIAL_TOLERANCE);
}

TEST_CASE("DistanceRestraintForceInternalTermsConserveTotalGradient") {
  constexpr int numAtoms = 4;
  ResdForce restraint(numAtoms);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  restraint.addRestraint({{0, 1}, {1, 2}}, {1.0, -1.0}, 2.0, -2.0, 1, 2);
  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(2.0f, 0.0f, 0.0f, 0.0f),
      make_float4(5.0f, 0.0f, 0.0f, 0.0f),
      make_float4(11.0f, -7.0f, 3.0f, 0.0f)};
  const double r01 = 2.0;
  const double r12 = 3.0;
  const double deviation = r01 - r12 - (-2.0);
  const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double pair01FirstGradient = derivative * 1.0 * (0.0 - 2.0) / r01;
  const double pair01SecondGradient = -pair01FirstGradient;
  const double pair12FirstGradient = derivative * -1.0 * (2.0 - 5.0) / r12;
  const double pair12SecondGradient = -pair12FirstGradient;
  const std::vector<double3> expectedGradients = {
      make_double3(pair01FirstGradient, 0.0, 0.0),
      make_double3(pair01SecondGradient + pair12FirstGradient, 0.0, 0.0),
      make_double3(pair12SecondGradient, 0.0, 0.0),
      make_double3(0.0, 0.0, 0.0)};

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD internal-gradient conservation",
                  expectedEnergy, expectedGradients);
  const std::vector<double3> observedGradients = CopyResdGradients(restraint);
  CheckZeroTotalGradient("RESD zero total gradient", observedGradients,
                         EXACT_GRADIENT_TOLERANCE);
  apo_test::CheckVectorsClose<double3>(
      "RESD unrelated atom gradient", {observedGradients[3]},
      {make_double3(0.0, 0.0, 0.0)}, EXACT_GRADIENT_TOLERANCE);
}

TEST_CASE("DistanceRestraintForceHonorsCalculationFlags") {
  const double distance = 2.0;
  const double displacement = 0.0 - 2.0;
  const double deviation = distance - 1.0;
  const double evaluatedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double firstAtomGradient = derivative * displacement / distance;
  const std::vector<double3> expectedGradients =
      MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient);
  const std::array<double, 9> expectedVirial = {
      -displacement * firstAtomGradient,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0};

  SECTION("CalcEnergyFalse") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    AddStandardRestraint(restraint);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    ClearAndCalculate(restraint, xyzq, false, true);
    CheckResdOutput(restraint, "RESD calcEnergy=false", 0.0, expectedGradients);
    CheckShiftedForcesAreZero(restraint, "RESD calcEnergy=false shifted forces",
                              EXACT_VIRIAL_TOLERANCE);
    CheckVirial("RESD manager virial with calcEnergy=false",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, EXACT_VIRIAL_TOLERANCE);
  }

  SECTION("CalcVirialFalse") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    AddStandardRestraint(restraint);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    ClearAndCalculate(restraint, xyzq, true, false);
    CheckResdOutput(restraint, "RESD calcVirial=false", evaluatedEnergy,
                    expectedGradients);
    CheckVirial("RESD no direct virial with calcVirial=false",
                CopyStoredVirial(restraint), ZERO_VIRIAL,
                EXACT_VIRIAL_TOLERANCE);
    CheckShiftedForcesAreZero(restraint, "RESD calcVirial=false shifted forces",
                              EXACT_VIRIAL_TOLERANCE);
  }

  SECTION("CalcEnergyAndCalcVirialFalse") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    AddStandardRestraint(restraint);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;

    ClearAndCalculate(restraint, xyzq, false, false);
    CheckResdOutput(restraint, "RESD both optional outputs false", 0.0,
                    expectedGradients);
    CheckVirial("RESD no direct optional output virial",
                CopyStoredVirial(restraint), ZERO_VIRIAL,
                EXACT_VIRIAL_TOLERANCE);
    CheckShiftedForcesAreZero(restraint, "RESD both-false shifted forces",
                              EXACT_VIRIAL_TOLERANCE);
  }
}

TEST_CASE("DistanceRestraintForceMultipleEvaluationsHaveNoStaleAccumulation") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  AddStandardRestraint(restraint);

  CudaContainer<float4> firstXyzq = STANDARD_COORDINATES;
  const double firstDistance = 2.0;
  const double firstDeviation = firstDistance - 1.0;
  const double firstEnergy = (2.0 / 2.0) * firstDeviation * firstDeviation;
  const double firstDerivative = 2.0 * firstDeviation;
  ClearAndCalculate(restraint, firstXyzq);
  CheckResdOutput(restraint, "RESD first evaluation", firstEnergy,
                  MakeExpectedXGradients(-firstDerivative, firstDerivative));

  CudaContainer<float4> secondXyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(3.0f, 0.0f, 0.0f, 0.0f),
      make_float4(5.0f, 0.0f, 0.0f, 0.0f)};
  const double secondDeviation = 3.0 - 1.0;
  const double secondEnergy = secondDeviation * secondDeviation;
  const double secondDerivative = 2.0 * secondDeviation;
  ClearAndCalculate(restraint, secondXyzq);
  CheckResdOutput(restraint, "RESD second evaluation", secondEnergy,
                  MakeExpectedXGradients(-secondDerivative, secondDerivative));

  CudaContainer<float4> thirdXyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(1.5f, 0.0f, 0.0f, 0.0f),
      make_float4(5.0f, 0.0f, 0.0f, 0.0f)};
  const double thirdDeviation = 1.5 - 1.0;
  const double thirdEnergy = thirdDeviation * thirdDeviation;
  const double thirdDerivative = 2.0 * thirdDeviation;
  ClearAndCalculate(restraint, thirdXyzq);
  CheckResdOutput(restraint, "RESD third evaluation", thirdEnergy,
                  MakeExpectedXGradients(-thirdDerivative, thirdDerivative));
}

TEST_CASE("DistanceRestraintForceGradientsMatchCentralFiniteDifferences") {
  SECTION("MultiPairTerm") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint({{0, 1}, {1, 2}}, {1.0, 0.5}, 1.5, 0.25, 1, 2);
    const std::vector<float4> coordinates = {
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(1.0f, 2.0f, 0.0f, 0.0f),
        make_float4(3.0f, 2.0f, 1.0f, 0.0f)};

    const double distance = std::sqrt(5.0);
    const double deviation = distance + 0.5 * distance - 0.25;
    const double expectedEnergy = (1.5 / 2.0) * deviation * deviation;
    const double derivative = 1.5 * deviation;
    const double pair01Coefficient = derivative / distance;
    const double pair12Coefficient = 0.5 * pair01Coefficient;
    const std::vector<double3> expectedGradients = {
        make_double3(-pair01Coefficient, -2.0 * pair01Coefficient, 0.0),
        make_double3(pair01Coefficient - 2.0 * pair12Coefficient,
                     2.0 * pair01Coefficient, -pair12Coefficient),
        make_double3(2.0 * pair12Coefficient, 0.0, pair12Coefficient)};

    CheckCentralFiniteDifference(restraint, "RESD multi-pair finite difference",
                                 coordinates, expectedEnergy,
                                 expectedGradients);
  }

  SECTION("NegativeCoefficient") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, {-0.75}, 1.25, 0.5, 1, 2);
    const std::vector<float4> coordinates = {
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(1.0f, 2.0f, 2.0f, 0.0f),
        make_float4(7.0f, -3.0f, 5.0f, 0.0f)};

    const double distance = 3.0;
    const double deviation = -0.75 * distance - 0.5;
    const double expectedEnergy = (1.25 / 2.0) * deviation * deviation;
    const double derivative = 1.25 * deviation;
    const double pairGradientCoefficient = derivative * -0.75 / distance;
    const std::vector<double3> expectedGradients = {
        make_double3(-pairGradientCoefficient, -2.0 * pairGradientCoefficient,
                     -2.0 * pairGradientCoefficient),
        make_double3(pairGradientCoefficient, 2.0 * pairGradientCoefficient,
                     2.0 * pairGradientCoefficient),
        make_double3(0.0, 0.0, 0.0)};

    CheckCentralFiniteDifference(
        restraint, "RESD negative-coefficient finite difference", coordinates,
        expectedEnergy, expectedGradients);
  }

  SECTION("DistanceExponentGreaterThanOne") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, {0.5}, 1.5, 2.0, 2, 2);
    const std::vector<float4> coordinates = {
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(1.0f, 2.0f, 2.0f, 0.0f),
        make_float4(7.0f, -3.0f, 5.0f, 0.0f)};

    const double distanceSquared = 9.0;
    const double deviation = 0.5 * distanceSquared - 2.0;
    const double expectedEnergy = (1.5 / 2.0) * deviation * deviation;
    const double derivative = 1.5 * deviation;
    const double pairGradientCoefficient = derivative * 0.5 * 2.0;
    const std::vector<double3> expectedGradients = {
        make_double3(-pairGradientCoefficient, -2.0 * pairGradientCoefficient,
                     -2.0 * pairGradientCoefficient),
        make_double3(pairGradientCoefficient, 2.0 * pairGradientCoefficient,
                     2.0 * pairGradientCoefficient),
        make_double3(0.0, 0.0, 0.0)};

    CheckCentralFiniteDifference(restraint, "RESD IVAL>1 finite difference",
                                 coordinates, expectedEnergy,
                                 expectedGradients);
  }

  SECTION("OddEnergyExponent") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 1.5, 3.5, 1, 3);
    const std::vector<float4> coordinates = {
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(1.0f, 2.0f, 2.0f, 0.0f),
        make_float4(7.0f, -3.0f, 5.0f, 0.0f)};

    const double distance = 3.0;
    const double deviation = distance - 3.5;
    const double expectedEnergy =
        (1.5 / 3.0) * deviation * deviation * deviation;
    const double derivative = 1.5 * deviation * deviation;
    const double pairGradientCoefficient = derivative / distance;
    const std::vector<double3> expectedGradients = {
        make_double3(-pairGradientCoefficient, -2.0 * pairGradientCoefficient,
                     -2.0 * pairGradientCoefficient),
        make_double3(pairGradientCoefficient, 2.0 * pairGradientCoefficient,
                     2.0 * pairGradientCoefficient),
        make_double3(0.0, 0.0, 0.0)};

    CheckCentralFiniteDifference(restraint, "RESD odd-EVAL finite difference",
                                 coordinates, expectedEnergy,
                                 expectedGradients);
  }

  SECTION("ActiveOneSidedTerm") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 1.5, 2.0, 1, 2,
                           DistanceRestraintCondition::POSITIVE);
    const std::vector<float4> coordinates = {
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(1.0f, 2.0f, 2.0f, 0.0f),
        make_float4(7.0f, -3.0f, 5.0f, 0.0f)};

    const double distance = 3.0;
    const double deviation = distance - 2.0;
    REQUIRE(deviation > 0.0);
    const double expectedEnergy = (1.5 / 2.0) * deviation * deviation;
    const double derivative = 1.5 * deviation;
    const double pairGradientCoefficient = derivative / distance;
    const std::vector<double3> expectedGradients = {
        make_double3(-pairGradientCoefficient, -2.0 * pairGradientCoefficient,
                     -2.0 * pairGradientCoefficient),
        make_double3(pairGradientCoefficient, 2.0 * pairGradientCoefficient,
                     2.0 * pairGradientCoefficient),
        make_double3(0.0, 0.0, 0.0)};

    CheckCentralFiniteDifference(
        restraint, "RESD active-one-sided finite difference", coordinates,
        expectedEnergy, expectedGradients);
  }
}

TEST_CASE("DistanceRestraintForceUsesRawCoordinatesAcrossBoxFaces") {
  SECTION("PositiveBoxFaceCrossing") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(19.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(-19.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(0.0f, 7.0f, 0.0f, 0.0f)};

    const double rawDisplacement = 19.5 - (-19.5);
    const double rawDistance = std::abs(rawDisplacement);
    const double deviation = rawDistance - 0.0;
    const double expectedEnergy = deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient = derivative * rawDisplacement / rawDistance;
    const std::array<double, 9> expectedVirial = {
        -rawDisplacement * firstAtomGradient,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD positive box-face crossing", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
    CheckShiftedForcesAreZero(restraint,
                              "RESD positive crossing shifted forces",
                              EXACT_VIRIAL_TOLERANCE);
    CheckVirial("RESD positive box-face virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, EXACT_VIRIAL_TOLERANCE);
  }

  SECTION("NegativeBoxFaceCrossing") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(-19.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(19.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(0.0f, 7.0f, 0.0f, 0.0f)};

    const double rawDisplacement = -19.5 - 19.5;
    const double rawDistance = std::abs(rawDisplacement);
    const double deviation = rawDistance - 0.0;
    const double expectedEnergy = deviation * deviation;
    const double derivative = 2.0 * deviation;
    const double firstAtomGradient = derivative * rawDisplacement / rawDistance;
    const std::array<double, 9> expectedVirial = {
        -rawDisplacement * firstAtomGradient,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(
        restraint, "RESD negative box-face crossing", expectedEnergy,
        MakeExpectedXGradients(firstAtomGradient, -firstAtomGradient));
    CheckShiftedForcesAreZero(restraint,
                              "RESD negative crossing shifted forces",
                              EXACT_VIRIAL_TOLERANCE);
    CheckVirial("RESD negative box-face virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, EXACT_VIRIAL_TOLERANCE);
  }

  SECTION("TranslatingOneAtomByOneFullBoxChangesRawDistance") {
    ResdForce restraint(NUM_ATOMS);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> baseXyzq =
        std::vector<float4>{make_float4(-0.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(0.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(0.0f, 7.0f, 0.0f, 0.0f)};
    CudaContainer<float4> shiftedXyzq =
        std::vector<float4>{make_float4(39.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(0.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(0.0f, 7.0f, 0.0f, 0.0f)};

    const double baseDisplacement = -0.5 - 0.5;
    const double baseDistance = std::abs(baseDisplacement);
    const double baseEnergy = baseDistance * baseDistance;
    const double baseFirstGradient = 2.0 * baseDisplacement;
    const std::array<double, 9> baseVirial = {
        -baseDisplacement * baseFirstGradient,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0};

    ClearAndCalculate(restraint, baseXyzq);
    CheckResdOutput(
        restraint, "RESD before one-box atom translation", baseEnergy,
        MakeExpectedXGradients(baseFirstGradient, -baseFirstGradient));
    CheckVirial(
        "RESD before one-box atom translation virial",
        CalculateManagerStyleVirial(restraint, baseXyzq, BOX_DIMENSIONS),
        baseVirial, EXACT_VIRIAL_TOLERANCE);

    const double shiftedDisplacement = 39.5 - 0.5;
    const double shiftedDistance = std::abs(shiftedDisplacement);
    const double shiftedEnergy = shiftedDistance * shiftedDistance;
    const double shiftedFirstGradient = 2.0 * shiftedDisplacement;
    const std::array<double, 9> shiftedVirial = {-shiftedDisplacement *
                                                     shiftedFirstGradient,
                                                 0.0,
                                                 0.0,
                                                 0.0,
                                                 0.0,
                                                 0.0,
                                                 0.0,
                                                 0.0,
                                                 0.0};

    ClearAndCalculate(restraint, shiftedXyzq);
    CheckResdOutput(
        restraint, "RESD after one-box atom translation", shiftedEnergy,
        MakeExpectedXGradients(shiftedFirstGradient, -shiftedFirstGradient));
    CheckVirial(
        "RESD after one-box atom translation virial",
        CalculateManagerStyleVirial(restraint, shiftedXyzq, BOX_DIMENSIONS),
        shiftedVirial, EXACT_VIRIAL_TOLERANCE);
    CHECK(shiftedEnergy != baseEnergy);
  }

  SECTION("SeveralPairsUseDifferentRawImageShifts") {
    constexpr int numAtoms = 4;
    ResdForce restraint(numAtoms);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    const std::vector<std::array<int, 2>> atomPairs = {{0, 1}, {0, 2}, {0, 3}};
    restraint.addRestraint(atomPairs, {1.0, 1.0, 1.0}, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(41.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(0.0f, -42.0f, 0.0f, 0.0f),
                            make_float4(0.0f, 0.0f, 43.0f, 0.0f)};

    const double r01 = 41.0;
    const double r02 = 42.0;
    const double r03 = 43.0;
    const double deviation = r01 + r02 + r03;
    const double expectedEnergy = deviation * deviation;
    const double derivative = 2.0 * deviation;
    const std::vector<double3> expectedGradients = {
        make_double3(-derivative, derivative, -derivative),
        make_double3(derivative, 0.0, 0.0), make_double3(0.0, -derivative, 0.0),
        make_double3(0.0, 0.0, derivative)};
    const std::array<double, 9> expectedVirial = {
        -r01 * derivative, 0.0, 0.0, 0.0, -r02 * derivative, 0.0, 0.0, 0.0,
        -r03 * derivative};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "RESD several raw image shifts", expectedEnergy,
                    expectedGradients);
    CheckZeroTotalGradient("RESD image-shift total gradient",
                           CopyResdGradients(restraint),
                           EXACT_GRADIENT_TOLERANCE);
    CheckShiftedForcesAreZero(restraint, "RESD several-shift shifted forces",
                              EXACT_VIRIAL_TOLERANCE);
    CheckVirial("RESD several raw image shifts virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, EXACT_VIRIAL_TOLERANCE);
  }
}

TEST_CASE("DistanceRestraintForceVirialMatchesFiniteStrainDerivative") {
  ResdForce restraint(NUM_ATOMS);
  restraint.setBoxDimensions(BOX_DIMENSIONS);
  restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
  const std::vector<float4> coordinates = {
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(1.0f, 2.0f, 3.0f, 0.0f),
      make_float4(4.0f, -5.0f, 6.0f, 0.0f)};
  CudaContainer<float4> xyzq = coordinates;

  const double dx = 0.0 - 1.0;
  const double dy = 0.0 - 2.0;
  const double dz = 0.0 - 3.0;
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double deviation = distance - 0.0;
  const double expectedEnergy = (2.0 / 2.0) * deviation * deviation;
  const double derivative = 2.0 * deviation;
  const double pairGradientCoefficient = derivative / distance;
  const double3 firstAtomGradient =
      make_double3(pairGradientCoefficient * dx, pairGradientCoefficient * dy,
                   pairGradientCoefficient * dz);
  const double3 secondAtomGradient = make_double3(
      -firstAtomGradient.x, -firstAtomGradient.y, -firstAtomGradient.z);
  const std::vector<double3> expectedGradients = {
      firstAtomGradient, secondAtomGradient, make_double3(0.0, 0.0, 0.0)};
  const std::array<double, 9> expectedVirial = {
      -dx * firstAtomGradient.x, -dx * firstAtomGradient.y,
      -dx * firstAtomGradient.z, -dy * firstAtomGradient.x,
      -dy * firstAtomGradient.y, -dy * firstAtomGradient.z,
      -dz * firstAtomGradient.x, -dz * firstAtomGradient.y,
      -dz * firstAtomGradient.z};

  ClearAndCalculate(restraint, xyzq);
  CheckResdOutput(restraint, "RESD finite-strain center", expectedEnergy,
                  expectedGradients, ANALYTICAL_ENERGY_TOLERANCE,
                  ANALYTICAL_GRADIENT_TOLERANCE);
  const std::array<double, 9> observedVirial =
      CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS);
  CheckVirial("RESD finite-strain center virial", observedVirial,
              expectedVirial, ANALYTICAL_VIRIAL_TOLERANCE);

  for (int gradientComponent = 0; gradientComponent < 3; gradientComponent++) {
    for (int coordinateComponent = 0; coordinateComponent < 3;
         coordinateComponent++) {
      std::vector<float4> plusCoordinates = coordinates;
      std::vector<float4> minusCoordinates = coordinates;

      for (std::size_t atomIndex = 0; atomIndex < coordinates.size();
           atomIndex++) {
        const float originalGradientCoordinate =
            GetCoordinateComponent(coordinates[atomIndex], gradientComponent);
        const float originalCoordinate =
            GetCoordinateComponent(coordinates[atomIndex], coordinateComponent);
        SetCoordinateComponent(plusCoordinates[atomIndex], gradientComponent,
                               originalGradientCoordinate +
                                   static_cast<float>(FINITE_DIFFERENCE_STEP) *
                                       originalCoordinate);
        SetCoordinateComponent(minusCoordinates[atomIndex], gradientComponent,
                               originalGradientCoordinate -
                                   static_cast<float>(FINITE_DIFFERENCE_STEP) *
                                       originalCoordinate);
      }

      const double plusEnergy = EvaluateEnergy(restraint, plusCoordinates);
      const double minusEnergy = EvaluateEnergy(restraint, minusCoordinates);
      const double numericalStrainDerivative =
          (plusEnergy - minusEnergy) / (2.0 * FINITE_DIFFERENCE_STEP);
      const double expectedStrainDerivative =
          -observedVirial[coordinateComponent * 3 + gradientComponent];

      INFO("gradient component: " << gradientComponent);
      INFO("coordinate component: " << coordinateComponent);
      CHECK(numericalStrainDerivative ==
            Approx(expectedStrainDerivative).margin(FINITE_STRAIN_TOLERANCE));
    }
  }
}

TEST_CASE("DistanceRestraintForceMatchesCharacterizedCharmmReferences") {
  SECTION("OddNegativeDeviation") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 3.0, 1, 3);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(2.0f, 0.0f, 0.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-2.0, 0.0, 0.0), make_double3(2.0, 0.0, 0.0)};
    const std::array<double, 9> expectedVirial = {-4.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0,  0.0, 0.0, 0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 15", -0.666666667,
                    expectedGradients, CHARMM_ENERGY_TOLERANCE,
                    CHARMM_GRADIENT_TOLERANCE);
    CheckVirial("CHARMM case 15 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("EvenNegativeDeviation") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 3.0, 1, 4);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(2.0f, 0.0f, 0.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(2.0, 0.0, 0.0), make_double3(-2.0, 0.0, 0.0)};
    const std::array<double, 9> expectedVirial = {4.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 16", 0.5, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckVirial("CHARMM case 16 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("PositiveBoundary") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 2.0, 1, 1,
                           DistanceRestraintCondition::POSITIVE);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(2.0f, 0.0f, 0.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-2.0, 0.0, 0.0), make_double3(2.0, 0.0, 0.0)};
    const std::array<double, 9> expectedVirial = {-4.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0,  0.0, 0.0, 0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 18", 0.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckVirial("CHARMM case 18 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("NegativeBoundary") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 2.0, 1, 1,
                           DistanceRestraintCondition::NEGATIVE);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(2.0f, 0.0f, 0.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-2.0, 0.0, 0.0), make_double3(2.0, 0.0, 0.0)};
    const std::array<double, 9> expectedVirial = {-4.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0,  0.0, 0.0, 0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 21", 0.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckVirial("CHARMM case 21 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("OrthorhombicBoxFaceCrossing") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(-19.5f, 0.0f, 0.0f, 0.0f),
                            make_float4(19.5f, 0.0f, 0.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-78.0, 0.0, 0.0), make_double3(78.0, 0.0, 0.0)};
    const std::array<double, 9> expectedVirial = {-3042.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0,     0.0, 0.0, 0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 31", 1521.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckShiftedForcesAreZero(restraint, "CHARMM case 31 shifted-force records",
                              CHARMM_VIRIAL_TOLERANCE);
    CheckVirial("CHARMM case 31 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("SignedMultiPairTerm") {
    ResdForce restraint(3);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint({{0, 1}, {1, 2}}, {1.0, -1.0}, 2.0, -2.0, 1, 2);
    CudaContainer<float4> xyzq = STANDARD_COORDINATES;
    const std::vector<double3> expectedGradients = {
        make_double3(-2.0, 0.0, 0.0), make_double3(4.0, 0.0, 0.0),
        make_double3(-2.0, 0.0, 0.0)};
    const std::array<double, 9> expectedVirial = {2.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 35", 1.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE,
                    CHARMM_MULTIPAIR_GRADIENT_TOLERANCE);
    CheckVirial("CHARMM case 35 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("OffAxisVirialAtOrigin") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(1.0f, 2.0f, 3.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-2.0, -4.0, -6.0), make_double3(2.0, 4.0, 6.0)};
    const std::array<double, 9> expectedVirial = {
        -2.0, -4.0, -6.0, -4.0, -8.0, -12.0, -6.0, -12.0, -18.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 63", 14.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckShiftedForcesAreZero(restraint, "CHARMM case 63 shifted-force records",
                              CHARMM_VIRIAL_TOLERANCE);
    CheckVirial("CHARMM case 63 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("OffAxisVirialAfterTranslation") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(7.0f, -11.0f, 13.0f, 0.0f),
                            make_float4(8.0f, -9.0f, 16.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-2.0, -4.0, -6.0), make_double3(2.0, 4.0, 6.0)};
    const std::array<double, 9> expectedVirial = {
        -2.0, -4.0, -6.0, -4.0, -8.0, -12.0, -6.0, -12.0, -18.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 64", 14.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckShiftedForcesAreZero(restraint, "CHARMM case 64 shifted-force records",
                              CHARMM_VIRIAL_TOLERANCE);
    CheckVirial("CHARMM case 64 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }

  SECTION("DiagonalOrthorhombicRawCoordinates") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 2.0, 0.0, 1, 2);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(-19.5f, -20.0f, -20.5f, 0.0f),
                            make_float4(19.5f, 20.0f, 20.5f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-78.0, -80.0, -82.0), make_double3(78.0, 80.0, 82.0)};
    const std::array<double, 9> expectedVirial = {-3042.0, -3120.0, -3198.0,
                                                  -3120.0, -3200.0, -3280.0,
                                                  -3198.0, -3280.0, -3362.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 66", 4802.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckShiftedForcesAreZero(restraint, "CHARMM case 66 shifted-force records",
                              CHARMM_VIRIAL_TOLERANCE);
    CheckVirial("CHARMM case 66 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_LARGE_VIRIAL_TOLERANCE);
  }

  SECTION("MixedFourTermAccumulation") {
    ResdForce restraint(2);
    restraint.setBoxDimensions(BOX_DIMENSIONS);
    restraint.addRestraint(SINGLE_PAIR, {1.0}, 2.0, 1.0, 1, 2);
    restraint.addRestraint(SINGLE_PAIR, {0.5}, 3.0, 1.0, 2, 3,
                           DistanceRestraintCondition::POSITIVE);
    restraint.addRestraint(SINGLE_PAIR, {-1.0}, 4.0, -1.0, 1, 4,
                           DistanceRestraintCondition::NEGATIVE);
    restraint.addRestraint(SINGLE_PAIR, {1.0}, 5.0, 3.0, 1, 1,
                           DistanceRestraintCondition::POSITIVE);
    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(2.0f, 0.0f, 0.0f, 0.0f)};
    const std::vector<double3> expectedGradients = {
        make_double3(-12.0, 0.0, 0.0), make_double3(12.0, 0.0, 0.0)};
    const std::array<double, 9> expectedVirial = {-24.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0,   0.0, 0.0, 0.0};

    ClearAndCalculate(restraint, xyzq);
    CheckResdOutput(restraint, "CHARMM case 70", 3.0, expectedGradients,
                    CHARMM_ENERGY_TOLERANCE, CHARMM_GRADIENT_TOLERANCE);
    CheckVirial("CHARMM case 70 internal virial",
                CalculateManagerStyleVirial(restraint, xyzq, BOX_DIMENSIONS),
                expectedVirial, CHARMM_VIRIAL_TOLERANCE);
  }
}

TEST_CASE("DistanceRestraintForceSubscribesBeforeForceManagerInitialization",
          "[distance-restraint][integration][lifecycle]") {
  ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
  auto restraint =
      MakeStandardIntegrationRestraint(system.context->getNumAtoms());

  REQUIRE(system.forceManager->isInitialized() == false);
  CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager, restraint,
                                              "resd_preinitialize"));
  REQUIRE(system.forceManager->isInitialized() == false);

  InitializeResdIntegrationSystem(system);
  REQUIRE(restraint->getForce()->size() == INTEGRATION_NUM_ATOMS);

  const ResdManagerOutput withRestraint = CalculateResdManagerOutput(system);

  CHECK_NOTHROW(system.forceManager->unsubscribe(restraint));
  const ResdManagerOutput withoutRestraint = CalculateResdManagerOutput(system);

  CheckResdManagerContribution(
      "subscription before ForceManager initialization", withoutRestraint,
      withRestraint, 14.0,
      MakeAntisymmetricTwoAtomContribution(-2.0, -4.0, -6.0));
}

TEST_CASE("DistanceRestraintForceSubscribesAfterForceManagerInitialization",
          "[distance-restraint][integration][lifecycle]") {
  ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
  InitializeResdIntegrationSystem(system);
  REQUIRE(system.forceManager->isInitialized() == true);

  const ResdManagerOutput baseline = CalculateResdManagerOutput(system);
  auto restraint =
      MakeStandardIntegrationRestraint(system.context->getNumAtoms());

  CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager, restraint,
                                              "resd_postinitialize"));
  REQUIRE(restraint->getForce()->size() == INTEGRATION_NUM_ATOMS);

  const ResdManagerOutput withRestraint = CalculateResdManagerOutput(system);

  CheckResdManagerContribution(
      "subscription after ForceManager initialization", baseline, withRestraint,
      14.0, MakeAntisymmetricTwoAtomContribution(-2.0, -4.0, -6.0));

  CHECK_NOTHROW(system.forceManager->unsubscribe(restraint));
  const ResdManagerOutput afterCleanup = CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "post-initialization subscription cleanup", baseline, afterCleanup, 0.0,
      MakeAntisymmetricTwoAtomContribution(0.0, 0.0, 0.0));
}

TEST_CASE("DistanceRestraintForceManagerInitializationValidatesAtomCount",
          "[distance-restraint][integration][lifecycle]") {
  SECTION("SubscriptionBeforeInitialization") {
    ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
    auto restraint =
        MakeStandardIntegrationRestraint(INTEGRATION_NUM_ATOMS + 1);

    CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager, restraint,
                                                "resd_wrong_count_before"));

    apo_test::CheckApoCharmmError(
        [&system]() -> void {
          system.context->setBoxDimensions(INTEGRATION_BOX_DIMENSIONS);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Initialization atom count mismatch; expected 3, observed 2");

    CHECK(system.forceManager->isInitialized() == false);
    CHECK_NOTHROW(system.forceManager->unsubscribe(restraint));
  }

  SECTION("SubscriptionAfterInitialization") {
    ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
    InitializeResdIntegrationSystem(system);
    const ResdManagerOutput baseline = CalculateResdManagerOutput(system);
    auto restraint =
        MakeStandardIntegrationRestraint(INTEGRATION_NUM_ATOMS + 1);

    apo_test::CheckApoCharmmError(
        [&system, &restraint]() -> void {
          SubscribeIntegrationRestraint(system.forceManager, restraint,
                                        "resd_wrong_count_after");
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Initialization atom count mismatch; expected 3, observed 2");

    const ResdManagerOutput afterRejectedSubscription =
        CalculateResdManagerOutput(system);
    CheckResdManagerContribution(
        "rejected atom-count mismatch", baseline, afterRejectedSubscription,
        0.0, MakeAntisymmetricTwoAtomContribution(0.0, 0.0, 0.0));
  }
}

TEST_CASE("DistanceRestraintForceUpdatesWhileSubscribed",
          "[distance-restraint][integration][lifecycle]") {
  ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
  InitializeResdIntegrationSystem(system);
  const ResdManagerOutput baseline = CalculateResdManagerOutput(system);

  auto restraint =
      MakeStandardIntegrationRestraint(system.context->getNumAtoms());
  CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager, restraint,
                                              "resd_lifecycle"));

  const ResdManagerOutput initiallySubscribed =
      CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "initial subscribed term", baseline, initiallySubscribed, 14.0,
      MakeAntisymmetricTwoAtomContribution(-2.0, -4.0, -6.0));

  restraint->setScale(3.0);
  const ResdManagerOutput afterScaleChange = CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "SCALE changed after subscription", baseline, afterScaleChange, 42.0,
      MakeAntisymmetricTwoAtomContribution(-6.0, -12.0, -18.0));

  restraint->addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 4.0, 0.0, 1, 2);
  const ResdManagerOutput afterTermAddition =
      CalculateResdManagerOutput(system);

  CheckResdManagerContribution(
      "term added after subscription", baseline, afterTermAddition, 126.0,
      MakeAntisymmetricTwoAtomContribution(-18.0, -36.0, -54.0));

  restraint->reset();
  const ResdManagerOutput afterReset = CalculateResdManagerOutput(system);

  CheckResdManagerContribution(
      "RESET after subscription", baseline, afterReset, 0.0,
      MakeAntisymmetricTwoAtomContribution(0.0, 0.0, 0.0));

  AddStandardRestraint(*restraint, 2.0, 0.0);
  const ResdManagerOutput afterNewTerm = CalculateResdManagerOutput(system);

  CheckResdManagerContribution(
      "new term after RESET", baseline, afterNewTerm, 14.0,
      MakeAntisymmetricTwoAtomContribution(-2.0, -4.0, -6.0));

  CHECK_NOTHROW(system.forceManager->unsubscribe(restraint));
  const ResdManagerOutput afterUnsubscription =
      CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "unsubscription removes active contribution", baseline,
      afterUnsubscription, 0.0,
      MakeAntisymmetricTwoAtomContribution(0.0, 0.0, 0.0));

  CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager, restraint,
                                              "resd_lifecycle"));
  const ResdManagerOutput afterResubscription =
      CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "resubscription restores active contribution", baseline,
      afterResubscription, 14.0,
      MakeAntisymmetricTwoAtomContribution(-2.0, -4.0, -6.0));

  CHECK_NOTHROW(system.forceManager->unsubscribe(restraint));
  const ResdManagerOutput afterCleanup = CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "lifecycle cleanup", baseline, afterCleanup, 0.0,
      MakeAntisymmetricTwoAtomContribution(0.0, 0.0, 0.0));
}

TEST_CASE("DistanceRestraintForceReceivesBoxChangesWhileSubscribed",
          "[distance-restraint][integration][lifecycle]") {
  ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
  InitializeResdIntegrationSystem(system);

  auto restraint =
      MakeStandardIntegrationRestraint(system.context->getNumAtoms());
  CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager, restraint,
                                              "resd_box_update"));

  CHECK_NOTHROW(
      system.context->setBoxDimensions(UPDATED_INTEGRATION_BOX_DIMENSIONS));
  apo_test::CheckVectorsClose<double>("updated CharmmContext box",
                                      system.context->getBoxDimensions(),
                                      UPDATED_INTEGRATION_BOX_DIMENSIONS, 0.0);
  apo_test::CheckVectorsClose<double>("updated ForceManager box",
                                      system.forceManager->getBoxDimensions(),
                                      UPDATED_INTEGRATION_BOX_DIMENSIONS, 0.0);

  const ResdManagerOutput withRestraintInUpdatedBox =
      CalculateResdManagerOutput(system);

  CHECK_NOTHROW(system.forceManager->unsubscribe(restraint));
  const ResdManagerOutput withoutRestraintInUpdatedBox =
      CalculateResdManagerOutput(system);

  CheckResdManagerContribution(
      "box update reaches subscribed RESD without minimum imaging",
      withoutRestraintInUpdatedBox, withRestraintInUpdatedBox, 14.0,
      MakeAntisymmetricTwoAtomContribution(-2.0, -4.0, -6.0));
}

TEST_CASE("DistanceRestraintForceSupportsMultipleSubscribedObjects",
          "[distance-restraint][integration][lifecycle]") {
  ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
  InitializeResdIntegrationSystem(system);
  const ResdManagerOutput baseline = CalculateResdManagerOutput(system);

  auto firstRestraint =
      MakeStandardIntegrationRestraint(system.context->getNumAtoms());
  auto secondRestraint =
      std::make_shared<ResdForce>(system.context->getNumAtoms());
  secondRestraint->addRestraint(SINGLE_PAIR, UNIT_COEFFICIENT, 4.0, 0.0, 1, 2);

  CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager,
                                              firstRestraint, "resd_primary"));
  CHECK_NOTHROW(SubscribeIntegrationRestraint(
      system.forceManager, secondRestraint, "resd_secondary"));

  const ResdManagerOutput withBothRestraints =
      CalculateResdManagerOutput(system);

  CheckResdManagerContribution(
      "two DistanceRestraintForce objects", baseline, withBothRestraints, 42.0,
      MakeAntisymmetricTwoAtomContribution(-6.0, -12.0, -18.0));

  CHECK_NOTHROW(system.forceManager->unsubscribe("resd_primary"));
  const ResdManagerOutput withSecondRestraintOnly =
      CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "secondary object remains after primary tag removal", baseline,
      withSecondRestraintOnly, 28.0,
      MakeAntisymmetricTwoAtomContribution(-4.0, -8.0, -12.0));

  CHECK_NOTHROW(system.forceManager->unsubscribe(secondRestraint));
  const ResdManagerOutput afterCleanup = CalculateResdManagerOutput(system);
  CheckResdManagerContribution(
      "multiple-object cleanup", baseline, afterCleanup, 0.0,
      MakeAntisymmetricTwoAtomContribution(0.0, 0.0, 0.0));
}

TEST_CASE("DistanceRestraintForcePropagatesWithLangevinThermostat",
          "[distance-restraint][integration][propagation]") {
  ResdIntegrationSystem system = MakeUninitializedResdIntegrationSystem();
  InitializeResdIntegrationSystem(system);

  auto restraint =
      MakeStandardIntegrationRestraint(system.context->getNumAtoms());
  CHECK_NOTHROW(SubscribeIntegrationRestraint(system.forceManager, restraint,
                                              "resd_propagation"));

  system.context->setRandomSeed(INTEGRATION_RANDOM_SEED);
  system.context->assignVelocitiesAtTemperature(INTEGRATION_TEMPERATURE);

  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(INTEGRATION_TIME_STEP);
  integrator->setReferenceTemperature(INTEGRATION_TEMPERATURE);
  integrator->setThermostatFriction(INTEGRATION_THERMOSTAT_FRICTION);
  integrator->setThermostatRngSeed(INTEGRATION_RANDOM_SEED);
  integrator->setCharmmContext(system.context);

  CHECK_NOTHROW(integrator->propagate(INTEGRATION_PROPAGATION_STEPS));
  CHECK(integrator->getCurrentPropagatedStep() ==
        INTEGRATION_PROPAGATION_STEPS);
  CHECK(integrator->getTotNumSteps() ==
        static_cast<unsigned long long int>(INTEGRATION_PROPAGATION_STEPS));

  apo_test::CheckFiniteTemperature(system.context->computeTemperature());
  apo_test::CheckFiniteTemperature(integrator->getInstantaneousTemperature());

  const std::vector<double4> finalCoordinates =
      apo_test::CopyToHost<double4>(system.context->getCoordinatesChargesDP());
  REQUIRE(finalCoordinates.size() == INTEGRATION_NUM_ATOMS);
  for (std::size_t i = 0; i < finalCoordinates.size(); i++) {
    INFO("propagated atom index: " << i);
    CHECK(std::isfinite(finalCoordinates[i].x));
    CHECK(std::isfinite(finalCoordinates[i].y));
    CHECK(std::isfinite(finalCoordinates[i].z));
    CHECK(std::isfinite(finalCoordinates[i].w));
  }

  CHECK_NOTHROW(system.forceManager->unsubscribe(restraint));
}
