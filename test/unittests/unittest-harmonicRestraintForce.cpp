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
#include "AtomSelection.h"
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "ForceManager.h"
#include "HarmonicRestraintForce.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <vector_types.h>

namespace {

constexpr int NUM_ATOMS = 3;
constexpr int RANDOM_SEED = 314159;
constexpr double TEMPERATURE = 300.0;
constexpr double TIME_STEP = 0.001;
constexpr double LANGEVIN_THERMOSTAT_FRICTION = 0.0;
constexpr double FORCE_TOLERANCE = 0.0;
constexpr double ENERGY_TOLERANCE = 0.0;
constexpr double INV_FORCE_SCALE_TEST = 1.0 / static_cast<double>(1LL << 40);
constexpr double FLOAT_EPSILON = 1.0 / static_cast<double>(1 << 23);

const std::vector<double> BOX_DIMENSIONS = {50.0, 50.0, 50.0};

AtomSelection MakeSelection(const int numAtoms,
                            const std::vector<int> &selectedAtoms) {
  AtomSelection selection(numAtoms, AtomSelection::InitialValue::NONE);

  for (const int atomIndex : selectedAtoms)
    selection.set(atomIndex);

  return selection;
}

std::shared_ptr<CharmmPSF> MakePsf(const int numAtoms) {
  auto psf = std::make_shared<CharmmPSF>();
  psf->setNumAtoms(numAtoms);
  return psf;
}

std::vector<double3>
CopyFixedPointForceToHost(const std::shared_ptr<Force<long long int>> &force) {
  std::vector<long long int> fx(force->size());
  std::vector<long long int> fy(force->size());
  std::vector<long long int> fz(force->size());

  force->getXYZ(fx.data(), fy.data(), fz.data());

  std::vector<double3> hostForces(force->size());
  for (std::size_t i = 0; i < hostForces.size(); i++) {
    hostForces[i].x = static_cast<double>(fx[i]) * INV_FORCE_SCALE_TEST;
    hostForces[i].y = static_cast<double>(fy[i]) * INV_FORCE_SCALE_TEST;
    hostForces[i].z = static_cast<double>(fz[i]) * INV_FORCE_SCALE_TEST;
  }

  return hostForces;
}

template <typename CT>
double GetHarmonicEnergy(HarmonicRestraintForce<long long int, CT> &restraint) {
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  restraint.getEnergyVirial()->copyToHost(*restraint.getStream());
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  return restraint.getEnergyVirial()->getEnergy("harm");
}

} // namespace

TEST_CASE("HarmonicRestraintForceConstructorAndDefaults") {
  auto psf = MakePsf(NUM_ATOMS);
  const std::weak_ptr<CharmmPSF> weakPsf = psf;
  HarmonicRestraintForce<long long int, float> restraint(psf);

  // Only the atom count is copied; the PSF is not retained.
  psf->setNumAtoms(NUM_ATOMS + 1);
  psf.reset();
  CHECK(weakPsf.expired());

  CHECK(decltype(restraint)::contributesVirial == false);
  CHECK(std::string(decltype(restraint)::energyTerm) == "harm");
  CHECK(restraint.getStream() != nullptr);
  CHECK(restraint.getForce() != nullptr);
  CHECK(restraint.getForce()->size() == NUM_ATOMS);
  CHECK(restraint.getForce()->stride() >= NUM_ATOMS);
  CHECK(restraint.getEnergyVirial() != nullptr);

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 2.0f, 3.0f, 0.0f), make_float4(4.0f, 5.0f, 6.0f, 0.0f),
      make_float4(7.0f, 8.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHarmonicEnergy(restraint) == Approx(0.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicRestraintForceConstructionAndInitializationValidation") {
  apo_test::CheckApoCharmmError(
      []() -> void {
        (void)HarmonicRestraintForce<long long int, float>(nullptr);
      },
      ApoCharmmErrorCode::InvalidArgument, "CharmmPSF must not be null");

  apo_test::CheckApoCharmmError(
      []() -> void {
        (void)HarmonicRestraintForce<long long int, float>(
            std::make_shared<CharmmPSF>());
      },
      ApoCharmmErrorCode::NotInitialized,
      "CharmmPSF atom count is not initialized; observed -1");

  apo_test::CheckApoCharmmError(
      []() -> void {
        (void)HarmonicRestraintForce<long long int, float>(MakePsf(0));
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Atom count must be positive; observed 0");

  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

  CHECK_NOTHROW(restraint.setBoxDimensions(BOX_DIMENSIONS));
  CHECK_NOTHROW(
      restraint.calcForce(xyzq.getDeviceArray().data(), false, false));
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.initialize(NUM_ATOMS + 1, BOX_DIMENSIONS);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Initialization atom count mismatch; expected 3, observed 4");

  CHECK_NOTHROW(restraint.initialize(NUM_ATOMS, BOX_DIMENSIONS));
}

TEST_CASE("HarmonicRestraintForceSelectionAndForceConstantValidation") {
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  const double infinity = std::numeric_limits<double>::infinity();

  AtomSelection wrongSelection(NUM_ATOMS + 1);
  apo_test::CheckApoCharmmError(
      [&restraint, &wrongSelection]() -> void {
        restraint.setSelection(wrongSelection);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Selection atom count mismatch; expected 3, observed 4");

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setForceConstant(infinity);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force constant must be finite; observed " + std::to_string(infinity));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void { restraint.setForceConstant(-1.0); },
      ApoCharmmErrorCode::InvalidArgument,
      "Force constant must be non-negative; observed " + std::to_string(-1.0));

  CHECK_NOTHROW(restraint.setForceConstant(0.0));
  CHECK_NOTHROW(restraint.setForceConstant(1.0));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void { restraint.setForceConstants({1.0, 2.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Force-constant array size mismatch; expected 3, observed 2");

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.setForceConstants({1.0, 2.0, 3.0, 4.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force-constant array size mismatch; expected 3, observed 4");

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setForceConstants({1.0, infinity, 3.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force constant at index 1 must be finite; observed " +
          std::to_string(infinity));

  CHECK_NOTHROW(restraint.setForceConstants({1.0, 2.0, -1.0}));
  CHECK_NOTHROW(restraint.setForceConstants({0.0, 1.0, 2.0}));
}

TEST_CASE("HarmonicRestraintForceReferenceCoordinateAndMassValidation") {
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  const double infinity = std::numeric_limits<double>::infinity();

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.setReferenceCoordinates(std::vector<double3>{
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference-coordinate array size mismatch; expected 3, observed 2");

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.setReferenceCoordinates(
            std::vector<std::vector<double>>{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference-coordinate array size mismatch; expected 3, observed 2");

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.setReferenceCoordinates(std::vector<std::vector<double>>{
            {0.0, 0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference coordinate at atom index 1 has invalid size; expected 3, "
      "observed 2");

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setReferenceCoordinates(std::vector<double3>{
            make_double3(infinity, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
            make_double3(0.0, 0.0, 0.0)});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference coordinate at atom index 0, X component must be finite; "
      "observed " +
          std::to_string(infinity));

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setReferenceCoordinates(std::vector<double3>{
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, infinity, 0.0),
            make_double3(0.0, 0.0, 0.0)});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference coordinate at atom index 1, Y component must be finite; "
      "observed " +
          std::to_string(infinity));

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setReferenceCoordinates(std::vector<std::vector<double>>{
            {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, infinity}});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Reference coordinate at atom index 2, Z component must be finite; "
      "observed " +
          std::to_string(infinity));

  CHECK_NOTHROW(
      restraint.setReferenceCoordinates(std::vector<std::vector<double>>{
          {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}}));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void { restraint.setMasses({1.0, 2.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Mass array size mismatch; expected 3, observed 2");

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setMasses({1.0, infinity, 3.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Mass at index 1 must be finite; observed " + std::to_string(infinity));

  CHECK_NOTHROW(restraint.setMasses({1.0, -1.0, 3.0}));
  CHECK_NOTHROW(restraint.setMasses({0.0, 1.0, 2.0}));
}

TEST_CASE("HarmonicRestraintForceExponentAndScaleFactorValidation") {
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  const double infinity = std::numeric_limits<double>::infinity();

  CHECK_NOTHROW(restraint.setExponent(-1));
  CHECK_NOTHROW(restraint.setExponent(0));
  CHECK_NOTHROW(restraint.setExponent(7));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void { restraint.setScaleFactors({1.0, 1.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Scale-factor array size mismatch; expected 3, observed 2");

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.setScaleFactors({1.0, 1.0, 1.0, 1.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Scale-factor array size mismatch; expected 3, observed 4");

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setScaleFactors({1.0, infinity, 1.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Scale factor at index 1 must be finite; observed " +
          std::to_string(infinity));

  CHECK_NOTHROW(restraint.setScaleFactors({2.0, 0.0, -1.0}));
}

TEST_CASE("HarmonicRestraintForceBoxDimensionValidation") {
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  const double infinity = std::numeric_limits<double>::infinity();

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void { restraint.setBoxDimensions({50.0, 50.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box-dimension array size mismatch; expected 3, observed 2");

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.setBoxDimensions({50.0, 50.0, 50.0, 50.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Box-dimension array size mismatch; expected 3, observed 4");

  apo_test::CheckApoCharmmError(
      [&restraint, infinity]() -> void {
        restraint.setBoxDimensions({50.0, infinity, 50.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimension at index 1 must be finite; observed " +
          std::to_string(infinity));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void { restraint.setBoxDimensions({50.0, 0.0, 50.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimension at index 1 must be positive; observed " +
          std::to_string(0.0));

  apo_test::CheckApoCharmmError(
      [&restraint]() -> void {
        restraint.setBoxDimensions({50.0, -1.0, 50.0});
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimension at index 1 must be positive; observed " +
          std::to_string(-1.0));

  CHECK_NOTHROW(restraint.setBoxDimensions(BOX_DIMENSIONS));
}

TEST_CASE("HarmonicRestraintForceForceManagerSubscriptionValidation") {
  ForceManager forceManager;
  auto restraint =
      std::make_shared<HarmonicRestraintForce<long long int, float>>(
          MakePsf(NUM_ATOMS));

  apo_test::CheckApoCharmmError(
      [&forceManager, &restraint]() -> void {
        forceManager.subscribe(restraint, "", restraint->getStream(),
                               restraint->getForce(),
                               restraint->getEnergyVirial());
      },
      ApoCharmmErrorCode::InvalidArgument, "Force tag must not be empty");

  CHECK_NOTHROW(forceManager.subscribe(
      restraint, "harm", restraint->getStream(), restraint->getForce(),
      restraint->getEnergyVirial()));

  apo_test::CheckApoCharmmError(
      [&forceManager, &restraint]() -> void {
        forceManager.subscribe(restraint, "harm", restraint->getStream(),
                               restraint->getForce(),
                               restraint->getEnergyVirial());
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force is already subscribed to this ForceManager");

  CHECK_NOTHROW(forceManager.unsubscribe(restraint));

  apo_test::CheckApoCharmmError(
      [&forceManager, &restraint]() -> void {
        forceManager.unsubscribe(restraint);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force is not subscribed to this ForceManager");
}

TEST_CASE("HarmonicRestraintForceCalculatesForceAndEnergy") {
  const AtomSelection selection = MakeSelection(NUM_ATOMS, {0, 1});

  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  restraint.setSelection(selection);
  restraint.setForceConstants({2.0, 0.5, 99.0});
  restraint.setMasses({1.5, 2.0, 5.0});
  restraint.setReferenceCoordinates(std::vector<double3>{
      make_double3(0.0, 0.0, 0.0), make_double3(1.0, 1.0, 1.0),
      make_double3(10.0, 10.0, 10.0)});
  restraint.initialize(NUM_ATOMS, BOX_DIMENSIONS);

  CudaContainer<float4> xyzq =
      std::vector<float4>{make_float4(1.0f, -2.0f, 0.5f, 0.0f),
                          make_float4(2.0f, 3.0f, -1.0f, 0.0f),
                          make_float4(9.0f, 8.0f, 7.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(6.0, -12.0, 3.0),
                                               make_double3(2.0, 4.0, -4.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);

  constexpr double expectedEnergy = 24.75;
  CHECK(GetHarmonicEnergy(restraint) ==
        Approx(expectedEnergy).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicRestraintForceSelectionMakesScalarForceConstant") {
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));

  restraint.setReferenceCoordinates(std::vector<double3>{
      make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
      make_double3(0.0, 0.0, 0.0)});
  restraint.setSelection(MakeSelection(NUM_ATOMS, {1}));
  restraint.setForceConstant(3.0);

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 2.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 3.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 12.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);

  constexpr double expectedEnergy = 12.0;
  CHECK(GetHarmonicEnergy(restraint) ==
        Approx(expectedEnergy).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicRestraintForceClearResetsForcesAndEnergy") {
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));

  restraint.setForceConstant(1.0);
  restraint.setReferenceCoordinates(std::vector<double3>{
      make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
      make_double3(0.0, 0.0, 0.0)});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 1.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 1.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  restraint.clear();
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);

  constexpr double expectedEnergy = 0.0;
  CHECK(GetHarmonicEnergy(restraint) ==
        Approx(expectedEnergy).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicRestraintForceFollowsCharmmExponents") {
  struct ExponentCase {
    int exponent;
    double energy;
    double firstFactor;
    double secondFactor;
  };

  // Atom 0 has C = 2 * 1.5 = 3 and S = 9, and atom 1 has C = -1 * 2 = -2 and
  // S = 1. Each gradient is the displacement times the factor DF of CHARMM's
  // branch for the exponent.
  const std::vector<ExponentCase> cases = {
      {2, 25.0, 6.0, -4.0},     {4, 241.0, 108.0, -8.0},
      {3, 79.0, 27.0, -6.0},    {1, 7.0, 1.0, -2.0},
      {0, 0.0, 0.0, 0.0},       {-3, 0.0, 0.0, 0.0},
      {5, 727.0, 405.0, -10.0}, {6, 2185.0, 1458.0, -12.0}};

  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstants({2.0, -1.0, 4.0});
  restraint.setMasses({1.5, 2.0, 1.0});
  restraint.setReferenceCoordinates(std::vector<double3>{
      make_double3(0.0, 0.0, 0.0), make_double3(1.0, 1.0, 1.0),
      make_double3(0.0, 0.0, 0.0)});

  // CHARMM skips atoms whose constant is zero, so the unselected atom's
  // coordinates never enter the arithmetic.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, -2.0f, 2.0f, 0.0f), make_float4(2.0f, 1.0f, 1.0f, 0.0f),
      make_float4(nan, nan, nan, 0.0f)};

  for (const ExponentCase &exponentCase : cases) {
    INFO("exponent: " << exponentCase.exponent);

    restraint.setExponent(exponentCase.exponent);
    restraint.clear();
    restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
    cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

    const double first = exponentCase.firstFactor;
    const double second = exponentCase.secondFactor;
    const std::vector<double3> expectedForces = {
        make_double3(first, -2.0 * first, 2.0 * first),
        make_double3(second, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)};

    apo_test::CheckVectorsClose<double3>(
        CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
        8.0 * FLOAT_EPSILON * 2.0 * std::abs(first));
    CHECK(GetHarmonicEnergy(restraint) ==
          Approx(exponentCase.energy)
              .margin(8.0 * FLOAT_EPSILON * std::abs(exponentCase.energy)));
  }
}

TEST_CASE("HarmonicRestraintForceFollowsCharmmScaleFactors") {
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(3.0);
  restraint.setReferenceCoordinates(std::vector<double3>{
      make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
      make_double3(0.0, 0.0, 0.0)});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 2.0f, 2.0f, 0.0f), make_float4(3.0f, 5.0f, 4.0f, 0.0f),
      make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

  // With C = 3, the harmonic branch gives E = 3 * S and the gradient
  // components d_a * 6 * s_a.
  SECTION("ScaledHarmonic") {
    restraint.setScaleFactors({2.0, 0.5, 1.25});

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void { restraint.setScaleFactors({1.0, 1.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Scale-factor array size mismatch; expected 3, observed 2");

    restraint.clear();
    restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
    cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

    // S = 9 for atom 0 and S = 50.5 for atom 1.
    const std::vector<double3> expectedForces = {make_double3(12.0, 6.0, 15.0),
                                                 make_double3(36.0, 15.0, 30.0),
                                                 make_double3(0.0, 0.0, 0.0)};

    apo_test::CheckVectorsClose<double3>(
        CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
        FORCE_TOLERANCE);
    CHECK(GetHarmonicEnergy(restraint) ==
          Approx(178.5).margin(ENERGY_TOLERANCE));
  }

  SECTION("NegativeAxisHarmonic") {
    restraint.setScaleFactors({1.0, -1.0, 1.0});

    restraint.clear();
    restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
    cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

    // S = 1 for atom 0 and S = 0 for atom 1, whose gradient is still nonzero.
    const std::vector<double3> expectedForces = {
        make_double3(6.0, -12.0, 12.0), make_double3(18.0, -30.0, 24.0),
        make_double3(0.0, 0.0, 0.0)};

    apo_test::CheckVectorsClose<double3>(
        CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
        FORCE_TOLERANCE);
    CHECK(GetHarmonicEnergy(restraint) == Approx(3.0).margin(ENERGY_TOLERANCE));
  }

  SECTION("ZeroAxisLinear") {
    restraint.setExponent(1);
    restraint.setScaleFactors({1.0, 0.0, 1.0});

    restraint.clear();
    restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
    cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

    // S = 5 for atom 0 and S = 25 for atom 1. The linear branch gives
    // E = 3 * sqrt(S) and the gradient components d_a * (3 / sqrt(S)) * s_a.
    const double root5 = std::sqrt(5.0);
    const std::vector<double3> expectedForces = {
        make_double3(3.0 / root5, 0.0, 6.0 / root5),
        make_double3(1.8, 0.0, 2.4), make_double3(0.0, 0.0, 0.0)};

    apo_test::CheckVectorsClose<double3>(
        CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
        8.0 * FLOAT_EPSILON * 3.0);
    CHECK(GetHarmonicEnergy(restraint) ==
          Approx(3.0 * root5 + 15.0).margin(8.0 * FLOAT_EPSILON * 25.0));
  }
}

TEST_CASE("HarmonicRestraintForceFollowsCharmmLinearTolerance") {
  // CHARMM's linear branch uses DF = C / sqrt(S) only when S > 1e-12 and a
  // zero gradient otherwise.
  HarmonicRestraintForce<long long int, float> restraint(MakePsf(NUM_ATOMS));
  restraint.setExponent(1);
  restraint.setForceConstant(3.0);
  restraint.setReferenceCoordinates(std::vector<double3>{
      make_double3(-1.0e-7, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
      make_double3(0.0, 1.0e-5, 0.0)});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  // S = 1e-14 for atom 0, S = 0 for atom 1, and S = 1e-10 for atom 2.
  const std::vector<double3> expectedForces = {make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, -3.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      8.0 * FLOAT_EPSILON * 3.0);
  CHECK(GetHarmonicEnergy(restraint) ==
        Approx(3.03e-5).margin(8.0 * FLOAT_EPSILON * 3.03e-5));
}

TEST_CASE("HarmonicRestraintForceArithmeticFollowsCT") {
  // A reference offset of 2^-30 angstrom is exact in double arithmetic and
  // rounds away when the reference is converted to float.
  const double offset = std::ldexp(1.0, -30);
  const double forceConstant = std::ldexp(1.0, 20);
  const AtomSelection selection = MakeSelection(NUM_ATOMS, {0});
  const std::vector<double3> referenceCoordinates = {
      make_double3(1.0 + offset, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
      make_double3(0.0, 0.0, 0.0)};

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

  HarmonicRestraintForce<long long int, double> doubleRestraint(
      MakePsf(NUM_ATOMS));
  doubleRestraint.setSelection(selection);
  doubleRestraint.setForceConstant(forceConstant);
  doubleRestraint.setReferenceCoordinates(referenceCoordinates);
  doubleRestraint.clear();
  doubleRestraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*doubleRestraint.getStream()));

  // The gradient 2 * 2^20 * -2^-30 = -2^-9 and the energy 2^20 * 2^-60 are
  // exact.
  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(doubleRestraint.getForce()),
      {make_double3(-std::ldexp(1.0, -9), 0.0, 0.0),
       make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)},
      FORCE_TOLERANCE);
  CHECK(GetHarmonicEnergy(doubleRestraint) == std::ldexp(1.0, -40));

  HarmonicRestraintForce<long long int, float> floatRestraint(
      MakePsf(NUM_ATOMS));
  floatRestraint.setSelection(selection);
  floatRestraint.setForceConstant(forceConstant);
  floatRestraint.setReferenceCoordinates(referenceCoordinates);
  floatRestraint.clear();
  floatRestraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*floatRestraint.getStream()));

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(floatRestraint.getForce()),
      {make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
       make_double3(0.0, 0.0, 0.0)},
      FORCE_TOLERANCE);
  CHECK(GetHarmonicEnergy(floatRestraint) == 0.0);
}

TEST_CASE("HarmonicRestraintForceSumsEnergyInDouble") {
  // The atom energies 2^24 and 1 are exact in float, but a float sum rounds
  // them to 2^24. Each atom's energy is evaluated in CT, and the sums are
  // double for both specializations.
  const std::vector<double> forceConstants = {std::ldexp(1.0, 18), 1.0, 0.0};
  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(8.0f, 0.0f, 0.0f, 0.0f), make_float4(1.0f, 0.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
  const std::vector<double3> expectedForces = {
      make_double3(std::ldexp(1.0, 22), 0.0, 0.0), make_double3(2.0, 0.0, 0.0),
      make_double3(0.0, 0.0, 0.0)};

  HarmonicRestraintForce<long long int, float> floatRestraint(
      MakePsf(NUM_ATOMS));
  floatRestraint.setForceConstants(forceConstants);
  floatRestraint.clear();
  floatRestraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*floatRestraint.getStream()));

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(floatRestraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHarmonicEnergy(floatRestraint) == std::ldexp(1.0, 24) + 1.0);

  HarmonicRestraintForce<long long int, double> doubleRestraint(
      MakePsf(NUM_ATOMS));
  doubleRestraint.setForceConstants(forceConstants);
  doubleRestraint.clear();
  doubleRestraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*doubleRestraint.getStream()));

  apo_test::CheckVectorsClose<double3>(
      CopyFixedPointForceToHost(doubleRestraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHarmonicEnergy(doubleRestraint) == std::ldexp(1.0, 24) + 1.0);
}

TEST_CASE("HarmonicRestraintForceCanBeSubscribedToForceManager") {
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

  auto restraint =
      std::make_shared<HarmonicRestraintForce<long long int, float>>(psf);
  restraint->setReferenceCoordinates(crd->getCoordinatesDP());
  restraint->setMasses(psf->getMasses());
  restraint->setSelection(MakeSelection(ctx->getNumAtoms(), {0, 1}));
  restraint->setForceConstant(10.0);

  auto fm = ctx->getForceManager();
  CHECK_NOTHROW(fm->subscribe(restraint, "harm", restraint->getStream(),
                              restraint->getForce(),
                              restraint->getEnergyVirial()));

  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  integrator->setReferenceTemperature(TEMPERATURE);
  integrator->setThermostatFriction(LANGEVIN_THERMOSTAT_FRICTION);
  integrator->setThermostatRngSeed(RANDOM_SEED);
  integrator->setCharmmContext(ctx);

  CHECK_NOTHROW(integrator->propagate(1));

  apo_test::CheckFiniteTemperature(ctx->computeTemperature());

  CHECK_NOTHROW(fm->unsubscribe(restraint));
}

TEST_CASE("HarmonicRestraintForceEnergyIsReportedAsHarm") {
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

  // A reference half an angstrom away gives the restraint a positive energy.
  std::vector<double3> reference = crd->getCoordinatesDP();
  reference[0].x += 0.5;

  auto restraint =
      std::make_shared<HarmonicRestraintForce<long long int, float>>(psf);
  restraint->setReferenceCoordinates(reference);
  restraint->setMasses(psf->getMasses());
  restraint->setSelection(MakeSelection(ctx->getNumAtoms(), {0, 1}));
  restraint->setForceConstant(1.0);

  auto fm = ctx->getForceManager();
  fm->subscribe(restraint, "harm", restraint->getStream(),
                restraint->getForce(), restraint->getEnergyVirial());
  ctx->calculatePotentialEnergy();

  // Like CHARMM's CONS HARM energy, the energy has its own term.
  const std::map<std::string, double> energyComponents =
      fm->getEnergyComponents();
  CHECK(energyComponents.at("harm") > 0.0);
  CHECK(energyComponents.at("harm") == GetHarmonicEnergy(*restraint));
  CHECK(energyComponents.at("user") == 0.0);

  fm->unsubscribe(restraint);
}
