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
#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "ForceManager.h"
#include "HarmonicCenterOfMassRestraintForce.h"
#include "apo_test_helpers.h"
#include "catch.hpp"
#include "cuda_utils.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <vector_types.h>

namespace {

using HmcmForce = HarmonicCenterOfMassRestraintForce<long long int, float>;

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
const std::vector<double> SMALL_BOX_DIMENSIONS = {20.0, 20.0, 20.0};

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
double GetHmcmEnergy(
    HarmonicCenterOfMassRestraintForce<long long int, CT> &restraint) {
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  restraint.getEnergyVirial()->copyToHost(*restraint.getStream());
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  return restraint.getEnergyVirial()->getEnergy("hmcm");
}

} // namespace

TEST_CASE("HarmonicCenterOfMassRestraintForceConstructorAndMetadata") {
  apo_test::CheckApoCharmmError([]() -> void { (void)HmcmForce(nullptr); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "CharmmPSF must not be null");

  apo_test::CheckApoCharmmError(
      []() -> void { (void)HmcmForce(std::make_shared<CharmmPSF>()); },
      ApoCharmmErrorCode::NotInitialized,
      "CharmmPSF atom count is not initialized; observed -1");

  apo_test::CheckApoCharmmError([]() -> void { (void)HmcmForce(MakePsf(0)); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Atom count must be positive; observed 0");

  auto psf = MakePsf(NUM_ATOMS);
  psf->getMasses().pop_back();
  apo_test::CheckApoCharmmError(
      [&psf]() -> void { (void)HmcmForce(psf); }, ApoCharmmErrorCode::Runtime,
      "PSF mass count mismatch; expected 3, observed 2");

  psf->getMasses() = {1.0, 1.0, 1.0};
  for (const double mass : {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()}) {
    psf->getMasses()[1] = mass;
    apo_test::CheckApoCharmmError([&psf]() -> void { (void)HmcmForce(psf); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Mass at index 1 must be finite; observed " +
                                      std::to_string(mass));
  }

  psf->getMasses()[1] = 1.0;
  const std::weak_ptr<CharmmPSF> weakPsf = psf;
  HmcmForce restraint(psf);
  psf.reset();
  CHECK(weakPsf.expired());

  CHECK(decltype(restraint)::contributesVirial == false);
  CHECK(std::string(decltype(restraint)::energyTerm) == "hmcm");
  CHECK(restraint.getStream() != nullptr);
  CHECK(restraint.getForce() != nullptr);
  CHECK(restraint.getForce()->size() == NUM_ATOMS);
  CHECK(restraint.getForce()->stride() >= NUM_ATOMS);
  CHECK(restraint.getEnergyVirial() != nullptr);
}

TEST_CASE("HarmonicCenterOfMassRestraintForceValidatesInput") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  const double infinity = std::numeric_limits<double>::infinity();

  SECTION("ForceConstantValidation") {
    CHECK_NOTHROW(restraint.setForceConstant(-1.0));

    apo_test::CheckApoCharmmError(
        [&restraint, infinity]() -> void {
          restraint.setForceConstant(infinity);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Force constant must be finite; observed " + std::to_string(infinity));

    CHECK_NOTHROW(restraint.setForceConstant(0.0));
    CHECK_NOTHROW(restraint.setForceConstant(1.0));
  }

  SECTION("ReferenceDistanceValidation") {
    CHECK_NOTHROW(restraint.setReferenceDistance(-1.0));

    apo_test::CheckApoCharmmError(
        [&restraint, infinity]() -> void {
          restraint.setReferenceDistance(infinity);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference distance must be finite; observed " +
            std::to_string(infinity));

    CHECK_NOTHROW(restraint.setReferenceDistance(0.0));
    CHECK_NOTHROW(restraint.setReferenceDistance(1.0));
  }

  SECTION("ReferencePositionValidation") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setReferencePosition({0.0, 0.0}, {1, 1, 1});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference-position array size mismatch; expected 3, observed 2");

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setReferencePosition({0.0, 0.0, 0.0}, {1, 1});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference-mask array size mismatch; expected 3, observed 2");

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setReferencePosition({0.0, 0.0, 0.0}, {0, 0, 0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference mask must activate at least one coordinate");

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setReferencePosition({0.0, 0.0, 0.0}, {1, 2, 1});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference mask at index 1 must be 0 or 1; observed 2");

    apo_test::CheckApoCharmmError(
        [&restraint, infinity]() -> void {
          restraint.setReferencePosition({0.0, infinity, 0.0}, {1, 1, 1});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Reference position at index 1 must be finite; observed " +
            std::to_string(infinity));

    CHECK_NOTHROW(restraint.setReferencePosition({0.0, 0.0, 0.0}));
    CHECK_NOTHROW(restraint.setReferencePosition({0.0, 0.0, 0.0}, {1, 0, 1}));
  }

  SECTION("BoxDimensionValidation") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void { restraint.setBoxDimensions({10.0, 10.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Box-dimension array size mismatch; expected 3, observed 2");

    apo_test::CheckApoCharmmError(
        [&restraint, infinity]() -> void {
          restraint.setBoxDimensions({10.0, infinity, 10.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index 1 must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setBoxDimensions({10.0, 0.0, 10.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index 1 must be positive; observed " +
            std::to_string(0.0));

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setBoxDimensions({10.0, -1.0, 10.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index 1 must be positive; observed " +
            std::to_string(-1.0));

    CHECK_NOTHROW(restraint.setBoxDimensions({10.0, 10.0, 10.0}));
  }

  SECTION("SelectionValidation") {
    AtomSelection emptySelection(NUM_ATOMS, AtomSelection::InitialValue::NONE);
    apo_test::CheckApoCharmmError(
        [&restraint, &emptySelection]() -> void {
          restraint.setSelection(emptySelection);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Selection must contain at least one atom; observed 0");

    AtomSelection wrongSizeSelection(NUM_ATOMS + 1,
                                     AtomSelection::InitialValue::ALL);
    apo_test::CheckApoCharmmError(
        [&restraint, &wrongSizeSelection]() -> void {
          restraint.setSelection(wrongSizeSelection);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Selection atom count mismatch; expected 3, observed 4");

    CHECK_NOTHROW(restraint.setSelection(MakeSelection(NUM_ATOMS, {0})));
  }

  SECTION("MassValidation") {
    apo_test::CheckApoCharmmError(
        [&restraint]() -> void { restraint.setMasses({1.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Mass array size mismatch; expected 3, observed 1");

    apo_test::CheckApoCharmmError(
        [&restraint, infinity]() -> void {
          restraint.setMasses({1.0, infinity, 1.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Mass at index 1 must be finite; observed " + std::to_string(infinity));

    CHECK_NOTHROW(restraint.setMasses({1.0, 1.0, -1.0}));
    CHECK_NOTHROW(restraint.setMasses({1.0, -3.0, 1.0}));

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setMasses({1.0e308, 1.0e308, 0.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Selected atoms must have a finite nonzero total weight; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.setMasses({1.0e308, -1.0e308, 1.0e-300});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Weight fraction at atom index 0 must be finite; observed " +
            std::to_string(infinity));

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void { restraint.setMasses({1.0, -1.0, 0.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Selected atoms must have a finite nonzero total weight; observed " +
            std::to_string(0.0));

    CHECK_NOTHROW(restraint.setMassWeighting(false));

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void { restraint.setMassWeighting(true); },
        ApoCharmmErrorCode::InvalidArgument,
        "Selected atoms must have a finite nonzero total weight; observed " +
            std::to_string(0.0));

    CHECK_NOTHROW(restraint.setMasses({1.0, 2.0, 3.0}));
  }

  SECTION("InitializeValidation") {
    CHECK_NOTHROW(restraint.initialize(NUM_ATOMS, {10.0, 10.0, 10.0}));

    apo_test::CheckApoCharmmError(
        [&restraint]() -> void {
          restraint.initialize(NUM_ATOMS + 1, {10.0, 10.0, 10.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Initialization atom count mismatch; expected 3, observed 4");
  }
}

TEST_CASE("HarmonicCenterOfMassRestraintForceEvaluatesWithoutBox") {
  // CHARMM's CONS HMCM never reads the unit cell.
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(2.5);
  restraint.setReferencePosition({1.0, 1.0, 1.0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 2.0f, 3.0f, 0.0f), make_float4(3.0f, 4.0f, 5.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(2.5, 5.0, 7.5),
                                               make_double3(2.5, 5.0, 7.5),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM forces without a box",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(35.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceChecksEveryKernelLaunchVariant") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(1.0);
  restraint.setReferencePosition({0.0, 0.0, 0.0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(2.0f, 0.0f, 0.0f, 0.0f),
      make_float4(3.0f, 0.0f, 0.0f, 0.0f)};

  const auto checkLaunches = [&restraint,
                              &xyzq](const bool calcEnergy,
                                     const bool calcVirial) -> void {
    restraint.clear();
    restraint.calcForce(xyzq.getDeviceArray().data(), calcEnergy, calcVirial);
    cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
    return;
  };

  CHECK_NOTHROW(checkLaunches(true, true));
  CHECK_NOTHROW(checkLaunches(true, false));
  CHECK_NOTHROW(checkLaunches(false, true));
  CHECK_NOTHROW(checkLaunches(false, false));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceNoopsForZeroForceConstant") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setReferencePosition({0.0, 0.0, 0.0});
  restraint.setForceConstant(0.0);

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(3.0f, 0.0f, 0.0f, 0.0f),
      make_float4(5.0f, 0.0f, 0.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM zero-force-constant forces",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(0.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceUnweightedPointRestraint") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(2.5);
  restraint.setReferencePosition({1.0, 1.0, 1.0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 2.0f, 3.0f, 0.0f), make_float4(3.0f, 4.0f, 5.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(2.5, 5.0, 7.5),
                                               make_double3(2.5, 5.0, 7.5),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM unweighted point restraint forces",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(35.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceCoordinateMask") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(2.5);
  restraint.setReferencePosition({1.0, 0.0, 0.0}, {1, 0, 0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 2.0f, 3.0f, 0.0f), make_float4(3.0f, 4.0f, 5.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(2.5, 0.0, 0.0),
                                               make_double3(2.5, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM coordinate-mask forces",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(2.5).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceMassWeighting") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setMasses({1.0, 3.0, 100.0});
  restraint.setForceConstant(2.0);
  restraint.setReferencePosition({1.0, 0.0, 0.0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(4.0f, 0.0f, 0.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(2.0, 0.0, 0.0),
                                               make_double3(6.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM mass-weighted forces",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(8.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceCanDisableMassWeighting") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setMasses({1.0, 3.0, 100.0});
  restraint.setMassWeighting(false);
  restraint.setForceConstant(2.0);
  restraint.setReferencePosition({1.0, 0.0, 0.0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(4.0f, 0.0f, 0.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(2.0, 0.0, 0.0),
                                               make_double3(2.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM unweighted-after-masses forces",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(2.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceCopiesPsfMasses") {
  // CHARMM's WEIG or MASS keyword weights the center with the PSF masses.
  auto psf = MakePsf(NUM_ATOMS);
  psf->getMasses() = {1.0, 3.0, 100.0};
  HmcmForce restraint(psf);
  psf->getMasses() = {1.0, 1.0, 1.0};
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(2.0);
  restraint.setReferencePosition({1.0, 0.0, 0.0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(4.0f, 0.0f, 0.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  // Weights stay uniform until mass weighting is enabled.
  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  apo_test::CheckVectorsClose<double3>(
      "HMCM forces before mass weighting",
      CopyFixedPointForceToHost(restraint.getForce()),
      {make_double3(2.0, 0.0, 0.0), make_double3(2.0, 0.0, 0.0),
       make_double3(0.0, 0.0, 0.0)},
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(2.0).margin(ENERGY_TOLERANCE));

  // The masses 1 and 3 copied at construction give the center x = 3.
  restraint.setMassWeighting(true);
  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  apo_test::CheckVectorsClose<double3>(
      "HMCM forces weighted by PSF masses",
      CopyFixedPointForceToHost(restraint.getForce()),
      {make_double3(2.0, 0.0, 0.0), make_double3(6.0, 0.0, 0.0),
       make_double3(0.0, 0.0, 0.0)},
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(8.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceReferenceDistance") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(3.0);
  restraint.setReferencePosition({0.0, 0.0, 0.0});
  restraint.setReferenceDistance(1.0);

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(3.0f, 0.0f, 0.0f, 0.0f), make_float4(5.0f, 0.0f, 0.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(9.0, 0.0, 0.0),
                                               make_double3(9.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM reference-distance forces",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(27.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceUsesRawCoordinates") {
  // As in CHARMM, neither the center nor the center-to-reference
  // displacement is imaged. The raw center x = 5.1 lies 6.1 angstroms from
  // the reference, more than half the box, although the minimum images of the
  // two atoms are 0.6 apart.
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions({10.0, 10.0, 10.0});
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setForceConstant(2.0);
  restraint.setReferencePosition({-1.0, 0.0, 0.0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(9.8f, 0.0f, 0.0f, 0.0f), make_float4(0.4f, 0.0f, 0.0f, 0.0f),
      make_float4(5.0f, 5.0f, 5.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  // d = 6.1 gives the energy 2 * 6.1^2 and the per-atom gradient
  // 6.1 * 2 * 2 / 2 * 6.1 / 6.1.
  const std::vector<double3> expectedForces = {make_double3(12.2, 0.0, 0.0),
                                               make_double3(12.2, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM raw-coordinate forces",
      CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
      8.0 * FLOAT_EPSILON * 12.2);
  CHECK(GetHmcmEnergy(restraint) ==
        Approx(74.42).margin(8.0 * FLOAT_EPSILON * 74.42));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceClearResetsForcesAndEnergy") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setBoxDimensions(SMALL_BOX_DIMENSIONS);
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));
  restraint.setReferencePosition({0.0, 0.0, 0.0});
  restraint.setForceConstant(1.0);

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(3.0f, 0.0f, 0.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), true, true);
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  CHECK(GetHmcmEnergy(restraint) > 0.0);

  restraint.clear();
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

  const std::vector<double3> expectedForces = {make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0),
                                               make_double3(0.0, 0.0, 0.0)};

  apo_test::CheckVectorsClose<double3>(
      "HMCM cleared forces", CopyFixedPointForceToHost(restraint.getForce()),
      expectedForces, FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(restraint) == Approx(0.0).margin(ENERGY_TOLERANCE));
}

TEST_CASE("HarmonicCenterOfMassRestraintForceFollowsCharmmSignConventions") {
  HmcmForce restraint(MakePsf(NUM_ATOMS));
  restraint.setSelection(MakeSelection(NUM_ATOMS, {0, 1}));

  SECTION("SignedParameters") {
    // W = -2 puts the center at x = 0.5. With k = -2 and r0 = -1, the
    // deviation is 1.5, the energy -2 * 1.5^2, and the axis factor
    // 0.5 * 2 * -2 / -2 * 1.5 / 0.5 = 3, which each atom scales by its mass.
    restraint.setMasses({1.0, -3.0, 100.0});
    restraint.setForceConstant(-2.0);
    restraint.setReferencePosition({0.0, 0.0, 0.0});
    restraint.setReferenceDistance(-1.0);

    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(2.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

    restraint.clear();
    restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
    cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

    const std::vector<double3> expectedForces = {make_double3(3.0, 0.0, 0.0),
                                                 make_double3(-9.0, 0.0, 0.0),
                                                 make_double3(0.0, 0.0, 0.0)};

    apo_test::CheckVectorsClose<double3>(
        "HMCM signed-parameter forces",
        CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
        FORCE_TOLERANCE);
    CHECK(GetHmcmEnergy(restraint) == Approx(-4.5).margin(ENERGY_TOLERANCE));
  }

  SECTION("ZeroDistance") {
    // CHARMM divides by the zero distance when r0 is nonzero; the restraint
    // keeps the energy k * r0^2 and applies no gradient.
    restraint.setForceConstant(3.0);
    restraint.setReferencePosition({2.0, 0.0, 0.0});
    restraint.setReferenceDistance(1.0);

    CudaContainer<float4> xyzq =
        std::vector<float4>{make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(3.0f, 0.0f, 0.0f, 0.0f),
                            make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

    restraint.clear();
    restraint.calcForce(xyzq.getDeviceArray().data(), true, false);
    cudaCheck(cudaStreamSynchronize(*restraint.getStream()));

    const std::vector<double3> expectedForces = {make_double3(0.0, 0.0, 0.0),
                                                 make_double3(0.0, 0.0, 0.0),
                                                 make_double3(0.0, 0.0, 0.0)};

    apo_test::CheckVectorsClose<double3>(
        "HMCM zero-distance forces",
        CopyFixedPointForceToHost(restraint.getForce()), expectedForces,
        FORCE_TOLERANCE);
    CHECK(GetHmcmEnergy(restraint) == Approx(3.0).margin(ENERGY_TOLERANCE));
  }
}

TEST_CASE("HarmonicCenterOfMassRestraintForceArithmeticFollowsCT") {
  // A reference offset of 2^-30 angstrom is exact in double arithmetic and
  // rounds away when the reference is converted to float.
  const double offset = std::ldexp(1.0, -30);
  const double forceConstant = std::ldexp(1.0, 20);
  const AtomSelection selection = MakeSelection(NUM_ATOMS, {0});

  CudaContainer<float4> xyzq = std::vector<float4>{
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(5.0f, 5.0f, 5.0f, 0.0f),
      make_float4(9.0f, 9.0f, 9.0f, 0.0f)};

  HarmonicCenterOfMassRestraintForce<long long int, double> doubleRestraint(
      MakePsf(NUM_ATOMS));
  doubleRestraint.setSelection(selection);
  doubleRestraint.setForceConstant(forceConstant);
  doubleRestraint.setReferencePosition({1.0 + offset, 0.0, 0.0});
  doubleRestraint.clear();
  doubleRestraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*doubleRestraint.getStream()));

  // The gradient -2^-30 * 2 * 2^20 and the energy 2^20 * 2^-60 are exact.
  apo_test::CheckVectorsClose<double3>(
      "HMCM double-arithmetic forces",
      CopyFixedPointForceToHost(doubleRestraint.getForce()),
      {make_double3(-std::ldexp(1.0, -9), 0.0, 0.0),
       make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)},
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(doubleRestraint) == std::ldexp(1.0, -40));

  HmcmForce floatRestraint(MakePsf(NUM_ATOMS));
  floatRestraint.setSelection(selection);
  floatRestraint.setForceConstant(forceConstant);
  floatRestraint.setReferencePosition({1.0 + offset, 0.0, 0.0});
  floatRestraint.clear();
  floatRestraint.calcForce(xyzq.getDeviceArray().data(), true, false);
  cudaCheck(cudaStreamSynchronize(*floatRestraint.getStream()));

  apo_test::CheckVectorsClose<double3>(
      "HMCM float-arithmetic forces",
      CopyFixedPointForceToHost(floatRestraint.getForce()),
      {make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0),
       make_double3(0.0, 0.0, 0.0)},
      FORCE_TOLERANCE);
  CHECK(GetHmcmEnergy(floatRestraint) == 0.0);
}

TEST_CASE("HarmonicCenterOfMassRestraintForceStaysOutOfManagerVirial") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCoordinates(std::vector<double3>{make_double3(1.0, 2.0, 3.0),
                                           make_double3(3.0, 4.0, 5.0)});
  ctx->useHolonomicConstraints(false);
  auto fm = ctx->getForceManager();

  ctx->calculateForces(false, true, true);
  fm->getPotentialEnergy().transferToHost();
  const double baselineEnergy = fm->getPotentialEnergy()[0];
  auto &baselineVirial = fm->getVirial();
  baselineVirial.transferToHost();
  std::vector<double> expectedVirial(9);
  for (int i = 0; i < 9; i++)
    expectedVirial[i] = baselineVirial[i];

  auto restraint = std::make_shared<HmcmForce>(psf);
  restraint->setForceConstant(2.5);
  restraint->setReferencePosition({1.0, 1.0, 1.0});
  fm->subscribe(restraint, "hmcm", restraint->getStream(),
                restraint->getForce(), restraint->getEnergyVirial());

  // CHARMM evaluates CONS HMCM after it forms its internal virial, so the
  // restraint adds its energy but leaves the aggregate virial unchanged.
  ctx->calculateForces(false, true, true);
  fm->getPotentialEnergy().transferToHost();
  CHECK(fm->getPotentialEnergy()[0] - baselineEnergy ==
        Approx(35.0).margin(8.0 * FLOAT_EPSILON * 35.0));

  auto &virial = fm->getVirial();
  virial.transferToHost();
  // The reciprocal-space virial is summed with device atomics, so repeated
  // evaluations can differ in the last bits.
  for (int i = 0; i < 9; i++)
    CHECK(virial[i] == Approx(expectedVirial[i]).margin(1.0e-8));

  fm->unsubscribe(restraint);
}

TEST_CASE("HarmonicCenterOfMassRestraintForceCanBeSubscribedToForceManager") {
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

  auto restraint = std::make_shared<HmcmForce>(psf);
  restraint->setSelection(MakeSelection(ctx->getNumAtoms(), {0, 1}));
  restraint->setReferencePosition({1.41, 1.41, 1.41});
  restraint->setForceConstant(1.0);

  auto fm = ctx->getForceManager();

  apo_test::CheckApoCharmmError(
      [&fm, &restraint]() -> void {
        fm->subscribe(restraint, "", restraint->getStream(),
                      restraint->getForce(), restraint->getEnergyVirial());
      },
      ApoCharmmErrorCode::InvalidArgument, "Force tag must not be empty");

  CHECK_NOTHROW(fm->subscribe(restraint, "hmcm", restraint->getStream(),
                              restraint->getForce(),
                              restraint->getEnergyVirial()));

  apo_test::CheckApoCharmmError(
      [&fm, &restraint]() -> void {
        fm->subscribe(restraint, "hmcm", restraint->getStream(),
                      restraint->getForce(), restraint->getEnergyVirial());
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force is already subscribed to this ForceManager");

  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  integrator->setReferenceTemperature(TEMPERATURE);
  integrator->setThermostatFriction(LANGEVIN_THERMOSTAT_FRICTION);
  integrator->setThermostatRngSeed(RANDOM_SEED);
  integrator->setCharmmContext(ctx);

  CHECK_NOTHROW(integrator->propagate(1));

  apo_test::CheckFiniteTemperature(ctx->computeTemperature());

  CHECK_NOTHROW(fm->unsubscribe(restraint));
  apo_test::CheckApoCharmmError(
      [&fm, &restraint]() -> void { fm->unsubscribe(restraint); },
      ApoCharmmErrorCode::InvalidArgument,
      "Force is not subscribed to this ForceManager");
}
