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
#include "AtomReference.h"
#include "AtomSelection.h"
#include "CharmmContext.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaContainer.h"
#include "ForceManager.h"
#include "GeometricRestraintForce.h"
#include "apo_test_helpers.h"
#include "catch.hpp"
#include "cuda_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename CT>
using GeoForce = GeometricRestraintForce<long long int, CT>;

constexpr int NUM_ATOMS = 12;
constexpr double INV_FORCE_SCALE_TEST = 1.0 / static_cast<double>(1LL << 40);
const std::array<double, 3> ORIGIN = {0.0, 0.0, 0.0};
const std::array<double, 3> X_DIRECTION = {1.0, 0.0, 0.0};
const std::vector<double> BOX_DIMS = {20.0, 30.0, 40.0};

struct Output {
  std::vector<double3> physicalForces;
  double energy;
  std::array<double, 9> virial;
};

std::shared_ptr<CharmmPSF> MakePsf(const int numAtoms = NUM_ATOMS) {
  auto psf = std::make_shared<CharmmPSF>();
  psf->setNumAtoms(numAtoms);
  for (int atom = 0; atom < numAtoms; atom++)
    psf->getMasses()[atom] = static_cast<double>(atom) + 1.0;
  return psf;
}

template <typename CT> Output ReadOutput(GeoForce<CT> &restraint) {
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  const auto force = restraint.getForce();
  std::vector<long long int> gx(force->size()), gy(force->size()),
      gz(force->size());
  force->getXYZ(gx.data(), gy.data(), gz.data());
  Output result;
  result.physicalForces.resize(force->size());
  for (int atom = 0; atom < force->size(); atom++) {
    // JEG261006: Store as physical forces, not native gradients.
    result.physicalForces[atom] =
        make_double3(-static_cast<double>(gx[atom]) * INV_FORCE_SCALE_TEST,
                     -static_cast<double>(gy[atom]) * INV_FORCE_SCALE_TEST,
                     -static_cast<double>(gz[atom]) * INV_FORCE_SCALE_TEST);
  }
  restraint.getEnergyVirial()->copyToHost(*restraint.getStream());
  cudaCheck(cudaStreamSynchronize(*restraint.getStream()));
  result.energy = restraint.getEnergyVirial()->getEnergy("mmfp");
  restraint.getEnergyVirial()->getVirial(result.virial.data());
  return result;
}

template <typename CT>
Output Evaluate(GeoForce<CT> &restraint, const std::vector<float4> &coordinates,
                const bool calcEnergy = true, const bool calcVirial = false,
                const bool clear = true) {
  CudaContainer<float4> xyzq(coordinates);
  if (clear)
    restraint.clear();
  restraint.calcForce(xyzq.getDeviceArray().data(), calcEnergy, calcVirial);
  return ReadOutput(restraint);
}

void CheckOutput(const std::string &label, const Output &observed,
                 const double expectedEnergy,
                 const std::vector<double3> &expectedForces,
                 const double energyTolerance = 0.0,
                 const double forceTolerance = 0.0) {
  apo_test::CheckVectorsClose<double>(label + " energy", {observed.energy},
                                      {expectedEnergy}, energyTolerance);
  apo_test::CheckVectorsClose<double3>(label + " physical forces",
                                       observed.physicalForces, expectedForces,
                                       forceTolerance);
  // JEG261006: CheckVectorsClose uses Approx::margin without disabling its
  // relative epsilon. These checks enforce the actual absolute numerical
  // contract.
  INFO("strict absolute check: " << label);
  CHECK(std::abs(observed.energy - expectedEnergy) <= energyTolerance);
  REQUIRE(observed.physicalForces.size() == expectedForces.size());
  for (std::size_t i = 0; i < expectedForces.size(); i++) {
    INFO("atom: " << i);
    CHECK(std::abs(observed.physicalForces[i].x - expectedForces[i].x) <=
          forceTolerance);
    CHECK(std::abs(observed.physicalForces[i].y - expectedForces[i].y) <=
          forceTolerance);
    CHECK(std::abs(observed.physicalForces[i].z - expectedForces[i].z) <=
          forceTolerance);
  }
  return;
}

struct ManagerOutput {
  std::vector<double3> gradients;
  double energy;
};

ManagerOutput EvaluateManager(const std::shared_ptr<CharmmContext> &context) {
  context->calculateForces(false, true, false);
  auto manager = context->getForceManager();
  auto forces = manager->getForces();
  std::vector<double> x(forces->size()), y(forces->size()), z(forces->size());
  forces->getXYZ(x.data(), y.data(), z.data());
  ManagerOutput result;
  for (int i = 0; i < forces->size(); i++)
    result.gradients.push_back(make_double3(x[i], y[i], z[i]));
  manager->getPotentialEnergy().transferToHost();
  result.energy = manager->getPotentialEnergy()[0];
  return result;
}

void CheckManagerDelta(const ManagerOutput &baseline,
                       const ManagerOutput &observed, const double energy,
                       const std::vector<double3> &gradients) {
  REQUIRE(observed.gradients.size() == baseline.gradients.size());
  REQUIRE(observed.gradients.size() == gradients.size());
  std::vector<double3> delta(observed.gradients.size());
  for (std::size_t i = 0; i < delta.size(); i++) {
    delta[i] = make_double3(observed.gradients[i].x - baseline.gradients[i].x,
                            observed.gradients[i].y - baseline.gradients[i].y,
                            observed.gradients[i].z - baseline.gradients[i].z);
  }
  apo_test::CheckVectorsClose<double3>("MMFP manager gradient difference",
                                       delta, gradients, 1.0e-8);
  apo_test::CheckVectorsClose<double>("MMFP manager energy difference",
                                      {observed.energy - baseline.energy},
                                      {energy}, 1.0e-8);
  CHECK(std::abs((observed.energy - baseline.energy) - energy) <= 1.0e-8);
  for (std::size_t i = 0; i < gradients.size(); i++) {
    CHECK(std::abs(delta[i].x - gradients[i].x) <= 1.0e-8);
    CHECK(std::abs(delta[i].y - gradients[i].y) <= 1.0e-8);
    CHECK(std::abs(delta[i].z - gradients[i].z) <= 1.0e-8);
  }
  return;
}

} // namespace

TEMPLATE_TEST_CASE("GeometricRestraintForceConstruction", "[mmfp]", float,
                   double) {
  STATIC_REQUIRE(!std::is_default_constructible<GeoForce<TestType>>::value);
  STATIC_REQUIRE(!std::is_copy_constructible<GeoForce<TestType>>::value);
  STATIC_REQUIRE(!std::is_copy_assignable<GeoForce<TestType>>::value);
  STATIC_REQUIRE(!std::is_move_constructible<GeoForce<TestType>>::value);
  STATIC_REQUIRE(!std::is_move_assignable<GeoForce<TestType>>::value);

  auto psf = MakePsf();

  SECTION("NullPsf") {
    apo_test::CheckApoCharmmError(
        [](void) -> void {
          GeoForce<TestType> force(nullptr);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument, "CharmmPSF must not be null");
  }

  SECTION("UnintializedPsf") {
    apo_test::CheckApoCharmmError(
        [](void) -> void {
          GeoForce<TestType> force(std::make_shared<CharmmPSF>());
          return;
        },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmPSF atom count is not initialized; observed -1");
  }

  SECTION("ZeroAtomPsf") {
    apo_test::CheckApoCharmmError(
        [](void) -> void {
          GeoForce<TestType> force(MakePsf(0));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom count must be positive; observed 0");
  }

  SECTION("InconsistentMassVector") {
    psf->getMasses().pop_back();
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          GeoForce<TestType> force(psf);
          return;
        },
        ApoCharmmErrorCode::Runtime,
        "PSF mass count mismatch; expected 12, observed 11");
  }

  SECTION("NonfiniteMasses") {
    for (const double mass : {std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity()}) {
      psf->getMasses()[0] = mass;
      apo_test::CheckApoCharmmError(
          [&](void) -> void {
            GeoForce<TestType> force(psf);
            return;
          },
          ApoCharmmErrorCode::InvalidArgument,
          "Mass at atom index 0 must be finite; observed " +
              std::to_string(mass));
    }
  }

  SECTION("SnapshotMutationIndependenceAndPsfLifetime") {
    // Retain signed and zero masses, including across reset
    psf->getMasses()[0] = -2.0;
    psf->getMasses()[1] = 0.0;
    const auto masses = psf->getMasses();
    std::weak_ptr<CharmmPSF> weak = psf;
    GeoForce<TestType> force(psf);
    CHECK(force.getNumAtoms() == NUM_ATOMS);
    CHECK(force.getMasses() == masses);
    CHECK(force.getNumTerms() == 0);
    REQUIRE(force.getStream() != nullptr);
    CHECK(*force.getStream() != nullptr);
    REQUIRE(force.getForce() != nullptr);
    CHECK(force.getForce()->size() == NUM_ATOMS);
    REQUIRE(force.getEnergyVirial() != nullptr);

    psf->getMasses()[0] = 99.0;
    psf->setNumAtoms(NUM_ATOMS + 1);
    psf.reset();
    CHECK(weak.expired());
    CHECK(force.getNumAtoms() == NUM_ATOMS);
    CHECK(force.getMasses() == masses);
    force.reset();
    CHECK(force.getMasses() == masses);
    CheckOutput("initial output", ReadOutput(force), 0.0,
                std::vector<double3>(NUM_ATOMS, make_double3(0.0, 0.0, 0.0)));
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceTypedInputs", "[mmfp]", float,
                   double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0].x = 3.0f;
  auto expected = std::vector<double3>(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
  expected[0].x = -2.0;

  SECTION("SelectionCopiedBeforeMutationAndDestruction") {
    AtomSelection selection(NUM_ATOMS);
    selection.set(0);
    force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
    selection.clear();
    selection.set(1);
  }

  SECTION("CartesianParameterValuesCopied") {
    AtomSelection selection(NUM_ATOMS);
    selection.set(0);
    auto origin = ORIGIN;
    auto direction = X_DIRECTION;
    force.addPlane(selection, origin, direction, 2.0, 2.0, A::SYMMETRIC);
    origin[0] = 20.0;
    direction[0] = -1.0;
  }

  SECTION("OneAtomReference") {
    AtomReference atom(psf, 0);
    force.addPlane(atom, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
  }

  SECTION("ForeignTopologyForceLocalValidIndexNoRetainedOwner") {
    std::weak_ptr<CharmmPSF> weak;
    {
      auto foreign = MakePsf(NUM_ATOMS + 5);
      weak = foreign;
      AtomReference atom(foreign, 0);
      force.addPlane(atom, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
    }
    CHECK(weak.expired());
  }

  SECTION("ForeignTopologyForceLocalInvalidIndex") {
    AtomReference atom(MakePsf(NUM_ATOMS + 1), NUM_ATOMS);
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(atom, ORIGIN, X_DIRECTION, 2.0);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom index is out of range; expected [0, 12), observed 12");
    CHECK(force.getNumTerms() == 0);
    return;
  }

  SECTION("EmptySelection") {
    // JEG261006: Native API decision, not CHARMM. CHARMM accepts "SELECT NONE"
    AtomSelection selection(NUM_ATOMS);
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Selection must contain at least one atom; observed 0");
    CHECK(force.getNumTerms() == 0);
    return;
  }

  SECTION("SelectionAtomCountMismatch") {
    AtomSelection selection(NUM_ATOMS + 1, AtomSelection::InitialValue::ALL);
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Selection atom count mismatch; expected 12, observed 13");
    CHECK(force.getNumTerms() == 0);
    return;
  }

  CHECK(force.getNumTerms() == 1);
  CheckOutput("plane_x_axis_positive_side", Evaluate(force, coordinates), 1.0,
              expected);
}

TEMPLATE_TEST_CASE("GeometricRestraintForceParameterValidation", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  G force(MakePsf());
  AtomSelection selection(NUM_ATOMS);
  selection.set(0);
  force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
  const double bad = std::numeric_limits<double>::quiet_NaN();

  SECTION("ValidityPlaneZeroDirection") {
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, ORIGIN, ORIGIN, 2.0);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Direction must have a positive finite norm");
  }

  SECTION("NonfiniteDirection") {
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, ORIGIN, {bad, 0.0, 1.0}, 2.0);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Direction component must be finite; index 0");
  }

  SECTION("NonfiniteOrigin") {
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, {bad, 0.0, 0.0}, X_DIRECTION, 2.0);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Origin component must be finite; index 0");
  }

  SECTION("NonfiniteForceConstant") {
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, ORIGIN, X_DIRECTION, bad);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Force constant must be finite; observed " + std::to_string(bad));
  }

  SECTION("NonfiniteOffset") {
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, bad);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Offset must be finite; observed " + std::to_string(bad));
  }

  SECTION("InvalidCastCreatedActivation") {
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, 2.0,
                         static_cast<A>(99));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Invalid MMFP activation value; observed 99");
  }

  CHECK(force.getNumTerms() == 1);
  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0].x = 3.0f;
  std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
  expected[0].x = -2.0;
  CheckOutput("failed addition preserves old term",
              Evaluate(force, coordinates), 1.0, expected);
}

TEMPLATE_TEST_CASE("GeometricRestraintForcePlanarOracle", "[mmfp]", float,
                   double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  struct Case {
    const char *id;
    std::array<double, 3> position, origin, direction;
    double k, offset;
    A activation;
    double energy;
    std::array<double, 3> physicalForce;
    double energyTolerance;
  };

  // JEG261006: Values below are retained from CHARMM, not runtime-derived
  // expectations. These coordinates are exactly representable as float4.
  const std::vector<Case> cases = {
      {"PlaneXAxisPositiveSide",
       {3.0, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       2.0,
       2.0,
       A::SYMMETRIC,
       1.0,
       {-2.0, 0.0, 0.0},
       0.0},
      {"PlaneXAxisNegativeSide",
       {-3.0, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       2.0,
       2.0,
       A::SYMMETRIC,
       25.0,
       {10.0, 0.0, 0.0},
       0},
      {"PlaneGeneralUnit",
       {3.0, 2.0, 4.0},
       {0.5, -1.0, 1.5},
       {0.2672612419124244, 0.5345224838248488, 0.8017837257372732},
       2.0,
       2.0,
       A::SYMMETRIC,
       5.1809948,
       {-1.2166693181, -2.4333386361, -3.6500079542},
       5.0e-8},
      {"PlaneGeneralNonunit",
       {3.0, 2.0, 4.0},
       {0.5, -1.0, 1.5},
       {2.0, 4.0, 6.0},
       2.0,
       2.0,
       A::SYMMETRIC,
       5.1809948,
       {-1.2166693181, -2.4333386361, -3.6500079542},
       5.0e-8},
      {"PlaneGeneralReversed",
       {3.0, 2.0, 4.0},
       {0.5, -1.0, 1.5},
       {-1.0, -2.0, -3.0},
       2.0,
       2.0,
       A::SYMMETRIC,
       39.3904338,
       {-3.3547592534, -6.7095185067, -10.0642777601},
       5.0e-8},
      {"PlaneOutsideZeroGeneral",
       {0.0, 0.0, 0.0},
       ORIGIN,
       {1.0, 2.0, 3.0},
       2.0,
       2.0,
       A::OUTSIDE,
       4.0,
       {1.0690449676, 2.1380899353, 3.2071349029},
       0.0},
      {"PlaneInsideNegdZeroGeneral",
       {0.0, 0.0, 0.0},
       ORIGIN,
       {1.0, 2.0, 3.0},
       2.0,
       -2.0,
       A::INSIDE,
       4.0,
       {-1.0690449676, -2.1380899353, -3.2071349029},
       0.0},
      {"PlaneInsideEqual",
       {2.0, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       2.0,
       2.0,
       A::INSIDE,
       0.0,
       {0.0, 0.0, 0.0},
       0.0},
      {"PlaneOutsideEqual",
       {2.0, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       2.0,
       2.0,
       A::OUTSIDE,
       0.0,
       {0.0, 0.0, 0.0},
       0.0},
      {"PlanePrimary",
       {1.0, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       2.0,
       2.0,
       A::SYMMETRIC,
       1.0,
       {2.0, 0.0, 0.0},
       0.0},
      {"PlaneAtomShiftX",
       {21.0, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       2.0,
       2.0,
       A::SYMMETRIC,
       361.0,
       {-38.0, 0.0, 0.0},
       0.0},
      {"PlaneReferenceShiftX",
       {1.0, 0.0, 0.0},
       {20.0, 0.0, 0.0},
       X_DIRECTION,
       2.0,
       2.0,
       A::SYMMETRIC,
       441.0,
       {42.0, 0.0, 0.0},
       0.0},
      {"PlaneBothShiftX",
       {21.0, 0.0, 0.0},
       {20.0, 0.0, 0.0},
       X_DIRECTION,
       2.0,
       2.0,
       A::SYMMETRIC,
       1.0,
       {2.0, 0.0, 0.0},
       0.0},
      {"PlaneCrossXBoundary",
       {-10.5, 0.0, 0.0},
       {9.0, 0.0, 0.0},
       X_DIRECTION,
       2.0,
       2.0,
       A::SYMMETRIC,
       462.25,
       {43.0, 0.0, 0.0},
       0.0},
      {"ValidityLargeFiniteCoordinateRepresentable",
       {1.0e8, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       1.0e-8,
       0.0,
       A::SYMMETRIC,
       50000000.0,
       {-1.0, 0.0, 0.0},
       0.5},
      {"ValidityExtremeFinitieCoordinate",
       {9.0e8, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       1.0e-18,
       0.0,
       A::SYMMETRIC,
       0.405,
       {-9.0e-10, 0.0, 0.0},
       5.0e-8},
      {"DroffPos1e150Balanced",
       {0.0, 0.0, 0.0},
       ORIGIN,
       X_DIRECTION,
       1.0e-300,
       1.0e150,
       A::SYMMETRIC,
       0.5,
       {0.0, 0.0, 0.0},
       5.0e-8},
      {"ReferencePos1e150Balanced",
       {0.0, 0.0, 0.0},
       {1.0e150, 0.0, 0.0},
       X_DIRECTION,
       1.0e-300,
       0.0,
       A::SYMMETRIC,
       0.5,
       {0.0, 0.0, 0.0},
       5.0e-8}};

  G force(MakePsf());
  AtomSelection selection(NUM_ATOMS);
  selection.set(0);
  for (const Case &test : cases) {
    INFO(test.id);
    force.reset();
    force.initialize(NUM_ATOMS, BOX_DIMS);
    force.addPlane(selection, test.origin, test.direction, test.k, test.offset,
                   test.activation);
    std::vector<float4> coordinates(NUM_ATOMS,
                                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    coordinates[0] = make_float4(static_cast<float>(test.position[0]),
                                 static_cast<float>(test.position[1]),
                                 static_cast<float>(test.position[2]), 0.0f);
    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    expected[0] = make_double3(test.physicalForce[0], test.physicalForce[1],
                               test.physicalForce[2]);
    const double magnitude = std::max({std::abs(test.physicalForce[0]),
                                       std::abs(test.physicalForce[1]),
                                       std::abs(test.physicalForce[2])});
    // JEG261006: CSV capture bound + fixed-point half quantum + optional float
    // rounding.
    const double tolerance =
        1.0e-10 + 0.5 * INV_FORCE_SCALE_TEST +
        (std::is_same<TestType, float>::value
             ? 0.5 * std::numeric_limits<float>::epsilon() * magnitude
             : 0.0);
    CheckOutput(test.id, Evaluate(force, coordinates), test.energy, expected,
                test.energyTolerance, tolerance);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceActivationAndSignedParameters",
                   "[mmfp]", float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  G force(MakePsf());
  AtomSelection selection(NUM_ATOMS);
  selection.set(0);
  const std::array<float, 7> positions = {-3.0f, -2.0f, -1.0f, 0.0f,
                                          1.0f,  2.0f,  3.0f};

  // JEG261006: Reference values collected from CHARMM establish the plane
  // activation, cusp, and boundary-equality behavoir. These exactly
  // representable native controls supplement the collected near-boundary
  // reference values.
  const std::array<std::array<double, 7>, 3> energies = {
      {{25.0, 16.0, 9.0, 4.0, 1.0, 0.0, 1.0},
       {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
       {0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0}}};
  const std::array<std::array<double, 7>, 3> forces = {
      {{10.0, 8.0, 6.0, 4.0, 2.0, 0.0, -2.0},
       {2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0},
       {0.0, 0.0, -2.0, 4.0, 2.0, 0.0, 0.0}}};
  const std::array<A, 3> modes = {A::SYMMETRIC, A::INSIDE, A::OUTSIDE};
  for (std::size_t mode = 0; mode < modes.size(); mode++) {
    force.reset();
    force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, 2.0, modes[mode]);
    for (std::size_t point = 0; point < positions.size(); point++) {
      INFO("activation=" << static_cast<int>(modes[mode])
                         << ", x=" << positions[point]);
      std::vector<float4> coordinates(NUM_ATOMS,
                                      make_float4(0.0f, 0.0f, 0.0f, 0.0f));
      coordinates[0].x = positions[point];
      std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
      expected[0].x = forces[mode][point];
      CheckOutput("CHARMM-based plane activation and boundary controls",
                  Evaluate(force, coordinates), energies[mode][point],
                  expected);
    }
  }

  // JEG261006: Reference values collected from CHARMM identify INSIDE as the
  // default activation mode, rather than SYMMETRIC.
  force.reset();
  force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, 2.0);
  {
    std::vector<float4> coordinates(NUM_ATOMS,
                                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    coordinates[0].x = -1.0f;
    CheckOutput("CHARMM default INSIDE activation",
                Evaluate(force, coordinates), 0.0, expected);
    // JEG261006: Reference values collected from CHARMM show zero contribution
    // for OUTSIDE plane restraints with a negative offset.
    force.reset();
    force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, -2.0, A::OUTSIDE);
    CheckOutput("CHARMM negative-offset OUTSIDE behavior",
                Evaluate(force, coordinates), 0.0, expected);
  }

  // JEG261006: Reference values collected from CHARMM include negative, zero,
  // and positive force constants.
  for (const double k : {-2.0, 0.0, 2.0}) {
    force.reset();
    force.addPlane(selection, ORIGIN, X_DIRECTION, k, 2.0, A::SYMMETRIC);
    std::vector<float4> coordinates(NUM_ATOMS,
                                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    coordinates[0].x = 3.0f;
    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    expected[0].x = -k;
    CheckOutput("CHARMM reference values: signed force constant",
                Evaluate(force, coordinates), k / 2.0, expected);
  }

  for (const double offset : {-2.0, 0.0, 2.0}) {
    force.reset();
    force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, offset, A::SYMMETRIC);
    std::vector<float4> coordinates(NUM_ATOMS,
                                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    coordinates[0].x = 3.0f;
    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    expected[0].x = -2.0 * (3.0 - offset);
    CheckOutput("CHARMM-based signed-offset control",
                Evaluate(force, coordinates), (3.0 - offset) * (3.0 - offset),
                expected);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceAtomwiseAccumulation", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0] = make_float4(3.0f, 0.0f, 0.0f, 0.0f);
  coordinates[1] = make_float4(4.0f, 3.0f, 0.0f, 0.0f);
  coordinates[2] = make_float4(1.0f, 4.0f, 0.0f, 0.0f);
  std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));

  SECTION("ThreeSelectedAtoms") {
    coordinates[1].y = 0.0f;
    coordinates[2].y = 0.0f;
    AtomSelection selection(NUM_ATOMS);
    selection.set(0);
    selection.set(1);
    selection.set(2);
    force.addPlane(selection, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
    expected[0].x = -2.0;
    expected[1].x = -4.0;
    expected[2].x = 2.0;
    CheckOutput("three selected atoms", Evaluate(force, coordinates), 6.0,
                expected);
  }

  SECTION("MultiplePlanarTerms") {
    coordinates[0] = make_float4(3.0, 2.0, 1.0, 0.0);
    coordinates[1] = make_float4(4.0, 1.0, 2.0, 0.0);
    force.addPlane(AtomReference(psf, 0), ORIGIN, X_DIRECTION, 1.0, 1.0,
                   A::SYMMETRIC);
    force.addPlane(AtomReference(psf, 1), ORIGIN, {0.0, 1.0, 0.0}, 1.5, 1.0,
                   A::SYMMETRIC);
    expected[0].x = -2.0;
    CheckOutput("multiple plane terms", Evaluate(force, coordinates), 2.0,
                expected);
  }

  SECTION("PlanarOnlyOverlapControl") {
    // JEG261006: Reference values collected from CHARMM include overlapping
    // selections with mixed geometries. This planar-only native control checks
    // the same additive rule with expected values calculated for the setup
    // below.
    AtomSelection first(NUM_ATOMS), second(NUM_ATOMS);
    first.set(0);
    first.set(1);
    second.set(1);
    second.set(2);
    force.addPlane(first, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
    force.addPlane(second, ORIGIN, {0.0, 1.0, 0.0}, 2.0, 2.0, A::SYMMETRIC);
    expected[0].x = -2.0;
    expected[1] = make_double3(-4.0, -2.0, 0.0);
    expected[2].y = -4.0;
    CheckOutput("overlap", Evaluate(force, coordinates), 10.0, expected);
  }

  SECTION("DuplicatePlanarTerms") {
    AtomReference atom(psf, 0);
    force.addPlane(atom, ORIGIN, X_DIRECTION, 1.0, 2.0, A::SYMMETRIC);
    expected[0].x = -1.0;
    CheckOutput("one planar term", Evaluate(force, coordinates), 0.5, expected);
    force.addPlane(atom, ORIGIN, X_DIRECTION, 1.0, 2.0, A::SYMMETRIC);
    expected[0].x = -2.0;
    CheckOutput("two identical planar terms", Evaluate(force, coordinates), 1.0,
                expected);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceInsertionOrder", "[mmfp]", float,
                   double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  const std::array<std::array<double, 3>, 6> constants = {{{1e16, -1e16, 1.0},
                                                           {-1e16, 1e16, 1.0},
                                                           {1e16, 1.0, -1e16},
                                                           {-1e16, 1.0, 1e16},
                                                           {1.0, 1e16, -1e16},
                                                           {1.0, -1e16, 1e16}}};
  // JEG261006: Reference values collected from CHARMM depend on term insertion
  // order. Cancelling the large terms first gives E=0.5 and physical Fx=-1; the
  // other four orders give zero energy and physical force.
  const std::array<const char *, 6> ids = {
      "K = (+1e16, -1e16, +1)", "K = (-1e16, +1e16, +1)",
      "K = (+1e16, +1, -1e16)", "K = (-1e16, +1, +1e16)",
      "K = (+1, +1e16, -1e16)", "K = (+1, -1e16, +1e16)"};
  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0].x = 3.0;
  for (std::size_t i = 0; i < constants.size(); i++) {
    force.reset();
    for (const double k : constants[i]) {
      force.addPlane(AtomReference(psf, 0), ORIGIN, X_DIRECTION, k, 2.0,
                     A::SYMMETRIC);
    }
    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    expected[0].x = (i < 2) ? -1.0 : 0.0;
    CheckOutput(std::string("insertion order / ") + ids[i],
                Evaluate(force, coordinates), (i < 2) ? 0.5 : 0.0, expected);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceResetClearAndInitialize", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  const auto masses = force.getMasses();
  force.initialize(NUM_ATOMS, BOX_DIMS);

  // JEG261006: Addition after initialize must invalidate derived device
  // membership tables.
  force.addPlane(AtomReference(psf, 0), ORIGIN, X_DIRECTION, 2.0, 2.0,
                 A::SYMMETRIC);
  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0] = make_float4(3.0f, 2.0f, 0.0f, 0.0f);
  std::vector<double3> zero(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
  auto expected = zero;
  expected[0].x = -2.0;
  CheckOutput("after initialize", Evaluate(force, coordinates), 1.0, expected);

  force.clear();
  CHECK(force.getNumTerms() == 1);
  CHECK(force.getMasses() == masses);
  CheckOutput("clear output only", ReadOutput(force), 0.0, zero);
  CheckOutput("clear preserved term", Evaluate(force, coordinates), 1.0,
              expected);
  expected[0].x = -4.0;
  CheckOutput("calcEnergy false preserves previous energy",
              Evaluate(force, coordinates, false, false, false), 1.0, expected);

  // JEG261006: Reference values collected from CHARMM confirm that GEO RESET
  // removes installed terms without changing PSF masses. The checks below also
  // verify the separate native output-clearing semantics.
  force.reset();
  force.reset();
  CHECK(force.getNumTerms() == 0);
  CHECK(force.getMasses() == masses);
  CheckOutput("reset is distinct from output clearing", ReadOutput(force), 1.0,
              expected);
  CheckOutput("no terms after reset", Evaluate(force, coordinates), 0.0, zero);

  // JEG261006: Reference values collected from CHARMM confirm that newly added
  // terms contribute after reset. This native reuse control uses only a plane.
  force.addPlane(AtomReference(psf, 0), ORIGIN, {0.0, 1.0, 0.0}, 2.0, 1.0,
                 A::SYMMETRIC);
  expected = zero;
  expected[0].y = -2.0;
  CheckOutput("new term after reset", Evaluate(force, coordinates), 1.0,
              expected);
  CHECK(force.getMasses() == masses);

  apo_test::CheckApoCharmmError(
      [&](void) -> void {
        force.initialize(NUM_ATOMS + 1, BOX_DIMS);
        return;
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Initialization atom count mismatch; expected 12, observed 13");
  apo_test::CheckApoCharmmError(
      [&](void) -> void {
        force.setBoxDimensions({1.0, 2.0});
        return;
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Box-dimension array size mismatch; expected 3, observed 2");
  for (const double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.setBoxDimensions({bad, 30.0, 40.0});
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index 0 must be " +
            std::string(std::isfinite(bad) ? "positive" : "finite") +
            "; observed " + std::to_string(bad));
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceBoxAndDeferredVirial", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  force.initialize(NUM_ATOMS, BOX_DIMS);
  force.addPlane(AtomReference(psf, 0), {9.0, 0.0, 0.0}, X_DIRECTION, 2.0, 2.0,
                 A::SYMMETRIC);

  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0].x = -10.5f;
  std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
  expected[0].x = 43.0;

  CheckOutput("original box", Evaluate(force, coordinates), 462.25, expected);

  force.setBoxDimensions({24.0, 36.0, 48.0});
  const Output result = Evaluate(force, coordinates, true, true);
  CheckOutput("changed box", result, 462.25, expected);

  // JEG261007: This checks the deferred-virial capability, not agreement with
  // virial reference values collected from CHARMM
  ForceView view(&force);
  CHECK(!view.contributesVirial());
  CHECK(view.supportsPBC(PBC::P1));
  CHECK(!view.supportsPBC(PBC::P21));
  CHECK(!view.supportsPBC(PBC::UNSET));
  apo_test::CheckVectorsClose<double>(
      "deferred virial remains cleared",
      std::vector<double>(result.virial.begin(), result.virial.end()),
      std::vector<double>(9, 0.0), 0.0);
}

TEMPLATE_TEST_CASE("GeometricRestraintForceGrowingSelectionsAndTerms", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  SECTION("MoreThan100SelectedAtomsAndANonmultipleOfBlockSize") {
    constexpr int count = 257;
    G force(MakePsf(count));
    AtomSelection all(count, AtomSelection::InitialValue::ALL);

    force.addPlane(all, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
    CheckOutput(
        "257 atomwise contributions",
        Evaluate(force,
                 std::vector<float4>(count, make_float4(3.0, 0.0, 0.0, 0.0))),
        257.0, std::vector<double3>(count, make_double3(-2.0, 0.0, 0.0)));
  }

  // JEG261007: Reference values collected from CHARMM confirm growth to 13 and
  // 14 unit-energy terms. The 300-term case is an additional native stress
  // test.
  SECTION("CharmmTermCapacityReferencesAndNativeGrowthExtension") {
    auto psf = MakePsf();
    G force(psf);
    std::vector<float4> coordinates(NUM_ATOMS,
                                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    coordinates[0].x = 3.0;
    for (int i = 1; i <= 300; i++) {
      force.addPlane(AtomReference(psf, 0), ORIGIN, X_DIRECTION, 2.0, 2.0,
                     A::SYMMETRIC);
      if ((i == 13) || (i == 14) || (i == 300)) {
        std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
        expected[0].x = -2.0 * static_cast<double>(i);
        CHECK(force.getNumTerms() == static_cast<std::size_t>(i));
        CheckOutput("term-capacity reference and native extension",
                    Evaluate(force, coordinates), static_cast<double>(i),
                    expected);
      }
    }
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceCheckedNumericalLimits", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  AtomReference atom(psf, 0);

  force.addPlane(atom, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);

  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0].x = 3.0;

  const Output previous = Evaluate(force, coordinates);

  force.reset();
  coordinates[0].x = 0.0;

  SECTION("CharmmReferenceForceOutsideTheNativeFixedPointRange") {
    // JEG261007: Reference values collected from CHARMM are E=5e-87 kcal/mol
    // and physical Fx=-1e7 kcal/mol/A. CHARMM accepts this case. The expected
    // exception below checks a native storage limit, not a CHARMM rejection:
    // both native specialization remains fixed-point AT.
    force.addPlane(atom, {-1e-93, 0.0, 0.0}, X_DIRECTION, 1e100, 0.0,
                   A::SYMMETRIC);
    CudaContainer<float4> xyzq(coordinates);
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.calcForce(xyzq.getDeviceArray().data(), true, false);
          return;
        },
        ApoCharmmErrorCode::Runtime,
        "MMFP gradient exceeds the fixed-point force range");
  }

  SECTION("NonfiniteEvaluatedArithmetic") {
    force.addPlane(atom, ORIGIN, X_DIRECTION, 1e308, -1e308, A::SYMMETRIC);
    CudaContainer<float4> xyzq(coordinates);
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.calcForce(xyzq.getDeviceArray().data(), true, false);
          return;
        },
        ApoCharmmErrorCode::Runtime,
        "MMFP evaluation produced a non-finite value");
  }

  SECTION("NullCoordinatePointerWithATerm") {
    force.addPlane(atom, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          force.calcForce(nullptr, true, false);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Coordinate-charge array must not be null");
  }

  CheckOutput("failed evaluation preserves output", ReadOutput(force),
              previous.energy, previous.physicalForces);

  force.reset();
  force.addPlane(atom, ORIGIN, X_DIRECTION, 5e6, 0.0, A::SYMMETRIC);
  coordinates[0].x = 1.0;

  const Output first = Evaluate(force, coordinates);
  CudaContainer<float4> xyzq(coordinates);

  apo_test::CheckApoCharmmError(
      [&](void) -> void {
        force.calcForce(xyzq.getDeviceArray().data(), true, false);
        return;
      },
      ApoCharmmErrorCode::Runtime,
      "MMFP accumulated gradient exceeds the fixed-point force range");
  CheckOutput("integer accumulated failure preserves output", ReadOutput(force),
              first.energy, first.physicalForces);
}

TEST_CASE("GeometricRestraintForceDoubleConversionAccuracy", "[mmfp]") {
  auto psf = MakePsf();
  GeoForce<float> single(psf);
  GeoForce<double> dual(psf);
  const AtomReference atom(psf, 0);

  single.addPlane(atom, {0.5, -1.0, 1.5}, {1.0, 2.0, 3.0}, 2.0, 2.0,
                  GeoForce<float>::Activation::SYMMETRIC);
  dual.addPlane(atom, {0.5, -1.0, 1.5}, {1.0, 2.0, 3.0}, 2.0, 2.0,
                GeoForce<double>::Activation::SYMMETRIC);

  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0] = make_float4(3.0f, 2.0f, 4.0f, 0.0f);

  const long double root = std::sqrt(14.0L);
  const long double deviation = 16.0L / root - 2.0L;
  std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
  expected[0] = make_double3(static_cast<double>(-2.0 * deviation / root),
                             static_cast<double>(-4.0 * deviation / root),
                             static_cast<double>(-6.0 * deviation / root));
  const double energy = static_cast<double>(deviation * deviation);

  // JEG261007: These independently calculated analytical values supplement the
  // rounded reference values collected from CHARMM for a general plane normal.
  CheckOutput("double analytical", Evaluate(dual, coordinates), energy,
              expected, 2.0e-13, 1.0e-12);
  CheckOutput("float conversion analytical", Evaluate(single, coordinates),
              energy, expected, 2.0e-13, 3.0e-7);

  const Output center = Evaluate(dual, coordinates);
  constexpr float step = 1.0f / 1024.0f;
  for (int i = 0; i < 3; i++) {
    auto plus = coordinates;
    auto minus = coordinates;
    double physicalForce = 0.0;
    if (i == 0) {
      plus[0].x += step;
      minus[0].x -= step;
      physicalForce = center.physicalForces[0].x;
    } else if (i == 1) {
      plus[0].y += step;
      minus[0].y -= step;
      physicalForce = center.physicalForces[0].y;
    } else if (i == 2) {
      plus[0].z += step;
      minus[0].z -= step;
      physicalForce = center.physicalForces[0].z;
    }
    const double plusEnergy = Evaluate(dual, plus).energy;
    const double minusEnergy = Evaluate(dual, minus).energy;
    const double numericalForce =
        -(plusEnergy - minusEnergy) / (2.0 * static_cast<double>(step));
    CHECK(std::abs(numericalForce - physicalForce) <= 2.0e-9);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceManagerIntegration", "[mmfp]", float,
                   double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  REQUIRE(psf->getNumAtoms() == 2);

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  auto fm = ctx->getForceManager();
  auto force = std::make_shared<G>(psf);

  SECTION("SubscriptionBeforeManagerInitialization") {
    fm->subscribe(force, "mmfp", force->getStream(), force->getForce(),
                  force->getEnergyVirial());
    ctx->setBoxDimensions({30.0, 32.0, 34.0});
  }

  SECTION("SubscriptionAfterManagerInitialization") {
    ctx->setBoxDimensions({30.0, 32.0, 34.0});
    fm->subscribe(force, "mmfp", force->getStream(), force->getForce(),
                  force->getEnergyVirial());
  }

  REQUIRE(fm->isInitialized());
  ctx->setCoordinates(std::vector<double3>{make_double3(3.0, 0.0, 0.0),
                                           make_double3(1.0, 2.0, 3.0)});
  ctx->useHolonomicConstraints(false);

  const ManagerOutput baseline = EvaluateManager(ctx);
  const auto forceOwner = force->getForce();
  const auto energyOwner = force->getEnergyVirial();
  const auto streamOwner = force->getStream();

  force->addPlane(AtomReference(psf, 0), ORIGIN, X_DIRECTION, 2.0, 2.0,
                  A::SYMMETRIC);
  CheckManagerDelta(baseline, EvaluateManager(ctx), 1.0,
                    {make_double3(2.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)});

  force->addPlane(AtomReference(psf, 1), ORIGIN, {0.0, 1.0, 0.0}, 3.0, 1.0,
                  A::SYMMETRIC);
  CheckManagerDelta(baseline, EvaluateManager(ctx), 2.5,
                    {make_double3(2.0, 0.0, 0.0), make_double3(0.0, 3.0, 0.0)});

  force->addCylinder(AtomReference(psf, 0), ORIGIN, {0.0, 0.0, 1.0}, 4.0, 2.0,
                     A::SYMMETRIC);
  CheckManagerDelta(baseline, EvaluateManager(ctx), 4.5,
                    {make_double3(6.0, 0.0, 0.0), make_double3(0.0, 3.0, 0.0)});

  force->addSphere(AtomReference(psf, 0), ORIGIN, 6.0, 2.0, A::SYMMETRIC);
  CheckManagerDelta(
      baseline, EvaluateManager(ctx), 7.5,
      {make_double3(12.0, 0.0, 0.0), make_double3(0.0, 3.0, 0.0)});

  ForceView mixedView(force.get());
  CHECK(mixedView.supportsPBC(PBC::P1));
  CHECK(!mixedView.supportsPBC(PBC::P21));
  CHECK(!mixedView.supportsPBC(PBC::UNSET));
  CHECK(!mixedView.contributesVirial());

  CHECK(force->getForce() == forceOwner);
  CHECK(force->getEnergyVirial() == energyOwner);
  CHECK(force->getStream() == streamOwner);

  apo_test::CheckApoCharmmError(
      [&](void) -> void {
        fm->setPeriodicBoundaryCondition(PBC::P21);
        return;
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Subscribed force does not support periodic boundary condition value 2");

  apo_test::CheckApoCharmmError(
      [&](void) -> void {
        fm->setPeriodicBoundaryCondition(PBC::UNSET);
        return;
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 0");

  CHECK(fm->getPeriodicBoundaryCondition() == PBC::P1);
  force->reset();
  CheckManagerDelta(baseline, EvaluateManager(ctx), 0.0,
                    {make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)});

  fm->unsubscribe(force);
  CheckManagerDelta(baseline, EvaluateManager(ctx), 0.0,
                    {make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)});

  // JEG261007: Reject an incompatible PBC at subscription, not only when
  // changing it
  fm->setPeriodicBoundaryCondition(PBC::P21);
  apo_test::CheckApoCharmmError(
      [&](void) -> void {
        fm->subscribe(force, "mmfp", force->getStream(), force->getForce(),
                      force->getEnergyVirial());
        return;
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Subscribed force does not support periodic boundary condition value 2");
}

TEMPLATE_TEST_CASE("GeometricRestraintForceRadialOracle", "[mmfp]", float,
                   double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;
  struct Case {
    const char *id;
    bool cylinder;
    std::array<double, 3> position, origin, direction;
    double energy;
    std::array<double, 3> physicalForce;
    double energyTolerance;
  };

  // JEG261007: CHARMM energy and physical-force value for SYMMETRIC terms with
  // forceConstant = 2 and offset = 2. Positions are exactly representable in
  // float4, so coordinate conversion does not change these inputs.
  const std::vector<Case> cases = {
      {"cylinder x axis",
       true,
       {4.0, 3.0, 0.0},
       {0.0, 0.0, 0.0},
       {1.0, 0.0, 0.0},
       1.0,
       {0.0, -2.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder y axis",
       true,
       {3.0, 4.0, 0.0},
       {0.0, 0.0, 0.0},
       {0.0, 1.0, 0.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder z axis",
       true,
       {3.0, 0.0, 4.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder general unit",
       true,
       {4.0, 1.0, -1.0},
       {1.0, -2.0, 3.0},
       {0.2672612419124244, 0.5345224838248488, 0.8017837257372732},
       14.254885700000001,
       {-4.2024438653, -4.4826067897, 4.3892191482},
       4.9999999999999998e-08},
      {"cylinder general nonunit",
       true,
       {4.0, 1.0, -1.0},
       {1.0, -2.0, 3.0},
       {2.0, 4.0, 6.0},
       14.254885700000001,
       {-4.2024438653, -4.4826067897, 4.3892191482},
       4.9999999999999998e-08},
      {"cylinder general reversed",
       true,
       {4.0, 1.0, -1.0},
       {1.0, -2.0, 3.0},
       {-1.0, -2.0, -3.0},
       14.254885700000001,
       {-4.2024438653, -4.4826067897, 4.3892191482},
       4.9999999999999998e-08},
      {"cylinder axis point",
       true,
       {0.0, 0.0, 5.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       0.0,
       {0.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder radial one",
       true,
       {1.0, 0.0, 5.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       1.0,
       {2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder radial target",
       true,
       {2.0, 0.0, 5.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       0.0,
       {0.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder radial three",
       true,
       {3.0, 0.0, 5.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder parallel translation a",
       true,
       {3.0, 0.0, -7.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder parallel translation b",
       true,
       {3.0, 0.0, 11.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder reference inside cell",
       true,
       {5.0, 3.0, 7.0},
       {2.0, 3.0, 4.0},
       {0.0, 0.0, 1.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"cylinder reference outside cell",
       true,
       {25.0, 3.0, 7.0},
       {22.0, 3.0, 4.0},
       {0.0, 0.0, 1.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"sphere general center center",
       false,
       {1.25, -2.5, 3.75},
       {1.25, -2.5, 3.75},
       {0.0, 0.0, 0.0},
       0.0,
       {0.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"sphere general center inside",
       false,
       {2.25, -2.5, 3.75},
       {1.25, -2.5, 3.75},
       {0.0, 0.0, 0.0},
       1.0,
       {2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"sphere general center target",
       false,
       {3.25, -2.5, 3.75},
       {1.25, -2.5, 3.75},
       {0.0, 0.0, 0.0},
       0.0,
       {0.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"sphere general center outside",
       false,
       {4.25, -2.5, 3.75},
       {1.25, -2.5, 3.75},
       {0.0, 0.0, 0.0},
       1.0,
       {-2.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"P1 primary atom primary origin",
       false,
       {3.0, 2.0, 1.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 0.0},
       3.0333704500000001,
       {-2.7928650971, -1.8619100647, -0.9309550324},
       5.0000000000000001e-09},
      {"P1 atom shift positive x",
       false,
       {23.0, 2.0, 1.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 0.0},
       445.56623999999999,
       {-42.0187706338, -3.6538061421, -1.826903071},
       5.0000000000000004e-06},
      {"P1 origin shift positive x",
       false,
       {3.0, 2.0, 1.0},
       {20.0, 0.0, 0.0},
       {0.0, 0.0, 0.0},
       229.414287,
       {30.0341594641, -3.5334305252, -1.7667152626},
       4.9999999999999998e-07},
      {"P1 both shift positive x",
       false,
       {23.0, 2.0, 1.0},
       {20.0, 0.0, 0.0},
       {0.0, 0.0, 0.0},
       3.0333704500000001,
       {-2.7928650971, -1.8619100647, -0.9309550324},
       5.0000000000000001e-09},
      {"P1 atom shift positive xyz",
       false,
       {23.0, 32.0, 41.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 0.0},
       3010.5269199999998,
       {-44.3822261186, -61.7491841649, -79.1161422113},
       5.0000000000000004e-06},
      {"P1 sphere crosses x boundary a",
       false,
       {9.5, 0.0, 0.0},
       {9.0, 0.0, 0.0},
       {0.0, 0.0, 0.0},
       2.25,
       {3.0, 0.0, 0.0},
       4.9999999999999998e-07},
      {"P1 sphere crosses x boundary b",
       false,
       {-10.5, 0.0, 0.0},
       {9.0, 0.0, 0.0},
       {0.0, 0.0, 0.0},
       306.25,
       {35.0, 0.0, 0.0},
       5.0000000000000002e-05},
      {"P1 cylinder primary",
       true,
       {1.0, 2.0, 3.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       0.055728090000000001,
       {-0.211145618, -0.422291236, 0.0},
       5.0000000000000001e-09},
      {"P1 cylinder atom shift x",
       true,
       {21.0, 2.0, 3.0},
       {0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       364.61990800000001,
       {-38.0180181096, -3.6207636295, 0.0},
       4.9999999999999998e-07},
      {"P1 cylinder reference shift x",
       true,
       {1.0, 2.0, 3.0},
       {20.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       292.580107,
       {34.0219781883, -3.5812608619, 0.0},
       4.9999999999999998e-07},
      {"P1 cylinder both shift x",
       true,
       {21.0, 2.0, 3.0},
       {20.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       0.055728090000000001,
       {-0.211145618, -0.422291236, 0.0},
       5.0000000000000001e-09},
      {"P1 cylinder cross x boundary",
       true,
       {-10.5, 0.0, 0.0},
       {9.0, 0.0, 0.0},
       {0.0, 0.0, 1.0},
       306.25,
       {35.0, 0.0, 0.0},
       5.0000000000000002e-05},
      {"P1 cylinder general cross xyz",
       true,
       {-10.5, -15.5, -20.5},
       {9.0, 14.0, 19.0},
       {1.0, 2.0, 3.0},
       17.8017319,
       {7.3656566134, 1.8414141533, -3.6828283067},
       4.9999999999999998e-08}};

  auto psf = MakePsf();
  G force(psf);
  const AtomReference atom(psf, 0);
  for (const Case &test : cases) {
    INFO(test.id);
    force.reset();
    force.initialize(NUM_ATOMS, BOX_DIMS);
    if (test.cylinder) {
      force.addCylinder(atom, test.origin, test.direction, 2.0, 2.0,
                        A::SYMMETRIC);
    } else
      force.addSphere(atom, test.origin, 2.0, 2.0, A::SYMMETRIC);

    std::vector<float4> coordinates(NUM_ATOMS,
                                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    for (const double x : test.position)
      REQUIRE(static_cast<double>(static_cast<float>(x)) == x);
    coordinates[0] = make_float4(static_cast<float>(test.position[0]),
                                 static_cast<float>(test.position[1]),
                                 static_cast<float>(test.position[2]), 0.0f);

    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    expected[0] = make_double3(test.physicalForce[0], test.physicalForce[1],
                               test.physicalForce[2]);
    const double magnitude = std::max({std::abs(test.physicalForce[0]),
                                       std::abs(test.physicalForce[1]),
                                       std::abs(test.physicalForce[2])});
    const double forceTolerance =
        1.0e-10 + 0.5 * INV_FORCE_SCALE_TEST +
        32.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, magnitude) +
        ((std::is_same<TestType, float>::value)
             ? 0.5 * std::numeric_limits<float>::epsilon() * magnitude
             : 0.0);
    const double energyTolerance =
        test.energyTolerance + 32.0 * std::numeric_limits<double>::epsilon() *
                                   std::max(1.0, std::abs(test.energy));

    CheckOutput(test.id, Evaluate(force, coordinates), test.energy, expected,
                energyTolerance, forceTolerance);

    // JEG261007: CHARMM uses raw Cartesian distances for these P1 restraints,
    // so changing only the box dimensions must leave energy and force
    // unchanged.
    force.setBoxDimensions({24.0, 36.0, 48.0});
    CheckOutput(std::string(test.id) + " changed box",
                Evaluate(force, coordinates), test.energy, expected,
                energyTolerance, forceTolerance);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceRadialActivation", "[mmfp]", float,
                   double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  const AtomReference atom(psf, 0);
  const std::array<A, 3> modes = {A::SYMMETRIC, A::INSIDE, A::OUTSIDE};
  // JEG261007: Rows are SYMMETRIC, INSIDE, OUTSIDE; columns are q=0,1,2,3
  const std::array<std::array<double, 4>, 3> energies = {
      {{0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 0.0}}};
  const std::array<std::array<double, 4>, 3> forces = {
      {{0.0, 2.0, 0.0, -2.0}, {0.0, 0.0, 0.0, -2.0}, {0.0, 2.0, 0.0, 0.0}}};

  for (const bool cylinder : {true, false}) {
    G force(psf);

    for (std::size_t i = 0; i < modes.size(); i++) {
      force.reset();

      if (cylinder)
        force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, 2.0, 2.0, modes[i]);
      else
        force.addSphere(atom, ORIGIN, 2.0, 2.0, modes[i]);

      for (int j = 0; j < 4; j++) {
        INFO("cylinder=" << cylinder << ", mode=" << i << ", q=" << j);

        std::vector<float4> coordinates(NUM_ATOMS,
                                        make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        coordinates[0] = make_float4(static_cast<float>(j), 0.0f,
                                     cylinder ? 7.0f : 0.0f, 0.0f);

        std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
        expected[0].x = forces[i][j];

        CheckOutput("radial activation and boundary controls",
                    Evaluate(force, coordinates), energies[i][j], expected);
      }
    }

    // JEG261007: CHARMM accepts signed force constants and offsets. Preserve
    // their signs in radial terms, as in the plane API.
    for (const double k : {-2.0, 0.0, 2.0}) {
      for (const double offset : {-2.0, 0.0, 2.0}) {
        for (const A mode : modes) {
          force.reset();

          if (cylinder)
            force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, k, offset, mode);
          else
            force.addSphere(atom, ORIGIN, k, offset, mode);

          std::vector<float4> coordinates(NUM_ATOMS,
                                          make_float4(0.0f, 0.0f, 0.0f, 0.0f));
          coordinates[0].x = 3.0f;

          const double deviation = 3.0 - offset;
          const bool active = mode != A::OUTSIDE;
          const double energy = active ? 0.5 * k * deviation * deviation : 0.0;
          std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
          expected[0].x = active ? -k * deviation : 0.0;

          CheckOutput("signed radial parameters", Evaluate(force, coordinates),
                      energy, expected);
        }
      }
    }

    // JEG261007: Defaults are offset=0, INSIDE. D=2 with no activation is
    // inactive at q=1.
    for (const bool defaultOffset : {true, false}) {
      force.reset();

      if (cylinder) {
        if (defaultOffset)
          force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, 2.0);
        else
          force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, 2.0, 2.0);
      } else {
        if (defaultOffset)
          force.addSphere(atom, ORIGIN, 2.0);
        else
          force.addSphere(atom, ORIGIN, 2.0, 2.0);
      }

      std::vector<float4> coordinates(NUM_ATOMS,
                                      make_float4(0.0f, 0.0f, 0.0f, 0.0f));
      coordinates[0].x = 1.0f;

      std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
      expected[0].x = defaultOffset ? -2.0 : 0.0;

      CheckOutput("radial API defaults", Evaluate(force, coordinates),
                  defaultOffset ? 1.0 : 0.0, expected);
    }
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceRadialSingularNeighborhood",
                   "[mmfp]", float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;
  struct Case {
    const char *id;
    double x, y;
    bool suppressed;
  };

  // JEG261007: Store tiny displacements in the double-precision origin and keep
  // the atom at zero. This tests CHARMM's radial suppression without float4
  // coordinate rounding moving the input across the threshold.
  const std::array<Case, 5> translated = {
      {{"exact center/axis", 0.0, 0.0, true},
       {"axis-aligned radius below suppression threshold", 9.9999e-6, 0.0,
        true},
       {"axis-aligned radius whose square exceeds threshold", 1.0e-5, 0.0,
        false},
       {"squared oblique radius at threshold", 6.0e-6, 8.0e-6, true},
       {"squared oblique radius above threshold", 6.0e-6, 8.0001e-6, false}}};

  auto psf = MakePsf();
  const AtomReference atom(psf, 0);

  for (const bool cylinder : {true, false}) {
    G force(psf);

    for (const A mode : {A::SYMMETRIC, A::OUTSIDE, A::INSIDE}) {
      const double offset = (mode == A::INSIDE) ? -2.0 : 2.0;

      if (cylinder) {
        force.reset();
        force.addCylinder(atom, {1.0, -2.0, 3.0}, {1.0, 2.0, 3.0}, 2.0, offset,
                          mode);

        std::vector<float4> coordinates(NUM_ATOMS,
                                        make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        coordinates[0] = make_float4(3.0f, 2.0f, 9.0f, 0.0f);

        CheckOutput(
            "general-axis singular suppression", Evaluate(force, coordinates),
            0.0, std::vector<double3>(NUM_ATOMS, make_double3(0.0, 0.0, 0.0)));
      }

      for (const Case &test : translated) {
        INFO(test.id << ", cylinder=" << cylinder
                     << ", mode=" << static_cast<int>(mode));
        force.reset();

        const std::array<double, 3> origin = {-test.x, -test.y,
                                              cylinder ? -5.0 : 0.0};

        if (cylinder)
          force.addCylinder(atom, origin, {0.0, 0.0, 1.0}, 2.0, offset, mode);
        else
          force.addSphere(atom, origin, 2.0, offset, mode);

        std::vector<float4> coordinates(NUM_ATOMS,
                                        make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
        double energy = 0.0;

        if (!test.suppressed) {
          const long double x = test.x, y = test.y;
          const long double q = std::sqrt(x * x + y * y);
          const long double deviation = q - static_cast<long double>(offset);
          energy = static_cast<double>(deviation * deviation);
          expected[0] =
              make_double3(static_cast<double>(-2.0L * deviation * x / q),
                           static_cast<double>(-2.0L * deviation * y / q), 0.0);
        }

        const double tolerance =
            test.suppressed
                ? 0.0
                : 1.0e-12 + 0.5 * INV_FORCE_SCALE_TEST +
                      (std::is_same<TestType, float>::value ? 2.5e-7 : 0.0);

        CheckOutput("translated axis and center suppression controls",
                    Evaluate(force, coordinates), energy, expected,
                    test.suppressed ? 0.0 : 1e-12, tolerance);
      }

      // JEG261007: Rounding 1e-5 to float moves it below the suppression
      // threshold. The next larger float lies above it and must contribute
      // normally. Expected values must therefore use the actual stored
      // coordinates.
      const float rounded = 1.0e-5f;
      const float above =
          std::nextafter(rounded, std::numeric_limits<float>::infinity());
      REQUIRE(static_cast<double>(rounded) < 1.0e-5);
      REQUIRE(static_cast<double>(above) > 1.0e-5);

      for (const float radial : {rounded, above}) {
        force.reset();

        if (cylinder)
          force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, 2.0, offset, mode);
        else
          force.addSphere(atom, ORIGIN, 2.0, offset, mode);

        std::vector<float4> coordinates(NUM_ATOMS,
                                        make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        coordinates[0] =
            make_float4(radial, 0.0f, cylinder ? 5.0f : 0.0f, 0.0f);

        std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
        const bool suppressed = radial == rounded;
        const double deviation = static_cast<double>(radial) - offset;
        expected[0].x = suppressed ? 0.0 : -2.0 * deviation;
        const double tolerance =
            suppressed
                ? 0.0
                : 1.0e-12 + 0.5 * INV_FORCE_SCALE_TEST +
                      (std::is_same<TestType, float>::value ? 2.5e-7 : 0.0);

        CheckOutput("native float4 threshold breacket",
                    Evaluate(force, coordinates),
                    suppressed ? 0.0 : deviation * deviation, expected,
                    suppressed ? 0.0 : 1.0e-12, tolerance);
      }
    }
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceRadialTypedInputsAndValidation",
                   "[mmfp]", float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();

  for (const bool cylinder : {true, false}) {
    INFO("cylinder=" << cylinder);

    G force(psf);
    const auto add = [&](const auto &input,
                         const std::array<double, 3> &origin = ORIGIN,
                         const double k = 2.0, const double offset = 2.0,
                         const A mode = A::SYMMETRIC) -> void {
      if (cylinder)
        force.addCylinder(input, origin, {0.0, 0.0, 1.0}, k, offset, mode);
      else
        force.addSphere(input, origin, k, offset, mode);
      return;
    };

    {
      AtomSelection selection(NUM_ATOMS);
      selection.set(0);

      auto origin = ORIGIN;
      std::array<double, 3> direction = {0.0, 0.0, 1.0};

      if (cylinder) {
        force.addCylinder(selection, origin, direction, 2.0, 2.0, A::SYMMETRIC);
      } else
        force.addSphere(selection, origin, 2.0, 2.0, A::SYMMETRIC);
      direction = X_DIRECTION;
      selection.clear();
      selection.set(1);
      origin[0] = 99.0;
    }

    std::weak_ptr<CharmmPSF> weak;
    {
      auto foreign = MakePsf(NUM_ATOMS + 1);
      weak = foreign;
      add(AtomReference(foreign, 0));
    }
    CHECK(weak.expired());
    REQUIRE(force.getNumTerms() == 2);

    std::vector<float4> coordinates(NUM_ATOMS,
                                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    coordinates[0].x = 3.0f;

    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    expected[0].x = -4.0;

    const Output previous = Evaluate(force, coordinates);
    CheckOutput("copied typed inputs", previous, 2.0, expected);

    AtomSelection empty(NUM_ATOMS);
    AtomSelection mismatch(NUM_ATOMS + 1, AtomSelection::InitialValue::ALL);
    const AtomReference outOfRange(MakePsf(NUM_ATOMS + 1), NUM_ATOMS);
    const AtomReference atom(psf, 0);
    const double bad = std::numeric_limits<double>::quiet_NaN();

    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          add(empty);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Selection must contain at least one atom; observed 0");

    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          add(mismatch);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Selection atom count mismatch; expected 12, observed 13");

    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          add(outOfRange);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom index is out of range; expected [0, 12), observed 12");

    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          add(atom, {bad, 0.0, 0.0});
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Origin component must be finite; index 0");

    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          add(atom, ORIGIN, bad);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Force constant must be finite; observed " + std::to_string(bad));

    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          add(atom, ORIGIN, 2.0, bad);
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Offset must be finite; observed " + std::to_string(bad));

    apo_test::CheckApoCharmmError(
        [&](void) -> void {
          add(atom, ORIGIN, 2.0, 2.0, static_cast<A>(99));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Invalid MMFP activation value; observed 99");

    if (cylinder) {
      apo_test::CheckApoCharmmError(
          [&](void) -> void {
            force.addCylinder(atom, ORIGIN, ORIGIN, 2.0);
            return;
          },
          ApoCharmmErrorCode::InvalidArgument,
          "Direction must have a positive finite norm");

      apo_test::CheckApoCharmmError(
          [&](void) -> void {
            force.addCylinder(atom, ORIGIN, {bad, 0.0, 1.0}, 2.0);
            return;
          },
          ApoCharmmErrorCode::InvalidArgument,
          "Direction component must be finite; index 0");
    }

    CHECK(force.getNumTerms() == 2);
    CheckOutput("failed additions preserve output", ReadOutput(force),
                previous.energy, previous.physicalForces);
    CheckOutput("failed additions preserve terms", Evaluate(force, coordinates),
                2.0, expected);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceMixedOracle", "[mmfp]", float,
                   double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  const AtomReference atom(psf, 0);

  force.addPlane(atom, ORIGIN, X_DIRECTION, 1.0, 1.0, A::SYMMETRIC);
  force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, 1.0, 1.0, A::SYMMETRIC);
  force.addSphere(atom, ORIGIN, 1.0, 1.0, A::SYMMETRIC);

  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0] = make_float4(3.0f, 2.0f, 1.0f, 0.0f);

  std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
  expected[0] = make_double3(-6.3661659799, -2.9107773199, -0.7327387581);

  const double tolerance =
      1.0e-10 + 0.5 * INV_FORCE_SCALE_TEST + 5.0e-14 +
      (std::is_same<TestType, float>::value
           ? 0.5 * std::numeric_limits<float>::epsilon() * 6.3661659799
           : 0.0);

  CheckOutput("same atom in plane, cylinder, and sphere",
              Evaluate(force, coordinates), 9.15279134, expected,
              5.0e-9 + 1.0e-13, tolerance);
}

TEMPLATE_TEST_CASE("GeometricRestraintForceMixedAtomwiseLifecycle", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);

  const auto masses = force.getMasses();
  const auto forceOwner = force.getForce();
  const auto energyOwner = force.getEnergyVirial();
  const auto streamOwner = force.getStream();

  AtomSelection first(NUM_ATOMS), second(NUM_ATOMS), allThree(NUM_ATOMS);
  first.set(0);
  first.set(1);
  second.set(1);
  second.set(2);
  allThree.set(0);
  allThree.set(1);
  allThree.set(2);

  const std::vector<double3> zero(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
  const std::array<std::array<int, 6>, 3> orders = {
      {{0, 1, 2, 0, 1, 2}, {2, 1, 0, 2, 1, 0}, {0, 0, 1, 1, 2, 2}}};
  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  for (int i = 0; i < 3; i++)
    coordinates[i].x = 3.0f;

  for (const auto &order : orders) {
    force.reset();
    force.initialize(NUM_ATOMS, BOX_DIMS);
    CheckOutput("reset removed all geometries", Evaluate(force, coordinates),
                0.0, zero);

    auto expected = zero;
    double energy = 0.0;
    std::size_t count = 0;
    for (const int geometry : order) {
      // JEG261014: Exact analytic contributions: E=1,2,3 and Fx=-2,-4,-6 per
      // selected atom for the plane, cylinder, and sphere, respectively.
      if (geometry == 0) {
        force.addPlane(first, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
        energy += 2.0;
        expected[0].x -= 2.0;
        expected[1].x -= 2.0;
      } else if (geometry == 1) {
        force.addCylinder(second, ORIGIN, {0.0, 0.0, 1.0}, 4.0, 2.0,
                          A::SYMMETRIC);
        energy += 4.0;
        expected[1].x -= 4.0;
        expected[2].x -= 4.0;
      } else {
        force.addSphere(allThree, ORIGIN, 6.0, 2.0, A::SYMMETRIC);
        energy += 9.0;
        for (int i = 0; i < 3; i++)
          expected[i].x -= 6.0;
      }
      count++;
      CHECK(force.getNumTerms() == count);
      CheckOutput("mixed addition after device synchronization",
                  Evaluate(force, coordinates), energy, expected);
    }

    CHECK(energy == 30.0);
    CHECK(expected[0].x == -16.0);
    CHECK(expected[1].x == -24.0);
    CHECK(expected[2].x == -20.0);
    force.clear();
    CHECK(force.getNumTerms() == 6);
    CheckOutput("clear preserves mixed terms", ReadOutput(force), 0.0, zero);
    CheckOutput("mixed force-only evaluation",
                Evaluate(force, coordinates, false), 0.0, expected);

    const Output withVirial = Evaluate(force, coordinates, true, true);
    CheckOutput("mixed derred virial", withVirial, energy, expected);
    for (const double component : withVirial.virial)
      CHECK(component == 0.0);
    CHECK(force.getMasses() == masses);
    CHECK(force.getForce() == forceOwner);
    CHECK(force.getEnergyVirial() == energyOwner);
    CHECK(force.getStream() == streamOwner);
  }

  // JEG261014: Moving the cylinder origin along the axis preserves its radial
  // geometry
  force.reset();
  force.addCylinder(AtomReference(psf, 0), {0.0, 0.0, 11.0}, {0.0, 0.0, -3.0},
                    2.0, 2.0, A::SYMMETRIC);
  auto expected = zero;
  expected[0].x = -2.0;
  CheckOutput("origin translation along infinite cylinder axis",
              Evaluate(force, coordinates), 1.0, expected);
}

TEMPLATE_TEST_CASE("GeometricRestraintForceMixedInsertionOrder", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  G force(psf);
  const AtomReference atom(psf, 0);
  const std::array<std::array<int, 3>, 6> orders = {
      {{0, 1, 2}, {1, 0, 2}, {0, 2, 1}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
  std::vector<float4> coordinates(NUM_ATOMS,
                                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  coordinates[0].x = 3.0f;

  for (std::size_t i = 0; i < orders.size(); i++) {
    force.reset();
    for (const int geometry : orders[i]) {
      if (geometry == 0)
        force.addPlane(atom, ORIGIN, X_DIRECTION, 1.0e16, 2.0, A::SYMMETRIC);
      else if (geometry == 1) {
        force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, -1.0e16, 2.0,
                          A::SYMMETRIC);
      } else
        force.addSphere(atom, ORIGIN, 1.0, 2.0, A::SYMMETRIC);
    }

    std::vector<double3> expected(NUM_ATOMS, make_double3(0.0, 0.0, 0.0));
    expected[0].x = (i < 2) ? -1.0 : 0.0;
    // JEG261014: CHARMM sums terms in insertion order; cancellation can discard
    // a small contribution. Preserve that order before conversion to CT.
    CheckOutput("mixed cancellation order", Evaluate(force, coordinates),
                (i < 2) ? 0.5 : 0.0, expected);
  }
}

TEMPLATE_TEST_CASE("GeometricRestraintForceMixedDynamicsStress", "[mmfp]",
                   float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  constexpr int count = 777;
  G force(MakePsf(count));
  AtomSelection all(count, AtomSelection::InitialValue::ALL);
  force.initialize(count, BOX_DIMS);
  const std::vector<float4> coordinates(count,
                                        make_float4(3.0f, 0.0f, 0.0f, 0.0f));

  // JEG261014: 303 terms and 777 atoms exercise multiple commit blocks and a
  // partial final block. Each atom contributes energy 1 and force (-2, 0, 0)
  // per term, so the expected totals are exact analytical sums.
  for (int i = 1; i <= 101; i++) {
    force.addPlane(all, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
    force.addCylinder(all, ORIGIN, {0.0, 0.0, 1.0}, 2.0, 2.0, A::SYMMETRIC);
    force.addSphere(all, ORIGIN, 2.0, 2.0, A::SYMMETRIC);
    if ((i == 1) || (i == 51) || (i == 101)) {
      const double terms = 3.0 * static_cast<double>(i);
      CHECK(force.getNumTerms() == static_cast<std::size_t>(3 * i));
      CheckOutput(
          "all atoms in every mixed term", Evaluate(force, coordinates),
          terms * static_cast<double>(count),
          std::vector<double3>(count, make_double3(-2.0 * terms, 0.0, 0.0)));
    }
  }

  force.reset();
  CHECK(force.getNumTerms() == 0);
  CheckOutput("reset large mixed tables", Evaluate(force, coordinates), 0.0,
              std::vector<double3>(count, make_double3(0.0, 0.0, 0.0)));
  force.addSphere(all, ORIGIN, 2.0, 2.0, A::SYMMETRIC);
  CheckOutput("reuse large mixed allocation", Evaluate(force, coordinates),
              static_cast<double>(count),
              std::vector<double3>(count, make_double3(-2.0, 0.0, 0.0)));
}

TEMPLATE_TEST_CASE("GeometricRestraintForceRadialTransactionalEvaluation",
                   "[mmfp]", float, double) {
  using G = GeoForce<TestType>;
  using A = typename G::Activation;

  auto psf = MakePsf();
  const AtomReference atom(psf, 0);
  for (const bool cylinder : {true, false}) {
    for (const bool calcEnergy : {true, false}) {
      INFO("cylinder=" << cylinder << ", calcEnergy=" << calcEnergy);
      G force(psf);
      force.addPlane(atom, ORIGIN, X_DIRECTION, 2.0, 2.0, A::SYMMETRIC);
      std::vector<float4> coordinates(NUM_ATOMS,
                                      make_float4(0.0f, 0.0f, 0.0f, 0.0f));
      coordinates[0].x = 3.0f;

      const Output previous = Evaluate(force, coordinates);
      if (cylinder) {
        force.addCylinder(atom, {1.0e308, 0.0, 0.0}, {0.0, 0.0, 1.0}, 2.0, 2.0,
                          A::SYMMETRIC);
      } else
        force.addSphere(atom, {1.0e308, 0.0, 0.0}, 2.0, 2.0, A::SYMMETRIC);

      CudaContainer<float4> xyzq(coordinates);

      apo_test::CheckApoCharmmError(
          [&](void) -> void {
            force.calcForce(xyzq.getDeviceArray().data(), calcEnergy, false);
            return;
          },
          ApoCharmmErrorCode::Runtime,
          "MMFP evaluation produced a non-finite value");
      CheckOutput("radial arithmetic failure preserves outputs",
                  ReadOutput(force), previous.energy, previous.physicalForces);

      force.reset();

      if (cylinder) {
        force.addCylinder(atom, ORIGIN, {0.0, 0.0, 1.0}, 1.0e7, 2.0,
                          A::SYMMETRIC);
      } else
        force.addSphere(atom, ORIGIN, 1.0e7, 2.0, A::SYMMETRIC);

      apo_test::CheckApoCharmmError(
          [&](void) -> void {
            force.calcForce(xyzq.getDeviceArray().data(), calcEnergy, false);
            return;
          },
          ApoCharmmErrorCode::Runtime,
          "MMFP gradient exceeds the fixed-point force range");
      CheckOutput("radial conversion failure preserves outputs",
                  ReadOutput(force), previous.energy, previous.physicalForces);
    }
  }
}
