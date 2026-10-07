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
