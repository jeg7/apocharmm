// BEGINLICENSE
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
#include "CudaDirectForceTypes.h"
#include "CudaEnergyVirial.h"
#include "Force.h"
#include "ForceManager.h"
#include "PBC.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr double TOLERANCE = 0.0;

class TestForce {
public:
  static constexpr bool contributesVirial = false;

  TestForce(void)
      : initializeCalls(0), clearCalls(0), calcForceCalls(0),
        setBoxDimensionsCalls(0), initializeNumAtoms(-1), lastXYZQ(nullptr),
        lastCalcEnergy(false), lastCalcVirial(false), lastBoxDimensions(),
        forceValues(std::make_shared<Force<long long int>>()),
        energyVirial(std::make_shared<CudaEnergyVirial>()) {}

  void initialize(const int numAtoms,
                  const std::vector<double> &boxDimensions) {
    initializeCalls++;
    initializeNumAtoms = numAtoms;
    lastBoxDimensions = boxDimensions;
    forceValues->realloc(numAtoms, 1.5f);
    return;
  }

  void clear(void) {
    clearCalls++;
    forceValues->clear();
    return;
  }

  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial) {
    calcForceCalls++;
    lastXYZQ = xyzq;
    lastCalcEnergy = calcEnergy;
    lastCalcVirial = calcVirial;
    return;
  }

  void setBoxDimensions(const std::vector<double> &boxDimensions) {
    setBoxDimensionsCalls++;
    lastBoxDimensions = boxDimensions;
    return;
  }

  std::shared_ptr<Force<long long int>> getForce(void) { return forceValues; }

  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void) {
    return energyVirial;
  }

public:
  int initializeCalls;
  int clearCalls;
  int calcForceCalls;
  int setBoxDimensionsCalls;
  int initializeNumAtoms;
  const float4 *lastXYZQ;
  bool lastCalcEnergy;
  bool lastCalcVirial;
  std::vector<double> lastBoxDimensions;

private:
  std::shared_ptr<Force<long long int>> forceValues;
  std::shared_ptr<CudaEnergyVirial> energyVirial;
};

class P1OnlyTestForce : public TestForce {
public:
  bool supportsPBC(const PBC pbc) const { return pbc == PBC::P1; }
};

class PBCCompatibilityTestForceManager : public ForceManager {
public:
  using ForceManager::ForceManager;

  void setPeriodicBoundaryConditionUnchecked(const PBC pbc) { m_Pbc = pbc; }

  std::vector<std::size_t> getSubscriptionVectorSizes(void) const {
    return {m_ForcePtrs.size(),   m_ForceViews.size(),
            m_ForceTags.size(),   m_ForceStreams.size(),
            m_ForceValues.size(), m_EnergyVirials.size()};
  }

  bool isReciprocalForceDirty(void) const { return m_ReciprocalForceDirty; }

  bool isDirectForceDirty(void) const { return m_DirectForceDirty; }
};

ForceManager CreateForceManager(void) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  return ForceManager(psf, prm);
}

PBCCompatibilityTestForceManager CreatePBCCompatibilityForceManager(void) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");

  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  return PBCCompatibilityTestForceManager(psf, prm);
}

} // namespace

TEST_CASE("ForceManagerDefaultConstructor") {
  ForceManager fm;

  CHECK(fm.hasCharmmContext() == false);
  CHECK(fm.getContext() == nullptr);
  CHECK(fm.getPsf() == nullptr);
  CHECK(fm.getPrm() == nullptr);
  CHECK(fm.isInitialized() == false);
  CHECK(fm.isComposite() == false);
  CHECK(fm.getChildren().empty() == true);

  apo_test::CheckVectorsClose<double>(
      "default box dimensions", fm.getBoxDimensions(),
      std::vector<double>{-9999.9999, -9999.9999, -9999.9999}, TOLERANCE);

  CHECK(fm.getKappa() == Approx(0.34).margin(TOLERANCE));
  CHECK(fm.getCutoff() == Approx(14.0).margin(TOLERANCE));
  CHECK(fm.getCtonnb() == Approx(12.0).margin(TOLERANCE));
  CHECK(fm.getCtofnb() == Approx(10.0).margin(TOLERANCE));
  apo_test::CheckVectorsEqual<int>("default FFT grid", fm.getFFTGrid(),
                                   std::vector<int>{-1, -1, -1});
  CHECK(fm.getPmeSplineOrder() == 4);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);
  CHECK(fm.getVdwType() == VDW_VFSW);
}

TEST_CASE("ForceManagerRejectsNullTopologyAndParameters") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  std::shared_ptr<CharmmPSF> nullPsf;
  std::shared_ptr<CharmmParameters> nullPrm;

  apo_test::CheckApoCharmmError(
      [&nullPsf, &prm](void) { (void)ForceManager(nullPsf, prm); },
      ApoCharmmErrorCode::InvalidArgument, "CharmmPSF must not be null");
  apo_test::CheckApoCharmmError(
      [&psf, &nullPrm](void) { (void)ForceManager(psf, nullPrm); },
      ApoCharmmErrorCode::InvalidArgument, "CharmmParameters must not be null");

  ForceManager fm;
  apo_test::CheckApoCharmmError([&fm, &nullPsf](void) { fm.setPsf(nullPsf); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "CharmmPSF must not be null");
  apo_test::CheckApoCharmmError([&fm, &nullPrm](void) { fm.setPrm(nullPrm); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "CharmmParameters must not be null");
  CHECK(fm.getPsf() == nullptr);
  CHECK(fm.getPrm() == nullptr);
}

TEST_CASE("ForceManagerConstructsFromTopologyAndParameters") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  ForceManager fm(psf, prm);

  CHECK(fm.getPsf() == psf);
  CHECK(fm.getPrm() == prm);
  CHECK(fm.getPsf()->getNumAtoms() == 2);
  CHECK(fm.isInitialized() == false);
  CHECK(fm.isComposite() == false);

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::NotInitialized,
      "Box dimensions must be set before initializing ForceManager");
}

TEST_CASE("ForceManagerSettersAndGetters") {
  auto fm = CreateForceManager();

  const std::vector<double> boxDimensions = {20.0, 21.0, 22.0};
  fm.setBoxDimensions(boxDimensions);
  apo_test::CheckVectorsClose<double>("box dimensions", fm.getBoxDimensions(),
                                      boxDimensions, TOLERANCE);

  fm.setKappa(0.45f);
  fm.setCutoff(9.0f);
  fm.setCtonnb(7.5f);
  fm.setCtofnb(8.5f);
  fm.setFFTGrid(32, 34, 36);
  fm.setPmeSplineOrder(6);
  fm.setPeriodicBoundaryCondition(PBC::P21);
  fm.setPrintEnergyDecomposition(true);

  CHECK(fm.getKappa() == Approx(0.45f).margin(TOLERANCE));
  CHECK(fm.getCutoff() == Approx(9.0f).margin(TOLERANCE));
  CHECK(fm.getCtonnb() == Approx(7.5f).margin(TOLERANCE));
  CHECK(fm.getCtofnb() == Approx(8.5f).margin(TOLERANCE));
  apo_test::CheckVectorsEqual<int>("FFT grid", fm.getFFTGrid(),
                                   std::vector<int>{32, 34, 36});
  CHECK(fm.getPmeSplineOrder() == 6);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P21);

  const std::vector<int> validVdwTypes = {VDW_VSH,  VDW_VSW, VDW_VFSW,
                                          VDW_VGSH, VDW_CUT, VDW_DBEXP};

  for (const int vdwType : validVdwTypes) {
    fm.setVdwType(vdwType);
    CHECK(fm.getVdwType() == vdwType);
  }
}

TEST_CASE("ForceManagerRejectsInvalidNonbondedParameters") {
  auto fm = CreateForceManager();
  const float nan = std::numeric_limits<float>::quiet_NaN();

  apo_test::CheckApoCharmmError([&fm, nan](void) { fm.setKappa(nan); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Kappa must be finite; observed " +
                                    std::to_string(nan));
  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setKappa(-1.0f); }, ApoCharmmErrorCode::InvalidArgument,
      "Kappa must be non-negative; observed -1.000000");

  apo_test::CheckApoCharmmError([&fm, nan](void) { fm.setCutoff(nan); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Cutoff must be finite; observed " +
                                    std::to_string(nan));
  apo_test::CheckApoCharmmError([&fm](void) { fm.setCutoff(0.0f); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Cutoff must be positive; observed 0.000000");

  apo_test::CheckApoCharmmError([&fm, nan](void) { fm.setCtonnb(nan); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Ctonnb must be finite; observed " +
                                    std::to_string(nan));
  apo_test::CheckApoCharmmError([&fm](void) { fm.setCtonnb(0.0f); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Ctonnb must be positive; observed 0.000000");

  apo_test::CheckApoCharmmError([&fm, nan](void) { fm.setCtofnb(nan); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Ctofnb must be finite; observed " +
                                    std::to_string(nan));
  apo_test::CheckApoCharmmError([&fm](void) { fm.setCtofnb(0.0f); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Ctofnb must be positive; observed 0.000000");

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setPmeSplineOrder(0); },
      ApoCharmmErrorCode::InvalidArgument,
      "PME spline order must be positive; observed 0");
  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setVdwType(0); }, ApoCharmmErrorCode::InvalidArgument,
      "Van der Waals type must be in [1, 6]; observed 0");
  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setVdwType(7); }, ApoCharmmErrorCode::InvalidArgument,
      "Van der Waals type must be in [1, 6]; observed 7");

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setPeriodicBoundaryCondition(PBC::UNSET); },
      ApoCharmmErrorCode::InvalidArgument,
      "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 0");
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setPeriodicBoundaryCondition(static_cast<PBC>(99)); },
      ApoCharmmErrorCode::InvalidArgument,
      "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 99");
}

TEST_CASE("ForceManagerRejectsInvalidBoxDimensions") {
  auto fm = CreateForceManager();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setBoxDimensions({20.0, 20.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box-dimension array size mismatch; expected 3, observed 2");
  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setBoxDimensions({20.0, 20.0, 20.0, 20.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box-dimension array size mismatch; expected 3, observed 4");
  apo_test::CheckApoCharmmError(
      [&fm, nan](void) { fm.setBoxDimensions({20.0, nan, 20.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimension at index 1 must be finite; observed " +
          std::to_string(nan));
  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setBoxDimensions({20.0, 0.0, 20.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimension at index 1 must be positive; observed 0.000000");
  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setBoxDimensions({20.0, -1.0, 20.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimension at index 1 must be positive; observed -1.000000");
}

TEST_CASE("ForceManagerRejectsInvalidInitializationInputs") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  SECTION("MissingCharmmPsf") {
    ForceManager fm;
    apo_test::CheckApoCharmmError(
        [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::NotInitialized,
        "CharmmPSF must be set before initializing ForceManager");
  }

  SECTION("MissingCharmmParameters") {
    ForceManager fm;
    fm.setPsf(psf);
    apo_test::CheckApoCharmmError(
        [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::NotInitialized,
        "CharmmParameters must be set before initializing ForceManager");
  }

  SECTION("MissingBoxDimensions") {
    ForceManager fm(psf, prm);
    apo_test::CheckApoCharmmError(
        [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::NotInitialized,
        "Box dimensions must be set before initializing ForceManager");
  }

  SECTION("CutoffLargerThanHalfOfBoxX") {
    ForceManager fm(psf, prm);
    fm.setBoxDimensions({20.0, 20.0, 20.0});
    fm.setCutoff(11.0f);
    apo_test::CheckApoCharmmError(
        [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::InvalidArgument,
        "Cutoff must be positive and not exceed half the X box dimension; "
        "cutoff 11.000000, X box dimension 20.000000");
  }

  SECTION("UninitializedAtomCount") {
    auto uninitializedPsf = std::make_shared<CharmmPSF>();
    ForceManager fm(uninitializedPsf, prm);
    fm.setBoxDimensions({20.0, 20.0, 20.0});
    fm.setCutoff(9.0f);
    apo_test::CheckApoCharmmError(
        [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::Runtime,
        "CharmmPSF atom count must be positive; observed -1");
  }

  SECTION("NonpositiveAtomCount") {
    auto emptyPsf = std::make_shared<CharmmPSF>();
    emptyPsf->setNumAtoms(0);
    ForceManager fm(emptyPsf, prm);
    fm.setBoxDimensions({20.0, 20.0, 20.0});
    fm.setCutoff(9.0f);
    apo_test::CheckApoCharmmError(
        [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::Runtime,
        "CharmmPSF atom count must be positive; observed 0");
  }
}

TEST_CASE("ForceManagerCopyConstructorCopiesConfigurationOnly") {
  ForceManager defaultManager;
  ForceManager defaultCopy(defaultManager);
  CHECK(defaultCopy.getPsf() == nullptr);
  CHECK(defaultCopy.getPrm() == nullptr);

  auto original = CreateForceManager();
  original.setBoxDimensions({30.0, 31.0, 32.0});
  original.setKappa(0.41f);
  original.setCutoff(10.0f);
  original.setCtonnb(8.0f);
  original.setCtofnb(9.0f);
  original.setFFTGrid(30, 32, 34);
  original.setPmeSplineOrder(5);
  original.setPeriodicBoundaryCondition(PBC::P21);
  original.setVdwType(VDW_DBEXP);

  ForceManager copy(original);

  CHECK(copy.getPsf() != original.getPsf());
  CHECK(copy.getPrm() != original.getPrm());
  CHECK(copy.getPsf()->getNumAtoms() == original.getPsf()->getNumAtoms());
  CHECK(copy.hasCharmmContext() == false);
  CHECK(copy.isInitialized() == false);

  apo_test::CheckVectorsClose<double>("copied box dimensions",
                                      copy.getBoxDimensions(),
                                      original.getBoxDimensions(), TOLERANCE);
  CHECK(copy.getKappa() == Approx(original.getKappa()).margin(TOLERANCE));
  CHECK(copy.getCutoff() == Approx(original.getCutoff()).margin(TOLERANCE));
  CHECK(copy.getCtonnb() == Approx(original.getCtonnb()).margin(TOLERANCE));
  CHECK(copy.getCtofnb() == Approx(original.getCtofnb()).margin(TOLERANCE));
  apo_test::CheckVectorsEqual<int>("copied FFT grid", copy.getFFTGrid(),
                                   original.getFFTGrid());
  CHECK(copy.getPmeSplineOrder() == original.getPmeSplineOrder());
  CHECK(copy.getPeriodicBoundaryCondition() ==
        original.getPeriodicBoundaryCondition());
  CHECK(copy.getVdwType() == original.getVdwType());

  original.setBoxDimensions({40.0, 41.0, 42.0});
  original.setKappa(0.11f);
  original.setVdwType(VDW_CUT);
  original.setPmeSplineOrder(7);

  apo_test::CheckVectorsClose<double>(
      "copied box dimensions remain independet", copy.getBoxDimensions(),
      std::vector<double>{30.0, 31.0, 32.0}, TOLERANCE);
  CHECK(copy.getKappa() == Approx(0.41f).margin(TOLERANCE));
  CHECK(copy.getVdwType() == VDW_DBEXP);
  CHECK(copy.getPmeSplineOrder() == 5);
}

TEST_CASE("ForceManagerRejectsOperationsBeforeInitialization") {
  auto fm = CreateForceManager();
  const std::string message =
      "ForceManager must be initialized before this operation";

  apo_test::CheckApoCharmmError([&fm](void) { (void)fm.getEnergyComponents(); },
                                ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError(
      [&fm](void) { (void)fm.getBondedForcevalues(); },
      ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError(
      [&fm](void) { (void)fm.getReciprocalForcevalues(); },
      ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError(
      [&fm](void) { (void)fm.getDirectForcevalues(); },
      ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError([&fm](void) { (void)fm.getTotalForcevalues(); },
                                ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError([&fm](void) { (void)fm.getForces(); },
                                ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError([&fm](void) { (void)fm.getForceStride(); },
                                ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError([&fm](void) { (void)fm.getPotentialEnergy(); },
                                ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError(
      [&fm](void) { (void)fm.getPotentialEnergies(); },
      ApoCharmmErrorCode::NotInitialized, message);
  apo_test::CheckApoCharmmError([&fm](void) { (void)fm.getVirial(); },
                                ApoCharmmErrorCode::NotInitialized, message);
}

TEST_CASE("ForceManagerRejectsChildPotentialEnergyEvaluation") {
  ForceManager fm;
  float4 xyzq{};

  apo_test::CheckApoCharmmError(
      [&fm](void) { (void)fm.computeAllChildrenPotentialEnergy(nullptr); },
      ApoCharmmErrorCode::InvalidArgument,
      "Coordinate-charge array must not be null");
  apo_test::CheckApoCharmmError(
      [&fm, &xyzq](void) { (void)fm.computeAllChildrenPotentialEnergy(&xyzq); },
      ApoCharmmErrorCode::Runtime,
      "ForceManager does not support child potential-energy evaluation");
}

TEST_CASE("ForceManagerAddForceManagerRejectsChildren") {
  ForceManager fm;

  apo_test::CheckApoCharmmError([&fm](void) { fm.addForceManager(nullptr); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Child ForceManager must not be null");

  auto child = std::make_shared<ForceManager>();
  apo_test::CheckApoCharmmError(
      [&fm, &child](void) { fm.addForceManager(child); },
      ApoCharmmErrorCode::Runtime,
      "ForceManager does not support child ForceManagers");
}

TEST_CASE("ForceManagerDefaultPBCCompatibilityIsUnrestricted") {
  const std::vector<std::size_t> emptySubscriptionSizes(6, 0);
  const std::vector<std::size_t> fullSubscriptionSizes(6, 1);

  auto fm = CreatePBCCompatibilityForceManager();
  auto force = std::make_shared<TestForce>();
  auto stream = std::make_shared<cudaStream_t>();

  SECTION("PBC::P1") {
    fm.subscribe(force, "default_tag", stream, force->getForce(),
                 force->getEnergyVirial());

    CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);
    CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  }

  SECTION("PBC::P21") {
    fm.setPeriodicBoundaryCondition(PBC::P21);
    fm.subscribe(force, "default_p21", stream, force->getForce(),
                 force->getEnergyVirial());

    CHECK(fm.getPeriodicBoundaryCondition() == PBC::P21);
    CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  }

  fm.unsubscribe(force);
  CHECK(fm.getSubscriptionVectorSizes() == emptySubscriptionSizes);
}

TEST_CASE("ForceManagerP1OnlySubscriptionValidationIsTransactional") {
  const std::vector<std::size_t> emptySubscriptionSizes(6, 0);
  const std::vector<std::size_t> fullSubscriptionSizes(6, 1);

  auto fm = CreatePBCCompatibilityForceManager();
  auto force = std::make_shared<P1OnlyTestForce>();
  auto stream = std::make_shared<cudaStream_t>();
  auto forceValues = force->getForce();
  auto energyVirial = force->getEnergyVirial();

  SECTION("PBC::P1") {
    fm.subscribe(force, "p1_only", stream, forceValues, energyVirial);

    CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);
    CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
    CHECK(force->initializeCalls == 0);

    fm.unsubscribe(force);

    CHECK(fm.getSubscriptionVectorSizes() == emptySubscriptionSizes);
  }

  SECTION("PBC::P21") {
    auto existingForce = std::make_shared<TestForce>();
    auto existingStream = std::make_shared<cudaStream_t>();

    fm.setPeriodicBoundaryCondition(PBC::P21);
    fm.subscribe(existingForce, "existing_default", existingStream,
                 existingForce->getForce(), existingForce->getEnergyVirial());

    REQUIRE(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);

    apo_test::CheckApoCharmmError(
        [&fm, &force, &stream, &forceValues, &energyVirial](void) {
          fm.subscribe(force, "p1_only", stream, forceValues, energyVirial);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Subscribed force does not support periodic boundary condition value "
        "2");

    CHECK(fm.getPeriodicBoundaryCondition() == PBC::P21);
    CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
    CHECK(force->initializeCalls == 0);

    fm.unsubscribe(existingForce);

    CHECK(fm.getSubscriptionVectorSizes() == emptySubscriptionSizes);
  }

  SECTION("PBC::UNSET") {
    auto existingForce = std::make_shared<TestForce>();
    auto existingStream = std::make_shared<cudaStream_t>();

    fm.subscribe(existingForce, "existing_default", existingStream,
                 existingForce->getForce(), existingForce->getEnergyVirial());

    REQUIRE(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);

    fm.setPeriodicBoundaryConditionUnchecked(PBC::UNSET);

    apo_test::CheckApoCharmmError(
        [&fm, &force, &stream, &forceValues, &energyVirial](void) {
          fm.subscribe(force, "p1_only", stream, forceValues, energyVirial);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 0");

    CHECK(fm.getPeriodicBoundaryCondition() == PBC::UNSET);
    CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
    CHECK(force->initializeCalls == 0);

    fm.unsubscribe(existingForce);

    CHECK(fm.getSubscriptionVectorSizes() == emptySubscriptionSizes);
  }

  SECTION("invalid cast-created PBC") {
    auto existingForce = std::make_shared<TestForce>();
    auto existingStream = std::make_shared<cudaStream_t>();
    const PBC invalidPBC = static_cast<PBC>(99);

    fm.subscribe(existingForce, "existing_default", existingStream,
                 existingForce->getForce(), existingForce->getEnergyVirial());

    REQUIRE(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);

    fm.setPeriodicBoundaryConditionUnchecked(invalidPBC);

    apo_test::CheckApoCharmmError(
        [&fm, &force, &stream, &forceValues, &energyVirial](void) {
          fm.subscribe(force, "p1_only", stream, forceValues, energyVirial);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 99");

    CHECK(fm.getPeriodicBoundaryCondition() == invalidPBC);
    CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
    CHECK(force->initializeCalls == 0);

    fm.unsubscribe(existingForce);

    CHECK(fm.getSubscriptionVectorSizes() == emptySubscriptionSizes);
  }
}

TEST_CASE("ForceManagerP1OnlyPBCMutationsIsTransactional") {
  const std::vector<std::size_t> emptySubscriptionSizes(6, 0);
  const std::vector<std::size_t> fullSubscriptionSizes(6, 1);

  auto fm = CreatePBCCompatibilityForceManager();
  fm.setBoxDimensions({50.0, 50.0, 50.0});

  auto force = std::make_shared<P1OnlyTestForce>();
  auto stream = std::make_shared<cudaStream_t>();

  fm.subscribe(force, "p1_only", stream, force->getForce(),
               force->getEnergyVirial());
  fm.initialize();

  REQUIRE(fm.isInitialized() == true);
  REQUIRE(fm.getPeriodicBoundaryCondition() == PBC::P1);
  REQUIRE(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  REQUIRE(fm.isReciprocalForceDirty() == false);
  REQUIRE(fm.isDirectForceDirty() == false);
  REQUIRE(force->initializeCalls == 1);

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setPeriodicBoundaryCondition(PBC::P21); },
      ApoCharmmErrorCode::InvalidArgument,
      "Subscribed force does not support periodic boundary condition value 2");

  CHECK(fm.isInitialized() == true);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);
  CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  CHECK(fm.isReciprocalForceDirty() == false);
  CHECK(fm.isDirectForceDirty() == false);
  CHECK(force->initializeCalls == 1);

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setPeriodicBoundaryCondition(PBC::UNSET); },
      ApoCharmmErrorCode::InvalidArgument,
      "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 0");

  CHECK(fm.isInitialized() == true);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);
  CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  CHECK(fm.isReciprocalForceDirty() == false);
  CHECK(fm.isDirectForceDirty() == false);
  CHECK(force->initializeCalls == 1);

  const PBC invalidPBC = static_cast<PBC>(99);

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.setPeriodicBoundaryCondition(invalidPBC); },
      ApoCharmmErrorCode::InvalidArgument,
      "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 99");

  CHECK(fm.isInitialized() == true);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);
  CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  CHECK(fm.isReciprocalForceDirty() == false);
  CHECK(fm.isDirectForceDirty() == false);
  CHECK(force->initializeCalls == 1);

  fm.unsubscribe(force);

  CHECK(fm.getSubscriptionVectorSizes() == emptySubscriptionSizes);

  CHECK_NOTHROW(fm.setPeriodicBoundaryCondition(PBC::P21));

  CHECK(fm.isInitialized() == true);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P21);
  CHECK(fm.getSubscriptionVectorSizes() == emptySubscriptionSizes);
  CHECK(fm.isReciprocalForceDirty() == true);
  CHECK(fm.isDirectForceDirty() == true);
}

TEST_CASE("ForceManagerInitializationRevalidatesPBCCompatibility") {
  const std::vector<std::size_t> fullSubscriptionSizes(6, 1);

  auto fm = CreatePBCCompatibilityForceManager();
  fm.setBoxDimensions({50.0, 50.0, 50.0});

  auto force = std::make_shared<P1OnlyTestForce>();
  auto stream = std::make_shared<cudaStream_t>();

  fm.subscribe(force, "p1_only", stream, force->getForce(),
               force->getEnergyVirial());

  REQUIRE(fm.isInitialized() == false);
  REQUIRE(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  REQUIRE(force->initializeCalls == 0);

  fm.setPeriodicBoundaryConditionUnchecked(PBC::P21);

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.initialize(); }, ApoCharmmErrorCode::InvalidArgument,
      "Subscribed force does not support periodic boundary condition value 2");

  CHECK(fm.isInitialized() == false);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P21);
  CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  CHECK(fm.isReciprocalForceDirty() == false);
  CHECK(fm.isDirectForceDirty() == false);
  CHECK(force->initializeCalls == 0);

  fm.setPeriodicBoundaryConditionUnchecked(PBC::P1);

  CHECK_NOTHROW(fm.initialize());

  CHECK(fm.isInitialized() == true);
  CHECK(fm.getPeriodicBoundaryCondition() == PBC::P1);
  CHECK(fm.getSubscriptionVectorSizes() == fullSubscriptionSizes);
  CHECK(fm.isReciprocalForceDirty() == false);
  CHECK(fm.isDirectForceDirty() == false);
  CHECK(force->initializeCalls == 1);
}

TEST_CASE("ForceManagerSubscriptionValidation") {
  auto fm = CreateForceManager();
  auto force = std::make_shared<TestForce>();
  auto stream = std::make_shared<cudaStream_t>();
  auto forceValues = force->getForce();
  auto energyVirial = force->getEnergyVirial();
  std::shared_ptr<TestForce> nullForce;
  std::shared_ptr<cudaStream_t> nullStream;
  std::shared_ptr<Force<long long int>> nullForceValues;
  std::shared_ptr<CudaEnergyVirial> nullEnergyVirial;

  apo_test::CheckApoCharmmError(
      [&fm, &nullForce, &stream, &forceValues, &energyVirial](void) {
        fm.subscribe(nullForce, "test_force", stream, forceValues,
                     energyVirial);
      },
      ApoCharmmErrorCode::InvalidArgument, "Subscribed force must not be null");
  apo_test::CheckApoCharmmError(
      [&fm, &force, &stream, &forceValues, &energyVirial](void) {
        fm.subscribe(force, "", stream, forceValues, energyVirial);
      },
      ApoCharmmErrorCode::InvalidArgument, "Force tag must not be empty");
  apo_test::CheckApoCharmmError(
      [&fm, &force, &nullStream, &forceValues, &energyVirial](void) {
        fm.subscribe(force, "test_force", nullStream, forceValues,
                     energyVirial);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Subscribed force stream must not be null");
  apo_test::CheckApoCharmmError(
      [&fm, &force, &stream, &nullForceValues, &energyVirial](void) {
        fm.subscribe(force, "test_force", stream, nullForceValues,
                     energyVirial);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Subscribed force storage must not be null");
  apo_test::CheckApoCharmmError(
      [&fm, &force, &stream, &forceValues, &nullEnergyVirial](void) {
        fm.subscribe(force, "test_force", stream, forceValues,
                     nullEnergyVirial);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Subscribed energy-virial storage must not be null");

  fm.subscribe(force, "test_force", stream, forceValues, energyVirial);

  apo_test::CheckApoCharmmError(
      [&fm, &force, &stream, &forceValues, &energyVirial](void) {
        fm.subscribe(force, "test_force", stream, forceValues, energyVirial);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force is already subscribed to this ForceManager");
  apo_test::CheckApoCharmmError(
      [&fm, &nullForce](void) { fm.unsubscribe(nullForce); },
      ApoCharmmErrorCode::InvalidArgument, "Subscribed force must not be null");

  auto otherForce = std::make_shared<TestForce>();
  apo_test::CheckApoCharmmError(
      [&fm, &otherForce](void) { fm.unsubscribe(otherForce); },
      ApoCharmmErrorCode::InvalidArgument,
      "Force is not subscribed to this ForceManager");

  fm.unsubscribe(force);
}

TEST_CASE("ForceManagerSubscribeAndUnsubscribePropagateBoxDimensions") {
  auto fm = CreateForceManager();

  auto force = std::make_shared<TestForce>();
  auto stream = std::make_shared<cudaStream_t>();

  fm.subscribe(force, "test_force", stream, force->getForce(),
               force->getEnergyVirial());

  fm.setBoxDimensions({20.0, 21.0, 22.0});

  CHECK(force->initializeCalls == 0);
  CHECK(force->setBoxDimensionsCalls == 1);
  apo_test::CheckVectorsClose<double>(
      "subscribed force box dimensions", force->lastBoxDimensions,
      std::vector<double>{20.0, 21.0, 22.0}, TOLERANCE);

  fm.unsubscribe(force);
  fm.setBoxDimensions({30.0, 31.0, 32.0});

  CHECK(force->setBoxDimensionsCalls == 1);
  apo_test::CheckVectorsClose<double>(
      "unsubscribed force box dimensions unchanged", force->lastBoxDimensions,
      std::vector<double>{20.0, 21.0, 22.0}, TOLERANCE);
}

TEST_CASE("ForceManagerUnsubscribeByTag") {
  auto fm = CreateForceManager();

  apo_test::CheckApoCharmmError([&fm](void) { fm.unsubscribe(""); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Force tag must not be empty");
  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.unsubscribe("missing_force"); },
      ApoCharmmErrorCode::InvalidArgument,
      "Force tag is not subscribed to this ForceManager");

  auto force = std::make_shared<TestForce>();
  auto stream = std::make_shared<cudaStream_t>();

  fm.subscribe(force, "test_force", stream, force->getForce(),
               force->getEnergyVirial());

  fm.setBoxDimensions({20.0, 21.0, 22.0});
  CHECK(force->setBoxDimensionsCalls == 1);

  fm.unsubscribe("test_force");
  fm.setBoxDimensions({30.0, 31.0, 32.0});

  CHECK(force->setBoxDimensionsCalls == 1);

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm.unsubscribe("test_force"); },
      ApoCharmmErrorCode::InvalidArgument,
      "Force tag is not subscribed to this ForceManager");
}

TEST_CASE("ForceManagerCheckedCudaLaunchesAndGraphCleanup") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions({50.0, 50.0, 50.0});
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);

  auto fm = ctx->getForceManager();
  REQUIRE(fm != nullptr);
  REQUIRE(fm->isInitialized() == true);

  auto force = std::make_shared<TestForce>();
  auto stream = std::make_shared<cudaStream_t>();

  fm->subscribe(force, "default_compatibility_force", stream, force->getForce(),
                force->getEnergyVirial());

  CHECK(force->initializeCalls == 1);
  CHECK(force->initializeNumAtoms == psf->getNumAtoms());

  CHECK_NOTHROW(ctx->calculateForces(false, false, false));

  CHECK(force->clearCalls == 1);
  CHECK(force->calcForceCalls == 1);
  CHECK(force->lastXYZQ != nullptr);
  CHECK(force->lastCalcEnergy == false);
  CHECK(force->lastCalcVirial == false);

  fm->unsubscribe(force);

  CHECK_NOTHROW(ctx->calculateForces(false, true, false));

  CHECK(force->clearCalls == 1);
  CHECK(force->calcForceCalls == 1);

  CudaContainer<double> &potentialEnergy = fm->getPotentialEnergy();
  potentialEnergy.transferToHost();
  REQUIRE(potentialEnergy.size() == 1);
  CHECK(std::isfinite(potentialEnergy[0]));

  apo_test::CheckApoCharmmError(
      [&fm](void) { fm->calcForcePart1(true, false, false); },
      ApoCharmmErrorCode::NotImplemented,
      "Force \"reset\" is not implemented (JEG260807: deprecate in future)");
}
