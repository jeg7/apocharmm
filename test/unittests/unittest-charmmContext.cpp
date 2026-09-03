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
#include "ForceManager.h"
#include "PBC.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace {

constexpr double TOLERANCE = 0.0;

const std::vector<double> BOX_DIMENSIONS = {40.0, 41.0, 42.0};

class CoutCapture {
public:
  CoutCapture(void) : m_PreviousBuffer(std::cout.rdbuf(m_Output.rdbuf())) {}

  ~CoutCapture(void) noexcept { std::cout.rdbuf(m_PreviousBuffer); }

  CoutCapture(const CoutCapture &) = delete;
  CoutCapture &operator=(const CoutCapture &) = delete;

  std::string str(void) const { return m_Output.str(); }

private:
  std::ostringstream m_Output;
  std::streambuf *m_PreviousBuffer;
};

} // namespace

TEST_CASE("CharmmContextForceManagerConstructorMirrorsBackendState") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions(BOX_DIMENSIONS);

  auto ctx = std::make_shared<CharmmContext>(fm);

  CHECK(ctx->getForceManager() == fm);
  CHECK(ctx->getPsf() == psf);
  CHECK(ctx->getPrm() == prm);

  apo_test::CheckVectorsClose<double>("CharmmContext box dimensions",
                                      ctx->getBoxDimensions(), BOX_DIMENSIONS,
                                      TOLERANCE);

  apo_test::CheckVectorsClose<double>("ForceManager box dimensions",
                                      fm->getBoxDimensions(), BOX_DIMENSIONS,
                                      TOLERANCE);

  CHECK(ctx->getPeriodicBoundaryCondition() ==
        fm->getPeriodicBoundaryCondition());

  CHECK(fm->isInitialized() == true);
}

TEST_CASE("CharmmContextStagedStateConfiguresForceManagerOnAttach") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  auto ctx = std::make_shared<CharmmContext>();
  ctx->setPsf(psf);
  ctx->setPrm(prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setPeriodicBoundaryCondition(PBC::P21);

  auto fm = std::make_shared<ForceManager>();
  ctx->setForceManager(fm);

  CHECK(ctx->getForceManager() == fm);

  CHECK(ctx->getPsf() == psf);
  CHECK(ctx->getPrm() == prm);
  CHECK(fm->getPsf() == psf);
  CHECK(fm->getPrm() == prm);

  apo_test::CheckVectorsClose<double>("CharmmContext staged box dimensions",
                                      ctx->getBoxDimensions(), BOX_DIMENSIONS,
                                      TOLERANCE);

  apo_test::CheckVectorsClose<double>("ForceManager staged box dimensions",
                                      fm->getBoxDimensions(), BOX_DIMENSIONS,
                                      TOLERANCE);

  CHECK(ctx->getPeriodicBoundaryCondition() == PBC::P21);
  CHECK(fm->getPeriodicBoundaryCondition() == PBC::P21);
}

TEST_CASE("CharmmContextStagedStateCanLoadCoordinatesAfterBackendInitialize") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>();
  ctx->setPsf(psf);
  ctx->setPrm(prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setPeriodicBoundaryCondition(PBC::P1);

  auto fm = std::make_shared<ForceManager>();
  ctx->setForceManager(fm);

  CHECK(fm->isInitialized() == true);

  CHECK_NOTHROW(ctx->setCoordinates(crd));

  CHECK(ctx->getNumAtoms() == psf->getNumAtoms());

  auto &coordinatesCharges = ctx->getCoordinatesChargesDP();
  auto &velocitiesInverseMasses = ctx->getVelocitiesInverseMasses();

  CHECK(coordinatesCharges.size() ==
        static_cast<std::size_t>(psf->getNumAtoms()));
  CHECK(velocitiesInverseMasses.size() ==
        static_cast<std::size_t>(psf->getNumAtoms()));

  coordinatesCharges.transferFromDevice();
  velocitiesInverseMasses.transferFromDevice();

  const std::vector<double> charges = psf->getCharges();
  const std::vector<double> masses = psf->getMasses();

  REQUIRE(charges.empty() == false);
  REQUIRE(masses.empty() == false);

  CHECK(coordinatesCharges[0].w == Approx(charges[0]).margin(TOLERANCE));
  CHECK(velocitiesInverseMasses[0].w ==
        Approx(1.0 / masses[0]).margin(TOLERANCE));
}

TEST_CASE("CharmmContextRejectsInvalidStagedBoxDimensions") {
  CharmmContext ctx;

  apo_test::CheckApoCharmmError(
      [&ctx](void) { ctx.setBoxDimensions({40.0, 40.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimensions must contain exactly 3 positive values");
  apo_test::CheckApoCharmmError(
      [&ctx](void) { ctx.setBoxDimensions({40.0, 40.0, 40.0, 40.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimensions must contain exactly 3 positive values");
  apo_test::CheckApoCharmmError(
      [&ctx](void) { ctx.setBoxDimensions({40.0, 0.0, 40.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimensions must contain exactly 3 positive values");
  apo_test::CheckApoCharmmError(
      [&ctx](void) { ctx.setBoxDimensions({40.0, -1.0, 40.0}); },
      ApoCharmmErrorCode::InvalidArgument,
      "Box dimensions must contain exactly 3 positive values");
}

TEST_CASE("ForceManagerContextBackPointerDoesNotOwnCharmmContext") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions(BOX_DIMENSIONS);

  {
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setCoordinates(crd);

    CHECK(fm->hasCharmmContext() == true);
    CHECK(fm->getContext() == ctx);
  }

  CHECK(fm->hasCharmmContext() == false);
  CHECK(fm->getContext() == nullptr);
}

TEST_CASE("CharmmContextCanAttachForceManagerBeforeStagingState") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  auto ctx = std::make_shared<CharmmContext>();
  auto fm = std::make_shared<ForceManager>();

  ctx->setForceManager(fm);

  CHECK(fm->isInitialized() == false);

  ctx->setPsf(psf);
  CHECK(fm->isInitialized() == false);

  ctx->setPrm(prm);
  CHECK(fm->isInitialized() == false);

  ctx->setBoxDimensions(BOX_DIMENSIONS);

  CHECK(fm->isInitialized() == true);
  CHECK(ctx->getPsf() == psf);
  CHECK(ctx->getPrm() == prm);
  CHECK(fm->getPsf() == psf);
  CHECK(fm->getPrm() == prm);

  apo_test::CheckVectorsClose<double>("CharmmContext box dimensions",
                                      ctx->getBoxDimensions(), BOX_DIMENSIONS,
                                      TOLERANCE);
  apo_test::CheckVectorsClose<double>("ForceManager box dimensions",
                                      fm->getBoxDimensions(), BOX_DIMENSIONS,
                                      TOLERANCE);
}

TEST_CASE("CharmmContextConstructsFromPsfAndParameters") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);

  REQUIRE(ctx->getForceManager() != nullptr);
  CHECK(ctx->getPsf() == psf);
  CHECK(ctx->getPrm() == prm);
  CHECK(ctx->getForceManager()->getPsf() == psf);
  CHECK(ctx->getForceManager()->getPrm() == prm);

  CHECK(ctx->getForceManager()->isInitialized() == false);

  ctx->setBoxDimensions(BOX_DIMENSIONS);

  CHECK(ctx->getForceManager()->isInitialized() == true);

  apo_test::CheckVectorsClose<double>("CharmmContext box dimensions",
                                      ctx->getBoxDimensions(), BOX_DIMENSIONS,
                                      TOLERANCE);

  apo_test::CheckVectorsClose<double>(
      "ForceManager box dimensions", ctx->getForceManager()->getBoxDimensions(),
      BOX_DIMENSIONS, TOLERANCE);

  ctx->setCoordinates(crd);

  CHECK(ctx->getNumAtoms() == psf->getNumAtoms());
  CHECK(ctx->getCoordinatesChargesDP().size() ==
        static_cast<std::size_t>(psf->getNumAtoms()));
  CHECK(ctx->getVelocitiesInverseMasses().size() ==
        static_cast<std::size_t>(psf->getNumAtoms()));
}

TEST_CASE("CharmmContextForwardsForceManagerConfiguration") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  auto fm = ctx->getForceManager();

  ctx->setKappa(0.45f);
  ctx->setCutoff(9.0f);
  ctx->setCtonnb(7.5f);
  ctx->setCtofnb(8.5f);
  ctx->setFFTGrid(32, 34, 36);
  ctx->setPmeSplineOrder(6);
  ctx->setVdwType(VDW_DBEXP);

  CHECK(ctx->getKappa() == Approx(0.45f));
  CHECK(ctx->getCutoff() == Approx(9.0f));
  CHECK(ctx->getCtonnb() == Approx(7.5f));
  CHECK(ctx->getCtofnb() == Approx(8.5f));
  apo_test::CheckVectorsEqual<int>("context FFT grid", ctx->getFFTGrid(),
                                   std::vector<int>{32, 34, 36});
  CHECK(ctx->getPmeSplineOrder() == 6);
  CHECK(ctx->getVdwType() == VDW_DBEXP);

  CHECK(fm->getKappa() == Approx(0.45f));
  CHECK(fm->getCutoff() == Approx(9.0f));
  CHECK(fm->getCtonnb() == Approx(7.5f));
  CHECK(fm->getCtofnb() == Approx(8.5f));
  apo_test::CheckVectorsEqual<int>("context FFT grid", fm->getFFTGrid(),
                                   std::vector<int>{32, 34, 36});
  CHECK(fm->getPmeSplineOrder() == 6);
  CHECK(fm->getVdwType() == VDW_DBEXP);
}

TEST_CASE("CharmmContextPrintsCharmmStyleEnergyTable") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCoordinates(crd);

  std::string observedOutput;
  {
    CoutCapture capture;
    ctx->calculatePotentialEnergy(false, true);
    observedOutput = capture.str();
  }

  const std::shared_ptr<ForceManager> fm = ctx->getForceManager();
  REQUIRE(fm != nullptr);

  const std::map<std::string, double> energyComponents =
      fm->getEnergyComponents();

  CHECK(energyComponents.size() == 12);

  CudaContainer<double> &potentialEnergy = fm->getPotentialEnergy();
  potentialEnergy.transferFromDevice();

  REQUIRE(potentialEnergy.size() == 1);
  const double totalEnergy = potentialEnergy[0];

  const double componentTotal =
      energyComponents.at("bond") + energyComponents.at("angle") +
      energyComponents.at("ureyb") + energyComponents.at("dihe") +
      energyComponents.at("imdihe") + energyComponents.at("cmap") +
      energyComponents.at("ewks") + energyComponents.at("ewse") +
      energyComponents.at("ewex") + energyComponents.at("elec") +
      energyComponents.at("vdw") + energyComponents.at("user");

  CHECK(totalEnergy == Approx(componentTotal).margin(1.0e-8));

  const std::shared_ptr<Force<double>> forces = fm->getForces();
  REQUIRE(forces != nullptr);

  const std::size_t numAtoms = static_cast<std::size_t>(ctx->getNumAtoms());

  std::vector<double> forceX(numAtoms);
  std::vector<double> forceY(numAtoms);
  std::vector<double> forceZ(numAtoms);

  forces->getXYZ(forceX.data(), forceY.data(), forceZ.data());

  double squaredGradientNorm = 0.0;
  for (std::size_t i = 0; i < numAtoms; i++) {
    squaredGradientNorm +=
        forceX[i] * forceX[i] + forceY[i] * forceY[i] + forceZ[i] * forceZ[i];
  }

  const double gradientRms =
      std::sqrt(squaredGradientNorm / static_cast<double>(numAtoms));

  const auto component =
      [&energyComponents](const std::string &name) -> double {
    return energyComponents.at(name);
  };

  std::ostringstream expectedOutput;

  expectedOutput << "ENER ENR:  Eval#     ENERgy      Delta-E         GRMS\n"
                 << "ENER INTERN:          BONDs       ANGLes       UREY-b"
                    "    DIHEdrals    IMPRopers\n"
                 << "ENER CROSS:           CMAPs        PMF1D        PMF2D"
                    "        PRIMO\n"
                 << "ENER EXTERN:        VDWaals         ELEC       HBONds"
                    "          ASP         USER\n"
                 << "ENER EWALD:          EWKSum       EWSElf       EWEXcl"
                    "       EWQCor       EWUTil\n"
                 << " ----------       ---------    ---------    ---------"
                    "    ---------    ---------\n";

  expectedOutput << std::fixed << std::setprecision(5) << std::right;

  expectedOutput << "ENER>" << std::setw(9) << 0 << std::setw(13) << totalEnergy
                 << std::setw(13) << 0.0 << std::setw(13) << gradientRms
                 << '\n';

  expectedOutput << std::left << std::setw(14) << "ENER INTERN>" << std::right
                 << std::setw(13) << component("bond") << std::setw(13)
                 << component("angle") << std::setw(13) << component("ureyb")
                 << std::setw(13) << component("dihe") << std::setw(13)
                 << component("imdihe") << '\n';

  expectedOutput << std::left << std::setw(14) << "ENER CROSS>" << std::right
                 << std::setw(13) << component("cmap") << std::setw(13) << 0.0
                 << std::setw(13) << 0.0 << std::setw(13) << 0.0 << '\n';

  expectedOutput << std::left << std::setw(14) << "ENER EXTERN>" << std::right
                 << std::setw(13) << component("vdw") << std::setw(13)
                 << component("elec") << std::setw(13) << 0.0 << std::setw(13)
                 << 0.0 << std::setw(13) << component("user") << '\n';

  expectedOutput << std::left << std::setw(14) << "ENER EWALD>" << std::right
                 << std::setw(13) << component("ewks") << std::setw(13)
                 << component("ewse") << std::setw(13) << component("ewex")
                 << std::setw(13) << 0.0 << std::setw(13) << 0.0 << '\n';

  expectedOutput << " ----------       ---------    ---------    ---------"
                    "    ---------    ---------\n";

  CHECK(observedOutput == expectedOutput.str());
}

TEST_CASE("CharmmContextEnergyTableTracksPrintedEvaluations") {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCoordinates(crd);

  std::string silentOutput;
  {
    CoutCapture capture;
    ctx->calculatePotentialEnergy(false, false);
    silentOutput = capture.str();
  }

  CHECK(silentOutput.empty());

  std::string firstOutput;
  {
    CoutCapture capture;
    ctx->calculatePotentialEnergy(false, true);
    firstOutput = capture.str();
  }

  CudaContainer<double> &potentialEnergy =
      ctx->getForceManager()->getPotentialEnergy();
  potentialEnergy.transferFromDevice();

  REQUIRE(potentialEnergy.size() == 1);
  const double firstEnergy = potentialEnergy[0];

  std::vector<double3> coordinates = crd->getCoordinatesDP();
  REQUIRE(coordinates.empty() == false);

  coordinates[0].x += 0.25;
  ctx->setCoordinates(coordinates);

  std::string secondOutput;
  {
    CoutCapture capture;
    ctx->calculatePotentialEnergy(false, true);
    secondOutput = capture.str();
  }

  potentialEnergy.transferFromDevice();
  const double secondEnergy = potentialEnergy[0];
  const double expectedDeltaEnergy = secondEnergy - firstEnergy;

  CHECK(std::abs(expectedDeltaEnergy) > 1.0e-6);

  std::istringstream firstTable(firstOutput);
  std::string firstRecordLine;

  for (int i = 0; i < 6; i++)
    REQUIRE(static_cast<bool>(std::getline(firstTable, firstRecordLine)));

  REQUIRE(static_cast<bool>(std::getline(firstTable, firstRecordLine)));

  std::string firstTag;
  std::uint64_t firstEvaluation = 0;
  double printedFirstEnergy = 0.0;
  double firstDeltaEnergy = 0.0;
  double firstGradientRms = 0.0;

  std::istringstream firstRecord(firstRecordLine);
  REQUIRE(static_cast<bool>(firstRecord >> firstTag >> firstEvaluation >>
                            printedFirstEnergy >> firstDeltaEnergy >>
                            firstGradientRms));

  CHECK(firstTag == "ENER>");
  CHECK(firstEvaluation == 0);
  CHECK(printedFirstEnergy == Approx(firstEnergy).margin(1.0e-5));
  CHECK(firstDeltaEnergy == Approx(0.0).margin(1.0e-5));
  CHECK(std::isfinite(firstGradientRms));
  CHECK(firstGradientRms >= 0.0);

  std::istringstream secondTable(secondOutput);
  std::string secondRecordLine;

  for (int i = 0; i < 6; i++)
    REQUIRE(static_cast<bool>(std::getline(secondTable, secondRecordLine)));

  REQUIRE(static_cast<bool>(std::getline(secondTable, secondRecordLine)));

  std::string secondTag;
  std::uint64_t secondEvaluation = 0;
  double printedSecondEnergy = 0.0;
  double secondDeltaEnergy = 0.0;
  double secondGradientRms = 0.0;

  std::istringstream secondRecord(secondRecordLine);
  REQUIRE(static_cast<bool>(secondRecord >> secondTag >> secondEvaluation >>
                            printedSecondEnergy >> secondDeltaEnergy >>
                            secondGradientRms));

  CHECK(secondTag == "ENER>");
  CHECK(secondEvaluation == 1);
  CHECK(printedSecondEnergy == Approx(secondEnergy).margin(1.0e-5));
  CHECK(secondDeltaEnergy == Approx(expectedDeltaEnergy).margin(1.0e-5));
  CHECK(std::isfinite(secondGradientRms));
  CHECK(secondGradientRms >= 0.0);
}

TEST_CASE("CharmmContextKineticEnergyReductionIsDeterministic") {
  const auto checkKineticEnergy = [](const int numAtoms) {
    CharmmContext context;
    context.setNumAtoms(numAtoms);

    std::vector<double4> velocitiesInverseMasses(
        static_cast<std::size_t>(numAtoms));

    for (double4 &velocityInverseMass : velocitiesInverseMasses) {
      velocityInverseMass.x = 1.0;
      velocityInverseMass.y = 2.0;
      velocityInverseMass.z = 3.0;
      velocityInverseMass.w = 0.5;
    }

    context.setVelocitiesInverseMasses(velocitiesInverseMasses);

    const double expectedKineticEnergy = 14.0 * static_cast<double>(numAtoms);
    const double firstKineticEnergy = context.getKineticEnergy();

    REQUIRE(firstKineticEnergy == expectedKineticEnergy);

    for (int repetition = 0; repetition < 8; repetition++)
      CHECK(context.getKineticEnergy() == firstKineticEnergy);
  };

  SECTION("SingleFirstPassBlock") { checkKineticEnergy(17); }

  SECTION("MultipleFinalPassIterations") {
    // With 256 threads and two atoms per thread, this produces 257
    // first-pass block sums. The final pass must therefore load the
    // partial-sum element at index 256.
    checkKineticEnergy(131073);
  }
}

TEST_CASE("CharmmContextParameterReplacementRebuildsAffectedForceBackends") {
  auto originalParameters = std::make_shared<CharmmParameters>(
      apo_test::GetTopparDir() / "toppar_water_ions.str");
  auto replacementParameters =
      std::make_shared<CharmmParameters>(*originalParameters);

  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto forceManager = std::make_shared<ForceManager>(psf, originalParameters);
  forceManager->setBoxDimensions(BOX_DIMENSIONS);

  auto context = std::make_shared<CharmmContext>(forceManager);
  context->setCoordinates(crd);

  struct ForceEvaluation {
    std::vector<double> forces;
    std::map<std::string, double> energyComponents;
  };

  const auto captureForceEvaluation =
      [](const std::shared_ptr<CharmmContext> &testContext) -> ForceEvaluation {
    testContext->calculatePotentialEnergy(false, false);

    const std::shared_ptr<Force<double>> force = testContext->getForces();
    REQUIRE(force != nullptr);

    const std::shared_ptr<ForceManager> testForceManager =
        testContext->getForceManager();
    REQUIRE(testForceManager != nullptr);

    const std::size_t numAtoms =
        static_cast<std::size_t>(testContext->getNumAtoms());

    ForceEvaluation evaluation;
    evaluation.forces.resize(3 * numAtoms);

    force->getXYZ(evaluation.forces.data(), evaluation.forces.data() + numAtoms,
                  evaluation.forces.data() + 2 * numAtoms);

    evaluation.energyComponents = testForceManager->getEnergyComponents();

    return evaluation;
  };

  const ForceEvaluation expected = captureForceEvaluation(context);

  const std::shared_ptr<cudaStream_t> bondedStream =
      forceManager->getBondedStream();
  const std::shared_ptr<cudaStream_t> reciprocalStream =
      forceManager->getReciprocalStream();
  const std::shared_ptr<cudaStream_t> directStream =
      forceManager->getDirectStream();

  const std::shared_ptr<Force<long long int>> bondedForceValues =
      forceManager->getBondedForcevalues();
  const std::shared_ptr<Force<long long int>> reciprocalForceValues =
      forceManager->getReciprocalForcevalues();
  const std::shared_ptr<Force<long long int>> directForceValues =
      forceManager->getDirectForcevalues();

  context->setPrm(replacementParameters);

  // Parameter replacement defers backend reconstruction but does not make the
  // configured manager unavailable.
  CHECK(forceManager->isInitialized());
  CHECK(context->getPrm() == replacementParameters);
  CHECK(forceManager->getPrm() == replacementParameters);

  const ForceEvaluation observed = captureForceEvaluation(context);

  apo_test::CheckVectorsClose<double>("reconfigured force values",
                                      observed.forces, expected.forces, 1.0e-6);

  REQUIRE(observed.energyComponents.size() == expected.energyComponents.size());

  for (const auto &entry : expected.energyComponents) {
    INFO("Energy component: " << entry.first);

    REQUIRE(observed.energyComponents.count(entry.first) == 1);

    CHECK(observed.energyComponents.at(entry.first) ==
          Approx(entry.second).margin(1.0e-6));
  }

  // Backend objects were reconstructed, but manager-owned component resources
  // must retain their identities.
  CHECK(forceManager->getBondedStream() == bondedStream);
  CHECK(forceManager->getReciprocalStream() == reciprocalStream);
  CHECK(forceManager->getDirectStream() == directStream);

  CHECK(forceManager->getBondedForcevalues() == bondedForceValues);
  CHECK(forceManager->getReciprocalForcevalues() == reciprocalForceValues);
  CHECK(forceManager->getDirectForcevalues() == directForceValues);
}
