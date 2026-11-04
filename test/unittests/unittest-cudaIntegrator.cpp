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
#include "CudaIntegrator.h"
#include "ForceManager.h"
#include "Subscriber.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector_functions.h>

namespace {

constexpr int RANDOM_SEED = 314159;
constexpr double TIME_STEP = 0.002;
constexpr double TEMPERATURE = 300.0;

class CudaIntegratorProbe final : public CudaIntegrator {
public:
  using CudaIntegrator::CudaIntegrator;

  int getInitializeCount(void) const { return m_InitializeCount; }

  int getPropagateOneStepCount(void) const { return m_PropagateOneStepCount; }

  void setCompletedStepForTest(const unsigned long long int step) {
    m_TotNumSteps = step;
    return;
  }

  void checkForNanEnergyForTest(void) {
    this->checkForNanEnergy();
    return;
  }

public:
  int failOnAttempt = 0;
  unsigned long long int restartCompletedSteps = 0;
  std::vector<unsigned long long int> completedBeforeStep;

protected:
  void initializeImpl(void) override {
    m_InitializeCount++;
    return;
  }

  void initializeFromRestartFileImpl(
      const std::filesystem::path &rstFilePath) override {
    static_cast<void>(rstFilePath);
    m_TotNumSteps = restartCompletedSteps;
    return;
  }

  void propagateOneStepImpl(void) override {
    completedBeforeStep.push_back(this->getTotNumSteps());
    if (failOnAttempt == m_PropagateOneStepCount + 1) {
      APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                      "Deliberate propagation failure");
    }
    m_PropagateOneStepCount++;
    return;
  }

private:
  int m_InitializeCount = 0;
  int m_PropagateOneStepCount = 0;
};

class CudaIntegratorPartialProbe final : public CudaIntegrator {
public:
  using CudaIntegrator::CudaIntegrator;

protected:
  void initializeImpl(void) override { return; }
};

std::shared_ptr<CharmmContext> MakeInitializedContext(void) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto context = std::make_shared<CharmmContext>(psf, prm);
  context->setBoxDimensions({50.0, 50.0, 50.0});
  context->setCoordinates(crd);
  context->useHolonomicConstraints(false);
  context->setRandomSeed(RANDOM_SEED);
  context->assignVelocitiesAtTemperature(TEMPERATURE);

  return context;
}

class CountingSubscriber : public Subscriber {
public:
  explicit CountingSubscriber(const int reportFrequency) : Subscriber() {
    this->setReportFrequency(reportFrequency);
  }

  void onPropagationStart(const unsigned long long int startStep,
                          const int numSteps) override {
    startSteps.push_back(startStep);
    requestedLengths.push_back(numSteps);
    if (failOnStart) {
      APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                      "Deliberate subscriber start failure");
    }
    return;
  }

  void update(void) override {
    updateCount++;
    completedSteps.push_back(m_Integrator->getTotNumSteps());
    if (failOnUpdate) {
      APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                      "Deliberate subscriber update failure");
    }
    return;
  }

  int updateCount = 0;
  bool failOnStart = false;
  bool failOnUpdate = false;
  std::vector<unsigned long long int> startSteps;
  std::vector<unsigned long long int> completedSteps;
  std::vector<int> requestedLengths;
};

} // namespace

TEST_CASE("CudaIntegratorBaseConstructionAndTimeStep") {
  SECTION("DefaultConstructor") {
    CudaIntegrator integrator;

    CHECK(integrator.getTimeStep() == Approx(0.0));
    CHECK(integrator.getCharmmContext() == nullptr);
    CHECK(integrator.getSubscribers().empty() == true);
    CHECK(integrator.getReportFreqList().empty() == true);
    CHECK(integrator.getCurrentPropagatedStep() == 0);
    CHECK(integrator.getTotNumSteps() == 0);
    CHECK(integrator.getNumSteps() == 0);
    CHECK(integrator.getCoordsDelta().size() == 0);
    CHECK(integrator.getCoordsDeltaPrevious().size() == 0);
  }

  SECTION("TimeStepConstructor") {
    CudaIntegrator integrator(TIME_STEP);

    CHECK(integrator.getTimeStep() == Approx(TIME_STEP));
  }

  SECTION("TimeStepAndDebugConstructor") {
    CudaIntegrator integrator(TIME_STEP, 7);

    CHECK(integrator.getTimeStep() == Approx(TIME_STEP));
  }

  SECTION("SetTimeStep") {
    CudaIntegrator integrator;

    integrator.setTimeStep(TIME_STEP);
    CHECK(integrator.getTimeStep() == Approx(TIME_STEP));

    integrator.setTimeStep(0.001);
    CHECK(integrator.getTimeStep() == Approx(0.001));
  }

  SECTION("RejectsInvalidTimeStep") {
    const double infinity = std::numeric_limits<double>::infinity();

    apo_test::CheckApoCharmmError(
        []() { static_cast<void>(CudaIntegrator(0.0)); },
        ApoCharmmErrorCode::InvalidArgument,
        "Time step must be positive; observed 0.000000");

    apo_test::CheckApoCharmmError(
        [infinity]() { static_cast<void>(CudaIntegrator(infinity)); },
        ApoCharmmErrorCode::InvalidArgument,
        "Time step must be finite; observed " + std::to_string(infinity));

    CudaIntegrator integrator;

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setTimeStep(0.0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Time step must be positive; observed 0.000000");

    apo_test::CheckApoCharmmError([&]() { integrator.setTimeStep(infinity); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Time step must be finite; observed " +
                                      std::to_string(infinity));
  }

  SECTION("ValidatesFrequencies") {
    CudaIntegrator integrator;

    CHECK_NOTHROW(integrator.setDebugPrintFrequency(10));
    CHECK_NOTHROW(integrator.setNonbondedListUpdateFrequency(5));
    CHECK_NOTHROW(integrator.setRemoveCenterOfMassFrequency(100));

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setDebugPrintFrequency(-1); },
        ApoCharmmErrorCode::InvalidArgument,
        "Debug print frequency must be non-negative; observed -1");

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setNonbondedListUpdateFrequency(0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Nonbonded-list update frequency must be positive; observed 0");

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setRemoveCenterOfMassFrequency(0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Center-of-mass removal frequency must be positive; observed 0");

    apo_test::CheckApoCharmmError(
        []() { static_cast<void>(CudaIntegrator(TIME_STEP, -1)); },
        ApoCharmmErrorCode::InvalidArgument,
        "Debug print frequency must be non-negative; observed -1");
  }
}

TEST_CASE("CudaIntegratorBaseInvalidOperations") {
  CudaIntegrator integrator(TIME_STEP);

  SECTION("InitializeRequiresContext") {
    apo_test::CheckApoCharmmError(
        [&]() { integrator.initialize(); }, ApoCharmmErrorCode::NotInitialized,
        "CharmmContext must be set before initialization");
  }

  SECTION("RestartInitializationRequiresContext") {
    const std::filesystem::path filePath = "restart.rst";

    apo_test::CheckApoCharmmError(
        [&]() { integrator.initializeFromRestartFile(filePath); },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmContext must be set before initializing from a restart file");
  }

  SECTION("OneStepPropagationRequiresContext") {
    apo_test::CheckApoCharmmError(
        [&]() { integrator.propagateOneStep(); },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmContext must be set before propagation");
  }

  SECTION("DefaultRestartAndStepHooksAreNotImplemented") {
    const std::filesystem::path filePath = "restart.rst";

    CudaIntegratorPartialProbe probe(TIME_STEP);
    probe.setCharmmContext(MakeInitializedContext());

    apo_test::CheckApoCharmmError(
        [&]() { probe.initializeFromRestartFile(filePath); },
        ApoCharmmErrorCode::NotImplemented,
        "CudaIntegrator::initializeFromRestartFile is not implemented by the "
        "base class");

    apo_test::CheckApoCharmmError(
        [&]() { probe.propagateOneStep(); }, ApoCharmmErrorCode::NotImplemented,
        "CudaIntegrator::propagateOneStep is not implemented by the base "
        "class");
  }

  SECTION("IntegratorDescriptorsAreNotImplemented") {
    apo_test::CheckApoCharmmError(
        [&]() { (void)integrator.getIntegratorDescriptors(); },
        ApoCharmmErrorCode::NotImplemented,
        "CudaIntegrator::getIntegratorDescriptors is not implemented by the "
        "base class");
  }
}

TEST_CASE("CudaIntegratorBaseContextValidation") {
  SECTION("RejectsNullCharmmContext") {
    CudaIntegratorProbe integrator(TIME_STEP);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setCharmmContext(nullptr); },
        ApoCharmmErrorCode::InvalidArgument, "CharmmContext must not be null");

    CHECK(integrator.getCharmmContext() == nullptr);
  }

  SECTION("RejectsUninitializedAtomCountWithoutPoisoningIntegrator") {
    CudaIntegratorProbe integrator(TIME_STEP);
    auto incompleteContext = std::make_shared<CharmmContext>();

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setCharmmContext(incompleteContext); },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmContext atom count is not initialized; observed -1");

    CHECK(integrator.getCharmmContext() == nullptr);

    auto initializedContext = MakeInitializedContext();
    CHECK_NOTHROW(integrator.setCharmmContext(initializedContext));
    CHECK(integrator.getCharmmContext() == initializedContext);
  }

  SECTION("InitializationFailureRollsBackAttachment") {
    CudaIntegrator integrator(TIME_STEP);
    auto context = MakeInitializedContext();

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setCharmmContext(context); },
        ApoCharmmErrorCode::NotImplemented,
        "CudaIntegrator::initialize is not implemented by the base class");

    CHECK(integrator.getCharmmContext() == nullptr);
  }

  SECTION("RejectsSecondCharmmContext") {
    CudaIntegratorProbe integrator(TIME_STEP);
    auto context = MakeInitializedContext();
    integrator.setCharmmContext(context);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.setCharmmContext(context); },
        ApoCharmmErrorCode::InvalidArgument,
        "A CharmmContext object was already set for this CudaIntegrator.");

    CHECK(integrator.getCharmmContext() == context);
  }
}

TEST_CASE("CudaIntegratorBasePropagationValidation") {
  SECTION("RejectsZeroSteps") {
    CudaIntegratorProbe integrator(TIME_STEP);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.propagate(0); }, ApoCharmmErrorCode::InvalidArgument,
        "Number of propagation steps must be positive; observed 0");
    CHECK(integrator.getNumSteps() == 0);
    CHECK(integrator.getTotNumSteps() == 0ULL);
  }

  SECTION("RejectsNegativeSteps") {
    CudaIntegratorProbe integrator(TIME_STEP);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.propagate(-1); },
        ApoCharmmErrorCode::InvalidArgument,
        "Number of propagation steps must be positive; observed -1");
    CHECK(integrator.getNumSteps() == 0);
    CHECK(integrator.getTotNumSteps() == 0ULL);
  }

  SECTION("RequiresCharmmContext") {
    CudaIntegratorProbe integrator(TIME_STEP);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.propagate(1); }, ApoCharmmErrorCode::NotInitialized,
        "CharmmContext must be set before propagation");
  }

  SECTION("RequiresForceManager") {
    auto sourceContext = MakeInitializedContext();
    auto contextWithoutForceManager =
        std::make_shared<CharmmContext>(*sourceContext);
    REQUIRE(contextWithoutForceManager->getForceManager() == nullptr);

    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(contextWithoutForceManager);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.propagate(1); }, ApoCharmmErrorCode::NotInitialized,
        "CharmmContext must have a ForceManager before propagation");
  }

  SECTION("RequiresInitializedForceManager") {
    auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                  "toppar_water_ions.str");
    auto psf =
        std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
    auto context = std::make_shared<CharmmContext>(psf, prm);
    context->useHolonomicConstraints(false);
    REQUIRE(context->getForceManager() != nullptr);
    REQUIRE(context->getForceManager()->isInitialized() == false);

    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.propagate(1); }, ApoCharmmErrorCode::NotInitialized,
        "ForceManager must be initialized before propagation");
  }

  SECTION("PositiveStepCountPropagates") {
    auto context = MakeInitializedContext();
    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);

    CHECK(integrator.getInitializeCount() == 1);
    CHECK(integrator.getPropagateOneStepCount() == 0);

    CHECK_NOTHROW(integrator.propagate(3));

    CHECK(integrator.getNumSteps() == 3);
    CHECK(integrator.getTotNumSteps() == 3ULL);
    CHECK(integrator.getCurrentPropagatedStep() == 3);
    CHECK(integrator.getPropagateOneStepCount() == 3);
    CHECK((integrator.completedBeforeStep ==
           std::vector<unsigned long long int>{0ULL, 1ULL, 2ULL}));

    CHECK_NOTHROW(integrator.propagate(2));
    CHECK(integrator.getTotNumSteps() == 5ULL);
    CHECK(integrator.getNumSteps() == 2);
  }

  SECTION("FailurePreservesOnlyCompletedSteps") {
    auto context = MakeInitializedContext();
    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);
    integrator.failOnAttempt = 3;

    apo_test::CheckApoCharmmError([&]() { integrator.propagate(5); },
                                  ApoCharmmErrorCode::Runtime,
                                  "Deliberate propagation failure");
    CHECK(integrator.getTotNumSteps() == 2ULL);
    CHECK(integrator.getCurrentPropagatedStep() == 2);
    CHECK(integrator.getNumSteps() == 5);

    integrator.failOnAttempt = 0;
    CHECK_NOTHROW(integrator.propagate(1));
    CHECK(integrator.getTotNumSteps() == 3ULL);
    CHECK(integrator.getNumSteps() == 1);
  }

  SECTION("WrappedGetterDerivesFromFullCount") {
    auto context = MakeInitializedContext();
    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);
    const auto maximum =
        static_cast<unsigned long long int>(std::numeric_limits<int>::max());
    integrator.setCompletedStepForTest(maximum - 1ULL);
    CHECK(integrator.getCurrentPropagatedStep() ==
          std::numeric_limits<int>::max() - 1);

    integrator.propagate(1);
    CHECK(integrator.getTotNumSteps() == maximum);
    CHECK(integrator.getCurrentPropagatedStep() ==
          std::numeric_limits<int>::max());
    integrator.propagate(1);
    CHECK(integrator.getTotNumSteps() == maximum + 1ULL);
    CHECK(integrator.getCurrentPropagatedStep() == 0);
    integrator.propagate(1);
    CHECK(integrator.getTotNumSteps() == maximum + 2ULL);
    CHECK(integrator.getCurrentPropagatedStep() == 1);
  }

  SECTION("RejectsCompletedCounterOverflowBeforeSteps") {
    auto context = MakeInitializedContext();
    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);
    const auto maximum = std::numeric_limits<unsigned long long int>::max();
    integrator.setCompletedStepForTest(maximum);
    apo_test::CheckApoCharmmError(
        [&]() { integrator.propagate(1); }, ApoCharmmErrorCode::InvalidArgument,
        "Requested propagation exceeds the completed-step counter range");
    CHECK(integrator.getTotNumSteps() == maximum);
    CHECK(integrator.getPropagateOneStepCount() == 0);
    CHECK(integrator.getNumSteps() == 0);
  }

  SECTION("SubscribersObserveRunAnnouncementsAndCompletedSteps") {
    auto context = MakeInitializedContext();
    context->getPotentialEnergy().setToValue(0.0);
    auto integrator = std::make_shared<CudaIntegratorProbe>(TIME_STEP);
    integrator->setCharmmContext(context);
    auto subscriber = std::make_shared<CountingSubscriber>(2);
    integrator->subscribe(subscriber);

    integrator->propagate(5);
    CHECK(
        (subscriber->startSteps == std::vector<unsigned long long int>{0ULL}));
    CHECK((subscriber->requestedLengths == std::vector<int>{5}));
    CHECK((subscriber->completedSteps ==
           std::vector<unsigned long long int>{2ULL, 4ULL}));

    // CHARMM counts ISTEP from the start of the run, not from each call: the
    // frequency-2 subscriber reports at step 6 although this call has one step.
    integrator->propagate(1);
    CHECK((subscriber->startSteps ==
           std::vector<unsigned long long int>{0ULL, 5ULL}));
    CHECK((subscriber->requestedLengths == std::vector<int>{5, 1}));
    CHECK((subscriber->completedSteps ==
           std::vector<unsigned long long int>{2ULL, 4ULL, 6ULL}));
    CHECK(integrator->getTotNumSteps() == 6ULL);
    CHECK(subscriber->getStartStep().value() == 0ULL);

    // A subscriber added later begins its own run at the step count it first
    // observes, as a restarted CHARMM run would.
    auto lateSubscriber = std::make_shared<CountingSubscriber>(4);
    integrator->subscribe(lateSubscriber);
    CHECK_FALSE(lateSubscriber->getStartStep().has_value());
    integrator->propagate(3);
    CHECK(lateSubscriber->getStartStep().value() == 6ULL);
    CHECK(lateSubscriber->completedSteps.empty());
    integrator->propagate(1);
    CHECK((lateSubscriber->completedSteps ==
           std::vector<unsigned long long int>{10ULL}));
    CHECK((subscriber->completedSteps ==
           std::vector<unsigned long long int>{2ULL, 4ULL, 6ULL, 8ULL, 10ULL}));
  }

  SECTION("SubscriberStartFailureDoesNotAdvanceClock") {
    auto context = MakeInitializedContext();
    auto integrator = std::make_shared<CudaIntegratorProbe>(TIME_STEP);
    integrator->setCharmmContext(context);
    auto subscriber = std::make_shared<CountingSubscriber>(1);
    integrator->subscribe(subscriber);
    subscriber->failOnStart = true;
    apo_test::CheckApoCharmmError([&]() { integrator->propagate(3); },
                                  ApoCharmmErrorCode::Runtime,
                                  "Deliberate subscriber start failure");
    CHECK(integrator->getTotNumSteps() == 0ULL);
    CHECK(integrator->getNumSteps() == 3);
    CHECK(integrator->getPropagateOneStepCount() == 0);
  }

  SECTION("SubscriberUpdateFailureKeepsCompletedStep") {
    auto context = MakeInitializedContext();
    context->getPotentialEnergy().setToValue(0.0);
    auto integrator = std::make_shared<CudaIntegratorProbe>(TIME_STEP);
    integrator->setCharmmContext(context);
    auto subscriber = std::make_shared<CountingSubscriber>(1);
    integrator->subscribe(subscriber);
    subscriber->failOnUpdate = true;
    apo_test::CheckApoCharmmError([&]() { integrator->propagate(3); },
                                  ApoCharmmErrorCode::Runtime,
                                  "Deliberate subscriber update failure");
    CHECK(integrator->getTotNumSteps() == 1ULL);
    CHECK((subscriber->completedSteps ==
           std::vector<unsigned long long int>{1ULL}));
  }

  SECTION("RestartImportStartsANewRunForAttachedSubscribers") {
    auto context = MakeInitializedContext();
    context->getPotentialEnergy().setToValue(0.0);
    auto integrator = std::make_shared<CudaIntegratorProbe>(TIME_STEP);
    integrator->setCharmmContext(context);
    auto subscriber = std::make_shared<CountingSubscriber>(4);
    integrator->subscribe(subscriber);
    integrator->propagate(3);
    CHECK(subscriber->getStartStep().value() == 0ULL);
    CHECK(subscriber->completedSteps.empty());

    // Resuming at NPRIV=1005 begins a new CHARMM-style run: the next report is
    // due at step 1009, not at the lifetime multiple 1008.
    integrator->restartCompletedSteps = 1005ULL;
    integrator->initializeFromRestartFile("unused.rst");
    CHECK_FALSE(subscriber->getStartStep().has_value());
    integrator->propagate(4);
    CHECK(subscriber->getStartStep().value() == 1005ULL);
    CHECK((subscriber->startSteps ==
           std::vector<unsigned long long int>{0ULL, 1005ULL}));
    CHECK((subscriber->completedSteps ==
           std::vector<unsigned long long int>{1009ULL}));
  }

  SECTION("DirectOneStepUsesValidatedBoundary") {
    auto context = MakeInitializedContext();
    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);

    CHECK_NOTHROW(integrator.propagateOneStep());
    CHECK(integrator.getPropagateOneStepCount() == 1);
    CHECK(integrator.getTotNumSteps() == 0ULL);
    CHECK(integrator.getCurrentPropagatedStep() == 0);
    CHECK(integrator.getNumSteps() == 0);
  }
}

TEST_CASE("CudaIntegratorBaseNanEnergyValidation") {
  SECTION("RejectsNanKineticEnergy") {
    auto context = MakeInitializedContext();
    context->getPotentialEnergy().setToValue(0.0);
    context->getVelocitiesInverseMasses().setToValue(
        make_double4(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0));

    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.checkForNanEnergyForTest(); },
        ApoCharmmErrorCode::Runtime, "Kinetic energy is NaN");
  }

  SECTION("RejectsNanPotentialEnergy") {
    auto context = MakeInitializedContext();
    context->getPotentialEnergy().setToValue(
        std::numeric_limits<double>::quiet_NaN());

    CudaIntegratorProbe integrator(TIME_STEP);
    integrator.setCharmmContext(context);

    apo_test::CheckApoCharmmError(
        [&]() { integrator.checkForNanEnergyForTest(); },
        ApoCharmmErrorCode::Runtime, "Potential energy is NaN");
  }
}

TEST_CASE("CudaIntegratorBaseSubscriberManagement") {
  auto integrator = std::make_shared<CudaIntegrator>(TIME_STEP);

  auto sub2 = std::make_shared<CountingSubscriber>(2);
  auto sub3 = std::make_shared<CountingSubscriber>(3);

  SECTION("SubscribeNullSubscriberThrows") {
    apo_test::CheckApoCharmmError([&]() { integrator->subscribe(nullptr); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Subscriber must not be null");

    CHECK(integrator->getSubscribers().empty() == true);
    CHECK(integrator->getReportFreqList().empty() == true);
  }

  SECTION("SubscribeRequiresSharedOwnership") {
    CudaIntegrator stackIntegrator(TIME_STEP);

    apo_test::CheckApoCharmmError(
        [&]() { stackIntegrator.subscribe(sub2); },
        ApoCharmmErrorCode::NotInitialized,
        "CudaIntegrator must be owned by std::shared_ptr before subscribing");

    CHECK(stackIntegrator.getSubscribers().empty() == true);
    CHECK(stackIntegrator.getReportFreqList().empty() == true);
  }

  SECTION("SubscribeSingleSubscriber") {
    integrator->subscribe(sub2);

    REQUIRE(integrator->getSubscribers().size() == 1);
    REQUIRE(integrator->getReportFreqList().size() == 1);

    CHECK(integrator->getSubscribers()[0] == sub2);
    CHECK(integrator->getReportFreqList()[0] == 2);
  }

  SECTION("SubscribeDuplicateSubscriberThrowsWithoutDuplicatingLists") {
    integrator->subscribe(sub2);

    apo_test::CheckApoCharmmError(
        [&]() { integrator->subscribe(sub2); },
        ApoCharmmErrorCode::InvalidArgument,
        "Subscriber is already subscribed to this CudaIntegrator");

    REQUIRE(integrator->getSubscribers().size() == 1);
    REQUIRE(integrator->getReportFreqList().size() == 1);
    CHECK(integrator->getSubscribers()[0] == sub2);
    CHECK(integrator->getReportFreqList()[0] == 2);
  }

  SECTION("SubscribeSubscriberList") {
    integrator->subscribe(std::vector<std::shared_ptr<Subscriber>>{sub2, sub3});

    REQUIRE(integrator->getSubscribers().size() == 2);
    REQUIRE(integrator->getReportFreqList().size() == 2);

    CHECK(integrator->getSubscribers()[0] == sub2);
    CHECK(integrator->getSubscribers()[1] == sub3);
    CHECK(integrator->getReportFreqList() == std::vector<int>{2, 3});
  }

  SECTION("UnsubscribeSingleSubscriber") {
    integrator->subscribe(std::vector<std::shared_ptr<Subscriber>>{sub2, sub3});

    integrator->unsubscribe(sub2);

    REQUIRE(integrator->getSubscribers().size() == 1);
    REQUIRE(integrator->getReportFreqList().size() == 1);

    CHECK(integrator->getSubscribers()[0] == sub3);
    CHECK(integrator->getReportFreqList()[0] == 3);
  }

  SECTION("UnsubscribeSubscriberList") {
    integrator->subscribe(std::vector<std::shared_ptr<Subscriber>>{sub2, sub3});

    integrator->unsubscribe(
        std::vector<std::shared_ptr<Subscriber>>{sub2, sub3});

    CHECK(integrator->getSubscribers().empty() == true);
    CHECK(integrator->getReportFreqList().empty() == true);
  }

  SECTION("UnsubscribeNullSubscriberThrows") {
    apo_test::CheckApoCharmmError([&]() { integrator->unsubscribe(nullptr); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Subscriber must not be null");

    CHECK(integrator->getSubscribers().empty() == true);
    CHECK(integrator->getReportFreqList().empty() == true);
  }

  SECTION("UnsubscribeMissingSubscriberThrows") {
    apo_test::CheckApoCharmmError([&]() { integrator->unsubscribe(sub2); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Subscriber not found (file \"\")");
  }
}
