// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaIntegrator.h"

#include "ApoCharmmError.h"
#include "Constants.h"
#include "Subscriber.h"
#include "cuda_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>

CudaIntegrator::CudaIntegrator(void)
    : m_TimeStep(0.0), m_DebugPrintFrequency(0), m_Context(nullptr),
      m_StepsSinceNeighborListUpdate(1), m_TotNumSteps(0), m_NumSteps(0),
      m_HolonomicConstraint(nullptr), m_CoordsRef(), m_CoordsDelta(),
      m_CoordsDeltaPrevious(), m_IntegratorStream(nullptr),
      m_IntegratorMemcpyStream(nullptr), m_UsingHolonomicConstraints(false),
      m_Subscribers(), m_ReportFreqList(), m_NonbondedListUpdateFrequency(20),
      m_RemoveCenterOfMassFrequency(1000),
      m_IntegratorTypeName("BaseClass integrator") {
  m_IntegratorStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_IntegratorStream.get()));

  try {
    m_IntegratorMemcpyStream = std::make_shared<cudaStream_t>();
    cudaCheck(cudaStreamCreate(m_IntegratorMemcpyStream.get()));
  } catch (...) {
    destroy_cuda_stream_noexcept(m_IntegratorStream.get());
    m_IntegratorStream.reset();
    throw;
  }
}

CudaIntegrator::CudaIntegrator(const double timeStep) : CudaIntegrator() {
  this->setTimeStep(timeStep);
}

CudaIntegrator::CudaIntegrator(const double timeStep,
                               const int debugPrintFrequency)
    : CudaIntegrator(timeStep) {
  this->setDebugPrintFrequency(debugPrintFrequency);
}

CudaIntegrator::~CudaIntegrator(void) noexcept {
  if (m_IntegratorMemcpyStream != nullptr) {
    destroy_cuda_stream_noexcept(m_IntegratorMemcpyStream.get());
    m_IntegratorMemcpyStream.reset();
  }

  if (m_IntegratorStream != nullptr) {
    destroy_cuda_stream_noexcept(m_IntegratorStream.get());
    m_IntegratorStream.reset();
  }
}

double CudaIntegrator::getTimeStep(void) const {
  return (m_TimeStep * charmm::constants::timfac);
}

void CudaIntegrator::setTimeStep(const double timeStep) {
  APOCHARMM_REQUIRE(
      std::isfinite(timeStep), ApoCharmmErrorCode::InvalidArgument,
      "Time step must be finite; observed " + std::to_string(timeStep));

  APOCHARMM_REQUIRE(timeStep > 0.0, ApoCharmmErrorCode::InvalidArgument,
                    "Time step must be positive; observed " +
                        std::to_string(timeStep));

  // Converting from ps to AKMA units ltm/consta_ltm
  m_TimeStep = timeStep / charmm::constants::timfac;

  return;
}

void CudaIntegrator::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  APOCHARMM_REQUIRE(ctx != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "CharmmContext must not be null");

  APOCHARMM_REQUIRE(
      m_Context == nullptr, ApoCharmmErrorCode::InvalidArgument,
      "A CharmmContext object was already set for this CudaIntegrator.");

  const int numAtoms = ctx->getNumAtoms();
  APOCHARMM_REQUIRE(numAtoms >= 0, ApoCharmmErrorCode::NotInitialized,
                    "CharmmContext atom count is not initialized; observed " +
                        std::to_string(numAtoms));

  m_Context = ctx;

  try {
    m_CoordsRef.resize(numAtoms);
    m_CoordsDelta.resize(numAtoms);
    m_CoordsDeltaPrevious.resize(numAtoms);

    m_UsingHolonomicConstraints = m_Context->usingHolonomicConstraints();
    if (m_UsingHolonomicConstraints) {
      m_HolonomicConstraint = std::make_shared<CudaHolonomicConstraint>();
      m_HolonomicConstraint->setCharmmContext(ctx);
      m_HolonomicConstraint->setup(m_TimeStep);
      m_HolonomicConstraint->setStream(m_IntegratorStream);
    }
    this->initializeImpl();
  } catch (...) {
    m_HolonomicConstraint.reset();
    m_UsingHolonomicConstraints = false;
    m_Context.reset();
    throw;
  }

  return;
}

const std::shared_ptr<CharmmContext>
CudaIntegrator::getCharmmContext(void) const {
  return m_Context;
}

std::shared_ptr<CharmmContext> CudaIntegrator::getCharmmContext(void) {
  return m_Context;
}

void CudaIntegrator::initialize(void) {
  APOCHARMM_REQUIRE(m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "CharmmContext must be set before initialization");

  this->initializeImpl();

  return;
}

void CudaIntegrator::initializeImpl(void) {
  APOCHARMM_THROW(
      ApoCharmmErrorCode::NotImplemented,
      "CudaIntegrator::initialize is not implemented by the base class");
}

void CudaIntegrator::initializeFromRestartFile(
    const std::filesystem::path &rstFilePath) {
  APOCHARMM_REQUIRE(
      m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before initializing from a restart file");

  this->requireCurrentState();
  this->initializeFromRestartFileImpl(rstFilePath);

  // Continuing from a restart begins a new CHARMM-style run for every attached
  // subscriber; the next propagate() fixes the new origin at the imported step.
  for (const auto &subscriber : m_Subscribers)
    subscriber->setStartStep(std::nullopt);

  return;
}

void CudaIntegrator::initializeFromRestartFileImpl(
    const std::filesystem::path &rstFilePath) {
  static_cast<void>(rstFilePath);
  APOCHARMM_THROW(ApoCharmmErrorCode::NotImplemented,
                  "CudaIntegrator::initializeFromRestartFile is not "
                  "implemented by the base class");
}

void CudaIntegrator::propagateOneStep(void) {
  this->requirePropagationReady();
  this->propagateOneStepImpl();
  return;
}

void CudaIntegrator::propagateOneStepImpl(void) {
  APOCHARMM_THROW(
      ApoCharmmErrorCode::NotImplemented,
      "CudaIntegrator::propagateOneStep is not implemented by the base class");
}

void CudaIntegrator::requirePropagationReady(void) const {
  APOCHARMM_REQUIRE(m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "CharmmContext must be set before propagation");

  const std::shared_ptr<ForceManager> forceManager =
      m_Context->getForceManager();

  APOCHARMM_REQUIRE(
      forceManager != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must have a ForceManager before propagation");

  APOCHARMM_REQUIRE(forceManager->isInitialized(),
                    ApoCharmmErrorCode::NotInitialized,
                    "ForceManager must be initialized before propagation");

  this->requireCurrentState();

  return;
}

void CudaIntegrator::requireCurrentState(void) const { return; }

void CudaIntegrator::propagate(const int numSteps) {
  APOCHARMM_REQUIRE(numSteps > 0, ApoCharmmErrorCode::InvalidArgument,
                    "Number of propagation steps must be positive; observed " +
                        std::to_string(numSteps));

  this->requirePropagationReady();

  const unsigned long long int requestedSteps =
      static_cast<unsigned long long int>(numSteps);
  APOCHARMM_REQUIRE(
      requestedSteps <=
          std::numeric_limits<unsigned long long int>::max() - m_TotNumSteps,
      ApoCharmmErrorCode::InvalidArgument,
      "Requested propagation exceeds the completed-step counter range");

  m_Context->resetNeighborList();

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  m_StepsSinceNeighborListUpdate = 0;

  int minReportFreq = 100000;

  // if there are subscribers, find the smallest report freq instead
  for (std::size_t i = 0; i < m_Subscribers.size(); i++) {
    if (m_ReportFreqList[i] < minReportFreq)
      minReportFreq = m_ReportFreqList[i];
  }

  m_NumSteps = numSteps;
  const unsigned long long int startStep = m_TotNumSteps;
  for (const auto &subscriber : m_Subscribers) {
    // Announce before fixing the origin so a subscriber can tell a fresh run
    // (empty origin) from a continuing one. CHARMM numbers ISTEP from the start
    // of a run; the first propagation after subscription (or after a subscriber
    // reset) fixes that origin.
    subscriber->onPropagationStart(startStep, numSteps);
    if (!subscriber->getStartStep().has_value())
      subscriber->setStartStep(startStep);
  }

  for (int completedInCall = 0; completedInCall < numSteps; completedInCall++) {
    const int step = completedInCall + 1;

    if (step % 10000 == 0) {
      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      std::chrono::steady_clock::duration duration = end - start;

      std::cout << "Step = " << step << " "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       duration)
                       .count()
                << "ms" << std::endl;
      start = std::chrono::steady_clock::now();
    }

    // if (step % removeCenterOfMassFrequency == 0) {
    //   context->removeCenterOfMassMotion();
    // }
    this->propagateOneStepImpl();

    m_TotNumSteps++;
    m_StepsSinceNeighborListUpdate++;

    // Check if we have nan-esque energy.
    if (step % minReportFreq == 0)
      this->checkForNanEnergy();

    // Check if report is needed for one or more of the subscribers
    this->reportIfNeeded();
  }

  return;
}

void CudaIntegrator::setDebugPrintFrequency(const int freq) {
  APOCHARMM_REQUIRE(freq >= 0, ApoCharmmErrorCode::InvalidArgument,
                    "Debug print frequency must be non-negative; observed " +
                        std::to_string(freq));

  m_DebugPrintFrequency = freq;

  return;
}

void CudaIntegrator::setNonbondedListUpdateFrequency(const int freq) {
  APOCHARMM_REQUIRE(
      freq > 0, ApoCharmmErrorCode::InvalidArgument,
      "Nonbonded-list update frequency must be positive; observed " +
          std::to_string(freq));

  m_NonbondedListUpdateFrequency = freq;

  return;
}

void CudaIntegrator::subscribe(std::shared_ptr<Subscriber> sub) {
  APOCHARMM_REQUIRE(sub != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "Subscriber must not be null");

  APOCHARMM_REQUIRE(std::find(m_Subscribers.begin(), m_Subscribers.end(),
                              sub) == m_Subscribers.end(),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Subscriber is already subscribed to this CudaIntegrator");

  const std::shared_ptr<CudaIntegrator> integrator =
      this->weak_from_this().lock();
  APOCHARMM_REQUIRE(
      integrator != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CudaIntegrator must be owned by std::shared_ptr before subscribing");

  sub->setCharmmContext(m_Context);

  // JEG250610: Subscribers can query integrator for time step. We don't need
  // N copies floating around sub->setTimeStepFromIntegrator(m_TimeStep *
  // charmm::constants::timfac);

  sub->setIntegrator(integrator);

  // A new subscription starts a new CHARMM-style run whose origin is fixed by
  // the next propagate() call, so a restart loaded after subscribing is
  // honored.
  sub->setStartStep(std::nullopt);
  m_Subscribers.push_back(sub);
  m_ReportFreqList.push_back(sub->getReportFrequency());

  return;
}

void CudaIntegrator::subscribe(
    const std::vector<std::shared_ptr<Subscriber>> &sublist) {
  for (std::shared_ptr<Subscriber> sub : sublist)
    this->subscribe(sub);
  return;
}

void CudaIntegrator::unsubscribe(std::shared_ptr<Subscriber> sub) {
  APOCHARMM_REQUIRE(sub != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "Subscriber must not be null");

  const auto subIterator =
      std::find(m_Subscribers.begin(), m_Subscribers.end(), sub);

  APOCHARMM_REQUIRE(
      subIterator != m_Subscribers.end(), ApoCharmmErrorCode::InvalidArgument,
      "Subscriber not found (file \"" + sub->getFilePath().string() + "\")");

  const std::size_t index = static_cast<std::size_t>(
      std::distance(m_Subscribers.begin(), subIterator));
  m_Subscribers.erase(subIterator);
  m_ReportFreqList.erase(m_ReportFreqList.begin() + index);

  return;
}

void CudaIntegrator::unsubscribe(
    const std::vector<std::shared_ptr<Subscriber>> &sublist) {
  for (std::size_t i = 0; i < sublist.size(); i++)
    this->unsubscribe(sublist[i]);
  return;
}

const std::vector<std::shared_ptr<Subscriber>> &
CudaIntegrator::getSubscribers(void) const {
  return m_Subscribers;
}

std::vector<std::shared_ptr<Subscriber>> &CudaIntegrator::getSubscribers(void) {
  return m_Subscribers;
}

const std::vector<int> &CudaIntegrator::getReportFreqList(void) const {
  return m_ReportFreqList;
}

std::vector<int> &CudaIntegrator::getReportFreqList(void) {
  return m_ReportFreqList;
}

void CudaIntegrator::setRemoveCenterOfMassFrequency(const int freq) {
  APOCHARMM_REQUIRE(
      freq > 0, ApoCharmmErrorCode::InvalidArgument,
      "Center-of-mass removal frequency must be positive; observed " +
          std::to_string(freq));

  m_RemoveCenterOfMassFrequency = freq;

  return;
}

const CudaContainer<double4> &CudaIntegrator::getCoordsDelta(void) const {
  this->requireCurrentState();
  return m_CoordsDelta;
}

CudaContainer<double4> &CudaIntegrator::getCoordsDelta(void) {
  this->requireCurrentState();
  return m_CoordsDelta;
}

const CudaContainer<double4> &
CudaIntegrator::getCoordsDeltaPrevious(void) const {
  this->requireCurrentState();
  return m_CoordsDeltaPrevious;
}

CudaContainer<double4> &CudaIntegrator::getCoordsDeltaPrevious(void) {
  this->requireCurrentState();
  return m_CoordsDeltaPrevious;
}

// JEG260814: Deprecate this.
std::map<std::string, std::string>
CudaIntegrator::getIntegratorDescriptors(void) {
  APOCHARMM_THROW(ApoCharmmErrorCode::NotImplemented,
                  "CudaIntegrator::getIntegratorDescriptors is not implemented "
                  "by the base class");
}

int CudaIntegrator::getCurrentPropagatedStep(void) const {
  return CudaIntegrator::wrapCurrentPropagatedStep(m_TotNumSteps);
}

unsigned long long int CudaIntegrator::getTotNumSteps(void) const {
  return m_TotNumSteps;
}

int CudaIntegrator::getNumSteps(void) const { return m_NumSteps; }

void CudaIntegrator::reportIfNeeded(void) {
  // Mirror CHARMM's MOD(ISTEP,NSAVC) test, where ISTEP counts the steps
  // completed since the run began rather than since this propagate() call.
  for (std::size_t i = 0; i < m_ReportFreqList.size(); i++) {
    const std::optional<unsigned long long int> &runStart =
        m_Subscribers[i]->getStartStep();

    APOCHARMM_REQUIRE(runStart.has_value() && *runStart <= m_TotNumSteps,
                      ApoCharmmErrorCode::NotInitialized,
                      "Subscriber run origin is missing or later than the "
                      "completed step count");

    const unsigned long long int stepsSinceRunStart = m_TotNumSteps - *runStart;
    if (stepsSinceRunStart %
            static_cast<unsigned long long int>(m_ReportFreqList[i]) ==
        0ULL) {
      m_Subscribers[i]->update();
    }
  }

  return;
}

void CudaIntegrator::checkForNanEnergy(void) {
  // Check if we have nan-esque energy
  CudaContainer<double> &potEnergyCC = m_Context->getPotentialEnergy();
  potEnergyCC.transferFromDevice();
  double potEnergy = potEnergyCC.getHostArray()[0];
  double kinEnergy = m_Context->getKineticEnergy();

  APOCHARMM_REQUIRE(!std::isnan(kinEnergy), ApoCharmmErrorCode::Runtime,
                    "Kinetic energy is NaN");
  APOCHARMM_REQUIRE(!std::isnan(potEnergy), ApoCharmmErrorCode::Runtime,
                    "Potential energy is NaN");

  return;
}

int CudaIntegrator::wrapCurrentPropagatedStep(
    const unsigned long long int propagatedStep) {
  constexpr unsigned long long int WRAP_MODULUS =
      static_cast<unsigned long long int>(std::numeric_limits<int>::max()) +
      1ULL;
  return static_cast<int>(propagatedStep % WRAP_MODULUS);
}
