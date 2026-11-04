// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaLangevinPistonIntegrator.h"

#include "ApoCharmmError.h"
#include "Constants.h"
#include "Crystal.h"
#include "CurandStateString.h"
#include "RestartIO.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <random>
#include <vector_functions.h>

CudaLangevinPistonIntegrator::CudaLangevinPistonIntegrator(
    const double timeStep)
    : CudaIntegrator(timeStep) {
  m_UsingHolonomicConstraints = true;

  m_UsingNoseHooverThermostat = true;

  m_ReferenceTemperature = 300.0;

  // Allocate memory for Nose-Hoover variables
  m_NoseHooverPistonMass.resize(1);
  m_NoseHooverPistonVelocity.resize(1);
  m_NoseHooverPistonVelocityPrevious.resize(1);
  m_NoseHooverPistonForce.resize(1);
  m_NoseHooverPistonForcePrevious.resize(1);

  // Set Nose-Hoover variables to default values
  m_NoseHooverPistonMass.setToValue(-9999.9999);
  m_NoseHooverPistonVelocity.setToValue(0.0);
  m_NoseHooverPistonVelocityPrevious.setToValue(0.0);
  m_NoseHooverPistonForce.setToValue(0.0);
  m_NoseHooverPistonForcePrevious.setToValue(0.0);

  // Allocate pressure variables
  m_HolonomicConstraintVirial.resize(9);
  m_KineticPressureTensor.resize(9);
  m_PressureTensor.resize(9);
  m_PressureScalar.resize(1);
  m_ReferencePressureTensor.resize(9);
  m_DeltaPressureTensor.resize(9);
  m_DeltaKineticPressureTensor.resize(9);
  m_StaticDeltaPressureTensor.resize(9);
  // m_PressureReductionPartialSums is allocated in initializeImpl().

  // Set pressure variables to default values
  m_LangevinPistonDegreesOfFreedom = -1;
  m_HolonomicConstraintVirial.setToValue(0.0);
  m_KineticPressureTensor.setToValue(0.0);
  m_PressureTensor.setToValue(0.0);
  m_PressureScalar.setToValue(0.0);
  m_ReferencePressureTensor.set({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  m_DeltaPressureTensor.setToValue(0.0);
  m_DeltaKineticPressureTensor.setToValue(0.0);
  m_StaticDeltaPressureTensor.setToValue(0.0);

  m_Pgamma = 0.0;

  m_OnStepCrystalFactor.resize(3);
  m_HalfStepCrystalFactor.resize(3);

  m_OnStepCrystalFactor.setToValue(0.0);
  m_HalfStepCrystalFactor.setToValue(0.0);

  std::random_device rd{};
  m_Seed = rd();
  m_RngSequencePos = 0;
  m_RngStates = nullptr;

  m_ConstantSurfaceTensionFlag = false;
  m_SurfaceTension.resize(1);

  m_MaxPredictorCorrectorIterations = 3;

  // 0 -> New "JUNG" T
  // 1 -> Old T
  m_AverageWindowSize = 0;
  m_KineticEnergy.resize(2);
  // m_KineticEnergyPartialSums is allocated in this->initialize();
  m_AverageTemperature.resize(2);
  m_AveragePressureTensor.resize(9);
  m_AveragePressureScalar.resize(1);

  m_KineticEnergy.setToValue(0.0);
  m_AverageTemperature.setToValue(0.0);
  m_AveragePressureTensor.setToValue(0.0);
  m_AveragePressureScalar.setToValue(0.0);

  m_UsingOldTemperature = false;
}

CudaLangevinPistonIntegrator::~CudaLangevinPistonIntegrator(void) noexcept {
  ::deallocate_noexcept(&m_RngStates);
}

void CudaLangevinPistonIntegrator::useNoseHooverThermostat(
    const bool usingNoseHooverThermostat) {
  this->requireCurrentState();
  m_UsingNoseHooverThermostat = usingNoseHooverThermostat;
  return;
}

void CudaLangevinPistonIntegrator::setReferenceTemperature(
    const double referenceTemperature) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(std::isfinite(referenceTemperature),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference temperature must be finite; observed " +
                        std::to_string(referenceTemperature));

  APOCHARMM_REQUIRE(referenceTemperature >= 0.0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference temperature must be non-negative; observed " +
                        std::to_string(referenceTemperature));

  m_ReferenceTemperature = referenceTemperature;

  if (m_LangevinPistonDegreesOfFreedom != -1) {
    // If the temperature changes, the friction variables need to be updated
    double oldGamma = m_Pgamma;
    this->setLangevinPistonFriction(oldGamma);
  }

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonMass(
    const double noseHooverPistonMass) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(std::isfinite(noseHooverPistonMass),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Nose-Hoover piston mass must be finite; observed " +
                        std::to_string(noseHooverPistonMass));

  APOCHARMM_REQUIRE(noseHooverPistonMass >= 0.0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Nose-Hoover piston mass must be non-negative; observed " +
                        std::to_string(noseHooverPistonMass));

  m_NoseHooverPistonMass.setToValue(noseHooverPistonMass);

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonVelocity(
    const double noseHooverPistonVelocity) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(std::isfinite(noseHooverPistonVelocity),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Nose-Hoover piston velocity must be finite; observed " +
                        std::to_string(noseHooverPistonVelocity));

  m_NoseHooverPistonVelocity.setToValue(noseHooverPistonVelocity);

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonVelocityPrevious(
    const double noseHooverPistonVelocityPrevious) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(
      std::isfinite(noseHooverPistonVelocityPrevious),
      ApoCharmmErrorCode::InvalidArgument,
      "Previous Nose-Hoover piston velocity must be finite; observed " +
          std::to_string(noseHooverPistonVelocityPrevious));

  m_NoseHooverPistonVelocityPrevious.setToValue(
      noseHooverPistonVelocityPrevious);

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonForce(
    const double noseHooverPistonForce) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(std::isfinite(noseHooverPistonForce),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Nose-Hoover piston force must be finite; observed " +
                        std::to_string(noseHooverPistonForce));

  m_NoseHooverPistonForce.setToValue(noseHooverPistonForce);

  return;
}

void CudaLangevinPistonIntegrator::setNoseHooverPistonForcePrevious(
    const double noseHooverPistonForcePrevious) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(
      std::isfinite(noseHooverPistonForcePrevious),
      ApoCharmmErrorCode::InvalidArgument,
      "Previous Nose-Hoover piston force must be finite; observed " +
          std::to_string(noseHooverPistonForcePrevious));

  m_NoseHooverPistonForcePrevious.setToValue(noseHooverPistonForcePrevious);

  return;
}

void CudaLangevinPistonIntegrator::setMaxPredictorCorrectorIterations(
    const int maxPredictorCorrectorIterations) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(
      maxPredictorCorrectorIterations > 0, ApoCharmmErrorCode::InvalidArgument,
      "Maximum predictor-corrector iterations must be positive; observed " +
          std::to_string(maxPredictorCorrectorIterations));

  m_MaxPredictorCorrectorIterations = maxPredictorCorrectorIterations;

  return;
}

void CudaLangevinPistonIntegrator::useOldTemperature(
    const bool usingOldTemperature) {
  this->requireCurrentState();
  m_UsingOldTemperature = usingOldTemperature;
  return;
}

void CudaLangevinPistonIntegrator::setReferencePressure(
    const std::vector<double> &referencePressureTensor) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(
      referencePressureTensor.size() == 9, ApoCharmmErrorCode::InvalidArgument,
      "Reference pressure tensor must contain exactly 9 elements; observed " +
          std::to_string(referencePressureTensor.size()));

  for (std::size_t i = 0; i < referencePressureTensor.size(); i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(referencePressureTensor[i]),
        ApoCharmmErrorCode::InvalidArgument,
        "Reference pressure tensor values must be finite; non-finite value at "
        "index " +
            std::to_string(i));
  }

  m_ReferencePressureTensor = referencePressureTensor;

  return;
}

void CudaLangevinPistonIntegrator::setConstantSurfaceTension(
    const bool constantSurfaceTensionFlag) {
  this->requireCurrentState();
  m_ConstantSurfaceTensionFlag = constantSurfaceTensionFlag;
  return;
}

void CudaLangevinPistonIntegrator::setLangevinPistonMass(
    const std::vector<double> &mass) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(m_LangevinPistonDegreesOfFreedom > 0,
                    ApoCharmmErrorCode::NotInitialized,
                    "Attach a CharmmContext with a crystal type before setting "
                    "the Langevin piston mass");

  APOCHARMM_REQUIRE(static_cast<int>(mass.size()) ==
                        m_LangevinPistonDegreesOfFreedom,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Langevin piston mass must contain " +
                        std::to_string(m_LangevinPistonDegreesOfFreedom) +
                        " element(s); observed " + std::to_string(mass.size()));

  for (std::size_t i = 0; i < mass.size(); i++) {
    APOCHARMM_REQUIRE(std::isfinite(mass[i]),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Langevin piston mass values must be finite; non-finite "
                      "value at index " +
                          std::to_string(i));

    APOCHARMM_REQUIRE(
        mass[i] >= 0.0, ApoCharmmErrorCode::InvalidArgument,
        "Langevin piston mass values must be non-negative; observed " +
            std::to_string(mass[i]) + " at index " + std::to_string(i));
  }

  for (int i = 0; i < m_LangevinPistonDegreesOfFreedom; i++) {
    m_LangevinPistonMass[i] = mass[i];
    if (mass[i] == 0.0)
      m_LangevinPistonInverseMass[i] = 0.0;
    else
      m_LangevinPistonInverseMass[i] = 1.0 / mass[i];
  }
  m_LangevinPistonMass.transferToDevice();
  m_LangevinPistonInverseMass.transferToDevice();

  // If the Langevin piston mass changes, the friction variables need to be
  // updated
  double oldGamma = m_Pgamma;
  this->setLangevinPistonFriction(oldGamma);

  return;
}

void CudaLangevinPistonIntegrator::setLangevinPistonFrictionSeed(
    const std::uint64_t seed) {
  this->requireCurrentState();
  m_Seed = seed;
  this->initializeRng();
  return;
}

void CudaLangevinPistonIntegrator::setRngSequencePos(
    const unsigned long long int rngSequencePos) {
  this->requireCurrentState();
  m_RngSequencePos = rngSequencePos;
  this->initializeRng();
  return;
}

void CudaLangevinPistonIntegrator::setRngStates(
    const std::string &rngStateString) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(
      m_LangevinPistonDegreesOfFreedom > 0, ApoCharmmErrorCode::NotInitialized,
      "Attach a CharmmContext with a crystal type before setting RNG states");

  unsigned long long int pos = 0;
  std::vector<curandStatePhilox4_32_10_t> states;
  apo::curand_states_from_string(pos, states, rngStateString);

  APOCHARMM_REQUIRE(states.size() == static_cast<std::size_t>(
                                         m_LangevinPistonDegreesOfFreedom),
                    ApoCharmmErrorCode::InvalidArgument,
                    "RNG state count must match Langevin piston degrees of "
                    "freedom; expected " +
                        std::to_string(m_LangevinPistonDegreesOfFreedom) +
                        ", observed " + std::to_string(states.size()));

  this->setRngStateData(pos, states);

  return;
}

void CudaLangevinPistonIntegrator::setLangevinPistonFriction(
    const double pgamma) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(m_LangevinPistonDegreesOfFreedom > 0,
                    ApoCharmmErrorCode::NotInitialized,
                    "Attach a CharmmContext with a crystal type before setting "
                    "the Langevin piston friction");

  APOCHARMM_REQUIRE(std::isfinite(pgamma), ApoCharmmErrorCode::InvalidArgument,
                    "Langevin piston friction must be finite; observed " +
                        std::to_string(pgamma));

  APOCHARMM_REQUIRE(pgamma >= 0.0, ApoCharmmErrorCode::InvalidArgument,
                    "Langevin piston friction must be non-negative; observed " +
                        std::to_string(pgamma));

  const double kbt = charmm::constants::kBoltz * m_ReferenceTemperature;
  const double pgam = charmm::constants::timfac * m_TimeStep * pgamma;

  m_Pgamma = pgamma;
  m_Palpha = (1.0 - 0.5 * pgam) / (1.0 + 0.5 * pgam);
  m_Pbfact = m_TimeStep * m_TimeStep / (1.0 + 0.5 * pgam);
  for (int i = 0; i < m_LangevinPistonDegreesOfFreedom; i++) {
    m_Prfwd[i] = std::sqrt(2.0 * m_LangevinPistonInverseMass[i] * pgam * kbt) /
                 m_TimeStep;
  }
  m_Prfwd.transferToDevice();

  return;
}

void CudaLangevinPistonIntegrator::resetAverages(void) {
  this->requireCurrentState();
  m_AverageWindowSize = 0;
  m_AverageTemperature.setToValue(0.0);
  m_AveragePressureTensor.setToValue(0.0);
  m_AveragePressureScalar.setToValue(0.0);
  return;
}

double CudaLangevinPistonIntegrator::getReferenceTemperature(void) const {
  this->requireCurrentState();
  return m_ReferenceTemperature;
}

bool CudaLangevinPistonIntegrator::usingOldTemperature(void) const {
  this->requireCurrentState();
  return m_UsingOldTemperature;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonMass(void) const {
  this->requireCurrentState();
  return m_NoseHooverPistonMass;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocity(void) const {
  this->requireCurrentState();
  return m_NoseHooverPistonVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocityPrevious(void) const {
  this->requireCurrentState();
  return m_NoseHooverPistonVelocityPrevious;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForce(void) const {
  this->requireCurrentState();
  return m_NoseHooverPistonForce;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForcePrevious(void) const {
  this->requireCurrentState();
  return m_NoseHooverPistonForcePrevious;
}

int CudaLangevinPistonIntegrator::getMaxPredictorCorrectorIterations(
    void) const {
  this->requireCurrentState();
  return m_MaxPredictorCorrectorIterations;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getKineticEnergy(void) const {
  this->requireCurrentState();
  return m_KineticEnergy;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getAverageTemperature(void) const {
  this->requireCurrentState();
  return m_AverageTemperature;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getReferencePressureTensor(void) const {
  this->requireCurrentState();
  return m_ReferencePressureTensor;
}

CRYSTAL CudaLangevinPistonIntegrator::getCrystalType(void) const {
  return (m_Context == nullptr) ? CRYSTAL::UNSET : m_Context->getCrystalType();
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonMass(void) const {
  this->requireCurrentState();
  return m_LangevinPistonMass;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepPosition(void) const {
  this->requireCurrentState();
  return m_LangevinPistonOnStepPosition;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepPosition(void) const {
  this->requireCurrentState();
  return m_LangevinPistonHalfStepPosition;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepVelocity(void) const {
  this->requireCurrentState();
  return m_LangevinPistonOnStepVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepVelocity(void) const {
  this->requireCurrentState();
  return m_LangevinPistonHalfStepVelocity;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPosition(void) const {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPosition;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPrevious(
    void) const {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPositionPrevious;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPredicted(
    void) const {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPositionPredicted;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPressure(void) const {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPressure;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureTensor(void) const {
  this->requireCurrentState();
  return m_PressureTensor;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureScalar(void) const {
  this->requireCurrentState();
  return m_PressureScalar;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureTensor(void) const {
  this->requireCurrentState();
  return m_AveragePressureTensor;
}

const CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureScalar(void) const {
  this->requireCurrentState();
  return m_AveragePressureScalar;
}

std::uint64_t
CudaLangevinPistonIntegrator::getLangevinPistonFrictionSeed(void) const {
  this->requireCurrentState();
  return m_Seed;
}

unsigned long long int
CudaLangevinPistonIntegrator::getRngSequencePos(void) const {
  this->requireCurrentState();
  return m_RngSequencePos;
}

std::string CudaLangevinPistonIntegrator::getRngStates(void) const {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(m_LangevinPistonDegreesOfFreedom > 0,
                    ApoCharmmErrorCode::NotInitialized,
                    "Attach a CharmmContext with a crystal type before "
                    "retrieving RNG states");

  APOCHARMM_REQUIRE(m_RngStates != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "RNG states are not initialized");

  std::vector<curandStatePhilox4_32_10_t> rngStates(
      m_LangevinPistonDegreesOfFreedom);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  cudaCheck(cudaMemcpy(static_cast<void *>(rngStates.data()),
                       static_cast<const void *>(m_RngStates),
                       rngStates.size() * sizeof(curandStatePhilox4_32_10_t),
                       cudaMemcpyDeviceToHost));

  return apo::curand_states_to_string(m_RngSequencePos, rngStates);
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonMass(void) {
  this->requireCurrentState();
  return m_NoseHooverPistonMass;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocity(void) {
  this->requireCurrentState();
  return m_NoseHooverPistonVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonVelocityPrevious(void) {
  this->requireCurrentState();
  return m_NoseHooverPistonVelocityPrevious;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForce(void) {
  this->requireCurrentState();
  return m_NoseHooverPistonForce;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getNoseHooverPistonForcePrevious(void) {
  this->requireCurrentState();
  return m_NoseHooverPistonForcePrevious;
}

CudaContainer<double> &CudaLangevinPistonIntegrator::getKineticEnergy(void) {
  this->requireCurrentState();
  return m_KineticEnergy;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getAverageTemperature(void) {
  this->requireCurrentState();
  return m_AverageTemperature;
}

double CudaLangevinPistonIntegrator::getInstantaneousTemperature(void) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(
      m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before computing instantaneous temperature");

  const double ndegf = static_cast<double>(m_Context->getNumDegreesOfFreedom());
  m_KineticEnergy.transferToHost();

  if (m_UsingOldTemperature)
    return (m_KineticEnergy[1] / (0.5 * ndegf * charmm::constants::kBoltz));

  return (m_KineticEnergy[0] / (0.5 * ndegf * charmm::constants::kBoltz));
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getReferencePressureTensor(void) {
  this->requireCurrentState();
  return m_ReferencePressureTensor;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonMass(void) {
  this->requireCurrentState();
  return m_LangevinPistonMass;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepPosition(void) {
  this->requireCurrentState();
  return m_LangevinPistonOnStepPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepPosition(void) {
  this->requireCurrentState();
  return m_LangevinPistonHalfStepPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonOnStepVelocity(void) {
  this->requireCurrentState();
  return m_LangevinPistonOnStepVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonHalfStepVelocity(void) {
  this->requireCurrentState();
  return m_LangevinPistonHalfStepVelocity;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPosition(void) {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPosition;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPrevious(void) {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPositionPrevious;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPositionPredicted(void) {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPositionPredicted;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureTensor(void) {
  this->requireCurrentState();
  return m_PressureTensor;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getInstantaneousPressureScalar(void) {
  this->requireCurrentState();
  return m_PressureScalar;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getLangevinPistonDeltaPressure(void) {
  this->requireCurrentState();
  return m_LangevinPistonDeltaPressure;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureTensor(void) {
  this->requireCurrentState();
  return m_AveragePressureTensor;
}

CudaContainer<double> &
CudaLangevinPistonIntegrator::getAveragePressureScalar(void) {
  this->requireCurrentState();
  return m_AveragePressureScalar;
}

double CudaLangevinPistonIntegrator::getInstantaneousSurfaceTension(void) {
  this->requireCurrentState();

  APOCHARMM_REQUIRE(
      m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before computing instantaneous surface "
      "tension");

  m_PressureTensor.transferToHost();

  return (0.5 * m_Context->getBoxDimensions()[2] *
          (m_PressureTensor[8] -
           0.5 * (m_PressureTensor[0] + m_PressureTensor[4])) /
          charmm::constants::surfaceTensionFactor);
}

/** @brief Updates the single-precision coordinates container (xyzq)
 */
__global__ static void UpdateSinglePrecisionCoordinatesKernel(
    float4 *__restrict__ xyzq, const double4 *__restrict__ coordsCharges,
    const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    xyzq[i].x = static_cast<float>(coordsCharges[i].x);
    xyzq[i].y = static_cast<float>(coordsCharges[i].y);
    xyzq[i].z = static_cast<float>(coordsCharges[i].z);
  }

  return;
}

__global__ static void
InitializationKernel(double4 *__restrict__ coordsDelta,
                     double4 *__restrict__ coordsDeltaPrevious,
                     const double4 *__restrict__ velMass, const int numAtoms,
                     const double *__restrict__ forces, const int forceStride,
                     const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    const double fx = forces[0 * forceStride + i];
    const double fy = forces[1 * forceStride + i];
    const double fz = forces[2 * forceStride + i];
    const double fact = 0.5 * timeStep2 * velMass[i].w;

    coordsDelta[i].x = timeStep * velMass[i].x - fact * fx;
    coordsDelta[i].y = timeStep * velMass[i].y - fact * fy;
    coordsDelta[i].z = timeStep * velMass[i].z - fact * fz;

    coordsDeltaPrevious[i].x = timeStep * velMass[i].x + fact * fx;
    coordsDeltaPrevious[i].y = timeStep * velMass[i].y + fact * fy;
    coordsDeltaPrevious[i].z = timeStep * velMass[i].z + fact * fz;
  }

  return;
}

__global__ static void
BackStepInitializationKernel1(double4 *__restrict__ coordsCharges,
                              const double4 *__restrict__ coordsDeltaPrevious,
                              const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsCharges[i].x -= coordsDeltaPrevious[i].x;
    coordsCharges[i].y -= coordsDeltaPrevious[i].y;
    coordsCharges[i].z -= coordsDeltaPrevious[i].z;
  }

  return;
}

__global__ static void
BackStepInitializationKernel2(double4 *__restrict__ coordsCharges,
                              double4 *__restrict__ coordsDeltaPrevious,
                              const double4 *__restrict__ coordsRef,
                              const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsDeltaPrevious[i].x = coordsRef[i].x - coordsCharges[i].x;
    coordsDeltaPrevious[i].y = coordsRef[i].y - coordsCharges[i].y;
    coordsDeltaPrevious[i].z = coordsRef[i].z - coordsCharges[i].z;

    coordsCharges[i].x = coordsRef[i].x;
    coordsCharges[i].y = coordsRef[i].y;
    coordsCharges[i].z = coordsRef[i].z;
  }

  return;
}

void CudaLangevinPistonIntegrator::initializeImpl(void) {
  const bool computeDefaultNoseHooverMass =
      m_NoseHooverPistonMass[0] == -9999.9999;
  const int previousPistonDof = m_LangevinPistonDegreesOfFreedom;

  try {
    const int requiredPistonDof = this->getContextPistonDegreesOfFreedom();
    if (previousPistonDof != requiredPistonDof) {
      // Publish the derived size before calling guarded helpers, then allocate
      // every crystal-sized array before any helper can use those arrays.
      m_LangevinPistonDegreesOfFreedom = requiredPistonDof;

      this->allocateLangevinPistonVariables();

      m_NoseHooverPistonVelocity.setToValue(0.0);
      m_NoseHooverPistonVelocityPrevious.setToValue(0.0);
      m_NoseHooverPistonForce.setToValue(0.0);
      m_NoseHooverPistonForcePrevious.setToValue(0.0);

      m_KineticEnergy.setToValue(0.0);
      m_PressureTensor.setToValue(0.0);
      m_PressureScalar.setToValue(0.0);

      this->resetAverages();
    }

    const int numAtoms = m_Context->getNumAtoms();
    constexpr int numThreads = 256;
    const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

    this->initializeRng();

    m_CoordsDeltaPredicted.resize(numAtoms);
    m_KineticEnergyPartialSums.resize(numBlocks * 2);
    m_PressureReductionPartialSums.resize(numBlocks * 9);

    m_HolonomicConstraintForces.resize(numAtoms);

    if (computeDefaultNoseHooverMass)
      m_NoseHooverPistonMass.setToValue(this->computeNoseHooverPistonMass());

    if (m_LangevinPistonMass[0] == -9999.9999) {
      double m = this->computeLangevinPistonMass();
      std::vector<double> mass(m_LangevinPistonDegreesOfFreedom, m);
      this->setLangevinPistonMass(mass);
    }

    double4 *coordsCharges =
        m_Context->getCoordinatesChargesDP().getDeviceArray().data();
    float4 *xyzq = m_Context->getCoordinatesChargesSP().getDeviceArray().data();

    if (m_UsingHolonomicConstraints) {
      copy_DtoD_async<double4>(coordsCharges,
                               m_CoordsRef.getDeviceArray().data(), numAtoms,
                               *m_IntegratorStream);

      m_HolonomicConstraint->handleHolonomicConstraints(
          m_CoordsRef.getDeviceArray().data());

      cudaCheckLaunch(
          UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                                   *m_IntegratorStream>>>(
              xyzq, coordsCharges, numAtoms));

      copy_DtoD_async<double4>(coordsCharges,
                               m_CoordsRef.getDeviceArray().data(), numAtoms,
                               *m_IntegratorStream);

      cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
    }

    m_Context->calculateForces();

    double4 *velMass =
        m_Context->getVelocitiesInverseMasses().getDeviceArray().data();
    double *forces = m_Context->getForces()->xyz();
    const int forceStride = m_Context->getForceStride();

    cudaCheckLaunch(
        InitializationKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
            m_CoordsDelta.getDeviceArray().data(),
            m_CoordsDeltaPrevious.getDeviceArray().data(), velMass, numAtoms,
            forces, forceStride, m_TimeStep));

    if (m_UsingHolonomicConstraints) {
      cudaCheckLaunch(BackStepInitializationKernel1<<<numBlocks, numThreads, 0,
                                                      *m_IntegratorStream>>>(
          coordsCharges, m_CoordsDeltaPrevious.getDeviceArray().data(),
          numAtoms));

      m_HolonomicConstraint->handleHolonomicConstraints(
          m_CoordsRef.getDeviceArray().data());

      cudaCheckLaunch(BackStepInitializationKernel2<<<numBlocks, numThreads, 0,
                                                      *m_IntegratorStream>>>(
          coordsCharges, m_CoordsDeltaPrevious.getDeviceArray().data(),
          m_CoordsRef.getDeviceArray().data(), numAtoms));
    }

    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  } catch (...) {
    // A partially rebuilt layout must not be observable; the next initialize()
    // then reallocates all crystal-sized state. A failed reinitialization of
    // the current layout keeps that layout and its configuration.
    if (m_LangevinPistonDegreesOfFreedom != previousPistonDof)
      m_LangevinPistonDegreesOfFreedom = -1;

    if (computeDefaultNoseHooverMass)
      m_NoseHooverPistonMass[0] = -9999.9999;

    throw;
  }

  return;
}

void CudaLangevinPistonIntegrator::initializeFromRestartFileImpl(
    const std::filesystem::path &rstFilePath) {
  // CudaIntegrator::initializeFromRestartFile() has already run
  // requireCurrentState(): the context box is valid and
  // m_LangevinPistonDegreesOfFreedom matches the context crystal.
  const int numAtoms = m_Context->getNumAtoms();
  const apo::RestartSnapshot snapshot =
      apo::ReadRestartFile(rstFilePath, numAtoms);

  APOCHARMM_REQUIRE(
      snapshot.crystal == m_Context->getCrystalType(),
      ApoCharmmErrorCode::InvalidArgument,
      "Restart crystal type does not match the configured CharmmContext");

  // JEG261102: We should support this one day and just reset the RNG, but for
  // now we will throw an error.
  APOCHARMM_REQUIRE(snapshot.nativeKind.empty() ||
                        snapshot.nativeKind == "LANGEVIN_PISTON",
                    ApoCharmmErrorCode::InvalidArgument,
                    "Restart was written by a different apoCHARMM integrator");

  // CHARMM recomputes NDEGF from its own setup after READYN, and its
  // conventions differ from this context's (Langevin dynamics, for example,
  // keeps the three center-of-mass degrees of freedom). Require equality only
  // for files written by apoCHARMM.
  APOCHARMM_REQUIRE(snapshot.nativeKind.empty() ||
                        snapshot.degreesOfFreedom ==
                            m_Context->getNumDegreesOfFreedom(),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Restart NDEGF does not match the configured context");

  // HDOT holds one compact displacement per piston degree of freedom; CHARMM
  // zeroes the remaining entries.
  for (int i = m_LangevinPistonDegreesOfFreedom; i < 6; i++) {
    APOCHARMM_REQUIRE(
        snapshot.hdot[i] == 0.0, ApoCharmmErrorCode::NotImplemented,
        "Restart contains unsupported cell-displacement components");
  }

  // Native files may retain dormant thermostat state after the caller has
  // disabled the thermostat. Restore that state without changing the setting.
  // JEG261102: I think we should probably allow this. I can't imagine a case
  // where a user would want to do this, but I don't think it should be
  // disallowed.
  APOCHARMM_REQUIRE(!snapshot.nativeKind.empty() ||
                        m_UsingNoseHooverThermostat ||
                        (snapshot.pnhv == 0.0 && snapshot.pnhf == 0.0),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Restart contains Nose-Hoover thermostat state but the "
                    "configured Langevin piston has its thermostat disabled");

  unsigned long long int restartRngPosition = 0;
  std::vector<curandStatePhilox4_32_10_t> restartRngStates;
  if (snapshot.hasNativeRng) {
    apo::curand_states_from_string(restartRngPosition, restartRngStates,
                                   snapshot.rngState);
    APOCHARMM_REQUIRE(
        restartRngStates.size() ==
            static_cast<std::size_t>(m_LangevinPistonDegreesOfFreedom),
        ApoCharmmErrorCode::InvalidArgument,
        "Restart RNG state count must match the piston degrees of freedom");
  }

  CudaContainer<double4> &coordinates = m_Context->getCoordinatesChargesDP();
  CudaContainer<double4> &velocities = m_Context->getVelocitiesInverseMasses();
  coordinates.transferToHost();
  velocities.transferToHost();
  m_CoordsDeltaPrevious.transferToHost();
  for (int i = 0; i < numAtoms; i++) {
    coordinates[i].x = snapshot.coordinates[i][0];
    coordinates[i].y = snapshot.coordinates[i][1];
    coordinates[i].z = snapshot.coordinates[i][2];
    velocities[i].x = snapshot.velocities[i][0];
    velocities[i].y = snapshot.velocities[i][1];
    velocities[i].z = snapshot.velocities[i][2];
    m_CoordsDeltaPrevious[i].x = snapshot.deltas[i][0];
    m_CoordsDeltaPrevious[i].y = snapshot.deltas[i][1];
    m_CoordsDeltaPrevious[i].z = snapshot.deltas[i][2];
  }

  m_Context->setBoxDimensions(
      {snapshot.box[0], snapshot.box[1], snapshot.box[2]});

  this->resetAverages();

  m_TotNumSteps = snapshot.completedSteps;
  m_NumSteps = snapshot.requestedSteps;
  m_StepsSinceNeighborListUpdate = 0;
  if (snapshot.hasNativeRng) {
    m_Seed = snapshot.rngSeed;
    this->setRngStateData(restartRngPosition, restartRngStates);
  } else {
    // CHARMM uses another generator. Keep the caller-configured apoCHARMM seed
    // and start a new stream instead of interpreting CHARMM's state.
    m_RngSequencePos = 0;
    this->initializeRng();
  }

  coordinates.transferToDevice();
  velocities.transferToDevice();
  m_CoordsDeltaPrevious.transferToDevice();
  m_CoordsDelta = m_CoordsDeltaPrevious;

  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;
  cudaCheckLaunch(
      UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                               *m_IntegratorStream>>>(
          m_Context->getCoordinatesChargesSP().getDeviceArray().data(),
          coordinates.getDeviceArray().data(), numAtoms));
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  if (snapshot.nativeKind.empty() && snapshot.heatingSteps == 0) {
    // CHARMM resumes a restart written with JHSTRT=0 through its two-step start
    // (IGVOPT=2): the previous displacement is regenerated from the velocities
    // and forces rather than taken from the !X, Y, Z block. Native apoCHARMM
    // files carry JHSTRT=1 once propagation has occurred. The neighbor list
    // still describes the pre-restart coordinates and calculateForces() does
    // not rebuild it, so rebuild it first.
    m_Context->resetNeighborList();
    this->initialize();
  }

  // The file's piston state is installed after any regeneration.
  for (int i = 0; i < m_LangevinPistonDegreesOfFreedom; i++)
    m_LangevinPistonDeltaPosition[i] = snapshot.hdot[i];
  m_LangevinPistonDeltaPosition.transferToDevice();
  this->setNoseHooverPistonVelocity(snapshot.pnhv);
  this->setNoseHooverPistonForce(snapshot.pnhf);

  return;
}

__global__ static void InvertDeltaAsymmetricKernel(
    double4 *__restrict__ coordsDeltaPrevious, const float4 *__restrict__ xyzq,
    const int2 *__restrict__ groups, const int numGroups, const float boxDimX) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numGroups; i += stride) {
    const int2 group = groups[i];

    float gx = 0.0f; //, gy = 0.0f, gz = 0.0f;
    for (int j = group.x; j <= group.y; j++) {
      gx += xyzq[j].x;
      // gy += xyzq[j].y;
      // gz += xyzq[j].z;
    }
    gx /= static_cast<float>(group.y - group.x + 1);
    // gy /= static_cast<float>(group.y - group.x + 1);
    // gz /= static_cast<float>(group.y - group.x + 1);

    if ((gx > 0.5 * boxDimX) || (gx < -0.5 * boxDimX)) {
      for (int j = group.x; j <= group.y; j++) {
        coordsDeltaPrevious[j].y *= -1.0;
        coordsDeltaPrevious[j].z *= -1.0;
      }
    }
  }

  return;
}

__global__ static void
HalfStepVelocityKernel(double4 *__restrict__ coordsCharges,
                       double4 *__restrict__ coordsDelta,
                       const double4 *__restrict__ coordsDeltaPrevious,
                       const double4 *__restrict__ velMass, const int numAtoms,
                       const double *__restrict__ forces, const int forceStride,
                       const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    const double fx = forces[0 * forceStride + i];
    const double fy = forces[1 * forceStride + i];
    const double fz = forces[2 * forceStride + i];
    const double fact = timeStep2 * velMass[i].w;

    coordsDelta[i].x = coordsDeltaPrevious[i].x - fact * fx;
    coordsDelta[i].y = coordsDeltaPrevious[i].y - fact * fy;
    coordsDelta[i].z = coordsDeltaPrevious[i].z - fact * fz;

    coordsCharges[i].x += coordsDelta[i].x;
    coordsCharges[i].y += coordsDelta[i].y;
    coordsCharges[i].z += coordsDelta[i].z;
  }

  return;
}

__global__ static void ComputeHolonomicConstraintForcesKernel(
    double4 *__restrict__ holonomicConstraintForces,
    const double4 *__restrict__ coordsCharges,
    const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ velMass, const int numAtoms,
    const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    const double fact = 1.0 / (timeStep2 * velMass[i].w);

    holonomicConstraintForces[i].x =
        fact * (coordsCharges[i].x - coordsRef[i].x - coordsDelta[i].x);
    holonomicConstraintForces[i].y =
        fact * (coordsCharges[i].y - coordsRef[i].y - coordsDelta[i].y);
    holonomicConstraintForces[i].z =
        fact * (coordsCharges[i].z - coordsRef[i].z - coordsDelta[i].z);
  }

  return;
}

/**
 * @brief Reduces component-major block partial sums into one tensor.
 *
 * Thread `t` visits partial-sum indices congruent to `t` modulo `BlockSize`
 * in increasing index order. Each iteration loads up to two thread-strided
 * records. The component values are then reduced together across one block.
 * Thread zero overwrites every output component.
 *
 * The first-pass layout is component-major:
 * `tensorPartialSums[component * numPartialSums + partialIndex]`.
 *
 * @tparam blockSize Compile-time number of threads in the CUDA block. It must
 * satisfy the requirements of `BlockReduceSumStorage`.
 * @tparam numComponents Positive compile-time tensor component count.
 * @param[out] tensor Non-NULL device array containing at least
 * `NumComponents` elements. Every element is overwritten.
 * @param[in] tensorPartialSums Non-NULL component-major device array
 * containing `NumComponents * numPartialSums` elements.
 * @param[in] numPartialSums Positive first-pass block count.
 *
 * @pre The kernel is launched with exactly one one-dimensional block
 * satisfying `blockDim.x == BlockSize`.
 * @pre Every block thread reaches the block-wide reduction.
 */
template <int blockSize, int numComponents>
__global__ static void
FinalizeTensorPartialSumsKernel(double *__restrict__ tensor,
                                const double *__restrict__ tensorPartialSums,
                                const int numPartialSums) {
  static_assert(numComponents > 0, "numComponents must be greater than zero");

  double values[numComponents] = {};

  for (int i = threadIdx.x; i < numPartialSums; i += 2 * blockSize) {
#pragma unroll
    for (int j = 0; j < numComponents; j++)
      values[j] += tensorPartialSums[j * numPartialSums + i];

    const int j = i + blockSize;
    if (j < numPartialSums) {
#pragma unroll
      for (int k = 0; k < numComponents; k++)
        values[k] += tensorPartialSums[k * numPartialSums + j];
    }
  }

  __shared__ BlockReduceSumStorage<double, blockSize, numComponents> cache;

  BlockReduceSum<double, blockSize, numComponents>(values, cache);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < numComponents; i++)
      tensor[i] = values[i];
  }

  return;
}

/**
 * @brief Computes one holonomic-constraint virial partial tensor per block.
 *
 * Each thread accumulates its atom-strided contribution to the row-major
 * nine-component tensor. The nine values are reduced together across the
 * block. Thread zero writes one component-major partial tensor without
 * atomics.
 *
 * @tparam blockSize Compile-time number of threads in each CUDA block. It must
 * satisfy the requirements of `BlockReduceSumStorage`.
 * @param[out] holonomicConstraintVirialPartialSums Non-NULL device array
 * containing at least `9 * gridDim.x` elements.
 * @param[in] coordsRef Non-NULL device array of `numAtoms` reference
 * coordinates.
 * @param[in] holonomicConstraintForces Non-NULL device array of `numAtoms`
 * constraint-force records.
 * @param[in] numAtoms Positive atom count.
 *
 * @pre The kernel is launched with one-dimensional blocks satisfying
 * `blockDim.x == BlockSize`.
 * @pre Every block thread reaches the block-wide reduction.
 */
template <int blockSize>
__global__ static void ComputeHolonomicConstraintVirialPartialSumsKernel(
    double *__restrict__ holonomicConstraintVirialPartialSums,
    const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ holonomicConstraintForces, const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  double vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = idx; i < numAtoms; i += stride) {
    vir[0] += coordsRef[i].x * holonomicConstraintForces[i].x;
    vir[1] += coordsRef[i].x * holonomicConstraintForces[i].y;
    vir[2] += coordsRef[i].x * holonomicConstraintForces[i].z;
    vir[3] += coordsRef[i].y * holonomicConstraintForces[i].x;
    vir[4] += coordsRef[i].y * holonomicConstraintForces[i].y;
    vir[5] += coordsRef[i].y * holonomicConstraintForces[i].z;
    vir[6] += coordsRef[i].z * holonomicConstraintForces[i].x;
    vir[7] += coordsRef[i].z * holonomicConstraintForces[i].y;
    vir[8] += coordsRef[i].z * holonomicConstraintForces[i].z;
  }

  __shared__ BlockReduceSumStorage<double, blockSize, 9> cache;

  BlockReduceSum<double, blockSize, 9>(vir, cache);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++)
      holonomicConstraintVirialPartialSums[i * gridDim.x + blockIdx.x] = vir[i];
  }

  return;
}

__global__ static void UpdateCoordsDeltaAfterHolonomicConstraintKernel(
    double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsCharges,
    const double4 *__restrict__ coordsRef, const int numAtoms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsDelta[i].x = coordsCharges[i].x - coordsRef[i].x;
    coordsDelta[i].y = coordsCharges[i].y - coordsRef[i].y;
    coordsDelta[i].z = coordsCharges[i].z - coordsRef[i].z;
  }

  return;
}
__global__ static void UpdateVirialWithHolonomicConstraintVirialKernel(
    double *__restrict__ virialTensor,
    const double *__restrict__ holonomicConstraintVirial) {
  if (threadIdx.x < 9)
    virialTensor[threadIdx.x] += holonomicConstraintVirial[threadIdx.x];
  return;
}

/**
 * @brief Computes one average kinetic-pressure partial tensor per block.
 *
 * The kernel uses the previous and next half-step displacements rather than
 * the on-step velocity. Each thread accumulates a row-major nine-component
 * tensor over its atom-strided input. The components are reduced together
 * across the block, and thread zero writes one component-major partial tensor.
 *
 * @tparam blockSize Compile-time number of threads in each CUDA block. It must
 * satisfy the requirements of `BlockReduceSumStorage`.
 * @param[out] kineticPressurePartialSums Non-NULL device array containing at
 * least `9 * gridDim.x` elements.
 * @param[in] velMass Non-NULL device array of `numAtoms`
 * `[vx, vy, vz, inverse_mass]` records.
 * @param[in] coordsDelta Non-NULL device array of `numAtoms` next-half-step
 * displacement records.
 * @param[in] coordsDeltaPrevious Non-NULL device array of `numAtoms`
 * previous-half-step displacement records.
 * @param[in] numAtoms Positive atom count.
 * @param[in] timeStep Positive time step in native integration units.
 *
 * @pre The kernel is launched with one-dimensional blocks satisfying
 * `blockDim.x == BlockSize`.
 * @pre Every block thread reaches the block-wide reduction.
 *
 * @note Computes the kinetic energy contribution to the pressure tensor using
 * previous and next half step velocities. One might think the on-step velocity
 * would be a better thing to use, but it would be unsound (Brooks 1987).
 */
template <int blockSize>
__global__ static void ComputeAverageKineticPressurePartialSumsKernel(
    double *__restrict__ kineticPressureTensorPartialSums,
    const double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsDeltaPrevious, const int numAtoms,
    const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  double prs[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = idx; i < numAtoms; i += stride) {
    const double fact = 0.5 / (timeStep2 * velMass[i].w);
    prs[0] += fact * (coordsDelta[i].x * coordsDelta[i].x +
                      coordsDeltaPrevious[i].x * coordsDeltaPrevious[i].x);
    prs[1] += fact * (coordsDelta[i].x * coordsDelta[i].y +
                      coordsDeltaPrevious[i].x * coordsDeltaPrevious[i].y);
    prs[2] += fact * (coordsDelta[i].x * coordsDelta[i].z +
                      coordsDeltaPrevious[i].x * coordsDeltaPrevious[i].z);
    prs[3] += fact * (coordsDelta[i].y * coordsDelta[i].x +
                      coordsDeltaPrevious[i].y * coordsDeltaPrevious[i].x);
    prs[4] += fact * (coordsDelta[i].y * coordsDelta[i].y +
                      coordsDeltaPrevious[i].y * coordsDeltaPrevious[i].y);
    prs[5] += fact * (coordsDelta[i].y * coordsDelta[i].z +
                      coordsDeltaPrevious[i].y * coordsDeltaPrevious[i].z);
    prs[6] += fact * (coordsDelta[i].z * coordsDelta[i].x +
                      coordsDeltaPrevious[i].z * coordsDeltaPrevious[i].x);
    prs[7] += fact * (coordsDelta[i].z * coordsDelta[i].y +
                      coordsDeltaPrevious[i].z * coordsDeltaPrevious[i].y);
    prs[8] += fact * (coordsDelta[i].z * coordsDelta[i].z +
                      coordsDeltaPrevious[i].z * coordsDeltaPrevious[i].z);
  }

  __shared__ BlockReduceSumStorage<double, blockSize, 9> cache;

  BlockReduceSum<double, blockSize, 9>(prs, cache);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++)
      kineticPressureTensorPartialSums[i * gridDim.x + blockIdx.x] = prs[i];
  }

  return;
}

__global__ static void UpdatePressureKernel(
    double *__restrict__ pressureTensor, double *__restrict__ pressureScalar,
    double *__restrict__ referencePressureTensor,
    double *__restrict__ deltaPressureTensor,
    const double *__restrict__ virialTensor,
    const double *__restrict__ kineticPressureTensor, const double volumeFactor,
    const double *__restrict__ surfaceTension,
    const double surfaceTensionFactor, const bool constantSurfaceTensionFlag) {
  if (threadIdx.x < 9) {
    pressureTensor[threadIdx.x] =
        volumeFactor *
        (kineticPressureTensor[threadIdx.x] + virialTensor[threadIdx.x]);

    deltaPressureTensor[threadIdx.x] =
        pressureTensor[threadIdx.x] - referencePressureTensor[threadIdx.x];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    *pressureScalar =
        (pressureTensor[0] + pressureTensor[4] + pressureTensor[8]) / 3.0;

    if (constantSurfaceTensionFlag == true) {
      referencePressureTensor[0] =
          referencePressureTensor[8] - surfaceTension[0] * surfaceTensionFactor;
      referencePressureTensor[4] = referencePressureTensor[0];
    }
  }

  return;
}

/**
 * @brief Computes one delta kinetic-pressure partial tensor per block.
 *
 * This is the next-half-step kinetic contribution updated during the
 * predictor-corrector loop. The previous-half-step contribution and virial
 * contribution are treated as invariant during that loop.
 *
 * Each thread accumulates a row-major nine-component tensor over its
 * atom-strided input. The components are reduced together across the block,
 * and thread zero writes one component-major partial tensor without atomics.
 *
 * @tparam blockSize Compile-time number of threads in each CUDA block. It must
 * satisfy the requirements of `BlockReduceSumStorage`.
 * @param[out] deltaKineticPressurePartialSums Non-NULL device array containing
 * at least `9 * gridDim.x` elements.
 * @param[in] velMass Non-NULL device array of `numAtoms`
 * `[vx, vy, vz, inverse_mass]` records.
 * @param[in] coordsDelta Non-NULL device array of `numAtoms` next-half-step
 * displacement records.
 * @param[in] numAtoms Positive atom count.
 * @param[in] volumeFactor Pressure conversion and volume factor.
 * @param[in] timeStep Positive time step in native integration units.
 *
 * @pre The kernel is launched with one-dimensional blocks satisfying
 * `blockDim.x == BlockSize`.
 * @pre Every block thread reaches the block-wide reduction.
 */
template <int blockSize>
__global__ static void ComputeDeltaKineticPressurePartialSumsKernel(
    double *__restrict__ deltaKineticPressureTensorPartialSums,
    const double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsDelta, const int numAtoms,
    const double volumeFactor, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;

  double prs[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = idx; i < numAtoms; i += stride) {
    const double fact = 0.5 * volumeFactor / (timeStep2 * velMass[i].w);
    prs[0] += fact * coordsDelta[i].x * coordsDelta[i].x;
    prs[1] += fact * coordsDelta[i].x * coordsDelta[i].y;
    prs[2] += fact * coordsDelta[i].x * coordsDelta[i].z;
    prs[3] += fact * coordsDelta[i].y * coordsDelta[i].x;
    prs[4] += fact * coordsDelta[i].y * coordsDelta[i].y;
    prs[5] += fact * coordsDelta[i].y * coordsDelta[i].z;
    prs[6] += fact * coordsDelta[i].z * coordsDelta[i].x;
    prs[7] += fact * coordsDelta[i].z * coordsDelta[i].y;
    prs[8] += fact * coordsDelta[i].z * coordsDelta[i].z;
  }

  __shared__ BlockReduceSumStorage<double, blockSize, 9> cache;

  BlockReduceSum<double, blockSize, 9>(prs, cache);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) {
      deltaKineticPressureTensorPartialSums[i * gridDim.x + blockIdx.x] =
          prs[i];
    }
  }

  return;
}

__global__ static void ComputeStaticDeltaPressureKernel(
    double *__restrict__ staticDeltaPressureTensor,
    const double *__restrict__ deltaPressureTensor,
    const double *__restrict__ deltaKineticPressureTensor) {
  if (threadIdx.x < 9) {
    staticDeltaPressureTensor[threadIdx.x] =
        deltaPressureTensor[threadIdx.x] -
        deltaKineticPressureTensor[threadIdx.x];
  }
  return;
}

__global__ static void UpdateLangevinPistonKernel(
    double *__restrict__ langevinPistonOnStepPosition,
    double *__restrict__ langevinPistonHalfStepPosition,
    double *__restrict__ langevinPistonDeltaPosition,
    double *__restrict__ langevinPistonDeltaPositionPrevious,
    double *__restrict__ langevinPistonOnStepVelocity,
    double *__restrict__ langevinPistonDeltaPressure,
    double *__restrict__ onStepCrystalFactor,
    double *__restrict__ halfStepCrystalFactor,
    double *__restrict__ boxDimensions,
    curandStatePhilox4_32_10_t *__restrict__ rngStates,
    const double *__restrict__ langevinPistonInverseMass,
    const double *__restrict__ deltaPressureTensor,
    const double *__restrict__ prfwd, const double palpha, const double pbfact,
    const double volumeFactor, const int dof, const CRYSTAL crystalType,
    const double timeStep) {
  if (threadIdx.x == 0) {
    switch (crystalType) {
    case CRYSTAL::CUBIC:
      langevinPistonDeltaPressure[0] =
          (deltaPressureTensor[0] + deltaPressureTensor[4] +
           deltaPressureTensor[8]) /
          boxDimensions[0];
      break;

    case CRYSTAL::TETRAGONAL:
      langevinPistonDeltaPressure[0] =
          (deltaPressureTensor[0] + deltaPressureTensor[4]) / boxDimensions[0];
      langevinPistonDeltaPressure[1] =
          deltaPressureTensor[8] / boxDimensions[2];
      break;

    case CRYSTAL::ORTHORHOMBIC:
      langevinPistonDeltaPressure[0] =
          deltaPressureTensor[0] / boxDimensions[0];
      langevinPistonDeltaPressure[1] =
          deltaPressureTensor[4] / boxDimensions[1];
      langevinPistonDeltaPressure[2] =
          deltaPressureTensor[8] / boxDimensions[2];
      break;

    default:
      break;
    }

    for (int i = 0; i < dof; i++) {
      curandStatePhilox4_32_10_t rngState = rngStates[i];

      langevinPistonDeltaPositionPrevious[i] = langevinPistonDeltaPosition[i];

      langevinPistonDeltaPosition[i] =
          palpha * langevinPistonDeltaPosition[i] +
          langevinPistonInverseMass[i] * langevinPistonDeltaPressure[i] *
              volumeFactor +
          pbfact * prfwd[i] * curand_normal_double(&rngState);

      langevinPistonOnStepVelocity[i] =
          0.5 *
          (langevinPistonDeltaPosition[i] +
           langevinPistonDeltaPositionPrevious[i]) /
          timeStep;

      rngStates[i] = rngState;
    }

    switch (crystalType) {
    case CRYSTAL::CUBIC:
      langevinPistonOnStepPosition[0] =
          (boxDimensions[0] + boxDimensions[1] + boxDimensions[2]) / 3.0;
      break;

    case CRYSTAL::TETRAGONAL:
      langevinPistonOnStepPosition[0] =
          (boxDimensions[0] + boxDimensions[1]) / 2.0;
      langevinPistonOnStepPosition[1] = boxDimensions[2];
      break;

    case CRYSTAL::ORTHORHOMBIC:
      langevinPistonOnStepPosition[0] = boxDimensions[0];
      langevinPistonOnStepPosition[1] = boxDimensions[1];
      langevinPistonOnStepPosition[2] = boxDimensions[2];
      break;

    default:
      break;
    }

    for (int i = 0; i < dof; i++) {
      langevinPistonOnStepPosition[i] += langevinPistonDeltaPosition[i];

      langevinPistonHalfStepPosition[i] = langevinPistonOnStepPosition[i] -
                                          0.5 * langevinPistonDeltaPosition[i];
    }

    switch (crystalType) {
    case CRYSTAL::CUBIC:
      onStepCrystalFactor[0] =
          timeStep * langevinPistonOnStepVelocity[0] / boxDimensions[0];
      onStepCrystalFactor[1] = onStepCrystalFactor[0];
      onStepCrystalFactor[2] = onStepCrystalFactor[0];

      halfStepCrystalFactor[0] =
          langevinPistonDeltaPosition[0] / langevinPistonHalfStepPosition[0];
      halfStepCrystalFactor[1] = halfStepCrystalFactor[0];
      halfStepCrystalFactor[2] = halfStepCrystalFactor[0];

      boxDimensions[0] = langevinPistonOnStepPosition[0];
      boxDimensions[1] = langevinPistonOnStepPosition[0];
      boxDimensions[2] = langevinPistonOnStepPosition[0];
      break;

    case CRYSTAL::TETRAGONAL:
      onStepCrystalFactor[0] =
          timeStep * langevinPistonOnStepVelocity[0] / boxDimensions[0];
      onStepCrystalFactor[1] = onStepCrystalFactor[0];
      onStepCrystalFactor[2] =
          timeStep * langevinPistonOnStepVelocity[1] / boxDimensions[2];

      halfStepCrystalFactor[0] =
          langevinPistonDeltaPosition[0] / langevinPistonHalfStepPosition[0];
      halfStepCrystalFactor[1] = halfStepCrystalFactor[0];
      halfStepCrystalFactor[2] =
          langevinPistonDeltaPosition[1] / langevinPistonHalfStepPosition[1];

      boxDimensions[0] = langevinPistonOnStepPosition[0];
      boxDimensions[1] = langevinPistonOnStepPosition[0];
      boxDimensions[2] = langevinPistonOnStepPosition[1];
      break;

    case CRYSTAL::ORTHORHOMBIC:
      onStepCrystalFactor[0] =
          timeStep * langevinPistonOnStepVelocity[0] / boxDimensions[0];
      onStepCrystalFactor[1] =
          timeStep * langevinPistonOnStepVelocity[1] / boxDimensions[1];
      onStepCrystalFactor[2] =
          timeStep * langevinPistonOnStepVelocity[2] / boxDimensions[2];

      halfStepCrystalFactor[0] =
          langevinPistonDeltaPosition[0] / langevinPistonHalfStepPosition[0];
      halfStepCrystalFactor[1] =
          langevinPistonDeltaPosition[1] / langevinPistonHalfStepPosition[1];
      halfStepCrystalFactor[2] =
          langevinPistonDeltaPosition[2] / langevinPistonHalfStepPosition[2];

      boxDimensions[0] = langevinPistonOnStepPosition[0];
      boxDimensions[1] = langevinPistonOnStepPosition[1];
      boxDimensions[2] = langevinPistonOnStepPosition[2];
      break;

    default:
      break;
    }
  }

  return;
}

/**
 * @brief Computes two kinetic-energy partial sums per CUDA block.
 *
 * Element zero is the three-point kinetic-energy estimator and element one is
 * the on-step estimator. Both values are reduced together across the block.
 *
 * @tparam blockSize Compile-time number of threads in each CUDA block. It must
 * satisfy the requirements of `BlockReduceSumStorage`.
 * @param[out] kineticEnergyPartialSums Non-NULL block-major device array with
 * at least `2 * gridDim.x` elements.
 * @param[in] velMass Non-NULL device array of `numAtoms`
 * `[vx, vy, vz, inverse_mass]` records.
 * @param[in] coordsDelta Non-NULL next-half-step displacement array.
 * @param[in] coordsDeltaPrevious Non-NULL previous-half-step displacement
 * array.
 * @param[in] numAtoms Positive atom count.
 * @param[in] timeStep Positive time step in native integration units.
 *
 * @pre The kernel is launched with one-dimensional blocks satisfying
 * `blockDim.x == BlockSize`.
 * @pre Every block thread reaches the block-wide reduction.
 */
template <int blockSize>
__global__ static void ComputeKineticEnergyPartialSumsKernel(
    double *__restrict__ kineticEnergyPartialSums,
    const double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsDeltaPrevious, const int numAtoms,
    const double timeStep) {
  constexpr double oneThird = 1.0 / 3.0;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double invTimeStep = 1.0 / timeStep;

  double ke[2] = {0.0, 0.0};
  for (int i = index; i < numAtoms; i += stride) {
    const double4 u1 =
        make_double4(coordsDeltaPrevious[i].x * invTimeStep,
                     coordsDeltaPrevious[i].y * invTimeStep,
                     coordsDeltaPrevious[i].z * invTimeStep, 0.0);
    const double4 v = velMass[i];
    const double4 u2 = make_double4(coordsDelta[i].x * invTimeStep,
                                    coordsDelta[i].y * invTimeStep,
                                    coordsDelta[i].z * invTimeStep, 0.0);

    const double oldHalfStepKineticEnergy =
        0.5 * (u1.x * u1.x + u1.y * u1.y + u1.z * u1.z) / v.w;
    const double onStepKineticEnergy =
        0.5 * (v.x * v.x + v.y * v.y + v.z * v.z) / v.w;
    const double newHalfStepKineticEnergy =
        0.5 * (u2.x * u2.x + u2.y * u2.y + u2.z * u2.z) / v.w;

    ke[0] += oneThird * (oldHalfStepKineticEnergy + onStepKineticEnergy +
                         newHalfStepKineticEnergy);
    ke[1] += onStepKineticEnergy;
  }

  __shared__ BlockReduceSumStorage<double, blockSize, 2> cache;

  BlockReduceSum<double, blockSize, 2>(ke, cache);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 2; i++)
      kineticEnergyPartialSums[blockIdx.x * 2 + i] = ke[i];
  }

  return;
}

/**
 * @brief Finalizes both kinetic-energy estimators and updates Nose-Hoover
 * state.
 *
 * Thread `t` visits partial-sum records congruent to `t` modulo `BlockSize` in
 * increasing record order. Each loop iteration loads up to two
 * thread-strided records. Both estimators are reduced together across the
 * block before thread zero updates the output and extended state.
 *
 * @tparam blockSize Compile-time number of threads in the CUDA block. It must
 * satisfy the requirements of `BlockReduceSumStorage`.
 * @param[out] kineticEnergy Non-NULL two-element output array.
 * @param[out] noseHooverPistonForce Non-NULL current velocity-increment scalar.
 * @param[in,out] noseHooverPistonForcePrevious Non-NULL previous increment.
 * @param[out] noseHooverPistonVelocity Non-NULL current piston velocity.
 * @param[in] noseHooverPistonVelocityPrevious Non-NULL previous velocity.
 * @param[in] noseHooverPistonMass Non-NULL coupling-mass scalar.
 * @param[in] kineticEnergyPartialSums Non-NULL block-major array containing
 * two values per first-pass block.
 * @param[in] numPartialSums Positive first-pass block count.
 * @param[in] referenceKineticEnergy Reference kinetic energy.
 * @param[in] usingOldTemperature Whether estimator element one is selected.
 * @param[in] timeStep Positive time step in native integration units.
 *
 * @pre The kernel is launched with exactly one one-dimensional block
 * satisfying `blockDim.x == BlockSize`.
 * @pre Every block thread reaches the block-wide reduction.
 */
template <int blockSize>
__global__ static void UpdateKineticEnergyAndNoseHooverPistonKernel(
    double *__restrict__ kineticEnergy,
    double *__restrict__ noseHooverPistonForce,
    double *__restrict__ noseHooverPistonForcePrevious,
    double *__restrict__ noseHooverPistonVelocity,
    const double *__restrict__ noseHooverPistonVelocityPrevious,
    const double *__restrict__ noseHooverPistonMass,
    const double *__restrict__ kineticEnergyPartialSums,
    const int numPartialSums, const double referenceKineticEnergy,
    const bool usingOldTemperature, const double timeStep) {
  double ke[2] = {0.0, 0.0};
  for (int i = threadIdx.x; i < numPartialSums; i += 2 * blockDim.x) {
    ke[0] += kineticEnergyPartialSums[i * 2 + 0];
    ke[1] += kineticEnergyPartialSums[i * 2 + 1];

    const int j = i + blockSize;
    if (j < numPartialSums) {
      ke[0] += kineticEnergyPartialSums[j * 2 + 0];
      ke[1] += kineticEnergyPartialSums[j * 2 + 1];
    }
  }

  __shared__ BlockReduceSumStorage<double, blockSize, 2> cache;

  BlockReduceSum<double, blockSize, 2>(ke, cache);

  if (threadIdx.x == 0) {
    kineticEnergy[0] = ke[0];
    kineticEnergy[1] = ke[1];

    const int i = (usingOldTemperature) ? 1 : 0;

    // Actually store the change in velocity (not the force)
    *noseHooverPistonForce = 2.0 * timeStep *
                             (kineticEnergy[i] - referenceKineticEnergy) /
                             *noseHooverPistonMass;

    if (*noseHooverPistonForcePrevious == 0.0)
      *noseHooverPistonForcePrevious = *noseHooverPistonForce;

    *noseHooverPistonVelocity =
        *noseHooverPistonVelocityPrevious +
        0.5 * (*noseHooverPistonForce + *noseHooverPistonForcePrevious);
  }

  return;
}

__global__ static void PredictorCorrectorKernel(
    double4 *__restrict__ coordsCharges, double4 *__restrict__ velMass,
    double4 *__restrict__ coordsDeltaPredicted,
    const double4 *__restrict__ coordsDelta,
    const double4 *__restrict__ coordsDeltaPrevious,
    const double4 *__restrict__ coordsRef, const int numAtoms,
    const double *__restrict__ noseHooverPistonVelocity,
    const bool usingNoseHooverThermostat,
    const double *__restrict__ onStepCrystalFactor,
    const double *__restrict__ halfStepCrystalFactor, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 / timeStep;

  const double lpFactX = timeStep * onStepCrystalFactor[0];
  const double lpFactY = timeStep * onStepCrystalFactor[1];
  const double lpFactZ = timeStep * onStepCrystalFactor[2];
  double nhFact = 0.0;
  if (usingNoseHooverThermostat)
    nhFact = timeStep * timeStep * noseHooverPistonVelocity[0];

  for (int i = idx; i < numAtoms; i += stride) {
    const double onStepVelocityX =
        fact * (coordsDeltaPrevious[i].x + coordsDeltaPredicted[i].x);
    const double onStepVelocityY =
        fact * (coordsDeltaPrevious[i].y + coordsDeltaPredicted[i].y);
    const double onStepVelocityZ =
        fact * (coordsDeltaPrevious[i].z + coordsDeltaPredicted[i].z);

    // v(t+ dt/2)*dt = [v(t-dt/2)*dt + f(t)*dt^2/m] - h.(t)/h(t)*v(t)*dt^2
    coordsDeltaPredicted[i].x =
        coordsDelta[i].x - (nhFact + lpFactX) * onStepVelocityX;
    coordsDeltaPredicted[i].y =
        coordsDelta[i].y - (nhFact + lpFactY) * onStepVelocityY;
    coordsDeltaPredicted[i].z =
        coordsDelta[i].z - (nhFact + lpFactZ) * onStepVelocityZ;

    velMass[i].x = onStepVelocityX;
    velMass[i].y = onStepVelocityY;
    velMass[i].z = onStepVelocityZ;

    const double halfStepPositionX =
        0.5 * (coordsRef[i].x + coordsCharges[i].x);
    const double halfStepPositionY =
        0.5 * (coordsRef[i].y + coordsCharges[i].y);
    const double halfStepPositionZ =
        0.5 * (coordsRef[i].z + coordsCharges[i].z);

    const double scaledCrystalVelocityX =
        halfStepCrystalFactor[0] * halfStepPositionX;
    const double scaledCrystalVelocityY =
        halfStepCrystalFactor[1] * halfStepPositionY;
    const double scaledCrystalVelocityZ =
        halfStepCrystalFactor[2] * halfStepPositionZ;

    coordsCharges[i].x =
        coordsRef[i].x + coordsDeltaPredicted[i].x + scaledCrystalVelocityX;
    coordsCharges[i].y =
        coordsRef[i].y + coordsDeltaPredicted[i].y + scaledCrystalVelocityY;
    coordsCharges[i].z =
        coordsRef[i].z + coordsDeltaPredicted[i].z + scaledCrystalVelocityZ;
  }

  return;
}

__global__ static void UpdateDeltaPressureKernel(
    double *__restrict__ deltaPressureTensor,
    const double *__restrict__ staticDeltaPressureTensor,
    const double *__restrict__ deltaKineticPressureTensor) {
  if (threadIdx.x < 9) {
    deltaPressureTensor[threadIdx.x] = staticDeltaPressureTensor[threadIdx.x] +
                                       deltaKineticPressureTensor[threadIdx.x];
  }
  return;
}

__global__ static void ApplyBarostatToReferenceCoordsKernel(
    double4 *__restrict__ coordsRef, const double4 *__restrict__ coordsCharges,
    const int numAtoms, const double *__restrict__ halfStepCrystalFactor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    const double halfStepPositionX =
        0.5 * (coordsRef[i].x + coordsCharges[i].x);
    const double halfStepPositionY =
        0.5 * (coordsRef[i].y + coordsCharges[i].y);
    const double halfStepPositionZ =
        0.5 * (coordsRef[i].z + coordsCharges[i].z);

    const double scaledCrystalVelocityX =
        halfStepCrystalFactor[0] * halfStepPositionX;
    const double scaledCrystalVelocityY =
        halfStepCrystalFactor[1] * halfStepPositionY;
    const double scaledCrystalVelocityZ =
        halfStepCrystalFactor[2] * halfStepPositionZ;

    coordsRef[i].x += scaledCrystalVelocityX;
    coordsRef[i].y += scaledCrystalVelocityY;
    coordsRef[i].z += scaledCrystalVelocityZ;
  }

  return;
}

__global__ static void
OnStepVelocityKernel(double4 *__restrict__ velMass,
                     const double4 *__restrict__ coordsDelta,
                     const double4 *__restrict__ coordsDeltaPrevious,
                     const int numAtoms, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 / timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    velMass[i].x = fact * (coordsDelta[i].x + coordsDeltaPrevious[i].x);
    velMass[i].y = fact * (coordsDelta[i].y + coordsDeltaPrevious[i].y);
    velMass[i].z = fact * (coordsDelta[i].z + coordsDeltaPrevious[i].z);
  }

  return;
}

__global__ static void
UpdateAverageTemperatureKernel(double *__restrict__ averageTemperature,
                               const double *__restrict__ kineticEnergy,
                               const int numDegreesOfFreedom,
                               const double kBoltz, const int step) {
  if (threadIdx.x < 2) {
    const double s = static_cast<double>(step + 1);
    const double ndegf = static_cast<double>(numDegreesOfFreedom);
    const double temperature =
        kineticEnergy[threadIdx.x] / (0.5 * ndegf * kBoltz);
    const double delta0 = temperature - averageTemperature[threadIdx.x];
    averageTemperature[threadIdx.x] += delta0 / s;
    // const double delta1 = temperature - averageTemperature[threadIdx.x];
    // varianceTemperature[threadIdx.x] += delta0 * delta1;
  }
  return;
}

__global__ static void
UpdateAveragePressureKernel(double *__restrict__ pressureTensor,
                            double *__restrict__ pressureScalar,
                            double *__restrict__ averagePressureTensor,
                            double *__restrict__ averagePressureScalar,
                            const double *__restrict__ virialTensor,
                            const double *__restrict__ kineticPressureTensor,
                            const double volumeFactor, const int step) {
  const double s = static_cast<double>(step + 1);

  if (threadIdx.x < 9) {
    pressureTensor[threadIdx.x] =
        volumeFactor *
        (kineticPressureTensor[threadIdx.x] + virialTensor[threadIdx.x]);
    const double delta0 =
        pressureTensor[threadIdx.x] - averagePressureTensor[threadIdx.x];
    averagePressureTensor[threadIdx.x] += delta0 / s;
    // const double delta1 =
    //     pressureTensor[threadIdx.x] - averagePressureTensor[threadIdx.x];
    // variancePressureTensor[threadIdx.x] += delta0 * delta1;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    *pressureScalar =
        (pressureTensor[0] + pressureTensor[4] + pressureTensor[8]) / 3.0;
    const double delta0 = *pressureScalar - *averagePressureScalar;
    *averagePressureScalar += delta0 / s;
    // const double delta1 = *pressureScalar - *averagePressureScalar;
    // *variancePressureScalar += delta0 * delta1;
  }

  return;
}

void CudaLangevinPistonIntegrator::propagateOneStepImpl(void) {
  const int numDegreesOfFreedom = m_Context->getNumDegreesOfFreedom();
  const double referenceKineticEnergy =
      0.5 * static_cast<double>(numDegreesOfFreedom) *
      charmm::constants::kBoltz * m_ReferenceTemperature;

  const int numAtoms = m_Context->getNumAtoms();
  double4 *coordsCharges =
      m_Context->getCoordinatesChargesDP().getDeviceArray().data();
  float4 *xyzq = m_Context->getCoordinatesChargesSP().getDeviceArray().data();
  double4 *velMass =
      m_Context->getVelocitiesInverseMasses().getDeviceArray().data();
  const int forceStride = m_Context->getForceStride();
  double *forces = m_Context->getForces()->xyz();

  CudaContainer<double> boxDimensions = m_Context->getBoxDimensions();
  const double volume = m_Context->getVolume();

  if (m_StepsSinceNeighborListUpdate % m_NonbondedListUpdateFrequency == 0) {
    if (m_Context->getForceManager()->getPeriodicBoundaryCondition() ==
        PBC::P21) {
      // Find a better place for this
      int numGroups =
          m_Context->getForceManager()->getPsf()->getGroups().size();
      int2 *groups = m_Context->getForceManager()
                         ->getPsf()
                         ->getGroups()
                         .getDeviceArray()
                         .data();

      constexpr int numThreads = 256;
      const int numBlocks = (numGroups + numThreads - 1) / numThreads;
      cudaCheckLaunch(InvertDeltaAsymmetricKernel<<<numBlocks, numThreads, 0,
                                                    *m_IntegratorStream>>>(
          m_CoordsDeltaPrevious.getDeviceArray().data(), xyzq, groups,
          numGroups, static_cast<float>(boxDimensions[0])));
      cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
    }
    m_Context->resetNeighborList();
  }

  if (m_TotNumSteps %
          static_cast<unsigned long long int>(m_RemoveCenterOfMassFrequency) ==
      0ULL) {
    this->removeCenterOfMassMotion();
  }

  copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceArray().data(),
                           numAtoms, *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  m_Context->calculateForces(false, true, true);

  copy_DtoD_async<double>(
      m_NoseHooverPistonVelocity.getDeviceArray().data(),
      m_NoseHooverPistonVelocityPrevious.getDeviceArray().data(), 1,
      *m_IntegratorStream);

  copy_DtoD_async<double>(
      m_NoseHooverPistonForce.getDeviceArray().data(),
      m_NoseHooverPistonForcePrevious.getDeviceArray().data(), 1,
      *m_IntegratorStream);

  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  cudaCheckLaunch(
      HalfStepVelocityKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          coordsCharges, m_CoordsDelta.getDeviceArray().data(),
          m_CoordsDeltaPrevious.getDeviceArray().data(), velMass, numAtoms,
          forces, forceStride, m_TimeStep));

  // JEG260114: This needs to be called after calculateForces becuase getVirial
  // actually updates the virial in the forceManager
  double *virialTensor = m_Context->getVirial().getDeviceArray().data();

  if (m_UsingHolonomicConstraints) {
    // JEG260107: Need sync here, otherwise race condition? I don't get it
    // either...
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceArray().data());

    cudaCheckLaunch(ComputeHolonomicConstraintForcesKernel<<<
                        numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        m_HolonomicConstraintForces.getDeviceArray().data(), coordsCharges,
        m_CoordsRef.getDeviceArray().data(),
        m_CoordsDelta.getDeviceArray().data(), velMass, numAtoms, m_TimeStep));

    cudaCheckLaunch(
        ComputeHolonomicConstraintVirialPartialSumsKernel<numThreads>
        <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
            m_PressureReductionPartialSums.getDeviceArray().data(),
            m_CoordsRef.getDeviceArray().data(),
            m_HolonomicConstraintForces.getDeviceArray().data(), numAtoms));

    cudaCheckLaunch(
        FinalizeTensorPartialSumsKernel<numThreads, 9>
        <<<1, numThreads, 0, *m_IntegratorStream>>>(
            m_HolonomicConstraintVirial.getDeviceArray().data(),
            m_PressureReductionPartialSums.getDeviceArray().data(), numBlocks));

    cudaCheckLaunch(UpdateCoordsDeltaAfterHolonomicConstraintKernel<<<
                        numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        m_CoordsDelta.getDeviceArray().data(), coordsCharges,
        m_CoordsRef.getDeviceArray().data(), numAtoms));

    cudaCheckLaunch(UpdateVirialWithHolonomicConstraintVirialKernel<<<
                        1, 32, 0, *m_IntegratorStream>>>(
        virialTensor, m_HolonomicConstraintVirial.getDeviceArray().data()));
  }

  cudaCheckLaunch(
      ComputeAverageKineticPressurePartialSumsKernel<numThreads>
      <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          m_PressureReductionPartialSums.getDeviceArray().data(), velMass,
          m_CoordsDelta.getDeviceArray().data(),
          m_CoordsDeltaPrevious.getDeviceArray().data(), numAtoms, m_TimeStep));

  cudaCheckLaunch(
      FinalizeTensorPartialSumsKernel<numThreads, 9>
      <<<1, numThreads, 0, *m_IntegratorStream>>>(
          m_KineticPressureTensor.getDeviceArray().data(),
          m_PressureReductionPartialSums.getDeviceArray().data(), numBlocks));

  cudaCheckLaunch(UpdatePressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
      m_PressureTensor.getDeviceArray().data(),
      m_PressureScalar.getDeviceArray().data(),
      m_ReferencePressureTensor.getDeviceArray().data(),
      m_DeltaPressureTensor.getDeviceArray().data(), virialTensor,
      m_KineticPressureTensor.getDeviceArray().data(),
      charmm::constants::patmos / volume,
      m_SurfaceTension.getDeviceArray().data(),
      charmm::constants::surfaceTensionFactor / boxDimensions[2],
      m_ConstantSurfaceTensionFlag));

  cudaCheckLaunch(ComputeDeltaKineticPressurePartialSumsKernel<numThreads>
                  <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
                      m_PressureReductionPartialSums.getDeviceArray().data(),
                      velMass, m_CoordsDelta.getDeviceArray().data(), numAtoms,
                      0.5 * charmm::constants::patmos / volume, m_TimeStep));

  cudaCheckLaunch(
      FinalizeTensorPartialSumsKernel<numThreads, 9>
      <<<1, numThreads, 0, *m_IntegratorStream>>>(
          m_DeltaKineticPressureTensor.getDeviceArray().data(),
          m_PressureReductionPartialSums.getDeviceArray().data(), numBlocks));

  cudaCheckLaunch(
      ComputeStaticDeltaPressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
          m_StaticDeltaPressureTensor.getDeviceArray().data(),
          m_DeltaPressureTensor.getDeviceArray().data(),
          m_DeltaKineticPressureTensor.getDeviceArray().data()));

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceArray().data(),
                           m_CoordsDeltaPredicted.getDeviceArray().data(),
                           numAtoms, *m_IntegratorStream);

  CudaContainer<double> boxDimensionsPredicted = boxDimensions;

  for (int iter = 0; iter < m_MaxPredictorCorrectorIterations; iter++) {
    copy_DtoD_async<double>(boxDimensions.getDeviceArray().data(),
                            boxDimensionsPredicted.getDeviceArray().data(), 3,
                            *m_IntegratorStream);

    copy_DtoD_async<double>(
        m_LangevinPistonDeltaPosition.getDeviceArray().data(),
        m_LangevinPistonDeltaPositionPredicted.getDeviceArray().data(),
        m_LangevinPistonDegreesOfFreedom, *m_IntegratorStream);

    cudaCheckLaunch(
        UpdateLangevinPistonKernel<<<1, 32, 0, *m_IntegratorStream>>>(
            m_LangevinPistonOnStepPosition.getDeviceArray().data(),
            m_LangevinPistonHalfStepPosition.getDeviceArray().data(),
            m_LangevinPistonDeltaPositionPredicted.getDeviceArray().data(),
            m_LangevinPistonDeltaPositionPrevious.getDeviceArray().data(),
            m_LangevinPistonOnStepVelocity.getDeviceArray().data(),
            m_LangevinPistonDeltaPressure.getDeviceArray().data(),
            m_OnStepCrystalFactor.getDeviceArray().data(),
            m_HalfStepCrystalFactor.getDeviceArray().data(),
            boxDimensionsPredicted.getDeviceArray().data(), m_RngStates,
            m_LangevinPistonInverseMass.getDeviceArray().data(),
            m_DeltaPressureTensor.getDeviceArray().data(),
            m_Prfwd.getDeviceArray().data(), m_Palpha, m_Pbfact,
            volume * m_Pbfact * charmm::constants::atmosp,
            m_LangevinPistonDegreesOfFreedom, m_Context->getCrystalType(),
            m_TimeStep));

    m_RngSequencePos++; // curand_normal_double iterates 1 step

    cudaCheckLaunch(ComputeKineticEnergyPartialSumsKernel<numThreads>
                    <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
                        m_KineticEnergyPartialSums.getDeviceArray().data(),
                        velMass, m_CoordsDelta.getDeviceArray().data(),
                        m_CoordsDeltaPrevious.getDeviceArray().data(), numAtoms,
                        m_TimeStep));

    cudaCheckLaunch(
        UpdateKineticEnergyAndNoseHooverPistonKernel<numThreads>
        <<<1, numThreads, 0, *m_IntegratorStream>>>(
            m_KineticEnergy.getDeviceArray().data(),
            m_NoseHooverPistonForce.getDeviceArray().data(),
            m_NoseHooverPistonForcePrevious.getDeviceArray().data(),
            m_NoseHooverPistonVelocity.getDeviceArray().data(),
            m_NoseHooverPistonVelocityPrevious.getDeviceArray().data(),
            m_NoseHooverPistonMass.getDeviceArray().data(),
            m_KineticEnergyPartialSums.getDeviceArray().data(), numBlocks,
            referenceKineticEnergy, m_UsingOldTemperature, m_TimeStep));

    cudaCheckLaunch(PredictorCorrectorKernel<<<numBlocks, numThreads, 0,
                                               *m_IntegratorStream>>>(
        coordsCharges, velMass, m_CoordsDeltaPredicted.getDeviceArray().data(),
        m_CoordsDelta.getDeviceArray().data(),
        m_CoordsDeltaPrevious.getDeviceArray().data(),
        m_CoordsRef.getDeviceArray().data(), numAtoms,
        m_NoseHooverPistonVelocity.getDeviceArray().data(),
        m_UsingNoseHooverThermostat,
        m_OnStepCrystalFactor.getDeviceArray().data(),
        m_HalfStepCrystalFactor.getDeviceArray().data(), m_TimeStep));

    cudaCheckLaunch(
        ComputeDeltaKineticPressurePartialSumsKernel<numThreads>
        <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
            m_PressureReductionPartialSums.getDeviceArray().data(), velMass,
            m_CoordsDeltaPredicted.getDeviceArray().data(), numAtoms,
            0.5 * charmm::constants::patmos / volume, m_TimeStep));

    cudaCheckLaunch(
        FinalizeTensorPartialSumsKernel<numThreads, 9>
        <<<1, numThreads, 0, *m_IntegratorStream>>>(
            m_DeltaKineticPressureTensor.getDeviceArray().data(),
            m_PressureReductionPartialSums.getDeviceArray().data(), numBlocks));

    cudaCheckLaunch(
        UpdateDeltaPressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
            m_DeltaPressureTensor.getDeviceArray().data(),
            m_StaticDeltaPressureTensor.getDeviceArray().data(),
            m_DeltaKineticPressureTensor.getDeviceArray().data()));
  }

  copy_DtoD_async<double>(boxDimensionsPredicted.getDeviceArray().data(),
                          boxDimensions.getDeviceArray().data(), 3,
                          *m_IntegratorStream);

  copy_DtoD_async<double>(
      m_LangevinPistonDeltaPositionPredicted.getDeviceArray().data(),
      m_LangevinPistonDeltaPosition.getDeviceArray().data(),
      m_LangevinPistonDegreesOfFreedom, *m_IntegratorStream);

  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceArray().data());

    cudaCheckLaunch(
        ApplyBarostatToReferenceCoordsKernel<<<numBlocks, numThreads, 0,
                                               *m_IntegratorStream>>>(
            m_CoordsRef.getDeviceArray().data(), coordsCharges, numAtoms,
            m_HalfStepCrystalFactor.getDeviceArray().data()));

    double4 *tmp = m_CoordsRef.getDeviceArray().data();
    m_Context->getCoordinatesChargesDP().getDeviceArray().assignData(tmp);

    m_HolonomicConstraint->handleHolonomicConstraints(coordsCharges);

    m_CoordsRef.getDeviceArray().assignData(tmp);
    m_Context->getCoordinatesChargesDP().getDeviceArray().assignData(
        coordsCharges);

    cudaCheckLaunch(UpdateCoordsDeltaAfterHolonomicConstraintKernel<<<
                        numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        m_CoordsDeltaPredicted.getDeviceArray().data(), coordsCharges,
        m_CoordsRef.getDeviceArray().data(), numAtoms));
  }

  copy_DtoD_async<double4>(m_CoordsDeltaPredicted.getDeviceArray().data(),
                           m_CoordsDelta.getDeviceArray().data(), numAtoms,
                           *m_IntegratorStream);

  cudaCheckLaunch(
      OnStepVelocityKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          velMass, m_CoordsDelta.getDeviceArray().data(),
          m_CoordsDeltaPrevious.getDeviceArray().data(), numAtoms, m_TimeStep));

  cudaCheckLaunch(
      ComputeKineticEnergyPartialSumsKernel<numThreads>
      <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          m_KineticEnergyPartialSums.getDeviceArray().data(), velMass,
          m_CoordsDelta.getDeviceArray().data(),
          m_CoordsDeltaPrevious.getDeviceArray().data(), numAtoms, m_TimeStep));

  cudaCheckLaunch(
      UpdateKineticEnergyAndNoseHooverPistonKernel<numThreads>
      <<<1, numThreads, 0, *m_IntegratorStream>>>(
          m_KineticEnergy.getDeviceArray().data(),
          m_NoseHooverPistonForce.getDeviceArray().data(),
          m_NoseHooverPistonForcePrevious.getDeviceArray().data(),
          m_NoseHooverPistonVelocity.getDeviceArray().data(),
          m_NoseHooverPistonVelocityPrevious.getDeviceArray().data(),
          m_NoseHooverPistonMass.getDeviceArray().data(),
          m_KineticEnergyPartialSums.getDeviceArray().data(), numBlocks,
          referenceKineticEnergy, m_UsingOldTemperature, m_TimeStep));

  cudaCheckLaunch(
      UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
          m_AverageTemperature.getDeviceArray().data(),
          m_KineticEnergy.getDeviceArray().data(), numDegreesOfFreedom,
          charmm::constants::kBoltz, m_AverageWindowSize));

  cudaCheckLaunch(
      ComputeAverageKineticPressurePartialSumsKernel<numThreads>
      <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          m_PressureReductionPartialSums.getDeviceArray().data(), velMass,
          m_CoordsDelta.getDeviceArray().data(),
          m_CoordsDeltaPrevious.getDeviceArray().data(), numAtoms, m_TimeStep));

  cudaCheckLaunch(
      FinalizeTensorPartialSumsKernel<numThreads, 9>
      <<<1, numThreads, 0, *m_IntegratorStream>>>(
          m_KineticPressureTensor.getDeviceArray().data(),
          m_PressureReductionPartialSums.getDeviceArray().data(), numBlocks));

  cudaCheckLaunch(
      UpdateAveragePressureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
          m_PressureTensor.getDeviceArray().data(),
          m_PressureScalar.getDeviceArray().data(),
          m_AveragePressureTensor.getDeviceArray().data(),
          m_AveragePressureScalar.getDeviceArray().data(), virialTensor,
          m_KineticPressureTensor.getDeviceArray().data(),
          charmm::constants::patmos / volume, m_AverageWindowSize));

  m_AverageWindowSize++;

  cudaCheckLaunch(
      UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                               *m_IntegratorStream>>>(
          xyzq, coordsCharges, numAtoms));

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceArray().data(),
                           m_CoordsDeltaPrevious.getDeviceArray().data(),
                           numAtoms, *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  boxDimensions.transferToHost();
  m_Context->setBoxDimensions(boxDimensions.getHostArray());

  // DO HFCTE CALCULATION HERE ONLY IN DEBUG MODE

  return;
}

void CudaLangevinPistonIntegrator::setRngStateData(
    const unsigned long long int position,
    const std::vector<curandStatePhilox4_32_10_t> &states) {
  m_RngSequencePos = position;
  this->alloc(m_LangevinPistonDegreesOfFreedom);

  cudaCheck(cudaMemcpy(static_cast<void *>(m_RngStates),
                       static_cast<const void *>(states.data()),
                       states.size() * sizeof(curandStatePhilox4_32_10_t),
                       cudaMemcpyHostToDevice));

  return;
}

double CudaLangevinPistonIntegrator::computeNoseHooverPistonMass(void) {
  CudaContainer<double4> velMass = m_Context->getVelocitiesInverseMasses();
  velMass.transferToHost();

  double totalMass = 0.0;
  for (std::size_t i = 0; i < velMass.size(); i++)
    totalMass += (1.0 / velMass[i].w);

  return (0.2 * totalMass);
}

double CudaLangevinPistonIntegrator::computeLangevinPistonMass(void) {
  CudaContainer<double4> velMass = m_Context->getVelocitiesInverseMasses();
  velMass.transferToHost();

  double totalMass = 0.0;
  for (std::size_t i = 0; i < velMass.size(); i++)
    totalMass += (1.0 / velMass[i].w);

  return (0.02 * totalMass);
}

void CudaLangevinPistonIntegrator::allocateLangevinPistonVariables(void) {
  // Allocate memory for Langevin piston variables
  m_LangevinPistonMass.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonInverseMass.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonOnStepPosition.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonHalfStepPosition.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonOnStepVelocity.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonHalfStepVelocity.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPosition.resize(m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPositionPrevious.resize(
      m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPositionPredicted.resize(
      m_LangevinPistonDegreesOfFreedom);
  m_LangevinPistonDeltaPressure.resize(m_LangevinPistonDegreesOfFreedom);
  m_Prfwd.resize(m_LangevinPistonDegreesOfFreedom);

  // Set pressure piston variables to default values
  m_LangevinPistonMass.setToValue(-9999.9999);
  m_LangevinPistonInverseMass.setToValue(-9999.9999);
  m_LangevinPistonOnStepPosition.setToValue(0.0);
  m_LangevinPistonHalfStepPosition.setToValue(0.0);
  m_LangevinPistonOnStepVelocity.setToValue(0.0);
  m_LangevinPistonHalfStepVelocity.setToValue(0.0);
  m_LangevinPistonDeltaPosition.setToValue(0.0);
  m_LangevinPistonDeltaPositionPrevious.setToValue(0.0);
  m_LangevinPistonDeltaPositionPredicted.setToValue(0.0);
  m_LangevinPistonDeltaPressure.setToValue(0.0);
  m_Prfwd.setToValue(0.0);

  return;
}

int CudaLangevinPistonIntegrator::getContextPistonDegreesOfFreedom(void) const {
  APOCHARMM_REQUIRE(m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "Langevin piston requires an attached CharmmContext");

  m_Context->requireCrystalBox();

  switch (m_Context->getCrystalType()) {
  case CRYSTAL::CUBIC:
    return 1;
  case CRYSTAL::TETRAGONAL:
    return 2;
  case CRYSTAL::ORTHORHOMBIC:
    return 3;
  default:
    APOCHARMM_THROW(ApoCharmmErrorCode::NotInitialized,
                    "Crystal type must be configured before this operation");
  }
}

void CudaLangevinPistonIntegrator::requireCurrentState(void) const {
  if (m_Context == nullptr)
    return;

  APOCHARMM_REQUIRE(this->getContextPistonDegreesOfFreedom() ==
                        m_LangevinPistonDegreesOfFreedom,
                    ApoCharmmErrorCode::NotInitialized,
                    "Langevin piston state does not match the context crystal "
                    "type; reinitialize the integrator");

  return;
}

__global__ static void
InitializeRngKernel(curandStatePhilox4_32_10_t *__restrict__ states,
                    const int n, const unsigned long long int seed,
                    const unsigned long long int offset,
                    const unsigned long long int nskip) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < n; i += stride) {
    curand_init(seed, i, offset, states + i);
    skipahead(2 * nskip, states + i);
  }

  return;
}

void CudaLangevinPistonIntegrator::initializeRng(void) {
  if (m_LangevinPistonDegreesOfFreedom == -1)
    return;

  this->alloc(m_LangevinPistonDegreesOfFreedom);
  constexpr int numThreads = 256;
  const int numBlocks =
      (m_LangevinPistonDegreesOfFreedom + numThreads - 1) / numThreads;
  cudaCheckLaunch(
      InitializeRngKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          m_RngStates, m_LangevinPistonDegreesOfFreedom, m_Seed, 0,
          m_RngSequencePos));

  return;
}

void CudaLangevinPistonIntegrator::removeCenterOfMassMotion(void) {
  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  const PBC pbc = m_Context->getForceManager()->getPeriodicBoundaryCondition();
  const int numAtoms = m_Context->getNumAtoms();
  CudaContainer<double4> &velMass = m_Context->getVelocitiesInverseMasses();

  velMass.transferToHost();
  m_CoordsDeltaPrevious.transferToHost();

  double3 com = make_double3(0.0, 0.0, 0.0);
  double totalMass = 0.0;
  for (int i = 0; i < numAtoms; i++) {
    double mass = 1.0 / velMass[i].w;

    com.x += m_CoordsDeltaPrevious[i].x * mass;
    if (pbc == PBC::P21) {
      com.y -= m_CoordsDeltaPrevious[i].y * mass;
      com.z -= m_CoordsDeltaPrevious[i].z * mass;
    } else {
      com.y += m_CoordsDeltaPrevious[i].y * mass;
      com.z += m_CoordsDeltaPrevious[i].z * mass;
    }

    totalMass += mass;
  }
  com.x /= totalMass;
  com.y /= totalMass;
  com.z /= totalMass;

  for (int i = 0; i < numAtoms; i++) {
    m_CoordsDeltaPrevious[i].x -= com.x;
    if (pbc == PBC::P21) {
      m_CoordsDeltaPrevious[i].y += com.y;
      m_CoordsDeltaPrevious[i].z += com.z;
    } else {
      m_CoordsDeltaPrevious[i].y -= com.y;
      m_CoordsDeltaPrevious[i].z -= com.z;
    }
  }
  m_CoordsDeltaPrevious.transferToDevice();

  return;
}

void CudaLangevinPistonIntegrator::alloc(const int n) {
  if (m_RngStates != nullptr) // Deallocate to be safe
    this->dealloc();
  // Allocate memory for RNG
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&m_RngStates),
                       n * sizeof(curandStatePhilox4_32_10_t)));
  return;
}

void CudaLangevinPistonIntegrator::dealloc(void) {
  ::deallocate(&m_RngStates);
  return;
}
