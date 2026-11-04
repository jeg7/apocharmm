// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaLangevinThermostatIntegrator.h"

#include "ApoCharmmError.h"
#include "Constants.h"
#include "Crystal.h"
#include "CurandStateString.h"
#include "RestartIO.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <random>
#include <vector_functions.h>

CudaLangevinThermostatIntegrator::CudaLangevinThermostatIntegrator(
    const double timeStep)
    : CudaIntegrator(timeStep) {
  m_UsingHolonomicConstraints = true;

  m_ReferenceTemperature = 300.0;
  this->setThermostatFriction(0.0);

  std::random_device rd{};
  m_Seed = rd();
  m_RngSequencePos = 0;
  m_RngStates = nullptr;

  // 0 -> New "Jung" T
  // 1 -> Old T
  m_AverageWindowSize = 0;
  m_KineticEnergy.resize(2);
  // m_KineticEnergyPartialSums is allocated in this->initialize();
  m_AverageTemperature.resize(2);

  m_KineticEnergy.setToValue(0.0);
  m_AverageTemperature.setToValue(0.0);
}

CudaLangevinThermostatIntegrator::~CudaLangevinThermostatIntegrator(
    void) noexcept {
  ::deallocate_noexcept(&m_RngStates);
}

void CudaLangevinThermostatIntegrator::setReferenceTemperature(
    const double referenceTemperature) {
  APOCHARMM_REQUIRE(std::isfinite(referenceTemperature),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference temperature must be finite; observed " +
                        std::to_string(referenceTemperature));

  APOCHARMM_REQUIRE(referenceTemperature >= 0.0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference temperature must be non-negative; observed " +
                        std::to_string(referenceTemperature));

  m_ReferenceTemperature = referenceTemperature;

  return;
}

void CudaLangevinThermostatIntegrator::setThermostatFriction(
    const double thermostatFriction) {
  APOCHARMM_REQUIRE(std::isfinite(thermostatFriction),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Thermostat friction must be finite; observed " +
                        std::to_string(thermostatFriction));

  APOCHARMM_REQUIRE(thermostatFriction >= 0.0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Thermostat friction must be non-negative; observed " +
                        std::to_string(thermostatFriction));

  m_ThermostatFriction = thermostatFriction;
  m_ThermostatGamma =
      m_TimeStep * charmm::constants::timfac * thermostatFriction;

  // JEG260409: If the friction changes before running a new simulation (i.e.
  // not from a restart file or continuing a previous one) AND the CHARMM
  // context has previous been set, initialize the derived state again so
  // m_CoordsDelta and m_CoordsDeltaPrevious can have correct values.
  if ((m_Context != nullptr) && (m_TotNumSteps == 0))
    this->initializeImpl();

  return;
}

void CudaLangevinThermostatIntegrator::setThermostatRngSeed(
    const std::uint64_t seed) {
  m_Seed = seed;
  this->initializeRng();
  return;
}

void CudaLangevinThermostatIntegrator::setRngSequencePos(
    const unsigned long long int rngSequencePos) {
  m_RngSequencePos = rngSequencePos;
  this->initializeRng();
  return;
}

void CudaLangevinThermostatIntegrator::setRngStates(
    const std::string &rngStateString) {
  APOCHARMM_REQUIRE(m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "CharmmContext must be set before RNG states");

  unsigned long long int pos = 0;
  std::vector<curandStatePhilox4_32_10_t> states;
  apo::curand_states_from_string(pos, states, rngStateString);

  const int numAtoms = m_Context->getNumAtoms();
  APOCHARMM_REQUIRE(states.size() == static_cast<std::size_t>(numAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "RNG state count must match number of atoms; expected " +
                        std::to_string(numAtoms) + ", observed " +
                        std::to_string(states.size()));

  this->setRngStateData(pos, states);

  return;
}

void CudaLangevinThermostatIntegrator::resetAverageTemperature(void) {
  m_AverageWindowSize = 0;
  m_AverageTemperature.setToValue(0.0);
  return;
}

double CudaLangevinThermostatIntegrator::getReferenceTemperature(void) const {
  return m_ReferenceTemperature;
}

double CudaLangevinThermostatIntegrator::getThermostatFriction(void) const {
  return m_ThermostatFriction;
}

std::uint64_t
CudaLangevinThermostatIntegrator::getThermostatRngSeed(void) const {
  return m_Seed;
}

unsigned long long int
CudaLangevinThermostatIntegrator::getRngSequencePos(void) const {
  return m_RngSequencePos;
}

std::string CudaLangevinThermostatIntegrator::getRngStates(void) const {
  APOCHARMM_REQUIRE(m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "CharmmContext must be set before retrieving RNG states");

  APOCHARMM_REQUIRE(m_RngStates != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "RNG states are not initialized");

  const int numAtoms = m_Context->getNumAtoms();
  std::vector<curandStatePhilox4_32_10_t> rngStates(numAtoms);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
  cudaCheck(cudaMemcpy(static_cast<void *>(rngStates.data()),
                       static_cast<const void *>(m_RngStates),
                       rngStates.size() * sizeof(curandStatePhilox4_32_10_t),
                       cudaMemcpyDeviceToHost));

  return apo::curand_states_to_string(m_RngSequencePos, rngStates);
}

int CudaLangevinThermostatIntegrator::getAverageWindowSize(void) const {
  return m_AverageWindowSize;
}

const CudaContainer<double> &
CudaLangevinThermostatIntegrator::getKineticEnergy(void) const {
  return m_KineticEnergy;
}

const CudaContainer<double> &
CudaLangevinThermostatIntegrator::getAverageTemperature(void) const {
  return m_AverageTemperature;
}

CudaContainer<double> &
CudaLangevinThermostatIntegrator::getKineticEnergy(void) {
  return m_KineticEnergy;
}

CudaContainer<double> &
CudaLangevinThermostatIntegrator::getAverageTemperature(void) {
  return m_AverageTemperature;
}

double CudaLangevinThermostatIntegrator::getInstantaneousTemperature(void) {
  APOCHARMM_REQUIRE(
      m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before computing instantaneous temperature");

  APOCHARMM_REQUIRE(m_KineticEnergy.size() == 2, ApoCharmmErrorCode::Runtime,
                    "Kinetic energy does not contain exactly 2 elements");

  const double ndegf = static_cast<double>(m_Context->getNumDegreesOfFreedom());
  m_KineticEnergy.transferToHost();
  return (m_KineticEnergy[0] / (0.5 * ndegf * charmm::constants::kBoltz));
}

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
                     const double gamma, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;
  const double halfGamma = 0.5 * gamma;
  const double alpha0 = timeStep * sqrt(1.0 + halfGamma);
  const double alpha1 = timeStep * (1.0 - halfGamma) / sqrt(1.0 + halfGamma);

  for (int i = idx; i < numAtoms; i += stride) {
    const double fx = forces[0 * forceStride + i];
    const double fy = forces[1 * forceStride + i];
    const double fz = forces[2 * forceStride + i];
    const double fact = 0.5 * timeStep2 * velMass[i].w;

    coordsDelta[i].x = alpha0 * velMass[i].x - fact * fx;
    coordsDelta[i].y = alpha0 * velMass[i].y - fact * fy;
    coordsDelta[i].z = alpha0 * velMass[i].z - fact * fz;

    coordsDeltaPrevious[i].x = alpha1 * velMass[i].x + fact * fx;
    coordsDeltaPrevious[i].y = alpha1 * velMass[i].y + fact * fy;
    coordsDeltaPrevious[i].z = alpha1 * velMass[i].z + fact * fz;
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

void CudaLangevinThermostatIntegrator::initializeImpl(void) {
  const int numAtoms = m_Context->getNumAtoms();
  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  this->initializeRng();

  m_KineticEnergyPartialSums.resize(numBlocks * 2);

  double4 *coordsCharges =
      m_Context->getCoordinatesChargesDP().getDeviceArray().data();
  float4 *xyzq = m_Context->getCoordinatesChargesSP().getDeviceArray().data();

  if (m_UsingHolonomicConstraints) {
    copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceArray().data(),
                             numAtoms, *m_IntegratorStream);

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceArray().data());

    cudaCheckLaunch(
        UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                                 *m_IntegratorStream>>>(
            xyzq, coordsCharges, numAtoms));

    copy_DtoD_async<double4>(coordsCharges, m_CoordsRef.getDeviceArray().data(),
                             numAtoms, *m_IntegratorStream);

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
          forces, forceStride, m_ThermostatGamma, m_TimeStep));

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

  return;
}

void CudaLangevinThermostatIntegrator::initializeFromRestartFileImpl(
    const std::filesystem::path &rstFilePath) {
  const int numAtoms = m_Context->getNumAtoms();
  const apo::RestartSnapshot snapshot =
      apo::ReadRestartFile(rstFilePath, numAtoms);

  m_Context->requireCrystalBox();

  APOCHARMM_REQUIRE(
      snapshot.crystal == m_Context->getCrystalType(),
      ApoCharmmErrorCode::InvalidArgument,
      "Restart crystal type does not match the configured CharmmContext");

  // JEG261102: We should support this one day and just reset the RNG, but for
  // now we will throw an error.
  APOCHARMM_REQUIRE(snapshot.nativeKind.empty() ||
                        snapshot.nativeKind == "LANGEVIN_THERMOSTAT",
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

  // JEG261102: Again, we should support this one day.
  APOCHARMM_REQUIRE((snapshot.hdot == std::array<double, 6>{}) &&
                        snapshot.pnhv == 0.0 && snapshot.pnhf == 0.0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "A Langevin thermostat cannot restore piston or "
                    "Nose-Hoover thermostat state");

  unsigned long long int restartRngPosition = 0;
  std::vector<curandStatePhilox4_32_10_t> restartRngStates;
  if (snapshot.hasNativeRng) {
    apo::curand_states_from_string(restartRngPosition, restartRngStates,
                                   snapshot.rngState);
    APOCHARMM_REQUIRE(restartRngStates.size() ==
                          static_cast<std::size_t>(numAtoms),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Restart RNG state count must match the atom count");
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

  this->resetAverageTemperature();

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

__global__ static void PositionUpdateKernel(
    double4 *__restrict__ coordsCharges, double4 *__restrict__ coordsDelta,
    curandStatePhilox4_32_10_t *__restrict__ rngStates,
    const double4 *__restrict__ coordsDeltaPrevious,
    const double4 *__restrict__ velMass, const int numAtoms,
    const double *__restrict__ forces, const int forceStride,
    const double gamma, const double kbt, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double timeStep2 = timeStep * timeStep;
  const double halfGamma = 0.5 * gamma;
  const double denomGamma = 1.0 / (1.0 + halfGamma);
  const double alpha = (1.0 - halfGamma) * denomGamma;

  for (int i = idx; i < numAtoms; i += stride) {
    curandStatePhilox4_32_10_t state = rngStates[i];
    const double rdf = sqrt(2.0 * gamma * kbt / velMass[i].w) / timeStep;
    const float4 rn = curand_normal4(&state);
    const double fact = timeStep2 * velMass[i].w * denomGamma;
    double fx = forces[0 * forceStride + i];
    double fy = forces[1 * forceStride + i];
    double fz = forces[2 * forceStride + i];
    fx += rdf * static_cast<double>(rn.x);
    fy += rdf * static_cast<double>(rn.y);
    fz += rdf * static_cast<double>(rn.z);

    coordsDelta[i].x = alpha * coordsDeltaPrevious[i].x - fact * fx;
    coordsDelta[i].y = alpha * coordsDeltaPrevious[i].y - fact * fy;
    coordsDelta[i].z = alpha * coordsDeltaPrevious[i].z - fact * fz;

    coordsCharges[i].x += coordsDelta[i].x;
    coordsCharges[i].y += coordsDelta[i].y;
    coordsCharges[i].z += coordsDelta[i].z;

    rngStates[i] = state;
  }

  return;
}

__global__ static void VelocityUpdateKernel(
    double4 *__restrict__ coordsDelta, double4 *__restrict__ velMass,
    const double4 *__restrict__ coordsCharges,
    const double4 *__restrict__ coordsRef,
    const double4 *__restrict__ coordsDeltaPrevious, const int numAtoms,
    const double gamma, const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 * sqrt(1.0 + 0.5 * gamma) / timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    coordsDelta[i].x = coordsCharges[i].x - coordsRef[i].x;
    coordsDelta[i].y = coordsCharges[i].y - coordsRef[i].y;
    coordsDelta[i].z = coordsCharges[i].z - coordsRef[i].z;

    velMass[i].x = fact * (coordsDelta[i].x + coordsDeltaPrevious[i].x);
    velMass[i].y = fact * (coordsDelta[i].y + coordsDeltaPrevious[i].y);
    velMass[i].z = fact * (coordsDelta[i].z + coordsDeltaPrevious[i].z);
  }

  return;
}

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
        0.5 * ((u1.x * u1.x) + (u1.y * u1.y) + (u1.z * u1.z)) / v.w;
    const double onStepKineticEnergy =
        0.5 * ((v.x * v.x) + (v.y * v.y) + (v.z * v.z)) / v.w;
    const double newHalfStepKineticEnergy =
        0.5 * ((u2.x * u2.x) + (u2.y * u2.y) + (u2.z * u2.z)) / v.w;

    ke[0] += oneThird * (oldHalfStepKineticEnergy + onStepKineticEnergy +
                         newHalfStepKineticEnergy);
    ke[1] += onStepKineticEnergy;
  }

  __shared__ BlockReduceSumStorage<double, blockSize, 2> cache;

  BlockReduceSum<double, blockSize, 2>(ke, cache);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 2; i++)
      kineticEnergyPartialSums[blockIdx.x * 2 + i] = ke[0];
  }

  return;
}

template <int blockSize>
__global__ static void
ComputeKineticEnergyKernel(double *__restrict__ kineticEnergy,
                           const double *__restrict__ kineticEnergyPartialSums,
                           const int numPartialSums) {
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

void CudaLangevinThermostatIntegrator::propagateOneStepImpl(void) {
  const double kbt = charmm::constants::kBoltz * m_ReferenceTemperature;
  const int numAtoms = m_Context->getNumAtoms();
  double4 *coordsCharges =
      m_Context->getCoordinatesChargesDP().getDeviceArray().data();
  float4 *xyzq = m_Context->getCoordinatesChargesSP().getDeviceArray().data();
  double4 *velMass =
      m_Context->getVelocitiesInverseMasses().getDeviceArray().data();
  const int forceStride = m_Context->getForceStride();
  double *forces = m_Context->getForces()->xyz();

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
      float boxDimX = static_cast<float>(m_Context->getBoxDimensions()[0]);

      constexpr int numThreads = 256;
      const int numBlocks = (numGroups + numThreads - 1) / numThreads;
      cudaCheckLaunch(InvertDeltaAsymmetricKernel<<<numBlocks, numThreads, 0,
                                                    *m_IntegratorStream>>>(
          m_CoordsDeltaPrevious.getDeviceArray().data(), xyzq, groups,
          numGroups, boxDimX));
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

  if (m_DebugPrintFrequency > 0 &&
      m_TotNumSteps %
              static_cast<unsigned long long int>(m_DebugPrintFrequency) ==
          0ULL) {
    m_Context->calculateForces(false, true, true);
  } else {
    m_Context->calculateForces(false, false, false);
  }

  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  cudaCheckLaunch(
      PositionUpdateKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          coordsCharges, m_CoordsDelta.getDeviceArray().data(), m_RngStates,
          m_CoordsDeltaPrevious.getDeviceArray().data(), velMass, numAtoms,
          forces, forceStride, m_ThermostatGamma, kbt, m_TimeStep));

  m_RngSequencePos += 4; // curand_normal4 iterates 4 steps

  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceArray().data());
  }

  cudaCheckLaunch(
      VelocityUpdateKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          m_CoordsDelta.getDeviceArray().data(), velMass, coordsCharges,
          m_CoordsRef.getDeviceArray().data(),
          m_CoordsDeltaPrevious.getDeviceArray().data(), numAtoms,
          m_ThermostatGamma, m_TimeStep));

  cudaCheck(cudaMemsetAsync(
      static_cast<void *>(m_KineticEnergy.getDeviceArray().data()), 0,
      2 * sizeof(double), *m_IntegratorStream));

  cudaCheckLaunch(
      ComputeKineticEnergyPartialSumsKernel<numThreads>
      <<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          m_KineticEnergyPartialSums.getDeviceArray().data(), velMass,
          m_CoordsDelta.getDeviceArray().data(),
          m_CoordsDeltaPrevious.getDeviceArray().data(), numAtoms, m_TimeStep));

  cudaCheckLaunch(
      ComputeKineticEnergyKernel<numThreads>
      <<<1, numThreads, 0, *m_IntegratorStream>>>(
          m_KineticEnergy.getDeviceArray().data(),
          m_KineticEnergyPartialSums.getDeviceArray().data(), numBlocks));

  cudaCheckLaunch(
      UpdateAverageTemperatureKernel<<<1, 32, 0, *m_IntegratorStream>>>(
          m_AverageTemperature.getDeviceArray().data(),
          m_KineticEnergy.getDeviceArray().data(),
          m_Context->getNumDegreesOfFreedom(), charmm::constants::kBoltz,
          m_AverageWindowSize));

  m_AverageWindowSize++;

  cudaCheckLaunch(
      UpdateSinglePrecisionCoordinatesKernel<<<numBlocks, numThreads, 0,
                                               *m_IntegratorStream>>>(
          xyzq, coordsCharges, numAtoms));

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceArray().data(),
                           m_CoordsDeltaPrevious.getDeviceArray().data(),
                           numAtoms, *m_IntegratorStream);

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  return;
}

void CudaLangevinThermostatIntegrator::setRngStateData(
    const unsigned long long int position,
    const std::vector<curandStatePhilox4_32_10_t> &states) {
  m_RngSequencePos = position;
  this->alloc(static_cast<int>(states.size()));

  cudaCheck(cudaMemcpy(static_cast<void *>(m_RngStates),
                       static_cast<const void *>(states.data()),
                       states.size() * sizeof(curandStatePhilox4_32_10_t),
                       cudaMemcpyHostToDevice));

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
    skipahead(nskip, states + i);
  }

  return;
}

void CudaLangevinThermostatIntegrator::initializeRng(void) {
  if (m_Context == nullptr)
    return;

  const int numAtoms = m_Context->getNumAtoms();
  this->alloc(numAtoms);
  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;
  cudaCheckLaunch(
      InitializeRngKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          m_RngStates, numAtoms, m_Seed, 0, m_RngSequencePos));

  return;
}

void CudaLangevinThermostatIntegrator::removeCenterOfMassMotion(void) {
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

void CudaLangevinThermostatIntegrator::alloc(const int n) {
  if (m_RngStates != nullptr) // Deallocate to be safe
    this->dealloc();

  // Allocate memory for RNG
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&m_RngStates),
                       n * sizeof(curandStatePhilox4_32_10_t)));

  return;
}

void CudaLangevinThermostatIntegrator::dealloc(void) {
  ::deallocate(&m_RngStates);
  return;
}
