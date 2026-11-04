// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaNoseHooverIntegrator.h"

#include "ApoCharmmError.h"
#include "Constants.h"
#include "Crystal.h"
#include "RestartIO.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <vector>
#include <vector_functions.h>

CudaNoseHooverIntegrator::CudaNoseHooverIntegrator(const double timeStep)
    : CudaIntegrator(timeStep) {
  m_UsingHolonomicConstraints = true;

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

  m_MaxPredictorCorrectorIterations = 3;

  // 0 -> New "Jung" T
  // 1 -> Old T
  m_AverageWindowSize = 0;
  m_KineticEnergy.resize(2);
  // m_KineticEnergyPartialSums is allocated in this->initialize();
  m_AverageTemperature.resize(2);

  m_KineticEnergy.setToValue(0.0);
  m_AverageTemperature.setToValue(0.0);

  m_UsingOldTemperature = false;
}

void CudaNoseHooverIntegrator::setReferenceTemperature(
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

void CudaNoseHooverIntegrator::setNoseHooverPistonMass(
    const double noseHooverPistonMass) {
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

void CudaNoseHooverIntegrator::setNoseHooverPistonVelocity(
    const double noseHooverPistonVelocity) {
  APOCHARMM_REQUIRE(std::isfinite(noseHooverPistonVelocity),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Nose-Hoover piston velocity must be finite; observed " +
                        std::to_string(noseHooverPistonVelocity));

  m_NoseHooverPistonVelocity.setToValue(noseHooverPistonVelocity);

  return;
}

void CudaNoseHooverIntegrator::setNoseHooverPistonVelocityPrevious(
    const double noseHooverPistonVelocityPrevious) {
  APOCHARMM_REQUIRE(
      std::isfinite(noseHooverPistonVelocityPrevious),
      ApoCharmmErrorCode::InvalidArgument,
      "Previous Nose-Hoover piston velocity must be finite; observed " +
          std::to_string(noseHooverPistonVelocityPrevious));

  m_NoseHooverPistonVelocityPrevious.setToValue(
      noseHooverPistonVelocityPrevious);

  return;
}

void CudaNoseHooverIntegrator::setNoseHooverPistonForce(
    const double noseHooverPistonForce) {
  APOCHARMM_REQUIRE(std::isfinite(noseHooverPistonForce),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Nose-Hoover piston force must be finite; observed " +
                        std::to_string(noseHooverPistonForce));

  m_NoseHooverPistonForce.setToValue(noseHooverPistonForce);

  return;
}

void CudaNoseHooverIntegrator::setNoseHooverPistonForcePrevious(
    const double noseHooverPistonForcePrevious) {
  APOCHARMM_REQUIRE(
      std::isfinite(noseHooverPistonForcePrevious),
      ApoCharmmErrorCode::InvalidArgument,
      "Previous Nose-Hoover piston force must be finite; observed " +
          std::to_string(noseHooverPistonForcePrevious));

  m_NoseHooverPistonForcePrevious.setToValue(noseHooverPistonForcePrevious);

  return;
}

void CudaNoseHooverIntegrator::setMaxPredictorCorrectorIterations(
    const int maxPredictorCorrectorIterations) {
  APOCHARMM_REQUIRE(
      maxPredictorCorrectorIterations > 0, ApoCharmmErrorCode::InvalidArgument,
      "Maximum predictor-corrector iterations must be positive; observed " +
          std::to_string(maxPredictorCorrectorIterations));

  m_MaxPredictorCorrectorIterations = maxPredictorCorrectorIterations;

  return;
}

void CudaNoseHooverIntegrator::useOldTemperature(
    const bool usingOldTemperature) {
  m_UsingOldTemperature = usingOldTemperature;
  return;
}

void CudaNoseHooverIntegrator::resetAverageTemperature(void) {
  m_AverageWindowSize = 0;
  m_AverageTemperature.setToValue(0.0);
  return;
}

double CudaNoseHooverIntegrator::getReferenceTemperature(void) const {
  return m_ReferenceTemperature;
}

bool CudaNoseHooverIntegrator::usingOldTemperature(void) const {
  return m_UsingOldTemperature;
}

const CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonMass(void) const {
  return m_NoseHooverPistonMass;
}

const CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonVelocity(void) const {
  return m_NoseHooverPistonVelocity;
}

const CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonVelocityPrevious(void) const {
  return m_NoseHooverPistonVelocityPrevious;
}

const CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonForce(void) const {
  return m_NoseHooverPistonForce;
}

const CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonForcePrevious(void) const {
  return m_NoseHooverPistonForcePrevious;
}

int CudaNoseHooverIntegrator::getMaxPredictorCorrectorIterations(void) const {
  return m_MaxPredictorCorrectorIterations;
}

const CudaContainer<double> &
CudaNoseHooverIntegrator::getKineticEnergy(void) const {
  return m_KineticEnergy;
}

const CudaContainer<double> &
CudaNoseHooverIntegrator::getAverageTemperature(void) const {
  return m_AverageTemperature;
}

CudaContainer<double> &CudaNoseHooverIntegrator::getNoseHooverPistonMass(void) {
  return m_NoseHooverPistonMass;
}

CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonVelocity(void) {
  return m_NoseHooverPistonVelocity;
}

CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonVelocityPrevious(void) {
  return m_NoseHooverPistonVelocityPrevious;
}

CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonForce(void) {
  return m_NoseHooverPistonForce;
}

CudaContainer<double> &
CudaNoseHooverIntegrator::getNoseHooverPistonForcePrevious(void) {
  return m_NoseHooverPistonForcePrevious;
}

CudaContainer<double> &CudaNoseHooverIntegrator::getKineticEnergy(void) {
  return m_KineticEnergy;
}

CudaContainer<double> &CudaNoseHooverIntegrator::getAverageTemperature(void) {
  return m_AverageTemperature;
}

double CudaNoseHooverIntegrator::getInstantaneousTemperature(void) {
  APOCHARMM_REQUIRE(
      m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before querying instantaneous temperature");

  const double ndegf = static_cast<double>(m_Context->getNumDegreesOfFreedom());
  m_KineticEnergy.transferToHost();
  if (m_UsingOldTemperature)
    return (m_KineticEnergy[1] / (0.5 * ndegf * charmm::constants::kBoltz));
  return (m_KineticEnergy[0] / (0.5 * ndegf * charmm::constants::kBoltz));
}

/** @brief Updates the single-prercision coordinates container (xyzq)
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

void CudaNoseHooverIntegrator::initializeImpl(void) {
  const int numAtoms = m_Context->getNumAtoms();
  constexpr int numThreads = 256;
  const int numBlocks = (numAtoms + numThreads - 1) / numThreads;

  m_CoordsDeltaPredicted.resize(numAtoms);
  m_KineticEnergyPartialSums.resize(numBlocks * 2);

  if (m_NoseHooverPistonMass[0] == -9999.9999)
    this->setNoseHooverPistonMass(this->computeNoseHooverPistonMass());

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

  return;
}

void CudaNoseHooverIntegrator::initializeFromRestartFileImpl(
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
                        snapshot.nativeKind == "NOSE_HOOVER",
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
  APOCHARMM_REQUIRE(
      (snapshot.hdot == std::array<double, 6>{}),
      ApoCharmmErrorCode::InvalidArgument,
      "A fixed-cell Nose-Hoover integrator cannot restore moving-cell state");

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

  // The file's thermostat state is installed after any regeneration.
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
      kineticEnergyPartialSums[blockIdx.x * 2 + i] = ke[i];
  }

  return;
}

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
      ke[0] += kineticEnergyPartialSums[i * 2 + 0];
      ke[1] += kineticEnergyPartialSums[i * 2 + 1];
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
    const double timeStep) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const double fact = 0.5 / timeStep;

  for (int i = idx; i < numAtoms; i += stride) {
    const double onStepVelocityX =
        fact * (coordsDeltaPrevious[i].x + coordsDeltaPredicted[i].x);
    const double onStepVelocityY =
        fact * (coordsDeltaPrevious[i].y + coordsDeltaPredicted[i].y);
    const double onStepVelocityZ =
        fact * (coordsDeltaPrevious[i].z + coordsDeltaPredicted[i].z);
    const double nhFact = timeStep * timeStep * noseHooverPistonVelocity[0];

    coordsDeltaPredicted[i].x = coordsDelta[i].x - nhFact * onStepVelocityX;
    coordsDeltaPredicted[i].y = coordsDelta[i].y - nhFact * onStepVelocityY;
    coordsDeltaPredicted[i].z = coordsDelta[i].z - nhFact * onStepVelocityZ;

    velMass[i].x = onStepVelocityX;
    velMass[i].y = onStepVelocityY;
    velMass[i].z = onStepVelocityZ;

    coordsCharges[i].x = coordsRef[i].x + coordsDeltaPredicted[i].x;
    coordsCharges[i].y = coordsRef[i].y + coordsDeltaPredicted[i].y;
    coordsCharges[i].z = coordsRef[i].z + coordsDeltaPredicted[i].z;
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

void CudaNoseHooverIntegrator::propagateOneStepImpl(void) {
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

  if (m_UsingHolonomicConstraints) {
    // JEG250506: Need sync here, otherwise race condition? I don't get it
    // either...
    cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceArray().data());

    cudaCheckLaunch(UpdateCoordsDeltaAfterHolonomicConstraintKernel<<<
                        numBlocks, numThreads, 0, *m_IntegratorStream>>>(
        m_CoordsDelta.getDeviceArray().data(), coordsCharges,
        m_CoordsRef.getDeviceArray().data(), numAtoms));
  }

  copy_DtoD_async<double4>(m_CoordsDelta.getDeviceArray().data(),
                           m_CoordsDeltaPredicted.getDeviceArray().data(),
                           numAtoms, *m_IntegratorStream);

  for (int iter = 0; iter < m_MaxPredictorCorrectorIterations; iter++) {
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
        m_NoseHooverPistonVelocity.getDeviceArray().data(), m_TimeStep));
  }

  if (m_UsingHolonomicConstraints) {
    m_HolonomicConstraint->handleHolonomicConstraints(
        m_CoordsRef.getDeviceArray().data());

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

double CudaNoseHooverIntegrator::computeNoseHooverPistonMass(void) {
  APOCHARMM_REQUIRE(
      m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before computing the Nose-Hoover piston mass");

  CudaContainer<double4> velMass = m_Context->getVelocitiesInverseMasses();
  velMass.transferToHost();

  double totalMass = 0.0;
  for (std::size_t i = 0; i < velMass.size(); i++)
    totalMass += (1.0 / velMass[i].w);

  return (0.2 * totalMass);
}

void CudaNoseHooverIntegrator::removeCenterOfMassMotion(void) {
  APOCHARMM_REQUIRE(
      m_Context != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmContext must be set before removing center-of-mass motion");

  cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));

  const PBC pbc = m_Context->getForceManager()->getPeriodicBoundaryCondition();
  const int numAtoms = m_Context->getNumAtoms();
  CudaContainer<double4> &velMass = m_Context->getVelocitiesInverseMasses();

  velMass.transferToHost();
  m_CoordsDeltaPrevious.transferToHost();

  double3 com = make_double3(0.0, 0.0, 0.0);
  double totalMass = 0.0;
  for (int i = 0; i < numAtoms; i++) {
    const double mass = 1.0 / velMass[i].w;

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
