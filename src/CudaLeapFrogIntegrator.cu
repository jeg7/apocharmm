// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaLeapFrogIntegrator.h"

#include <iostream>

CudaLeapFrogIntegrator::CudaLeapFrogIntegrator(const double timeStep)
    : CudaIntegrator(timeStep) {}

void CudaLeapFrogIntegrator::initializeImpl(void) { return; }

__global__ static void leapFrogKernel(const int numAtoms, const int stride,
                                      const double timeStep,
                                      float4 *__restrict__ xyzq,
                                      double4 *__restrict__ velMass,
                                      const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    float fx = force[index];
    float fy = force[index + stride];
    float fz = force[index + 2 * stride];

    velMass[index].x -= timeStep * fx * velMass[index].w;
    velMass[index].y -= timeStep * fy * velMass[index].w;
    velMass[index].z -= timeStep * fz * velMass[index].w;

    xyzq[index].x += timeStep * velMass[index].x;
    xyzq[index].y += timeStep * velMass[index].y;
    xyzq[index].z += timeStep * velMass[index].z;
  }
}

void CudaLeapFrogIntegrator::propagateOneStepImpl(void) {
  float4 *xyzq = m_Context->getCoordinatesChargesSP().getDeviceArray().data();
  double4 *velMass =
      m_Context->getVelocitiesInverseMasses().getDeviceArray().data();

  if (m_StepsSinceNeighborListUpdate % 20 == 0)
    m_Context->resetNeighborList();

  // A subscriber that reports after this step reads its energies and virial.
  const bool calcEnergy = this->isReportDueAfterStep();
  m_Context->calculateForces(false, calcEnergy, calcEnergy);
  auto force = m_Context->getForces();

  m_Context->calculateKineticEnergy();
  auto ke = m_Context->getKineticEnergy();
  auto peContainer = m_Context->getPotentialEnergy();
  peContainer.transferFromDevice();
  auto pe = peContainer[0];

  int numAtoms = m_Context->getNumAtoms();
  int stride = m_Context->getForceStride();

  int numThreads = 1024;
  int numBlocks = (numAtoms - 1) / numThreads + 1;

  gpu_range_start("leapFrog");
  leapFrogKernel<<<numBlocks, numThreads>>>(numAtoms, stride, m_TimeStep, xyzq,
                                            velMass, force->xyz());
  cudaCheck(cudaDeviceSynchronize());

  gpu_range_stop();

  return;
}
