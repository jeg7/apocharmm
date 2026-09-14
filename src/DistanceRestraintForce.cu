// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II
//
// ENDLICENSE

#include "DistanceRestraintForce.h"

#include "ApoCharmmError.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

template <typename AT, typename CT>
DistanceRestraintForce<AT, CT>::DistanceRestraintForce(const int numAtoms)
    : m_NumAtoms(numAtoms), m_Scale(1.0), m_BoxDimX(0.0), m_BoxDimY(0.0),
      m_BoxDimZ(0.0), m_DeviceDataDirty(true), m_TermPairOffsets(),
      m_FirstAtomIndices(), m_SecondAtomIndices(), m_PairCoefficients(),
      m_ForceConstants(), m_ReferenceValues(), m_DistanceExponents(),
      m_EnergyExponents(), m_ActivationConditions(), m_EnergyVirial(nullptr),
      m_Forces(nullptr), m_Stream(nullptr) {
  APOCHARMM_REQUIRE(numAtoms > 0, ApoCharmmErrorCode::InvalidArgument,
                    "Atom count must be positive; observed " +
                        std::to_string(numAtoms));

  m_TermPairOffsets.getHostArray().push_back(0);

  m_EnergyVirial = std::make_shared<CudaEnergyVirial>();
  m_EnergyVirial->insert("resd");

  m_Forces = std::make_shared<Force<AT>>();
  m_Forces->realloc(numAtoms, 1.5f);

  m_Stream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_Stream.get()));
}

template <typename AT, typename CT>
DistanceRestraintForce<AT, CT>::~DistanceRestraintForce(void) noexcept {
  this->dealloc();
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::setScale(const double scale) {
  APOCHARMM_REQUIRE(std::isfinite(scale), ApoCharmmErrorCode::InvalidArgument,
                    "Scale must be finite; observed " + std::to_string(scale));

  m_Scale = scale;

  return;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::addRestraint(
    const std::vector<std::array<int, 2>> &atomPairs,
    const std::vector<double> &coefficients, const double forceConstant,
    const double referenceValue, const int distanceExponent,
    const int energyExponent, const DistanceRestraintCondition condition) {
  APOCHARMM_REQUIRE(
      !atomPairs.empty(), ApoCharmmErrorCode::InvalidArgument,
      "A distance-restraint term must contain at least one atom pair");

  APOCHARMM_REQUIRE(atomPairs.size() == coefficients.size(),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Pair and coefficient counts must match; observed " +
                        std::to_string(atomPairs.size()) + " pairs and " +
                        std::to_string(coefficients.size()) + " coefficients");

  const std::size_t maxIntCount =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  const std::size_t currentTermCount = m_ForceConstants.size();
  const std::size_t currentPairCount = m_FirstAtomIndices.size();

  APOCHARMM_REQUIRE(currentTermCount < maxIntCount,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Adding a distance-restraint term would exceed the int "
                    "term-count limit; maximum " +
                        std::to_string(maxIntCount) + ", current " +
                        std::to_string(currentTermCount));

  APOCHARMM_REQUIRE(currentPairCount <= maxIntCount,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Existing flattened pair count exceeds the int "
                    "representation limit; maximum " +
                        std::to_string(maxIntCount) + ", observed " +
                        std::to_string(currentPairCount));

  APOCHARMM_REQUIRE(atomPairs.size() <= maxIntCount - currentPairCount,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Adding the requested atom pairs would exceed the int "
                    "flattened-pair limit; maximum " +
                        std::to_string(maxIntCount) + ", current " +
                        std::to_string(currentPairCount) + ", requested " +
                        std::to_string(atomPairs.size()));

  for (std::size_t pairIndex = 0; pairIndex < atomPairs.size(); pairIndex++) {
    const int firstAtomIndex = atomPairs[pairIndex][0];
    const int secondAtomIndex = atomPairs[pairIndex][1];

    APOCHARMM_REQUIRE(
        (firstAtomIndex >= 0) && (firstAtomIndex < m_NumAtoms),
        ApoCharmmErrorCode::InvalidArgument,
        "First atom index at pair index " + std::to_string(pairIndex) +
            " is out of range; expected [0, " + std::to_string(m_NumAtoms) +
            "), observed " + std::to_string(firstAtomIndex));

    APOCHARMM_REQUIRE(
        (secondAtomIndex >= 0) && (secondAtomIndex < m_NumAtoms),
        ApoCharmmErrorCode::InvalidArgument,
        "Second atom index at pair index " + std::to_string(pairIndex) +
            " is out of range; expected [0, " + std::to_string(m_NumAtoms) +
            "), observed " + std::to_string(secondAtomIndex));

    APOCHARMM_REQUIRE(
        firstAtomIndex != secondAtomIndex, ApoCharmmErrorCode::InvalidArgument,
        "Atom indices at pair index " + std::to_string(pairIndex) +
            " must be distinct; observed (" + std::to_string(firstAtomIndex) +
            ", " + std::to_string(secondAtomIndex) + ")");

    APOCHARMM_REQUIRE(std::isfinite(coefficients[pairIndex]),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Coefficient at pair index " + std::to_string(pairIndex) +
                          " must be finite; observed " +
                          std::to_string(coefficients[pairIndex]));

    APOCHARMM_REQUIRE(coefficients[pairIndex] != 0.0,
                      ApoCharmmErrorCode::InvalidArgument,
                      "Coefficient at pair index " + std::to_string(pairIndex) +
                          " must be nonzero");
  }

  APOCHARMM_REQUIRE(std::isfinite(forceConstant),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Force constant must be finite; observed " +
                        std::to_string(forceConstant));

  APOCHARMM_REQUIRE(forceConstant != 0.0, ApoCharmmErrorCode::InvalidArgument,
                    "Force constant must be nonzero");

  APOCHARMM_REQUIRE(std::isfinite(referenceValue),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference value must be finite; observed " +
                        std::to_string(referenceValue));

  const bool validDistanceExponent =
      (distanceExponent == -2) || (distanceExponent == -1) ||
      (distanceExponent == 0) || (distanceExponent == 1) ||
      (distanceExponent == 2) || (distanceExponent == 6) ||
      (distanceExponent == 7);
  APOCHARMM_REQUIRE(validDistanceExponent, ApoCharmmErrorCode::InvalidArgument,
                    "Distance exponent must be one of the characterized values "
                    "{-2, -1, 0, 1, 2, 6, 7}; observed " +
                        std::to_string(distanceExponent));

  const bool validEnergyExponent =
      (energyExponent == 1) || (energyExponent == 2) || (energyExponent == 3) ||
      (energyExponent == 4);
  APOCHARMM_REQUIRE(validEnergyExponent, ApoCharmmErrorCode::InvalidArgument,
                    "Energy exponent must be one of the characterized values "
                    "{1, 2, 3, 4}; observed " +
                        std::to_string(energyExponent));

  const int conditionValue = static_cast<int>(condition);
  APOCHARMM_REQUIRE(
      (condition == DistanceRestraintCondition::NONE) ||
          (condition == DistanceRestraintCondition::POSITIVE) ||
          (condition == DistanceRestraintCondition::NEGATIVE),
      ApoCharmmErrorCode::InvalidArgument,
      "Activation condition must be NONE, POSITIVE, or NEGATIVE; observed " +
          std::to_string(conditionValue));

  std::vector<int> termPairOffsets = m_TermPairOffsets.getHostArray();
  std::vector<int> firstAtomIndices = m_FirstAtomIndices.getHostArray();
  std::vector<int> secondAtomIndices = m_SecondAtomIndices.getHostArray();
  std::vector<double> pairCoefficients = m_PairCoefficients.getHostArray();
  std::vector<double> forceConstants = m_ForceConstants.getHostArray();
  std::vector<double> referenceValues = m_ReferenceValues.getHostArray();
  std::vector<int> distanceExponents = m_DistanceExponents.getHostArray();
  std::vector<int> energyExponents = m_EnergyExponents.getHostArray();
  std::vector<int> activationConditions = m_ActivationConditions.getHostArray();

  const std::size_t newPairCount = currentPairCount + atomPairs.size();
  const std::size_t newTermCount = currentTermCount + 1;

  firstAtomIndices.reserve(newPairCount);
  secondAtomIndices.reserve(newPairCount);
  pairCoefficients.reserve(newPairCount);
  forceConstants.reserve(newTermCount);
  referenceValues.reserve(newTermCount);
  distanceExponents.reserve(newTermCount);
  energyExponents.reserve(newTermCount);
  activationConditions.reserve(newTermCount);
  termPairOffsets.reserve(newTermCount + 1);

  for (const std::array<int, 2> &atomPair : atomPairs)
    firstAtomIndices.push_back(atomPair[0]);

  for (const std::array<int, 2> &atomPair : atomPairs)
    secondAtomIndices.push_back(atomPair[1]);

  pairCoefficients.insert(pairCoefficients.end(), coefficients.begin(),
                          coefficients.end());

  forceConstants.push_back(forceConstant);
  referenceValues.push_back(referenceValue);
  distanceExponents.push_back(distanceExponent);
  energyExponents.push_back(energyExponent);
  activationConditions.push_back(conditionValue);

  termPairOffsets.push_back(static_cast<int>(newPairCount));

  m_FirstAtomIndices.getHostArray().swap(firstAtomIndices);
  m_SecondAtomIndices.getHostArray().swap(secondAtomIndices);
  m_PairCoefficients.getHostArray().swap(pairCoefficients);
  m_ForceConstants.getHostArray().swap(forceConstants);
  m_ReferenceValues.getHostArray().swap(referenceValues);
  m_DistanceExponents.getHostArray().swap(distanceExponents);
  m_EnergyExponents.getHostArray().swap(energyExponents);
  m_ActivationConditions.getHostArray().swap(activationConditions);
  m_TermPairOffsets.getHostArray().swap(termPairOffsets);

  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::reset(void) {
  std::vector<int> termPairOffsets(1, 0);

  m_FirstAtomIndices.getHostArray().clear();
  m_SecondAtomIndices.getHostArray().clear();
  m_PairCoefficients.getHostArray().clear();
  m_ForceConstants.getHostArray().clear();
  m_ReferenceValues.getHostArray().clear();
  m_DistanceExponents.getHostArray().clear();
  m_EnergyExponents.getHostArray().clear();
  m_ActivationConditions.getHostArray().clear();
  m_TermPairOffsets.getHostArray().swap(termPairOffsets);

  m_Scale = 1.0;
  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::setBoxDimensions(
    const std::vector<double> &boxDimensions) {
  APOCHARMM_REQUIRE(boxDimensions.size() == 3,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Box-dimension array size mismatch; expected 3, observed " +
                        std::to_string(boxDimensions.size()));

  for (std::size_t i = 0; i < 3; i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(boxDimensions[i]), ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index " + std::to_string(i) +
            " must be finite; observed " + std::to_string(boxDimensions[i]));

    APOCHARMM_REQUIRE(
        boxDimensions[i] > 0.0, ApoCharmmErrorCode::InvalidArgument,
        "Box dimension at index " + std::to_string(i) +
            " must be positive; observed " + std::to_string(boxDimensions[i]));
  }

  m_BoxDimX = boxDimensions[0];
  m_BoxDimY = boxDimensions[1];
  m_BoxDimZ = boxDimensions[2];

  return;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::initialize(
    const int numAtoms, const std::vector<double> &boxDimensions) {
  APOCHARMM_REQUIRE(m_NumAtoms == numAtoms, ApoCharmmErrorCode::InvalidArgument,
                    "Initialization atom count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(numAtoms));

  this->setBoxDimensions(boxDimensions);

  return;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::clear(void) {
  m_EnergyVirial->clear(*m_Stream);
  m_Forces->clear(*m_Stream);
  return;
}

template <typename T>
static __forceinline__ __device__ T integerPower(const T base,
                                                 const int exponent) {
  T result = static_cast<T>(1);
  T factor = base;

  unsigned int remainingExponent = 0U;
  if (exponent < 0) {
    factor = static_cast<T>(1) / factor;
    remainingExponent =
        static_cast<unsigned int>(-static_cast<long long int>(exponent));
  } else
    remainingExponent = static_cast<unsigned int>(exponent);

  while (remainingExponent != 0U) {
    if ((remainingExponent & 1U) != 0U)
      result *= factor;

    remainingExponent >>= 1U;
    if (remainingExponent != 0U)
      factor *= factor;
  }

  return result;
}

template <typename AT, typename CT, int blockSize, bool calcEnergy,
          bool calcVirial>
__global__ static void DistanceRestraintForceKernel(
    AT *__restrict__ forces, double *__restrict__ energy, const int forceStride,
    const float4 *__restrict__ xyzq, const double scale,
    const int *__restrict__ termPairOffsets,
    const int *__restrict__ firstAtomIndices,
    const int *__restrict__ secondAtomIndices,
    const double *__restrict__ pairCoefficients,
    const double *__restrict__ forceConstants,
    const double *__restrict__ referenceValues,
    const int *__restrict__ distanceExponents,
    const int *__restrict__ energyExponents,
    const int *__restrict__ activationConditions) {
  const int pairBegin = termPairOffsets[blockIdx.x];
  const int pairEnd = termPairOffsets[blockIdx.x + 1];
  const int distanceExponent = distanceExponents[blockIdx.x];

  __shared__ BlockReduceSumStorage<double, blockSize, 1> cache;
  __shared__ double deviation;
  __shared__ double derivative;
  __shared__ int active;
  __shared__ int zeroDistanceDetected;

  if (threadIdx.x == 0)
    zeroDistanceDetected = 0;
  __syncthreads();

  double reactionCoordinate[1] = {0.0};
  for (int i = pairBegin + threadIdx.x; i < pairEnd; i += blockDim.x) {
    const int iatom = firstAtomIndices[i];
    const int jatom = secondAtomIndices[i];

    const double dx =
        static_cast<double>(xyzq[iatom].x) - static_cast<double>(xyzq[jatom].x);
    const double dy =
        static_cast<double>(xyzq[iatom].y) - static_cast<double>(xyzq[jatom].y);
    const double dz =
        static_cast<double>(xyzq[iatom].z) - static_cast<double>(xyzq[jatom].z);
    const double dr2 = dx * dx + dy * dy + dz * dz;

    if (dr2 == 0.0) {
      atomicExch(&zeroDistanceDetected, 1);
      continue;
    }

    const double dr = sqrt(dr2);
    reactionCoordinate[0] +=
        pairCoefficients[i] * integerPower<double>(dr, distanceExponent);
  }

  __syncthreads();

  if (zeroDistanceDetected != 0) {
    if (threadIdx.x == 0)
      asm volatile("trap;");
    return;
  }

  BlockReduceSum<double, blockSize, 1>(reactionCoordinate, cache);

  if (threadIdx.x == 0) {
    deviation = reactionCoordinate[0] - referenceValues[blockIdx.x];

    const int condition = activationConditions[blockIdx.x];
    active =
        (condition == static_cast<int>(DistanceRestraintCondition::NONE)) ||
        ((condition ==
          static_cast<int>(DistanceRestraintCondition::POSITIVE)) &&
         (deviation >= 0.0)) ||
        ((condition ==
          static_cast<int>(DistanceRestraintCondition::NEGATIVE)) &&
         (deviation <= 0.0));

    if (active != 0) {
      const double forceConstant = forceConstants[blockIdx.x];
      const int energyExponent = energyExponents[blockIdx.x];

      derivative = scale * forceConstant *
                   integerPower<double>(deviation, energyExponent - 1);

      if constexpr (calcEnergy) {
        const double en =
            scale * (forceConstant / static_cast<double>(energyExponent)) *
            integerPower<double>(deviation, energyExponent);
        atomicAdd(energy, en);
      }
    } else {
      derivative = 0.0;
    }
  }
  __syncthreads();

  if (active == 0)
    return;

  for (int i = pairBegin + threadIdx.x; i < pairEnd; i += blockDim.x) {
    const int iatom = firstAtomIndices[i];
    const int jatom = secondAtomIndices[i];

    const double dx =
        static_cast<double>(xyzq[iatom].x) - static_cast<double>(xyzq[jatom].x);
    const double dy =
        static_cast<double>(xyzq[iatom].y) - static_cast<double>(xyzq[jatom].y);
    const double dz =
        static_cast<double>(xyzq[iatom].z) - static_cast<double>(xyzq[jatom].z);
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const double dr = sqrt(dr2);

    if (distanceExponent == 0)
      continue;

    const double pairGradientCoefficient =
        derivative * pairCoefficients[i] *
        static_cast<double>(distanceExponent) *
        integerPower<double>(dr, distanceExponent - 2);

    AT gradX = static_cast<AT>(0);
    AT gradY = static_cast<AT>(0);
    AT gradZ = static_cast<AT>(0);
    calc_component_force<AT, CT>(static_cast<CT>(pairGradientCoefficient),
                                 static_cast<CT>(dx), static_cast<CT>(dy),
                                 static_cast<CT>(dz), gradX, gradY, gradZ);

    write_force<AT>(gradX, gradY, gradZ, iatom, forceStride, forces);
    write_force<AT>(-gradX, -gradY, -gradZ, jatom, forceStride, forces);
  }

  if constexpr (calcVirial) {
    // JEG260901: Raw-coordinate RESD creates no shifted-force virial record.
    // ForceManager constructs the final virial from these gradients and the
    // primary coordinates.
  }

  return;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::calcForce(const float4 *xyzq,
                                               const bool calcEnergy,
                                               const bool calcVirial) {
  const int numTerms = static_cast<int>(m_ForceConstants.size());
  if ((numTerms == 0) || (m_Scale == 0.0))
    return;

  this->syncDeviceData();

  constexpr int numThreads = 256;
  const int numBlocks = numTerms;

  if ((calcEnergy == true) && (calcVirial == true)) {
    cudaCheckLaunch(
        DistanceRestraintForceKernel<AT, CT, numThreads, true, true>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_EnergyVirial->getEnergyPointer("resd"),
            m_Forces->stride(), xyzq, m_Scale,
            m_TermPairOffsets.getDeviceArray().data(),
            m_FirstAtomIndices.getDeviceArray().data(),
            m_SecondAtomIndices.getDeviceArray().data(),
            m_PairCoefficients.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_ReferenceValues.getDeviceArray().data(),
            m_DistanceExponents.getDeviceArray().data(),
            m_EnergyExponents.getDeviceArray().data(),
            m_ActivationConditions.getDeviceArray().data()));
  } else if ((calcEnergy == true) && (calcVirial == false)) {
    cudaCheckLaunch(
        DistanceRestraintForceKernel<AT, CT, numThreads, true, false>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_EnergyVirial->getEnergyPointer("resd"),
            m_Forces->stride(), xyzq, m_Scale,
            m_TermPairOffsets.getDeviceArray().data(),
            m_FirstAtomIndices.getDeviceArray().data(),
            m_SecondAtomIndices.getDeviceArray().data(),
            m_PairCoefficients.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_ReferenceValues.getDeviceArray().data(),
            m_DistanceExponents.getDeviceArray().data(),
            m_EnergyExponents.getDeviceArray().data(),
            m_ActivationConditions.getDeviceArray().data()));
  } else if ((calcEnergy == false) && (calcVirial == true)) {
    cudaCheckLaunch(
        DistanceRestraintForceKernel<AT, CT, numThreads, false, true>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_EnergyVirial->getEnergyPointer("resd"),
            m_Forces->stride(), xyzq, m_Scale,
            m_TermPairOffsets.getDeviceArray().data(),
            m_FirstAtomIndices.getDeviceArray().data(),
            m_SecondAtomIndices.getDeviceArray().data(),
            m_PairCoefficients.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_ReferenceValues.getDeviceArray().data(),
            m_DistanceExponents.getDeviceArray().data(),
            m_EnergyExponents.getDeviceArray().data(),
            m_ActivationConditions.getDeviceArray().data()));
  } else if ((calcEnergy == false) && (calcVirial == false)) {
    cudaCheckLaunch(
        DistanceRestraintForceKernel<AT, CT, numThreads, false, false>
        <<<numBlocks, numThreads, 0, *m_Stream>>>(
            m_Forces->xyz(), m_EnergyVirial->getEnergyPointer("resd"),
            m_Forces->stride(), xyzq, m_Scale,
            m_TermPairOffsets.getDeviceArray().data(),
            m_FirstAtomIndices.getDeviceArray().data(),
            m_SecondAtomIndices.getDeviceArray().data(),
            m_PairCoefficients.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_ReferenceValues.getDeviceArray().data(),
            m_DistanceExponents.getDeviceArray().data(),
            m_EnergyExponents.getDeviceArray().data(),
            m_ActivationConditions.getDeviceArray().data()));
  }

  return;
}

template <typename AT, typename CT>
std::shared_ptr<cudaStream_t> DistanceRestraintForce<AT, CT>::getStream(void) {
  return m_Stream;
}

template <typename AT, typename CT>
std::shared_ptr<Force<AT>> DistanceRestraintForce<AT, CT>::getForce(void) {
  return m_Forces;
}

template <typename AT, typename CT>
std::shared_ptr<CudaEnergyVirial>
DistanceRestraintForce<AT, CT>::getEnergyVirial(void) {
  return m_EnergyVirial;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::syncDeviceData(void) {
  if (m_DeviceDataDirty == false)
    return;

  m_TermPairOffsets.resize(m_TermPairOffsets.size());
  m_FirstAtomIndices.resize(m_FirstAtomIndices.size());
  m_SecondAtomIndices.resize(m_SecondAtomIndices.size());
  m_PairCoefficients.resize(m_PairCoefficients.size());
  m_ForceConstants.resize(m_ForceConstants.size());
  m_ReferenceValues.resize(m_ReferenceValues.size());
  m_DistanceExponents.resize(m_DistanceExponents.size());
  m_EnergyExponents.resize(m_EnergyExponents.size());
  m_ActivationConditions.resize(m_ActivationConditions.size());

  m_TermPairOffsets.transferToDevice();
  m_FirstAtomIndices.transferToDevice();
  m_SecondAtomIndices.transferToDevice();
  m_PairCoefficients.transferToDevice();
  m_ForceConstants.transferToDevice();
  m_ReferenceValues.transferToDevice();
  m_DistanceExponents.transferToDevice();
  m_EnergyExponents.transferToDevice();
  m_ActivationConditions.transferToDevice();

  m_DeviceDataDirty = false;

  return;
}

template <typename AT, typename CT>
void DistanceRestraintForce<AT, CT>::dealloc(void) noexcept {
  if (m_Stream != nullptr) {
    destroy_cuda_stream_noexcept(m_Stream.get());
    m_Stream.reset();
  }
  return;
}

//
// Explicit instances of DistanceRestraintForce
//
template class DistanceRestraintForce<long long int, float>;
template class DistanceRestraintForce<long long int, double>;
