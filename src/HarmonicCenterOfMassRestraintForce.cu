// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II
//
// ENDLICENSE

#include "HarmonicCenterOfMassRestraintForce.h"

#include "ApoCharmmError.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

template <typename AT, typename CT>
HarmonicCenterOfMassRestraintForce<AT, CT>::HarmonicCenterOfMassRestraintForce(
    const std::shared_ptr<const CharmmPSF> &psf)
    : m_NumAtoms(0), m_NumSelected(0), m_ForceConstant(0.0),
      m_ReferenceDistance(0.0), m_UseMassWeighting(false), m_AtomIndices(),
      m_AtomWeights(), m_Masses(), m_PartialSums(), m_RestraintState(),
      m_BoxDimensions{0.0, 0.0, 0.0}, m_DeviceDataDirty(true),
      m_ReferencePosition{0.0, 0.0, 0.0}, m_ReferenceMask{1, 1, 1},
      m_Selection(0, AtomSelection::InitialValue::ALL), m_EnergyVirial(nullptr),
      m_Forces(nullptr), m_Stream(nullptr) {
  APOCHARMM_REQUIRE(psf != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "CharmmPSF must not be null");

  const int numAtoms = psf->getNumAtoms();
  APOCHARMM_REQUIRE(numAtoms >= 0, ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF atom count is not initialized; observed " +
                        std::to_string(numAtoms));
  APOCHARMM_REQUIRE(numAtoms > 0, ApoCharmmErrorCode::InvalidArgument,
                    "Atom count must be positive; observed " +
                        std::to_string(numAtoms));

  const std::vector<double> &masses = psf->getMasses();
  APOCHARMM_REQUIRE(masses.size() == static_cast<std::size_t>(numAtoms),
                    ApoCharmmErrorCode::Runtime,
                    "PSF mass count mismatch; expected " +
                        std::to_string(numAtoms) + ", observed " +
                        std::to_string(masses.size()));
  for (std::size_t i = 0; i < masses.size(); i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(masses[i]), ApoCharmmErrorCode::InvalidArgument,
        "Mass at index " + std::to_string(i) + " must be finite; observed " +
            std::to_string(masses[i]));
  }

  m_NumAtoms = numAtoms;

  m_Masses = masses;
  m_PartialSums.resize(1);
  m_RestraintState.resize(1);
  m_Selection.setNumAtoms(numAtoms, AtomSelection::InitialValue::ALL);

  m_PartialSums.set(CT4{});
  m_RestraintState.set(CT4{});

  this->updateSelectedAtoms();

  m_EnergyVirial = std::make_shared<CudaEnergyVirial>();
  m_EnergyVirial->insert("hmcm");

  m_Forces = std::make_shared<Force<AT>>();
  m_Forces->realloc(numAtoms, 1.5f);

  m_Stream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_Stream.get()));
}

template <typename AT, typename CT>
HarmonicCenterOfMassRestraintForce<AT, CT>::~HarmonicCenterOfMassRestraintForce(
    void) noexcept {
  if (m_Stream != nullptr) {
    destroy_cuda_stream_noexcept(m_Stream.get());
    m_Stream.reset();
  }
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setSelection(
    const AtomSelection &selection) {
  APOCHARMM_REQUIRE(selection.getNumAtoms() == m_NumAtoms,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Selection atom count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(selection.getNumAtoms()));

  APOCHARMM_REQUIRE(selection.getNumSelected() > 0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Selection must contain at least one atom; observed " +
                        std::to_string(selection.getNumSelected()));

  m_Selection = selection;
  this->updateSelectedAtoms();

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setForceConstant(
    const double forceConstant) {
  APOCHARMM_REQUIRE(std::isfinite(forceConstant),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Force constant must be finite; observed " +
                        std::to_string(forceConstant));

  m_ForceConstant = forceConstant;

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setReferencePosition(
    const std::vector<double> &referencePosition) {
  this->setReferencePosition(referencePosition, {1, 1, 1});
  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setReferencePosition(
    const std::vector<double> &referencePosition,
    const std::vector<int> &referenceMask) {
  APOCHARMM_REQUIRE(
      referencePosition.size() == 3, ApoCharmmErrorCode::InvalidArgument,
      "Reference-position array size mismatch; expected 3, observed " +
          std::to_string(referencePosition.size()));

  APOCHARMM_REQUIRE(
      referenceMask.size() == 3, ApoCharmmErrorCode::InvalidArgument,
      "Reference-mask array size mismatch; expected 3, observed " +
          std::to_string(referenceMask.size()));

  int numActiveComponents = 0;
  for (std::size_t i = 0; i < 3; i++) {
    APOCHARMM_REQUIRE(std::isfinite(referencePosition[i]),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Reference position at index " + std::to_string(i) +
                          " must be finite; observed " +
                          std::to_string(referencePosition[i]));

    APOCHARMM_REQUIRE((referenceMask[i] == 0) || (referenceMask[i] == 1),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Reference mask at index " + std::to_string(i) +
                          " must be 0 or 1; observed " +
                          std::to_string(referenceMask[i]));

    numActiveComponents += referenceMask[i];
  }

  APOCHARMM_REQUIRE(numActiveComponents > 0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference mask must activate at least one coordinate");

  std::copy(referencePosition.begin(), referencePosition.end(),
            m_ReferencePosition.begin());

  std::copy(referenceMask.begin(), referenceMask.end(),
            m_ReferenceMask.begin());

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setReferenceDistance(
    const double referenceDistance) {
  APOCHARMM_REQUIRE(std::isfinite(referenceDistance),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference distance must be finite; observed " +
                        std::to_string(referenceDistance));

  m_ReferenceDistance = referenceDistance;

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setMasses(
    const std::vector<double> &masses) {
  APOCHARMM_REQUIRE(masses.size() == static_cast<std::size_t>(m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Mass array size mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(masses.size()));

  for (std::size_t i = 0; i < masses.size(); i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(masses[i]), ApoCharmmErrorCode::InvalidArgument,
        "Mass at index " + std::to_string(i) + " must be finite; observed " +
            std::to_string(masses[i]));
  }

  m_Masses = masses;
  m_UseMassWeighting = true;
  this->updateSelectedAtoms();

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setMassWeighting(
    const bool useMassWeighting) {
  m_UseMassWeighting = useMassWeighting;
  this->updateSelectedAtoms();
  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::setBoxDimensions(
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

  if ((m_BoxDimensions[0] == boxDimensions[0]) &&
      (m_BoxDimensions[1] == boxDimensions[1]) &&
      (m_BoxDimensions[2] == boxDimensions[2]))
    return;

  std::copy(boxDimensions.begin(), boxDimensions.end(),
            m_BoxDimensions.begin());

  return;
}

template <typename AT, typename CT>
std::shared_ptr<CudaEnergyVirial>
HarmonicCenterOfMassRestraintForce<AT, CT>::getEnergyVirial(void) {
  return m_EnergyVirial;
}

template <typename AT, typename CT>
std::shared_ptr<Force<AT>>
HarmonicCenterOfMassRestraintForce<AT, CT>::getForce(void) {
  return m_Forces;
}

template <typename AT, typename CT>
std::shared_ptr<cudaStream_t>
HarmonicCenterOfMassRestraintForce<AT, CT>::getStream(void) {
  return m_Stream;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::initialize(
    const int numAtoms, const std::vector<double> &boxDimensions) {
  APOCHARMM_REQUIRE(m_NumAtoms == numAtoms, ApoCharmmErrorCode::InvalidArgument,
                    "Initialization atom count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(numAtoms));

  this->setBoxDimensions(boxDimensions);
  this->syncDeviceData();

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::clear(void) {
  m_EnergyVirial->clear(*m_Stream);
  m_Forces->clear(*m_Stream);
  return;
}

/**
 * @brief Reduces raw selected coordinates and weights by CUDA block.
 *
 * As in CHARMM, the selected coordinates are summed without periodic
 * unwrapping. Each thread accumulates `w_i * r_i` and `w_i` in `CT`, and the
 * four independent quantities are reduced together across each block. One
 * record is written per launch block as `{sum_x, sum_y, sum_z, sum_weight}`.
 *
 * @tparam CT Arithmetic representation of the products and the reduction.
 * @tparam CT4 Four-component vector type matching `CT`.
 * @tparam blockSize Compile-time number of threads in each CUDA block. It must
 * be at least `warpsize`, no greater than 1024, and an integer multiple of
 * `warpsize`.
 * @param[out] partialSums Non-NULL device array with at least `gridDim.x`
 * elements. Element `blockIdx.x` is overwritten by thread zero in that block.
 * @param[in] atomIndices Non-NULL device array of `numSelected` zero-based atom
 * indices.
 * @param[in] atomWeights Non-NULL device array of `numSelected` finite weights
 * corresponding to `atomIndices`.
 * @param[in] xyzq Non-NULL device array containing every indexed `float4`
 * coordinate-charge record. Cartesian components use angstroms; component `w`
 * is ignored.
 * @param[in] numSelected Positive dimensionless selected-atom count.
 *
 * @pre The kernel is launched with one-dimensional blocks satisfying
 * `blockDim.x == blockSize`.
 * @pre Every thread in each block reaches the block-wide reduction.
 * @pre The three input arrays remain valid for the complete kernel execution.
 */
template <typename CT, typename CT4, int blockSize>
__global__ static void PartialSumsKernel(CT4 *__restrict__ partialSums,
                                         const int *__restrict__ atomIndices,
                                         const double *__restrict__ atomWeights,
                                         const float4 *__restrict__ xyzq,
                                         const int numSelected) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  CT sums[4] = {static_cast<CT>(0), static_cast<CT>(0), static_cast<CT>(0),
                static_cast<CT>(0)};
  for (int i = index; i < numSelected; i += stride) {
    const CT weight = static_cast<CT>(atomWeights[i]);
    const float4 xyzqi = xyzq[atomIndices[i]];

    sums[0] += weight * static_cast<CT>(xyzqi.x);
    sums[1] += weight * static_cast<CT>(xyzqi.y);
    sums[2] += weight * static_cast<CT>(xyzqi.z);
    sums[3] += weight;
  }

  __shared__ BlockReduceSumStorage<CT, blockSize, 4> cache;
  BlockReduceSum<CT, blockSize, 4>(sums, cache);

  if (threadIdx.x == 0)
    partialSums[blockIdx.x] = CT4{sums[0], sums[1], sums[2], sums[3]};

  return;
}

/**
 * @brief Converts block partials into gradient factors and optional energy.
 *
 * The kernel reduces all block records and normalizes the raw center. With the
 * active components of `d = C - R`, `s = norm(d)`, and the deviation `s - r0`,
 * it writes the per-axis factors `d_a * 2 * k / W * (s - r0) / s` in CHARMM's
 * operation order to state components x, y, and z. A zero `s` writes zero
 * factors. When requested, thread zero converts `k * (s - r0) * (s - r0)` to
 * double and adds it to `energy`.
 *
 * @tparam CT Arithmetic representation of the reduction, the center, the
 * gradient factors, and the energy value.
 * @tparam CT4 Four-component vector type matching `CT`.
 * @tparam blockSize Compile-time number of threads in the CUDA block. It must
 * be at least `warpsize`, no greater than 1024, and an integer multiple of
 * `warpsize`.
 * @tparam calcEnergy Whether thread zero atomically adds the current energy to
 * `energy`.
 * @param[out] restraintState Non-NULL device array of at least one record. The
 * kernel overwrites element zero.
 * @param[in,out] energy Non-NULL device pointer to the `hmcm` scalar in
 * kilocalories per mole. It is modified only when `calcEnergy` is `true`.
 * @param[in] partialSums Non-NULL device array of `numPartialSums` block
 * records produced by `PartialSumsKernel`.
 * @param[in] numPartialSums Positive dimensionless partial-record count.
 * @param[in] forceConstant Coefficient in kilocalories per mole per square
 * angstrom.
 * @param[in] referencePosition Reference `[x, y, z]` position in angstroms;
 * component `w` is ignored.
 * @param[in] referenceMaskX Dimensionless X activation flag, zero or one.
 * @param[in] referenceMaskY Dimensionless Y activation flag, zero or one.
 * @param[in] referenceMaskZ Dimensionless Z activation flag, zero or one.
 * @param[in] referenceDistance Target distance in angstroms.
 *
 * @pre The kernel is launched with exactly one one-dimensional block satisfying
 * `blockDim.x == blockSize`.
 * @pre Every thread in the block reaches the block-wide reduction.
 */
template <typename CT, typename CT4, int blockSize, bool calcEnergy>
__global__ static void
StateKernel(CT4 *__restrict__ restraintState, double *__restrict__ energy,
            const CT4 *__restrict__ partialSums, const int numPartialSums,
            const CT forceConstant, const CT4 referencePosition,
            const int referenceMaskX, const int referenceMaskY,
            const int referenceMaskZ, const CT referenceDistance) {
  constexpr CT ZERO = static_cast<CT>(0);

  CT sums[4] = {static_cast<CT>(0), static_cast<CT>(0), static_cast<CT>(0),
                static_cast<CT>(0)};
  for (int i = threadIdx.x; i < numPartialSums; i += blockDim.x) {
    const CT4 partialSum = partialSums[i];
    sums[0] += partialSum.x;
    sums[1] += partialSum.y;
    sums[2] += partialSum.z;
    sums[3] += partialSum.w;
  }

  __shared__ BlockReduceSumStorage<CT, blockSize, 4> cache;
  BlockReduceSum<CT, blockSize, 4>(sums, cache);

  if (threadIdx.x == 0) {
    const CT totalWeight = sums[3];

    CT dx = ZERO, dy = ZERO, dz = ZERO;
    if (referenceMaskX != 0)
      dx = sums[0] / totalWeight - referencePosition.x;
    if (referenceMaskY != 0)
      dy = sums[1] / totalWeight - referencePosition.y;
    if (referenceMaskZ != 0)
      dz = sums[2] / totalWeight - referencePosition.z;

    const CT distance = sqrt(dx * dx + dy * dy + dz * dz);
    const CT deviation = distance - referenceDistance;

    // CHARMM gives zero factors at a zero distance only when the deviation is
    // also zero and divides by zero otherwise. Both cases use zero here.
    CT4 factors = CT4{ZERO, ZERO, ZERO, ZERO};
    if (distance != ZERO) {
      constexpr CT TWO = static_cast<CT>(2);
      factors.x = dx * TWO * forceConstant / totalWeight * deviation / distance;
      factors.y = dy * TWO * forceConstant / totalWeight * deviation / distance;
      factors.z = dz * TWO * forceConstant / totalWeight * deviation / distance;
    }
    restraintState[0] = factors;

    if constexpr (calcEnergy)
      atomicAdd(energy,
                static_cast<double>(forceConstant * deviation * deviation));
  }

  return;
}

/**
 * @brief Distributes the gradient factors to the selected atoms.
 *
 * As in CHARMM, selected atom `i` receives its weight times the per-axis
 * factors from `StateKernel`. Each component is rounded once to the
 * fixed-point force representation and added atomically.
 *
 * @tparam AT Device force-accumulator representation written to `forces`.
 * @tparam CT Arithmetic representation of the weight-factor products.
 * @tparam CT4 Four-component vector type matching `CT`.
 * @param[in,out] forces Non-NULL device structure-of-arrays storage containing
 * at least `3 * forceStride` elements. Contributions are added atomically.
 * @param[in] forceStride Dimensionless element offset between X, Y, and Z
 * component arrays.
 * @param[in] atomIndices Non-NULL device array of `numSelected` zero-based atom
 * indices.
 * @param[in] atomWeights Non-NULL device array of `numSelected` weights
 * corresponding to `atomIndices`.
 * @param[in] numSelected Positive dimensionless selected-atom count.
 * @param[in] restraintState Non-NULL device array whose element zero was
 * produced by `StateKernel`.
 */
template <typename AT, typename CT, typename CT4>
__global__ static void
ApplyForcesKernel(AT *__restrict__ forces, const int forceStride,
                  const int *__restrict__ atomIndices,
                  const double *__restrict__ atomWeights, const int numSelected,
                  const CT4 *__restrict__ restraintState) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  // Gradient state
  const CT4 factors = restraintState[0];

  for (int i = index; i < numSelected; i += stride) {
    AT gradX = static_cast<AT>(0);
    AT gradY = static_cast<AT>(0);
    AT gradZ = static_cast<AT>(0);

    calc_component_force<AT, CT>(static_cast<CT>(atomWeights[i]), factors.x,
                                 factors.y, factors.z, gradX, gradY, gradZ);

    write_force<AT>(gradX, gradY, gradZ, atomIndices[i], forceStride, forces);
  }

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::calcForce(
    const float4 *xyzq, const bool calcEnergy, const bool calcVirial) {
  static_cast<void>(calcVirial);

  if (m_NumSelected <= 0 || m_ForceConstant == 0.0)
    return;

  this->syncDeviceData();

  constexpr int numThreads = 256;
  const int numBlocks = (m_NumSelected + numThreads - 1) / numThreads;
  const int numPartialSums = static_cast<int>(m_PartialSums.size());
  const CT4 referencePosition = {static_cast<CT>(m_ReferencePosition[0]),
                                 static_cast<CT>(m_ReferencePosition[1]),
                                 static_cast<CT>(m_ReferencePosition[2]),
                                 static_cast<CT>(0)};

  // One stream establishes the reduction -> state -> distribution dependency
  // without host synchronization. Immediate launch checks do not report later
  // asynchronous execution failures.
  cudaCheckLaunch(
      PartialSumsKernel<CT, CT4, numThreads>
      <<<numBlocks, numThreads, 0, *m_Stream>>>(
          m_PartialSums.getDeviceArray().data(),
          m_AtomIndices.getDeviceArray().data(),
          m_AtomWeights.getDeviceArray().data(), xyzq, m_NumSelected));

  if (calcEnergy) {
    cudaCheckLaunch(
        StateKernel<CT, CT4, numThreads, true><<<1, numThreads, 0, *m_Stream>>>(
            m_RestraintState.getDeviceArray().data(),
            m_EnergyVirial->getEnergyPointer("hmcm"),
            m_PartialSums.getDeviceArray().data(), numPartialSums,
            static_cast<CT>(m_ForceConstant), referencePosition,
            m_ReferenceMask[0], m_ReferenceMask[1], m_ReferenceMask[2],
            static_cast<CT>(m_ReferenceDistance)));
  } else {
    cudaCheckLaunch(
        StateKernel<CT, CT4, numThreads, false>
        <<<1, numThreads, 0, *m_Stream>>>(
            m_RestraintState.getDeviceArray().data(),
            m_EnergyVirial->getEnergyPointer("hmcm"),
            m_PartialSums.getDeviceArray().data(), numPartialSums,
            static_cast<CT>(m_ForceConstant), referencePosition,
            m_ReferenceMask[0], m_ReferenceMask[1], m_ReferenceMask[2],
            static_cast<CT>(m_ReferenceDistance)));
  }

  cudaCheckLaunch(ApplyForcesKernel<AT, CT, CT4>
                  <<<numBlocks, numThreads, 0, *m_Stream>>>(
                      m_Forces->xyz(), m_Forces->stride(),
                      m_AtomIndices.getDeviceArray().data(),
                      m_AtomWeights.getDeviceArray().data(), m_NumSelected,
                      m_RestraintState.getDeviceArray().data()));

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::updateSelectedAtoms(void) {
  const std::vector<int> selectedAtomIndices = m_Selection.getAtomIndices();

  m_NumSelected = static_cast<int>(selectedAtomIndices.size());

  std::vector<double> selectedAtomWeights(selectedAtomIndices.size());
  double totalWeight = 0.0;
  for (std::size_t i = 0; i < selectedAtomIndices.size(); i++) {
    const int atomIndex = selectedAtomIndices[i];
    const double weight =
        (m_UseMassWeighting == true) ? m_Masses[atomIndex] : 1.0;
    selectedAtomWeights[i] = weight;
    totalWeight += weight;
  }

  APOCHARMM_REQUIRE(
      std::isfinite(totalWeight) && totalWeight != 0.0,
      ApoCharmmErrorCode::InvalidArgument,
      "Selected atoms must have a finite nonzero total weight; observed " +
          std::to_string(totalWeight));

  for (std::size_t i = 0; i < selectedAtomWeights.size(); i++) {
    const double fraction = selectedAtomWeights[i] / totalWeight;
    APOCHARMM_REQUIRE(
        std::isfinite(fraction), ApoCharmmErrorCode::InvalidArgument,
        "Weight fraction at atom index " +
            std::to_string(selectedAtomIndices[i]) +
            " must be finite; observed " + std::to_string(fraction));
  }

  m_AtomIndices.getHostArray() = selectedAtomIndices;
  m_AtomWeights.getHostArray() = selectedAtomWeights;

  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void HarmonicCenterOfMassRestraintForce<AT, CT>::syncDeviceData(void) {
  if (!m_DeviceDataDirty)
    return;

  m_AtomIndices.resize(m_AtomIndices.size());
  m_AtomWeights.resize(m_AtomWeights.size());
  m_AtomIndices.transferToDeviceAsync(*m_Stream);
  m_AtomWeights.transferToDeviceAsync(*m_Stream);

  constexpr int numThreads = 256;
  const int numBlocks = (m_NumSelected + numThreads - 1) / numThreads;

  // The host copies of both reduction records are zero and never written, so
  // clearing the device copies keeps the mirrors equal.
  m_PartialSums.resize(numBlocks);
  clear_gpu_array<CT4>(m_PartialSums.getDeviceArray().data(),
                       m_PartialSums.size(), *m_Stream);
  clear_gpu_array<CT4>(m_RestraintState.getDeviceArray().data(),
                       m_RestraintState.size(), *m_Stream);
  cudaCheck(cudaStreamSynchronize(*m_Stream));

  m_DeviceDataDirty = false;

  return;
}

//
// Explicit instances of HarmonicCenterOfMassRestraintForce
//
template class HarmonicCenterOfMassRestraintForce<long long int, float>;
template class HarmonicCenterOfMassRestraintForce<long long int, double>;
