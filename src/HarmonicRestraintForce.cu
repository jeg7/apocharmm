// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "HarmonicRestraintForce.h"

#include "ApoCharmmError.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

template <typename AT, typename CT>
HarmonicRestraintForce<AT, CT>::HarmonicRestraintForce(
    const std::shared_ptr<const CharmmPSF> &psf)
    : m_NumAtoms(0), m_ForceConstants(), m_ReferenceCoordinates(), m_Masses(),
      m_BoxDimensions{0.0, 0.0, 0.0}, m_DeviceDataDirty(false), m_Exponent(2),
      m_ScaleFactors{1.0, 1.0, 1.0},
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

  m_NumAtoms = numAtoms;
  m_ForceConstants.resize(static_cast<std::size_t>(numAtoms));
  m_ReferenceCoordinates.resize(static_cast<std::size_t>(numAtoms));
  m_Masses.resize(static_cast<std::size_t>(numAtoms));
  m_Selection.setNumAtoms(numAtoms, AtomSelection::InitialValue::ALL);

  m_ForceConstants.set(0.0);

  m_ReferenceCoordinates.set(make_double3(0.0, 0.0, 0.0));

  m_Masses.set(1.0);

  m_EnergyVirial = std::make_shared<CudaEnergyVirial>();
  m_EnergyVirial->insert("harm");

  m_Forces = std::make_shared<Force<AT>>();
  m_Forces->realloc(numAtoms, 1.5f);

  m_Stream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_Stream.get()));
}

template <typename AT, typename CT>
HarmonicRestraintForce<AT, CT>::~HarmonicRestraintForce(void) noexcept {
  if (m_Stream != nullptr) {
    destroy_cuda_stream_noexcept(m_Stream.get());
    m_Stream.reset();
  }
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setSelection(
    const AtomSelection &selection) {
  APOCHARMM_REQUIRE(selection.getNumAtoms() == m_NumAtoms,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Selection atom count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(selection.getNumAtoms()));

  m_Selection = selection;

  for (int i = 0; i < m_NumAtoms; i++) {
    if (!selection.contains(i))
      m_ForceConstants[i] = 0.0;
  }

  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setForceConstant(
    const double forceConstant) {
  APOCHARMM_REQUIRE(std::isfinite(forceConstant),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Force constant must be finite; observed " +
                        std::to_string(forceConstant));

  APOCHARMM_REQUIRE(forceConstant >= 0.0, ApoCharmmErrorCode::InvalidArgument,
                    "Force constant must be non-negative; observed " +
                        std::to_string(forceConstant));

  std::fill(m_ForceConstants.getHostArray().begin(),
            m_ForceConstants.getHostArray().end(), 0.0);

  for (const int i : m_Selection.getAtomIndices())
    m_ForceConstants[i] = forceConstant;

  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setForceConstants(
    const std::vector<double> &forceConstants) {
  APOCHARMM_REQUIRE(forceConstants.size() ==
                        static_cast<std::size_t>(m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Force-constant array size mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(forceConstants.size()));

  const std::vector<int> selectedAtomIndices = m_Selection.getAtomIndices();

  for (const int i : selectedAtomIndices) {
    APOCHARMM_REQUIRE(
        std::isfinite(forceConstants[i]), ApoCharmmErrorCode::InvalidArgument,
        "Force constant at index " + std::to_string(i) +
            " must be finite; observed " + std::to_string(forceConstants[i]));
  }

  std::fill(m_ForceConstants.getHostArray().begin(),
            m_ForceConstants.getHostArray().end(), 0.0);

  for (const int i : selectedAtomIndices)
    m_ForceConstants[i] = forceConstants[i];

  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setReferenceCoordinates(
    const std::vector<double3> &referenceCoordinates) {
  APOCHARMM_REQUIRE(referenceCoordinates.size() ==
                        static_cast<std::size_t>(m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference-coordinate array size mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(referenceCoordinates.size()));

  for (int i = 0; i < m_NumAtoms; i++) {
    APOCHARMM_REQUIRE(std::isfinite(referenceCoordinates[i].x),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Reference coordinate at atom index " +
                          std::to_string(i) +
                          ", X component must be finite; observed " +
                          std::to_string(referenceCoordinates[i].x));

    APOCHARMM_REQUIRE(std::isfinite(referenceCoordinates[i].y),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Reference coordinate at atom index " +
                          std::to_string(i) +
                          ", Y component must be finite; observed " +
                          std::to_string(referenceCoordinates[i].y));

    APOCHARMM_REQUIRE(std::isfinite(referenceCoordinates[i].z),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Reference coordinate at atom index " +
                          std::to_string(i) +
                          ", Z component must be finite; observed " +
                          std::to_string(referenceCoordinates[i].z));

    m_ReferenceCoordinates[i] = referenceCoordinates[i];
  }

  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setReferenceCoordinates(
    const std::vector<std::vector<double>> &referenceCoordinates) {
  APOCHARMM_REQUIRE(referenceCoordinates.size() ==
                        static_cast<std::size_t>(m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Reference-coordinate array size mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(referenceCoordinates.size()));

  std::vector<double3> convertedCoordinates(referenceCoordinates.size());

  for (int i = 0; i < m_NumAtoms; i++) {
    APOCHARMM_REQUIRE(referenceCoordinates[i].size() == 3,
                      ApoCharmmErrorCode::InvalidArgument,
                      "Reference coordinate at atom index " +
                          std::to_string(i) +
                          " has invalid size; expected 3, observed " +
                          std::to_string(referenceCoordinates[i].size()));

    convertedCoordinates[i] =
        make_double3(referenceCoordinates[i][0], referenceCoordinates[i][1],
                     referenceCoordinates[i][2]);
  }

  this->setReferenceCoordinates(convertedCoordinates);

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setMasses(
    const std::vector<double> &masses) {
  APOCHARMM_REQUIRE(masses.size() == static_cast<std::size_t>(m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Mass array size mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(masses.size()));

  for (int i = 0; i < m_NumAtoms; i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(masses[i]), ApoCharmmErrorCode::InvalidArgument,
        "Mass at index " + std::to_string(i) + " must be finite; observed " +
            std::to_string(masses[i]));

    m_Masses[i] = masses[i];
  }

  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setExponent(const int exponent) {
  m_Exponent = exponent;
  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setScaleFactors(
    const std::vector<double> &scaleFactors) {
  APOCHARMM_REQUIRE(scaleFactors.size() == 3,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Scale-factor array size mismatch; expected 3, observed " +
                        std::to_string(scaleFactors.size()));

  for (std::size_t i = 0; i < 3; i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(scaleFactors[i]), ApoCharmmErrorCode::InvalidArgument,
        "Scale factor at index " + std::to_string(i) +
            " must be finite; observed " + std::to_string(scaleFactors[i]));
  }

  for (std::size_t i = 0; i < 3; i++)
    m_ScaleFactors[i] = scaleFactors[i];

  return;
}

template <typename AT, typename CT>
std::shared_ptr<cudaStream_t> HarmonicRestraintForce<AT, CT>::getStream(void) {
  return m_Stream;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::initialize(
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
void HarmonicRestraintForce<AT, CT>::clear(void) {
  m_EnergyVirial->clear(*m_Stream);
  m_Forces->clear(*m_Stream);
  return;
}

/**
 * @brief Accumulates CONS HARM ABSOLUTE gradients and optional energy.
 *
 * Each thread follows CHARMM's per-atom branches and operation order: atoms
 * whose product `C = k * m` is zero are skipped, `S` is the scaled squared
 * displacement, and the exponent selects the energy and gradient factor. Each
 * gradient component is rounded once to the fixed-point force representation
 * and added atomically. Each atom's energy is evaluated in `CT` and converted
 * to double; the thread sums and the block reduction are double, and thread
 * zero adds each block total to `energy`.
 *
 * @tparam AT Device force-accumulator representation written to `forces`.
 * @tparam CT Arithmetic representation of the per-atom force and energy
 * evaluation.
 * @tparam calcEnergy Whether block totals are added to `energy`.
 * @param[in,out] forces Non-NULL device structure-of-arrays storage containing
 * at least `3 * forceStride` elements. Contributions are added atomically.
 * @param[in] forceStride Dimensionless element offset between X, Y, and Z
 * component arrays.
 * @param[in,out] energy Non-NULL device pointer to the `harm` scalar in
 * kilocalories per mole. It is modified only when `calcEnergy` is `true`.
 * @param[in] forceConstants Non-NULL device array of `numAtoms` force
 * constants.
 * @param[in] xyzq Non-NULL device array of `numAtoms` coordinate-charge
 * records. Cartesian components use angstroms; component `w` is ignored.
 * @param[in] coordsRef Non-NULL device array of `numAtoms` reference positions
 * in angstroms.
 * @param[in] masses Non-NULL device array of `numAtoms` masses or weights.
 * @param[in] numAtoms Positive dimensionless atom count.
 * @param[in] exponent Dimensionless integer exponent.
 * @param[in] scaleX Dimensionless X scale factor.
 * @param[in] scaleY Dimensionless Y scale factor.
 * @param[in] scaleZ Dimensionless Z scale factor.
 *
 * @pre Every thread in each block reaches the block-wide reduction.
 */
template <typename AT, typename CT, bool calcEnergy>
__global__ static void HarmonicRestraintForceKernel(
    AT *__restrict__ forces, const int forceStride, double *__restrict__ energy,
    const double *__restrict__ forceConstants, const float4 *__restrict__ xyzq,
    const double3 *__restrict__ coordsRef, const double *__restrict__ masses,
    const int numAtoms, const int exponent, const CT scaleX, const CT scaleY,
    const CT scaleZ) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  constexpr CT ZERO = static_cast<CT>(0);

  double epot = 0.0;
  for (int i = index; i < numAtoms; i += stride) {
    // CHARMM's KCNSTR is the FORCe or WEIGhting value, multiplied by the atom
    // mass when MASS is given.
    const CT c =
        static_cast<CT>(forceConstants[i]) * static_cast<CT>(masses[i]);
    if (c == ZERO)
      continue;

    const CT ax = static_cast<CT>(xyzq[i].x) - static_cast<CT>(coordsRef[i].x);
    const CT ay = static_cast<CT>(xyzq[i].y) - static_cast<CT>(coordsRef[i].y);
    const CT az = static_cast<CT>(xyzq[i].z) - static_cast<CT>(coordsRef[i].z);
    const CT s = scaleX * ax * ax + scaleY * ay * ay + scaleZ * az * az;

    CT e = ZERO;
    CT df = ZERO;
    if (exponent == 1) {
      const CT r = sqrt(s);
      e = c * r;
      if (s > static_cast<CT>(1.0e-12))
        df = c / r;
    } else if (exponent == 2) {
      e = c * s;
      df = static_cast<CT>(2) * c;
    } else if (exponent == 3) {
      const CT r = sqrt(s);
      e = c * s * r;
      df = c * static_cast<CT>(3) * r;
    } else if (exponent == 4) {
      e = c * s * s;
      df = c * static_cast<CT>(4) * s;
    } else if (exponent > 4) {
      const CT p = c * pow(s, (static_cast<CT>(exponent) - static_cast<CT>(4)) *
                                  static_cast<CT>(0.5));
      e = p * s * s;
      df = p * static_cast<CT>(exponent) * s;
    }

    if constexpr (calcEnergy)
      epot += static_cast<double>(e);

    write_force<AT>(roundCTtoAT<AT, CT>(ax * df * scaleX),
                    roundCTtoAT<AT, CT>(ay * df * scaleY),
                    roundCTtoAT<AT, CT>(az * df * scaleZ), i, forceStride,
                    forces);
  }

  if constexpr (calcEnergy) {
    epot = BlockReduceSum<double>(epot);
    if (threadIdx.x == 0)
      atomicAdd(energy, epot);
  }

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::calcForce(const float4 *xyzq,
                                               const bool calcEnergy,
                                               const bool calcVirial) {
  // JEG260802: Did not add error checking here because this function is called
  // frequently. i.e. Did not want to slow dynamics down. Users should not be
  // calling this function themselves.

  // JEG261113: We can add this calculation later if we implement the external
  // virial. For now, we only have the internal virial, and absolute harmonic
  // restraints only contribute to the external virial.
  static_cast<void>(calcVirial);

  this->syncDeviceData();

  constexpr int numThreads = 256;
  const int numBlocks = (m_NumAtoms + numThreads - 1) / numThreads;
  const CT scaleX = static_cast<CT>(m_ScaleFactors[0]);
  const CT scaleY = static_cast<CT>(m_ScaleFactors[1]);
  const CT scaleZ = static_cast<CT>(m_ScaleFactors[2]);

  if (calcEnergy) {
    cudaCheckLaunch(HarmonicRestraintForceKernel<AT, CT, true>
                    <<<numBlocks, numThreads, 0, *m_Stream>>>(
                        m_Forces->xyz(), m_Forces->stride(),
                        m_EnergyVirial->getEnergyPointer("harm"),
                        m_ForceConstants.getDeviceArray().data(), xyzq,
                        m_ReferenceCoordinates.getDeviceArray().data(),
                        m_Masses.getDeviceArray().data(), m_NumAtoms,
                        m_Exponent, scaleX, scaleY, scaleZ));
  } else {
    cudaCheckLaunch(HarmonicRestraintForceKernel<AT, CT, false>
                    <<<numBlocks, numThreads, 0, *m_Stream>>>(
                        m_Forces->xyz(), m_Forces->stride(),
                        m_EnergyVirial->getEnergyPointer("harm"),
                        m_ForceConstants.getDeviceArray().data(), xyzq,
                        m_ReferenceCoordinates.getDeviceArray().data(),
                        m_Masses.getDeviceArray().data(), m_NumAtoms,
                        m_Exponent, scaleX, scaleY, scaleZ));
  }

  return;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::setBoxDimensions(
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

  // JEG260615: After some thought, I am not entirely convinced this is correct.
  // So I have commented it out.
  /* *
  if ((m_BoxDimensions[0] != 0.0) && (m_BoxDimensions[1] != 0.0) &&
      (m_BoxDimensions[2] != 0.0)) {
    // JEG260512: This only works for cubic, tetragonal, and orthorhombic boxes.
    // If we ever want to do other crystals (e.g. triclinic) we would need to
    // update all of the code to use the A, B, C, alpha, beta, gamma matrix.
    const double dx = boxDimensions[0] / m_BoxDimensions[0];
    const double dy = boxDimensions[1] / m_BoxDimensions[1];
    const double dz = boxDimensions[2] / m_BoxDimensions[2];

    for (int i = 0; i < m_NumAtoms; i++) {
      m_ReferenceCoordinates[i].x *= dx;
      m_ReferenceCoordinates[i].y *= dy;
      m_ReferenceCoordinates[i].z *= dz;
    }
    m_ReferenceCoordinates.transferToDevice();
  }
  * */

  std::copy(boxDimensions.begin(), boxDimensions.end(),
            m_BoxDimensions.begin());

  return;
}

template <typename AT, typename CT>
std::shared_ptr<Force<AT>> HarmonicRestraintForce<AT, CT>::getForce(void) {
  return m_Forces;
}

template <typename AT, typename CT>
std::shared_ptr<CudaEnergyVirial>
HarmonicRestraintForce<AT, CT>::getEnergyVirial(void) {
  return m_EnergyVirial;
}

template <typename AT, typename CT>
void HarmonicRestraintForce<AT, CT>::syncDeviceData(void) {
  if (!m_DeviceDataDirty)
    return;

  m_ForceConstants.transferToDeviceAsync(*m_Stream);
  m_ReferenceCoordinates.transferToDeviceAsync(*m_Stream);
  m_Masses.transferToDeviceAsync(*m_Stream);
  cudaCheck(cudaStreamSynchronize(*m_Stream));

  m_DeviceDataDirty = false;

  return;
}

//
// Explicit instances of HarmonicRestraintForce
//
template class HarmonicRestraintForce<long long int, float>;
template class HarmonicRestraintForce<long long int, double>;
