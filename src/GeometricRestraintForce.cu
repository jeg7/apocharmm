// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "GeometricRestraintForce.h"

#include "ApoCharmmError.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <string>

namespace {

// JEG261002: Private device error bits. No failed evaluation is committed to
// public output.
constexpr int NONFINITE_EVALUATION = 1;
constexpr int UNREPRESENTABLE_GRADIENT = 2;
constexpr int ACCUMULATION_OVERFLOW = 4;
constexpr int UNSUPPORTED_TERM = 8;

} // namespace

template <typename AT, typename CT>
GeometricRestraintForce<AT, CT>::GeometricRestraintForce(
    const std::shared_ptr<const CharmmPSF> &psf)
    : m_NumAtoms(0), m_BoxDimensions{0.0, 0.0, 0.0}, m_DeviceDataDirty(true) {
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
        "Mass at atom index " + std::to_string(i) +
            " must be finite; observed " + std::to_string(masses[i]));
  }

  m_NumAtoms = numAtoms;
  m_Masses = masses;
  m_TermOffsets.getHostArray().push_back(0);
  m_PendingForces.resize(static_cast<std::size_t>(numAtoms));
  m_PendingEnergy.resize(1);
  m_EvaluationError.resize(1);
  m_EnergyVirial = std::make_shared<CudaEnergyVirial>();
  m_EnergyVirial->insert("mmfp");
  m_Forces = std::make_shared<Force<AT>>();
  m_Forces->realloc(numAtoms, 1.5f);
  m_Stream = std::make_shared<cudaStream_t>();
  try {
    cudaCheck(cudaStreamCreate(m_Stream.get()));
    this->clear();
    cudaCheck(cudaStreamSynchronize(*m_Stream));
  } catch (...) {
    destroy_cuda_stream_noexcept(m_Stream.get());
    throw;
  }
}

template <typename AT, typename CT>
GeometricRestraintForce<AT, CT>::~GeometricRestraintForce(void) noexcept {
  if (m_Stream != nullptr)
    destroy_cuda_stream_noexcept(m_Stream.get());
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addPlane(
    const AtomSelection &selection, const std::array<double, 3> &origin,
    const std::array<double, 3> &direction, const double forceConstant,
    const double offset, const Activation activation) {
  APOCHARMM_REQUIRE(selection.getNumAtoms() == m_NumAtoms,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Selection atom count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(selection.getNumAtoms()));

  this->appendPlane(selection.getAtomIndices(), origin, direction,
                    forceConstant, offset, activation);

  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addPlane(
    const AtomReference &atom, const std::array<double, 3> &origin,
    const std::array<double, 3> &direction, const double forceConstant,
    const double offset, const Activation activation) {
  this->appendPlane({atom.getAtomIndex()}, origin, direction, forceConstant,
                    offset, activation);
  return;
}

template <typename AT, typename CT>
int GeometricRestraintForce<AT, CT>::getNumAtoms(void) const noexcept {
  return m_NumAtoms;
}

template <typename AT, typename CT>
std::size_t GeometricRestraintForce<AT, CT>::getNumTerms(void) const noexcept {
  return m_TermTypes.size();
}

template <typename AT, typename CT>
const std::vector<double> &
GeometricRestraintForce<AT, CT>::getMasses(void) const noexcept {
  return m_Masses.getHostArray();
}

template <typename AT, typename CT>
std::shared_ptr<cudaStream_t> GeometricRestraintForce<AT, CT>::getStream(void) {
  return m_Stream;
}

template <typename AT, typename CT>
std::shared_ptr<Force<AT>> GeometricRestraintForce<AT, CT>::getForce(void) {
  return m_Forces;
}

template <typename AT, typename CT>
std::shared_ptr<CudaEnergyVirial>
GeometricRestraintForce<AT, CT>::getEnergyVirial(void) {
  return m_EnergyVirial;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::reset(void) {
  std::vector<std::size_t> emptyOffsets(1, 0);

  m_TermOffsets.getHostArray().swap(emptyOffsets);
  m_AtomIndices.getHostArray().clear();
  m_TermTypes.getHostArray().clear();
  m_Origins.getHostArray().clear();
  m_Directions.getHostArray().clear();
  m_Offsets.getHostArray().clear();
  m_ForceConstants.getHostArray().clear();
  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::setBoxDimensions(
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

  std::copy(boxDimensions.begin(), boxDimensions.end(),
            m_BoxDimensions.begin());

  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::initialize(
    const int numAtoms, const std::vector<double> &boxDimensions) {
  APOCHARMM_REQUIRE(numAtoms == m_NumAtoms, ApoCharmmErrorCode::InvalidArgument,
                    "Initialization atom count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(numAtoms));

  this->setBoxDimensions(boxDimensions);
  this->syncDeviceData();

  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::clear(void) {
  m_EnergyVirial->clear(*m_Stream);
  m_Forces->clear(*m_Stream);
  return;
}

template <typename AT, typename CT, int blockSize, bool calcEnergy>
__global__ static void EvaluateGeometricTermsKernel(
    double3 *__restrict__ membershipGradients,
    double *__restrict__ termEnergies, int *__restrict__ error,
    const float4 *__restrict__ xyzq, const int *__restrict__ atomIndices,
    const std::size_t *__restrict__ termOffsets,
    const int4 *__restrict__ termTypes, const double3 *__restrict__ origins,
    const double3 *__restrict__ directions, const double *__restrict__ offsets,
    const double *__restrict__ forceConstants, const std::size_t numTerms) {
  using GeoForce = GeometricRestraintForce<AT, CT>;
  __shared__ BlockReduceSumStorage<double, blockSize, 1> cache;

  for (std::size_t i = blockIdx.x; i < numTerms; i += gridDim.x) {
    const int4 types = termTypes[i];
    const bool supported =
        (types.x == static_cast<int>(GeoForce::Geometry::PLANE)) &&
        (types.z == static_cast<int>(GeoForce::SelectionMode::ATOMWISE)) &&
        (types.w == static_cast<int>(GeoForce::PotentialType::HARMONIC));
    if (!supported && (threadIdx.x == 0))
      atomicOr(error, UNSUPPORTED_TERM);

    const double3 origin = origins[i];
    const double3 normal = directions[i];
    const double forceConstant = forceConstants[i];
    const double offset = offsets[i];
    double partialEnergy[1] = {0.0};

    for (std::size_t j = termOffsets[i] + threadIdx.x; j < termOffsets[i + 1];
         j += blockDim.x) {
      double3 gradient = make_double3(0.0, 0.0, 0.0);
      double energy = 0.0;

      if (supported && forceConstant != 0.0) {
        const float4 position = xyzq[atomIndices[j]];

        // JEG261006: Plane geometry uses the raw Cartesian displacement
        const double dx = static_cast<double>(position.x) - origin.x;
        const double dy = static_cast<double>(position.y) - origin.y;
        const double dz = static_cast<double>(position.z) - origin.z;
        const double signedDistance =
            normal.x * dx + normal.y * dy + normal.z * dz;

        // JEG261006: Activation maps the oriented coordinate to a slab distance
        // only for one-sided modes. The exact-zero tie uses +normal
        const bool symmetric =
            types.y == static_cast<int>(GeoForce::Activation::SYMMETRIC);
        const double coordinate =
            symmetric ? signedDistance : fabs(signedDistance);
        const double derivativeSign =
            (!symmetric && signedDistance < 0.0) ? -1.0 : 1.0;
        const double deviation = coordinate - offset;
        const bool active =
            symmetric ||
            ((types.y == static_cast<int>(GeoForce::Activation::INSIDE)) &&
             (deviation > 0.0)) ||
            ((types.y == static_cast<int>(GeoForce::Activation::OUTSIDE)) &&
             (deviation < 0.0));

        if (!isfinite(signedDistance) || !isfinite(deviation))
          atomicOr(error, NONFINITE_EVALUATION);
        else if (active && (deviation != 0.0)) {
          // JEG261006: Harmonic potential and energetic gradient
          const double derivative = forceConstant * deviation;
          const double orientedDerivative = derivative * derivativeSign;
          gradient = make_double3(orientedDerivative * normal.x,
                                  orientedDerivative * normal.y,
                                  orientedDerivative * normal.z);
          if constexpr (calcEnergy)
            energy = 0.5 * derivative * deviation;
        }
      }

      if (!isfinite(gradient.x) || !isfinite(gradient.y) ||
          !isfinite(gradient.z) || !isfinite(energy))
        atomicOr(error, NONFINITE_EVALUATION);

      membershipGradients[j] = gradient;
      partialEnergy[0] += energy;
    }

    BlockReduceSum<double, blockSize, 1>(partialEnergy, cache);
    if (threadIdx.x == 0) {
      termEnergies[i] = partialEnergy[0];
      if (!isfinite(partialEnergy[0]))
        atomicOr(error, NONFINITE_EVALUATION);
    }
    __syncthreads();
  }

  return;
}

template <typename AT, typename CT, bool calcEnergy>
__global__ static void StageGeometricOutputKernel(
    longlong3 *__restrict__ pendingForces, double *__restrict__ pendingEnergy,
    int *__restrict__ error, const std::size_t *__restrict__ atomMemberships,
    const double3 *__restrict__ membershipGradients,
    const std::size_t *__restrict__ atomOffsets, const int numAtoms,
    const AT *__restrict__ forces, const int forceStride,
    const double *__restrict__ energy, const double *__restrict__ termEnergies,
    const std::size_t numTerms) {
  if constexpr (calcEnergy) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      double total = *energy;
      // JEG261006: Do not sort, reassociate, or atomically sum separate terms
      for (std::size_t i = 0; i < numTerms; i++)
        total += termEnergies[i];
      if (!isfinite(total))
        atomicOr(error, NONFINITE_EVALUATION);
      *pendingEnergy = total;
    }
  }

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    double gradient[3] = {0.0, 0.0, 0.0};
    for (std::size_t j = atomOffsets[i]; j < atomOffsets[i + 1]; j++) {
      const double3 contribution = membershipGradients[atomMemberships[j]];
      gradient[0] += contribution.x;
      gradient[1] += contribution.y;
      gradient[2] += contribution.z;
    }

    AT result[3] = {static_cast<AT>(0), static_cast<AT>(0), static_cast<AT>(0)};
    bool valid = true;
    const double upperBound = -static_cast<double>(LLONG_MIN) * INV_FORCE_SCALE;
    for (int k = 0; k < 3; k++) {
      if (!isfinite(gradient[k])) {
        atomicOr(error, NONFINITE_EVALUATION);
        valid = false;
        continue;
      }
      // JEG261006: Check before narrowing as well as after rounding to CT. In
      // particular, CT=float can round an otherwise representable value to the
      // upper bound
      if ((gradient[k] < -upperBound) || (gradient[k] >= upperBound)) {
        atomicOr(error, UNREPRESENTABLE_GRADIENT);
        valid = false;
        continue;
      }
      const CT converted = static_cast<CT>(gradient[k]);
      if ((static_cast<double>(converted) < -upperBound) ||
          (static_cast<double>(converted) >= upperBound)) {
        atomicOr(error, UNREPRESENTABLE_GRADIENT);
        valid = false;
        continue;
      }

      const AT increment = roundCTtoAT<AT, CT>(converted);
      const AT previous = forces[k * forceStride + i];
      if (((increment > 0) && (previous > LLONG_MAX - increment)) ||
          ((increment < 0) && (previous < LLONG_MIN - increment))) {
        atomicOr(error, ACCUMULATION_OVERFLOW);
        valid = false;
        continue;
      }

      result[k] = previous + increment;
    }

    if (valid)
      pendingForces[i] = make_longlong3(result[0], result[1], result[2]);
  }

  return;
}

template <typename AT, bool calcEnergy>
__global__ static void CommitGeometricOutputKernel(
    double *__restrict__ energy, AT *__restrict__ forces, const int forceStride,
    const longlong3 *__restrict__ pendingForces, const int numAtoms,
    const double *__restrict__ pendingEnergy) {
  if constexpr (calcEnergy) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
      *energy = *pendingEnergy;
  }

  const int idx = blockIdx.x * gridDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < numAtoms; i += stride) {
    const longlong3 value = pendingForces[i];
    forces[forceStride * 0 + i] = value.x;
    forces[forceStride * 1 + i] = value.y;
    forces[forceStride * 2 + i] = value.z;
  }

  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::calcForce(const float4 *xyzq,
                                                const bool calcEnergy,
                                                const bool calcVirial) {
  static_cast<void>(calcVirial);

  if (this->getNumTerms() == 0)
    return;

  APOCHARMM_REQUIRE(xyzq != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "Coordinate-charge array must not be null");
  this->syncDeviceData();

  constexpr unsigned int numThreads = 256;
  const unsigned int termBlocks = static_cast<unsigned int>(
      std::min<std::size_t>(this->getNumTerms(), 65535));
  const unsigned int atomBlocks =
      static_cast<unsigned int>(std::min<std::size_t>(
          (static_cast<std::size_t>(m_NumAtoms) + numThreads - 1) / numThreads,
          65535));
  int *error = m_EvaluationError.getDeviceArray().data();
  double *energy = m_EnergyVirial->getEnergyPointer("mmfp");
  cudaCheck(cudaMemsetAsync(error, 0, sizeof(int), *m_Stream));

  if (calcEnergy) {
    cudaCheckLaunch(
        EvaluateGeometricTermsKernel<AT, CT, numThreads, true>
        <<<termBlocks, numThreads, 0, *m_Stream>>>(
            m_MembershipGradients.getDeviceArray().data(),
            m_TermEnergies.getDeviceArray().data(), error, xyzq,
            m_AtomIndices.getDeviceArray().data(),
            m_TermOffsets.getDeviceArray().data(),
            m_TermTypes.getDeviceArray().data(),
            m_Origins.getDeviceArray().data(),
            m_Directions.getDeviceArray().data(),
            m_Offsets.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(), this->getNumTerms()));
    cudaCheckLaunch(
        StageGeometricOutputKernel<AT, CT, true>
        <<<atomBlocks, numThreads, 0, *m_Stream>>>(
            m_PendingForces.getDeviceArray().data(),
            m_PendingEnergy.getDeviceArray().data(), error,
            m_AtomMemberships.getDeviceArray().data(),
            m_MembershipGradients.getDeviceArray().data(),
            m_AtomOffsets.getDeviceArray().data(), m_NumAtoms, m_Forces->xyz(),
            m_Forces->stride(), energy, m_TermEnergies.getDeviceArray().data(),
            this->getNumTerms()));
  } else {
    cudaCheckLaunch(
        EvaluateGeometricTermsKernel<AT, CT, numThreads, false>
        <<<termBlocks, numThreads, 0, *m_Stream>>>(
            m_MembershipGradients.getDeviceArray().data(),
            m_TermEnergies.getDeviceArray().data(), error, xyzq,
            m_AtomIndices.getDeviceArray().data(),
            m_TermOffsets.getDeviceArray().data(),
            m_TermTypes.getDeviceArray().data(),
            m_Origins.getDeviceArray().data(),
            m_Directions.getDeviceArray().data(),
            m_Offsets.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(), this->getNumTerms()));
    cudaCheckLaunch(
        StageGeometricOutputKernel<AT, CT, false>
        <<<atomBlocks, numThreads, 0, *m_Stream>>>(
            m_PendingForces.getDeviceArray().data(),
            m_PendingEnergy.getDeviceArray().data(), error,
            m_AtomMemberships.getDeviceArray().data(),
            m_MembershipGradients.getDeviceArray().data(),
            m_AtomOffsets.getDeviceArray().data(), m_NumAtoms, m_Forces->xyz(),
            m_Forces->stride(), energy, m_TermEnergies.getDeviceArray().data(),
            this->getNumTerms()));
  }

  int observedError = 0;
  cudaCheck(cudaMemcpyAsync(&observedError, error, sizeof(int),
                            cudaMemcpyDeviceToHost, *m_Stream));
  cudaCheck(cudaStreamSynchronize(*m_Stream));

  APOCHARMM_REQUIRE((observedError & UNSUPPORTED_TERM) == 0,
                    ApoCharmmErrorCode::Runtime,
                    "MMFP evaluation encountered an unsupported term");
  APOCHARMM_REQUIRE((observedError & NONFINITE_EVALUATION) == 0,
                    ApoCharmmErrorCode::Runtime,
                    "MMFP evaluation produced a non-finite value");
  APOCHARMM_REQUIRE((observedError & UNREPRESENTABLE_GRADIENT) == 0,
                    ApoCharmmErrorCode::Runtime,
                    "MMFP gradient exceeds the fixed-point force range");
  APOCHARMM_REQUIRE(
      (observedError & ACCUMULATION_OVERFLOW) == 0, ApoCharmmErrorCode::Runtime,
      "MMFP accumulated gradient exceeds the fixed-point force range");

  if (calcEnergy) {
    cudaCheckLaunch(CommitGeometricOutputKernel<AT, true>
                    <<<atomBlocks, numThreads, 0, *m_Stream>>>(
                        energy, m_Forces->xyz(), m_Forces->stride(),
                        m_PendingForces.getDeviceArray().data(), m_NumAtoms,
                        m_PendingEnergy.getDeviceArray().data()));
  } else {
    cudaCheckLaunch(CommitGeometricOutputKernel<AT, false>
                    <<<atomBlocks, numThreads, 0, *m_Stream>>>(
                        energy, m_Forces->xyz(), m_Forces->stride(),
                        m_PendingForces.getDeviceArray().data(), m_NumAtoms,
                        m_PendingEnergy.getDeviceArray().data()));
  }

  return;
}

template <typename AT, typename CT>
bool GeometricRestraintForce<AT, CT>::supportsPBC(
    const PBC pbc) const noexcept {
  return pbc == PBC::P1;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::appendPlane(
    const std::vector<int> &atomIndices, const std::array<double, 3> &origin,
    const std::array<double, 3> &direction, const double forceConstant,
    const double offset, const Activation activation) {
  APOCHARMM_REQUIRE(!atomIndices.empty(), ApoCharmmErrorCode::InvalidArgument,
                    "Selection must contain at least one atom; observed 0");
  APOCHARMM_REQUIRE(
      atomIndices.size() <=
          std::numeric_limits<std::size_t>::max() - m_AtomIndices.size(),
      ApoCharmmErrorCode::InvalidArgument,
      "Adding the selection would exceed the membership representation limit");
  for (const int i : atomIndices) {
    APOCHARMM_REQUIRE(
        (i >= 0) && (i < m_NumAtoms), ApoCharmmErrorCode::InvalidArgument,
        "Atom index is out of range; expected [0, " +
            std::to_string(m_NumAtoms) + "), observed " + std::to_string(i));
  }

  for (std::size_t i = 0; i < 3; i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(origin[i]), ApoCharmmErrorCode::InvalidArgument,
        "Origin component must be finite; index " + std::to_string(i));
    APOCHARMM_REQUIRE(
        std::isfinite(direction[i]), ApoCharmmErrorCode::InvalidArgument,
        "Direction component must be finite; index " + std::to_string(i));
  }

  APOCHARMM_REQUIRE(std::isfinite(forceConstant),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Force constant must be finite; observed " +
                        std::to_string(forceConstant));

  APOCHARMM_REQUIRE(std::isfinite(offset), ApoCharmmErrorCode::InvalidArgument,
                    "Offset must be finite; observed " +
                        std::to_string(offset));

  APOCHARMM_REQUIRE((activation == Activation::SYMMETRIC) ||
                        (activation == Activation::INSIDE) ||
                        (activation == Activation::OUTSIDE),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Invalid MMFP activation value; observed " +
                        std::to_string(static_cast<int>(activation)));

  const double norm = std::hypot(direction[0], direction[1], direction[2]);
  APOCHARMM_REQUIRE(std::isfinite(norm) && (norm > 0.0),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Direction must have a positive finite norm");

  const double3 normal = make_double3(direction[0] / norm, direction[1] / norm,
                                      direction[2] / norm);

  // JEG261002: Build replacements first so partial host allocation cannot
  // install a term
  std::vector<int> indices = m_AtomIndices.getHostArray();
  std::vector<std::size_t> termOffsets = m_TermOffsets.getHostArray();
  std::vector<int4> types = m_TermTypes.getHostArray();
  std::vector<double3> origins = m_Origins.getHostArray();
  std::vector<double3> directions = m_Directions.getHostArray();
  std::vector<double> offsets = m_Offsets.getHostArray();
  std::vector<double> forceConstants = m_ForceConstants.getHostArray();

  indices.insert(indices.end(), atomIndices.begin(), atomIndices.end());
  termOffsets.push_back(indices.size());
  types.push_back(make_int4(static_cast<int>(Geometry::PLANE),
                            static_cast<int>(activation),
                            static_cast<int>(SelectionMode::ATOMWISE),
                            static_cast<int>(PotentialType::HARMONIC)));
  origins.push_back(make_double3(origin[0], origin[1], origin[2]));
  directions.push_back(normal);
  offsets.push_back(offset);
  forceConstants.push_back(forceConstant);

  m_AtomIndices.getHostArray().swap(indices);
  m_TermOffsets.getHostArray().swap(termOffsets);
  m_TermTypes.getHostArray().swap(types);
  m_Origins.getHostArray().swap(origins);
  m_Directions.getHostArray().swap(directions);
  m_Offsets.getHostArray().swap(offsets);
  m_ForceConstants.getHostArray().swap(forceConstants);
  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::syncDeviceData(void) {
  if (!m_DeviceDataDirty)
    return;

  cudaCheck(cudaStreamSynchronize(*m_Stream));

  std::vector<std::size_t> atomOffsets(m_NumAtoms + 1, 0);
  for (const int atomIndex : m_AtomIndices.getHostArray())
    atomOffsets[atomIndex + 1]++;
  for (std::size_t atomIndex = 1; atomIndex < atomOffsets.size(); atomIndex++)
    atomOffsets[atomIndex] += atomOffsets[atomIndex - 1];

  std::vector<std::size_t> positions = atomOffsets;
  std::vector<std::size_t> memberships(m_AtomIndices.size());
  for (std::size_t member = 0; member < m_AtomIndices.size(); member++)
    memberships[positions[m_AtomIndices[member]]++] = member;

  m_AtomOffsets.getHostArray().swap(atomOffsets);
  m_AtomMemberships.getHostArray().swap(memberships);

  // JEG261002: CudaContainer modifiers above deliberately changed only the host
  // mirrors
  const auto upload = [](auto &container) -> void {
    container.resize(container.size());
    container.transferToDevice();
    return;
  };

  upload(m_TermOffsets);
  upload(m_AtomIndices);
  upload(m_TermTypes);
  upload(m_Origins);
  upload(m_Directions);
  upload(m_Offsets);
  upload(m_ForceConstants);
  upload(m_AtomOffsets);
  upload(m_AtomMemberships);

  m_MembershipGradients.resize(m_AtomIndices.size());
  m_TermEnergies.resize(m_TermTypes.size());

  m_DeviceDataDirty = false;

  return;
}

// JEG261002: Explicit instances of GeometricRestraintForce
template class GeometricRestraintForce<long long int, float>;
template class GeometricRestraintForce<long long int, double>;
