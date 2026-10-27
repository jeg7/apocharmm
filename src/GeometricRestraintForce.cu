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
#include <utility>

namespace {

// JEG261002: Private device error bits. No failed evaluation is committed to
// public output.
constexpr int NONFINITE_EVALUATION = 1;
constexpr int UNREPRESENTABLE_GRADIENT = 2;
constexpr int ACCUMULATION_OVERFLOW = 4;
constexpr int UNSUPPORTED_TERM = 8;
constexpr unsigned int RCM_CHUNK_SIZE = 256;

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

  // JEG261022: Match Force::realloc's 256-byte stride and both applications of
  // its growth factor, using wide arithmetic before any allocation or
  // narrowing.
  constexpr float forceAllocationFactor = 1.5f;
  const std::size_t forceStride =
      (((static_cast<std::size_t>(numAtoms) + 31) * sizeof(AT) - 1) / 256 + 1) *
      256 / sizeof(AT);
  const double requestedForceCapacity =
      static_cast<double>(3 * forceStride) * forceAllocationFactor;
  const double allocatedForceCapacity =
      std::floor(requestedForceCapacity) * forceAllocationFactor;
  APOCHARMM_REQUIRE(allocatedForceCapacity <=
                        static_cast<double>(std::numeric_limits<int>::max()),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Atom count exceeds the fixed-point force storage "
                    "representation limit; observed " +
                        std::to_string(numAtoms));

  const std::vector<double> &masses = psf->getMasses();
  APOCHARMM_REQUIRE(masses.size() == static_cast<std::size_t>(numAtoms),
                    ApoCharmmErrorCode::Runtime,
                    "PSF mass count mismatch; expected " +
                        std::to_string(numAtoms) + ", observed " +
                        std::to_string(masses.size()));

  m_NumAtoms = numAtoms;

  this->validateMasses(masses);

  m_Masses = masses;
  m_TermOffsets.getHostArray().push_back(0);
  m_PendingForces.resize(static_cast<std::size_t>(numAtoms));
  m_PendingEnergy.resize(1);

  const std::size_t virialBlocks = std::min<std::size_t>(
      (static_cast<std::size_t>(numAtoms) + RCM_CHUNK_SIZE - 1) /
          RCM_CHUNK_SIZE,
      65535);
  m_VirialPartials.resize(9 * virialBlocks);
  m_PendingVirial.resize(9);
  m_EvaluationError.resize(1);
  m_EnergyVirial = std::make_shared<CudaEnergyVirial>();
  m_EnergyVirial->insert("mmfp");
  m_Forces = std::make_shared<Force<AT>>();
  m_Forces->realloc(numAtoms, forceAllocationFactor);
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
    const double offset, const Activation activation,
    const SelectionMode selectionMode) {
  this->appendTerm(this->getSelectionIndices(selection), Geometry::PLANE,
                   origin, direction, forceConstant, offset, activation,
                   selectionMode);
  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addPlane(
    const AtomReference &atom, const std::array<double, 3> &origin,
    const std::array<double, 3> &direction, const double forceConstant,
    const double offset, const Activation activation,
    const SelectionMode selectionMode) {
  this->appendTerm({atom.getAtomIndex()}, Geometry::PLANE, origin, direction,
                   forceConstant, offset, activation, selectionMode);
  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addCylinder(
    const AtomSelection &selection, const std::array<double, 3> &origin,
    const std::array<double, 3> &direction, const double forceConstant,
    const double offset, const Activation activation,
    const SelectionMode selectionMode) {
  this->appendTerm(this->getSelectionIndices(selection), Geometry::CYLINDER,
                   origin, direction, forceConstant, offset, activation,
                   selectionMode);
  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addCylinder(
    const AtomReference &atom, const std::array<double, 3> &origin,
    const std::array<double, 3> &direction, const double forceConstant,
    const double offset, const Activation activation,
    const SelectionMode selectionMode) {
  this->appendTerm({atom.getAtomIndex()}, Geometry::CYLINDER, origin, direction,
                   forceConstant, offset, activation, selectionMode);
  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addSphere(
    const AtomSelection &selection, const std::array<double, 3> &origin,
    const double forceConstant, const double offset,
    const Activation activation, const SelectionMode selectionMode) {
  this->appendTerm(this->getSelectionIndices(selection), Geometry::SPHERE,
                   origin, {0.0, 0.0, 0.0}, forceConstant, offset, activation,
                   selectionMode);
  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::addSphere(
    const AtomReference &atom, const std::array<double, 3> &origin,
    const double forceConstant, const double offset,
    const Activation activation, const SelectionMode selectionMode) {
  this->appendTerm({atom.getAtomIndex()}, Geometry::SPHERE, origin,
                   {0.0, 0.0, 0.0}, forceConstant, offset, activation,
                   selectionMode);
  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::setMasses(
    const std::vector<double> &masses) {
  this->validateMasses(masses);

  const std::vector<int4> &types = m_TermTypes.getHostArray();
  const std::vector<std::size_t> &offsets = m_TermOffsets.getHostArray();
  const std::vector<int> &indices = m_AtomIndices.getHostArray();
  for (std::size_t i = 0; i < types.size(); i++) {
    if (types[i].z == static_cast<int>(SelectionMode::RCM))
      this->validateRcmMasses(masses, indices, offsets[i], offsets[i + 1]);
  }

  // JEG261019: Allocate replacement storage only after every RCM selection
  // validated. Recompute derived totals into that storage; no caller data is
  // retained.
  std::vector<double> totals(this->getNumTerms(), 0.0);
  for (std::size_t i = 0; i < types.size(); i++) {
    if (types[i].z == static_cast<int>(SelectionMode::RCM)) {
      totals[i] =
          this->validateRcmMasses(masses, indices, offsets[i], offsets[i + 1]);
    }
  }

  // JEG261019: Finish old readers before releasing either old device
  // allocation.
  cudaCheck(cudaStreamSynchronize(*m_Stream));
  CudaContainer<double> replacementMasses(masses);
  CudaContainer<double> replacementTotals(std::move(totals));

  // JEG261019: CudaContainer move assignment is noexcept and transfers both
  // mirrors. No throwing operation may be inserted between these two commits.
  m_Masses = std::move(replacementMasses);
  m_SelectedMassTotals = std::move(replacementTotals);

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
  m_SelectedMassTotals.getHostArray().clear();
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
  this->clear(true, true);
  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::clear(const bool calcEnergy,
                                            const bool calcVirial) {
  if (calcEnergy && calcVirial)
    m_EnergyVirial->clear(*m_Stream);
  else if (calcEnergy && !calcVirial)
    m_EnergyVirial->clearEnergy(*m_Stream);
  else if (!calcEnergy && calcVirial)
    m_EnergyVirial->clearVirial(*m_Stream);
  m_Forces->clear(*m_Stream);
  return;
}

template <int blockSize>
__global__ static void BuildRcmPartialsKernel(
    double3 *__restrict__ partials, int *__restrict__ error,
    const float4 *__restrict__ xyzq, const int *__restrict__ atomIndices,
    const double *__restrict__ masses,
    const std::size_t *__restrict__ chunkBegins,
    const std::size_t *__restrict__ chunkEnds, const std::size_t numChunks) {
  static_assert(blockSize == RCM_CHUNK_SIZE,
                "RCM partial block size must equal the membership chunk size");

  __shared__ BlockReduceSumStorage<double, blockSize, 3> cache;

  for (std::size_t chunk = blockIdx.x; chunk < numChunks; chunk += gridDim.x) {
    double weighted[3] = {0.0, 0.0, 0.0};
    const std::size_t member = chunkBegins[chunk] + threadIdx.x;
    if (member < chunkEnds[chunk]) {
      const int atom = atomIndices[member];
      const float4 position = xyzq[atom];
      const double mass = masses[atom];
      weighted[0] = mass * static_cast<double>(position.x);
      weighted[1] = mass * static_cast<double>(position.y);
      weighted[2] = mass * static_cast<double>(position.z);
      if (!isfinite(weighted[0]) || !isfinite(weighted[1]) ||
          !isfinite(weighted[2])) {
        atomicOr(error, NONFINITE_EVALUATION);
      }
    }

    BlockReduceSum<double, blockSize, 3>(weighted, cache);
    if (threadIdx.x == 0) {
      partials[chunk] = make_double3(weighted[0], weighted[1], weighted[2]);
      if (!isfinite(weighted[0]) || !isfinite(weighted[1]) ||
          !isfinite(weighted[2])) {
        atomicOr(error, NONFINITE_EVALUATION);
      }
    }
    __syncthreads();
  }

  return;
}

template <typename AT, typename CT, bool calcEnergy>
__device__ static void
EvaluateGeometricPoint(double3 &gradient, double &energy, int *error,
                       const double3 position, const int4 types,
                       const double3 origin, const double3 normal,
                       const double offset, const double forceConstant) {
  using GeoForce = GeometricRestraintForce<AT, CT>;
  gradient = make_double3(0.0, 0.0, 0.0);
  energy = 0.0;
  if (forceConstant == 0.0)
    return;

  // JEG261019: CHARMM uses raw Cartesian displacements for atomwise and RCM
  // restraints. Selected atoms are not unwrapped, and the center-reference
  // displacement is not replaced by a periodic image.
  const double dx = position.x - origin.x;
  const double dy = position.y - origin.y;
  const double dz = position.z - origin.z;
  const bool symmetric =
      types.y == static_cast<int>(GeoForce::Activation::SYMMETRIC);

  double coordinate = 0.0;
  double derivativeSign = 1.0;
  double3 coordinateGradient = normal;
  bool evaluate = true;

  if (types.x == static_cast<int>(GeoForce::Geometry::PLANE)) {
    // JEG261019: A symmetric plane uses signed distance. One-sided planes use
    // its magnitude; CHARMM chooses +normal at exactly zero distance.
    const double signedDistance = normal.x * dx + normal.y * dy + normal.z * dz;
    coordinate = (symmetric) ? signedDistance : fabs(signedDistance);
    derivativeSign = (!symmetric && (signedDistance < 0.0)) ? -1.0 : 1.0;
    if (!isfinite(signedDistance)) {
      atomicOr(error, NONFINITE_EVALUATION);
      evaluate = false;
    }
  } else {
    double3 radial = make_double3(dx, dy, dz);
    if (types.x == static_cast<int>(GeoForce::Geometry::CYLINDER)) {
      const double axialDistance =
          normal.x * dx + normal.y * dy + normal.z * dz;
      radial = make_double3(dx - axialDistance * normal.x,
                            dy - axialDistance * normal.y,
                            dz - axialDistance * normal.z);
    }

    // JEG261019: Squaring perpendicular components avoids cancellation near the
    // cylinder axis. CHARMM suppresses both energy and gradient near the axis
    // or sphere center, so apply the cutoff before activation or deviation.
    const double radiusSquared =
        radial.x * radial.x + radial.y * radial.y + radial.z * radial.z;
    if (!isfinite(radiusSquared)) {
      atomicOr(error, NONFINITE_EVALUATION);
      evaluate = false;
    } else if (radiusSquared <= 1.0e-10)
      evaluate = false;
    else {
      coordinate = sqrt(radiusSquared);
      coordinateGradient = make_double3(
          radial.x / coordinate, radial.y / coordinate, radial.z / coordinate);
    }
  }

  if (evaluate) {
    const double deviation = coordinate - offset;
    const bool active =
        symmetric ||
        ((types.y == static_cast<int>(GeoForce::Activation::INSIDE)) &&
         (deviation > 0.0)) ||
        ((types.y == static_cast<int>(GeoForce::Activation::OUTSIDE)) &&
         (deviation < 0.0));

    if (!isfinite(deviation))
      atomicOr(error, NONFINITE_EVALUATION);
    else if (active && (deviation != 0.0)) {
      // JEG261020: CHARMM squares the devition before scaling the harmonic
      // energy, then accumulates gradients only when that energy is nonzero.
      // This gate applies even when energy output is disabled.
      const double squaredDeviation = deviation * deviation;
      const double evaluatedEnergy = (0.5 * forceConstant) * squaredDeviation;

      if (!isfinite(evaluatedEnergy)) {
        atomicOr(error, NONFINITE_EVALUATION);
        return;
      }

      if (evaluatedEnergy != 0.0) {
        const double derivative = forceConstant * deviation;
        const double orientedDerivative = derivative * derivativeSign;
        gradient = make_double3(orientedDerivative * coordinateGradient.x,
                                orientedDerivative * coordinateGradient.y,
                                orientedDerivative * coordinateGradient.z);

        // JEG261019: CHARMM's harmonic energy is 0.5 * K * deviation^2.
        // derivative already contains one factor of K.
        if constexpr (calcEnergy)
          energy = evaluatedEnergy;
      }
    }
  }

  if (!isfinite(gradient.x) || !isfinite(gradient.y) || !isfinite(gradient.z) ||
      !isfinite(energy)) {
    atomicOr(error, NONFINITE_EVALUATION);
  }

  return;
}

template <typename AT, typename CT, int blockSize, bool calcEnergy>
__global__ static void EvaluateGeometricTermsKernel(
    double3 *__restrict__ membershipGradients,
    double *__restrict__ termEnergies, double3 *__restrict__ rcmCenters,
    double3 *__restrict__ rcmCenterGradients, int *__restrict__ error,
    const float4 *__restrict__ xyzq, const int *__restrict__ atomIndices,
    const double *__restrict__ masses,
    const std::size_t *__restrict__ termOffsets,
    const int4 *__restrict__ termTypes, const double3 *__restrict__ origins,
    const double3 *__restrict__ directions, const double *__restrict__ offsets,
    const double *__restrict__ forceConstants,
    const double *__restrict__ selectedMassTotals,
    const std::size_t *__restrict__ rcmChunkOffsets,
    const double3 *__restrict__ rcmPartials, const std::size_t numTerms) {
  using GeoForce = GeometricRestraintForce<AT, CT>;
  __shared__ BlockReduceSumStorage<double, blockSize, 1> cache;

  for (std::size_t i = blockIdx.x; i < numTerms; i += gridDim.x) {
    const int4 types = termTypes[i];
    const bool rcm = types.z == static_cast<int>(GeoForce::SelectionMode::RCM);
    const bool supportedGeometry =
        (types.x == static_cast<int>(GeoForce::Geometry::PLANE)) ||
        (types.x == static_cast<int>(GeoForce::Geometry::CYLINDER)) ||
        (types.x == static_cast<int>(GeoForce::Geometry::SPHERE));
    const bool supported =
        supportedGeometry &&
        (rcm ||
         (types.z == static_cast<int>(GeoForce::SelectionMode::ATOMWISE))) &&
        (types.w == static_cast<int>(GeoForce::PotentialType::HARMONIC));
    if (!supported && (threadIdx.x == 0))
      atomicOr(error, UNSUPPORTED_TERM);

    if (rcm) {
      if (threadIdx.x == 0) {
        double3 weighted = make_double3(0.0, 0.0, 0.0);
        for (std::size_t chunk = rcmChunkOffsets[i];
             chunk < rcmChunkOffsets[i + 1]; chunk++) {
          const double3 partial = rcmPartials[chunk];
          weighted.x += partial.x;
          weighted.y += partial.y;
          weighted.z += partial.z;
        }

        // JEG261019: CHARMM divides the signed mass-weighted coordinate sum by
        // the signed total mass. Divide directly because forming a reciprocal
        // first can overflow for tiny totals. Retain the raw center; no
        // periodic image shift if applied. The tensor uses distributed atom
        // gradients.
        const double total = selectedMassTotals[i];
        const double3 center = make_double3(
            weighted.x / total, weighted.y / total, weighted.z / total);

        double3 gradient = make_double3(0.0, 0.0, 0.0);
        double energy = 0.0;

        if (!isfinite(center.x) || !isfinite(center.y) || !isfinite(center.z))
          atomicOr(error, NONFINITE_EVALUATION);
        else if (supported) {
          EvaluateGeometricPoint<AT, CT, calcEnergy>(
              gradient, energy, error, center, types, origins[i], directions[i],
              offsets[i], forceConstants[i]);
        }

        rcmCenters[i] = center;
        rcmCenterGradients[i] = gradient;
        termEnergies[i] = energy;
      }
      __syncthreads();

      const double3 centerGradient = rcmCenterGradients[i];
      const double total = selectedMassTotals[i];
      for (std::size_t j = termOffsets[i] + threadIdx.x; j < termOffsets[i + 1];
           j += blockDim.x) {
        const double fraction = masses[atomIndices[j]] / total;
        const double3 gradient = make_double3(fraction * centerGradient.x,
                                              fraction * centerGradient.y,
                                              fraction * centerGradient.z);
        if (!isfinite(gradient.x) || !isfinite(gradient.y) ||
            !isfinite(gradient.z)) {
          atomicOr(error, NONFINITE_EVALUATION);
        }

        membershipGradients[j] = gradient;
      }
    } else {
      double partialEnergy[1] = {0.0};

      for (std::size_t j = termOffsets[i] + threadIdx.x; j < termOffsets[i + 1];
           j += blockDim.x) {
        double3 gradient = make_double3(0.0, 0.0, 0.0);
        double energy = 0.0;

        if (supported) {
          const float4 position = xyzq[atomIndices[j]];

          EvaluateGeometricPoint<AT, CT, calcEnergy>(
              gradient, energy, error,
              make_double3(position.x, position.y, position.z), types,
              origins[i], directions[i], offsets[i], forceConstants[i]);
        }

        membershipGradients[j] = gradient;
        partialEnergy[0] += energy;
      }

      BlockReduceSum<double, blockSize, 1>(partialEnergy, cache);
      if (threadIdx.x == 0) {
        rcmCenters[i] = make_double3(0.0, 0.0, 0.0);
        rcmCenterGradients[i] = make_double3(0.0, 0.0, 0.0);
        termEnergies[i] = partialEnergy[0];
        if (!isfinite(partialEnergy[0]))
          atomicOr(error, NONFINITE_EVALUATION);
      }
    }
    __syncthreads();
  }

  return;
}

template <typename AT, typename CT, int blockSize, bool calcEnergy,
          bool calcVirial>
__global__ static void StageGeometricOutputKernel(
    longlong3 *__restrict__ pendingForces, double *__restrict__ pendingEnergy,
    double *__restrict__ virialPartials, int *__restrict__ error,
    const float4 *__restrict__ xyzq,
    const std::size_t *__restrict__ atomMemberships,
    const double3 *__restrict__ membershipGradients,
    const std::size_t *__restrict__ atomOffsets, const int numAtoms,
    const AT *__restrict__ forces, const int forceStride,
    const double *__restrict__ energy, const double *__restrict__ termEnergies,
    const std::size_t numTerms) {
  __shared__ BlockReduceSumStorage<double, blockSize, 9> virialCache;
  double moment[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

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

    if constexpr (calcVirial) {
      if (atomOffsets[i] != atomOffsets[i + 1]) {
        // JEG261020: CHARMM adds overlapping term gradients in insertion order,
        // so form the external moment from that sum and the raw atom
        // coordinate. Use only this evaluation's double gradient, before
        // fixed-point conversion.
        const float4 position = xyzq[i];
        const double coordinate[3] = {static_cast<double>(position.x),
                                      static_cast<double>(position.y),
                                      static_cast<double>(position.z)};
        for (int r = 0; r < 3; r++) {
          for (int c = 0; c < 3; c++) {
            const int comp = r * 3 + c;
            moment[comp] += coordinate[r] * gradient[c];
            if (!isfinite(moment[comp]))
              atomicOr(error, NONFINITE_EVALUATION);
          }
        }
      }
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
      // particular, CT=float can round an otherwise representable value to
      // the upper bound
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

  if constexpr (calcVirial) {
    BlockReduceSum<double, blockSize, 9>(moment, virialCache);
    if (threadIdx.x == 0) {
      for (int comp = 0; comp < 9; comp++) {
        if (!isfinite(moment[comp]))
          atomicOr(error, NONFINITE_EVALUATION);
        virialPartials[blockIdx.x * 9 + comp] = moment[comp];
      }
    }
  }

  return;
}

__global__ static void StageGeometricVirialKernel(
    double *__restrict__ pendingVirial, int *__restrict__ error,
    const double *__restrict__ virial, const double *__restrict__ partials,
    const unsigned int numBlocks) {
  if (threadIdx.x < 9) {
    double total = virial[threadIdx.x];
    for (unsigned int block = 0; block < numBlocks; block++)
      total += partials[block * 9 + threadIdx.x];

    if (!isfinite(total))
      atomicOr(error, NONFINITE_EVALUATION);

    pendingVirial[threadIdx.x] = total;
  }

  return;
}

template <typename AT, bool calcEnergy, bool calcVirial>
__global__ static void CommitGeometricOutputKernel(
    AT *__restrict__ forces, double *__restrict__ energy,
    double *__restrict__ virial, const int forceStride,
    const longlong3 *__restrict__ pendingForces, const int numAtoms,
    const double *__restrict__ pendingEnergy,
    const double *__restrict__ pendingVirial) {
  if constexpr (calcEnergy) {
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
      *energy = *pendingEnergy;
  }

  if constexpr (calcVirial) {
    if ((blockIdx.x == 0) && (threadIdx.x < 9))
      virial[threadIdx.x] = pendingVirial[threadIdx.x];
  }

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
  if (this->getNumTerms() == 0)
    return;

  APOCHARMM_REQUIRE(xyzq != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "Coordinate-charge array must not be null");
  this->syncDeviceData();

  constexpr unsigned int numThreads = RCM_CHUNK_SIZE;
  const unsigned int termBlocks = static_cast<unsigned int>(
      std::min<std::size_t>(this->getNumTerms(), 65535));
  const unsigned int atomBlocks =
      static_cast<unsigned int>(std::min<std::size_t>(
          (static_cast<std::size_t>(m_NumAtoms) + numThreads - 1) / numThreads,
          65535));
  int *error = m_EvaluationError.getDeviceArray().data();
  double *energy = m_EnergyVirial->getEnergyPointer("mmfp");
  double *virial = m_EnergyVirial->getVirialPointer()->virmat;

  cudaCheck(cudaMemsetAsync(error, 0, sizeof(int), *m_Stream));

  if (m_RcmChunkBegins.size() != 0) {
    const unsigned int chunkBlocks = static_cast<unsigned int>(
        std::min<std::size_t>(m_RcmChunkBegins.size(), 65535));
    cudaCheckLaunch(BuildRcmPartialsKernel<numThreads>
                    <<<chunkBlocks, numThreads, 0, *m_Stream>>>(
                        m_RcmPartials.getDeviceArray().data(), error, xyzq,
                        m_AtomIndices.getDeviceArray().data(),
                        m_Masses.getDeviceArray().data(),
                        m_RcmChunkBegins.getDeviceArray().data(),
                        m_RcmChunkEnds.getDeviceArray().data(),
                        m_RcmChunkBegins.size()););
  }

  if (calcEnergy && calcVirial) {
    cudaCheckLaunch(
        EvaluateGeometricTermsKernel<AT, CT, numThreads, true>
        <<<termBlocks, numThreads, 0, *m_Stream>>>(
            m_MembershipGradients.getDeviceArray().data(),
            m_TermEnergies.getDeviceArray().data(),
            m_RcmCenters.getDeviceArray().data(),
            m_RcmCenterGradients.getDeviceArray().data(), error, xyzq,
            m_AtomIndices.getDeviceArray().data(),
            m_Masses.getDeviceArray().data(),
            m_TermOffsets.getDeviceArray().data(),
            m_TermTypes.getDeviceArray().data(),
            m_Origins.getDeviceArray().data(),
            m_Directions.getDeviceArray().data(),
            m_Offsets.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_SelectedMassTotals.getDeviceArray().data(),
            m_RcmChunkOffsets.getDeviceArray().data(),
            m_RcmPartials.getDeviceArray().data(), this->getNumTerms()));
    cudaCheckLaunch(
        StageGeometricOutputKernel<AT, CT, numThreads, true, true>
        <<<atomBlocks, numThreads, 0, *m_Stream>>>(
            m_PendingForces.getDeviceArray().data(),
            m_PendingEnergy.getDeviceArray().data(),
            m_VirialPartials.getDeviceArray().data(), error, xyzq,
            m_AtomMemberships.getDeviceArray().data(),
            m_MembershipGradients.getDeviceArray().data(),
            m_AtomOffsets.getDeviceArray().data(), m_NumAtoms, m_Forces->xyz(),
            m_Forces->stride(), energy, m_TermEnergies.getDeviceArray().data(),
            this->getNumTerms()));
  } else if (calcEnergy && !calcVirial) {
    cudaCheckLaunch(
        EvaluateGeometricTermsKernel<AT, CT, numThreads, true>
        <<<termBlocks, numThreads, 0, *m_Stream>>>(
            m_MembershipGradients.getDeviceArray().data(),
            m_TermEnergies.getDeviceArray().data(),
            m_RcmCenters.getDeviceArray().data(),
            m_RcmCenterGradients.getDeviceArray().data(), error, xyzq,
            m_AtomIndices.getDeviceArray().data(),
            m_Masses.getDeviceArray().data(),
            m_TermOffsets.getDeviceArray().data(),
            m_TermTypes.getDeviceArray().data(),
            m_Origins.getDeviceArray().data(),
            m_Directions.getDeviceArray().data(),
            m_Offsets.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_SelectedMassTotals.getDeviceArray().data(),
            m_RcmChunkOffsets.getDeviceArray().data(),
            m_RcmPartials.getDeviceArray().data(), this->getNumTerms()));
    cudaCheckLaunch(
        StageGeometricOutputKernel<AT, CT, numThreads, true, false>
        <<<atomBlocks, numThreads, 0, *m_Stream>>>(
            m_PendingForces.getDeviceArray().data(),
            m_PendingEnergy.getDeviceArray().data(),
            m_VirialPartials.getDeviceArray().data(), error, xyzq,
            m_AtomMemberships.getDeviceArray().data(),
            m_MembershipGradients.getDeviceArray().data(),
            m_AtomOffsets.getDeviceArray().data(), m_NumAtoms, m_Forces->xyz(),
            m_Forces->stride(), energy, m_TermEnergies.getDeviceArray().data(),
            this->getNumTerms()));
  } else if (!calcEnergy && calcVirial) {
    cudaCheckLaunch(
        EvaluateGeometricTermsKernel<AT, CT, numThreads, false>
        <<<termBlocks, numThreads, 0, *m_Stream>>>(
            m_MembershipGradients.getDeviceArray().data(),
            m_TermEnergies.getDeviceArray().data(),
            m_RcmCenters.getDeviceArray().data(),
            m_RcmCenterGradients.getDeviceArray().data(), error, xyzq,
            m_AtomIndices.getDeviceArray().data(),
            m_Masses.getDeviceArray().data(),
            m_TermOffsets.getDeviceArray().data(),
            m_TermTypes.getDeviceArray().data(),
            m_Origins.getDeviceArray().data(),
            m_Directions.getDeviceArray().data(),
            m_Offsets.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_SelectedMassTotals.getDeviceArray().data(),
            m_RcmChunkOffsets.getDeviceArray().data(),
            m_RcmPartials.getDeviceArray().data(), this->getNumTerms()));
    cudaCheckLaunch(
        StageGeometricOutputKernel<AT, CT, numThreads, false, true>
        <<<atomBlocks, numThreads, 0, *m_Stream>>>(
            m_PendingForces.getDeviceArray().data(),
            m_PendingEnergy.getDeviceArray().data(),
            m_VirialPartials.getDeviceArray().data(), error, xyzq,
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
            m_TermEnergies.getDeviceArray().data(),
            m_RcmCenters.getDeviceArray().data(),
            m_RcmCenterGradients.getDeviceArray().data(), error, xyzq,
            m_AtomIndices.getDeviceArray().data(),
            m_Masses.getDeviceArray().data(),
            m_TermOffsets.getDeviceArray().data(),
            m_TermTypes.getDeviceArray().data(),
            m_Origins.getDeviceArray().data(),
            m_Directions.getDeviceArray().data(),
            m_Offsets.getDeviceArray().data(),
            m_ForceConstants.getDeviceArray().data(),
            m_SelectedMassTotals.getDeviceArray().data(),
            m_RcmChunkOffsets.getDeviceArray().data(),
            m_RcmPartials.getDeviceArray().data(), this->getNumTerms()));
    cudaCheckLaunch(
        StageGeometricOutputKernel<AT, CT, numThreads, false, false>
        <<<atomBlocks, numThreads, 0, *m_Stream>>>(
            m_PendingForces.getDeviceArray().data(),
            m_PendingEnergy.getDeviceArray().data(),
            m_VirialPartials.getDeviceArray().data(), error, xyzq,
            m_AtomMemberships.getDeviceArray().data(),
            m_MembershipGradients.getDeviceArray().data(),
            m_AtomOffsets.getDeviceArray().data(), m_NumAtoms, m_Forces->xyz(),
            m_Forces->stride(), energy, m_TermEnergies.getDeviceArray().data(),
            this->getNumTerms()));
  }

  if (calcVirial) {
    cudaCheckLaunch(StageGeometricVirialKernel<<<1, 32, 0, *m_Stream>>>(
        m_PendingVirial.getDeviceArray().data(), error, virial,
        m_VirialPartials.getDeviceArray().data(), atomBlocks));
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

  if (calcEnergy && calcVirial) {
    cudaCheckLaunch(CommitGeometricOutputKernel<AT, true, true>
                    <<<atomBlocks, numThreads, 0, *m_Stream>>>(
                        m_Forces->xyz(), energy, virial, m_Forces->stride(),
                        m_PendingForces.getDeviceArray().data(), m_NumAtoms,
                        m_PendingEnergy.getDeviceArray().data(),
                        m_PendingVirial.getDeviceArray().data()));
  } else if (calcEnergy && !calcVirial) {
    cudaCheckLaunch(CommitGeometricOutputKernel<AT, true, false>
                    <<<atomBlocks, numThreads, 0, *m_Stream>>>(
                        m_Forces->xyz(), energy, virial, m_Forces->stride(),
                        m_PendingForces.getDeviceArray().data(), m_NumAtoms,
                        m_PendingEnergy.getDeviceArray().data(),
                        m_PendingVirial.getDeviceArray().data()));
  } else if (!calcEnergy && calcVirial) {
    cudaCheckLaunch(CommitGeometricOutputKernel<AT, false, true>
                    <<<atomBlocks, numThreads, 0, *m_Stream>>>(
                        m_Forces->xyz(), energy, virial, m_Forces->stride(),
                        m_PendingForces.getDeviceArray().data(), m_NumAtoms,
                        m_PendingEnergy.getDeviceArray().data(),
                        m_PendingVirial.getDeviceArray().data()));
  } else {
    cudaCheckLaunch(CommitGeometricOutputKernel<AT, false, false>
                    <<<atomBlocks, numThreads, 0, *m_Stream>>>(
                        m_Forces->xyz(), energy, virial, m_Forces->stride(),
                        m_PendingForces.getDeviceArray().data(), m_NumAtoms,
                        m_PendingEnergy.getDeviceArray().data(),
                        m_PendingVirial.getDeviceArray().data()));
  }

  return;
}

template <typename AT, typename CT>
bool GeometricRestraintForce<AT, CT>::supportsPBC(
    const PBC pbc) const noexcept {
  return pbc == PBC::P1;
}

template <typename AT, typename CT>
std::vector<int> GeometricRestraintForce<AT, CT>::getSelectionIndices(
    const AtomSelection &selection) const {
  APOCHARMM_REQUIRE(selection.getNumAtoms() == m_NumAtoms,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Selection atom count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(selection.getNumAtoms()));
  return selection.getAtomIndices();
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::appendTerm(
    const std::vector<int> &atomIndices, const Geometry geometry,
    const std::array<double, 3> &origin, const std::array<double, 3> &direction,
    const double forceConstant, const double offset,
    const Activation activation, const SelectionMode selectionMode) {
  APOCHARMM_REQUIRE(atomIndices.size() <=
                        std::numeric_limits<std::size_t>::max() -
                            m_AtomIndices.size(),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Adding the selection would exceed the membership "
                    "representation limit");
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

  double3 normal = make_double3(0.0, 0.0, 0.0);
  if (geometry != Geometry::SPHERE) {
    const double norm = std::hypot(direction[0], direction[1], direction[2]);
    APOCHARMM_REQUIRE(std::isfinite(norm) && (norm > 0.0),
                      ApoCharmmErrorCode::InvalidArgument,
                      "Direction must have a positive finite norm");

    normal = make_double3(direction[0] / norm, direction[1] / norm,
                          direction[2] / norm);
  }

  APOCHARMM_REQUIRE((selectionMode == SelectionMode::ATOMWISE) ||
                        (selectionMode == SelectionMode::RCM),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Selection mode must be ATOMWISE or RCM; observed " +
                        std::to_string(static_cast<int>(selectionMode)));

  // JEG261027: CHARMM accepts an empty atomwise selection with zero
  // contribution. Validate parameters, then preserve existing terms, outputs
  // and device state. The native RCM no-op also skips center construction and
  // mass division.
  if (atomIndices.empty())
    return;

  const double selectedMassTotal =
      (selectionMode == SelectionMode::RCM)
          ? this->validateRcmMasses(m_Masses.getHostArray(), atomIndices, 0,
                                    atomIndices.size())
          : 0.0;

  // JEG261002: Build replacements first so partial host allocation cannot
  // install a term
  std::vector<int> indices = m_AtomIndices.getHostArray();
  std::vector<std::size_t> termOffsets = m_TermOffsets.getHostArray();
  std::vector<int4> types = m_TermTypes.getHostArray();
  std::vector<double3> origins = m_Origins.getHostArray();
  std::vector<double3> directions = m_Directions.getHostArray();
  std::vector<double> offsets = m_Offsets.getHostArray();
  std::vector<double> forceConstants = m_ForceConstants.getHostArray();
  std::vector<double> selectedMassTotals = m_SelectedMassTotals.getHostArray();

  indices.insert(indices.end(), atomIndices.begin(), atomIndices.end());
  termOffsets.push_back(indices.size());
  types.push_back(make_int4(static_cast<int>(geometry),
                            static_cast<int>(activation),
                            static_cast<int>(selectionMode),
                            static_cast<int>(PotentialType::HARMONIC)));
  origins.push_back(make_double3(origin[0], origin[1], origin[2]));
  directions.push_back(normal);
  offsets.push_back(offset);
  forceConstants.push_back(forceConstant);
  selectedMassTotals.push_back(selectedMassTotal);

  m_AtomIndices.getHostArray().swap(indices);
  m_TermOffsets.getHostArray().swap(termOffsets);
  m_TermTypes.getHostArray().swap(types);
  m_Origins.getHostArray().swap(origins);
  m_Directions.getHostArray().swap(directions);
  m_Offsets.getHostArray().swap(offsets);
  m_ForceConstants.getHostArray().swap(forceConstants);
  m_SelectedMassTotals.getHostArray().swap(selectedMassTotals);
  m_DeviceDataDirty = true;

  return;
}

template <typename AT, typename CT>
void GeometricRestraintForce<AT, CT>::validateMasses(
    const std::vector<double> &masses) const {
  APOCHARMM_REQUIRE(masses.size() == static_cast<std::size_t>(m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Mass count mismatch; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(masses.size()));

  for (std::size_t i = 0; i < masses.size(); i++) {
    APOCHARMM_REQUIRE(
        std::isfinite(masses[i]), ApoCharmmErrorCode::InvalidArgument,
        "Mass at atom index " + std::to_string(i) +
            " must be finite; observed " + std::to_string(masses[i]));
  }

  return;
}

template <typename AT, typename CT>
double GeometricRestraintForce<AT, CT>::validateRcmMasses(
    const std::vector<double> &masses, const std::vector<int> &indices,
    const std::size_t begin, const std::size_t end) const {
  double total = 0.0;

  for (std::size_t i = begin; i < end; i++)
    total += masses[indices[i]];

  APOCHARMM_REQUIRE(std::isfinite(total) && (total != 0.0),
                    ApoCharmmErrorCode::InvalidArgument,
                    "RCM total mass must be finite and nonzero; observed " +
                        std::to_string(total));

  for (std::size_t i = begin; i < end; i++) {
    APOCHARMM_REQUIRE(std::isfinite(masses[indices[i]] / total),
                      ApoCharmmErrorCode::InvalidArgument,
                      "RCM mass fraction must be finite at atom index " +
                          std::to_string(indices[i]));
  }

  return total;
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

  // JEG261019: Chunk sizes bound individual reduction work, never term size.
  // Atomwise terms occupy empty chunk ranges and never access masses.
  std::vector<std::size_t> chunkOffsets(m_TermTypes.size() + 1, 0);
  std::vector<std::size_t> chunkBegins, chunkEnds;
  for (std::size_t term = 0; term < m_TermTypes.size(); term++) {
    if (m_TermTypes[term].z == static_cast<int>(SelectionMode::RCM)) {
      const std::size_t end = m_TermOffsets[term + 1];
      std::size_t begin = m_TermOffsets[term];
      while (begin < end) {
        const std::size_t count =
            std::min<std::size_t>(end - begin, RCM_CHUNK_SIZE);
        chunkBegins.push_back(begin);
        chunkEnds.push_back(begin + count);
        begin += count;
      }
    }
    chunkOffsets[term + 1] = chunkBegins.size();
  }

  m_AtomOffsets.getHostArray().swap(atomOffsets);
  m_AtomMemberships.getHostArray().swap(memberships);
  m_RcmChunkOffsets.getHostArray().swap(chunkOffsets);
  m_RcmChunkBegins.getHostArray().swap(chunkBegins);
  m_RcmChunkEnds.getHostArray().swap(chunkEnds);

  // JEG261002: CudaContainer modifiers above deliberately changed only the
  // host mirrors
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
  upload(m_SelectedMassTotals);
  upload(m_RcmChunkOffsets);
  upload(m_RcmChunkBegins);
  upload(m_RcmChunkEnds);
  upload(m_AtomOffsets);
  upload(m_AtomMemberships);

  m_MembershipGradients.resize(m_AtomIndices.size());
  m_TermEnergies.resize(m_TermTypes.size());
  m_RcmPartials.resize(m_RcmChunkBegins.size());
  m_RcmCenters.resize(m_TermTypes.size());
  m_RcmCenterGradients.resize(m_TermTypes.size());

  m_DeviceDataDirty = false;

  return;
}

// JEG261002: Explicit instances of GeometricRestraintForce
template class GeometricRestraintForce<long long int, float>;
template class GeometricRestraintForce<long long int, double>;
