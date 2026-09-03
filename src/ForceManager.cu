// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "ForceManager.h"

#include "CharmmContext.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

ForceManager::ForceManager(void) {
  m_Psf = nullptr;
  m_Prm = nullptr;

  m_IsInitialized = false;

  m_BondedForceDirty = false;
  m_ReciprocalForceDirty = false;
  m_DirectForceDirty = false;

  m_BondedStream = nullptr;
  m_ReciprocalStream = nullptr;
  m_DirectStream = nullptr;
  m_ForceManagerStream = nullptr;

  m_BondedForceValues = nullptr;
  m_ReciprocalForceValues = nullptr;
  m_DirectForceValues = nullptr;
  m_TotalForceValues = nullptr;

  m_BoxX = -9999.9999f;
  m_BoxY = -9999.9999f;
  m_BoxZ = -9999.9999f;
  m_BoxDimensions = {-9999.9999, -9999.9999, -9999.9999};

  m_Kappa = 0.34;
  m_Cutoff = 14.0;
  m_Ctonnb = 12.0;
  m_Ctofnb = 10.0;

  m_NfftX = -1;
  m_NfftY = -1;
  m_NfftZ = -1;

  m_PmeSplineOrder = 4;

  m_Pbc = PBC::P1;

  m_BondedForcePtr = nullptr;
  m_ReciprocalForcePtr = nullptr;
  m_DirectForcePtr = nullptr;

  m_BondedVirial.resize(9);
  m_ReciprocalVirial.resize(9);
  m_DirectVirial.resize(9);
  m_SubscribedForceVirial.resize(9);
  m_TotalVirial.resize(9);

  m_BondedVirial.setToValue(0.0);
  m_ReciprocalVirial.setToValue(0.0);
  m_DirectVirial.setToValue(0.0);
  m_SubscribedForceVirial.setToValue(0.0);
  m_TotalVirial.setToValue(0.0);

  m_ClearGraphCreated = false;
  m_ClearGraph = nullptr;
  m_ClearGraphInstance = nullptr;

  m_ComputeDirectSpaceForces = true;

  m_VdwType = VDW_VFSW;

  m_PrintEnergyDecomposition = false;
}

ForceManager::ForceManager(std::shared_ptr<CharmmPSF> psf,
                           std::shared_ptr<CharmmParameters> prm)
    : ForceManager() {
  this->setPsf(psf);
  this->setPrm(prm);
}

ForceManager::ForceManager(const ForceManager &other) : ForceManager() {
  if (other.m_Psf != nullptr)
    m_Psf = std::make_shared<CharmmPSF>(*other.m_Psf);
  if (other.m_Prm != nullptr)
    m_Prm = std::make_shared<CharmmParameters>(*other.m_Prm);

  m_BoxX = other.m_BoxX;
  m_BoxY = other.m_BoxY;
  m_BoxZ = other.m_BoxZ;
  m_BoxDimensions = other.m_BoxDimensions;

  m_Kappa = other.m_Kappa;
  m_Cutoff = other.m_Cutoff;
  m_Ctonnb = other.m_Ctonnb;
  m_Ctofnb = other.m_Ctofnb;

  m_NfftX = other.m_NfftX;
  m_NfftY = other.m_NfftY;
  m_NfftZ = other.m_NfftZ;

  m_PmeSplineOrder = other.m_PmeSplineOrder;

  m_Pbc = other.m_Pbc;

  m_VdwType = other.m_VdwType;

  // TODO unittest this (check that copied fm has same attributes, check that
  // acting on copied fm does not change original fm attributes)
}

ForceManager::~ForceManager(void) noexcept { this->dealloc(); }

void ForceManager::setContext(std::shared_ptr<CharmmContext> ctx) {
  if (ctx == nullptr) {
    m_Context.reset();
    return;
  }

  m_Context = ctx;

  return;
}

void ForceManager::setPsf(std::shared_ptr<CharmmPSF> psf) {
  APOCHARMM_REQUIRE(psf != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "CharmmPSF must not be null");

  m_Psf = psf;
  // If changing the CharmmPSF, set "initialized" flag to FALSE
  m_IsInitialized = false;

  return;
}

void ForceManager::setPrm(std::shared_ptr<CharmmParameters> prm) {
  APOCHARMM_REQUIRE(prm != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "CharmmParameters must not be null");

  m_Prm = prm;

  if (m_IsInitialized) {
    m_BondedForceDirty = true;
    m_DirectForceDirty = true;
  }

  return;
}

void ForceManager::addPsf(const std::filesystem::path &psfPath) {
  m_Psf = std::make_shared<CharmmPSF>(psfPath);
  // If changing the CharmmPSF, set "initialized" flag to FALSE
  m_IsInitialized = false;
  return;
}

void ForceManager::addPrm(const std::filesystem::path &prmPath) {
  this->setPrm(std::make_shared<CharmmParameters>(prmPath));
  return;
}

void ForceManager::addPrm(const std::vector<std::filesystem::path> &prmList) {
  this->setPrm(std::make_shared<CharmmParameters>(prmList));
  return;
}

void ForceManager::setBoxDimensions(const std::vector<double> &boxDimensions) {
  this->checkBoxDimensions(boxDimensions);
  m_BoxDimensions = boxDimensions;

  m_BoxX = static_cast<float>(boxDimensions[0]);
  m_BoxY = static_cast<float>(boxDimensions[1]);
  m_BoxZ = static_cast<float>(boxDimensions[2]);

  if (m_BondedForcePtr != nullptr)
    m_BondedForcePtr->setBoxDimensions(boxDimensions);
  if (m_ReciprocalForcePtr != nullptr)
    m_ReciprocalForcePtr->setBoxDimensions(boxDimensions);
  if (m_DirectForcePtr != nullptr)
    m_DirectForcePtr->setBoxDimensions(boxDimensions);

  for (ForceView &forceView : m_ForceViews)
    forceView.setBoxDimensions(boxDimensions);

  return;
}

void ForceManager::setKappa(const float kappa) {
  APOCHARMM_REQUIRE(std::isfinite(kappa), ApoCharmmErrorCode::InvalidArgument,
                    "Kappa must be finite; observed " + std::to_string(kappa));

  APOCHARMM_REQUIRE(kappa >= 0.0f, ApoCharmmErrorCode::InvalidArgument,
                    "Kappa must be non-negative; observed " +
                        std::to_string(kappa));

  if (m_Kappa == kappa)
    return;

  m_Kappa = kappa;

  if (m_IsInitialized) {
    m_ReciprocalForceDirty = true;
    m_DirectForceDirty = true;
  }

  return;
}

void ForceManager::setCutoff(const float cutoff) {
  APOCHARMM_REQUIRE(std::isfinite(cutoff), ApoCharmmErrorCode::InvalidArgument,
                    "Cutoff must be finite; observed " +
                        std::to_string(cutoff));

  APOCHARMM_REQUIRE(cutoff > 0.0f, ApoCharmmErrorCode::InvalidArgument,
                    "Cutoff must be positive; observed " +
                        std::to_string(cutoff));

  if (m_Cutoff == cutoff)
    return;

  m_Cutoff = cutoff;

  if (m_IsInitialized)
    m_DirectForceDirty = true;

  return;
}

void ForceManager::setCtonnb(const float ctonnb) {
  APOCHARMM_REQUIRE(std::isfinite(ctonnb), ApoCharmmErrorCode::InvalidArgument,
                    "Ctonnb must be finite; observed " +
                        std::to_string(ctonnb));

  APOCHARMM_REQUIRE(ctonnb > 0.0f, ApoCharmmErrorCode::InvalidArgument,
                    "Ctonnb must be positive; observed " +
                        std::to_string(ctonnb));

  if (m_Ctonnb == ctonnb)
    return;

  m_Ctonnb = ctonnb;

  if (m_IsInitialized)
    m_DirectForceDirty = true;

  return;
}

void ForceManager::setCtofnb(const float ctofnb) {
  APOCHARMM_REQUIRE(std::isfinite(ctofnb), ApoCharmmErrorCode::InvalidArgument,
                    "Ctofnb must be finite; observed " +
                        std::to_string(ctofnb));

  APOCHARMM_REQUIRE(ctofnb > 0.0f, ApoCharmmErrorCode::InvalidArgument,
                    "Ctofnb must be positive; observed " +
                        std::to_string(ctofnb));

  if (m_Ctofnb == ctofnb)
    return;

  m_Ctofnb = ctofnb;

  if (m_IsInitialized)
    m_DirectForceDirty = true;

  return;
}

void ForceManager::setFFTGrid(const int nfftx, const int nffty,
                              const int nfftz) {
  APOCHARMM_REQUIRE(nfftx > 0, ApoCharmmErrorCode::InvalidArgument,
                    "NFFTX must be positive; observed " +
                        std::to_string(nfftx));
  APOCHARMM_REQUIRE(nffty > 0, ApoCharmmErrorCode::InvalidArgument,
                    "NFFTY must be positive; observed " +
                        std::to_string(nffty));
  APOCHARMM_REQUIRE(nfftz > 0, ApoCharmmErrorCode::InvalidArgument,
                    "NFFTZ must be positive; observed " +
                        std::to_string(nfftz));

  if ((m_NfftX == nfftx) && (m_NfftY == nffty) && (m_NfftZ == nfftz))
    return;

  m_NfftX = nfftx;
  m_NfftY = nffty;
  m_NfftZ = nfftz;

  if (m_IsInitialized)
    m_ReciprocalForceDirty = true;

  return;
}

void ForceManager::setPmeSplineOrder(const int pmeSplineOrder) {
  APOCHARMM_REQUIRE(pmeSplineOrder > 0, ApoCharmmErrorCode::InvalidArgument,
                    "PME spline order must be positive; observed " +
                        std::to_string(pmeSplineOrder));

  if (m_PmeSplineOrder == pmeSplineOrder)
    return;

  m_PmeSplineOrder = pmeSplineOrder;

  if (m_IsInitialized)
    m_ReciprocalForceDirty = true;

  return;
}

void ForceManager::setPeriodicBoundaryCondition(const PBC pbc) {
  if (m_Pbc == pbc)
    return;

  m_Pbc = pbc;

  if (m_IsInitialized) {
    m_ReciprocalForceDirty = true;
    m_DirectForceDirty = true;
  }

  return;
}

void ForceManager::setVdwType(const int vdwType) {
  APOCHARMM_REQUIRE((vdwType >= VDW_VSH) && (vdwType <= VDW_DBEXP),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Van der Waals type must be in [1, 6]; observed " +
                        std::to_string(vdwType));

  if (m_VdwType == vdwType)
    return;

  m_VdwType = vdwType;

  if (m_IsInitialized)
    m_DirectForceDirty = true;

  return;
}

void ForceManager::setPrintEnergyDecomposition(
    const bool printEnergyDecomposition) {
  m_PrintEnergyDecomposition = printEnergyDecomposition;
  return;
}

void ForceManager::addForceManager(std::shared_ptr<ForceManager> fm) {
  // JEG260802: This check is done to prevent irrelevant compiler warnings,
  // since the function does nothing but throw an error.
  APOCHARMM_REQUIRE(fm != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "Child ForceManager must not be null");
  APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                  "ForceManager does not support child ForceManagers");
}

std::shared_ptr<CharmmContext> ForceManager::getContext(void) {
  return m_Context.lock();
}

bool ForceManager::hasCharmmContext(void) const { return !m_Context.expired(); }

std::shared_ptr<CharmmPSF> ForceManager::getPsf(void) { return m_Psf; }

std::shared_ptr<CharmmParameters> ForceManager::getPrm(void) { return m_Prm; }

bool ForceManager::isInitialized(void) const { return m_IsInitialized; }

const CudaContainer<int4> &ForceManager::getShakeAtoms(void) const {
  return m_ShakeAtoms;
}

CudaContainer<int4> &ForceManager::getShakeAtoms(void) { return m_ShakeAtoms; }

const CudaContainer<float4> &ForceManager::getShakeParams(void) const {
  return m_ShakeParams;
}

CudaContainer<float4> &ForceManager::getShakeParams(void) {
  return m_ShakeParams;
}

const CudaEnergyVirial &ForceManager::getBondedEnergyVirial(void) const {
  return m_BondedEnergyVirial;
}

CudaEnergyVirial &ForceManager::getBondedEnergyVirial(void) {
  return m_BondedEnergyVirial;
}

const CudaEnergyVirial &ForceManager::getReciprocalEnergyVirial(void) const {
  return m_ReciprocalEnergyVirial;
}

CudaEnergyVirial &ForceManager::getReciprocalEnergyVirial(void) {
  return m_ReciprocalEnergyVirial;
}

const CudaEnergyVirial &ForceManager::getDirectEnergyVirial(void) const {
  return m_DirectEnergyVirial;
}

CudaEnergyVirial &ForceManager::getDirectEnergyVirial(void) {
  return m_DirectEnergyVirial;
}

std::map<std::string, double> ForceManager::getEnergyComponents(void) {
  this->checkInitialized();

  std::map<std::string, double> energyDecompositionMap;

  energyDecompositionMap["bond"] = m_BondedEnergyVirial.getEnergy("bond");
  energyDecompositionMap["angle"] = m_BondedEnergyVirial.getEnergy("angle");
  energyDecompositionMap["ureyb"] = m_BondedEnergyVirial.getEnergy("ureyb");
  energyDecompositionMap["dihe"] = m_BondedEnergyVirial.getEnergy("dihe");
  energyDecompositionMap["imdihe"] = m_BondedEnergyVirial.getEnergy("imdihe");
  energyDecompositionMap["cmap"] = m_BondedEnergyVirial.getEnergy("cmap");

  energyDecompositionMap["ewks"] = m_ReciprocalEnergyVirial.getEnergy("ewks");
  energyDecompositionMap["ewse"] = m_ReciprocalEnergyVirial.getEnergy("ewse");

  energyDecompositionMap["ewex"] = m_DirectEnergyVirial.getEnergy("ewex");
  energyDecompositionMap["elec"] = m_DirectEnergyVirial.getEnergy("elec");
  energyDecompositionMap["vdw"] = m_DirectEnergyVirial.getEnergy("vdw");

  // JEG260811: This should eventually be put in its own bucket. However, it is
  // fine for now. (CONS HARM, CONS HMCM, CONS RESD, etc. should have their own
  // buckets).
  double userEnergy = 0.0;
  for (const std::shared_ptr<CudaEnergyVirial> &energyVirial :
       m_EnergyVirials) {
    userEnergy += energyVirial->getEnergy();
  }
  energyDecompositionMap["user"] = userEnergy;

  return energyDecompositionMap;
}

std::shared_ptr<cudaStream_t> ForceManager::getBondedStream(void) {
  return m_BondedStream;
}

std::shared_ptr<cudaStream_t> ForceManager::getReciprocalStream(void) {
  return m_ReciprocalStream;
}

std::shared_ptr<cudaStream_t> ForceManager::getDirectStream(void) {
  return m_DirectStream;
}

std::shared_ptr<cudaStream_t> ForceManager::getForceManagerStream(void) {
  return m_ForceManagerStream;
}

std::shared_ptr<Force<long long int>> ForceManager::getBondedForcevalues(void) {
  this->checkInitialized();
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_BondedForceValues;
}
std::shared_ptr<Force<long long int>>
ForceManager::getReciprocalForcevalues(void) {
  this->checkInitialized();
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_ReciprocalForceValues;
}
std::shared_ptr<Force<long long int>> ForceManager::getDirectForcevalues(void) {
  this->checkInitialized();
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_DirectForceValues;
}
std::shared_ptr<Force<double>> ForceManager::getTotalForcevalues(void) {
  this->checkInitialized();
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_TotalForceValues;
}

std::shared_ptr<Force<double>> ForceManager::getForces(void) {
  this->checkInitialized();
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));
  return m_TotalForceValues;
}

int ForceManager::getForceStride(void) const {
  this->checkInitialized();
  return m_TotalForceValues->stride();
}

const std::vector<double> &ForceManager::getBoxDimensions(void) const {
  return m_BoxDimensions;
}

std::vector<double> &ForceManager::getBoxDimensions(void) {
  return m_BoxDimensions;
}

float ForceManager::getKappa(void) const { return m_Kappa; }

float ForceManager::getCutoff(void) const { return m_Cutoff; }

float ForceManager::getCtonnb(void) const { return m_Ctonnb; }

float ForceManager::getCtofnb(void) const { return m_Ctofnb; }

std::vector<int> ForceManager::getFFTGrid(void) const {
  return {m_NfftX, m_NfftY, m_NfftZ};
}

int ForceManager::getPmeSplineOrder(void) const { return m_PmeSplineOrder; }

PBC ForceManager::getPeriodicBoundaryCondition(void) const { return m_Pbc; }

CudaContainer<double> &ForceManager::getPotentialEnergy(void) {
  this->checkInitialized();
  return m_TotalPotentialEnergy;
}

float ForceManager::getPotentialEnergies(void) {
  this->checkInitialized();
  // TODO : Don't do this
  // Pb : this should not be done on the Host side
  // Copy every energy-virial to host
  m_DirectEnergyVirial.copyToHost();
  m_BondedEnergyVirial.copyToHost();
  m_ReciprocalEnergyVirial.copyToHost();
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

  // Add every energy component
  float totalBondedEnergy =
      static_cast<float>(m_BondedEnergyVirial.getEnergy("bond")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("angle")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("ureyb")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("dihe")) +
      static_cast<float>(m_BondedEnergyVirial.getEnergy("imdihe"));

  float totalNonBondedEnergy =
      static_cast<float>(m_ReciprocalEnergyVirial.getEnergy("ewks")) +
      static_cast<float>(m_ReciprocalEnergyVirial.getEnergy("ewse")) +
      static_cast<float>(m_DirectEnergyVirial.getEnergy("ewex")) +
      static_cast<float>(m_DirectEnergyVirial.getEnergy("elec")) +
      static_cast<float>(m_DirectEnergyVirial.getEnergy("vdw"));

  return (totalBondedEnergy + totalNonBondedEnergy);
}

CudaContainer<double> &ForceManager::getVirial(void) {
  this->checkInitialized();

  m_BondedEnergyVirial.getVirial(m_BondedVirial);
  m_ReciprocalEnergyVirial.getVirial(m_ReciprocalVirial);
  m_DirectEnergyVirial.getVirial(m_DirectVirial);

  m_BondedVirial.transferFromDevice();
  m_ReciprocalVirial.transferFromDevice();
  m_DirectVirial.transferFromDevice();

  if (m_Pbc == PBC::P21) {
    for (int i = 0; i < 9; i++)
      m_ReciprocalVirial[i] /= 2.0;
    m_ReciprocalVirial.transferToDevice();
  }

  for (int i = 0; i < 9; i++) {
    m_TotalVirial[i] =
        m_BondedVirial[i] + m_ReciprocalVirial[i] + m_DirectVirial[i];
  }

  for (std::size_t i = 0; i < m_EnergyVirials.size(); i++) {
    if (m_ForceViews[i].contributesVirial() == false)
      continue;

    m_EnergyVirials[i]->getVirial(m_SubscribedForceVirial);
    m_SubscribedForceVirial.transferToHost();

    for (int j = 0; j < 9; j++)
      m_TotalVirial[j] += m_SubscribedForceVirial[j];
  }

  m_TotalVirial.transferToDevice();

  return m_TotalVirial;
}

int ForceManager::getVdwType(void) const { return m_VdwType; }

// const std::vector<Bond> &ForceManager::getBonds(void) const {
//   return m_Psf->getBonds();
// }

// std::vector<Bond> &ForceManager::getBonds(void) { return m_Psf->getBonds(); }

bool ForceManager::isComposite(void) const { return false; }

const std::vector<std::shared_ptr<ForceManager>> &
ForceManager::getChildren(void) const {
  return m_Children;
}

std::vector<std::shared_ptr<ForceManager>> &ForceManager::getChildren(void) {
  return m_Children;
}

void ForceManager::initialize(void) {
  // Some sanity checks before starting
  APOCHARMM_REQUIRE(m_Psf != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF must be set before initializing ForceManager");
  APOCHARMM_REQUIRE(
      m_Prm != nullptr, ApoCharmmErrorCode::NotInitialized,
      "CharmmParameters must be set before initializing ForceManager");
  APOCHARMM_REQUIRE(
      (m_BoxX != -9999.9999f) && (m_BoxY != -9999.9999f) &&
          (m_BoxZ != -9999.9999f),
      ApoCharmmErrorCode::NotInitialized,
      "Box dimensions must be set before initializing ForceManager");

  APOCHARMM_REQUIRE((m_Cutoff > 0.0f) && (m_Cutoff <= m_BoxX / 2.0f),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Cutoff must be positive and not exceed half the X box "
                    "dimension; cutoff " +
                        std::to_string(m_Cutoff) + ", X box dimension " +
                        std::to_string(m_BoxX));

  // If nfft not given, use values via truncating
  if ((m_NfftX <= 0) || (m_NfftY <= 0) || (m_NfftZ <= 0)) {
    std::vector<int> nfft = this->computeFFTGridSize();
    this->setFFTGrid(nfft[0], nfft[1], nfft[2]);
  }

  const int numAtoms = m_Psf->getNumAtoms();

  APOCHARMM_REQUIRE(numAtoms > 0, ApoCharmmErrorCode::Runtime,
                    "CharmmPSF atom count must be positive; observed " +
                        std::to_string(numAtoms));

  // Bonded
  m_BondedStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_BondedStream.get()));

  m_BondedForceValues = std::make_shared<Force<long long int>>();
  m_BondedForceValues->realloc(numAtoms, 1.5f);

  this->rebuildBondedForce();

  // Reciprocal
  m_ReciprocalStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_ReciprocalStream.get()));

  m_ReciprocalForceValues = std::make_shared<Force<long long int>>();
  m_ReciprocalForceValues->realloc(numAtoms, 1.5f);

  this->rebuildReciprocalForce();

  // Direct
  m_DirectStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_DirectStream.get()));

  m_DirectForceValues = std::make_shared<Force<long long int>>();
  m_DirectForceValues->realloc(numAtoms, 1.5f);

  this->rebuildDirectForce();

  // Initialize any forces that are already subscribed
  for (ForceView &forceView : m_ForceViews) {
    forceView.initialize(numAtoms, {static_cast<double>(m_BoxX),
                                    static_cast<double>(m_BoxY),
                                    static_cast<double>(m_BoxZ)});
  }

  m_ForceManagerStream = std::make_shared<cudaStream_t>();
  cudaCheck(cudaStreamCreate(m_ForceManagerStream.get()));

  m_TotalForceValues = std::make_shared<Force<double>>();
  m_TotalForceValues->realloc(numAtoms, 1.5f);

  cudaCheck(cudaDeviceSynchronize());
  m_TotalPotentialEnergy.resize(1); // doing it for diffave and difflc; for Now

  m_BondedForceDirty = false;
  m_ReciprocalForceDirty = false;
  m_DirectForceDirty = false;

  m_IsInitialized = true;

  return;
}

void ForceManager::resetNeighborList(const float4 *xyzq) {
  // JEG260802: Did not add error checking here because this function is called
  // frequently. i.e. Did not want to slow dynamics down. Users should not be
  // calling this function themselves.

  if (m_DirectForceDirty) {
    // JEG260903: Reconstructing the direct backend also builds its new neighbor
    // list.
    this->rebuildDirtyForces(xyzq);
    return;
  }

  m_DirectForcePtr->resetNeighborList(xyzq, m_Psf->getNumAtoms());

  return;
}

void ForceManager::calcForcePart1(const bool reset, const bool calcEnergy,
                                  const bool calcVirial) {
  APOCHARMM_REQUIRE(
      !reset, ApoCharmmErrorCode::NotImplemented,
      "Force \"reset\" is not implemented (JEG260807: deprecate in future)");

  if (!m_ClearGraphCreated) {
    cudaCheck(cudaStreamBeginCapture(*m_ForceManagerStream,
                                     cudaStreamCaptureModeGlobal));
    m_BondedForceValues->clear(*m_ForceManagerStream);
    m_ReciprocalForceValues->clear(*m_ForceManagerStream);
    m_DirectForceValues->clear(*m_ForceManagerStream);
    cudaCheck(cudaStreamEndCapture(*m_ForceManagerStream, &m_ClearGraph));
    cudaCheck(cudaGraphInstantiate(&m_ClearGraphInstance, m_ClearGraph, NULL,
                                   NULL, 0));
    m_ClearGraphCreated = true;
  }
  cudaCheck(cudaGraphLaunch(m_ClearGraphInstance, *m_ForceManagerStream));

  // Clear the virials and energy
  if ((calcEnergy == true) && (calcVirial == true)) {
    m_BondedEnergyVirial.clear(*m_BondedStream);
    m_ReciprocalEnergyVirial.clear(*m_ReciprocalStream);
    m_DirectEnergyVirial.clear(*m_DirectStream);
  } else if (calcEnergy == true) {
    m_BondedEnergyVirial.clearEnergy(*m_BondedStream);
    m_ReciprocalEnergyVirial.clearEnergy(*m_ReciprocalStream);
    m_DirectEnergyVirial.clearEnergy(*m_DirectStream);
  } else if (calcVirial == true) {
    m_BondedEnergyVirial.clearVirial(*m_BondedStream);
    m_ReciprocalEnergyVirial.clearVirial(*m_ReciprocalStream);
    m_DirectEnergyVirial.clearVirial(*m_DirectStream);
  }

  for (ForceView &forceView : m_ForceViews)
    forceView.clear();

  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

  return;
}

void ForceManager::calcForcePart2(const float4 *xyzq, const bool calcEnergy,
                                  const bool calcVirial) {
  // JEG260902: This is added here so the neighbor list can be rebuilt. It
  // should not normally be called during dynamics unless you're doing something
  // absolutely wild.
  if (m_BondedForceDirty || m_ReciprocalForceDirty || m_DirectForceDirty)
    this->rebuildDirtyForces(xyzq);

  // JEG260802: Did not add error checking here because this function is called
  // frequently. i.e. Did not want to slow dynamics down. Users should not be
  // calling this function themselves.
  gpu_range_start("bonded");
  m_BondedForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  gpu_range_start("reciprocal");
  m_ReciprocalForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  gpu_range_start("direct");
  m_DirectForcePtr->calc_force(xyzq, calcEnergy, calcVirial);
  gpu_range_stop();

  for (ForceView &forceView : m_ForceViews) {
    const bool forceCalcVirial = calcVirial && forceView.contributesVirial();
    forceView.calcForce(xyzq, calcEnergy, forceCalcVirial);
  }

  return;
}

__global__ void convertLLIToFloat(int numAtoms, int stride,
                                  const long long int *__restrict__ forceLLI,
                                  float *__restrict__ forceF) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    float fx = ((float)forceLLI[index]) * INV_FORCE_SCALE;
    float fy = ((float)forceLLI[index + stride]) * INV_FORCE_SCALE;
    float fz = ((float)forceLLI[index + 2 * stride]) * INV_FORCE_SCALE;

    forceF[index] = fx;
    forceF[index + stride] = fy;
    forceF[index + 2 * stride] = fz;
  }
}

/**
 * @brief Sums the eleven built-in potential-energy components on the device.
 *
 * A single thread writes the result. All input and output pointers address one
 * device-resident `double` in kilocalories per mole.
 *
 * @param[out] potentialEnergy Aggregate energy destination.
 * @param[in] bond Bond energy.
 * @param[in] angl Angle energy.
 * @param[in] urey Urey-Bradley energy.
 * @param[in] dihe Dihedral energy.
 * @param[in] impr Improper-dihedral energy.
 * @param[in] cmap CMAP energy.
 * @param[in] ewks reciprocal-space k-space energy.
 * @param[in] ewse reciprocal-space self energy.
 * @param[in] ewex Ewald exclusion energy.
 * @param[in] elec Direct-space electrostatic energy.
 * @param[in] vdwe Van der Waals energy.
 *
 * @internal
 */
__global__ static void UpdatePotentialEnergyKernel(
    double *__restrict__ potentialEnergy, const double *__restrict__ bond,
    const double *__restrict__ angl, const double *__restrict__ urey,
    const double *__restrict__ dihe, const double *__restrict__ impr,
    const double *__restrict__ cmap, const double *__restrict__ ewks,
    const double *__restrict__ ewse, const double *__restrict__ ewex,
    const double *__restrict__ elec, const double *__restrict__ vdwe) {
  if (threadIdx.x == 0) {
    *potentialEnergy = *bond + *angl + *urey + *dihe + *impr + *cmap + *ewks +
                       *ewse + *ewex + *elec + *vdwe;
  }
  return;
}

/**
 * @brief Adds one subscribed-force energy to the aggregate device value.
 *
 * @param[in,out] pe Device pointer to one aggregate energy value in
 * kilocalories per mole.
 * @param[in] en Device pointer to one subscribed-force energy value in
 * kilocalories per mole.
 *
 * @internal
 */
__global__ static void
UpdatePotentialEnergyKernel2(double *__restrict__ pe,
                             const double *__restrict__ en) {
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
      (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
    *pe += *en;
  return;
}

void ForceManager::calcForcePart3(const float4 *xyzq, const bool calcEnergy,
                                  const bool calcVirial) {
  // JEG260802: Did not add error checking here because this function is called
  // frequently. i.e. Did not want to slow dynamics down. Users should not be
  // calling this function themselves.
  m_TotalForceValues->clear(*m_ForceManagerStream);

  cudaCheck(cudaStreamSynchronize(*m_BondedStream));
  m_TotalForceValues->add<double>(*m_BondedForceValues, *m_ForceManagerStream);

  cudaCheck(cudaStreamSynchronize(*m_ReciprocalStream));
  m_TotalForceValues->add<double>(*m_ReciprocalForceValues,
                                  *m_ForceManagerStream);

  cudaCheck(cudaStreamSynchronize(*m_DirectStream));
  m_TotalForceValues->add<double>(*m_DirectForceValues, *m_ForceManagerStream);

  for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
    cudaCheck(cudaStreamSynchronize(*m_ForceStreams[i]));
    m_TotalForceValues->add<double>(*m_ForceViews[i].getForce(),
                                    *m_ForceManagerStream);
  }

  // TODO : find a better way.
  // For now, as virial requires forces to be double
  cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

  if (calcVirial) {
    // Are we not computing virial twice ? I thought
    // directForcePtr->calc_force(xyzq, calcEnergy, calcVirial) would already
    // do it
    m_BondedForceValues->convert<double>(*m_BondedStream);
    m_ReciprocalForceValues->convert<double>(*m_ReciprocalStream);
    m_DirectForceValues->convert<double>(*m_DirectStream);
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      if (m_ForceViews[i].contributesVirial())
        m_ForceValues[i]->convert<double>(*m_ForceStreams[i]);
    }

    cudaCheck(cudaStreamSynchronize(*m_BondedStream));
    cudaCheck(cudaStreamSynchronize(*m_ReciprocalStream));
    cudaCheck(cudaStreamSynchronize(*m_DirectStream));
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      if (m_ForceViews[i].contributesVirial())
        cudaCheck(cudaStreamSynchronize(*m_ForceStreams[i]));
    }

    const int numAtoms = m_Psf->getNumAtoms();
    const int forceStride = m_TotalForceValues->stride();

    m_BondedEnergyVirial.calcVirial(
        numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
        m_BoxDimensions[2], forceStride,
        reinterpret_cast<double *>(m_BondedForceValues->xyz()),
        *m_BondedStream);
    // Reciprocal space virial has already been calculated in the scalar_sum
    // m_ReciprocalEnergyVirial.calcVirial(
    //     numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
    //     m_BoxDimensions[2], this->getForceStride(),
    //     reinterpret_cast<double *>(m_ReciprocalForceValues->xyz()),
    //     *m_ReciprocalStream);
    m_DirectEnergyVirial.calcVirial(
        numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
        m_BoxDimensions[2], forceStride,
        reinterpret_cast<double *>(m_DirectForceValues->xyz()),
        *m_DirectStream);
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      if (m_ForceViews[i].contributesVirial()) {
        m_EnergyVirials[i]->calcVirial(
            numAtoms, xyzq, m_BoxDimensions[0], m_BoxDimensions[1],
            m_BoxDimensions[2], forceStride,
            reinterpret_cast<double *>(m_ForceValues[i]->xyz()),
            *m_ForceStreams[i]);
      }
    }

    cudaCheck(cudaStreamSynchronize(*m_BondedStream));
    cudaCheck(cudaStreamSynchronize(*m_ReciprocalStream));
    cudaCheck(cudaStreamSynchronize(*m_DirectStream));
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      if (m_ForceViews[i].contributesVirial())
        cudaCheck(cudaStreamSynchronize(*m_ForceStreams[i]));
    }
  }

  // Copy everything (all EnergyVirials) to Host, add together
  if (calcEnergy) {
    m_BondedEnergyVirial.copyToHost();
    m_ReciprocalEnergyVirial.copyToHost();
    m_DirectEnergyVirial.copyToHost();
    for (std::size_t i = 0; i < m_ForceViews.size(); i++)
      m_EnergyVirials[i]->copyToHost();
    cudaCheck(cudaDeviceSynchronize());

    // totalBondedEnergy = bondedEnergyVirial.getEnergy("bond") +
    //                     bondedEnergyVirial.getEnergy("angle") +
    //                     bondedEnergyVirial.getEnergy("ureyb") +
    //                     bondedEnergyVirial.getEnergy("dihe") +
    //                     bondedEnergyVirial.getEnergy("imdihe") +
    //                     bondedEnergyVirial.getEnergy("cmap");

    // totalNonBondedEnergy = directEnergyVirial.getEnergy("ewex") +
    //                        directEnergyVirial.getEnergy("elec") +
    //                        directEnergyVirial.getEnergy("vdw") +
    //                        reciprocalEnergyVirial.getEnergy("ewks") +
    //                        reciprocalEnergyVirial.getEnergy("ewse");

    // JEG260802: Did not add error checking here because this function is
    // called frequently. i.e. Did not want to slow dynamics down. Users should
    // not be calling this function themselves.
    UpdatePotentialEnergyKernel<<<1, 32, 0, *m_ForceManagerStream>>>(
        m_TotalPotentialEnergy.getDeviceArray().data(),
        m_BondedEnergyVirial.getEnergyPointer("bond"),
        m_BondedEnergyVirial.getEnergyPointer("angle"),
        m_BondedEnergyVirial.getEnergyPointer("ureyb"),
        m_BondedEnergyVirial.getEnergyPointer("dihe"),
        m_BondedEnergyVirial.getEnergyPointer("imdihe"),
        m_BondedEnergyVirial.getEnergyPointer("cmap"),
        m_ReciprocalEnergyVirial.getEnergyPointer("ewks"),
        m_ReciprocalEnergyVirial.getEnergyPointer("ewse"),
        m_DirectEnergyVirial.getEnergyPointer("ewex"),
        m_DirectEnergyVirial.getEnergyPointer("elec"),
        m_DirectEnergyVirial.getEnergyPointer("vdw"));

    // JEG260514: For now, we are assuming that all added forces only have a
    // single energy component. This CudaEnergyVirial interface is not very
    // flexibile. Should be overhauled at some point.
    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      // JEG260802: Did not add error checking here because this function is
      // called frequently. i.e. Did not want to slow dynamics down. Users
      // should not be calling this function themselves.
      UpdatePotentialEnergyKernel2<<<1, 32, 0, *m_ForceManagerStream>>>(
          m_TotalPotentialEnergy.getDeviceArray().data(),
          m_EnergyVirials[i]->getEnergyPointer());
    }

    cudaCheck(cudaStreamSynchronize(*m_ForceManagerStream));

    if (m_PrintEnergyDecomposition) {
      std::cout << "bond energy         : "
                << m_BondedEnergyVirial.getEnergy("bond") << "\n";
      std::cout << "angle energy        : "
                << m_BondedEnergyVirial.getEnergy("angle") << "\n";
      std::cout << "ureyb energy        : "
                << m_BondedEnergyVirial.getEnergy("ureyb") << "\n";
      std::cout << "dihe energy         : "
                << m_BondedEnergyVirial.getEnergy("dihe") << "\n";
      std::cout << "imdihe energy       : "
                << m_BondedEnergyVirial.getEnergy("imdihe") << "\n";
      std::cout << "cmap energy         : "
                << m_BondedEnergyVirial.getEnergy("cmap") << "\n";

      std::cout << "recip kspace energy : "
                << m_ReciprocalEnergyVirial.getEnergy("ewks") << "\n";
      std::cout << "recip  self energy  : "
                << m_ReciprocalEnergyVirial.getEnergy("ewse") << "\n";

      std::cout << "ewex energy         : "
                << m_DirectEnergyVirial.getEnergy("ewex") << "\n";
      std::cout << "elec energy         : "
                << m_DirectEnergyVirial.getEnergy("elec") << "\n";
      std::cout << "vdw energy          : "
                << m_DirectEnergyVirial.getEnergy("vdw") << "\n";

      for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
        std::cout << m_ForceTags[i] << ": " << m_EnergyVirials[i]->getEnergy()
                  << "\n";
      }

      m_TotalPotentialEnergy.transferToHost();

      std::cout << "Total potential energy : " << m_TotalPotentialEnergy[0]
                << std::endl;
    }
  }

  return;
}

void ForceManager::calcForce(const float4 *xyzq, const bool reset,
                             const bool calcEnergy, const bool calcVirial) {
  // JEG260802: Did not add error checking here because this function is called
  // frequently. i.e. Did not want to slow dynamics down. Users should not be
  // calling this function themselves.
  this->calcForcePart1(reset, calcEnergy, calcVirial);
  this->calcForcePart2(xyzq, calcEnergy, calcVirial);
  this->calcForcePart3(xyzq, calcEnergy, calcVirial);
  return;
}

CudaContainer<double>
ForceManager::computeAllChildrenPotentialEnergy(const float4 *xyzq) {
  // JEG260802: This check is done to prevent irrelevant compiler warnings,
  // since the function does nothing but throw an error.
  APOCHARMM_REQUIRE(xyzq != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "Coordinate-charge array must not be null");
  APOCHARMM_THROW(
      ApoCharmmErrorCode::Runtime,
      "ForceManager does not support child potential-energy evaluation");
}

/**
 * @brief Tests the leading character of an atom-type string for hydrogen.
 *
 * @param[in] atomType Non-empty atom-type string.
 * @return `true` when the first character is `H`; otherwise `false`.
 *
 * @pre `atomType` is not empty.
 * @internal
 */
inline bool isHydrogen(const std::string &atomType) {
  return (atomType[0] == 'H');
}

/**
 * @brief Groups selected heavy-atom-hydrogen bonds into SHAKE records.
 *
 * The first pass builds a hydrogen-neighbor list per heavy atom. The second
 * pass encodes each non-empty group into one `int4` atom record and one
 * `float4` parameter record before assigning both host vectors to CUDA
 * containers.
 *
 * The implementation currently assumes no selected heavy atom has more than
 * three hydrogens and uses one hydrogen-mass and equilibrium-bond value for the
 * generated group.
 *
 * @internal
 */
void ForceManager::initializeHolonomicConstraintsVariables(void) {
  const int numAtoms = m_Psf->getNumAtoms();

  const auto &bonds = m_Psf->getBonds();
  const auto &atomNames = m_Psf->getAtomNames();
  const auto &atomTypes = m_Psf->getAtomTypes();
  const auto &atomMasses = m_Psf->getMasses();

  std::vector<int> numBondsH(numAtoms, 0);
  std::vector<std::vector<int>> hydrogenBonds(numAtoms);

  std::vector<int4> shakeAtoms;
  std::vector<float4> shakeParams;
  auto bondParams = m_Prm->getBondParams();

  for (const auto &bond : bonds) {
    // TODO : refine these selection criteria
    if (isHydrogen(atomTypes[bond.iatom]) ||
        isHydrogen(atomTypes[bond.jatom])) {
      if (!((atomTypes[bond.iatom] == "OT" and atomTypes[bond.jatom] == "HT") ||
            (atomTypes[bond.iatom] == "HT" and atomTypes[bond.jatom] == "OT") ||
            (atomTypes[bond.iatom][0] == 'H' and
             atomTypes[bond.jatom][0] == 'H'))) {
        int heavyAtom = -1, hydrogenAtom = -1;
        if (isHydrogen(atomTypes[bond.iatom])) {
          heavyAtom = bond.jatom;
          hydrogenAtom = bond.iatom;
        } else {
          heavyAtom = bond.iatom;
          hydrogenAtom = bond.jatom;
        }
        numBondsH[heavyAtom]++;
        hydrogenBonds[heavyAtom].push_back(hydrogenAtom);
      }
    }
  }

  for (int i = 0; i < numAtoms; i++) {
    if (numBondsH[i]) {
      std::vector<int> group;
      group = {i, -1, -1, -1};
      float totalMass = static_cast<float>(atomMasses[i]);
      float hydrogenMass = 0.0f;
      BondValues bondValue(0.0f, 0.0f);
      for (std::size_t j = 0; j < hydrogenBonds[i].size(); j++) {
        int hyd = hydrogenBonds[i][j];
        group[j + 1] = hyd;
        totalMass += atomMasses[j];
        hydrogenMass = atomMasses[hyd];

        std::string atomType0 = "", atomType1 = "";
        if (atomTypes[i] < atomTypes[hyd]) {
          atomType0 = atomTypes[i];
          atomType1 = atomTypes[hyd];
        } else {
          atomType0 = atomTypes[hyd];
          atomType1 = atomTypes[i];
        }

        BondKey bondKey(atomType0, atomType1);
        bondValue = bondParams[bondKey];
      }

      float avgMass =
          totalMass / static_cast<float>(hydrogenBonds[i].size() + 1);
      shakeAtoms.push_back({group[0], group[1], group[2], group[3]});

      float4 p = make_float4(1.0f / static_cast<float>(atomMasses[i]), avgMass,
                             bondValue.b0 * bondValue.b0, 1.0f / hydrogenMass);
      shakeParams.push_back(p);
    }
  }

  m_ShakeAtoms = shakeAtoms;
  m_ShakeParams = shakeParams;

  return;
}

std::vector<int> ForceManager::computeFFTGridSize(void) {
  this->checkBoxDimensions(m_BoxDimensions);
  int fx = 2 * (static_cast<int>(m_BoxX) / 2);
  int fy = 2 * (static_cast<int>(m_BoxY) / 2);
  int fz = 2 * (static_cast<int>(m_BoxZ) / 2);
  if (fx < 2) {
    fx = 2;
    std::cout << "Warning: boxx seems very small (" << m_BoxX
              << "), setting associated fft grid size to2\n";
  }
  if (fy < 2) {
    fy = 2;
    std::cout << "Warning: boxy seems very small (" << m_BoxY
              << "), setting associated fft grid size to2\n";
  }
  if (fz < 2) {
    fz = 2;
    std::cout << "Warning: boxz seems very small (" << m_BoxZ
              << "), setting associated fft grid size to2\n";
  }
  return {fx, fy, fz};
}

void ForceManager::checkBoxDimensions(
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

  return;
}

void ForceManager::rebuildBondedForce(void) {
  auto bondedParamsAndList = m_Prm->getBondedParamsAndLists(m_Psf);

  if (m_BondedForcePtr != nullptr)
    cudaCheck(cudaStreamSynchronize(*m_BondedStream));

  m_BondedForcePtr.reset();

  auto rebuiltForce = std::make_unique<CudaBondedForce<long long int, float>>(
      m_BondedEnergyVirial, "bond", "ureyb", "angle", "dihe", "imdihe", "cmap");

  rebuiltForce->setup_list(bondedParamsAndList.listsSize,
                           bondedParamsAndList.listVal, *m_BondedStream);
  rebuiltForce->setup_coef(bondedParamsAndList.paramsSize,
                           bondedParamsAndList.paramsVal);
  rebuiltForce->setBoxDimensions(m_BoxDimensions);
  rebuiltForce->setForce(m_BondedForceValues);
  rebuiltForce->setStream(m_BondedStream);

  m_BondedForcePtr = std::move(rebuiltForce);

  // SHAKE equilibrium bond lengths also depend on CharmmParameters
  this->initializeHolonomicConstraintsVariables();

  return;
}

void ForceManager::rebuildReciprocalForce(void) {
  if (m_ReciprocalForcePtr != nullptr)
    cudaCheck(cudaStreamSynchronize(*m_ReciprocalStream));

  // Destroy the old PME object before configuring the new one. This avoids
  // overlapping ownership of legacy reciprocal-grid texture state.
  m_ReciprocalForcePtr.reset();

  auto rebuiltForce =
      std::make_unique<CudaPMEReciprocalForce>(m_ReciprocalEnergyVirial);

  rebuiltForce->setPBC(m_Pbc);
  rebuiltForce->setParameters(m_NfftX, m_NfftY, m_NfftZ, m_PmeSplineOrder,
                              m_Kappa, *m_ReciprocalStream);
  rebuiltForce->setNumAtoms(m_Psf->getNumAtoms());
  rebuiltForce->setBoxDimensions(m_BoxDimensions);
  rebuiltForce->setForce(m_ReciprocalForceValues);
  rebuiltForce->setStream(m_ReciprocalStream);

  m_ReciprocalForcePtr = std::move(rebuiltForce);

  return;
}

void ForceManager::rebuildDirectForce(void) {
  const bool calculateVdw =
      (m_DirectForcePtr == nullptr) ? true : m_DirectForcePtr->get_calc_vdw();

  const bool calculateElec =
      (m_DirectForcePtr == nullptr) ? true : m_DirectForcePtr->get_calc_elec();

  auto iblo14 = m_Psf->getIblo14();
  auto inb14 = m_Psf->getInb14();
  auto vdwParamsAndTypes = m_Prm->getVdwParamsAndTypes(m_Psf);
  auto inExLists = m_Psf->getInclusionExclusionLists();

  if (m_DirectForcePtr != nullptr)
    cudaCheck(cudaStreamSynchronize(*m_DirectStream));

  // This must precede construction of the replacement. The direct-force
  // constructor requires its legacy texture references to be unbound.
  m_DirectForcePtr.reset();

  auto rebuiltForce =
      std::make_unique<CudaPMEDirectForce<long long int, float>>(
          m_DirectEnergyVirial, "vdw", "elec", "ewex");

  const bool qP21 = (m_Pbc == PBC::P21);

  // TODO this seems to do the job twice ?
  // 1. "directForcePtr->setup(boxx, kappa, ctofnb, (...) );"
  // 2. "directForcePtr->setBoxDimensions({boxx...});"

  // JEG260626: ctonnb and ctofnb were switched?
  // m_DirectForcePtr->setup(m_BoxX, m_BoxY, m_BoxZ, m_Kappa, m_Ctofnb,
  // m_Ctonnb,
  //                         1.0, m_VdwType, EWALD, q_p21);
  rebuiltForce->setup(m_BoxX, m_BoxY, m_BoxZ, m_Kappa, m_Ctonnb, m_Ctofnb, 1.0,
                      m_VdwType, EWALD, qP21);
  rebuiltForce->setBoxDimensions(m_BoxDimensions);
  rebuiltForce->setStream(m_DirectStream);
  rebuiltForce->setForce(m_DirectForceValues);
  rebuiltForce->setNumAtoms(m_Psf->getNumAtoms());
  rebuiltForce->setCutoff(m_Cutoff);
  rebuiltForce->setupSorted(m_Psf->getNumAtoms());
  rebuiltForce->setupTopologicalExclusions(m_Psf->getNumAtoms(), iblo14, inb14);
  rebuiltForce->setupNeighborList(m_Psf->getNumAtoms());
  rebuiltForce->set_vdwparam(vdwParamsAndTypes.vdwParams);
  rebuiltForce->set_vdwparam14(vdwParamsAndTypes.vdw14Params);
  rebuiltForce->set_vdwtype(vdwParamsAndTypes.vdwTypes);
  rebuiltForce->set_vdwtype14(vdwParamsAndTypes.vdw14Types);
  rebuiltForce->set_14_list(inExLists.sizes, inExLists.in14_ex14);

  // setup() enables both components. Preserve any component-selection state
  // maintained by a specialized ForceManager.
  rebuiltForce->set_calc_vdw(calculateVdw);
  rebuiltForce->set_calc_elec(calculateElec);

  m_DirectForcePtr = std::move(rebuiltForce);

  return;
}

void ForceManager::rebuildDirtyForces(const float4 *xyzq) {
  if (!m_BondedForceDirty && !m_ReciprocalForceDirty && !m_DirectForceDirty)
    return;

  if (m_DirectForceDirty) {
    APOCHARMM_REQUIRE(xyzq != nullptr, ApoCharmmErrorCode::InvalidArgument,
                      "Coordinate-charge array must not be null when "
                      "rebuilding the direct-space force");
  }

  // Leave the manager uninitialized if any reconstruction or neighbor-list
  // build fails. Dirty flags are cleared only after all requested work
  // succeeds.
  m_IsInitialized = false;

  if (m_BondedForceDirty)
    this->rebuildBondedForce();

  if (m_ReciprocalForceDirty)
    this->rebuildReciprocalForce();

  if (m_DirectForceDirty)
    this->rebuildDirectForce();

  // A newly constructed direct backend owns a newly-constructed, but empty,
  // neighbor-list object. Build it before making the manager usable again.
  if (m_DirectForceDirty)
    m_DirectForcePtr->resetNeighborList(xyzq, m_Psf->getNumAtoms());

  m_BondedForceDirty = false;
  m_ReciprocalForceDirty = false;
  m_DirectForceDirty = false;

  m_IsInitialized = true;

  return;
}

void ForceManager::checkInitialized(void) const {
  APOCHARMM_REQUIRE(m_IsInitialized, ApoCharmmErrorCode::NotInitialized,
                    "ForceManager must be initialized before this operation");
  return;
}

void ForceManager::dealloc(void) noexcept {
  if (m_BondedStream != nullptr) {
    destroy_cuda_stream_noexcept(m_BondedStream.get());
    m_BondedStream.reset();
  }

  if (m_ReciprocalStream != nullptr) {
    destroy_cuda_stream_noexcept(m_ReciprocalStream.get());
    m_ReciprocalStream.reset();
  }

  if (m_DirectStream != nullptr) {
    destroy_cuda_stream_noexcept(m_DirectStream.get());
    m_DirectStream.reset();
  }

  if (m_ForceManagerStream != nullptr) {
    destroy_cuda_stream_noexcept(m_ForceManagerStream.get());
    m_ForceManagerStream.reset();
  }

  if (m_ClearGraph != nullptr) {
    (void)cudaGraphDestroy(m_ClearGraph);
    m_ClearGraph = nullptr;
  }

  if (m_ClearGraphInstance != nullptr) {
    (void)cudaGraphExecDestroy(m_ClearGraphInstance);
    m_ClearGraphInstance = nullptr;
  }

  m_ClearGraphCreated = false;

  return;
}
