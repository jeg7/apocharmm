// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II
//
// ENDLICENSE

#include "DynaSubscriber.h"

#include "ApoCharmmError.h"
#include "CharmmContext.h"
#include "Crystal.h"
#include "CudaContainer.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaLeapFrogIntegrator.h"
#include "CudaNoseHooverIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "CudaVerletIntegrator.h"
#include "EnergyTable.h"
#include "ForceManager.h"
#include "str_utils.h"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

// CHARMM prints DYNA> lines every NPRINT = 10 steps by default.
DynaSubscriber::DynaSubscriber(const std::filesystem::path &filePath)
    : Subscriber(filePath, 10) {}

DynaSubscriber::DynaSubscriber(const std::filesystem::path &filePath,
                               const int reportFrequency)
    : Subscriber(filePath, reportFrequency) {}

void DynaSubscriber::update(void) {
  this->writeBlock();
  return;
}

void DynaSubscriber::onPropagationStart(const unsigned long long int startStep,
                                        const int numSteps) {
  static_cast<void>(startStep);
  static_cast<void>(numSteps);

  // An empty origin marks a new run.
  if (!m_StartStep.has_value())
    this->writeBlock();

  return;
}

void DynaSubscriber::writeBlock(void) {
  APOCHARMM_REQUIRE(!m_FilePath.empty(), ApoCharmmErrorCode::NotInitialized,
                    "DynaSubscriber requires an output file before update");

  APOCHARMM_REQUIRE(m_CharmmContext != nullptr,
                    ApoCharmmErrorCode::NotInitialized,
                    "DynaSubscriber requires a CharmmContext before update");

  APOCHARMM_REQUIRE(m_Integrator != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "DynaSubscriber requires an integrator before update");

  APOCHARMM_REQUIRE(m_FileStream.is_open(), ApoCharmmErrorCode::NotInitialized,
                    "DynaSubscriber output file is not open for writing: " +
                        m_FilePath.string());

  // These integrators compute the energies and the virial in each step that
  // ends on a report, so a block never shows stale values.
  const std::shared_ptr<CudaLangevinPistonIntegrator> piston =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(m_Integrator);
  APOCHARMM_REQUIRE(
      piston != nullptr ||
          std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(
              m_Integrator) != nullptr ||
          std::dynamic_pointer_cast<CudaNoseHooverIntegrator>(m_Integrator) !=
              nullptr ||
          std::dynamic_pointer_cast<CudaVelocityVerletIntegrator>(
              m_Integrator) != nullptr ||
          std::dynamic_pointer_cast<CudaLeapFrogIntegrator>(m_Integrator) !=
              nullptr,
      ApoCharmmErrorCode::NotImplemented,
      "DynaSubscriber supports only CudaLangevinThermostatIntegrator, "
      "CudaNoseHooverIntegrator, CudaLangevinPistonIntegrator, "
      "CudaVelocityVerletIntegrator, and CudaLeapFrogIntegrator");

  CudaContainer<double> &potentialEnergy =
      m_CharmmContext->getPotentialEnergy();
  const std::shared_ptr<ForceManager> forceManager =
      m_CharmmContext->getForceManager();
  APOCHARMM_REQUIRE(!forceManager->isComposite(),
                    ApoCharmmErrorCode::NotImplemented,
                    "DynaSubscriber does not support composite ForceManagers");

  // CHARMM's constant-pressure dynamics also prints the crystal parameters.
  std::string_view crystal;
  if (piston != nullptr) {
    switch (m_CharmmContext->getCrystalType()) {
    case CRYSTAL::CUBIC:
      crystal = "CUBI";
      break;
    case CRYSTAL::TETRAGONAL:
      crystal = "TETR";
      break;
    case CRYSTAL::ORTHORHOMBIC:
      crystal = "ORTH";
      break;
    default:
      APOCHARMM_THROW(ApoCharmmErrorCode::NotInitialized,
                      "DynaSubscriber requires a CUBIC, TETRAGONAL, or "
                      "ORTHORHOMBIC crystal type for the Langevin piston");
    }
  }

  const unsigned long long int completedSteps = m_Integrator->getTotNumSteps();
  const bool startsRun = !m_StartStep.has_value();

  // The integrator has not computed the energies of a run's starting
  // coordinates.
  if (startsRun)
    m_CharmmContext->calculateForces(false, true, true);

  potentialEnergy.transferToHost(cudaStreamLegacy);

  // getVirial() sums the virial on the host, so its host values are current.
  const CudaContainer<double> &virial = m_CharmmContext->getVirial();

  apo::EnergyProperties properties;
  properties.kineticEnergy = m_CharmmContext->getKineticEnergy();
  properties.potentialEnergy = potentialEnergy[0];
  properties.totalEnergy =
      properties.potentialEnergy + properties.kineticEnergy;
  properties.temperature = m_CharmmContext->computeTemperature();
  properties.internalVirial = (virial[0] + virial[4] + virial[8]) / 3.0;
  properties.virialKineticEnergy =
      -1.5 * (properties.internalVirial + properties.externalVirial);
  properties.volume = m_CharmmContext->getVolume();

  std::vector<double> pressureTensor;
  if (piston != nullptr) {
    const CudaContainer<double> &pressureScalar =
        piston->getInstantaneousPressureScalar();
    pressureScalar.transferToHost(cudaStreamLegacy);
    properties.internalPressure = pressureScalar[0];
    properties.surfaceTension = piston->getInstantaneousSurfaceTension();

    const CudaContainer<double> &pressure =
        piston->getInstantaneousPressureTensor();
    pressure.transferToHost(cudaStreamLegacy);
    pressureTensor = pressure.getHostArray();
  }

  const unsigned long long int runStart = m_StartStep.value_or(completedSteps);
  const double time =
      m_Integrator->getTimeStep() * static_cast<double>(completedSteps);

  apo::WriteEnergyTable(m_FileStream, apo::EnergyTableType::DYNA, startsRun,
                        completedSteps - runStart, time, properties,
                        forceManager->getEnergyComponents());

  // PRNXTLD (source/image/crystal.F90) writes these lines. apoCHARMM boxes are
  // orthogonal, and it has no lattice gradient, whose norm prints as zero.
  if (piston != nullptr) {
    const std::vector<double> &box = m_CharmmContext->getBoxDimensions();
    m_FileStream
        << " Crystal Parameters : Crystal Type = " << crystal << '\n'
        << "      DYNA A     = " << apo::format_fortran_fixed(box[0], 10, 5)
        << " B    = " << apo::format_fortran_fixed(box[1], 10, 5)
        << " C     = " << apo::format_fortran_fixed(box[2], 10, 5) << '\n'
        << "      DYNA Alpha = " << apo::format_fortran_fixed(90.0, 10, 5)
        << " Beta = " << apo::format_fortran_fixed(90.0, 10, 5)
        << " Gamma = " << apo::format_fortran_fixed(90.0, 10, 5) << '\n'
        << "      DYNA PIXX =  "
        << apo::format_fortran_fixed(pressureTensor[0], 10, 2)
        << " PIYY = " << apo::format_fortran_fixed(pressureTensor[4], 10, 2)
        << " PIZZ = " << apo::format_fortran_fixed(pressureTensor[8], 10, 2)
        << '\n'
        << "      DYNA PIXY =  "
        << apo::format_fortran_fixed(pressureTensor[1], 10, 2)
        << " PIXZ = " << apo::format_fortran_fixed(pressureTensor[2], 10, 2)
        << " PIYZ = " << apo::format_fortran_fixed(pressureTensor[5], 10, 2)
        << '\n'
        << "      DYNA Gradient Norm = "
        << apo::format_fortran_fixed(0.0, 10, 5) << "\n\n";
  }

  m_FileStream.flush();
  APOCHARMM_REQUIRE(m_FileStream.good(), ApoCharmmErrorCode::Runtime,
                    "Failed to write DYNA output: " + m_FilePath.string());

  m_StartStep = runStart;

  return;
}
