// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "RestartSubscriber.h"

#include "ApoCharmmError.h"
#include "CharmmContext.h"
#include "Constants.h"
#include "Crystal.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaNoseHooverIntegrator.h"
#include "DcdSubscriber.h"
#include "str_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

RestartSubscriber::RestartSubscriber(void) : Subscriber() {}

RestartSubscriber::RestartSubscriber(const std::filesystem::path &filePath)
    : Subscriber(filePath) {}

RestartSubscriber::RestartSubscriber(const std::filesystem::path &filePath,
                                     const int reportFrequency)
    : Subscriber(filePath, reportFrequency) {}

void RestartSubscriber::update(void) {
  APOCHARMM_REQUIRE(!m_FilePath.empty(), ApoCharmmErrorCode::NotInitialized,
                    "RestartSubscriber requires an output file before update");

  APOCHARMM_REQUIRE(m_CharmmContext != nullptr,
                    ApoCharmmErrorCode::NotInitialized,
                    "RestartSubscriber requires a CharmmContext before update");

  APOCHARMM_REQUIRE(m_Integrator != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "RestartSubscriber requires an integrator before update");

  auto nh = std::dynamic_pointer_cast<CudaNoseHooverIntegrator>(m_Integrator);
  auto lp =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(m_Integrator);
  auto lt =
      std::dynamic_pointer_cast<CudaLangevinThermostatIntegrator>(m_Integrator);

  APOCHARMM_REQUIRE(
      nh != nullptr || lp != nullptr || lt != nullptr,
      ApoCharmmErrorCode::NotImplemented,
      "RestartSubscriber supports only CudaNoseHooverIntegrator, "
      "CudaLangevinPistonIntegrator, and CudaLangevinThermostatIntegrator");

  APOCHARMM_REQUIRE(m_Integrator->getCharmmContext() != nullptr,
                    ApoCharmmErrorCode::NotInitialized,
                    "RestartSubscriber integrator has no CharmmContext");

  APOCHARMM_REQUIRE(m_Integrator->getCharmmContext() == m_CharmmContext,
                    ApoCharmmErrorCode::Runtime,
                    "RestartSubscriber CharmmContext does not match the "
                    "integrator CharmmContext");

  m_CharmmContext->requireCrystalBox();

  const std::vector<double> &box = m_CharmmContext->getBoxDimensions();
  const int natom = m_CharmmContext->getNumAtoms();
  const int ndegf = m_CharmmContext->getNumDegreesOfFreedom();
  const unsigned long long int completed = m_Integrator->getTotNumSteps();
  APOCHARMM_REQUIRE(
      natom > 0 && ndegf > 0, ApoCharmmErrorCode::NotInitialized,
      "RestartSubscriber requires atoms and positive degrees of freedom");
  APOCHARMM_REQUIRE(completed <= static_cast<unsigned long long int>(
                                     std::numeric_limits<std::int32_t>::max()),
                    ApoCharmmErrorCode::InvalidArgument,
                    "RestartSubscriber step exceeds the supported CHARMM "
                    "32-bit integer range");

  // XTLTYP and the number of piston degrees of freedom follow the context's
  // crystal type: CUBI 1, TETR 2 (a, c), ORTH 3 (a, b, c)
  std::string_view crystal;
  std::size_t pistonDegreesOfFreedom = 0;
  switch (m_CharmmContext->getCrystalType()) {
  case CRYSTAL::CUBIC:
    crystal = "CUBI";
    pistonDegreesOfFreedom = 1;
    break;
  case CRYSTAL::TETRAGONAL:
    crystal = "TETR";
    pistonDegreesOfFreedom = 2;
    break;
  case CRYSTAL::ORTHORHOMBIC:
    crystal = "ORTH";
    pistonDegreesOfFreedom = 3;
    break;
  default:
    APOCHARMM_THROW(
        ApoCharmmErrorCode::NotImplemented,
        "RestartSubscriber requires CUBIC, TETRAGONAL, or ORTHORHOMBIC");
  }

  // NSAVC is the coordinate-saving interval of the attached DCD output, or zero
  // when no trajectory is written. The integrator's cached interval is the one
  // it actually schedules
  const auto &subscribers = m_Integrator->getSubscribers();
  const auto &frequencies = m_Integrator->getReportFreqList();
  APOCHARMM_REQUIRE(subscribers.size() == frequencies.size(),
                    ApoCharmmErrorCode::Runtime,
                    "Integrator subscriber and frequency lists do not match");

  int nsavc = 0;
  for (std::size_t i = 0; i < subscribers.size(); i++) {
    if (!std::dynamic_pointer_cast<DcdSubscriber>(subscribers[i]))
      continue;
    APOCHARMM_REQUIRE(nsavc == 0 || nsavc == frequencies[i],
                      ApoCharmmErrorCode::NotImplemented,
                      "A CHARMM restart cannot encode multiple coordinate "
                      "saving frequencies");
    nsavc = frequencies[i];
  }

  std::array<double, 6> hdot{};
  double pnhv = 0.0;
  double pnhf = 0.0;
  std::string_view nativeKind = "NOSE_HOOVER";
  std::string_view rngKind = "NONE";
  std::uint64_t rngSeed = 0;
  std::string rngState;
  if (nh) {
    auto &velocity = nh->getNoseHooverPistonVelocity();
    auto &force = nh->getNoseHooverPistonForce();
    APOCHARMM_REQUIRE(velocity.size() == 1 && force.size() == 1,
                      ApoCharmmErrorCode::NotInitialized,
                      "Nose-Hoover restart state has invalid dimensions");
    velocity.transferToHost();
    force.transferToHost();
    pnhv = velocity[0];
    pnhf = force[0];
  } else if (lp) {
    auto &velocity = lp->getNoseHooverPistonVelocity();
    auto &force = lp->getNoseHooverPistonForce();
    auto &delta = lp->getLangevinPistonDeltaPosition();
    APOCHARMM_REQUIRE(velocity.size() == 1 && force.size() == 1 &&
                          delta.size() == pistonDegreesOfFreedom,
                      ApoCharmmErrorCode::NotInitialized,
                      "Langevin-piston restart state has invalid dimensions");
    velocity.transferToHost();
    force.transferToHost();
    delta.transferToHost();
    pnhv = velocity[0];
    pnhf = force[0];
    for (std::size_t i = 0; i < pistonDegreesOfFreedom; i++)
      hdot[i] = delta[i];
    nativeKind = "LANGEVIN_PISTON";
    rngKind = "PHILOX4_32_10";
    rngSeed = lp->getLangevinPistonFrictionSeed();
    rngState = lp->getRngStates();
  } else if (lt) {
    nativeKind = "LANGEVIN_THERMOSTAT";
    rngKind = "PHILOX4_32_10";
    rngSeed = lt->getThermostatRngSeed();
    rngState = lt->getRngStates();
  }

  auto &coordinates = m_CharmmContext->getCoordinatesChargesDP();
  auto &velocities = m_CharmmContext->getVelocitiesInverseMasses();
  auto &deltas = m_Integrator->getCoordsDeltaPrevious();
  APOCHARMM_REQUIRE(coordinates.size() == static_cast<std::size_t>(natom) &&
                        velocities.size() == static_cast<std::size_t>(natom) &&
                        deltas.size() == static_cast<std::size_t>(natom),
                    ApoCharmmErrorCode::NotInitialized,
                    "Restart per-atom state does not match NATOM");
  coordinates.transferToHost();
  velocities.transferToHost();
  deltas.transferToHost();

  // JHSTRT counts the steps in CHARMM's running temperature average selects its
  // two-step start when zero. An evolved snapshot starts a one-sample window
  // without requesting that initialization
  const int jhstrt = (completed == 0) ? 0 : 1;
  const double avetem =
      (completed == 0) ? 0.0 : m_CharmmContext->computeTemperature();
  constexpr std::size_t width = 22;
  constexpr int precision = 15;
  constexpr std::size_t lenenp = 96;
  constexpr std::size_t lenent = 192;
  constexpr std::size_t lenenv = 50;
  constexpr double fmark = -999.0;
  const std::array<double, 6> xtlabc{box[0], 0.0, box[1], 0.0, 0.0, box[2]};
  // PNH=0 resets the thermal-piston reference-energy origin
  const std::array<double, 3> thermostat{0.0, pnhv, pnhf};
  const std::array<double, 3> fit{0.0, 0.0, avetem}; // FITA, FITP, AVETEM
  // (EPROP(I),EPRPP(I),EPRP2P(I),I=1,LENENP): CHARMM recomputes the previous
  // energies it finds set to FMARK (PEPR, KEPR2, XTLPEP, XTLKEP, XTLKP2)
  std::array<double, 3 * lenenp> propertyHistory{};
  for (const int index : {21, 23, 30, 31, 32})
    propertyHistory[3 * index] = fmark;
  // Every other statistic is reset; the largest group is 3 * LENENT values
  const std::vector<double> zeros(3 * lenent, 0.0);

  // Finish validation and formatting before truncating an existing output
  std::ostringstream output;
  output.imbue(std::locale::classic());

  // Header (A4,2I6,2X,A4,2X,A4): REST, VERNUM, LDYNA, XTLTYP, NSED. READYN
  // reads at most one further A4 (the DHDGB field at offsets 28-31), so the
  // marker at offsets 32-34 is never consumed by CHARMM; it tells apoCHARMM
  // readers that the trailing RNG extension must be present
  output << "REST" << std::setw(6) << charmm::constants::vernum << std::setw(6)
         << 1 << "  " << crystal << "          " << "APO" << '\n'
         << '\n';
  output << std::setw(8) << 2 << " !NTITLE followed by title\n";
  output << std::left << std::setw(80)
         << "* APOCHARMM RESTART FILE; DIAGNOSTIC HISTORY RESET" << '\n'
         << std::setw(80)
         << "* LEAPFROG STATE; RESEED WHEN SWITCHING RANDOM NUMBER GENERATORS"
         << '\n'
         << std::right << '\n';

  // Crystal block, every record (3D22.15): XTLABC, HDOT, PNH/PNHV/PNHF, UC1A,
  // UC2A, UC1B, UC2B, then GRAD1A, GRAD1B, GRAD2A, GRAD2B
  output << " !CRYSTAL PARAMETERS\n";
  apo::write_fixed_width_fortran_values(output, xtlabc.data(), xtlabc.size(), 3,
                                        width, precision);
  apo::write_fixed_width_fortran_values(output, hdot.data(), hdot.size(), 3,
                                        width, precision);
  apo::write_fixed_width_fortran_values(output, thermostat.data(),
                                        thermostat.size(), 3, width, precision);
  apo::write_fixed_width_fortran_values(output, zeros.data(), 24, 3, width,
                                        precision);
  apo::write_fixed_width_fortran_values(output, zeros.data(), 4, 3, width,
                                        precision);

  // Metadata record (7I12,D22.15,I12,2I22): the scalar seed and zero CHARMM RNG
  // counts, so READYN keeps its own generator state
  output << "\n !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n";
  output << std::setw(12) << natom << std::setw(12) << completed
         << std::setw(12) << m_Integrator->getNumSteps() << std::setw(12)
         << nsavc << std::setw(12) << 0 << std::setw(12) << jhstrt
         << std::setw(12) << ndegf
         << apo::format_fortran_double(1.0, width, precision) << std::setw(12)
         << 0 << std::setw(22) << 0 << std::setw(22) << 0 << '\n';

  // Energies and statistics: default all-T selections, (I8,3D22.15), then per
  // group every (3D22.15) history record followed by every (2D22.15) record
  output << "\n !ENERGIES and STATISTICS\n"
         << std::string(lenenp, 'T') << '\n'
         << std::string(lenent, 'T') << '\n';
  output << std::setw(8) << 0; // ISTPSA
  apo::write_fixed_width_fortran_values(output, fit.data(), fit.size(), 3,
                                        width, precision);
  apo::write_fixed_width_fortran_values(output, propertyHistory.data(),
                                        propertyHistory.size(), 3, width,
                                        precision);
  apo::write_fixed_width_fortran_values(output, zeros.data(), 2 * lenenp, 2,
                                        width, precision);
  apo::write_fixed_width_fortran_values(output, zeros.data(), 3 * lenent, 3,
                                        width, precision);
  apo::write_fixed_width_fortran_values(output, zeros.data(), 2 * lenent, 2,
                                        width, precision);
  apo::write_fixed_width_fortran_values(output, zeros.data(), 3 * lenenv, 3,
                                        width, precision);
  apo::write_fixed_width_fortran_values(output, zeros.data(), 2 * lenenv, 2,
                                        width, precision);

  // Per-atom records, (3D22.15) each. WRIDYN's call site passes the current
  // coordinates as XOLD and the previous leapfrog displacement as X
  output << "\n !XOLD, YOLD, ZOLD\n";
  for (int i = 0; i < natom; i++) {
    const std::array<double, 3> row{coordinates[i].x, coordinates[i].y,
                                    coordinates[i].z};
    apo::write_fixed_width_fortran_values(output, row.data(), row.size(), 3,
                                          width, precision);
  }

  output << "\n !VX, VY, VZ\n";
  for (int i = 0; i < natom; i++) {
    const std::array<double, 3> row{velocities[i].x, velocities[i].y,
                                    velocities[i].z};
    apo::write_fixed_width_fortran_values(output, row.data(), row.size(), 3,
                                          width, precision);
  }

  output << "\n !X, Y, Z\n";
  for (int i = 0; i < natom; i++) {
    const std::array<double, 3> row{deltas[i].x, deltas[i].y, deltas[i].z};
    apo::write_fixed_width_fortran_values(output, row.data(), row.size(), 3,
                                          width, precision);
  }

  output << "\n !APOCHARMM RNG STATE\n1 " << nativeKind << ' ' << rngKind
         << '\n'
         << rngSeed << '\n'
         << rngState << '\n';
  APOCHARMM_REQUIRE(output.good(), ApoCharmmErrorCode::Runtime,
                    "Failed to format restart snapshot");

  this->openFile();
  m_FileStream << output.str();
  m_FileStream.close();
  APOCHARMM_REQUIRE(!m_FileStream.fail(), ApoCharmmErrorCode::Runtime,
                    "Failed to write restart file: " + m_FilePath.string());

  return;
}
