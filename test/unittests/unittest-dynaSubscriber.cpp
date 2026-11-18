// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "ApoCharmmError.h"
#include "AtomSelection.h"
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "Crystal.h"
#include "CudaIntegrator.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaNoseHooverIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "DynaSubscriber.h"
#include "EnergyTable.h"
#include "ForceManager.h"
#include "HarmonicRestraintForce.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int REPORT_FREQUENCY = 10;
constexpr int PRE_RUN_STEPS = 3;
constexpr int RANDOM_SEED = 314159;
constexpr double TIME_STEP = 0.001;
constexpr double REFERENCE_TEMPERATURE = 300.0;
constexpr double NOSE_HOOVER_PISTON_MASS = 300.0;
constexpr double LANGEVIN_THERMOSTAT_FRICTION = 1.0;
constexpr double LANGEVIN_PISTON_MASS = 5.0;
constexpr double LANGEVIN_PISTON_FRICTION = 5.0;
constexpr std::uint64_t LANGEVIN_RNG_SEED = 987654321ULL;
constexpr double FIELD_TOLERANCE = 1.0e-5;

const std::vector<double> BOX_DIMENSIONS = {50.0, 50.0, 50.0};
const std::vector<double> REFERENCE_PRESSURE_TENSOR = {1.0, 0.0, 0.0, 0.0, 1.0,
                                                       0.0, 0.0, 0.0, 1.0};
const std::string SEPARATOR = " ----------       ---------    ---------    "
                              "---------    ---------    ---------";

std::shared_ptr<CharmmContext> CreateContext(void) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCrystalType(CRYSTAL::CUBIC);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->setRandomSeed(RANDOM_SEED);
  ctx->assignVelocitiesAtTemperature(REFERENCE_TEMPERATURE);

  return ctx;
}

std::shared_ptr<CudaLangevinThermostatIntegrator>
CreateLangevinThermostatIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(TIME_STEP);
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setThermostatFriction(LANGEVIN_THERMOSTAT_FRICTION);
  integrator->setThermostatRngSeed(LANGEVIN_RNG_SEED);
  integrator->setCharmmContext(ctx);
  return integrator;
}

std::shared_ptr<CudaLangevinPistonIntegrator>
CreateLangevinPistonIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(TIME_STEP);
  integrator->useNoseHooverThermostat(true);
  integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
  integrator->setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
  integrator->setReferencePressure(REFERENCE_PRESSURE_TENSOR);
  integrator->setLangevinPistonFrictionSeed(LANGEVIN_RNG_SEED);
  integrator->setRemoveCenterOfMassFrequency(1000);
  integrator->setCharmmContext(ctx);
  integrator->setLangevinPistonMass(std::vector<double>(
      integrator->getLangevinPistonMass().size(), LANGEVIN_PISTON_MASS));
  integrator->setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION);
  return integrator;
}

std::vector<std::string> ReadLines(const std::filesystem::path &filePath) {
  std::istringstream input(apo_test::ReadTextFile(filePath));
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);)
    lines.push_back(line);
  return lines;
}

// Reads the numbers that follow the label of a DYNA line.
std::vector<double> ReadValues(const std::string &line,
                               const std::size_t labelWidth) {
  std::istringstream input(line.substr(labelWidth));
  std::vector<double> values;
  for (double value = 0.0; input >> value;)
    values.push_back(value);
  return values;
}

} // namespace

TEST_CASE("DynaSubscriberDefaultsToCharmmPrintFrequency") {
  const std::filesystem::path path = "tmpDynaSubscriberFrequency.out";
  apo_test::RemoveIfExists(path);

  {
    DynaSubscriber defaultSubscriber(path);
    CHECK(defaultSubscriber.getReportFrequency() == 10);

    DynaSubscriber explicitSubscriber(path, 3);
    CHECK(explicitSubscriber.getReportFrequency() == 3);
  }

  apo_test::RemoveIfExists(path);
}

TEST_CASE("DynaSubscriberValidatesRequiredState") {
  const std::filesystem::path path = "tmpDynaSubscriberState.out";
  apo_test::RemoveIfExists(path);

  {
    DynaSubscriber subscriber(path);
    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotInitialized,
        "DynaSubscriber requires a CharmmContext before update");

    // The base integrator does not compute energies on report steps.
    subscriber.setCharmmContext(CreateContext());
    subscriber.setIntegrator(std::make_shared<CudaIntegrator>(TIME_STEP));
    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotImplemented,
        "DynaSubscriber supports only CudaLangevinThermostatIntegrator, "
        "CudaNoseHooverIntegrator, CudaLangevinPistonIntegrator, "
        "CudaVelocityVerletIntegrator, and CudaLeapFrogIntegrator");
  }

  apo_test::RemoveIfExists(path);
}

TEST_CASE("DynaSubscriberWritesCharmmBlocksDuringDynamics") {
  const std::filesystem::path path = "tmpDynaSubscriberBlocks.out";
  apo_test::RemoveIfExists(path);

  auto ctx = CreateContext();
  auto integrator = CreateLangevinThermostatIntegrator(ctx);
  auto subscriber = std::make_shared<DynaSubscriber>(path, REPORT_FREQUENCY);

  // Steps taken before the subscription do not count toward the run's steps.
  integrator->propagate(PRE_RUN_STEPS);
  integrator->subscribe(subscriber);
  integrator->propagate(2 * REPORT_FREQUENCY);
  integrator->unsubscribe(subscriber);

  // The ions have no bonded terms, so there is no INTERN or CROSS line.
  const std::vector<std::string> header = {
      "DYNA DYN: Step         Time      TOTEner        TOTKe       ENERgy"
      "  TEMPerature",
      "DYNA PROP:             GRMS      HFCTote        HFCKe       EHFCor"
      "        VIRKe",
      "DYNA EXTERN:        VDWaals         ELEC       HBONds          ASP"
      "         USER",
      "DYNA EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor"
      "       EWUTil",
      "DYNA PRESS:            VIRE         VIRI       PRESSE       PRESSI"
      "       VOLUme",
      SEPARATOR};
  constexpr std::size_t BLOCK_SIZE = 6;

  const std::vector<std::string> lines = ReadLines(path);
  REQUIRE(lines.size() == header.size() + 3 * BLOCK_SIZE);

  for (std::size_t i = 0; i < header.size(); i++)
    CHECK(lines[i] == header[i]);

  // Blocks follow the header at steps 0, 10, and 20 of the run, and the time
  // column also counts the steps before it.
  for (std::size_t block = 0; block < 3; block++) {
    const std::size_t first = header.size() + block * BLOCK_SIZE;
    CHECK(lines[first + 1].compare(0, 14, "DYNA PROP>    ") == 0);
    CHECK(lines[first + 2].compare(0, 14, "DYNA EXTERN>  ") == 0);
    CHECK(lines[first + 3].compare(0, 14, "DYNA EWALD>   ") == 0);
    CHECK(lines[first + 5] == SEPARATOR);

    REQUIRE(lines[first].compare(0, 5, "DYNA>") == 0);
    const std::vector<double> dyn = ReadValues(lines[first], 5);
    REQUIRE(dyn.size() == 6);
    CHECK(dyn[0] == static_cast<double>(block * REPORT_FREQUENCY));
    CHECK(dyn[1] ==
          Approx(static_cast<double>(PRE_RUN_STEPS + block * REPORT_FREQUENCY) *
                 TIME_STEP)
              .margin(FIELD_TOLERANCE));
    CHECK(dyn[2] == Approx(dyn[3] + dyn[4]).margin(2.0 * FIELD_TOLERANCE));

    REQUIRE(lines[first + 4].compare(0, 14, "DYNA PRESS>   ") == 0);
    const std::vector<double> press = ReadValues(lines[first + 4], 14);
    REQUIRE(press.size() == 5);
    CHECK(press[0] == 0.0);
    CHECK(press[2] == 0.0);
    CHECK(press[3] == 0.0);
    CHECK(press[4] == Approx(ctx->getVolume()));
  }

  // The last block holds the values of the last step, not the starting ones.
  const std::vector<double> firstValues = ReadValues(lines[header.size()], 5);
  const std::vector<double> lastValues =
      ReadValues(lines[header.size() + 2 * BLOCK_SIZE], 5);
  CHECK(lastValues[4] != firstValues[4]);
  CHECK(lastValues[4] ==
        Approx(apo_test::CopyScalarToHost(ctx->getPotentialEnergy()))
            .margin(FIELD_TOLERANCE));
  CHECK(lastValues[3] ==
        Approx(ctx->getKineticEnergy()).margin(FIELD_TOLERANCE));

  apo_test::RemoveIfExists(path);
}

TEST_CASE("DynaSubscriberPrintsRestraintsInCharmmLines") {
  const std::filesystem::path path = "tmpDynaSubscriberRestraint.out";
  apo_test::RemoveIfExists(path);

  auto ctx = CreateContext();

  // A reference half an angstrom away gives the restraint a positive energy.
  CharmmCrd crd(apo_test::GetDataDir() / "nacl_pair.cor");
  std::vector<double3> reference = crd.getCoordinatesDP();
  reference[0].x += 0.5;

  auto restraint =
      std::make_shared<HarmonicRestraintForce<long long int, float>>(
          ctx->getPsf());
  restraint->setReferenceCoordinates(reference);
  restraint->setMasses(ctx->getPsf()->getMasses());
  restraint->setSelection(
      AtomSelection(ctx->getNumAtoms(), AtomSelection::InitialValue::ALL));
  restraint->setForceConstant(1.0);

  std::shared_ptr<ForceManager> fm = ctx->getForceManager();
  fm->subscribe(restraint, "harm", restraint->getStream(),
                restraint->getForce(), restraint->getEnergyVirial());

  auto integrator = CreateLangevinThermostatIntegrator(ctx);
  auto subscriber = std::make_shared<DynaSubscriber>(path, REPORT_FREQUENCY);

  integrator->subscribe(subscriber);
  integrator->propagate(REPORT_FREQUENCY);
  integrator->unsubscribe(subscriber);
  fm->unsubscribe(restraint);

  const std::vector<std::string> lines = ReadLines(path);
  REQUIRE(lines.size() == 7 + 2 * 7);
  CHECK(lines[4] == "DYNA CONSTR:       HARMonic    CDIHedral          CIC"
                    "     RESDistance       NOE");

  // The restraint energy is in the HARMonic column, and USER stays zero.
  REQUIRE(lines[9].compare(0, 14, "DYNA EXTERN>  ") == 0);
  const std::vector<double> external = ReadValues(lines[9], 14);
  REQUIRE(external.size() == 5);
  CHECK(external[4] == 0.0);

  REQUIRE(lines[11].compare(0, 14, "DYNA CONSTR>  ") == 0);
  const std::vector<double> constraint = ReadValues(lines[11], 14);
  REQUIRE(constraint.size() == 5);
  CHECK(constraint[0] > 0.0);
  CHECK(constraint[1] == 0.0);
  CHECK(constraint[3] == 0.0);

  apo_test::RemoveIfExists(path);
}

TEST_CASE("DynaSubscriberPrintsPistonPressureAndCrystalParameters") {
  const std::filesystem::path path = "tmpDynaSubscriberPiston.out";
  apo_test::RemoveIfExists(path);

  auto ctx = CreateContext();
  auto integrator = CreateLangevinPistonIntegrator(ctx);
  auto subscriber = std::make_shared<DynaSubscriber>(path, REPORT_FREQUENCY);

  integrator->subscribe(subscriber);
  integrator->propagate(REPORT_FREQUENCY);
  integrator->unsubscribe(subscriber);

  const std::vector<std::string> lines = ReadLines(path);

  int numCrystalBlocks = 0;
  std::size_t firstPress = lines.size();
  std::size_t lastPress = lines.size();
  for (std::size_t i = 0; i < lines.size(); i++) {
    if (lines[i].compare(0, 14, "DYNA PRESS>   ") == 0) {
      if (firstPress == lines.size())
        firstPress = i;
      lastPress = i;
    }

    if (lines[i] != " Crystal Parameters : Crystal Type = CUBI")
      continue;

    numCrystalBlocks++;
    REQUIRE(i + 6 < lines.size());
    CHECK(lines[i + 1].compare(0, 19, "      DYNA A     = ") == 0);
    CHECK(lines[i + 2] == "      DYNA Alpha =   90.00000 Beta =   90.00000"
                          " Gamma =   90.00000");
    CHECK(lines[i + 3].compare(0, 19, "      DYNA PIXX =  ") == 0);
    CHECK(lines[i + 4].compare(0, 19, "      DYNA PIXY =  ") == 0);
    CHECK(lines[i + 5] == "      DYNA Gradient Norm =    0.00000");
    CHECK(lines[i + 6].empty());
  }

  CHECK(numCrystalBlocks == 2);

  // A new piston has computed no pressure before its first step.
  REQUIRE(firstPress < lines.size());
  const std::vector<double> firstPressValues =
      ReadValues(lines[firstPress], 14);
  REQUIRE(firstPressValues.size() == 5);
  CHECK(firstPressValues[3] == 0.0);

  REQUIRE(lastPress < lines.size());
  const std::vector<double> press = ReadValues(lines[lastPress], 14);
  REQUIRE(press.size() == 5);
  CHECK(press[3] == Approx(apo_test::CopyScalarToHost(
                               integrator->getInstantaneousPressureScalar()))
                        .margin(FIELD_TOLERANCE));
  CHECK(press[4] == Approx(ctx->getVolume()).margin(FIELD_TOLERANCE));

  apo_test::RemoveIfExists(path);
}

TEST_CASE("DynaSubscriberShowsCurrentEnergiesForEachIntegrator") {
  const std::filesystem::path path = "tmpDynaSubscriberIntegrators.out";

  // Each integrator computes the energies of the steps it reports, so the last
  // block shows the last evaluation rather than the starting energy.
  const auto checkLastBlock =
      [&path](std::shared_ptr<CharmmContext> ctx,
              std::shared_ptr<CudaIntegrator> integrator) -> void {
    apo_test::RemoveIfExists(path);
    auto subscriber = std::make_shared<DynaSubscriber>(path, REPORT_FREQUENCY);
    integrator->subscribe(subscriber);
    integrator->propagate(REPORT_FREQUENCY);
    integrator->unsubscribe(subscriber);

    std::vector<std::string> records;
    for (const std::string &line : ReadLines(path)) {
      if (line.compare(0, 5, "DYNA>") == 0)
        records.push_back(line);
    }

    REQUIRE(records.size() == 2);
    const std::vector<double> first = ReadValues(records[0], 5);
    const std::vector<double> last = ReadValues(records[1], 5);
    REQUIRE(first.size() == 6);
    REQUIRE(last.size() == 6);
    CHECK(last[4] != first[4]);
    CHECK(last[4] ==
          Approx(apo_test::CopyScalarToHost(ctx->getPotentialEnergy()))
              .margin(FIELD_TOLERANCE));

    apo_test::RemoveIfExists(path);
  };

  SECTION("CudaNoseHooverIntegrator") {
    auto ctx = CreateContext();
    auto integrator = std::make_shared<CudaNoseHooverIntegrator>(TIME_STEP);
    integrator->setReferenceTemperature(REFERENCE_TEMPERATURE);
    integrator->setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS);
    integrator->setRemoveCenterOfMassFrequency(1000);
    integrator->setCharmmContext(ctx);
    checkLastBlock(ctx, integrator);
  }

  SECTION("CudaVelocityVerletIntegrator") {
    auto ctx = CreateContext();
    auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(TIME_STEP);
    integrator->setRemoveCenterOfMassFrequency(1000);
    integrator->setCharmmContext(ctx);
    checkLastBlock(ctx, integrator);
  }
}

TEST_CASE("EnergyTableFollowsCharmmLineRulesAndFields") {
  std::map<std::string, double> terms;
  for (const char *key :
       {"bond", "angle", "ureyb", "dihe", "imdihe", "cmap", "vdw", "elec",
        "user", "ewks", "ewse", "ewex", "harm", "resd", "hmcm", "geo"}) {
    terms[key] = 0.0;
  }

  SECTION("ENER") {
    // setfmt drops a decimal above 1e6 and switches to E13.5 above 1e11, and
    // terms below 0.49e-6 leave their line out.
    terms["bond"] = 2000000.123456;
    terms["hmcm"] = 1.0e-7;
    terms["geo"] = 2.0e11;

    apo::EnergyProperties properties;
    properties.potentialEnergy = 1.25;
    properties.energyChange = -0.5;
    properties.gradientRms = 3.0;

    std::ostringstream output;
    apo::WriteEnergyTable(output, apo::EnergyTableType::ENER, true, 3, 0.0,
                          properties, terms);
    CHECK(
        output.str() ==
        "ENER ENR:  Eval#     ENERgy      Delta-E         GRMS\n"
        "ENER INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals"
        "    IMPRopers\n"
        "ENER MMFP:              GEO        MDIP           SSBP        SHEL"
        "       DROFfa\n" +
            SEPARATOR +
            "\n"
            "ENER>        3      1.25000     -0.50000      3.00000\n"
            "ENER INTERN>   2000000.1235       0.0000       0.0000       0.0000"
            "       0.0000\n"
            "ENER MMFP>      0.20000E+12  0.00000E+00  0.00000E+00  0.00000E+00"
            "  0.00000E+00\n" +
            SEPARATOR + "\n");
  }

  SECTION("DYNA") {
    // A nonzero surface tension adds the XTLE line, and a step beyond the I9
    // field prints as asterisks.
    terms["vdw"] = -12.5;
    terms["elec"] = -300.25;

    apo::EnergyProperties properties;
    properties.totalEnergy = -302.75;
    properties.kineticEnergy = 10.0;
    properties.potentialEnergy = -312.75;
    properties.temperature = 300.123456;
    properties.virialKineticEnergy = -6.0;
    properties.internalVirial = 4.0;
    properties.internalPressure = -123.456789;
    properties.volume = 125000.0;
    properties.surfaceTension = 12.5;

    std::ostringstream output;
    apo::WriteEnergyTable(output, apo::EnergyTableType::DYNA, false,
                          1234567890ULL, 0.02, properties, terms);
    CHECK(output.str() ==
          "DYNA>*********      0.02000   -302.75000     10.00000   -312.75000"
          "    300.12346\n"
          "DYNA PROP>          0.00000      0.00000      0.00000      0.00000"
          "     -6.00000\n"
          "DYNA EXTERN>      -12.50000   -300.25000      0.00000      0.00000"
          "      0.00000\n"
          "DYNA PRESS>         0.00000      4.00000      0.00000   -123.45679"
          " 125000.00000\n"
          "DYNA XTLE>                       0.00000     12.50000      0.00000"
          "      0.00000\n" +
              SEPARATOR + "\n");
  }
}
