// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "RestartIO.h"

#include "ApoCharmmError.h"
#include "Crystal.h"
#include "str_utils.h"

#include <array>
#include <cstddef>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>

namespace apo {

RestartSnapshot ReadRestartFile(const std::filesystem::path &path,
                                const int expectedAtoms) {
  APOCHARMM_REQUIRE(expectedAtoms > 0, ApoCharmmErrorCode::NotInitialized,
                    "Restart loading requires a nonempty initialized context");

  std::ifstream input(path);
  APOCHARMM_REQUIRE(input.is_open(), ApoCharmmErrorCode::Runtime,
                    "Could not open restart file: " + path.string());

  const std::string context = "restart file \"" + path.string() + "\"";
  std::size_t lineNumber = 0;
  std::string line;
  RestartSnapshot snapshot;

  // Header record, (A4,2I6,2X,A4,2X,A4): REST, VERNUM, LDYNA, XTLTYP, NSED
  apo::get_line(line, lineNumber, input, "restart header", context);

  // apoCHARMM writers place "APO" at columns 32-34, beyond every field READYN
  // consumes. The marker promises the trailing apoCHARMM RNG extension.
  const bool nativeHeader =
      (line.length() >= 35) && (line.compare(32, 3, "APO") == 0);

  APOCHARMM_REQUIRE(
      apo::get_fixed_width_field(line, 0, 4, "record type", context) == "REST",
      ApoCharmmErrorCode::InvalidArgument,
      "Expected REST header in " + context);

  snapshot.version = apo::parse_fixed_width_int(line, 4, 6, "VERSION", context);
  APOCHARMM_REQUIRE(snapshot.version >= 24, ApoCharmmErrorCode::NotImplemented,
                    "Supported restart versions are 24 or greater in " +
                        context);

  snapshot.dynamicsType =
      apo::parse_fixed_width_int(line, 10, 6, "LDYNA", context);
  APOCHARMM_REQUIRE(
      snapshot.dynamicsType == 1, ApoCharmmErrorCode::NotImplemented,
      "Only ordinary leapfrog (LDYNA=1) restarts are supported in " + context);

  snapshot.crystalLabel =
      apo::get_fixed_width_field(line, 18, 4, "CRYSTAL", context);
  if (snapshot.crystalLabel == "CUBI")
    snapshot.crystal = CRYSTAL::CUBIC;
  else if (snapshot.crystalLabel == "TETR")
    snapshot.crystal = CRYSTAL::TETRAGONAL;
  else if (snapshot.crystalLabel == "ORTH")
    snapshot.crystal = CRYSTAL::ORTHORHOMBIC;
  else
    APOCHARMM_THROW(ApoCharmmErrorCode::NotImplemented,
                    "Only CUBI, TETR, and ORTH restarts are supported in " +
                        context);

  APOCHARMM_REQUIRE(
      apo::get_fixed_width_field(line, 24, 4, "NSEO", context) == "    ",
      ApoCharmmErrorCode::NotImplemented,
      "CHARMM NOSE restart arrays are not supported in " + context);

  // Beyond the five standard fields only blanks (plus the apoCHARMM marker at
  // columns 32-34) are accepted; a nonblank DHDGB field is unsupported
  std::string headerExtension = line.substr(28);
  if (nativeHeader)
    headerExtension.replace(4, 3, "   ");
  APOCHARMM_REQUIRE(
      apo::trim(headerExtension).empty(), ApoCharmmErrorCode::NotImplemented,
      "DHDGB or another restart header extension is unsupported in " + context);

  // Title records, (/I8,A) then one line per title. Titles are free text in
  // both programs; the header marker, not a title prefix, identifies apoCHARMM
  // output. They are retained as written.
  apo::get_nonblank_line(line, lineNumber, input, "NTITLE", context);
  const int titleCount =
      apo::parse_fixed_width_int(line, 0, 8, "NTITLE", context);
  APOCHARMM_REQUIRE(titleCount >= 0, ApoCharmmErrorCode::InvalidArgument,
                    "Invalid NTITLE in " + context);

  snapshot.titles.reserve(titleCount);
  for (int i = 0; i < titleCount; i++) {
    apo::get_line(line, lineNumber, input, "title", context);
    snapshot.titles.push_back(line);
  }

  // !CRYSTAL PARAMETERS, every record (3D22.15): XTLABC, HDOT, PNH/PNHV/PNHF,
  // UC1A, UC2A, UC1B, UC2B, then GRAD1A, GRAD1B, GRAD2A, GRAD2B
  apo::get_expected_line(lineNumber, input, " !CRYSTAL PARAMETERS", context);
  apo::read_fixed_width_fortran_values(
      lineNumber, input, snapshot.xtlabc.data(), snapshot.xtlabc.size(), 3, 22,
      "XTLABC", context);
  APOCHARMM_REQUIRE(
      (snapshot.xtlabc[0] > 0.0) && (snapshot.xtlabc[1] == 0.0) &&
          (snapshot.xtlabc[2] > 0.0) && (snapshot.xtlabc[3] == 0.0) &&
          (snapshot.xtlabc[4] == 0.0) && (snapshot.xtlabc[5] > 0.0),
      ApoCharmmErrorCode::InvalidArgument,
      "Restart cell must have positive lengths and zero off-diagonal "
      "components in " +
          context);

  snapshot.box = {snapshot.xtlabc[0], snapshot.xtlabc[2], snapshot.xtlabc[5]};
  APOCHARMM_REQUIRE((snapshot.crystal != CRYSTAL::CUBIC ||
                     (snapshot.box[0] == snapshot.box[1] &&
                      snapshot.box[0] == snapshot.box[2])) &&
                        (snapshot.crystal != CRYSTAL::TETRAGONAL ||
                         snapshot.box[0] == snapshot.box[1]),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Restart crystal label does not match its cell in " +
                        context);

  apo::read_fixed_width_fortran_values(lineNumber, input, snapshot.hdot.data(),
                                       snapshot.hdot.size(), 3, 22, "HDOT",
                                       context);

  std::array<double, 3> thermostat{};
  apo::read_fixed_width_fortran_values(lineNumber, input, thermostat.data(),
                                       thermostat.size(), 3, 22,
                                       "PNH/PNHV/PNHF", context);
  snapshot.pnh = thermostat[0];
  snapshot.pnhv = thermostat[1];
  snapshot.pnhf = thermostat[2];

  apo::read_fixed_width_fortran_values(lineNumber, input, snapshot.uc1a.data(),
                                       snapshot.uc1a.size(), 3, 22, "UC1A",
                                       context);
  apo::read_fixed_width_fortran_values(lineNumber, input, snapshot.uc2a.data(),
                                       snapshot.uc2a.size(), 3, 22, "UC2A",
                                       context);
  apo::read_fixed_width_fortran_values(lineNumber, input, snapshot.uc1b.data(),
                                       snapshot.uc1b.size(), 3, 22, "UC1B",
                                       context);
  apo::read_fixed_width_fortran_values(lineNumber, input, snapshot.uc2b.data(),
                                       snapshot.uc2b.size(), 3, 22, "UC2B",
                                       context);
  apo::read_fixed_width_fortran_values(lineNumber, input,
                                       snapshot.cellGradients.data(),
                                       snapshot.cellGradients.size(), 3, 22,
                                       "GRAD1A,GRAD1B,GRAD2A,GRAD2B", context);

  // Metadata record, (7I12,D22.15,I12,2I22,<n>I22): NATOM, NPRIV, NSTEP, NSAVC,
  // NSAVV, JHSTRT, NDEGF, SEED, NSAVL, then CHARMM's RNG node and seed counts
  // followed by its seed words
  apo::get_expected_line(
      lineNumber, input,
      " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL", context);
  apo::get_line(line, lineNumber, input, "integer and RNG record", context);

  snapshot.numAtoms = apo::parse_fixed_width_int(line, 0, 12, "NATOM", context);
  APOCHARMM_REQUIRE(snapshot.numAtoms == expectedAtoms,
                    ApoCharmmErrorCode::InvalidArgument,
                    "NATOM mismatch in " + context);

  snapshot.completedSteps =
      apo::parse_fixed_width_ull(line, 12, 12, "NPRIV", context);
  snapshot.requestedSteps =
      apo::parse_fixed_width_int(line, 24, 12, "NSTEP", context);
  snapshot.coordinateSaveFrequency =
      apo::parse_fixed_width_int(line, 36, 12, "NSAVC", context);
  snapshot.velocitySaveFrequency =
      apo::parse_fixed_width_int(line, 48, 12, "NSAVV", context);
  snapshot.heatingSteps =
      apo::parse_fixed_width_int(line, 60, 12, "JHSTRT", context);
  snapshot.degreesOfFreedom =
      apo::parse_fixed_width_int(line, 72, 12, "NDEGF", context);
  APOCHARMM_REQUIRE(
      snapshot.requestedSteps >= 0 && snapshot.coordinateSaveFrequency >= 0 &&
          snapshot.velocitySaveFrequency >= 0 && snapshot.heatingSteps >= 0 &&
          snapshot.degreesOfFreedom > 0,
      ApoCharmmErrorCode::InvalidArgument,
      "Invalid dynamics counters in " + context);

  snapshot.seed =
      apo::parse_fixed_width_fortran_double(line, 84, 22, "SEED", context);
  snapshot.lambdaSaveFrequency =
      apo::parse_fixed_width_int(line, 106, 12, "NSAVL", context);
  APOCHARMM_REQUIRE(
      snapshot.lambdaSaveFrequency == 0, ApoCharmmErrorCode::NotImplemented,
      "Lambda-dynamics restart state is unsupported in " + context);

  const int nodeCount =
      apo::parse_fixed_width_int(line, 118, 22, "RNG node count", context);
  const int seedsPerNode =
      apo::parse_fixed_width_int(line, 140, 22, "RNG seed count", context);
  APOCHARMM_REQUIRE(nodeCount >= 0 && seedsPerNode >= 0 &&
                        ((nodeCount == 0) == (seedsPerNode == 0)),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Invalid CHARMM RNG counts in " + context);

  const std::size_t availableSeedFields = (line.size() - 162) / 22;
  APOCHARMM_REQUIRE(
      seedsPerNode == 0 ||
          static_cast<std::size_t>(nodeCount) <=
              availableSeedFields / static_cast<std::size_t>(seedsPerNode),
      ApoCharmmErrorCode::Runtime, "Truncated CHARMM RNG fields in " + context);

  const std::size_t totalSeedFields = static_cast<std::size_t>(nodeCount) *
                                      static_cast<std::size_t>(seedsPerNode);
  for (std::size_t i = 0; i < totalSeedFields; i++) {
    const std::string field = apo::trim(apo::get_fixed_width_field(
        line, 162 + i * 22, 22, "CHARMM RNG state", context));
    const std::size_t firstDigit =
        !field.empty() && (field.front() == '+' || field.front() == '-') ? 1
                                                                         : 0;
    APOCHARMM_REQUIRE(field.size() > firstDigit &&
                          field.find_first_not_of("0123456789", firstDigit) ==
                              std::string::npos,
                      ApoCharmmErrorCode::Runtime,
                      "Malformed CHARMM RNG state in " + context);
  }
  APOCHARMM_REQUIRE(
      apo::trim(std::string_view(line).substr(162 + totalSeedFields * 22))
          .empty(),
      ApoCharmmErrorCode::Runtime,
      "Unexpected data after CHARMM RNG fields in " + context);
  // CHARMM's RNG words belong to a different generator and are not retained

  // !ENERGIES and STATISTICS. QEPROP and QETERM are (256L1) records whose
  // lengths are LENENP and LENENT; only CHARMM's default all-T selection is
  // supported
  apo::get_expected_line(lineNumber, input, " !ENERGIES and STATISTICS",
                         context);
  apo::get_line(line, lineNumber, input, "QEPROP", context);
  snapshot.propertyFlags = apo::trim(line);
  APOCHARMM_REQUIRE(snapshot.propertyFlags.size() >= 50 &&
                        snapshot.propertyFlags.size() <= 96 &&
                        snapshot.propertyFlags.find_first_not_of('T') ==
                            std::string::npos,
                    ApoCharmmErrorCode::NotImplemented,
                    "Only default all-T CHARMM energy-selection flags are "
                    "supported for QEPROP in " +
                        context);

  apo::get_line(line, lineNumber, input, "QETERM", context);
  snapshot.termFlags = apo::trim(line);
  APOCHARMM_REQUIRE(
      snapshot.termFlags.size() >= 50 && snapshot.termFlags.size() <= 192 &&
          snapshot.termFlags.find_first_not_of('T') == std::string::npos,
      ApoCharmmErrorCode::NotImplemented,
      "Only default all-T CHARMM energy-selection flags are supported for "
      "QETERM in " +
          context);

  const std::size_t propertyCount = snapshot.propertyFlags.size();
  const std::size_t termCount = snapshot.termFlags.size();

  // (I8,3D22.15): ISTPSA, FITA, FITP, AVETEM
  apo::get_line(line, lineNumber, input, "ISTPSA,FITA,FITP,AVETEM", context);
  snapshot.averagingSteps =
      apo::parse_fixed_width_int(line, 0, 8, "ISTPSA", context);
  APOCHARMM_REQUIRE(snapshot.averagingSteps >= 0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Negative ISTPSA in " + context);

  std::array<double, 3> fit{};
  apo::parse_fixed_width_fortran_doubles(fit.data(), fit.size(), line, 8, 22,
                                         "FITA,FITP,AVETEM", context);
  snapshot.fitA = fit[0];
  snapshot.fitP = fit[1];
  snapshot.averageTemperature = fit[2];

  // Per group, WRIDYN writes every (3D22.15) history record and then every
  // (2D22.15) long-term record. Both are retained although apoCHARMM resets
  // them; LENENV is 50.
  apo::read_fixed_width_fortran_rows(lineNumber, input,
                                     snapshot.propertyHistory, propertyCount,
                                     22, "EPROP,EPRPP,EPRP2P", context);
  apo::read_fixed_width_fortran_rows(lineNumber, input,
                                     snapshot.propertyAverages, propertyCount,
                                     22, "EPRPA,EPRP2A", context);
  apo::read_fixed_width_fortran_rows(lineNumber, input, snapshot.termHistory,
                                     termCount, 22, "ETERM,ETRMP,ETRM2P",
                                     context);
  apo::read_fixed_width_fortran_rows(lineNumber, input, snapshot.termAverages,
                                     termCount, 22, "ETRMA,ETRM2A", context);
  apo::read_fixed_width_fortran_rows(lineNumber, input,
                                     snapshot.pressureHistory, 50, 22,
                                     "EPRESS,EPRSP,EPRS2P", context);
  apo::read_fixed_width_fortran_rows(lineNumber, input,
                                     snapshot.pressureAverages, 50, 22,
                                     "EPRSA,EPRS2A", context);

  // Per-atom records, (3D22.15) each. CHARMM's call site passes the current
  // coordinates as XOLD and the previous leapfrog displacement as X.
  const std::size_t numAtoms = static_cast<std::size_t>(snapshot.numAtoms);

  apo::get_expected_line(lineNumber, input, " !XOLD, YOLD, ZOLD", context);
  apo::read_fixed_width_fortran_rows(lineNumber, input, snapshot.coordinates,
                                     numAtoms, 22, "XOLD,YOLD,ZOLD", context);

  apo::get_expected_line(lineNumber, input, " !VX, VY, VZ", context);
  apo::read_fixed_width_fortran_rows(lineNumber, input, snapshot.velocities,
                                     numAtoms, 22, "VX,VY,VZ", context);

  apo::get_expected_line(lineNumber, input, " !X, Y, Z", context);
  apo::read_fixed_width_fortran_rows(lineNumber, input, snapshot.deltas,
                                     numAtoms, 22, "X,Y,Z", context);

  // Trailing apoCHARMM RNG extension. READYN stops before this point, so CHARMM
  // never sees it; apoCHARMM requires it whenever the header carries the
  // marker.
  bool seenNativeFooter = false;
  while (apo::try_get_line(line, lineNumber, input)) {
    if (apo::trim(line).empty())
      continue;

    APOCHARMM_REQUIRE(!seenNativeFooter && line == " !APOCHARMM RNG STATE",
                      ApoCharmmErrorCode::NotImplemented,
                      "Unsupported or duplicate trailing restart state in " +
                          context + ": " + line);

    seenNativeFooter = true;
    apo::get_line(line, lineNumber, input, "APOCHARMM RNG descriptor", context);

    std::istringstream descriptor(line);
    descriptor.imbue(std::locale::classic());

    int footerVersion = 0;
    std::string algorithm;
    descriptor >> footerVersion >> snapshot.nativeKind >> algorithm;
    APOCHARMM_REQUIRE(!descriptor.fail(), ApoCharmmErrorCode::Runtime,
                      "Malformed APOCHARMM RNG descriptor in " + context);

    descriptor >> std::ws;
    APOCHARMM_REQUIRE(
        descriptor.eof() && footerVersion == 1 &&
            ((snapshot.nativeKind == "NOSE_HOOVER" && algorithm == "NONE") ||
             ((snapshot.nativeKind == "LANGEVIN_PISTON" ||
               snapshot.nativeKind == "LANGEVIN_THERMOSTAT") &&
              algorithm == "PHILOX4_32_10")),
        ApoCharmmErrorCode::NotImplemented,
        "Unsupported APOCHARMM RNG descriptor in " + context);

    apo::get_line(line, lineNumber, input, "APOCHARMM RNG seed", context);
    snapshot.rngSeed = apo::parse_ull(line, "native RNG seed", context);

    apo::get_line(line, lineNumber, input, "APOCHARMM RNG state", context);
    snapshot.rngState = apo::trim(line);
    snapshot.hasNativeRng = algorithm != "NONE";
    APOCHARMM_REQUIRE(
        snapshot.hasNativeRng
            ? !snapshot.rngState.empty()
            : (snapshot.rngSeed == 0 && snapshot.rngState.empty()),
        ApoCharmmErrorCode::Runtime,
        "Missing or inconsistent APOCHARMM RNG state in " + context);
  }

  APOCHARMM_REQUIRE(!input.bad(), ApoCharmmErrorCode::Runtime,
                    "Failed while reading " + context);
  APOCHARMM_REQUIRE(
      !nativeHeader || seenNativeFooter, ApoCharmmErrorCode::Runtime,
      "APOCHARMM restart is missing its RNG footer in " + context);
  APOCHARMM_REQUIRE(
      nativeHeader || !seenNativeFooter, ApoCharmmErrorCode::Runtime,
      "APOCHARMM RNG footer requires the APO header marker in " + context);

  // A foreign file with JHSTRT=0 is accepted: CHARMM resumes it through its
  // two-step start (IGVOPT=2), and the integrators regenerate the previous
  // displacement instead of using the !X, Y, Z block.

  return snapshot;
}

} // namespace apo
