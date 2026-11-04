// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "Crystal.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace apo {

/**
 * @brief Everything a supported CHARMM restart file records except CHARMM's own
 * random-number fields.
 *
 * Each comment names the CHARMM variable written by `WRIDYN` in
 * `source/dynamc/dynio.F90`. Values are retained even when apoCHARMM does not
 * use them, so a file can be inspected or carried forward without reparsing.
 * Units follow the file: angstroms and AKMA time.
 */
struct RestartSnapshot {
  // Header record (A4,2I6,2X,A4,2X,A4)
  int version = 0;                  // `VERNUM` (52 for the reference CHARMM)
  int dynamicsType = 0;             //`LDYNA` (1 is ordinary leapfrog)
  std::string crystalLabel;         // `XTLTYP` (CUBI, TETR, or ORTH)
  CRYSTAL crystal = CRYSTAL::UNSET; // `XTLTYP` (mapped to enum)
  std::vector<std::string> titles;  // Title lines after `!NTITLE` (as written)

  // !CRYSTAL PARAMETERS
  std::array<double, 6> xtlabc{}; //`XTLABC` (symmetric shape matrix)
  std::array<double, 3> box{};    // Diagonal of `XTLABC` (the box lengths)
  std::array<double, 6> hdot{};   //`HDOT` (compact crystal displacements)
  double pnh = 0.0;               // `PNH` (thermal-piston position)
  double pnhv = 0.0;              //`PNHV` (thermal-piston velocity)
  double pnhf = 0.0;              // `PNHF` (thermal-piston velocity increment)
  std::array<double, 6> uc1a{};   //`UC1A` (cell statistics)
  std::array<double, 6> uc2a{};   //`UC2A` (cell statistics)
  std::array<double, 6> uc1b{};   //`UC1B` (cell statistics)
  std::array<double, 6> uc2b{};   //`UC2B` (cell statistics)
  std::array<double, 4> cellGradients{}; //`GRAD1A, GRAD1B, GRAD2A, GRAD2B`

  // !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL
  int numAtoms = 0;                          // `NATOM`
  unsigned long long int completedSteps = 0; // `NPRIV`
  int requestedSteps = 0;                    // `NSTEP`
  int coordinateSaveFrequency = 0;           // `NSAVC`
  int velocitySaveFrequency = 0;             // `NSAVV`
  int heatingSteps = 0;                      // `JHSTRT`
  int degreesOfFreedom = 0;                  // `NDEGF`
  double seed = 0.0;                         // `SEED`
  int lambdaSaveFrequency = 0;               // `NSAVL`

  // !ENERGIES and STATISTICS
  std::string propertyFlags;       // `QEPROP`
  std::string termFlags;           // `QETERM`
  int averagingSteps = 0;          // `ISTPSA`
  double fitA = 0.0;               // `FITA`
  double fitP = 0.0;               // `FITP`
  double averageTemperature = 0.0; // `AVETEM`
  // Below are per property, energy term, and pressure item
  std::vector<std::array<double, 3>> propertyHistory;  // `EPROP, EPRPP, EPRP2P`
  std::vector<std::array<double, 2>> propertyAverages; // `EPRPA, EPRP2A`
  std::vector<std::array<double, 3>> termHistory;      // `ETERM, ETRMP, ETRM2P`
  std::vector<std::array<double, 2>> termAverages;     // `ETRMA, ETRM2A`
  std::vector<std::array<double, 3>> pressureHistory; // `EPRESS, EPRSP. EPRS2P`
  std::vector<std::array<double, 2>> pressureAverages; // `EPRSA, EPRS2A`

  // Per-atom blocks
  std::vector<std::array<double, 3>>
      coordinates; // `!XOLD, YOLD, ZOLD` (current coordinates)
  std::vector<std::array<double, 3>> velocities; // `!VX, VY, VZ`
  std::vector<std::array<double, 3>>
      deltas; // `!X, Y, Z` (previous leapfrog displacement)

  // apoCHARMM RNG extension, present when the header carries the APO marker.
  // CHARMM's own RNG counts and seeds on the metadata record are validated but
  // not retained.
  std::string
      nativeKind; // Integrator named by the extension (empty for CHARMM files)
  bool hasNativeRng = false; // True when the extension carries Philox state
  std::uint64_t rngSeed =
      0;                // Exact native seed from the extension or legacy field
  std::string rngState; // Serialized native RNG state
};

// JEG261029: Parse and validate the supported CHARMM restart layout without
// modifying a context or integrator. The caller validates and installs the
// returned state.
RestartSnapshot ReadRestartFile(const std::filesystem::path &path,
                                const int expectedAtoms);

} // namespace apo
