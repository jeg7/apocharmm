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

#include <map>
#include <ostream>
#include <string>

namespace apo {

/**
 * @brief Selects one of the CHARMM energy tables that apoCHARMM writes.
 *
 * Each enumerator is the table's CHARMM name (`ENAME`). `ENER` is the table of
 * CHARMM's `ENER` command (`ETYPE` `ENR`), and `DYNA` is the table printed
 * during dynamics (`ETYPE` `DYN`).
 */
enum class EnergyTableType { ENER, DYNA };

/**
 * @brief Holds the energy properties that CHARMM's `PRINTE` writes.
 *
 * Each comment names the CHARMM `EPROP` element (`source/energy/energym.F90`),
 * or the `PRINTE` variable for `energyChange`. Energies are in kilocalories
 * per mole, temperatures in kelvin, pressures in atmospheres, the volume in
 * cubic angstroms, and the surface tension in dyn/cm. A property that
 * apoCHARMM does not compute stays zero.
 */
struct EnergyProperties {
  double totalEnergy = 0.0;            // TOTE
  double kineticEnergy = 0.0;          // TOTKE
  double potentialEnergy = 0.0;        // EPOT
  double temperature = 0.0;            // TEMPS
  double energyChange = 0.0;           // EDIF (the ENER Delta-E column)
  double gradientRms = 0.0;            // GRMS
  double hfcTotalEnergy = 0.0;         // HFCTE
  double hfcKineticEnergy = 0.0;       // HFCKE
  double hfcCorrection = 0.0;          // EHFC
  double virialKineticEnergy = 0.0;    // VIRKE
  double externalVirial = 0.0;         // VIRE
  double internalVirial = 0.0;         // VIRI
  double externalPressure = 0.0;       // PRESSE
  double internalPressure = 0.0;       // PRESSI
  double volume = 0.0;                 // VOLUME
  double crystalTotalEnergy = 0.0;     // XTLTE
  double surfaceTension = 0.0;         // XTLKE (where PSCALE stores it)
  double crystalPotentialEnergy = 0.0; // XTLPE
  double crystalTemperature = 0.0;     // XTLTEM
};

/**
 * @brief Writes one CHARMM energy table.
 *
 * The layout is that of CHARMM's `PRINTE` (`source/energy/printe.F90`) in a
 * build with CHARMM's default keywords, which include CMAP, EPMF, PRIMO,
 * CPATH, and HMCOM. A line of energy terms appears only when one of its terms
 * exceeds `0.49e-6` in magnitude, and CHARMM's `setfmt` picks the fields of
 * each line from its largest magnitude: `F13.5`, one decimal fewer above each
 * power of ten from `1e6`, and `E13.5` above `1e11`. Each table ends with
 * CHARMM's separator line. A `DYNA` table always has its `PROP>` and `PRESS>`
 * lines, and its `XTLE>` line when `crystalPotentialEnergy` or
 * `surfaceTension` is nonzero.
 *
 * Restraint energies take CHARMM's columns: `harm` and `resd` in `CONSTR>`,
 * `hmcm` in the `RESTR>` column that CHARMM fills with `CQRT + PATH + HMCM`,
 * and `geo` in `MMFP>`. CHARMM's `IMAGES>` line never appears, because the
 * nonbonded terms already include the image interactions, as with CHARMM's
 * DOMDEC.
 *
 * @param[out] output Stream that receives the table.
 * @param[in] type Table to write.
 * @param[in] header Whether to write the header lines first. CHARMM writes
 * them for every `ENER` table and at the start of each `DYNA` command.
 * @param[in] cycle Integer column: the evaluation number of an `ENER` table,
 * or the step of a `DYNA` table.
 * @param[in] time Simulation time of a `DYNA` table, in picoseconds. An `ENER`
 * table ignores it.
 * @param[in] properties Energy properties to print.
 * @param[in] terms Energy terms under the keys that
 * `ForceManager::getEnergyComponents()` reports.
 *
 * @throws std::out_of_range If `terms` lacks one of those keys.
 * @warning The stream state is not checked. Callers that need to detect a
 * failed write check the stream afterward.
 */
void WriteEnergyTable(std::ostream &output, const EnergyTableType type,
                      const bool header, const unsigned long long int cycle,
                      const double time, const EnergyProperties &properties,
                      const std::map<std::string, double> &terms);

} // namespace apo
