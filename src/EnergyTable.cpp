// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "EnergyTable.h"

#include "str_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace {

// PRINTE writes a line of terms only when one of them exceeds this magnitude.
constexpr double ACCU = 0.49e-6;

constexpr const char *SEPARATOR = " ----------       ---------    ---------    "
                                  "---------    ---------    ---------";

/**
 * @brief Returns the decimals that CHARMM's `setfmt` picks for a line.
 *
 * `setfmt` (`source/energy/printe.F90`) takes the largest magnitude of its
 * five arguments, starting from the first. The fields are `F13.5`, one decimal
 * fewer above each power of ten from `1e6` to `1e10`, and `E13.5` above
 * `1e11`, returned here as `-1`.
 */
int SelectDecimals(const std::vector<double> &values) {
  double largest = std::abs(values[0]);
  for (std::size_t i = 1; i < values.size(); i++) {
    if (std::abs(values[i]) > largest)
      largest = std::abs(values[i]);
  }

  int decimals = 5;
  if (largest > 1.0e6)
    decimals = 4;
  if (largest > 1.0e7)
    decimals = 3;
  if (largest > 1.0e8)
    decimals = 2;
  if (largest > 1.0e9)
    decimals = 1;
  if (largest > 1.0e10)
    decimals = 0;
  if (largest > 1.0e11)
    decimals = -1;

  return decimals;
}

/**
 * @brief Writes a label and then each value in the field that `setfmt` chose.
 */
void WriteValues(std::ostream &output, const std::string &label,
                 const std::vector<double> &values, const int decimals) {
  output << label;

  for (const double value : values) {
    if (decimals < 0 && std::isfinite(value)) {
      // Fortran writes E13.5 like D13.5 with an E.
      std::string field = apo::format_fortran_double(value, 13, 5);
      std::replace(field.begin(), field.end(), 'D', 'E');
      output << field;
    } else
      output << apo::format_fortran_fixed(value, 13, std::max(decimals, 0));
  }

  output << '\n';

  return;
}

/**
 * @brief Reports whether a magnitude exceeds `ACCU`, PRINTE's line test.
 */
bool ExceedsAccuracy(const std::vector<double> &values) {
  return std::any_of(values.begin(), values.end(),
                     [](const double value) { return std::abs(value) > ACCU; });
}

} // namespace

void apo::WriteEnergyTable(std::ostream &output, const EnergyTableType type,
                           const bool header,
                           const unsigned long long int cycle,
                           const double time,
                           const EnergyProperties &properties,
                           const std::map<std::string, double> &terms) {
  const bool dynamics = type == EnergyTableType::DYNA;
  const std::string name = dynamics ? "DYNA" : "ENER";

  const double bond = terms.at("bond");
  const double angle = terms.at("angle");
  const double ureyb = terms.at("ureyb");
  const double dihe = terms.at("dihe");
  const double imdihe = terms.at("imdihe");
  const double cmap = terms.at("cmap");
  const double vdw = terms.at("vdw");
  const double elec = terms.at("elec");
  const double user = terms.at("user");
  const double ewks = terms.at("ewks");
  const double ewse = terms.at("ewse");
  const double ewex = terms.at("ewex");
  const double harm = terms.at("harm");
  const double resd = terms.at("resd");
  const double hmcm = terms.at("hmcm");
  const double geo = terms.at("geo");

  // CHARMM's QINTER, QCROSS, QEXTER, QEWALD, QCONST, QRESTR, QMMFP, and QPXTL.
  // The CHARMM terms that apoCHARMM does not have are zero.
  const bool internal = ExceedsAccuracy({bond, angle, ureyb, dihe, imdihe});
  const bool cross = ExceedsAccuracy({cmap});
  const bool external = ExceedsAccuracy({vdw, elec, user});
  const bool ewald = ExceedsAccuracy({ewks, ewse, ewex});
  const bool constraint = ExceedsAccuracy({harm, resd});
  const bool restraint = ExceedsAccuracy({hmcm});
  const bool mmfp = ExceedsAccuracy({geo});
  const bool crystal =
      dynamics && ExceedsAccuracy({properties.crystalPotentialEnergy,
                                   properties.surfaceTension});

  if (header) {
    if (dynamics) {
      output << name
             << " DYN: Step         Time      TOTEner        TOTKe       ENERgy"
                "  TEMPerature\n"
             << name
             << " PROP:             GRMS      HFCTote        HFCKe       EHFCor"
                "        VIRKe\n";
    } else
      output << name << " ENR:  Eval#     ENERgy      Delta-E         GRMS\n";

    if (internal) {
      output << name
             << " INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals"
                "    IMPRopers\n";
    }

    if (cross) {
      output << name
             << " CROSS:           CMAPs        PMF1D        PMF2D        PRIMO"
                "\n";
    }

    if (external) {
      output << name
             << " EXTERN:        VDWaals         ELEC       HBONds          ASP"
                "         USER\n";
    }

    if (ewald) {
      output << name
             << " EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor"
                "       EWUTil\n";
    }

    if (constraint) {
      output << name
             << " CONSTR:       HARMonic    CDIHedral          CIC     "
                "RESDistance       NOE\n";
    }

    if (restraint) {
      output << name
             << " RESTR:        CDROplet    EHARmonic        SHAPe        DMC"
                "            RGY\n";
    }

    if (mmfp) {
      output << name
             << " MMFP:              GEO        MDIP           SSBP        SHEL"
                "       DROFfa\n";
    }

    if (dynamics) {
      output << name
             << " PRESS:            VIRE         VIRI       PRESSE       PRESSI"
                "       VOLUme\n";
    }

    if (crystal) {
      output << name
             << " XTLE:                       XTLTe         SURFtension  XTLPe"
                "        XTLtemp\n";
    }

    output << SEPARATOR << '\n';
  }

  // The integer column is Fortran's I9, which overflows to asterisks.
  std::string cycleField = std::to_string(cycle);
  if (cycleField.size() > 9)
    cycleField = std::string(9, '*');
  cycleField.insert(0, 9 - cycleField.size(), ' ');

  if (dynamics) {
    WriteValues(output, name + '>' + cycleField,
                {time, properties.totalEnergy, properties.kineticEnergy,
                 properties.potentialEnergy, properties.temperature},
                SelectDecimals(
                    {properties.totalEnergy, properties.kineticEnergy,
                     properties.potentialEnergy, properties.temperature, 0.0}));

    const std::vector<double> prop = {
        properties.gradientRms, properties.hfcTotalEnergy,
        properties.hfcKineticEnergy, properties.hfcCorrection,
        properties.virialKineticEnergy};
    WriteValues(output, name + " PROP>    ", prop, SelectDecimals(prop));
  } else {
    WriteValues(
        output, name + '>' + cycleField,
        {properties.potentialEnergy, properties.energyChange,
         properties.gradientRms},
        SelectDecimals({properties.potentialEnergy, properties.gradientRms,
                        properties.energyChange, 0.0, 0.0}));
  }

  if (internal) {
    WriteValues(output, name + " INTERN>  ", {bond, angle, ureyb, dihe, imdihe},
                SelectDecimals({bond, angle, ureyb, dihe, imdihe}));
  }

  if (cross) {
    WriteValues(output, name + " CROSS>   ", {cmap, 0.0, 0.0, 0.0},
                SelectDecimals({cmap, 0.0, 0.0, 0.0, 0.0}));
  }

  if (external) {
    WriteValues(output, name + " EXTERN>  ", {vdw, elec, 0.0, 0.0, user},
                SelectDecimals({vdw, elec, 0.0, 0.0, user}));
  }

  if (ewald) {
    WriteValues(output, name + " EWALD>   ", {ewks, ewse, ewex, 0.0, 0.0},
                SelectDecimals({ewks, ewse, ewex, 0.0, 0.0}));
  }

  if (constraint) {
    WriteValues(output, name + " CONSTR>  ", {harm, 0.0, 0.0, resd, 0.0},
                SelectDecimals({harm, 0.0, 0.0, resd, 0.0}));
  }

  if (restraint) {
    // CHARMM picks this line's fields from CQRT, EHARM, SHAP, DMC, and RGY, so
    // the PATH and HMCM it adds to the first column do not count.
    WriteValues(output, name + " RESTR>   ", {hmcm, 0.0, 0.0, 0.0, 0.0},
                SelectDecimals({0.0, 0.0, 0.0, 0.0, 0.0}));
  }

  if (mmfp) {
    WriteValues(output, name + " MMFP>    ", {geo, 0.0, 0.0, 0.0, 0.0},
                SelectDecimals({geo, 0.0, 0.0, 0.0, 0.0}));
  }

  if (dynamics) {
    const std::vector<double> press = {
        properties.externalVirial, properties.internalVirial,
        properties.externalPressure, properties.internalPressure,
        properties.volume};
    WriteValues(output, name + " PRESS>   ", press, SelectDecimals(press));
  }

  if (crystal) {
    WriteValues(output, name + " XTLE>                 ",
                {properties.crystalTotalEnergy, properties.surfaceTension,
                 properties.crystalPotentialEnergy,
                 properties.crystalTemperature},
                SelectDecimals({properties.crystalTotalEnergy,
                                properties.surfaceTension,
                                properties.crystalPotentialEnergy,
                                properties.crystalTemperature, 0.0}));
  }

  output << SEPARATOR << '\n';

  return;
}
