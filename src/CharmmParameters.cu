// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmParameters.h"
#include "CmapSpline.h"

#include "ApoCharmmError.h"
#include "str_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

std::string GetPrmValueContext(const std::string_view recordType,
                               const std::string &line,
                               const std::string &fileName,
                               const std::size_t lineNumber) {
  return std::string(recordType) + " parameter record in file \"" + fileName +
         "\" at line " + std::to_string(lineNumber) + ": " + line;
}

std::string CleanPrmLine(const std::string &rawLine) {
  std::string line = rawLine;

  const std::size_t commentPosition = line.find('!');
  if (commentPosition != std::string::npos)
    line.erase(commentPosition);

  std::replace(line.begin(), line.end(), '\t', ' ');
  apo::trim_ip(line);
  apo::to_upper_ip(line);

  return line;
}

} // namespace

CharmmParameters::CharmmParameters(void)
    : m_BondParams(), m_UreybParams(), m_AngleParams(), m_DihedralParams(),
      m_ImproperParams(), m_CmapParams(), m_NbfixParams(), m_VdwParams(), 
      m_Vdw14Params(), m_PrmFilePaths() {}

CharmmParameters::CharmmParameters(const std::filesystem::path &filePath)
    : CharmmParameters() {
  m_PrmFilePaths.push_back(filePath);
  this->readCharmmParameterFile(filePath);
}

CharmmParameters::CharmmParameters(
    const std::vector<std::filesystem::path> &filePaths)
    : CharmmParameters() {
  APOCHARMM_REQUIRE(!filePaths.empty(), ApoCharmmErrorCode::InvalidArgument,
                    "At least one CHARMM parameter file is required");

  for (const std::filesystem::path &filePath : filePaths) {
    m_PrmFilePaths.push_back(filePath);
    this->readCharmmParameterFile(filePath);
  }
}

const std::map<BondKey, BondValues> &
CharmmParameters::getBondParams(void) const {
  return m_BondParams;
}

const std::map<AngleKey, BondValues> &
CharmmParameters::getUreybParams(void) const {
  return m_UreybParams;
}

const std::map<AngleKey, AngleValues> &
CharmmParameters::getAngleParams(void) const {
  return m_AngleParams;
}

const std::map<DihedralKey, std::vector<DihedralValues>> &
CharmmParameters::getDihedralParams(void) const {
  return m_DihedralParams;
}

const std::map<DihedralKey, ImDihedralValues> &
CharmmParameters::getImproperParams(void) const {
  return m_ImproperParams;
}

const std::map<CmapKey, CmapValues> &
CharmmParameters::getCmapParams(void) const {
  return m_CmapParams;
}

const std::map<std::string, VdwParameters> &
CharmmParameters::getVdwParams(void) const {
  return m_VdwParams;
}

const std::map<std::string, VdwParameters> &
CharmmParameters::getVdw14Params(void) const {
  return m_Vdw14Params;
}

const std::vector<std::filesystem::path> &
CharmmParameters::getPrmFilePaths(void) const {
  return m_PrmFilePaths;
}

BondedParamsAndLists CharmmParameters::getBondedParamsAndLists(
    const std::shared_ptr<CharmmPSF> &psf) const {
  APOCHARMM_REQUIRE(psf != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "CharmmPSF must not be null");

  std::vector<int> paramsSize;
  std::vector<std::vector<float>> paramsVal;

  std::vector<int> listsSize;
  std::vector<std::vector<int>> listVal;

  // JEG260817: The four outputs are concatenated in the fixed order expected by
  // CudaBondedForce: bond, Urey-Bradley, angle, proper dihedral, improper
  // dihedral, and CMAP.

  const std::vector<std::string> &atomTypes = psf->getAtomTypes();
  const std::vector<std::string> &atomNames = psf->getAtomNames();
  const std::vector<Bond> &bonds = psf->getBonds();
  const std::vector<Angle> &angles = psf->getAngles();
  const std::vector<Dihedral> &dihedrals = psf->getDihedrals();
  const std::vector<Dihedral> &impropers = psf->getImpropers();
  const std::vector<CrossTerm> &cmaps = psf->getCrossTerms();

  std::vector<BondKey> bondKeysPresent;
  std::vector<AngleKey> ureybKeysPresent;
  std::vector<AngleKey> angleKeysPresent;
  std::vector<DihedralKey> dihedralKeysPresent;
  std::vector<DihedralKey> improperKeysPresent;
  std::vector<CmapKey> cmapKeysPresent;

  for (int bond = 0; bond < psf->getNumBonds(); bond++) {
    std::string atom1 = atomTypes[bonds[bond].iatom];
    std::string atom2 = atomTypes[bonds[bond].jatom];
    if (atom1 > atom2)
      std::swap(atom1, atom2);
    auto key = BondKey(atom1, atom2);

    if (m_BondParams.count(key)) {
      auto findResult =
          std::find(bondKeysPresent.begin(), bondKeysPresent.end(), key);
      if (findResult == std::end(bondKeysPresent)) {
        bondKeysPresent.push_back(key);
        const BondValues &value = m_BondParams.at(key);
        paramsVal.push_back(
            {static_cast<float>(value.b0), static_cast<float>(value.kb)});
      }
      findResult =
          std::find(bondKeysPresent.begin(), bondKeysPresent.end(), key);
      int bondType = findResult - std::begin(bondKeysPresent);
      listVal.push_back({bonds[bond].iatom, bonds[bond].jatom, bondType, 13});
    } else {
      APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                      "Bond parameters were not found for bond " +
                          std::to_string(bond) + " with atom types \"" + atom1 +
                          "\" and \"" + atom2 + "\"");
    }
  }
  paramsSize.push_back(bondKeysPresent.size());
  listsSize.push_back(listVal.size());

  // Most angles do not have a Urey-Bradley contribution, so only register terms
  // whose force constant is nonzero.
  for (int angle = 0; angle < psf->getNumAngles(); angle++) {
    std::string atom1 = atomTypes[angles[angle].iatom];
    std::string atom2 = atomTypes[angles[angle].jatom];
    std::string atom3 = atomTypes[angles[angle].katom];
    if (atom1 > atom3)
      std::swap(atom1, atom3);
    auto key = AngleKey(atom1, atom2, atom3);

    const auto parameter = m_UreybParams.find(key);
    if (parameter == m_UreybParams.end())
      continue;

    const BondValues &value = m_UreybParams.at(key);
    if (std::abs(value.kb) <= 0.01)
      continue;

    auto findResult =
        std::find(ureybKeysPresent.begin(), ureybKeysPresent.end(), key);
    if (findResult == ureybKeysPresent.end()) {
      ureybKeysPresent.push_back(key);
      paramsVal.push_back(
          {static_cast<float>(value.b0), static_cast<float>(value.kb)});
    }
    findResult =
        std::find(ureybKeysPresent.begin(), ureybKeysPresent.end(), key);
    int ureybType = findResult - ureybKeysPresent.begin();
    listVal.push_back(
        {angles[angle].iatom, angles[angle].katom, ureybType, 13});
  }
  paramsSize.push_back(ureybKeysPresent.size());
  listsSize.push_back(listVal.size() - listsSize[0]);

  for (int angle = 0; angle < psf->getNumAngles(); angle++) {
    std::string atom1 = atomTypes[angles[angle].iatom];
    std::string atom2 = atomTypes[angles[angle].jatom];
    std::string atom3 = atomTypes[angles[angle].katom];
    if (atom1 > atom3)
      std::swap(atom1, atom3);
    auto key = AngleKey(atom1, atom2, atom3);

    if (m_AngleParams.count(key)) {
      auto findResult =
          std::find(angleKeysPresent.begin(), angleKeysPresent.end(), key);
      if (findResult == angleKeysPresent.end()) {
        angleKeysPresent.push_back(key);
        const AngleValues &value = m_AngleParams.at(key);
        paramsVal.push_back({static_cast<float>(value.theta0),
                             static_cast<float>(value.kTheta)});
      }
      findResult =
          std::find(angleKeysPresent.begin(), angleKeysPresent.end(), key);
      int angleType = findResult - angleKeysPresent.begin();
      listVal.push_back({angles[angle].iatom, angles[angle].jatom,
                         angles[angle].katom, angleType, 13, 13});

    } else {
      APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                      "Angle parameters were not found for angle " +
                          std::to_string(angle) + " with atom types \"" +
                          atom1 + "\", \"" + atom2 + "\", and \"" + atom3 +
                          "\"");
    }
  }
  paramsSize.push_back(angleKeysPresent.size());
  listsSize.push_back(listVal.size() - listsSize[0] - listsSize[1]);

  int dihedralParamsPresent = 0;
  int startParamDihedral = paramsVal.size();
  std::map<DihedralKey, int> indexOfKeyInParamsVal;
  const double pi_180 = std::acos(-1) / 180.0;
  for (int dihedral = 0; dihedral < psf->getNumDihedrals(); dihedral++) {
    std::string atom1 = atomTypes[dihedrals[dihedral].iatom];
    std::string atom2 = atomTypes[dihedrals[dihedral].jatom];
    std::string atom3 = atomTypes[dihedrals[dihedral].katom];
    std::string atom4 = atomTypes[dihedrals[dihedral].latom];

    if (atom1 > atom4) {
      std::swap(atom1, atom4);
      std::swap(atom2, atom3);
    }
    if ((atom1 == atom4) && (atom2 > atom3))
      std::swap(atom2, atom3);
    auto key = DihedralKey(atom1, atom2, atom3, atom4);

    if (m_DihedralParams.count(key)) {
      auto findResult = std::find(dihedralKeysPresent.begin(),
                                  dihedralKeysPresent.end(), key);
      if (findResult == dihedralKeysPresent.end()) {
        const std::vector<DihedralValues> &values = m_DihedralParams.at(key);

        for (std::size_t i = 0; i < values.size(); i++) {
          if (i > 0)
            paramsVal[paramsVal.size() - 1][0] *= -1;
          else {
            indexOfKeyInParamsVal[key] =
                static_cast<int>(paramsVal.size() - startParamDihedral);
          }

          const DihedralValues &value = values[i];
          const double cpsin = std::sin(value.delta * pi_180);
          const double cpcos = std::cos(value.delta * pi_180);
          paramsVal.push_back(
              {static_cast<float>(value.n), static_cast<float>(value.kChi),
               static_cast<float>(cpsin), static_cast<float>(cpcos)});
          dihedralParamsPresent++;
        }
        dihedralKeysPresent.push_back(key);
      }
      int dihedralType = indexOfKeyInParamsVal[key];
      listVal.push_back({dihedrals[dihedral].iatom, dihedrals[dihedral].jatom,
                         dihedrals[dihedral].katom, dihedrals[dihedral].latom,
                         dihedralType, 13, 13, 13});
    } else {
      if (atom2 > atom3)
        std::swap(atom2, atom3);
      key = DihedralKey("X", atom2, atom3, "X");

      if (m_DihedralParams.count(key)) {
        auto findResult = std::find(dihedralKeysPresent.begin(),
                                    dihedralKeysPresent.end(), key);
        if (findResult == dihedralKeysPresent.end()) {
          const std::vector<DihedralValues> &values = m_DihedralParams.at(key);

          for (std::size_t i = 0; i < values.size(); i++) {
            if (i > 0)
              paramsVal[paramsVal.size() - 1][0] *= -1;
            else {
              indexOfKeyInParamsVal[key] =
                  static_cast<int>(paramsVal.size() - startParamDihedral);
            }

            const DihedralValues &value = values[i];
            const double cpsin = std::sin(value.delta * pi_180);
            const double cpcos = std::cos(value.delta * pi_180);
            paramsVal.push_back(
                {static_cast<float>(value.n), static_cast<float>(value.kChi),
                 static_cast<float>(cpsin), static_cast<float>(cpcos)});
            dihedralParamsPresent++;
          }
          dihedralKeysPresent.push_back(key);
        }
        int dihedralType = indexOfKeyInParamsVal[key];
        listVal.push_back({dihedrals[dihedral].iatom, dihedrals[dihedral].jatom,
                           dihedrals[dihedral].katom, dihedrals[dihedral].latom,
                           dihedralType, 13, 13, 13});
      } else {
        APOCHARMM_THROW(
            ApoCharmmErrorCode::Runtime,
            "Dihedral parameters were not found for dihedral " +
                std::to_string(dihedral) +
                "; atom types: " + atomTypes[dihedrals[dihedral].iatom] + " " +
                atomTypes[dihedrals[dihedral].jatom] + " " +
                atomTypes[dihedrals[dihedral].katom] + " " +
                atomTypes[dihedrals[dihedral].latom] +
                "; atom names: " + atomNames[dihedrals[dihedral].iatom] + " " +
                atomNames[dihedrals[dihedral].jatom] + " " +
                atomNames[dihedrals[dihedral].katom] + " " +
                atomNames[dihedrals[dihedral].latom]);
      }
    }
  }
  paramsSize.push_back(dihedralParamsPresent);
  listsSize.push_back(listVal.size() - listsSize[0] - listsSize[1] -
                      listsSize[2]);

  for (int improper = 0; improper < psf->getNumImpropers(); improper++) {
    std::string atom1 = atomTypes[impropers[improper].iatom];
    std::string atom2 = atomTypes[impropers[improper].jatom];
    std::string atom3 = atomTypes[impropers[improper].katom];
    std::string atom4 = atomTypes[impropers[improper].latom];
    if (atom1 > atom4) {
      std::swap(atom1, atom4);
      std::swap(atom2, atom3);
    }
    if ((atom1 == atom4) && (atom2 > atom3))
      std::swap(atom2, atom3);
    auto key = DihedralKey(atom1, atom2, atom3, atom4);

    if (m_ImproperParams.count(key)) {
      auto findResult = std::find(improperKeysPresent.begin(),
                                  improperKeysPresent.end(), key);
      if (findResult == improperKeysPresent.end()) {
        improperKeysPresent.push_back(key);
        const ImDihedralValues &value = m_ImproperParams.at(key);
        paramsVal.push_back({static_cast<float>(value.psi0),
                             static_cast<float>(value.kPsi), 0.0f, 1.0f});
      }
      findResult = std::find(improperKeysPresent.begin(),
                             improperKeysPresent.end(), key);
      int improperType = findResult - improperKeysPresent.begin();
      listVal.push_back({impropers[improper].iatom, impropers[improper].jatom,
                         impropers[improper].katom, impropers[improper].latom,
                         improperType, 13, 13, 13});
    } else {
      key = DihedralKey(atom1, "X", "X", atom4);

      if (m_ImproperParams.count(key)) {
        auto findResult = std::find(improperKeysPresent.begin(),
                                    improperKeysPresent.end(), key);
        if (findResult == improperKeysPresent.end()) {
          improperKeysPresent.push_back(key);
          const ImDihedralValues &value = m_ImproperParams.at(key);
          paramsVal.push_back({static_cast<float>(value.psi0),
                               static_cast<float>(value.kPsi), 0.0f, 1.0f});
        }
        findResult = std::find(improperKeysPresent.begin(),
                               improperKeysPresent.end(), key);
        int improperType = findResult - improperKeysPresent.begin();
        listVal.push_back({impropers[improper].iatom, impropers[improper].jatom,
                           impropers[improper].katom, impropers[improper].latom,
                           improperType, 13, 13, 13});
      } else {
        APOCHARMM_THROW(
            ApoCharmmErrorCode::Runtime,
            "Improper parameters were not found for improper " +
                std::to_string(improper) +
                "; atom types: " + atomTypes[impropers[improper].iatom] + " " +
                atomTypes[impropers[improper].jatom] + " " +
                atomTypes[impropers[improper].katom] + " " +
                atomTypes[impropers[improper].latom] +
                "; atom names: " + atomNames[impropers[improper].iatom] + " " +
                atomNames[impropers[improper].jatom] + " " +
                atomNames[impropers[improper].katom] + " " +
                atomNames[impropers[improper].latom]);
      }
    }
  }
  paramsSize.push_back(improperKeysPresent.size());
  listsSize.push_back(listVal.size() - listsSize[0] - listsSize[1] -
                      listsSize[2] - listsSize[3]);

  for (int cmap = 0; cmap < psf->getNumCrossTerms(); cmap++) {
    const CrossTerm &crossTerm = cmaps[cmap];

    std::string atom1 = atomTypes[crossTerm.iatom1];
    std::string atom2 = atomTypes[crossTerm.jatom1];
    std::string atom3 = atomTypes[crossTerm.katom1];
    std::string atom4 = atomTypes[crossTerm.latom1];

    const DihedralKey dihe1(atom1, atom2, atom3, atom4);

    atom1 = atomTypes[crossTerm.iatom2];
    atom2 = atomTypes[crossTerm.jatom2];
    atom3 = atomTypes[crossTerm.katom2];
    atom4 = atomTypes[crossTerm.latom2];

    const DihedralKey dihe2(atom1, atom2, atom3, atom4);
    const CmapKey key(dihe1, dihe2);

    const auto parameter = m_CmapParams.find(key);

    APOCHARMM_REQUIRE(
        parameter != m_CmapParams.end(), ApoCharmmErrorCode::Runtime,
        "CMAP parameters were not found for cross term " + std::to_string(cmap));

    auto findResult =
        std::find(cmapKeysPresent.begin(), cmapKeysPresent.end(), key);

    if (findResult == cmapKeysPresent.end()) {
      cmapKeysPresent.push_back(key);

      const CmapValues &value = parameter->second;

      APOCHARMM_REQUIRE(
          value.values.size() == CmapValues::numValues &&
          value.coeff.size() == CmapValues::numCoefficients,
          ApoCharmmErrorCode::Runtime,
          "CMAP coefficient data has an invalid size");

      for (const double coefficient : value.coeff)
        paramsVal.push_back({static_cast<float>(coefficient)});
    }

    const int cmapType = static_cast<int>(std::find(cmapKeysPresent.begin(),
                         cmapKeysPresent.end(), key) - cmapKeysPresent.begin());

    listVal.push_back({crossTerm.iatom1, crossTerm.jatom1,
                       crossTerm.katom1, crossTerm.latom1,
                       crossTerm.iatom2, crossTerm.jatom2,
                       crossTerm.katom2, crossTerm.latom2,
                       cmapType, 13, 13, 13, 13, 13, 13
    });
  }

  paramsSize.push_back(
      static_cast<int>(cmapKeysPresent.size() * CmapValues::numCoefficients));

  listsSize.push_back(
      listVal.size() - listsSize[0] - listsSize[1] -
      listsSize[2] - listsSize[3] - listsSize[4]);

  return BondedParamsAndLists(paramsSize, paramsVal, listsSize, listVal);
}

VdwParamsAndTypes
CharmmParameters::getVdwParamsAndTypes(std::shared_ptr<CharmmPSF> &psf) const {
  APOCHARMM_REQUIRE(psf != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "CharmmPSF must not be null");

  std::vector<float> psfVdwParams, psfVdw14Params;
  std::vector<int> psfVdwTypes, psfVdw14Types;
  std::set<std::string> vdwAtomTypesMap, vdw14AtomTypesMap;

  for (const std::string &atomType : psf->getAtomTypes()) {
    vdwAtomTypesMap.insert(atomType);
    auto findResult = m_Vdw14Params.find(atomType);
    if (findResult != m_Vdw14Params.end())
      vdw14AtomTypesMap.insert(atomType);
  }

  std::vector<std::string> vdwAtomTypes(vdwAtomTypesMap.begin(),
                                        vdwAtomTypesMap.end());
  std::vector<std::string> vdw14AtomTypes(vdw14AtomTypesMap.begin(),
                                          vdw14AtomTypesMap.end());

  // JEG260817: std::set establishes stable ascending lexicographic type
  // indices. The nested pair loops emit the lower triangle at pair
  // i * (i + 1) / 2 + j, with each pair stored as [C6, C12].

  for (const std::string &atomType : vdwAtomTypes) {
    APOCHARMM_REQUIRE(m_VdwParams.find(atomType) != m_VdwParams.end(),
                      ApoCharmmErrorCode::Runtime,
                      "NONBONDED parameters were not found for atom type \"" +
                          atomType + "\"");
  }

  for (std::size_t i = 0; i < vdwAtomTypes.size(); i++) {
    for (std::size_t j = 0; j <= i; j++) {
      std::string iType = vdwAtomTypes[i];
      std::string jType = vdwAtomTypes[j];

      double epsilon, rmin;

      std::tuple<std::string, std::string> nbfixKey{jType, iType};
      if (m_NbfixParams.find(nbfixKey) != m_NbfixParams.end()) {
        const NBFixParameters &nbfix = m_NbfixParams.at(nbfixKey);
        epsilon = nbfix.emin;
        rmin = nbfix.rmin;
      } else {
        const VdwParameters &iParameters = m_VdwParams.at(iType);
        const VdwParameters &jParameters = m_VdwParams.at(jType);

        const double epsilonI = iParameters.epsilon;
        const double epsilonJ = jParameters.epsilon;
        const double rmin_2I = iParameters.rmin_2;
        const double rmin_2J = jParameters.rmin_2;

        epsilon = std::sqrt(epsilonI * epsilonJ);
        rmin = rmin_2I + rmin_2J;
      }

      double c12 = epsilon * std::pow(rmin, 12);
      double c6 = 2 * epsilon * std::pow(rmin, 6);

      psfVdwParams.push_back(static_cast<float>(c6));
      psfVdwParams.push_back(static_cast<float>(c12));
    }
  }

  for (std::size_t i = 0; i < vdwAtomTypes.size(); i++) {
    for (std::size_t j = 0; j <= i; j++) {
      std::string iType = vdwAtomTypes[i];
      std::string jType = vdwAtomTypes[j];

      double epsilon, rmin;

      std::tuple<std::string, std::string> nbfixKey{jType, iType};
      if (m_NbfixParams.find(nbfixKey) != m_NbfixParams.end()) {
        const NBFixParameters &nbfix = m_NbfixParams.at(nbfixKey);
        epsilon = nbfix.emin;
        rmin = nbfix.rmin;
      } else {
        const VdwParameters &iParameters = m_VdwParams.at(iType);
        const VdwParameters &jParameters = m_VdwParams.at(jType);

        double epsilonI = iParameters.epsilon;
        double epsilonJ = jParameters.epsilon;
        double rmin_2I = iParameters.rmin_2;
        double rmin_2J = jParameters.rmin_2;

        if (std::find(vdw14AtomTypes.begin(), vdw14AtomTypes.end(), iType) !=
            vdw14AtomTypes.end()) {
          const VdwParameters &i14Parameters = m_Vdw14Params.at(iType);
          epsilonI = i14Parameters.epsilon;
          rmin_2I = i14Parameters.rmin_2;
        }

        if (std::find(vdw14AtomTypes.begin(), vdw14AtomTypes.end(), jType) !=
            vdw14AtomTypes.end()) {
          const VdwParameters &j14Parameters = m_Vdw14Params.at(jType);
          epsilonJ = j14Parameters.epsilon;
          rmin_2J = j14Parameters.rmin_2;
        }

        epsilon = std::sqrt(epsilonI * epsilonJ);
        rmin = rmin_2I + rmin_2J;
      }

      double c12 = epsilon * std::pow(rmin, 12);
      double c6 = 2 * epsilon * std::pow(rmin, 6);

      psfVdw14Params.push_back(static_cast<float>(c6));
      psfVdw14Params.push_back(static_cast<float>(c12));
    }
  }

  int index = 0;
  for (const std::string &atomType : psf->getAtomTypes()) {
    auto result = std::find(vdwAtomTypes.begin(), vdwAtomTypes.end(), atomType);
    int pos = result - vdwAtomTypes.begin();
    psfVdwTypes.push_back(pos);
    psfVdw14Types.push_back(pos);
    index++;
  }

  return VdwParamsAndTypes(psfVdwParams, psfVdw14Params, psfVdwTypes,
                           psfVdw14Types);
}

void CharmmParameters::readCharmmParameterFile(
    const std::filesystem::path &filePath) {
  enum class Section {
    NONE,
    ATOMS,
    BONDS,
    ANGLES,
    DIHEDRALS,
    IMPROPERS,
    CMAP,
    NONBONDED,
    NBFIX,
    HBOND
  };

  APOCHARMM_REQUIRE(!filePath.empty(), ApoCharmmErrorCode::InvalidArgument,
                    "CHARMM parameter file path must not be empty");

  const std::string fileName = filePath.string();

  std::ifstream prmFile(filePath);
  APOCHARMM_REQUIRE(prmFile.is_open(), ApoCharmmErrorCode::Runtime,
                    "Failed to open CHARMM parameter file \"" + fileName +
                        "\"");

  std::size_t lineNumber = 0;
  std::string line = "";

  const std::string upperFileName = apo::to_upper(filePath.filename().string());
  if (upperFileName.find("TOPPAR") != std::string::npos) {
    bool parameterBlockFound = false;

    while (std::getline(prmFile, line)) {
      lineNumber++;
      line = CleanPrmLine(line);

      if (line.empty() || (line.front() == '*'))
        continue;

      if ((line.find("READ") != std::string::npos) &&
          (line.find("PARA") != std::string::npos)) {
        parameterBlockFound = true;
        break;
      }
    }

    APOCHARMM_REQUIRE(parameterBlockFound, ApoCharmmErrorCode::Runtime,
                      "CHARMM parameter block was not found in file \"" +
                          fileName + "\"");
  }

  Section section = Section::NONE;

  while (std::getline(prmFile, line)) {
    lineNumber++;
    line = CleanPrmLine(line);

    if (line.empty() || (line.front() == '*'))
      continue;

    std::vector<std::string> tokens = apo::split(line);
    const std::string &keyword = tokens.front();

    if (keyword == "ATOMS") {
      section = Section::ATOMS;
      continue;
    }

    if (keyword == "BONDS") {
      section = Section::BONDS;
      continue;
    }

    if (keyword == "ANGLES") {
      section = Section::ANGLES;
      continue;
    }

    if (keyword == "DIHEDRALS") {
      section = Section::DIHEDRALS;
      continue;
    }

    if (keyword.rfind("IMPR", 0) == 0) {
      section = Section::IMPROPERS;
      continue;
    }

    if (keyword == "CMAP") {
      section = Section::CMAP;
      continue;
    }

    if (keyword == "NONBONDED") {
      section = Section::NONBONDED;
      const std::size_t headerLineNumber = lineNumber;

      while (!tokens.empty() && (tokens.back() == "-")) {
        bool continuationFound = false;

        while (std::getline(prmFile, line)) {
          lineNumber++;
          line = CleanPrmLine(line);

          if (line.empty() || (line.front() == '*'))
            continue;

          tokens = apo::split(line);
          continuationFound = true;
          break;
        }

        APOCHARMM_REQUIRE(continuationFound, ApoCharmmErrorCode::Runtime,
                          "Unexpected end of file while reading the NONBONDED "
                          "header in file \"" +
                              fileName + "\" beginning at line " +
                              std::to_string(headerLineNumber));
      }

      continue;
    }

    if (keyword == "NBFIX") {
      section = Section::NBFIX;
      continue;
    }

    if (keyword == "HBOND") {
      section = Section::HBOND;
      continue;
    }

    if (keyword == "END") {
      section = Section::NONE;
      continue;
    }

    switch (section) {
    case Section::BONDS:
      this->parseBondRecord(tokens, line, fileName, lineNumber);
      break;
    case Section::ANGLES:
      this->parseAngleRecord(tokens, line, fileName, lineNumber);
      break;
    case Section::DIHEDRALS:
      this->parseDihedralRecord(tokens, line, fileName, lineNumber);
      break;
    case Section::IMPROPERS:
      this->parseImproperRecord(tokens, line, fileName, lineNumber);
      break;
    case Section::CMAP:
      this->parseCmapRecord(tokens, prmFile, line, fileName, lineNumber);
      break;
    case Section::NONBONDED:
      this->parseNonbondedRecord(tokens, line, fileName, lineNumber);
      break;
    case Section::NBFIX:
      this->parseNbfixRecord(tokens, line, fileName, lineNumber);
      break;
    case Section::NONE:
    case Section::ATOMS:
    case Section::HBOND:
      break;
    }
  }

  APOCHARMM_REQUIRE(!prmFile.bad(), ApoCharmmErrorCode::Runtime,
                    "Failed while reading CHARMM parameter file \"" + fileName +
                        "\"");

  return;
}

void CharmmParameters::parseBondRecord(const std::vector<std::string> &tokens,
                                       const std::string &line,
                                       const std::string &fileName,
                                       const std::size_t lineNumber) {
  APOCHARMM_REQUIRE(tokens.size() == 4, ApoCharmmErrorCode::Runtime,
                    "Invalid BONDS parameter record in file \"" + fileName +
                        "\" at line " + std::to_string(lineNumber) + ": " +
                        line);

  std::string atom1 = tokens[0];
  std::string atom2 = tokens[1];
  if (atom1 > atom2)
    std::swap(atom1, atom2);

  const std::string valueContext =
      GetPrmValueContext("BONDS", line, fileName, lineNumber);
  const double kb = apo::parse_double(tokens[2], "kb", valueContext);
  const double b0 = apo::parse_double(tokens[3], "b0", valueContext);

  m_BondParams.insert({BondKey(atom1, atom2), BondValues(kb, b0)});

  return;
}

void CharmmParameters::parseAngleRecord(const std::vector<std::string> &tokens,
                                        const std::string &line,
                                        const std::string &fileName,
                                        const std::size_t lineNumber) {
  APOCHARMM_REQUIRE(
      (tokens.size() == 5) || (tokens.size() == 7), ApoCharmmErrorCode::Runtime,
      "Invalid ANGLES parameter record in file \"" + fileName + "\" at line " +
          std::to_string(lineNumber) + ": " + line);

  std::string atom1 = tokens[0];
  const std::string &atom2 = tokens[1];
  std::string atom3 = tokens[2];
  if (atom1 > atom3)
    std::swap(atom1, atom3);

  const std::string valueContext =
      GetPrmValueContext("ANGLES", line, fileName, lineNumber);
  const double degreesToRadians = std::acos(-1.0) / 180.0;
  const double kTheta = apo::parse_double(tokens[3], "kTheta", valueContext);
  const double theta0 =
      degreesToRadians * apo::parse_double(tokens[4], "theta0", valueContext);

  double kub = 0.0;
  double s0 = 0.0;
  if (tokens.size() == 7) {
    kub = apo::parse_double(tokens[5], "kub", valueContext);
    s0 = apo::parse_double(tokens[6], "s0", valueContext);
  }

  const AngleKey key(atom1, atom2, atom3);
  m_AngleParams.insert({key, AngleValues(kTheta, theta0)});
  m_UreybParams.insert({key, BondValues(kub, s0)});

  return;
}

void CharmmParameters::parseDihedralRecord(
    const std::vector<std::string> &tokens, const std::string &line,
    const std::string &fileName, const std::size_t lineNumber) {
  APOCHARMM_REQUIRE(tokens.size() == 7, ApoCharmmErrorCode::Runtime,
                    "Invalid DIHEDRALS parameter record in file \"" + fileName +
                        "\" at line " + std::to_string(lineNumber) + ": " +
                        line);

  std::string atom1 = tokens[0];
  std::string atom2 = tokens[1];
  std::string atom3 = tokens[2];
  std::string atom4 = tokens[3];

  if (atom1 > atom4) {
    std::swap(atom1, atom4);
    std::swap(atom2, atom3);
  } else if ((atom1 == atom4) && (atom2 > atom3))
    std::swap(atom2, atom3);

  const std::string valueContext =
      GetPrmValueContext("DIHEDRALS", line, fileName, lineNumber);
  const double kChi = apo::parse_double(tokens[4], "kChi", valueContext);
  const int multiplicity =
      apo::parse_int(tokens[5], "multiplicity", valueContext);
  const double delta = apo::parse_double(tokens[6], "delta", valueContext);

  const DihedralKey key(atom1, atom2, atom3, atom4);
  m_DihedralParams[key].push_back(DihedralValues(kChi, multiplicity, delta));

  return;
}

void CharmmParameters::parseImproperRecord(
    const std::vector<std::string> &tokens, const std::string &line,
    const std::string &fileName, const std::size_t lineNumber) {
  APOCHARMM_REQUIRE(tokens.size() == 7, ApoCharmmErrorCode::Runtime,
                    "Invalid IMPROPER parameter record in file \"" + fileName +
                        "\" at line " + std::to_string(lineNumber) + ": " +
                        line);

  std::string atom1 = tokens[0];
  std::string atom2 = tokens[1];
  std::string atom3 = tokens[2];
  std::string atom4 = tokens[3];

  if (atom1 > atom4) {
    std::swap(atom1, atom4);
    std::swap(atom2, atom3);
  } else if ((atom1 == atom4) && (atom2 > atom3))
    std::swap(atom2, atom3);

  const std::string valueContext =
      GetPrmValueContext("IMPROPER", line, fileName, lineNumber);
  const double kPsi = apo::parse_double(tokens[4], "kPsi", valueContext);
  static_cast<void>(
      apo::parse_double(tokens[5], "ignored multiplicity", valueContext));
  const double psi0 = apo::parse_double(tokens[6], "psi0", valueContext);

  m_ImproperParams.insert(
      {DihedralKey(atom1, atom2, atom3, atom4), ImDihedralValues(kPsi, psi0)});

  return;
}

void CharmmParameters::parseCmapRecord(
    const std::vector<std::string> &tokens, std::ifstream &prmFile,
    const std::string &line, const std::string &fileName,
    std::size_t &lineNumber) {

  APOCHARMM_REQUIRE(tokens.size() == 9, ApoCharmmErrorCode::Runtime,
      "Invalid CMAP parameter record in file \"" + fileName +
          "\" at line " + std::to_string(lineNumber) + ": " + line);

  const std::string valueContext =
      GetPrmValueContext("CMAP", line, fileName, lineNumber);

  const int gridSize =
      apo::parse_int(tokens[8], "grid size", valueContext);

  APOCHARMM_REQUIRE(
      gridSize == CmapValues::gridSize,
      ApoCharmmErrorCode::Runtime,
      "Unsupported CMAP grid size in file \"" + fileName +
          "\" at line " + std::to_string(lineNumber) +
          ": expected " + std::to_string(CmapValues::gridSize) +
          ", observed " + std::to_string(gridSize));

  std::vector<double> values;
  values.reserve(CmapValues::numValues);

  while (values.size() < CmapValues::numValues) {
    std::string cmapLine;

    APOCHARMM_REQUIRE(
        static_cast<bool>(std::getline(prmFile, cmapLine)),
        ApoCharmmErrorCode::Runtime,
        "Unexpected end of file while reading CMAP parameter in file \"" +
            fileName + "\" after line " +
            std::to_string(lineNumber));

    lineNumber++;
    cmapLine = CleanPrmLine(cmapLine);

    if (cmapLine.empty() || cmapLine.front() == '*')
      continue;

    const std::vector<std::string> cmapTokens = apo::split(cmapLine);

    for (const std::string &token : cmapTokens) {
      const double value =
          apo::parse_double(token, "CMAP value", valueContext);

      values.push_back(value);

      APOCHARMM_REQUIRE(
          values.size() <= CmapValues::numValues,
          ApoCharmmErrorCode::Runtime,
          "Too many CMAP values in file \"" + fileName +
              "\" at line " + std::to_string(lineNumber));
    }
  }

  const DihedralKey dihe1(tokens[0], tokens[1], tokens[2], tokens[3]);
  const DihedralKey dihe2(tokens[4], tokens[5], tokens[6], tokens[7]);
  const CmapKey key(dihe1, dihe2);

  CmapValues cmap(values);

  const double dx = 360.0 / static_cast<double>(CmapValues::gridSize);

  // CHARMM uses xm = grid / 2.
  const int xm = CmapValues::gridSize / 2;

  cmap_set_spline(CmapValues::gridSize, xm, dx, cmap.values, cmap.coeff);

  m_CmapParams.insert_or_assign(key, std::move(cmap));

  return;
}

void CharmmParameters::parseNonbondedRecord(
    const std::vector<std::string> &tokens, const std::string &line,
    const std::string &fileName, const std::size_t lineNumber) {
  APOCHARMM_REQUIRE(
      (tokens.size() == 4) || (tokens.size() == 7), ApoCharmmErrorCode::Runtime,
      "Invalid NONBONDED parameter record in file \"" + fileName +
          "\" at line " + std::to_string(lineNumber) + ": " + line);

  const std::string valueContext =
      GetPrmValueContext("NONBONDED", line, fileName, lineNumber);

  static_cast<void>(
      apo::parse_double(tokens[1], "ignored value", valueContext));
  const double epsilon = apo::parse_double(tokens[2], "epsilon", valueContext);
  const double rmin_2 = apo::parse_double(tokens[3], "rmin/2", valueContext);

  m_VdwParams.insert_or_assign(tokens[0], VdwParameters(epsilon, rmin_2));

  if (tokens.size() == 7) {
    static_cast<void>(
        apo::parse_double(tokens[4], "ignored 1-4", valueContext));
    const double epsilon14 =
        apo::parse_double(tokens[5], "1-4 epsilon", valueContext);
    const double rmin_2_14 =
        apo::parse_double(tokens[6], "1-4 rmin/2", valueContext);

    m_Vdw14Params.insert_or_assign(tokens[0],
                                   VdwParameters(epsilon14, rmin_2_14));
  }

  return;
}

void CharmmParameters::parseNbfixRecord(const std::vector<std::string> &tokens,
                                        const std::string &line,
                                        const std::string &fileName,
                                        const std::size_t lineNumber) {
  APOCHARMM_REQUIRE(
      (tokens.size() == 4) || (tokens.size() == 6), ApoCharmmErrorCode::Runtime,
      "Invalid NBFIX parameter record in file \"" + fileName + "\" at line " +
          std::to_string(lineNumber) + ": " + line);

  std::string atom1 = tokens[0];
  std::string atom2 = tokens[1];
  if (atom1 > atom2)
    std::swap(atom1, atom2);

  const std::string valueContext =
      GetPrmValueContext("NBFIX", line, fileName, lineNumber);

  const double emin =
      std::abs(apo::parse_double(tokens[2], "emin", valueContext));
  const double rmin = apo::parse_double(tokens[3], "rmin", valueContext);

  double emin14 = emin;
  double rmin14 = rmin;
  if (tokens.size() == 6) {
    emin14 = std::abs(apo::parse_double(tokens[4], "1-4 emin", valueContext));
    rmin14 = apo::parse_double(tokens[5], "1-4 rmin", valueContext);
  }

  const NBFixParameters parameters{atom1, atom2, emin, rmin, emin14, rmin14};
  const std::tuple<std::string, std::string> key{atom1, atom2};
  m_NbfixParams.insert({key, parameters});

  return;
}
