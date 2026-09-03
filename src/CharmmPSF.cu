// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmPSF.h"

#include "ApoCharmmError.h"
#include "str_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector_functions.h>

namespace {

std::string GetPsfValueContext(const std::string_view sectionName,
                               const std::string &fileName,
                               const std::size_t lineNumber) {
  return std::string(sectionName) + " section of PSF \"" + fileName +
         "\" at line " + std::to_string(lineNumber);
}

std::vector<std::string> FindPsfSection(
    std::string &line, std::size_t &position, std::size_t &lineNumber,
    const std::string_view fileData, const std::string_view marker,
    const std::string_view sectionName, const std::string &fileName) {
  while (position < fileData.size()) {
    apo::get_line(line, position, lineNumber, fileData, "section headers",
                  "PSF \"" + fileName + "\"");

    std::vector<std::string> tokens = apo::split(line);
    if ((tokens.size() >= 2) && (tokens[1] == marker))
      return tokens;
  }

  APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                  "Could not find " + std::string(sectionName) +
                      " section in PSF \"" + fileName + "\"");
}

void RequireSupportedPsfCount(const unsigned long long int count,
                              const std::string_view sectionName,
                              const std::string &fileName,
                              const std::size_t lineNumber) {
  APOCHARMM_REQUIRE(
      count <=
          static_cast<unsigned long long int>(std::numeric_limits<int>::max()),
      ApoCharmmErrorCode::Runtime,
      std::string(sectionName) + " count exceeds supported range in PSF \"" +
          fileName + "\" at line " + std::to_string(lineNumber));
  return;
}

int ParsePsfAtomNumber(const std::string &token,
                       const std::string_view sectionName,
                       const std::string &fileName,
                       const std::size_t lineNumber, const int numAtoms) {
  const int atomNumber =
      apo::parse_int(token, "atom index",
                     GetPsfValueContext(sectionName, fileName, lineNumber));

  APOCHARMM_REQUIRE((atomNumber >= 1) && (atomNumber <= numAtoms),
                    ApoCharmmErrorCode::Runtime,
                    std::string(sectionName) + " atom index \"" + token +
                        "\" is out of range in PSF \"" + fileName +
                        "\" at line " + std::to_string(lineNumber));

  return atomNumber;
}

} // namespace

CharmmPSF::CharmmPSF(void)
    : m_NumAtoms(-1), m_SegmentIdentifiers(), m_ResidueIdentifiers(),
      m_ResidueNames(), m_AtomNames(), m_AtomTypes(), m_Charges(), m_Masses(),
      m_NumBonds(-1), m_Bonds(), m_NumAngles(-1), m_Angles(),
      m_NumDihedrals(-1), m_Dihedrals(), m_NumImpropers(-1), m_Impropers(),
      m_NumCrossTerms(-1), m_CrossTerms(), m_Connected12(), m_Connected13(),
      m_Connected14(), m_Iblo14(), m_Inb14(), m_WaterMolecules(), m_Residues(),
      m_Groups(), m_FilePath() {}

CharmmPSF::CharmmPSF(const std::filesystem::path &filePath) : CharmmPSF() {
  this->readCharmmPSF(filePath);
  this->initializeWaterMolecules();
  this->createConnectedComponents();
  this->buildTopologicalExclusions();
}

CharmmPSF::CharmmPSF(const CharmmPSF &other)
    : m_NumAtoms(other.m_NumAtoms),
      m_SegmentIdentifiers(other.m_SegmentIdentifiers),
      m_ResidueIdentifiers(other.m_ResidueIdentifiers),
      m_ResidueNames(other.m_ResidueNames), m_AtomNames(other.m_AtomNames),
      m_AtomTypes(other.m_AtomTypes), m_Charges(other.m_Charges),
      m_Masses(other.m_Masses), m_NumBonds(other.m_NumBonds),
      m_Bonds(other.m_Bonds), m_NumAngles(other.m_NumAngles),
      m_Angles(other.m_Angles), m_NumDihedrals(other.m_NumDihedrals),
      m_Dihedrals(other.m_Dihedrals), m_NumImpropers(other.m_NumImpropers),
      m_Impropers(other.m_Impropers), m_NumCrossTerms(other.m_NumCrossTerms),
      m_CrossTerms(other.m_CrossTerms), m_Connected12(other.m_Connected12),
      m_Connected13(other.m_Connected13), m_Connected14(other.m_Connected14),
      m_Iblo14(other.m_Iblo14), m_Inb14(other.m_Inb14),
      m_WaterMolecules(other.m_WaterMolecules), m_Residues(other.m_Residues),
      m_Groups(other.m_Groups), m_FilePath(other.m_FilePath) {}

CharmmPSF::CharmmPSF(const CharmmPSF &&other)
    : m_NumAtoms(other.m_NumAtoms),
      m_SegmentIdentifiers(other.m_SegmentIdentifiers),
      m_ResidueIdentifiers(other.m_ResidueIdentifiers),
      m_ResidueNames(other.m_ResidueNames), m_AtomNames(other.m_AtomNames),
      m_AtomTypes(other.m_AtomTypes), m_Charges(other.m_Charges),
      m_Masses(other.m_Masses), m_NumBonds(other.m_NumBonds),
      m_Bonds(other.m_Bonds), m_NumAngles(other.m_NumAngles),
      m_Angles(other.m_Angles), m_NumDihedrals(other.m_NumDihedrals),
      m_Dihedrals(other.m_Dihedrals), m_NumImpropers(other.m_NumImpropers),
      m_Impropers(other.m_Impropers), m_NumCrossTerms(other.m_NumCrossTerms),
      m_CrossTerms(other.m_CrossTerms), m_Connected12(other.m_Connected12),
      m_Connected13(other.m_Connected13), m_Connected14(other.m_Connected14),
      m_Iblo14(other.m_Iblo14), m_Inb14(other.m_Inb14),
      m_WaterMolecules(other.m_WaterMolecules), m_Residues(other.m_Residues),
      m_Groups(other.m_Groups), m_FilePath(other.m_FilePath) {}

void CharmmPSF::setNumAtoms(const int numAtoms) {
  APOCHARMM_REQUIRE(numAtoms >= 0, ApoCharmmErrorCode::InvalidArgument,
                    "Number of atoms must be nonnegative; observed " +
                        std::to_string(numAtoms));

  const std::size_t count = static_cast<std::size_t>(numAtoms);

  m_NumAtoms = numAtoms;
  m_SegmentIdentifiers.resize(count);
  m_ResidueIdentifiers.resize(count);
  m_ResidueNames.resize(count);
  m_AtomNames.resize(count);
  m_AtomTypes.resize(count);
  m_Charges.resize(count);
  m_Masses.resize(count);
  return;
}

void CharmmPSF::setAtomCharges(const std::vector<double> &charges) {
  APOCHARMM_REQUIRE(m_NumAtoms >= 0, ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF atom count is not initialized");

  APOCHARMM_REQUIRE(charges.size() == static_cast<std::size_t>(m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "CharmmPSF charge count must match atom count; expected " +
                        std::to_string(m_NumAtoms) + ", observed " +
                        std::to_string(charges.size()));

  m_Charges = charges;

  return;
}

int CharmmPSF::getNumAtoms(void) const { return m_NumAtoms; }

int CharmmPSF::getNumBonds(void) const { return m_NumBonds; }

int CharmmPSF::getNumAngles(void) const { return m_NumAngles; }

int CharmmPSF::getNumDihedrals(void) const { return m_NumDihedrals; }

int CharmmPSF::getNumImpropers(void) const { return m_NumImpropers; }

int CharmmPSF::getNumCrossTerms(void) const { return m_NumCrossTerms; }

const std::vector<std::string> &CharmmPSF::getSegmentIdentifiers(void) const {
  return m_SegmentIdentifiers;
}

const std::vector<int> &CharmmPSF::getResidueIdentifiers(void) const {
  return m_ResidueIdentifiers;
}

const std::vector<std::string> &CharmmPSF::getResidueNames(void) const {
  return m_ResidueNames;
}

const std::vector<std::string> &CharmmPSF::getAtomNames(void) const {
  return m_AtomNames;
}

const std::vector<std::string> &CharmmPSF::getAtomTypes(void) const {
  return m_AtomTypes;
}

const std::vector<double> &CharmmPSF::getCharges(void) const {
  return m_Charges;
}

const std::vector<double> &CharmmPSF::getMasses(void) const { return m_Masses; }

const std::vector<Bond> &CharmmPSF::getBonds(void) const { return m_Bonds; }

const std::vector<Angle> &CharmmPSF::getAngles(void) const { return m_Angles; }

const std::vector<Dihedral> &CharmmPSF::getDihedrals(void) const {
  return m_Dihedrals;
}

const std::vector<Dihedral> &CharmmPSF::getImpropers(void) const {
  return m_Impropers;
}

const std::vector<CrossTerm> &CharmmPSF::getCrossTerms(void) const {
  return m_CrossTerms;
}

const std::vector<std::set<int>> &CharmmPSF::getConnected12(void) const {
  return m_Connected12;
}

const std::vector<std::set<int>> &CharmmPSF::getConnected13(void) const {
  return m_Connected13;
}

const std::vector<std::set<int>> &CharmmPSF::getConnected14(void) const {
  return m_Connected14;
}

const std::vector<int> &CharmmPSF::getIblo14(void) const { return m_Iblo14; }

const std::vector<int> &CharmmPSF::getInb14(void) const { return m_Inb14; }

const CudaContainer<int4> &CharmmPSF::getWaterMolecules(void) const {
  return m_WaterMolecules;
}

const CudaContainer<int2> &CharmmPSF::getResidues(void) const {
  return m_Residues;
}

const CudaContainer<int2> &CharmmPSF::getGroups(void) const { return m_Groups; }

const std::filesystem::path &CharmmPSF::getFilePath(void) const {
  return m_FilePath;
}

std::vector<std::string> &CharmmPSF::getSegmentIdentifiers(void) {
  return m_SegmentIdentifiers;
}

std::vector<int> &CharmmPSF::getResidueIdentifiers(void) {
  return m_ResidueIdentifiers;
}

std::vector<std::string> &CharmmPSF::getResidueNames(void) {
  return m_ResidueNames;
}

std::vector<std::string> &CharmmPSF::getAtomNames(void) { return m_AtomNames; }

std::vector<std::string> &CharmmPSF::getAtomTypes(void) { return m_AtomTypes; }

std::vector<double> &CharmmPSF::getCharges(void) { return m_Charges; }

std::vector<double> &CharmmPSF::getMasses(void) { return m_Masses; }

std::vector<Bond> &CharmmPSF::getBonds(void) { return m_Bonds; }

std::vector<Angle> &CharmmPSF::getAngles(void) { return m_Angles; }

std::vector<Dihedral> &CharmmPSF::getDihedrals(void) { return m_Dihedrals; }

std::vector<Dihedral> &CharmmPSF::getImpropers(void) { return m_Impropers; }

std::vector<CrossTerm> &CharmmPSF::getCrossTerms(void) { return m_CrossTerms; }

std::vector<std::set<int>> &CharmmPSF::getConnected12(void) {
  return m_Connected12;
}

std::vector<std::set<int>> &CharmmPSF::getConnected13(void) {
  return m_Connected13;
}

std::vector<std::set<int>> &CharmmPSF::getConnected14(void) {
  return m_Connected14;
}

std::vector<int> &CharmmPSF::getIblo14(void) { return m_Iblo14; }

std::vector<int> &CharmmPSF::getInb14(void) { return m_Inb14; }

CudaContainer<int4> &CharmmPSF::getWaterMolecules(void) {
  return m_WaterMolecules;
}

CudaContainer<int2> &CharmmPSF::getResidues(void) { return m_Residues; }

CudaContainer<int2> &CharmmPSF::getGroups(void) { return m_Groups; }

std::filesystem::path &CharmmPSF::getFilePath(void) { return m_FilePath; }

double CharmmPSF::getNetCharge(void) const {
  APOCHARMM_REQUIRE(m_NumAtoms >= 0, ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF atom count is not initialized");

  APOCHARMM_REQUIRE(
      m_Charges.size() == static_cast<std::size_t>(m_NumAtoms),
      ApoCharmmErrorCode::Runtime,
      "CharmmPSF charge count does not match atom count; expected " +
          std::to_string(m_NumAtoms) + ", observed " +
          std::to_string(m_Charges.size()));

  double netCharge = 0.0;
  for (int i = 0; i < m_NumAtoms; i++)
    netCharge += m_Charges[i];

  return netCharge;
}

double CharmmPSF::getTotalMass(void) const {
  APOCHARMM_REQUIRE(m_NumAtoms >= 0, ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF atom count is not initialized");

  APOCHARMM_REQUIRE(
      m_Masses.size() == static_cast<std::size_t>(m_NumAtoms),
      ApoCharmmErrorCode::Runtime,
      "CharmmPSF mass count does not match atom count; expected " +
          std::to_string(m_NumAtoms) + ", observed " +
          std::to_string(m_Masses.size()));

  double totalMass = 0.0;
  for (int i = 0; i < m_NumAtoms; i++)
    totalMass += m_Masses[i];

  return totalMass;
}

InclusionExclusion CharmmPSF::getInclusionExclusionLists(void) const {
  APOCHARMM_REQUIRE(m_NumAtoms >= 0, ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF atom count is not initialized");

  const std::size_t expectedSize = static_cast<std::size_t>(m_NumAtoms);
  APOCHARMM_REQUIRE(
      (m_Connected12.size() == expectedSize) &&
          (m_Connected13.size() == expectedSize) &&
          (m_Connected14.size() == expectedSize),
      ApoCharmmErrorCode::Runtime,
      "CharmmPSF connectivity list sizes do not match atom count; expected " +
          std::to_string(m_NumAtoms) + ", observed " +
          std::to_string(m_Connected12.size()) + ", " +
          std::to_string(m_Connected13.size()) + ", " +
          std::to_string(m_Connected14.size()));

  // Fill in from 1-2, 1-3, 1-4 connections
  std::vector<int> inclusion, exclusion;
  for (int iatom = 0; iatom < m_NumAtoms; iatom++) {
    // Inclusions
    for (const int jatom : m_Connected14[iatom]) {
      if (jatom > iatom) {
        if ((m_Connected12[iatom].find(jatom) == m_Connected12[iatom].end()) &&
            (m_Connected13[iatom].find(jatom) == m_Connected13[iatom].end())) {
          inclusion.push_back(iatom);
          inclusion.push_back(jatom);
        }
      }
    }

    // Exclusions
    std::set<int> ex;
    for (const int jatom : m_Connected12[iatom]) {
      if (jatom > iatom)
        ex.insert(jatom);
    }
    for (const int jatom : m_Connected13[iatom]) {
      if (jatom > iatom)
        ex.insert(jatom);
    }
    for (const int jatom : ex) {
      exclusion.push_back(iatom);
      exclusion.push_back(jatom);
    }
  }

  std::vector<int> sizes = {static_cast<int>(inclusion.size() / 2),
                            static_cast<int>(exclusion.size() / 2)};
  std::vector<int> in14_ex14;
  in14_ex14.insert(in14_ex14.end(), inclusion.begin(), inclusion.end());
  in14_ex14.insert(in14_ex14.end(), exclusion.begin(), exclusion.end());
  // for (std::size_t i = 0; i < inclusion.size(); i++)
  //   std::cout << "inclusion[" << i << "] = " << inclusion[i] << std::endl;
  // for (std::size_t i = 0; i < exclusion.size(); i++)
  //   std::cout << "exclusion[" << i << "] = " << exclusion[i] << std::endl;
  // for (std::size_t i = 0; i < in14_ex14.size(); i++)
  //   std::cout << "in14_ex14[" << i << "] = " << in14_ex14[i] << std::endl;

  return InclusionExclusion(sizes, in14_ex14);
}

void CharmmPSF::initializeWaterMolecules(void) {
  m_WaterMolecules.clear();

  for (int i = 0; i < m_NumAtoms - 2; i++) {
    if ((m_AtomTypes[i + 0] == "OT") && (m_AtomTypes[i + 1] == "HT") &&
        (m_AtomTypes[i + 2] == "HT")) {
      m_WaterMolecules.push_back(make_int4(i + 0, i + 1, i + 2, 0));
      i += 2;
    }
  }
  m_WaterMolecules.shrink_to_fit();

  return;
}

int find(int node, const std::vector<int> &link) {
  while (node != link[node])
    node = link[node];
  return node;
}

void CharmmPSF::createConnectedComponents(void) {
  std::vector<int> link(m_NumAtoms);
  for (int i = 0; i < m_NumAtoms; i++)
    link[i] = i;

  for (int i = 0; i < m_NumBonds; i++) {
    const int iatom = m_Bonds[i].iatom;
    const int jatom = m_Bonds[i].jatom;
    const int rep1 = find(iatom, link);
    const int rep2 = find(jatom, link);
    if (rep1 != rep2) {
      if (rep1 > rep2)
        link[rep2] = rep1;
      else
        link[rep1] = rep2;
    }
  }

  int startAtom = 0;
  while (startAtom < m_NumAtoms) {
    int endAtom = find(startAtom, link);
    m_Groups.push_back(make_int2(startAtom, endAtom));
    startAtom = endAtom + 1;
  }
  m_Groups.shrink_to_fit();

  return;
}

void CharmmPSF::buildTopologicalExclusions(void) {
  m_Connected12.resize(m_NumAtoms);
  m_Connected13.resize(m_NumAtoms);
  m_Connected14.resize(m_NumAtoms);

  // 1-2 exclusions
  for (const Bond bond : m_Bonds) {
    m_Connected12[bond.iatom].insert(bond.jatom);
    m_Connected12[bond.jatom].insert(bond.iatom);
  }

  // 1-3 exclusions
  for (const Bond bond : m_Bonds) {
    for (const int iatom : m_Connected12[bond.jatom]) {
      if (iatom != bond.iatom)
        m_Connected13[bond.iatom].insert(iatom);
    }
    for (const int jatom : m_Connected12[bond.iatom]) {
      if (jatom != bond.jatom)
        m_Connected13[bond.jatom].insert(jatom);
    }
  }

  // 1-4 exclusions
  for (int iatom = 0; iatom < m_NumAtoms; iatom++) {
    for (const int jatom : m_Connected13[iatom]) {
      for (const int katom : m_Connected12[jatom]) {
        if (m_Connected12[iatom].count(katom) == 0)
          m_Connected14[iatom].insert(katom);
      }
    }
  }

  m_Iblo14.clear();
  m_Inb14.clear();

  for (int iatom = 0; iatom < m_NumAtoms; iatom++) {
    std::set<int> connectedAtoms;
    for (const int jatom : m_Connected12[iatom])
      connectedAtoms.insert(jatom);
    for (const int katom : m_Connected13[iatom])
      connectedAtoms.insert(katom);
    for (const int latom : m_Connected14[iatom])
      connectedAtoms.insert(latom);
    for (const int connectedAtom : connectedAtoms) {
      if (connectedAtom > iatom)
        m_Inb14.push_back(connectedAtom + 1);
    }
    m_Iblo14.push_back(static_cast<int>(m_Inb14.size()));
  }
  m_Iblo14.shrink_to_fit();
  m_Inb14.shrink_to_fit();

  return;
}

void CharmmPSF::readCharmmPSF(const std::filesystem::path &filePath) {
  APOCHARMM_REQUIRE(!filePath.empty(), ApoCharmmErrorCode::InvalidArgument,
                    "CHARMM PSF file path must not be empty");

  m_FilePath = filePath;

  std::string fileData = "";
  apo::read_file_into_string(fileData, filePath);

  const std::string fileName = filePath.string();
  const std::string psfSource = "PSF \"" + fileName + "\"";

  std::size_t pos = 0;
  std::size_t lineNumber = 0;
  std::string line = "";
  std::vector<std::string> tokens;

  // Parse TITLE section
  tokens = FindPsfSection(line, pos, lineNumber, fileData, "!NTITLE", "TITLE",
                          fileName);
  const unsigned long long int ntitle = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("TITLE", fileName, lineNumber));
  const unsigned long long int nlineTitle = ntitle;
  for (unsigned long long int i = 0; i < nlineTitle; i++)
    apo::get_line(line, pos, lineNumber, fileData, "TITLE records", psfSource);

  // Parse ATOM section
  tokens = FindPsfSection(line, pos, lineNumber, fileData, "!NATOM", "ATOM",
                          fileName);
  const unsigned long long int natom = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("ATOM", fileName, lineNumber));
  RequireSupportedPsfCount(natom, "ATOM", fileName, lineNumber);

  const unsigned long long int nlineAtom = natom;
  this->setNumAtoms(static_cast<int>(natom));

  bool hasResidue = false;
  int resiOld = 0;
  int resiStartIdx = 0;
  int resiEndIdx = -1;

  for (unsigned long long int i = 0; i < nlineAtom; i++) {
    apo::get_line(line, pos, lineNumber, fileData, "ATOM records", psfSource);
    tokens = apo::split(line);

    APOCHARMM_REQUIRE(tokens.size() >= 8, ApoCharmmErrorCode::Runtime,
                      "Invalid ATOM record in PSF \"" + fileName +
                          "\" at line " + std::to_string(lineNumber) + ": " +
                          line);

    const std::string valueContext =
        GetPsfValueContext("ATOM", fileName, lineNumber);

    const std::string &segi = tokens[1];
    const int resi =
        apo::parse_int(tokens[2], "residue identifier", valueContext);
    const std::string &resn = tokens[3];
    const std::string &anam = tokens[4];
    const std::string &atyp = tokens[5];
    const double chrg = apo::parse_double(tokens[6], "charge", valueContext);
    const double mass = apo::parse_double(tokens[7], "mass", valueContext);

    const std::size_t atomIndex = static_cast<std::size_t>(i);
    m_SegmentIdentifiers[atomIndex] = segi;
    m_ResidueIdentifiers[atomIndex] = resi;
    m_ResidueNames[atomIndex] = resn;
    m_AtomNames[atomIndex] = anam;
    m_AtomTypes[atomIndex] = atyp;
    m_Charges[atomIndex] = chrg;
    m_Masses[atomIndex] = mass;

    if (!hasResidue) {
      resiOld = resi;
      hasResidue = true;
    }

    if (resiOld != resi) {
      resiEndIdx = static_cast<int>(i) - 1;
      m_Residues.push_back(make_int2(resiStartIdx, resiEndIdx));
      resiStartIdx = static_cast<int>(i);
    }

    resiOld = resi;
  }

  if (m_NumAtoms > 0)
    m_Residues.push_back(make_int2(resiStartIdx, m_NumAtoms - 1));

  m_Residues.shrink_to_fit();

  // Ensure no extra whitespace in string variables
  for (std::string &segi : m_SegmentIdentifiers)
    apo::trim_ip(segi);
  for (std::string &resn : m_ResidueNames)
    apo::trim_ip(resn);
  for (std::string &anam : m_AtomNames)
    apo::trim_ip(anam);
  for (std::string &atyp : m_AtomTypes)
    apo::trim_ip(atyp);

  // Parse BOND section
  tokens = FindPsfSection(line, pos, lineNumber, fileData, "!NBOND:", "BOND",
                          fileName);
  const unsigned long long int nbond = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("BOND", fileName, lineNumber));
  RequireSupportedPsfCount(nbond, "BOND", fileName, lineNumber);

  const unsigned long long int nlineBond =
      nbond / 4 + ((nbond % 4 == 0) ? 0 : 1);
  m_NumBonds = static_cast<int>(nbond);
  m_Bonds.resize(nbond);

  unsigned long long int ibond = 0;
  for (unsigned long long int i = 0; i < nlineBond; i++) {
    apo::get_line(line, pos, lineNumber, fileData, "BOND records", psfSource);
    tokens = apo::split(line);

    const std::size_t termsOnLine = static_cast<std::size_t>(
        std::min<unsigned long long int>(4, nbond - ibond));
    APOCHARMM_REQUIRE(
        tokens.size() == 2 * termsOnLine, ApoCharmmErrorCode::Runtime,
        "Invalid BOND record in PSF \"" + fileName + "\" at line " +
            std::to_string(lineNumber) + ": " + line);

    for (std::size_t term = 0; term < termsOnLine; term++) {
      const std::size_t offset = 2 * term;
      Bond &bond = m_Bonds[ibond];
      bond.iatom = ParsePsfAtomNumber(tokens[offset + 0], "BOND", fileName,
                                      lineNumber, m_NumAtoms) -
                   1;
      bond.jatom = ParsePsfAtomNumber(tokens[offset + 1], "BOND", fileName,
                                      lineNumber, m_NumAtoms) -
                   1;
      ibond++;
    }
  }

  // Parse ANGLe section
  tokens = FindPsfSection(line, pos, lineNumber, fileData, "!NTHETA:", "ANGLE",
                          fileName);
  const unsigned long long int ntheta = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("ANGLE", fileName, lineNumber));
  RequireSupportedPsfCount(ntheta, "ANGLE", fileName, lineNumber);

  const unsigned long long int nlineTheta =
      ntheta / 3 + ((ntheta % 3 == 0) ? 0 : 1);
  m_NumAngles = static_cast<int>(ntheta);
  m_Angles.resize(ntheta);

  unsigned long long int itheta = 0;
  for (unsigned long long int i = 0; i < nlineTheta; i++) {
    apo::get_line(line, pos, lineNumber, fileData, "ANGLE records", psfSource);
    tokens = apo::split(line);

    const std::size_t termsOnLine = static_cast<std::size_t>(
        std::min<unsigned long long int>(3, ntheta - itheta));
    APOCHARMM_REQUIRE(
        tokens.size() == 3 * termsOnLine, ApoCharmmErrorCode::Runtime,
        "Invalid ANGLE record in PSF \"" + fileName + "\" at line " +
            std::to_string(lineNumber) + ": " + line);

    for (std::size_t term = 0; term < termsOnLine; term++) {
      const std::size_t offset = 3 * term;
      Angle &angle = m_Angles[itheta];
      angle.iatom = ParsePsfAtomNumber(tokens[offset + 0], "ANGLE", fileName,
                                       lineNumber, m_NumAtoms) -
                    1;
      angle.jatom = ParsePsfAtomNumber(tokens[offset + 1], "ANGLE", fileName,
                                       lineNumber, m_NumAtoms) -
                    1;
      angle.katom = ParsePsfAtomNumber(tokens[offset + 2], "ANGLE", fileName,
                                       lineNumber, m_NumAtoms) -
                    1;
      itheta++;
    }
  }

  // Parse DIHEdral section
  tokens = FindPsfSection(line, pos, lineNumber, fileData, "!NPHI:", "DIHEDRAL",
                          fileName);
  const unsigned long long int nphi = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("DIHEDRAL", fileName, lineNumber));
  RequireSupportedPsfCount(nphi, "DIHEDRAL", fileName, lineNumber);

  const unsigned long long int nlinePhi = nphi / 2 + ((nphi % 2 == 0) ? 0 : 1);
  m_NumDihedrals = static_cast<int>(nphi);
  m_Dihedrals.resize(nphi);

  unsigned long long int iphi = 0;
  for (unsigned long long int i = 0; i < nlinePhi; i++) {
    apo::get_line(line, pos, lineNumber, fileData, "DIHEDRAL records",
                  psfSource);
    tokens = apo::split(line);

    const std::size_t termsOnLine = static_cast<std::size_t>(
        std::min<unsigned long long int>(2, nphi - iphi));
    APOCHARMM_REQUIRE(
        tokens.size() == 4 * termsOnLine, ApoCharmmErrorCode::Runtime,
        "Invalid DIHEDRAL record in PSF \"" + fileName + "\" at line " +
            std::to_string(lineNumber) + ": " + line);

    for (std::size_t term = 0; term < termsOnLine; term++) {
      const std::size_t offset = 4 * term;
      Dihedral &dihedral = m_Dihedrals[iphi];
      dihedral.iatom = ParsePsfAtomNumber(tokens[offset + 0], "DIHEDRAL",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      dihedral.jatom = ParsePsfAtomNumber(tokens[offset + 1], "DIHEDRAL",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      dihedral.katom = ParsePsfAtomNumber(tokens[offset + 2], "DIHEDRAL",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      dihedral.latom = ParsePsfAtomNumber(tokens[offset + 3], "DIHEDRAL",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      iphi++;
    }
  }

  // Parse IMPRoper dihedral section
  tokens = FindPsfSection(line, pos, lineNumber, fileData,
                          "!NIMPHI:", "IMPROPER", fileName);
  const unsigned long long int nimphi = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("IMPROPER", fileName, lineNumber));
  RequireSupportedPsfCount(nimphi, "IMPROPER", fileName, lineNumber);

  const unsigned long long int nlineImphi =
      nimphi / 2 + ((nimphi % 2 == 0) ? 0 : 1);
  m_NumImpropers = static_cast<int>(nimphi);
  m_Impropers.resize(nimphi);

  unsigned long long int iimphi = 0;
  for (unsigned long long int i = 0; i < nlineImphi; i++) {
    apo::get_line(line, pos, lineNumber, fileData, "IMPROPER records",
                  psfSource);
    tokens = apo::split(line);

    const std::size_t termsOnLine = static_cast<std::size_t>(
        std::min<unsigned long long int>(2, nimphi - iimphi));
    APOCHARMM_REQUIRE(
        tokens.size() == 4 * termsOnLine, ApoCharmmErrorCode::Runtime,
        "Invalid IMPROPER record in PSF \"" + fileName + "\" at line " +
            std::to_string(lineNumber) + ": " + line);

    for (std::size_t term = 0; term < termsOnLine; term++) {
      const std::size_t offset = 4 * term;
      Dihedral &improper = m_Impropers[iimphi];
      improper.iatom = ParsePsfAtomNumber(tokens[offset + 0], "IMPROPER",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      improper.jatom = ParsePsfAtomNumber(tokens[offset + 1], "IMPROPER",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      improper.katom = ParsePsfAtomNumber(tokens[offset + 2], "IMPROPER",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      improper.latom = ParsePsfAtomNumber(tokens[offset + 3], "IMPROPER",
                                          fileName, lineNumber, m_NumAtoms) -
                       1;
      iimphi++;
    }
  }

  // Parse DONOr section
  tokens = FindPsfSection(line, pos, lineNumber, fileData, "!NDON:", "DONOR",
                          fileName);
  const unsigned long long int ndon = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("DONOR", fileName, lineNumber));
  const unsigned long long int nlineDon = ndon / 4 + ((ndon % 4 == 0) ? 0 : 1);
  // unsigned long long int idon = 0;
  for (unsigned long long int i = 0; i < nlineDon; i++)
    apo::get_line(line, pos, lineNumber, fileData, "DONOR records", psfSource);

  // Parse ACCEptor section
  tokens = FindPsfSection(line, pos, lineNumber, fileData, "!NACC:", "ACCEPTOR",
                          fileName);
  const unsigned long long int nacc = apo::parse_ull(
      tokens[0], "count", GetPsfValueContext("ACCEPTOR", fileName, lineNumber));
  const unsigned long long int nlineAcc = nacc / 4 + ((nacc % 4 == 0) ? 0 : 1);
  // unsigned long long int iacc = 0;
  for (unsigned long long int i = 0; i < nlineAcc; i++) {
    apo::get_line(line, pos, lineNumber, fileData, "ACCEPTOR records",
                  psfSource);
  }

  // Sections between ACCEPTOR and CROSS-TERM are ignored.

  // Parse CRoss-TERM section
  tokens = FindPsfSection(line, pos, lineNumber, fileData,
                          "!NCRTERM:", "CROSS-TERM", fileName);
  const unsigned long long int ncrterm =
      apo::parse_ull(tokens[0], "count",
                     GetPsfValueContext("CROSS-TERM", fileName, lineNumber));
  RequireSupportedPsfCount(ncrterm, "CROSS-TERM", fileName, lineNumber);

  const unsigned long long int nlineCrterm = ncrterm;
  m_NumCrossTerms = static_cast<int>(ncrterm);
  m_CrossTerms.resize(ncrterm);

  for (unsigned long long int i = 0; i < nlineCrterm; i++) {
    apo::get_line(line, pos, lineNumber, fileData, "CROSS-TERM records",
                  psfSource);
    tokens = apo::split(line);

    APOCHARMM_REQUIRE(tokens.size() == 8, ApoCharmmErrorCode::Runtime,
                      "Invalid CROSS-TERM record in PSF \"" + fileName +
                          "\" at line " + std::to_string(lineNumber) + ": " +
                          line);

    CrossTerm &crossTerm = m_CrossTerms[i];
    crossTerm.iatom1 = ParsePsfAtomNumber(tokens[0], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
    crossTerm.jatom1 = ParsePsfAtomNumber(tokens[1], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
    crossTerm.katom1 = ParsePsfAtomNumber(tokens[2], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
    crossTerm.latom1 = ParsePsfAtomNumber(tokens[3], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
    crossTerm.iatom2 = ParsePsfAtomNumber(tokens[4], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
    crossTerm.jatom2 = ParsePsfAtomNumber(tokens[5], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
    crossTerm.katom2 = ParsePsfAtomNumber(tokens[6], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
    crossTerm.latom2 = ParsePsfAtomNumber(tokens[7], "CROSS-TERM", fileName,
                                          lineNumber, m_NumAtoms) - 1;
  }

  return;
}
