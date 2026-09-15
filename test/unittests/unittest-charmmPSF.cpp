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
#include "CharmmPSF.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <string>
#include <type_traits>
#include <utility>

static_assert(std::is_nothrow_destructible_v<CharmmPSF>);
static_assert(std::is_copy_constructible_v<CharmmPSF>);
static_assert(std::is_copy_assignable_v<CharmmPSF>);
static_assert(std::is_nothrow_move_constructible_v<CharmmPSF>);
static_assert(std::is_nothrow_move_assignable_v<CharmmPSF>);

namespace {

const std::string SINGLE_ATOM_PSF_PREFIX = R"PSF(PSF

       0 !NTITLE
       1 !NATOM
       1 SEG1     1 RES1 C1   CT1    0.000000  12.0110           0
)PSF";

const std::string EMPTY_BOND_SECTION = "       0 !NBOND: bonds\n";
const std::string EMPTY_ANGLE_SECTION = "       0 !NTHETA: angles\n";
const std::string EMPTY_DIHEDRAL_SECTION = "       0 !NPHI: dihedrals\n";
const std::string EMPTY_IMPROPER_SECTION = "       0 !NIMPHI: impropers\n";
const std::string EMPTY_DONOR_SECTION = "       0 !NDON: donors\n";
const std::string EMPTY_ACCEPTOR_SECTION = "       0 !NACC: acceptors\n";

void CheckDefaultPsfState(const CharmmPSF &psf) {
  CHECK(psf.getNumAtoms() == -1);
  CHECK(psf.getNumBonds() == -1);
  CHECK(psf.getNumAngles() == -1);
  CHECK(psf.getNumDihedrals() == -1);
  CHECK(psf.getNumImpropers() == -1);
  CHECK(psf.getNumCrossTerms() == -1);

  CHECK(psf.getSegmentIdentifiers().empty() == true);
  CHECK(psf.getResidueIdentifiers().empty() == true);
  CHECK(psf.getResidueNames().empty() == true);
  CHECK(psf.getAtomNames().empty() == true);
  CHECK(psf.getAtomTypes().empty() == true);
  CHECK(psf.getCharges().empty() == true);
  CHECK(psf.getMasses().empty() == true);

  CHECK(psf.getBonds().empty() == true);
  CHECK(psf.getAngles().empty() == true);
  CHECK(psf.getDihedrals().empty() == true);
  CHECK(psf.getImpropers().empty() == true);
  CHECK(psf.getCrossTerms().empty() == true);

  CHECK(psf.getConnected12().empty() == true);
  CHECK(psf.getConnected13().empty() == true);
  CHECK(psf.getConnected14().empty() == true);
  CHECK(psf.getIblo14().empty() == true);
  CHECK(psf.getInb14().empty() == true);

  CHECK(psf.getWaterMolecules().size() == 0);
  CHECK(psf.getWaterMolecules().getDeviceArray().data() == nullptr);
  CHECK(psf.getResidues().size() == 0);
  CHECK(psf.getResidues().getDeviceArray().data() == nullptr);
  CHECK(psf.getGroups().size() == 0);
  CHECK(psf.getGroups().getDeviceArray().data() == nullptr);
  CHECK(psf.getFilePath().empty() == true);
}

void CheckPsfFileError(const std::string &fileName, const std::string &contents,
                       const ApoCharmmErrorCode expectedCode,
                       const std::string &expectedMessage) {
  apo_test::RemoveIfExists(fileName);
  apo_test::WriteTextFile(fileName, contents);

  apo_test::CheckApoCharmmError(
      [&fileName](void) {
        CharmmPSF psf(fileName);
        static_cast<void>(psf);
      },
      expectedCode, expectedMessage);

  apo_test::RemoveIfExists(fileName);

  return;
}

} // namespace

TEST_CASE("CharmmPSFDefaultConstructor") {
  CharmmPSF psf;
  CheckDefaultPsfState(psf);
}

TEST_CASE("CharmmPSFParsesLinearChain") {
  const std::string fileName = "tmp_charmm_psf_linear_chain.psf";
  const std::string psfText = R"PSF(PSF

       1 !NTITLE
 REMARKS generated CharmmPSF linear-chain unit test
       4 !NATOM
       1 SEG1     1 RES1 C1   CT1   -0.300000  12.0110           0
       2 SEG1     1 RES1 H1   HA1    0.100000   1.0080           0
       3 SEG1     2 RES2 C2   CT2    0.100000  12.0110           0
       4 SEG1     2 RES2 H2   HA2    0.100000   1.0080           0
       3 !NBOND: bonds
       1       2       2       3       3       4
       2 !NTHETA: angles
       1       2       3       2       3       4
       1 !NPHI: dihedrals
       1       2       3       4
       0 !NIMPHI: impropers
       0 !NDON: donors
       0 !NACC: acceptors
       0 !NCRTERM: cross-terms
)PSF";

  apo_test::WriteTextFile(fileName, psfText);

  CharmmPSF psf(fileName);

  SECTION("CountsAndFilePath") {
    CHECK(psf.getFilePath() == fileName);
    CHECK(psf.getNumAtoms() == 4);
    CHECK(psf.getNumBonds() == 3);
    CHECK(psf.getNumAngles() == 2);
    CHECK(psf.getNumDihedrals() == 1);
    CHECK(psf.getNumImpropers() == 0);
    CHECK(psf.getNumCrossTerms() == 0);
  }

  SECTION("AtomMetadata") {
    CHECK(psf.getSegmentIdentifiers() ==
          std::vector<std::string>{"SEG1", "SEG1", "SEG1", "SEG1"});
    CHECK(psf.getResidueIdentifiers() == std::vector<int>{1, 1, 2, 2});
    CHECK(psf.getResidueNames() ==
          std::vector<std::string>{"RES1", "RES1", "RES2", "RES2"});
    CHECK(psf.getAtomNames() ==
          std::vector<std::string>{"C1", "H1", "C2", "H2"});
    CHECK(psf.getAtomTypes() ==
          std::vector<std::string>{"CT1", "HA1", "CT2", "HA2"});

    REQUIRE(psf.getCharges().size() == 4);
    CHECK(psf.getCharges()[0] == Approx(-0.3));
    CHECK(psf.getCharges()[1] == Approx(0.1));
    CHECK(psf.getCharges()[2] == Approx(0.1));
    CHECK(psf.getCharges()[3] == Approx(0.1));

    REQUIRE(psf.getMasses().size() == 4);
    CHECK(psf.getMasses()[0] == Approx(12.011));
    CHECK(psf.getMasses()[1] == Approx(1.008));
    CHECK(psf.getMasses()[2] == Approx(12.011));
    CHECK(psf.getMasses()[3] == Approx(1.008));

    CHECK(psf.getNetCharge() == Approx(0.0).margin(1.0e-12));
    CHECK(psf.getTotalMass() == Approx(26.038));
  }

  SECTION("BondedTerms") {
    REQUIRE(psf.getBonds().size() == 3);
    CHECK(psf.getBonds()[0].iatom == 0);
    CHECK(psf.getBonds()[0].jatom == 1);
    CHECK(psf.getBonds()[1].iatom == 1);
    CHECK(psf.getBonds()[1].jatom == 2);
    CHECK(psf.getBonds()[2].iatom == 2);
    CHECK(psf.getBonds()[2].jatom == 3);

    REQUIRE(psf.getAngles().size() == 2);
    CHECK(psf.getAngles()[0].iatom == 0);
    CHECK(psf.getAngles()[0].jatom == 1);
    CHECK(psf.getAngles()[0].katom == 2);
    CHECK(psf.getAngles()[1].iatom == 1);
    CHECK(psf.getAngles()[1].jatom == 2);
    CHECK(psf.getAngles()[1].katom == 3);

    REQUIRE(psf.getDihedrals().size() == 1);
    CHECK(psf.getDihedrals()[0].iatom == 0);
    CHECK(psf.getDihedrals()[0].jatom == 1);
    CHECK(psf.getDihedrals()[0].katom == 2);
    CHECK(psf.getDihedrals()[0].latom == 3);

    CHECK(psf.getImpropers().empty() == true);
    CHECK(psf.getCrossTerms().empty() == true);
  }

  SECTION("ResiduesGroupsAndWater") {
    const std::vector<int2> &residues = psf.getResidues().getHostArray();
    REQUIRE(residues.size() == 2);
    CHECK(residues[0].x == 0);
    CHECK(residues[0].y == 1);
    CHECK(residues[1].x == 2);
    CHECK(residues[1].y == 3);

    const std::vector<int2> &groups = psf.getGroups().getHostArray();
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].x == 0);
    CHECK(groups[0].y == 3);

    CHECK(psf.getWaterMolecules().size() == 0);
    CHECK(psf.getWaterMolecules().getHostArray().empty() == true);
  }

  SECTION("ConnectedComponentsAndExclusions") {
    const std::vector<std::set<int>> expected12 = {
        std::set<int>{1}, std::set<int>{0, 2}, std::set<int>{1, 3},
        std::set<int>{2}};
    const std::vector<std::set<int>> expected13 = {
        std::set<int>{2}, std::set<int>{3}, std::set<int>{0}, std::set<int>{1}};
    const std::vector<std::set<int>> expected14 = {
        std::set<int>{3}, std::set<int>{}, std::set<int>{}, std::set<int>{0}};

    CHECK(psf.getConnected12() == expected12);
    CHECK(psf.getConnected13() == expected13);
    CHECK(psf.getConnected14() == expected14);

    CHECK(psf.getIblo14() == std::vector<int>{3, 5, 6, 6});
    CHECK(psf.getInb14() == std::vector<int>{2, 3, 4, 3, 4, 4});

    const InclusionExclusion exclusionLists = psf.getInclusionExclusionLists();

    CHECK(exclusionLists.sizes == std::vector<int>{1, 5});
    CHECK(exclusionLists.in14_ex14 ==
          std::vector<int>{0, 3, 0, 1, 0, 2, 1, 2, 1, 3, 2, 3});
  }

  SECTION("CopyConstructorDeepCopy") {
    CharmmPSF copy(psf);

    CHECK(copy.getFilePath() == psf.getFilePath());
    CHECK(copy.getNumAtoms() == psf.getNumAtoms());
    CHECK(copy.getNumBonds() == psf.getNumBonds());
    CHECK(copy.getCharges().data() != psf.getCharges().data());
    CHECK(copy.getAtomNames().data() != psf.getAtomNames().data());

    copy.getCharges()[0] = 9.0;
    copy.getAtomNames()[0] = "XX";

    CHECK(psf.getCharges()[0] == Approx(-0.3));
    CHECK(psf.getAtomNames()[0] == "C1");
    CHECK(copy.getCharges()[0] == Approx(9.0));
    CHECK(copy.getAtomNames()[0] == "XX");
    CHECK(copy.getNetCharge() == Approx(9.3));
  }

  SECTION("MoveConstructorTransfersStorage") {
    CharmmPSF source(fileName);

    double *const chargesData = source.getCharges().data();
    int2 *const residuesHostData = source.getResidues().getHostArray().data();
    int2 *const residuesDeviceData =
        source.getResidues().getDeviceArray().data();

    CharmmPSF moved(std::move(source));

    CHECK(moved.getCharges().data() == chargesData);
    CHECK(moved.getResidues().getHostArray().data() == residuesHostData);
    CHECK(moved.getResidues().getDeviceArray().data() == residuesDeviceData);

    CHECK(moved.getFilePath() == fileName);
    CHECK(moved.getNumAtoms() == 4);
    CHECK(moved.getNumBonds() == 3);
    CHECK(moved.getAtomNames() ==
          std::vector<std::string>{"C1", "H1", "C2", "H2"});
    CHECK(moved.getNetCharge() == Approx(0.0).margin(1.0e-12));

    CheckDefaultPsfState(source);
  }

  SECTION("MoveAssignmentTransfersStorage") {
    CharmmPSF source(fileName);

    double *const chargesData = source.getCharges().data();
    int2 *const residuesHostData = source.getResidues().getHostArray().data();
    int2 *const residuesDeviceData =
        source.getResidues().getDeviceArray().data();

    CharmmPSF assigned(fileName);
    assigned = std::move(source);

    CHECK(assigned.getCharges().data() == chargesData);
    CHECK(assigned.getResidues().getHostArray().data() == residuesHostData);
    CHECK(assigned.getResidues().getDeviceArray().data() == residuesDeviceData);

    CHECK(assigned.getFilePath() == fileName);
    CHECK(assigned.getNumAtoms() == 4);
    CHECK(assigned.getNumBonds() == 3);
    CHECK(assigned.getAtomNames() ==
          std::vector<std::string>{"C1", "H1", "C2", "H2"});
    CHECK(assigned.getNetCharge() == Approx(0.0).margin(1.0e-12));

    CheckDefaultPsfState(source);
  }

  SECTION("SelfMoveAssignmentIsNoOp") {
    CharmmPSF psf(fileName);

    double *const chargesData = psf.getCharges().data();
    int2 *const residuesHostData = psf.getResidues().getHostArray().data();
    int2 *const residuesDeviceData = psf.getResidues().getDeviceArray().data();

    psf = std::move(psf);

    CHECK(psf.getCharges().data() == chargesData);
    CHECK(psf.getResidues().getHostArray().data() == residuesHostData);
    CHECK(psf.getResidues().getDeviceArray().data() == residuesDeviceData);
    CHECK(psf.getNumAtoms() == 4);
    CHECK(psf.getNumBonds() == 3);
    CHECK(psf.getNetCharge() == Approx(0.0).margin(1.0e-12));
  }

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("CharmmPSFDetectsWaterMolecules") {
  const std::string fileName = "tmp_charmm_psf_water.psf";
  const std::string psfText = R"PSF(PSF

       1 !NTITLE
 REMARKS generated CharmmPSF water unit test
       3 !NATOM
       1 WAT      1 TIP3 OH2  OT    -0.834000  15.9994           0
       2 WAT      1 TIP3 H1   HT     0.417000   1.0080           0
       3 WAT      1 TIP3 H2   HT     0.417000   1.0080           0
       2 !NBOND: bonds
       1       2       1       3
       1 !NTHETA: angles
       2       1       3
       0 !NPHI: dihedrals
       0 !NIMPHI: impropers
       0 !NDON: donors
       0 !NACC: acceptors
       0 !NCRTERM: cross-terms
)PSF";

  apo_test::WriteTextFile(fileName, psfText);

  CharmmPSF psf(fileName);

  CHECK(psf.getNumAtoms() == 3);
  CHECK(psf.getNumBonds() == 2);
  CHECK(psf.getNumAngles() == 1);
  CHECK(psf.getNetCharge() == Approx(0.0).margin(1.0e-12));

  const std::vector<int4> &waterMolecules =
      psf.getWaterMolecules().getHostArray();
  REQUIRE(waterMolecules.size() == 1);
  CHECK(waterMolecules[0].x == 0);
  CHECK(waterMolecules[0].y == 1);
  CHECK(waterMolecules[0].z == 2);
  CHECK(waterMolecules[0].w == 0);

  const std::vector<int2> &residues = psf.getResidues().getHostArray();
  REQUIRE(residues.size() == 1);
  CHECK(residues[0].x == 0);
  CHECK(residues[0].y == 2);

  const std::vector<int2> &groups = psf.getGroups().getHostArray();
  REQUIRE(groups.size() == 1);
  CHECK(groups[0].x == 0);
  CHECK(groups[0].y == 2);

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("CharmmPSFParserValidationUsesApoCharmmError") {
  SECTION("EmptyFileName") {
    apo_test::CheckApoCharmmError(
        [](void) {
          CharmmPSF psf("");
          static_cast<void>(psf);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "CHARMM PSF file path must not be empty");
  }

  SECTION("MissingFile") {
    const std::string fileName = "tmp_charmm_psf_missing.psf";
    apo_test::RemoveIfExists(fileName);

    apo_test::CheckApoCharmmError(
        [&fileName](void) {
          CharmmPSF psf(fileName);
          static_cast<void>(psf);
        },
        ApoCharmmErrorCode::Runtime,
        "Failed to open file \"" + fileName + "\"");
  }

  SECTION("MissingTitleSection") {
    const std::string fileName = "tmp_charmm_psf_missing_title.psf";
    CheckPsfFileError(fileName, "PSF\n", ApoCharmmErrorCode::Runtime,
                      "Could not find TITLE section in PSF \"" + fileName +
                          "\"");
  }

  SECTION("MissingAtomSection") {
    const std::string fileName = "tmp_charmm_psf_missing_atom.psf";
    CheckPsfFileError(
        fileName, "PSF\n\n       0 !NTITLE\n", ApoCharmmErrorCode::Runtime,
        "Could not find ATOM section in PSF \"" + fileName + "\"");
  }

  SECTION("MissingBondSection") {
    const std::string fileName = "tmp_charmm_psf_missing_bond.psf";
    CheckPsfFileError(
        fileName, SINGLE_ATOM_PSF_PREFIX, ApoCharmmErrorCode::Runtime,
        "Could not find BOND section in PSF \"" + fileName + "\"");
  }

  SECTION("MissingAngleSection") {
    const std::string fileName = "tmp_charmm_psf_missing_angle.psf";
    CheckPsfFileError(fileName, SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION,
                      ApoCharmmErrorCode::Runtime,
                      "Could not find ANGLE section in PSF \"" + fileName +
                          "\"");
  }

  SECTION("MissingDihedralSection") {
    const std::string fileName = "tmp_charmm_psf_missing_dihedral.psf";
    CheckPsfFileError(
        fileName,
        SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION + EMPTY_ANGLE_SECTION,
        ApoCharmmErrorCode::Runtime,
        "Could not find DIHEDRAL section in PSF \"" + fileName + "\"");
  }

  SECTION("MissingImproperSection") {
    const std::string fileName = "tmp_charmm_psf_missing_improper.psf";
    CheckPsfFileError(fileName,
                      SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION +
                          EMPTY_ANGLE_SECTION + EMPTY_DIHEDRAL_SECTION,
                      ApoCharmmErrorCode::Runtime,
                      "Could not find IMPROPER section in PSF \"" + fileName +
                          "\"");
  }

  SECTION("MissingDonorSection") {
    const std::string fileName = "tmp_charmm_psf_missing_donor.psf";
    CheckPsfFileError(
        fileName,
        SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION + EMPTY_ANGLE_SECTION +
            EMPTY_DIHEDRAL_SECTION + EMPTY_IMPROPER_SECTION,
        ApoCharmmErrorCode::Runtime,
        "Could not find DONOR section in PSF \"" + fileName + "\"");
  }

  SECTION("MissingAcceptorSection") {
    const std::string fileName = "tmp_charmm_psf_missing_acceptor.psf";
    CheckPsfFileError(fileName,
                      SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION +
                          EMPTY_ANGLE_SECTION + EMPTY_DIHEDRAL_SECTION +
                          EMPTY_IMPROPER_SECTION + EMPTY_DONOR_SECTION,
                      ApoCharmmErrorCode::Runtime,
                      "Could not find ACCEPTOR section in PSF \"" + fileName +
                          "\"");
  }

  SECTION("MissingCrossTermSection") {
    const std::string fileName = "tmp_charmm_psf_missing_cross_term.psf";
    CheckPsfFileError(
        fileName,
        SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION + EMPTY_ANGLE_SECTION +
            EMPTY_DIHEDRAL_SECTION + EMPTY_IMPROPER_SECTION +
            EMPTY_DONOR_SECTION + EMPTY_ACCEPTOR_SECTION,
        ApoCharmmErrorCode::Runtime,
        "Could not find CROSS-TERM section in PSF \"" + fileName + "\"");
  }

  SECTION("TruncatedTitleRecords") {
    const std::string fileName = "tmp_charmm_psf_truncated_title.psf";
    CheckPsfFileError(
        fileName, "PSF\n\n       1 !NTITLE\n", ApoCharmmErrorCode::Runtime,
        "Unexpected end of file while reading TITLE records in PSF \"" +
            fileName + "\"");
  }

  SECTION("UnsupportedAtomCount") {
    const std::string fileName = "tmp_charmm_psf_atom_count_range.psf";
    CheckPsfFileError(fileName, "PSF\n\n       0 !NTITLE\n2147483648 !NATOM\n",
                      ApoCharmmErrorCode::Runtime,
                      "ATOM count exceeds supported range in PSF \"" +
                          fileName + "\" at line 4");
  }

  SECTION("InvalidAtomRecord") {
    const std::string fileName = "tmp_charmm_psf_invalid_atom_record.psf";
    const std::string contents =
        "PSF\n\n       0 !NTITLE\n       1 !NATOM\n1 SEG1 1 RES1 C1\n";
    CheckPsfFileError(fileName, contents, ApoCharmmErrorCode::Runtime,
                      "Invalid ATOM record in PSF \"" + fileName +
                          "\" at line 5: 1 SEG1 1 RES1 C1");
  }

  SECTION("InvalidAtomCharge") {
    const std::string fileName = "tmp_charmm_psf_invalid_atom_charge.psf";
    const std::string contents = "PSF\n\n       0 !NTITLE\n       1 !NATOM\n"
                                 "1 SEG1 1 RES1 C1 CT1 BAD 12.0110 0\n";
    CheckPsfFileError(fileName, contents, ApoCharmmErrorCode::Runtime,
                      "Invalid charge value \"BAD\" in ATOM section of PSF \"" +
                          fileName + "\" at line 5");
  }

  SECTION("InvalidBondRecord") {
    const std::string fileName = "tmp_charmm_psf_invalid_bond_record.psf";
    CheckPsfFileError(
        fileName, SINGLE_ATOM_PSF_PREFIX + "       1 !NBOND: bonds\n1\n",
        ApoCharmmErrorCode::Runtime,
        "Invalid BOND record in PSF \"" + fileName + "\" at line 7: 1");
  }

  SECTION("BondAtomIndexOutOfRange") {
    const std::string fileName = "tmp_charmm_psf_bond_index_range.psf";
    CheckPsfFileError(fileName,
                      SINGLE_ATOM_PSF_PREFIX + "       1 !NBOND: bonds\n1 2\n",
                      ApoCharmmErrorCode::Runtime,
                      "BOND atom index \"2\" is out of range in PSF \"" +
                          fileName + "\" at line 7");
  }

  SECTION("InvalidAngleRecord") {
    const std::string fileName = "tmp_charmm_psf_invalid_angle_record.psf";
    CheckPsfFileError(fileName,
                      SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION +
                          "       1 !NTHETA: angles\n1 1\n",
                      ApoCharmmErrorCode::Runtime,
                      "Invalid ANGLE record in PSF \"" + fileName +
                          "\" at line 8: 1 1");
  }

  SECTION("InvalidDihedralRecord") {
    const std::string fileName = "tmp_charmm_psf_invalid_dihedral_record.psf";
    CheckPsfFileError(
        fileName,
        SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION + EMPTY_ANGLE_SECTION +
            "       1 !NPHI: dihedrals\n1 1 1\n",
        ApoCharmmErrorCode::Runtime,
        "Invalid DIHEDRAL record in PSF \"" + fileName + "\" at line 9: 1 1 1");
  }

  SECTION("InvalidImproperRecord") {
    const std::string fileName = "tmp_charmm_psf_invalid_improper_record.psf";
    CheckPsfFileError(fileName,
                      SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION +
                          EMPTY_ANGLE_SECTION + EMPTY_DIHEDRAL_SECTION +
                          "       1 !NIMPHI: impropers\n1 1 1\n",
                      ApoCharmmErrorCode::Runtime,
                      "Invalid IMPROPER record in PSF \"" + fileName +
                          "\" at line 10: 1 1 1");
  }

  SECTION("InvalidCrossTermRecord") {
    const std::string fileName = "tmp_charmm_psf_invalid_cross_term_record.psf";
    CheckPsfFileError(fileName,
                      SINGLE_ATOM_PSF_PREFIX + EMPTY_BOND_SECTION +
                          EMPTY_ANGLE_SECTION + EMPTY_DIHEDRAL_SECTION +
                          EMPTY_IMPROPER_SECTION + EMPTY_DONOR_SECTION +
                          EMPTY_ACCEPTOR_SECTION +
                          "       1 !NCRTERM: cross-terms\n1 1 1 1\n",
                      ApoCharmmErrorCode::Runtime,
                      "Invalid CROSS-TERM record in PSF \"" + fileName +
                          "\" at line 13: 1 1 1 1");
  }
}

TEST_CASE("CharmmPSFSetterValidationUsesApoCharmmError") {
  SECTION("NegativeAtomCount") {
    CharmmPSF psf;
    apo_test::CheckApoCharmmError(
        [&psf](void) { psf.setNumAtoms(-1); },
        ApoCharmmErrorCode::InvalidArgument,
        "Number of atoms must be nonnegative; observed -1");
  }

  SECTION("ChargesBeforeAtomCount") {
    CharmmPSF psf;
    apo_test::CheckApoCharmmError([&psf](void) { psf.setAtomCharges({}); },
                                  ApoCharmmErrorCode::NotInitialized,
                                  "CharmmPSF atom count is not initialized");
  }

  SECTION("ChargeCountMismatch") {
    CharmmPSF psf;
    psf.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&psf](void) { psf.setAtomCharges({0.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "CharmmPSF charge count must match atom count; expected 2, observed 1");
  }
}

TEST_CASE("CharmmPSFAggregateValidationUsesApoCharmmError") {
  SECTION("NetChargeBeforeInitialization") {
    CharmmPSF psf;
    apo_test::CheckApoCharmmError(
        [&psf](void) { static_cast<void>(psf.getNetCharge()); },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmPSF atom count is not initialized");
  }

  SECTION("TotalMassBeforeInitialization") {
    CharmmPSF psf;
    apo_test::CheckApoCharmmError(
        [&psf](void) { static_cast<void>(psf.getTotalMass()); },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmPSF atom count is not initialized");
  }

  SECTION("InclusionExclusionBeforeInitialization") {
    CharmmPSF psf;
    apo_test::CheckApoCharmmError(
        [&psf](void) { static_cast<void>(psf.getInclusionExclusionLists()); },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmPSF atom count is not initialized");
  }

  SECTION("ChargeCountMismatch") {
    CharmmPSF psf;
    psf.setNumAtoms(2);
    psf.getCharges().resize(1);
    apo_test::CheckApoCharmmError(
        [&psf](void) { static_cast<void>(psf.getNetCharge()); },
        ApoCharmmErrorCode::Runtime,
        "CharmmPSF charge count does not match atom count; expected 2, "
        "observed 1");
  }

  SECTION("MassCountMismatch") {
    CharmmPSF psf;
    psf.setNumAtoms(2);
    psf.getMasses().resize(1);
    apo_test::CheckApoCharmmError(
        [&psf](void) { static_cast<void>(psf.getTotalMass()); },
        ApoCharmmErrorCode::Runtime,
        "CharmmPSF mass count does not match atom count; expected 2, observed "
        "1");
  }

  SECTION("ConnectivityCountMismatch") {
    CharmmPSF psf;
    psf.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&psf](void) { static_cast<void>(psf.getInclusionExclusionLists()); },
        ApoCharmmErrorCode::Runtime,
        "CharmmPSF connectivity list sizes do not match atom count; expected "
        "2, observed 0, 0, 0");
  }
}
