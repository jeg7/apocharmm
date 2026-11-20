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
#include "CharmmParameters.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <filesystem>
#include <type_traits>

static_assert(std::is_nothrow_destructible_v<CharmmParameters>);

namespace {

constexpr double DOUBLE_TOLERANCE = 1.0e-12;
constexpr double FLOAT_TOLERANCE = 1.0e-6;

const std::string TEST_PARAMETER_TEXT =
    R"PRM(* generated CharmmParameters unit test file
*

BONDS
A      B      1.0D+02   1.1
B      C      200.0     1.2
C      D      300.0     1.3

ANGLES
A      B      C       50.0    120.0    10.0    2.5
B      C      D       60.0    110.0

DIHEDRALS
A      B      C      D        0.5      3       180.0
X      B      C      X        0.2      2         0.0

IMPROPER
A      B      C      D        1.5      0       180.0

NONBONDED NBXMOD 5 ATOM CDIEL FSHIFT VATOM VDISTANCE VFSWITCH -
CUTNB 14.0 CTOFNB 12.0 CTONNB 10.0 EPS 1.0 E14FAC 1.0 WMIN 1.5
A      0.0    -0.1     1.0
B      0.0    -0.2     1.1
C      0.0    -0.3     1.2
D      0.0    -0.4     1.3
E      0.0    -0.5     1.4      0.0    -0.05    1.0

END

READ PARA CARD APPEND
NBFIX
A      D      -0.05    2.5

END
)PRM";

const std::string TEST_PARAMETER_SUPPLEMENT_TEXT =
    R"PRM(* generated supplemental CharmmParameters unit test file
*

BONDS
E      F      400.0     1.4

NONBONDED
F      0.0    -0.6      1.5

END
)PRM";

const std::string MISSING_BOND_PARAMETER_TEXT =
    R"PRM(* generated incomplete CharmmParameters unit test file
*

BONDS
A      B      100.0     1.1
B      C      200.0     1.2

END
)PRM";

const std::string MISSING_ANGLE_PARAMETER_TEXT =
    R"PRM(BONDS
A B 100.0 1.1
B C 200.0 1.2
C D 300.0 1.3

END
)PRM";

const std::string MISSING_DIHEDRAL_PARAMETER_TEXT =
    R"PRM(BONDS
A B 100.0 1.1
B C 200.0 1.2
C D 300.0 1.3

ANGLES
A B C 50.0 120.0
B C D 60.0 110.0

END
)PRM";

const std::string MISSING_IMPROPER_PARAMETER_TEXT =
    R"PRM(BONDS
A B 100.0 1.1
B C 200.0 1.2
C D 300.0 1.3

ANGLES
A B C 50.0 120.0
B C D 60.0 110.0

DIHEDRALS
A B C D 0.5 3 180.0

END
)PRM";

const std::string MISSING_VDW_PARAMETER_TEXT =
    R"PRM(NONBONDED
A 0.0 -0.1 1.0
B 0.0 -0.2 1.1
C 0.0 -0.3 1.2

END
)PRM";

const std::string MALFORMED_PARAMETER_TEXT = "NONBONDED\n"
                                             "A 0.0 -0.1 1.0 2.0\n"
                                             "END\n";

const std::string TEST_PSF_TEXT = R"PSF(PSF

       1 !NTITLE
 REMARKS generated CharmmParameters unit test PSF
       4 !NATOM
       1 SEG1     1 RES1 A1   A      0.000000  12.0110           0
       2 SEG1     1 RES1 B1   B      0.000000  12.0110           0
       3 SEG1     1 RES1 C1   C      0.000000  12.0110           0
       4 SEG1     1 RES1 D1   D      0.000000  12.0110           0
       3 !NBOND: bonds
       1       2       2       3       3       4
       2 !NTHETA: angles
       1       2       3       2       3       4
       1 !NPHI: dihedrals
       1       2       3       4
       1 !NIMPHI: impropers
       1       2       3       4
       0 !NDON: donors
       0 !NACC: acceptors
       0 !NCRTERM: cross-terms
)PSF";

// TEST_PSF_TEXT plus one cross term whose second torsion is the first one read
// backward, and the same cross term with both torsions listed the other way.
const std::string TEST_CMAP_PSF_TEXT =
    TEST_PSF_TEXT.substr(0, TEST_PSF_TEXT.find("       0 !NCRTERM")) +
    "       1 !NCRTERM: cross-terms\n"
    "       1       2       3       4       4       3       2       1\n";

const std::string TEST_REVERSED_CMAP_PSF_TEXT =
    TEST_PSF_TEXT.substr(0, TEST_PSF_TEXT.find("       0 !NCRTERM")) +
    "       1 !NCRTERM: cross-terms\n"
    "       4       3       2       1       1       2       3       4\n";

// TEST_PARAMETER_TEXT plus one map for the torsions A-B-C-D and D-C-B-A whose
// energies are 0.25 * phiIndex - 0.5 * psiIndex, written phi-major as in
// CHARMM parameter files.
const std::string TEST_CMAP_PARAMETER_TEXT = [](void) {
  std::string text = TEST_PARAMETER_TEXT + "CMAP\nA B C D D C B A 24\n";
  for (int i = 0; i < CmapValues::gridSize; i++) {
    for (int j = 0; j < CmapValues::gridSize; j++)
      text += std::to_string(0.25 * i - 0.5 * j) + " ";
    text += "\n";
  }
  return text + "END\n";
}();

void CheckParameterFileError(const std::string &fileName,
                             const std::string &contents,
                             const ApoCharmmErrorCode expectedCode,
                             const std::string &expectedMessage) {
  apo_test::RemoveIfExists(fileName);
  apo_test::WriteTextFile(fileName, contents);

  apo_test::CheckApoCharmmError(
      [&fileName](void) {
        CharmmParameters parameters(fileName);
        static_cast<void>(parameters);
      },
      expectedCode, expectedMessage);

  apo_test::RemoveIfExists(fileName);
  return;
}

void CheckBondedParameterError(const std::string &parameterContents,
                               const std::string &psfContents,
                               const ApoCharmmErrorCode expectedCode,
                               const std::string &expectedMessage) {
  const std::string parameterFile = "tmp_charmm_parameters_bonded_error.prm";
  const std::string psfFile = "tmp_charmm_parameters_bonded_error.psf";

  apo_test::RemoveIfExists(parameterFile);
  apo_test::RemoveIfExists(psfFile);
  apo_test::WriteTextFile(parameterFile, parameterContents);
  apo_test::WriteTextFile(psfFile, psfContents);

  CharmmParameters parameters(parameterFile);
  const std::shared_ptr<CharmmPSF> psf = std::make_shared<CharmmPSF>(psfFile);

  apo_test::CheckApoCharmmError(
      [&parameters, &psf](void) {
        static_cast<void>(parameters.getBondedParamsAndLists(psf));
      },
      expectedCode, expectedMessage);

  apo_test::RemoveIfExists(parameterFile);
  apo_test::RemoveIfExists(psfFile);
  return;
}

} // namespace

TEST_CASE("CharmmParametersParsesSingleFile") {
  const std::string parameterFile = "tmp_charmm_parameters_full.prm";
  apo_test::WriteTextFile(parameterFile, TEST_PARAMETER_TEXT);

  CharmmParameters parameters(parameterFile);

  SECTION("OriginalFilePath") {
    CHECK(parameters.getPrmFilePaths() ==
          std::vector<std::filesystem::path>{parameterFile});
  }

  SECTION("BondParameters") {
    const std::map<BondKey, BondValues> bonds = parameters.getBondParams();

    REQUIRE(bonds.size() == 3);
    REQUIRE(bonds.count(BondKey("A", "B")) == 1);
    REQUIRE(bonds.count(BondKey("B", "C")) == 1);
    REQUIRE(bonds.count(BondKey("C", "D")) == 1);

    CHECK(bonds.at(BondKey("A", "B")).kb == Approx(100.0f));
    CHECK(bonds.at(BondKey("A", "B")).b0 == Approx(1.1f));

    CHECK(bonds.at(BondKey("B", "C")).kb == Approx(200.0f));
    CHECK(bonds.at(BondKey("B", "C")).b0 == Approx(1.2f));

    CHECK(bonds.at(BondKey("C", "D")).kb == Approx(300.0f));
    CHECK(bonds.at(BondKey("C", "D")).b0 == Approx(1.3f));
  }

  SECTION("AngleAndUreyBradleyParameters") {
    const std::map<AngleKey, AngleValues> angles = parameters.getAngleParams();
    const std::map<AngleKey, BondValues> ureyBradleys =
        parameters.getUreybParams();

    REQUIRE(angles.size() == 2);
    REQUIRE(ureyBradleys.size() == 2);

    REQUIRE(angles.count(AngleKey("A", "B", "C")) == 1);
    CHECK(angles.at(AngleKey("A", "B", "C")).kTheta == Approx(50.0f));
    CHECK(angles.at(AngleKey("A", "B", "C")).theta0 ==
          Approx(static_cast<float>(apo_test::Deg2Rad(120.0)))
              .margin(FLOAT_TOLERANCE));

    REQUIRE(angles.count(AngleKey("B", "C", "D")) == 1);
    CHECK(angles.at(AngleKey("B", "C", "D")).kTheta == Approx(60.0f));
    CHECK(angles.at(AngleKey("B", "C", "D")).theta0 ==
          Approx(static_cast<float>(apo_test::Deg2Rad(110.0)))
              .margin(FLOAT_TOLERANCE));

    CHECK(ureyBradleys.at(AngleKey("A", "B", "C")).kb == Approx(10.0f));
    CHECK(ureyBradleys.at(AngleKey("A", "B", "C")).b0 == Approx(2.5f));

    CHECK(ureyBradleys.at(AngleKey("B", "C", "D")).kb == Approx(0.0f));
    CHECK(ureyBradleys.at(AngleKey("B", "C", "D")).b0 == Approx(0.0f));
  }

  SECTION("DihedralAndImproperParameters") {
    const std::map<DihedralKey, std::vector<DihedralValues>> dihedrals =
        parameters.getDihedralParams();
    const std::map<DihedralKey, ImDihedralValues> impropers =
        parameters.getImproperParams();

    REQUIRE(dihedrals.count(DihedralKey("A", "B", "C", "D")) == 1);
    REQUIRE(dihedrals.at(DihedralKey("A", "B", "C", "D")).size() == 1);
    CHECK(dihedrals.at(DihedralKey("A", "B", "C", "D"))[0].kChi ==
          Approx(0.5f));
    CHECK(dihedrals.at(DihedralKey("A", "B", "C", "D"))[0].n == 3);
    CHECK(dihedrals.at(DihedralKey("A", "B", "C", "D"))[0].delta ==
          Approx(180.0f));

    REQUIRE(dihedrals.count(DihedralKey("X", "B", "C", "X")) == 1);
    REQUIRE(dihedrals.at(DihedralKey("X", "B", "C", "X")).size() == 1);
    CHECK(dihedrals.at(DihedralKey("X", "B", "C", "X"))[0].kChi ==
          Approx(0.2f));
    CHECK(dihedrals.at(DihedralKey("X", "B", "C", "X"))[0].n == 2);
    CHECK(dihedrals.at(DihedralKey("X", "B", "C", "X"))[0].delta ==
          Approx(0.0f));

    REQUIRE(impropers.count(DihedralKey("A", "B", "C", "D")) == 1);
    CHECK(impropers.at(DihedralKey("A", "B", "C", "D")).kPsi == Approx(1.5f));
    CHECK(impropers.at(DihedralKey("A", "B", "C", "D")).psi0 == Approx(180.0f));
  }

  SECTION("VdwParameters") {
    const std::map<std::string, VdwParameters> vdw = parameters.getVdwParams();
    const std::map<std::string, VdwParameters> vdw14 =
        parameters.getVdw14Params();

    REQUIRE(vdw.size() == 5);
    REQUIRE(vdw.count("A") == 1);
    REQUIRE(vdw.count("B") == 1);
    REQUIRE(vdw.count("C") == 1);
    REQUIRE(vdw.count("D") == 1);
    REQUIRE(vdw.count("E") == 1);

    CHECK(vdw.at("A").epsilon == Approx(-0.1).margin(DOUBLE_TOLERANCE));
    CHECK(vdw.at("A").rmin_2 == Approx(1.0).margin(DOUBLE_TOLERANCE));
    CHECK(vdw.at("D").epsilon == Approx(-0.4).margin(DOUBLE_TOLERANCE));
    CHECK(vdw.at("D").rmin_2 == Approx(1.3).margin(DOUBLE_TOLERANCE));

    REQUIRE(vdw14.size() == 1);
    REQUIRE(vdw14.count("E") == 1);
    CHECK(vdw14.at("E").epsilon == Approx(-0.05).margin(DOUBLE_TOLERANCE));
    CHECK(vdw14.at("E").rmin_2 == Approx(1.0).margin(DOUBLE_TOLERANCE));
  }

  apo_test::RemoveIfExists(parameterFile);
}

TEST_CASE("CharmmParametersParsesMultipleFiles") {
  const std::string parameterFile = "tmp_charmm_parameters_full.prm";
  const std::string supplementalFile = "tmp_charmm_parameters_supplement.prm";
  apo_test::WriteTextFile(parameterFile, TEST_PARAMETER_TEXT);
  apo_test::WriteTextFile(supplementalFile, TEST_PARAMETER_SUPPLEMENT_TEXT);

  CharmmParameters parameters({parameterFile, supplementalFile});

  CHECK(parameters.getPrmFilePaths() ==
        std::vector<std::filesystem::path>{parameterFile, supplementalFile});

  const std::map<BondKey, BondValues> bonds = parameters.getBondParams();
  REQUIRE(bonds.count(BondKey("A", "B")) == 1);
  REQUIRE(bonds.count(BondKey("E", "F")) == 1);

  CHECK(bonds.at(BondKey("E", "F")).kb == Approx(400.0f));
  CHECK(bonds.at(BondKey("E", "F")).b0 == Approx(1.4f));

  const std::map<std::string, VdwParameters> vdw = parameters.getVdwParams();
  REQUIRE(vdw.count("F") == 1);
  CHECK(vdw.at("F").epsilon == Approx(-0.6).margin(DOUBLE_TOLERANCE));
  CHECK(vdw.at("F").rmin_2 == Approx(1.5).margin(DOUBLE_TOLERANCE));

  apo_test::RemoveIfExists(parameterFile);
  apo_test::RemoveIfExists(supplementalFile);
}

TEST_CASE("CharmmParametersBuildsBondedParamsAndLists") {
  const std::string parameterFile = "tmp_charmm_parameters_full.prm";
  const std::string psfFile = "tmp_charmm_parameters_test.psf";
  apo_test::WriteTextFile(parameterFile, TEST_PARAMETER_TEXT);
  apo_test::WriteTextFile(psfFile, TEST_PSF_TEXT);

  CharmmParameters parameters(parameterFile);
  std::shared_ptr<CharmmPSF> psf = std::make_shared<CharmmPSF>(psfFile);

  const BondedParamsAndLists bonded = parameters.getBondedParamsAndLists(psf);

  CHECK(bonded.paramsSize == std::vector<int>{3, 1, 2, 1, 1, 0});
  CHECK(bonded.listsSize == std::vector<int>{3, 1, 2, 1, 1, 0});

  const std::vector<std::vector<float>> expectedParams = {
      {1.1f, 100.0f},
      {1.2f, 200.0f},
      {1.3f, 300.0f},
      {2.5f, 10.0f},
      {static_cast<float>(apo_test::Deg2Rad(120.0)), 50.0f},
      {static_cast<float>(apo_test::Deg2Rad(110.0)), 60.0f},
      {3.0f, 0.5f, 0.0f, -1.0f},
      {180.0f, 1.5f, 0.0f, 1.0f},
  };

  apo_test::CheckVectorsClose2D<float>(bonded.paramsVal, expectedParams,
                                       FLOAT_TOLERANCE);

  const std::vector<std::vector<int>> expectedLists = {
      {0, 1, 0, 13},
      {1, 2, 1, 13},
      {2, 3, 2, 13},
      {0, 2, 0, 13},
      {0, 1, 2, 0, 13, 13},
      {1, 2, 3, 1, 13, 13},
      {0, 1, 2, 3, 0, 13, 13, 13},
      {0, 1, 2, 3, 0, 13, 13, 13},
  };

  CHECK(bonded.listVal == expectedLists);

  apo_test::RemoveIfExists(parameterFile);
  apo_test::RemoveIfExists(psfFile);
}

TEST_CASE("CharmmParametersParsesAndPacksCmapRecords") {
  const std::string parameterFile = "tmp_charmm_parameters_cmap.prm";
  const std::string psfFile = "tmp_charmm_parameters_cmap.psf";
  apo_test::WriteTextFile(parameterFile, TEST_CMAP_PARAMETER_TEXT);
  apo_test::WriteTextFile(psfFile, TEST_CMAP_PSF_TEXT);

  CharmmParameters parameters(parameterFile);
  std::shared_ptr<CharmmPSF> psf = std::make_shared<CharmmPSF>(psfFile);

  const CmapKey key(DihedralKey("A", "B", "C", "D"),
                    DihedralKey("D", "C", "B", "A"));
  const std::map<CmapKey, CmapValues> &cmaps = parameters.getCmapParams();

  REQUIRE(cmaps.size() == 1);
  REQUIRE(cmaps.count(key) == 1);
  REQUIRE(cmaps.at(key).values.size() == CmapValues::numValues);
  REQUIRE(cmaps.at(key).coeff.size() == CmapValues::numCoefficients);
  CHECK(cmaps.at(key).values[1 * CmapValues::gridSize + 2] ==
        Approx(-0.75).margin(DOUBLE_TOLERANCE));

  const BondedParamsAndLists bonded = parameters.getBondedParamsAndLists(psf);

  CHECK(bonded.paramsSize == std::vector<int>{3, 1, 2, 1, 1, 9216});
  CHECK(bonded.listsSize == std::vector<int>{3, 1, 2, 1, 1, 1});
  CHECK(bonded.listVal.back() ==
        std::vector<int>{0, 1, 2, 3, 3, 2, 1, 0, 0, 13, 13, 13, 13, 13, 13});

  // Within cell (1, 2), t runs along phi and u along psi. The bicubic patch
  // reproduces its four corner energies: E(0, 0) is coefficient 0, E(1, 0)
  // sums coefficients 4 * m, E(0, 1) sums coefficients n, and E(1, 1) sums
  // all 16.
  const std::size_t cellStart =
      8 + (1 * CmapValues::gridSize + 2) * CmapValues::coefficientsPerCell;
  REQUIRE(bonded.paramsVal.size() >= cellStart + 16);

  double phiEdge = 0.0;
  double psiEdge = 0.0;
  double corner = 0.0;
  for (std::size_t m = 0; m < 4; m++) {
    for (std::size_t n = 0; n < 4; n++) {
      const double coefficient = bonded.paramsVal[cellStart + 4 * m + n][0];

      if (n == 0)
        phiEdge += coefficient;
      if (m == 0)
        psiEdge += coefficient;
      corner += coefficient;
    }
  }

  CHECK(bonded.paramsVal[cellStart][0] ==
        Approx(-0.75).margin(FLOAT_TOLERANCE));
  CHECK(phiEdge == Approx(-0.5).margin(1.0e-5));
  CHECK(psiEdge == Approx(-1.25).margin(1.0e-5));
  CHECK(corner == Approx(-1.0).margin(1.0e-5));

  apo_test::RemoveIfExists(parameterFile);
  apo_test::RemoveIfExists(psfFile);
}

TEST_CASE("CharmmParametersBuildsVdwParamsAndTypes") {
  const std::string parameterFile = "tmp_charmm_parameters_full.prm";
  const std::string psfFile = "tmp_charmm_parameters_test.psf";
  apo_test::WriteTextFile(parameterFile, TEST_PARAMETER_TEXT);
  apo_test::WriteTextFile(psfFile, TEST_PSF_TEXT);

  CharmmParameters parameters(parameterFile);
  std::shared_ptr<CharmmPSF> psf = std::make_shared<CharmmPSF>(psfFile);

  const VdwParamsAndTypes vdw = parameters.getVdwParamsAndTypes(psf);

  CHECK(vdw.vdwTypes == std::vector<int>{0, 1, 2, 3});
  CHECK(vdw.vdw14Types == std::vector<int>{0, 1, 2, 3});

  // Four unique atom types produce a lower-triangular 4x4 packed matrix.
  REQUIRE(vdw.vdwParams.size() == 20);
  REQUIRE(vdw.vdw14Params.size() == 20);

  const double aaEpsilon = std::sqrt((-0.1) * (-0.1));
  const double aaRmin = 1.0 + 1.0;
  const float expectedAAC6 =
      static_cast<float>(2.0 * aaEpsilon * std::pow(aaRmin, 6));
  const float expectedAAC12 =
      static_cast<float>(aaEpsilon * std::pow(aaRmin, 12));

  CHECK(vdw.vdwParams[0] == Approx(expectedAAC6).margin(FLOAT_TOLERANCE));
  CHECK(vdw.vdwParams[1] == Approx(expectedAAC12).margin(FLOAT_TOLERANCE));

  // Pair A-D is packed at triangular pair index 6: (A,A), (B,A), (B,B),
  // (C,A), (C,B), (C,C), (D,A). It should use the NBFIX record.
  const std::size_t adPairIndex = 6;
  const double adEpsilon = 0.05;
  const double adRmin = 2.5;
  const float expectedADC6 =
      static_cast<float>(2.0 * adEpsilon * std::pow(adRmin, 6));
  const float expectedADC12 =
      static_cast<float>(adEpsilon * std::pow(adRmin, 12));

  CHECK(vdw.vdwParams[2 * adPairIndex + 0] ==
        Approx(expectedADC6).margin(FLOAT_TOLERANCE));
  CHECK(vdw.vdwParams[2 * adPairIndex + 1] ==
        Approx(expectedADC12).margin(FLOAT_TOLERANCE));

  // No A-D 1-4-specific NBFIX values were supplied, so the current
  // implementation uses the same NBFIX values for vdw14Params.
  CHECK(vdw.vdw14Params[2 * adPairIndex + 0] ==
        Approx(expectedADC6).margin(FLOAT_TOLERANCE));
  CHECK(vdw.vdw14Params[2 * adPairIndex + 1] ==
        Approx(expectedADC12).margin(FLOAT_TOLERANCE));

  apo_test::RemoveIfExists(parameterFile);
  apo_test::RemoveIfExists(psfFile);
}

TEST_CASE("CharmmParametersInputValidationUsesApoCharmmError") {
  SECTION("EmptyPath") {
    apo_test::CheckApoCharmmError(
        [](void) {
          CharmmParameters parameters("");
          static_cast<void>(parameters);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "CHARMM parameter file path must not be empty");
  }

  SECTION("EmptyFileList") {
    apo_test::CheckApoCharmmError(
        [](void) {
          CharmmParameters parameters{std::vector<std::filesystem::path>{}};
          static_cast<void>(parameters);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "At least one CHARMM parameter file is required");
  }

  SECTION("MissingFile") {
    const std::string parameterFile = "tmp_charmm_parameters_missing.prm";
    apo_test::RemoveIfExists(parameterFile);

    apo_test::CheckApoCharmmError(
        [&parameterFile](void) {
          CharmmParameters parameters(parameterFile);
          static_cast<void>(parameters);
        },
        ApoCharmmErrorCode::Runtime,
        "Failed to open CHARMM parameter file \"" + parameterFile + "\"");
  }
}

TEST_CASE("CharmmParametersPsfValidationUsesApoCharmmError") {
  const std::string parameterFile = "tmp_charmm_parameters_full.prm";
  apo_test::WriteTextFile(parameterFile, TEST_PARAMETER_TEXT);

  CharmmParameters parameters(parameterFile);
  std::shared_ptr<CharmmPSF> psf;

  SECTION("NullBondedPsf") {
    apo_test::CheckApoCharmmError(
        [&parameters, &psf](void) {
          static_cast<void>(parameters.getBondedParamsAndLists(psf));
        },
        ApoCharmmErrorCode::InvalidArgument, "CharmmPSF must not be null");
  }

  SECTION("NullVdwPsf") {
    apo_test::CheckApoCharmmError(
        [&parameters, &psf](void) {
          static_cast<void>(parameters.getVdwParamsAndTypes(psf));
        },
        ApoCharmmErrorCode::InvalidArgument, "CharmmPSF must not be null");
  }

  apo_test::RemoveIfExists(parameterFile);
}

TEST_CASE("CharmmParametersParserValidationUsesApoCharmmError") {
  SECTION("MissingTopparParameterBlock") {
    const std::string parameterFile =
        "tmp_charmm_parameters_toppar_missing_block.str";

    CheckParameterFileError(parameterFile, "BONDS\nA B 100.0 1.0\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "CHARMM parameter block was not found in file \"" +
                                parameterFile + "\"");
  }

  SECTION("TruncatedNonbondedHeader") {
    const std::string parameterFile =
        "tmp_charmm_parameters_nonbonded_header.prm";

    CheckParameterFileError(
        parameterFile, "NONBONDED -\n", ApoCharmmErrorCode::Runtime,
        "Unexpected end of file while reading the NONBONDED header in file \"" +
            parameterFile + "\" beginning at line 1");
  }

#if defined(__linux__)
  SECTION("ReadFailure") {
    apo_test::CheckApoCharmmError(
        [](void) {
          CharmmParameters parameters(".");
          static_cast<void>(parameters);
        },
        ApoCharmmErrorCode::Runtime,
        "Failed while reading CHARMM parameter file \".\"");
  }
#endif

  SECTION("InvalidBondRecord") {
    const std::string parameterFile = "tmp_charmm_parameters_invalid_bond.prm";

    CheckParameterFileError(parameterFile, "BONDS\nA B 100.0\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Invalid BONDS parameter record in file \"" +
                                parameterFile + "\" at line 2: A B 100.0");
  }

  SECTION("InvalidAngleRecord") {
    const std::string parameterFile = "tmp_charmm_parameters_invalid_angle.prm";

    CheckParameterFileError(parameterFile, "ANGLES\nA B C 50.0\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Invalid ANGLES parameter record in file \"" +
                                parameterFile + "\" at line 2: A B C 50.0");
  }

  SECTION("InvalidDihedralRecord") {
    const std::string parameterFile =
        "tmp_charmm_parameters_invalid_dihedral.prm";

    CheckParameterFileError(parameterFile, "DIHEDRALS\nA B C D 0.5 3\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Invalid DIHEDRALS parameter record in file \"" +
                                parameterFile + "\" at line 2: A B C D 0.5 3");
  }

  SECTION("InvalidImproperRecord") {
    const std::string parameterFile =
        "tmp_charmm_parameters_invalid_improper.prm";

    CheckParameterFileError(parameterFile, "IMPROPER\nA B C D 1.5 0\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Invalid IMPROPER parameter record in file \"" +
                                parameterFile + "\" at line 2: A B C D 1.5 0");
  }

  SECTION("InvalidNonbondedRecord") {
    const std::string parameterFile =
        "tmp_charmm_parameters_invalid_nonbonded.prm";

    CheckParameterFileError(
        parameterFile, MALFORMED_PARAMETER_TEXT, ApoCharmmErrorCode::Runtime,
        "Invalid NONBONDED parameter record in file \"" + parameterFile +
            "\" at line 2: A 0.0 -0.1 1.0 2.0");
  }

  SECTION("InvalidNbfixRecord") {
    const std::string parameterFile = "tmp_charmm_parameters_invalid_nbfix.prm";

    CheckParameterFileError(parameterFile, "NBFIX\nA B -0.05 2.5 0.0\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Invalid NBFIX parameter record in file \"" +
                                parameterFile +
                                "\" at line 2: A B -0.05 2.5 0.0");
  }

  SECTION("InvalidCmapRecord") {
    const std::string parameterFile = "tmp_charmm_parameters_invalid_cmap.prm";

    CheckParameterFileError(parameterFile, "CMAP\nA B C D B C D E\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Invalid CMAP parameter record in file \"" +
                                parameterFile +
                                "\" at line 2: A B C D B C D E");
  }

  SECTION("UnsupportedCmapGridSize") {
    const std::string parameterFile = "tmp_charmm_parameters_cmap_grid.prm";

    CheckParameterFileError(parameterFile, "CMAP\nA B C D B C D E 12\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Unsupported CMAP grid size in file \"" +
                                parameterFile +
                                "\" at line 2: expected 24, observed 12");
  }

  SECTION("TruncatedCmapGrid") {
    const std::string parameterFile =
        "tmp_charmm_parameters_cmap_truncated.prm";

    CheckParameterFileError(
        parameterFile, "CMAP\nA B C D B C D E 24\n0.0 0.0\n",
        ApoCharmmErrorCode::Runtime,
        "Unexpected end of file while reading CMAP parameter in file \"" +
            parameterFile + "\" after line 3");
  }

  SECTION("InvalidCmapEnergy") {
    const std::string parameterFile = "tmp_charmm_parameters_cmap_energy.prm";

    CheckParameterFileError(parameterFile,
                            "CMAP\nA B C D B C D E 24\n0.0 BAD\nEND\n",
                            ApoCharmmErrorCode::Runtime,
                            "Invalid grid energy value \"BAD\" in CMAP "
                            "parameter record in file \"" +
                                parameterFile + "\" at line 3: 0.0 BAD");
  }

  SECTION("InvalidFloatingPointValue") {
    const std::string parameterFile = "tmp_charmm_parameters_invalid_float.prm";

    CheckParameterFileError(
        parameterFile, "BONDS\nA B BAD 1.0\nEND\n", ApoCharmmErrorCode::Runtime,
        "Invalid kb value \"BAD\" in BONDS parameter record in file \"" +
            parameterFile + "\" at line 2: A B BAD 1.0");
  }

  SECTION("FloatingPointValueWithTrailingCharacters") {
    const std::string parameterFile =
        "tmp_charmm_parameters_trailing_float.prm";

    CheckParameterFileError(
        parameterFile, "BONDS\nA B 100.0BAD 1.0\nEND\n",
        ApoCharmmErrorCode::Runtime,
        "Invalid kb value \"100.0BAD\" in BONDS parameter record in file \"" +
            parameterFile + "\" at line 2: A B 100.0BAD 1.0");
  }

  SECTION("OutOfRangeFloatingPointValue") {
    const std::string parameterFile =
        "tmp_charmm_parameters_out_of_range_float.prm";

    CheckParameterFileError(
        parameterFile, "BONDS\nA B 1E9999 1.0\nEND\n",
        ApoCharmmErrorCode::Runtime,
        "Invalid kb value \"1E9999\" in BONDS parameter record in file \"" +
            parameterFile + "\" at line 2: A B 1E9999 1.0");
  }

  SECTION("NonfiniteFloatingPointValue") {
    const std::string parameterFile =
        "tmp_charmm_parameters_nonfinite_float.prm";

    CheckParameterFileError(
        parameterFile, "BONDS\nA B NAN 1.0\nEND\n", ApoCharmmErrorCode::Runtime,
        "Invalid kb value \"NAN\" in BONDS parameter record in file \"" +
            parameterFile + "\" at line 2: A B NAN 1.0");
  }

  SECTION("InvalidIntegerValue") {
    const std::string parameterFile =
        "tmp_charmm_parameters_invalid_integer.prm";

    CheckParameterFileError(
        parameterFile, "DIHEDRALS\nA B C D 0.5 BAD 180.0\nEND\n",
        ApoCharmmErrorCode::Runtime,
        "Invalid multiplicity value \"BAD\" in DIHEDRALS parameter record in "
        "file \"" +
            parameterFile + "\" at line 2: A B C D 0.5 BAD 180.0");
  }
}

TEST_CASE("CharmmParametersAssemblyValidationUsesApoCharmmError") {
  SECTION("MissingBondParameter") {
    CheckBondedParameterError(
        MISSING_BOND_PARAMETER_TEXT, TEST_PSF_TEXT, ApoCharmmErrorCode::Runtime,
        "Bond parameters were not found for bond 2 with atom types \"C\" and "
        "\"D\"");
  }

  SECTION("MissingAngleParameter") {
    CheckBondedParameterError(
        MISSING_ANGLE_PARAMETER_TEXT, TEST_PSF_TEXT,
        ApoCharmmErrorCode::Runtime,
        "Angle parameters were not found for angle 0 with atom types \"A\", "
        "\"B\", and \"C\"");
  }

  SECTION("MissingDihedralParameter") {
    CheckBondedParameterError(
        MISSING_DIHEDRAL_PARAMETER_TEXT, TEST_PSF_TEXT,
        ApoCharmmErrorCode::Runtime,
        "Dihedral parameters were not found for dihedral 0; atom types: A B C "
        "D; atom names: A1 B1 C1 D1");
  }

  SECTION("MissingImproperParameter") {
    CheckBondedParameterError(
        MISSING_IMPROPER_PARAMETER_TEXT, TEST_PSF_TEXT,
        ApoCharmmErrorCode::Runtime,
        "Improper parameters were not found for improper 0; atom types: A B C "
        "D; atom names: A1 B1 C1 D1");
  }

  SECTION("MissingCmapParameter") {
    // As in CHARMM's flexible parameter reader, a map does not match the same
    // torsions listed in another order.
    CheckBondedParameterError(
        TEST_CMAP_PARAMETER_TEXT, TEST_REVERSED_CMAP_PSF_TEXT,
        ApoCharmmErrorCode::Runtime,
        "CMAP parameters were not found for cross term 0; atom types: D C B A, "
        "A B C D; atom names: D1 C1 B1 A1, A1 B1 C1 D1");
  }

  SECTION("MissingNonbondedParameter") {
    const std::string parameterFile =
        "tmp_charmm_parameters_missing_nonbonded.prm";
    const std::string psfFile = "tmp_charmm_parameters_missing_nonbonded.psf";

    apo_test::WriteTextFile(parameterFile, MISSING_VDW_PARAMETER_TEXT);
    apo_test::WriteTextFile(psfFile, TEST_PSF_TEXT);

    CharmmParameters parameters(parameterFile);
    std::shared_ptr<CharmmPSF> psf = std::make_shared<CharmmPSF>(psfFile);

    apo_test::CheckApoCharmmError(
        [&parameters, &psf](void) {
          static_cast<void>(parameters.getVdwParamsAndTypes(psf));
        },
        ApoCharmmErrorCode::Runtime,
        "NONBONDED parameters were not found for atom type \"D\"");

    apo_test::RemoveIfExists(parameterFile);
    apo_test::RemoveIfExists(psfFile);
  }
}
