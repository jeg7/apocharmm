// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "AtomSelector.h"
#include "CharmmPSF.h"
#include "catch.hpp"
#include "compare.h"
#include "test_paths.h"
#include <memory>
#include <string>
#include <vector>

static std::vector<int> MakeRange(const int first, const int last) {
  std::vector<int> values;
  for (int value = first; value < last; value++)
    values.push_back(value);
  return values;
}

TEST_CASE("nacl") {
  const std::string dataPath = getDataPath();
  auto psf = std::make_shared<CharmmPSF>(dataPath + "1lvz.psf");

  SECTION("selection") {
    // ILE LYS GLU ASN LEU LYS ASP CYS GLY LEU PHE
    auto selector = std::make_shared<AtomSelector>(psf);

    // None
    auto selection = selector->select("none");
    CHECK(selection.getNumSelected() == 0);

    // All
    selection = selector->select("all");
    CHECK(selection.getNumSelected() == 183);

    // By number
    selection = selector->select("bynum 1:5");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), MakeRange(0, 5)));

    // By number out of range
    selection = selector->select("bynu 180 : 200");
    CHECK(
        CompareVectors1<int>(selection.getAtomIndices(), MakeRange(179, 183)));

    // Atom name
    selection = selector->select("type CA");
    CHECK(
        CompareVectors1<int>(selection.getAtomIndices(),
                             {4, 23, 45, 60, 74, 93, 115, 127, 138, 145, 164}));

    // Atom type
    selection = selector->select("chemical CT1");
    CHECK(CompareVectors1<int>(
        selection.getAtomIndices(),
        {4, 6, 23, 45, 60, 74, 79, 93, 115, 127, 145, 150, 164}));

    // Segment
    selection = selector->select("segid A000");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), MakeRange(0, 183)));

    // Residue identifier
    selection = selector->select("resid 1");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), MakeRange(0, 21)));

    // Residue identifier range
    selection = selector->select("resi 1:3");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), MakeRange(0, 58)));

    // Residue name
    selection = selector->select("resname GLU");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), MakeRange(43, 58)));

    selection = selector->select("resn LYS");
    const std::vector<int> lys2 = MakeRange(21, 43);
    const std::vector<int> lys6 = MakeRange(91, 113);
    std::vector<int> lys = lys2;
    lys.insert(lys.end(), lys6.begin(), lys6.end());
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), lys));

    // Atom selection
    selection = selector->select("atom A000 1 CA");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), {4}, 0.0, true));

    // And
    selection = selector->select("type CA .and. resn GLU");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), {45}));

    // Or and precedence
    // This should parse as: type CA .or. (type N .and. resn ILE)
    selection = selector->select("type CA .or. type N .and. resn ILE");
    CHECK(CompareVectors1<int>(
        selection.getAtomIndices(),
        {0, 4, 23, 45, 60, 74, 93, 115, 127, 138, 145, 164}, 0.0, true));

    // Parenthesis should change the result
    selection = selector->select("(type CA .or. type N) .and. resn ILE");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), {0, 4}));

    // Not + wildcard
    selection = selector->select(".not. type H*");
    CHECK(selection.getNumSelected() == 89);

    // Wildcards
    selection = selector->select("type H*");
    CHECK(selection.getNumSelected() == 94);

    selection = selector->select("type HG+");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(),
                               {29, 30, 51, 52, 99, 100, 133}));

    selection = selector->select("type HG#");
    CHECK(CompareVectors1<int>(
        selection.getAtomIndices(),
        {9, 10, 11, 13, 14, 29, 30, 51, 52, 80, 99, 100, 133, 151}));

    selection = selector->select("type H%");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(),
                               {5,   7,   22,  24,  44,  46,  59,  61,
                                73,  75,  80,  92,  94,  114, 116, 126,
                                128, 137, 144, 146, 151, 163, 165, 175}));

    selection = selector->select("type C#");
    CHECK(CompareVectors1<int>(
        selection.getAtomIndices(),
        {19, 41, 56, 70, 89, 111, 123, 134, 141, 160, 180}));

    selection = selector->select("type C%");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(),
                               {4,   6,   15,  23,  25,  28,  31,  34,  45,
                                47,  50,  53,  60,  62,  65,  74,  76,  79,
                                93,  95,  98,  101, 104, 115, 117, 120, 127,
                                129, 138, 145, 147, 150, 164, 166, 169, 174}));

    selection = selector->select("type C*");
    CHECK(selection.getNumSelected() == 57);

    // By residue
    // OE* occurs in residue GLU only, so .byres. should select the whole GLU
    // residue
    selection = selector->select(".byres. type OE*");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), MakeRange(43, 58)));

    // selection = selector->select(".byres. ( type CA .and. resn GLU )");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), MakeRange(43, 58)));

    // Bonded
    selection = selector->select(".bonded. bynum 1");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), {1, 2, 3, 4}));

    // Atom A000 1 CA is CHARMM atom 5, zero-based 4.
    // It is bonded to CHARMM atoms 1, 6, 7, 20 -> zero-based 0, 5, 6, 19
    selection = selector->select(".bonded. (atom A000 1 CA)");
    CHECK(CompareVectors1<int>(selection.getAtomIndices(), {0, 5, 6, 19}));

    // Syntax error
    CHECK_THROWS(selector->select(".and. type CA"));
    CHECK_THROWS(selector->select("type CA resn GLU"));
    CHECK_THROWS(selector->select("(type CA .or. type N"));
    CHECK_THROWS(selector->select("bynu A:C"));

    // Not implemented yet
    CHECK_THROWS(selector->select(".around. type CA"));
  }
}
