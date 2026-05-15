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
#include "test_paths.h"
#include <memory>
#include <string>

TEST_CASE("nacl") {
  const std::string dataPath = getDataPath();
  auto psf = std::make_shared<CharmmPSF>(dataPath + "nacl_pair.psf");

  SECTION("selection") {
    auto selector = std::make_shared<AtomSelector>(psf);

    // SEGID
    selector->seleSegi("NACL");
    CHECK(selector->getSelection()[0] == 1);
    CHECK(selector->getSelection()[1] == 1);
    CHECK(selector->getNumSelected() == 2);
    selector->clear();
    selector->seleSegi("AAAA");
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 0);
    selector->clear();

    // RESID
    selector->seleResi(1);
    CHECK(selector->getSelection()[0] == 1);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleResi(2);
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 1);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleResi(3);
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 0);
    selector->clear();

    // RESN
    selector->seleResn("SOD");
    CHECK(selector->getSelection()[0] == 1);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleResn("CLA");
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 1);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleResn("ALA");
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 0);
    selector->clear();

    // NAME
    selector->seleName("SOD");
    CHECK(selector->getSelection()[0] == 1);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleName("CLA");
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 1);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleName("CH3");
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 0);
    selector->clear();

    // TYPE
    selector->seleType("SOD");
    CHECK(selector->getSelection()[0] == 1);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleType("CLA");
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 1);
    CHECK(selector->getNumSelected() == 1);
    selector->clear();
    selector->seleType("CH3");
    CHECK(selector->getSelection()[0] == 0);
    CHECK(selector->getSelection()[1] == 0);
    CHECK(selector->getNumSelected() == 0);
    selector->clear();
  }
}
