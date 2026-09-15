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

#include "ApoCharmmError.h"
#include "SelectionParser.h"
#include "SelectionTokenizer.h"

#include <string>
#include <utility>
#include <vector>

AtomSelector::AtomSelector(std::shared_ptr<const CharmmPSF> psf) : m_Psf(psf) {
  APOCHARMM_REQUIRE(m_Psf != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "AtomSelector requires a non-null PSF");

  APOCHARMM_REQUIRE(m_Psf->getNumAtoms() >= 0,
                    ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF atom count is not initialized; observed " +
                        std::to_string(m_Psf->getNumAtoms()));
}

AtomSelection
AtomSelector::select(const std::string_view selectionString) const {
  std::vector<SelectionToken> tokens =
      SelectionTokenizer::tokenize(selectionString);

  return SelectionParser::parse(m_Psf, std::move(tokens));
}

AtomReference
AtomSelector::selectAtom(const std::string_view selectionString) const {
  const AtomSelection selection = this->select(selectionString);
  const int numSelected = selection.getNumSelected();

  APOCHARMM_REQUIRE(numSelected == 1, ApoCharmmErrorCode::InvalidArgument,
                    "Atom selection must match exactly one atom; observed " +
                        std::to_string(numSelected));

  const std::vector<int> atomIndices = selection.getAtomIndices();
  const int atomIndex = atomIndices.front();

  return AtomReference(m_Psf, atomIndex);
}
