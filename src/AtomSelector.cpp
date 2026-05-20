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

#include "SelectionParser.h"
#include "SelectionTokenizer.h"
#include "str_utils.h"
#include <stdexcept>
#include <utility>

AtomSelector::AtomSelector(std::shared_ptr<const CharmmPSF> psf) : m_Psf(psf) {}

AtomSelection
AtomSelector::select(const std::string_view selectionString) const {
  if (m_Psf == nullptr)
    throw std::runtime_error("AtomSelector does not have a PSF");

  std::vector<SelectionToken> tokens =
      SelectionTokenizer::tokenize(selectionString);

  return SelectionParser::parse(m_Psf, std::move(tokens));
}
