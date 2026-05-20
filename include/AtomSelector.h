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

#include "AtomSelection.h"
#include "CharmmPSF.h"
#include <memory>
#include <string_view>

class AtomSelector {
public:
  AtomSelector(void) = delete;
  explicit AtomSelector(std::shared_ptr<const CharmmPSF> psf);

public:
  AtomSelection select(const std::string_view selectionString) const;

private:
  std::shared_ptr<const CharmmPSF> m_Psf;
};
