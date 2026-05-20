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

#include "SelectionToken.h"
#include <string_view>
#include <vector>

class SelectionTokenizer {
public:
  static std::vector<SelectionToken>
  tokenize(const std::string_view selectionString);

private:
  static bool isBareTokenCharacter(const char c);
  static bool isInteger(const std::string_view str);
  static bool isReal(const std::string_view str);
  static SelectionTokenType getBareTokenType(const std::string_view str);
  static SelectionTokenType getDottedTokenType(const std::string_view str);
};
