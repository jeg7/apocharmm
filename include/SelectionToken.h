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

#include <string>

enum class SelectionTokenType {
  End, // 0

  Identifier, // 1
  Integer,    // 2
  Real,       // 3

  LeftParenthesis,  // 4
  RightParenthesis, // 5
  Colon,            // 6

  All,  // 7
  None, // 8

  And, // 9
  Or,  // 10
  Not, // 11

  ByResidue, // 12
  ByGroup,   // 13
  Bonded,    // 14

  Type,              // 15
  Chemical,          // 16
  SegmentIdentifier, // 17
  ResidueIdentifier, // 18
  ResidueName,       // 19
  Atom,              // 20
  ByNumber           // 21
};

struct SelectionToken {
  SelectionTokenType type;
  std::string text;
  int pos;
};
