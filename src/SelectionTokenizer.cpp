// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "SelectionTokenizer.h"

#include "str_utils.h"
#include <cctype>
#include <stdexcept>

std::vector<SelectionToken>
SelectionTokenizer::tokenize(const std::string_view selectionString) {
  std::vector<SelectionToken> tokens;

  std::size_t pos = 0;
  while (pos < selectionString.length()) {
    const char character = selectionString[pos];

    // Ignore blank spaces
    if (std::isspace(static_cast<unsigned char>(character))) {
      pos++;
      continue;
    } else if (character == '(') {
      tokens.push_back(SelectionToken{SelectionTokenType::LeftParenthesis, "(",
                                      static_cast<int>(pos)});
      pos++;
      continue;
    } else if (character == ')') {
      tokens.push_back(SelectionToken{SelectionTokenType::RightParenthesis, ")",
                                      static_cast<int>(pos)});
      pos++;
      continue;
    } else if (character == ':') {
      tokens.push_back(SelectionToken{SelectionTokenType::Colon, ":",
                                      static_cast<int>(pos)});
      pos++;
      continue;
    } else if (character == '.') {
      const std::size_t start = pos;
      const std::size_t end = selectionString.find('.', start + 1);

      if (end == std::string::npos) {
        throw std::runtime_error(
            "Unterminated dotted selection operator at position " +
            std::to_string(start));
      }

      const std::string_view text =
          selectionString.substr(start, end - start + 1);
      const SelectionTokenType type =
          SelectionTokenizer::getDottedTokenType(text);
      tokens.push_back(
          SelectionToken{type, std::string{text}, static_cast<int>(start)});

      pos = end + 1;
      continue;
    }

    const std::size_t start = pos;
    while ((pos < selectionString.length()) &&
           (SelectionTokenizer::isBareTokenCharacter(selectionString[pos]))) {
      pos++;
    }

    if (start == pos) {
      throw std::runtime_error(
          "Unexpected character in atom selection at position " +
          std::to_string(start));
    }

    const std::string_view text = selectionString.substr(start, pos - start);
    const SelectionTokenType type = SelectionTokenizer::getBareTokenType(text);
    tokens.push_back(
        SelectionToken{type, std::string{text}, static_cast<int>(start)});
  }

  tokens.push_back(SelectionToken{SelectionTokenType::End, "",
                                  static_cast<int>(selectionString.length())});

  return tokens;
}

bool SelectionTokenizer::isBareTokenCharacter(const char c) {
  if (std::isspace(static_cast<unsigned char>(c)))
    return false;
  if ((c == '(') || (c == ')') || (c == ':'))
    return false;
  return true;
}

bool SelectionTokenizer::isInteger(const std::string_view str) {
  if (str.empty() == true)
    return false;

  std::size_t pos = 0;
  if (str[pos] == '-') // Check for negative sign
    pos++;

  if (pos == str.length())
    return false;

  while (pos < str.length()) {
    if (std::isdigit(static_cast<unsigned char>(str[pos])) == false)
      return false;
    pos++;
  }

  return true;
}

bool SelectionTokenizer::isReal(const std::string_view str) {
  if (str.empty())
    return false;

  std::size_t pos = 0;
  if (str[pos] == '-') // Check for negative sign
    pos++;

  bool foundDigit = false, foundDecimal = false;
  while (pos < str.length()) {
    if (std::isdigit(static_cast<unsigned char>(str[pos])))
      foundDigit = true;
    else if ((str[pos] == '.') && (foundDecimal == false))
      foundDecimal = true;
    else
      return false;
    pos++;
  }

  return ((foundDigit == true) && (foundDecimal == true));
}

SelectionTokenType
SelectionTokenizer::getBareTokenType(const std::string_view str) {
  const std::string tmp = apo::trim(apo::to_upper(str));
  const std::size_t len = (tmp.length() < 4) ? tmp.length() : 4;
  const std::string_view upperStr = std::string_view(tmp).substr(0, len);
  if (upperStr == "ALL")
    return SelectionTokenType::All;
  if (upperStr == "NONE")
    return SelectionTokenType::None;
  if (upperStr == "TYPE")
    return SelectionTokenType::Type;
  if (upperStr == "CHEM")
    return SelectionTokenType::Chemical;
  if (upperStr == "SEGI")
    return SelectionTokenType::SegmentIdentifier;
  if (upperStr == "RESI")
    return SelectionTokenType::ResidueIdentifier;
  if (upperStr == "RESN")
    return SelectionTokenType::ResidueName;
  if (upperStr == "ATOM")
    return SelectionTokenType::Atom;
  if (upperStr == "BYNU")
    return SelectionTokenType::ByNumber;
  if (SelectionTokenizer::isInteger(str))
    return SelectionTokenType::Integer;
  if (SelectionTokenizer::isReal(str))
    return SelectionTokenType::Real;
  return SelectionTokenType::Identifier;
}

SelectionTokenType
SelectionTokenizer::getDottedTokenType(const std::string_view str) {
  const std::string upperStr = apo::to_upper(str);

  if (upperStr == ".AND.")
    return SelectionTokenType::And;
  if (upperStr == ".OR.")
    return SelectionTokenType::Or;
  if (upperStr == ".NOT.")
    return SelectionTokenType::Not;
  if (upperStr == ".BYRES.")
    return SelectionTokenType::ByResidue;
  if (upperStr == ".BYGROUP.")
    return SelectionTokenType::ByGroup;
  if (upperStr == ".BONDED.")
    return SelectionTokenType::Bonded;

  throw std::invalid_argument("Unknown dotted atom selection operator \"" +
                              std::string{str} + "\"");

  return SelectionTokenType::End;
}
