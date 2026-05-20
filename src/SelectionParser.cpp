// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "SelectionParser.h"

#include "str_utils.h"
#include <string>
#include <utility>

#include <iostream>

AtomSelection SelectionParser::parse(std::shared_ptr<const CharmmPSF> psf,
                                     std::vector<SelectionToken> tokens) {
  SelectionParser parser(psf, std::move(tokens));
  return parser.parse_impl();
}

SelectionParser::SelectionParser(std::shared_ptr<const CharmmPSF> psf,
                                 std::vector<SelectionToken> tokens)
    : m_Psf(psf), m_Tokens(std::move(tokens)), m_Position(0), m_OperatorStack(),
      m_SelectionStack() {
  if (m_Psf == nullptr)
    throw std::invalid_argument("SelectionParser requires a non-null PSF");

  this->buildResidueIndex();
  this->buildGroupIndex();
}

AtomSelection SelectionParser::parse_impl(void) {
  bool expectingSelection = true;

  while (this->peek().type != SelectionTokenType::End) {
    const SelectionTokenType tokenType = this->peek().type;

    if (expectingSelection == true) {
      if (SelectionParser::isPrefixOperator(tokenType) == true) {
        m_OperatorStack.push_back(tokenType);
        this->consume();
        continue;
      }

      if (tokenType == SelectionTokenType::LeftParenthesis) {
        m_OperatorStack.push_back(tokenType);
        this->consume();
        continue;
      }

      if (SelectionParser::isPrimarySelectionStart(tokenType) == true) {
        AtomSelection selection = this->readPrimarySelection();
        m_SelectionStack.push_back(std::move(selection));

        this->applyPendingPrefixOperators();

        expectingSelection = false;
        continue;
      }

      this->throwErrorAtCurrent("Expected an atom selection");
    }

    if (SelectionParser::isBinaryOperator(tokenType) == true) {
      this->reduceBinaryOperatorsFor(tokenType);

      m_OperatorStack.push_back(tokenType);
      this->consume();

      expectingSelection = true;
      continue;
    }

    if (tokenType == SelectionTokenType::RightParenthesis) {
      this->reduceUntilLeftParenthesis();
      this->consume();

      this->applyPendingPrefixOperators();

      expectingSelection = false;
      continue;
    }

    this->throwErrorAtCurrent("Expected .AND., .OR., ')', or end of selection");
  }

  if (expectingSelection == true) {
    this->throwErrorAtCurrent(
        "Selection ended while expecting an atom selection");
  }

  this->reduceRemainingOperators();

  if (m_SelectionStack.size() != 1)
    throw std::runtime_error("Invalud atom selection expression");

  return std::move(m_SelectionStack.back());
}

AtomSelection SelectionParser::readPrimarySelection(void) {
  if (this->match(SelectionTokenType::All) == true) {
    return AtomSelection(m_Psf->getNumAtoms(),
                         AtomSelection::InitialValue::ALL);
  }

  if (this->match(SelectionTokenType::None) == true) {
    return AtomSelection(m_Psf->getNumAtoms(),
                         AtomSelection::InitialValue::NONE);
  }

  if (this->match(SelectionTokenType::Atom) == true)
    return this->readAtomSelection();

  const SelectionTokenType tokenType = this->peek().type;

  if (SelectionParser::isFieldToken(tokenType) == true) {
    this->consume();
    return this->readFieldSelection(tokenType,
                                    SelectionParser::getFieldName(tokenType));
  }

  this->throwErrorAtCurrent("Expected a primary atom selection");

  return AtomSelection(m_Psf->getNumAtoms(), AtomSelection::InitialValue::NONE);
}

AtomSelection SelectionParser::readAtomSelection(void) {
  const SelectionToken segmentValue = this->consumeSelectionValue("ATOM");
  const SelectionToken residueValue = this->consumeSelectionValue("ATOM");
  const SelectionToken atomValue = this->consumeSelectionValue("ATOM");

  AtomSelection segmentSelection = this->makeFieldSelection(
      SelectionTokenType::SegmentIdentifier, segmentValue.text);

  AtomSelection residueSelection = this->makeFieldSelection(
      SelectionTokenType::ResidueIdentifier, residueValue.text);

  AtomSelection atomSelection =
      this->makeFieldSelection(SelectionTokenType::Type, atomValue.text);

  return (segmentSelection & residueSelection & atomSelection);
}

AtomSelection
SelectionParser::makeFieldSelection(const SelectionTokenType fieldTokenType,
                                    const std::string_view fieldName) const {
  const int numAtoms = m_Psf->getNumAtoms();

  AtomSelection selection(numAtoms, AtomSelection::InitialValue::NONE);

  for (int i = 0; i < numAtoms; i++) {
    const std::string_view value = this->getFieldValue(fieldTokenType, i);

    if (SelectionParser::doSelectionPatternsMatch(value, fieldName) == true)
      selection.set(i);
  }

  return selection;
}

AtomSelection
SelectionParser::makeRangeSelection(const SelectionTokenType fieldTokenType,
                                    const std::string_view firstText,
                                    const std::string_view lastText) const {
  const int numAtoms = m_Psf->getNumAtoms();

  AtomSelection selection(numAtoms, AtomSelection::InitialValue::NONE);

  if (fieldTokenType == SelectionTokenType::ByNumber) {
    long long int firstAtomNumber = 0;
    long long int lastAtomNumber = 0;

    if ((SelectionParser::parseInteger(firstAtomNumber, firstText) == false) ||
        (SelectionParser::parseInteger(lastAtomNumber, lastText) == false)) {
      throw std::runtime_error("BYNU range requires integer atom numbers");
    }

    if (firstAtomNumber > lastAtomNumber)
      std::swap(firstAtomNumber, lastAtomNumber);

    const long long int firstClamped = std::max(firstAtomNumber, 1LL);
    const long long int lastClamped = std::min(
        lastAtomNumber, static_cast<long long int>(m_Psf->getNumAtoms()));

    for (long long int i = firstClamped; i <= lastClamped; i++)
      selection.set(static_cast<int>(i - 1));

    return selection;
  }

  long long int firstInteger = 0;
  long long int lastInteger = 0;

  const bool isIntegerRange =
      ((SelectionParser::parseInteger(firstInteger, firstText) == true) &&
       (SelectionParser::parseInteger(lastInteger, lastText) == true));

  if ((isIntegerRange == true) && (firstInteger > lastInteger))
    std::swap(firstInteger, lastInteger);

  const std::string firstUpper = apo::to_upper(firstText);
  const std::string lastUpper = apo::to_upper(lastText);
  const std::string lowText = std::min(firstUpper, lastUpper);
  const std::string highText = std::max(firstUpper, lastUpper);

  for (int i = 0; i < numAtoms; i++) {
    const std::string_view value = this->getFieldValue(fieldTokenType, i);

    if (isIntegerRange == true) {
      long long int valueInteger = 0;

      if ((SelectionParser::parseInteger(valueInteger, value) == true) &&
          (valueInteger >= firstInteger) && (valueInteger <= lastInteger)) {
        selection.set(i);
      }

      continue;
    }

    const std::string valueUpper = apo::to_upper(value);
    if ((valueUpper >= lowText) && (valueUpper <= highText))
      selection.set(i);
  }

  return selection;
}

AtomSelection
SelectionParser::readFieldSelection(const SelectionTokenType fieldTokenType,
                                    const std::string_view fieldName) {
  const SelectionToken firstValue = this->consumeSelectionValue(fieldName);

  if (this->match(SelectionTokenType::Colon) == true) {
    const SelectionToken lastValue = this->consumeSelectionValue(fieldName);

    return this->makeRangeSelection(fieldTokenType, firstValue.text,
                                    lastValue.text);
  }

  return this->makeFieldSelection(fieldTokenType, firstValue.text);
}

AtomSelection SelectionParser::popSelection(void) {
  if (m_SelectionStack.empty() == true)
    throw std::runtime_error("Expected an atom selection before operator");

  AtomSelection selection = std::move(m_SelectionStack.back());
  m_SelectionStack.pop_back();

  return selection;
}

AtomSelection
SelectionParser::invertSelection(const AtomSelection &selection) const {
  const int numAtoms = m_Psf->getNumAtoms();

  if (selection.getNumAtoms() != numAtoms) {
    throw std::runtime_error(
        "Cannot invert AtomSelection with wrong atom count");
  }

  AtomSelection inverted(numAtoms, AtomSelection::InitialValue::NONE);

  for (int i = 0; i < numAtoms; i++) {
    if (selection.contains(i) == false)
      inverted.set(i);
  }

  return inverted;
}

AtomSelection
SelectionParser::expandByResidue(const AtomSelection &selection) const {
  AtomSelection expanded(m_Psf->getNumAtoms(),
                         AtomSelection::InitialValue::NONE);

  const std::vector<int2> &residues = m_Psf->getResidues().getHostArray();

  for (const int i : selection.getAtomIndices()) {
    const int j = m_ResidueIndex[i];

    if ((j < 0) || (j >= static_cast<int>(residues.size()))) {
      throw std::runtime_error(
          "Invalid residue index while expanding selection by residue");
    }

    const int2 resi = residues[j];

    for (int k = resi.x; k <= resi.y; k++)
      expanded.set(k);
  }

  return expanded;
}

AtomSelection
SelectionParser::expandByGroup(const AtomSelection &selection) const {
  AtomSelection expanded(m_Psf->getNumAtoms(),
                         AtomSelection::InitialValue::NONE);

  const std::vector<int2> &groups = m_Psf->getResidues().getHostArray();

  for (const int i : selection.getAtomIndices()) {
    const int j = m_GroupIndex[i];

    if ((j < 0) || (j >= static_cast<int>(groups.size()))) {
      throw std::runtime_error(
          "Invalid group index while expanding selection by group");
    }

    const int2 grp = groups[j];

    for (int k = grp.x; k <= grp.y; k++)
      expanded.set(k);
  }

  return expanded;
}

AtomSelection
SelectionParser::expandBonded(const AtomSelection &selection) const {
  AtomSelection expanded(m_Psf->getNumAtoms(),
                         AtomSelection::InitialValue::NONE);

  const std::vector<std::set<int>> &connected12 = m_Psf->getConnected12();

  if (static_cast<int>(connected12.size()) != m_Psf->getNumAtoms()) {
    throw std::runtime_error("CharmmPSF bonded-connectivity array size does "
                             "not match number of atoms");
  }

  for (const int i : selection.getAtomIndices()) {
    for (const int j : connected12[i])
      expanded.set(j);
  }

  return expanded;
}

void SelectionParser::applyTopOperator(void) {
  if (m_OperatorStack.empty() == true)
    throw std::runtime_error("Internal parser error: Empty operator stack");

  const SelectionTokenType opType = m_OperatorStack.back();
  m_OperatorStack.pop_back();

  switch (opType) {
  case SelectionTokenType::And: {
    AtomSelection right = this->popSelection();
    AtomSelection left = this->popSelection();

    left &= right;
    m_SelectionStack.push_back(std::move(left));

    return;
  }

  case SelectionTokenType::Or: {
    AtomSelection right = this->popSelection();
    AtomSelection left = this->popSelection();

    left |= right;
    m_SelectionStack.push_back(std::move(left));

    return;
  }

  case SelectionTokenType::Not: {
    AtomSelection selection = this->popSelection();

    m_SelectionStack.push_back(this->invertSelection(selection));

    return;
  }

  case SelectionTokenType::ByResidue: {
    AtomSelection selection = this->popSelection();

    m_SelectionStack.push_back(this->expandByResidue(selection));

    return;
  }

  case SelectionTokenType::ByGroup: {
    AtomSelection selection = this->popSelection();

    m_SelectionStack.push_back(this->expandByGroup(selection));

    return;
  }

  case SelectionTokenType::Bonded: {
    AtomSelection selection = this->popSelection();

    m_SelectionStack.push_back(this->expandBonded(selection));

    return;
  }

  case SelectionTokenType::LeftParenthesis:
    throw std::runtime_error("Internal parser error: Tried to apply '('");

  default:
    throw std::runtime_error(
        "Internal parser error: Tried to apply a non-operator token");
  }

  return;
}

void SelectionParser::applyPendingPrefixOperators(void) {
  while ((m_OperatorStack.empty() == false) &&
         (SelectionParser::isPrefixOperator(m_OperatorStack.back()) == true)) {
    this->applyTopOperator();
  }
  return;
}

void SelectionParser::reduceBinaryOperatorsFor(
    const SelectionTokenType currOp) {
  while (m_OperatorStack.empty() == false) {
    const SelectionTokenType prevOp = m_OperatorStack.back();

    if (SelectionParser::isBinaryOperator(prevOp) == false)
      return;

    if (SelectionParser::getOperatorPrecedence(prevOp) <
        SelectionParser::getOperatorPrecedence(currOp))
      return;

    this->applyTopOperator();
  }

  return;
}

void SelectionParser::reduceUntilLeftParenthesis(void) {
  while ((m_OperatorStack.empty() == false) &&
         (m_OperatorStack.back() != SelectionTokenType::LeftParenthesis)) {
    this->applyTopOperator();
  }

  if (m_OperatorStack.empty() == true)
    this->throwErrorAtCurrent("Found ')' without matching '('");

  m_OperatorStack.pop_back();

  return;
}

void SelectionParser::reduceRemainingOperators(void) {
  while (m_OperatorStack.empty() == false) {
    if (m_OperatorStack.back() == SelectionTokenType::LeftParenthesis)
      throw std::runtime_error("Found '(' without matching ')'");

    this->applyTopOperator();
  }

  return;
}

const SelectionToken &SelectionParser::peek(void) const {
  if (m_Position >= m_Tokens.size()) {
    throw std::runtime_error(
        "Internal parser error: Token position is out of range");
  }
  return m_Tokens[m_Position];
}

const SelectionToken &SelectionParser::consume(void) {
  const SelectionToken &token = this->peek();
  m_Position++;
  return token;
}

const SelectionToken &
SelectionParser::consumeSelectionValue(const std::string_view context) {
  const SelectionToken &token = this->peek();

  if (SelectionParser::isSelectionValueToken(token.type) == false) {
    this->throwErrorAtCurrent("Expected selection value after " +
                              std::string{context});
  }

  return this->consume();
}

bool SelectionParser::match(const SelectionTokenType type) {
  if (this->peek().type != type)
    return false;
  this->consume();
  return true;
}

const std::string_view
SelectionParser::getFieldValue(const SelectionTokenType fieldTokenType,
                               const int atomIndex) const {
  switch (fieldTokenType) {
  case SelectionTokenType::Type:
    // CHARMM TYPE maps to the PSF atom-name field
    return m_Psf->getAtomNames()[atomIndex];

  case SelectionTokenType::Chemical:
    // CHARMM CHEMICAL maps to the PSF atom-type field
    return m_Psf->getAtomTypes()[atomIndex];

  case SelectionTokenType::SegmentIdentifier:
    return m_Psf->getSegmentIdentifiers()[atomIndex];

  case SelectionTokenType::ResidueIdentifier:
    return std::to_string(m_Psf->getResidueIdentifiers()[atomIndex]);

  case SelectionTokenType::ResidueName:
    return m_Psf->getResidueNames()[atomIndex];

  case SelectionTokenType::ByNumber:
    return std::to_string(atomIndex + 1);

  default:
    throw std::runtime_error(
        "Internal parser error: Token is not a field selection token");
  }
}

void SelectionParser::buildResidueIndex(void) {
  const int numAtoms = m_Psf->getNumAtoms();

  m_ResidueIndex.assign(numAtoms, -1);

  const std::vector<int2> &residues = m_Psf->getResidues().getHostArray();

  for (std::size_t i = 0; i < residues.size(); i++) {
    const int2 resi = residues[i];

    if ((resi.x < 0) || (resi.y < resi.x) || (resi.y >= numAtoms))
      throw std::runtime_error("Residue atom range is out of bounds");

    for (int j = resi.x; j <= resi.y; j++)
      m_ResidueIndex[j] = static_cast<int>(i);
  }

  return;
}

void SelectionParser::buildGroupIndex(void) {
  const int numAtoms = m_Psf->getNumAtoms();

  m_GroupIndex.assign(numAtoms, -1);

  const std::vector<int2> &groups = m_Psf->getGroups().getHostArray();

  for (std::size_t i = 0; i < groups.size(); i++) {
    const int2 grp = groups[i];

    if ((grp.x < 0) || (grp.y < grp.x) || (grp.y >= numAtoms))
      throw std::runtime_error("Group atom range is out of bounds");

    for (int j = grp.x; j <= grp.y; j++)
      m_GroupIndex[j] = static_cast<int>(i);
  }

  return;
}

bool SelectionParser::parseInteger(long long int &value,
                                   const std::string_view str) {
  try {
    std::size_t pos = 0;
    value = std::stoll(std::string{str}, &pos, 10);
    return (pos == str.length());
  } catch (...) {
    return false;
  }
}

bool SelectionParser::doSelectionPatternsMatch(const std::string_view value,
                                               const std::string_view pattern) {
  const std::string upperValue = apo::to_upper(value);
  const std::string upperPattern = apo::to_upper(pattern);

  if (apo::contains_wildcard(upperPattern) == false)
    return (upperValue == upperPattern);

  const std::size_t valueSize = upperValue.length();
  const std::size_t patternSize = upperPattern.length();

  std::vector<char> prev(patternSize + 1, 0);
  std::vector<char> curr(patternSize + 1, 0);

  prev[0] = 1;

  for (std::size_t i = 1; i <= patternSize; i++) {
    const char patternCharacter = upperPattern[i - 1];

    if ((patternCharacter == '*') || (patternCharacter == '#'))
      prev[i] = prev[i - 1];
  }

  for (std::size_t i = 1; i <= valueSize; i++) {
    curr.assign(patternSize + 1, 0);
    const char valueCharacter = upperValue[i - 1];

    for (std::size_t j = 1; j <= patternSize; j++) {
      const char patternCharacter = upperPattern[j - 1];

      if (patternCharacter == '*') {
        const bool matchZeroCharacters = (curr[j - 1] != 0);
        const bool matchOneMoreCharacter = (prev[j] != 0);
        curr[j] = (matchZeroCharacters || matchOneMoreCharacter);
        continue;
      }

      if (patternCharacter == '#') {
        const bool matchZeroDigits = (curr[j - 1] != 0);
        const bool matchOneMoreDigit =
            ((std::isdigit(static_cast<unsigned char>(valueCharacter)) != 0) &&
             (prev[j] != 0));
        curr[j] = (matchZeroDigits || matchOneMoreDigit);
        continue;
      }

      if (patternCharacter == '%') {
        curr[j] = prev[j - 1];
        continue;
      }

      if (patternCharacter == '+') {
        curr[j] =
            ((std::isdigit(static_cast<unsigned char>(valueCharacter)) != 0) &&
             (prev[j - 1] != 0));
        continue;
      }

      curr[j] = ((valueCharacter == patternCharacter) && (prev[j - 1] != 0));
    }
    prev.swap(curr);
  }

  return (prev[patternSize] != 0);
}

bool SelectionParser::isSelectionValueToken(const SelectionTokenType type) {
  switch (type) {
  case SelectionTokenType::Identifier:
  case SelectionTokenType::Integer:
  case SelectionTokenType::Real:

    // JEG260519: These are normally keywords, but they can also be literal
    // values after a field token (e.g. "type TYPE" or "ren ALL").
  case SelectionTokenType::All:
  case SelectionTokenType::None:
  case SelectionTokenType::Type:
  case SelectionTokenType::Chemical:
  case SelectionTokenType::SegmentIdentifier:
  case SelectionTokenType::ResidueIdentifier:
  case SelectionTokenType::ResidueName:
  case SelectionTokenType::Atom:
  case SelectionTokenType::ByNumber:
    return true;

  default:
    return false;
  }
}

bool SelectionParser::isFieldToken(const SelectionTokenType type) {
  switch (type) {
  case SelectionTokenType::Type:
  case SelectionTokenType::Chemical:
  case SelectionTokenType::SegmentIdentifier:
  case SelectionTokenType::ResidueIdentifier:
  case SelectionTokenType::ResidueName:
  case SelectionTokenType::ByNumber:
    return true;

  default:
    return false;
  }
}

bool SelectionParser::isPrimarySelectionStart(const SelectionTokenType type) {
  return ((type == SelectionTokenType::All) ||
          (type == SelectionTokenType::None) ||
          (type == SelectionTokenType::Atom) ||
          (SelectionParser::isFieldToken(type) == true));
}

bool SelectionParser::isPrefixOperator(const SelectionTokenType type) {
  return ((type == SelectionTokenType::Not) ||
          (type == SelectionTokenType::ByResidue) ||
          (type == SelectionTokenType::ByGroup) ||
          (type == SelectionTokenType::Bonded));
}

bool SelectionParser::isBinaryOperator(const SelectionTokenType type) {
  return ((type == SelectionTokenType::And) ||
          (type == SelectionTokenType::Or));
}

int SelectionParser::getOperatorPrecedence(const SelectionTokenType type) {
  switch (type) {
  case SelectionTokenType::Or:
    return 1;

  case SelectionTokenType::And:
    return 2;

  default:
    return 0;
  }
}

const std::string_view
SelectionParser::getFieldName(const SelectionTokenType type) {
  switch (type) {
  case SelectionTokenType::Type:
    return "type";

  case SelectionTokenType::Chemical:
    return "chem";

  case SelectionTokenType::SegmentIdentifier:
    return "segi";

  case SelectionTokenType::ResidueIdentifier:
    return "resi";

  case SelectionTokenType::ResidueName:
    return "resn";

  case SelectionTokenType::ByNumber:
    return "bynu";

  default:
    throw std::runtime_error(
        "Internal parser error: Token is not a field selection token");
  }
}

void SelectionParser::throwErrorAtCurrent(
    const std::string_view message) const {
  throw std::runtime_error(std::string{message} + " at position " +
                           std::to_string(this->peek().pos));
  return;
}
