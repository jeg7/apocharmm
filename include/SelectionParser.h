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
#include "SelectionToken.h"
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

class SelectionParser {
public:
  static AtomSelection parse(std::shared_ptr<const CharmmPSF> psf,
                             std::vector<SelectionToken> tokens);

private:
  SelectionParser(std::shared_ptr<const CharmmPSF> psf,
                  std::vector<SelectionToken> tokens);

  AtomSelection parse_impl(void);

private:
  AtomSelection readPrimarySelection(void);
  AtomSelection readAtomSelection(void);
  AtomSelection readFieldSelection(const SelectionTokenType fieldTokenType,
                                   const std::string_view fieldName);

  AtomSelection makeFieldSelection(const SelectionTokenType fieldTokenType,
                                   const std::string_view fieldName) const;
  AtomSelection makeRangeSelection(const SelectionTokenType fieldTokenType,
                                   const std::string_view firstText,
                                   const std::string_view lastText) const;

private:
  AtomSelection popSelection(void);
  AtomSelection invertSelection(const AtomSelection &selection) const;
  AtomSelection expandByResidue(const AtomSelection &selection) const;
  AtomSelection expandByGroup(const AtomSelection &selection) const;
  AtomSelection expandBonded(const AtomSelection &selection) const;
  void applyTopOperator(void);
  void applyPendingPrefixOperators(void);

  void reduceBinaryOperatorsFor(const SelectionTokenType currOp);
  void reduceUntilLeftParenthesis(void);
  void reduceRemainingOperators(void);

private:
  const SelectionToken &peek(void) const;
  const SelectionToken &consume(void);
  const SelectionToken &consumeSelectionValue(const std::string_view context);
  bool match(const SelectionTokenType type);

  const std::string_view getFieldValue(const SelectionTokenType fieldTokenType,
                                       const int atomIndex) const;

private:
  void buildResidueIndex(void);
  void buildGroupIndex(void);

private:
  static bool parseInteger(long long int &value, const std::string_view str);
  static bool doSelectionPatternsMatch(const std::string_view value,
                                       const std::string_view pattern);

  static bool isSelectionValueToken(const SelectionTokenType type);
  static bool isFieldToken(const SelectionTokenType type);
  static bool isPrimarySelectionStart(const SelectionTokenType type);
  static bool isPrefixOperator(const SelectionTokenType type);
  static bool isBinaryOperator(const SelectionTokenType type);
  static int getOperatorPrecedence(const SelectionTokenType type);
  static const std::string_view getFieldName(const SelectionTokenType type);

private:
  void throwErrorAtCurrent(const std::string_view message) const;

private:
  std::shared_ptr<const CharmmPSF> m_Psf;
  std::vector<SelectionToken> m_Tokens;
  std::size_t m_Position;

  std::vector<SelectionTokenType> m_OperatorStack;
  std::vector<AtomSelection> m_SelectionStack;

  std::vector<int> m_ResidueIndex;
  std::vector<int> m_GroupIndex;
};
