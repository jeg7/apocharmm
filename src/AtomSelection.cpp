// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "AtomSelection.h"

#include "ApoCharmmError.h"

#include <string>
#include <utility>

AtomSelection::AtomSelection(const int numAtoms,
                             const InitialValue initialValue)
    : m_NumAtoms(0), m_Words() {
  this->setNumAtoms(numAtoms, initialValue);
}

AtomSelection::AtomSelection(const AtomSelection &other)
    : m_NumAtoms(other.m_NumAtoms), m_Words(other.m_Words) {}

AtomSelection::AtomSelection(AtomSelection &&other) noexcept
    : m_NumAtoms(0), m_Words() {
  std::swap(m_NumAtoms, other.m_NumAtoms);
  m_Words.swap(other.m_Words);
}

AtomSelection &AtomSelection::operator=(const AtomSelection &other) {
  m_NumAtoms = other.m_NumAtoms;
  m_Words = other.m_Words;
  return *this;
}

AtomSelection &AtomSelection::operator=(AtomSelection &&other) noexcept {
  if (this != &other) {
    AtomSelection replacement(std::move(other));
    std::swap(m_NumAtoms, replacement.m_NumAtoms);
    m_Words.swap(replacement.m_Words);
  }

  return *this;
}

AtomSelection &AtomSelection::operator&=(const AtomSelection &other) {
  this->checkCompatible(other);

  for (std::size_t i = 0; i < m_Words.size(); i++)
    m_Words[i] &= other.m_Words[i];

  return *this;
}

AtomSelection &AtomSelection::operator|=(const AtomSelection &other) {
  this->checkCompatible(other);

  for (std::size_t i = 0; i < m_Words.size(); i++)
    m_Words[i] |= other.m_Words[i];

  return *this;
}

int AtomSelection::getNumAtoms(void) const { return m_NumAtoms; }

int AtomSelection::getNumSelected(void) const {
  int count = 0;

  for (const std::uint64_t word : m_Words)
    count += AtomSelection::countBits(word);

  return count;
}

std::vector<int> AtomSelection::getAtomIndices(void) const {
  std::vector<int> atomIndices;
  atomIndices.reserve(static_cast<std::size_t>(this->getNumSelected()));

  for (int i = 0; i < m_NumAtoms; i++) {
    if (this->contains(i) == true)
      atomIndices.push_back(i);
  }

  return atomIndices;
}

void AtomSelection::setNumAtoms(const int numAtoms,
                                const InitialValue initialValue) {
  APOCHARMM_REQUIRE(numAtoms >= 0, ApoCharmmErrorCode::InvalidArgument,
                    "Atom count must be non-negative; observed " +
                        std::to_string(numAtoms));

  std::vector<std::uint64_t> words(AtomSelection::getWordCount(numAtoms),
                                   (initialValue == InitialValue::ALL)
                                       ? ~std::uint64_t{0}
                                       : std::uint64_t{0});

  m_NumAtoms = numAtoms;
  m_Words.swap(words);
  this->maskUnusedBits();

  return;
}

bool AtomSelection::contains(const int atomIndex) const {
  this->checkIndex(atomIndex);

  const int wordIndex = atomIndex / 64;
  const int bitIndex = atomIndex % 64;

  return ((m_Words[wordIndex] & (std::uint64_t{1} << bitIndex)) != 0);
}

void AtomSelection::set(const int atomIndex, const bool isSelected) {
  this->checkIndex(atomIndex);

  const int wordIndex = atomIndex / 64;
  const int bitIndex = atomIndex % 64;

  if (isSelected == true)
    m_Words[wordIndex] |= (std::uint64_t{1} << bitIndex);
  else
    m_Words[wordIndex] &= ~(std::uint64_t{1} << bitIndex);

  return;
}

void AtomSelection::clear(void) {
  for (std::uint64_t &word : m_Words)
    word = std::uint64_t{0};
  return;
}

void AtomSelection::fill(void) {
  for (std::uint64_t &word : m_Words)
    word = ~std::uint64_t{0};
  this->maskUnusedBits();
  return;
}

std::size_t AtomSelection::getWordCount(const int numAtoms) {
  return (static_cast<std::size_t>((numAtoms) + std::size_t{63}) /
          std::size_t{64});
}

int AtomSelection::countBits(std::uint64_t word) {
  int count = 0;

  while (word != 0) {
    word &= (word - 1);
    count++;
  }

  return count;
}

void AtomSelection::checkIndex(const int atomIndex) const {
  APOCHARMM_REQUIRE((atomIndex >= 0) && (atomIndex < m_NumAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Atom index is out of range; expected [0, " +
                        std::to_string(m_NumAtoms) + "), observed " +
                        std::to_string(atomIndex));
  return;
}

void AtomSelection::checkCompatible(const AtomSelection &other) const {
  APOCHARMM_REQUIRE(
      m_NumAtoms == other.m_NumAtoms, ApoCharmmErrorCode::InvalidArgument,
      "Selection atom count mismatch; expected " + std::to_string(m_NumAtoms) +
          ", observed " + std::to_string(other.m_NumAtoms));
  return;
}

void AtomSelection::maskUnusedBits(void) {
  if ((m_NumAtoms <= 0) || (m_Words.empty())) // Nothing to mask
    return;

  // JEG260518: This should be updated if std::uint64_t is ever changed
  const int numUsedBitsInLastWord = m_NumAtoms % 64;

  if (numUsedBitsInLastWord == 0) // Nothing to mask
    return;

  // Generate a bit mask that is all 0's except for the bits that are used
  const std::uint64_t mask =
      (std::uint64_t{1} << numUsedBitsInLastWord) - std::uint64_t{1};

  // Preserves used bits in last word and sets all unused bits to 0
  m_Words.back() &= mask;

  return;
}
