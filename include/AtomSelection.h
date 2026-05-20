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

#include <cstdint>
#include <vector>

// JEG260518
// We use an array of uint64_t (8 byte) variables as a bit array. This is done
// for two reasons. 1.) We can decrease memory usage of atom selections by a
// factor of 32x treating each bit as an atom. 2.) For logical selection
// operations (e.g. .NOT., .AND., .OR.) we can operate on sets of 64 atoms at
// once improving performance. The code for manipulating the bits and things is
// all here.

class AtomSelection {
public:
  enum class InitialValue { NONE, ALL };

public:
  AtomSelection(void) = delete;
  AtomSelection(const int numAtoms,
                const InitialValue initialValue = InitialValue::NONE);
  AtomSelection(const AtomSelection &other);
  AtomSelection(const AtomSelection &&other);

public:
  AtomSelection &operator=(const AtomSelection &other);
  AtomSelection &operator=(const AtomSelection &&other);
  AtomSelection &operator&=(const AtomSelection &other);
  AtomSelection &operator|=(const AtomSelection &other);

public:
  int getNumAtoms(void) const;
  int getNumSelected(void) const;
  std::vector<int> getAtomIndices(void) const;

public:
  void setNumAtoms(const int numAtoms,
                   const InitialValue initialValue = InitialValue::NONE);

public:
  bool contains(const int atomIndex) const;
  void set(const int atomIndex, const bool isSelected = true);

  void clear(void);
  void fill(void);

private:
  static std::size_t getWordCount(const int numAtoms);
  static int countBits(std::uint64_t word);

  void checkIndex(const int atomIndex) const;
  void checkCompatible(const AtomSelection &other) const;
  void maskUnusedBits(void);

private:
  int m_NumAtoms;
  std::vector<std::uint64_t> m_Words;
};

inline AtomSelection operator&(AtomSelection left, const AtomSelection &right) {
  left &= right;
  return left;
}

inline AtomSelection operator|(AtomSelection left, const AtomSelection &right) {
  left |= right;
  return left;
}
