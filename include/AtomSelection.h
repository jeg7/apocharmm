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

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief Owns a compact host-resident set of zero-based atom indices.
 *
 * `AtomSelection` associates one selection bit with every atom in the range
 * `[0, getNumAtoms())`. Internally, consecutive groups of 64 atoms occupy one
 * `std::uint64_t`; atom `i` uses word `i / 64` and the bit at offset `i % 64`.
 * Unused high bits in the final word are always cleared. Counts and indices are
 * dimensionless.
 *
 * The object exclusively owns its host storage. Copy construction and copy
 * assignment create independent storage. Move construction and move assignment
 * transfer the owned word storage without allocation and leave the source as a
 * valid zero-atom selection. Logical operations require both operands to
 * describe the same atom count. Resizing resets the complete selection rather
 * than preserving an overlap.
 *
 * This class performs no CUDA allocation, transfer, stream operation, or
 * synchronization. The compiler-generated destructor only releases host
 * storage. The class provides no internal locking: concurrent read-only calls
 * are valid only while no thread mutates or destroys the object.
 *
 * @note The normal representation invariant is
 * `m_Words.size() == ceil(getNumAtoms() / 64)` with all out-of-range bits zero.
 * @warning Copy assignment is sequential rather than transactional. If vector
 * assignment fails after the atom count changes, the destination can violate
 * its representation invariant.
 * @see atom_selection
 */
class AtomSelection {
public:
  /**
   * @brief Selects the value assigned to every atom during initialization.
   */
  enum class InitialValue {
    /** Leaves every valid atom index unselected. */
    NONE,
    /** Selects every valid atom index. */
    ALL
  };

public:
  /**
   * @brief Prevents construction without an explicit atom count.
   */
  AtomSelection(void) = delete;

  /**
   * @brief Constructs a selection for a fixed number of atoms.
   *
   * The object allocates enough host words to represent `numAtoms` bits and
   * initializes every valid bit according to `initialValue`.
   *
   * @param[in] numAtoms Dimensionless atom count. The value must be
   * non-negative.
   * @param[in] initialValue Initial selection state applied to every atom.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `numAtoms` is negative.
   * @throws std::bad_alloc If bit-storage or diagnostic allocation fails.
   * @throws std::length_error If the requested storage or diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post `getNumAtoms() == numAtoms`.
   * @post `getNumSelected()` is zero for `InitialValue::NONE` and `numAtoms`
   * for `InitialValue::ALL`.
   */
  AtomSelection(const int numAtoms,
                const InitialValue initialValue = InitialValue::NONE);

  /**
   * @brief Constructs an independent copy of another selection.
   *
   * @param[in] other Selection borrowed for the duration of construction. No
   * reference to it is retained.
   * @throws std::bad_alloc If the owned word vector cannot be copied.
   * @throws std::length_error If the copied storage exceeds an
   * implementation-defined limit.
   *
   * @post The new object has the same atom count and selected indices as
   * `other`, without aliasing its storage.
   */
  AtomSelection(const AtomSelection &other);

  /**
   * @brief Constructs a selection by transferring another selection's storage.
   *
   * No word storage is allocated or copied. The source is reset to the valid
   * zero-atom representation.
   *
   * @param[in,out] other Selection whose owned word storage is transferred.
   *
   * @post The new object has the same atom count and selected indices that
   * `other` had before construction.
   * @post `other.getNumAtoms() == 0` and `other.getNumSelected() == 0`.
   * @note This operation performs no allocation and does not throw.
   */
  AtomSelection(AtomSelection &&other) noexcept;

public:
  /**
   * @brief Replaces this selection with an independent copy.
   *
   * @param[in] other Selection borrowed for the duration of assignment. No
   * reference to it is retained.
   * @return A borrowed mutable reference to this object.
   * @throws std::bad_alloc If the owned word vector cannot be copied.
   * @throws std::length_error If the copied storage exceeds an
   * implementation-defined limit.
   *
   * @post On success, this object has the same atom count and selected indices
   * as `other` without aliasing its storage.
   * @warning The atom count is assigned before the word vector. A failed vector
   * assignment can leave this object with a new count and old word storage.
   */
  AtomSelection &operator=(const AtomSelection &other);

  /**
   * @brief Replaces this selection by transferring another selection's storage.
   *
   * The destinations's previous word storage is released through normal
   * non-throwing container destruction. Self-move assignment is a no-op.
   *
   * @param[in,out] other Selection whose owned word storage is transferred.
   * @return A borrowed mutable reference to this object.
   *
   * @post For distinct objects, this object has the atom count and selected
   * indices that `other` had before assignment.
   * @post For distinct objects, `other.getNumAtoms() == 0` and
   * `other.getNumSelected() == 0`.
   * @post Self-move assignment leaves the object unchanged.
   * @note This operation performs no allocation and does not throw.
   */
  AtomSelection &operator=(AtomSelection &&other) noexcept;

  /**
   * @brief Intersects this selection with another selection.
   *
   * @param[in] other Selection whose bits are combined with this object. The
   * operand is borrowed and remains unchanged.
   * @return A borrowed mutable reference to this object.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if the
   * atom counts differ.
   * @throws std::bad_alloc If mismatch-diagnostic allocation fails.
   * @throws std::length_error If a mismatch diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, an atom is selected exactly when it was selected in both
   * operands.
   * @post If the atom counts differ, this object is unchanged.
   */
  AtomSelection &operator&=(const AtomSelection &other);

  /**
   * @brief Unites this selection with another selection.
   *
   * @param[in] other Selection whose bits are combined with this object. The
   * operand is borrowed and remains unchanged.
   * @return A borrowed mutable reference to this object.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if the
   * atom counts differ.
   * @throws std::bad_alloc If mismatch-diagnostic allocation fails.
   * @throws std::length_error If a mismatch diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, an atom is selected when it was selected in either
   * operand.
   * @post If the atom counts differ, this object is unchanged.
   */
  AtomSelection &operator|=(const AtomSelection &other);

public:
  /**
   * @brief Returns the represented atom count.
   *
   * @return The non-negative, dimensionless count established at construction
   * or by `setNumAtoms()`.
   */
  int getNumAtoms(void) const;

  /**
   * @brief Returns the number of selected atoms.
   *
   * @return A dimensionless value in `[0, getNumAtoms()]` computed from the
   * stored bits.
   */
  int getNumSelected(void) const;

  /**
   * @brief Returns all selected atom indices in ascending order.
   *
   * @return A newly owned host vector of zero-based indices. Its length equals
   * `getNumSelected()`, and it does not alias this object.
   * @throws std::bad_alloc If the result vector cannot allocate storage.
   * @throws std::length_error If the result exceeds an implementation-defined
   * vector limit.
   */
  std::vector<int> getAtomIndices(void) const;

public:
  /**
   * @brief Resets the selection for a new atom count.
   *
   * This operation replaces all existing bits. It does not preserve selected
   * atoms from the old range.
   *
   * @param[in] numAtoms New dimensionless atom count. The value must be
   * non-negative.
   * @param[in] initialValue State assigned to every atom in the new range.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `numAtoms` is negative.
   * @throws std::bad_alloc If replacement storage or diagnostic allocation
   * fails.
   * @throws std::length_error If replacement storage or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, `getNumAtoms() == numAtoms` and the selected count is
   * either zero or `numAtoms` according to `initialValue`.
   * @post On failure, the previous atom count and selected bits are unchanged.
   */
  void setNumAtoms(const int numAtoms,
                   const InitialValue initialValue = InitialValue::NONE);

public:
  /**
   * @brief Tests whether one atom index is selected.
   *
   * @param[in] atomIndex Zero-based, dimensionless atom index. The value must
   * be in `[0, getNumAtoms())`.
   * @return `true` when the corresponding bit is set; otherwise `false`.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `atomIndex` is outside the represented range.
   * @throws std::bad_alloc If range-error diagnostic allocation fails.
   * @throws std::length_error If a range-error diagnostic exceeds an
   * implementation-defined limit.
   */
  bool contains(const int atomIndex) const;

  /**
   * @brief Sets or clears one atom-selection bit.
   *
   * @param[in] atomIndex Zero-based, dimensionless atom index. The value must
   * be in `[0, getNumAtoms())`.
   * @param[in] isSelected `true` to select the atom or `false` to clear it.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `atomIndex` is outside the represented range.
   * @throws std::bad_alloc If range-error diagnostic allocation fails.
   * @throws std::length_error If a range-error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post On success, `contains(atomIndex) == isSelected` and all other bits
   * are unchanged.
   * @post If `atomIndex` is invalid, the selection is unchanged.
   */
  void set(const int atomIndex, const bool isSelected = true);

  /**
   * @brief Clears every atom-selection bit.
   *
   * @post `getNumSelected() == 0` and the atom count is unchanged.
   */
  void clear(void);

  /**
   * @brief Selects every represented atom.
   *
   * @post `getNumSelected() == getNumAtoms()` and unused final-word bits remain
   * zero.
   */
  void fill(void);

private:
  /**
   * @brief Computes the number of 64-bit words required for an atom count.
   *
   * @param[in] numAtoms Non-negative, dimensionless atom count.
   * @return `ceil(numAtoms / 64)` without allocating storage.
   */
  static std::size_t getWordCount(const int numAtoms);

  /**
   * @brief Counts set bits in one storage word.
   *
   * @param[in] word Dimensionless bit word copied by value.
   * @return The number of set bits in `[0, 64]`.
   */
  static int countBits(std::uint64_t word);

  /**
   * @brief Validates one atom index against the represented range.
   *
   * @param[in] atomIndex Zero-based, dimensionless atom index to validate.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` when the
   * index is outside `[0, m_NumAtoms)`.
   */
  void checkIndex(const int atomIndex) const;

  /**
   * @brief Validates that another selection has the same atom count.
   *
   * @param[in] other Selection borrowed for validation.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` when the
   * atom counts differ.
   */
  void checkCompatible(const AtomSelection &other) const;

  /**
   * @brief Clears out-of-range bits in the final storage word.
   *
   * @post Every bit representing an index greater than or equal to
   * `m_NumAtoms` is zero.
   */
  void maskUnusedBits(void);

private:
  /** Stores the non-negative, dimensionless number of represented atoms. */
  int m_NumAtoms;
  /** Owns the host bit words in ascending groups of 64 atom indices. */
  std::vector<std::uint64_t> m_Words;
};

/**
 * @brief Returns the intersection of two selections.
 *
 * The left operand is copied by value and modified; neither caller-owned input
 * is mutated.
 *
 * @param[in] left Selection copied into the owned result.
 * @param[in] right Selection borrowed for the operation.
 * @return A newly owned selection containing indices present in both operands.
 * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if the atom
 * counts differ.
 * @throws std::bad_alloc If result-copy or diagnostic allocation fails.
 * @throws std::length_error If copied storage or a diagnostic exceeds an
 * implementation-defined limit.
 */
inline AtomSelection operator&(AtomSelection left, const AtomSelection &right) {
  left &= right;
  return left;
}

/**
 * @brief Returns the union of two selections.
 *
 * The left operand is copied by value and modified; neither caller-owned input
 * is mutated.
 *
 * @param[in] left Selection copied into the owned result.
 * @param[in] right Selection borrowed for the operation.
 * @return A newly owned selection containing indices present in either operand.
 * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if the atom
 * counts differ.
 * @throws std::bad_alloc If result-copy or diagnostic allocation fails.
 * @throws std::length_error If copied storage or a diagnostic exceeds an
 * implementation-defined limit.
 */
inline AtomSelection operator|(AtomSelection left, const AtomSelection &right) {
  left |= right;
  return left;
}
