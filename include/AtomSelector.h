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

#include "AtomReference.h"
#include "AtomSelection.h"
#include "CharmmPSF.h"

#include <memory>
#include <string_view>

/**
 * @brief Evaluates CHARMM-style atom-selection expressions against a PSF.
 *
 * `AtomSelector` retains shared ownership of one const @ref CharmmPSF and uses
 * its host-resident atom metadata, residue and group intervals, and direct
 * bonded-connectivity table. Each call to `select()` tokenizes and parses the
 * supplied expression synchronously, then returns an independent
 * @ref AtomSelection. Each call to `selectAtom()` delegates to `select()`,
 * requires exactly one selected atom, and returns an @ref AtomReference.
 * AtomSelection results retain neither this selector nor its PSF. AtomReference
 * results share ownership of the same native PSF object.
 *
 * The selector does not snapshot the topology. Mutating the shared PSF through
 * another non-const owner can change later results or violate parser
 * preconditions. No CUDA allocation, transfer, stream operation, or
 * synchronization is performed by this class.
 *
 * Compiler-generated copy operations share the same PSF; generated move
 * operations transfer that shared pointer. The class provides no internal
 * locking. Concurrent calls are valid only while the retained PSF remains
 * immutable and no thread destroys the selector.
 *
 * @see atom_selection
 * @see SelectionTokenizer
 * @see SelectionParser
 */
class AtomSelector {
public:
  /**
   * @brief Prevents construction without a topology.
   */
  AtomSelector(void) = delete;

  /**
   * @brief Constructs a selector that shares ownership of a PSF.
   *
   * @param[in] psf Shared pointer to the topology used by every future
   * selection. The pointer is copied and retained; the PSF is not cloned and
   * may not be null.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if `psf`
   * is null.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::NotInitialized` if the PSF
   * atom count is negative.
   * @throws std::bad_alloc If retained ownership or diagnostic allocation
   * fails.
   * @throws std::length_error If a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post The selector retains shared ownership of the same native PSF object.
   */
  explicit AtomSelector(std::shared_ptr<const CharmmPSF> psf);

public:
  /**
   * @brief Evaluates an atom-selection expression against the retained PSF.
   *
   * Tokenization is case-insensitive for recognized keywords and dotted
   * operators. Matching, ranges, precedence, expansion operators, and wildcard
   * behavior are defined on @ref atom_selection. The input view is borrowed
   * only for this call; token text is copied before parsing and no view is
   * retained.
   *
   * @param[in] selectionString Expression bytes to parse. The view may refer to
   * non-null-terminated storage. Embedded control bytes, including `\0`, are
   * rejected by the tokenizer.
   * @return A newly owned selection with the same atom count as the retained
   * PSF. The result is independent of this selector and the PSF.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` for a
   * lexical or syntax error, an unknown operator, an invalid `BYNU` range,
   * unmatched parentheses, or an out-of-range neighbor stored in the PSF
   * bonded-connectivity table.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Runtime` if retained PSF
   * residue, group, or bonded-connectivity state violates parser invariants, or
   * if an internal token or operator-stack invariant fails.
   * @throws std::bad_alloc If token, parser-stack, index, selection, result, or
   * diagnostic allocation fails.
   * @throws std::length_error If token text, a parser container, a selection,
   * the result, or a diagnostic exceeds an implementation-defined limit.
   *
   * @pre The retained PSF remains alive, has a non-negative atom count, and has
   * per-atom metadata arrays consistent with that count.
   * @post The retained PSF and this selector are unchanged.
   * @note The parser reads the host mirrors of residue and group containers. It
   * performs no host-to-device or device-to-host transfer and does not
   * synchronize a CUDA stream.
   */
  AtomSelection select(const std::string_view selectionString) const;

  /**
   * @brief Evaluates an expression that must select exactly one atom.
   *
   * This method delegates tokenization, parsing, and all selection-language
   * behavior to `select()`. It requires exactly one selected atom, copies the
   * sole zero-based index, releases the temporary @ref AtomSelection storage,
   * and returns an @ref AtomReference that retains shared ownership of the
   * selector's PSF.
   *
   * @param[in] selectionString Expression bytes to parse. The view has the same
   * borrowing, termination, and control-byte behavior as for `select()`.
   * @return A topology-aware reference to the sole selected atom. The result
   * remains valid after this selector and the caller's other shared owners of
   * the PSF are destroyed.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` for any
   * lexical, syntax, range, or connectivity-index error reported by `select()`,
   * or when the resulting selection contains zero or multiple atoms.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Runtime` for any malformed
   * retained PSF or internal parser state reported by `select()`.
   * @throws std::bad_alloc If token, parser-stack, selection, index-vector, or
   * diagnostic allocation fails.
   * @throws std::length_error If token text, a parser container, selection
   * storage, an index vector, or a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @pre The retained PSF satisfies the same requirements as for `select()`.
   * @post The retained PSF and this selector are unchanged.
   * @post No temporary @ref AtomSelection or its bitset storage is retained by
   * the returned reference.
   */
  AtomReference selectAtom(const std::string_view selectionString) const;

private:
  /** Shares ownership of the immutable topology used for selection. */
  std::shared_ptr<const CharmmPSF> m_Psf;
};
