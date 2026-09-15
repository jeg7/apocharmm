// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

/**
 * @file
 * @brief Declares a topology-aware reference to one atom.
 */

#pragma once

#include "CharmmPSF.h"

#include <memory>

/**
 * @brief Retains one zero-based atom index and shared ownership of its
 * topology.
 *
 * `AtomReference` stores a valid atom index together with a
 * `std::shared_ptr<const CharmmPSF>` for the native topology that defines that
 * index. It does not retain an @ref AtomSelection, allocate per-atom storage,
 * store coordinates, or perform CUDA work.
 *
 * Topology identity is native object identity. Two references have the same
 * topology only when their shared pointers address the same @ref CharmmPSF
 * object. Equality additionally requires the same atom index.
 *
 * The object exposes no setters. Copy operations share ownership of the same
 * topology, while move operations transfer the stored shared pointer.
 * Whole-object copy and move assignment remain available. A moved-from object
 * remains valid for destruction or assignment, but its query results are
 * unspecified.
 *
 * The class provides no internal locking. Concurrent read-only queries are
 * valid only while no thread assigns to or destroys either reference and no
 * thread mutates the shared topology through another owner.
 *
 * @see AtomSelector::selectAtom
 */
class AtomReference {
public:
  /**
   * @brief Prevents construction without a topology and atom index.
   */
  AtomReference(void) = delete;

  /**
   * @brief Constructs a reference to one atom in a native PSF.
   *
   * Validation is performed in this order: the PSF pointer must be non-null,
   * its atom count must be initialized and non-negative, and `atomIndex` must
   * be in `[0, psf->getNumAtoms())`.
   *
   * @param[in] psf Shared pointer to the originating topology. The pointer is
   * retained without cloning the PSF.
   * @param[in] atomIndex Zero-based, dimensionless atom index.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if `psf`
   * is null.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::NotInitialized` if the PSF
   * atom count is negative.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `atomIndex` is outside `[0, psf->getNumAtoms())`.
   * @throws std::bad_alloc If diagnostic allocation fails.
   * @throws std::length_error If a diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post The new object retains shared ownership of the supplied native PSF.
   * @post `getAtomIndex() == atomIndex`.
   */
  AtomReference(std::shared_ptr<const CharmmPSF> psf, const int atomIndex);

  /**
   * @brief Constructs a reference with the same topology and atom index.
   *
   * @param[in] other Reference to copy. The source remains unchanged.
   * @post The new object compares equal to `other` and shares topology
   * ownership with it.
   */
  AtomReference(const AtomReference &other) noexcept = default;

  /**
   * @brief Constructs a reference by moving another reference.
   *
   * @param[in,out] other Reference whose stored shared pointer is transferred.
   * @post The new object has the topology and atom index held by `other` before
   * the move. The moved-from object remains valid for destruction or
   * assignment.
   */
  AtomReference(AtomReference &&other) noexcept = default;

public:
  /**
   * @brief Replaces this value with a copy of another reference.
   *
   * @param[in] other Reference to copy. The source remains unchanged.
   * @return A borrowed mutable reference to this object.
   * @post This object compares equal to `other` and shares topology ownership
   * with it.
   */
  AtomReference &operator=(const AtomReference &other) noexcept = default;

  /**
   * @brief Replaces this value by moving another reference.
   *
   * @param[in,out] other Reference whose stored shared pointer is transferred.
   * @return A borrowed mutable reference to this object.
   * @post This object has the topology and atom index held by `other` before
   * the move. The moved-from object remains valid for destruction or
   * assignment.
   */
  AtomReference &operator=(AtomReference &&other) noexcept = default;

public:
  /**
   * @brief Returns the referenced atom index.
   *
   * @return The zero-based, dimensionless atom index supplied at construction
   * or assignment.
   */
  int getAtomIndex(void) const noexcept;

  /**
   * @brief Returns shared ownership of the originating topology.
   *
   * @return A copied `std::shared_ptr<const CharmmPSF>` addressing the same
   * native PSF object retained by this reference.
   */
  std::shared_ptr<const CharmmPSF> getPsf(void) const noexcept;

  /**
   * @brief Tests whether another reference originates from the same topology.
   *
   * @param[in] other Reference whose native PSF object identity is compared.
   * @return `true` only when both shared pointers address the same native
   * @ref CharmmPSF object; otherwise `false`.
   */
  bool hasSameTopology(const AtomReference &other) const noexcept;

  /**
   * @brief Tests topology-aware atom-reference equality.
   *
   * @param[in] other Reference to compare.
   * @return `true` only when both references have the same native PSF object
   * identity and the same atom index; otherwise `false`.
   */
  bool operator==(const AtomReference &other) const noexcept;

  /**
   * @brief Tests topology-aware atom-reference inequality.
   *
   * @param[in] other Reference to compare.
   * @return `true` when the references differ in native PSF object identity,
   * atom index, or both; otherwise `false`.
   */
  bool operator!=(const AtomReference &other) const noexcept;

private:
  /** Shares ownership of the originating immutable topology. */
  std::shared_ptr<const CharmmPSF> m_Psf;

  /** Stores the zero-based atom index in the originating topology. */
  int m_AtomIndex;
};
