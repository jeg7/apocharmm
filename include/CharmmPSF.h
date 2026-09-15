// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "CudaContainer.h"

#include <filesystem>
#include <set>
#include <string>
#include <vector>
#include <vector_types.h>

/**
 * @brief Stores one covalent bond as two atom indices.
 *
 * PSF parsing converts both atom numbers from the file's one-based convention
 * to zero-based indices into the per-atom arrays owned by @ref CharmmPSF.
 *
 * @note Direct aggregate construction and mutation perform no validation.
 */
struct Bond {
  /**
   * @brief Stores the zero-based endpoint indices in `iatom`, `jatom` order.
   */
  int iatom, jatom;
};

/**
 * @brief Stores one valence angle as three atom indices.
 *
 * PSF parsing converts all atom numbers from the file's one-based convention to
 * zero-based indices into the per-atom arrays owned by @ref CharmmPSF. `jatom`
 * is the central atom.
 *
 * @note Direct aggregate construction and mutation perform no validation.
 */
struct Angle {
  /**
   * @brief Stores the zero-based indices in `iatom`, `jatom`, `katom` order.
   */
  int iatom, jatom, katom;
};

/**
 * @brief Stores one proper or improper dihedral as four atom indices.
 *
 * PSF parsing converts all atom numbers from the file's one-based convention to
 * zero-based indices into the per-atom arrays owned by @ref CharmmPSF.
 *
 * @note Direct aggregate construction and mutation perform no validation.
 */
struct Dihedral {
  /**
   * @brief Stores the zero-based indices in file order from `iatom` through
   * `latom`.
   */
  int iatom, jatom, katom, latom;
};

/**
 * @brief Stores the two four-atom tuples in one PSF cross-term record.
 *
 * Each tuple identifies one of the coupled dihedrals in a CHARMM CMAP
 * cross-term. The current CharmmPSF parser validates every atom number against
 * the atom count.
 *
 * @warning The current implementation retains these eight values in the PSF
 * file's one-based convention. This differs from @ref Bond, @ref Angle, and
 * @ref Dihedral and is tracked as an implementation defect rather than a stable
 * indexing guarantee.
 *
 * @note Direct aggregate construction and mutation perform no validation.
 */
struct CrossTerm {
  /**
   * @brief Stores the first tuple in `iatom1`, `jatom1`, `katom1`, `latom1`
   * order.
   */
  int iatom1, jatom1, katom1, latom1;

  /**
   * @brief Stores the second tuple in `iatom2`, `jatom2`, `katom2`, `latom2`
   * order.
   */
  int iatom2, jatom2, katom2, latom2;
};

/**
 * @brief Owns the flattened inclusion and exclusion lists derived from a PSF.
 *
 * Values returned by @ref CharmmPSF::getInclusionExclusionLists contain two
 * pair counts and one concatenated zero-based atom-index array. Inclusion pairs
 * precede exclusion pairs. This value type owns independent host-side copies
 * and performs no CUDA allocation or transfer.
 */
struct InclusionExclusion {
  /**
   * @brief Constructs an owned inclusion/exclusion value from two host vectors.
   *
   * The constructor copies both inputs without validating their lengths or
   * relationship.
   *
   * @param[in] _sizes Pair-count vector to copy.
   * @param[in] _in14_ex14 Flattened atom-index vector to copy.
   * @throws std::bad_alloc If either vector copy cannot allocate storage.
   * @throws std::length_error If either input exceeds an implementation-defined
   * vector size limit.
   *
   * @post `sizes` and `in14_ex14` are independent copies of the inputs.
   */
  InclusionExclusion(const std::vector<int> &_sizes,
                     const std::vector<int> &_in14_ex14)
      : sizes(_sizes), in14_ex14(_in14_ex14) {}

  /**
   * @brief Stores the dimensionless inclusion and exclusion pair counts.
   *
   * For values returned by CharmmPSF, the vector has length two:
   * `sizes[0]` is the number of inclusion pairs and `sizes[1]` is the number of
   * exclusion pairs.
   */
  std::vector<int> sizes;

  /**
   * @brief Stores concatenated zero-based atom-index pairs.
   *
   * For values returned by CharmmPSF, the vector length is
   * `2 * (sizes[0] + sizes[1])`. The first `2 * sizes[0]` elements are
   * inclusion pairs. The remaining elements are exclusion pairs.
   */
  std::vector<int> in14_ex14;
};

/**
 * @brief Owns CHARMM PSF topology, per-atom metadata, and derived topology
 * tables.
 *
 * A CharmmPSF stores segment and residue identifiers, atom names and types,
 * charges, masses, bonded topology, topological connectivity, CHARMM-style
 * exclusion arrays, residue intervals, connected-component intervals, and
 * recognized water molecules. Per-atom metadata and bonded records are
 * host-resident. Residue, connected-component, and water records are stored in
 * @ref CudaContainer objects with independent host and device mirrors.
 *
 * The file constructor parses the PSF synchronously on the calling host thread.
 * It then derives residue intervals, water tuples, connected components, and
 * topological exclusions. The object stores the supplied path but does not
 * retain a reference to the caller's string.
 *
 * Parsed bonds, angles, proper dihedrals, and improper dihedrals use zero-based
 * atom indices. The current cross-term parser retains one-based indices; see
 * @ref CrossTerm and @ref charmm_psf for the associated implementation defect.
 *
 * Const accessors return borrowed references to object-owned storage. Mutable
 * accessors are unchecked escape hatches: they do not update stored counts,
 * dependent tables, or host/device coherence. See @ref charmm_psf_mutation.
 *
 * Copy construction deep-copies all host vectors and both mirrors of each
 * CudaContainer. Copy assignment is explicitly defaulted and performs
 * sequential memberwise assignment, so a later allocation or CUDA failure can
 * leave partially assigned state. Move construction and move assignment
 * transfer all owned host and device storage and leave the source in the exact
 * default-constructed state.
 *
 * The compiler-generated destructor is non-throwing. Nested CUDA-owning
 * containers use non-throwing destruction and discard CUDA cleanup failures
 * during destruction.
 *
 * The class provides no internal locking. Callers must externally synchronize
 * concurrent mutation, destruction, or access that can overlap CUDA operations.
 *
 * @warning Parsed and derived state is consistent only until unchecked mutable
 * access or a partial low-level reconfiguration changes one component without
 * rebuilding its dependents.
 *
 * @see charmm_psf
 */
class CharmmPSF {
public:
  /**
   * @brief Constructs an uninitialized, empty PSF object.
   *
   * The atom, bond, angle, proper-dihedral, improper-dihedral, and cross-term
   * counts are initialized to `-1`. All host vectors and CudaContainer members
   * are empty, and the stored file name is empty.
   *
   * @post Aggregate methods requiring an initialized atom count report
   * `ApoCharmmErrorCode::NotInitialized`.
   */
  CharmmPSF(void);

  /**
   * @brief Constructs and derives topology from a CHARMM PSF file.
   *
   * The file is read completely into host memory and parsed in section order.
   * The parser requires TITLE, ATOM, BOND, ANGLE, DIHEDRAL, IMPROPER, DONOR,
   * ACCEPTOR, and CROSS-TERM sections. Donor and acceptor records, and sections
   * between ACCEPTOR and CROSS-TERM, are skipped rather than retained.
   *
   * Atom metadata is stored in file-record order. Charges and masses must be
   * finite. Topology atom numbers must be in the one-based range
   * `[1, getNumAtoms()]`. After parsing, the constructor derives water,
   * connected-component, residue, and exclusion data.
   *
   * @param[in] filePath Borrowed non-empty path to a PSF file. The path is
   * copied into the object, retained without canonicalization, and not retained
   * by reference.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `filePath` is empty.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened or read, a required section or record is missing, a record
   * has an invalid field count, a numeric field is malformed or non-finite, a
   * topology atom number is out of range, or a supported section count exceeds
   * `int`.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if allocation,
   * copying, launch checking, or cleanup for a derived CudaContainer fails.
   * @throws std::bad_alloc If host-side parsing, topology storage, or
   * diagnostic construction cannot allocate memory.
   * @throws std::length_error If a file, container, or diagnostic exceeds an
   * implementation-defined size limit.
   *
   * @post On success, all stored nonnegative counts and parser-owned vectors
   * describe the parsed PSF, and all constructor-derived tables have been
   * created.
   * @note Construction does not select a CUDA device or stream and does not
   * issue a CharmmPSF-level device-wide synchronization.
   */
  CharmmPSF(const std::filesystem::path &filePath);

  /**
   * @brief Constructs an independent deep copy of another PSF object.
   *
   * All scalar values and host vectors are copied. Each CudaContainer copies
   * its host and device mirrors independently, preserving any pre-existing
   * host/device divergence in `other`.
   *
   * @param[in] other PSF object to copy. The source remains unchanged and is
   * not retained.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if copying a
   * CudaContainer device allocation fails.
   * @throws std::bad_alloc If a host or device-copy diagnostic allocation
   * fails.
   * @throws std::length_error If copied storage exceeds an
   * implementation-defined limit.
   *
   * @post On success, the new object owns storage independent of `other`.
   */
  CharmmPSF(const CharmmPSF &other);

  /**
   * @brief Constructs a PSF by transferring another PSF's complete state.
   *
   * Host vectors, connectivity sets, file-path storage, and all three
   * CudaContainer mirrors are transferred without allocation, copying,
   * transfer, or synchronization.
   *
   * @param[in,out] other PSF whose complete owned state is transferred.
   *
   * @post This object contains the state that `other` held before construction.
   * @post `other` is equivalent to a default-constructed `CharmmPSF`: all six
   * counts are `-1`, all host containers are empty, all CudaContainers are
   * empty, and the stored file path is empty.
   */
  CharmmPSF(CharmmPSF &&other) noexcept;

public:
  /**
   * @brief Replaces this PSF with an independent memberwise copy.
   *
   * @param[in] other PSF whose complete stored state is copied.
   * @return A borrowed mutable reference to this object.
   *
   * @warning Assignment is sequential rather than transactional. A later host
   * allocation or CUDA failure can leave earlier members assigned.
   */
  CharmmPSF &operator=(const CharmmPSF &other) = default;

  /**
   * @brief Replaces this PSF by transferring another PSF's complete state.
   *
   * The destination's previous state is released through normal non-throwing
   * destruction. Self-move assignment is a no-op.
   *
   * @param[in,out] other PSF whose complete owned state is transferred.
   * @return A borrowed mutable reference to this object.
   *
   * @post For distinct objects, this object contains the state that `other`
   * held before assignment.
   * @post For distinct objects, `other` is equivalent to a
   * default-constructed `CharmmPSF`.
   * @post Self-move assignment leaves the PSF unchanged.
   * @warning CUDA cleanup failure while releasing the destination's former
   * CudaContainer storage is discarded, consistent with destruction.
   */
  CharmmPSF &operator=(CharmmPSF &&other) noexcept;

public:
  /**
   * @brief Sets the atom count and resizes the seven per-atom host arrays.
   *
   * The segment identifiers, residue identifiers, residue names, atom names,
   * atom types, charges, and masses are resized to `numAtoms`. Newly added
   * strings are empty, newly added residue identifiers are zero, and newly
   * added charges and masses are `0.0`.
   *
   * This method does not change any bonded-topology count or record. It also
   * does not rebuild residue intervals, water tuples, connected components,
   * topological connectivity, or CHARMM exclusion arrays.
   *
   * @param[in] numAtoms New dimensionless atom count. The value must be
   * nonnegative.
   * @throws ApoCharmmError With
   * `ApoCharmmErrorCode::InvalidArgument` if `numAtoms` is negative.
   * @throws std::bad_alloc If resizing a per-atom host vector cannot allocate
   * storage.
   * @throws std::length_error If `numAtoms` exceeds a host vector's maximum
   * size.
   *
   * @post `getNumAtoms() == numAtoms` and every per-atom host vector has length
   * `numAtoms`.
   * @warning Existing element references, pointers, and iterators into a
   * resized vector can be invalidated.
   * @warning Calling this method on a parsed object can leave topology and
   * derived tables inconsistent with the new atom count.
   */
  void setNumAtoms(const int numAtoms);

  /**
   * @brief Replaces all per-atom partial charges.
   *
   * The input is copied into host-owned storage. No topology, aggregate cache,
   * CudaContainer, or device allocation is changed.
   *
   * @param[in] charges Host vector containing exactly `getNumAtoms()` finite or
   * non-finite `double` values in elementary-charge units. This setter does not
   * validate finiteness or physical range.
   * @throws ApoCharmmError With
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is negative.
   * @throws ApoCharmmError With
   * `ApoCharmmErrorCode::InvalidArgument` if `charges.size()` differs from the
   * initialized atom count.
   * @throws std::bad_alloc If copying the vector cannot allocate storage.
   * @throws std::length_error If the input exceeds an implementation-defined
   * vector size limit.
   *
   * @post On success, `getCharges()` contains an independent copy of `charges`.
   * @warning Existing references, pointers, and iterators into the charge
   * vector can be invalidated.
   */
  void setAtomCharges(const std::vector<double> &charges);

public:
  /**
   * @brief Returns the stored atom count.
   *
   * @return Dimensionless atom count, or `-1` for a default-constructed object
   * whose atom count has not been initialized.
   * @note The value is stored independently from the lengths of the mutable
   * per-atom vectors.
   */
  int getNumAtoms(void) const;

  /**
   * @brief Returns the stored bond count.
   *
   * @return Dimensionless bond count, or `-1` before a PSF has initialized the
   * bond section.
   * @note Unchecked mutation of `getBonds()` does not update this value.
   */
  int getNumBonds(void) const;

  /**
   * @brief Returns the stored angle count.
   *
   * @return Dimensionless angle count, or `-1` before a PSF has initialized the
   * angle section.
   * @note Unchecked mutation of `getAngles()` does not update this value.
   */
  int getNumAngles(void) const;

  /**
   * @brief Returns the stored proper-dihedral count.
   *
   * @return Dimensionless proper-dihedral count, or `-1` before a PSF has
   * initialized the proper-dihedral section.
   * @note Unchecked mutation of `getDihedrals()` does not update this value.
   */
  int getNumDihedrals(void) const;

  /**
   * @brief Returns the stored improper-dihedral count.
   *
   * @return Dimensionless improper-dihedral count, or `-1` before a PSF has
   * initialized the improper section.
   * @note Unchecked mutation of `getImpropers()` does not update this value.
   */
  int getNumImpropers(void) const;

  /**
   * @brief Returns the stored cross-term count.
   *
   * @return Dimensionless cross-term count, or `-1` before a PSF has
   * initialized the cross-term section.
   * @note Unchecked mutation of `getCrossTerms()` does not update this value.
   */
  int getNumCrossTerms(void) const;

  /**
   * @brief Returns the per-atom segment identifiers.
   *
   * @return Borrowed const reference to the host vector owned by this object.
   * Values are in atom-record order. No ownership is transferred, and the
   * reference must not outlive this CharmmPSF.
   * @note A normally parsed vector has `getNumAtoms()` entries, but unchecked
   * mutable access can violate that relationship.
   */
  const std::vector<std::string> &getSegmentIdentifiers(void) const;

  /**
   * @brief Returns the per-atom residue identifiers.
   *
   * @return Borrowed const reference to the host vector owned by this object.
   * Values are dimensionless integers in atom-record order. No ownership is
   * transferred, and the reference must not outlive this CharmmPSF.
   * @note The parser preserves the integer identifiers written in the PSF.
   */
  const std::vector<int> &getResidueIdentifiers(void) const;

  /**
   * @brief Returns the per-atom residue names.
   *
   * @return Borrowed const reference to the host vector owned by this object.
   * Values are in atom-record order. No ownership is transferred, and the
   * reference must not outlive this CharmmPSF.
   */
  const std::vector<std::string> &getResidueNames(void) const;

  /**
   * @brief Returns the per-atom atom names.
   *
   * @return Borrowed const reference to the host vector owned by this object.
   * Values are in atom-record order. No ownership is transferred, and the
   * reference must not outlive this CharmmPSF.
   */
  const std::vector<std::string> &getAtomNames(void) const;

  /**
   * @brief Returns the per-atom CHARMM atom types.
   *
   * @return Borrowed const reference to the host vector owned by this object.
   * Values are in atom-record order. No ownership is transferred, and the
   * reference must not outlive this CharmmPSF.
   */
  const std::vector<std::string> &getAtomTypes(void) const;

  /**
   * @brief Returns the per-atom partial charges.
   *
   * @return Borrowed const reference to the host vector owned by this object.
   * Elements are in atom-record order and use elementary-charge units. No
   * ownership is transferred, and the reference must not outlive this
   * CharmmPSF.
   * @note PSF parsing requires finite values but does not impose a physical
   * charge range.
   */
  const std::vector<double> &getCharges(void) const;

  /**
   * @brief Returns the per-atom masses.
   *
   * @return Borrowed const reference to the host vector owned by this object.
   * Elements are in atom-record order and use atomic mass units. No ownership
   * is transferred, and the reference must not outlive this CharmmPSF.
   * @note PSF parsing requires finite values but does not require positive
   * masses.
   */
  const std::vector<double> &getMasses(void) const;

  /**
   * @brief Returns the parsed covalent bonds.
   *
   * @return Borrowed const reference to object-owned host storage. Parsed
   * records contain zero-based atom indices in PSF order. No ownership is
   * transferred.
   */
  const std::vector<Bond> &getBonds(void) const;

  /**
   * @brief Returns the parsed valence angles.
   *
   * @return Borrowed const reference to object-owned host storage. Parsed
   * records contain zero-based atom indices in PSF order. No ownership is
   * transferred.
   */
  const std::vector<Angle> &getAngles(void) const;

  /**
   * @brief Returns the parsed proper dihedrals.
   *
   * @return Borrowed const reference to object-owned host storage. Parsed
   * records contain zero-based atom indices in PSF order. No ownership is
   * transferred.
   */
  const std::vector<Dihedral> &getDihedrals(void) const;

  /**
   * @brief Returns the parsed improper dihedrals.
   *
   * @return Borrowed const reference to object-owned host storage. Parsed
   * records contain zero-based atom indices in PSF order. No ownership is
   * transferred.
   */
  const std::vector<Dihedral> &getImpropers(void) const;

  /**
   * @brief Returns the parsed CMAP cross-term records.
   *
   * @return Borrowed const reference to object-owned host storage. No ownership
   * is transferred.
   * @warning The current parser stores these atom numbers in one-based PSF
   * form, unlike the other topology records. See @ref CrossTerm.
   */
  const std::vector<CrossTerm> &getCrossTerms(void) const;

  /**
   * @brief Returns the per-atom direct-bond connectivity sets.
   *
   * @return Borrowed const reference to a host vector normally containing one
   * set per atom. Each set stores zero-based atoms connected by one bond. No
   * ownership is transferred.
   */
  const std::vector<std::set<int>> &getConnected12(void) const;

  /**
   * @brief Returns the per-atom two-bond connectivity sets.
   *
   * @return Borrowed const reference to a host vector normally containing one
   * set per atom. Each set stores zero-based atoms reached by the current
   * two-bond construction. No ownership is transferred.
   */
  const std::vector<std::set<int>> &getConnected13(void) const;

  /**
   * @brief Returns the per-atom three-bond connectivity sets.
   *
   * @return Borrowed const reference to a host vector normally containing one
   * set per atom. Each set stores zero-based atoms reached by the current
   * three-bond construction. No ownership is transferred.
   * @note These sets are filtered against 1-2 and 1-3 connectivity when
   * `getInclusionExclusionLists()` produces explicit inclusion pairs.
   */
  const std::vector<std::set<int>> &getConnected14(void) const;

  /**
   * @brief Returns the cumulative CHARMM exclusion offsets.
   *
   * @return Borrowed const reference to object-owned host storage. A normally
   * parsed object has one dimensionless entry per atom. `getIblo14()[i]` is the
   * cumulative number of entries in `getInb14()` through atom `i`.
   */
  const std::vector<int> &getIblo14(void) const;

  /**
   * @brief Returns the flattened CHARMM exclusion atom numbers.
   *
   * @return Borrowed const reference to object-owned host storage. Entries use
   * CHARMM's one-based atom-number convention and include only neighbors whose
   * zero-based index is greater than the source atom's index.
   */
  const std::vector<int> &getInb14(void) const;

  /**
   * @brief Returns recognized three-site water tuples.
   *
   * @return Borrowed const reference to the object-owned
   * `CudaContainer<int4>`. Each host/device element stores zero-based
   * `(oxygen, hydrogen1, hydrogen2, 0)` components in `x`, `y`, `z`, `w`
   * order. No ownership is transferred.
   * @note File construction recognizes only consecutive atom types
   * `OT`, `HT`, `HT`.
   */
  const CudaContainer<int4> &getWaterMolecules(void) const;

  /**
   * @brief Returns contiguous residue intervals.
   *
   * @return Borrowed const reference to the object-owned
   * `CudaContainer<int2>`. Each element stores a zero-based inclusive
   * `[x, y]` atom interval. No ownership is transferred.
   * @warning The current parser starts a new interval only when the numeric
   * residue identifier changes; a segment change alone does not split an
   * interval.
   */
  const CudaContainer<int2> &getResidues(void) const;

  /**
   * @brief Returns contiguous connected-component intervals.
   *
   * @return Borrowed const reference to the object-owned
   * `CudaContainer<int2>`. Each element stores a zero-based inclusive
   * `[x, y]` interval. No ownership is transferred.
   * @warning The current construction assumes every bonded component occupies
   * one contiguous atom-index interval.
   */
  const CudaContainer<int2> &getGroups(void) const;

  /**
   * @brief Returns the path supplied to the file constructor.
   *
   * @return Borrowed const reference to the object-owned file-system path. The
   * path is retained without canonicalization.
   */
  const std::filesystem::path &getFilePath(void) const;

  /**
   * @brief Returns unchecked mutable access to segment identifiers.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Resizing does not update the atom count. Mutation does not rebuild
   * residue intervals or any topology-derived state. See
   * @ref charmm_psf_mutation.
   */
  std::vector<std::string> &getSegmentIdentifiers(void);

  /**
   * @brief Returns unchecked mutable access to residue identifiers.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not rebuild `getResidues()` and resizing does not
   * update the atom count. See @ref charmm_psf_mutation.
   */
  std::vector<int> &getResidueIdentifiers(void);

  /**
   * @brief Returns unchecked mutable access to residue names.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Resizing does not update the atom count or any derived state. See
   * @ref charmm_psf_mutation.
   */
  std::vector<std::string> &getResidueNames(void);

  /**
   * @brief Returns unchecked mutable access to atom names.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Resizing does not update the atom count or any derived state. See
   * @ref charmm_psf_mutation.
   */
  std::vector<std::string> &getAtomNames(void);

  /**
   * @brief Returns unchecked mutable access to atom types.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not rebuild recognized water tuples, and resizing
   * does not update the atom count. See @ref charmm_psf_mutation.
   */
  std::vector<std::string> &getAtomTypes(void);

  /**
   * @brief Returns unchecked mutable access to per-atom charges.
   *
   * @return Borrowed mutable reference to object-owned host storage in
   * elementary-charge units. No ownership is transferred.
   * @warning Resizing does not update the atom count. A length mismatch causes
   * `getNetCharge()` to throw `ApoCharmmErrorCode::Runtime`. See
   * @ref charmm_psf_mutation.
   */
  std::vector<double> &getCharges(void);

  /**
   * @brief Returns unchecked mutable access to per-atom masses.
   *
   * @return Borrowed mutable reference to object-owned host storage in atomic
   * mass units. No ownership is transferred.
   * @warning Resizing does not update the atom count. A length mismatch causes
   * `getTotalMass()` to throw `ApoCharmmErrorCode::Runtime`. See
   * @ref charmm_psf_mutation.
   */
  std::vector<double> &getMasses(void);

  /**
   * @brief Returns unchecked mutable access to bond records.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not update the bond count, connectivity,
   * connected-component intervals, or exclusion arrays. Atom indices are not
   * validated. See @ref charmm_psf_mutation.
   */
  std::vector<Bond> &getBonds(void);

  /**
   * @brief Returns unchecked mutable access to angle records.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not update the angle count or any dependent force
   * data held by collaborators. Atom indices are not validated. See
   * @ref charmm_psf_mutation.
   */
  std::vector<Angle> &getAngles(void);

  /**
   * @brief Returns unchecked mutable access to proper-dihedral records.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not update the proper-dihedral count or dependent
   * force data. Atom indices are not validated. See @ref charmm_psf_mutation.
   */
  std::vector<Dihedral> &getDihedrals(void);

  /**
   * @brief Returns unchecked mutable access to improper-dihedral records.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not update the improper-dihedral count or dependent
   * force data. Atom indices are not validated. See @ref charmm_psf_mutation.
   */
  std::vector<Dihedral> &getImpropers(void);

  /**
   * @brief Returns unchecked mutable access to cross-term records.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not update the cross-term count. The current parsed
   * representation is one-based, but direct mutation is not validated. See
   * @ref charmm_psf_mutation.
   */
  std::vector<CrossTerm> &getCrossTerms(void);

  /**
   * @brief Returns unchecked mutable access to direct-bond connectivity.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not update 1-3 or 1-4 connectivity, `getIblo14()`,
   * `getInb14()`, or bonded records. See @ref charmm_psf_mutation.
   */
  std::vector<std::set<int>> &getConnected12(void);

  /**
   * @brief Returns unchecked mutable access to two-bond connectivity.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation does not update 1-4 connectivity or either CHARMM
   * exclusion array. See @ref charmm_psf_mutation.
   */
  std::vector<std::set<int>> &getConnected13(void);

  /**
   * @brief Returns unchecked mutable access to three-bond connectivity.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation affects later inclusion-list construction but does not
   * update either CHARMM exclusion array. See @ref charmm_psf_mutation.
   */
  std::vector<std::set<int>> &getConnected14(void);

  /**
   * @brief Returns unchecked mutable access to cumulative exclusion offsets.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation is not validated against `getInb14()` or the atom count.
   * See @ref charmm_psf_mutation.
   */
  std::vector<int> &getIblo14(void);

  /**
   * @brief Returns unchecked mutable access to flattened exclusion atom
   * numbers.
   *
   * @return Borrowed mutable reference to object-owned host storage. No
   * ownership is transferred.
   * @warning Mutation is not validated against `getIblo14()`, the atom count,
   * or the one-based CHARMM atom-number convention. See
   * @ref charmm_psf_mutation.
   */
  std::vector<int> &getInb14(void);

  /**
   * @brief Returns unchecked mutable access to recognized water tuples.
   *
   * @return Borrowed mutable reference to the object-owned
   * `CudaContainer<int4>`. No ownership is transferred.
   * @warning Host and device mirrors are not automatically coherent. Host
   * mutation requires `transferToDevice()` before device consumers can observe
   * it; device mutation requires `transferToHost()` before host consumers can
   * observe it. See @ref charmm_psf_mutation.
   */
  CudaContainer<int4> &getWaterMolecules(void);

  /**
   * @brief Returns unchecked mutable access to residue intervals.
   *
   * @return Borrowed mutable reference to the object-owned
   * `CudaContainer<int2>`. No ownership is transferred.
   * @warning Mutation does not update residue metadata. Host and device mirrors
   * require explicit transfer after one-sided mutation. See
   * @ref charmm_psf_mutation.
   */
  CudaContainer<int2> &getResidues(void);

  /**
   * @brief Returns unchecked mutable access to connected-component intervals.
   *
   * @return Borrowed mutable reference to the object-owned
   * `CudaContainer<int2>`. No ownership is transferred.
   * @warning Mutation does not update bonds or connectivity. Host and device
   * mirrors require explicit transfer after one-sided mutation. See
   * @ref charmm_psf_mutation.
   */
  CudaContainer<int2> &getGroups(void);

  /**
   * @brief Returns unchecked mutable access to the stored file path.
   *
   * @return Borrowed mutable reference to object-owned file-system path. No
   * ownership is transferred.
   * @warning Changing this path does not read another file, reparse topology,
   * or change any other stored data. See @ref charmm_psf_mutation.
   */
  std::filesystem::path &getFilePath(void);

  /**
   * @brief Computes the sum of all stored atom charges.
   *
   * @return Net partial charge in elementary-charge units. An initialized
   * zero-atom object returns `0.0`.
   * @throws ApoCharmmError With
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is negative.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Runtime` if the charge
   * vector length differs from the atom count.
   * @throws std::bad_alloc If constructing an error diagnostic fails.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post The PSF object is unchanged.
   */
  double getNetCharge(void) const;

  /**
   * @brief Computes the sum of all stored atom masses.
   *
   * @return Total mass in atomic mass units. An initialized zero-atom object
   * returns `0.0`.
   * @throws ApoCharmmError With
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is negative.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Runtime` if the mass
   * vector length differs from the atom count.
   * @throws std::bad_alloc If constructing an error diagnostic fails.
   * @throws std::length_error If an error diagnostic exceeds an
   * implementation-defined limit.
   *
   * @post The PSF object is unchanged.
   */
  double getTotalMass(void) const;

  /**
   * @brief Builds owned 1-4 inclusion and 1-2/1-3 exclusion pair lists.
   *
   * Every pair is emitted once with its lower zero-based atom index first.
   * Inclusion pairs are 1-4 connections not also present in the source atom's
   * 1-2 or 1-3 sets. Exclusion pairs are the ordered union of the source atom's
   * 1-2 and 1-3 sets. Inclusion pairs precede exclusion pairs in the returned
   * flattened vector.
   *
   * @return Owned host-side @ref InclusionExclusion value. `sizes[0]` is the
   * inclusion-pair count, `sizes[1]` is the exclusion-pair count, and
   * `in14_ex14` contains the corresponding zero-based pairs.
   * @throws ApoCharmmError With
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is negative.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Runtime` if any outer
   * connectivity vector length differs from the atom count.
   * @throws std::bad_alloc If temporary or returned host vectors cannot
   * allocate storage.
   * @throws std::length_error If temporary or returned storage exceeds an
   * implementation-defined vector size limit.
   *
   * @post The PSF object and its connectivity sets are unchanged.
   */
  InclusionExclusion getInclusionExclusionLists(void) const;

private:
  /**
   * @brief Exchanges complete owned PSF state with another object.
   *
   * @param[in,out] other PSF whose scalar, host, device, and path state is
   * exchanged with this object.
   *
   * @post Each object contains the complete previous state of the other.
   * @note The operation performs no allocation, data copy, CUDA transfer, or
   * synchronization.
   */
  void swap(CharmmPSF &other) noexcept;

  /**
   * @brief Rebuilds recognized three-site water tuples from atom types.
   *
   * The existing water container is cleared. The method scans atom types in
   * atom-index order and records every consecutive `OT`, `HT`, `HT` triple as
   * `(oxygen, hydrogen1, hydrogen2, 0)`. Recognized triples do not overlap.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if clearing,
   * appending, copying, allocation, launch checking, or shrinking the
   * CudaContainer fails.
   * @throws std::bad_alloc If host-side growth or diagnostic construction
   * cannot allocate storage.
   * @throws std::length_error If derived storage exceeds an
   * implementation-defined limit.
   *
   * @pre `m_NumAtoms` is nonnegative and `m_AtomTypes` contains at least
   * `m_NumAtoms` entries.
   * @post The host and device water mirrors contain the recognized tuples.
   */
  void initializeWaterMolecules(void);

  /**
   * @brief Builds contiguous intervals for bonded connected components.
   *
   * A union-find array is constructed from the current bonds. The component
   * representative is then interpreted as the inclusive end atom of a
   * contiguous component interval and appended to `m_Groups`.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if appending or
   * shrinking the groups CudaContainer fails.
   * @throws std::bad_alloc If host-side union-find or container growth cannot
   * allocate storage.
   * @throws std::length_error If derived storage exceeds an
   * implementation-defined limit.
   *
   * @pre `m_NumAtoms` and `m_NumBonds` are nonnegative, every bond index is in
   * `[0, m_NumAtoms)`, and every bonded component occupies a contiguous atom
   * interval.
   * @post One inclusive zero-based interval has been appended per interpreted
   * component and both CudaContainer mirrors contain the resulting list.
   * @warning The method does not clear `m_Groups` before appending and is
   * currently called only once during file construction.
   */
  void createConnectedComponents(void);

  /**
   * @brief Builds 1-2, 1-3, and 1-4 connectivity and CHARMM exclusion arrays.
   *
   * The outer connectivity vectors are resized to the atom count. The method
   * then inserts symmetric bond-derived connectivity, constructs cumulative
   * `m_Iblo14` offsets, and stores higher-index neighbors as one-based entries
   * in `m_Inb14`.
   *
   * @throws std::bad_alloc If a vector or set cannot allocate storage.
   * @throws std::length_error If derived storage exceeds an
   * implementation-defined limit.
   *
   * @pre `m_NumAtoms` is nonnegative and every bond index is in
   * `[0, m_NumAtoms)`.
   * @post `m_Iblo14` has one entry per atom and `m_Inb14` contains the
   * flattened one-based exclusion atom numbers.
   * @warning Resizing the outer connectivity vectors does not clear sets that
   * already exist at retained indices. The method is currently called once
   * during file construction.
   */
  void buildTopologicalExclusions(void);

  /**
   * @brief Parses primary PSF records and residue intervals from one file.
   *
   * The method reads the complete file into host memory, initializes per-atom
   * vectors, parses supported topology sections, converts bond/angle/dihedral
   * atom numbers to zero-based indices, and appends residue intervals while
   * processing ATOM records. Water, group, and exclusion derivation occurs
   * later in the file constructor.
   *
   * @param[in] filePath Borrowed non-empty path copied into `m_FilePath`.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `filePath` is empty.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Runtime` for file I/O,
   * missing sections, premature end of file, unsupported counts, malformed
   * records, non-finite numeric fields, or out-of-range atom numbers.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if appending or
   * shrinking residue intervals fails in a CudaContainer operation.
   * @throws std::bad_alloc If parsing or storage allocation fails.
   * @throws std::length_error If the file or parsed storage exceeds an
   * implementation-defined limit.
   *
   * @post On success, parser-owned counts, per-atom metadata, primary topology,
   * residue intervals, and the stored file name describe the input file.
   */
  void readCharmmPSF(const std::filesystem::path &filePath);

private:
  /** Stores the atom count, using `-1` as the uninitialized sentinel. */
  int m_NumAtoms;

  /**
   * Owns host-resident per-atom metadata in atom-record order. Normal parsed
   * state keeps every vector length equal to `m_NumAtoms`.
   */
  std::vector<std::string> m_SegmentIdentifiers;
  std::vector<int> m_ResidueIdentifiers;
  std::vector<std::string> m_ResidueNames;
  std::vector<std::string> m_AtomNames;
  std::vector<std::string> m_AtomTypes;
  std::vector<double> m_Charges;
  std::vector<double> m_Masses;

  /** Stores the bond count and zero-based bond records. */
  int m_NumBonds;
  std::vector<Bond> m_Bonds;

  /** Stores the angle count and zero-based angle records. */
  int m_NumAngles;
  std::vector<Angle> m_Angles;

  /** Stores the proper-dihedral count and zero-based records. */
  int m_NumDihedrals;
  std::vector<Dihedral> m_Dihedrals;

  /** Stores the improper-dihedral count and zero-based records. */
  int m_NumImpropers;
  std::vector<Dihedral> m_Impropers;

  /** Stores the cross-term count and the current parsed records. */
  int m_NumCrossTerms;
  std::vector<CrossTerm> m_CrossTerms;

  /** Owns host-resident per-atom 1-2, 1-3, and 1-4 connectivity sets. */
  std::vector<std::set<int>> m_Connected12;
  std::vector<std::set<int>> m_Connected13;
  std::vector<std::set<int>> m_Connected14;

  /**
   * Owns the cumulative offsets and one-based atom numbers used by the CHARMM
   * exclusion representation.
   */
  std::vector<int> m_Iblo14;
  std::vector<int> m_Inb14;

  /**
   * Owns host/device mirrors for recognized waters, residue intervals, and
   * interpreted connected-component intervals.
   */
  CudaContainer<int4> m_WaterMolecules;
  CudaContainer<int2> m_Residues;
  CudaContainer<int2> m_Groups;

  /** Owns the path supplied to successful file construction. */
  std::filesystem::path m_FilePath;
};
