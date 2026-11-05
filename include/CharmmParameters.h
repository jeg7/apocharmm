// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE
//

#pragma once

#include "CharmmPSF.h"

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

/**
 * @brief Stores an ordered pair of CHARMM atom-type names for bond lookup.
 *
 * The parser canonicalizes a bond by sorting its two atom types before it
 * constructs this value. Direct construction performs no sorting or case
 * conversion.
 */
struct BondKey {
public:
  /**
   * @brief Constructs a bond key from two atom-type names.
   *
   * @param[in] iatom First atom-type name to copy.
   * @param[in] jatom Second atom-type name to copy.
   *
   * @post `atom1 == iatom` and `atom2 == jatom`.
   * @throws std::bad_alloc If copying either atom-type name requires an
   * allocation that fails.
   */
  BondKey(const std::string &iatom, const std::string &jatom)
      : atom1(iatom), atom2(jatom) {}

public:
  /**
   * @brief Returns whether two ordered bond keys are equal.
   *
   * @param[in] other Key to compare with this key.
   * @return `true` when both atom-type names match in the same order;
   * otherwise, `false`.
   */
  bool operator==(const BondKey &other) const {
    return this->atom1 == other.atom1 && this->atom2 == other.atom2;
  }

  /**
   * @brief Orders bond keys in descending lexicographic atom-type order.
   *
   * @param[in] first Left key to compare.
   * @param[in] second Right key to compare.
   * @return `true` when `first` sorts before `second` by descending `atom1`
   * and then descending `atom2`.
   */
  friend bool operator<(const BondKey &first, const BondKey &second) {
    if (second.atom1 != first.atom1)
      return second.atom1 < first.atom1;
    else
      return second.atom2 < first.atom2;
  }

  /**
   * @brief Writes the two atom-type names followed by a trailing space.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] key Bond key to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output, const BondKey &key) {
    output << key.atom1 << " " << key.atom2 << " ";
    return output;
  }

public:
  /** Stores the first atom-type name exactly as supplied. */
  std::string atom1;
  /** Stores the second atom-type name exactly as supplied. */
  std::string atom2;
};

/**
 * @brief Stores harmonic bond or Urey-Bradley equilibrium parameters.
 *
 * The represented potential is `kb * (r - b0)^2` in the apoCHARMM bonded
 * force backend.
 */
class BondValues {
public:
  /**
   * @brief Constructs zero-valued bond parameters.
   *
   * @post `kb == 0.0` and `b0 == 0.0`.
   */
  BondValues(void) : kb(0.0), b0(0.0) {}

  /**
   * @brief Constructs bond parameters from a force constant and distance.
   *
   * @param[in] kb Harmonic force constant in kcal mol^-1 angstrom^-2.
   * @param[in] b0 Equilibrium distance in angstroms.
   *
   * @note The constructor performs no range or finiteness validation.
   */
  BondValues(const double kb, const double b0) : kb(kb), b0(b0) {}

  /**
   * @brief Copies bond parameters.
   *
   * @param[in] other Value to copy.
   */
  BondValues(const BondValues &other) = default;

public:
  /**
   * @brief Writes the force constant and equilibrium distance plus a newline.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] bv Bond parameters to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output, const BondValues &bv) {
    output << "(kb: " << bv.kb << ", b0:" << bv.b0 << ")\n";
    return output;
  }

public:
  /** Harmonic force constant in kcal mol^-1 angstrom^-2. */
  double kb;
  /** Equilibrium distance in angstroms. */
  double b0;
};

/**
 * @brief Stores an ordered triple of CHARMM atom-type names for angle lookup.
 *
 * The parser canonicalizes an angle by sorting its outer atom types before it
 * constructs this value. Direct construction performs no sorting or case
 * conversion.
 */
struct AngleKey {
public:
  /**
   * @brief Constructs an angle key from three atom-type names.
   *
   * @param[in] iatom First outer atom-type name to copy.
   * @param[in] jatom Central atom-type name to copy.
   * @param[in] katom Second outer atom-type name to copy.
   *
   * @throws std::bad_alloc If copying an atom-type name requires an allocation
   * that fails.
   */
  AngleKey(const std::string &iatom, const std::string &jatom,
           const std::string &katom)
      : atom1(iatom), atom2(jatom), atom3(katom) {}

public:
  /**
   * @brief Returns whether two ordered angle keys are equal.
   *
   * @param[in] other Key to compare with this key.
   * @return `true` when all three names match in the same order; otherwise,
   * `false`.
   */
  bool operator==(const AngleKey &other) const {
    return this->atom1 == other.atom1 && this->atom2 == other.atom2 &&
           this->atom3 == other.atom3;
  }

  /**
   * @brief Orders angle keys in descending lexicographic atom-type order.
   *
   * @param[in] first Left key to compare.
   * @param[in] second Right key to compare.
   * @return `true` when `first` sorts before `second` by descending `atom1`,
   * `atom2`, and then `atom3`.
   */
  friend bool operator<(const AngleKey &first, const AngleKey &second) {
    if (second.atom1 != first.atom1)
      return second.atom1 < first.atom1;
    else if (second.atom2 != first.atom2)
      return second.atom2 < first.atom2;
    else
      return second.atom3 < first.atom3;
  }

  /**
   * @brief Writes the three atom-type names followed by a trailing space.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] key Angle key to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output, const AngleKey &key) {
    output << key.atom1 << " " << key.atom2 << " " << key.atom3 << " ";
    return output;
  }

public:
  /** Stores the first outer atom-type name exactly as supplied. */
  std::string atom1;
  /** Stores the central atom-type name exactly as supplied. */
  std::string atom2;
  /** Stores the second outer atom-type name exactly as supplied. */
  std::string atom3;
};

/**
 * @brief Stores harmonic angle equilibrium parameters.
 *
 * The represented potential is `kTheta * (theta - theta0)^2` in the
 * apoCHARMM bonded force backend.
 */
struct AngleValues {
public:
  /**
   * @brief Constructs zero-valued angle parameters.
   *
   * @post `kTheta == 0.0` and `theta0 == 0.0`.
   */
  AngleValues(void) : kTheta(0.0), theta0(0.0) {}

  /**
   * @brief Constructs angle parameters from a force constant and angle.
   *
   * @param[in] kTheta Harmonic force constant in kcal mol^-1 radian^-2.
   * @param[in] theta0 Equilibrium angle in radians.
   *
   * @note The constructor performs no range or finiteness validation.
   */
  AngleValues(const double kTheta, const double theta0)
      : kTheta(kTheta), theta0(theta0) {}

  /**
   * @brief Copies angle parameters.
   *
   * @param[in] other Value to copy.
   */
  AngleValues(const AngleValues &other) = default;

public:
  /**
   * @brief Writes the force constant and equilibrium angle plus a newline.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] av Angle parameters to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output, const AngleValues &av) {
    output << "(kTheta: " << av.kTheta << ", theta0: " << av.theta0 << ")\n";
    return output;
  }

public:
  /** Harmonic force constant in kcal mol^-1 radian^-2. */
  double kTheta;
  /** Equilibrium angle in radians. */
  double theta0;
};

/**
 * @brief Stores an ordered quartet of CHARMM atom types for torsion lookup.
 *
 * The parameter parser canonicalizes forward and reverse atom-type order before
 * constructing this value. Direct construction performs no canonicalization or
 * case conversion. The name `X` is used by wildcard matching code.
 */
struct DihedralKey {
public:
  /**
   * @brief Constructs a torsion key from four atom-type names.
   *
   * @param[in] iatom First atom-type name to copy.
   * @param[in] jatom Second atom-type name to copy.
   * @param[in] katom Third atom-type name to copy.
   * @param[in] latom Fourth atom-type name to copy.
   *
   * @throws std::bad_alloc If copying an atom-type name requires an allocation
   * that fails.
   */
  DihedralKey(const std::string &iatom, const std::string &jatom,
              const std::string &katom, const std::string &latom)
      : atom1(iatom), atom2(jatom), atom3(katom), atom4(latom) {}

public:
  /**
   * @brief Orders torsion keys in descending lexicographic atom-type order.
   *
   * @param[in] first Left key to compare.
   * @param[in] second Right key to compare.
   * @return `true` when `first` sorts before `second` by descending `atom1`,
   * `atom2`, `atom3`, and then `atom4`.
   */
  friend bool operator<(const DihedralKey &first, const DihedralKey &second) {
    if (second.atom1 != first.atom1)
      return second.atom1 < first.atom1;
    else if (second.atom2 != first.atom2)
      return second.atom2 < first.atom2;
    else if (second.atom3 != first.atom3)
      return second.atom3 < first.atom3;
    else
      return second.atom4 < first.atom4;
  }

  /**
   * @brief Writes the four atom-type names followed by a trailing space.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] dk Torsion key to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output, const DihedralKey &dk) {
    output << dk.atom1 << " " << dk.atom2 << " " << dk.atom3 << " " << dk.atom4
           << " ";
    return output;
  }

  /**
   * @brief Returns whether two ordered torsion keys are equal.
   *
   * @param[in] other Key to compare with this key.
   * @return `true` when all four names match in the same order; otherwise,
   * `false`.
   */
  bool operator==(const DihedralKey &other) const {
    return this->atom1 == other.atom1 && this->atom2 == other.atom2 &&
           this->atom3 == other.atom3 && this->atom4 == other.atom4;
  }

public:
  /** Stores the first atom-type name exactly as supplied. */
  std::string atom1;
  /** Stores the second atom-type name exactly as supplied. */
  std::string atom2;
  /** Stores the third atom-type name exactly as supplied. */
  std::string atom3;
  /** Stores the fourth atom-type name exactly as supplied. */
  std::string atom4;
};

/**
 * @brief Stores one CHARMM proper-dihedral Fourier term.
 *
 * The represented potential is `kChi * (1 + cos(n * phi - delta))`. The parser
 * accepts any representable integer multiplicity, but the packed CUDA
 * continuation encoding assumes a positive terminal multiplicity.
 */
struct DihedralValues {
public:
  /**
   * @brief Constructs a zero-valued proper-dihedral term.
   *
   * @post `kChi == 0.0`, `delta == 0.0`, and `n == 0`.
   */
  DihedralValues(void) : kChi(0.0), delta(0.0), n(0) {}

  /**
   * @brief Constructs a proper-dihedral Fourier term.
   *
   * @param[in] kChi Fourier amplitude in kcal mol^-1.
   * @param[in] n Dimensionless integer multiplicity.
   * @param[in] delta Phase offset in degrees.
   *
   * @note The constructor performs no range or finiteness validation.
   */
  DihedralValues(const double kChi, const int n, const double delta)
      : kChi(kChi), delta(delta), n(n) {}

  /**
   * @brief Copies a proper-dihedral term.
   *
   * @param[in] other Value to copy.
   */
  DihedralValues(const DihedralValues &other) = default;

public:
  /**
   * @brief Writes the amplitude, phase, and multiplicity plus a newline.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] dv Proper-dihedral term to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output,
                                  const DihedralValues &dv) {
    output << "(kChi: " << dv.kChi << ", delta : " << dv.delta
           << ", n : " << dv.n << ")\n";
    return output;
  }

public:
  /** Fourier amplitude in kcal mol^-1. */
  double kChi;
  /** Phase offset in degrees. */
  double delta;
  /** Dimensionless integer multiplicity. */
  int n;
};

/**
 * @brief Stores one CHARMM harmonic improper-dihedral parameter record.
 *
 * The intended potential is `kPsi * (psi - psi0)^2`. The parser stores the
 * phase in degrees without conversion. See @ref charmm_parameters for the
 * current packed-backend limitation.
 */
struct ImDihedralValues {
public:
  /**
   * @brief Constructs zero-valued improper-dihedral parameters.
   *
   * @post `kPsi == 0.0` and `psi0 == 0.0`.
   */
  ImDihedralValues(void) : kPsi(0.0), psi0(0.0) {}

  /**
   * @brief Constructs harmonic improper-dihedral parameters.
   *
   * @param[in] kPsi Harmonic force constant in kcal mol^-1 radian^-2.
   * @param[in] psi0 Equilibrium improper angle in degrees as stored by the
   * parser.
   *
   * @note The constructor performs no range or finiteness validation.
   */
  ImDihedralValues(const double kPsi, const double psi0)
      : kPsi(kPsi), psi0(psi0) {}

  /**
   * @brief Copies improper-dihedral parameters.
   *
   * @param[in] other Value to copy.
   */
  ImDihedralValues(const ImDihedralValues &other) = default;

public:
  /**
   * @brief Writes the force constant and phase plus a newline.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] idv Improper-dihedral parameters to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output,
                                  const ImDihedralValues &idv) {
    output << "(kPsi : " << idv.kPsi << ", psi0 : " << idv.psi0 << ")\n";
    return output;
  }

public:
  /** Harmonic force constant in kcal mol^-1 radian^-2. */
  double kPsi;
  /** Equilibrium improper angle in degrees as parsed. */
  double psi0;
};

/**
 * @brief Stores the two torsion keys that identify one CMAP interaction.
 *
 */
struct CmapKey {
public:
  /**
   * @brief Constructs a CMAP key by copying two torsion keys.
   *
   * @param[in] d1 First torsion key.
   * @param[in] d2 Second torsion key.
   *
   * @throws std::bad_alloc If copying either key requires an allocation that
   * fails.
   */
  CmapKey(const DihedralKey &d1, const DihedralKey &d2) : dih1(d1), dih2(d2) {}

  friend bool operator<(const CmapKey &lhs, const CmapKey &rhs) {
    if (lhs.dih1 < rhs.dih1)
      return true;
    if (rhs.dih1 < lhs.dih1)
      return false;
    return lhs.dih2 < rhs.dih2;
  }

  friend bool operator==(const CmapKey &lhs, const CmapKey &rhs) {
    return lhs.dih1 == rhs.dih1 && lhs.dih2 == rhs.dih2;
  }
  
  /**
   * @brief Writes the two torsion keys that identify the CMAP interaction.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] ck CMAP key to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output, const CmapKey &ck) {
    output << ck.dih1 << ck.dih2;
    return output;
  }

public:
  /** Stores the first torsion key. */
  DihedralKey dih1;
  /** Stores the second torsion key. */
  DihedralKey dih2;
};

/**
 * @brief Stores one CHARMM CMAP parameter record.
 *
 * A CMAP parameter consists of a 24 x 24 grid of energy correction values
 * and 16 bicubic polynomial coefficients for each grid cell.
 */
struct CmapValues {
public:
  static constexpr int gridSize = 24;
  static constexpr std::size_t numValues = gridSize * gridSize;
  static constexpr std::size_t coefficientsPerCell = 16;
  static constexpr std::size_t numCoefficients = coefficientsPerCell * numValues;

  CmapValues(void) : values(), coeff() {}

  explicit CmapValues(const std::vector<double> &values)
      : values(values), coeff() {}

  CmapValues(const CmapValues &other) = default;

public:
  friend std::ostream &operator<<(std::ostream &output, const CmapValues &cv) {
    output << "(gridSize : " << CmapValues::gridSize
           << ", values : " << cv.values.size()
           << ", coeff : " << cv.coeff.size() << ")\n";

    if (cv.values.size() != CmapValues::numValues)
      return output;

    for (int i = 0; i < CmapValues::gridSize; ++i) {
      for (int j = 0; j < CmapValues::gridSize; ++j)
        output << cv.values[i * CmapValues::gridSize + j] << " ";
      output << "\n";
    }

    return output;
  }

public:
  // Original CHARMM CMAP energy grid.
  std::vector<double> values;

  // 16 bicubic polynomial coefficients per grid cell.
  //
  // Cell (i,j) occupies:
  //
  //  ((i * gridSize + j) * 16) ... + 15
  //
  // and is evaluated as
  //
  //   E(t,u) = sum_i sum_j c[i*4+j] t^i u^j
  //
  // with t,u in [0,1].
  std::vector<double> coeff;
};

/**
 * @brief Stores one atom type's CHARMM Lennard-Jones parameters.
 *
 * The parser stores values exactly as represented in a NONBONDED record. It
 * performs no physical-range or sign validation.
 */
class VdwParameters {
public:
  /**
   * @brief Constructs zero-valued Lennard-Jones parameters.
   *
   * @post `epsilon == 0.0` and `rmin_2 == 0.0`.
   */
  VdwParameters(void) : epsilon(0.0), rmin_2(0.0) {}

  /**
   * @brief Constructs Lennard-Jones parameters.
   *
   * @param[in] eps Well-depth value in kcal mol^-1 as written in the CHARMM
   * parameter record.
   * @param[in] rm2 Half of the minimum-energy pair distance in angstroms.
   */
  VdwParameters(const double eps, const double rm2)
      : epsilon(eps), rmin_2(rm2) {}

public:
  /**
   * @brief Writes the well depth and half-distance plus a newline.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] vdw Lennard-Jones parameters to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output,
                                  const VdwParameters &vdw) {
    output << "(epsilon: " << vdw.epsilon << ", rmin_2: " << vdw.rmin_2
           << ")\n";
    return output;
  }

public:
  /** Well-depth value in kcal mol^-1 as parsed. */
  double epsilon;
  /** Half of the minimum-energy pair distance in angstroms. */
  double rmin_2;
};

// JEG260812: This and CMAP need to be fixed
/**
 * @brief Stores a pair-specific CHARMM NBFIX override.
 *
 * Parsing canonicalizes `atom1` and `atom2`, stores non-negative well-depth
 * magnitudes, and copies the regular values into the 1-4 fields when the input
 * omits explicit 1-4 values. The aggregate itself performs no
 * canonicalization or validation.
 *
 * @warning Default-initializing this type leaves `emin`, `rmin`, `emin14`, and
 * `rmin14` indeterminate. Value-initialize the object or initialize all six
 * members before reading it.
 */
class NBFixParameters {
public:
  // NBFixParameters(std::string a1, std::string a2, double e, double r,
  //                 double e14, double r14)
  //     : atom1(a1), atom2(a2), emin(e), rmin(r), emin14(e14), rmin14(r14) {}
  // // copy constructor
  // NBFixParameters(const NBFixParameters &nbfix) = default;
  // NBFixParameters(const NBFixParameters &nbfix)
  //     : atom1(nbfix.atom1), atom2(nbfix.atom2), emin(nbfix.emin),
  //       rmin(nbfix.rmin), emin14(nbfix.emin14), rmin14(nbfix.rmin14) {}

  // // assignment operator
  // NBFixParameters &operator=(const NBFixParameters &nbfix) {
  //   atom1 = nbfix.atom1;
  //   atom2 = nbfix.atom2;
  //   emin = nbfix.emin;
  //   rmin = nbfix.rmin;
  //   emin14 = nbfix.emin14;
  //   rmin14 = nbfix.rmin14;
  //   return *this;
  // }

  /**
   * @brief Writes atom types and all regular and 1-4 values plus a newline.
   *
   * @param[in,out] output Stream that receives the text.
   * @param[in] nbfix NBFIX override to write.
   * @return `output` after the write.
   */
  friend std::ostream &operator<<(std::ostream &output,
                                  const NBFixParameters &nbfix) {
    output << "(" << nbfix.atom1 << "," << nbfix.atom2 << "," << nbfix.emin
           << "," << nbfix.rmin << "," << nbfix.emin14 << "," << nbfix.rmin14
           << ")\n";
    return output;
  }

public:
  /**
   * Stores the first and second atom-type names. Parser output is canonical.
   */
  std::string atom1, atom2;

  /**
   * Stores regular well depth in kcal mol^-1, regular minimum distance in
   * angstroms, 1-4 well depth in kcal mol^-1, and 1-4 minimum distance in
   * angstroms, respectively.
   */
  double emin, rmin, emin14, rmin14;
};

/**
 * @brief Stores host-side bonded coefficient rows and topology rows.
 *
 * `paramsSize` and `listsSize` each contain six entries in this order:
 * bond, Urey-Bradley, angle, proper dihedral, improper dihedral, and CMAP.
 * `paramsVal` concatenates the coefficient rows for those categories:
 *
 * - bond: `[b0, kb]` in `[angstrom, kcal mol^-1 angstrom^-2]`;
 * - Urey-Bradley: `[s0, kub]` in
 *   `[angstrom, kcal mol^-1 angstrom^-2]`;
 * - angle: `[theta0_radians, kTheta]` in
 *   `[radian, kcal mol^-1 radian^-2]`;
 * - proper dihedral: `[signed_n, kChi, sin(delta), cos(delta)]` in
 *   `[dimensionless, kcal mol^-1, dimensionless, dimensionless]`, where
 *   nonterminal rows for a multi-term key use a negative multiplicity;
 * - improper dihedral: `[psi0_degrees, kPsi, 0, 1]` in
 *   `[degree, kcal mol^-1 radian^-2, dimensionless, dimensionless]`;
 * - CMAP: 16 bicubic polynomial coefficients per grid cell, 
 *   stored as one float per row.
 *
 * `listVal` concatenates zero-based PSF atom-index rows. Bond and
 * Urey-Bradley rows are `[i, j, type, 13]`; angle rows are
 * `[i, j, k, type, 13, 13]`; torsion rows are
 * `[i, j, k, l, type, 13, 13, 13]`. Type indices are local to each category.
 * The shift sentinel `13` denotes the zero periodic image.
 *
 * All storage is owned host memory. ForceManager later copies these rows to
 * CUDA force objects.
 */
struct BondedParamsAndLists {
public:
  /**
   * @brief Constructs packed bonded data by copying all supplied vectors.
   *
   * @param[in] pSize Six coefficient-row counts in bonded-category order.
   * @param[in] pVal Concatenated host coefficient rows.
   * @param[in] lSize Six topology-row counts in bonded-category order.
   * @param[in] lVal Concatenated host topology rows.
   *
   * @post The object owns independent copies of all four inputs.
   * @throws std::bad_alloc If copying the host vectors requires an allocation
   * that fails.
   * @note The constructor does not validate sizes, row widths, or indices.
   */
  BondedParamsAndLists(const std::vector<int> &pSize,
                       const std::vector<std::vector<float>> &pVal,
                       const std::vector<int> &lSize,
                       const std::vector<std::vector<int>> &lVal)
      : paramsSize(pSize), paramsVal(pVal), listsSize(lSize), listVal(lVal) {}

public:
  /** Stores the six coefficient-row counts in bonded-category order. */
  std::vector<int> paramsSize;

  /** Stores the concatenated host coefficient rows. */
  std::vector<std::vector<float>> paramsVal;

  /** Stores the six topology-row counts in bonded-category order. */
  std::vector<int> listsSize;

  /** Stores the concatenated zero-based PSF topology rows. */
  std::vector<std::vector<int>> listVal;
};

/**
 * @brief Stores host-side Lennard-Jones pair tables and per-atom type indices.
 *
 * For `N` distinct PSF atom types, both coefficient arrays contain
 * `N * (N + 1)` floats. Distinct types are assigned ascending lexicographic
 * indices. Pair `(i, j)`, where `0 <= j <= i < N`, starts at float offset
 * `2 * (i * (i + 1) / 2 + j)` and stores `[C6, C12]`, with units
 * `[kcal mol^-1 angstrom^6, kcal mol^-1 angstrom^12]`. `vdwTypes` and
 * `vdw14Types` contain one zero-based type index per PSF atom; the current
 * implementation produces identical type-index arrays.
 *
 * All storage is owned host memory. ForceManager later copies these arrays to
 * the direct-space CUDA force object.
 */
struct VdwParamsAndTypes {
public:
  /**
   * @brief Constructs packed Lennard-Jones data by copying all inputs.
   *
   * @param[in] vdwParams Flattened regular `[C6, C12]` pair table in
   * `[kcal mol^-1 angstrom^6, kcal mol^-1 angstrom^12]` units.
   * @param[in] vdw14Params Flattened 1-4 `[C6, C12]` pair table in the same
   * units.
   * @param[in] vdwTypes Per-atom regular type indices.
   * @param[in] vdw14Types Per-atom 1-4 type indices.
   *
   * @post The object owns independent copies of all four inputs.
   * @throws std::bad_alloc If copying the host arrays requires an allocation
   * that fails.
   * @note The constructor does not validate lengths or type indices.
   */
  VdwParamsAndTypes(const std::vector<float> &vdwParams,
                    const std::vector<float> &vdw14Params,
                    const std::vector<int> &vdwTypes,
                    const std::vector<int> &vdw14Types)
      : vdwParams(vdwParams), vdw14Params(vdw14Params), vdwTypes(vdwTypes),
        vdw14Types(vdw14Types) {}

public:
  /**
   * Stores regular and 1-4 flattened pair coefficients in
   * `[C6, C12]` order.
   */
  std::vector<float> vdwParams, vdw14Params;

  /** Stores regular and 1-4 zero-based type indices for every PSF atom. */
  std::vector<int> vdwTypes, vdw14Types;
};

/**
 * @brief Stores and packs host-resident CHARMM force-field parameters.
 *
 * The object parses one or more CHARMM `.prm` or `.str` files into bonded,
 * Lennard-Jones, and NBFIX lookup tables. It does not store topology, charges,
 * coordinates, device memory, CUDA streams, or synchronization state.
 * CharmmPSF supplies topology and atom types when packed force data is needed.
 *
 * Parsed record text is comment-stripped, trimmed, and converted to uppercase;
 * constructor file paths are retained as `std::filesystem::path` objects
 * without canonicalization. Compiler-generated copy construction and assignment
 * deep-copy all owned containers; move construction and assignment transfer
 * their contents and leave the source valid but otherwise unspecified.
 * Destruction is non-throwing, releases the host storage, and performs no CUDA
 * work.
 *
 * The class provides no internal synchronization. Concurrent const access
 * requires that no thread mutate, assign, move, or destroy the same object.
 *
 * @see charmm_parameters
 */
class CharmmParameters {
public:
  /**
   * @brief Constructs an empty parameter set.
   *
   * @post All parameter tables and the constructor-file list are empty.
   */
  CharmmParameters(void);

  /**
   * @brief Constructs a parameter set from one `.prm` or `.str` file.
   *
   * @param[in] filePath Non-empty file-system path. The path is copied and is
   * not retained as a pointer or reference.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::InvalidArgument` if
   * `filePath` is empty.
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened or read, a required TOPPAR parameter block is absent, a
   * NONBONDED header continuation is unterminated, or a parsed record has an
   * invalid shape or numeric field.
   * @throws std::bad_alloc If host-side string or container allocation fails.
   *
   * @post On successful construction, `filePath` is the sole entry returned by
   * getPrmFilePaths().
   */
  CharmmParameters(const std::filesystem::path &filePath);

  /**
   * @brief Constructs a parameter set by reading files in the supplied order.
   *
   * Later files are merged into the existing tables. Duplicate precedence
   * depends on the parameter section and is described on @ref
   * charmm_parameters.
   *
   * @param[in] filePaths Non-empty ordered list of non-empty `.prm` or `.str`
   * paths. All strings are copied.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::InvalidArgument` if
   * `filePaths` is empty or any contained path is empty.
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if any file
   * cannot be opened or read, a required TOPPAR parameter block is absent, a
   * NONBONDED header continuation is unterminated, or a parsed record has an
   * invalid shape or numeric field.
   * @throws std::bad_alloc If host-side string or container allocation fails.
   *
   * @post On successful construction, getPrmFilePaths() equals `filePaths` in
   * the original order.
   */
  CharmmParameters(const std::vector<std::filesystem::path> &filePaths);

public: // Getters
  /**
   * @brief Returns the parsed harmonic bond-parameter map.
   *
   * @return Borrowed const reference to the map owned by this object. Keys are
   * canonical atom-type pairs and values use AKMA bond units.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Later successful
   * reads are observable through the same map.
   */
  const std::map<BondKey, BondValues> &getBondParams(void) const;

  /**
   * @brief Returns the parsed Urey-Bradley parameter map.
   *
   * Every parsed ANGLES record creates an entry; records without explicit
   * Urey-Bradley fields store zero values.
   *
   * @return Borrowed const reference to the map owned by this object. Values
   * use AKMA bond units.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Later successful
   * reads are observable through the same map.
   */
  const std::map<AngleKey, BondValues> &getUreybParams(void) const;

  /**
   * @brief Returns the parsed harmonic angle-parameter map.
   *
   * @return Borrowed const reference to the map owned by this object.
   * Equilibrium angles are stored in radians.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Later successful
   * reads are observable through the same map.
   */
  const std::map<AngleKey, AngleValues> &getAngleParams(void) const;

  /**
   * @brief Returns the parsed proper-dihedral parameter map.
   *
   * Each key maps to all Fourier terms encountered for that key, in file-read
   * order.
   *
   * @return Borrowed const reference to the map owned by this object. Phase
   * offsets are stored in degrees.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Appending a later
   * matching term can invalidate references and iterators into that key's
   * `std::vector<DihedralValues>`.
   */
  const std::map<DihedralKey, std::vector<DihedralValues>> &
  getDihedralParams(void) const;

  /**
   * @brief Returns the parsed harmonic improper-dihedral parameter map.
   *
   * @return Borrowed const reference to the map owned by this object.
   * Equilibrium phases are stored in degrees.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Later successful
   * reads are observable through the same map.
   */
  const std::map<DihedralKey, ImDihedralValues> &getImproperParams(void) const;

  /**
   * @brief Returns the parsed CMAP parameter map.
   *
   * Each key identifies the two canonical dihedrals defining the CMAP
   * interaction and maps to its square energy grid.
   *
   * @return Borrowed const reference to the map owned by this object.
   * CMAP energy values are stored in kcal mol^-1 in row-major order.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Later successful
   * reads are observable through the same map.
   */
  const std::map<CmapKey, CmapValues> &getCmapParams(void) const;

  /**
   * @brief Returns the parsed regular Lennard-Jones parameter map.
   *
   * @return Borrowed const reference to the atom-type map owned by this object.
   * Values contain epsilon in kcal mol^-1 and `Rmin/2` in angstroms.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Later NONBONDED
   * records can replace mapped values in place.
   */
  const std::map<std::string, VdwParameters> &getVdwParams(void) const;

  /**
   * @brief Returns the parsed explicit 1-4 Lennard-Jones parameter map.
   *
   * Only NONBONDED records containing the optional three 1-4 fields contribute
   * entries.
   *
   * @return Borrowed const reference to the atom-type map owned by this object.
   * Values contain epsilon in kcal mol^-1 and `Rmin/2` in angstroms.
   *
   * @note The reference aliases this object and remains valid only while the
   * object is alive and has not been assigned from or moved. Later explicit
   * 1-4 records can replace mapped values in place.
   */
  const std::map<std::string, VdwParameters> &getVdw14Params(void) const;

  /**
   * @brief Returns the paths supplied to a file-reading constructor.
   *
   * Calls to readCharmmParameterFile() do not append to this list.
   *
   * @return Borrowed const reference to the owned file-system paths in
   * constructor order. The paths are not canonicalized.
   *
   * @note The reference remains valid only while this object is alive and has
   * not been assigned from or moved.
   */
  const std::vector<std::filesystem::path> &getPrmFilePaths(void) const;

public:
  /**
   * @brief Packs bonded parameters and PSF topology into host vectors.
   *
   * Exact parameter matching is attempted first. Proper dihedrals then permit
   * an `X-middle-middle-X` wildcard, and impropers permit an
   * `outer-X-X-outer` wildcard. Missing Urey-Bradley terms are allowed, and
   * terms with `abs(kub) <= 0.01` are omitted. CMAP output is always empty.
   *
   * @param[in] psf Borrowed shared pointer to a non-null CharmmPSF. The pointee
   * is read but not mutated or retained after return.
   * @return Owned host-side copies in BondedParamsAndLists layout.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::InvalidArgument` if
   * `psf` is null.
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if a
   * required bond, angle, proper-dihedral, or improper-dihedral
   * parameter cannot be matched to the PSF.
   * @throws std::bad_alloc If host-side packing allocation fails.
   *
   * @post Neither this object nor `*psf` is modified.
   * @note The operation performs no CUDA transfer or synchronization.
   * @warning The current improper row layout does not match `imdihe_pot`'s
   * mode-field contract; see @ref charmm_parameters.
   */
  BondedParamsAndLists
  getBondedParamsAndLists(const std::shared_ptr<CharmmPSF> &psf) const;

  /**
   * @brief Packs Lennard-Jones pair coefficients and PSF type indices.
   *
   * Distinct PSF atom types are sorted lexicographically. Normal pair values
   * use CHARMM geometric epsilon and additive `Rmin/2` combination unless an
   * NBFIX entry overrides the pair. The 1-4 table substitutes explicit per-type
   * 1-4 values where available.
   *
   * @param[in] psf Borrowed shared pointer to a non-null CharmmPSF. The pointee
   * is read but not mutated or retained after return.
   * @return Owned host-side copies in VdwParamsAndTypes layout. Each
   * coefficient pair stores `C6 = 2 * epsilon * Rmin^6` followed by
   * `C12 = epsilon * Rmin^12`.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::InvalidArgument` if
   * `psf` is null.
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if any PSF
   * atom type lacks a regular NONBONDED parameter.
   * @throws std::bad_alloc If host-side table construction fails.
   *
   * @post Neither this object nor `*psf` is modified.
   * @note The operation performs no CUDA transfer or synchronization.
   * @warning The current implementation uses regular NBFIX values when packing
   * both normal and 1-4 pair tables, even when explicit NBFIX 1-4 values were
   * parsed.
   */
  VdwParamsAndTypes getVdwParamsAndTypes(std::shared_ptr<CharmmPSF> &psf) const;

  /**
   * @brief Reads and merges one CHARMM parameter file.
   *
   * The parser recognizes BONDS, ANGLES, DIHEDRALS, IMPROPER, NONBONDED, and
   * NBFIX records. ATOMS, CMAP, and HBOND sections are currently ignored.
   * Record text is stripped of `!` comments, normalized to uppercase, and
   * parsed on the host. This method does not append `filePath` to the
   * constructor path list returned by getPrmFileNames().
   *
   * Bond, angle, Urey-Bradley, improper, and NBFIX maps keep the first matching
   * key. Proper-dihedral records append Fourier terms. Regular and explicit 1-4
   * NONBONDED values replace earlier values for the same atom type.
   *
   * @param[in] filePath Non-empty file-system path. The path is used only for
   * this call; parsed records, not the path string, are retained.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::InvalidArgument` if
   * `filePath` is empty.
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened or read, a file whose final path component contains
   * `TOPPAR` has no CHARMM parameter block, a NONBONDED header continuation is
   * unterminated, or a recognized record has an invalid token count or numeric
   * field.
   * @throws std::bad_alloc If host-side parsing or map growth requires an
   * allocation that fails.
   *
   * @post On success, supported records are merged into the existing host maps.
   * @warning The update is not transactional. A failure can leave records read
   * earlier in the same call observable in this object.
   * @note The operation performs no CUDA transfer or synchronization.
   */
  void readCharmmParameterFile(const std::filesystem::path &filePath);

private:
  /**
   * @brief Parses and inserts one normalized BONDS record.
   *
   * @param[in] tokens Four normalized record tokens.
   * @param[in] line Normalized record text used in diagnostics.
   * @param[in] fileName Source path used in diagnostics.
   * @param[in] lineNumber One-based source line number.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the token
   * count or either numeric field is invalid.
   * @post The first value for a canonical bond key is retained.
   */
  void parseBondRecord(const std::vector<std::string> &tokens,
                       const std::string &line, const std::string &fileName,
                       const std::size_t lineNumber);

  /**
   * @brief Parses and inserts one normalized ANGLES record.
   *
   * @param[in] tokens Five tokens without Urey-Bradley data or seven
   * tokens with it.
   * @param[in] line Normalized record text used in diagnostics.
   * @param[in] fileName Source path used in diagnostics.
   * @param[in] lineNumber One-based source line number.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the token
   * count or a numeric field is invalid.
   * @post The angle in degrees is converted to radians. The first angle and
   * Urey-Bradley values for the canonical key are retained.
   */
  void parseAngleRecord(const std::vector<std::string> &tokens,
                        const std::string &line, const std::string &fileName,
                        const std::size_t lineNumber);

  /**
   * @brief Parses and appends one normalized DIHEDRALS record.
   *
   * @param[in] tokens Seven normalized record tokens.
   * @param[in] line Normalized record text used in diagnostics.
   * @param[in] fileName Source path used in diagnostics.
   * @param[in] lineNumber One-based source line number.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the token
   * count, multiplicity, or another numeric field is invalid.
   * @post The new Fourier term is appended in read order under its canonical
   * torsion key; the phase remains in degrees.
   */
  void parseDihedralRecord(const std::vector<std::string> &tokens,
                           const std::string &line, const std::string &fileName,
                           const std::size_t lineNumber);

  /**
   * @brief Parses and inserts one normalized IMPROPER record.
   *
   * @param[in] tokens Seven normalized record tokens.
   * @param[in] line Normalized record text used in diagnostics.
   * @param[in] fileName Source path used in diagnostics.
   * @param[in] lineNumber One-based source line number.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the token
   * count or a numeric field is invalid.
   * @post The first value for the canonical key is retained; `psi0` remains in
   * degrees and the input multiplicity field is parsed but discarded.
   */
  void parseImproperRecord(const std::vector<std::string> &tokens,
                           const std::string &line, const std::string &fileName,
                           const std::size_t lineNumber);

  /**
   * @brief Parses one CMAP header and its associated grid values.
   *
   * @param[in] tokens Nine normalized CMAP header tokens containing two
   * torsion keys followed by the grid size.
   * @param[in,out] prmFile Parameter file stream used to read the CMAP grid.
   * @param[in] line Normalized CMAP header text used in diagnostics.
   * @param[in] fileName Source path used in diagnostics.
   * @param[in,out] lineNumber One-based source line number, updated as CMAP
   * grid lines are consumed.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the
   * header, grid size, or grid values are invalid, or if the file ends before
   * the complete grid is read.
   * @post The CMAP grid is stored under its two canonical torsion keys.
   */
  void parseCmapRecord(
      const std::vector<std::string> &tokens, std::ifstream &prmFile,
      const std::string &line, const std::string &fileName,
      std::size_t &lineNumber);

  /**
   * @brief Parses and stores one normalized NONBONDED record.
   *
   * @param[in] tokens Four regular tokens or seven tokens including explicit
   * 1-4 values.
   * @param[in] line Normalized record text used in diagnostics.
   * @param[in] fileName Source path used in diagnostics.
   * @param[in] lineNumber One-based source line number.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the token
   * count or a numeric field is invalid.
   * @post The regular value replaces an earlier value for the atom type. The
   * second token is parsed and discarded. The 1-4 value is replaced only when
   * the optional fields are present; their first token is parsed and discarded.
   */
  void parseNonbondedRecord(const std::vector<std::string> &tokens,
                            const std::string &line,
                            const std::string &fileName,
                            const std::size_t lineNumber);

  /**
   * @brief Parses and inserts one normalized NBFIX record.
   *
   * @param[in] tokens Four regular tokens or six tokens including explicit 1-4
   * values.
   * @param[in] line Normalized record text used in diagnostics.
   * @param[in] fileName Source path used in diagnostics.
   * @param[in] lineNumber One-based source line number.
   *
   * @throws ApoCharmmError with code `ApoCharmmErrorCode::Runtime` if the token
   * count or a numeric field is invalid.
   * @post Epsilon values are stored as non-negative magnitudes. Missing 1-4
   * fields copy the regular values, and the first canonical pair is retained.
   */
  void parseNbfixRecord(const std::vector<std::string> &tokens,
                        const std::string &line, const std::string &fileName,
                        const std::size_t lineNumber);

private:
  /** Stores canonical harmonic bond parameters. */
  std::map<BondKey, BondValues> m_BondParams;
  /** Stores canonical Urey-Bradley parameters keyed like angles. */
  std::map<AngleKey, BondValues> m_UreybParams;
  /** Stores canonical harmonic angle parameters. */
  std::map<AngleKey, AngleValues> m_AngleParams;
  /** Stores ordered Fourier terms for each canonical proper-dihedral key. */
  std::map<DihedralKey, std::vector<DihedralValues>> m_DihedralParams;
  /** Stores one harmonic value for each canonical improper-dihedral key. */
  std::map<DihedralKey, ImDihedralValues> m_ImproperParams;

  /** Stores one CMAP grid for each pair of torsion keys. */
  std::map<CmapKey, CmapValues> m_CmapParams;

  /** Stores canonical pair-specific NBFIX overrides. */
  std::map<std::tuple<std::string, std::string>, NBFixParameters> m_NbfixParams;

  /** Stores regular NONBONDED values by uppercase atom type. */
  std::map<std::string, VdwParameters> m_VdwParams;
  /** Stores explicit 1-4 NONBONDED values by uppercase atom type. */
  std::map<std::string, VdwParameters> m_Vdw14Params;

  /** Stores only the paths supplied to the successful constructor. */
  std::vector<std::filesystem::path> m_PrmFilePaths;
};
