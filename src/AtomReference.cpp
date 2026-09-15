// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "AtomReference.h"

#include "ApoCharmmError.h"

#include <string>
#include <utility>

AtomReference::AtomReference(std::shared_ptr<const CharmmPSF> psf,
                             const int atomIndex)
    : m_Psf(std::move(psf)), m_AtomIndex(atomIndex) {
  APOCHARMM_REQUIRE(m_Psf != nullptr, ApoCharmmErrorCode::InvalidArgument,
                    "AtomReference requires a non-null PSF");

  const int numAtoms = m_Psf->getNumAtoms();

  APOCHARMM_REQUIRE(numAtoms >= 0, ApoCharmmErrorCode::NotInitialized,
                    "CharmmPSF atom count is not initialized; observed " +
                        std::to_string(numAtoms));

  APOCHARMM_REQUIRE((m_AtomIndex >= 0) && (m_AtomIndex < numAtoms),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Atom index is out of range; expected [0, " +
                        std::to_string(numAtoms) + "), observed " +
                        std::to_string(m_AtomIndex));
}

int AtomReference::getAtomIndex(void) const noexcept { return m_AtomIndex; }

std::shared_ptr<const CharmmPSF> AtomReference::getPsf(void) const noexcept {
  return m_Psf;
}

bool AtomReference::hasSameTopology(const AtomReference &other) const noexcept {
  return m_Psf.get() == other.m_Psf.get();
}

bool AtomReference::operator==(const AtomReference &other) const noexcept {
  return this->hasSameTopology(other) && (m_AtomIndex == other.m_AtomIndex);
}

bool AtomReference::operator!=(const AtomReference &other) const noexcept {
  return !(*this == other);
}
