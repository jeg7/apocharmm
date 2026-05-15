// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "AtomSelector.h"

#include <stdexcept>

AtomSelector::AtomSelector(void) : m_Psf(nullptr), m_Selection() {}

AtomSelector::AtomSelector(const std::shared_ptr<CharmmPSF> psf)
    : m_Psf(psf), m_Selection(psf->getNumAtoms()) {
  m_Selection.set(0);
}

void AtomSelector::setPsf(const std::shared_ptr<CharmmPSF> psf) {
  m_Psf = psf;
  m_Selection.resize(psf->getNumAtoms());
  m_Selection.set(0);
  return;
}

int AtomSelector::getNumSelected(void) const {
  const int numAtoms = m_Psf->getNumAtoms();
  int sum = 0;
  for (int i = 0; i < numAtoms; i++)
    sum += m_Selection[i];
  return sum;
}

const CudaContainer<int> &AtomSelector::getSelection(void) const {
  return m_Selection;
}

CudaContainer<int> &AtomSelector::getSelection(void) { return m_Selection; }

void AtomSelector::seleSegi(const std::string &segi) {
  if (m_Psf == nullptr)
    throw std::runtime_error("PSF is undefined");

  const int numAtoms = m_Psf->getNumAtoms();
  const std::vector<std::string> &segis = m_Psf->getSegmentIdentifiers();

  for (int i = 0; i < numAtoms; i++)
    m_Selection[i] = (segis[i] == segi) ? 1 : 0;
  m_Selection.transferToDevice();

  return;
}

void AtomSelector::seleResi(const int resi) {
  if (m_Psf == nullptr)
    throw std::runtime_error("PSF is undefined");

  const int numAtoms = m_Psf->getNumAtoms();
  const std::vector<int> &resis = m_Psf->getResidueIdentifiers();

  for (int i = 0; i < numAtoms; i++)
    m_Selection[i] = (resis[i] == resi) ? 1 : 0;
  m_Selection.transferToDevice();

  return;
}

void AtomSelector::seleResn(const std::string &resn) {
  if (m_Psf == nullptr)
    throw std::runtime_error("PSF is undefined");

  const int numAtoms = m_Psf->getNumAtoms();
  const std::vector<std::string> &resns = m_Psf->getResidueNames();

  for (int i = 0; i < numAtoms; i++)
    m_Selection[i] = (resns[i] == resn) ? 1 : 0;
  m_Selection.transferToDevice();

  return;
}

void AtomSelector::seleName(const std::string &anam) {
  if (m_Psf == nullptr)
    throw std::runtime_error("PSF is undefined");

  const int numAtoms = m_Psf->getNumAtoms();
  const std::vector<std::string> &anams = m_Psf->getAtomNames();

  for (int i = 0; i < numAtoms; i++)
    m_Selection[i] = (anams[i] == anam) ? 1 : 0;
  m_Selection.transferToDevice();

  return;
}

void AtomSelector::seleType(const std::string &atyp) {
  if (m_Psf == nullptr)
    throw std::runtime_error("PSF is undefined");

  const int numAtoms = m_Psf->getNumAtoms();
  const std::vector<std::string> &atyps = m_Psf->getAtomTypes();

  for (int i = 0; i < numAtoms; i++)
    m_Selection[i] = (atyps[i] == atyp) ? 1 : 0;
  m_Selection.transferToDevice();

  return;
}

void AtomSelector::selectAnd() { return; }

void AtomSelector::selectOr() { return; }

void AtomSelector::selectNot() { return; }

void AtomSelector::clear(void) {
  m_Selection.set(0);
  return;
}
