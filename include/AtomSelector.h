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

#include "CharmmPSF.h"
#include "CudaContainer.h"
#include <memory>

class AtomSelector {
public:
  AtomSelector(void);
  AtomSelector(const std::shared_ptr<CharmmPSF> psf);

public:
  void setPsf(const std::shared_ptr<CharmmPSF> psf);

public:
  int getNumSelected(void) const;

  /**
   * @brief Returns a CudaContainer that has the indices of the atoms that have
   * been selected
   */
  const CudaContainer<int> &getSelection(void) const;
  CudaContainer<int> &getSelection(void);

public:
  void seleSegi(const std::string &segi);
  void seleResi(const int resi);
  void seleResn(const std::string &resn);
  void seleName(const std::string &anam);
  void seleType(const std::string &atyp);

  void selectAnd();
  void selectOr();
  void selectNot();

  void clear(void);

private:
  std::shared_ptr<CharmmPSF> m_Psf;
  CudaContainer<int> m_Selection;
};
