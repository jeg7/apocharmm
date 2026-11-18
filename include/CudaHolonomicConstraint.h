// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "CudaContainer.h"
#include <memory>

class CharmmContext;

/**
 * @brief Class for holonomic constraint
 *
 */
class CudaHolonomicConstraint {
public:
  /**
   * @brief Construct a new Cuda Holonoma object
   *
   */
  CudaHolonomicConstraint(void);

public:
  void setCharmmContext(std::shared_ptr<CharmmContext> ctx);
  void setStream(std::shared_ptr<cudaStream_t> stream);

public:
  void setup(const double timeStep);

  /**
   * @brief Constrains the context coordinates to the reference geometry.
   *
   * @param[in] coordsRef Device coordinates that supply the constraint
   * directions.
   * @param[in] adjustVelocities Whether to add the constraint displacement,
   * divided by the time step, to the context velocities. A dynamics step needs
   * that correction; a projection that is not a step forward in time, such as
   * the start-up back step, passes `false`.
   */
  void handleHolonomicConstraints(const double4 *coordsRef,
                                  const bool adjustVelocities = true);
  void removeForceAlongHolonomicConstraints(void);

private:
  void constrainWaterMolecules(const double4 *coordsRef);
  void constrainShakeAtoms(const double4 *coordsRef);
  void updateVelocities(void);

private:
  std::shared_ptr<CharmmContext> m_Context;

  CudaContainer<int4> m_SettleAtoms;
  CudaContainer<int4> m_ShakeAtoms;
  CudaContainer<int2> m_AllConstrainedAtomPairs; // for projecting out the
                                                 // forces during minimization
  CudaContainer<float4> m_ShakeParams;

  CudaContainer<double4> m_CoordsStored;
  double m_TimeStep;

  std::shared_ptr<cudaStream_t> m_Stream;

  double mO;
  double mH;
  double mH2O;
  double mO_div_mH2O;
  double mH_div_mH2O;
  double rOHsq;
  double rHHsq;
  double ra;
  double ra_inv;
  double rb;
  double rc;
  double rc2;
};
