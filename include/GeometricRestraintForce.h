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

template <typename AT, typename CT> class GeometricRestraintForce {
public:
  static constexpr bool contributesVirial = true;

public:
  enum class Geometry : int { PLANE = 0, CYLINDER = 1, SPHERE = 2 };
  enum class Activation : int { SYMMETRIC = 0, INSIDE = 1, OUTSIDE = -1 };
  enum class SelectionMode : int { ATOMWISE = 0, RCM = 1 };
  enum class PotentialType : int { HARMONIC = 0 };

public:
  GeometricRestraintForce(void) = delete;

  explicit GeometricRestraintForce(const std::shared_ptr<const CharmmPSF> &psf);

  GeometricRestraintForce(const GeometricRestraintForce &other) = delete;

  GeometricRestraintForce(GeometricRestraintForce &&other) = delete;

  GeometricRestraintForce &
  operator=(const GeometricRestraintForce &other) = delete;

  GeometricRestraintForce &operator=(GeometricRestraintForce &&other) = delete;

  ~GeometricRestraintForce(void) = noexcept;
};
