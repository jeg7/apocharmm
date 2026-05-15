// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

namespace charmm {

namespace constants {

constexpr double kBoltz =
    1.987191E-03; // Gas constant in AKMA units :  N_a * k_b / kcal
constexpr double atmosp = 1.4584007E-05;
constexpr double patmos = 1.0 / atmosp;
constexpr double surfaceTensionFactor = 98.6923; // converts dyne/cm to atm*angs

} // namespace constants

} // namespace charmm
