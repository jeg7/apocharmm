// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

namespace charmm {

namespace constants {

// KBOLTZ is Boltzman constant AKMA units (KBOLTZ = N *K / 1 Kcal)
//                                                   a  b
constexpr double kBoltz =
    1.987191E-03; // Gas constant in AKMA units :  N_a * k_b / kcal

// ATMOSP is the converting factor to calculate the boundary pressure
//     1 ATMOSP = 1.01325   10**5  Joule/meter**3
//              = 1.4584007 10**-5 Kcal/Mol/angs.**3
constexpr double atmosp = 1.4584007E-05;

// PATMOS is the inverse of ATMOSP.
constexpr double patmos = 1.0 / atmosp;

constexpr double surfaceTensionFactor = 98.6923; // converts dyne/cm to atm*angs

// TIMFAC is the conversion factor from AKMA time to picoseconds.
//        (TIMFAC = SQRT ( ( 1A )**2 *1amu * Na / 1Kcal )
//        Multiply AKMA time by timfac to obtain picoseconds; divide picoseconds
//        by timfac to obtain AKMA time. The DCD DELTA field stores AKMA time.
constexpr double timfac = 4.88882129E-02;

// VERNUM of the reference CHARMM source (version 52). DCD control records and
// restart headers carry this value.
constexpr int vernum = 52;

} // namespace constants

} // namespace charmm
