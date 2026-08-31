// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Julian Melendez Delgado
//
// ENDLICENSE

/**
 * @file
 * @brief Declares function for setting CMAP spline coefficients.
 */

#pragma once

#include <vector>

/**
 * @brief Generate bicubic coefficients for a periodic square CMAP grid.
 *
 * @param num Number of grid points per dimension.
 * @param xm Number of periodic padding points added on each side.
 * @param dx Angular spacing between grid points in degrees.
 * @param grid Energy values in row-major order; must contain num * num values.
 * @param coefficients Output containing 16 coefficients for each CMAP cell.
 */
void cmap_set_spline(const int num, const int xm, const double dx, 
                     const std::vector<double> &grid, 
                     std::vector<double> &coefficients);