// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Julian Melendez Delgado
//
// ENDLICENSE

#include "CmapSpline.h"

#include <cmath>
#include <vector>

// Evaluate a 1D cubic spline and its first derivative using the
// precomputed second derivatives at the grid points.
static void cmap_spline_interp(const double xmin, const double dx,
                               const double *ya, const double *y2a,
                               const double x, double &y, double &y1) {
  
  const int inx = static_cast<int>(std::floor((x - xmin) / dx)) + 1;
  const double x1 = xmin + static_cast<double>(inx) * dx;
  const double x2 = xmin + static_cast<double>(inx - 1) * dx;

  const double a = (x1 - x) / dx;
  const double b = (x - x2) / dx;

  y = a * ya[inx - 1] + b * ya[inx] + ((a * a * a - a) * y2a[inx - 1] +
       (b * b * b - b) * y2a[inx]) * (dx * dx) / 6.0;

  y1 = (ya[inx] - ya[inx - 1]) / dx - (3.0 * a * a - 1.0) * dx * 
        y2a[inx - 1] / 6.0 + (3.0 * b * b - 1.0) * dx * y2a[inx] / 6.0;
}

// Solve for the second derivative at each point of a uniformly spaced
// natural cubic spline using the tridiagonal Thomas algorithm.
static void cmap_spline(const double dx, const double *y, const int n,
                        double *u, double *y2) {
  y2[0] = 0.0;
  u[0] = 0.0;

  const double dxinv = 1.0 / dx;

  // Forward elimination: compute c_i and d_i for M_i = c_i M_{i+1} + d_i.
  for (int i = 1; i < n - 1; ++i) {
    const double pinv = 1.0 / (y2[i - 1] + 4.0);

    y2[i] = -pinv;

    u[i] = ((6.0 * y[i + 1] - 12.0 * y[i] + 6.0 * y[i - 1]) * dxinv * 
             dxinv - u[i - 1]) * pinv;
  }

  y2[n - 1] = 0.0;

  // Back substitution: solve M_i = c_i M_{i+1} + d_i,
  // overwriting c_i in y2 with the second derivative M_i.
  for (int i = n - 2; i >= 0; --i)
    y2[i] = y2[i] * y2[i + 1] + u[i];
}

// Generate bicubic polynomial coefficients for a periodic CMAP.
//
// For every grid cell, 16 coefficients are stored:
//
//   c[i * 4 + j]
//
// such that
//
//   E(t,u) = sum_i sum_j c[i*4+j] * t^i * u^j
//
// where t and u are normalized coordinates in [0,1] within
// the grid cell.
void cmap_set_spline(const int num, const int xm, const double dx,
                     const std::vector<double> &grid,
                     std::vector<double> &coeff) {

  const int n = num + 2 * xm;
  // 16 polynomial coefficients per grid cell.
  coeff.resize(static_cast<std::size_t>(num) * static_cast<std::size_t>(num) * 16);

  // Periodically extended CMAP and spline work arrays for derivatives.
  std::vector<double> tgmap(static_cast<std::size_t>(n) * n);
  std::vector<double> t2(static_cast<std::size_t>(n) * n);

  std::vector<double> u(n);
  std::vector<double> u2(n);
  std::vector<double> yytmp(n);
  std::vector<double> y1tmp(n);

  const double xmin = -180.0 - static_cast<double>(xm) * dx;

  // Periodically extend the original CMAP grid.
  for (int i = 0; i < n; ++i) {
    const int ii = ((i + num - xm) % num + num) % num;

    for (int j = 0; j < n; ++j) {
      const int jj = ((j + num - xm) % num + num) % num;

      tgmap[i * n + j] = grid[ii * num + jj];
    }
  }

  // Compute spline second derivatives along the psi dimension.
  for (int i = 0; i < n; ++i) {
    cmap_spline(dx, &tgmap[i * n], n, u.data(), &t2[i * n]);
  }

  // Temporary derivative grids.
  std::vector<double> dphi(static_cast<std::size_t>(num) * num);
  std::vector<double> dpsi(static_cast<std::size_t>(num) * num);
  std::vector<double> dphidpsi(static_cast<std::size_t>(num) * num);

  // Generate the spline derivatives at every original CMAP grid point.
  for (int i = xm; i < num + xm; ++i) {
    const double phi = static_cast<double>(i - xm) * dx - 180.0;

    for (int j = xm; j < num + xm; ++j) {
      const double psi = static_cast<double>(j - xm) * dx - 180.0;

      // At fixed psi, evaluate every phi row to obtain E and dE/dpsi
      // as functions of phi.
      for (int k = 0; k < n; ++k) {
        cmap_spline_interp(xmin, dx, &tgmap[k * n], &t2[k * n], 
                           psi, yytmp[k], y1tmp[k]);
      }

      // Spline E(phi, psi) along phi at fixed psi, then evaluate the
      // energy and dE/dphi at the current phi.
      cmap_spline(dx, yytmp.data(), n, u.data(), u2.data());

      double v;
      double v1;

      cmap_spline_interp(xmin, dx, yytmp.data(), u2.data(), phi, v, v1);

      // Spline dE/dpsi along phi at fixed psi, then evaluate dE/dpsi
      // and the mixed derivative d2E/(dphi dpsi) at the current phi.
      cmap_spline(dx, y1tmp.data(), n, u.data(), u2.data());

      double v2;
      double v12;

      cmap_spline_interp(xmin, dx, y1tmp.data(), u2.data(), phi, v2, v12);

      const int ii = i - xm;
      const int jj = j - xm;

      // Store the derivatives using the original CMAP grid indexing.
      dphi[ii * num + jj] = v1;
      dpsi[ii * num + jj] = v2;
      dphidpsi[ii * num + jj] = v12;
    }
  }

  // Fixed bicubic Hermite transformation matrix. For each CMAP cell,
  // it converts the four corner energies and their first and mixed
  // derivatives into the cell's 16 polynomial coefficients.
  // WT is shared by all cells and CMAP types; it is not a coefficient map.
  static const int WT[16][16] = {
      { 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      { 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
      {-3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1,  0,  0,  0,  0},
      { 2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0},
      { 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
      { 0,  0,  0,  0, -3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1},
      { 0,  0,  0,  0,  2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1},
      {-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      { 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0},
      { 9, -9,  9, -9,  6,  3, -3, -6,  6, -6, -3,  3,  4,  2,  1,  2},
      {-6,  6, -6,  6, -4, -2,  2,  4, -3,  3,  3, -3, -2, -1, -1, -2},
      { 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      { 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0},
      {-6,  6, -6,  6, -3, -3,  3,  3, -4,  4,  2, -2, -2, -2, -1, -1},
      { 4, -4,  4, -4,  2,  2, -2, -2,  2, -2, -2,  2,  1,  1,  1,  1}};

  // Build coefficients for each cell.
  for (int i = 0; i < num; ++i) {
    const int i1 = (i + 1) % num;

    for (int j = 0; j < num; ++j) {
      const int j1 = (j + 1) % num;

      // Indices of the four grid points bounding this cell.
      const int i00 = i * num + j;
      const int i10 = i1 * num + j;
      const int i01 = i * num + j1;
      const int i11 = i1 * num + j1;

      const double f00 = grid[i00];
      const double f10 = grid[i10];
      const double f01 = grid[i01];
      const double f11 = grid[i11];

      const double fx00 = dphi[i00] * dx;
      const double fx10 = dphi[i10] * dx;
      const double fx01 = dphi[i01] * dx;
      const double fx11 = dphi[i11] * dx;

      const double fy00 = dpsi[i00] * dx;
      const double fy10 = dpsi[i10] * dx;
      const double fy01 = dpsi[i01] * dx;
      const double fy11 = dpsi[i11] * dx;

      const double fxy00 = dphidpsi[i00] * dx * dx;
      const double fxy10 = dphidpsi[i10] * dx * dx;
      const double fxy01 = dphidpsi[i01] * dx * dx;
      const double fxy11 = dphidpsi[i11] * dx * dx;

      // Bicubic constraints at the four cell corners: energy, normalized
      // phi derivative, normalized psi derivative, and mixed derivative.
      const double input[16] = {f00,  f10,  f11,  f01,
                                fx00, fx10, fx11, fx01,
                                fy00, fy10, fy11, fy01,
                                fxy00, fxy10, fxy11, fxy01};

      const std::size_t base = (static_cast<std::size_t>(i) * num + j) * 16;
      
      // Generate this cell's 16 bicubic coefficients from its corner data.
      for (int row = 0; row < 16; ++row) {
        double value = 0.0;
        for (int col = 0; col < 16; ++col) {
          value += static_cast<double>(WT[row][col]) * input[col];
        }

        coeff[base + row] = value;
      }
    }
  }
}