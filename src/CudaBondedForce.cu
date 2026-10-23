// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#ifndef NOCUDAC
#include "CudaBondedForce.h"
#include "cuda_utils.h"
#include "gpu_utils.h"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Energy and virial in device memory
// static __device__ BondedEnergyVirial_t d_energy_virial;

//
// Reduces energy values
//
__forceinline__ __device__ void reduce_energy(const double epot,
                                              volatile double *sh_epot,
                                              double *global_epot) {
#if __CUDA_ARCH__ < 300
  sh_epot[threadIdx.x] = epot;
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    int t = threadIdx.x + i;
    double epot_val = (t < blockDim.x) ? sh_epot[t] : 0.0;
    __syncthreads();
    sh_epot[threadIdx.x] = sh_epot[threadIdx.x] + epot_val;
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(global_epot, sh_epot[0]);
#else
  const int tid = threadIdx.x & (warpsize - 1);
  const int base = (threadIdx.x / warpsize);
  // Reduce within warps
  double energy = epot;
  for (int d = warpsize / 2; d >= 1; d /= 2) {
    energy += __hiloint2double(SHFL(__double2hiint(energy), tid + d),
                               SHFL(__double2loint(energy), tid + d));
  }
  // Reduce between warps
  // NOTE: this __syncthreads() is needed because we're using a single shared
  // memory buffer
  __syncthreads();
  if (tid == 0)
    sh_epot[base] = energy;
  __syncthreads();
  if (base == 0) {
    energy = (tid < blockDim.x / warpsize) ? sh_epot[tid] : 0.0;
    for (int d = warpsize / 2; d >= 1; d /= 2) {
      energy += __hiloint2double(SHFL(__double2hiint(energy), tid + d),
                                 SHFL(__double2loint(energy), tid + d));
    }
  }
  if (threadIdx.x == 0)
    atomicAdd(global_epot, energy);
#endif
}

//
// Templated sqrt() -function
//
template <typename T>
__forceinline__ __device__ double sqrt_template(const T x) {
  if (sizeof(T) == 4) {
    return sqrtf(x);
  } else {
    return sqrt(x);
  }
}

template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__device__ void calc_bond_force_device(
    const int pos, const bondlist_t *__restrict__ bondlist,
    const float2 *__restrict__ bondcoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double &epot, Virial_t *__restrict__ virial) {
  int ii = bondlist[pos].i;
  int jj = bondlist[pos].j;
  int ic = bondlist[pos].itype;
  int ish = bondlist[pos].ishift;

  // Calculate shift for i-atom
  CT shx, shy, shz;
  calc_box_shift<CT>(ish, boxx, boxy, boxz, shx, shy, shz);

  float4 xyzqi = xyzq[ii];
  float4 xyzqj = xyzq[jj];

  CT dx = xyzqi.x + shx - xyzqj.x;
  CT dy = xyzqi.y + shy - xyzqj.y;
  CT dz = xyzqi.z + shz - xyzqj.z;

  CT r = sqrt_template<CT>(dx * dx + dy * dy + dz * dz);

  float2 bondcoef_val = bondcoef[ic];
  CT db = r - (CT)bondcoef_val.x;
  CT fij = db * (CT)bondcoef_val.y;

  if (calc_energy) {
    epot += (double)(fij * db);
  }
  fij *= ((CT)2) / r;

  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij, ii, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, jj, stride, force);

  // Store virial
  if (calc_virial) {
#ifdef USE_DP_SFORCE
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(fij * dx));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(fij * dy));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(fij * dz));
    }
#else
    if (ish != 13) {
      fxij /= CONVERT_TO_VIR;
      fyij /= CONVERT_TO_VIR;
      fzij /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][0],
                llitoulli(fxij));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][1],
                llitoulli(fyij));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][2],
                llitoulli(fzij));
    }
#endif
  }
}

//
// bondcoef.x = cbb
// bondcoef.y = cbc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_bond_force_kernel(
    const int nbondlist, const bondlist_t *__restrict__ bondlist,
    const float2 *__restrict__ bondcoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_bond,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // CUDA_ARCH <  300: blockDim.x*sizeof(double)
  // CUDA_ARCH >= 300: (blockDim.x/warpsize)*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy) {
    epot = 0.0;
  }

  while (pos < nbondlist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>(
        pos, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz, force, epot,
        virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_bond);
  }
  // if (threadIdx.x == 0 )
  //  printf("after pos : %d, energy: %f\n ", pos - blockDim.x * gridDim.x,
  //  *energy_bond);
}

//
// bondcoef.x = cbb
// bondcoef.y = cbc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_ureyb_force_kernel(
    const int nureyblist, const bondlist_t *__restrict__ ureyblist,
    const float2 *__restrict__ ureybcoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_ureyb,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy) {
    epot = 0.0;
  }

  while (pos < nureyblist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>(
        pos, ureyblist, ureybcoef, xyzq, stride, boxx, boxy, boxz, force, epot,
        virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_ureyb);
  }
}

template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__device__ void calc_angle_force_device(
    const int pos, const anglelist_t *__restrict__ anglelist,
    const float2 *__restrict__ anglecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double &epot, Virial_t *__restrict__ virial) {
  int ii = anglelist[pos].i;
  int jj = anglelist[pos].j;
  int kk = anglelist[pos].k;
  int ic = anglelist[pos].itype;
  int ish = anglelist[pos].ishift1;
  int ksh = anglelist[pos].ishift2;

  // Calculate shift for i-atom
  CT ishx, ishy, ishz;
  calc_box_shift<CT>(ish, boxx, boxy, boxz, ishx, ishy, ishz);

  // Calculate shift for k-atom
  CT kshx, kshy, kshz;
  calc_box_shift<CT>(ksh, boxx, boxy, boxz, kshx, kshy, kshz);

  CT dxij = xyzq[ii].x + ishx - xyzq[jj].x;
  CT dyij = xyzq[ii].y + ishy - xyzq[jj].y;
  CT dzij = xyzq[ii].z + ishz - xyzq[jj].z;

  CT dxkj = xyzq[kk].x + kshx - xyzq[jj].x;
  CT dykj = xyzq[kk].y + kshy - xyzq[jj].y;
  CT dzkj = xyzq[kk].z + kshz - xyzq[jj].z;

  CT rij = sqrtf(dxij * dxij + dyij * dyij + dzij * dzij);
  CT rkj = sqrtf(dxkj * dxkj + dykj * dykj + dzkj * dzkj);

  CT rij_inv = ((CT)1) / rij;
  CT rkj_inv = ((CT)1) / rkj;

  CT dxijr = dxij * rij_inv;
  CT dyijr = dyij * rij_inv;
  CT dzijr = dzij * rij_inv;
  CT dxkjr = dxkj * rkj_inv;
  CT dykjr = dykj * rkj_inv;
  CT dzkjr = dzkj * rkj_inv;
  CT cst = dxijr * dxkjr + dyijr * dykjr + dzijr * dzkjr;

  // anglecoef.x = ctb
  // anglecoef.y = ctc
  float2 anglecoef_val = anglecoef[ic];
  // printf("%d %d %d %d %f %f\n", pos, ii, jj, kk, anglecoef_val.x*57.295,
  // anglecoef_val.y);

  // Restrict values of cst to the interval [-0.999 ... 0.999]
  // NOTE: we are ignoring the fancy stuff that is done on the CPU version
  cst = min((CT)0.999, max(-(CT)0.999, cst));

  CT at = acosf(cst);
  CT da = at - (CT)anglecoef_val.x;
  CT df = ((CT)anglecoef_val.y) * da;
  if (calc_energy) {
    epot += (double)(df * da);
  }
  CT st2r = ((CT)1.0) / (((CT)1.0) - cst * cst);
  CT str = sqrtf(st2r);
  df = -((CT)2.0) * df * str;

  CT dtxi = rij_inv * (dxkjr - cst * dxijr);
  CT dtxj = rkj_inv * (dxijr - cst * dxkjr);
  CT dtyi = rij_inv * (dykjr - cst * dyijr);
  CT dtyj = rkj_inv * (dyijr - cst * dykjr);
  CT dtzi = rij_inv * (dzkjr - cst * dzijr);
  CT dtzj = rkj_inv * (dzijr - cst * dzkjr);

  AT AT_dtxi, AT_dtyi, AT_dtzi;
  AT AT_dtxj, AT_dtyj, AT_dtzj;
  calc_component_force<AT, CT>(df, dtxi, dtyi, dtzi, AT_dtxi, AT_dtyi, AT_dtzi);
  calc_component_force<AT, CT>(df, dtxj, dtyj, dtzj, AT_dtxj, AT_dtyj, AT_dtzj);

  write_force<AT>(AT_dtxi, AT_dtyi, AT_dtzi, ii, stride, force);
  write_force<AT>(AT_dtxj, AT_dtyj, AT_dtzj, kk, stride, force);
  write_force<AT>(-AT_dtxi - AT_dtxj, -AT_dtyi - AT_dtyj, -AT_dtzi - AT_dtzj,
                  jj, stride, force);

  // Store virial
  if (calc_virial) {
#ifdef USE_DP_SFORCE
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(df * dtxi));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(df * dtyi));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(df * dtzi));
    }
    if (ksh != 13) {
      atomicAdd(&virial->sforce_dp[ksh][0], (double)(df * dtxj));
      atomicAdd(&virial->sforce_dp[ksh][1], (double)(df * dtyj));
      atomicAdd(&virial->sforce_dp[ksh][2], (double)(df * dtzj));
    }
#else
    if (ish != 13) {
      AT_dtxi /= CONVERT_TO_VIR;
      AT_dtyi /= CONVERT_TO_VIR;
      AT_dtzi /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][0],
                llitoulli(AT_dtxi));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][1],
                llitoulli(AT_dtyi));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][2],
                llitoulli(AT_dtzi));
    }
    if (ksh != 13) {
      AT_dtxj /= CONVERT_TO_VIR;
      AT_dtyj /= CONVERT_TO_VIR;
      AT_dtzj /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ksh][0],
                llitoulli(AT_dtxj));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ksh][1],
                llitoulli(AT_dtyj));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ksh][2],
                llitoulli(AT_dtzj));
    }
#endif
  }
}

//
// anglecoef.x = ctb
// anglecoef.y = ctc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_angle_force_kernel(
    const int nanglelist, const anglelist_t *__restrict__ anglelist,
    const float2 *__restrict__ anglecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_angle,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy)
    epot = 0.0;

  while (pos < nanglelist) {
    calc_angle_force_device<AT, CT, calc_energy, calc_virial>(
        pos, anglelist, anglecoef, xyzq, stride, boxx, boxy, boxz, (AT *)force,
        epot, virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_angle);
  }
}

//
// Dihedral potential
//
// dihecoef.x = cpd (integer)
// dihecoef.y = cpc
// dihecoef.z = cpsin
// dihecoef.w = cpcos
//
// Out: df, e
//
template <typename T, bool calc_energy>
__forceinline__ __device__ void dihe_pot(const float4 *dihecoef,
                                         const int ic_in, const T st,
                                         const T ct, T &df, double &e) {
  df = (T)0;
  if (calc_energy)
    e = 0.0;
  int ic = ic_in;

  bool lrep = true;
  while (lrep) {
    float4 dihecoef_val = dihecoef[ic];
    // printf("%d %d %.2f %.1f %.1f\n", threadIdx.x,
    // dihecoef_val.x,dihecoef_val.y ,dihecoef_val.z,   dihecoef_val.w );

    int iper = (int)dihecoef_val.x;
    lrep = (iper > 0) ? false : true;
    iper = abs(iper);

    T e1 = (T)1;
    T df1 = (T)0;
    T ddf1 = (T)0;

    // Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
    for (int nper = 1; nper <= iper; nper++) {
      ddf1 = e1 * ct - df1 * st;
      df1 = e1 * st + df1 * ct;
      e1 = ddf1;
    }
    if (calc_energy)
      e1 = e1 * dihecoef_val.w + df1 * dihecoef_val.z;
    df1 = df1 * dihecoef_val.w - ddf1 * dihecoef_val.z;
    df1 = -iper * df1;
    if (calc_energy) {
      e1 += (T)1;
      if (iper == 0)
        e1 = (T)1;
    }

    float arg = dihecoef_val.y;
    if (calc_energy)
      e += arg * e1;
    df += arg * df1;

    ic++;
  }
}

//
// Improper dihedral potential
//
// imdihecoef.x = cid (integer)
// imdihecoef.y = cic
// imdihecoef.z = cisin
// imdihecoef.w = cicos
//
// Out: df, e
//
template <typename T, bool calc_energy>
__forceinline__ __device__ void imdihe_pot(const float4 *dihecoef,
                                           const int ic_in, const T st,
                                           const T ct, T &df, double &e) {
  df = (T)0;
  if (calc_energy)
    e = 0.0;

  float4 dihecoef_val = dihecoef[ic_in];

  if ((int)dihecoef_val.x != 0) {
    int ic = ic_in;
    bool lrep = true;
    while (lrep) {
      int iper = (int)dihecoef_val.x;
      lrep = (iper > 0) ? false : true;
      iper = abs(iper);

      T e1 = (T)1;
      T df1 = (T)0;
      T ddf1 = (T)0;

      // Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
      for (int nper = 1; nper <= iper; nper++) {
        ddf1 = e1 * ct - df1 * st;
        df1 = e1 * st + df1 * ct;
        e1 = ddf1;
      }
      if (calc_energy)
        e1 = e1 * dihecoef_val.w + df1 * dihecoef_val.z;
      df1 = df1 * dihecoef_val.w - ddf1 * dihecoef_val.z;
      df1 = -iper * df1;
      if (calc_energy) {
        e1 += (T)1;
        if (iper == 0)
          e1 = (T)1;
      }

      float arg = dihecoef_val.y;
      if (calc_energy)
        e += arg * e1;
      df += arg * df1;

      ic++;
      if (lrep)
        dihecoef_val = dihecoef[ic];
    }
    // use harmonic potential
  } else {
    // calcul of cos(phi-phi0),sin(phi-phi0) and (phi-phi0).

    T ca = ct * dihecoef_val.w + st * dihecoef_val.z;
    T sa = st * dihecoef_val.w - ct * dihecoef_val.z;
    T ap;
    if (ca > (T)0.1) {
      ap = asinf(sa);
    } else {
      // ap = sign(acos(max(ca,-(T)1)),sa);
      ap = acosf(max(ca, -(T)1));
      ap = (sa > (T)0) ? ap : -ap;
      // warning is now triggered at deltaphi=84.26...deg (used to be 90).
      // nbent = nbent + 1;
    }

    df = dihecoef_val.y * ap;
    if (calc_energy)
      e = df * ap;
    df *= (T)2;
  }
}

template <typename T, bool calc_energy>
__forceinline__ __device__ void cmap_pot(const float *cmapcoef, 
                                         const int cmap_type, const T phi, 
                                         const T psi, T &dE_dphi, T &dE_dpsi, 
                                         double &e) {

  constexpr int n = 24;
  constexpr int ncell = n * n;
  constexpr int ncoef = 16;

  constexpr T xmin = (T)-180.0;
  constexpr T dx = (T)15.0;
  constexpr T inv_dx = (T)(1.0 / 15.0);

  // Convert angles to CMAP cell coordinates.
  const T x = (phi - xmin) * inv_dx;
  const T y = (psi - xmin) * inv_dx;

  const int ix = (int)floorf((float)x);
  const int iy = (int)floorf((float)y);

  const T tx = x - (T)ix;
  const T ty = y - (T)iy;

  // Periodic cell index.
  const int i = ((ix % n) + n) % n;
  const int j = ((iy % n) + n) % n;

  const int cell = i * n + j;

  // 16 bicubic polynomial coefficients.
  const float *c = cmapcoef + (cmap_type * ncell + cell) * ncoef;

  // Coefficients for each power of ty.
  const T a0 = c[0];
  const T a1 = c[4];
  const T a2 = c[8];
  const T a3 = c[12];

  const T b0 = c[1];
  const T b1 = c[5];
  const T b2 = c[9];
  const T b3 = c[13];

  const T c0 = c[2];
  const T c1 = c[6];
  const T c2 = c[10];
  const T c3 = c[14];

  const T d0 = c[3];
  const T d1 = c[7];
  const T d2 = c[11];
  const T d3 = c[15];

  // Cubic polynomials in x.
  const T px0 = ((a3 * tx + a2) * tx + a1) * tx + a0;
  const T px1 = ((b3 * tx + b2) * tx + b1) * tx + b0;
  const T px2 = ((c3 * tx + c2) * tx + c1) * tx + c0;
  const T px3 = ((d3 * tx + d2) * tx + d1) * tx + d0;

  // Energy: cubic in y whose coefficients are the cubic-in-x evaluations above.
  if constexpr (calc_energy) {
    e = (double)(((px3 * ty + px2) * ty + px1) * ty + px0);
  }

  // dE/dx.
  const T dx0 = (3.0 * a3 * tx + 2.0 * a2) * tx + a1;
  const T dx1 = (3.0 * b3 * tx + 2.0 * b2) * tx + b1;
  const T dx2 = (3.0 * c3 * tx + 2.0 * c2) * tx + c1;
  const T dx3 = (3.0 * d3 * tx + 2.0 * d2) * tx + d1;
  const T de_dx = ((dx3 * ty + dx2) * ty + dx1) * ty + dx0;

  // dE/dy.
  const T de_dy = ((3.0 * px3 * ty + 2.0 * px2) * ty + px1);

  // Convert from normalized cell coordinates to degrees.
  dE_dphi = de_dx * inv_dx;
  dE_dpsi = de_dy * inv_dx;
}

template <typename AT, typename CT, bool q_dihe, bool calc_energy,
          bool calc_virial>
__device__ void calc_dihe_force_device(
    const int pos, const dihelist_t *__restrict__ dihelist,
    const float4 *__restrict__ dihecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double &epot, Virial_t *__restrict__ virial) {
  int ii = dihelist[pos].i;
  int jj = dihelist[pos].j;
  int kk = dihelist[pos].k;
  int ll = dihelist[pos].l;
  int ic = dihelist[pos].itype;
  int ish = dihelist[pos].ishift1;
  int jsh = dihelist[pos].ishift2;
  int lsh = dihelist[pos].ishift3;

  // Calculate shift for i-atom
  CT ishx, ishy, ishz;
  calc_box_shift<CT>(ish, boxx, boxy, boxz, ishx, ishy, ishz);

  // Calculate shift for j-atom
  CT jshx, jshy, jshz;
  calc_box_shift<CT>(jsh, boxx, boxy, boxz, jshx, jshy, jshz);

  // Calculate shift for l-atom
  CT lshx, lshy, lshz;
  calc_box_shift<CT>(lsh, boxx, boxy, boxz, lshx, lshy, lshz);

  CT fx = (xyzq[ii].x + ishx) - (xyzq[jj].x + jshx);
  CT fy = (xyzq[ii].y + ishy) - (xyzq[jj].y + jshy);
  CT fz = (xyzq[ii].z + ishz) - (xyzq[jj].z + jshz);

  CT gx = xyzq[jj].x + jshx - xyzq[kk].x;
  CT gy = xyzq[jj].y + jshy - xyzq[kk].y;
  CT gz = xyzq[jj].z + jshz - xyzq[kk].z;

  CT hx = xyzq[ll].x + lshx - xyzq[kk].x;
  CT hy = xyzq[ll].y + lshy - xyzq[kk].y;
  CT hz = xyzq[ll].z + lshz - xyzq[kk].z;

  // A=F^G, B=H^G.
  CT ax = fy * gz - fz * gy;
  CT ay = fz * gx - fx * gz;
  CT az = fx * gy - fy * gx;
  CT bx = hy * gz - hz * gy;
  CT by = hz * gx - hx * gz;
  CT bz = hx * gy - hy * gx;

  CT ra2 = ax * ax + ay * ay + az * az;
  CT rb2 = bx * bx + by * by + bz * bz;
  CT rg = sqrtf(gx * gx + gy * gy + gz * gz);

  //    if((ra2 <= rxmin2) .or. (rb2 <= rxmin2) .or. (rg <= rxmin)) then
  //          nlinear = nlinear + 1
  //       endif

  CT rgr = 1.0f / rg;
  CT ra2r = 1.0f / ra2;
  CT rb2r = 1.0f / rb2;
  CT rabr = sqrtf(ra2r * rb2r);

  // ct=cos(phi)
  CT ct = (ax * bx + ay * by + az * bz) * rabr;
  //
  // ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
  // which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
  CT st = rg * rabr * (ax * hx + ay * hy + az * hz);
  //
  //     Energy and derivative contributions.

  CT df;
  double e;
  if (q_dihe) {
    dihe_pot<CT, calc_energy>(dihecoef, ic, st, ct, df, e);
  } else {
    imdihe_pot<CT, calc_energy>(dihecoef, ic, st, ct, df, e);
  }

  if (calc_energy)
    epot += e;

  //
  //     Compute derivatives wrt catesian coordinates.
  //
  // GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
  //  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)

  CT fg = fx * gx + fy * gy + fz * gz;
  CT hg = hx * gx + hy * gy + hz * gz;
  ra2r *= df;
  rb2r *= df;
  CT fga = fg * ra2r * rgr;
  CT hgb = hg * rb2r * rgr;
  CT gaa = ra2r * rg;
  CT gbb = rb2r * rg;
  // DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.

  // Store forces
  AT dfx, dfy, dfz;
  calc_component_force<AT, CT>(-gaa, ax, ay, az, dfx, dfy, dfz);
  write_force<AT>(dfx, dfy, dfz, ii, stride, force);

  AT dgx, dgy, dgz;
  calc_component_force<AT, CT>(fga, ax, ay, az, -hgb, bx, by, bz, dgx, dgy,
                               dgz);
  write_force<AT>(dgx - dfx, dgy - dfy, dgz - dfz, jj, stride, force);

  AT dhx, dhy, dhz;
  calc_component_force<AT, CT>(gbb, bx, by, bz, dhx, dhy, dhz);
  write_force<AT>(-dhx - dgx, -dhy - dgy, -dhz - dgz, kk, stride, force);
  write_force<AT>(dhx, dhy, dhz, ll, stride, force);

  // Store virial
  if (calc_virial) {
#ifdef USE_DP_SFORCE
    if (ish != 13) {
      atomicAdd(&virial->sforce_dp[ish][0], (double)(-gaa * ax));
      atomicAdd(&virial->sforce_dp[ish][1], (double)(-gaa * ay));
      atomicAdd(&virial->sforce_dp[ish][2], (double)(-gaa * az));
    }
    if (jsh != 13) {
      atomicAdd(&virial->sforce_dp[jsh][0],
                (double)(fga * ax - hgb * bx + gaa * ax));
      atomicAdd(&virial->sforce_dp[jsh][1],
                (double)(fga * ay - hgb * by + gaa * ay));
      atomicAdd(&virial->sforce_dp[jsh][2],
                (double)(fga * az - hgb * bz + gaa * az));
    }
    if (lsh != 13) {
      atomicAdd(&virial->sforce_dp[lsh][0], (double)(gbb * bx));
      atomicAdd(&virial->sforce_dp[lsh][1], (double)(gbb * by));
      atomicAdd(&virial->sforce_dp[lsh][2], (double)(gbb * bz));
    }
#else
    dfx /= CONVERT_TO_VIR;
    dfy /= CONVERT_TO_VIR;
    dfz /= CONVERT_TO_VIR;
    dgx /= CONVERT_TO_VIR;
    dgy /= CONVERT_TO_VIR;
    dgz /= CONVERT_TO_VIR;
    if (ish != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][0],
                llitoulli(dfx));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][1],
                llitoulli(dfy));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish][2],
                llitoulli(dfz));
    }
    if (jsh != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh][0],
                llitoulli(dgx - dfx));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh][1],
                llitoulli(dgy - dfy));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh][2],
                llitoulli(dgz - dfz));
    }
    if (lsh != 13) {
      dhx /= CONVERT_TO_VIR;
      dhy /= CONVERT_TO_VIR;
      dhz /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh][0],
                llitoulli(dhx));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh][1],
                llitoulli(dhy));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh][2],
                llitoulli(dhz));
    }
#endif
  }
}

//
// dihecoef.x = cpd (integer)
// dihecoef.y = cpc
// dihecoef.z = cpsin
// dihecoef.w = cpcos
//
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_dihe_force_kernel(
    const int ndihelist, const dihelist_t *__restrict__ dihelist,
    const float4 *__restrict__ dihecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_dihe,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy)
    epot = 0.0;

  while (pos < ndihelist) {
    calc_dihe_force_device<AT, CT, true, calc_energy, calc_virial>(
        pos, dihelist, dihecoef, xyzq, stride, boxx, boxy, boxz, (AT *)force,
        epot, virial);
    // printf("epot %f\n", epot);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_dihe);
  }
  // if (pos == 8064 - blockDim.x * gridDim.x) printf("pos: %d, epot: %.6f\n",
  // pos, epot); printf("pos: %d, epot: %.6f type : %d\n", pos - blockDim.x *
  // gridDim.x, epot, dihelist[pos - blockDim.x * gridDim.x].itype);
}

//
// dihecoef.x = cpd (integer)
// dihecoef.y = cpc
// dihecoef.z = cpsin
// dihecoef.w = cpcos
//
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_imdihe_force_kernel(
    const int nimdihelist, const dihelist_t *__restrict__ imdihelist,
    const float4 *__restrict__ imdihecoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_imdihe,
    Virial_t *__restrict__ virial) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy)
    epot = 0.0;

  while (pos < nimdihelist) {
    calc_dihe_force_device<AT, CT, false, calc_energy, calc_virial>(
        pos, imdihelist, imdihecoef, xyzq, stride, boxx, boxy, boxz,
        (AT *)force, epot, virial);
    pos += blockDim.x * gridDim.x;
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_imdihe);
  }
}

template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__device__ void calc_cmap_force_device(
    const int pos, const cmaplist_t *__restrict__ cmaplist,
    const float *__restrict__ cmapcoef, const float4 *__restrict__ xyzq,
    const int stride, const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double &epot, Virial_t *__restrict__ virial) {

  const int i1 = cmaplist[pos].i1;
  const int j1 = cmaplist[pos].j1;
  const int k1 = cmaplist[pos].k1;
  const int l1 = cmaplist[pos].l1;

  const int i2 = cmaplist[pos].i2;
  const int j2 = cmaplist[pos].j2;
  const int k2 = cmaplist[pos].k2;
  const int l2 = cmaplist[pos].l2;

  const int ic = cmaplist[pos].itype;

  const int ish1 = cmaplist[pos].ishift1;
  const int jsh1 = cmaplist[pos].ishift2;
  const int lsh1 = cmaplist[pos].ishift3;
  const int ish2 = cmaplist[pos].ishift4;
  const int jsh2 = cmaplist[pos].ishift5;
  const int lsh2 = cmaplist[pos].ishift6;

  // Calculate shifts for the first dihedral.
  CT ish1x, ish1y, ish1z;
  calc_box_shift<CT>(ish1, boxx, boxy, boxz, ish1x, ish1y, ish1z);
  CT jsh1x, jsh1y, jsh1z;
  calc_box_shift<CT>(jsh1, boxx, boxy, boxz, jsh1x, jsh1y, jsh1z);
  CT lsh1x, lsh1y, lsh1z;
  calc_box_shift<CT>(lsh1, boxx, boxy, boxz, lsh1x, lsh1y, lsh1z);

  // Calculate shifts for the second dihedral.
  CT ish2x, ish2y, ish2z;
  calc_box_shift<CT>(ish2, boxx, boxy, boxz, ish2x, ish2y, ish2z);
  CT jsh2x, jsh2y, jsh2z;
  calc_box_shift<CT>(jsh2, boxx, boxy, boxz, jsh2x, jsh2y, jsh2z);
  CT lsh2x, lsh2y, lsh2z;
  calc_box_shift<CT>(lsh2, boxx, boxy, boxz, lsh2x, lsh2y, lsh2z);

  // First dihedral: i1-j1-k1-l1.
  const CT fx1 = (xyzq[i1].x + ish1x) - (xyzq[j1].x + jsh1x);
  const CT fy1 = (xyzq[i1].y + ish1y) - (xyzq[j1].y + jsh1y);
  const CT fz1 = (xyzq[i1].z + ish1z) - (xyzq[j1].z + jsh1z);

  const CT gx1 = xyzq[j1].x + jsh1x - xyzq[k1].x;
  const CT gy1 = xyzq[j1].y + jsh1y - xyzq[k1].y;
  const CT gz1 = xyzq[j1].z + jsh1z - xyzq[k1].z;

  const CT hx1 = xyzq[l1].x + lsh1x - xyzq[k1].x;
  const CT hy1 = xyzq[l1].y + lsh1y - xyzq[k1].y;
  const CT hz1 = xyzq[l1].z + lsh1z - xyzq[k1].z;

  const CT ax1 = fy1 * gz1 - fz1 * gy1;
  const CT ay1 = fz1 * gx1 - fx1 * gz1;
  const CT az1 = fx1 * gy1 - fy1 * gx1;

  const CT bx1 = hy1 * gz1 - hz1 * gy1;
  const CT by1 = hz1 * gx1 - hx1 * gz1;
  const CT bz1 = hx1 * gy1 - hy1 * gx1;

  const CT ra21 = ax1 * ax1 + ay1 * ay1 + az1 * az1;
  const CT rb21 = bx1 * bx1 + by1 * by1 + bz1 * bz1;
  const CT rg1 = sqrtf(gx1 * gx1 + gy1 * gy1 + gz1 * gz1);

  const CT ra2r1 = 1.0f / ra21;
  const CT rb2r1 = 1.0f / rb21;
  const CT rabr1 = sqrtf(ra2r1 * rb2r1);

  const CT ct1 = (ax1 * bx1 + ay1 * by1 + az1 * bz1) * rabr1;

  const CT st1 = rg1 * rabr1 * (ax1 * hx1 + ay1 * hy1 + az1 * hz1);

  // Second dihedral: i2-j2-k2-l2.
  const CT fx2 = (xyzq[i2].x + ish2x) - (xyzq[j2].x + jsh2x);
  const CT fy2 = (xyzq[i2].y + ish2y) - (xyzq[j2].y + jsh2y);
  const CT fz2 = (xyzq[i2].z + ish2z) - (xyzq[j2].z + jsh2z);

  const CT gx2 = xyzq[j2].x + jsh2x - xyzq[k2].x;
  const CT gy2 = xyzq[j2].y + jsh2y - xyzq[k2].y;
  const CT gz2 = xyzq[j2].z + jsh2z - xyzq[k2].z;

  const CT hx2 = xyzq[l2].x + lsh2x - xyzq[k2].x;
  const CT hy2 = xyzq[l2].y + lsh2y - xyzq[k2].y;
  const CT hz2 = xyzq[l2].z + lsh2z - xyzq[k2].z;

  const CT ax2 = fy2 * gz2 - fz2 * gy2;
  const CT ay2 = fz2 * gx2 - fx2 * gz2;
  const CT az2 = fx2 * gy2 - fy2 * gx2;

  const CT bx2 = hy2 * gz2 - hz2 * gy2;
  const CT by2 = hz2 * gx2 - hx2 * gz2;
  const CT bz2 = hx2 * gy2 - hy2 * gx2;

  const CT ra22 = ax2 * ax2 + ay2 * ay2 + az2 * az2;
  const CT rb22 = bx2 * bx2 + by2 * by2 + bz2 * bz2;
  const CT rg2 = sqrtf(gx2 * gx2 + gy2 * gy2 + gz2 * gz2);

  const CT ra2r2 = 1.0f / ra22;
  const CT rb2r2 = 1.0f / rb22;
  const CT rabr2 = sqrtf(ra2r2 * rb2r2);

  const CT ct2 = (ax2 * bx2 + ay2 * by2 + az2 * bz2) * rabr2;
  const CT st2 = rg2 * rabr2 * (ax2 * hx2 + ay2 * hy2 + az2 * hz2);

  // Convert sin/cos to dihedral angles in degrees.
  const CT pi = static_cast<CT>(3.14159265358979323846);
  const CT phi1 = atan2f(st1, ct1) * (180.0 / pi);
  const CT phi2 = atan2f(st2, ct2) * (180.0 / pi);

  CT dE_dphi;
  CT dE_dpsi;
  double e;

  cmap_pot<CT, calc_energy>(cmapcoef, ic, phi1, phi2, dE_dphi, dE_dpsi, e);

  if (calc_energy)
    epot += e;

  // cmap_pot returns derivatives per degree, while the Cartesian dihedral
  // derivatives are with respect to angles in radians.
  const CT radians_to_degrees = (CT)180.0 / pi;
  const CT df1 = dE_dphi * radians_to_degrees;
  const CT df2 = dE_dpsi * radians_to_degrees;

  // Cartesian force contributions from the first dihedral.
  const CT fg1 = fx1 * gx1 + fy1 * gy1 + fz1 * gz1;
  const CT hg1 = hx1 * gx1 + hy1 * gy1 + hz1 * gz1;
  const CT rgr1 = (CT)1.0 / rg1;
  const CT ra2df1 = ra2r1 * df1;
  const CT rb2df1 = rb2r1 * df1;
  const CT fga1 = fg1 * ra2df1 * rgr1;
  const CT hgb1 = hg1 * rb2df1 * rgr1;
  const CT gaa1 = ra2df1 * rg1;
  const CT gbb1 = rb2df1 * rg1;

  AT dfx1, dfy1, dfz1;
  calc_component_force<AT, CT>(-gaa1, ax1, ay1, az1, dfx1, dfy1, dfz1);

  AT dgx1, dgy1, dgz1;
  calc_component_force<AT, CT>(fga1, ax1, ay1, az1, -hgb1, bx1, by1, bz1,
                               dgx1, dgy1, dgz1);

  AT dhx1, dhy1, dhz1;
  calc_component_force<AT, CT>(gbb1, bx1, by1, bz1, dhx1, dhy1, dhz1);

  write_force<AT>(dfx1, dfy1, dfz1, i1, stride, force);
  write_force<AT>(dgx1 - dfx1, dgy1 - dfy1, dgz1 - dfz1, j1, stride, force);
  write_force<AT>(-dhx1 - dgx1, -dhy1 - dgy1, -dhz1 - dgz1, k1, stride,
                  force);
  write_force<AT>(dhx1, dhy1, dhz1, l1, stride, force);

  // Cartesian force contributions from the second dihedral.
  const CT fg2 = fx2 * gx2 + fy2 * gy2 + fz2 * gz2;
  const CT hg2 = hx2 * gx2 + hy2 * gy2 + hz2 * gz2;
  const CT rgr2 = (CT)1.0 / rg2;
  const CT ra2df2 = ra2r2 * df2;
  const CT rb2df2 = rb2r2 * df2;
  const CT fga2 = fg2 * ra2df2 * rgr2;
  const CT hgb2 = hg2 * rb2df2 * rgr2;
  const CT gaa2 = ra2df2 * rg2;
  const CT gbb2 = rb2df2 * rg2;

  AT dfx2, dfy2, dfz2;
  calc_component_force<AT, CT>(-gaa2, ax2, ay2, az2, dfx2, dfy2, dfz2);

  AT dgx2, dgy2, dgz2;
  calc_component_force<AT, CT>(fga2, ax2, ay2, az2, -hgb2, bx2, by2, bz2,
                               dgx2, dgy2, dgz2);

  AT dhx2, dhy2, dhz2;
  calc_component_force<AT, CT>(gbb2, bx2, by2, bz2, dhx2, dhy2, dhz2);

  write_force<AT>(dfx2, dfy2, dfz2, i2, stride, force);
  write_force<AT>(dgx2 - dfx2, dgy2 - dfy2, dgz2 - dfz2, j2, stride, force);
  write_force<AT>(-dhx2 - dgx2, -dhy2 - dgy2, -dhz2 - dgz2, k2, stride, force);
  write_force<AT>(dhx2, dhy2, dhz2, l2, stride, force);

  // Store periodic-image virial corrections for both dihedrals.
  if (calc_virial) {
#ifdef USE_DP_SFORCE
    if (ish1 != 13) {
      atomicAdd(&virial->sforce_dp[ish1][0], (double)(-gaa1 * ax1));
      atomicAdd(&virial->sforce_dp[ish1][1], (double)(-gaa1 * ay1));
      atomicAdd(&virial->sforce_dp[ish1][2], (double)(-gaa1 * az1));
    }
    if (jsh1 != 13) {
      atomicAdd(&virial->sforce_dp[jsh1][0],
                (double)(fga1 * ax1 - hgb1 * bx1 + gaa1 * ax1));
      atomicAdd(&virial->sforce_dp[jsh1][1],
                (double)(fga1 * ay1 - hgb1 * by1 + gaa1 * ay1));
      atomicAdd(&virial->sforce_dp[jsh1][2],
                (double)(fga1 * az1 - hgb1 * bz1 + gaa1 * az1));
    }
    if (lsh1 != 13) {
      atomicAdd(&virial->sforce_dp[lsh1][0], (double)(gbb1 * bx1));
      atomicAdd(&virial->sforce_dp[lsh1][1], (double)(gbb1 * by1));
      atomicAdd(&virial->sforce_dp[lsh1][2], (double)(gbb1 * bz1));
    }

    if (ish2 != 13) {
      atomicAdd(&virial->sforce_dp[ish2][0], (double)(-gaa2 * ax2));
      atomicAdd(&virial->sforce_dp[ish2][1], (double)(-gaa2 * ay2));
      atomicAdd(&virial->sforce_dp[ish2][2], (double)(-gaa2 * az2));
    }
    if (jsh2 != 13) {
      atomicAdd(&virial->sforce_dp[jsh2][0],
                (double)(fga2 * ax2 - hgb2 * bx2 + gaa2 * ax2));
      atomicAdd(&virial->sforce_dp[jsh2][1],
                (double)(fga2 * ay2 - hgb2 * by2 + gaa2 * ay2));
      atomicAdd(&virial->sforce_dp[jsh2][2],
                (double)(fga2 * az2 - hgb2 * bz2 + gaa2 * az2));
    }
    if (lsh2 != 13) {
      atomicAdd(&virial->sforce_dp[lsh2][0], (double)(gbb2 * bx2));
      atomicAdd(&virial->sforce_dp[lsh2][1], (double)(gbb2 * by2));
      atomicAdd(&virial->sforce_dp[lsh2][2], (double)(gbb2 * bz2));
    }
#else
    dfx1 /= CONVERT_TO_VIR;
    dfy1 /= CONVERT_TO_VIR;
    dfz1 /= CONVERT_TO_VIR;
    dgx1 /= CONVERT_TO_VIR;
    dgy1 /= CONVERT_TO_VIR;
    dgz1 /= CONVERT_TO_VIR;
    if (ish1 != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish1][0],
                llitoulli(dfx1));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish1][1],
                llitoulli(dfy1));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish1][2],
                llitoulli(dfz1));
    }
    if (jsh1 != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh1][0],
                llitoulli(dgx1 - dfx1));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh1][1],
                llitoulli(dgy1 - dfy1));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh1][2],
                llitoulli(dgz1 - dfz1));
    }
    if (lsh1 != 13) {
      dhx1 /= CONVERT_TO_VIR;
      dhy1 /= CONVERT_TO_VIR;
      dhz1 /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh1][0],
                llitoulli(dhx1));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh1][1],
                llitoulli(dhy1));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh1][2],
                llitoulli(dhz1));
    }

    dfx2 /= CONVERT_TO_VIR;
    dfy2 /= CONVERT_TO_VIR;
    dfz2 /= CONVERT_TO_VIR;
    dgx2 /= CONVERT_TO_VIR;
    dgy2 /= CONVERT_TO_VIR;
    dgz2 /= CONVERT_TO_VIR;
    if (ish2 != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish2][0],
                llitoulli(dfx2));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish2][1],
                llitoulli(dfy2));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[ish2][2],
                llitoulli(dfz2));
    }
    if (jsh2 != 13) {
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh2][0],
                llitoulli(dgx2 - dfx2));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh2][1],
                llitoulli(dgy2 - dfy2));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[jsh2][2],
                llitoulli(dgz2 - dfz2));
    }
    if (lsh2 != 13) {
      dhx2 /= CONVERT_TO_VIR;
      dhy2 /= CONVERT_TO_VIR;
      dhz2 /= CONVERT_TO_VIR;
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh2][0],
                llitoulli(dhx2));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh2][1],
                llitoulli(dhy2));
      atomicAdd((unsigned long long int *)&virial->sforce_fp[lsh2][2],
                llitoulli(dhz2));
    }
#endif
  }
}

//
// cmapcoef contains flattened 24x24 CMAP grids.
// Each grid has 576 consecutive float values.
//
// Grid for cmapType:
//   cmapcoef[cmapType * 576 ... cmapType * 576 + 575]
//
// cmapType is stored in cmaplist.itype.
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_cmap_force_kernel(
    const int ncmaplist, const cmaplist_t *__restrict__ cmaplist,
    const float *__restrict__ cmapcoef,
    const float4 *__restrict__ xyzq, const int stride,
    const CT boxx, const CT boxy, const CT boxz,
    AT *__restrict__ force, double *__restrict__ energy_cmap,
    Virial_t *__restrict__ virial) {
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  double epot;
  if (calc_energy)
    epot = 0.0;

  while (pos < ncmaplist) {
    calc_cmap_force_device<AT, CT, calc_energy, calc_virial>(
        pos, cmaplist, cmapcoef, xyzq, stride, boxx, boxy, boxz,
        (AT *)force, epot, virial);

    pos += blockDim.x * gridDim.x;
  }

  if (calc_energy) {
    reduce_energy(epot, sh_epot, energy_cmap);
  }
}

/*
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_all_forces_kernel() {

  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  double epot;
  if (calc_energy) {
    epot = 0.0;
  }

  if (pos < d_setup.nbondlist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>
      (pos, d_setup.bondlist, d_setup.bondcoef, d_setup.xyzq,
       d_setup.stride, d_setup.boxx, d_setup.boxy, d_setup.boxz,
       (AT *)d_setup.force, epot);
  } else if (pos < d_setup.nureyblist + d_setup.nbondlist) {
    calc_bond_force_device<AT, CT, calc_energy, calc_virial>
      (pos - d_setup.nbondlist, d_setup.ureyblist, d_setup.ureybcoef,
d_setup.xyzq,
       d_setup.stride, d_setup.boxx, d_setup.boxy, d_setup.boxz,
       (AT *)d_setup.force, epot);
  } else if (pos < d_setup.nanglelist + d_setup.nureyblist + d_setup.nbondlist)
{
    calc_angle_force_device<AT, CT, calc_energy, calc_virial>
      (pos - d_setup.nureyblist - d_setup.nbondlist,
       d_setup.anglelist, d_setup.anglecoef, d_setup.xyzq, d_setup.stride,
       d_setup.boxx, d_setup.boxy, d_setup.boxz, (AT *)d_setup.force, epot);
  } else if (pos < d_setup.ndihelist + d_setup.nanglelist +
             d_setup.nureyblist + d_setup.nbondlist) {
    calc_dihe_force_device<AT, CT, true, calc_energy, calc_virial>
      (pos - d_setup.nanglelist - d_setup.nureyblist - d_setup.nbondlist,
       d_setup.dihelist, d_setup.dihecoef, d_setup.xyzq, d_setup.stride,
       d_setup.boxx, d_setup.boxy, d_setup.boxz, (AT *)d_setup.force, epot);
  } else if (pos < d_setup.nimdihelist + d_setup.ndihelist + d_setup.nanglelist
+
             d_setup.nureyblist + d_setup.nbondlist) {
    calc_dihe_force_device<AT, CT, false, calc_energy, calc_virial>
      (pos - d_setup.ndihelist - d_setup.nanglelist - d_setup.nureyblist -
d_setup.nbondlist,
       d_setup.imdihelist, d_setup.imdihecoef, d_setup.xyzq, d_setup.stride,
       d_setup.boxx, d_setup.boxy, d_setup.boxz, (AT *)d_setup.force, epot);
  }

  // Reduce energy
  if (calc_energy) {
    reduce_energy(epot, sh_epot, &d_energy_virial.energy_bond);
  }

}
*/

//
// Calculates all forces in a single kernel call
// NOTE: Energy calculation is disabled here because we cannot get individual
// energy terms out
//
template <typename AT, typename CT, bool calc_virial>
__global__ void calc_all_forces_kernel(
    const int nbondlist, const bondlist_t *__restrict__ bondlist,
    const float2 *__restrict__ bondcoef,

    const int nureyblist, const bondlist_t *__restrict__ ureyblist,
    const float2 *__restrict__ ureybcoef,

    const int nanglelist, const anglelist_t *__restrict__ anglelist,
    const float2 *__restrict__ anglecoef,

    const int ndihelist, const dihelist_t *__restrict__ dihelist,
    const float4 *__restrict__ dihecoef,

    const int nimdihelist, const dihelist_t *__restrict__ imdihelist,
    const float4 *__restrict__ imdihecoef,

    const int ncmaplist, const cmaplist_t *__restrict__ cmaplist,
    const float *__restrict__ cmapcoef,

    const float4 *__restrict__ xyzq, const int stride, const CT boxx,
    const CT boxy, const CT boxz, AT *__restrict__ force,
    Virial_t *__restrict__ virial) {
  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  // Dummy variable
  double epot;

  if (pos < nbondlist) {
    calc_bond_force_device<AT, CT, false, calc_virial>(
        pos, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz, force, epot,
        virial);
  } else if (pos < nureyblist + nbondlist) {
    calc_bond_force_device<AT, CT, false, calc_virial>(
        pos - nbondlist, ureyblist, ureybcoef, xyzq, stride, boxx, boxy, boxz,
        force, epot, virial);
  } else if (pos < nanglelist + nureyblist + nbondlist) {
    calc_angle_force_device<AT, CT, false, calc_virial>(
        pos - nureyblist - nbondlist, anglelist, anglecoef, xyzq, stride, boxx,
        boxy, boxz, force, epot, virial);
  } else if (pos < ndihelist + nanglelist + nureyblist + nbondlist) {
    calc_dihe_force_device<AT, CT, true, false, calc_virial>(
        pos - nanglelist - nureyblist - nbondlist, dihelist, dihecoef, xyzq,
        stride, boxx, boxy, boxz, force, epot, virial);
  } else if (pos <
             nimdihelist + ndihelist + nanglelist + nureyblist + nbondlist) {
    calc_dihe_force_device<AT, CT, false, false, calc_virial>(
        pos - ndihelist - nanglelist - nureyblist - nbondlist, imdihelist,
        imdihecoef, xyzq, stride, boxx, boxy, boxz, force, epot, virial);
  } else if (pos < ncmaplist + nimdihelist + ndihelist + nanglelist +
                         nureyblist + nbondlist) {
    calc_cmap_force_device<AT, CT, false, calc_virial>(
        pos - nimdihelist - ndihelist - nanglelist - nureyblist - nbondlist,
        cmaplist, cmapcoef, xyzq, stride, boxx, boxy, boxz, force, epot, virial);
  }
}

//---------------------------------------------------------------------------------------------------------
//
// Setups lists
//

__device__ void setup_bondlist_kernel(const int i,
                                      const int *__restrict__ bond_tbl,
                                      const bond_t *__restrict__ bond,
                                      bondlist_t *__restrict__ bondlist,
                                      const float4 *__restrict__ xyzq,
                                      const float3 half_box,
                                      const int *__restrict__ glo2loc_ind) {
  int j = bond_tbl[i];
  bond_t bondv = bond[j];
  bondlist_t bondlistv;
  bondlistv.i = glo2loc_ind[bondv.i];
  bondlistv.j = glo2loc_ind[bondv.j];
  bondlistv.itype = bondv.itype;
  float4 xyzq_i = xyzq[bondlistv.i];
  float4 xyzq_j = xyzq[bondlistv.j];
  bondlistv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
  bondlist[i] = bondlistv;
}

__device__ void setup_anglelist_kernel(const int i,
                                       const int *__restrict__ angle_tbl,
                                       const angle_t *__restrict__ angle,
                                       anglelist_t *__restrict__ anglelist,
                                       const float4 *__restrict__ xyzq,
                                       const float3 half_box,
                                       const int *__restrict__ glo2loc_ind) {
  int j = angle_tbl[i];
  angle_t anglev = angle[j];
  anglelist_t anglelistv;
  anglelistv.i = glo2loc_ind[anglev.i];
  anglelistv.j = glo2loc_ind[anglev.j];
  anglelistv.k = glo2loc_ind[anglev.k];
  anglelistv.itype = anglev.itype;
  float4 xyzq_i = xyzq[anglelistv.i];
  float4 xyzq_j = xyzq[anglelistv.j];
  float4 xyzq_k = xyzq[anglelistv.k];
  anglelistv.ishift1 = calc_ishift(xyzq_i, xyzq_j, half_box);
  anglelistv.ishift2 = calc_ishift(xyzq_k, xyzq_j, half_box);
  anglelist[i] = anglelistv;
}

__device__ void setup_dihelist_kernel(const int i,
                                      const int *__restrict__ dihe_tbl,
                                      const dihe_t *__restrict__ dihe,
                                      dihelist_t *__restrict__ dihelist,
                                      const float4 *__restrict__ xyzq,
                                      const float3 half_box,
                                      const int *__restrict__ glo2loc_ind) {
  int j = dihe_tbl[i];
  dihe_t dihev = dihe[j];
  dihelist_t dihelistv;
  dihelistv.i = glo2loc_ind[dihev.i];
  dihelistv.j = glo2loc_ind[dihev.j];
  dihelistv.k = glo2loc_ind[dihev.k];
  dihelistv.l = glo2loc_ind[dihev.l];
  dihelistv.itype = dihev.itype;
  float4 xyzq_i = xyzq[dihelistv.i];
  float4 xyzq_j = xyzq[dihelistv.j];
  float4 xyzq_k = xyzq[dihelistv.k];
  float4 xyzq_l = xyzq[dihelistv.l];
  dihelistv.ishift1 = calc_ishift(xyzq_i, xyzq_k, half_box);
  dihelistv.ishift2 = calc_ishift(xyzq_j, xyzq_k, half_box);
  dihelistv.ishift3 = calc_ishift(xyzq_l, xyzq_k, half_box);
  dihelist[i] = dihelistv;
}

__device__ void setup_cmaplist_kernel(const int i,
                                      const int *__restrict__ cmap_tbl,
                                      const cmap_t *__restrict__ cmap,
                                      cmaplist_t *__restrict__ cmaplist,
                                      const float4 *__restrict__ xyzq,
                                      const float3 half_box,
                                      const int *__restrict__ glo2loc_ind) {
  int j = cmap_tbl[i];
  cmap_t cmapv = cmap[j];
  cmaplist_t cmaplistv;
  cmaplistv.i1 = glo2loc_ind[cmapv.i1];
  cmaplistv.j1 = glo2loc_ind[cmapv.j1];
  cmaplistv.k1 = glo2loc_ind[cmapv.k1];
  cmaplistv.l1 = glo2loc_ind[cmapv.l1];
  cmaplistv.i2 = glo2loc_ind[cmapv.i2];
  cmaplistv.j2 = glo2loc_ind[cmapv.j2];
  cmaplistv.k2 = glo2loc_ind[cmapv.k2];
  cmaplistv.l2 = glo2loc_ind[cmapv.l2];
  cmaplistv.itype = cmapv.itype;
  float4 xyzq_i1 = xyzq[cmaplistv.i1];
  float4 xyzq_j1 = xyzq[cmaplistv.j1];
  float4 xyzq_k1 = xyzq[cmaplistv.k1];
  float4 xyzq_l1 = xyzq[cmaplistv.l1];
  float4 xyzq_i2 = xyzq[cmaplistv.i2];
  float4 xyzq_j2 = xyzq[cmaplistv.j2];
  float4 xyzq_k2 = xyzq[cmaplistv.k2];
  float4 xyzq_l2 = xyzq[cmaplistv.l2];
  // JM260901: We can probably get away with less ishift calculations
  cmaplistv.ishift1 = calc_ishift(xyzq_i1, xyzq_k1, half_box);
  cmaplistv.ishift2 = calc_ishift(xyzq_j1, xyzq_k1, half_box);
  cmaplistv.ishift3 = calc_ishift(xyzq_l1, xyzq_k1, half_box);
  cmaplistv.ishift4 = calc_ishift(xyzq_i2, xyzq_k2, half_box);
  cmaplistv.ishift5 = calc_ishift(xyzq_j2, xyzq_k2, half_box);
  cmaplistv.ishift6 = calc_ishift(xyzq_l2, xyzq_k2, half_box);
  cmaplist[i] = cmaplistv;
}

__global__ void setup_list_kernel(
    const int nbond_tbl, const int *__restrict__ bond_tbl,
    const bond_t *__restrict__ bond, bondlist_t *__restrict__ bondlist,
    const int nureyb_tbl, const int *__restrict__ ureyb_tbl,
    const bond_t *__restrict__ ureyb, bondlist_t *__restrict__ ureyblist,
    const int nangle_tbl, const int *__restrict__ angle_tbl,
    const angle_t *__restrict__ angle, anglelist_t *__restrict__ anglelist,
    const int ndihe_tbl, const int *__restrict__ dihe_tbl,
    const dihe_t *__restrict__ dihe, dihelist_t *__restrict__ dihelist,
    const int nimdihe_tbl, const int *__restrict__ imdihe_tbl,
    const dihe_t *__restrict__ imdihe, dihelist_t *__restrict__ imdihelist,
    const int ncmap_tbl, const int *__restrict__ cmap_tbl,
    const cmap_t *__restrict__ cmap, cmaplist_t *__restrict__ cmaplist,
    const float4 *__restrict__ xyzq, const float3 half_box,
    const int *__restrict__ glo2loc_ind) {
  int pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < nbond_tbl) {
    setup_bondlist_kernel(pos, bond_tbl, bond, bondlist, xyzq, half_box,
                          glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl) {
    setup_bondlist_kernel(pos - nbond_tbl, ureyb_tbl, ureyb, ureyblist, xyzq,
                          half_box, glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl) {
    setup_anglelist_kernel(pos - nbond_tbl - nureyb_tbl, angle_tbl, angle,
                           anglelist, xyzq, half_box, glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl) {
    setup_dihelist_kernel(pos - nbond_tbl - nureyb_tbl - nangle_tbl, dihe_tbl,
                          dihe, dihelist, xyzq, half_box, glo2loc_ind);
  } else if (pos <
             nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl) {
    setup_dihelist_kernel(pos - nbond_tbl - nureyb_tbl - nangle_tbl - ndihe_tbl,
                          imdihe_tbl, imdihe, imdihelist, xyzq, half_box,
                          glo2loc_ind);
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl +
                       nimdihe_tbl + ncmap_tbl) {
    setup_cmaplist_kernel(
        pos - nbond_tbl - nureyb_tbl - nangle_tbl - ndihe_tbl - nimdihe_tbl,
        cmap_tbl, cmap, cmaplist, xyzq, half_box, glo2loc_ind);
  }
}

//-----------------------------------------------------------------------------------------------------------

// #############################################################################################

//
// Dummy constructor
//
/*template <typename AT, typename CT>
CudaBondedForce<AT, CT>::CudaBondedForce(){
}*/

//
// Class creator
//
template <typename AT, typename CT>
CudaBondedForce<AT, CT>::CudaBondedForce(
    CudaEnergyVirial &energyVirial, const char *nameBond, const char *nameUreyb,
    const char *nameAngle, const char *nameDihe, const char *nameImdihe,
    const char *nameCmap)
    : energyVirial(energyVirial) {
  // Insert energy terms
  if (nameBond != NULL) {
    energyVirial.insert(nameBond);
    strBond = nameBond;
  }

  if (nameUreyb != NULL) {
    energyVirial.insert(nameUreyb);
    strUreyb = nameUreyb;
  }

  if (nameAngle != NULL) {
    energyVirial.insert(nameAngle);
    strAngle = nameAngle;
  }

  if (nameDihe != NULL) {
    energyVirial.insert(nameDihe);
    strDihe = nameDihe;
  }

  if (nameImdihe != NULL) {
    energyVirial.insert(nameImdihe);
    strImdihe = nameImdihe;
  }

  if (nameCmap != NULL) {
    energyVirial.insert(nameCmap);
    strCmap = nameCmap;
  }

  nbondlist = 0;
  nbondcoef = 0;
  bondlist_len = 0;
  bondlist = NULL;
  bondcoef_len = 0;
  bondcoef = NULL;

  nureyblist = 0;
  nureybcoef = 0;
  ureyblist_len = 0;
  ureyblist = NULL;
  ureybcoef_len = 0;
  ureybcoef = NULL;

  nanglelist = 0;
  nanglecoef = 0;
  anglelist_len = 0;
  anglelist = NULL;
  anglecoef_len = 0;
  anglecoef = NULL;

  ndihelist = 0;
  ndihecoef = 0;
  dihelist_len = 0;
  dihelist = NULL;
  dihecoef_len = 0;
  dihecoef = NULL;

  nimdihelist = 0;
  nimdihecoef = 0;
  imdihelist_len = 0;
  imdihelist = NULL;
  imdihecoef_len = 0;
  imdihecoef = NULL;

  ncmaplist = 0;
  ncmapcoef = 0;
  cmaplist_len = 0;
  cmaplist = NULL;
  cmapcoef_len = 0;
  cmapcoef = NULL;

  //  allocate_host<BondedEnergyVirial_t>(&h_energy_virial, 1);
}

// Move Constructor
template <typename AT, typename CT>
CudaBondedForce<AT, CT>::CudaBondedForce(CudaBondedForce &&other)
    : energyVirial(other.energyVirial), strBond(other.strBond),
      strUreyb(other.strUreyb), strAngle(other.strAngle),
      strDihe(other.strDihe), strImdihe(other.strImdihe),
      strCmap(other.strCmap) {

  nbondlist = other.nbondlist;
  nbondcoef = other.nbondcoef;
  bondlist_len = other.bondlist_len;
  bondlist = other.bondlist;
  bondcoef_len = other.bondcoef_len;
  bondcoef = other.bondcoef;

  other.nbondlist = 0;
  other.nbondcoef = 0;
  other.bondlist_len = 0;
  other.bondlist = NULL;
  other.bondcoef_len = 0;
  other.bondcoef = NULL;

  nureyblist = other.nureyblist;
  nureybcoef = other.nureybcoef;
  ureyblist_len = other.ureyblist_len;
  ureyblist = other.ureyblist;
  ureybcoef_len = other.ureybcoef_len;
  ureybcoef = other.ureybcoef;

  other.nureyblist = 0;
  other.nureybcoef = 0;
  other.ureyblist_len = 0;
  other.ureyblist = NULL;
  other.ureybcoef_len = 0;
  other.ureybcoef = NULL;

  nanglelist = other.nanglelist;
  nanglecoef = other.nanglecoef;
  anglelist_len = other.anglelist_len;
  anglelist = other.anglelist;
  anglecoef_len = other.anglecoef_len;
  anglecoef = other.anglecoef;

  other.nanglelist = 0;
  other.nanglecoef = 0;
  other.anglelist_len = 0;
  other.anglelist = NULL;
  other.anglecoef_len = 0;
  other.anglecoef = NULL;

  ndihelist = other.ndihelist;
  ndihecoef = other.ndihecoef;
  dihelist_len = other.dihelist_len;
  dihelist = other.dihelist;
  dihecoef_len = other.dihecoef_len;
  dihecoef = other.dihecoef;

  other.ndihelist = 0;
  other.ndihecoef = 0;
  other.dihelist_len = 0;
  other.dihelist = NULL;
  other.dihecoef_len = 0;
  other.dihecoef = NULL;

  nimdihelist = other.nimdihelist;
  nimdihecoef = other.nimdihecoef;
  imdihelist_len = other.imdihelist_len;
  imdihelist = other.imdihelist;
  imdihecoef_len = other.imdihecoef_len;
  imdihecoef = other.imdihecoef;

  other.nimdihelist = 0;
  other.nimdihecoef = 0;
  other.imdihelist_len = 0;
  other.imdihelist = NULL;
  other.imdihecoef_len = 0;
  other.imdihecoef = NULL;

  ncmaplist = other.ncmaplist;
  ncmapcoef = other.ncmapcoef;
  cmaplist_len = other.cmaplist_len;
  cmaplist = other.cmaplist;
  cmapcoef_len = other.cmapcoef_len;
  cmapcoef = other.cmapcoef;

  other.ncmaplist = 0;
  other.ncmapcoef = 0;
  other.cmaplist_len = 0;
  other.cmaplist = NULL;
  other.cmapcoef_len = 0;
  other.cmapcoef = NULL;

  forceVal = other.forceVal;
  bondedStream = other.bondedStream;
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaBondedForce<AT, CT>::~CudaBondedForce() noexcept {
  deallocate_noexcept<bondlist_t>(&bondlist);
  deallocate_noexcept<float2>(&bondcoef);

  deallocate_noexcept<bondlist_t>(&ureyblist);
  deallocate_noexcept<float2>(&ureybcoef);

  deallocate_noexcept<anglelist_t>(&anglelist);
  deallocate_noexcept<float2>(&anglecoef);

  deallocate_noexcept<dihelist_t>(&dihelist);
  deallocate_noexcept<float4>(&dihecoef);

  deallocate_noexcept<dihelist_t>(&imdihelist);
  deallocate_noexcept<float4>(&imdihecoef);

  deallocate_noexcept<cmaplist_t>(&cmaplist);
  deallocate_noexcept<float>(&cmapcoef);
}

//
// Setup the coefficients using vectors
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_coef(
    const std::vector<int> &size, const std::vector<std::vector<float>> &val) {
  assert(size.size() == 6);
  assert(val.size() == std::accumulate(size.begin(), size.end(), 0));

  nbondcoef = size[0];
  nureybcoef = size[1];
  nanglecoef = size[2];
  ndihecoef = size[3];
  nimdihecoef = size[4];
  ncmapcoef = size[5];
  size_t pos = 0;
  if (nbondcoef > 0) {
    float2 *h_bondcoef;
    h_bondcoef = (float2 *)malloc(nbondcoef * sizeof(float2));
    for (int i = 0; i < nbondcoef; ++i) {
      float2 elem;
      elem.x = val[i][0];
      elem.y = val[i][1];
      h_bondcoef[i] = elem;
    }
    reallocate<float2>(&bondcoef, &bondcoef_len, nbondcoef, 1.2f);
    copy_HtoD<float2>(h_bondcoef, bondcoef, nbondcoef);
    pos += nbondcoef;
    free(h_bondcoef);
  }

  if (nureybcoef > 0) {
    std::vector<float2> h_ureybcoef(nureybcoef);
    for (int i = 0; i < nureybcoef; i++)
      h_ureybcoef[i] = make_float2(val[pos + i][0], val[pos + i][1]);
    reallocate<float2>(&ureybcoef, &ureybcoef_len, nureybcoef, 1.2f);
    copy_HtoD<float2>(h_ureybcoef.data(), ureybcoef, nureybcoef);
    pos += nureybcoef;
  }

  if (nanglecoef > 0) {
    std::vector<float2> h_anglecoef(nanglecoef);
    for (int i = 0; i < nanglecoef; i++)
      h_anglecoef[i] = make_float2(val[pos + i][0], val[pos + i][1]);
    reallocate<float2>(&anglecoef, &anglecoef_len, nanglecoef, 1.2f);
    copy_HtoD<float2>(h_anglecoef.data(), anglecoef, nanglecoef);
    pos += nanglecoef;
  }

  if (ndihecoef > 0) {
    std::vector<float4> h_dihecoef(ndihecoef);
    for (int i = 0; i < ndihecoef; ++i) {
      h_dihecoef[i] = make_float4(val[pos + i][0], val[pos + i][1],
                                  val[pos + i][2], val[pos + i][3]);
    }
    reallocate<float4>(&dihecoef, &dihecoef_len, ndihecoef, 1.2f);
    copy_HtoD<float4>(h_dihecoef.data(), dihecoef, ndihecoef);
    pos += ndihecoef;
  }

  if (nimdihecoef > 0) {
    std::vector<float4> h_imdihecoef(nimdihecoef);
    for (int i = 0; i < nimdihecoef; i++) {
      h_imdihecoef[i] = make_float4(val[pos + i][0], val[pos + i][1],
                                    val[pos + i][2], val[pos + i][3]);
    }
    reallocate<float4>(&imdihecoef, &imdihecoef_len, nimdihecoef, 1.2f);
    copy_HtoD<float4>(h_imdihecoef.data(), imdihecoef, nimdihecoef);
    pos += nimdihecoef;
  }

  if (ncmapcoef > 0) {
    std::vector<float> h_cmapcoef(ncmapcoef);
    for (int i = 0; i < ncmapcoef; i++)
      h_cmapcoef[i] = val[pos + i][0];

    reallocate<float>(&cmapcoef, &cmapcoef_len, ncmapcoef, 1.2f);
    copy_HtoD<float>(h_cmapcoef.data(), cmapcoef, ncmapcoef);
    pos += ncmapcoef;
  }
  assert(pos == val.size());
}

//
// Setup coefficients (copies them from CPU to GPU)
// NOTE: This only has to be once in the beginning of the simulation
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_coef(
    const int nbondcoef, const float2 *h_bondcoef, const int nureybcoef,
    const float2 *h_ureybcoef, const int nanglecoef, const float2 *h_anglecoef,
    const int ndihecoef, const float4 *h_dihecoef, const int nimdihecoef,
    const float4 *h_imdihecoef, const int ncmapcoef, const float *h_cmapcoef) {
  assert((nureybcoef == 0) || (nureybcoef > 0 && nureybcoef == nanglecoef));

  this->nbondcoef = nbondcoef;
  if (nbondcoef > 0) {
    reallocate<float2>(&bondcoef, &bondcoef_len, nbondcoef, 1.2f);
    copy_HtoD<float2>(h_bondcoef, bondcoef, nbondcoef);
  }

  this->nureybcoef = nureybcoef;
  if (nureybcoef > 0) {
    reallocate<float2>(&ureybcoef, &ureybcoef_len, nureybcoef, 1.2f);
    copy_HtoD<float2>(h_ureybcoef, ureybcoef, nureybcoef);
  }

  this->nanglecoef = nanglecoef;
  if (nanglecoef > 0) {
    reallocate<float2>(&anglecoef, &anglecoef_len, nanglecoef, 1.2f);
    copy_HtoD<float2>(h_anglecoef, anglecoef, nanglecoef);
  }

  this->ndihecoef = ndihecoef;
  if (ndihecoef > 0) {
    reallocate<float4>(&dihecoef, &dihecoef_len, ndihecoef, 1.2f);
    copy_HtoD<float4>(h_dihecoef, dihecoef, ndihecoef);
  }

  this->nimdihecoef = nimdihecoef;
  if (nimdihecoef > 0) {
    reallocate<float4>(&imdihecoef, &imdihecoef_len, nimdihecoef, 1.2f);
    copy_HtoD<float4>(h_imdihecoef, imdihecoef, nimdihecoef);
  }

  this->ncmapcoef = ncmapcoef;
  if (ncmapcoef > 0) {
    reallocate<float>(&cmapcoef, &cmapcoef_len, ncmapcoef, 1.2f);
    copy_HtoD<float>(h_cmapcoef, cmapcoef, ncmapcoef);
  }
}

//
// Setup bondlists from vectors
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_list(
    const std::vector<int> &size, const std::vector<std::vector<int>> &val,
    cudaStream_t stream) {
  nbondlist = size[0];
  nureyblist = size[1];
  nanglelist = size[2];
  ndihelist = size[3];
  nimdihelist = size[4];
  ncmaplist = size[5];
  // assert((nureyblist == 0) || (nureyblist > 0 && nureyblist ==
  // nanglelist));
  size_t pos = 0;
  if (nbondlist > 0) {
    std::vector<bondlist_t> h_bondlist(nbondlist);
    for (int i = 0; i < nbondlist; i++) {
      h_bondlist[i].i = val[i][0];
      h_bondlist[i].j = val[i][1];
      h_bondlist[i].itype = val[i][2];
      h_bondlist[i].ishift = val[i][3];
    }
    reallocate<bondlist_t>(&bondlist, &bondlist_len, nbondlist, 1.2f);
    copy_HtoD<bondlist_t>(h_bondlist.data(), bondlist, nbondlist, stream);
    pos += nbondlist;
  }

  if (nureyblist > 0) {
    std::vector<bondlist_t> h_ureyblist(nureyblist);
    for (int i = 0; i < nureyblist; i++) {
      h_ureyblist[i].i = val[pos + i][0];
      h_ureyblist[i].j = val[pos + i][1];
      h_ureyblist[i].itype = val[pos + i][2];
      h_ureyblist[i].ishift = val[pos + i][3];
    }
    reallocate<bondlist_t>(&ureyblist, &ureyblist_len, nureyblist, 1.2f);
    copy_HtoD<bondlist_t>(h_ureyblist.data(), ureyblist, nureyblist, stream);
    pos += nureyblist;
  }

  if (nanglelist > 0) {
    std::vector<anglelist_t> h_anglelist(nanglelist);
    for (int i = 0; i < nanglelist; i++) {
      h_anglelist[i].i = val[pos + i][0];
      h_anglelist[i].j = val[pos + i][1];
      h_anglelist[i].k = val[pos + i][2];
      h_anglelist[i].itype = val[pos + i][3];
      h_anglelist[i].ishift1 = val[pos + i][4];
      h_anglelist[i].ishift2 = val[pos + i][5];
    }
    reallocate<anglelist_t>(&anglelist, &anglelist_len, nanglelist, 1.2f);
    copy_HtoD<anglelist_t>(h_anglelist.data(), anglelist, nanglelist, stream);
    pos += nanglelist;
  }

  if (ndihelist > 0) {
    std::vector<dihelist_t> h_dihelist(ndihelist);
    for (int i = 0; i < ndihelist; i++) {
      h_dihelist[i].i = val[pos + i][0];
      h_dihelist[i].j = val[pos + i][1];
      h_dihelist[i].k = val[pos + i][2];
      h_dihelist[i].l = val[pos + i][3];
      h_dihelist[i].itype = val[pos + i][4];
      h_dihelist[i].ishift1 = val[pos + i][5];
      h_dihelist[i].ishift2 = val[pos + i][6];
      h_dihelist[i].ishift3 = val[pos + i][7];
    }
    reallocate<dihelist_t>(&dihelist, &dihelist_len, ndihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_dihelist.data(), dihelist, ndihelist, stream);
    pos += ndihelist;
  }

  if (nimdihelist > 0) {
    std::vector<dihelist_t> h_imdihelist(nimdihelist);
    for (int i = 0; i < nimdihelist; i++) {
      h_imdihelist[i].i = val[pos + i][0];
      h_imdihelist[i].j = val[pos + i][1];
      h_imdihelist[i].k = val[pos + i][2];
      h_imdihelist[i].l = val[pos + i][3];
      h_imdihelist[i].itype = val[pos + i][4];
      h_imdihelist[i].ishift1 = val[pos + i][5];
      h_imdihelist[i].ishift2 = val[pos + i][6];
      h_imdihelist[i].ishift3 = val[pos + i][7];
    }
    reallocate<dihelist_t>(&imdihelist, &imdihelist_len, nimdihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_imdihelist.data(), imdihelist, nimdihelist, stream);
    pos += nimdihelist;
  }

  if (ncmaplist > 0) {
    std::vector<cmaplist_t> h_cmaplist(ncmaplist);
    for (int i = 0; i < ncmaplist; i++) {
      h_cmaplist[i].i1 = val[pos + i][0];
      h_cmaplist[i].j1 = val[pos + i][1];
      h_cmaplist[i].k1 = val[pos + i][2];
      h_cmaplist[i].l1 = val[pos + i][3];
      h_cmaplist[i].i2 = val[pos + i][4];
      h_cmaplist[i].j2 = val[pos + i][5];
      h_cmaplist[i].k2 = val[pos + i][6];
      h_cmaplist[i].l2 = val[pos + i][7];
      h_cmaplist[i].itype = val[pos + i][8];
      h_cmaplist[i].ishift1 = val[pos + i][9];
      h_cmaplist[i].ishift2 = val[pos + i][10];
      h_cmaplist[i].ishift3 = val[pos + i][11];
      h_cmaplist[i].ishift4 = val[pos + i][12];
      h_cmaplist[i].ishift5 = val[pos + i][13];
      h_cmaplist[i].ishift6 = val[pos + i][14];
    }
    reallocate<cmaplist_t>(&cmaplist, &cmaplist_len, ncmaplist, 1.2f);
    copy_HtoD<cmaplist_t>(h_cmaplist.data(), cmaplist, ncmaplist, stream);
    pos += ncmaplist;
  }
}

//
// Setup bondlists (copies them from CPU to GPU)
// NOTE: This has to be done after neighborlist update
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_list(
    const int nbondlist, const bondlist_t *h_bondlist, const int nureyblist,
    const bondlist_t *h_ureyblist, const int nanglelist,
    const anglelist_t *h_anglelist, const int ndihelist,
    const dihelist_t *h_dihelist, const int nimdihelist,
    const dihelist_t *h_imdihelist, const int ncmaplist,
    const cmaplist_t *h_cmaplist, cudaStream_t stream) {
  assert((nureyblist == 0) || (nureyblist > 0 && nureyblist == nanglelist));

  this->nbondlist = nbondlist;
  if (nbondlist > 0) {
    reallocate<bondlist_t>(&bondlist, &bondlist_len, nbondlist, 1.2f);
    copy_HtoD<bondlist_t>(h_bondlist, bondlist, nbondlist, stream);
  }

  this->nureyblist = nureyblist;
  if (nureyblist > 0) {
    reallocate<bondlist_t>(&ureyblist, &ureyblist_len, nureyblist, 1.2f);
    copy_HtoD<bondlist_t>(h_ureyblist, ureyblist, nureyblist, stream);
  }

  this->nanglelist = nanglelist;
  if (nanglelist > 0) {
    reallocate<anglelist_t>(&anglelist, &anglelist_len, nanglelist, 1.2f);
    copy_HtoD<anglelist_t>(h_anglelist, anglelist, nanglelist, stream);
  }

  this->ndihelist = ndihelist;
  if (ndihelist > 0) {
    reallocate<dihelist_t>(&dihelist, &dihelist_len, ndihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_dihelist, dihelist, ndihelist, stream);
  }

  this->nimdihelist = nimdihelist;
  if (nimdihelist > 0) {
    reallocate<dihelist_t>(&imdihelist, &imdihelist_len, nimdihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_imdihelist, imdihelist, nimdihelist, stream);
  }

  this->ncmaplist = ncmaplist;
  if (ncmaplist > 0) {
    reallocate<cmaplist_t>(&cmaplist, &cmaplist_len, ncmaplist, 1.2f);
    copy_HtoD<cmaplist_t>(h_cmaplist, cmaplist, ncmaplist, stream);
  }
}

//
// Setup lists from device memory using global bond data:
// bond[]                  = global bond data
// bond_tbl[0:nbond_tbl-1] = the index of bond in bond[]
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setup_list(
    const float4 *xyzq, const CT boxx, const CT boxy, const CT boxz,
    const int *glo2loc_ind, const int nbond_tbl, const int *bond_tbl,
    const bond_t *bond, const int nureyb_tbl, const int *ureyb_tbl,
    const bond_t *ureyb, const int nangle_tbl, const int *angle_tbl,
    const angle_t *angle, const int ndihe_tbl, const int *dihe_tbl,
    const dihe_t *dihe, const int nimdihe_tbl, const int *imdihe_tbl,
    const dihe_t *imdihe, const int ncmap_tbl, const int *cmap_tbl,
    const cmap_t *cmap, cudaStream_t stream) {
  this->nbondlist = nbond_tbl;
  if (nbondlist > 0)
    reallocate<bondlist_t>(&bondlist, &bondlist_len, nbondlist, 1.2f);

  this->nureyblist = nureyb_tbl;
  if (nureyblist > 0)
    reallocate<bondlist_t>(&ureyblist, &ureyblist_len, nureyblist, 1.2f);

  this->nanglelist = nangle_tbl;
  if (nanglelist > 0)
    reallocate<anglelist_t>(&anglelist, &anglelist_len, nanglelist, 1.2f);

  this->ndihelist = ndihe_tbl;
  if (ndihelist > 0)
    reallocate<dihelist_t>(&dihelist, &dihelist_len, ndihelist, 1.2f);

  this->nimdihelist = nimdihe_tbl;
  if (nimdihelist > 0)
    reallocate<dihelist_t>(&imdihelist, &imdihelist_len, nimdihelist, 1.2f);

  this->ncmaplist = ncmap_tbl;
  if (ncmaplist > 0)
    reallocate<cmaplist_t>(&cmaplist, &cmaplist_len, ncmaplist, 1.2f);

  float3 half_box;
  half_box.x = boxx * 0.5f;
  half_box.y = boxy * 0.5f;
  half_box.z = boxz * 0.5f;

  int nthread = 512;
  int nblock = (nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl +
                ncmap_tbl - 1) /
                   nthread +
               1;
  setup_list_kernel<<<nblock, nthread, 0, stream>>>(
      nbond_tbl, bond_tbl, bond, bondlist, nureyb_tbl, ureyb_tbl, ureyb,
      ureyblist, nangle_tbl, angle_tbl, angle, anglelist, ndihe_tbl, dihe_tbl,
      dihe, dihelist, nimdihe_tbl, imdihe_tbl, imdihe, imdihelist, ncmap_tbl,
      cmap_tbl, cmap, cmaplist, xyzq, half_box, glo2loc_ind);
  cudaCheck(cudaGetLastError());
}

//
// Print info
//
template <typename AT, typename CT> void CudaBondedForce<AT, CT>::print() {
  int maxnum = nbondlist;
  maxnum = std::max(maxnum, nureyblist);
  maxnum = std::max(maxnum, nanglelist);
  maxnum = std::max(maxnum, ndihelist);
  maxnum = std::max(maxnum, nimdihelist);
  maxnum = std::max(maxnum, ncmaplist);
  maxnum = std::max(maxnum, nbondcoef);
  maxnum = std::max(maxnum, nureybcoef);
  maxnum = std::max(maxnum, nanglecoef);
  maxnum = std::max(maxnum, ndihecoef);
  maxnum = std::max(maxnum, nimdihecoef);
  maxnum = std::max(maxnum, ncmapcoef);
  int maxw = (int)log10(maxnum + 1) + 2;
  std::cout << "BOND:   " << std::setw(maxw) << nbondlist << " "
            << std::setw(maxw) << nbondcoef << std::endl;
  std::cout << "UREYB:  " << std::setw(maxw) << nureyblist << " "
            << std::setw(maxw) << nureybcoef << std::endl;
  std::cout << "ANGLE:  " << std::setw(maxw) << nanglelist << " "
            << std::setw(maxw) << nanglecoef << std::endl;
  std::cout << "DIHE:   " << std::setw(maxw) << ndihelist << " "
            << std::setw(maxw) << ndihecoef << std::endl;
  std::cout << "IMDIHE: " << std::setw(maxw) << nimdihelist << " "
            << std::setw(maxw) << nimdihecoef << std::endl;
}

template <typename AT, typename CT> void CudaBondedForce<AT, CT>::clear(void) {
  energyVirial.clear(*bondedStream);
  forceVal->clear(*bondedStream);
  return;
}

//
// Calculates forces
//
template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::calc_force(
    const float4 *xyzq, const CT boxx, const CT boxy, const CT boxz,
    const bool calc_energy, const bool calc_virial, const int stride, AT *force,
    const bool calc_bond, const bool calc_ureyb, const bool calc_angle,
    const bool calc_dihe, const bool calc_imdihe, const bool calc_cmap,
    cudaStream_t stream) {

  if (calc_energy) {
    if (calc_bond) {
      int nthread = 512;
      int nblock = (nbondlist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_bond_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nbondlist, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strBond),
                energyVirial.getVirialPointer());
      } else {
        calc_bond_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nbondlist, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strBond), NULL);
      }
      cudaCheck(cudaGetLastError());
    }

    if (calc_ureyb) {
      int nthread = 512;
      int nblock = (nureyblist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_ureyb_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nureyblist, ureyblist, ureybcoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strUreyb),
                energyVirial.getVirialPointer());
      } else {
        calc_ureyb_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nureyblist, ureyblist, ureybcoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strUreyb), NULL);
      }
      cudaCheck(cudaGetLastError());
    }
    if (calc_angle) {
      // int nthread = 512;
      int nthread = 128;
      int nblock = (nanglelist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_angle_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nanglelist, anglelist, anglecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strAngle),
                energyVirial.getVirialPointer());
      } else {
        calc_angle_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nanglelist, anglelist, anglecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strAngle), NULL);
      }
      cudaCheck(cudaGetLastError());
    }

    if (calc_dihe) {
      // int nthread = 512;
      int nthread = 128;
      int nblock = (ndihelist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_dihe_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                ndihelist, dihelist, dihecoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strDihe),
                energyVirial.getVirialPointer());
      } else {
        calc_dihe_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                ndihelist, dihelist, dihecoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strDihe), NULL);
      }
      cudaCheck(cudaGetLastError());
    }

    if (calc_imdihe) {
      // int nthread = 512;
      int nthread = 128;
      int nblock = (nimdihelist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);
      if (calc_virial) {
        calc_imdihe_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                nimdihelist, imdihelist, imdihecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strImdihe),
                energyVirial.getVirialPointer());
      } else {
        calc_imdihe_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                nimdihelist, imdihelist, imdihecoef, xyzq, stride, boxx, boxy,
                boxz, force, energyVirial.getEnergyPointer(strImdihe), NULL);
      }
      cudaCheck(cudaGetLastError());
    }

    //JM260901 should we add size check to other kernel launches?
    if (calc_cmap && ncmaplist > 0) {
      int nthread = 128;
      int nblock = (ncmaplist - 1) / nthread + 1;
      int shmem_size =
          (nthread / ((get_cuda_arch() < 300) ? 1 : warpsize)) * sizeof(double);

      if (calc_virial) {
        calc_cmap_force_kernel<AT, CT, true, true>
            <<<nblock, nthread, shmem_size, stream>>>(
                ncmaplist, cmaplist, cmapcoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strCmap),
                energyVirial.getVirialPointer());
      } else {
        calc_cmap_force_kernel<AT, CT, true, false>
            <<<nblock, nthread, shmem_size, stream>>>(
                ncmaplist, cmaplist, cmapcoef, xyzq, stride, boxx, boxy, boxz,
                force, energyVirial.getEnergyPointer(strCmap), NULL);
      }

      cudaCheck(cudaGetLastError());
    }
  } else {
    int nbondlist_loc = (calc_bond) ? nbondlist : 0;
    int nureyblist_loc = (calc_ureyb) ? nureyblist : 0;
    int nanglelist_loc = (calc_angle) ? nanglelist : 0;
    int ndihelist_loc = (calc_dihe) ? ndihelist : 0;
    int nimdihelist_loc = (calc_imdihe) ? nimdihelist : 0;
    int ncmaplist_loc = (calc_cmap) ? ncmaplist : 0;

    // int nthread = 512;
    int nthread = 128;
    int nblock = (nbondlist_loc + nureyblist_loc + nanglelist_loc +
                  ndihelist_loc + nimdihelist_loc + ncmaplist_loc - 1) /
                     nthread +
                 1;
    int shmem_size = 0;

    if (calc_virial) {
      calc_all_forces_kernel<AT, CT, true>
          <<<nblock, nthread, shmem_size, stream>>>(
              nbondlist_loc, bondlist, bondcoef, nureyblist_loc, ureyblist,
              ureybcoef, nanglelist_loc, anglelist, anglecoef, ndihelist_loc,
              dihelist, dihecoef, nimdihelist_loc, imdihelist, imdihecoef, 
              ncmaplist_loc, cmaplist, cmapcoef, xyzq, stride, boxx, boxy, boxz,
              force, energyVirial.getVirialPointer());
    } else {
      calc_all_forces_kernel<AT, CT, false>
          <<<nblock, nthread, shmem_size, stream>>>(
              nbondlist_loc, bondlist, bondcoef, nureyblist_loc, ureyblist,
              ureybcoef, nanglelist_loc, anglelist, anglecoef, ndihelist_loc,
              dihelist, dihecoef, nimdihelist_loc, imdihelist, imdihecoef, 
              ncmaplist_loc, cmaplist, cmapcoef, xyzq, stride, boxx, boxy, boxz, 
              force, NULL);
    }
    cudaCheck(cudaGetLastError());
  }
}

template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setForce(
    std::shared_ptr<Force<long long int>> &forceValIn) {
  forceVal = forceValIn;
}

template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::setStream(
    std::shared_ptr<cudaStream_t> streamIn) {
  bondedStream = streamIn;
}

template <typename AT, typename CT>
void CudaBondedForce<AT, CT>::calc_force(const float4 *xyzq, bool calcEnergy,
                                         bool calcVirial) {

  calc_force(xyzq, boxDimensions[0], boxDimensions[1], boxDimensions[2],
             calcEnergy, calcVirial, forceVal->stride(), forceVal->xyz(), true,
             true, true, true, true, true, *bondedStream);
}

template <typename AT, typename CT>
std::shared_ptr<Force<long long int>> CudaBondedForce<AT, CT>::getForce(void) {
  return forceVal;
}

//
// Explicit instances of CudaBondedForce
//
template class CudaBondedForce<long long int, float>;
template class CudaBondedForce<long long int, double>;
#endif // NOCUDAC
