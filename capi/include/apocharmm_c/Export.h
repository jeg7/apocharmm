// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_EXPORT_H__
#define __APOCHARMM_C_EXPORT_H__

#if defined(_WIN32)
#if defined(APOCHARMM_C_BUILD)
#define APOCHARMM_C_API __declspec(dllexport)
#else
#define APOCHARMM_C_API __declspec(dllimport)
#endif
#else
#define APOCHARMM_C_API __attribute__((visibility("default")))
#endif

#endif
