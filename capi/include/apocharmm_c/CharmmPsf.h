// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __APOCHARMM_C_CHARMM_PSF_H__
#define __APOCHARMM_C_CHARMM_PSF_H__

#include "apocharmm_c/Export.h"
#include "apocharmm_c/Status.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apo_charmm_psf apo_charmm_psf;

APOCHARMM_C_API apo_status apo_charmm_psf_create(apo_charmm_psf **out,
                                                 const char *path);

APOCHARMM_C_API void apo_charmm_psf_destroy(apo_charmm_psf *psf);

APOCHARMM_C_API apo_status
apo_charmm_psf_get_num_atoms(size_t *num_atoms, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status
apo_charmm_psf_get_num_bonds(size_t *num_bonds, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status
apo_charmm_psf_get_num_angles(size_t *num_angles, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_num_dihedrals(
    size_t *num_dihedrals, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_num_impropers(
    size_t *num_impropers, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_num_cross_terms(
    size_t *num_cross_terms, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_segment_identifiers(
    char *segis, const size_t len, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_residue_identifiers(
    int *resis, const size_t len, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_residue_names(
    char *resns, const size_t len, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_atom_names(
    char *anams, const size_t len, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_atom_types(
    char *atyps, const size_t len, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_charges(
    double *charges, const size_t len, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_masses(double *masses,
                                                     const size_t len,
                                                     const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status
apo_charmm_psf_get_net_charge(double *net_charge, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status
apo_charmm_psf_get_total_mass(double *total_mass, const apo_charmm_psf *psf);

APOCHARMM_C_API apo_status apo_charmm_psf_get_file_name(
    char *file_name, const size_t len, const apo_charmm_psf *psf);

#ifdef __cplusplus
}
#endif

#endif
