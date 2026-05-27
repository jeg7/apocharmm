// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/Validation.h"

#include "CharmmPSF.h"

#include <memory>
#include <string>

extern "C" apo_status apo_charmm_psf_create(apo_charmm_psf **out,
                                            const char *path) {
  const char *function_name = "apo_charmm_psf_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_charmm_psf>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_c_string(path, function_name, "PSF path"));

        std::unique_ptr<apo_charmm_psf> handle(new apo_charmm_psf());
        handle->object = std::make_shared<CharmmPSF>(std::string(path));

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_charmm_psf_destroy(apo_charmm_psf *psf) { delete psf; }

extern "C" apo_status apo_charmm_psf_get_num_atoms(size_t *num_atoms,
                                                   const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_num_atoms";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_atoms, function_name, "num_atoms"));

        *num_atoms = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const int n = psf->object->getNumAtoms();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_psf_get_num_atoms: CharmmPSF returned a negative "
              "atom count");
        }

        *num_atoms = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_num_bonds(size_t *num_bonds,
                                                   const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_num_bonds";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_bonds, function_name, "num_bonds"));

        *num_bonds = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const int n = psf->object->getNumBonds();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_psf_get_num_bonds: CharmmPSF returned a negative "
              "bond count");
        }

        *num_bonds = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_num_angles(size_t *num_angles,
                                                    const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_num_angles";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_angles, function_name, "num_angles"));

        *num_angles = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const int n = psf->object->getNumAngles();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_psf_get_num_angles: CharmmPSF returned a negative "
              "angle count");
        }

        *num_angles = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_psf_get_num_dihedrals(size_t *num_dihedrals,
                                 const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_num_dihedrals";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_dihedrals, function_name, "num_dihedrals"));

        *num_dihedrals = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const int n = psf->object->getNumDihedrals();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_psf_get_num_dihedrals: CharmmPSF returned a negative "
              "dihedral count");
        }

        *num_dihedrals = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_psf_get_num_impropers(size_t *num_impropers,
                                 const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_num_impropers";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_impropers, function_name, "num_impropers"));

        *num_impropers = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const int n = psf->object->getNumImpropers();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_charmm_psf_get_num_impropers: CharmmPSF returned a negative "
              "improper count");
        }

        *num_impropers = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_psf_get_num_cross_terms(size_t *num_cross_terms,
                                   const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_num_cross_terms";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_cross_terms, function_name, "num_cross_terms"));

        *num_cross_terms = 0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const int n = psf->object->getNumCrossTerms();

        if (n < 0) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR, "apo_charmm_psf_get_num_cross_terms: "
                                        "CharmmPSF returned a negative "
                                        "cross_term count");
        }

        *num_cross_terms = static_cast<size_t>(n);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_psf_get_segment_identifiers(char *segis, const size_t len,
                                       const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_segment_identifiers";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::vector<std::string> &segids =
            psf->object->getSegmentIdentifiers();
        const size_t req_len =
            8 * segids.size(); // Assume that each segi is 8 characters max

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<char>(
            segis, len, req_len, function_name, "Segment identifier buffer"));

        for (size_t i = 0; i < segids.size(); i++) {
          for (size_t j = 0; j < 8; j++)
            segis[i * 8 + j] = (j < segids[i].length()) ? segids[i][j] : ' ';
        }

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_psf_get_residue_identifiers(int *resis, const size_t len,
                                       const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_residue_identifiers";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::vector<int> &resids = psf->object->getResidueIdentifiers();
        const size_t req_len = resids.size();

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<int>(
            resis, len, req_len, function_name, "Residue identifier buffer"));

        for (size_t i = 0; i < resids.size(); i++)
          resis[i] = resids[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_charmm_psf_get_residue_names(char *resns, const size_t len,
                                 const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_residue_names";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::vector<std::string> &resnames =
            psf->object->getResidueNames();
        const size_t req_len =
            8 * resnames.size(); // Assume that each resn is 8 characters max

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<char>(
            resns, len, req_len, function_name, "Residue name buffer"));

        for (size_t i = 0; i < resnames.size(); i++) {
          for (size_t j = 0; j < 8; j++) {
            resns[i * 8 + j] =
                (j < resnames[i].length()) ? resnames[i][j] : ' ';
          }
        }

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_atom_names(char *anams,
                                                    const size_t len,
                                                    const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_atom_names";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::vector<std::string> &atomnames = psf->object->getAtomNames();
        const size_t req_len =
            8 *
            atomnames.size(); // Assume that each atom name is 8 characters max

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<char>(
            anams, len, req_len, function_name, "Atom name buffer"));

        for (size_t i = 0; i < atomnames.size(); i++) {
          for (size_t j = 0; j < 8; j++) {
            anams[i * 8 + j] =
                (j < atomnames[i].length()) ? atomnames[i][j] : ' ';
          }
        }

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_atom_types(char *atyps,
                                                    const size_t len,
                                                    const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_atom_types";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::vector<std::string> &atomtypes = psf->object->getAtomTypes();
        const size_t req_len =
            8 *
            atomtypes.size(); // Assume that each atom type is 8 characters max

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<char>(
            atyps, len, req_len, function_name, "Atom name buffer"));

        for (size_t i = 0; i < atomtypes.size(); i++) {
          for (size_t j = 0; j < 8; j++) {
            atyps[i * 8 + j] =
                (j < atomtypes[i].length()) ? atomtypes[i][j] : ' ';
          }
        }

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_charges(double *charges,
                                                 const size_t len,
                                                 const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_charges";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::vector<double> &chrgs = psf->object->getCharges();
        const size_t req_len = chrgs.size();

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            charges, len, req_len, function_name, "Charge buffer"));

        for (size_t i = 0; i < chrgs.size(); i++)
          charges[i] = chrgs[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_masses(double *masses,
                                                const size_t len,
                                                const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_masses";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::vector<double> &amass = psf->object->getMasses();
        const size_t req_len = amass.size();

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            masses, len, req_len, function_name, "Charge buffer"));

        for (size_t i = 0; i < amass.size(); i++)
          masses[i] = amass[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_net_charge(double *net_charge,
                                                    const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_net_charge";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            net_charge, function_name, "net_charge"));

        *net_charge = 0.0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        *net_charge = psf->object->getNetCharge();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_total_mass(double *total_mass,
                                                    const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_total_mass";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            total_mass, function_name, "total_mass"));

        *total_mass = 0.0;

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        *total_mass = psf->object->getNetCharge();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_charmm_psf_get_file_name(char *file_name,
                                                   const size_t len,
                                                   const apo_charmm_psf *psf) {
  const char *function_name = "apo_charmm_psf_get_file_name";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_handle_object<apo_charmm_psf>(
                psf, function_name, "CharmmPsf"));

        const std::string &fname = psf->object->getFileName();
        const size_t req_len = fname.length();

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<char>(
            file_name, len, req_len, function_name, "File name buffer"));

        for (size_t i = 0; i < len; i++)
          file_name[i] = (i < fname.length()) ? fname[i] : ' ';

        return APO_STATUS_OK;
      },
      function_name);
}
