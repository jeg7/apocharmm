# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import ctypes

from ._base import _ApoObject
from ._lib import encode_path, lib
from ._types import FilePath
from .error import check_status

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_charmm_psf_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
    ]
    lib().apo_charmm_psf_create.restype = ctypes.c_int

    lib().apo_charmm_psf_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_charmm_psf_destroy.restype = None

    lib().apo_charmm_psf_get_num_atoms.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_num_atoms.restype = ctypes.c_int

    lib().apo_charmm_psf_get_num_bonds.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_num_bonds.restype = ctypes.c_int

    lib().apo_charmm_psf_get_num_angles.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_num_angles.restype = ctypes.c_int

    lib().apo_charmm_psf_get_num_dihedrals.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_num_dihedrals.restype = ctypes.c_int

    lib().apo_charmm_psf_get_num_impropers.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_num_impropers.restype = ctypes.c_int

    lib().apo_charmm_psf_get_num_cross_terms.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_num_cross_terms.restype = ctypes.c_int

    lib().apo_charmm_psf_get_segment_identifiers.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_segment_identifiers.restype = ctypes.c_int

    lib().apo_charmm_psf_get_residue_identifiers.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_residue_identifiers.restype = ctypes.c_int

    lib().apo_charmm_psf_get_residue_names.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_residue_names.restype = ctypes.c_int

    lib().apo_charmm_psf_get_atom_names.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_atom_names.restype = ctypes.c_int

    lib().apo_charmm_psf_get_atom_types.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_atom_types.restype = ctypes.c_int

    lib().apo_charmm_psf_get_charges.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_charges.restype = ctypes.c_int

    lib().apo_charmm_psf_get_masses.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_masses.restype = ctypes.c_int

    lib().apo_charmm_psf_get_net_charge.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_net_charge.restype = ctypes.c_int

    lib().apo_charmm_psf_get_total_mass.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_total_mass.restype = ctypes.c_int

    lib().apo_charmm_psf_get_file_name.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_psf_get_file_name.restype = ctypes.c_int

    _prototypes_initialized = True

    return


class CharmmPsf(_ApoObject):
    _destroy_function_name = "apo_charmm_psf_destroy"

    def __init__(self, path: FilePath) -> None:
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        encoded_path: bytes = encode_path(path)
        c_path: ctypes.c_char_p = ctypes.c_char_p(encoded_path)

        status = lib().apo_charmm_psf_create(ctypes.byref(handle), c_path)

        check_status(status, "CharmmPsf construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_charmm_psf_create returned success but produced a NULL handle"
            )

        self._handle = handle

        return

    def getNumAtoms(self) -> int:
        _initialize_prototypes()

        num_atoms = ctypes.c_size_t()

        status = lib().apo_charmm_psf_get_num_atoms(
            ctypes.byref(num_atoms), self.handle
        )

        check_status(status, "CharmmPsf.getNumAtoms() failed")

        return int(num_atoms.value)

    def getNumBonds(self) -> int:
        _initialize_prototypes()

        num_bonds = ctypes.c_size_t()

        status = lib().apo_charmm_psf_get_num_bonds(
            ctypes.byref(num_bonds), self.handle
        )

        check_status(status, "CharmmPsf.getNumBonds() failed")

        return int(num_bonds.value)

    def getNumAngles(self) -> int:
        _initialize_prototypes()

        num_angles = ctypes.c_size_t()

        status = lib().apo_charmm_psf_get_num_angles(
            ctypes.byref(num_angles), self.handle
        )

        check_status(status, "CharmmPsf.getNumAngles() failed")

        return int(num_angles.value)

    def getNumDihedrals(self) -> int:
        _initialize_prototypes()

        num_dihedrals = ctypes.c_size_t()

        status = lib().apo_charmm_psf_get_num_dihedrals(
            ctypes.byref(num_dihedrals), self.handle
        )

        check_status(status, "CharmmPsf.getNumDihedrals() failed")

        return int(num_dihedrals.value)

    def getNumImpropers(self) -> int:
        _initialize_prototypes()

        num_impropers = ctypes.c_size_t()

        status = lib().apo_charmm_psf_get_num_impropers(
            ctypes.byref(num_impropers), self.handle
        )

        check_status(status, "CharmmPsf.getNumImpropers() failed")

        return int(num_impropers.value)

    def getNumCrossTerms(self) -> int:
        _initialize_prototypes()

        num_cross_terms = ctypes.c_size_t()

        status = lib().apo_charmm_psf_get_num_cross_terms(
            ctypes.byref(num_cross_terms), self.handle
        )

        check_status(status, "CharmmPsf.getNumCrossTerms() failed")

        return int(num_cross_terms.value)

    def getSegmentIdentifiers(self) -> list[str]:
        _initialize_prototypes()

        num_segis: int = self.getNumAtoms()
        buffer_len: int = 8 * num_segis
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_char * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_segment_identifiers(
            buffer, c_buffer_len, self.handle
        )

        check_status(status, "CharmmPsf.getSegmentIdentifiers() failed")

        raw = buffer.raw

        segis: list[str] = []
        for i in range(num_segis):
            start = i * 8
            stop = start + 8
            segi = raw[start:stop].decode("utf-8")
            segis.append(segi.replace(" ", ""))  # Remove whitespace

        return segis

    def getResidueIdentifiers(self) -> list[int]:
        _initialize_prototypes()

        num_resis = self.getNumAtoms()
        buffer_len = num_resis
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_int * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_residue_identifiers(
            buffer, c_buffer_len, self.handle
        )

        check_status(status, "CharmmPsf.getResidueIdentifiers() failed")

        resis: list[int] = []
        for i in range(num_resis):
            resis.append(int(buffer[i]))

        return resis

    def getResidueNames(self) -> list[str]:
        _initialize_prototypes()

        num_resns = self.getNumAtoms()
        buffer_len = 8 * num_resns
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_char * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_residue_names(
            buffer, c_buffer_len, self.handle
        )

        check_status(status, "CharmmPsf.getResidueNames() failed")

        raw = buffer.raw

        resns: list[str] = []
        for i in range(num_resns):
            start = i * 8
            stop = start + 8
            resn = raw[start:stop].decode("utf-8")
            resns.append(resn.replace(" ", ""))  # Remove whitespace

        return resns

    def getAtomNames(self) -> list[str]:
        _initialize_prototypes()

        num_names = self.getNumAtoms()
        buffer_len = 8 * num_names
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_char * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_atom_names(buffer, c_buffer_len, self.handle)

        check_status(status, "CharmmPsf.getAtomNames() failed")

        raw = buffer.raw

        atom_names: list[str] = []
        for i in range(num_names):
            start = i * 8
            stop = start + 8
            atom_name = raw[start:stop].decode("utf-8")
            atom_names.append(atom_name.replace(" ", ""))  # Remove whitespace

        return atom_names

    def getAtomTypes(self) -> list[str]:
        _initialize_prototypes()

        num_types = self.getNumAtoms()
        buffer_len = 8 * num_types
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_char * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_atom_types(buffer, c_buffer_len, self.handle)

        check_status(status, "CharmmPsf.getAtomTypes() failed")

        raw = buffer.raw

        atom_types: list[str] = []
        for i in range(num_types):
            start = i * 8
            stop = start + 8
            atom_type = raw[start:stop].decode("utf-8")
            atom_types.append(atom_type.replace(" ", ""))  # Remove whitespace

        return atom_types

    def getCharges(self) -> list[float]:
        _initialize_prototypes()

        num_atoms = self.getNumAtoms()
        buffer_len = num_atoms
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_double * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_charges(buffer, c_buffer_len, self.handle)

        check_status(status, "CharmmPsf.getCharges() failed")

        charges: list[float] = []
        for i in range(num_atoms):
            charges.append(float(buffer[i]))

        return charges

    def getMasses(self) -> list[float]:
        _initialize_prototypes()

        num_atoms = self.getNumAtoms()
        buffer_len = num_atoms
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_double * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_masses(buffer, c_buffer_len, self.handle)

        check_status(status, "CharmmPsf.getMasses() failed")

        masses: list[float] = []
        for i in range(num_atoms):
            masses.append(float(buffer[i]))

        return masses

    def getNetCharge(self) -> float:
        _initialize_prototypes()

        net_charge = ctypes.c_double()

        status = lib().apo_charmm_psf_get_net_charge(
            ctypes.byref(net_charge), self.handle
        )

        check_status(status, "CharmmPsf.getNetCharge() failed")

        return float(net_charge.value)

    def getTotalMass(self) -> float:
        _initialize_prototypes()

        total_mass = ctypes.c_double()

        status = lib().apo_charmm_psf_get_total_mass(
            ctypes.byref(total_mass), self.handle
        )

        check_status(status, "CharmmPsf.getTotalMass() failed")

        return float(total_mass.value)

    def getFileName(self) -> str:
        _initialize_prototypes()

        # We are going to assume that the file name is less than 1024 chars
        buffer_len = 1024
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(buffer_len)

        buffer_type = ctypes.c_char * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_psf_get_file_name(buffer, c_buffer_len, self.handle)

        check_status(status, "CharmmPsf.getFileName() failed")

        file_name = buffer.raw.decode("utf-8")

        return file_name.rstrip(" ")
