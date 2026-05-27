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

    lib().apo_charmm_crd_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
    ]
    lib().apo_charmm_crd_create.restype = ctypes.c_int

    lib().apo_charmm_crd_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_charmm_crd_destroy.restype = None

    lib().apo_charmm_crd_get_num_atoms.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_crd_get_num_atoms.restype = ctypes.c_int

    lib().apo_charmm_crd_get_coordinates.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_crd_get_coordinates.restype = ctypes.c_int

    _prototypes_initialized = True

    return


class CharmmCrd(_ApoObject):
    _destroy_function_name = "apo_charmm_crd_destroy"

    def __init__(self, path: FilePath) -> None:
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        status = lib().apo_charmm_crd_create(
            ctypes.byref(handle), ctypes.c_char_p(encode_path(path))
        )

        check_status(status, "CharmmCrd construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_charmm_crd_create returned success but produced a NULL handle"
            )

        self._handle = handle

        return

    def getNumAtoms(self) -> int:
        _initialize_prototypes()

        num_atoms = ctypes.c_size_t()

        status = lib().apo_charmm_crd_get_num_atoms(
            ctypes.byref(num_atoms), self.handle
        )

        check_status(status, "CharmmCrd.getNumAtoms() failed")

        return int(num_atoms.value)

    def getCoordinates(self) -> list[list[float]]:
        _initialize_prototypes()

        num_atoms = self.getNumAtoms()
        buffer_len = 3 * num_atoms

        buffer_type = ctypes.c_double * buffer_len
        buffer = buffer_type()

        status = lib().apo_charmm_crd_get_coordinates(buffer, buffer_len, self.handle)

        check_status(status, "CharmmCrd.getCoordinates() failed")

        coordinates: list[list[float]] = []
        for i in range(num_atoms):
            coordinates.append(
                [
                    float(buffer[i * 3 + 0]),
                    float(buffer[i * 3 + 1]),
                    float(buffer[i * 3 + 2]),
                ]
            )

        return coordinates
