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
from ._types import FilePaths
from .error import check_status

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_charmm_parameters_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
    ]
    lib().apo_charmm_parameters_create.restype = ctypes.c_int

    lib().apo_charmm_parameters_create_from_files.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
    ]
    lib().apo_charmm_parameters_create_from_files.restype = ctypes.c_int

    lib().apo_charmm_parameters_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_charmm_parameters_destroy.restype = None

    _prototypes_initialized = True

    return


class CharmmParameters(_ApoObject):
    _destroy_function_name = "apo_charmm_parameters_destroy"

    def __init__(self, paths: FilePaths) -> None:
        _initialize_prototypes()
        super().__init__()

        handle: ctypes.c_void_p = ctypes.c_void_p()

        if isinstance(paths, (list, tuple)):
            if len(paths) == 0:
                raise ValueError("CharmmParameters requires at least one file")

            encoded_paths: list[bytes] = [encode_path(path) for path in paths]
            num_paths: int = len(encoded_paths)
            c_num_paths: ctypes.c_size_t = ctypes.c_size_t(num_paths)

            path_array_type = ctypes.c_char_p * len(encoded_paths)
            path_array = path_array_type(*encoded_paths)

            status = lib().apo_charmm_parameters_create_from_files(
                ctypes.byref(handle), path_array, c_num_paths
            )

            function_name = "apo_charmm_parameters_create_from_files"
        else:
            encoded_path: bytes = encode_path(paths)
            c_path: ctypes.c_char_p = ctypes.c_char_p(encoded_path)

            status = lib().apo_charmm_parameters_create(ctypes.byref(handle), c_path)

            function_name = "apo_charmm_parameters_create"

        check_status(status, "CharmmParameters construction failed")

        if handle.value is None:
            raise RuntimeError(
                "{} returned success but prodced a NULL handle".format(function_name)
            )

        self._handle = handle

        return
