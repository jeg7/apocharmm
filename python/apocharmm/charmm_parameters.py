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
from .errors import check_status

_prototypes_initialized = False


def _initialize_prototypes():
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_charmm_parameters_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
    ]
    lib().apo_charmm_parameters_create.restype = ctypes.c_int

    lib().apo_charmm_parameters_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_charmm_parameters_destroy.restype = None

    _prototypes_initialized = True


class CharmmParameters(_ApoObject):
    _destroy_function_name = "apo_charmm_parameters_destroy"

    def __init__(self, path):
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        status = lib().apo_charmm_parameters_create(
            ctypes.byref(handle), ctypes.c_char_p(encode_path(path))
        )

        check_status(status, "CharmmParameters construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_charmm_parameters_create returned success but prodced a NULL handle"
            )

        self._handle = handle
