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

    lib().apo_force_manager_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib().apo_force_manager_create.restype = ctypes.c_int

    lib().apo_force_manager_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_force_manager_destroy.restype = None

    lib().apo_force_manager_set_box_dimensions.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib().apo_force_manager_set_box_dimensions.restype = ctypes.c_int

    lib().apo_force_manager_get_box_dimensions.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_force_manager_get_box_dimensions.restype = ctypes.c_int

    _prototypes_initialized = True


class ForceManager(_ApoObject):
    _destroy_function_name = "apo_force_manager_destroy"

    def __init__(self, psf, parameters):
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        status = lib().apo_force_manager_create(
            ctypes.byref(handle), psf.handle, parameters.handle
        )

        check_status(status, "ForceManager construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_force_manager_create returned success but produced a NULL handle"
            )

        self._handle = handle

        self._psf = psf
        self._parameters = parameters

    def setBoxDimensions(self, box_dimensions):
        _initialize_prototypes()

        values = list(box_dimensions)

        if len(values) != 3:
            raise RuntimeError(
                "ForceManager.setBoxDimensions expects exactly three values"
            )

        status = lib().apo_force_manager_set_box_dimensions(
            self.handle, float(values[0]), float(values[1]), float(values[2])
        )

        check_status(status, "ForceManager.setBoxDimensions() failed")

    # def getBoxDimensions(self):
    #     _initialize_prototypes()

    #     buffer_len = 3

    #     buffer_type = ctypes.c_double * buffer_len
    #     buffer = buffer_type()

    #     status = lib().apo_force_manager_get_box_dimensions(
    #         buffer, buffer_len, self._handle
    #     )

    #     check_status(status, "ForceManager.getBoxDimensions() failed")

    #     box_dimensions = [float(buffer[0]), float(buffer[1]), float(buffer[2])]

    #     return box_dimensions
