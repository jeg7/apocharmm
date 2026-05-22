# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import ctypes
import os

_library = None


def lib():
    global _library

    if _library is not None:
        return _library

    library_path = os.environ.get("APOCHARMM_LIBRARY_PATH")
    if library_path is None or library_path == "":
        raise RuntimeError("APOCHARMM_LIBRARY_PATH must point to libapocharmm_c.so")

    _library = ctypes.CDLL(library_path)

    _library.apo_last_error.argtypes = []
    _library.apo_last_error.restype = ctypes.c_char_p

    return _library


def encode_path(path):
    return os.fsencode(path)
