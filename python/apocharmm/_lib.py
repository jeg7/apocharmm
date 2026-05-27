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

from ._types import FilePath

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library

    if _library is not None:
        return _library

    library_path = os.environ.get("APOCHARMM_LIBRARY_PATH")
    if library_path is None or library_path == "":
        raise RuntimeError("APOCHARMM_LIBRARY_PATH must point to libapocharmm_c.so")

    library = ctypes.CDLL(library_path)

    library.apo_last_error.argtypes = []
    library.apo_last_error.restype = ctypes.c_char_p

    _library = library

    return library


def encode_path(path: FilePath) -> bytes:
    return os.fsencode(path)
