# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import ctypes

from ._lib import lib


class _ApoObject:
    _destroy_function_name = None

    def __init__(self):
        self._handle = ctypes.c_void_p()

    @property
    def handle(self):
        if self._handle is None or self._handle.value is None:
            raise RuntimeError("apoCHARMM object has been destroyed")
        return self._handle

    def close(self):
        if self._handle is None or self._handle.value is None:
            return

        if self._destroy_function_name is None:
            raise RuntimeError("No destroy function has been set for this object")

        destroy = getattr(lib(), self._destroy_function_name)
        destroy(self._handle)

        self._handle = ctypes.c_void_p()

    def destroy(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
