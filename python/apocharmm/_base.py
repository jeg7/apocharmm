# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import ctypes

from types import TracebackType
from typing import ClassVar, Self

from ._lib import lib


class _ApoObject:
    _destroy_function_name: ClassVar[str | None] = None

    def __init__(self) -> None:
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        return

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None or self._handle.value is None:
            raise RuntimeError("apoCHARMM object has been destroyed")
        return self._handle

    def close(self) -> None:
        if self._handle is None or self._handle.value is None:
            return

        destroy_function_name = self._destroy_function_name
        if destroy_function_name is None:
            raise RuntimeError("No destroy function has been set for this object")

        destroy = getattr(lib(), destroy_function_name)
        destroy(self._handle)

        self._handle = ctypes.c_void_p()

        return

    def destroy(self) -> None:
        self.close()
        return

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        return
