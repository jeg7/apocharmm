# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
Shared validation for Python integers crossing fixed-width C ABI boundaries.
"""

import ctypes

_C_INT_BITS: int = ctypes.sizeof(ctypes.c_int) * 8
C_INT_MAX: int = (1 << (_C_INT_BITS - 1)) - 1
C_INT_MIN: int = -C_INT_MAX - 1

_UINT64_BITS: int = ctypes.sizeof(ctypes.c_uint64) * 8
UINT64_MAX: int = (1 << _UINT64_BITS) - 1


def _require_integer_in_range(
    value: object,
    argument_name: str,
    minimum: int,
    maximum: int,
    ctype_name: str,
    *,
    allow_bool: bool,
) -> int:
    if not isinstance(value, int) or (isinstance(value, bool) and not allow_bool):
        raise TypeError(f"{argument_name} must be an int")

    integer_value: int = int(value)

    if integer_value < minimum or integer_value > maximum:
        raise ValueError(f"{argument_name} must fit in {ctype_name}")

    return integer_value


def require_c_int(
    value: object, argument_name: str, *, allow_bool: bool = False
) -> int:
    """
    Returns a Python integer representable by signed C ``int``.

    Boolean values are rejected unless ``allow_bool`` is explicitly enabled.
    """
    return _require_integer_in_range(
        value, argument_name, C_INT_MIN, C_INT_MAX, "int", allow_bool=allow_bool
    )


def require_c_uint64(
    value: object, argument_name: str, *, allow_bool: bool = False
) -> int:
    """
    Returns a Python integer representable by C ``uint64_t``.

    Boolean values are rejected unless ``allow_bool`` is explicitly enabled.
    """
    return _require_integer_in_range(
        value, argument_name, 0, UINT64_MAX, "uint64_t", allow_bool=allow_bool
    )
