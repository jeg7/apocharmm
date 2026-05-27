# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

APO_STATUS_OK: int = 0
APO_STATUS_INVALID_ARGUMENT: int = 1
APO_STATUS_RUNTIME_ERROR: int = 2
APO_STATUS_CUDA_ERROR: int = 3
APO_STATUS_NOT_INITIALIZED: int = 4
APO_STATUS_NOT_IMPLEMENTED: int = 5


class ApoCharmmError(RuntimeError):
    pass


def check_status(status: int, context: str) -> None:
    if status == APO_STATUS_OK:
        return

    from ._lib import lib

    message = lib().apo_last_error()
    if message is None:
        decoded: str = "Unknown apoCHARMM C API error"
    else:
        decoded = message.decode("utf-8", errors="replace")

    raise ApoCharmmError("{}: {}".format(context, decoded))
