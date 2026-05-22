# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

APO_STATUS_OK = 0


class ApoCharmmError(RuntimeError):
    pass


def check_status(status, context):
    if status == APO_STATUS_OK:
        return

    from ._lib import lib

    message = lib().apo_last_error()
    if message is None:
        decoded = "Unknown apoCHARMM C API error"
    else:
        decoded = message.decode("utf-8", errors="replace")

    raise ApoCharmmError(f"{context}: {decoded}")
