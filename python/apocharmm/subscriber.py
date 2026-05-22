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
from ._lib import lib
from .errors import check_status

_prototypes_initialized = False


def _initialize_prototypes():
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_subscriber_set_report_frequency.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib().apo_subscriber_set_report_frequency.restype = ctypes.c_int

    lib().apo_subscriber_get_report_frequency.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
    ]
    lib().apo_subscriber_get_report_frequency.restype = ctypes.c_int

    _prototypes_initialized = True


class Subscriber(_ApoObject):
    def __init__(self):
        super().__init__()
        self._subscriber_handle = ctypes.c_void_p()

    @property
    def subscriber_handle(self):
        if self._subscriber_handle is None or self._subscriber_handle.value is None:
            raise RuntimeError("apoCHARMM subscriber object has been destroyed")
        return self._subscriber_handle

    def setReportFrequency(self, report_frequency):
        _initialize_prototypes()

        status = lib().apo_subscriber_set_report_frequency(
            self.subscriber_handle, int(report_frequency)
        )

        check_status(status, "Subscriber.setReportFrequency(report_frequency) failed")

    def getReportFrequency(self):
        _initialize_prototypes()

        value = ctypes.c_int()

        status = lib().apo_subscriber_get_report_frequency(
            ctypes.byref(value), self.subscriber_handle
        )

        check_status(status, "Subscriber.getReportFrequency() failed")

        return int(value.value)
