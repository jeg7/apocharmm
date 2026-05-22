# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import ctypes

from ._lib import encode_path, lib
from .errors import check_status
from .subscriber import Subscriber

_prototypes_initialized = False


def _initialize_prototypes():
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_dcd_subscriber_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
    ]
    lib().apo_dcd_subscriber_create.restype = ctypes.c_int

    lib().apo_dcd_subscriber_create_with_report_frequency.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib().apo_dcd_subscriber_create_with_report_frequency.restype = ctypes.c_int

    lib().apo_dcd_subscriber_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_dcd_subscriber_destroy.restype = None

    lib().apo_dcd_subscriber_as_subscriber.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib().apo_dcd_subscriber_as_subscriber.restype = ctypes.c_int

    _prototypes_initialized = True


class DcdSubscriber(Subscriber):
    _destroy_function_name = "apo_dcd_subscriber_destroy"

    def __init__(self, file_name, report_frequency=None):
        _initialize_prototypes()
        super().__init__

        handle = ctypes.c_void_p()
        encoded_file_name = encode_path(file_name)

        if report_frequency is None:
            status = lib().apo_dcd_subscriber_create(
                ctypes.byref(handle), encoded_file_name
            )
        else:
            status = lib().apo_dcd_subscriber_create_with_report_frequency(
                ctypes.byref(handle), encoded_file_name, int(report_frequency)
            )

        check_status(status, "DcdSubscriber construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_dcd_subscriber_create returned success but produced a NULL handle"
            )

        self._handle = handle

        subscriber_handle = ctypes.c_void_p()

        status = lib().apo_dcd_subscriber_as_subscriber(
            ctypes.byref(subscriber_handle), self.handle
        )

        check_status(status, "DcdSubscriber base-subscriber conversion failed")

        if subscriber_handle.value is None:
            raise RuntimeError(
                "apo_dcd_subscriber_as_subscriber returned success but produced a NULL handle"
            )

        self._subscriber_handle = subscriber_handle
        self._file_name = file_name

    def close(self):
        super().close()
        self._subscriber_handle = ctypes.c_void_p()
