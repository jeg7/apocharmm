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
from ._types import FilePath
from .error import check_status

from .charmm_context import CharmmContext
from .subscriber import Subscriber

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_cuda_integrator_set_time_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    lib().apo_cuda_integrator_set_time_step.restype = ctypes.c_int

    lib().apo_cuda_integrator_set_charmm_context.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib().apo_cuda_integrator_set_charmm_context.restype = ctypes.c_int

    lib().apo_cuda_integrator_subscribe.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib().apo_cuda_integrator_subscribe.restype = ctypes.c_int

    lib().apo_cuda_integrator_propagate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib().apo_cuda_integrator_propagate.restype = ctypes.c_int

    lib().apo_cuda_integrator_initialize_from_restart_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib().apo_cuda_integrator_initialize_from_restart_file.restype = ctypes.c_int

    _prototypes_initialized = True

    return


class CudaIntegrator(_ApoObject):
    def __init__(self) -> None:
        super().__init__()

        self._integrator_handle: ctypes.c_void_p = ctypes.c_void_p()
        self._context: CharmmContext = None
        self._subscribers: list[Subscriber] = []

        return

    @property
    def integrator_handle(self) -> ctypes.c_void_p:
        if self._integrator_handle is None or self._integrator_handle.value is None:
            raise RuntimeError("apoCHARMM integrator object has been destroyed")

        return self._integrator_handle

    def close(self) -> None:
        super().close()

        self._integrator_handle = ctypes.c_void_p()
        self._context = None
        self._subscribers = []

        return

    def setTimeStep(self, time_step: float) -> None:
        _initialize_prototypes()

        status = lib().apo_cuda_integrator_set_time_step(
            self.integrator_handle, time_step
        )

        check_status(status, "CudaIntegrator.setTimeStep(time_step) failed")

        return

    def setCharmmContext(self, context: CharmmContext) -> None:
        _initialize_prototypes()

        if not isinstance(context, CharmmContext):
            raise TypeError("CudaIntegrator.setCharmmContext expects a CharmmContext")

        status = lib().apo_cuda_integrator_set_charmm_context(
            self.integrator_handle, context.handle
        )

        check_status(status, "CudaIntegrator.setCharmmContext(context) failed")

        self._context = context

        return

    def subscribe(self, subscriber: Subscriber) -> None:
        _initialize_prototypes()

        if not isinstance(subscriber, Subscriber):
            raise TypeError("CudaIntegrator.subscribe expects a Subscriber")

        status = lib().apo_cuda_integrator_subscribe(
            self.integrator_handle, subscriber.subscriber_handle
        )

        check_status(status, "CudaIntegrator.subscribe(subscriber) failed")

        self._subscribers.append(subscriber)

        return

    def propagate(self, num_steps: int) -> None:
        _initialize_prototypes()

        status = lib().apo_cuda_integrator_propagate(self.integrator_handle, num_steps)

        check_status(status, "CudaIntegrator.propagate(num_steps) failed")

        return

    def initializeFromRestartFile(self, path: FilePath) -> None:
        _initialize_prototypes()

        status = lib().apo_cuda_integrator_initialize_from_restart_file(
            self.integrator_handle, ctypes.c_char_p(encode_path(path))
        )

        check_status(status, "CudaIntegrator.initializeFromRestartFile(path) failed")

        return
