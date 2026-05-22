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
from .charmm_context import CharmmContext
from .errors import check_status
from .subscriber import Subscriber

_prototypes_initialized = False


def _initialize_prototypes():
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

    _prototypes_initialized = True


class CudaIntegrator(_ApoObject):
    def __init__(self):
        super().__init__()

        self._integrator_handle = ctypes.c_void_p()
        self._context = None
        self._subscribers = []

    @property
    def integrator_handle(self):
        if self._integrator_handle is None or self._integrator_handle.value is None:
            raise RuntimeError("apoCHARMM integrator object has been destroyed")

        return self._integrator_handle

    def close(self):
        super().close()

        self._integrator_handle = ctypes.c_void_p()
        self._context = None
        self._subscribers = []

    def setTimeStep(self, time_step):
        _initialize_prototypes()

        status = lib().apo_cuda_integrator_set_time_step(
            self.integrator_handle, float(time_step)
        )

        check_status(status, "CudaIntegrator.setTimeStep(time_step) failed")

    def setCharmmContext(self, context):
        _initialize_prototypes()

        if not isinstance(context, CharmmContext):
            raise TypeError("CudaIntegrator.setCharmmContext expects a CharmmContext")

        status = lib().apo_cuda_integrator_set_charmm_context(
            self.integrator_handle, context.handle
        )

        check_status(status, "CudaIntegrator.setCharmmContext(context) failed")

        self._context = context

    def subscribe(self, subscriber):
        _initialize_prototypes()

        if not isinstance(subscriber, Subscriber):
            raise TypeError("CudaIntegrator.subscribe expects a Subscriber")

        status = lib().apo_cuda_integrator_subscribe(
            self.integrator_handle, subscriber.subscriber_handle
        )

        check_status(status, "CudaIntegrator.subscribe(subscriber) failed")

        self._subscribers.append(subscriber)

    def propagate(self, num_steps):
        _initialize_prototypes()

        status = lib().apo_cuda_integrator_propagate(
            self.integrator_handle, int(num_steps)
        )

        check_status(status, "CudaIntegrator.propagate(num_steps) failed")
