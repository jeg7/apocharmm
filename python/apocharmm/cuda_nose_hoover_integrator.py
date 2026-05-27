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
from .error import check_status
from .cuda_integrator import CudaIntegrator

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_cuda_nose_hoover_integrator_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_double,
    ]
    lib().apo_cuda_nose_hoover_integrator_create.restype = ctypes.c_int

    lib().apo_cuda_nose_hoover_integrator_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_cuda_nose_hoover_integrator_destroy.restype = None

    lib().apo_cuda_nose_hoover_integrator_set_reference_temperature.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    lib().apo_cuda_nose_hoover_integrator_set_reference_temperature.restype = (
        ctypes.c_int
    )

    lib().apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    lib().apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass.restype = (
        ctypes.c_int
    )

    lib().apo_cuda_nose_hoover_integrator_use_old_temperature.argtypes = [
        ctypes.c_void_p,
        ctypes.c_bool,
    ]
    lib().apo_cuda_nose_hoover_integrator_use_old_temperature.restype = ctypes.c_int

    lib().apo_cuda_nose_hoover_integrator_reset_average_temperature.argtypes = [
        ctypes.c_void_p
    ]
    lib().apo_cuda_nose_hoover_integrator_reset_average_temperature.restype = (
        ctypes.c_int
    )

    lib().apo_cuda_nose_hoover_integrator_get_reference_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_cuda_nose_hoover_integrator_get_reference_temperature.restype = (
        ctypes.c_int
    )

    lib().apo_cuda_nose_hoover_integrator_get_average_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_cuda_nose_hoover_integrator_get_average_temperature.restype = ctypes.c_int

    lib().apo_cuda_nose_hoover_integrator_get_instantaneous_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_cuda_nose_hoover_integrator_get_instantaneous_temperature.restype = (
        ctypes.c_int
    )

    lib().apo_cuda_nose_hoover_integrator_as_cuda_integrator.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib().apo_cuda_nose_hoover_integrator_as_cuda_integrator.restype = ctypes.c_int

    _prototypes_initialized = True

    return


class CudaNoseHooverIntegrator(CudaIntegrator):
    _destroy_function_name = "apo_cuda_nose_hoover_integrator_destroy"

    def __init__(self, time_step: float) -> None:
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        status = lib().apo_cuda_nose_hoover_integrator_create(
            ctypes.byref(handle), time_step
        )

        check_status(status, "CudaNoseHooverIntegrator construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_cuda_nose_hoover_integrator_create returned success but produced a NULL handle"
            )

        self._handle = handle

        integrator_handle = ctypes.c_void_p()

        status = lib().apo_cuda_nose_hoover_integrator_as_cuda_integrator(
            ctypes.byref(integrator_handle), self.handle
        )

        check_status(
            status, "CudaNoseHooverIntegrator base-integrator conversion failed"
        )

        if integrator_handle.value is None:
            raise RuntimeError(
                "apo_cuda_nose_hoover_integrator_as_cuda_integrator returned success but produced a NULL handle"
            )

        self._integrator_handle = integrator_handle

        return

    def setReferenceTemperature(self, temperature: float) -> None:
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_set_reference_temperature(
            self.handle, temperature
        )

        check_status(
            status,
            "CudaNoseHooverIntegrator.setReferenceTemperature(temperature) failed",
        )

        return

    def setNoseHooverPistonMass(self, mass: float) -> None:
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass(
            self.handle, mass
        )

        check_status(
            status,
            "CudaNoseHooverIntegrator.setNoseHooverPistonMass(mass) failed",
        )

        return

    def useOldTemperature(self, flag: bool = True) -> None:
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_use_old_temperature(
            self.handle, flag
        )

        check_status(status, "CudaNoseHooverIntegrator.useOldTemperature(flag) failed")

        return

    def resetAverageTemperature(self) -> None:
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_reset_average_temperature(
            self.handle
        )

        check_status(
            status,
            "CudaNoseHooverIntegrator.resetAverageTemperature() failed",
        )

    def getReferenceTemperature(self) -> float:
        _initialize_prototypes()

        reference_temperature = ctypes.c_double()

        status = lib().apo_cuda_nose_hoover_integrator_get_reference_temperature(
            ctypes.byref(reference_temperature), self.handle
        )

        check_status(
            status,
            "CudaNoseHooverIntegrator.getReferenceTemperature() failed",
        )

        return float(reference_temperature.value)

    def getAverageTemperature(self) -> float:
        _initialize_prototypes()

        average_temperature = ctypes.c_double()

        status = lib().apo_cuda_nose_hoover_integrator_get_average_temperature(
            ctypes.byref(average_temperature), self.handle
        )

        check_status(status, "CudaNoseHooverIntegrator.getAverageTemperature() failed")

        return float(average_temperature.value)

    def getInstantaneousTemperature(self) -> float:
        _initialize_prototypes()

        instantaneous_temperature = ctypes.c_double()

        status = lib().apo_cuda_nose_hoover_integrator_get_instantaneous_temperature(
            ctypes.byref(instantaneous_temperature), self.handle
        )

        check_status(
            status,
            "CudaNoseHooverIntegrator.getInstantaneousTemperature() failed",
        )

        return float(instantaneous_temperature.value)
