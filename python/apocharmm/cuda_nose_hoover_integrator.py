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
from .errors import check_status
from .cuda_integrator import CudaIntegrator

_prototypes_initialized = False


def _initialize_prototypes():
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

    #################
    #### Setters ####
    #######################################################################
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
    #######################################################################

    #################
    #### Getters ####
    #######################################################################
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
    #######################################################################

    ###################
    #### Functions ####
    #######################################################################
    lib().apo_cuda_nose_hoover_integrator_as_cuda_integrator.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib().apo_cuda_nose_hoover_integrator_as_cuda_integrator.restype = ctypes.c_int
    #######################################################################

    _prototypes_initialized = True


class CudaNoseHooverIntegrator(CudaIntegrator):
    _destroy_function_name = "apo_cuda_nose_hoover_integrator_destroy"

    def __init__(self, time_step):
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        status = lib().apo_cuda_nose_hoover_integrator_create(
            ctypes.byref(handle), float(time_step)
        )

        check_status(status, "CudaNoseHooverThermostatIntegrator construction failed")

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

    #################
    #### Setters ####
    #######################################################################
    def setReferenceTemperature(self, temperature):
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_set_reference_temperature(
            self.handle, temperature
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.setReferenceTemperature(temperature) failed",
        )

    def setNoseHooverPistonMass(self, mass):
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass(
            self.handle, mass
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.setNoseHooverPistonMass(mass) failed",
        )

    def useOldTemperature(self, flag=True):
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_use_old_temperature(
            self.handle, flag
        )

        check_status(
            status, "CudaNoseHooverThermostatIntegrator.useOldTemperature(flag) failed"
        )

    def resetAverageTemperature(self):
        _initialize_prototypes()

        status = lib().apo_cuda_nose_hoover_integrator_reset_average_temperature(
            self.handle
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.resetAverageTemperature() failed",
        )

    #######################################################################

    #################
    #### Getters ####
    #######################################################################
    def getReferenceTemperature(self):
        _initialize_prototypes()

        value = ctypes.c_double()

        status = lib().apo_cuda_nose_hoover_integrator_get_reference_temperature(
            ctypes.byref(value), self.handle
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.getReferenceTemperature() failed",
        )

        return float(value.value)

    def getAverageTemperature(self):
        _initialize_prototypes()

        value = ctypes.c_double()

        status = lib().apo_cuda_nose_hoover_integrator_get_average_temperature(
            ctypes.byref(value), self.handle
        )

        check_status(
            status, "CudaNoseHooverThermostatIntegrator.getAverageTemperature() failed"
        )

        return float(value.value)

    def getInstantaneousTemperature(self):
        _initialize_prototypes()

        value = ctypes.c_double()

        status = lib().apo_cuda_nose_hoover_integrator_get_instantaneous_temperature(
            ctypes.byref(value), self.handle
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.getInstantaneousTemperature() failed",
        )

        return float(value.value)

    #######################################################################

    ###################
    #### Functions ####
    #######################################################################
    #######################################################################
