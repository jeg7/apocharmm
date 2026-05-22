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
from .errors import check_status

_prototypes_initialized = False

# cnht --> cuda_nose_hoover_thermostat


def _initialize_prototypes():
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_cnht_integrator_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_double,
    ]
    lib().apo_cnht_integrator_create.restype = ctypes.c_int

    lib().apo_cnht_integrator_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_cnht_integrator_destroy.restype = None

    #################
    #### Setters ####
    #######################################################################
    lib().apo_cnht_integrator_set_time_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    lib().apo_cnht_integrator_set_time_step.restype = ctypes.c_int

    lib().apo_cnht_integrator_set_charmm_context.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib().apo_cnht_integrator_set_charmm_context.restype = ctypes.c_int

    lib().apo_cnht_integrator_set_reference_temperature.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    lib().apo_cnht_integrator_set_reference_temperature.restype = ctypes.c_int

    lib().apo_cnht_integrator_set_nose_hoover_piston_mass.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    lib().apo_cnht_integrator_set_nose_hoover_piston_mass.restype = ctypes.c_int

    lib().apo_cnht_integrator_use_old_temperature.argtypes = [
        ctypes.c_void_p,
        ctypes.c_bool,
    ]
    lib().apo_cnht_integrator_use_old_temperature.restype = ctypes.c_int

    lib().apo_cnht_integrator_reset_average_temperature.argtypes = [ctypes.c_void_p]
    lib().apo_cnht_integrator_reset_average_temperature.restype = ctypes.c_int
    #######################################################################

    #################
    #### Getters ####
    #######################################################################
    lib().apo_cnht_integrator_get_reference_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_cnht_integrator_get_reference_temperature.restype = ctypes.c_int

    lib().apo_cnht_integrator_get_average_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_cnht_integrator_get_average_temperature.restype = ctypes.c_int

    lib().apo_cnht_integrator_get_instantaneous_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_cnht_integrator_get_instantaneous_temperature.restype = ctypes.c_int
    #######################################################################

    ###################
    #### Functions ####
    #######################################################################
    lib().apo_cnht_integrator_propagate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib().apo_cnht_integrator_propagate.restype = ctypes.c_int
    #######################################################################

    _prototypes_initialized = True


class CudaNoseHooverThermostatIntegrator(_ApoObject):
    _destroy_function_name = "apo_cuda_nose_hoover_thermostat_integrator_destroy"

    def __init__(self, time_step):
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        status = lib().apo_cnht_integrator_create(ctypes.byref(handle), time_step)

        check_status(status, "CudaNoseHooverThermostatIntegrator construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_cnht_integrator_create returned success but produced a NULL handle"
            )

        self._handle = handle

    #################
    #### Setters ####
    #######################################################################
    def setTimeStep(self, time_step):
        _initialize_prototypes()

        status = lib().apo_cnht_integrator_set_time_step(self.handle, time_step)

        check_status(
            status, "CudaNoseHooverThermostatIntegrator.setTimeStep(time_step) failed"
        )

    def setCharmmContext(self, context):
        _initialize_prototypes()

        status = lib().apo_cnht_integrator_set_charmm_context(
            self.handle, context.handle
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.setCharmmContext(context) failed",
        )

        self._context = context

    def setReferenceTemperature(self, temperature):
        _initialize_prototypes()

        status = lib().apo_cnht_integrator_set_reference_temperature(
            self.handle, temperature
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.setReferenceTemperature(temperature) failed",
        )

    def setNoseHooverPistonMass(self, mass):
        _initialize_prototypes()

        status = lib().apo_cnht_integrator_set_nose_hoover_piston_mass(
            self.handle, mass
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.setNoseHooverPistonMass(mass) failed",
        )

    def useOldTemperature(self, flag=True):
        _initialize_prototypes()

        status = lib().apo_cnht_integrator_use_old_temperature(self.handle, flag)

        check_status(
            status, "CudaNoseHooverThermostatIntegrator.useOldTemperature(flag) failed"
        )

    def resetAverageTemperature(self):
        _initialize_prototypes()

        status = lib().apo_cnht_integrator_reset_average_temperature(self.handle)

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

        buffer_type = ctypes.c_double * 1
        buffer = buffer_type()

        status = lib().apo_cnht_integrator_get_reference_temperature(
            buffer, self.handle
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.getReferenceTemperature() failed",
        )

        return float(buffer[0])

    def getAverageTemperature(self):
        _initialize_prototypes()

        buffer_type = ctypes.c_double * 1
        buffer = buffer_type()

        status = lib().apo_cnht_integrator_get_average_temperature(buffer, self.handle)

        check_status(
            status, "CudaNoseHooverThermostatIntegrator.getAverageTemperature() failed"
        )

        return float(buffer[0])

    def getInstantaneousTemperature(self):
        _initialize_prototypes()

        buffer_type = ctypes.c_double * 1
        buffer = buffer_type()

        status = lib().apo_cnht_integrator_get_instantaneous_temperature(
            buffer, self.handle
        )

        check_status(
            status,
            "CudaNoseHooverThermostatIntegrator.getInstantaneousTemperature() failed",
        )

        return float(buffer[0])

    #######################################################################

    ###################
    #### Functions ####
    #######################################################################
    def propagate(self, num_steps):
        _initialize_prototypes()

        status = lib().apo_cnht_integrator_propagate(self.handle, num_steps)

        check_status(status, "CudaNoseHooverThermostatIntegrator.propagate() failed")

    #######################################################################
