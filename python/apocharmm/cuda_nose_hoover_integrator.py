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
from .error import configure_status_function
from .cuda_integrator import CudaIntegrator

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_double],
        "CudaNoseHooverIntegrator construction",
    )

    lib().apo_cuda_nose_hoover_integrator_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_cuda_nose_hoover_integrator_destroy.restype = None

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_set_reference_temperature,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaNoseHooverIntegrator.setReferenceTemperature(temperature)",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaNoseHooverIntegrator.setNoseHooverPistonMass(mass)",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_use_old_temperature,
        [ctypes.c_void_p, ctypes.c_bool],
        "CudaNoseHooverIntegrator.useOldTemperature(flag)",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_reset_average_temperature,
        [ctypes.c_void_p],
        "CudaNoseHooverIntegrator.resetAverageTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_get_reference_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaNoseHooverIntegrator.getReferenceTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_get_nose_hoover_piston_mass,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaNoseHooverIntegrator.getNoseHooverPistonMass()",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_get_average_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaNoseHooverIntegrator.getAverageTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_get_instantaneous_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaNoseHooverIntegrator.getInstantaneousTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_nose_hoover_integrator_as_cuda_integrator,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "CudaNoseHooverIntegrator base-integrator coversion",
    )

    _prototypes_initialized = True

    return


class CudaNoseHooverIntegrator(CudaIntegrator):
    """
    @brief Provides deterministic Nose-Hoover temperature control.

    The wrapper owns a concrete Nose-Hoover C handle and inherits context,
    restart, subscription, propagation, and closure behavior from
    `CudaIntegrator`.

    @anchor python_cuda_nose_hoover_integrator
    @see cuda_integrators
    """

    _destroy_function_name = "apo_cuda_nose_hoover_integrator_destroy"

    def __init__(self, time_step: float) -> None:
        """
        @brief Constructs a Python-owned Nose-Hoover integrator.

        @param[in] time_step Value accepted by `ctypes.c_double`, interpreted as
        a finite positive time step in picoseconds.
        @throws TypeError If conversion to a C `double` fails.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR` if construction fails.
        @throws RuntimeError If native construction or base-view conversion
        reports success but produces a `NULL` handle.
        @post The wrapper owns the concrete handle and stores a borrowed base
        view.
        """
        _initialize_prototypes()
        super().__init__()

        handle: ctypes.c_void_p = ctypes.c_void_p()
        c_time_step: ctypes.c_double = ctypes.c_double(time_step)

        lib().apo_cuda_nose_hoover_integrator_create(ctypes.byref(handle), c_time_step)

        if handle.value is None:
            raise RuntimeError(
                "apo_cuda_nose_hoover_integrator_create returned success but produced a NULL handle"
            )

        self._handle = handle

        integrator_handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_cuda_nose_hoover_integrator_as_cuda_integrator(
            ctypes.byref(integrator_handle), self.handle
        )

        if integrator_handle.value is None:
            raise RuntimeError(
                "apo_cuda_nose_hoover_integrator_as_cuda_integrator returned success but produced a NULL handle"
            )

        self._integrator_handle = integrator_handle

        return

    def setReferenceTemperature(self, temperature: float) -> None:
        """
        @brief Sets the Nose-Hoover reference temperature.

        @param[in] temperature Value converted to C `double`, in kelvin. It must
        be finite and non-negative.
        @return `None`.
        @throws TypeError If C scalar conversion fails.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` for an invalid value or
        `APO_STATUS_RUNTIME_ERROR` for another native failure.
        """
        _initialize_prototypes()

        c_temperature: ctypes.c_double = ctypes.c_double(temperature)

        lib().apo_cuda_nose_hoover_integrator_set_reference_temperature(
            self.handle, c_temperature
        )

        return

    def setNoseHooverPistonMass(self, mass: float) -> None:
        """
        @brief Sets the scalar Nose-Hoover coupling mass.

        @param[in] mass Value converted to C `double`. It must be finite and
        non-negative. The exact dimensional convention is not established by
        the repository.
        @return `None`.
        @throws TypeError If C scalar conversion fails.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR`.
        @warning Zero is accepted, but active propagation divides by the mass.
        """
        _initialize_prototypes()

        c_mass: ctypes.c_double = ctypes.c_double(mass)

        lib().apo_cuda_nose_hoover_integrator_set_nose_hoover_piston_mass(
            self.handle, c_mass
        )

        return

    def useOldTemperature(self, flag: bool = True) -> None:
        """
        @brief Selects the native on-step temperature estimator.

        @param[in] flag Value converted with `ctypes.c_bool`. A truthy value
        selects kinetic-energy element 1 for native feedback and instantaneous
        temperature.
        @return `None`.
        @throws TypeError If C boolean conversion rejects the value.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError If the native handle is rejected.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool(flag)

        lib().apo_cuda_nose_hoover_integrator_use_old_temperature(self.handle, c_flag)

        return

    def resetAverageTemperature(self) -> None:
        """
        @brief Resets both running-temperature values and their sample count.

        @return `None`.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR`.
        """
        _initialize_prototypes()

        lib().apo_cuda_nose_hoover_integrator_reset_average_temperature(self.handle)

        return

    def getReferenceTemperature(self) -> float:
        """
        @brief Returns the reference temperature.

        @return A new Python `float` in kelvin.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError If the native handle or output operation fails.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_nose_hoover_integrator_get_reference_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)

    def getNoseHooverPistonMass(self) -> float:
        """
        @brief Returns the scalar Nose-Hoover coupling mass.

        @return A new Python `float` in the native coupling-mass convention.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError If the native handle or output operation fails.
        """
        _initialize_prototypes()

        c_mass = ctypes.c_double()

        lib().apo_cuda_nose_hoover_integrator_get_nose_hoover_piston_mass(
            ctypes.byref(c_mass), self.handle
        )

        return float(c_mass.value)

    def getAverageTemperature(self) -> float:
        """
        @brief Returns one native running-average temperature.

        @return A new Python `float` in kelvin.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR`.
        @note Returns the on-step estimator when old-temperature mode is enabled
        and the three-point estimator otherwise, as `getInstantaneousTemperature()`
        does.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_nose_hoover_integrator_get_average_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)

    def getInstantaneousTemperature(self) -> float:
        """
        @brief Returns the native selected instantaneous temperature.

        @return A new Python `float` in kelvin. Native selection uses element 1
        when old-temperature mode is enabled and element 0 otherwise.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_NOT_INITIALIZED` before context attachment,
        `APO_STATUS_CUDA_ERROR` for transfer failure, or another documented
        native status.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_nose_hoover_integrator_get_instantaneous_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)
