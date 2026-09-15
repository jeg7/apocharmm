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
from ._validation import require_c_uint64
from .error import configure_status_function
from .cuda_integrator import CudaIntegrator

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_double],
        "CudaLangevinThermostatIntegrator construction",
    )

    lib().apo_cuda_langevin_thermostat_integrator_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_cuda_langevin_thermostat_integrator_destroy.restype = None

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_set_reference_temperature,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaLangevinThermostatIntegrator.setReferenceTemperature(temperature)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_set_thermostat_friction,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaLangevinThermostatIntegrator.setThermostatFriction(friction)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_set_thermostat_rng_seed,
        [ctypes.c_void_p, ctypes.c_uint64],
        "CudaLangevinThermostatIntegrator.setThermostatRngSeed(seed)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_reset_average_temperature,
        [ctypes.c_void_p],
        "CudaLangevinThermostatIntegrator.resetAverageTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_get_reference_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinThermostatIntegrator.getReferenceTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_get_thermostat_friction,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinThermostatIntegrator.getThermostatFriction()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_get_thermostat_rng_seed,
        [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p],
        "CudaLangevinThermostatIntegrator.getThermostatRngSeed()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_get_average_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinThermostatIntegrator.getAverageTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_get_instantaneous_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinThermostatIntegrator.getInstantaneousTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_thermostat_integrator_as_cuda_integrator,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "CudaLangevinThermostatIntegrator base-integrator conversion",
    )

    _prototypes_initialized = True

    return


class CudaLangevinThermostatIntegrator(CudaIntegrator):
    """
    @brief Provides stochastic Langevin temperature control.

    The wrapper owns a concrete thermostat C handle. Native state includes one
    Philox generator per context atom and two kinetic and temperature
    estimators.

    @anchor python_cuda_langevin_thermostat_integrator
    @see cuda_integrators
    """

    _destroy_function_name = "apo_cuda_langevin_thermostat_integrator_destroy"

    def __init__(self, time_step: float) -> None:
        """
        @brief Constructs a Python-owned Langevin thermostat integrator.

        @param[in] time_step Value converted to C `double`, interpreted as a
        finite positive time step in picoseconds.
        @throws TypeError If C scalar conversion fails.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR`.
        @throws RuntimeError If a successful native call produces a `NULL`
        concrete or base-view handle.
        """
        _initialize_prototypes()
        super().__init__()

        handle: ctypes.c_void_p = ctypes.c_void_p()
        c_time_step: ctypes.c_double = ctypes.c_double(time_step)

        lib().apo_cuda_langevin_thermostat_integrator_create(
            ctypes.byref(handle), c_time_step
        )

        if handle.value is None:
            raise RuntimeError(
                "apo_cuda_langevin_thermostat_integrator_create returned success but produced a NULL handle"
            )

        self._handle = handle

        integrator_handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_cuda_langevin_thermostat_integrator_as_cuda_integrator(
            ctypes.byref(integrator_handle), self.handle
        )

        if integrator_handle.value is None:
            raise RuntimeError(
                "apo_cuda_langevin_thermostat_integrator_as_cuda_integrator returned success but produced a NULL handle"
            )

        self._integrator_handle = integrator_handle

        return

    def setReferenceTemperature(self, temperature: float) -> None:
        """
        @brief Sets the thermostat reference temperature.

        @param[in] temperature Value converted to C `double`, in kelvin. It must
        be finite and non-negative.
        @return `None`.
        @throws TypeError If conversion fails.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or execution fails.
        """
        _initialize_prototypes()

        c_temperature: ctypes.c_double = ctypes.c_double(temperature)

        lib().apo_cuda_langevin_thermostat_integrator_set_reference_temperature(
            self.handle, c_temperature
        )

        return

    def setThermostatFriction(self, friction: float) -> None:
        """
        @brief Sets the thermostat friction coefficient.

        @param[in] friction Value converted to C `double`, in inverse
        picoseconds. It must be finite and non-negative.
        @return `None`.
        @throws TypeError If conversion fails.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR`.
        @note Before the first propagation request, changing friction after
        context attachment reruns native concrete initialization.
        """
        _initialize_prototypes()

        c_friction: ctypes.c_double = ctypes.c_double(friction)

        lib().apo_cuda_langevin_thermostat_integrator_set_thermostat_friction(
            self.handle, c_friction
        )

        return

    def setThermostatRngSeed(self, seed: int) -> None:
        """
        @brief Sets the thermostat RNG seed.

        @param[in] seed Python integer in `[0, 2**64 - 1]`. The value is
        converted to `ctypes.c_uint64`.
        @return `None`.
        @throws TypeError If comparison or unsigned conversion rejects the
        object.
        @throws ValueError If the integer is outside the uint64 range.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If attached-context RNG allocation or
        initialization fails.
        @post The current native sequence position is preserved.
        """
        _initialize_prototypes()

        seed_value: int = require_c_uint64(seed, "seed", allow_bool=True)
        c_seed: ctypes.c_uint64 = ctypes.c_uint64(seed_value)

        lib().apo_cuda_langevin_thermostat_integrator_set_thermostat_rng_seed(
            self.handle, c_seed
        )

        return

    def resetAverageTemperature(self) -> None:
        """
        @brief Resets both running-temperature values and their sample count.

        @return `None`.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or device reset fails.
        """
        _initialize_prototypes()

        lib().apo_cuda_langevin_thermostat_integrator_reset_average_temperature(
            self.handle
        )

        return

    def getReferenceTemperature(self) -> float:
        """
        @brief Returns the reference temperature.

        @return A new Python `float` in kelvin.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or output fails.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_langevin_thermostat_integrator_get_reference_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)

    def getThermostatFriction(self) -> float:
        """
        @brief Returns the thermostat friction coefficient.

        @return A new Python `float` in inverse picoseconds.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or output fails.
        """
        _initialize_prototypes()

        c_friction = ctypes.c_double()

        lib().apo_cuda_langevin_thermostat_integrator_get_thermostat_friction(
            ctypes.byref(c_friction), self.handle
        )

        return float(c_friction.value)

    def getThermostatRngSeed(self) -> int:
        """
        @brief Returns the thermostat RNG seed.

        @return A new non-negative Python `int` in the uint64 range.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or output fails.
        """
        _initialize_prototypes()

        c_seed = ctypes.c_uint64()

        lib().apo_cuda_langevin_thermostat_integrator_get_thermostat_rng_seed(
            ctypes.byref(c_seed), self.handle
        )

        return int(c_seed.value)

    def getAverageTemperature(self) -> float:
        """
        @brief Returns the three-point running-average temperature.

        @return A new Python `float` containing native estimator element 0 in
        kelvin.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_CUDA_ERROR` for transfer failure,
        `APO_STATUS_RUNTIME_ERROR` for an invalid native array shape, or another
        documented native status.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_langevin_thermostat_integrator_get_average_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)

    def getInstantaneousTemperature(self) -> float:
        """
        @brief Returns the instantaneous three-point temperature.

        @return A new Python `float` in kelvin, derived from kinetic-energy
        element 0.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_NOT_INITIALIZED` before context attachment,
        `APO_STATUS_CUDA_ERROR` for transfer failure, or
        `APO_STATUS_RUNTIME_ERROR` for an invalid native array shape.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_langevin_thermostat_integrator_get_instantaneous_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)
