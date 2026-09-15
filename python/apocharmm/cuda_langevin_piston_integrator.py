# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from collections.abc import Sequence
import ctypes

from ._lib import lib
from ._validation import require_c_uint64
from .enums import CrystalType
from .error import configure_status_function
from .cuda_integrator import CudaIntegrator

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_double],
        "CudaLangevinPistonIntegrator construction",
    )

    lib().apo_cuda_langevin_piston_integrator_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_cuda_langevin_piston_integrator_destroy.restype = None

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_use_nose_hoover_thermostat,
        [ctypes.c_void_p, ctypes.c_bool],
        "CudaLangevinPistonIntegrator.useNoseHooverThermostat(flag)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_reference_temperature,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaLangevinPistonIntegrator.setReferenceTemperature(temperature)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_nose_hoover_piston_mass,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaLangevinPistonIntegrator.setNoseHooverPistonMass(mass)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_use_old_temperature,
        [ctypes.c_void_p, ctypes.c_bool],
        "CudaLangevinPistonIntegrator.useOldTemperature(flag)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_reference_pressure,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CudaLangevinPistonIntegrator.setReferencePressure(pressure_tensor)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_constant_surface_tension,
        [ctypes.c_void_p, ctypes.c_bool],
        "CudaLangevinPistonIntegrator.setConstantSurfaceTension(flag)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_crystal_type,
        [ctypes.c_void_p, ctypes.c_int],
        "CudaLangevinPistonIntegrator.setCrystalType(crystal_type)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_langevin_piston_mass,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CudaLangevinPistonIntegrator.setLangevinPistonMass(mass)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_langevin_piston_friction_seed,
        [ctypes.c_void_p, ctypes.c_uint64],
        "CudaLangevinPistonIntegrator.setLangevinPistonFrictionSeed(seed)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_set_langevin_piston_friction,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaLangevinPistonIntegrator.setLangevinPistonFriction(friction)",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_reset_averages,
        [ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.resetAverages()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_reference_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getReferenceTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_nose_hoover_piston_mass,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getNoseHooverPistonMass()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_average_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getAverageTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_reference_pressure_tensor,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getReferencePressureTensor()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_crystal_type,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getCrystalType()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_langevin_piston_mass,
        [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
        "CudaLangevinPistonIntegrator.getLangevinPistonMass()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_tensor,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getInstantaneousPressureTensor()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_scalar,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getInstantaneousPressureScalar()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_average_pressure_tensor,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getAveragePressureTensor()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_average_pressure_scalar,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getAveragePressureScalar()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_langevin_piston_friction_seed,
        [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getLangevinPistonFrictionSeed()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_get_instantaneous_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator.getInstantaneousTemperature()",
    )

    configure_status_function(
        lib().apo_cuda_langevin_piston_integrator_as_cuda_integrator,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "CudaLangevinPistonIntegrator base-integrator conversion",
    )

    _prototypes_initialized = True

    return


class CudaLangevinPistonIntegrator(CudaIntegrator):
    """
    @brief Provides pressure-controlled Langevin-piston dynamics.

    Select a supported `CrystalType` before attaching a context. The wrapper
    owns the concrete piston C handle and inherits common context, propagation,
    restart, subscriber, and closure operations.

    @anchor python_cuda_langevin_piston_integrator
    @see cuda_integrators
    """

    _destroy_function_name = "apo_cuda_langevin_piston_integrator_destroy"

    def __init__(self, time_step: float) -> None:
        """
        @brief Constructs a Python-owned Langevin-piston integrator.

        @param[in] time_step Value converted to C `double`, interpreted as a
        finite positive time step in picoseconds.
        @throws TypeError If conversion fails.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR`.
        @throws RuntimeError If a successful native call produces a `NULL`
        concrete or base-view handle.
        @post The crystal type remains `CrystalType.NONE` until explicitly set.
        """
        _initialize_prototypes()
        super().__init__()

        handle: ctypes.c_void_p = ctypes.c_void_p()
        c_time_step: ctypes.c_double = ctypes.c_double(time_step)

        lib().apo_cuda_langevin_piston_integrator_create(
            ctypes.byref(handle), c_time_step
        )

        if handle.value is None:
            raise RuntimeError(
                "apo_cuda_langevin_piston_integrator_create returned success but produced a NULL handle"
            )

        self._handle = handle

        integrator_handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_cuda_langevin_piston_integrator_as_cuda_integrator(
            ctypes.byref(integrator_handle), self.handle
        )

        if integrator_handle.value is None:
            raise RuntimeError(
                "apo_cuda_langevin_piston_integrator_as_cuda_integrator returned success but produced a NULL handle"
            )

        self._integrator_handle = integrator_handle

        return

    def useNoseHooverThermostat(self, flag: bool = True) -> None:
        """
        @brief Enables or disables Nose-Hoover temperature coupling.

        @param[in] flag Value converted with `ctypes.c_bool`. A truthy value
        enables the coupling.
        @return `None`.
        @throws TypeError If C boolean conversion rejects the value.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If the native handle is rejected.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool(flag)

        lib().apo_cuda_langevin_piston_integrator_use_nose_hoover_thermostat(
            self.handle, c_flag
        )

        return

    def setReferenceTemperature(self, temperature: float) -> None:
        """
        @brief Sets the piston reference temperature.

        @param[in] temperature Value converted to C `double`, in kelvin. It must
        be finite and non-negative.
        @return `None`.
        @throws TypeError If conversion fails.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, factor recomputation, or
        device transfer fails.
        """
        _initialize_prototypes()

        c_temperature: ctypes.c_double = ctypes.c_double(temperature)

        lib().apo_cuda_langevin_piston_integrator_set_reference_temperature(
            self.handle, c_temperature
        )

        return

    def setNoseHooverPistonMass(self, mass: float) -> None:
        """
        @brief Sets the scalar Nose-Hoover coupling mass.

        @param[in] mass Value converted to C `double`. It must be finite and
        non-negative. Its exact dimensional convention is not established.
        @return `None`.
        @throws TypeError If conversion fails.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or device update fails.
        @warning Zero is accepted, but enabled Nose-Hoover propagation divides
        by this value.
        """
        _initialize_prototypes()

        c_mass: ctypes.c_double = ctypes.c_double(mass)

        lib().apo_cuda_langevin_piston_integrator_set_nose_hoover_piston_mass(
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
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If the native handle is rejected.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool(flag)

        lib().apo_cuda_langevin_piston_integrator_use_old_temperature(
            self.handle, c_flag
        )

        return

    def setReferencePressure(self, pressure_tensor: Sequence[Sequence[float]]) -> None:
        """
        @brief Sets the row-major reference pressure tensor.

        The wrapper iterates the outer object, iterates each inner object,
        converts every value with `float()`, and flattens all values into a
        temporary contiguous C buffer. Native validation requires exactly nine
        total values but the wrapper does not independently require a 3-by-3
        nested shape.

        @param[in] pressure_tensor Iterable of iterables whose flattened values
        are ordered `xx`, `xy`, `xz`, `yx`, `yy`, `yz`, `zx`, `zy`, `zz` and
        interpreted in atmospheres.
        @return `None`.
        @throws TypeError If an input is not iterable or a value cannot be
        converted to `float`.
        @throws ValueError If `float()` rejects a value.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` for a non-nine-element or non-finite
        tensor, `APO_STATUS_CUDA_ERROR` for transfer failure, or
        `APO_STATUS_RUNTIME_ERROR` for another native failure.
        """
        _initialize_prototypes()

        flattened_pressure_tensor: list[float] = []

        for pressure in pressure_tensor:
            flattened_pressure_tensor.extend(float(value) for value in pressure)

        c_buffer_type = ctypes.c_double * len(flattened_pressure_tensor)
        c_buffer = c_buffer_type(*flattened_pressure_tensor)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(flattened_pressure_tensor))

        lib().apo_cuda_langevin_piston_integrator_set_reference_pressure(
            self.handle, c_buffer, c_buffer_len
        )

        return

    def setConstantSurfaceTension(self, flag: bool = True) -> None:
        """
        @brief Enables or disables the constant-surface-tension branch.

        @param[in] flag Value converted with `ctypes.c_bool`.
        @return `None`.
        @throws TypeError If C boolean conversion rejects the value.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If the native handle is rejected.
        @warning The current API has no target-surface-tension setter and the
        native target value is not initialized after allocation. Do not enable
        this mode until that implementation defect is corrected.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool(flag)

        lib().apo_cuda_langevin_piston_integrator_set_constant_surface_tension(
            self.handle, c_flag
        )

        return

    def setCrystalType(self, crystal_type: CrystalType | int) -> None:
        """
        @brief Selects crystal symmetry and reallocates piston state.

        @param[in] crystal_type `CrystalType.CUBIC`,
        `CrystalType.TETRAGONAL`, `CrystalType.ORTHORHOMBIC`, or another object
        accepted by `int()` whose resulting value names one of those enums.
        @return `None`.
        @throws TypeError If `int()` or `ctypes.c_int` rejects the value.
        @throws ValueError If integer conversion fails or the native enum value
        is unsupported, including `CrystalType.NONE`.
        @throws OverflowError If the value cannot be represented by
        `ctypes.c_int`.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR`.
        @warning Reconfiguration resets crystal-sized state and can invalidate
        native aliases.
        """
        _initialize_prototypes()

        c_crystal_type: ctypes.c_int = ctypes.c_int(int(crystal_type))

        lib().apo_cuda_langevin_piston_integrator_set_crystal_type(
            self.handle, c_crystal_type
        )

        return

    def setLangevinPistonMass(self, masses: Sequence[float]) -> None:
        """
        @brief Sets crystal-sized Langevin-piston masses.

        The sequence is eagerly converted with `float()` and copied into a
        temporary contiguous C buffer.

        @param[in] masses One, two, or three finite non-negative values whose
        length exactly matches the selected crystal degree-of-freedom count.
        The exact dimensional convention is not established by the repository.
        @return `None`.
        @throws TypeError If `masses` is not iterable or a value cannot be
        converted to `float`.
        @throws ValueError If `float()` rejects a value.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_NOT_INITIALIZED` before crystal selection,
        `APO_STATUS_INVALID_ARGUMENT` for length or value validation,
        `APO_STATUS_CUDA_ERROR` for device update failure, or
        `APO_STATUS_RUNTIME_ERROR` for another native failure.
        """
        _initialize_prototypes()

        mass_values: list[float] = [float(value) for value in masses]

        c_buffer_type = ctypes.c_double * len(mass_values)
        c_buffer = c_buffer_type(*mass_values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(mass_values))

        lib().apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
            self.handle, c_buffer, c_buffer_len
        )

        return

    def setLangevinPistonFrictionSeed(self, seed: int) -> None:
        """
        @brief Sets the piston RNG seed.

        @param[in] seed Python integer in `[0, 2**64 - 1]`.
        @return `None`.
        @throws TypeError If comparison or uint64 conversion rejects the value.
        @throws ValueError If the value is outside the uint64 range.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If selected-crystal RNG allocation or
        initialization fails.
        @post The recorded native sequence position is preserved.
        """
        _initialize_prototypes()

        seed_value: int = require_c_uint64(seed, "seed", allow_bool=True)
        c_seed: ctypes.c_uint64 = ctypes.c_uint64(seed_value)

        lib().apo_cuda_langevin_piston_integrator_set_langevin_piston_friction_seed(
            self.handle, c_seed
        )

        return

    def setLangevinPistonFriction(self, friction: float) -> None:
        """
        @brief Sets the Langevin-piston friction coefficient.

        @param[in] friction Value converted to C `double`, in inverse
        picoseconds. It must be finite and non-negative.
        @return `None`.
        @throws TypeError If conversion fails.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_NOT_INITIALIZED` before crystal selection,
        `APO_STATUS_INVALID_ARGUMENT` for an invalid value,
        `APO_STATUS_CUDA_ERROR` for factor transfer failure, or
        `APO_STATUS_RUNTIME_ERROR` for another native failure.
        """
        _initialize_prototypes()

        c_friction: ctypes.c_double = ctypes.c_double(friction)

        lib().apo_cuda_langevin_piston_integrator_set_langevin_piston_friction(
            self.handle, c_friction
        )

        return

    def resetAverages(self) -> None:
        """
        @brief Resets temperature and pressure running averages.

        @return `None`.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or a device reset fails.
        """
        _initialize_prototypes()

        lib().apo_cuda_langevin_piston_integrator_reset_averages(self.handle)

        return

    def getReferenceTemperature(self) -> float:
        """
        @brief Returns the piston reference temperature.

        @return A new Python `float` in kelvin.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or output fails.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_langevin_piston_integrator_get_reference_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)

    def getNoseHooverPistonMass(self) -> float:
        """
        @brief Returns the scalar Nose-Hoover coupling mass.

        @return A new Python `float` in the native coupling-mass convention.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, shape checking, transfer,
        or output fails.
        """
        _initialize_prototypes()

        c_mass = ctypes.c_double()

        lib().apo_cuda_langevin_piston_integrator_get_nose_hoover_piston_mass(
            ctypes.byref(c_mass), self.handle
        )

        return float(c_mass.value)

    def getAverageTemperature(self) -> float:
        """
        @brief Returns one running-average temperature.

        @return A new Python `float` in kelvin.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, shape checking, or transfer
        fails.
        @warning Current C ABI behavior returns estimator element 0 when
        old-temperature mode is enabled and element 1 when it is disabled,
        opposite the native instantaneous-temperature selector.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_langevin_piston_integrator_get_average_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)

    def getReferencePressureTensor(self) -> list[list[float]]:
        """
        @brief Returns the reference pressure tensor.

        @return A newly allocated 3-by-3 nested list of Python `float` values in
        row-major order and atmospheres. The result does not alias native
        storage.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, shape checking, transfer,
        or output fails.
        """
        _initialize_prototypes()

        c_buffer_type = ctypes.c_double * 9
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(9)

        lib().apo_cuda_langevin_piston_integrator_get_reference_pressure_tensor(
            c_buffer, c_buffer_len, self.handle
        )

        pressure_tensor: list[list[float]] = []
        for i in range(3):
            pressure_tensor.append(
                [
                    float(c_buffer[i * 3 + 0]),
                    float(c_buffer[i * 3 + 1]),
                    float(c_buffer[i * 3 + 2]),
                ]
            )

        return pressure_tensor

    def getCrystalType(self) -> CrystalType:
        """
        @brief Returns the selected crystal type.

        @return A new `CrystalType`, including `CrystalType.NONE` before
        configuration.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or enum conversion fails.
        @throws ValueError If the returned C integer is not a Python
        `CrystalType` value.
        """
        _initialize_prototypes()

        c_crystal_type = ctypes.c_int()

        lib().apo_cuda_langevin_piston_integrator_get_crystal_type(
            ctypes.byref(c_crystal_type), self.handle
        )

        return CrystalType(c_crystal_type.value)

    def getLangevinPistonMass(self) -> list[float]:
        """
        @brief Returns the crystal-sized Langevin-piston masses.

        @return A newly allocated list containing zero, one, two, or three
        Python `float` values in active crystal-degree-of-freedom order.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, count checking, transfer,
        or output fails.
        """
        _initialize_prototypes()

        c_num_mass: ctypes.c_size_t = ctypes.c_size_t()

        c_buffer_type = ctypes.c_double * 3
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(3)

        lib().apo_cuda_langevin_piston_integrator_get_langevin_piston_mass(
            ctypes.byref(c_num_mass), c_buffer, c_buffer_len, self.handle
        )

        num_mass: int = int(c_num_mass.value)

        masses: list[float] = []
        for i in range(num_mass):
            masses.append(float(c_buffer[i]))

        return masses

    def getInstantaneousPressureTensor(self) -> list[list[float]]:
        """
        @brief Returns the instantaneous pressure tensor.

        @return A newly allocated 3-by-3 row-major nested list in atmospheres.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, shape checking, transfer,
        or output fails.
        """
        _initialize_prototypes()

        c_buffer_type = ctypes.c_double * 9
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(9)

        lib().apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_tensor(
            c_buffer, c_buffer_len, self.handle
        )

        pressure_tensor: list[list[float]] = []
        for i in range(3):
            pressure_tensor.append(
                [
                    float(c_buffer[i * 3 + 0]),
                    float(c_buffer[i * 3 + 1]),
                    float(c_buffer[i * 3 + 2]),
                ]
            )

        return pressure_tensor

    def getInstantaneousPressureScalar(self) -> float:
        """
        @brief Returns the instantaneous scalar pressure.

        @return A new Python `float` in atmospheres.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, shape checking, transfer,
        or output fails.
        """
        _initialize_prototypes()

        c_pressure_scalar = ctypes.c_double()

        lib().apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_scalar(
            ctypes.byref(c_pressure_scalar), self.handle
        )

        return float(c_pressure_scalar.value)

    def getAveragePressureTensor(self) -> list[list[float]]:
        """
        @brief Returns the running-average pressure tensor.

        @return A newly allocated 3-by-3 row-major nested list in atmospheres.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, shape checking, transfer,
        or output fails.
        """
        _initialize_prototypes()

        c_buffer_type = ctypes.c_double * 9
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(9)

        lib().apo_cuda_langevin_piston_integrator_get_average_pressure_tensor(
            c_buffer, c_buffer_len, self.handle
        )

        pressure_tensor: list[list[float]] = []
        for i in range(3):
            pressure_tensor.append(
                [
                    float(c_buffer[i * 3 + 0]),
                    float(c_buffer[i * 3 + 1]),
                    float(c_buffer[i * 3 + 2]),
                ]
            )

        return pressure_tensor

    def getAveragePressureScalar(self) -> float:
        """
        @brief Returns the running-average scalar pressure.

        @return A new Python `float` in atmospheres.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation, shape checking, transfer,
        or output fails.
        """
        _initialize_prototypes()

        c_pressure_scalar = ctypes.c_double()

        lib().apo_cuda_langevin_piston_integrator_get_average_pressure_scalar(
            ctypes.byref(c_pressure_scalar), self.handle
        )

        return float(c_pressure_scalar.value)

    def getLangevinPistonFrictionSeed(self) -> int:
        """
        @brief Returns the Langevin-piston RNG seed.

        @return A new non-negative Python `int` in the uint64 range.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError If native validation or output fails.
        """
        _initialize_prototypes()

        c_seed = ctypes.c_uint64()

        lib().apo_cuda_langevin_piston_integrator_get_langevin_piston_friction_seed(
            ctypes.byref(c_seed), self.handle
        )

        return int(c_seed.value)

    def getInstantaneousTemperature(self) -> float:
        """
        @brief Returns the native selected instantaneous temperature.

        @return A new Python `float` in kelvin. Native selection uses
        kinetic-energy element 1 when old-temperature mode is enabled and
        element 0 otherwise.
        @throws RuntimeError If the wrapper is closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_NOT_INITIALIZED` before context attachment,
        `APO_STATUS_CUDA_ERROR` for transfer failure, or another documented
        native status.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_cuda_langevin_piston_integrator_get_instantaneous_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)
