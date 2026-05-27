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
from .error import check_status

from .charmm_crd import CharmmCrd
from .force_manager import ForceManager

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_charmm_context_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_context_create.restype = ctypes.c_int

    lib().apo_charmm_context_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_charmm_context_destroy.restype = None

    lib().apo_charmm_context_set_coordinates.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib().apo_charmm_context_set_coordinates.restype = ctypes.c_int

    lib().apo_charmm_context_set_random_seed_for_velocities.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    lib().apo_charmm_context_set_random_seed_for_velocities.restype = ctypes.c_int

    lib().apo_charmm_context_use_holonomic_constraints.argtypes = [
        ctypes.c_void_p,
        ctypes.c_bool,
    ]
    lib().apo_charmm_context_use_holonomic_constraints.restype = ctypes.c_int

    lib().apo_charmm_context_assign_velocities_at_temperature.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    lib().apo_charmm_context_assign_velocities_at_temperature.restype = ctypes.c_int

    lib().apo_charmm_context_compute_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    ]
    lib().apo_charmm_context_compute_temperature.restype = ctypes.c_int

    _prototypes_initialized = True

    return


class CharmmContext(_ApoObject):
    _destroy_function_name = "apo_charmm_context_destroy"

    def __init__(self, force_manager: ForceManager) -> None:
        _initialize_prototypes()
        super().__init__()

        handle = ctypes.c_void_p()

        status = lib().apo_charmm_context_create(
            ctypes.byref(handle), force_manager.handle
        )

        check_status(status, "CharmmContext construction failed")

        if handle.value is None:
            raise RuntimeError(
                "apo_charmm_context_create returned success but produced a NULL handle"
            )

        self._handle = handle
        self._force_manager = force_manager

        return

    def setCoordinates(self, crd: CharmmCrd) -> None:
        _initialize_prototypes()

        status = lib().apo_charmm_context_set_coordinates(self.handle, crd.handle)

        check_status(status, "CharmmContext.setCoordinates(crd) failed")

        return

    def setRandomSeedForVelocities(self, seed: int) -> None:
        _initialize_prototypes()

        status = lib().apo_charmm_context_set_random_seed_for_velocities(
            self.handle, seed
        )

        check_status(status, "CharmmContext.setRandomSeedForVelocities(seed) failed")

        return

    def useHolonomicConstraints(self, use_holonomic_constraints: bool) -> None:
        _initialize_prototypes()

        status = lib().apo_charmm_context_use_holonomic_constraints(
            self.handle, use_holonomic_constraints
        )

        check_status(
            status,
            "CharmmContext.useHolonomicConstraints(useHolonomicConstraints) failed",
        )

        return

    def assignVelocitiesAtTemperature(self, temperature: float) -> None:
        _initialize_prototypes()

        status = lib().apo_charmm_context_assign_velocities_at_temperature(
            self.handle, temperature
        )

        check_status(
            status, "CharmmContext.assignVelocitiesAtTemperature(temperature) failed"
        )

        return

    def computeTemperature(self) -> float:
        _initialize_prototypes()

        temperature = ctypes.c_double()

        status = lib().apo_charmm_context_compute_temperature(
            ctypes.byref(temperature), self.handle
        )

        check_status(status, "CharmmContext.computeTemperature() failed")

        return float(temperature.value)
