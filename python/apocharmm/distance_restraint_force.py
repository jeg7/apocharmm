# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Exposes the owning Python wrapper for distance restraints.

`DistanceRestraintForce` copies Python restraint definitions through the C ABI
and can be subscribed to `ForceManager`. Force calculation and native output
access are not exposed directly by this module.

@anchor python_distance_restraint_force_module
@see distance_restraint_force
"""

from collections.abc import Sequence
import ctypes

from ._base import _ApoObject
from ._lib import lib
from ._validation import require_c_int
from .atom_reference import AtomReference
from .enums import DistanceRestraintCondition
from .error import configure_status_function
from .force_manager import ForceManager

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_distance_restraint_force_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        "DistanceRestraintForce construction",
    )

    lib().apo_distance_restraint_force_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_distance_restraint_force_destroy.restype = None

    configure_status_function(
        lib().apo_distance_restraint_force_set_scale,
        [ctypes.c_void_p, ctypes.c_double],
        "DistanceRestraintForce.setScale(scale)",
    )

    configure_status_function(
        lib().apo_distance_restraint_force_add_restraint,
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ],
        "DistanceRestraintForce.addRestraint(atom_pairs, coefficients, force_constant, reference_value, distance_exponent, energy_exponent, condition)",
    )

    configure_status_function(
        lib().apo_distance_restraint_force_reset,
        [ctypes.c_void_p],
        "DistanceRestraintForce.reset()",
    )

    configure_status_function(
        lib().apo_force_manager_subscribe_distance_restraint_force,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p],
        "ForceManager.subscribe(DistanceRestraintForce)",
    )

    configure_status_function(
        lib().apo_force_manager_unsubscribe_distance_restraint_force,
        [ctypes.c_void_p, ctypes.c_void_p],
        "ForceManager.unsubscribe(DistanceRestraintForce)",
    )

    _prototypes_initialized = True

    return


class DistanceRestraintForce(_ApoObject):
    """
    @brief Configures an owned native absolute distance restraint.

    One object may own multiple restraint terms. For each term, native code
    forms the coefficient-weighted sum of powered pair distances, subtracts the
    reference value, applis the selected one-sided condition, and accumulates
    the normalized energy, gradient, and internal virial.

    This wrapper owns one C handle for the native
    `DistanceRestraintForce<long long int, float>` specialization. Python pair
    and coefficient sequences are materialized into temporary contiguous
    `ctypes` arrays and copied again into native storage. No caller sequence or
    buffer is retained.

    `ForceManager.subscribe()` retains this wrapper after native subscription
    succeeds. The native manager independently retains the restraint and its
    CUDA resources. Unsubscribe before calling `close()`: closing a subscribed
    wrapper destroys the C handle but does not remove the native subscription,
    and the closed wrapper can no longer identify that subscription for normal
    unsubscription.

    `close()`, `destroy()`, context-manager exit, and finalization release the
    owned C handle. Explicit closure is idempotent. Methods that access
    `self.handle` after closure raise `RuntimeError`.

    The wrapper and native object provide no internal synchronization. Do not
    overlap configuration, manager subscription changes, calculation, or
    closure across threads.

    @anchor python_distance_restraint_force
    @see distance_restraint_force
    """

    _destroy_function_name = "apo_distance_restraint_force_destroy"

    def __init__(self, num_atoms: int) -> None:
        """
        @brief Constructs a distance restraint for a fixed atom count.

        `num_atoms` is checked as a Python `int`, converted to signed C `int`,
        and passed to the native constructor. The wrapper owns the newly
        returned handle and sets `default_force_tag` to `"resd"`.

        @param[in] num_atoms Dimensionless positive atom count representable by
        signed 32-bit C `int`. The current `isinstance(value, int)` check also
        accepts Python `bool`; `True` reaches native construction as one and
        `False` is rejected natively as zero.
        @throws TypeError If `num_atoms` is not a Python `int`.
        @throws ValueError If `num_atoms` is outside the signed 32-bit C `int`
        range.
        @throws ApoCharmmError If native validation rejects a non-positive
        value, native CUDA setup fails, or native allocation fails.
        @throws RuntimeError If native construction reports success but
        produces a NULL handle.
        @throws MemoryError If Python or `ctypes` bookkeeping cannot allocate
        storage.

        @post On success, this wrapper owns a live native handle and
        `default_force_tag == "resd"`.
        """
        super().__init__()

        if not isinstance(num_atoms, int):
            raise TypeError("num_atoms must be an int")

        if num_atoms < -(2**31) or num_atoms > 2**31 - 1:
            raise ValueError("num_atoms must fit in int")

        _initialize_prototypes()

        handle: ctypes.c_void_p = ctypes.c_void_p()
        c_num_atoms: ctypes.c_int = ctypes.c_int(num_atoms)

        lib().apo_distance_restraint_force_create(ctypes.byref(handle), c_num_atoms)

        if handle.value is None:
            raise RuntimeError(
                "apo_distance_restraint_force_create returned success but produced a NULL handle"
            )

        self._handle = handle
        self._default_force_tag: str = "resd"

        return

    @property
    def default_force_tag(self) -> str:
        """
        @brief Returns the default manager energy tag.

        This property reads Python-owned text and does not access the native
        handle, so it remains available after `close()`.

        @return The Python string `"resd"`.
        """
        return self._default_force_tag

    def setScale(self, scale: float) -> None:
        """
        @brief Sets the global scale applied to every stored restraint term.

        `scale` is converted with `float()` and then to C `double`. Native
        validation defines the accepted finite values and preserves the stored
        restraint definitions.

        @param[in] scale Value accepted by `float()` and native scale
        validation.
        @throws TypeError If `float(scale)` rejects the object.
        @throws ValueError If Python conversion rejects the value.
        @throws OverflowError If Python conversion overflows.
        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError If the C ABI reports a failure.
        """
        _initialize_prototypes()

        c_scale: ctypes.c_double = ctypes.c_double(float(scale))
        lib().apo_distance_restraint_force_set_scale(self.handle, c_scale)

        return

    def addRestraint(
        self,
        atom_pairs: Sequence[Sequence[AtomReference]],
        coefficients: Sequence[float],
        force_constant: float,
        reference_value: float,
        distance_exponent: int,
        energy_exponent: int,
        condition: DistanceRestraintCondition,
    ) -> None:
        """
        @brief Appends one distance-restraint term.

        Every element of `atom_pairs` must contain exactly two live
        `AtomReference` objects. Raw integer endpoint pairs are not accepted.

        Pair topology provenance is deliberately ignored by the native force.
        Endpoints may originate from different native PSF objects. Native code
        extracts and stores only their zero-based atom indices, validates those
        indices against the force-local atom count, and retains neither the
        Python wrappers, C handles, native AtomReference values, nor source
        PSFs.

        The outer pair count must equal the coefficient count. Endpoint handles
        and coefficients are placed in temporary contiguous arrays and borrowed
        only for the C call. Pair ordering and coefficient correspondence are
        preserved.

        @param[in] atom_pairs Sequence of two-AtomReference sequences.
        @param[in] coefficients Sequence containing one coefficient per pair.
        @param[in] force_constant Value accepted by `float()`.
        @param[in] reference_value Value accepted by `float()`.
        @param[in] distance_exponent Required signed C `int` exponent.
        @param[in] energy_exponent Required signed C `int` exponent.
        @param[in] condition Required one-sided activation condition.
        @throws TypeError If an endpoint is not an `AtomReference`, an exponent
        is not an `int`, an exponent is `bool`, `condition` has the wrong type,
        a supplied sequence is not iterable, or a scalar cannot be converted by
        `float()`.
        @throws ValueError If a pair does not contain exactly two values, an
        exponent does not fit signed C `int`, or Python scalar conversion rejects
        a value.
        @throws OverflowError If Python floating-point conversion overflows.
        @throws RuntimeError If this wrapper or an endpoint AtomReference has
        been closed.
        @throws ApoCharmmError If native validation or allocation fails.
        """
        _initialize_prototypes()

        if not isinstance(condition, DistanceRestraintCondition):
            raise TypeError("condition must be a DistanceRestraintCondition")

        first_atom_references: list[AtomReference] = []
        second_atom_references: list[AtomReference] = []
        for pair_index, atom_pair in enumerate(atom_pairs):
            atom_pair_values: list[AtomReference] = list(atom_pair)

            if len(atom_pair_values) != 2:
                raise ValueError(
                    f"atom_pairs[{pair_index}] must contain exactly 2 values"
                )

            for endpoint_index, atom_reference in enumerate(atom_pair_values):
                if not isinstance(atom_reference, AtomReference):
                    raise TypeError(
                        f"atom_pairs[{pair_index}][{endpoint_index}] must be an AtomReference"
                    )

            first_atom_references.append(atom_pair_values[0])
            second_atom_references.append(atom_pair_values[1])

        coefficient_values: list[float] = [float(value) for value in coefficients]

        distance_exponent_value: int = require_c_int(
            distance_exponent, "distance_exponent"
        )
        energy_exponent_value: int = require_c_int(energy_exponent, "energy_exponent")

        c_first_atom_references_type: type[ctypes.Array[ctypes.c_void_p]] = (
            ctypes.c_void_p * len(first_atom_references)
        )

        c_first_atom_references: ctypes.Array[ctypes.c_void_p] = (
            c_first_atom_references_type(
                *(reference.handle for reference in first_atom_references)
            )
        )

        c_first_atom_references_len: ctypes.c_size_t = ctypes.c_size_t(
            len(first_atom_references)
        )

        c_second_atom_references_type: type[ctypes.Array[ctypes.c_void_p]] = (
            ctypes.c_void_p * len(second_atom_references)
        )

        c_second_atom_references: ctypes.Array[ctypes.c_void_p] = (
            c_second_atom_references_type(
                *(reference.handle for reference in second_atom_references)
            )
        )

        c_second_atom_references_len: ctypes.c_size_t = ctypes.c_size_t(
            len(second_atom_references)
        )

        c_coefficients_type: type[ctypes.Array[ctypes.c_double]] = (
            ctypes.c_double * len(coefficient_values)
        )

        c_coefficients: ctypes.Array[ctypes.c_double] = c_coefficients_type(
            *coefficient_values
        )

        c_coefficients_len: ctypes.c_size_t = ctypes.c_size_t(len(coefficient_values))

        c_force_constant: ctypes.c_double = ctypes.c_double(float(force_constant))
        c_reference_value: ctypes.c_double = ctypes.c_double(float(reference_value))
        c_distance_exponent: ctypes.c_int = ctypes.c_int(distance_exponent_value)
        c_energy_exponent: ctypes.c_int = ctypes.c_int(energy_exponent_value)
        c_condition: ctypes.c_int = ctypes.c_int(condition.value)

        lib().apo_distance_restraint_force_add_restraint(
            self.handle,
            c_first_atom_references,
            c_first_atom_references_len,
            c_second_atom_references,
            c_second_atom_references_len,
            c_coefficients,
            c_coefficients_len,
            c_force_constant,
            c_reference_value,
            c_distance_exponent,
            c_energy_exponent,
            c_condition,
        )

        return

    def reset(self) -> None:
        """
        @brief Removes all stored restraint terms and restores native scale one.

        The operation changes restraint configuration only. It does not replace
        force-manager output clearing.

        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError If the C ABI reports a failure.
        """

        _initialize_prototypes()

        lib().apo_distance_restraint_force_reset(self.handle)

        return

    def _subscribe_to_force_manager(
        self, force_manager: ForceManager, force_tag: str | None = None
    ) -> None:
        """
        @brief Implements the `ForceManager.subscribe()` callback.

        `None` selects `default_force_tag`; otherwise `force_tag` must be a
        Python `str`. The selected text is encoded as UTF-8 and passed as a
        borrowed null-terminated C string. Native code copies the tag and
        retains the restraint and calculation resources on success.

        This callback performs the native subscription. The calling
        `ForceManager.subscribe()` method retains this Python wrapper only after
        the callback succeeds.

        @param[in] force_manager Live `ForceManager` receiving the restraint.
        @param[in] force_tag Python `str` tag, or `None` to use `"resd"`.
        @throws TypeError If `force_manager` is not a `ForceManager` or a
        non-`None` tag is not a `str`.
        @throws UnicodeEncodeError If `force_tag` cannot be encoded as UTF-8.
        @throws RuntimeError If this wrapper or `force_manager` has been closed.
        @throws ApoCharmmError If the tag is empty, either native handle is
        invalid, the restraint is already subscribed, initialization detects a
        mismatch, or native allocation or CUDA setup fails.

        @warning An embedded NUL character terminates the C string and causes
        native code to observe only the preceding tag prefix.
        """
        _initialize_prototypes()

        if not isinstance(force_manager, ForceManager):
            raise TypeError("force_manager must be a ForceManager")

        if force_tag is None:
            force_tag_value: str = self._default_force_tag
        else:
            if not isinstance(force_tag, str):
                raise TypeError("force_tag must be a str")

            force_tag_value = force_tag

        encoded_force_tag: bytes = force_tag_value.encode("utf-8")
        c_force_tag: ctypes.c_char_p = ctypes.c_char_p(encoded_force_tag)

        lib().apo_force_manager_subscribe_distance_restraint_force(
            force_manager.handle, self.handle, c_force_tag
        )

        return

    def _unsubscribe_from_force_manager(self, force_manager: ForceManager) -> None:
        """
        @brief Implements the `ForceManager.unsubscribe()` callback.

        Native unsubscription releases the manager's retained restraint,
        stream, force-array, and energy owners. After this callback succeeds,
        the calling `ForceManager.unsubscribe()` method releases its retained
        Python reference.

        @param[in] force_manager Live `ForceManager` currently containing this
        restraint.
        @throws RuntimeError If this wrapper or `force_manager` has been closed.
        @throws ApoCharmmError If either native handle is invalid, this
        restraint is not subscribed to the manager, or an unexpected native
        runtime failure occurs.
        """
        _initialize_prototypes()

        lib().apo_force_manager_unsubscribe_distance_restraint_force(
            force_manager.handle, self.handle
        )

        return
