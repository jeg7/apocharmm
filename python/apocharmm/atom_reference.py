# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Provides the Python topology-aware atom-reference value type.

`AtomReference` identifies one zero-based atom index in one native `CharmmPsf`
object. Topology identity is native PSF object identity rather than file-name or
file-content equality.

@anchor python_atom_reference_module
@see atom_selection
"""

import ctypes

from ._base import _ApoObject
from ._lib import lib
from ._validation import require_c_int
from .charmm_psf import CharmmPsf
from .error import configure_status_function

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_atom_reference_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int],
        "AtomReference construction",
    )

    lib().apo_atom_reference_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_atom_reference_destroy.restype = None

    configure_status_function(
        lib().apo_atom_reference_get_atom_index,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "AtomReference.getAtomIndex()",
    )

    configure_status_function(
        lib().apo_atom_reference_has_same_topology,
        [ctypes.POINTER(ctypes.c_bool), ctypes.c_void_p, ctypes.c_void_p],
        "AtomReference.hasSameTopology(other)",
    )

    configure_status_function(
        lib().apo_atom_reference_equals,
        [ctypes.POINTER(ctypes.c_bool), ctypes.c_void_p, ctypes.c_void_p],
        "AtomReference.__eq__(other)",
    )

    _prototypes_initialized = True

    return


class AtomReference(_ApoObject):
    """
    @brief Owns one topology-aware zero-based atom identity.

    The native object retains shared const ownership of its source PSF and one
    validated atom index. This Python wrapper owns only the returned C handle;
    it deliberately does not retain the source `CharmmPsf` Python wrapper.

    Two references have the same topology only when their native PSF shared
    owners point to the same native `CharmmPSF` object. Equality additionally
    requires equal atom indices. Parsing the same PSF file into two independent
    `CharmmPsf` objects therefore produces different topologies.

    `close()`, `destroy()`, context-manager exit, or finalization releases the
    handle and is idempotent. Methods on a closed reference raise
    `RuntimeError`. References are equality-comparable but unordered and
    unhashable. Construction and queries are host-only and have no coordinate,
    CUDA, stream, or synchronization state.

    @anchor python_atom_reference
    @see atom_selection
    """

    _destroy_function_name = "apo_atom_reference_destroy"

    def __init__(self, psf: CharmmPsf, atom_index: int) -> None:
        """
        @brief Constructs a reference to one atom in one PSF.

        @param[in] psf Live `CharmmPsf` instance. The Python object is borrowed
        only during construction; native shared PSF ownership is copied into the
        new reference.
        @param[in] atom_index Zero-based atom index representable by signed C
        `int`.
        @throws TypeError If `psf` is not a `CharmmPsf`, if `atom_index` is not
        an `int`, or if `atom_index` is `bool`.
        @throws ValueError If `atom_index` does not fit signed C `int`.
        @throws RuntimeError If `psf` has been closed or if the C ABI reports
        success but returns a NULL handle.
        @throws ApoCharmmError With native status
        `APO_STATUS_NOT_INITIALIZED` when the PSF atom count is negative,
        `APO_STATUS_INVALID_ARGUMENT` when the native PSF handle or atom index is
        invalid, or `APO_STATUS_RUNTIME_ERROR` for allocation or another
        translated native failure.

        @post On success, this wrapper owns one atom-reference handle and does
        not retain `psf` as a Python object.
        """
        _initialize_prototypes()
        super().__init__()

        if not isinstance(psf, CharmmPsf):
            raise TypeError("AtomReference expects a CharmmPsf")

        atom_index_value = require_c_int(atom_index, "atom_index")

        handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_atom_reference_create(
            ctypes.byref(handle), psf.handle, ctypes.c_int(atom_index_value)
        )

        if handle.value is None:
            raise RuntimeError(
                "apo_atom_reference_create returned success but produced a NULL handle"
            )

        self._handle = handle

        return

    @classmethod
    def _from_handle(cls, handle: ctypes.c_void_p) -> "AtomReference":
        """
        @brief Adopts one newly owned non-NULL C atom-reference handle.

        This factory is for wrappers such as `AtomSelector.selectAtom()` that
        already own a newly created C handle. It does not repeat PSF or index
        construction.

        @param[in] handle Newly owned `ctypes.c_void_p` handle.
        @return A new wrapper that adopts `handle` without retaining another
        Python object.
        @throws TypeError If `handle` is not a `ctypes.c_void_p`.
        @throws RuntimeError If `handle` is NULL.
        """
        _initialize_prototypes()

        if not isinstance(handle, ctypes.c_void_p):
            raise TypeError("AtomReference expects a ctypes.c_void_p handle")

        if handle.value is None:
            raise RuntimeError("AtomReference construction failed: NULL handle")

        instance = cls.__new__(cls)
        _ApoObject.__init__(instance)
        instance._handle = handle

        return instance

    def getAtomIndex(self) -> int:
        """
        @brief Returns the stored zero-based atom index.

        @return The dimensionless zero-based index supplied at construction or
        selected by `AtomSelector.selectAtom()`.
        @throws RuntimeError If this reference has been closed.
        @throws ApoCharmmError If the native handle is invalid or another native
        failure is translated.
        """
        _initialize_prototypes()

        atom_index: ctypes.c_int = ctypes.c_int()
        lib().apo_atom_reference_get_atom_index(ctypes.byref(atom_index), self.handle)

        return int(atom_index.value)

    def hasSameTopology(self, other: "AtomReference") -> bool:
        """
        @brief Tests native PSF object identity.

        @param[in] other Live `AtomReference` to compare.
        @return `True` only when both references retain the same native
        `CharmmPSF` object.
        @throws TypeError If `other` is not an `AtomReference`.
        @throws RuntimeError If either reference has been closed.
        @throws ApoCharmmError If either native handle is invalid or another
        native failure is translated.
        """
        _initialize_prototypes()

        if not isinstance(other, AtomReference):
            raise TypeError("other must be an AtomReference")

        has_same_topology: ctypes.c_bool = ctypes.c_bool()
        lib().apo_atom_reference_has_same_topology(
            ctypes.byref(has_same_topology), self.handle, other.handle
        )

        return bool(has_same_topology.value)

    def __eq__(self, other: object) -> bool:
        """
        @brief Tests topology identity and atom-index equality.

        @param[in] other Object to compare.
        @return `True` only for another live `AtomReference` with the same native
        PSF object and index. Returns `NotImplemented` for unrelated Python
        types so Python can apply normal reflected-comparison behavior.
        @throws RuntimeError If either compared `AtomReference` has been closed.
        @throws ApoCharmmError If either native handle is invalid or another
        native failure is translated.
        """
        if not isinstance(other, AtomReference):
            return NotImplemented

        _initialize_prototypes()

        equals: ctypes.c_bool = ctypes.c_bool()
        lib().apo_atom_reference_equals(ctypes.byref(equals), self.handle, other.handle)

        return bool(equals.value)
