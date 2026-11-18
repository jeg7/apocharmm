# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Provides the owning Python wrapper for atom-selection results.

`AtomSelection` exposes immutable count, index-list, and membership queries for
one native selection. User code normally obtains an instance from
`AtomSelector.select()` rather than constructing it from a raw handle.

@anchor python_atom_selection_module
@see atom_selection
"""

import ctypes

from ._base import _ApoObject
from ._lib import lib
from ._validation import require_c_int
from .error import configure_status_function

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    lib().apo_atom_selection_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_atom_selection_destroy.restype = None

    configure_status_function(
        lib().apo_atom_selection_get_num_atoms,
        [ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p],
        "AtomSelection.getNumAtoms()",
    )

    configure_status_function(
        lib().apo_atom_selection_get_num_selected,
        [ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p],
        "AtomSelection.getNumSelected()",
    )

    configure_status_function(
        lib().apo_atom_selection_get_atom_indices,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_void_p],
        "AtomSelection.getAtomIndices()",
    )

    configure_status_function(
        lib().apo_atom_selection_contains,
        [ctypes.POINTER(ctypes.c_bool), ctypes.c_void_p, ctypes.c_int],
        "AtomSelection.contains(atom_index)",
    )

    _prototypes_initialized = True

    return


class AtomSelection(_ApoObject):
    """
    @brief Owns one immutable native atom-selection handle.

    Atom indices are zero-based, sorted in ascending order when returned as a
    list, and dimensionless. The wrapper stores no coordinates and performs no
    CUDA transfer or synchronization.

    The wrapper owns its C handle. `close()`, `destroy()`, context-manager exit,
    or finalization releases that handle; explicit closure is idempotent.
    Accessing any query after closure raises `RuntimeError`. Closing the selector
    or PSF that produced this object does not invalidate the selection.

    The class provides no internal synchronization. Do not overlap `close()`
    with a query from another thread.

    @anchor python_atom_selection
    @see atom_selection
    """

    _destroy_function_name = "apo_atom_selection_destroy"

    def __init__(self, handle: ctypes.c_void_p) -> None:
        """
        @brief Constructs an owning wrapper around a native selection handle.

        This low-level constructor takes ownership of the pointed-to C handle
        without copying the native selection. User code should normally call
        `AtomSelector.select()` instead. After successful construction, the
        caller must not destroy or transfer the same native handle separately.

        @param[in] handle Exact `ctypes.c_void_p` containing a non-NULL,
        newly owned `apo_atom_selection` pointer.
        @throws TypeError If `handle` is not a `ctypes.c_void_p` instance.
        @throws RuntimeError If `handle.value` is `None`.

        @post On success, this wrapper owns the native handle until closure.
        """
        _initialize_prototypes()
        super().__init__()

        if not isinstance(handle, ctypes.c_void_p):
            raise TypeError("AtomSelection expects a ctypes.c_void_p handle")

        if handle.value is None:
            raise RuntimeError("AtomSelection construction failed: NULL handle")

        self._handle = handle

        return

    def getNumAtoms(self) -> int:
        """
        @brief Returns the number of atoms represented by the selection.

        @return A non-negative, dimensionless Python `int`.
        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError If the native C ABI rejects the live handle or
        reports an unexpected runtime failure.
        """
        _initialize_prototypes()

        c_num_atoms: ctypes.c_size_t = ctypes.c_size_t()

        lib().apo_atom_selection_get_num_atoms(ctypes.byref(c_num_atoms), self.handle)

        return int(c_num_atoms.value)

    def getNumSelected(self) -> int:
        """
        @brief Returns the number of selected atoms.

        @return A dimensionless Python `int` in `[0, getNumAtoms()]`.
        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError If the native C ABI rejects the live handle or
        reports an unexpected runtime failure.
        """
        _initialize_prototypes()

        c_num_selected: ctypes.c_size_t = ctypes.c_size_t()

        lib().apo_atom_selection_get_num_selected(
            ctypes.byref(c_num_selected), self.handle
        )

        return int(c_num_selected.value)

    def getAtomIndices(self) -> list[int]:
        """
        @brief Returns all selected atom indices.

        The method allocates a new `ctypes` buffer and a new Python list. The
        returned list never aliases native storage and remains valid after this
        wrapper is closed.

        @return A newly owned `list[int]` of zero-based, dimensionless indices in
        strictly ascending order. Its length equals `getNumSelected()`.
        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError If either native count or buffer-copy operation
        fails.
        @throws MemoryError If the temporary buffer or Python list cannot be
        allocated.
        """
        _initialize_prototypes()

        num_selected: int = self.getNumSelected()

        c_buffer_type = ctypes.c_int * num_selected
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(num_selected)

        lib().apo_atom_selection_get_atom_indices(c_buffer, c_buffer_len, self.handle)

        atom_indices: list[int] = []
        for i in range(num_selected):
            atom_indices.append(int(c_buffer[i]))

        return atom_indices

    def contains(self, atom_index: int) -> bool:
        """
        @brief Tests whether one atom index is selected.

        @param[in] atom_index Zero-based, dimensionless Python `int`. Boolean
        values are rejected even though `bool` subclasses `int`. The value must
        first fit a signed C `int` and then lie in `[0, getNumAtoms())`.
        @return `True` when the atom is selected; otherwise `False`.
        @throws TypeError If `atom_index` is not an `int` or is a `bool`.
        @throws ValueError If `atom_index` is outside the signed 32-bit C `int`
        range.
        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the converted index is outside the
        selection's atom range, or with another status if the native call fails.
        """
        _initialize_prototypes()

        c_is_selected: ctypes.c_bool = ctypes.c_bool()
        c_atom_index: ctypes.c_int = ctypes.c_int(
            require_c_int(atom_index, "atom_index")
        )

        lib().apo_atom_selection_contains(
            ctypes.byref(c_is_selected), self.handle, c_atom_index
        )

        return bool(c_is_selected.value)
