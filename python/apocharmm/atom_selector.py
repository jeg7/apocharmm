# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Provides the Python atom-selection evaluator.

`AtomSelector` evaluates CHARMM-style selection strings against one
`CharmmPsf`. `select()` returns independently owned set-valued `AtomSelection`
objects, while `selectAtom()` requires exactly one match and returns an
independently owned topology-aware `AtomReference`.

@anchor python_atom_selector_module
@see atom_selection
"""

import ctypes

from ._base import _ApoObject
from ._lib import lib
from ._validation import require_c_int
from .atom_reference import AtomReference
from .atom_selection import AtomSelection
from .charmm_psf import CharmmPsf
from .error import configure_status_function

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_atom_selector_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "AtomSelector construction",
    )

    lib().apo_atom_selector_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_atom_selector_destroy.restype = None

    configure_status_function(
        lib().apo_atom_selector_select,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
        "AtomSelector.select(selection_string)",
    )

    configure_status_function(
        lib().apo_atom_selector_select_atom,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
        "AtomSelector.selectAtom(selection_string)",
    )

    _prototypes_initialized = True

    return


class AtomSelector(_ApoObject):
    """
    @brief Evaluates atom-selection expressions against one native PSF.

    The wrapper retains the supplied `CharmmPsf` Python object in `_psf`. The
    native selector also shares ownership of the underlying native PSF, so
    closing the original PSF wrapper after construction does not invalidate the
    selector. The topology is shared rather than copied.

    Each successful `select()` call returns a new owning set-valued
    `AtomSelection` that remains valid after this selector is closed. Each
    successful `selectAtom()` call requires exactly one selected atom and
    returns a new owning topology-aware `AtomReference` that retains shared
    native PSF ownership. Selection evaluation is host-only and performs no CUDA
    transfer or synchronization.

    `close()`, `destroy()`, context-manager exit, or finalization releases the
    selector handle and is idempotent. Calling either selection method after
    closure raises `RuntimeError`. The wrapper provides no internal
    synchronization; do not overlap closure with selection from another thread.

    @anchor python_atom_selector
    @see atom_selection
    """

    _destroy_function_name = "apo_atom_selector_destroy"

    def __init__(self, psf: CharmmPsf) -> None:
        """
        @brief Constructs a selector for one parsed PSF.

        The wrapper retains `psf` as a Python reference and the C ABI copies
        shared native ownership. No topology data is copied or moved.

        @param[in] psf Live `CharmmPsf` instance whose atom metadata and derived
        topology tables define every future selection.
        @throws TypeError If `psf` is not a `CharmmPsf` instance.
        @throws RuntimeError If `psf` has been closed or if the C ABI reports
        success but returns a NULL selector handle.
        @throws ApoCharmmError With native status
        `APO_STATUS_NOT_INITIALIZED` if the PSF atom count is not initialized,
        with `APO_STATUS_INVALID_ARGUMENT` if the native handle is invalid, or
        with `APO_STATUS_RUNTIME_ERROR` if native allocation fails.

        @post On success, this wrapper owns one selector handle and retains
        `psf`.
        """
        _initialize_prototypes()
        super().__init__()

        if not isinstance(psf, CharmmPsf):
            raise TypeError("AtomSelector expects a CharmmPsf")

        handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_atom_selector_create(ctypes.byref(handle), psf.handle)

        if handle.value is None:
            raise RuntimeError(
                "apo_atom_selector_create returned success but produced a NULL handle"
            )

        self._handle = handle
        self._psf: CharmmPsf = psf

        return

    def select(self, selection_string: str) -> AtomSelection:
        """
        @brief Evaluates an atom-selection string as a set-valued result.

        The string is encoded as UTF-8 and passed to the null-terminated C ABI.
        Recognized keywords and dotted operators use ASCII spellings and
        case-insensitive matching. Native diagnostic positions are UTF-8 byte
        offsets, not Python character indices. See @ref atom_selection for the
        complete language.

        Example:

        @code{.py}
        selector = apo.AtomSelector(psf)
        alpha_carbons = selector.select("type CA")
        indices = alpha_carbons.getAtomIndices()
        @endcode

        @param[in] selection_string Python `str` containing one complete
        expression. The encoded bytes are copied during the native call and are
        not retained.
        @return A newly owned set-valued `AtomSelection` independent of this
        selector and the source PSF wrapper. Zero, one, or many atoms may be
        selected.
        @throws TypeError If `selection_string` is not a `str`.
        @throws UnicodeEncodeError If UTF-8 encoding rejects the string, such as
        for an unpaired surrogate.
        @throws RuntimeError If this selector has been closed or if the C ABI
        reports success but returns a NULL selection handle.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the string is empty or the expression
        has a lexical, syntax, operator, range, or parenthesis error, or if a
        PSF bonded-neighbor index is out of range; or with
        `APO_STATUS_RUNTIME_ERROR` for malformed PSF state, allocation failure,
        or an internal parser failure.

        @post This selector and its PSF are unchanged.
        @warning The current wrapper does not reject embedded `"\\0"`
        characters. Because the C ABI accepts a C string, only the prefix before
        the first embedded null byte is parsed. Avoid embedded null characters.
        """
        _initialize_prototypes()

        if not isinstance(selection_string, str):
            raise TypeError("selection_string must be a str")

        handle: ctypes.c_void_p = ctypes.c_void_p()

        encoded_selection: bytes = selection_string.encode("utf-8")
        c_selection_string: ctypes.c_char_p = ctypes.c_char_p(encoded_selection)

        lib().apo_atom_selector_select(
            ctypes.byref(handle), self.handle, c_selection_string
        )

        if handle.value is None:
            raise RuntimeError(
                "apo_atom_selector_select returned success but produced a NULL handle"
            )

        return AtomSelection(handle)

    def selectAtom(self, selection_string: str) -> AtomReference:
        """
        @brief Evaluates an expression that must match exactly one atom.

        The method uses the same UTF-8 conversion and native grammar as
        `select()`. Cardinality is checked only by native
        `AtomSelector::selectAtom()`: zero or multiple matches are errors rather
        than Python-side special cases.

        Example:

        @code{.py}
        selector = apo.AtomSelector(psf)
        alpha_carbon = selector.selectAtom("atom PROA 42 CA")
        atom_index = alpha_carbon.getAtomIndex()
        @endcode

        @param[in] selection_string Python `str` containing one complete
        expression. The encoded bytes are copied during the native call and are
        not retained.
        @return A newly owned `AtomReference` that retains shared native
        ownership of the selector's PSF and remains valid after the selector or
        source PSF wrapper is closed.
        @throws TypeError If `selection_string` is not a `str`.
        @throws UnicodeEncodeError If UTF-8 encoding rejects the string, such as
        for an unpaired surrogate.
        @throws RuntimeError If this selector has been closed or if the C ABI
        reports success but returns a NULL atom-reference handle.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the string is empty, the expression is
        invalid, a stored PSF bonded-neighbor index is out of range, or the
        expression selects anything other than exactly one atom; or with
        `APO_STATUS_RUNTIME_ERROR` for malformed PSF state, allocation failure,
        or an internal parser failure.

        @post This selector and its PSF are unchanged.
        @warning The current wrapper does not reject embedded `"\\0"`
        characters. Because the C ABI accepts a C string, only the prefix before
        the first embedded null byte is parsed. Avoid embedded null characters.
        """
        _initialize_prototypes()

        if not isinstance(selection_string, str):
            raise TypeError("selection_string must be a str")

        handle: ctypes.c_void_p = ctypes.c_void_p()

        encoded_selection: bytes = selection_string.encode("utf-8")
        c_selection_string: ctypes.c_char_p = ctypes.c_char_p(encoded_selection)

        lib().apo_atom_selector_select_atom(
            ctypes.byref(handle), self.handle, c_selection_string
        )

        if handle.value is None:
            raise RuntimeError(
                "apo_atom_selector_select_atom returned success but produced a NULL handle"
            )

        return AtomReference._from_handle(handle)
