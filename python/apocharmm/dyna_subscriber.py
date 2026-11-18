# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Provides the Python CHARMM DYNA energy-output subscriber.

`DynaSubscriber` owns a concrete native DYNA-writer handle and exposes its
borrowed base-subscriber view to `CudaIntegrator`.

@anchor python_dyna_subscriber_module
@see subscriber
"""

import ctypes

from ._lib import encode_path, lib
from ._types import FilePath
from ._validation import require_c_int
from .error import configure_status_function
from .subscriber import Subscriber

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_dyna_subscriber_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
        "DynaSubscriber construction",
    )

    configure_status_function(
        lib().apo_dyna_subscriber_create_with_report_frequency,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_int],
        "DynaSubscriber construction",
    )

    lib().apo_dyna_subscriber_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_dyna_subscriber_destroy.restype = None

    configure_status_function(
        lib().apo_dyna_subscriber_as_subscriber,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "DynaSubscriber base-subscriber conversion",
    )

    _prototypes_initialized = True

    return


class DynaSubscriber(Subscriber):
    """
    @brief Writes CHARMM's DYNA energy blocks to a text file.

    Construction immediately creates or truncates the text output. The blocks
    have the layout that CHARMM prints during dynamics. When a run starts, the
    native writer writes the header and a step-0 block, whose energies it
    evaluates at the starting coordinates. It then writes one block after every
    reporting interval of the run. A run starts with the first propagation
    after subscription or after a restart load. Each block holds the values the
    integrator computed in the step just completed; for the Langevin
    thermostat, Nose-Hoover, and Langevin-piston integrators these describe
    the step before the one in the step column. Restraints appear in CHARMM's
    lines, fields that apoCHARMM does not compute print as zero, and only the
    Langevin piston fills the pressure fields and adds the crystal-parameter
    lines.

    The wrapper owns the concrete C handle. Its `subscriber_handle` property is
    a borrowed view into that handle. Use the object as a context manager or
    call `close()` exactly through the wrapper; never destroy the borrowed view.

    @warning Unsubscribe before closing the wrapper. Closing invalidates the C
    base view even when the native integrator still retains the underlying C++
    subscriber.
    @warning DYNA updates are not transactional. A native write failure can
    leave a partial block after some dynamics steps have completed.

    @anchor python_dyna_subscriber
    @see subscriber
    """

    _destroy_function_name = "apo_dyna_subscriber_destroy"

    def __init__(self, path: FilePath, report_frequency: int | None = None) -> None:
        """
        @brief Constructs and owns a native DYNA writer.

        `path` is converted with `os.fsencode()` through the package helper and
        copied by native construction. Accepted path objects are `str`, `bytes`,
        `os.PathLike[str]`, and `os.PathLike[bytes]`. The original Python object
        is retained in `_path`, but native file access does not borrow it.

        `report_frequency=None` selects the native default of 10 steps,
        CHARMM's default `NPRINT`. An explicit ordinary Python `int` must fit a
        signed 32-bit C `int` and be positive. Booleans are rejected.

        @param[in] path Filesystem path to create or truncate. It must encode to
        a nonempty native path; when the path has a nonempty parent path, that
        parent path must exist. Encoded bytes are passed as a null-terminated C
        string, so an embedded null byte truncates the path observed by native
        code.
        @param[in] report_frequency Optional positive, dimensionless number of
        propagated steps between DYNA blocks.
        @throws TypeError If `path` is not accepted by `os.fsencode()`, or an
        explicit frequency is not an `int` or is a `bool`.
        @throws UnicodeEncodeError If the platform filesystem encoding cannot
        encode a string path.
        @throws ValueError If an explicit frequency is outside the signed
        32-bit C `int` range.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if native construction rejects the path or
        frequency, or `APO_STATUS_RUNTIME_ERROR` if file opening, allocation,
        base-view conversion, or an unexpected native exception fails.
        @throws RuntimeError If a native function reports success but returns a
        null concrete or base handle.

        @post On success, the file has been created or truncated and this
        wrapper owns both the concrete handle and a live borrowed base view.
        """
        _initialize_prototypes()
        super().__init__()

        handle: ctypes.c_void_p = ctypes.c_void_p()

        encoded_path: bytes = encode_path(path)
        c_path: ctypes.c_char_p = ctypes.c_char_p(encoded_path)

        if report_frequency is None:
            function_name = "apo_dyna_subscriber_create"
            lib().apo_dyna_subscriber_create(ctypes.byref(handle), c_path)
        else:
            c_report_frequency: ctypes.c_int = ctypes.c_int(
                require_c_int(report_frequency, "report_frequency")
            )
            function_name = "apo_dyna_subscriber_create_with_report_frequency"

            lib().apo_dyna_subscriber_create_with_report_frequency(
                ctypes.byref(handle), c_path, c_report_frequency
            )

        if handle.value is None:
            raise RuntimeError(
                "{} returned success but produced a NULL handle".format(function_name)
            )

        self._handle = handle

        subscriber_handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_dyna_subscriber_as_subscriber(
            ctypes.byref(subscriber_handle), self.handle
        )

        if subscriber_handle.value is None:
            raise RuntimeError(
                "apo_dyna_subscriber_as_subscriber returned success but produced a NULL handle"
            )

        self._subscriber_handle = subscriber_handle
        self._path: FilePath = path

        return

    def close(self) -> None:
        """
        @brief Releases the owned concrete C handle and clears its base view.

        The inherited close operation is idempotent. On the first call it invokes
        `apo_dyna_subscriber_destroy`, then this override clears the borrowed
        `subscriber_handle` regardless of whether the concrete handle had
        already been released.

        @return `None`.
        @post Later subscriber operations raise `RuntimeError`.
        @warning Call `CudaIntegrator.unsubscribe(self)` before closure. Native
        unsubscription requires the borrowed base view that this method clears.
        """
        super().close()
        self._subscriber_handle = ctypes.c_void_p()
        return
