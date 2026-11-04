# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Provides the Python CHARMM-style restart-file subscriber.

`RestartSubscriber` owns a concrete native restart-writer handle and exposes
its borrowed base-subscriber view to `CudaIntegrator`.

@anchor python_restart_subscriber_module
@see subscriber
"""

import ctypes

from ._lib import encode_path, lib
from ._types import FilePath
from .error import configure_status_function
from .subscriber import Subscriber

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_restart_subscriber_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
        "RestartSubscriber construction",
    )

    configure_status_function(
        lib().apo_restart_subscriber_create_with_report_frequency,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_int],
        "RestartSubscriber construction",
    )

    lib().apo_restart_subscriber_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_restart_subscriber_destroy.restype = None

    configure_status_function(
        lib().apo_restart_subscriber_as_subscriber,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "RestartSubscriber base-subscriber conversion",
    )

    _prototypes_initialized = True

    return


class RestartSubscriber(Subscriber):
    """
    @brief Replaces a CHARMM-style restart file with current simulation state.

    Construction immediately creates or truncates the text output. During
    propagation, each scheduled native update rewrites the complete file with
    current context state and supported-integrator state. Box lengths and
    coordinates use angstroms, velocities use angstroms per AKMA time unit, and
    average temperature uses kelvin. The standard core uses CHARMM version-52
    field sizes and D22.15 precision. A trailing apoCHARMM RNG extension retains
    native Philox state. Decimal serialization is not bitwise lossless.

    Energy/statistics history and the thermostat reference-energy origin are
    explicitly reset. Supported imports use ordinary leapfrog state with CUBI,
    TETR, or ORTH cells and default CHARMM energy-selection flags. Match
    topology, timestep, constraints, potential, and thermostat/piston
    configuration in both programs. Configure a new RNG seed before importing a
    foreign restart; foreign RNG state is never interpreted as apoCHARMM Philox
    state.

    Call `CharmmContext.setCrystalType()` before writing or loading a restart.
    The context's explicit crystal type supplies the output label and must
    match the label on import. Box lengths do not select or change that type.

    Native updates support `CudaNoseHooverIntegrator`,
    `CudaLangevinPistonIntegrator`, and `CudaLangevinThermostatIntegrator`.
    Device-backed state is copied to host, and each transfer synchronizes the
    current CUDA device.

    The wrapper owns the concrete C handle. Its `subscriber_handle` property is
    a borrowed view into that handle. Use the object as a context manager or
    call `close()` through the wrapper; never destroy the borrowed view.

    @warning Unsubscribe before closing the wrapper. Closing invalidates the C
    base view even when the native integrator still retains the underlying C++
    subscriber.
    @warning Each update truncates the target before all sections are written. A
    native failure can replace a usable restart with a partial file.

    @anchor python_restart_subscriber
    @see subscriber
    """

    _destroy_function_name = "apo_restart_subscriber_destroy"

    def __init__(self, path: FilePath, report_frequency: int | None = None) -> None:
        """
        @brief Constructs and owns a native restart writer.

        `path` is converted with `os.fsencode()` through the package helper and
        copied by native construction. Accepted path objects are `str`, `bytes`,
        `os.PathLike[str]`, and `os.PathLike[bytes]`. The original Python object
        is retained in `_path`, but native file access does not borrow it.

        `report_frequency=None` selects the native default of 1000 steps. An
        explicit ordinary Python `int` must fit a signed 32-bit C `int` and be
        positive. Boolean values follow Python integer conversion: `True`
        becomes one and `False` is rejected by native validation.

        @param[in] path Filesystem path to create or truncate. It must encode to
        a nonempty native path; when the path has a nonempty parent path, that
        parent path must exist. Encoded bytes are passed as a null-terminated C
        string, so an embedded null byte truncates the path observed by native
        code.
        @param[in] report_frequency Optional positive, dimensionless number of
        propagated steps between restart replacements.
        @throws TypeError If `path` is not accepted by `os.fsencode()`, or an
        explicit frequency cannot be compared with integer bounds or converted
        by `ctypes.c_int`.
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
            create_function_name: str = "apo_restart_subscriber_create"
            lib().apo_restart_subscriber_create(ctypes.byref(handle), c_path)
        else:
            if report_frequency < -(2**31) or report_frequency > 2**31 - 1:
                raise ValueError("report_frequency must fit in int")

            c_report_frequency: ctypes.c_int = ctypes.c_int(report_frequency)

            create_function_name: str = (
                "apo_restart_subscriber_create_with_report_frequency"
            )

            lib().apo_restart_subscriber_create_with_report_frequency(
                ctypes.byref(handle), c_path, c_report_frequency
            )

        if handle.value is None:
            raise RuntimeError(
                "{} returned success but produced a NULL handle".format(
                    create_function_name
                )
            )

        self._handle = handle

        subscriber_handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_restart_subscriber_as_subscriber(
            ctypes.byref(subscriber_handle), self.handle
        )

        if subscriber_handle.value is None:
            raise RuntimeError(
                "apo_restart_subscriber_as_subscriber returned success but produced a NULL handle"
            )

        self._subscriber_handle = subscriber_handle
        self._path: FilePath = path

        return

    def close(self) -> None:
        """
        @brief Releases the owned concrete C handle and clears its base view.

        The inherited close operation is idempotent. On the first call it invokes
        `apo_restart_subscriber_destroy`, then this override clears the borrowed
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
