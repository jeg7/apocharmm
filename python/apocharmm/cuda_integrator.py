# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import ctypes
import signal
import threading

from ._base import _ApoObject
from ._lib import encode_path, lib
from ._types import FilePath
from .error import configure_status_function

from .charmm_context import CharmmContext
from .subscriber import Subscriber

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_cuda_integrator_set_time_step,
        [ctypes.c_void_p, ctypes.c_double],
        "CudaIntegrator.setTimeStep(time_step)",
    )

    configure_status_function(
        lib().apo_cuda_integrator_set_charmm_context,
        [ctypes.c_void_p, ctypes.c_void_p],
        "CudaIntegrator.setCharmmContext(context)",
    )

    configure_status_function(
        lib().apo_cuda_integrator_initialize,
        [ctypes.c_void_p],
        "CudaIntegrator.initialize()",
    )

    configure_status_function(
        lib().apo_cuda_integrator_subscribe,
        [ctypes.c_void_p, ctypes.c_void_p],
        "CudaIntegrator.subscribe(subscriber)",
    )

    configure_status_function(
        lib().apo_cuda_integrator_unsubscribe,
        [ctypes.c_void_p, ctypes.c_void_p],
        "CudaIntegrator.unsubscribe(subscriber)",
    )

    configure_status_function(
        lib().apo_cuda_integrator_propagate,
        [ctypes.c_void_p, ctypes.c_int],
        "CudaIntegrator.propagate(num_steps)",
    )

    configure_status_function(
        lib().apo_cuda_integrator_initialize_from_restart_file,
        [ctypes.c_void_p, ctypes.c_char_p],
        "CudaIntegrator.initializeFromRestartFile(path)",
    )

    _prototypes_initialized = True

    return


class CudaIntegrator(_ApoObject):
    """
    @brief Provides the common Python interface to native CUDA integrators.

    This base wrapper does not construct a native integrator itself. Concrete
    subclasses own a concrete C handle and install a borrowed base-integrator
    view in `_integrator_handle`.

    The wrapper retains the successfully attached `CharmmContext` and each
    successfully subscribed `Subscriber`. It provides no internal
    synchronization; serialize operations involving the same integrator,
    context, or subscribers.

    @anchor python_cuda_integrator
    @see cuda_integrators
    """

    def __init__(self) -> None:
        """
        @brief Constructs an empty Python base-integrator view.

        This constructor allocates no native object. Concrete subclasses call it
        before creating their owned handle and borrowed base view.

        @post `integrator_handle` raises `RuntimeError` until a concrete
        subclass installs a live view.
        """
        super().__init__()

        self._integrator_handle: ctypes.c_void_p = ctypes.c_void_p()
        self._context: CharmmContext | None = None
        self._subscribers: list[Subscriber] = []

        return

    @property
    def integrator_handle(self) -> ctypes.c_void_p:
        """
        @brief Returns the borrowed C base-integrator pointer.

        @return The live `ctypes.c_void_p` stored by this wrapper. It aliases an
        embedded view owned by the concrete C handle and must not be destroyed
        or retained past wrapper closure.
        @throws RuntimeError If no concrete view has been installed or the
        wrapper has been closed.
        """
        if self._integrator_handle is None or self._integrator_handle.value is None:
            raise RuntimeError("apoCHARMM integrator object has been destroyed")

        return self._integrator_handle

    def close(self) -> None:
        """
        @brief Closes the concrete handle and invalidates the base view.

        Closure is idempotent. The method destroys the owned concrete C handle,
        clears the borrowed base view, and releases retained Python context and
        subscriber references.

        @return `None`.
        @throws RuntimeError If a subclass has not configured a native destroy
        function.
        @post Later handle access raises `RuntimeError`.
        @warning The method does not implicitly unsubscribe native subscribers.
        Unsubscribe before closure to avoid retaining a native ownership cycle.
        """
        super().close()

        self._integrator_handle = ctypes.c_void_p()
        self._context = None
        self._subscribers = []

        return

    def setTimeStep(self, time_step: float) -> None:
        """
        @brief Sets the public integrator time step.

        `time_step` is converted with `ctypes.c_double` and passed by value.

        @param[in] time_step Finite positive time step in picoseconds.
        @return `None`.
        @throws TypeError If `time_step` cannot be converted to a C `double`.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the converted value is non-finite or
        not positive, or `APO_STATUS_RUNTIME_ERROR` for an unexpected native
        failure.
        @warning The native setter does not reinitialize attached constraint or
        concrete-integrator derived state.
        """
        _initialize_prototypes()

        c_time_step: ctypes.c_double = ctypes.c_double(time_step)

        lib().apo_cuda_integrator_set_time_step(self.integrator_handle, c_time_step)

        return

    def setCharmmContext(self, context: CharmmContext) -> None:
        """
        @brief Attaches a context and initializes concrete native state.

        The argument must be a live `CharmmContext`. The wrapper retains the
        same Python object only after the native call succeeds.

        @param[in] context Context wrapper to attach.
        @return `None`.
        @throws TypeError If `context` is not a `CharmmContext`.
        @throws RuntimeError If this integrator or `context` has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` for null or duplicate attachment,
        `APO_STATUS_NOT_INITIALIZED` for missing context or concrete
        prerequisites, `APO_STATUS_NOT_IMPLEMENTED` for direct base
        initialization, `APO_STATUS_CUDA_ERROR` for CUDA initialization
        failure, or `APO_STATUS_RUNTIME_ERROR` for another native failure.
        @post On success, `_context` retains `context`.
        """
        _initialize_prototypes()

        if not isinstance(context, CharmmContext):
            raise TypeError("CudaIntegrator.setCharmmContext expects a CharmmContext")

        lib().apo_cuda_integrator_set_charmm_context(
            self.integrator_handle, context.handle
        )

        self._context = context

        return

    def initialize(self) -> None:
        """
        @brief Reinitializes concrete state using the attached context.

        @return `None`.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status `APO_STATUS_NOT_INITIALIZED`
        when no context is attached or a concrete prerequisite such as the
        crystal type is absent, `APO_STATUS_INVALID_ARGUMENT` when the
        configured context fails concrete validation,
        `APO_STATUS_NOT_IMPLEMENTED` for the direct base hook,
        `APO_STATUS_CUDA_ERROR` for CUDA initialization failure, or
        `APO_STATUS_RUNTIME_ERROR` for another native failure.
        @note Call between completed propagation calls after changing the
        context's crystal type. Integration history and RNG state are rebuilt;
        the completed-step counter, friction, RNG seed/sequence, reference
        temperature/pressure, Nose-Hoover mass, and atomic velocities are
        retained. When the piston layout changes, piston storage is rebuilt with
        default masses and piston/Nose-Hoover dynamic history and phase averages
        are reset; configure custom new-layout masses afterward if needed.
        Reinitializing an unchanged type preserves the selected masses and
        friction. Use `initializeFromRestartFile()` to resume an exact restart.
        @warning Initialization is not transactional. Serialize it with other
        operations on this integrator and its context.
        """
        _initialize_prototypes()

        lib().apo_cuda_integrator_initialize(self.integrator_handle)

        return

    def subscribe(self, subscriber: Subscriber) -> None:
        """
        @brief Subscribes one output wrapper.

        The native integrator caches the subscriber's current reporting
        frequency and retains its native object. After native success, this
        wrapper appends the same Python subscriber to `_subscribers`.

        @param[in] subscriber Live concrete `Subscriber` wrapper.
        @return `None`.
        @throws TypeError If `subscriber` is not a `Subscriber`.
        @throws RuntimeError If either wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` for null or duplicate attachment,
        `APO_STATUS_NOT_INITIALIZED` if shared native ownership is unavailable,
        or `APO_STATUS_RUNTIME_ERROR` for allocation or another native failure.
        @throws MemoryError If the Python retention list cannot grow after
        native attachment.
        @warning A `MemoryError` after native success can leave native and
        Python retention state inconsistent.
        """
        _initialize_prototypes()

        if not isinstance(subscriber, Subscriber):
            raise TypeError("CudaIntegrator.subscribe expects a Subscriber")

        lib().apo_cuda_integrator_subscribe(
            self.integrator_handle, subscriber.subscriber_handle
        )

        self._subscribers.append(subscriber)

        return

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """
        @brief Removes one subscriber from native and Python scheduling state.

        The native unsubscription runs first. On native success, the matching
        Python object is removed from `_subscribers`.

        @param[in] subscriber Live `Subscriber` wrapper identifying the native
        object to remove.
        @return `None`.
        @throws TypeError If `subscriber` is not a `Subscriber`.
        @throws RuntimeError If either wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the subscriber is not attached or a
        handle is rejected, or `APO_STATUS_RUNTIME_ERROR` for another native
        failure.
        @throws ValueError If native and Python retention state is inconsistent
        and the Python list does not contain `subscriber`.
        """
        _initialize_prototypes()

        if not isinstance(subscriber, Subscriber):
            raise TypeError("CudaIntegrator.unsubscribe expects a Subscriber")

        lib().apo_cuda_integrator_unsubscribe(
            self.integrator_handle, subscriber.subscriber_handle
        )

        self._subscribers.remove(subscriber)

        return

    def propagate(self, num_steps: int) -> None:
        """
        @brief Propagates dynamics and scheduled subscribers.

        `num_steps` must fit a signed 32-bit C `int`. Python booleans follow
        integer rules: `True` becomes one, while `False` reaches native
        validation as zero. On the main Python thread, the wrapper temporarily
        installs the default `SIGINT` handler during the native call and restores
        the previous handler in a `finally` block.

        @param[in] num_steps Positive dimensionless number of steps.
        @return `None`.
        @throws TypeError If comparison or conversion to `ctypes.c_int` rejects
        the supplied object.
        @throws ValueError If the value is outside the signed 32-bit C `int`
        range.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status `APO_STATUS_INVALID_ARGUMENT`,
        `APO_STATUS_NOT_INITIALIZED`, `APO_STATUS_NOT_IMPLEMENTED`,
        `APO_STATUS_CUDA_ERROR`, or `APO_STATUS_RUNTIME_ERROR` according to the
        documented native failure.
        @warning Propagation is not transactional. The native completed-step
        counter advances after each successful concrete step. Earlier state and
        output remain after failure; the failing step may partly modify state.
        """
        _initialize_prototypes()

        if num_steps < -(2**31) or num_steps > 2**31 - 1:
            raise ValueError("num_steps must fit in int")

        c_num_steps: ctypes.c_int = ctypes.c_int(num_steps)

        restore_signal_handler: bool = (
            threading.current_thread() is threading.main_thread()
        )
        previous_sigint_handler = None

        if restore_signal_handler:
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        try:
            lib().apo_cuda_integrator_propagate(self.integrator_handle, c_num_steps)
        finally:
            if restore_signal_handler:
                signal.signal(signal.SIGINT, previous_sigint_handler)

        return

    def initializeFromRestartFile(self, path: FilePath) -> None:
        """
        @brief Initializes concrete state from a restart file.

        `path` may be `str`, `bytes`, `os.PathLike[str]`, or
        `os.PathLike[bytes]`. It is encoded with `os.fsencode()`, passed as a
        temporary null-terminated byte string, and not retained.

        @param[in] path Restart-file path.
        @return `None`.
        @throws TypeError If `path` is not accepted by `os.fsencode()`.
        @throws ValueError If path encoding rejects an embedded null byte.
        @throws OSError If filesystem path conversion fails.
        @throws RuntimeError If the wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT`, `APO_STATUS_NOT_INITIALIZED`,
        `APO_STATUS_NOT_IMPLEMENTED`, `APO_STATUS_CUDA_ERROR`, or
        `APO_STATUS_RUNTIME_ERROR` according to restart validation, parsing,
        transfer, or concrete support.
        @warning Restart loading is not transactional.
        """
        _initialize_prototypes()

        encoded_path: bytes = encode_path(path)
        c_path: ctypes.c_char_p = ctypes.c_char_p(encoded_path)

        lib().apo_cuda_integrator_initialize_from_restart_file(
            self.integrator_handle, c_path
        )

        return
