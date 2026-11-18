# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Provides the Python base view shared by concrete subscribers.

`Subscriber` exposes the common reporting-frequency API used by
`DcdSubscriber`, `RestartSubscriber`, and `CudaIntegrator`. User code should
construct a concrete subscriber rather than instantiate the base wrapper.

@anchor python_subscriber_module
@see subscriber
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

    configure_status_function(
        lib().apo_subscriber_set_report_frequency,
        [ctypes.c_void_p, ctypes.c_int],
        "Subscriber.setReportFrequency(report_frequency)",
    )

    configure_status_function(
        lib().apo_subscriber_get_report_frequency,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "Subscriber.getReportFrequency()",
    )

    _prototypes_initialized = True

    return


class Subscriber(_ApoObject):
    """
    @brief Provides the common Python interface to a native subscriber.

    The base wrapper does not create or own a native subscriber. Concrete
    subclasses own a concrete C handle and store a borrowed embedded base view
    in `_subscriber_handle`. That view is valid only until the concrete wrapper
    is closed.

    A `CudaIntegrator` retains each successfully subscribed Python wrapper in
    its `_subscribers` list and the native integrator retains the native C++
    subscriber. Unsubscribe before closing a concrete wrapper so its borrowed C
    view remains available to the native unsubscribe call.

    The wrapper provides no internal synchronization. Do not overlap closure,
    frequency changes, subscription, or propagation involving the same object
    from multiple threads.

    @anchor python_subscriber
    @see subscriber
    """

    def __init__(self) -> None:
        """
        @brief Constructs an empty Python base-subscriber view.

        This constructor allocates no native object. It is used by concrete
        subclasses before they install a borrowed base view.

        @post `subscriber_handle` raises `RuntimeError` until a concrete
        subclass installs a live native view.
        """
        super().__init__()
        self._subscriber_handle: ctypes.c_void_p = ctypes.c_void_p()
        return

    @property
    def subscriber_handle(self) -> ctypes.c_void_p:
        """
        @brief Returns the borrowed C base-subscriber pointer.

        @return The live `ctypes.c_void_p` stored by this wrapper. The returned
        object aliases wrapper state and must not be destroyed or retained past
        wrapper closure.
        @throws RuntimeError If no concrete view has been installed or the
        wrapper has been closed.
        """
        if self._subscriber_handle is None or self._subscriber_handle.value is None:
            raise RuntimeError("apoCHARMM subscriber object has been destroyed")
        return self._subscriber_handle

    def setReportFrequency(self, report_frequency: int) -> None:
        """
        @brief Sets the native reporting interval.

        The value must be a Python `int` that fits a signed 32-bit C `int`;
        booleans are rejected. Configure the interval before subscription
        because the native integrator caches it when attaching the subscriber.

        @param[in] report_frequency Positive, dimensionless number of propagated
        steps between scheduled updates.
        @return `None`.
        @throws TypeError If `report_frequency` is not an `int` or is a `bool`.
        @throws ValueError If the value is outside the signed 32-bit C `int`
        range.
        @throws RuntimeError If the concrete wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the converted value is not positive, or
        `APO_STATUS_RUNTIME_ERROR` if an unexpected native exception occurs.

        @post A native validation failure leaves the previous frequency
        unchanged.
        """
        _initialize_prototypes()

        c_report_frequency: ctypes.c_int = ctypes.c_int(
            require_c_int(report_frequency, "report_frequency")
        )

        lib().apo_subscriber_set_report_frequency(
            self.subscriber_handle, c_report_frequency
        )

        return

    def getReportFrequency(self) -> int:
        """
        @brief Returns the native reporting interval.

        @return A positive, dimensionless Python `int`.
        @throws RuntimeError If the concrete wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the base view is rejected, or
        `APO_STATUS_RUNTIME_ERROR` if an unexpected native exception occurs.
        """
        _initialize_prototypes()

        c_report_frequency = ctypes.c_int()

        lib().apo_subscriber_get_report_frequency(
            ctypes.byref(c_report_frequency), self.subscriber_handle
        )

        return int(c_report_frequency.value)
