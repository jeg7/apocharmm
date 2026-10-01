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
from typing import Protocol, cast

from ._base import _ApoObject
from ._lib import lib
from .enums import PeriodicBoundaryCondition, VdwType
from .error import configure_status_function

from .charmm_parameters import CharmmParameters
from .charmm_psf import CharmmPsf

_prototypes_initialized: bool = False


class _SubscribableForce(Protocol):
    """
    @brief Defines the private Python subscription extension protocol.

    An object accepted by `ForceManager.subscribe()` must provide matching
    subscribe and unsubscribe callbacks. The manager retains a strong Python
    reference only after the subscribe callback succeeds.

    Native `_ApoObject` implementations normally forward these callbacks to
    their C ABI subscription functions. Pure Python implementations may define
    other behavior but remain responsible for honoring the manager lifecycle.
    """

    def _subscribe_to_force_manager(
        self, force_manager: "ForceManager", force_tag: str | None = None
    ) -> None:
        """
        @brief Connects this object to a ForceManager.

        @param[in] force_manager Live manager receiving the subscription.
        @param[in] force_tag Optional non-empty force label. `None` permits a
        native force implementation to choose its default tag.
        """
        return

    def _unsubscribe_from_force_manager(self, force_manager: "ForceManager") -> None:
        """
        @brief Disconnects this object from a ForceManager.

        @param[in] force_manager Live manager from which this object is being
        removed.
        """
        return


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_force_manager_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p],
        "ForceManager construction",
    )

    lib().apo_force_manager_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_force_manager_destroy.restype = None

    configure_status_function(
        lib().apo_force_manager_set_box_dimensions,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "ForceManager.setBoxDimensions(box_dimensions)",
    )

    configure_status_function(
        lib().apo_force_manager_set_kappa,
        [ctypes.c_void_p, ctypes.c_double],
        "ForceManager.setKappa(kappa)",
    )

    configure_status_function(
        lib().apo_force_manager_set_cutoff,
        [ctypes.c_void_p, ctypes.c_double],
        "ForceManager.setCutoff(cutoff)",
    )

    configure_status_function(
        lib().apo_force_manager_set_ctonnb,
        [ctypes.c_void_p, ctypes.c_double],
        "ForceManager.setCtonnb(ctonnb)",
    )

    configure_status_function(
        lib().apo_force_manager_set_ctofnb,
        [ctypes.c_void_p, ctypes.c_double],
        "ForceManager.setCtofnb",
    )

    configure_status_function(
        lib().apo_force_manager_set_fft_grid,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t],
        "ForceManager.setFFTGrid(grid)",
    )

    configure_status_function(
        lib().apo_force_manager_set_pme_spline_order,
        [ctypes.c_void_p, ctypes.c_int],
        "ForceManager.setPmeSplineOrder(order)",
    )

    configure_status_function(
        lib().apo_force_manager_set_periodic_boundary_condition,
        [ctypes.c_void_p, ctypes.c_int],
        "ForceManager.setPeriodicBoundaryCondition(pbc)",
    )

    configure_status_function(
        lib().apo_force_manager_set_vdw_type,
        [ctypes.c_void_p, ctypes.c_int],
        "ForceManager.setVdwType(vdw_type)",
    )

    configure_status_function(
        lib().apo_force_manager_set_print_energy_decomposition,
        [ctypes.c_void_p, ctypes.c_bool],
        "ForceManager.setPrintEnergyDecomposition(flag)",
    )

    configure_status_function(
        lib().apo_force_manager_get_num_atoms,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "ForceManager.getNumAtoms()",
    )

    configure_status_function(
        lib().apo_force_manager_is_initialized,
        [ctypes.POINTER(ctypes.c_bool), ctypes.c_void_p],
        "ForceManager.isInitialized()",
    )

    configure_status_function(
        lib().apo_force_manager_get_box_dimensions,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_void_p],
        "ForceManager.getBoxDimensions()",
    )

    configure_status_function(
        lib().apo_force_manager_get_kappa,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "ForceManager.getKappa()",
    )

    configure_status_function(
        lib().apo_force_manager_get_cutoff,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "ForceManager.getCutoff",
    )

    configure_status_function(
        lib().apo_force_manager_get_ctonnb,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "ForceManager.getCtonnb",
    )

    configure_status_function(
        lib().apo_force_manager_get_ctofnb,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "ForceManager.getCtofnb",
    )

    configure_status_function(
        lib().apo_force_manager_get_fft_grid,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_void_p],
        "ForceManager.getFFTGrid()",
    )

    configure_status_function(
        lib().apo_force_manager_get_pme_spline_order,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "ForceManager.getPmeSplineOrder()",
    )

    configure_status_function(
        lib().apo_force_manager_get_periodic_boundary_condition,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "ForceManager.getPeriodicBoundaryCondition()",
    )

    configure_status_function(
        lib().apo_force_manager_get_vdw_type,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "ForceManager.getVdwType()",
    )

    configure_status_function(
        lib().apo_force_manager_is_composite,
        [ctypes.POINTER(ctypes.c_bool), ctypes.c_void_p],
        "ForceManager.isComposite()",
    )

    _prototypes_initialized = True

    return


class ForceManager(_ApoObject):
    """
    @anchor python_force_manager
    @brief Configures and owns a native apoCHARMM ForceManager.

    Construct the wrapper from a live `CharmmPsf` and
    `CharmmParameters`. The wrapper owns its C handle, while the C handle
    retains native shared ownership of both collaborators. The Python wrapper
    also retains the collaborator wrappers.

    Configuration methods copy Python scalar or sequence data through the C
    ABI. Force backends are initialized when the manager is attached to a
    sufficiently configured `CharmmContext`, or through direct native C++
    initialization; initialization is not exposed as a Python method.

    `subscribe()` retains each force wrapper after its subscription callback
    succeeds. `close()` releases the manager handle, clears retained subscribed
    forces, and is idempotent. Public methods that access `self.handle` after
    closure raise `RuntimeError`.

    Native status failures are raised as `ApoCharmmError`. The wrapper and
    native manager provide no internal synchronization for concurrent host
    access.

    @see force_manager
    """

    _destroy_function_name = "apo_force_manager_destroy"

    def __init__(self, psf: CharmmPsf, parameters: CharmmParameters) -> None:
        """
        @brief Constructs an owning Python ForceManager wrapper.

        The native manager retains shared ownership of the underlying PSF and
        parameter objects but remains uninitialized. This wrapper also keeps
        strong references to `psf` and `parameters`.

        @param[in] psf Live `CharmmPsf` wrapper.
        @param[in] parameters Live `CharmmParameters` wrapper.

        @throws TypeError If `psf` is not a `CharmmPsf` or `parameters` is not
        a `CharmmParameters`.
        @throws RuntimeError If either collaborator wrapper is closed or native
        construction reports success but produces a NULL handle.
        @throws ApoCharmmError If native handle validation, allocation, or
        CUDA-backed manager construction fails.
        """
        _initialize_prototypes()
        super().__init__()

        if not isinstance(psf, CharmmPsf):
            raise TypeError("ForceManager expects a CharmmPsf")

        if not isinstance(parameters, CharmmParameters):
            raise TypeError("ForceManager expects a CharmmParameters")

        handle = ctypes.c_void_p()

        lib().apo_force_manager_create(
            ctypes.byref(handle), psf.handle, parameters.handle
        )

        if handle.value is None:
            raise RuntimeError(
                "apo_force_manager_create returned success but produced a NULL handle"
            )

        self._handle = handle

        self._psf: CharmmPsf | None = psf
        self._parameters: CharmmParameters | None = parameters
        self._subscribed_forces: list[_SubscribableForce] = []

        return

    @classmethod
    def _from_handle(cls, handle: ctypes.c_void_p) -> "ForceManager":
        """
        @brief Wraps an already owned C ForceManager handle.

        This internal constructor takes ownership of `handle`. It is used for a
        newly owned handle returned by another C ABI object, such as
        `CharmmContext.getForceManager()`. It does not reconstruct or retain
        Python PSF and parameter wrappers because their native lifetimes are
        already preserved by the C handle.

        @param[in] handle Non-NULL newly owned `ctypes.c_void_p`. The caller
        must not destroy or wrap the same ownership again.

        @return New owning `ForceManager` wrapper.
        @throws RuntimeError If `handle` is NULL.
        """
        _initialize_prototypes()

        if handle.value is None:
            raise RuntimeError("ForceManager._from_handle received a NULL handle")

        obj = cls.__new__(cls)
        _ApoObject.__init__(obj)

        obj._handle = handle
        obj._psf = None
        obj._parameters = None
        obj._subscribed_forces = []

        return obj

    def close(self) -> None:
        """
        @brief Releases the owned native manager handle.

        The operation is idempotent. After native destruction, retained
        subscribed-force references are cleared. Retained PSF and parameter
        wrapper attributes are not cleared by this override.

        @post Public operations that access `self.handle` raise `RuntimeError`.
        """
        super().close()

        if hasattr(self, "_subscribed_forces"):
            self._subscribed_forces = []

        return

    def subscribe(
        self, force: _SubscribableForce, force_tag: str | None = None
    ) -> None:
        """
        @brief Subscribes a native or protocol-compatible force.

        `force` must provide
        `_subscribe_to_force_manager(force_manager, force_tag)`. The callback
        runs before this wrapper retains a strong reference. On successful
        return, the reference prevents the force wrapper from being collected
        while it remains subscribed.

        Native `_ApoObject` forces perform empty-tag and duplicate validation
        at the native boundary. Pure Python protocol objects are checked in
        this method before their callback runs.

        @param[in] force Native force wrapper or protocol-compatible Python
        object.
        @param[in] force_tag Optional tag. It must be a `str` when supplied.
        `None` lets a native force select its default tag.

        @throws TypeError If `force_tag` is not a `str`, or `force` does not
        expose a callable subscription method.
        @throws ValueError If a pure Python force uses an empty tag or is
        already retained by identity.
        @throws RuntimeError If this manager or a native force wrapper is
        closed.
        @throws ApoCharmmError If a native subscription rejects a null
        resource, empty tag, duplicate force, or other native setup.
        """
        _initialize_prototypes()

        if force_tag is not None and not isinstance(force_tag, str):
            raise TypeError("force_tag must be a str")

        is_native_force: bool = isinstance(force, _ApoObject)

        if force_tag == "" and not is_native_force:
            raise ValueError("force_tag must not be empty")

        subscribe_method = getattr(force, "_subscribe_to_force_manager", None)
        if not callable(subscribe_method):
            raise TypeError(
                "ForceManager.subscribe expects an object with _subscribe_to_force_manager(force_manager, force_tag)"
            )

        if not is_native_force and any(
            existing_force is force for existing_force in self._subscribed_forces
        ):
            raise ValueError("force is already subscribed to this ForceManager")

        subscribable_force: _SubscribableForce = cast(_SubscribableForce, force)
        subscribable_force._subscribe_to_force_manager(self, force_tag)

        self._subscribed_forces.append(subscribable_force)

        return

    def unsubscribe(self, force: _SubscribableForce) -> None:
        """
        @brief Unsubscribes a native or protocol-compatible force.

        The object's `_unsubscribe_from_force_manager(force_manager)` callback
        runs before this wrapper removes its retained reference. The retained
        reference is removed only after the callback succeeds.

        @param[in] force Force object to remove by Python object identity.

        @throws TypeError If `force` does not expose a callable unsubscription
        method.
        @throws ValueError If a pure Python force is not currently retained.
        @throws RuntimeError If this manager or a native force wrapper is
        closed.
        @throws ApoCharmmError If native unsubscription rejects the object or
        reports another failure.
        """
        _initialize_prototypes()

        is_native_force: bool = isinstance(force, _ApoObject)

        unsubscribe_method = getattr(force, "_unsubscribe_from_force_manager", None)
        if not callable(unsubscribe_method):
            raise TypeError(
                "ForceManager.unsubscribe expects an object with _unsubscribe_from_force_manager(force_manager)"
            )

        index_to_remove: int | None = None
        for index, existing_force in enumerate(self._subscribed_forces):
            if existing_force is force:
                index_to_remove = index
                break

        if index_to_remove is None and not is_native_force:
            raise ValueError("force is not subscribed to this ForceManager")

        subscribable_force: _SubscribableForce = cast(_SubscribableForce, force)
        subscribable_force._unsubscribe_from_force_manager(self)

        if index_to_remove is not None:
            del self._subscribed_forces[index_to_remove]

        return

    def setBoxDimensions(self, box_dimensions: Sequence[float]) -> None:
        """
        @brief Sets three orthorhombic box dimensions.

        Every element is converted with `float()`, copied into a temporary
        contiguous `ctypes.c_double` buffer, and copied again by the native
        manager.

        @param[in] box_dimensions Iterable containing exactly three finite
        positive values in `[x, y, z]` order, in angstroms.

        @throws TypeError If `box_dimensions` is not iterable or an element
        cannot be converted with `float()`.
        @throws ValueError If element conversion fails.
        @throws OverflowError If conversion to a Python float overflows.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native length or value validation fails,
        or a native or subscribed force rejects the update.
        """
        _initialize_prototypes()

        box_values: list[float] = [float(value) for value in box_dimensions]

        c_buffer_type = ctypes.c_double * len(box_values)
        c_buffer = c_buffer_type(*box_values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(box_values))

        lib().apo_force_manager_set_box_dimensions(self.handle, c_buffer, c_buffer_len)

        return

    def setKappa(self, kappa: float) -> None:
        """
        @brief Sets the Ewald splitting parameter.

        The value is converted to `ctypes.c_double` and narrowed to native
        `float`.

        @param[in] kappa Finite non-negative value in inverse angstroms.

        @throws TypeError If `kappa` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the narrowed native value is non-finite or
        negative, or native validation otherwise fails.
        """
        _initialize_prototypes()

        c_kappa: ctypes.c_double = ctypes.c_double(kappa)

        lib().apo_force_manager_set_kappa(self.handle, c_kappa)

        return

    def setCutoff(self, cutoff: float) -> None:
        """
        @brief Sets the direct-space pair-list cutoff.

        @param[in] cutoff Value accepted by `ctypes.c_double`, in angstroms. The
        narrowed native value must be finite and positive.

        @throws TypeError If `cutoff` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If native cutoff validation fails.
        """
        _initialize_prototypes()

        c_cutoff: ctypes.c_double = ctypes.c_double(cutoff)

        lib().apo_force_manager_set_cutoff(self.handle, c_cutoff)

        return

    def setCtonnb(self, ctonnb: float) -> None:
        """
        @brief Sets the outer switching distance exposed as `ctonnb`.

        The current native backend uses this value as the distance at which the
        switching function reaches zero.

        @param[in] ctonnb Value accepted by `ctypes.c_double`, in angstroms.
        The narrowed native value must be finite and positive.

        @throws TypeError If `ctonnb` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If native value validation fails.
        """
        _initialize_prototypes()

        c_ctonnb: ctypes.c_double = ctypes.c_double(ctonnb)

        lib().apo_force_manager_set_ctonnb(self.handle, c_ctonnb)

        return

    def setCtofnb(self, ctofnb: float) -> None:
        """
        @brief Sets the inner switching distance exposed as `ctofnb`.

        The current native backend uses this value as the distance at which
        switching begins.

        @param[in] ctofnb Value accepted by `ctypes.c_double`, in angstroms.
        The narrowed native value must be finite and positive.

        @throws TypeError If `ctofnb` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If native value validation fails.
        """
        _initialize_prototypes()

        c_ctofnb: ctypes.c_double = ctypes.c_double(ctofnb)

        lib().apo_force_manager_set_ctofnb(self.handle, c_ctofnb)

        return

    def setFFTGrid(self, grid: Sequence[int]) -> None:
        """
        @brief Sets the three PME FFT grid dimensions.

        Every element is converted with `int()`, copied into a temporary
        contiguous `ctypes.c_int` buffer, and copied by the native manager.

        @param[in] grid Iterable containing exactly three positive values in
        `[nfftx, nffty, nfftz]` order. Each value is converted with `int()`.

        @throws TypeError If `grid` is not iterable or an element cannot be
        converted with `int()`.
        @throws ValueError If integer conversion fails.
        @throws OverflowError If Python integer conversion overflows.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native length or positivity validation
        fails.
        """
        _initialize_prototypes()

        grid_values: list[int] = [int(value) for value in grid]

        c_buffer_type = ctypes.c_int * len(grid_values)
        c_buffer = c_buffer_type(*grid_values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(grid_values))

        lib().apo_force_manager_set_fft_grid(self.handle, c_buffer, c_buffer_len)

        return

    def setPmeSplineOrder(self, order: int) -> None:
        """
        @brief Sets the PME interpolation spline order.

        @param[in] order Positive dimensionless value accepted by
        `ctypes.c_int`.

        @throws TypeError If `order` cannot initialize `ctypes.c_int`.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the resulting native order is not positive.
        """
        _initialize_prototypes()

        c_order: ctypes.c_int = ctypes.c_int(order)

        lib().apo_force_manager_set_pme_spline_order(self.handle, c_order)

        return

    def setPeriodicBoundaryCondition(
        self, pbc: PeriodicBoundaryCondition | int
    ) -> None:
        """
        @brief Sets the periodic boundary condition.

        `pbc` is first converted to `PeriodicBoundaryCondition`, then forwarded
        as its integer C representation.

        @param[in] pbc `PeriodicBoundaryCondition` or corresponding integer.
        `P1` and `P21`, with values `1` and `2`, are accepted. `UNSET`, with
        value `0`, is declared but rejected by the native setter.

        @throws ValueError If `pbc` is not a declared enum value.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If `pbc` is `PeriodicBoundaryCondition.UNSET`, or
        native handle or enum validation otherwise fails.
        """
        _initialize_prototypes()

        try:
            pbc_value: PeriodicBoundaryCondition = PeriodicBoundaryCondition(pbc)
        except ValueError as exc:
            raise ValueError(f"invalid pbc: {pbc!r}") from exc

        c_pbc: ctypes.c_int = ctypes.c_int(int(pbc_value))

        lib().apo_force_manager_set_periodic_boundary_condition(self.handle, c_pbc)

        return

    def setVdwType(self, vdw_type: VdwType | int) -> None:
        """
        @brief Sets the native van der Waals model.

        `vdw_type` is normalized through `VdwType`. Operational native values
        are `VdwType.VSH` through `VdwType.DBEXP`, with integer values `1`
        through `6`. `VdwType.NONE` is representable by the Python enum but is
        rejected by the native setter.

        @param[in] vdw_type `VdwType` or corresponding integer.

        @throws ValueError If `vdw_type` is not declared by `VdwType`.
        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the normalized value is `VdwType.NONE` or the
        native setter otherwise rejects it.
        """
        _initialize_prototypes()

        try:
            vdw_type_value: VdwType = VdwType(vdw_type)
        except ValueError as exc:
            raise ValueError(f"invalid vdw_type: {vdw_type!r}") from exc

        c_vdw_type: ctypes.c_int = ctypes.c_int(int(vdw_type_value))

        lib().apo_force_manager_set_vdw_type(self.handle, c_vdw_type)

        return

    def setPrintEnergyDecomposition(self, flag: bool = True) -> None:
        """
        @brief Enables or disables native energy-decomposition printing.

        `flag` is converted with `ctypes.c_bool`; the wrapper does not require
        that its runtime type be exactly `bool`.

        @param[in] flag Truth-valued object selecting output. Omitting the
        argument enables printing.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If native handle validation fails.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool(flag)

        lib().apo_force_manager_set_print_energy_decomposition(self.handle, c_flag)

        return

    def getNumAtoms(self) -> int:
        """
        @brief Returns the atom count reported by the native PSF.

        @return New Python `int`.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native manager has no PSF or output
        validation fails.

        @warning This function will be deprecated in the future.
        """
        _initialize_prototypes()

        c_num_atoms: ctypes.c_int = ctypes.c_int()

        lib().apo_force_manager_get_num_atoms(ctypes.byref(c_num_atoms), self.handle)

        return int(c_num_atoms.value)

    def isInitialized(self) -> bool:
        """
        @brief Reports whether native force initialization completed.

        @return New Python `bool`.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If native handle or output validation fails.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool()

        lib().apo_force_manager_is_initialized(ctypes.byref(c_flag), self.handle)

        return bool(c_flag.value)

    def getBoxDimensions(self) -> tuple[float, float, float]:
        """
        @brief Returns the three stored box dimensions.

        The native values are copied into a new fixed-size C buffer and then
        into a new Python tuple.

        @return New `(x, y, z)` tuple in angstroms.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If native handle or output-buffer validation
        fails.
        """
        _initialize_prototypes()

        c_buffer_type = ctypes.c_double * 3
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(3)

        lib().apo_force_manager_get_box_dimensions(c_buffer, c_buffer_len, self.handle)

        return (float(c_buffer[0]), float(c_buffer[1]), float(c_buffer[2]))

    def getKappa(self) -> float:
        """
        @brief Returns the stored Ewald splitting parameter.

        @return Python `float` in inverse angstroms.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        c_kappa: ctypes.c_double = ctypes.c_double()

        lib().apo_force_manager_get_kappa(ctypes.byref(c_kappa), self.handle)

        return float(c_kappa.value)

    def getCutoff(self) -> float:
        """
        @brief Returns the stored pair-list cutoff.

        @return Python `float` in angstroms.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        c_cutoff: ctypes.c_double = ctypes.c_double()

        lib().apo_force_manager_get_cutoff(ctypes.byref(c_cutoff), self.handle)

        return float(c_cutoff.value)

    def getCtonnb(self) -> float:
        """
        @brief Returns the outer switching distance exposed as `ctonnb`.

        @return Python `float` in angstroms.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        c_ctonnb: ctypes.c_double = ctypes.c_double()

        lib().apo_force_manager_get_ctonnb(ctypes.byref(c_ctonnb), self.handle)

        return float(c_ctonnb.value)

    def getCtofnb(self) -> float:
        """
        @brief Returns the inner switching distance exposed as `ctofnb`.

        @return Python `float` in angstroms.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        c_ctofnb: ctypes.c_double = ctypes.c_double()

        lib().apo_force_manager_get_ctofnb(ctypes.byref(c_ctofnb), self.handle)

        return float(c_ctofnb.value)

    def getFFTGrid(self) -> tuple[int, int, int]:
        """
        @brief Returns the three stored PME FFT dimensions.

        @return New `(nfftx, nffty, nfftz)` tuple of Python integers. Before
        explicit configuration or successful native initialization, the base
        defaults are `(-1, -1, -1)`.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If native handle or output-buffer validation
        fails.
        """
        _initialize_prototypes()

        c_buffer_type = ctypes.c_int * 3
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(3)

        lib().apo_force_manager_get_fft_grid(c_buffer, c_buffer_len, self.handle)

        return (int(c_buffer[0]), int(c_buffer[1]), int(c_buffer[2]))

    def getPmeSplineOrder(self) -> int:
        """
        @brief Returns the stored PME interpolation spline order.

        The native value is an integer. The current wrapper implementation
        converts it to a Python `float` despite this method's `int` return
        annotation.

        @return Python `float` containing the dimensionless order.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        c_order: ctypes.c_int = ctypes.c_int()

        lib().apo_force_manager_get_pme_spline_order(ctypes.byref(c_order), self.handle)

        return float(c_order.value)

    def getPeriodicBoundaryCondition(self) -> PeriodicBoundaryCondition:
        """
        @brief Returns the stored periodic boundary condition.

        @return `PeriodicBoundaryCondition` enum value.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native handle is invalid or the native
        PBC value cannot be converted by the C ABI.
        """
        _initialize_prototypes()

        c_pbc = ctypes.c_int()

        lib().apo_force_manager_get_periodic_boundary_condition(
            ctypes.byref(c_pbc), self.handle
        )

        return PeriodicBoundaryCondition(c_pbc.value)

    def getVdwType(self) -> VdwType:
        """
        @brief Returns the stored native van der Waals model.

        @return Corresponding `VdwType` enum value.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native getter fails.
        @throws ValueError If native state contains an integer not declared by
        `VdwType`.
        """
        _initialize_prototypes()

        c_vdw_type: ctypes.c_int = ctypes.c_int()

        lib().apo_force_manager_get_vdw_type(ctypes.byref(c_vdw_type), self.handle)

        return VdwType(c_vdw_type.value)

    def isComposite(self) -> bool:
        """
        @brief Reports whether the wrapped native manager is composite.

        @return Python `bool`. A directly constructed base `ForceManager`
        returns `False`.

        @throws RuntimeError If this wrapper is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool()

        lib().apo_force_manager_is_composite(ctypes.byref(c_flag), self.handle)

        return bool(c_flag.value)
