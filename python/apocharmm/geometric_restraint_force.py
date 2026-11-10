# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

"""
@brief Exposes the owning Python wrapper for MMFP geometric restraints.

`GeometricRestraintForce` copies harmonic plane, cylinder, and sphere terms and
force-wide masses through the C ABI. Every term-addition method accepts exactly
one typed `AtomSelection` or `AtomReference`; raw atom indices are not part of
this interface. Force calculation and native output access are not exposed
directly by this module.

@anchor python_geometric_restraint_force_module
@see geometric_restraint_force
"""

from collections.abc import Sequence
import ctypes

from ._base import _ApoObject
from ._lib import lib
from .atom_reference import AtomReference
from .atom_selection import AtomSelection
from .charmm_psf import CharmmPsf
from .enums import GeometricRestraintActivation, GeometricRestraintSelectionMode
from .error import configure_status_function

_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_geometric_restraint_force_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "GeometricRestraintForce construction",
    )

    lib().apo_geometric_restraint_force_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_geometric_restraint_force_destroy.restype = None

    configure_status_function(
        lib().apo_geometric_restraint_force_add_plane_from_selection,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ],
        "GeometricRestraintForce.addPlane(atoms, origin, direction, "
        "force_constant, offset, activation, selection_mode)",
    )

    configure_status_function(
        lib().apo_geometric_restraint_force_add_plane_from_atom_reference,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ],
        "GeometricRestraintForce.addPlane(atoms, origin, direction, "
        "force_constant, offset, activation, selection_mode)",
    )

    configure_status_function(
        lib().apo_geometric_restraint_force_add_cylinder_from_selection,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ],
        "GeometricRestraintForce.addCylinder(atoms, origin, direction, "
        "force_constant, offset, activation, selection_mode)",
    )

    configure_status_function(
        lib().apo_geometric_restraint_force_add_cylinder_from_atom_reference,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ],
        "GeometricRestraintForce.addCylinder(atoms, origin, direction, "
        "force_constant, offset, activation, selection_mode)",
    )

    configure_status_function(
        lib().apo_geometric_restraint_force_add_sphere_from_selection,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ],
        "GeometricRestraintForce.addSphere(atoms, center, force_constant, "
        "offset, activation, selection_mode)",
    )

    configure_status_function(
        lib().apo_geometric_restraint_force_add_sphere_from_atom_reference,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ],
        "GeometricRestraintForce.addSphere(atoms, center, force_constant, "
        "offset, activation, selection_mode)",
    )

    configure_status_function(
        lib().apo_geometric_restraint_force_set_masses,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "GeometricRestraintForce.setMasses(masses)",
    )

    configure_status_function(
        lib().apo_geometric_restraint_force_reset,
        [ctypes.c_void_p],
        "GeometricRestraintForce.reset()",
    )

    _prototypes_initialized = True

    return


class GeometricRestraintForce(_ApoObject):
    """
    @brief Configures an owned native MMFP geometric restraint.

    One object owns an ordered collection of harmonic plane, infinite-cylinder,
    and sphere terms. Each term uses `INSIDE`, `OUTSIDE`, or `SYMMETRIC`
    activation and either evaluates every selected atom independently
    (`ATOMWISE`) or evaluates one signed mass-weighted center (`RCM`). As in
    CHARMM, separate terms may overlap or repeat and contribute additively in
    insertion order. Harmonic terms are the only supported potential form. See
    the subsystem page for the reaction coordinates and the energy, gradient,
    and virial definitions. Positions, origins, centers, and offsets use
    angstroms; force constants use kilocalories per mole per square angstrom;
    masses normally use atomic mass units.

    This wrapper owns one C handle for the native
    `GeometricRestraintForce<long long int, float>` specialization.
    Construction borrows a `CharmmPsf` only for the call: native code copies its
    atom count and complete mass vector, and this wrapper retains neither the
    Python PSF wrapper nor its native PSF. Closing or collecting that wrapper,
    or later changes to the native PSF, do not affect the restraint.

    Every term-addition method accepts exactly one live `AtomSelection` or
    `AtomReference`, borrowed only for the call. A selection must represent this
    restraint's atom count; its topology is not compared. As in CHARMM, an empty
    selection is valid: it is accepted after parameter validation and adds no
    term. A reference is a one-atom selection that contributes only its
    zero-based index, which must lie in this restraint's force-local atom range;
    its topology is neither compared nor retained. Raw `int` indices, integer
    sequences, sequences of `AtomReference` objects, mixed collections, and
    other objects that merely expose a `handle` attribute are rejected with
    `TypeError`; no collection is converted into an `AtomSelection`.

    Native code copies the selected indices and every parameter, so atom
    objects may be closed as soon as a call returns. Python vectors and masses
    are materialized once, converted with `float()`, and passed as temporary
    contiguous C `double` arrays with explicit lengths. No caller sequence,
    temporary array, or atom object is retained.

    `RCM` terms read one force-wide mass vector. It starts as the PSF masses
    copied at construction, which are the masses CHARMM uses for `RCM` centers.
    `setMasses()` replaces it for current and later `RCM` terms after validating
    every installed `RCM` selection; a rejected replacement leaves the previous
    masses in effect. As in CHARMM, individual masses may be positive, zero, or
    negative, provided every nonempty `RCM` selection keeps a finite nonzero
    total and finite mass fractions. `ATOMWISE` terms ignore masses.

    `reset()` mirrors CHARMM's GEO RESET: it removes every term but keeps the
    active masses, including masses installed by `setMasses()`. It is not output
    clearing: existing force, energy, and virial values are left for the force
    manager's next clearing or evaluation.

    `ForceManager` accepts this restraint only under
    `PeriodicBoundaryCondition.P1` and rejects a later change to `P21` while it
    is subscribed. Under P1, atom positions, `RCM` centers, and reference
    displacements come from raw Cartesian coordinates, without unwrapping or
    minimum-image replacement, exactly as in CHARMM. A native manager
    subscription retains independent shared ownership of the restraint and its
    calculation resources, so term addition, `setMasses()`, and `reset()` remain
    available after subscription and take effect at the next manager evaluation
    without resubscription. Closing a subscribed wrapper destroys its
    identifying C handle but does not remove the native subscription. This class
    does not define the private `ForceManager` subscription callbacks, so
    `ForceManager.subscribe()` rejects it with `TypeError` before any native
    call.

    `close()`, `destroy()`, context-manager exit, and finalization release the
    owned C handle. Explicit closure is idempotent. Methods that access
    `self.handle` after closure raise `RuntimeError` before any native call;
    `default_force_tag` remains available because it is Python-owned text.

    The wrapper and native object provide no internal synchronization. Do not
    overlap configuration, closure, manager subscription changes, or manager
    evaluations of this restraint across threads, and do not close a borrowed
    PSF, selection, or reference while a call is using it.

    @anchor python_geometric_restraint_force
    @see geometric_restraint_force
    """

    _destroy_function_name = "apo_geometric_restraint_force_destroy"

    def __init__(self, psf: CharmmPsf) -> None:
        """
        @brief Constructs an empty restraint from a PSF snapshot.

        The native constructor borrows `psf` only for the call. It copies the
        atom count and complete mass vector, creates a private CUDA stream,
        fixed-point force storage, and an energy-virial object containing the
        named term `mmfp`, and clears every output. The new restraint has no
        terms. The wrapper owns the returned handle, does not retain `psf`, and
        sets `default_force_tag` to `"mmfp"`.

        @param[in] psf Live `CharmmPsf` with a positive atom count and exactly
        one finite mass per atom. As in CHARMM, positive, zero, and negative
        finite masses are accepted.
        @throws TypeError If `psf` is not a `CharmmPsf`.
        @throws RuntimeError If `psf` has been closed or if native construction
        reports success but produces a NULL handle.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the atom count is zero or exceeds the
        fixed-point force-storage representation limit or a mass is NaN or
        infinite; `APO_STATUS_NOT_INITIALIZED` if the PSF atom count is
        negative; `APO_STATUS_CUDA_ERROR` if stream creation, allocation,
        transfer, output clearing, or synchronization fails; or
        `APO_STATUS_RUNTIME_ERROR` if the PSF mass count differs from its atom
        count or native allocation fails.
        @throws MemoryError If Python or `ctypes` bookkeeping cannot allocate
        storage.

        @post On success, this wrapper owns a live native handle,
        `default_force_tag == "mmfp"`, and closing `psf` does not change the
        restraint's atom count or masses.
        @note Construction uses synchronous device transfers and synchronizes
        the private stream, so it can wait for other work on the device.
        """
        super().__init__()

        if not isinstance(psf, CharmmPsf):
            raise TypeError("GeometricRestraintForce expects a CharmmPsf")

        _initialize_prototypes()

        handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_geometric_restraint_force_create(ctypes.byref(handle), psf.handle)

        if handle.value is None:
            raise RuntimeError(
                "apo_geometric_restraint_force_create returned success but produced a NULL handle"
            )

        self._handle = handle
        self._default_force_tag: str = "mmfp"

        return

    @property
    def default_force_tag(self) -> str:
        """
        @brief Returns the default force-manager energy tag.

        The tag equals the native energy-term name. This property reads
        Python-owned text and does not access the native handle, so it remains
        available after `close()`.

        @return The Python string `"mmfp"`.
        """
        return self._default_force_tag

    def addPlane(
        self,
        atoms: AtomSelection | AtomReference,
        origin: Sequence[float],
        direction: Sequence[float],
        force_constant: float,
        offset: float,
        activation: GeometricRestraintActivation,
        selection_mode: GeometricRestraintSelectionMode,
    ) -> None:
        """
        @brief Appends one harmonic plane term.

        The plane passes through `origin`. Native code normalizes `direction`
        with double-precision `std::hypot` to obtain the normal `n`. As in
        CHARMM, positive scaling leaves the normal unchanged and reversal
        changes its orientation; `SYMMETRIC` uses the signed distance
        `dot(n, r - origin)`, and `INSIDE` and `OUTSIDE` use its absolute value,
        with gradient `+n` at exactly zero distance. A zero direction is
        rejected because it cannot define a normal.

        `atoms` is checked first. Each vector is then consumed once as an
        iterable and converted with `float()`, and `force_constant` and `offset`
        are converted with `float()`, before the C call. The C ABI requires
        three values per vector, and native code copies the selected indices and
        every parameter. No atom object, caller sequence, or temporary array is
        retained, and this call performs no CUDA work.

        @param[in] atoms Live `AtomSelection` representing this restraint's
        atom count, or live `AtomReference` whose index lies in `[0, N)` for
        restraint atom count `N`. A valid empty selection adds no term.
        @param[in] origin Iterable of exactly three values accepted by
        `float()`, in `[x, y, z]` order and angstrom units. Converted values
        must be finite.
        @param[in] direction Iterable of exactly three dimensionless values
        accepted by `float()`. Converted values must be finite and have a
        positive finite Euclidean norm.
        @param[in] force_constant Value accepted by `float()`: finite signed
        `K` in kilocalories per mole per square angstrom. Zero is accepted.
        @param[in] offset Value accepted by `float()`: finite signed CHARMM
        `DROFF` in angstroms.
        @param[in] activation Required `GeometricRestraintActivation` member.
        @param[in] selection_mode Required `GeometricRestraintSelectionMode`
        member. A nonempty `RCM` selection requires a finite nonzero total of
        its active masses and finite mass fractions, even when `force_constant`
        is zero.
        @throws TypeError If `atoms` is not an `AtomSelection` or
        `AtomReference`, `activation` or `selection_mode` is not a member of its
        enumeration, a vector is not iterable, or a value cannot be converted by
        `float()`.
        @throws ValueError If Python conversion rejects a value.
        @throws OverflowError If Python conversion overflows.
        @throws RuntimeError If this wrapper or `atoms` has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if a vector does not contain exactly three
        values, a converted value is NaN or infinite, the direction norm is not
        positive and finite, the selection atom count differs, the reference
        index is out of range, or an `RCM` mass requirement is violated; or
        `APO_STATUS_RUNTIME_ERROR` if native term storage cannot grow.

        @post On success with a nonempty selection or a reference, one term is
        appended. On failure, previously installed terms are unchanged.
        """
        _initialize_prototypes()

        if not isinstance(atoms, (AtomSelection, AtomReference)):
            raise TypeError("atoms must be an AtomSelection or AtomReference")

        if not isinstance(activation, GeometricRestraintActivation):
            raise TypeError("activation must be a GeometricRestraintActivation")

        if not isinstance(selection_mode, GeometricRestraintSelectionMode):
            raise TypeError("selection_mode must be a GeometricRestraintSelectionMode")

        origin_values: list[float] = [float(value) for value in origin]
        c_origin_type: type[ctypes.Array[ctypes.c_double]] = ctypes.c_double * len(
            origin_values
        )
        c_origin: ctypes.Array[ctypes.c_double] = c_origin_type(*origin_values)
        c_origin_len: ctypes.c_size_t = ctypes.c_size_t(len(origin_values))

        direction_values: list[float] = [float(value) for value in direction]
        c_direction_type: type[ctypes.Array[ctypes.c_double]] = ctypes.c_double * len(
            direction_values
        )
        c_direction: ctypes.Array[ctypes.c_double] = c_direction_type(*direction_values)
        c_direction_len: ctypes.c_size_t = ctypes.c_size_t(len(direction_values))

        c_force_constant: ctypes.c_double = ctypes.c_double(float(force_constant))
        c_offset: ctypes.c_double = ctypes.c_double(float(offset))
        c_activation: ctypes.c_int = ctypes.c_int(activation.value)
        c_selection_mode: ctypes.c_int = ctypes.c_int(selection_mode.value)

        if isinstance(atoms, AtomSelection):
            lib().apo_geometric_restraint_force_add_plane_from_selection(
                self.handle,
                atoms.handle,
                c_origin,
                c_origin_len,
                c_direction,
                c_direction_len,
                c_force_constant,
                c_offset,
                c_activation,
                c_selection_mode,
            )
        else:
            lib().apo_geometric_restraint_force_add_plane_from_atom_reference(
                self.handle,
                atoms.handle,
                c_origin,
                c_origin_len,
                c_direction,
                c_direction_len,
                c_force_constant,
                c_offset,
                c_activation,
                c_selection_mode,
            )

        return

    def addCylinder(
        self,
        atoms: AtomSelection | AtomReference,
        origin: Sequence[float],
        direction: Sequence[float],
        force_constant: float,
        offset: float,
        activation: GeometricRestraintActivation,
        selection_mode: GeometricRestraintSelectionMode,
    ) -> None:
        """
        @brief Appends one harmonic infinite-cylinder term.

        The axis passes through `origin` along `direction`, which native code
        normalizes with double-precision `std::hypot`. As in CHARMM, the
        reaction coordinate is the raw Cartesian perpendicular distance to that
        axis, positive scaling or reversal of `direction` leaves the geometry
        unchanged, and a computed squared radial distance at or below `1e-10`
        square angstroms suppresses the whole evaluation before activation. A
        zero direction is rejected because it cannot define an axis.

        Input checking, conversion, copying, and retention match `addPlane()`.
        This call performs no CUDA work.

        @param[in] atoms Live `AtomSelection` representing this restraint's
        atom count, or live `AtomReference` whose index lies in `[0, N)` for
        restraint atom count `N`. A valid empty selection adds no term.
        @param[in] origin Iterable of exactly three axis-reference values
        accepted by `float()`, in `[x, y, z]` order and angstrom units.
        Converted values must be finite.
        @param[in] direction Iterable of exactly three dimensionless axis
        values accepted by `float()`. Converted values must be finite and have a
        positive finite Euclidean norm.
        @param[in] force_constant Value accepted by `float()`: finite signed
        `K` in kilocalories per mole per square angstrom. Zero is accepted.
        @param[in] offset Value accepted by `float()`: finite signed CHARMM
        `DROFF` in angstroms.
        @param[in] activation Required `GeometricRestraintActivation` member.
        @param[in] selection_mode Required `GeometricRestraintSelectionMode`
        member. A nonempty `RCM` selection requires a finite nonzero total of
        its active masses and finite mass fractions, even when `force_constant`
        is zero.
        @throws TypeError If `atoms` is not an `AtomSelection` or
        `AtomReference`, `activation` or `selection_mode` is not a member of its
        enumeration, a vector is not iterable, or a value cannot be converted by
        `float()`.
        @throws ValueError If Python conversion rejects a value.
        @throws OverflowError If Python conversion overflows.
        @throws RuntimeError If this wrapper or `atoms` has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if a vector does not contain exactly three
        values, a converted value is NaN or infinite, the direction norm is not
        positive and finite, the selection atom count differs, the reference
        index is out of range, or an `RCM` mass requirement is violated; or
        `APO_STATUS_RUNTIME_ERROR` if native term storage cannot grow.

        @post On success with a nonempty selection or a reference, one term is
        appended. On failure, previously installed terms are unchanged.
        """
        _initialize_prototypes()

        if not isinstance(atoms, (AtomSelection, AtomReference)):
            raise TypeError("atoms must be an AtomSelection or AtomReference")

        if not isinstance(activation, GeometricRestraintActivation):
            raise TypeError("activation must be a GeometricRestraintActivation")

        if not isinstance(selection_mode, GeometricRestraintSelectionMode):
            raise TypeError("selection_mode must be a GeometricRestraintSelectionMode")

        origin_values: list[float] = [float(value) for value in origin]
        c_origin_type: type[ctypes.Array[ctypes.c_double]] = ctypes.c_double * len(
            origin_values
        )
        c_origin: ctypes.Array[ctypes.c_double] = c_origin_type(*origin_values)
        c_origin_len: ctypes.c_size_t = ctypes.c_size_t(len(origin_values))

        direction_values: list[float] = [float(value) for value in direction]
        c_direction_type: type[ctypes.Array[ctypes.c_double]] = ctypes.c_double * len(
            direction_values
        )
        c_direction: ctypes.Array[ctypes.c_double] = c_direction_type(*direction_values)
        c_direction_len: ctypes.c_size_t = ctypes.c_size_t(len(direction_values))

        c_force_constant: ctypes.c_double = ctypes.c_double(float(force_constant))
        c_offset: ctypes.c_double = ctypes.c_double(float(offset))
        c_activation: ctypes.c_int = ctypes.c_int(activation.value)
        c_selection_mode: ctypes.c_int = ctypes.c_int(selection_mode.value)

        if isinstance(atoms, AtomSelection):
            lib().apo_geometric_restraint_force_add_cylinder_from_selection(
                self.handle,
                atoms.handle,
                c_origin,
                c_origin_len,
                c_direction,
                c_direction_len,
                c_force_constant,
                c_offset,
                c_activation,
                c_selection_mode,
            )
        else:
            lib().apo_geometric_restraint_force_add_cylinder_from_atom_reference(
                self.handle,
                atoms.handle,
                c_origin,
                c_origin_len,
                c_direction,
                c_direction_len,
                c_force_constant,
                c_offset,
                c_activation,
                c_selection_mode,
            )

        return

    def addSphere(
        self,
        atoms: AtomSelection | AtomReference,
        center: Sequence[float],
        force_constant: float,
        offset: float,
        activation: GeometricRestraintActivation,
        selection_mode: GeometricRestraintSelectionMode,
    ) -> None:
        """
        @brief Appends one harmonic sphere term.

        As in CHARMM, the reaction coordinate is the raw Cartesian distance to
        `center`, without periodic imaging, and a computed squared distance at
        or below `1e-10` square angstroms suppresses the whole evaluation
        before activation.

        Input checking, conversion, copying, and retention match `addPlane()`.
        A sphere has no direction. This call performs no CUDA work.

        @param[in] atoms Live `AtomSelection` representing this restraint's
        atom count, or live `AtomReference` whose index lies in `[0, N)` for
        restraint atom count `N`. A valid empty selection adds no term.
        @param[in] center Iterable of exactly three values accepted by
        `float()`, in `[x, y, z]` order and angstrom units. Converted values
        must be finite.
        @param[in] force_constant Value accepted by `float()`: finite signed
        `K` in kilocalories per mole per square angstrom. Zero is accepted.
        @param[in] offset Value accepted by `float()`: finite signed CHARMM
        `DROFF` in angstroms.
        @param[in] activation Required `GeometricRestraintActivation` member.
        @param[in] selection_mode Required `GeometricRestraintSelectionMode`
        member. A nonempty `RCM` selection requires a finite nonzero total of
        its active masses and finite mass fractions, even when `force_constant`
        is zero.
        @throws TypeError If `atoms` is not an `AtomSelection` or
        `AtomReference`, `activation` or `selection_mode` is not a member of its
        enumeration, `center` is not iterable, or a value cannot be converted by
        `float()`.
        @throws ValueError If Python conversion rejects a value.
        @throws OverflowError If Python conversion overflows.
        @throws RuntimeError If this wrapper or `atoms` has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if `center` does not contain exactly three
        values, a converted value is NaN or infinite, the selection atom count
        differs, the reference index is out of range, or an `RCM` mass
        requirement is violated; or `APO_STATUS_RUNTIME_ERROR` if native term
        storage cannot grow.

        @post On success with a nonempty selection or a reference, one term is
        appended. On failure, previously installed terms are unchanged.
        @note Native diagnostics name a nonfinite center component as an origin
        component.
        """
        _initialize_prototypes()

        if not isinstance(atoms, (AtomSelection, AtomReference)):
            raise TypeError("atoms must be an AtomSelection or AtomReference")

        if not isinstance(activation, GeometricRestraintActivation):
            raise TypeError("activation must be a GeometricRestraintActivation")

        if not isinstance(selection_mode, GeometricRestraintSelectionMode):
            raise TypeError("selection_mode must be a GeometricRestraintSelectionMode")

        center_values: list[float] = [float(value) for value in center]
        c_center_type: type[ctypes.Array[ctypes.c_double]] = ctypes.c_double * len(
            center_values
        )
        c_center: ctypes.Array[ctypes.c_double] = c_center_type(*center_values)
        c_center_len: ctypes.c_size_t = ctypes.c_size_t(len(center_values))

        c_force_constant: ctypes.c_double = ctypes.c_double(float(force_constant))
        c_offset: ctypes.c_double = ctypes.c_double(float(offset))
        c_activation: ctypes.c_int = ctypes.c_int(activation.value)
        c_selection_mode: ctypes.c_int = ctypes.c_int(selection_mode.value)

        if isinstance(atoms, AtomSelection):
            lib().apo_geometric_restraint_force_add_sphere_from_selection(
                self.handle,
                atoms.handle,
                c_center,
                c_center_len,
                c_force_constant,
                c_offset,
                c_activation,
                c_selection_mode,
            )
        else:
            lib().apo_geometric_restraint_force_add_sphere_from_atom_reference(
                self.handle,
                atoms.handle,
                c_center,
                c_center_len,
                c_force_constant,
                c_offset,
                c_activation,
                c_selection_mode,
            )

        return

    def setMasses(self, masses: Sequence[float]) -> None:
        """
        @brief Replaces the force-wide mass vector used by every RCM term.

        The method consumes `masses` once as an iterable, converts every element
        with `float()`, materializes one new Python list, and copies that list
        into a temporary contiguous C `double` array passed with an explicit
        length. Native code copies the values again, validates the complete
        vector and every installed `RCM` selection, prepares independent host
        and device replacement storage, and commits only after every check
        passes. No caller sequence or buffer is retained.

        As in CHARMM, individual masses may be positive, zero, or negative, and
        selected totals may be tiny, huge, negative, or close to zero, provided
        every installed `RCM` selection keeps a finite nonzero total and finite
        mass fractions. A replacement that would make a selected total zero is
        rejected, because the mass-weighted center is then undefined. The masses
        apply to current and later `RCM` terms, are ignored by `ATOMWISE` terms,
        and are preserved by `reset()`.

        @param[in] masses Iterable producing exactly one value per atom, in
        atom-index order and normally in atomic mass units. Every element must
        be accepted by `float()`, and every converted value must be finite.
        @throws TypeError If `masses` is not iterable or an element cannot be
        converted with `float()`.
        @throws ValueError If element conversion rejects a value.
        @throws OverflowError If element conversion overflows.
        @throws MemoryError If the temporary list or C array cannot be
        allocated.
        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError With native status
        `APO_STATUS_INVALID_ARGUMENT` if the length differs from the restraint
        atom count, a converted mass is NaN or infinite, or an installed `RCM`
        total or mass fraction would become invalid; `APO_STATUS_CUDA_ERROR` if
        stream synchronization, replacement allocation, or transfer fails; or
        `APO_STATUS_RUNTIME_ERROR` if native allocation fails.

        @post On success, every current and later `RCM` term uses the copied
        masses. On failure, the previously active masses, their derived totals,
        and the installed terms remain in effect. Python conversion failures
        occur before the C call.
        @note Native replacement synchronizes the restraint's private CUDA
        stream and uses synchronous device transfers, so it can wait for other
        work on the device.
        """
        _initialize_prototypes()

        mass_values: list[float] = [float(value) for value in masses]
        c_masses_type: type[ctypes.Array[ctypes.c_double]] = ctypes.c_double * len(
            mass_values
        )
        c_masses: ctypes.Array[ctypes.c_double] = c_masses_type(*mass_values)
        c_masses_len: ctypes.c_size_t = ctypes.c_size_t(len(mass_values))

        lib().apo_geometric_restraint_force_set_masses(
            self.handle, c_masses, c_masses_len
        )

        return

    def reset(self) -> None:
        """
        @brief Removes every geometric term while preserving masses and output.

        Like CHARMM's GEO RESET, this method removes every installed term, may
        be repeated while no terms remain, allows new terms afterward, and
        leaves masses unchanged. Native host term storage is emptied, and device
        mirrors are rebuilt before the next evaluation. The active masses,
        including masses installed by `setMasses()`, the stored box, the private
        CUDA stream, and existing force, energy, and virial values are
        preserved. Reset is not output clearing and performs no CUDA work.

        @throws RuntimeError If this wrapper has been closed.
        @throws ApoCharmmError With native status `APO_STATUS_RUNTIME_ERROR` if
        replacement empty term storage cannot be allocated.

        @post On success, the restraint owns no terms and its active masses are
        unchanged. On failure, installed terms are unchanged.
        """
        _initialize_prototypes()

        lib().apo_geometric_restraint_force_reset(self.handle)

        return
