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

from ._base import _ApoObject
from ._lib import encode_path, lib
from ._types import FilePath
from ._validation import require_c_uint64
from .enums import PeriodicBoundaryCondition, VdwType
from .error import configure_status_function

from .charmm_crd import CharmmCrd
from .charmm_parameters import CharmmParameters
from .charmm_psf import CharmmPsf
from .force_manager import ForceManager


def _flatten_rows(
    values: Sequence[Sequence[float]], width: int, argument_name: str
) -> list[float]:
    """
    @brief Converts fixed-width nested rows to one flat float list.

    Each scalar is converted with `float()` before row-width validation.

    @param[in] values Borrowed outer sequence of borrowed row sequences.
    @param[in] width Required number of scalar values per row.
    @param[in] argument_name Name used in a row-width diagnostic.
    @return Newly allocated row-major list.

    @throws TypeError If a row is not iterable or a scalar cannot be converted
    with `float()`.
    @throws ValueError If scalar conversion fails or a converted row does not
    contain exactly `width` elements.
    """
    flattened: list[float] = []

    for index, row in enumerate(values):
        row_values: list[float] = [float(value) for value in row]
        if len(row_values) != width:
            raise ValueError(
                f"{argument_name}[{index}] must contain exactly {width} values; observed {len(row_values)}"
            )
        flattened.extend(row_values)

    return flattened


_prototypes_initialized: bool = False


def _initialize_prototypes() -> None:
    global _prototypes_initialized

    if _prototypes_initialized:
        return

    configure_status_function(
        lib().apo_charmm_context_create,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "CharmmContext construction from ForceManager",
    )

    configure_status_function(
        lib().apo_charmm_context_create_from_psf_parameters,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p],
        "CharmmContext construction from CharmmPsf/CharmmParameters",
    )

    lib().apo_charmm_context_destroy.argtypes = [ctypes.c_void_p]
    lib().apo_charmm_context_destroy.restype = None

    configure_status_function(
        lib().apo_charmm_context_set_prm,
        [ctypes.c_void_p, ctypes.c_void_p],
        "CharmmContext.setPrm(parameters)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_psf,
        [ctypes.c_void_p, ctypes.c_void_p],
        "CharmmContext.setPsf(psf)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_force_manager,
        [ctypes.c_void_p, ctypes.c_void_p],
        "CharmmContext.setForceManager(force_manager)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_coordinates_charges,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CharmmContext.setCoordinatesCharges(coordinates_charges)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_coordinates_from_array,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CharmmContext.setCoordinates(coordinates)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_coordinates,
        [ctypes.c_void_p, ctypes.c_void_p],
        "CharmmContext.setCoordinates(crd)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_charges,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CharmmContext.setCharges(charges)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_velocities_inverse_masses,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CharmmContext.setVelocitiesInverseMasses(velocities_inverse_masses)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_velocities,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CharmmContext.setVelocities(velocities)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_velocities_from_charmm_velocity_file,
        [ctypes.c_void_p, ctypes.c_char_p],
        "CharmmContext.setVelocitiesFromCHARMMVelocityFile(file_name)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_masses,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CharmmContext.setMasses(masses)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_temperature,
        [ctypes.c_void_p, ctypes.c_double],
        "CharmmContext.setTemperature(temperature)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_periodic_boundary_condition,
        [ctypes.c_void_p, ctypes.c_int],
        "CharmmContext.setPeriodicBoundaryCondition(pbc)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_box_dimensions,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
        "CharmmContext.setBoxDimensions(box_dimensions)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_random_seed,
        [ctypes.c_void_p, ctypes.c_uint64],
        "CharmmContext.setRandomSeed(seed)",
    )

    configure_status_function(
        lib().apo_charmm_context_use_holonomic_constraints,
        [ctypes.c_void_p, ctypes.c_bool],
        "CharmmContext.useHolonomicConstraints(flag)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_kappa,
        [ctypes.c_void_p, ctypes.c_double],
        "CharmmContext.setKappa(kappa)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_cutoff,
        [ctypes.c_void_p, ctypes.c_double],
        "CharmmContext.setCutoff(cutoff)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_ctonnb,
        [ctypes.c_void_p, ctypes.c_double],
        "CharmmContext.setCtonnb(ctonnb)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_ctofnb,
        [ctypes.c_void_p, ctypes.c_double],
        "CharmmContext.setCtofnb(ctofnb)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_fft_grid,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t],
        "CharmmContext.setFFTGrid(grid)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_pme_spline_order,
        [ctypes.c_void_p, ctypes.c_int],
        "CharmmContext.setPmeSplineOrder(order)",
    )

    configure_status_function(
        lib().apo_charmm_context_set_vdw_type,
        [ctypes.c_void_p, ctypes.c_int],
        "CharmmContext.setVdwType(vdw_type)",
    )

    configure_status_function(
        lib().apo_charmm_context_get_num_atoms,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "CharmmContext.getNumAtoms()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_num_degrees_of_freedom,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "CharmmContext.getNumDegreesOfFreedom()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_coordinates_charges,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_void_p],
        "CharmmContext.getCoordinatesCharges()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_velocity_mass,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_void_p],
        "CharmmContext.getVelocityMass()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_periodic_boundary_condition,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "CharmmContext.getPeriodicBoundaryCondition()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_box_dimensions,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_void_p],
        "CharmmContext.getBoxDimensions()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_random_seed,
        [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p],
        "CharmmContext.getRandomSeed()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_volume,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CharmmContext.getVolume()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_kappa,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CharmmContext.getKappa()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_cutoff,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CharmmContext.getCutoff()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_ctonnb,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CharmmContext.getCtonnb()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_ctofnb,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CharmmContext.getCtofnb()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_fft_grid,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_void_p],
        "CharmmContext.getFFTGrid()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_pme_spline_order,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "CharmmContext.getPmeSplineOrder()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_vdw_type,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],
        "CharmmContext.getVdwType()",
    )

    configure_status_function(
        lib().apo_charmm_context_get_force_manager,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "CharmmContext.getForceManager()",
    )

    configure_status_function(
        lib().apo_charmm_context_assign_velocities_at_temperature,
        [ctypes.c_void_p, ctypes.c_double],
        "CharmmContext.assignVelocitiesAtTemperature(temperature)",
    )

    configure_status_function(
        lib().apo_charmm_context_compute_temperature,
        [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p],
        "CharmmContext.computeTemperature()",
    )

    configure_status_function(
        lib().apo_charmm_context_calculate_potential_energy,
        [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool],
        "CharmmContext.calculatePotentialEnergy(reset, print_energy)",
    )

    _prototypes_initialized = True

    return


class CharmmContext(_ApoObject):
    """
    @anchor python_charmm_context
    @brief Represents mutable molecular state for apoCHARMM calculations.

    Construct the wrapper from a `ForceManager`, or from a `CharmmPsf` together
    with `CharmmParameters`. The wrapper owns its C handle and retains Python
    references to collaborators whose native objects participate in the
    context.

    Array setters copy Python values into temporary C buffers. Array getters
    return new Python lists or tuples; they do not expose native storage.

    `close()` releases the owned context handle and is idempotent. The wrapper
    supports the context-manager protocol. Using a method that accesses
    `self.handle` after closure raises `RuntimeError`. Native nonzero statuses
    are raised as `ApoCharmmError`.

    The wrapper and its native object provide no internal thread
    synchronization.

    @see charmm_context
    """

    _destroy_function_name = "apo_charmm_context_destroy"

    def __init__(
        self,
        force_manager_or_psf: ForceManager | CharmmPsf,
        parameters: CharmmParameters | None = None,
    ) -> None:
        """
        @brief Constructs a Python CharmmContext wrapper.

        Pass either a live `ForceManager` and leave `parameters` as `None`, or
        pass a live `CharmmPsf` together with a live `CharmmParameters`.
        Collaborator wrappers are retained to preserve their Python and native
        lifetimes.

        @param[in] force_manager_or_psf `ForceManager` or `CharmmPsf` selecting
        the construction path.
        @param[in] parameters `CharmmParameters` required with `CharmmPsf`, or
        `None` with `ForceManager`.

        @throws TypeError If the argument combination does not match either
        supported construction path.
        @throws RuntimeError If a supplied wrapper is closed, or native
        construction reports success but produces a NULL handle.
        @throws ApoCharmmError If native validation, GPU setup, allocation, or
        force-manager initialization fails.
        """
        _initialize_prototypes()
        super().__init__()

        handle: ctypes.c_void_p = ctypes.c_void_p()

        if isinstance(force_manager_or_psf, ForceManager) and parameters is None:
            lib().apo_charmm_context_create(
                ctypes.byref(handle), force_manager_or_psf.handle
            )
            self._force_manager: ForceManager | None = force_manager_or_psf
            self._psf: CharmmPsf | None = None
            self._parameters: CharmmParameters | None = None

        elif isinstance(force_manager_or_psf, CharmmPsf) and isinstance(
            parameters, CharmmParameters
        ):
            lib().apo_charmm_context_create_from_psf_parameters(
                ctypes.byref(handle), force_manager_or_psf.handle, parameters.handle
            )
            self._force_manager = None
            self._psf = force_manager_or_psf
            self._parameters = parameters

        else:
            raise TypeError(
                "CharmmContext expects either ForceManager or (CharmmPsf, CharmmParameters)"
            )

        if handle.value is None:
            raise RuntimeError(
                "CharmmContext construction returned success but produced a NULL handle"
            )

        self._handle = handle

        return

    def setPrm(self, parameters: CharmmParameters) -> None:
        """
        @brief Sets the CHARMM parameter set.

        The wrapper requires a `CharmmParameters`, passes its borrowed handle to
        the C ABI, and retains the Python object after success.

        @param[in] parameters Live `CharmmParameters` wrapper.

        @throws TypeError If `parameters` is not a `CharmmParameters`.
        @throws RuntimeError If this context or `parameters` is closed.
        @throws ApoCharmmError If native synchronization or initialization
        fails.
        """
        _initialize_prototypes()

        if not isinstance(parameters, CharmmParameters):
            raise TypeError("CharmmContext.setPrm expects CharmmParameters")

        lib().apo_charmm_context_set_prm(self.handle, parameters.handle)
        self._parameters = parameters

        return

    def setPsf(self, psf: CharmmPsf) -> None:
        """
        @brief Sets the PSF and imports its atom charges and masses.

        The wrapper requires a `CharmmPsf`, passes its borrowed handle to the C
        ABI, and retains the Python object after success.

        @param[in] psf Live `CharmmPsf` wrapper.

        @throws TypeError If `psf` is not a `CharmmPsf`.
        @throws RuntimeError If this context or `psf` is closed.
        @throws ApoCharmmError If the atom count conflicts, storage or transfer
        fails, or force-manager initialization fails.
        """
        _initialize_prototypes()

        if not isinstance(psf, CharmmPsf):
            raise TypeError("CharmmContext.setPsf expects CharmmPsf")

        lib().apo_charmm_context_set_psf(self.handle, psf.handle)
        self._psf = psf

        return

    def setForceManager(self, force_manager: ForceManager) -> None:
        """
        @brief Sets the ForceManager associated with the context.

        The wrapper retains `force_manager` after a successful native state
        reconciliation and clears retained PSF and parameter wrappers because
        the new manager becomes their native source.

        @param[in] force_manager Live `ForceManager` wrapper.

        @throws TypeError If `force_manager` is not a `ForceManager`.
        @throws RuntimeError If this context or `force_manager` is closed.
        @throws ApoCharmmError If native state reconciliation or initialization
        fails.
        """
        _initialize_prototypes()

        if not isinstance(force_manager, ForceManager):
            raise TypeError("CharmmContext.setForceManager expects ForceManager")

        lib().apo_charmm_context_set_force_manager(self.handle, force_manager.handle)
        self._force_manager = force_manager
        self._psf = None
        self._parameters = None

        return

    def setCoordinatesCharges(
        self, coordinates_charges: Sequence[Sequence[float]]
    ) -> None:
        """
        @brief Sets coordinates and charges from nested Python rows.

        `coordinates_charges` must contain one row per atom and exactly four
        values per row in `[x, y, z, charge]` order. Coordinates use angstroms
        and charges use elementary-charge units. Every scalar is converted with
        `float()`, flattened into a temporary `ctypes` buffer, and copied by the
        native context.

        @param[in] coordinates_charges Sequence of fixed-width row sequences.

        @throws TypeError If a row is not iterable or a scalar cannot be
        converted to `float`.
        @throws ValueError If scalar conversion fails or a row does not contain
        exactly four values.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the atom count is unset, the row count is
        wrong, or native transfer or neighbor-list work fails.
        """
        _initialize_prototypes()

        values: list[float] = _flatten_rows(
            coordinates_charges, 4, "coordinates_charges"
        )
        c_buffer_type = ctypes.c_double * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_coordinates_charges(
            self.handle, c_buffer, c_buffer_len
        )

        return

    def setCoordinates(
        self, coordinates: CharmmCrd | Sequence[Sequence[float]]
    ) -> None:
        """
        @brief Sets coordinates from a CharmmCrd or nested Python rows.

        A `CharmmCrd` is borrowed for the native call and is not retained.
        Otherwise, `coordinates` must contain one `[x, y, z]` row per atom.
        Scalars are converted with `float()` and copied through a temporary C
        buffer. Coordinates use angstroms and existing charges are preserved.

        @param[in] coordinates Live `CharmmCrd` wrapper or sequence of
        three-value row sequences.

        @throws TypeError If a row is not iterable, a scalar cannot be
        converted, or a closed `CharmmCrd` is accessed.
        @throws ValueError If scalar conversion fails or a row does not contain
        exactly three values.
        @throws RuntimeError If this context or a supplied `CharmmCrd` is
        closed.
        @throws ApoCharmmError If the atom count is unset, the coordinate count
        is wrong, or native transfer or neighbor-list work fails.
        """
        _initialize_prototypes()

        if isinstance(coordinates, CharmmCrd):
            lib().apo_charmm_context_set_coordinates(self.handle, coordinates.handle)
            return

        values: list[float] = _flatten_rows(coordinates, 3, "coordinates")
        c_buffer_type = ctypes.c_double * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_coordinates_from_array(
            self.handle, c_buffer, c_buffer_len
        )

        return

    def setCharges(self, charges: Sequence[float]) -> None:
        """
        @brief Sets per-atom charges while preserving coordinates.

        Each scalar is converted with `float()`, copied to a temporary C buffer,
        and then copied by the native context. Charges use elementary-charge
        units.

        @param[in] charges Sequence containing exactly one value per atom.

        @throws TypeError If `charges` is not iterable or a value cannot be
        converted with `float()`.
        @throws ValueError If scalar conversion raises `ValueError`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the atom count is unset, the sequence length
        is wrong, or native transfer fails.
        """
        _initialize_prototypes()

        values: list[float] = [float(value) for value in charges]
        c_buffer_type = ctypes.c_double * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_charges(self.handle, c_buffer, c_buffer_len)

        return

    def setVelocitiesInverseMasses(
        self, velocities_inverse_masses: Sequence[Sequence[float]]
    ) -> None:
        """
        @brief Sets velocities and inverse masses from nested rows.

        Every row must be `[vx, vy, vz, inverse_mass]`. Velocity components use
        angstroms per AKMA time unit and inverse mass uses reciprocal atomic mass
        units. Scalars are converted with `float()` and copied through a
        temporary C buffer.

        @param[in] velocities_inverse_masses Sequence of four-value rows.

        @throws TypeError If a row is not iterable or a scalar cannot be
        converted to `float`.
        @throws ValueError If scalar conversion fails or a row does not contain
        exactly four values.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the atom count is unset, the row count is
        wrong, or native transfer fails.
        """
        _initialize_prototypes()

        values: list[float] = _flatten_rows(
            velocities_inverse_masses, 4, "velocities_inverse_masses"
        )
        c_buffer_type = ctypes.c_double * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_velocities_inverse_masses(
            self.handle, c_buffer, c_buffer_len
        )

        return

    def setVelocities(self, velocities: Sequence[Sequence[float]]) -> None:
        """
        @brief Sets velocity components while preserving inverse masses.

        Every row must be `[vx, vy, vz]` in angstroms per AKMA time unit.
        Scalars are converted with `float()` and copied through a temporary C
        buffer.

        @param[in] velocities Sequence containing one three-value row per atom.

        @throws TypeError If a row is not iterable or a scalar cannot be
        converted to `float`.
        @throws ValueError If scalar conversion fails or a row does not contain
        exactly three values.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the atom count is unset, the row count is
        wrong, or native transfer fails.
        """
        _initialize_prototypes()

        values: list[float] = _flatten_rows(velocities, 3, "velocities")
        c_buffer_type = ctypes.c_double * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_velocities(self.handle, c_buffer, c_buffer_len)

        return

    def setVelocitiesFromCHARMMVelocityFile(self, path: FilePath) -> None:
        """
        @brief Loads velocities from a CHARMM velocity file.

        `path` is encoded with `os.fsencode()` through `encode_path()` and is
        borrowed by the C ABI for the duration of the call. The native parser
        requires a matching atom count and one parseable velocity record per
        atom.

        @param[in] path `str`, `bytes`, or `os.PathLike` path accepted by
        `os.fsencode()`.

        @throws TypeError If `path` cannot be encoded as a filesystem path.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the atom count is unset, the path is empty,
        the file cannot be opened or parsed, the atom count differs, or native
        transfer fails.
        """
        _initialize_prototypes()

        encoded_path: bytes = encode_path(path)
        c_path: ctypes.c_char_p = ctypes.c_char_p(encoded_path)

        lib().apo_charmm_context_set_velocities_from_charmm_velocity_file(
            self.handle, c_path
        )

        return

    def setMasses(self, masses: Sequence[float]) -> None:
        """
        @brief Sets masses and stores native inverse masses.

        Each scalar is converted with `float()` and interpreted as an atomic
        mass-unit value. The native context stores its reciprocal and preserves
        existing velocity components.

        @param[in] masses Sequence containing exactly one mass per atom.

        @throws TypeError If `masses` is not iterable or a value cannot be
        converted to `float`.
        @throws ValueError If scalar conversion raises `ValueError`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the atom count is unset, the sequence length
        is wrong, or native transfer fails.

        @warning The current native implementation does not reject zero,
        negative, infinite, or NaN masses before division.
        """
        _initialize_prototypes()

        values: list[float] = [float(value) for value in masses]
        c_buffer_type = ctypes.c_double * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_masses(self.handle, c_buffer, c_buffer_len)

        return

    def setTemperature(self, temperature: float) -> None:
        """
        @brief Sets the stored target temperature.

        This method does not generate or rescale velocities.

        @param[in] temperature Python `int` or `float` accepted by
        `ctypes.c_double`, in kelvin. The native value must be finite and
        non-negative.

        @throws TypeError If `temperature` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the native value is non-finite or negative.
        """
        _initialize_prototypes()

        c_temperature: ctypes.c_double = ctypes.c_double(temperature)

        lib().apo_charmm_context_set_temperature(self.handle, c_temperature)

        return

    def setPeriodicBoundaryCondition(
        self, pbc: PeriodicBoundaryCondition | int
    ) -> None:
        """
        @brief Sets the periodic boundary condition.

        `pbc` is normalized through `PeriodicBoundaryCondition` before its
        integer value is passed to the C ABI.

        @param[in] pbc `PeriodicBoundaryCondition` or matching integer value
        `0`, `1`, or `2`.

        @throws TypeError If `pbc` cannot be interpreted by the enumeration.
        @throws ValueError If the integer is not a declared enum value.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If native propagation, initialization, imaging,
        or neighbor-list rebuilding fails.
        """
        _initialize_prototypes()

        try:
            pbc_value: PeriodicBoundaryCondition = PeriodicBoundaryCondition(pbc)
        except ValueError as exc:
            raise ValueError(f"invalid pbc: {pbc!r}") from exc

        c_pbc: ctypes.c_int = ctypes.c_int(int(pbc_value))

        lib().apo_charmm_context_set_periodic_boundary_condition(self.handle, c_pbc)

        return

    def setBoxDimensions(self, box_dimensions: Sequence[float]) -> None:
        """
        @brief Sets orthorhombic box dimensions.

        `box_dimensions` is converted to a temporary contiguous `double`
        buffer. It must contain exactly `[x, y, z]` lengths in angstroms.

        @param[in] box_dimensions Sequence of exactly three numeric values.

        @throws TypeError If the input is not iterable or a value cannot be
        converted with `float()`.
        @throws ValueError If scalar conversion raises `ValueError`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the length is not three, a dimension is not
        positive, attached force configuration is invalid, or native
        initialization or neighbor-list work fails.
        """
        _initialize_prototypes()

        values: list[float] = [float(value) for value in box_dimensions]

        c_buffer_type = ctypes.c_double * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_box_dimensions(self.handle, c_buffer, c_buffer_len)

        return

    def setRandomSeed(self, seed: int) -> None:
        """
        @brief Sets the random seed used for velocity assignment.

        @param[in] seed Python `int` in the inclusive range
        `[0, 2**64 - 1]`.

        @throws TypeError If `seed` is not comparable with integers or cannot
        initialize `ctypes.c_uint64`.
        @throws ValueError If `seed` is outside the unsigned 64-bit range.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the native context rejects the update.
        """
        _initialize_prototypes()

        seed_value: int = require_c_uint64(seed, "seed", allow_bool=True)
        c_seed: ctypes.c_uint64 = ctypes.c_uint64(seed_value)

        lib().apo_charmm_context_set_random_seed(self.handle, c_seed)

        return

    def useHolonomicConstraints(self, flag: bool) -> None:
        """
        @brief Selects constrained degree-of-freedom accounting.

        The value is converted with `ctypes.c_bool`. The native context
        recomputes its degree-of-freedom count but does not execute a
        constraint solver.

        @param[in] flag Boolean selection.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the native context has no PSF or no force
        manager.
        """
        _initialize_prototypes()

        c_flag: ctypes.c_bool = ctypes.c_bool(flag)

        lib().apo_charmm_context_use_holonomic_constraints(self.handle, c_flag)

        return

    def setKappa(self, kappa: float) -> None:
        """
        @brief Sets the Ewald splitting parameter.

        @param[in] kappa Python `int` or `float` accepted by
        `ctypes.c_double`, in inverse angstroms. The native float-converted
        value must be finite and non-negative.

        @throws TypeError If `kappa` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or native
        validation fails.

        @warning Configure this value before force-manager initialization.
        """
        _initialize_prototypes()

        c_kappa: ctypes.c_double = ctypes.c_double(kappa)

        lib().apo_charmm_context_set_kappa(self.handle, c_kappa)

        return

    def setCutoff(self, cutoff: float) -> None:
        """
        @brief Sets the direct-space cutoff.

        @param[in] cutoff Python `int` or `float` accepted by
        `ctypes.c_double`, in angstroms. The native float-converted value must
        be finite and positive.

        @throws TypeError If `cutoff` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or native
        validation fails.

        @warning Configure this value before force-manager initialization.
        """
        _initialize_prototypes()

        c_cutoff: ctypes.c_double = ctypes.c_double(cutoff)

        lib().apo_charmm_context_set_cutoff(self.handle, c_cutoff)

        return

    def setCtonnb(self, ctonnb: float) -> None:
        """
        @brief Sets the nonbonded distance exposed as `ctonnb`.

        @param[in] ctonnb Python `int` or `float` accepted by
        `ctypes.c_double`, in angstroms. The native float-converted value must
        be finite and positive.

        @throws TypeError If `ctonnb` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or native
        validation fails.

        @warning The current backend forwards this value as `roff`. Configure
        it before force-manager initialization.
        """
        _initialize_prototypes()

        c_ctonnb: ctypes.c_double = ctypes.c_double(ctonnb)

        lib().apo_charmm_context_set_ctonnb(self.handle, c_ctonnb)

        return

    def setCtofnb(self, ctofnb: float) -> None:
        """
        @brief Sets the nonbonded distance exposed as `ctofnb`.

        @param[in] ctofnb Python `int` or `float` accepted by
        `ctypes.c_double`, in angstroms. The native float-converted value must
        be finite and positive.

        @throws TypeError If `ctofnb` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or native
        validation fails.

        @warning The current backend forwards this value as `ron`. Configure
        it before force-manager initialization.
        """
        _initialize_prototypes()

        c_ctofnb: ctypes.c_double = ctypes.c_double(ctofnb)

        lib().apo_charmm_context_set_ctofnb(self.handle, c_ctofnb)

        return

    def setFFTGrid(self, grid: Sequence[int]) -> None:
        """
        @brief Sets the three-dimensional PME FFT grid.

        Each element is converted with `int()` and copied to a temporary C
        buffer. The sequence must contain exactly three positive dimensions in
        X, Y, Z order.

        @param[in] grid Sequence of three integer-convertible values.

        @throws TypeError If `grid` is not iterable or an element cannot be
        converted with `int()`.
        @throws ValueError If integer conversion fails.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the length is not three, a dimension is not
        positive, or no force manager is attached.

        @warning Configure the grid before force-manager initialization.
        """
        _initialize_prototypes()

        values: list[int] = [int(value) for value in grid]
        c_buffer_type = ctypes.c_int * len(values)
        c_buffer = c_buffer_type(*values)
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(len(values))

        lib().apo_charmm_context_set_fft_grid(self.handle, c_buffer, c_buffer_len)

        return

    def setPmeSplineOrder(self, order: int) -> None:
        """
        @brief Sets the PME interpolation spline order.

        @param[in] order Python integer accepted by `ctypes.c_int`. The native
        value must be positive.

        @throws TypeError If `order` cannot initialize `ctypes.c_int`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If `order` is not positive or no force manager
        is attached.

        @warning Configure the order before force-manager initialization.
        """
        _initialize_prototypes()

        c_order: ctypes.c_int = ctypes.c_int(order)

        lib().apo_charmm_context_set_pme_spline_order(self.handle, c_order)

        return

    def setVdwType(self, vdw_type: VdwType | int) -> None:
        """
        @brief Sets the van der Waals model identifier.

        The argument is normalized through `VdwType`. Native CharmmContext
        accepts values `VdwType.VSH` through `VdwType.DBEXP`.

        @param[in] vdw_type `VdwType` or matching integer value.

        @throws TypeError If `vdw_type` cannot be interpreted by the enum.
        @throws ValueError If it is not a declared Python enum value.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the normalized value is `VdwType.NONE`, no
        force manager is attached, or native validation fails.

        @warning Configure the model before force-manager initialization.
        """
        _initialize_prototypes()

        try:
            vdw_type_value: VdwType = VdwType(vdw_type)
        except ValueError as exc:
            raise ValueError(f"invalid vdw_type: {vdw_type!r}") from exc

        c_vdw_type: ctypes.c_int = ctypes.c_int(int(vdw_type_value))

        lib().apo_charmm_context_set_vdw_type(self.handle, c_vdw_type)

        return

    def getNumAtoms(self) -> int:
        """
        @brief Returns the context atom count.

        @return Python `int` containing the dimensionless atom count.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        num_atoms: ctypes.c_int = ctypes.c_int()

        lib().apo_charmm_context_get_num_atoms(ctypes.byref(num_atoms), self.handle)

        return int(num_atoms.value)

    def getNumDegreesOfFreedom(self) -> int:
        """
        @brief Returns the current degree-of-freedom count.

        @return Python `int` containing the dimensionless count.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        ndegf: ctypes.c_int = ctypes.c_int()

        lib().apo_charmm_context_get_num_degrees_of_freedom(
            ctypes.byref(ndegf), self.handle
        )

        return int(ndegf.value)

    def getCoordinatesCharges(self) -> list[list[float]]:
        """
        @brief Returns a Python copy of coordinates and charges.

        Native device storage is transferred to host before copying. The result
        contains one `[x, y, z, charge]` list per atom. Coordinates use
        angstroms and charges use elementary-charge units.

        @return Newly allocated `list[list[float]]` with shape `(N, 4)`.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If atom-count retrieval, output validation, or
        the native device-to-host transfer fails.
        """
        _initialize_prototypes()

        num_atoms: int = self.getNumAtoms()

        c_buffer_type = ctypes.c_double * (num_atoms * 4)
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(num_atoms * 4)

        lib().apo_charmm_context_get_coordinates_charges(
            c_buffer, c_buffer_len, self.handle
        )

        xyzq: list[list[float]] = []
        for i in range(num_atoms):
            xyzq.append(
                [
                    float(c_buffer[i * 4 + 0]),
                    float(c_buffer[i * 4 + 1]),
                    float(c_buffer[i * 4 + 2]),
                    float(c_buffer[i * 4 + 3]),
                ]
            )

        return xyzq

    def getVelocityMass(self) -> list[list[float]]:
        """
        @brief Returns a Python copy of velocities and inverse masses.

        Despite the method name, each fourth component is inverse mass, not
        mass. Rows are `[vx, vy, vz, inverse_mass]`; velocity uses angstroms per
        AKMA time unit and inverse mass uses reciprocal atomic mass units.

        @return Newly allocated `list[list[float]]` with shape `(N, 4)`.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If atom-count retrieval, output validation, or
        the native device-to-host transfer fails.
        """
        _initialize_prototypes()

        num_atoms: int = self.getNumAtoms()

        c_buffer_type = ctypes.c_double * (num_atoms * 4)
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(num_atoms * 4)

        lib().apo_charmm_context_get_velocity_mass(c_buffer, c_buffer_len, self.handle)

        xyzm: list[list[float]] = []
        for i in range(num_atoms):
            xyzm.append(
                [
                    float(c_buffer[i * 4 + 0]),
                    float(c_buffer[i * 4 + 1]),
                    float(c_buffer[i * 4 + 2]),
                    float(c_buffer[i * 4 + 3]),
                ]
            )

        return xyzm

    def getPeriodicBoundaryCondition(self) -> PeriodicBoundaryCondition:
        """
        @brief Returns the periodic boundary condition.

        @return `PeriodicBoundaryCondition` mapped from the native value.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the C ABI getter fails.
        @throws ValueError If a successful C ABI call returns an integer not
        represented by `PeriodicBoundaryCondition`.
        """
        _initialize_prototypes()

        c_pbc = ctypes.c_int()

        lib().apo_charmm_context_get_periodic_boundary_condition(
            ctypes.byref(c_pbc), self.handle
        )

        return PeriodicBoundaryCondition(c_pbc.value)

    def getBoxDimensions(self) -> tuple[float, float, float]:
        """
        @brief Returns the three stored box dimensions.

        @return New `(x, y, z)` tuple of lengths in angstroms.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If native output validation fails.
        """
        _initialize_prototypes()

        c_buffer_type = ctypes.c_double * 3
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(3)

        lib().apo_charmm_context_get_box_dimensions(c_buffer, c_buffer_len, self.handle)

        return (float(c_buffer[0]), float(c_buffer[1]), float(c_buffer[2]))

    def getRandomSeed(self) -> int:
        """
        @brief Returns the stored random seed.

        @return Python `int` in the unsigned 64-bit range.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the native getter fails.
        """
        _initialize_prototypes()

        c_seed: ctypes.c_uint64 = ctypes.c_uint64()

        lib().apo_charmm_context_get_random_seed(ctypes.byref(c_seed), self.handle)

        return int(c_seed.value)

    def getVolume(self) -> float:
        """
        @brief Returns the orthorhombic box volume.

        @return Python `float` in cubic angstroms.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If valid positive box dimensions have not been
        set or the native getter otherwise fails.
        """
        _initialize_prototypes()

        c_volume: ctypes.c_double = ctypes.c_double()

        lib().apo_charmm_context_get_volume(ctypes.byref(c_volume), self.handle)

        return float(c_volume.value)

    def getKappa(self) -> float:
        """
        @brief Returns the stored Ewald splitting parameter.

        @return Python `float` in inverse angstroms.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or the native
        getter fails.
        """
        _initialize_prototypes()

        c_kappa: ctypes.c_double = ctypes.c_double()

        lib().apo_charmm_context_get_kappa(ctypes.byref(c_kappa), self.handle)

        return float(c_kappa.value)

    def getCutoff(self) -> float:
        """
        @brief Returns the stored direct-space cutoff.

        @return Python `float` in angstroms.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or the native
        getter fails.
        """
        _initialize_prototypes()

        c_cutoff: ctypes.c_double = ctypes.c_double()

        lib().apo_charmm_context_get_cutoff(ctypes.byref(c_cutoff), self.handle)

        return float(c_cutoff.value)

    def getCtonnb(self) -> float:
        """
        @brief Returns the distance exposed as `ctonnb`.

        @return Python `float` in angstroms.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or the native
        getter fails.
        """
        _initialize_prototypes()

        c_ctonnb: ctypes.c_double = ctypes.c_double()

        lib().apo_charmm_context_get_ctonnb(ctypes.byref(c_ctonnb), self.handle)

        return float(c_ctonnb.value)

    def getCtofnb(self) -> float:
        """
        @brief Returns the distance exposed as `ctofnb`.

        @return Python `float` in angstroms.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or the native
        getter fails.
        """
        _initialize_prototypes()

        c_ctofnb: ctypes.c_double = ctypes.c_double()

        lib().apo_charmm_context_get_ctofnb(ctypes.byref(c_ctofnb), self.handle)

        return float(c_ctofnb.value)

    def getFFTGrid(self) -> tuple[int, int, int]:
        """
        @brief Returns the stored PME FFT grid.

        @return New `(nfftx, nffty, nfftz)` tuple of dimensionless grid sizes.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached, the native grid
        does not contain exactly three values, or output validation fails.
        """
        _initialize_prototypes()

        c_buffer_type = ctypes.c_int * 3
        c_buffer = c_buffer_type()
        c_buffer_len: ctypes.c_size_t = ctypes.c_size_t(3)

        lib().apo_charmm_context_get_fft_grid(c_buffer, c_buffer_len, self.handle)

        return (int(c_buffer[0]), int(c_buffer[1]), int(c_buffer[2]))

    def getPmeSplineOrder(self) -> int:
        """
        @brief Returns the stored PME interpolation spline order.

        @return Python `int` containing the dimensionless order.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or the native
        getter fails.
        """
        _initialize_prototypes()

        c_order: ctypes.c_int = ctypes.c_int()

        lib().apo_charmm_context_get_pme_spline_order(
            ctypes.byref(c_order), self.handle
        )

        return int(c_order.value)

    def getVdwType(self) -> VdwType:
        """
        @brief Returns the stored van der Waals model identifier.

        @return `VdwType` mapped from the native model code.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If no force manager is attached or the native
        getter fails.
        @throws ValueError If a successful native call returns an integer not
        represented by `VdwType`.
        """
        _initialize_prototypes()

        c_vdw_type: ctypes.c_int = ctypes.c_int()

        lib().apo_charmm_context_get_vdw_type(ctypes.byref(c_vdw_type), self.handle)

        return VdwType(c_vdw_type.value)

    def getForceManager(self) -> ForceManager:
        """
        @brief Returns a Python wrapper for the associated ForceManager.

        When construction or `setForceManager()` retained a wrapper, that same
        Python object is returned. Otherwise, the C ABI creates a newly owned
        force-manager handle sharing the native manager; the wrapper takes
        ownership of that handle and is cached by this context.

        Closing the returned `ForceManager` releases only its C handle. The
        native context continues to retain its native manager.

        @return Cached or newly created `ForceManager` wrapper.

        @throws RuntimeError If the context must access its native handle after
        closure.
        @throws ApoCharmmError If the native context has no force manager or C
        handle creation fails.
        """
        _initialize_prototypes()

        if self._force_manager is not None:
            return self._force_manager

        fm_handle: ctypes.c_void_p = ctypes.c_void_p()

        lib().apo_charmm_context_get_force_manager(ctypes.byref(fm_handle), self.handle)

        self._force_manager = ForceManager._from_handle(fm_handle)

        return self._force_manager

    def assignVelocitiesAtTemperature(self, temperature: float) -> None:
        """
        @brief Assigns sampled velocities at a temperature.

        The native context samples independent Cartesian Gaussian components
        using its stored seed. It does not remove center-of-mass motion or
        rescale the sampled result.

        @param[in] temperature Python `int` or `float` accepted by
        `ctypes.c_double`, in kelvin. The value must be finite and
        non-negative.

        @throws TypeError If `temperature` cannot initialize `ctypes.c_double`.
        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the atom count is unset, native temperature
        validation fails, or the velocity transfer fails.
        """
        _initialize_prototypes()

        c_temperature: ctypes.c_double = ctypes.c_double(temperature)

        lib().apo_charmm_context_assign_velocities_at_temperature(
            self.handle, c_temperature
        )

        return

    def computeTemperature(self) -> float:
        """
        @brief Computes the instantaneous kinetic temperature.

        The native calculation evaluates kinetic energy on the GPU and divides
        by `0.5 * ndegf * k_B`.

        @return Python `float` in kelvin.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If atom, velocity, or degree-of-freedom state is
        incomplete, or native CUDA work fails.
        """
        _initialize_prototypes()

        c_temperature = ctypes.c_double()

        lib().apo_charmm_context_compute_temperature(
            ctypes.byref(c_temperature), self.handle
        )

        return float(c_temperature.value)

    def calculatePotentialEnergy(self, print_energy: bool = False) -> None:
        """
        @brief Computes forces, potential energy, and virial.

        The Python wrapper always passes `reset=False`. When `print_energy` is
        truthy, the native context writes a CHARMM-style energy table to
        standard output.

        @param[in] print_energy Boolean selecting native energy-table output.

        @throws RuntimeError If this context is closed.
        @throws ApoCharmmError If the force manager is not initialized,
        composite-manager printing is unsupported, or native force or CUDA work
        fails.
        """
        _initialize_prototypes()

        c_reset: ctypes.c_bool = ctypes.c_bool(False)
        c_print: ctypes.c_bool = ctypes.c_bool(print_energy)

        lib().apo_charmm_context_calculate_potential_energy(
            self.handle, c_reset, c_print
        )

        return
