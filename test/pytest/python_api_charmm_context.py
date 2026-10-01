# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from pathlib import Path
import sys

import apocharmm as apo

import apo_test_helpers as apo_test

TOLERANCE: float = 1.0e-7
BOX_DIMENSIONS: tuple[float, float, float] = (40.0, 41.0, 42.0)
FFT_GRID: tuple[int, int, int] = (32, 34, 36)
RANDOM_SEED: int = 0x1234ABCD
ASSIGNMENT_TEMPERATURE: float = 300.0

COORDINATES_CHARGES: tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] = (
    (1.0, 2.0, 3.0, -0.75),
    (-4.0, 5.0, -6.0, 0.75),
)

COORDINATES: tuple[
    tuple[float, float, float],
    tuple[float, float, float],
] = (
    (7.0, 8.0, 9.0),
    (-10.0, 11.0, -12.0),
)

CHARGES: tuple[float, float] = (-0.5, 0.5)

VELOCITIES_INVERSE_MASSES: tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] = (
    (0.25, -0.50, 0.75, 0.125),
    (-1.00, 1.25, -1.50, 0.250),
)

VELOCITIES: tuple[
    tuple[float, float, float],
    tuple[float, float, float],
] = (
    (1.5, -2.5, 3.5),
    (-4.5, 5.5, -6.5),
)

MASSES: tuple[float, float] = (4.0, 8.0)

WATER_PSF_TEXT: str = """PSF

       1 !NTITLE
 REMARKS generated CharmmContext Python API test PSF
       3 !NATOM
       1 WAT      1 TIP3 OH2  OT    -0.834000  15.9994           0
       2 WAT      1 TIP3 H1   HT     0.417000   1.0080           0
       3 WAT      1 TIP3 H2   HT     0.417000   1.0080           0
       2 !NBOND: bonds
       1       2       1       3
       1 !NTHETA: angles
       2       1       3
       0 !NPHI: dihedrals
       0 !NIMPHI: impropers
       0 !NDON: donors
       0 !NACC: acceptors
       0 !NCRTERM: cross-terms
"""

VELOCITY_FILE_TEXT: str = """* generated CharmmContext Python API test velocity file
*
       2
       1       1 IONS SOD   1.25  -2.50   3.75 IONS 1 0.0
       2       2 IONS CLA  -4.50   5.25  -6.00 IONS 2 0.0
"""

VELOCITY_COUNT_MISMATCH_TEXT: str = """* generated count-mismatch velocity file
*
       1
       1       1 IONS SOD   1.25  -2.50   3.75 IONS 1 0.0
"""

VELOCITY_BAD_COUNT_TEXT: str = """* generated malformed-count velocity file
*
not-an-atom-count
"""

VELOCITY_BAD_RECORD_TEXT: str = """* generated malformed-record velocity file
*
       2
       1       1 IONS SOD   1.25  -2.50   3.75 IONS 1 0.0
not-a-velocity-record
"""

STATUS_NAMES: dict[int, str] = {
    apo.APO_STATUS_INVALID_ARGUMENT: "APO_STATUS_INVALID_ARGUMENT",
    apo.APO_STATUS_RUNTIME_ERROR: "APO_STATUS_RUNTIME_ERROR",
    apo.APO_STATUS_CUDA_ERROR: "APO_STATUS_CUDA_ERROR",
    apo.APO_STATUS_NOT_INITIALIZED: "APO_STATUS_NOT_INITIALIZED",
    apo.APO_STATUS_NOT_IMPLEMENTED: "APO_STATUS_NOT_IMPLEMENTED",
}


def combine_xyz_and_scalar(
    xyz: Sequence[Sequence[float]], scalar: Sequence[float]
) -> list[list[float]]:
    if len(xyz) != len(scalar):
        raise ValueError("xyz and scalar lengths must match")

    combined: list[list[float]] = []
    for i, row in enumerate(xyz):
        if len(row) != 3:
            raise ValueError(f"xyz[{i}] must contain exactly 3 values")

        combined.append([float(row[0]), float(row[1]), float(row[2]), float(scalar[i])])

    return combined


def inverse_masses(masses: Sequence[float]) -> list[float]:
    return [1.0 / float(mass) for mass in masses]


def check_construction_and_default_state(
    parameters: apo.CharmmParameters,
    psf: apo.CharmmPsf,
) -> None:
    print("Checking CharmmContext construction and default state...")

    context = apo.CharmmContext(psf, parameters)
    force_manager: apo.ForceManager | None = None

    try:
        num_atoms: int = psf.getNumAtoms()
        psf_charges: list[float] = psf.getCharges()
        psf_inverse_masses: list[float] = inverse_masses(psf.getMasses())

        apo_test.assert_equal(
            "CharmmContext.getNumAtoms default",
            context.getNumAtoms(),
            num_atoms,
        )
        apo_test.assert_equal(
            "CharmmContext.getNumDegreesOfFreedom default",
            context.getNumDegreesOfFreedom(),
            3,
        )
        apo_test.assert_equal(
            "CharmmContext.getPeriodicBoundaryCondition default",
            context.getPeriodicBoundaryCondition(),
            apo.PeriodicBoundaryCondition.P1,
        )
        apo_test.assert_sequence_close(
            "CharmmContext.getBoxDimensions default",
            context.getBoxDimensions(),
            (-9999.9999, -9999.9999, -9999.9999),
            TOLERANCE,
        )

        apo_test.assert_close(
            "CharmmContext.getKappa default",
            context.getKappa(),
            0.34,
            TOLERANCE,
        )
        apo_test.assert_close(
            "CharmmContext.getCutoff default", context.getCutoff(), 14.0, TOLERANCE
        )
        apo_test.assert_close(
            "CharmmContext.getCtonnb default", context.getCtonnb(), 12.0, TOLERANCE
        )
        apo_test.assert_close(
            "CharmmContext.getCtofnb default", context.getCtofnb(), 10.0, TOLERANCE
        )
        apo_test.assert_equal(
            "CharmmContext.getFFTGrid default",
            context.getFFTGrid(),
            (-1, -1, -1),
        )
        apo_test.assert_equal(
            "CharmmContext.getPmeSplineOrder default",
            context.getPmeSplineOrder(),
            4,
        )
        apo_test.assert_equal(
            "CharmmContext.getVdwType default",
            context.getVdwType(),
            apo.VdwType.VFSW,
        )

        expected_xyzq: list[list[float]] = combine_xyz_and_scalar(
            ((0.0, 0.0, 0.0),) * num_atoms, psf_charges
        )
        expected_xyzm: list[list[float]] = combine_xyz_and_scalar(
            ((0.0, 0.0, 0.0),) * num_atoms, psf_inverse_masses
        )

        apo_test.assert_nested_sequence_close(
            "CharmmContext coordinates/charges default",
            context.getCoordinatesCharges(),
            expected_xyzq,
            TOLERANCE,
        )
        apo_test.assert_nested_sequence_close(
            "CharmmContext velocities/inverse masses default",
            context.getVelocityMass(),
            expected_xyzm,
            TOLERANCE,
        )

        random_seed: int = context.getRandomSeed()
        if random_seed < 0 or random_seed > 2**64 - 1:
            raise AssertionError(
                f"CharmmContext default random seed is outside uint64_t: {random_seed}"
            )

        force_manager = context.getForceManager()
        apo_test.assert_equal(
            "CharmmContext.getForceManager returns cached wrapper",
            context.getForceManager() is force_manager,
            True,
        )
        apo_test.assert_equal(
            "CharmmContext ForceManager atom count",
            force_manager.getNumAtoms(),
            num_atoms,
        )
        apo_test.assert_equal(
            "CharmmContext ForceManager initially uninitialized",
            force_manager.isInitialized(),
            False,
        )
    finally:
        context.close()
        if force_manager is not None:
            force_manager.close()

    backend = apo.ForceManager(psf, parameters)
    backend.setKappa(0.41)
    backend.setCutoff(8.75)
    backend_context: apo.CharmmContext | None = None

    try:
        backend_context = apo.CharmmContext(backend)

        apo_test.assert_equal(
            "CharmmContext ForceManager constructor preserves wrapper identity",
            backend_context.getForceManager() is backend,
            True,
        )
        apo_test.assert_equal(
            "CharmmContext ForceManager constructor atom count",
            backend_context.getNumAtoms(),
            psf.getNumAtoms(),
        )
        apo_test.assert_close(
            "CharmmContext ForceManager constructor kappa",
            backend_context.getKappa(),
            0.41,
            TOLERANCE,
        )
        apo_test.assert_close(
            "CharmmContext ForceManager constructor cutoff",
            backend_context.getCutoff(),
            8.75,
            TOLERANCE,
        )
        apo_test.assert_equal(
            "CharmmContext ForceManager constructor leaves degrees of freedom unset",
            backend_context.getNumDegreesOfFreedom(),
            -1,
        )
    finally:
        if backend_context is not None:
            backend_context.close()
        backend.close()

    return


def check_object_setters(
    parameter_path: str,
    psf_path: str,
    parameters: apo.CharmmParameters,
    psf: apo.CharmmPsf,
) -> None:
    print("Checking CharmmContext object setters...")

    replacement_parameters = apo.CharmmParameters(parameter_path)
    replacement_psf = apo.CharmmPsf(psf_path)
    replacement_force_manager = apo.ForceManager(
        replacement_psf, replacement_parameters
    )
    context = apo.CharmmContext(psf, parameters)

    try:
        context.setPrm(replacement_parameters)
        context.setPsf(replacement_psf)
        apo_test.assert_equal(
            "CharmmContext.setPsf preserves atom count",
            context.getNumAtoms(),
            replacement_psf.getNumAtoms(),
        )

        context.setForceManager(replacement_force_manager)
        apo_test.assert_equal(
            "CharmmContext.setForceManager updates cached wrapper",
            context.getForceManager() is replacement_force_manager,
            True,
        )
        apo_test.assert_equal(
            "CharmmContext.setForceManager preserves atom count",
            context.getNumAtoms(),
            replacement_force_manager.getNumAtoms(),
        )
    finally:
        context.close()
        replacement_force_manager.close()
        replacement_psf.close()
        replacement_parameters.close()

    return


def check_array_state_round_trips(
    parameters: apo.CharmmParameters,
    psf: apo.CharmmPsf,
    crd: apo.CharmmCrd,
) -> None:
    print(
        "Checking CharmmContext coordinate, charge, velocity, and mass round trips..."
    )

    context = apo.CharmmContext(psf, parameters)

    try:
        context.setCoordinatesCharges(COORDINATES_CHARGES)
        apo_test.assert_nested_sequence_close(
            "CharmmContext.setCoordinatesCharges",
            context.getCoordinatesCharges(),
            COORDINATES_CHARGES,
            TOLERANCE,
        )

        context.setCoordinates(COORDINATES)
        expected_xyzq: list[list[float]] = combine_xyz_and_scalar(
            COORDINATES,
            [row[3] for row in COORDINATES_CHARGES],
        )
        apo_test.assert_nested_sequence_close(
            "CharmmContext.setCoordinates preserves charges",
            context.getCoordinatesCharges(),
            expected_xyzq,
            TOLERANCE,
        )

        context.setCharges(CHARGES)
        expected_xyzq = combine_xyz_and_scalar(COORDINATES, CHARGES)
        apo_test.assert_nested_sequence_close(
            "CharmmContext.setCharges preserves coordinates",
            context.getCoordinatesCharges(),
            expected_xyzq,
            TOLERANCE,
        )

        context.setCoordinates(crd)
        expected_xyzq = combine_xyz_and_scalar(crd.getCoordinates(), CHARGES)
        apo_test.assert_nested_sequence_close(
            "CharmmContext.setCoordinates CharmmCrd overload",
            context.getCoordinatesCharges(),
            expected_xyzq,
            TOLERANCE,
        )

        context.setVelocitiesInverseMasses(VELOCITIES_INVERSE_MASSES)
        apo_test.assert_nested_sequence_close(
            "CharmmContext.setVelocitiesInverseMasses",
            context.getVelocityMass(),
            VELOCITIES_INVERSE_MASSES,
            TOLERANCE,
        )

        context.setVelocities(VELOCITIES)
        expected_xyzm: list[list[float]] = combine_xyz_and_scalar(
            VELOCITIES,
            [row[3] for row in VELOCITIES_INVERSE_MASSES],
        )
        apo_test.assert_nested_sequence_close(
            "CharmmContext.setVelocities preserves inverse masses",
            context.getVelocityMass(),
            expected_xyzm,
            TOLERANCE,
        )

        context.setMasses(MASSES)
        expected_xyzm = combine_xyz_and_scalar(VELOCITIES, inverse_masses(MASSES))
        apo_test.assert_nested_sequence_close(
            "CharmmContext.setMasses stores inverse masses",
            context.getVelocityMass(),
            expected_xyzm,
            TOLERANCE,
        )
    finally:
        context.close()

    return


def check_velocity_file_loading(
    parameters: apo.CharmmParameters,
    psf: apo.CharmmPsf,
    valid_path: Path,
    mismatch_path: Path,
    bad_count_path: Path,
    bad_record_path: Path,
    missing_path: Path,
) -> None:
    print("Checking CharmmContext CHARMM velocity-file loading...")

    context = apo.CharmmContext(psf, parameters)

    try:
        apo_test.expect_apo_error(
            "CharmmContext velocity file rejects empty path",
            lambda: context.setVelocitiesFromCHARMMVelocityFile(""),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "CHARMM velocity file is NULL or empty",
            "CharmmContext.setVelocitiesFromCHARMMVelocityFile(file_name)",
        )
        apo_test.expect_apo_error(
            "CharmmContext velocity file reports open failure",
            lambda: context.setVelocitiesFromCHARMMVelocityFile(missing_path),
            apo.APO_STATUS_RUNTIME_ERROR,
            f'Could not open CHARMM velocity file "{missing_path}"',
            "CharmmContext.setVelocitiesFromCHARMMVelocityFile(file_name)",
        )
        apo_test.expect_apo_error(
            "CharmmContext velocity file rejects atom-count mismatch",
            lambda: context.setVelocitiesFromCHARMMVelocityFile(mismatch_path),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "CHARMM velocity atom count mismatch; expected 2, observed 1",
            "CharmmContext.setVelocitiesFromCHARMMVelocityFile(file_name)",
        )
        apo_test.expect_apo_error(
            "CharmmContext velocity file reports malformed atom count",
            lambda: context.setVelocitiesFromCHARMMVelocityFile(bad_count_path),
            apo.APO_STATUS_RUNTIME_ERROR,
            "Could not parse atom count from CHARMM velocity file",
            "CharmmContext.setVelocitiesFromCHARMMVelocityFile(file_name)",
        )
        apo_test.expect_apo_error(
            "CharmmContext velocity file reports malformed record",
            lambda: context.setVelocitiesFromCHARMMVelocityFile(bad_record_path),
            apo.APO_STATUS_RUNTIME_ERROR,
            "Could not parse velocity record 2 from CHARMM velocity file",
            "CharmmContext.setVelocitiesFromCHARMMVelocityFile(file_name)",
        )

        context.setMasses(MASSES)
        context.setVelocitiesFromCHARMMVelocityFile(valid_path)

        expected_xyzm: list[list[float]] = combine_xyz_and_scalar(
            ((1.25, -2.50, 3.75), (-4.50, 5.25, -6.00)),
            inverse_masses(MASSES),
        )
        apo_test.assert_nested_sequence_close(
            "CharmmContext CHARMM velocity file values",
            context.getVelocityMass(),
            expected_xyzm,
            TOLERANCE,
        )
    finally:
        context.close()

    return


def check_configuration_and_initialized_state(
    parameters: apo.CharmmParameters,
    psf: apo.CharmmPsf,
    crd: apo.CharmmCrd,
) -> None:
    print("Checking CharmmContext configuration and initialized backend state...")

    context = apo.CharmmContext(psf, parameters)
    force_manager: apo.ForceManager | None = None

    try:
        context.setTemperature(275.0)
        context.setKappa(0.45)
        context.setCutoff(9.0)
        context.setCtonnb(7.5)
        context.setCtofnb(8.5)
        context.setFFTGrid(FFT_GRID)
        context.setPmeSplineOrder(6)

        for pbc in (
            apo.PeriodicBoundaryCondition.P1,
            apo.PeriodicBoundaryCondition.P21,
        ):
            context.setPeriodicBoundaryCondition(pbc)
            apo_test.assert_equal(
                f"CharmmContext periodic boundary condition {pbc}",
                context.getPeriodicBoundaryCondition(),
                pbc,
            )

        apo_test.expect_invalid_argument(
            "CharmmContext.setPeriodicBoundaryCondition rejects UNSET",
            lambda: context.setPeriodicBoundaryCondition(
                apo.PeriodicBoundaryCondition.UNSET
            ),
            "apoCHARMM error [InvalidArgument]: Periodic boundary condition "
            "must be PBC::P1 or PBC::P21; observed 0",
        )
        apo_test.assert_equal(
            "CharmmContext PBC remains unchanged after rejected UNSET",
            context.getPeriodicBoundaryCondition(),
            apo.PeriodicBoundaryCondition.P21,
        )

        context.setPeriodicBoundaryCondition(apo.PeriodicBoundaryCondition.P1)
        context.useHolonomicConstraints(False)
        apo_test.assert_equal(
            "CharmmContext degrees of freedom without holonomic constraints",
            context.getNumDegreesOfFreedom(),
            3,
        )

        for vdw_type in (
            apo.VdwType.VSH,
            apo.VdwType.VSW,
            apo.VdwType.VFSW,
            apo.VdwType.VGSH,
            apo.VdwType.CUT,
            apo.VdwType.DBEXP,
        ):
            context.setVdwType(vdw_type)
            apo_test.assert_equal(
                f"CharmmContext van der Waals type {vdw_type}",
                context.getVdwType(),
                vdw_type,
            )

        context.setRandomSeed(0)
        apo_test.assert_equal(
            "CharmmContext zero random seed", context.getRandomSeed(), 0
        )
        context.setRandomSeed(2**64 - 1)
        apo_test.assert_equal(
            "CharmmContext maximum random seed",
            context.getRandomSeed(),
            2**64 - 1,
        )
        context.setRandomSeed(RANDOM_SEED)
        apo_test.assert_equal(
            "CharmmContext configured random seed",
            context.getRandomSeed(),
            RANDOM_SEED,
        )

        apo_test.assert_close(
            "CharmmContext configured kappa",
            context.getKappa(),
            0.45,
            TOLERANCE,
        )
        apo_test.assert_close(
            "CharmmContext configured cutoff",
            context.getCutoff(),
            9.0,
            TOLERANCE,
        )
        apo_test.assert_close(
            "CharmmContext configured ctonnb",
            context.getCtonnb(),
            7.5,
            TOLERANCE,
        )
        apo_test.assert_close(
            "CharmmContext configured ctofnb",
            context.getCtofnb(),
            8.5,
            TOLERANCE,
        )
        apo_test.assert_equal(
            "CharmmContext configured FFT grid",
            context.getFFTGrid(),
            FFT_GRID,
        )
        apo_test.assert_equal(
            "CharmmContext configured PME spline order",
            context.getPmeSplineOrder(),
            6,
        )
        apo_test.assert_equal(
            "CharmmContext configured van der Waals type",
            context.getVdwType(),
            apo.VdwType.DBEXP,
        )

        context.setBoxDimensions(BOX_DIMENSIONS)

        apo_test.assert_sequence_close(
            "CharmmContext configured box dimensions",
            context.getBoxDimensions(),
            BOX_DIMENSIONS,
            TOLERANCE,
        )
        apo_test.assert_close(
            "CharmmContext configured volume",
            context.getVolume(),
            BOX_DIMENSIONS[0] * BOX_DIMENSIONS[1] * BOX_DIMENSIONS[2],
            TOLERANCE,
        )

        force_manager = context.getForceManager()
        apo_test.assert_equal(
            "CharmmContext backend initialized after complete state",
            force_manager.isInitialized(),
            True,
        )
        apo_test.assert_sequence_close(
            "CharmmContext backend box dimensions",
            force_manager.getBoxDimensions(),
            BOX_DIMENSIONS,
            TOLERANCE,
        )
        apo_test.assert_close(
            "CharmmContext backend kappa", force_manager.getKappa(), 0.45, TOLERANCE
        )
        apo_test.assert_close(
            "CharmmContext backend cutoff", force_manager.getCutoff(), 9.0, TOLERANCE
        )
        apo_test.assert_equal(
            "CharmmContext backend FFT grid", force_manager.getFFTGrid(), FFT_GRID
        )
        apo_test.assert_equal(
            "CharmmContext backend PME spline order",
            force_manager.getPmeSplineOrder(),
            6,
        )
        apo_test.assert_equal(
            "CharmmContext backend van der Waals type",
            force_manager.getVdwType(),
            apo.VdwType.DBEXP,
        )

        context.setCoordinates(crd)
        expected_xyzq: list[list[float]] = combine_xyz_and_scalar(
            crd.getCoordinates(), psf.getCharges()
        )
        apo_test.assert_nested_sequence_close(
            "CharmmContext initialized coordinate loading",
            context.getCoordinatesCharges(),
            expected_xyzq,
            TOLERANCE,
        )
    finally:
        context.close()
        if force_manager is not None:
            force_manager.close()

    return


def check_random_velocity_assignment_and_temperature(
    parameters: apo.CharmmParameters,
    psf: apo.CharmmPsf,
) -> None:
    print("Checking CharmmContext random velocity assignment and temperature...")

    first = apo.CharmmContext(psf, parameters)
    second = apo.CharmmContext(psf, parameters)

    try:
        first.setRandomSeed(RANDOM_SEED)
        second.setRandomSeed(RANDOM_SEED)

        first.assignVelocitiesAtTemperature(ASSIGNMENT_TEMPERATURE)
        second.assignVelocitiesAtTemperature(ASSIGNMENT_TEMPERATURE)

        first_values: list[list[float]] = first.getVelocityMass()
        second_values: list[list[float]] = second.getVelocityMass()

        apo_test.assert_finite_nested_sequence(
            "CharmmContext assigned velocities", first_values
        )
        apo_test.assert_nested_sequence_close(
            "CharmmContext deterministic velocity assignment",
            first_values,
            second_values,
            TOLERANCE,
        )
        apo_test.assert_sequence_close(
            "CharmmContext assigned inverse masses",
            [row[3] for row in first_values],
            inverse_masses(psf.getMasses()),
            TOLERANCE,
        )

        first_temperature: float = first.computeTemperature()
        second_temperature: float = second.computeTemperature()
        apo_test.assert_finite_temperature(
            "CharmmContext first computed", first_temperature
        )
        apo_test.assert_finite_temperature(
            "CharmmContext second computed", second_temperature
        )
        apo_test.assert_close(
            "CharmmContext deterministic computed temperature",
            first_temperature,
            second_temperature,
            TOLERANCE,
        )

        first.setVelocities(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        apo_test.assert_close(
            "CharmmContext zero-velocity temperature",
            first.computeTemperature(),
            0.0,
            TOLERANCE,
        )
        first.setTemperature(0.0)
    finally:
        first.close()
        second.close()

    backend = apo.ForceManager(psf, parameters)
    backend_context: apo.CharmmContext | None = None

    try:
        backend_context = apo.CharmmContext(backend)
        apo_test.assert_equal(
            "CharmmContext ForceManager-constructor degrees of freedom sentinel",
            backend_context.getNumDegreesOfFreedom(),
            -1,
        )

        apo_test.expect_apo_error(
            "CharmmContext.computeTemperature rejects unset degrees of freedom",
            backend_context.computeTemperature,
            apo.APO_STATUS_NOT_INITIALIZED,
            "Degrees of freedom must be initialized before computing temperature",
            "CharmmContext.computeTemperature()",
        )

        backend_context.useHolonomicConstraints(False)
        backend_context.setVelocities(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        apo_test.assert_close(
            "CharmmContext initialized degrees-of-freedom temperature",
            backend_context.computeTemperature(),
            0.0,
            TOLERANCE,
        )
    finally:
        if backend_context is not None:
            backend_context.close()
        backend.close()

    return


def check_holonomic_constraints(
    parameters: apo.CharmmParameters,
    water_psf_path: Path,
) -> None:
    print("Checking CharmmContext holonomic-constraint degrees of freedom...")

    water_psf = apo.CharmmPsf(water_psf_path)
    context = apo.CharmmContext(water_psf, parameters)

    try:
        apo_test.assert_equal(
            "water CharmmContext atom count", context.getNumAtoms(), 3
        )
        apo_test.assert_equal(
            "water CharmmContext constrained P1 degrees of freedom",
            context.getNumDegreesOfFreedom(),
            3,
        )

        context.useHolonomicConstraints(False)
        apo_test.assert_equal(
            "water CharmmContext unconstrained P1 degrees of freedom",
            context.getNumDegreesOfFreedom(),
            6,
        )

        context.useHolonomicConstraints(True)
        apo_test.assert_equal(
            "water CharmmContext re-enabled constraints",
            context.getNumDegreesOfFreedom(),
            3,
        )

        context.setPeriodicBoundaryCondition(apo.PeriodicBoundaryCondition.P21)
        context.useHolonomicConstraints(True)
        apo_test.assert_equal(
            "water CharmmContext constrained P21 degrees of freedom",
            context.getNumDegreesOfFreedom(),
            5,
        )
    finally:
        context.close()
        water_psf.close()

    return


def check_validation(
    parameter_path: str,
    parameters: apo.CharmmParameters,
    psf: apo.CharmmPsf,
    water_psf_path: Path,
) -> None:
    print("Checking CharmmContext Python and native validation...")

    force_manager = apo.ForceManager(psf, parameters)
    context = apo.CharmmContext(psf, parameters)
    water_psf = apo.CharmmPsf(water_psf_path)

    try:
        apo_test.expect_exception(
            "CharmmContext rejects unsupported constructor input",
            TypeError,
            lambda: apo.CharmmContext(object()),  # type: ignore[arg-type]
        )
        apo_test.expect_exception(
            "CharmmContext rejects CharmmPsf without parameters",
            TypeError,
            lambda: apo.CharmmContext(psf),
        )
        apo_test.expect_exception(
            "CharmmContext rejects invalid parameters object",
            TypeError,
            lambda: apo.CharmmContext(psf, object()),  # type: ignore[arg-type]
        )
        apo_test.expect_exception(
            "CharmmContext rejects parameters with ForceManager constructor",
            TypeError,
            lambda: apo.CharmmContext(force_manager, parameters),
        )

        apo_test.expect_exception(
            "CharmmContext.setPrm rejects non-CharmmParameters",
            TypeError,
            lambda: context.setPrm(object()),  # type: ignore[arg-type]
        )
        apo_test.expect_exception(
            "CharmmContext.setPsf rejects non-CharmmPsf",
            TypeError,
            lambda: context.setPsf(object()),  # type: ignore[arg-type]
        )
        apo_test.expect_exception(
            "CharmmContext.setForceManager rejects non-ForceManager",
            TypeError,
            lambda: context.setForceManager(object()),  # type: ignore[arg-type]
        )

        apo_test.expect_apo_error(
            "CharmmContext.setPsf rejects atom-count mismatch",
            lambda: context.setPsf(water_psf),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Atom count mismatch between CharmmContext and CharmmPSF; expected 2, observed 3",
            "CharmmContext.setPsf(psf)",
        )

        apo_test.expect_exception(
            "CharmmContext.setCoordinatesCharges rejects wrong row width",
            ValueError,
            lambda: context.setCoordinatesCharges(((1.0, 2.0, 3.0),)),
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCoordinatesCharges rejects wrong atom count",
            lambda: context.setCoordinatesCharges(((1.0, 2.0, 3.0, 0.0),)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Coordinate and charge count mismatch; expected 2, observed 1",
            "CharmmContext.setCoordinatesCharges(coordinates_charges)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCoordinatesCharges rejects empty input",
            lambda: context.setCoordinatesCharges(()),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "xyzq must not be empty",
            "CharmmContext.setCoordinatesCharges(coordinates_charges)",
        )

        apo_test.expect_exception(
            "CharmmContext.setCoordinates rejects wrong row width",
            ValueError,
            lambda: context.setCoordinates(((1.0, 2.0),)),
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCoordinates rejects wrong atom count",
            lambda: context.setCoordinates(((1.0, 2.0, 3.0),)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Coordinate count mismatch; expected 2, observed 1",
            "CharmmContext.setCoordinates(coordinates)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCoordinates rejects empty input",
            lambda: context.setCoordinates(()),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "xyz must not be empty",
            "CharmmContext.setCoordinates(coordinates)",
        )

        apo_test.expect_apo_error(
            "CharmmContext.setCharges rejects wrong atom count",
            lambda: context.setCharges((0.0,)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Charge count mismatch; expected 2, observed 1",
            "CharmmContext.setCharges(charges)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCharges rejects empty input",
            lambda: context.setCharges(()),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Charge count mismatch; expected 2, observed 0",
            "CharmmContext.setCharges(charges)",
        )

        apo_test.expect_exception(
            "CharmmContext.setVelocitiesInverseMasses rejects wrong row width",
            ValueError,
            lambda: context.setVelocitiesInverseMasses(((1.0, 2.0, 3.0),)),
        )
        apo_test.expect_apo_error(
            "CharmmContext.setVelocitiesInverseMasses rejects wrong atom count",
            lambda: context.setVelocitiesInverseMasses(((1.0, 2.0, 3.0, 1.0),)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Velocity and inverse-mass count mismatch; expected 2, observed 1",
            "CharmmContext.setVelocitiesInverseMasses(velocities_inverse_masses)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setVelocitiesInverseMasses rejects empty input",
            lambda: context.setVelocitiesInverseMasses(()),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "xyzm must not be empty",
            "CharmmContext.setVelocitiesInverseMasses(velocities_inverse_masses)",
        )

        apo_test.expect_exception(
            "CharmmContext.setVelocities rejects wrong row width",
            ValueError,
            lambda: context.setVelocities(((1.0, 2.0),)),
        )
        apo_test.expect_apo_error(
            "CharmmContext.setVelocities rejects wrong atom count",
            lambda: context.setVelocities(((1.0, 2.0, 3.0),)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Velocity count mismatch; expected 2, observed 1",
            "CharmmContext.setVelocities(velocities)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setVelocities rejects empty input",
            lambda: context.setVelocities(()),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "xyz must not be empty",
            "CharmmContext.setVelocities(velocities)",
        )

        apo_test.expect_apo_error(
            "CharmmContext.setMasses rejects wrong atom count",
            lambda: context.setMasses((1.0,)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Mass count mismatch; expected 2, observed 1",
            "CharmmContext.setMasses(masses)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setMasses rejects empty input",
            lambda: context.setMasses(()),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Mass count mismatch; expected 2, observed 0",
            "CharmmContext.setMasses(masses)",
        )

        apo_test.expect_apo_error(
            "CharmmContext.setTemperature rejects NaN",
            lambda: context.setTemperature(math.nan),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Temperature must be finite and non-negative",
            "CharmmContext.setTemperature(temperature)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setTemperature rejects negative values",
            lambda: context.setTemperature(-1.0),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Temperature must be finite and non-negative",
            "CharmmContext.setTemperature(temperature)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.assignVelocitiesAtTemperature rejects NaN",
            lambda: context.assignVelocitiesAtTemperature(math.nan),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Temperature must be finite and non-negative",
            "CharmmContext.assignVelocitiesAtTemperature(temperature)",
        )

        apo_test.expect_exception(
            "CharmmContext.setPeriodicBoundaryCondition rejects invalid value",
            ValueError,
            lambda: context.setPeriodicBoundaryCondition(99),
        )
        apo_test.expect_exception(
            "CharmmContext.setVdwType rejects invalid value",
            ValueError,
            lambda: context.setVdwType(99),
        )
        apo_test.expect_apo_error(
            "CharmmContext.setVdwType rejects VdwType.NONE",
            lambda: context.setVdwType(apo.VdwType.NONE),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Van der Waals type must be in [1, 6]; observed 0",
            "CharmmContext.setVdwType(vdw_type)",
        )

        apo_test.expect_apo_error(
            "CharmmContext.setBoxDimensions rejects short input",
            lambda: context.setBoxDimensions((20.0, 20.0)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "box_dimensions must contain exactly 3 elements",
            "CharmmContext.setBoxDimensions(box_dimensions)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setBoxDimensions rejects long input",
            lambda: context.setBoxDimensions((20.0, 20.0, 20.0, 20.0)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "box_dimensions must contain exactly 3 elements",
            "CharmmContext.setBoxDimensions(box_dimensions)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setBoxDimensions rejects zero dimensions",
            lambda: context.setBoxDimensions((20.0, 0.0, 20.0)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Box dimensions must contain exactly 3 positive values",
            "CharmmContext.setBoxDimensions(box_dimensions)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setBoxDimensions rejects negative dimensions",
            lambda: context.setBoxDimensions((20.0, -1.0, 20.0)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Box dimensions must contain exactly 3 positive values",
            "CharmmContext.setBoxDimensions(box_dimensions)",
        )

        apo_test.expect_apo_error(
            "CharmmContext.getVolume rejects unset box dimensions",
            context.getVolume,
            apo.APO_STATUS_NOT_INITIALIZED,
            "Box dimensions are not set",
            "CharmmContext.getVolume()",
        )

        apo_test.expect_apo_error(
            "CharmmContext.setKappa rejects NaN",
            lambda: context.setKappa(math.nan),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Kappa must be finite; observed nan",
            "CharmmContext.setKappa(kappa)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setKappa rejects negative values",
            lambda: context.setKappa(-1.0),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Kappa must be non-negative; observed -1.000000",
            "CharmmContext.setKappa(kappa)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCutoff rejects zero",
            lambda: context.setCutoff(0.0),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Cutoff must be positive; observed 0.000000",
            "CharmmContext.setCutoff(cutoff)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCtonnb rejects zero",
            lambda: context.setCtonnb(0.0),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Ctonnb must be positive; observed 0.000000",
            "CharmmContext.setCtonnb(ctonnb)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setCtofnb rejects zero",
            lambda: context.setCtofnb(0.0),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Ctofnb must be positive; observed 0.000000",
            "CharmmContext.setCtofnb(ctofnb)",
        )

        apo_test.expect_apo_error(
            "CharmmContext.setFFTGrid rejects short input",
            lambda: context.setFFTGrid((32, 34)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "grid must contain exactly 3 elements",
            "CharmmContext.setFFTGrid(grid)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setFFTGrid rejects nonpositive X dimension",
            lambda: context.setFFTGrid((0, 32, 32)),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "NFFTX must be positive; observed 0",
            "CharmmContext.setFFTGrid(grid)",
        )
        apo_test.expect_apo_error(
            "CharmmContext.setPmeSplineOrder rejects zero",
            lambda: context.setPmeSplineOrder(0),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "PME spline order must be positive; observed 0",
            "CharmmContext.setPmeSplineOrder(order)",
        )

        apo_test.expect_exception(
            "CharmmContext.setRandomSeed rejects negative values",
            ValueError,
            lambda: context.setRandomSeed(-1),
        )
        apo_test.expect_exception(
            "CharmmContext.setRandomSeed rejects values above uint64_t",
            ValueError,
            lambda: context.setRandomSeed(2**64),
        )

        replacement_parameters = apo.CharmmParameters(parameter_path)
        replacement_parameters.close()
        apo_test.expect_exception(
            "CharmmContext.setPrm rejects a closed CharmmParameters handle",
            RuntimeError,
            lambda: context.setPrm(replacement_parameters),
        )
    finally:
        context.close()
        water_psf.close()
        force_manager.close()

    return


def main(argc: int, argv: list[str]) -> int:
    repo_root: Path = apo_test.get_repo_root()
    output_dir: Path = repo_root / "test/pytest"
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_path: str = apo_test.require_file(
        apo_test.get_toppar_dir() / "toppar_water_ions.str"
    )
    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")
    crd_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.cor")

    water_psf_path: Path = output_dir / "tmp_python_api_charmm_context_water.psf"
    velocity_path: Path = output_dir / "tmp_python_api_charmm_context.vel"
    velocity_mismatch_path: Path = (
        output_dir / "tmp_python_api_charmm_context_count_mismatch.vel"
    )
    velocity_bad_count_path: Path = (
        output_dir / "tmp_python_api_charmm_context_bad_count.vel"
    )
    velocity_bad_record_path: Path = (
        output_dir / "tmp_python_api_charmm_context_bad_record.vel"
    )
    missing_velocity_path: Path = (
        output_dir / "tmp_python_api_charmm_context_missing.vel"
    )

    temporary_paths: tuple[Path, ...] = (
        water_psf_path,
        velocity_path,
        velocity_mismatch_path,
        velocity_bad_count_path,
        velocity_bad_record_path,
        missing_velocity_path,
    )

    for path in temporary_paths:
        apo_test.remove_if_exists(path)

    parameters: apo.CharmmParameters | None = None
    psf: apo.CharmmPsf | None = None
    crd: apo.CharmmCrd | None = None

    try:
        apo_test.write_text_file(water_psf_path, WATER_PSF_TEXT)
        apo_test.write_text_file(velocity_path, VELOCITY_FILE_TEXT)
        apo_test.write_text_file(
            velocity_mismatch_path,
            VELOCITY_COUNT_MISMATCH_TEXT,
        )
        apo_test.write_text_file(velocity_bad_count_path, VELOCITY_BAD_COUNT_TEXT)
        apo_test.write_text_file(velocity_bad_record_path, VELOCITY_BAD_RECORD_TEXT)

        parameters = apo.CharmmParameters(parameter_path)
        psf = apo.CharmmPsf(psf_path)
        crd = apo.CharmmCrd(crd_path)

        check_construction_and_default_state(parameters, psf)
        check_object_setters(parameter_path, psf_path, parameters, psf)
        check_array_state_round_trips(parameters, psf, crd)
        check_velocity_file_loading(
            parameters,
            psf,
            velocity_path,
            velocity_mismatch_path,
            velocity_bad_count_path,
            velocity_bad_record_path,
            missing_velocity_path,
        )
        check_configuration_and_initialized_state(parameters, psf, crd)
        check_random_velocity_assignment_and_temperature(parameters, psf)
        check_holonomic_constraints(parameters, water_psf_path)
        check_validation(parameter_path, parameters, psf, water_psf_path)
    finally:
        print("Cleaning up CharmmContext Python API test files...")

        if crd is not None:
            crd.close()
        if psf is not None:
            psf.close()
        if parameters is not None:
            parameters.close()

        for path in temporary_paths:
            apo_test.remove_if_exists(path)

    print("\033[32m" + "PASS: CharmmContext Python API tests completed." + "\033[0m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
