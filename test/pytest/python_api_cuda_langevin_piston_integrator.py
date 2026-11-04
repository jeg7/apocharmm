# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from __future__ import annotations

from pathlib import Path
import sys

import apocharmm as apo

import apo_test_helpers as apo_test

BOX_DIMENSIONS: list[float] = [50.0, 50.0, 50.0]
RANDOM_SEED: int = 314159
TEMPERATURE: float = 300.0
TIME_STEP: float = 0.001
REFERENCE_TEMPERATURE: float = 315.0
NOSE_HOOVER_PISTON_MASS: float = 500.0
LANGEVIN_PISTON_MASS: list[float] = [5.0]
LANGEVIN_PISTON_FRICTION: float = 5.0
LANGEVIN_PISTON_SEED: int = 987654321
REFERENCE_PRESSURE_TENSOR: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
NUM_STEPS: int = 3
TOLERANCE: float = 1.0e-10
DETERMINISTIC_TOLERANCE: float = 1.0e-14


def create_context(configure_crystal: bool = True) -> apo.CharmmContext:
    prm_path: str = apo_test.require_file(
        apo_test.get_toppar_dir() / "toppar_water_ions.str"
    )
    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")
    crd_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.cor")

    prm = apo.CharmmParameters(prm_path)
    psf = apo.CharmmPsf(psf_path)
    crd = apo.CharmmCrd(crd_path)

    ctx = apo.CharmmContext(psf, prm)
    if configure_crystal:
        ctx.setCrystalType(apo.CrystalType.CUBIC)
    ctx.setBoxDimensions(BOX_DIMENSIONS)
    ctx.setCoordinates(crd)
    ctx.useHolonomicConstraints(False)
    ctx.setRandomSeed(RANDOM_SEED)
    ctx.assignVelocitiesAtTemperature(TEMPERATURE)

    return ctx


def create_integrator(context: apo.CharmmContext) -> apo.CudaLangevinPistonIntegrator:
    integrator = apo.CudaLangevinPistonIntegrator(TIME_STEP)
    integrator.useNoseHooverThermostat(True)
    integrator.setReferenceTemperature(REFERENCE_TEMPERATURE)
    integrator.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS)
    integrator.setReferencePressure(REFERENCE_PRESSURE_TENSOR)
    integrator.setLangevinPistonFrictionSeed(LANGEVIN_PISTON_SEED)
    integrator.setCharmmContext(context)
    integrator.setLangevinPistonMass(LANGEVIN_PISTON_MASS)
    integrator.setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION)
    return integrator


def check_setters_and_getters() -> None:
    print("Checking CudaLangevinPistonIntegrator setters/getters...")

    context = create_context()
    integrator = create_integrator(context)

    apo_test.assert_close(
        "CudaLangevinPistonIntegrator.getReferenceTemperature",
        integrator.getReferenceTemperature(),
        REFERENCE_TEMPERATURE,
        TOLERANCE,
    )
    apo_test.assert_close(
        "CudaLangevinPistonIntegrator.getNoseHooverPistonMass",
        integrator.getNoseHooverPistonMass(),
        NOSE_HOOVER_PISTON_MASS,
        TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        "CudaLangevinPistonIntegrator.getReferencePressureTensor",
        integrator.getReferencePressureTensor(),
        REFERENCE_PRESSURE_TENSOR,
        TOLERANCE,
    )
    apo_test.assert_equal(
        "CudaLangevinPistonIntegrator.getCrystalType",
        integrator.getCrystalType(),
        apo.CrystalType.CUBIC,
    )
    apo_test.assert_sequence_close(
        "CudaLangevinPistonIntegrator.getLangevinPistonMass",
        integrator.getLangevinPistonMass(),
        LANGEVIN_PISTON_MASS,
        TOLERANCE,
    )
    apo_test.assert_equal(
        "CudaLangevinPistonIntegrator.getLangevinPistonFrictionSeed",
        integrator.getLangevinPistonFrictionSeed(),
        LANGEVIN_PISTON_SEED,
    )

    integrator.resetAverages()

    apo_test.assert_close(
        "CudaLangevinPistonIntegrator.getAverageTemperature after reset",
        integrator.getAverageTemperature(),
        0.0,
        TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        "CudaLangevinPistonIntegrator.getAveragePressureTensor after reset",
        integrator.getAveragePressureTensor(),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        TOLERANCE,
    )
    apo_test.assert_close(
        "CudaLangevinPistonIntegrator.getAveragePressureScalar after reset",
        integrator.getAveragePressureScalar(),
        0.0,
        TOLERANCE,
    )

    integrator.close()

    return


def check_validation() -> None:
    print("Checking CudaLangevinPistonIntegrator validation...")

    apo_test.expect_apo_error(
        "constructor rejects zero time step",
        lambda: apo.CudaLangevinPistonIntegrator(0.0),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Time step must be positive; observed 0.000000",
        "CudaLangevinPistonIntegrator construction",
    )

    apo_test.expect_apo_error(
        "constructor rejects infinite time step",
        lambda: apo.CudaLangevinPistonIntegrator(float("inf")),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Time step must be finite; observed",
        "CudaLangevinPistonIntegrator construction",
    )

    integrator = apo.CudaLangevinPistonIntegrator(TIME_STEP)

    apo_test.expect_apo_error(
        "setReferencePressure rejects wrong row length",
        lambda: integrator.setReferencePressure(
            [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "pressure_tensor must contain exactly 9 elements",
        "CudaLangevinPistonIntegrator.setReferencePressure(pressure_tensor)",
    )

    apo_test.expect_apo_error(
        "setReferencePressure rejects wrong flattened length",
        lambda: integrator.setReferencePressure([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "pressure_tensor must contain exactly 9 elements",
        "CudaLangevinPistonIntegrator.setReferencePressure(pressure_tensor)",
    )

    apo_test.expect_apo_error(
        "setReferencePressure rejects non-finite entries",
        lambda: integrator.setReferencePressure(
            [
                [1.0, 0.0, 0.0],
                [0.0, float("inf"), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Reference pressure tensor values must be finite; non-finite value at index 4",
        "CudaLangevinPistonIntegrator.setReferencePressure(pressure_tensor)",
    )

    apo_test.expect_apo_error(
        "setReferenceTemperature rejects negative temperature",
        lambda: integrator.setReferenceTemperature(-1.0),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Reference temperature must be non-negative; observed -1.000000",
        "CudaLangevinPistonIntegrator.setReferenceTemperature(temperature)",
    )

    apo_test.expect_apo_error(
        "setNoseHooverPistonMass rejects negative mass",
        lambda: integrator.setNoseHooverPistonMass(-1.0),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Nose-Hoover piston mass must be non-negative; observed -1.000000",
        "CudaLangevinPistonIntegrator.setNoseHooverPistonMass(mass)",
    )

    apo_test.expect_apo_error(
        "setLangevinPistonMass rejects missing crystal type",
        lambda: integrator.setLangevinPistonMass([1.0]),
        apo.APO_STATUS_NOT_INITIALIZED,
        "Attach a CharmmContext with a crystal type before setting the Langevin piston mass",
        "CudaLangevinPistonIntegrator.setLangevinPistonMass(mass)",
    )

    apo_test.expect_apo_error(
        "setLangevinPistonFriction rejects missing context",
        lambda: integrator.setLangevinPistonFriction(1.0),
        apo.APO_STATUS_NOT_INITIALIZED,
        "Attach a CharmmContext with a crystal type before setting the Langevin piston friction",
        "CudaLangevinPistonIntegrator.setLangevinPistonFriction(friction)",
    )

    apo_test.expect_apo_error(
        "getInstantaneousTemperature rejects missing context",
        lambda: integrator.getInstantaneousTemperature(),
        apo.APO_STATUS_NOT_INITIALIZED,
        "CharmmContext must be set before computing instantaneous temperature",
        "CudaLangevinPistonIntegrator.getInstantaneousTemperature()",
    )

    apo_test.expect_exception(
        "setLangevinPistonFrictionSeed rejects negative seed",
        ValueError,
        lambda: integrator.setLangevinPistonFrictionSeed(-1),
    )

    configured_context = create_context()
    integrator.setCharmmContext(configured_context)

    apo_test.expect_apo_error(
        "setLangevinPistonMass rejects wrong length for cubic crystal",
        lambda: integrator.setLangevinPistonMass([1.0, 2.0]),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Langevin piston mass must contain 1 element(s); observed 2",
        "CudaLangevinPistonIntegrator.setLangevinPistonMass(mass)",
    )

    apo_test.expect_apo_error(
        "setLangevinPistonMass rejects negative mass",
        lambda: integrator.setLangevinPistonMass([-1.0]),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Langevin piston mass values must be non-negative; observed -1.000000 at index 0",
        "CudaLangevinPistonIntegrator.setLangevinPistonMass(mass)",
    )

    apo_test.expect_apo_error(
        "setLangevinPistonMass rejects infinite mass",
        lambda: integrator.setLangevinPistonMass([float("inf")]),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Langevin piston mass values must be finite; non-finite value at index 0",
        "CudaLangevinPistonIntegrator.setLangevinPistonMass(mass)",
    )

    apo_test.expect_apo_error(
        "setLangevinPistonFriction rejects negative friction",
        lambda: integrator.setLangevinPistonFriction(-1.0),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Langevin piston friction must be non-negative; observed -1.000000",
        "CudaLangevinPistonIntegrator.setLangevinPistonFriction(friction)",
    )

    apo_test.expect_apo_error(
        "setLangevinPistonFriction rejects infinite friction",
        lambda: integrator.setLangevinPistonFriction(float("inf")),
        apo.APO_STATUS_INVALID_ARGUMENT,
        "Langevin piston friction must be finite; observed",
        "CudaLangevinPistonIntegrator.setLangevinPistonFriction(friction)",
    )

    missing_crystal_integrator = apo.CudaLangevinPistonIntegrator(TIME_STEP)
    context = create_context(configure_crystal=False)

    apo_test.expect_apo_error(
        "setCharmmContext rejects missing crystal type",
        lambda: missing_crystal_integrator.setCharmmContext(context),
        apo.APO_STATUS_NOT_INITIALIZED,
        "Crystal type must be configured before this operation",
        "CudaIntegrator.setCharmmContext(context)",
    )

    integrator.close()
    missing_crystal_integrator.close()

    return


def check_crystal_phase_change() -> None:
    print("Checking CudaLangevinPistonIntegrator crystal phase changes...")

    context = create_context()
    integrator = apo.CudaLangevinPistonIntegrator(TIME_STEP)

    try:
        apo_test.expect_apo_error(
            "initialize requires an attached context",
            integrator.initialize,
            apo.APO_STATUS_NOT_INITIALIZED,
            "CharmmContext must be set before initialization",
            "CudaIntegrator.initialize()",
        )

        integrator.setCharmmContext(context)
        integrator.setLangevinPistonMass(LANGEVIN_PISTON_MASS)
        integrator.setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION)
        integrator.propagate(1)

        context.setCrystalType(apo.CrystalType.CUBIC)
        apo_test.assert_sequence_close(
            "unchanged selection preserves usable piston masses",
            integrator.getLangevinPistonMass(),
            LANGEVIN_PISTON_MASS,
            TOLERANCE,
        )

        context.setCrystalType(apo.CrystalType.ORTHORHOMBIC)
        apo_test.assert_equal(
            "piston crystal getter follows the context",
            integrator.getCrystalType(),
            apo.CrystalType.ORTHORHOMBIC,
        )

        stale: str = (
            "Langevin piston state does not match the context crystal type; reinitialize the integrator"
        )
        orth_masses: list[float] = [5.0, 6.0, 7.0]
        for label, operation, function in [
            (
                "stale piston mass getter",
                integrator.getLangevinPistonMass,
                "CudaLangevinPistonIntegrator.getLangevinPistonMass()",
            ),
            (
                "stale piston mass setter",
                lambda: integrator.setLangevinPistonMass(orth_masses),
                "CudaLangevinPistonIntegrator.setLangevinPistonMass(mass)",
            ),
            (
                "stale piston friction setter",
                lambda: integrator.setLangevinPistonFriction(6.0),
                "CudaLangevinPistonIntegrator.setLangevinPistonFriction(friction)",
            ),
            (
                "stale propagation",
                lambda: integrator.propagate(1),
                "CudaIntegrator.propagate(num_steps)",
            ),
            (
                "stale restart import",
                lambda: integrator.initializeFromRestartFile(
                    "unused_while_piston_state_is_stale.rst"
                ),
                "CudaIntegrator.initializeFromRestartFile(path)",
            ),
        ]:
            apo_test.expect_apo_error(
                label, operation, apo.APO_STATUS_NOT_INITIALIZED, stale, function
            )

        integrator.initialize()
        apo_test.assert_equal(
            "ORTH allocates three piston masses",
            len(integrator.getLangevinPistonMass()),
            3,
        )
        integrator.setLangevinPistonMass(orth_masses)
        integrator.initialize()
        apo_test.assert_sequence_close(
            "same-type initialize preserves masses",
            integrator.getLangevinPistonMass(),
            orth_masses,
            TOLERANCE,
        )
        integrator.propagate(1)
        context.setBoxDimensions((50.0, 51.0, 52.0))
        apo_test.expect_apo_error(
            "unequal ORTH box rejects reverse change to CUBIC",
            lambda: context.setCrystalType(apo.CrystalType.CUBIC),
            apo.APO_STATUS_INVALID_ARGUMENT,
            "Box dimensions are incompatible with the crystal type",
            "CharmmContext.setCrystalType(crystal_type)",
        )
        apo_test.assert_equal(
            "failed change preserves context crystal",
            context.getCrystalType(),
            apo.CrystalType.ORTHORHOMBIC,
        )
        apo_test.assert_sequence_close(
            "failed change leaves piston state usable",
            integrator.getLangevinPistonMass(),
            orth_masses,
            TOLERANCE,
        )
    finally:
        integrator.close()
        context.close()

    apo_test.expect_exception(
        "initialize rejects a closed wrapper", RuntimeError, integrator.initialize
    )

    return


def check_short_propagation() -> None:
    print("Checking CudaLangevinPistonIntegrator short propagation...")

    ctx = create_context()
    integrator = create_integrator(ctx)
    integrator.resetAverages()

    initial_coordinates_charges: list[list[float]] = ctx.getCoordinatesCharges()

    integrator.propagate(NUM_STEPS)

    final_coordinates_charges: list[list[float]] = ctx.getCoordinatesCharges()

    apo_test.assert_finite_temperature(
        "CharmmContext.computeTemperature", ctx.computeTemperature()
    )
    apo_test.assert_finite_temperature(
        "CudaLangevinPistonIntegrator.getInstantaneousTemperature",
        integrator.getInstantaneousTemperature(),
    )
    apo_test.assert_finite_temperature(
        "CudaLangevinPistonIntegrator.getAverageTemperature",
        integrator.getAverageTemperature(),
    )

    apo_test.assert_finite_nested_sequence(
        "CudaLangevinPistonIntegrator.getInstantaneousPressureTensor",
        integrator.getInstantaneousPressureTensor(),
    )
    apo_test.assert_finite_scalar(
        "CudaLangevinPistonIntegrator.getInstantaneousPressureScalar",
        integrator.getInstantaneousPressureScalar(),
    )
    apo_test.assert_finite_nested_sequence(
        "CudaLangevinPistonIntegrator.getAveragePressureTensor",
        integrator.getAveragePressureTensor(),
    )
    apo_test.assert_finite_scalar(
        "CudaLangevinPistonIntegrator.getAveragePressureScalar",
        integrator.getAveragePressureScalar(),
    )

    if final_coordinates_charges[0][:3] == initial_coordinates_charges[0][:3]:
        raise AssertionError("short propagation did not move atom 0 coordinates")

    integrator.close()

    return


def check_deterministic_trajectories() -> None:
    print("Checking deterministic Langevin piston trajectories...")

    ctx1 = create_context()
    ctx2 = create_context()

    apo_test.assert_nested_sequence_close(
        "initial coordinates/charges",
        ctx1.getCoordinatesCharges(),
        ctx2.getCoordinatesCharges(),
        DETERMINISTIC_TOLERANCE,
    )

    integrator1 = create_integrator(ctx1)
    integrator2 = create_integrator(ctx2)

    apo_test.assert_close(
        "deterministic initial reference temperature",
        integrator1.getReferenceTemperature(),
        integrator2.getReferenceTemperature(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        "deterministic initial reference pressure tensor",
        integrator1.getReferencePressureTensor(),
        integrator2.getReferencePressureTensor(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_sequence_close(
        "deterministic initial Langevin piston mass",
        integrator1.getLangevinPistonMass(),
        integrator2.getLangevinPistonMass(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_equal(
        "deterministic initial Langevin piston seed",
        integrator1.getLangevinPistonFrictionSeed(),
        integrator2.getLangevinPistonFrictionSeed(),
    )

    initial_coordinates_charges1: list[list[float]] = ctx1.getCoordinatesCharges()
    initial_coordinates_charges2: list[list[float]] = ctx2.getCoordinatesCharges()

    integrator1.propagate(NUM_STEPS)
    integrator2.propagate(NUM_STEPS)

    final_coordinates_charges1: list[list[float]] = ctx1.getCoordinatesCharges()
    final_coordinates_charges2: list[list[float]] = ctx2.getCoordinatesCharges()

    apo_test.assert_nested_sequence_close(
        "deterministic final coordinates/charges",
        final_coordinates_charges1,
        final_coordinates_charges2,
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_close(
        "deterministic final instantaneous temperature",
        integrator1.getInstantaneousTemperature(),
        integrator2.getInstantaneousTemperature(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_close(
        "deterministic final average temperature",
        integrator1.getAverageTemperature(),
        integrator2.getAverageTemperature(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        "determinstic final instantaneous pressure tensor",
        integrator1.getInstantaneousPressureTensor(),
        integrator2.getInstantaneousPressureTensor(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_close(
        "determinstic final instantaneous pressure scalar",
        integrator1.getInstantaneousPressureScalar(),
        integrator2.getInstantaneousPressureScalar(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        "determinstic final average pressure tensor",
        integrator1.getAveragePressureTensor(),
        integrator2.getAveragePressureTensor(),
        DETERMINISTIC_TOLERANCE,
    )
    apo_test.assert_close(
        "determinstic final average pressure scalar",
        integrator1.getAveragePressureScalar(),
        integrator2.getAveragePressureScalar(),
        DETERMINISTIC_TOLERANCE,
    )

    if final_coordinates_charges1[0][:3] == initial_coordinates_charges1[0][:3]:
        raise AssertionError("deterministic trajectory 1 did not move atom 0")
    if final_coordinates_charges2[0][:3] == initial_coordinates_charges2[0][:3]:
        raise AssertionError("deterministic trajectory 2 did not move atom 0")

    integrator1.close()
    integrator2.close()

    return


def main(argc: int, argv: list[str]) -> int:
    check_setters_and_getters()
    check_validation()
    check_crystal_phase_change()
    check_short_propagation()
    check_deterministic_trajectories()

    print(
        "\033[32m"
        + "PASS: CudaLangevinPistonIntegrator Python API tests completed."
        + "\033[0m"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
