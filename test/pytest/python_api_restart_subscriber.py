# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import apocharmm as apo

import apo_test_helpers as apo_test

BOX_DIMENSIONS: list[float] = [50.0, 50.0, 50.0]
RANDOM_SEED: int = 314159
REFERENCE_TEMPERATURE: float = 300.0
TIME_STEP: float = 0.001
NUM_STEPS: int = 3
REPORT_FREQUENCY: int = NUM_STEPS
NOSE_HOOVER_PISTON_MASS: float = 500.0
LANGEVIN_THERMOSTAT_FRICTION: float = 1.0
LANGEVIN_PISTON_MASS: list[float] = [5.0]
LANGEVIN_PISTON_FRICTION: float = 5.0
LANGEVIN_RNG_SEED: int = 987654321
REFERENCE_PRESSURE_TENSOR: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
# Covers CHARMM's fifteen significant digits and one resumed step in this fixture
TOLERANCE: float = 1.0e-10

IntegratorFactory = Callable[[apo.CharmmContext], apo.CudaIntegrator]


def assert_restart_file_has_required_section(path: Path) -> None:
    text: str = path.read_text(encoding="utf-8")

    # CHARMM's (A4,2I6,2X,A4,2X,A4) header occupies offsets 0-27; apoCHARMM adds
    # its marker at offsets 32-34, past the DHDGB field READYN may read.
    header: str = text.splitlines()[0]
    apo_test.assert_equal("restart header record type", header[:4], "REST")
    apo_test.assert_equal("restart header NSED/DHDGB fields", header[24:32], " " * 8)
    apo_test.assert_equal("restart header APO marker", header[32:35], "APO")

    for section in (
        "!CRYSTAL PARAMETERS",
        "!NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL",
        "!ENERGIES and STATISTICS",
        "!XOLD, YOLD, ZOLD",
        "!VX, VY, VZ",
        "!X, Y, Z",
    ):
        if section not in text:
            raise AssertionError(f"restart file is missing section: {section}")

    return


def create_context(assign_velocities: bool) -> apo.CharmmContext:
    prm_path: str = apo_test.require_file(
        apo_test.get_toppar_dir() / "toppar_water_ions.str"
    )
    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")
    crd_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.cor")

    prm = apo.CharmmParameters(prm_path)
    psf = apo.CharmmPsf(psf_path)
    crd = apo.CharmmCrd(crd_path)

    ctx = apo.CharmmContext(psf, prm)
    ctx.setBoxDimensions(BOX_DIMENSIONS)
    ctx.setCrystalType(apo.CrystalType.CUBIC)
    ctx.setCoordinates(crd)
    ctx.useHolonomicConstraints(False)

    if assign_velocities == True:
        ctx.setRandomSeed(RANDOM_SEED)
        ctx.assignVelocitiesAtTemperature(REFERENCE_TEMPERATURE)

    return ctx


def create_nose_hoover_integrator(
    ctx: apo.CharmmContext,
) -> apo.CudaNoseHooverIntegrator:
    integrator = apo.CudaNoseHooverIntegrator(TIME_STEP)
    integrator.setReferenceTemperature(REFERENCE_TEMPERATURE)
    integrator.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS)
    integrator.useOldTemperature(False)
    integrator.setCharmmContext(ctx)
    return integrator


def create_langevin_thermostat_integrator(
    ctx: apo.CharmmContext,
) -> apo.CudaLangevinThermostatIntegrator:
    integrator = apo.CudaLangevinThermostatIntegrator(TIME_STEP)
    integrator.setReferenceTemperature(REFERENCE_TEMPERATURE)
    integrator.setThermostatFriction(LANGEVIN_THERMOSTAT_FRICTION)
    integrator.setThermostatRngSeed(LANGEVIN_RNG_SEED)
    integrator.setCharmmContext(ctx)
    return integrator


def create_langevin_piston_integrator(
    ctx: apo.CharmmContext,
) -> apo.CudaLangevinPistonIntegrator:
    integrator = apo.CudaLangevinPistonIntegrator(TIME_STEP)
    integrator.useNoseHooverThermostat(True)
    integrator.setReferenceTemperature(REFERENCE_TEMPERATURE)
    integrator.setNoseHooverPistonMass(NOSE_HOOVER_PISTON_MASS)
    integrator.useOldTemperature(False)
    integrator.setReferencePressure(REFERENCE_PRESSURE_TENSOR)
    integrator.setConstantSurfaceTension(False)
    integrator.setLangevinPistonFrictionSeed(LANGEVIN_RNG_SEED)
    integrator.setCharmmContext(ctx)
    integrator.setLangevinPistonMass(LANGEVIN_PISTON_MASS)
    integrator.setLangevinPistonFriction(LANGEVIN_PISTON_FRICTION)
    return integrator


def check_restart_equivalence(
    label: str, rst_path: Path, create_integrator: IntegratorFactory
) -> None:
    print(f"Checking {label} restart trajectory equivalence...")

    ctx1 = create_context(assign_velocities=True)
    ctx2 = create_context(assign_velocities=True)
    ctx3 = create_context(assign_velocities=False)

    integrator1 = create_integrator(ctx1)
    integrator2 = create_integrator(ctx2)
    integrator3 = create_integrator(ctx3)

    integrator1.propagate(NUM_STEPS + 1)

    rst = apo.RestartSubscriber(rst_path, REPORT_FREQUENCY)
    integrator2.subscribe(rst)
    integrator2.propagate(NUM_STEPS)
    integrator2.unsubscribe(rst)
    assert_restart_file_has_required_section(rst_path)
    integrator2.propagate(1)

    integrator3.initializeFromRestartFile(rst_path)
    integrator3.propagate(1)

    apo_test.assert_nested_sequence_close(
        f"{label} ctx1 vs ctx2 coordinates/charges",
        ctx1.getCoordinatesCharges(),
        ctx2.getCoordinatesCharges(),
        TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        f"{label} ctx1 vs ctx3 coordinates/charges",
        ctx1.getCoordinatesCharges(),
        ctx3.getCoordinatesCharges(),
        TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        f"{label} ctx2 vs ctx3 coordinates/charges",
        ctx2.getCoordinatesCharges(),
        ctx3.getCoordinatesCharges(),
        TOLERANCE,
    )

    apo_test.assert_nested_sequence_close(
        f"{label} ctx1 vs ctx2 velocities/masses",
        ctx1.getVelocityMass(),
        ctx2.getVelocityMass(),
        TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        f"{label} ctx1 vs ctx3 velocities/masses",
        ctx1.getVelocityMass(),
        ctx3.getVelocityMass(),
        TOLERANCE,
    )
    apo_test.assert_nested_sequence_close(
        f"{label} ctx2 vs ctx3 velocities/masses",
        ctx2.getVelocityMass(),
        ctx3.getVelocityMass(),
        TOLERANCE,
    )

    integrator1.close()
    integrator2.close()
    integrator3.close()

    rst.close()

    return


def check_construction_and_validation(output_dir: Path) -> None:
    print("Checking RestartSubscriber construction and validation...")

    default_path: Path = output_dir / "tmp_python_api_restart_default.rst"
    frequency_path: Path = output_dir / "tmp_python_api_restart_frequency.rst"

    apo_test.remove_if_exists(default_path)
    apo_test.remove_if_exists(frequency_path)

    default_rst = apo.RestartSubscriber(default_path)
    apo_test.assert_equal(
        "default RestartSubscriber report frequency",
        default_rst.getReportFrequency(),
        1000,
    )
    default_rst.close()

    rst = apo.RestartSubscriber(frequency_path, REPORT_FREQUENCY)
    apo_test.assert_equal(
        "RestartSubscriber report frequency", rst.getReportFrequency(), REPORT_FREQUENCY
    )
    rst.setReportFrequency(1)
    apo_test.assert_equal(
        "RestartSubscriber updated report frequency", rst.getReportFrequency(), 1
    )
    rst.close()

    zero_error = apo_test.expect_invalid_argument(
        "RestartSubscriber rejects zero report frequency",
        lambda: apo.RestartSubscriber(output_dir / "tmp_zero.rst", 0),
        "Subscriber report frequency must be positive; observed 0",
    )
    apo_test.assert_equal(
        "RestartSubscriber zero report frequency context",
        zero_error.context,
        "RestartSubscriber construction",
    )

    negative_error = apo_test.expect_invalid_argument(
        "RestartSubscriber rejects negative report frequency",
        lambda: apo.RestartSubscriber(output_dir / "tmp_negative.rst", -1),
        "Subscriber report frequency must be positive; observed -1",
    )
    apo_test.assert_equal(
        "RestartSubscriber negative report frequency context",
        negative_error.context,
        "RestartSubscriber construction",
    )

    empty_path_error = apo_test.expect_invalid_argument(
        "RestartSubscriber rejects empty output path",
        lambda: apo.RestartSubscriber(""),
        "Output file name must not be empty",
    )
    apo_test.assert_equal(
        "RestartSubscriber empty output path context",
        empty_path_error.context,
        "RestartSubscriber construction",
    )

    apo_test.expect_exception(
        "RestartSubscriber rejects too-large report frequency",
        ValueError,
        lambda: apo.RestartSubscriber(output_dir / "tmp_large.rst", 2**31),
    )
    apo_test.expect_exception(
        "RestartSubscriber rejects too-small report frequency",
        ValueError,
        lambda: apo.RestartSubscriber(output_dir / "tmp_small.rst", -(2**31) - 1),
    )

    missing_directory: Path = output_dir / "missing_dir"

    default_missing_error = apo_test.expect_invalid_argument(
        "RestartSubscriber default constructor rejects missing output directory",
        lambda: apo.RestartSubscriber(missing_directory / "tmp.rst"),
        f"Output directory does not exist: {missing_directory}",
    )
    apo_test.assert_equal(
        "RestartSubscriber default missing directory context",
        default_missing_error.context,
        "RestartSubscriber construction",
    )

    frequency_missing_error = apo_test.expect_invalid_argument(
        "RestartSubscriber frequency constructor rejects missing output directory",
        lambda: apo.RestartSubscriber(
            missing_directory / "tmp_frequency.rst", REPORT_FREQUENCY
        ),
        f"Output directory does not exist: {missing_directory}",
    )
    apo_test.assert_equal(
        "RestartSubscriber frequency missing directory context",
        frequency_missing_error.context,
        "RestartSubscriber construction",
    )

    apo_test.remove_if_exists(default_path)
    apo_test.remove_if_exists(frequency_path)

    return


def check_close_invalidates_handle(output_dir: Path) -> None:
    print("Checking RestartSubscriber close invalidates handle...")

    rst_path: Path = output_dir / "tmp_python_api_rst_closed.rst"
    apo_test.remove_if_exists(rst_path)

    rst = apo.RestartSubscriber(rst_path, REPORT_FREQUENCY)
    rst.close()

    apo_test.expect_exception(
        "RestartSubscriber.getReportFrequency rejects closed handle",
        RuntimeError,
        lambda: rst.getReportFrequency(),
    )

    apo_test.remove_if_exists(rst_path)

    return


def main(argc: int, argv: list[str]) -> int:
    output_dir: Path = apo_test.get_repo_root() / "test/pytest"
    output_dir.mkdir(parents=True, exist_ok=True)

    rst_files: tuple[Path, ...] = (
        output_dir / "tmp_python_api_restart_nose_hoover.rst",
        output_dir / "tmp_python_api_restart_langevin_thermostat.rst",
        output_dir / "tmp_python_api_langevin_piston.rst",
    )

    try:
        check_construction_and_validation(output_dir)
        check_close_invalidates_handle(output_dir)

        check_restart_equivalence(
            "Nose-Hoover", rst_files[0], create_nose_hoover_integrator
        )
        check_restart_equivalence(
            "Langevin thermostat", rst_files[1], create_langevin_thermostat_integrator
        )
        check_restart_equivalence(
            "Langevin piston", rst_files[2], create_langevin_piston_integrator
        )
    finally:
        print("Cleaning up RestartSubscriber Python API test files...")
        for path in rst_files:
            apo_test.remove_if_exists(path)

    print(
        "\033[32m" + "PASS: RestartSubscriber Python API tests completed." + "\033[0m"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
