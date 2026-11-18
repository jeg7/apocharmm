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
THERMOSTAT_FRICTION: float = 0.0
REPORT_FREQUENCY: int = 3
NUM_STEPS: int = 6
FIELD_TOLERANCE: float = 1.0e-5

SEPARATOR: str = (
    " ----------       ---------    ---------    ---------    ---------    ---------"
)


def create_context() -> apo.CharmmContext:
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
    ctx.setCoordinates(crd)
    ctx.useHolonomicConstraints(False)
    ctx.setRandomSeed(RANDOM_SEED)
    ctx.assignVelocitiesAtTemperature(TEMPERATURE)

    return ctx


def create_integrator(ctx: apo.CharmmContext) -> apo.CudaLangevinThermostatIntegrator:
    integrator = apo.CudaLangevinThermostatIntegrator(TIME_STEP)
    integrator.setReferenceTemperature(TEMPERATURE)
    integrator.setThermostatFriction(THERMOSTAT_FRICTION)
    integrator.setThermostatRngSeed(RANDOM_SEED)
    integrator.setCharmmContext(ctx)
    return integrator


def check_constructor_validation(output_dir: Path) -> None:
    print("Checking DynaSubscriber construction and validation...")

    default_path: Path = output_dir / "tmp_python_api_dyna_default.out"
    frequency_path: Path = output_dir / "tmp_python_api_dyna_frequency.out"

    apo_test.remove_if_exists(default_path)
    apo_test.remove_if_exists(frequency_path)

    default_dyna = apo.DynaSubscriber(default_path)
    apo_test.assert_equal(
        "default DynaSubscriber report frequency", default_dyna.getReportFrequency(), 10
    )
    default_dyna.close()

    dyna = apo.DynaSubscriber(frequency_path, REPORT_FREQUENCY)
    apo_test.assert_equal(
        "DynaSubscriber report frequency", dyna.getReportFrequency(), REPORT_FREQUENCY
    )
    dyna.setReportFrequency(2)
    apo_test.assert_equal(
        "DynaSubscriber updated report frequency", dyna.getReportFrequency(), 2
    )
    dyna.close()

    zero_error = apo_test.expect_invalid_argument(
        "DynaSubscriber rejects zero report frequency",
        lambda: apo.DynaSubscriber(output_dir / "tmp_zero.out", 0),
        "Subscriber report frequency must be positive; observed 0",
    )
    apo_test.assert_equal(
        "DynaSubscriber zero report frequency context",
        zero_error.context,
        "DynaSubscriber construction",
    )

    negative_error = apo_test.expect_invalid_argument(
        "DynaSubscriber rejects negative report frequency",
        lambda: apo.DynaSubscriber(output_dir / "tmp_negative.out", -1),
        "Subscriber report frequency must be positive; observed -1",
    )
    apo_test.assert_equal(
        "DynaSubscriber negative report frequency context",
        negative_error.context,
        "DynaSubscriber construction",
    )

    empty_path_error = apo_test.expect_invalid_argument(
        "DynaSubscriber rejects empty output path",
        lambda: apo.DynaSubscriber(""),
        "Output file name must not be empty",
    )
    apo_test.assert_equal(
        "DynaSubscriber empty output path context",
        empty_path_error.context,
        "DynaSubscriber construction",
    )

    apo_test.expect_exception(
        "DynaSubscriber rejects too-large report frequency",
        ValueError,
        lambda: apo.DynaSubscriber(output_dir / "tmp_large.out", 2**31),
    )
    apo_test.expect_exception(
        "DynaSubscriber rejects too-small report frequency",
        ValueError,
        lambda: apo.DynaSubscriber(output_dir / "tmp_small.out", -(2**31) - 1),
    )

    missing_directory: Path = output_dir / "missing_dir"
    missing_directory_error = apo_test.expect_invalid_argument(
        "DynaSubscriber rejects missing output directory",
        lambda: apo.DynaSubscriber(missing_directory / "tmp.out"),
        f"Output directory does not exist: {missing_directory}",
    )
    apo_test.assert_equal(
        "DynaSubscriber missing output directory context",
        missing_directory_error.context,
        "DynaSubscriber construction",
    )

    frequency_missing_directory_error = apo_test.expect_invalid_argument(
        "DynaSubscriber frequency constructor rejects missing output directory",
        lambda: apo.DynaSubscriber(
            missing_directory / "tmp_frequency.out", REPORT_FREQUENCY
        ),
        f"Output directory does not exist: {missing_directory}",
    )
    apo_test.assert_equal(
        "DynaSubscriber frequency missing output directory context",
        frequency_missing_directory_error.context,
        "DynaSubscriber construction",
    )

    open_error = apo_test.expect_exception(
        "DynaSubscriber maps native output open failure",
        apo.ApoCharmmError,
        lambda: apo.DynaSubscriber(output_dir),
    )
    apo_test.assert_equal(
        "DynaSubscriber output open failure status",
        open_error.status,
        apo.APO_STATUS_RUNTIME_ERROR,
    )
    apo_test.assert_equal(
        "DynaSubscriber output open failure status name",
        open_error.status_name,
        "APO_STATUS_RUNTIME_ERROR",
    )
    apo_test.assert_equal(
        "DynaSubscriber output open failure context",
        open_error.context,
        "DynaSubscriber construction",
    )

    expected_open_diagnostic: str = (
        f"Failed to open subscriber output file for writing: {output_dir}"
    )
    if expected_open_diagnostic not in open_error.native_diagnostic:
        raise AssertionError(
            "DynaSubscriber output open failure: expected native diagnostic "
            f"to contain {expected_open_diagnostic!r}, observed "
            f"{open_error.native_diagnostic!r}"
        )

    frequency_open_error = apo_test.expect_exception(
        "DynaSubscriber frequency constructor maps native output open failure",
        apo.ApoCharmmError,
        lambda: apo.DynaSubscriber(output_dir, REPORT_FREQUENCY),
    )
    apo_test.assert_equal(
        "DynaSubscriber frequency output open failure status",
        frequency_open_error.status,
        apo.APO_STATUS_RUNTIME_ERROR,
    )
    apo_test.assert_equal(
        "DynaSubscriber frequency output open failure status name",
        frequency_open_error.status_name,
        "APO_STATUS_RUNTIME_ERROR",
    )
    apo_test.assert_equal(
        "DynaSubscriber frequency output open failure context",
        frequency_open_error.context,
        "DynaSubscriber construction",
    )

    if expected_open_diagnostic not in frequency_open_error.native_diagnostic:
        raise AssertionError(
            "DynaSubscriber frequency output open failure: expected native "
            f"diagnostic to contain {expected_open_diagnostic!r}, observed "
            f"{frequency_open_error.native_diagnostic!r}"
        )

    apo_test.remove_if_exists(default_path)
    apo_test.remove_if_exists(frequency_path)

    return


def check_dyna_output(dyna_path: Path) -> None:
    print("Checking DynaSubscriber DYNA output...")

    apo_test.remove_if_exists(dyna_path)

    ctx = create_context()
    integrator = create_integrator(ctx)

    dyna = apo.DynaSubscriber(dyna_path, REPORT_FREQUENCY)
    integrator.subscribe(dyna)

    # Two calls form one CHARMM-style run: blocks at steps 0, 3, and 6, with a
    # single header.
    integrator.propagate(1)
    integrator.propagate(NUM_STEPS - 1)

    integrator.unsubscribe(dyna)
    dyna.close()
    integrator.close()

    apo_test.assert_file_created(dyna_path)

    lines: list[str] = dyna_path.read_text().splitlines()

    # The ions have no bonded terms, so there is no INTERN or CROSS line.
    header: list[str] = [
        "DYNA DYN: Step         Time      TOTEner        TOTKe       ENERgy  TEMPerature",
        "DYNA PROP:             GRMS      HFCTote        HFCKe       EHFCor        VIRKe",
        "DYNA EXTERN:        VDWaals         ELEC       HBONds          ASP         USER",
        "DYNA EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor       EWUTil",
        "DYNA PRESS:            VIRE         VIRI       PRESSE       PRESSI       VOLUme",
        SEPARATOR,
    ]
    block_size: int = 6
    num_blocks: int = NUM_STEPS // REPORT_FREQUENCY + 1

    apo_test.assert_equal(
        "DYNA line count", len(lines), len(header) + num_blocks * block_size
    )

    for i, expected in enumerate(header):
        apo_test.assert_equal(f"DYNA header line {i}", lines[i], expected)

    for block in range(num_blocks):
        first: int = len(header) + block * block_size
        apo_test.assert_equal(f"DYNA block {block} label", lines[first][:5], "DYNA>")
        apo_test.assert_equal(
            f"DYNA block {block} separator", lines[first + 5], SEPARATOR
        )

        values: list[float] = [float(token) for token in lines[first][5:].split()]
        apo_test.assert_equal(f"DYNA block {block} field count", len(values), 6)
        apo_test.assert_equal(
            f"DYNA block {block} step", values[0], float(block * REPORT_FREQUENCY)
        )
        apo_test.assert_close(
            f"DYNA block {block} time",
            values[1],
            block * REPORT_FREQUENCY * TIME_STEP,
            FIELD_TOLERANCE,
        )
        apo_test.assert_close(
            f"DYNA block {block} total energy",
            values[2],
            values[3] + values[4],
            2.0 * FIELD_TOLERANCE,
        )

    return


def check_close_invalidates_handle(output_dir: Path) -> None:
    print("Checking DynaSubscriber close invalidates handle...")

    dyna_path: Path = output_dir / "tmp_python_api_dyna_closed.out"
    apo_test.remove_if_exists(dyna_path)

    dyna = apo.DynaSubscriber(dyna_path, REPORT_FREQUENCY)
    dyna.close()

    apo_test.expect_exception(
        "DynaSubscriber.getReportFrequency rejects closed handle",
        RuntimeError,
        lambda: dyna.getReportFrequency(),
    )

    apo_test.remove_if_exists(dyna_path)

    return


def main(argc: int, argv: list[str]) -> int:
    output_dir: Path = apo_test.get_repo_root() / "test/pytest"
    output_dir.mkdir(parents=True, exist_ok=True)

    dyna_path: Path = output_dir / "tmp_python_api_dyna_subscriber.out"

    try:
        check_constructor_validation(output_dir)
        check_dyna_output(dyna_path)
        check_close_invalidates_handle(output_dir)
    finally:
        print("Cleaning up DynaSubscriber Python API test files...")
        apo_test.remove_if_exists(dyna_path)

    print("\033[32m" + "PASS: DynaSubscriber Python API tests completed." + "\033[0m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
