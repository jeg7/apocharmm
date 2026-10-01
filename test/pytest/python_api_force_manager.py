# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from __future__ import annotations

import math
from pathlib import Path
import sys

import apocharmm as apo

import apo_test_helpers as apo_test

TOLERANCE: float = 1.0e-7
BOX_DIMENSIONS: tuple[float, float, float] = (20.0, 21.0, 22.0)
FFT_GRID: tuple[int, int, int] = (32, 34, 36)


class TestForce:
    def __init__(self) -> None:
        self.subscribe_calls: list[tuple[apo.ForceManager, str | None]] = []
        self.unsubscribe_calls: list[apo.ForceManager] = []
        return

    def _subscribe_to_force_manager(
        self, force_manager: apo.ForceManager, force_tag: str | None = None
    ) -> None:
        self.subscribe_calls.append((force_manager, force_tag))
        return

    def _unsubscribe_from_force_manager(self, force_manager: apo.ForceManager) -> None:
        self.unsubscribe_calls.append(force_manager)
        return


def create_force_manager() -> apo.ForceManager:
    prm_path: str = apo_test.require_file(
        apo_test.get_toppar_dir() / "toppar_water_ions.str"
    )
    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")

    prm = apo.CharmmParameters(prm_path)
    psf = apo.CharmmPsf(psf_path)

    return apo.ForceManager(psf, prm)


def check_default_state() -> None:
    print("Checking ForceManager default state...")

    fm = create_force_manager()

    apo_test.assert_equal("ForceManager.getNumAtoms", fm.getNumAtoms(), 2)
    apo_test.assert_equal("ForceManager.isInitialized", fm.isInitialized(), False)
    apo_test.assert_equal("ForceManager.isComposite", fm.isComposite(), False)

    apo_test.assert_sequence_close(
        "ForceManager.getBoxDimensions default",
        fm.getBoxDimensions(),
        (-9999.9999, -9999.9999, -9999.9999),
        TOLERANCE,
    )

    apo_test.assert_close(
        "ForceManager.getKappa default", fm.getKappa(), 0.34, TOLERANCE
    )
    apo_test.assert_close(
        "ForceManager.getCutoff default", fm.getCutoff(), 14.0, TOLERANCE
    )
    apo_test.assert_close(
        "ForceManager.getCtonnb default", fm.getCtonnb(), 12.0, TOLERANCE
    )
    apo_test.assert_close(
        "ForceManager.getCtofnb default", fm.getCtofnb(), 10.0, TOLERANCE
    )
    apo_test.assert_equal(
        "ForceManager.getFFTGrid default", fm.getFFTGrid(), (-1, -1, -1)
    )
    apo_test.assert_equal(
        "ForceManager.getPmeSplineOrder default", fm.getPmeSplineOrder(), 4
    )
    apo_test.assert_equal(
        "ForceManager.getPeriodicBoundaryCondition default",
        fm.getPeriodicBoundaryCondition(),
        apo.PeriodicBoundaryCondition.P1,
    )
    apo_test.assert_equal(
        "ForceManager.getVdwType default", fm.getVdwType(), apo.VdwType.VFSW
    )

    return


def check_setters_and_getters() -> None:
    print("Checking ForceManager setter/getter round trips...")

    fm = create_force_manager()

    fm.setBoxDimensions(BOX_DIMENSIONS)
    fm.setKappa(0.45)
    fm.setCutoff(9.0)
    fm.setCtonnb(8.5)
    fm.setCtofnb(7.5)
    fm.setFFTGrid(FFT_GRID)
    fm.setPmeSplineOrder(6)
    fm.setPeriodicBoundaryCondition(apo.PeriodicBoundaryCondition.P21)
    fm.setVdwType(apo.VdwType.DBEXP)
    fm.setPrintEnergyDecomposition(True)
    fm.setPrintEnergyDecomposition(False)

    apo_test.assert_sequence_close(
        "ForceManager.getBoxDimensions",
        fm.getBoxDimensions(),
        BOX_DIMENSIONS,
        TOLERANCE,
    )
    apo_test.assert_close("ForceManager.getKappa", fm.getKappa(), 0.45, TOLERANCE)
    apo_test.assert_close("ForceManager.getCutoff", fm.getCutoff(), 9.0, TOLERANCE)
    apo_test.assert_close("ForceManager.getCtonnb", fm.getCtonnb(), 8.5, TOLERANCE)
    apo_test.assert_close("ForceManager.getCtofnb", fm.getCtofnb(), 7.5, TOLERANCE)
    apo_test.assert_equal("ForceManager.getFFTGRid", fm.getFFTGrid(), FFT_GRID)
    apo_test.assert_equal("ForceManager.getPmeSplineOrder", fm.getPmeSplineOrder(), 6)
    apo_test.assert_equal(
        "ForceManager.getPeriodicBoundaryCondition",
        fm.getPeriodicBoundaryCondition(),
        apo.PeriodicBoundaryCondition.P21,
    )
    apo_test.assert_equal("ForceManager.getVdwType", fm.getVdwType(), apo.VdwType.DBEXP)

    for vdw_type in (
        apo.VdwType.VSH,
        apo.VdwType.VSW,
        apo.VdwType.VFSW,
        apo.VdwType.VGSH,
        apo.VdwType.CUT,
        apo.VdwType.DBEXP,
    ):
        fm.setVdwType(vdw_type)
        apo_test.assert_equal(
            f"ForceManager.getVdwType {vdw_type}", fm.getVdwType(), vdw_type
        )

    return


def check_validation() -> None:
    print("Checking ForceManager validation...")

    prm_path: str = apo_test.require_file(
        apo_test.get_toppar_dir() / "toppar_water_ions.str"
    )
    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")

    prm = apo.CharmmParameters(prm_path)
    psf = apo.CharmmPsf(psf_path)

    fm = apo.ForceManager(psf, prm)

    apo_test.expect_exception(
        "ForceManager rejects non-CharmmPsf psf",
        TypeError,
        lambda: apo.ForceManager(object(), prm),
    )
    apo_test.expect_exception(
        "ForceManager rejects non-CharmmParameters parameters",
        TypeError,
        lambda: apo.ForceManager(psf, object()),
    )

    length_error = apo_test.expect_invalid_argument(
        "ForceManager.setBoxDimensions rejects short input",
        lambda: fm.setBoxDimensions([1.0, 2.0]),
        "apo_force_manager_set_box_dimensions: box_dimensions must contain exactly 3 elements",
    )
    apo_test.assert_equal(
        "ForceManager.setBoxDimensions error context",
        length_error.context,
        "ForceManager.setBoxDimensions(box_dimensions)",
    )

    apo_test.expect_invalid_argument(
        "ForceManager.setBoxDimensions rejects long input",
        lambda: fm.setBoxDimensions([1.0, 2.0, 3.0, 4.0]),
        "apo_force_manager_set_box_dimensions: box_dimensions must contain exactly 3 elements",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setBoxDimensions rejects negative dimensions",
        lambda: fm.setBoxDimensions([20.0, -1.0, 20.0]),
        "apoCHARMM error [InvalidArgument]: Box dimension at index 1 must be positive; observed -1.000000",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setBoxDimensions rejects zero dimensions",
        lambda: fm.setBoxDimensions([20.0, 0.0, 20.0]),
        "apoCHARMM error [InvalidArgument]: Box dimension at index 1 must be positive; observed 0.000000",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setBoxDimensions rejects NaN",
        lambda: fm.setBoxDimensions([20.0, math.nan, 20.0]),
        "apoCHARMM error [InvalidArgument]: Box dimension at index 1 must be finite; observed nan",
    )

    kappa_error = apo_test.expect_invalid_argument(
        "ForceManager.setKappa rejects NaN",
        lambda: fm.setKappa(math.nan),
        "apoCHARMM error [InvalidArgument]: Kappa must be finite; observed nan",
    )
    apo_test.assert_equal(
        "ForceManager.setKappa error context",
        kappa_error.context,
        "ForceManager.setKappa(kappa)",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setKappa rejects negative values",
        lambda: fm.setKappa(-1.0),
        "apoCHARMM error [InvalidArgument]: Kappa must be non-negative; observed -1.000000",
    )

    apo_test.expect_invalid_argument(
        "ForceManager.setCutoff rejects NaN",
        lambda: fm.setCutoff(math.nan),
        "apoCHARMM error [InvalidArgument]: Cutoff must be finite; observed nan",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setCutoff rejects zero",
        lambda: fm.setCutoff(0.0),
        "apoCHARMM error [InvalidArgument]: Cutoff must be positive; observed 0.000000",
    )

    apo_test.expect_invalid_argument(
        "ForceManager.setCtonnb rejects NaN",
        lambda: fm.setCtonnb(math.nan),
        "apoCHARMM error [InvalidArgument]: Ctonnb must be finite; observed nan",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setCtonnb rejects zero",
        lambda: fm.setCtonnb(0.0),
        "apoCHARMM error [InvalidArgument]: Ctonnb must be positive; observed 0.000000",
    )

    apo_test.expect_invalid_argument(
        "ForceManager.setCtofnb rejects NaN",
        lambda: fm.setCtofnb(math.nan),
        "apoCHARMM error [InvalidArgument]: Ctofnb must be finite; observed nan",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setCtofnb rejects zero",
        lambda: fm.setCtofnb(0.0),
        "apoCHARMM error [InvalidArgument]: Ctofnb must be positive; observed 0.000000",
    )

    apo_test.expect_invalid_argument(
        "ForceManager.setFFTGrid rejects short input",
        lambda: fm.setFFTGrid([32, 34]),
        "apo_force_manager_set_fft_grid: grid must contain exactly 3 elements",
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setPmeSplineOrder rejects zero",
        lambda: fm.setPmeSplineOrder(0),
        "apoCHARMM error [InvalidArgument]: PME spline order must be positive; observed 0",
    )

    apo_test.expect_invalid_argument(
        "ForceManager.setPeriodicBoundaryCondition rejects UNSET",
        lambda: fm.setPeriodicBoundaryCondition(apo.PeriodicBoundaryCondition.UNSET),
        "apoCHARMM error [InvalidArgument]: Periodic boundary condition must "
        "be PBC::P1 or PBC::P21; observed 0",
    )
    apo_test.assert_equal(
        "ForceManager PBC remains unchanged after rejected UNSET",
        fm.getPeriodicBoundaryCondition(),
        apo.PeriodicBoundaryCondition.P1,
    )
    apo_test.expect_exception(
        "ForceManager.setPeriodicBoundaryCondition rejects invalid enum value",
        ValueError,
        lambda: fm.setPeriodicBoundaryCondition(99),
    )
    apo_test.expect_exception(
        "ForceManager.setVdwType rejects invalid enum value",
        ValueError,
        lambda: fm.setVdwType(99),
    )
    apo_test.expect_invalid_argument(
        "ForceManager.setVdwType rejects VdwType.NONE through C ABI",
        lambda: fm.setVdwType(apo.VdwType.NONE),
        "apoCHARMM error [InvalidArgument]: Van der Waals type must be in [1, 6]; observed 0",
    )

    return


def check_subscription_wrapper() -> None:
    print("Checking ForceManager Python subscription bookkeeping...")

    fm = create_force_manager()

    force = TestForce()
    fm.subscribe(force)

    apo_test.assert_equal(
        "ForceManager.subscribe call count", len(force.subscribe_calls), 1
    )
    apo_test.assert_equal(
        "ForceManager.subscribe manager identity",
        force.subscribe_calls[0][0] is fm,
        True,
    )
    apo_test.assert_equal(
        "ForceManager.subscribe default tag", force.subscribe_calls[0][1], None
    )

    apo_test.expect_exception(
        "ForceManager.subscribe rejects duplicate force",
        ValueError,
        lambda: fm.subscribe(force),
    )

    fm.unsubscribe(force)

    apo_test.assert_equal(
        "ForceManager.unsubscribe call count", len(force.unsubscribe_calls), 1
    )
    apo_test.assert_equal(
        "ForceManager.unsubscribe manager identity",
        force.unsubscribe_calls[0] is fm,
        True,
    )

    tagged_force = TestForce()
    fm.subscribe(tagged_force, "custom_tag")

    apo_test.assert_equal(
        "ForceManager.subscribe custom tag",
        tagged_force.subscribe_calls[0][1],
        "custom_tag",
    )
    fm.unsubscribe(tagged_force)

    apo_test.expect_exception(
        "ForceManager.subscribe rejects non-subscribable object",
        TypeError,
        lambda: fm.subscribe(object()),
    )
    apo_test.expect_exception(
        "ForceManager.subscribe rejects non-string force_tag",
        TypeError,
        lambda: fm.subscribe(TestForce(), 1),
    )
    apo_test.expect_exception(
        "ForceManager.subscribe rejects empty force_tag",
        ValueError,
        lambda: fm.subscribe(TestForce(), ""),
    )
    apo_test.expect_exception(
        "ForceManager.unsubscribe rejects non-subscribable object",
        TypeError,
        lambda: fm.unsubscribe(object()),
    )
    apo_test.expect_exception(
        "ForceManager.unsubscribe rejects unregistered force",
        ValueError,
        lambda: fm.unsubscribe(TestForce()),
    )

    return


def main(argc: int, argv: list[str]) -> int:
    check_default_state()
    check_setters_and_getters()
    check_validation()
    check_subscription_wrapper()

    print("\033[32m" + "PASS: ForceManager Python API tests completed." + "\033[0m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
