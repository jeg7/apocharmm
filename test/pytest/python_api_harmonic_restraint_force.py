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
import sys

import apocharmm as apo

import apo_test_helpers as apo_test

BOX_DIMENSIONS: list[float] = [50.0, 50.0, 50.0]
RANDOM_SEED: int = 314159
TEMPERATURE: float = 300.0
TIME_STEP: float = 0.001
FLOAT_RELATIVE_TOLERANCE: float = 64.0 * 2.0**-24


def create_system() -> tuple[
    apo.CharmmParameters,
    apo.CharmmPsf,
    apo.CharmmCrd,
    apo.ForceManager,
    apo.CharmmContext,
]:
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

    fm = ctx.getForceManager()

    return prm, psf, crd, fm, ctx


def create_configured_restraint(
    psf: apo.CharmmPsf, crd: apo.CharmmCrd
) -> apo.HarmonicRestraintForce:
    selector = apo.AtomSelector(psf)
    selection = selector.select("all")

    restraint = apo.HarmonicRestraintForce(psf)
    restraint.setSelection(selection)
    restraint.setReferenceCoordinates(crd.getCoordinates())
    restraint.setMasses(psf.getMasses())
    restraint.setForceConstant(10.0)
    restraint.setForceConstants([10.0 for _ in range(psf.getNumAtoms())])
    restraint.setBoxDimensions(BOX_DIMENSIONS)

    return restraint


def check_construction_and_setters() -> None:
    print("Checking HarmonicRestraintForce construction and setters...")

    _, psf, crd, _, _ = create_system()

    restraint = create_configured_restraint(psf, crd)

    apo_test.assert_equal(
        "HarmonicRestraintForce.default_force_tag", restraint.default_force_tag, "harm"
    )

    restraint.close()

    return


def check_subscription_and_short_propagation() -> None:
    print("Checking HarmonicRestraintForce ForceManager subscription...")

    _, psf, crd, fm, ctx = create_system()

    restraint = create_configured_restraint(psf, crd)

    fm.subscribe(restraint)

    apo_test.expect_invalid_argument(
        "ForceManager rejects duplicate HarmonicRestraintForce subscription",
        lambda: fm.subscribe(restraint),
        "Force is already subscribed to this ForceManager",
        expected_context="ForceManager.subscribe(HarmonicRestraintForce)",
    )

    integrator = apo.CudaLangevinThermostatIntegrator(TIME_STEP)
    integrator.setReferenceTemperature(TEMPERATURE)
    integrator.setThermostatFriction(0.0)
    integrator.setThermostatRngSeed(RANDOM_SEED)
    integrator.setCharmmContext(ctx)

    integrator.propagate(10)

    apo_test.assert_finite_temperature(
        "post harmonic restraint propagation", ctx.computeTemperature()
    )

    fm.unsubscribe(restraint)

    apo_test.expect_invalid_argument(
        "ForceManager rejects missing HarmonicRestraintForce unsubscription",
        lambda: fm.unsubscribe(restraint),
        "Force is not subscribed to this ForceManager",
        expected_context="ForceManager.unsubscribe(HarmonicRestraintForce)",
    )

    integrator.close()
    restraint.close()
    fm.close()
    ctx.close()

    return


def check_validation() -> None:
    print("Checking HarmonicRestraintForce validation...")

    _, psf, crd, fm, _ = create_system()
    restraint = apo.HarmonicRestraintForce(psf)
    selector = apo.AtomSelector(psf)
    selection = selector.select("all")

    atom_count_error: TypeError = apo_test.expect_exception(
        "HarmonicRestraintForce rejects an atom count",
        TypeError,
        lambda: apo.HarmonicRestraintForce(2),  # type: ignore[arg-type]
    )
    apo_test.assert_equal(
        "HarmonicRestraintForce atom-count rejection message",
        str(atom_count_error),
        "HarmonicRestraintForce expects a CharmmPsf",
    )
    apo_test.expect_exception(
        "setSelection rejects non-AtomSelection",
        TypeError,
        lambda: restraint.setSelection(object()),  # type: ignore[arg-type]
    )
    force_constant_error = apo_test.expect_invalid_argument(
        "setForceConstant rejects negative force constant",
        lambda: restraint.setForceConstant(-1.0),
        "Force constant must be non-negative",
        expected_context="HarmonicRestraintForce.setForceConstant(force_constant)",
    )
    apo_test.assert_equal(
        "setForceConstant rendered native function occurrence count",
        force_constant_error.message.count(
            "apo_harmonic_restraint_force_set_force_constant"
        ),
        1,
    )
    apo_test.expect_invalid_argument(
        "setForceConstant rejects non-finite force constant",
        lambda: restraint.setForceConstant(math.inf),
        "Force constant must be finite",
    )
    restraint.setForceConstant(0.0)
    apo_test.expect_invalid_argument(
        "setForceConstants rejects wrong length",
        lambda: restraint.setForceConstants([1.0]),
        "Force-constant array size mismatch",
    )
    apo_test.expect_invalid_argument(
        "setForceConstants rejects non-finite value",
        lambda: restraint.setForceConstants([1.0, math.inf]),
        "Force constant at index 1 must be finite",
    )
    restraint.setForceConstants([1.0, -1.0])
    apo_test.expect_exception(
        "setReferenceCoordinates rejects wrong coordinate length",
        ValueError,
        lambda: restraint.setReferenceCoordinates([[0.0, 0.0]]),
    )
    apo_test.expect_invalid_argument(
        "setReferenceCoordinates rejects wrong atom count",
        lambda: restraint.setReferenceCoordinates([[0.0, 0.0, 0.0]]),
        "Reference-coordinate array size mismatch; expected 2, observed 1",
    )
    apo_test.expect_invalid_argument(
        "setReferenceCoordinates rejects non-finite coordinate",
        lambda: restraint.setReferenceCoordinates(
            [[0.0, 0.0, 0.0], [0.0, math.inf, 0.0]]
        ),
        "Reference coordinate at atom index 1, Y component must be finite",
    )
    apo_test.expect_invalid_argument(
        "setMasses rejects wrong length",
        lambda: restraint.setMasses([1.0]),
        "Mass array size mismatch; expected 2, observed 1",
    )
    apo_test.expect_invalid_argument(
        "setMasses rejects non-finite mass",
        lambda: restraint.setMasses([1.0, math.inf]),
        "Mass at index 1 must be finite",
    )
    restraint.setMasses([1.0, -1.0])
    restraint.setMasses([0.0, 1.0])
    apo_test.expect_exception(
        "setExponent rejects non-int exponent",
        TypeError,
        lambda: restraint.setExponent(2.0),  # type: ignore[arg-type]
    )
    apo_test.expect_exception(
        "setExponent rejects bool exponent",
        TypeError,
        lambda: restraint.setExponent(True),
    )
    apo_test.expect_exception(
        "setExponent rejects out-of-range exponent",
        ValueError,
        lambda: restraint.setExponent(2**31),
    )
    restraint.setExponent(-1)
    restraint.setExponent(2)
    apo_test.expect_invalid_argument(
        "setScaleFactors rejects wrong length",
        lambda: restraint.setScaleFactors([1.0, 1.0]),
        "Scale-factor array size mismatch; expected 3, observed 2",
        expected_context="HarmonicRestraintForce.setScaleFactors(scale_factors)",
    )
    apo_test.expect_invalid_argument(
        "setScaleFactors rejects non-finite factor",
        lambda: restraint.setScaleFactors([1.0, math.inf, 1.0]),
        "Scale factor at index 1 must be finite",
    )
    restraint.setScaleFactors([2.0, 0.0, -1.0])
    restraint.setScaleFactors([1.0, 1.0, 1.0])
    apo_test.expect_invalid_argument(
        "setBoxDimensions rejects wrong length",
        lambda: restraint.setBoxDimensions([50.0, 50.0]),
        "Box-dimension array size mismatch; expected 3, observed 2",
    )
    apo_test.expect_invalid_argument(
        "setBoxDimensions rejects non-finite dimension",
        lambda: restraint.setBoxDimensions([50.0, math.inf, 50.0]),
        "Box dimension at index 1 must be finite",
    )
    apo_test.expect_invalid_argument(
        "setBoxDimensions rejects zero dimension",
        lambda: restraint.setBoxDimensions([50.0, 0.0, 50.0]),
        "Box dimension at index 1 must be positive",
    )
    apo_test.expect_invalid_argument(
        "setBoxDimensions rejects negative dimension",
        lambda: restraint.setBoxDimensions([50.0, -1.0, 50.0]),
        "Box dimension at index 1 must be positive",
    )
    restraint.setBoxDimensions(BOX_DIMENSIONS)
    apo_test.expect_exception(
        "subscribe rejects non-ForceManager",
        TypeError,
        lambda: restraint._subscribe_to_force_manager(object()),  # type: ignore[arg-type]
    )
    apo_test.expect_invalid_argument(
        "subscribe rejects empty force tag",
        lambda: fm.subscribe(restraint, ""),
        "Force tag must not be empty",
    )

    restraint.setSelection(selection)
    restraint.setReferenceCoordinates(crd.getCoordinates())
    restraint.setMasses(psf.getMasses())
    restraint.setForceConstant(1.0)

    fm.subscribe(restraint, "custom-harmonic-restraint")
    fm.unsubscribe(restraint)

    restraint.close()
    fm.close()

    return


def check_charmm_exponent_and_scale_factors() -> None:
    print("Checking HarmonicRestraintForce CHARMM exponent and scale factors...")

    _, psf, _, fm, ctx = create_system()

    ctx.calculatePotentialEnergy()
    baseline_energy: float = fm.getPotentialEnergy()
    baseline_forces: list[list[float]] = fm.getForces()

    coordinates: list[list[float]] = [
        record[:3] for record in ctx.getCoordinatesCharges()
    ]
    displacements: list[list[float]] = [[1.0, -2.0, 2.0], [-0.5, 1.0, 0.25]]
    force_constants: list[float] = [2.0, -1.5]
    masses: list[float] = psf.getMasses()
    scale_factors: list[float] = [1.0, 0.5, 0.25]

    restraint = apo.HarmonicRestraintForce(psf)
    restraint.setReferenceCoordinates(
        [
            [position[k] - displacement[k] for k in range(3)]
            for position, displacement in zip(coordinates, displacements)
        ]
    )
    restraint.setForceConstants(force_constants)
    restraint.setMasses(masses)
    restraint.setExponent(4)
    restraint.setScaleFactors(scale_factors)

    fm.subscribe(restraint)
    ctx.calculatePotentialEnergy()

    # CHARMM's quartic branch: E = C * S * S and DF = C * 4 * S, so the
    # physical force component is -d_a * DF * s_a.
    expected_energy: float = 0.0
    expected_forces: list[list[float]] = []
    for atom, displacement in enumerate(displacements):
        c: float = force_constants[atom] * masses[atom]
        s: float = sum(
            scale * component * component
            for scale, component in zip(scale_factors, displacement)
        )
        expected_energy += c * s * s
        expected_forces.append(
            [
                -component * (c * 4.0 * s) * scale
                for scale, component in zip(scale_factors, displacement)
            ]
        )

    observed_forces: list[list[float]] = fm.getForces()

    apo_test.assert_close(
        "CHARMM quartic scaled energy",
        fm.getPotentialEnergy() - baseline_energy,
        expected_energy,
        FLOAT_RELATIVE_TOLERANCE * abs(expected_energy),
    )
    apo_test.assert_nested_sequence_close(
        "CHARMM quartic scaled forces",
        [
            [observed - baseline for observed, baseline in zip(row, baseline_row)]
            for row, baseline_row in zip(observed_forces, baseline_forces)
        ],
        expected_forces,
        FLOAT_RELATIVE_TOLERANCE
        * max(abs(component) for row in expected_forces for component in row),
    )

    fm.unsubscribe(restraint)
    restraint.close()
    fm.close()
    ctx.close()

    return


def check_close_invalidates_handle() -> None:
    print("Checking HarmonicRestraintForce close invalidates handle...")

    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")
    psf = apo.CharmmPsf(psf_path)
    restraint = apo.HarmonicRestraintForce(psf)
    psf.close()
    restraint.setForceConstant(1.0)
    restraint.close()

    apo_test.expect_exception(
        "closed HarmonicRestraintForce rejects setter",
        RuntimeError,
        lambda: restraint.setForceConstant(1.0),
    )
    apo_test.expect_exception(
        "HarmonicRestraintForce rejects a closed CharmmPsf",
        RuntimeError,
        lambda: apo.HarmonicRestraintForce(psf),
    )


def main(argc: int, argv: list[str]) -> int:
    check_construction_and_setters()
    check_subscription_and_short_propagation()
    check_validation()
    check_charmm_exponent_and_scale_factors()
    check_close_invalidates_handle()

    print(
        "\033[32m"
        + "PASS: HarmonicRestraintForce Python API tests completed."
        + "\033[0m"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
