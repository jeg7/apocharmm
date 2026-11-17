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
    psf: apo.CharmmPsf,
) -> apo.HarmonicCenterOfMassRestraintForce:
    selector = apo.AtomSelector(psf)
    selection = selector.select("all")

    restraint = apo.HarmonicCenterOfMassRestraintForce(psf)
    restraint.setSelection(selection)
    restraint.setReferencePosition([1.41, 1.41, 1.41])
    restraint.setReferencePosition([1.41, 1.41, 1.41], [1, 1, 1])
    restraint.setReferenceDistance(0.0)
    restraint.setMasses(psf.getMasses())
    restraint.setMassWeighting(False)
    restraint.setForceConstant(1.0)

    return restraint


def check_construction_and_setters() -> None:
    print("Checking HarmonicCenterOfMassRestraintForce construction and setters...")

    _, psf, _, _, _ = create_system()

    restraint = create_configured_restraint(psf)

    apo_test.assert_equal(
        "HarmonicCenterOfMassRestraintForce.default_force_tag",
        restraint.default_force_tag,
        "hmcm",
    )

    restraint.close()

    return


def check_subscription_and_short_propagation() -> None:
    print("Checking HarmonicCenterOfMassRestraintForce ForceManager subscription...")

    _, psf, crd, fm, ctx = create_system()

    restraint = create_configured_restraint(psf)

    fm.subscribe(restraint)

    apo_test.expect_invalid_argument(
        "ForceManager rejects duplicate HarmonicCenterOfMassRestraintForce subscription",
        lambda: fm.subscribe(restraint),
        "Force is already subscribed to this ForceManager",
        expected_context="ForceManager.subscribe(HarmonicCenterOfMassRestraintForce)",
    )

    integrator = apo.CudaLangevinThermostatIntegrator(TIME_STEP)
    integrator.setReferenceTemperature(TEMPERATURE)
    integrator.setThermostatFriction(0.0)
    integrator.setThermostatRngSeed(RANDOM_SEED)
    integrator.setCharmmContext(ctx)

    integrator.propagate(10)

    apo_test.assert_finite_temperature(
        "post harmonic center-of-mass restraint propagation", ctx.computeTemperature()
    )

    fm.unsubscribe(restraint)

    apo_test.expect_invalid_argument(
        "ForceManager rejects missing HarmonicCenterOfMassRestraintForce unsubscription",
        lambda: fm.unsubscribe(restraint),
        "Force is not subscribed to this ForceManager",
        expected_context=(
            "ForceManager.unsubscribe(HarmonicCenterOfMassRestraintForce)"
        ),
    )

    integrator.close()
    restraint.close()
    fm.close()
    ctx.close()

    return


def check_validation() -> None:
    print("Checking HarmonicCenterOfMassRestraintForce validation...")

    _, psf, _, fm, _ = create_system()
    num_atoms: int = psf.getNumAtoms()
    restraint = apo.HarmonicCenterOfMassRestraintForce(psf)
    selector = apo.AtomSelector(psf)
    selection = selector.select("all")

    atom_count_error: TypeError = apo_test.expect_exception(
        "HarmonicCenterOfMassRestraintForce rejects an atom count",
        TypeError,
        lambda: apo.HarmonicCenterOfMassRestraintForce(2),  # type: ignore[arg-type]
    )
    apo_test.assert_equal(
        "HarmonicCenterOfMassRestraintForce atom-count rejection message",
        str(atom_count_error),
        "HarmonicCenterOfMassRestraintForce expects a CharmmPsf",
    )
    apo_test.expect_exception(
        "setSelection rejects non-AtomSelection",
        TypeError,
        lambda: restraint.setSelection(object()),  # type: ignore[arg-type]
    )

    restraint.setForceConstant(-1.0)
    force_constant_error = apo_test.expect_invalid_argument(
        "setForceConstant rejects non-finite force constant",
        lambda: restraint.setForceConstant(math.inf),
        "Force constant must be finite",
        expected_context=(
            "HarmonicCenterOfMassRestraintForce.setForceConstant(force_constant)"
        ),
    )
    apo_test.assert_equal(
        "setForceConstant rendered native function occurrence count",
        force_constant_error.message.count(
            "apo_harmonic_center_of_mass_restraint_force_set_force_constant"
        ),
        1,
    )
    restraint.setForceConstant(0.0)

    apo_test.expect_invalid_argument(
        "setReferencePosition rejects wrong coordinate length",
        lambda: restraint.setReferencePosition([0.0, 0.0]),
        "Reference-position array size mismatch; expected 3, observed 2",
        expected_context=(
            "HarmonicCenterOfMassRestraintForce.setReferencePosition(reference_position)"
        ),
    )
    apo_test.expect_invalid_argument(
        "setReferencePosition rejects wrong reference mask length",
        lambda: restraint.setReferencePosition([0.0, 0.0, 0.0], [1, 1]),
        "Reference-mask array size mismatch; expected 3, observed 2",
    )
    apo_test.expect_invalid_argument(
        "setReferencePosition rejects inactive reference mask",
        lambda: restraint.setReferencePosition([0.0, 0.0, 0.0], [0, 0, 0]),
        "Reference mask must activate at least one coordinate",
    )
    apo_test.expect_invalid_argument(
        "setReferencePosition rejects invalid reference mask value",
        lambda: restraint.setReferencePosition([0.0, 0.0, 0.0], [1, 2, 1]),
        "Reference mask at index 1 must be 0 or 1; observed 2",
    )
    apo_test.expect_invalid_argument(
        "setReferencePosition rejects non-finite coordinate",
        lambda: restraint.setReferencePosition([0.0, math.inf, 0.0]),
        "Reference position at index 1 must be finite",
    )

    restraint.setReferenceDistance(-1.0)
    apo_test.expect_invalid_argument(
        "setReferenceDistance rejects non-finite reference distance",
        lambda: restraint.setReferenceDistance(math.inf),
        "Reference distance must be finite",
        expected_context=(
            "HarmonicCenterOfMassRestraintForce.setReferenceDistance(reference_distance)"
        ),
    )

    apo_test.expect_invalid_argument(
        "setMasses rejects wrong length",
        lambda: restraint.setMasses([1.0]),
        f"Mass array size mismatch; expected {num_atoms}, observed 1",
        expected_context="HarmonicCenterOfMassRestraintForce.setMasses(masses)",
    )

    nonfinite_masses: list[float] = [1.0 for _ in range(num_atoms)]
    nonfinite_masses[1] = math.inf
    apo_test.expect_invalid_argument(
        "setMasses rejects non-finite mass",
        lambda: restraint.setMasses(nonfinite_masses),
        "Mass at index 1 must be finite",
    )

    signed_masses: list[float] = [1.0 for _ in range(num_atoms)]
    signed_masses[1] = -3.0
    restraint.setMasses(signed_masses)

    zero_masses: list[float] = [0.0 for _ in range(num_atoms)]
    apo_test.expect_invalid_argument(
        "setMasses rejects zero selected total weight",
        lambda: restraint.setMasses(zero_masses),
        "Selected atoms must have a finite nonzero total weight",
    )

    restraint.setMassWeighting(False)
    apo_test.expect_invalid_argument(
        "setMassWeighting rejects zero selected total weight",
        lambda: restraint.setMassWeighting(True),
        "Selected atoms must have a finite nonzero total weight",
        expected_context=("HarmonicCenterOfMassRestraintForce.setMassWeighting(flag)"),
    )
    restraint.setMasses(psf.getMasses())

    apo_test.expect_exception(
        "subscribe rejects non-ForceManager",
        TypeError,
        lambda: restraint._subscribe_to_force_manager(object()),  # type: ignore[arg-type]
    )
    apo_test.expect_invalid_argument(
        "subscribe rejects empty force tag",
        lambda: fm.subscribe(restraint, ""),
        "Force tag must not be empty",
        expected_context=("ForceManager.subscribe(HarmonicCenterOfMassRestraintForce)"),
    )

    restraint.setSelection(selection)
    restraint.setReferencePosition([1.41, 1.41, 1.41])
    restraint.setReferenceDistance(0.0)
    restraint.setForceConstant(1.0)
    restraint.setMassWeighting(False)

    fm.subscribe(restraint, "custom-hmcm")
    fm.unsubscribe(restraint)

    restraint.close()
    fm.close()

    return


def check_charmm_raw_center_and_signed_parameters() -> None:
    print("Checking HarmonicCenterOfMassRestraintForce CHARMM raw center and signs...")

    _, psf, _, fm, ctx = create_system()

    ctx.setCoordinates([[-20.0, 1.0, 2.0], [22.0, -3.0, 4.0]])
    ctx.calculatePotentialEnergy()
    baseline_energy: float = fm.getPotentialEnergy()
    baseline_forces: list[list[float]] = fm.getForces()

    coordinates: list[list[float]] = [
        record[:3] for record in ctx.getCoordinatesCharges()
    ]
    masses: list[float] = [3.0, -1.0]
    reference_position: list[float] = [1.0, 1.0, 5.0]
    reference_mask: list[int] = [1, 1, 0]
    force_constant: float = -2.0
    reference_distance: float = -0.5

    restraint = apo.HarmonicCenterOfMassRestraintForce(psf)
    restraint.setMasses(masses)
    restraint.setReferencePosition(reference_position, reference_mask)
    restraint.setReferenceDistance(reference_distance)
    restraint.setForceConstant(force_constant)

    fm.subscribe(restraint)
    ctx.calculatePotentialEnergy()

    # CHARMM builds the center from the raw coordinates. The two atoms are 42
    # angstroms apart in x, so their minimum images would give another center.
    total: float = sum(masses)
    center: list[float] = [
        sum(mass * position[k] for mass, position in zip(masses, coordinates)) / total
        for k in range(3)
    ]
    displacement: list[float] = [
        center[k] - reference_position[k] if reference_mask[k] else 0.0
        for k in range(3)
    ]
    distance: float = math.sqrt(
        sum(component * component for component in displacement)
    )
    deviation: float = distance - reference_distance
    factors: list[float] = [
        component * 2.0 * force_constant / total * deviation / distance
        for component in displacement
    ]
    expected_energy: float = force_constant * deviation * deviation
    expected_forces: list[list[float]] = [
        [-mass * factor for factor in factors] for mass in masses
    ]

    observed_forces: list[list[float]] = fm.getForces()

    apo_test.assert_close(
        "CHARMM raw-center signed energy",
        fm.getPotentialEnergy() - baseline_energy,
        expected_energy,
        FLOAT_RELATIVE_TOLERANCE * abs(expected_energy),
    )
    apo_test.assert_nested_sequence_close(
        "CHARMM raw-center signed forces",
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


def check_psf_masses_weight_the_center() -> None:
    print("Checking HarmonicCenterOfMassRestraintForce PSF mass weighting...")

    _, psf, _, fm, ctx = create_system()

    ctx.setCoordinates([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    ctx.calculatePotentialEnergy()
    baseline_energy: float = fm.getPotentialEnergy()

    restraint = apo.HarmonicCenterOfMassRestraintForce(psf)
    restraint.setMassWeighting(True)
    restraint.setReferencePosition([0.0, 0.0, 0.0])
    restraint.setForceConstant(2.0)

    fm.subscribe(restraint)
    ctx.calculatePotentialEnergy()

    # As with CHARMM's WEIG or MASS keyword, the center uses the PSF masses.
    masses: list[float] = psf.getMasses()
    center: float = 4.0 * masses[1] / sum(masses)
    expected_energy: float = 2.0 * center * center

    apo_test.assert_close(
        "PSF mass-weighted center energy",
        fm.getPotentialEnergy() - baseline_energy,
        expected_energy,
        FLOAT_RELATIVE_TOLERANCE * expected_energy,
    )

    fm.unsubscribe(restraint)
    restraint.close()
    fm.close()
    ctx.close()

    return


def check_close_invalidates_handle() -> None:
    print("Checking HarmonicCenterOfMassRestraintForce close invalidates handle...")

    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")
    psf = apo.CharmmPsf(psf_path)
    restraint = apo.HarmonicCenterOfMassRestraintForce(psf)
    psf.close()
    restraint.setMassWeighting(True)
    restraint.close()

    apo_test.expect_exception(
        "closed HarmonicCenterOfMassRestraintForce rejects setter",
        RuntimeError,
        lambda: restraint.setForceConstant(1.0),
    )
    apo_test.expect_exception(
        "HarmonicCenterOfMassRestraintForce rejects a closed CharmmPsf",
        RuntimeError,
        lambda: apo.HarmonicCenterOfMassRestraintForce(psf),
    )

    return


def main(argc: int, argv: list[str]) -> int:
    check_construction_and_setters()
    check_subscription_and_short_propagation()
    check_validation()
    check_charmm_raw_center_and_signed_parameters()
    check_psf_masses_weight_the_center()
    check_close_invalidates_handle()

    print(
        "\033[32m"
        + "PASS: HarmonicCenterOfMassRestraintForce Python API tests completed."
        + "\033[0m"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
