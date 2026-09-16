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

ADD_RESTRAINT_CONTEXT: str = (
    "DistanceRestraintForce.addRestraint(atom_pairs, coefficients, "
    "force_constant, reference_value, distance_exponent, energy_exponent, "
    "condition)"
)
BOX_DIMENSIONS: list[float] = [50.0, 50.0, 50.0]
PAIR_DISTANCE: float = math.sqrt(3.0) * 2.82
RANDOM_SEED: int = 314159
TEMPERATURE: float = 300.0
TIME_STEP: float = 0.001

AtomPair = tuple[apo.AtomReference, apo.AtomReference]


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


def close_system(
    prm: apo.CharmmParameters,
    psf: apo.CharmmPsf,
    crd: apo.CharmmCrd,
    fm: apo.ForceManager,
    ctx: apo.CharmmContext,
) -> None:
    fm.close()
    ctx.close()
    crd.close()
    psf.close()
    prm.close()
    return


def create_atom_pair(psf: apo.CharmmPsf) -> AtomPair:
    first = apo.AtomReference(psf, 0)

    try:
        second = apo.AtomReference(psf, 1)
    except BaseException:
        first.close()
        raise

    return first, second


def close_atom_pair(atom_pair: AtomPair) -> None:
    atom_pair[1].close()
    atom_pair[0].close()
    return


def add_stable_one_pair_term(
    restraint: apo.DistanceRestraintForce,
    atom_pair: AtomPair,
    condition: apo.DistanceRestraintCondition,
    reference_value: float = PAIR_DISTANCE - 0.1,
) -> None:
    restraint.addRestraint([atom_pair], [1.0], 0.1, reference_value, 1, 2, condition)
    return


def check_exports_construction_default_tag_and_scale() -> None:
    print(
        "Checking DistanceRestraintCondition/DistanceRestraintForce exports, "
        "construction, default tag, and SCALE..."
    )

    apo_test.assert_equal(
        "DistanceRestraintCondition __all__ export",
        "DistanceRestraintCondition" in apo.__all__,
        True,
    )
    apo_test.assert_equal(
        "DistanceRestraintForce __all__ export",
        "DistanceRestraintForce" in apo.__all__,
        True,
    )
    apo_test.assert_equal(
        "DistanceRestraintCondition.NONE value",
        apo.DistanceRestraintCondition.NONE.value,
        0,
    )
    apo_test.assert_equal(
        "DistanceRestraintCondition.POSITIVE value",
        apo.DistanceRestraintCondition.POSITIVE.value,
        1,
    )
    apo_test.assert_equal(
        "DistanceRestraintCondition.NEGATIVE value",
        apo.DistanceRestraintCondition.NEGATIVE.value,
        -1,
    )

    restraint = apo.DistanceRestraintForce(2)
    try:
        apo_test.assert_equal(
            "DistanceRestraintForce.default_force_tag",
            restraint.default_force_tag,
            "resd",
        )

        restraint.setScale(0.0)
        restraint.setScale(-2.0)
        restraint.setScale(3.0)
    finally:
        restraint.close()

    return


def check_term_conditions_reset_and_reuse() -> None:
    print(
        "Checking one-pair, multi-pair, multiple-term, condition, RESET, and "
        "reuse configuration..."
    )

    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")
    psf = apo.CharmmPsf(psf_path)
    atom_pair = create_atom_pair(psf)
    restraint = apo.DistanceRestraintForce(psf.getNumAtoms())

    try:
        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.NONE,
        )

        restraint.addRestraint(
            [atom_pair, atom_pair],
            [1.0, -0.5],
            0.1,
            0.5 * PAIR_DISTANCE,
            1,
            2,
            apo.DistanceRestraintCondition.NONE,
        )

        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.POSITIVE,
            reference_value=PAIR_DISTANCE - 1.0,
        )

        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.NEGATIVE,
            reference_value=PAIR_DISTANCE + 1.0,
        )

        restraint.setScale(-3.0)
        restraint.reset()

        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.NONE,
        )
    finally:
        restraint.close()
        close_atom_pair(atom_pair)
        psf.close()

    return


def check_validation() -> None:
    print("Checking DistanceRestraintForce validation and error propagation...")

    prm, psf, crd, fm, ctx = create_system()
    restraint = apo.DistanceRestraintForce(psf.getNumAtoms())
    atom_pair = create_atom_pair(psf)

    try:
        apo_test.expect_exception(
            "DistanceRestraintForce rejects non-int num_atoms",
            TypeError,
            lambda: apo.DistanceRestraintForce(2.0),  # type: ignore[arg-type]
        )

        apo_test.expect_invalid_argument(
            "DistanceRestraintForce rejects zero num_atoms",
            lambda: apo.DistanceRestraintForce(0),
            "Atom count must be positive; observed 0",
            expected_context="DistanceRestraintForce construction",
        )

        apo_test.expect_invalid_argument(
            "DistanceRestraintForce rejects negative num_atoms",
            lambda: apo.DistanceRestraintForce(-1),
            "Atom count must be positive; observed -1",
            expected_context="DistanceRestraintForce construction",
        )

        apo_test.expect_exception(
            "addRestraint rejects raw integer endpoint pairs",
            TypeError,
            lambda: restraint.addRestraint(
                [[0, 1]],  # type: ignore[list-item]
                [1.0],
                1.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
        )

        apo_test.expect_exception(
            "addRestraint rejects a pair with the wrong number of references",
            ValueError,
            lambda: restraint.addRestraint(
                [[atom_pair[0]]],
                [1.0],
                1.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
        )

        apo_test.expect_invalid_argument(
            "addRestraint rejects an empty pair list",
            lambda: restraint.addRestraint(
                [],
                [],
                1.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
            "A distance-restraint term must contain at least one atom pair",
            expected_context=ADD_RESTRAINT_CONTEXT,
        )

        apo_test.expect_invalid_argument(
            "addRestraint rejects pair/coefficient count mismatch",
            lambda: restraint.addRestraint(
                [atom_pair],
                [1.0, -0.5],
                1.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
            "Pair and coefficient counts must match",
            expected_context=ADD_RESTRAINT_CONTEXT,
        )

        larger_psf_path: str = apo_test.require_file(
            apo_test.get_data_dir() / "1lvz.psf"
        )
        larger_psf = apo.CharmmPsf(larger_psf_path)
        outside_reference = apo.AtomReference(larger_psf, psf.getNumAtoms())

        try:
            apo_test.expect_invalid_argument(
                "addRestraint applies force-local index validation",
                lambda: restraint.addRestraint(
                    [[atom_pair[0], outside_reference]],
                    [1.0],
                    1.0,
                    0.0,
                    1,
                    2,
                    apo.DistanceRestraintCondition.NONE,
                ),
                "Second atom index at pair index 0 is out of range",
                expected_context=ADD_RESTRAINT_CONTEXT,
            )
        finally:
            outside_reference.close()
            larger_psf.close()

        second_psf_path: str = apo_test.require_file(
            apo_test.get_data_dir() / "nacl_pair.psf"
        )
        second_psf = apo.CharmmPsf(second_psf_path)
        second_pair = create_atom_pair(second_psf)

        try:
            restraint.addRestraint(
                [[atom_pair[0], second_pair[1]]],
                [1.0],
                1.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            )
            restraint.reset()

            apo_test.expect_invalid_argument(
                "addRestraint rejects the same extracted index from "
                "different topologies",
                lambda: restraint.addRestraint(
                    [[atom_pair[0], second_pair[0]]],
                    [1.0],
                    1.0,
                    0.0,
                    1,
                    2,
                    apo.DistanceRestraintCondition.NONE,
                ),
                "Atom indices at pair index 0 must be distinct; " "observed (0, 0)",
                expected_context=ADD_RESTRAINT_CONTEXT,
            )
        finally:
            close_atom_pair(second_pair)
            second_psf.close()

        closed_reference = apo.AtomReference(psf, 0)
        closed_reference.close()

        apo_test.expect_exception(
            "addRestraint rejects a closed AtomReference",
            RuntimeError,
            lambda: restraint.addRestraint(
                [[closed_reference, atom_pair[1]]],
                [1.0],
                1.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
        )

        apo_test.expect_invalid_argument(
            "addRestraint propagates zero force constant as ApoCharmmError",
            lambda: restraint.addRestraint(
                [atom_pair],
                [1.0],
                0.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
            "Force constant must be nonzero",
            expected_context=ADD_RESTRAINT_CONTEXT,
        )

        apo_test.expect_invalid_argument(
            "addRestraint propagates nonfinite reference value as " "ApoCharmmError",
            lambda: restraint.addRestraint(
                [atom_pair],
                [1.0],
                1.0,
                math.inf,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
            "Reference value must be finite",
            expected_context=ADD_RESTRAINT_CONTEXT,
        )

        apo_test.expect_invalid_argument(
            "addRestraint rejects uncharacterized distance exponent",
            lambda: restraint.addRestraint(
                [atom_pair],
                [1.0],
                1.0,
                0.0,
                3,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
            "Distance exponent must be one of the characterized values",
            expected_context=ADD_RESTRAINT_CONTEXT,
        )

        apo_test.expect_invalid_argument(
            "addRestraint rejects negative energy exponent",
            lambda: restraint.addRestraint(
                [atom_pair],
                [1.0],
                1.0,
                0.0,
                1,
                -1,
                apo.DistanceRestraintCondition.NONE,
            ),
            "Energy exponent must be one of the characterized values",
            expected_context=ADD_RESTRAINT_CONTEXT,
        )

        apo_test.expect_exception(
            "addRestraint rejects a non-enum condition",
            TypeError,
            lambda: restraint.addRestraint(
                [atom_pair],
                [1.0],
                1.0,
                0.0,
                1,
                2,
                0,  # type: ignore[arg-type]
            ),
        )

        apo_test.expect_exception(
            "subscribe rejects a non-ForceManager target",
            TypeError,
            lambda: restraint._subscribe_to_force_manager(  # type: ignore[arg-type]
                object()
            ),
        )

        apo_test.expect_invalid_argument(
            "subscribe rejects an empty force tag",
            lambda: fm.subscribe(restraint, ""),
            "Force tag must not be empty",
            expected_context="ForceManager.subscribe(DistanceRestraintForce)",
        )
    finally:
        close_atom_pair(atom_pair)
        restraint.close()
        close_system(prm, psf, crd, fm, ctx)

    return


def check_subscription_mutation_and_short_integration() -> None:
    print(
        "Checking default/custom subscription, duplicate rejection, "
        "unsubscription, subscribed mutation, and short integration..."
    )

    prm, psf, crd, fm, ctx = create_system()
    restraint = apo.DistanceRestraintForce(psf.getNumAtoms())
    atom_pair = create_atom_pair(psf)
    integrator: apo.CudaLangevinThermostatIntegrator | None = None

    try:
        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.NONE,
        )

        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.POSITIVE,
            reference_value=PAIR_DISTANCE - 1.0,
        )

        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.NEGATIVE,
            reference_value=PAIR_DISTANCE + 1.0,
        )

        fm.subscribe(restraint)

        try:
            ctx.calculatePotentialEnergy()

            apo_test.expect_invalid_argument(
                "ForceManager rejects duplicate DistanceRestaintForce " "subscription",
                lambda: fm.subscribe(restraint),
                "Force is already subscribed to this ForceManager",
                expected_context=("ForceManager.subscribe(DistanceRestraintForce)"),
            )
        finally:
            fm.unsubscribe(restraint)

        fm.subscribe(restraint, "custom-resd")

        try:
            restraint.setScale(3.0)

            restraint.addRestraint(
                [atom_pair, atom_pair],
                [1.0, -0.5],
                0.1,
                0.5 * PAIR_DISTANCE,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            )

            ctx.calculatePotentialEnergy()

            restraint.setScale(-0.25)
            ctx.calculatePotentialEnergy()

            restraint.reset()
            ctx.calculatePotentialEnergy()

            add_stable_one_pair_term(
                restraint,
                atom_pair,
                apo.DistanceRestraintCondition.NONE,
            )

            ctx.calculatePotentialEnergy()

            integrator = apo.CudaLangevinThermostatIntegrator(TIME_STEP)
            integrator.setReferenceTemperature(TEMPERATURE)
            integrator.setThermostatFriction(0.0)
            integrator.setThermostatRngSeed(RANDOM_SEED)
            integrator.setCharmmContext(ctx)
            integrator.propagate(10)

            apo_test.assert_finite_temperature(
                "post distance-restraint propagation",
                ctx.computeTemperature(),
            )

            coordinate_charge_rows: list[list[float]] = ctx.getCoordinatesCharges()

            coordinates: list[list[float]] = [
                [row[0], row[1], row[2]] for row in coordinate_charge_rows
            ]

            apo_test.assert_finite_nested_sequence(
                "post distance-restraint coordinates",
                coordinates,
            )
        finally:
            fm.unsubscribe(restraint)
    finally:
        if integrator is not None:
            integrator.close()

        close_atom_pair(atom_pair)
        restraint.close()
        close_system(prm, psf, crd, fm, ctx)

    return


def check_reference_handles_are_not_retained() -> None:
    print("Checking AtomReference handle lifetime after addRestraint...")

    prm, psf, crd, fm, ctx = create_system()

    source_psf_path: str = apo_test.require_file(
        apo_test.get_data_dir() / "nacl_pair.psf"
    )
    source_psf = apo.CharmmPsf(source_psf_path)
    atom_pair = create_atom_pair(source_psf)

    restraint = apo.DistanceRestraintForce(psf.getNumAtoms())
    subscribed: bool = False

    try:
        add_stable_one_pair_term(
            restraint,
            atom_pair,
            apo.DistanceRestraintCondition.NONE,
        )

        close_atom_pair(atom_pair)
        source_psf.close()

        fm.subscribe(restraint)
        subscribed = True

        ctx.calculatePotentialEnergy()
    finally:
        if subscribed:
            fm.unsubscribe(restraint)

        close_atom_pair(atom_pair)
        source_psf.close()
        restraint.close()
        close_system(prm, psf, crd, fm, ctx)

    return


def check_closed_handle_rejection() -> None:
    print("Checking closed DistanceRestraintForce handle rejection...")

    psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "nacl_pair.psf")
    psf = apo.CharmmPsf(psf_path)
    atom_pair = create_atom_pair(psf)

    restraint = apo.DistanceRestraintForce(psf.getNumAtoms())
    restraint.close()

    try:
        apo_test.expect_exception(
            "closed DistanceRestraintForce rejects addRestraint",
            RuntimeError,
            lambda: restraint.addRestraint(
                [atom_pair],
                [1.0],
                1.0,
                0.0,
                1,
                2,
                apo.DistanceRestraintCondition.NONE,
            ),
        )

        apo_test.expect_exception(
            "closed DistanceRestraintForce rejects reset",
            RuntimeError,
            restraint.reset,
        )
    finally:
        close_atom_pair(atom_pair)
        psf.close()

    return


def main(argc: int, argv: list[str]) -> int:
    check_exports_construction_default_tag_and_scale()
    check_term_conditions_reset_and_reuse()
    check_validation()
    check_subscription_mutation_and_short_integration()
    check_reference_handles_are_not_retained()
    check_closed_handle_rejection()

    print(
        "\033[32m"
        + "PASS: DistanceRestraintForce Python API tests completed."
        + "\033[0m"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
