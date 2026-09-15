# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from __future__ import annotations

import ctypes
import gc
from pathlib import Path
import sys
from typing import cast
import weakref

import apocharmm as apo
import apocharmm._lib as apo_lib

import apo_test_helpers as apo_test

NUM_ATOMS: int = 10
TEST_PSF_TEXT: str = """PSF

       1 !NTITLE
 REMARKS generated AtomReference Python API test PSF
      10 !NATOM
       1 SEG1     1 ALA  N    NH1   -0.300000  14.0070           0
       2 SEG1     1 ALA  CA   CT1    0.100000  12.0110           0
       3 SEG1     1 ALA  CB   CT2    0.000000  12.0110           0
       4 SEG1     1 ALA  HB1  HA     0.100000   1.0080           0
       5 SEG2     2 GLY  N    NH1   -0.200000  14.0070           0
       6 SEG2     2 GLY  CA   CT1    0.100000  12.0110           0
       7 SEG2     2 GLY  HA1  HA     0.100000   1.0080           0
       8 WAT      3 TIP3 OH2  OT    -0.834000  15.9994           0
       9 WAT      3 TIP3 H1   HT     0.417000   1.0080           0
      10 WAT      3 TIP3 H2   HT     0.417000   1.0080           0
       7 !NBOND: bonds
       1       2       2       3       3       4       5       6
       6       7       8       9       8      10
       4 !NTHETA: angles
       1       2       3       2       3       4       5       6       7
       9       8      10
       1 !NPHI: dihedrals
       1       2       3       4
       0 !NIMPHI: impropers
       0 !NDON: donors
       0 !NACC: acceptors
       0 !NCRTERM: cross-terms
"""


def get_last_error_text() -> str:
    message_address: int | None = apo_lib.lib().apo_last_error()
    if message_address is None:
        return ""

    return ctypes.string_at(message_address).decode("utf-8", errors="replace")


def check_top_level_export() -> None:
    print("Checking AtomReference top-level export...")

    apo_test.assert_equal(
        "AtomReference top-level class name",
        apo.AtomReference.__name__,
        "AtomReference",
    )
    apo_test.assert_equal(
        "AtomReference top-level module",
        apo.AtomReference.__module__,
        "apocharmm.atom_reference",
    )
    apo_test.assert_equal(
        "AtomReference in apocharmm.__all__", "AtomReference" in apo.__all__, True
    )

    return


def check_direct_construction(psf: apo.CharmmPsf) -> None:
    print("Checking direct AtomReference construction and validation...")

    first = apo.AtomReference(psf, 0)
    last = apo.AtomReference(psf, NUM_ATOMS - 1)

    apo_test.assert_equal("first atom index", first.getAtomIndex(), 0)
    apo_test.assert_equal("last atom index", last.getAtomIndex(), NUM_ATOMS - 1)

    apo_test.expect_exception(
        "AtomReference rejects wrong PSF type",
        TypeError,
        lambda: apo.AtomReference(cast(apo.CharmmPsf, object()), 0),
    )
    apo_test.expect_exception(
        "AtomReference rejects float atom index",
        TypeError,
        lambda: apo.AtomReference(psf, cast(int, 1.0)),
    )
    apo_test.expect_exception(
        "AtomReference rejects string atom index",
        TypeError,
        lambda: apo.AtomReference(psf, cast(int, "1")),
    )
    apo_test.expect_exception(
        "AtomReference rejects bool atom index",
        TypeError,
        lambda: apo.AtomReference(psf, True),
    )

    c_int_bits: int = ctypes.sizeof(ctypes.c_int) * 8
    c_int_min: int = -(1 << (c_int_bits - 1))
    c_int_max: int = (1 << (c_int_bits - 1)) - 1

    apo_test.expect_exception(
        "AtomReference rejects C-int underflow",
        ValueError,
        lambda: apo.AtomReference(psf, c_int_min - 1),
    )
    apo_test.expect_exception(
        "AtomReference rejects C-int overflow",
        ValueError,
        lambda: apo.AtomReference(psf, c_int_max + 1),
    )

    negative_error = apo_test.expect_invalid_argument(
        "AtomReference rejects negative atom index",
        lambda: apo.AtomReference(psf, -1),
        f"Atom index is out of range; expected [0, {NUM_ATOMS}), observed -1",
        expected_context="AtomReference construction",
    )
    apo_test.assert_equal(
        "negative index native function occurrence count",
        negative_error.message.count("apo_atom_reference_create"),
        1,
    )

    upper_error = apo_test.expect_invalid_argument(
        "AtomReference rejects atom count as index",
        lambda: apo.AtomReference(psf, NUM_ATOMS),
        f"Atom index is out of range; expected [0, {NUM_ATOMS}), observed {NUM_ATOMS}",
        expected_context="AtomReference construction",
    )
    apo_test.assert_equal(
        "upper index native function occurrence count",
        upper_error.message.count("apo_atom_reference_create"),
        1,
    )

    first.close()
    last.close()

    return


def check_topology_and_equality(psf_path: Path) -> None:
    print("Checking AtomReference topology identity and equality...")

    first_psf = apo.CharmmPsf(str(psf_path))
    second_psf = apo.CharmmPsf(str(psf_path))

    try:
        first_zero = apo.AtomReference(first_psf, 0)
        first_zero_copy = apo.AtomReference(first_psf, 0)
        first_one = apo.AtomReference(first_psf, 1)
        second_zero = apo.AtomReference(second_psf, 0)
        second_one = apo.AtomReference(second_psf, 1)

        try:
            apo_test.assert_equal(
                "same PSF has same topology",
                first_zero.hasSameTopology(first_one),
                True,
            )
            apo_test.assert_equal(
                "same file parsed twice has different topology",
                first_zero.hasSameTopology(second_zero),
                False,
            )

            apo_test.expect_exception(
                "hasSameTopology rejects unrelated type",
                TypeError,
                lambda: first_zero.hasSameTopology(cast(apo.AtomReference, object())),
            )

            apo_test.assert_equal(
                "same topology and index compare equal",
                first_zero == first_zero_copy,
                True,
            )
            apo_test.assert_equal(
                "same topology and different index compare unequal",
                first_zero == first_one,
                False,
            )
            apo_test.assert_equal(
                "different topology and same index compare unequal",
                first_zero == second_zero,
                False,
            )
            apo_test.assert_equal(
                "different topology and different index compare unequal",
                first_zero == second_one,
                False,
            )
            apo_test.assert_equal(
                "inequality negates equality", first_zero != first_zero_copy, False
            )
            apo_test.assert_equal(
                "inequality detects different index", first_zero != first_one, True
            )
            apo_test.assert_equal(
                "direct equality returns NotImplemented for unrelated type",
                first_zero.__eq__(object()) is NotImplemented,
                True,
            )
            apo_test.assert_equal(
                "operator equality with unrelated type", first_zero == object(), False
            )
            apo_test.assert_equal(
                "operator inequality with unrelated type", first_zero != object(), True
            )
            apo_test.expect_exception(
                "AtomReference remains unhashable", TypeError, lambda: hash(first_zero)
            )
        finally:
            first_zero.close()
            first_zero_copy.close()
            first_one.close()
            second_zero.close()
            second_one.close()
    finally:
        first_psf.close()
        second_psf.close()

    return


def check_select_atom(psf: apo.CharmmPsf) -> None:
    print("Checking AtomSelector.selectAtom()...")

    selector = apo.AtomSelector(psf)

    try:
        reference = selector.selectAtom("atom SEG1 1 CA")
        try:
            apo_test.assert_equal(
                "selectAtom selected index", reference.getAtomIndex(), 1
            )
        finally:
            reference.close()

        apo_test.expect_exception(
            "selectAtom rejects non-string selection",
            TypeError,
            lambda: selector.selectAtom(cast(str, 1)),
        )

        apo_test.expect_invalid_argument(
            "selectAtom rejects zero matches",
            lambda: selector.selectAtom("none"),
            "Atom selection must match exactly one atom; observed 0",
            expected_context="AtomSelector.selectAtom(selection_string)",
        )
        apo_test.expect_invalid_argument(
            "selectAtom rejects multiple matches",
            lambda: selector.selectAtom("type CA"),
            "Atom selection must match exactly one atom; observed 2",
            expected_context="AtomSelector.selectAtom(selection_string)",
        )

        parser_error = apo_test.expect_invalid_argument(
            "selectAtom propagates parser errors",
            lambda: selector.selectAtom(".around. type CA"),
            'Unknown dotted atom selection operator ".around."',
            expected_context="AtomSelector.selectAtom(selection_string)",
        )
        apo_test.assert_equal(
            "selectAtom rendered context occurrence count",
            parser_error.message.count(parser_error.context),
            1,
        )
        apo_test.assert_equal(
            "selectAtom rendered native function occurrence count",
            parser_error.message.count("apo_atom_selector_select_atom"),
            1,
        )
        apo_test.assert_equal(
            "selectAtom rendered failed text count",
            parser_error.message.count("failed"),
            0,
        )
    finally:
        selector.close()

    return


def check_lifetimes(psf_path: Path) -> None:
    print("Checking AtomReference native and Python-wrapper lifetimes...")

    psf = apo.CharmmPsf(str(psf_path))
    selector = apo.AtomSelector(psf)
    reference = selector.selectAtom("atom SEG1 1 CA")

    selector.close()
    apo_test.assert_equal(
        "selector-created reference after selector close", reference.getAtomIndex(), 1
    )

    psf.close()
    apo_test.assert_equal(
        "selector-created reference after PSF-wrapper close",
        reference.getAtomIndex(),
        1,
    )
    reference.close()

    source_psf = apo.CharmmPsf(str(psf_path))
    source_psf_weak = weakref.ref(source_psf)
    direct_reference = apo.AtomReference(source_psf, 2)

    source_psf.close()
    del source_psf
    gc.collect()

    apo_test.assert_equal(
        "AtomReference does not retain Python CharmmPsf wrapper",
        source_psf_weak() is None,
        True,
    )
    apo_test.assert_equal(
        "direct reference retains native PSF after wrapper collection",
        direct_reference.getAtomIndex(),
        2,
    )
    direct_reference.close()

    return


def check_closure(psf: apo.CharmmPsf) -> None:
    print("Checking AtomReference closure behavior...")

    reference = apo.AtomReference(psf, 0)
    other = apo.AtomReference(psf, 1)

    reference.close()
    reference.close()

    apo_test.expect_exception(
        "closed reference rejects getAtomIndex",
        RuntimeError,
        lambda: reference.getAtomIndex(),
    )
    apo_test.expect_exception(
        "closed reference rejects hasSameTopology",
        RuntimeError,
        lambda: reference.hasSameTopology(other),
    )
    apo_test.expect_exception(
        "closed reference rejects equality",
        RuntimeError,
        lambda: reference == other,
    )

    other.close()

    with apo.AtomReference(psf, 3) as context_reference:
        apo_test.assert_equal(
            "context-manager reference index", context_reference.getAtomIndex(), 3
        )

    apo_test.expect_exception(
        "context-manager exit closes reference",
        RuntimeError,
        lambda: context_reference.getAtomIndex(),
    )

    return


def check_stale_diagnostics(psf: apo.CharmmPsf) -> None:
    print("Checking successful AtomReference operations clear stale diagnostics...")

    apo_test.expect_invalid_argument(
        "seed stale direct-construction diagnostic",
        lambda: apo.AtomReference(psf, -1),
        "Atom index is out of range",
    )
    apo_test.assert_equal(
        "construction failure leaves diagnostic", get_last_error_text() != "", True
    )

    reference = apo.AtomReference(psf, 0)
    apo_test.assert_equal(
        "successful construction clears diagnostic", get_last_error_text(), ""
    )
    other = apo.AtomReference(psf, 0)

    apo_test.expect_invalid_argument(
        "seed stale accessor diagnostic",
        lambda: apo.AtomReference(psf, NUM_ATOMS),
        "Atom index is out of range",
    )
    apo_test.assert_equal(
        "accessor seed leaves diagnostic", get_last_error_text() != "", True
    )

    apo_test.assert_equal("successful getAtomIndex result", reference.getAtomIndex(), 0)
    apo_test.assert_equal(
        "successful accessor clears diagnostic", get_last_error_text(), ""
    )

    apo_test.expect_invalid_argument(
        "seed stale topology diagnostic",
        lambda: apo.AtomReference(psf, NUM_ATOMS),
        "Atom index is out of range",
    )
    apo_test.assert_equal(
        "successful hasSameTopology result", reference.hasSameTopology(other), True
    )
    apo_test.assert_equal(
        "successful topology query clears diagnostic", get_last_error_text(), ""
    )

    apo_test.expect_invalid_argument(
        "seed stale equality diagnostic",
        lambda: apo.AtomReference(psf, NUM_ATOMS),
        "Atom index is out of range",
    )
    apo_test.assert_equal("successful equality result", reference == other, True)
    apo_test.assert_equal(
        "successful equality query clears diagnostic", get_last_error_text(), ""
    )

    selector = apo.AtomSelector(psf)
    try:
        apo_test.expect_invalid_argument(
            "seed stale selectAtom diagnostic",
            lambda: selector.selectAtom("none"),
            "Atom selection must match exactly one atom; observed 0",
        )
        apo_test.assert_equal(
            "selectAtom failure leaves diagnostic", get_last_error_text() != "", True
        )

        selected_reference = selector.selectAtom("bynu 2")
        try:
            apo_test.assert_equal(
                "successful selectAtom clears diagnostic", get_last_error_text(), ""
            )
            apo_test.assert_equal(
                "selected reference index", selected_reference.getAtomIndex(), 1
            )
        finally:
            selected_reference.close()
    finally:
        selector.close()
        other.close()
        reference.close()

    return


def main(argc: int, argv: list[str]) -> int:
    repo_root: Path = Path(argv[1]) if argc > 1 else Path(".")
    output_dir: Path = repo_root / "test/pytest"
    output_dir.mkdir(parents=True, exist_ok=True)

    psf_path: Path = output_dir / "tmp_python_api_atom_reference.psf"
    apo_test.remove_if_exists(psf_path)

    try:
        psf_path.write_text(TEST_PSF_TEXT, encoding="utf-8")

        check_top_level_export()
        check_topology_and_equality(psf_path)
        check_lifetimes(psf_path)

        psf = apo.CharmmPsf(str(psf_path))
        try:
            check_direct_construction(psf)
            check_select_atom(psf)
            check_closure(psf)
            check_stale_diagnostics(psf)
        finally:
            psf.close()
    finally:
        print("Cleaning up AtomReference Python API test files...")
        apo_test.remove_if_exists(psf_path)

    print("\033[32m" + "PASS: AtomReference Python API tests completed." + "\033[0m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
