# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from __future__ import annotations

import array
from collections.abc import Callable, Sequence
import ctypes
from fractions import Fraction
import gc
import math
from pathlib import Path
import sys
from typing import NamedTuple, cast
from unittest.mock import patch
import weakref

import apocharmm as apo
import apocharmm._lib as apo_lib

import apo_test_helpers as apo_test

NUM_ATOMS: int = 12
LARGER_NUM_ATOMS: int = 183
TEST_PSF_TEXT: str = """PSF

       1 !NTITLE
 REMARKS Python API MMFP zero-interaction system
      12 !NATOM
       1 MMFP     1 DUM  DUM  DUM    0.000000   1.0000           0
       2 MMFP     2 DUM  DUM  DUM    0.000000   1.5000           0
       3 MMFP     3 DUM  DUM  DUM    0.000000   2.0000           0
       4 MMFP     4 DUM  DUM  DUM    0.000000   2.5000           0
       5 MMFP     5 DUM  DUM  DUM    0.000000   3.0000           0
       6 MMFP     6 DUM  DUM  DUM    0.000000   3.5000           0
       7 MMFP     7 DUM  DUM  DUM    0.000000   4.0000           0
       8 MMFP     8 DUM  DUM  DUM    0.000000   4.5000           0
       9 MMFP     9 DUM  DUM  DUM    0.000000   5.0000           0
      10 MMFP    10 DUM  DUM  DUM    0.000000   5.5000           0
      11 MMFP    11 DUM  DUM  DUM    0.000000   6.0000           0
      12 MMFP    12 DUM  DUM  DUM    0.000000   6.5000           0
       0 !NBOND: bonds
       0 !NTHETA: angles
       0 !NPHI: dihedrals
       0 !NIMPHI: impropers
       0 !NDON: donors
       0 !NACC: acceptors
       0 !NCRTERM: cross-terms
"""
STANDARD_MASSES: list[float] = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
]
CHARMM_ACCEPTED_RCM_PAIR_MASSES: dict[str, tuple[float, float]] = {
    "RCM with a zero-mass member": (0.0, 2.0),
    "RCM with a negative member and positive total": (-1.0, 3.0),
    "RCM with a negative selected total": (1.0, -3.0),
    "RCM with a small positive total after cancellation": (1.0, -0.9999),
    "RCM with a small negative total after cancellation": (1.0, -1.0001),
    "RCM total of 1e-10 with a zero-mass partner": (1.0e-10, 0.0),
    "RCM total of -1e-10 with a zero-mass partner": (-1.0e-10, 0.0),
    "RCM total of 1e-5 with a zero-mass partner": (1.0e-5, 0.0),
    "RCM total of -1e-5 with a zero-mass partner": (-1.0e-5, 0.0),
    "RCM total of 1e300 with a zero-mass partner": (1.0e300, 0.0),
    "RCM total of -1e300 with a zero-mass partner": (-1.0e300, 0.0),
    "RCM total of 1e-300 with a zero-mass partner": (1.0e-300, 0.0),
    "RCM total of -1e-300 with a zero-mass partner": (-1.0e-300, 0.0),
}
CHARMM_ZERO_TOTAL_RCM_PAIR_MASSES: tuple[float, float] = (1.0, -1.0)
CHARMM_UNSELECTED_MASS: float = 2.0

ORIGIN: list[float] = [0.0, 0.0, 0.0]
X_DIRECTION: list[float] = [1.0, 0.0, 0.0]
Z_DIRECTION: list[float] = [0.0, 0.0, 1.0]

SYMMETRIC: apo.GeometricRestraintActivation = apo.GeometricRestraintActivation.SYMMETRIC
INSIDE: apo.GeometricRestraintActivation = apo.GeometricRestraintActivation.INSIDE
OUTSIDE: apo.GeometricRestraintActivation = apo.GeometricRestraintActivation.OUTSIDE
ATOMWISE: apo.GeometricRestraintSelectionMode = (
    apo.GeometricRestraintSelectionMode.ATOMWISE
)
RCM: apo.GeometricRestraintSelectionMode = apo.GeometricRestraintSelectionMode.RCM

ADD_PLANE_CONTEXT: str = (
    "GeometricRestraintForce.addPlane(atoms, origin, direction, "
    "force_constant, offset, activation, selection_mode)"
)
ADD_CYLINDER_CONTEXT: str = (
    "GeometricRestraintForce.addCylinder(atoms, origin, direction, "
    "force_constant, offset, activation, selection_mode)"
)
ADD_SPHERE_CONTEXT: str = (
    "GeometricRestraintForce.addSphere(atoms, center, force_constant, "
    "offset, activation, selection_mode)"
)
SET_MASSES_CONTEXT: str = "GeometricRestraintForce.setMasses(masses)"
SET_MASSES_FUNCTION_NAME: str = "apo_geometric_restraint_force_set_masses"
CLOSED_HANDLE_MESSAGE: str = "apoCHARMM object has been destroyed"
ATOMS_TYPE_MESSAGE: str = "atoms must be an AtomSelection or AtomReference"
RCM_TOTAL_DIAGNOSTIC: str = "RCM total mass must be finite and nonzero"

TEST_PARAMETER_TEXT: str = """NONBONDED
DUM 0.0 0.0 1.0
END
"""
BOX_DIMENSIONS: list[float] = [20.0, 30.0, 40.0]
CUTOFF: float = 9.0
CTONNB: float = 8.5
CTOFNB: float = 7.5
DEFAULT_COORDINATES: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (3.0, 4.0, 5.0),
    (-3.0, -4.0, -5.0),
    (7.0, 1.0, -2.0),
    (-7.0, -1.0, 2.0),
    (9.0, 9.0, 9.0),
    (-9.0, -9.0, -9.0),
    (2.0, 5.0, -6.0),
    (-2.0, -5.0, 6.0),
]
EQUAL_MASSES: list[float] = [2.0] * NUM_ATOMS
UNEQUAL_MASSES: list[float] = [
    1.0,
    3.0,
    2.0,
    4.0,
    1.5,
    5.0,
    2.5,
    6.0,
    3.5,
    7.0,
    4.5,
    8.0,
]
NEGATIVE_TOTAL_MASSES: list[float] = [1.0, -3.0] + [2.0] * (NUM_ATOMS - 2)
ZERO_MEMBER_MASSES: list[float] = [0.0] + [2.0] * (NUM_ATOMS - 1)
INV_FORCE_SCALE: float = 2.0**-40
FLOAT_EPSILON: float = 2.0**-23
ARITHMETIC_TOLERANCE: float = 64.0 * FLOAT_EPSILON
DUPLICATE_TERM_COUNT: int = 150

SUBSCRIBE_CONTEXT: str = "ForceManager.subscribe(GeometricRestraintForce)"
UNSUBSCRIBE_CONTEXT: str = "ForceManager.unsubscribe(GeometricRestraintForce)"
SET_PBC_CONTEXT: str = "ForceManager.setPeriodicBoundaryCondition(pbc)"

AtomInput = apo.AtomSelection | apo.AtomReference


class TermAddition(NamedTuple):
    name: str
    context: str
    selection_function: str
    atom_reference_function: str
    add: Callable[[object], None]


class CharmmTerm(NamedTuple):
    geometry: str  # "plane", "cylinder", or "sphere"
    activation: apo.GeometricRestraintActivation
    selection_mode: apo.GeometricRestraintSelectionMode
    atoms: str
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    force_constant: float
    offset: float


class CharmmPhase(NamedTuple):
    name: str
    terms: list[CharmmTerm]
    energy: float
    forces: dict[int, tuple[float, float, float]]
    energy_tolerance: float
    force_tolerance: float


class CharmmCase(NamedTuple):
    name: str
    coordinate_prefix: list[tuple[float, float, float]]
    masses: list[float]
    phases: list[CharmmPhase]


class ManagerOutput(NamedTuple):
    energy: float
    forces: list[list[float]]
    virial: list[list[float]]


class DuckTypedAtoms:
    def __init__(self, handle: ctypes.c_void_p) -> None:
        self._handle: ctypes.c_void_p = handle
        return

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._handle


class RecordingLibrary:
    def __init__(self, library: ctypes.CDLL) -> None:
        self._library: ctypes.CDLL = library
        self.destroyed_handles: list[int | None] = []
        self.force_tags: list[bytes | None] = []
        return

    def __getattr__(self, name: str) -> object:
        return getattr(self._library, name)

    def apo_geometric_restraint_force_destroy(self, handle: ctypes.c_void_p) -> None:
        self.destroyed_handles.append(handle.value)
        self._library.apo_geometric_restraint_force_destroy(handle)
        return

    def apo_force_manager_subscribe_geometric_restraint_force(
        self,
        force_manager_handle: ctypes.c_void_p,
        restraint_handle: ctypes.c_void_p,
        force_tag: ctypes.c_char_p,
    ) -> int:
        self.force_tags.append(force_tag.value)
        return int(
            self._library.apo_force_manager_subscribe_geometric_restraint_force(
                force_manager_handle, restraint_handle, force_tag
            )
        )


CHARMM_CASES: list[CharmmCase] = [
    CharmmCase(
        "INSIDE plane above its offset",
        [(3.0, 0.5, -0.25)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        INSIDE,
                        ATOMWISE,
                        "bynu 1",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                1.0,
                {0: (-2.0, 0.0, 0.0)},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "Symmetric cylinder off its axis",
        [(3.0, 0.5, -0.25)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "cylinder",
                        SYMMETRIC,
                        ATOMWISE,
                        "bynu 1",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0),
                        2.0,
                        2.0,
                    )
                ],
                1.08447494,
                {0: (-2.0544243047, -0.3424040508, 0.0)},
                5e-09,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "Sphere measured from the raw P1 coordinate",
        [(-9.5, 1.0, 0.5)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "sphere",
                        SYMMETRIC,
                        ATOMWISE,
                        "bynu 1",
                        (9.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                273.364988,
                {0: (33.0072846537, -1.7841775488, -0.8920887744)},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "OUTSIDE plane through its atom",
        [],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        OUTSIDE,
                        ATOMWISE,
                        "bynu 1",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                4.0,
                {0: (4.0, 0.0, 0.0)},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "OUTSIDE sphere centered on its atom",
        [(1.25, -2.5, 3.75)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "sphere",
                        OUTSIDE,
                        ATOMWISE,
                        "bynu 1",
                        (1.25, -2.5, 3.75),
                        (0.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                0.0,
                {},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "Plane term over every atom",
        [(3.0, 0.0, 0.0)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        SYMMETRIC,
                        ATOMWISE,
                        "all",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                328.0,
                {
                    0: (-2.0, 0.0, 0.0),
                    1: (2.0, 0.0, 0.0),
                    2: (4.0, 0.0, 0.0),
                    3: (4.0, 0.0, 0.0),
                    4: (-2.0, 0.0, 0.0),
                    5: (10.0, 0.0, 0.0),
                    6: (-10.0, 0.0, 0.0),
                    7: (18.0, 0.0, 0.0),
                    8: (-14.0, 0.0, 0.0),
                    9: (22.0, 0.0, 0.0),
                    11: (8.0, 0.0, 0.0),
                },
                5e-05,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "RCM sphere over three equal masses",
        [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
        EQUAL_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "sphere",
                        SYMMETRIC,
                        RCM,
                        "bynu 1:3",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                2.14359354,
                {
                    0: (-0.5635329744, -0.5635329744, -0.5635329744),
                    1: (-0.5635329744, -0.5635329744, -0.5635329744),
                    2: (-0.5635329744, -0.5635329744, -0.5635329744),
                },
                5e-09,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "Plane, cylinder, and RCM sphere terms together",
        [(3.0, 2.0, 1.0), (4.0, 1.0, 2.0), (5.0, 3.0, 1.0)],
        UNEQUAL_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        SYMMETRIC,
                        ATOMWISE,
                        "bynu 1:2",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        1.0,
                        1.5,
                    ),
                    CharmmTerm(
                        "cylinder",
                        OUTSIDE,
                        ATOMWISE,
                        "bynu 2:3",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0),
                        1.25,
                        2.0,
                    ),
                    CharmmTerm(
                        "sphere",
                        SYMMETRIC,
                        RCM,
                        "bynu 1:3",
                        (1.0, 1.0, 1.0),
                        (0.0, 0.0, 0.0),
                        0.75,
                        2.5,
                    ),
                ],
                4.49751867,
                {
                    0: (-1.5970852864, -0.0255487596, -0.0153292557),
                    1: (-2.7912558591, -0.0766462787, -0.0459877672),
                    2: (-0.1941705727, -0.0510975191, -0.0306585115),
                },
                5e-09,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "OUTSIDE plane beyond its offset",
        [(3.0, 0.5, -0.25)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        OUTSIDE,
                        ATOMWISE,
                        "bynu 1",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                0.0,
                {},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "OUTSIDE cylinder beyond its offset",
        [(3.0, 0.5, -0.25)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "cylinder",
                        OUTSIDE,
                        ATOMWISE,
                        "bynu 1",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0),
                        2.0,
                        2.0,
                    )
                ],
                0.0,
                {},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "OUTSIDE sphere beyond its offset",
        [(3.0, 0.5, -0.25)],
        STANDARD_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "sphere",
                        OUTSIDE,
                        ATOMWISE,
                        "bynu 1",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                0.0,
                {},
                5e-07,
                1e-10,
            )
        ],
    ),
]
RESET_REUSE_CASE: CharmmCase = CharmmCase(
    "Reset then reuse with an RCM term",
    [(1.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
    UNEQUAL_MASSES,
    [
        CharmmPhase(
            "Before reset",
            [
                CharmmTerm(
                    "plane",
                    SYMMETRIC,
                    ATOMWISE,
                    "bynu 1",
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    2.0,
                    2.0,
                )
            ],
            1.0,
            {0: (2.0, 0.0, 0.0)},
            5e-07,
            1e-10,
        ),
        CharmmPhase(
            "New RCM term after reset",
            [
                CharmmTerm(
                    "plane",
                    SYMMETRIC,
                    RCM,
                    "bynu 1:2",
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    2.0,
                    2.0,
                )
            ],
            4.0,
            {0: (-1.0, 0.0, 0.0), 1: (-3.0, 0.0, 0.0)},
            5e-07,
            1e-10,
        ),
    ],
)
RCM_MASS_CASES: list[CharmmCase] = [
    CharmmCase(
        "RCM plane with equal masses",
        [(1.0, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)],
        EQUAL_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        SYMMETRIC,
                        RCM,
                        "bynu 1:2",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                1.0,
                {0: (-1.0, 0.0, 0.0), 1: (-1.0, 0.0, 0.0)},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "RCM plane with unequal masses",
        [(1.0, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)],
        UNEQUAL_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        SYMMETRIC,
                        RCM,
                        "bynu 1:2",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                4.0,
                {0: (-1.0, 0.0, 0.0), 1: (-3.0, 0.0, 0.0)},
                5e-07,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "RCM plane with a negative selected total",
        [(1.0, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)],
        NEGATIVE_TOTAL_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        SYMMETRIC,
                        RCM,
                        "bynu 1:2",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                25.0,
                {0: (5.0, 0.0, 0.0), 1: (-15.0, 0.0, 0.0)},
                5e-06,
                1e-10,
            )
        ],
    ),
    CharmmCase(
        "RCM plane with a zero-mass member",
        [(1.0, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)],
        ZERO_MEMBER_MASSES,
        [
            CharmmPhase(
                "Restraint evaluated",
                [
                    CharmmTerm(
                        "plane",
                        SYMMETRIC,
                        RCM,
                        "bynu 1:2",
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        2.0,
                        2.0,
                    )
                ],
                9.0,
                {1: (-6.0, 0.0, 0.0)},
                5e-07,
                1e-10,
            )
        ],
    ),
]


def select(psf: apo.CharmmPsf, selection_string: str) -> apo.AtomSelection:
    selector = apo.AtomSelector(psf)

    try:
        return selector.select(selection_string)
    finally:
        selector.close()


def select_atom(psf: apo.CharmmPsf, selection_string: str) -> apo.AtomReference:
    selector = apo.AtomSelector(psf)

    try:
        return selector.selectAtom(selection_string)
    finally:
        selector.close()


def masses_with(atom_index: int, mass: float) -> list[float]:
    masses: list[float] = list(STANDARD_MASSES)
    masses[atom_index] = mass
    return masses


def make_charmm_masses(pair_masses: tuple[float, float]) -> list[float]:
    unselected_masses: list[float] = [CHARMM_UNSELECTED_MASS] * (NUM_ATOMS - 2)
    return [pair_masses[0], pair_masses[1]] + unselected_masses


def make_term_additions(
    restraint: apo.GeometricRestraintForce,
    activation: apo.GeometricRestraintActivation,
    selection_mode: apo.GeometricRestraintSelectionMode,
) -> list[TermAddition]:
    return [
        TermAddition(
            "addPlane",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane_from_selection",
            "apo_geometric_restraint_force_add_plane_from_atom_reference",
            lambda atoms: restraint.addPlane(
                cast(AtomInput, atoms),
                ORIGIN,
                X_DIRECTION,
                2.0,
                1.0,
                activation,
                selection_mode,
            ),
        ),
        TermAddition(
            "addCylinder",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder_from_selection",
            "apo_geometric_restraint_force_add_cylinder_from_atom_reference",
            lambda atoms: restraint.addCylinder(
                cast(AtomInput, atoms),
                ORIGIN,
                Z_DIRECTION,
                2.0,
                1.0,
                activation,
                selection_mode,
            ),
        ),
        TermAddition(
            "addSphere",
            ADD_SPHERE_CONTEXT,
            "apo_geometric_restraint_force_add_sphere_from_selection",
            "apo_geometric_restraint_force_add_sphere_from_atom_reference",
            lambda atoms: restraint.addSphere(
                cast(AtomInput, atoms), ORIGIN, 2.0, 1.0, activation, selection_mode
            ),
        ),
    ]


def expect_closed_handle(label: str, action: Callable[[], object]) -> None:
    error: RuntimeError = apo_test.expect_exception(label, RuntimeError, action)

    apo_test.assert_equal(f"{label} exception type", type(error) is RuntimeError, True)
    apo_test.assert_equal(f"{label} message", str(error), CLOSED_HANDLE_MESSAGE)

    return


def expect_atoms_type_error(label: str, action: Callable[[], object]) -> None:
    error: TypeError = apo_test.expect_exception(label, TypeError, action)

    apo_test.assert_equal(f"{label} message", str(error), ATOMS_TYPE_MESSAGE)

    return


def assert_no_rcm_term(label: str, restraint: apo.GeometricRestraintForce) -> None:
    try:
        restraint.setMasses([0.0] * NUM_ATOMS)
    except apo.ApoCharmmError as exc:
        raise AssertionError(f"{label}: an RCM term is installed") from exc

    restraint.setMasses(STANDARD_MASSES)

    return


def make_force_manager(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> apo.ForceManager:
    force_manager = apo.ForceManager(psf, parameters)

    try:
        force_manager.setBoxDimensions(BOX_DIMENSIONS)
        force_manager.setCutoff(CUTOFF)
        force_manager.setCtonnb(CTONNB)
        force_manager.setCtofnb(CTOFNB)
        force_manager.setPeriodicBoundaryCondition(apo.PeriodicBoundaryCondition.P1)
    except BaseException:
        force_manager.close()
        raise

    return force_manager


def make_context(force_manager: apo.ForceManager) -> apo.CharmmContext:
    context = apo.CharmmContext(force_manager)

    apo_test.assert_equal(
        "CharmmContext initializes the configured ForceManager",
        force_manager.isInitialized(),
        True,
    )

    return context


def evaluated_coordinates(case: CharmmCase) -> list[list[float]]:
    coordinates: list[list[float]] = [
        list(position) for position in DEFAULT_COORDINATES
    ]

    for index, position in enumerate(case.coordinate_prefix):
        coordinates[index] = list(position)

    return coordinates


def evaluate(
    force_manager: apo.ForceManager,
    context: apo.CharmmContext,
    coordinates: Sequence[Sequence[float]],
) -> ManagerOutput:
    context.setCoordinates(coordinates)
    context.calculatePotentialEnergy()

    return ManagerOutput(
        force_manager.getPotentialEnergy(),
        force_manager.getForces(),
        force_manager.getVirial(),
    )


def add_charmm_terms(
    restraint: apo.GeometricRestraintForce,
    psf: apo.CharmmPsf,
    terms: Sequence[CharmmTerm],
    use_atom_references: bool,
) -> None:
    for term in terms:
        atoms: AtomInput = (
            select_atom(psf, term.atoms)
            if use_atom_references
            else select(psf, term.atoms)
        )

        try:
            if term.geometry == "plane":
                restraint.addPlane(
                    atoms,
                    term.origin,
                    term.direction,
                    term.force_constant,
                    term.offset,
                    term.activation,
                    term.selection_mode,
                )
            elif term.geometry == "cylinder":
                restraint.addCylinder(
                    atoms,
                    term.origin,
                    term.direction,
                    term.force_constant,
                    term.offset,
                    term.activation,
                    term.selection_mode,
                )
            else:
                restraint.addSphere(
                    atoms,
                    term.origin,
                    term.force_constant,
                    term.offset,
                    term.activation,
                    term.selection_mode,
                )
        finally:
            atoms.close()

    return


def check_manager_output(
    label: str, phase: CharmmPhase, baseline: ManagerOutput, observed: ManagerOutput
) -> None:
    force_magnitude: float = max(
        (abs(component) for force in phase.forces.values() for component in force),
        default=0.0,
    )
    energy_tolerance: float = phase.energy_tolerance + ARITHMETIC_TOLERANCE * abs(
        phase.energy
    )
    force_tolerance: float = (
        phase.force_tolerance
        + 0.5 * INV_FORCE_SCALE
        + 0.5 * FLOAT_EPSILON * force_magnitude
        + ARITHMETIC_TOLERANCE * max(1.0, force_magnitude)
    )

    apo_test.assert_equal(f"{label}: force rows", len(observed.forces), NUM_ATOMS)
    apo_test.assert_equal(
        f"{label}: virial shape", [len(row) for row in observed.virial], [3, 3, 3]
    )

    apo_test.assert_close(
        f"{label}: aggregate energy",
        observed.energy - baseline.energy,
        phase.energy,
        energy_tolerance,
    )

    for atom in range(NUM_ATOMS):
        expected_force: tuple[float, float, float] = phase.forces.get(
            atom, (0.0, 0.0, 0.0)
        )

        for k in range(3):
            apo_test.assert_close(
                f"{label}: force on atom {atom}, component {'xyz'[k]}",
                observed.forces[atom][k] - baseline.forces[atom][k],
                expected_force[k],
                force_tolerance,
            )

    apo_test.assert_equal(f"{label}: virial", observed.virial, baseline.virial)

    return


def check_no_contribution(
    label: str, baseline: ManagerOutput, observed: ManagerOutput
) -> None:
    apo_test.assert_equal(f"{label}: energy", observed.energy, baseline.energy)
    apo_test.assert_equal(f"{label}: forces", observed.forces, baseline.forces)
    apo_test.assert_equal(f"{label}: virial", observed.virial, baseline.virial)
    return


def run_charmm_case(
    psf: apo.CharmmPsf,
    force_manager: apo.ForceManager,
    context: apo.CharmmContext,
    case: CharmmCase,
    use_atom_references: bool,
) -> None:
    route: str = "AtomReference" if use_atom_references else "AtomSelection"
    coordinates: list[list[float]] = evaluated_coordinates(case)
    restraint = apo.GeometricRestraintForce(psf)

    try:
        force_manager.subscribe(restraint)

        try:
            baseline: ManagerOutput = evaluate(force_manager, context, coordinates)
            restraint.setMasses(case.masses)

            for phase in case.phases:
                add_charmm_terms(restraint, psf, phase.terms, use_atom_references)
                check_manager_output(
                    f"{case.name} through {route} inputs, {phase.name}",
                    phase,
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )
        finally:
            force_manager.unsubscribe(restraint)
    finally:
        restraint.close()

    print(f"Matched CHARMM through {route} inputs: {case.name}")

    return


def check_exports_and_enums() -> None:
    print("Checking GeometricRestraintForce exports and enumerations...")

    for name in (
        "GeometricRestraintActivation",
        "GeometricRestraintSelectionMode",
        "GeometricRestraintForce",
    ):
        apo_test.assert_equal(f"{name} in apocharmm.__all__", name in apo.__all__, True)

    apo_test.assert_equal(
        "GeometricRestraintForce top-level module",
        apo.GeometricRestraintForce.__module__,
        "apocharmm.geometric_restraint_force",
    )
    apo_test.assert_equal(
        "GeometricRestraintActivation module",
        apo.GeometricRestraintActivation.__module__,
        "apocharmm.enums",
    )
    apo_test.assert_equal(
        "GeometricRestraintSelectionMode module",
        apo.GeometricRestraintSelectionMode.__module__,
        "apocharmm.enums",
    )
    apo_test.assert_equal(
        "GeometricRestraintActivation members",
        [(member.name, member.value) for member in apo.GeometricRestraintActivation],
        [("SYMMETRIC", 0), ("INSIDE", 1), ("OUTSIDE", -1)],
    )
    apo_test.assert_equal(
        "GeometricRestraintSelectionMode members",
        [(member.name, member.value) for member in apo.GeometricRestraintSelectionMode],
        [("ATOMWISE", 0), ("RCM", 1)],
    )
    apo_test.assert_equal(
        "GeometricRestraintForce public API",
        sorted(
            name
            for name in dir(apo.GeometricRestraintForce)
            if not name.startswith("_")
        ),
        [
            "addCylinder",
            "addPlane",
            "addSphere",
            "close",
            "default_force_tag",
            "destroy",
            "handle",
            "reset",
            "setMasses",
        ],
    )

    return


def check_construction(psf_path: Path, psf: apo.CharmmPsf) -> None:
    print("Checking GeometricRestraintForce construction and default tag...")

    rejected_psfs: list[tuple[str, object]] = [
        ("an arbitrary object", object()),
        ("an atom count", NUM_ATOMS),
        ("a PSF path string", str(psf_path)),
        ("a PSF Path", psf_path),
        ("None", None),
    ]

    for label, rejected_psf in rejected_psfs:
        error: TypeError = apo_test.expect_exception(
            f"GeometricRestraintForce rejects {label}",
            TypeError,
            lambda value=rejected_psf: apo.GeometricRestraintForce(
                cast(apo.CharmmPsf, value)
            ),
        )
        apo_test.assert_equal(
            f"GeometricRestraintForce rejects {label} message",
            str(error),
            "GeometricRestraintForce expects a CharmmPsf",
        )

    closed_psf = apo.CharmmPsf(str(psf_path))
    closed_psf.close()

    expect_closed_handle(
        "GeometricRestraintForce rejects a closed CharmmPsf",
        lambda: apo.GeometricRestraintForce(closed_psf),
    )

    restraint = apo.GeometricRestraintForce(psf)

    try:
        apo_test.assert_equal(
            "constructed GeometricRestraintForce handle is non-NULL",
            restraint.handle.value is not None,
            True,
        )
        apo_test.assert_equal(
            "GeometricRestraintForce.default_force_tag",
            restraint.default_force_tag,
            "mmfp",
        )
        apo_test.expect_exception(
            "GeometricRestraintForce.default_force_tag is read-only",
            AttributeError,
            lambda: setattr(restraint, "default_force_tag", "geo"),
        )

        restraint.reset()
        assert_no_rcm_term("new restraint", restraint)
    finally:
        restraint.close()

    return


def check_psf_lifetime(psf_path: Path, larger_psf: apo.CharmmPsf) -> None:
    print("Checking PSF lifetime after GeometricRestraintForce construction...")

    source_psf = apo.CharmmPsf(str(psf_path))
    source_psf_weak: weakref.ref[apo.CharmmPsf] = weakref.ref(source_psf)
    restraint = apo.GeometricRestraintForce(source_psf)

    source_psf.close()
    del source_psf
    gc.collect()

    apo_test.assert_equal(
        "GeometricRestraintForce does not retain the Python CharmmPsf wrapper",
        source_psf_weak() is None,
        True,
    )

    selection_psf = apo.CharmmPsf(str(psf_path))
    pair = select(selection_psf, "bynu 1:2")
    last_atom = select_atom(selection_psf, f"bynu {NUM_ATOMS}")
    outside_atom = select_atom(larger_psf, f"bynu {NUM_ATOMS + 1}")

    try:
        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, RCM)
        restraint.addSphere(last_atom, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        apo_test.expect_invalid_argument(
            "copied atom count bounds AtomReference indices after PSF closure",
            lambda: restraint.addSphere(
                outside_atom, ORIGIN, 2.0, 1.0, INSIDE, ATOMWISE
            ),
            f"Atom index is out of range; expected [0, {NUM_ATOMS}), "
            f"observed {NUM_ATOMS}",
            expected_context=ADD_SPHERE_CONTEXT,
        )

        apo_test.expect_invalid_argument(
            "RCM terms use the PSF masses copied at construction",
            lambda: restraint.setMasses(masses_with(NUM_ATOMS - 1, 0.0)),
            RCM_TOTAL_DIAGNOSTIC,
            expected_context=SET_MASSES_CONTEXT,
        )
    finally:
        outside_atom.close()
        last_atom.close()
        pair.close()
        selection_psf.close()
        restraint.close()

    return


def check_closure(psf: apo.CharmmPsf) -> None:
    print("Checking GeometricRestraintForce closure and context management...")

    pair = select(psf, "bynu 1:2")
    atom = apo.AtomReference(psf, 0)
    restraint = apo.GeometricRestraintForce(psf)

    try:
        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, ATOMWISE)

        restraint.close()

        expect_closed_handle(
            "closed restraint rejects handle access", lambda: restraint.handle
        )

        for atoms in (pair, atom):
            for addition in make_term_additions(restraint, SYMMETRIC, ATOMWISE):
                expect_closed_handle(
                    f"closed restraint rejects {addition.name} with "
                    f"{type(atoms).__name__}",
                    lambda addition=addition, atoms=atoms: addition.add(atoms),
                )

        expect_closed_handle(
            "closed restraint rejects setMasses",
            lambda: restraint.setMasses(STANDARD_MASSES),
        )
        expect_closed_handle("closed restraint rejects reset", restraint.reset)

        apo_test.assert_equal(
            "default_force_tag remains available after close",
            restraint.default_force_tag,
            "mmfp",
        )
        apo_test.assert_equal(
            "closing the restraint leaves its AtomSelection usable",
            pair.getAtomIndices(),
            [0, 1],
        )
        apo_test.assert_equal(
            "closing the restraint leaves its AtomReference usable",
            atom.getAtomIndex(),
            0,
        )

        restraint.close()
        restraint.destroy()
        restraint.close()

        with apo.GeometricRestraintForce(psf) as managed:
            managed.addSphere(atom, ORIGIN, 2.0, 1.0, OUTSIDE, ATOMWISE)

        expect_closed_handle("context-manager exit closes the restraint", managed.reset)

        entered_restraint = apo.GeometricRestraintForce(psf)

        with entered_restraint as entered:
            apo_test.assert_equal(
                "context manager returns the same wrapper",
                entered is entered_restraint,
                True,
            )

        expect_closed_handle(
            "context-manager exit closes the entered wrapper",
            entered_restraint.reset,
        )

        failing_restraint = apo.GeometricRestraintForce(psf)

        def fail_inside_context() -> None:
            with failing_restraint:
                raise LookupError("context body failure")

        apo_test.expect_exception(
            "context-manager exit propagates the body exception",
            LookupError,
            fail_inside_context,
        )
        expect_closed_handle(
            "context-manager exit closes the restraint after an exception",
            failing_restraint.reset,
        )
    finally:
        restraint.close()
        atom.close()
        pair.close()

    return


def check_finalizer_ownership(psf: apo.CharmmPsf) -> None:
    print("Checking GeometricRestraintForce finalizer ownership...")

    apo_test.assert_equal(
        "GeometricRestraintForce destroy function name",
        apo.GeometricRestraintForce._destroy_function_name,
        "apo_geometric_restraint_force_destroy",
    )

    recording_library = RecordingLibrary(apo_lib.lib())

    restraint = apo.GeometricRestraintForce(psf)
    restraint_weak: weakref.ref[apo.GeometricRestraintForce] = weakref.ref(restraint)
    handle_value: int | None = restraint.handle.value

    closed_restraint = apo.GeometricRestraintForce(psf)
    closed_restraint.close()
    closed_restraint_weak: weakref.ref[apo.GeometricRestraintForce] = weakref.ref(
        closed_restraint
    )

    gc.collect()

    with patch("apocharmm._base.lib", return_value=recording_library):
        del restraint
        del closed_restraint
        gc.collect()

    apo_test.assert_equal(
        "finalized GeometricRestraintForce wrapper was collected",
        restraint_weak() is None,
        True,
    )
    apo_test.assert_equal(
        "closed GeometricRestraintForce wrapper was collected",
        closed_restraint_weak() is None,
        True,
    )
    apo_test.assert_equal(
        "finalization destroys only the open owned handle, exactly once",
        recording_library.destroyed_handles,
        [handle_value],
    )

    return


def check_enum_validation(psf: apo.CharmmPsf) -> None:
    print("Checking activation and selection-mode enum validation...")

    pair = select(psf, "bynu 1:2")
    atom = apo.AtomReference(psf, 0)
    restraint = apo.GeometricRestraintForce(psf)

    rejected_activations: list[tuple[str, object]] = [
        ("int 0", 0),
        ("int 1", 1),
        ("int -1", -1),
        ("bool", True),
        ("selection-mode member", RCM),
        ("distance-condition member", apo.DistanceRestraintCondition.POSITIVE),
        ("member name", "INSIDE"),
        ("None", None),
    ]
    rejected_selection_modes: list[tuple[str, object]] = [
        ("int 0", 0),
        ("int 1", 1),
        ("bool", True),
        ("activation member", INSIDE),
        ("distance-condition member", apo.DistanceRestraintCondition.NONE),
        ("member name", "RCM"),
        ("None", None),
    ]

    try:
        for atoms in (pair, atom):
            for label, activation in rejected_activations:
                for addition in make_term_additions(
                    restraint, cast(apo.GeometricRestraintActivation, activation), RCM
                ):
                    error: TypeError = apo_test.expect_exception(
                        f"{addition.name} rejects activation {label}",
                        TypeError,
                        lambda addition=addition, atoms=atoms: addition.add(atoms),
                    )
                    apo_test.assert_equal(
                        f"{addition.name} rejects activation {label} message",
                        str(error),
                        "activation must be a GeometricRestraintActivation",
                    )

            for label, selection_mode in rejected_selection_modes:
                for addition in make_term_additions(
                    restraint,
                    INSIDE,
                    cast(apo.GeometricRestraintSelectionMode, selection_mode),
                ):
                    error = apo_test.expect_exception(
                        f"{addition.name} rejects selection_mode {label}",
                        TypeError,
                        lambda addition=addition, atoms=atoms: addition.add(atoms),
                    )
                    apo_test.assert_equal(
                        f"{addition.name} rejects selection_mode {label} message",
                        str(error),
                        "selection_mode must be a GeometricRestraintSelectionMode",
                    )

        assert_no_rcm_term("rejected enum values install no RCM term", restraint)

        for atoms in (pair, atom):
            for activation in apo.GeometricRestraintActivation:
                for selection_mode in apo.GeometricRestraintSelectionMode:
                    for addition in make_term_additions(
                        restraint, activation, selection_mode
                    ):
                        addition.add(atoms)
    finally:
        restraint.close()
        atom.close()
        pair.close()

    return


def check_atom_inputs(psf: apo.CharmmPsf, second_psf: apo.CharmmPsf) -> None:
    print("Checking AtomSelection and AtomReference term inputs...")

    restraint = apo.GeometricRestraintForce(psf)
    all_atoms = select(psf, "all")
    pair = select(psf, "bynu 1:2")
    other_topology_pair = select(second_psf, "bynu 1:2")
    selected_atom = select_atom(psf, "bynu 1")
    direct_atom = apo.AtomReference(psf, NUM_ATOMS - 1)

    accepted_atoms: list[tuple[str, AtomInput]] = [
        ("all-atom AtomSelection", all_atoms),
        ("pair AtomSelection", pair),
        ("pair AtomSelection from another topology", other_topology_pair),
        ("selector AtomReference", selected_atom),
        ("directly constructed AtomReference", direct_atom),
    ]

    try:
        for label, atoms in accepted_atoms:
            for selection_mode in apo.GeometricRestraintSelectionMode:
                for addition in make_term_additions(
                    restraint, SYMMETRIC, selection_mode
                ):
                    addition.add(atoms)

            print(f"Accepted {label} for every geometry and selection mode")

        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, RCM)

        apo_test.expect_invalid_argument(
            "pair RCM terms use copied indices 0 and 1",
            lambda: restraint.setMasses(masses_with(0, -1.5)),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )
        apo_test.expect_invalid_argument(
            "single-atom RCM term uses the copied AtomReference index",
            lambda: restraint.setMasses(masses_with(NUM_ATOMS - 1, 0.0)),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )
    finally:
        direct_atom.close()
        selected_atom.close()
        other_topology_pair.close()
        pair.close()
        all_atoms.close()
        restraint.close()

    return


def check_rejected_atom_forms(psf: apo.CharmmPsf) -> None:
    print("Checking rejected raw-index, collection, and duck-typed atom inputs...")

    restraint = apo.GeometricRestraintForce(psf)
    pair = select(psf, "bynu 1:2")
    first = apo.AtomReference(psf, 0)
    second = apo.AtomReference(psf, 1)

    rejected_atoms: list[tuple[str, object]] = [
        ("an int index", 0),
        ("a bool index", True),
        ("an int list", [0, 1]),
        ("a one-element int tuple", (0,)),
        ("an int range", range(2)),
        ("an AtomReference list", [first, second]),
        ("a one-element AtomReference list", [first]),
        ("an AtomReference tuple", (first, second)),
        ("a mixed AtomSelection and AtomReference list", [pair, first]),
        ("a mixed AtomReference and int list", [first, 1]),
        ("a one-element AtomSelection tuple", (pair,)),
        ("a duck-typed AtomSelection handle", DuckTypedAtoms(pair.handle)),
        ("a duck-typed AtomReference handle", DuckTypedAtoms(first.handle)),
        ("a raw ctypes handle", pair.handle),
        ("a CharmmPsf", psf),
        ("None", None),
    ]

    try:
        for label, atoms in rejected_atoms:
            for addition in make_term_additions(restraint, SYMMETRIC, RCM):
                expect_atoms_type_error(
                    f"{addition.name} rejects {label}",
                    lambda addition=addition, atoms=atoms: addition.add(atoms),
                )

        assert_no_rcm_term("rejected atom forms install no term", restraint)
    finally:
        second.close()
        first.close()
        pair.close()
        restraint.close()

    return


def check_closed_atom_inputs(psf: apo.CharmmPsf) -> None:
    print("Checking closed AtomSelection and AtomReference inputs...")

    restraint = apo.GeometricRestraintForce(psf)
    closed_pair = select(psf, "bynu 1:2")
    closed_atom = apo.AtomReference(psf, 0)

    closed_pair.close()
    closed_atom.close()

    try:
        for label, atoms in (
            ("a closed AtomSelection", closed_pair),
            ("a closed AtomReference", closed_atom),
        ):
            for addition in make_term_additions(restraint, SYMMETRIC, RCM):
                expect_closed_handle(
                    f"{addition.name} rejects {label}",
                    lambda addition=addition, atoms=atoms: addition.add(atoms),
                )

        assert_no_rcm_term("closed atom inputs install no term", restraint)
    finally:
        restraint.close()

    return


def check_empty_selection(psf: apo.CharmmPsf, larger_psf: apo.CharmmPsf) -> None:
    print("Checking that empty AtomSelections add no term, as in CHARMM...")

    restraint = apo.GeometricRestraintForce(psf)
    empty = select(psf, "none")
    larger_empty = select(larger_psf, "none")
    first = apo.AtomReference(psf, 0)

    try:
        apo_test.assert_equal("empty selection count", empty.getNumSelected(), 0)

        restraint.setMasses([0.0] * NUM_ATOMS)

        for selection_mode in apo.GeometricRestraintSelectionMode:
            for addition in make_term_additions(restraint, INSIDE, selection_mode):
                addition.add(empty)

        apo_test.expect_invalid_argument(
            "zero masses still reject a nonempty RCM term",
            lambda: restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_SPHERE_CONTEXT,
        )

        restraint.setMasses([0.0] * NUM_ATOMS)
        restraint.setMasses(STANDARD_MASSES)

        apo_test.expect_invalid_argument(
            "empty selection still validates the force constant",
            lambda: restraint.addPlane(
                empty, ORIGIN, X_DIRECTION, math.inf, 1.0, INSIDE, ATOMWISE
            ),
            "Force constant must be finite",
            expected_context=ADD_PLANE_CONTEXT,
        )
        apo_test.expect_invalid_argument(
            "empty selection still validates the direction",
            lambda: restraint.addCylinder(
                empty, ORIGIN, [0.0, 0.0, 0.0], 2.0, 1.0, INSIDE, ATOMWISE
            ),
            "Direction must have a positive finite norm",
            expected_context=ADD_CYLINDER_CONTEXT,
        )
        apo_test.expect_invalid_argument(
            "empty selection still validates the center length",
            lambda: restraint.addSphere(empty, [0.0, 0.0], 2.0, 1.0, INSIDE, RCM),
            "center must contain exactly 3 elements",
            expected_context=ADD_SPHERE_CONTEXT,
        )

        for addition in make_term_additions(restraint, INSIDE, RCM):
            apo_test.expect_invalid_argument(
                f"{addition.name} rejects an empty selection of another atom count",
                lambda addition=addition: addition.add(larger_empty),
                f"Selection atom count mismatch; expected {NUM_ATOMS}, "
                f"observed {LARGER_NUM_ATOMS}",
                expected_context=addition.context,
            )

        assert_no_rcm_term("empty selections install no term", restraint)
    finally:
        first.close()
        larger_empty.close()
        empty.close()
        restraint.close()

    return


def check_force_local_validation(psf: apo.CharmmPsf, larger_psf: apo.CharmmPsf) -> None:
    print("Checking atom-count and force-local AtomReference validation...")

    restraint = apo.GeometricRestraintForce(psf)
    larger_all = select(larger_psf, "all")
    larger_first = select_atom(larger_psf, "bynu 1")
    larger_last_valid = select_atom(larger_psf, f"bynu {NUM_ATOMS}")
    larger_outside = select_atom(larger_psf, f"bynu {NUM_ATOMS + 1}")
    larger_end = apo.AtomReference(larger_psf, LARGER_NUM_ATOMS - 1)

    try:
        for selection_mode in apo.GeometricRestraintSelectionMode:
            for addition in make_term_additions(restraint, SYMMETRIC, selection_mode):
                mismatch_error = apo_test.expect_invalid_argument(
                    f"{addition.name} rejects an AtomSelection atom-count mismatch",
                    lambda addition=addition: addition.add(larger_all),
                    f"Selection atom count mismatch; expected {NUM_ATOMS}, "
                    f"observed {LARGER_NUM_ATOMS}",
                    expected_context=addition.context,
                )
                apo_test.assert_equal(
                    f"{addition.name} mismatch uses the AtomSelection entry point",
                    mismatch_error.message.count(addition.selection_function),
                    1,
                )

                addition.add(larger_first)
                addition.add(larger_last_valid)

                outside_error = apo_test.expect_invalid_argument(
                    f"{addition.name} rejects a force-local index of {NUM_ATOMS}",
                    lambda addition=addition: addition.add(larger_outside),
                    f"Atom index is out of range; expected [0, {NUM_ATOMS}), "
                    f"observed {NUM_ATOMS}",
                    expected_context=addition.context,
                )
                apo_test.assert_equal(
                    f"{addition.name} invalid index uses the AtomReference entry "
                    "point",
                    outside_error.message.count(addition.atom_reference_function),
                    1,
                )

                apo_test.expect_invalid_argument(
                    f"{addition.name} rejects a force-local index of "
                    f"{LARGER_NUM_ATOMS - 1}",
                    lambda addition=addition: addition.add(larger_end),
                    f"Atom index is out of range; expected [0, {NUM_ATOMS}), "
                    f"observed {LARGER_NUM_ATOMS - 1}",
                    expected_context=addition.context,
                )

        apo_test.expect_invalid_argument(
            "cross-topology AtomReference RCM terms use force-local indices",
            lambda: restraint.setMasses(masses_with(NUM_ATOMS - 1, 0.0)),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )
    finally:
        larger_end.close()
        larger_outside.close()
        larger_last_valid.close()
        larger_first.close()
        larger_all.close()
        restraint.close()

    return


def check_input_lifetime(psf_path: Path, psf: apo.CharmmPsf) -> None:
    print("Checking atom-input lifetime after term addition...")

    restraint = apo.GeometricRestraintForce(psf)

    try:
        source_psf = apo.CharmmPsf(str(psf_path))
        pair = select(source_psf, "bynu 1:2")
        atom = select_atom(source_psf, "bynu 3")

        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, RCM)
        restraint.addCylinder(pair, ORIGIN, Z_DIRECTION, 2.0, 1.0, OUTSIDE, ATOMWISE)
        restraint.addSphere(atom, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        source_psf_weak: weakref.ref[apo.CharmmPsf] = weakref.ref(source_psf)
        pair_weak: weakref.ref[apo.AtomSelection] = weakref.ref(pair)
        atom_weak: weakref.ref[apo.AtomReference] = weakref.ref(atom)

        pair.close()
        atom.destroy()
        source_psf.close()
        del pair, atom, source_psf
        gc.collect()

        apo_test.assert_equal(
            "restraint retains no AtomSelection wrapper", pair_weak() is None, True
        )
        apo_test.assert_equal(
            "restraint retains no AtomReference wrapper", atom_weak() is None, True
        )
        apo_test.assert_equal(
            "restraint retains no atom-source CharmmPsf wrapper",
            source_psf_weak() is None,
            True,
        )

        apo_test.expect_invalid_argument(
            "pair RCM term outlives its closed AtomSelection",
            lambda: restraint.setMasses(masses_with(1, -1.0)),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )
        apo_test.expect_invalid_argument(
            "atom RCM term outlives its destroyed AtomReference",
            lambda: restraint.setMasses(masses_with(2, 0.0)),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )

        restraint.reset()
        assert_no_rcm_term(
            "reset removes RCM terms added from closed inputs", restraint
        )
    finally:
        restraint.close()

    return


def check_mass_conversion(psf: apo.CharmmPsf) -> None:
    print("Checking setMasses() conversion and transport...")

    restraint = apo.GeometricRestraintForce(psf)
    first = apo.AtomReference(psf, 0)
    second = apo.AtomReference(psf, 1)

    try:
        integer_masses: tuple[int, ...] = (0,) + tuple(range(2, NUM_ATOMS + 1))
        restraint.setMasses(integer_masses)

        apo_test.expect_invalid_argument(
            "integer zero mass reaches native RCM validation as 0.0",
            lambda: restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_SPHERE_CONTEXT,
        )
        restraint.addSphere(second, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        restraint.reset()

        restraint.setMasses(cast(Sequence[float], [Fraction(1, 3)] * NUM_ATOMS))
        restraint.setMasses(array.array("d", STANDARD_MASSES))
        restraint.setMasses(
            cast(Sequence[float], (mass for mass in masses_with(0, 0.0)))
        )

        apo_test.expect_invalid_argument(
            "single-pass iterable masses reach native code",
            lambda: restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_SPHERE_CONTEXT,
        )
        restraint.addSphere(second, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        restraint.reset()
        restraint.setMasses(STANDARD_MASSES)

        conversion_failures: list[tuple[str, type[Exception], object]] = [
            ("a non-iterable value", TypeError, 1.0),
            ("None", TypeError, None),
            ("a non-numeric element", TypeError, [object()] * NUM_ATOMS),
            (
                "an unparsable string element",
                ValueError,
                [0.0, "mass"] + [1.0] * (NUM_ATOMS - 2),
            ),
            (
                "an overflowing integer element",
                OverflowError,
                [0.0, 10**400] + [1.0] * (NUM_ATOMS - 2),
            ),
        ]

        for label, exception_type, masses in conversion_failures:
            apo_test.expect_exception(
                f"setMasses rejects {label} before the C call",
                exception_type,
                lambda masses=masses: restraint.setMasses(
                    cast(Sequence[float], masses)
                ),
            )

        restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM)
    finally:
        second.close()
        first.close()
        restraint.close()

    return


def check_mass_buffer_copying(psf: apo.CharmmPsf) -> None:
    print("Checking setMasses() caller-buffer copying...")

    restraint = apo.GeometricRestraintForce(psf)
    first = apo.AtomReference(psf, 0)
    second = apo.AtomReference(psf, 1)

    try:
        caller_masses: list[float] = masses_with(0, 0.0)
        restraint.setMasses(caller_masses)

        caller_masses[0] = 1.0
        caller_masses[1] = 0.0

        apo_test.expect_invalid_argument(
            "mutating the caller list does not change copied atom 0",
            lambda: restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_SPHERE_CONTEXT,
        )
        restraint.addSphere(second, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        restraint.reset()

        caller_buffer: array.array[float] = array.array("d", masses_with(1, 0.0))
        restraint.setMasses(caller_buffer)

        caller_buffer[1] = 1.0
        caller_buffer[0] = 0.0
        del caller_buffer
        gc.collect()

        apo_test.expect_invalid_argument(
            "mutating and releasing the caller buffer does not change atom 1",
            lambda: restraint.addSphere(second, ORIGIN, 2.0, 1.0, INSIDE, RCM),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_SPHERE_CONTEXT,
        )
        restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM)
    finally:
        second.close()
        first.close()
        restraint.close()

    return


def check_mass_length_and_finiteness(psf: apo.CharmmPsf) -> None:
    print("Checking setMasses() length and finite-value validation...")

    restraint = apo.GeometricRestraintForce(psf)

    length_failures: list[tuple[str, list[float]]] = [
        ("empty", []),
        ("short", STANDARD_MASSES[:-1]),
        ("long", STANDARD_MASSES + [7.0]),
    ]
    nonfinite_failures: list[tuple[str, int, float]] = [
        ("NaN", 4, math.nan),
        ("positive infinity", 0, math.inf),
        ("negative infinity", NUM_ATOMS - 1, -math.inf),
    ]

    try:
        for label, masses in length_failures:
            error = apo_test.expect_invalid_argument(
                f"setMasses rejects {label} masses",
                lambda masses=masses: restraint.setMasses(masses),
                f"Mass count mismatch; expected {NUM_ATOMS}, observed {len(masses)}",
                expected_context=SET_MASSES_CONTEXT,
            )
            apo_test.assert_equal(
                f"setMasses {label} diagnostic names the C entry point",
                error.message.count(SET_MASSES_FUNCTION_NAME),
                1,
            )

        for label, atom_index, mass in nonfinite_failures:
            apo_test.expect_invalid_argument(
                f"setMasses rejects {label}",
                lambda atom_index=atom_index, mass=mass: restraint.setMasses(
                    masses_with(atom_index, mass)
                ),
                f"Mass at atom index {atom_index} must be finite",
                expected_context=SET_MASSES_CONTEXT,
            )
    finally:
        restraint.close()

    return


def check_masses_charmm_accepts(psf: apo.CharmmPsf) -> None:
    print("Checking RCM masses that CHARMM accepts and rejects...")

    restraint = apo.GeometricRestraintForce(psf)
    pair = select(psf, "bynu 1:2")

    try:
        for label, pair_masses in CHARMM_ACCEPTED_RCM_PAIR_MASSES.items():
            restraint.reset()
            restraint.setMasses(STANDARD_MASSES)
            restraint.addPlane(pair, ORIGIN, X_DIRECTION, 0.0, 0.0, SYMMETRIC, RCM)

            restraint.setMasses(make_charmm_masses(pair_masses))

            restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 0.0, SYMMETRIC, RCM)
            restraint.addSphere(pair, ORIGIN, 2.0, 0.0, SYMMETRIC, RCM)

            print(f"Accepted RCM masses: {label}")

        zero_total_masses: list[float] = make_charmm_masses(
            CHARMM_ZERO_TOTAL_RCM_PAIR_MASSES
        )

        restraint.reset()
        restraint.setMasses(STANDARD_MASSES)
        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 0.0, SYMMETRIC, RCM)

        apo_test.expect_invalid_argument(
            "setMasses() rejects a zero RCM total for an installed RCM term",
            lambda: restraint.setMasses(zero_total_masses),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )

        restraint.reset()
        restraint.setMasses(zero_total_masses)

        apo_test.expect_invalid_argument(
            "addPlane() rejects a new RCM term with a zero selected total",
            lambda: restraint.addPlane(
                pair, ORIGIN, X_DIRECTION, 2.0, 0.0, SYMMETRIC, RCM
            ),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_PLANE_CONTEXT,
        )

        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 0.0, SYMMETRIC, ATOMWISE)
    finally:
        pair.close()
        restraint.close()

    return


def check_failed_mass_replacement(psf: apo.CharmmPsf) -> None:
    print("Checking failed setMasses() replacement behavior...")

    restraint = apo.GeometricRestraintForce(psf)
    first = apo.AtomReference(psf, 0)

    native_failures: list[tuple[str, list[float], str]] = [
        (
            "an installed RCM total would become zero",
            masses_with(0, 0.0),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
        ),
        (
            "the replacement length is wrong",
            [0.0] * (NUM_ATOMS - 1),
            f"Mass count mismatch; expected {NUM_ATOMS}, observed {NUM_ATOMS - 1}",
        ),
        (
            "a replacement mass is not finite",
            [0.0] + [1.0] * 4 + [math.nan] + [1.0] * (NUM_ATOMS - 6),
            "Mass at atom index 5 must be finite",
        ),
    ]

    try:
        restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        for label, masses, diagnostic in native_failures:
            apo_test.expect_invalid_argument(
                f"setMasses fails when {label}",
                lambda masses=masses: restraint.setMasses(masses),
                diagnostic,
                expected_context=SET_MASSES_CONTEXT,
            )

            restraint.addPlane(first, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, RCM)

            print(f"Previous masses remain active after failure: {label}")

        apo_test.expect_exception(
            "setMasses Python conversion failure precedes the C call",
            ValueError,
            lambda: restraint.setMasses(
                cast(Sequence[float], [0.0, "mass"] + [1.0] * (NUM_ATOMS - 2))
            ),
        )
        restraint.addCylinder(first, ORIGIN, Z_DIRECTION, 2.0, 1.0, OUTSIDE, RCM)

        apo_test.expect_invalid_argument(
            "installed RCM terms survive failed replacements",
            lambda: restraint.setMasses(masses_with(0, 0.0)),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )

        restraint.setMasses(masses_with(1, 0.0))
    finally:
        first.close()
        restraint.close()

    return


def check_reset(psf: apo.CharmmPsf) -> None:
    print("Checking reset(), which removes every term as CHARMM's GEO RESET does...")

    restraint = apo.GeometricRestraintForce(psf)
    pair = select(psf, "bynu 1:2")
    first = apo.AtomReference(psf, 0)

    try:
        restraint.reset()
        restraint.reset()

        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, RCM)
        restraint.addCylinder(pair, ORIGIN, Z_DIRECTION, 2.0, 1.0, OUTSIDE, ATOMWISE)
        restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        apo_test.expect_invalid_argument(
            "installed RCM terms constrain replacement masses before reset",
            lambda: restraint.setMasses([0.0] * NUM_ATOMS),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )

        restraint.reset()
        assert_no_rcm_term("reset removes every installed RCM term", restraint)

        restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        apo_test.expect_invalid_argument(
            "terms added after reset are installed",
            lambda: restraint.setMasses([0.0] * NUM_ATOMS),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=SET_MASSES_CONTEXT,
        )

        restraint.reset()
        restraint.reset()
        assert_no_rcm_term("repeated reset cycles remove every term", restraint)
    finally:
        first.close()
        pair.close()
        restraint.close()

    return


def check_reset_preserves_masses(psf: apo.CharmmPsf) -> None:
    print("Checking that reset() keeps the masses, as CHARMM's GEO RESET does...")

    restraint = apo.GeometricRestraintForce(psf)
    pair = select(psf, "bynu 1:2")
    first = apo.AtomReference(psf, 0)
    second = apo.AtomReference(psf, 1)

    try:
        restraint.setMasses([0.0, -1.5] + STANDARD_MASSES[2:])
        restraint.addSphere(second, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        restraint.reset()

        apo_test.expect_invalid_argument(
            "reset keeps the replacement zero mass instead of the PSF mass",
            lambda: restraint.addSphere(first, ORIGIN, 2.0, 1.0, INSIDE, RCM),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_SPHERE_CONTEXT,
        )
        restraint.addSphere(second, ORIGIN, 2.0, 1.0, INSIDE, RCM)

        restraint.reset()
        restraint.reset()

        apo_test.expect_invalid_argument(
            "repeated reset keeps the replacement masses",
            lambda: restraint.addPlane(
                first, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, RCM
            ),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_PLANE_CONTEXT,
        )
        restraint.addPlane(pair, ORIGIN, X_DIRECTION, 2.0, 1.0, SYMMETRIC, RCM)
    finally:
        second.close()
        first.close()
        pair.close()
        restraint.close()

    return


def check_geometry_arguments(psf: apo.CharmmPsf) -> None:
    print("Checking geometry-specific argument validation...")

    restraint = apo.GeometricRestraintForce(psf)
    pair = select(psf, "bynu 1:2")
    first = apo.AtomReference(psf, 0)

    invalid_cases: list[tuple[str, str, str, str, Callable[[AtomInput], None]]] = [
        (
            "addPlane rejects a two-element origin",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "origin must contain exactly 3 elements",
            lambda atoms: restraint.addPlane(
                atoms, [0.0, 0.0], X_DIRECTION, 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a four-element origin",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "origin must contain exactly 3 elements",
            lambda atoms: restraint.addPlane(
                atoms, [0.0, 0.0, 0.0, 0.0], X_DIRECTION, 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects an empty origin",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "origin must contain exactly 3 elements",
            lambda atoms: restraint.addPlane(
                atoms, [], X_DIRECTION, 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a two-element direction",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "direction must contain exactly 3 elements",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, [1.0, 0.0], 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a nonfinite origin",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "Origin component must be finite; index 1",
            lambda atoms: restraint.addPlane(
                atoms, [0.0, math.nan, 0.0], X_DIRECTION, 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a nonfinite direction",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "Direction component must be finite; index 2",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, [1.0, 0.0, math.inf], 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a zero direction, which cannot define a normal",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "Direction must have a positive finite norm",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, [0.0, 0.0, 0.0], 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a direction whose norm overflows",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "Direction must have a positive finite norm",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, [1.7e308, 1.7e308, 0.0], 2.0, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a nonfinite force constant",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "Force constant must be finite",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, X_DIRECTION, math.inf, 1.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a nonfinite offset",
            ADD_PLANE_CONTEXT,
            "apo_geometric_restraint_force_add_plane",
            "Offset must be finite",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, X_DIRECTION, 2.0, math.nan, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "addCylinder rejects a two-element origin",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder",
            "origin must contain exactly 3 elements",
            lambda atoms: restraint.addCylinder(
                atoms, [0.0, 0.0], Z_DIRECTION, 2.0, 1.0, OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addCylinder rejects a four-element direction",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder",
            "direction must contain exactly 3 elements",
            lambda atoms: restraint.addCylinder(
                atoms, ORIGIN, [0.0, 0.0, 1.0, 0.0], 2.0, 1.0, OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addCylinder rejects a nonfinite origin",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder",
            "Origin component must be finite; index 0",
            lambda atoms: restraint.addCylinder(
                atoms, [-math.inf, 0.0, 0.0], Z_DIRECTION, 2.0, 1.0, OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addCylinder rejects a nonfinite direction",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder",
            "Direction component must be finite; index 0",
            lambda atoms: restraint.addCylinder(
                atoms, ORIGIN, [math.nan, 0.0, 1.0], 2.0, 1.0, OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addCylinder rejects a zero direction, which cannot define an axis",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder",
            "Direction must have a positive finite norm",
            lambda atoms: restraint.addCylinder(
                atoms, ORIGIN, [0.0, 0.0, 0.0], 2.0, 1.0, OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addCylinder rejects a nonfinite force constant",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder",
            "Force constant must be finite",
            lambda atoms: restraint.addCylinder(
                atoms, ORIGIN, Z_DIRECTION, -math.inf, 1.0, OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addCylinder rejects a nonfinite offset",
            ADD_CYLINDER_CONTEXT,
            "apo_geometric_restraint_force_add_cylinder",
            "Offset must be finite",
            lambda atoms: restraint.addCylinder(
                atoms, ORIGIN, Z_DIRECTION, 2.0, math.inf, OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addSphere rejects a two-element center",
            ADD_SPHERE_CONTEXT,
            "apo_geometric_restraint_force_add_sphere",
            "center must contain exactly 3 elements",
            lambda atoms: restraint.addSphere(
                atoms, [0.0, 0.0], 2.0, 1.0, INSIDE, ATOMWISE
            ),
        ),
        (
            "addSphere rejects a four-element center",
            ADD_SPHERE_CONTEXT,
            "apo_geometric_restraint_force_add_sphere",
            "center must contain exactly 3 elements",
            lambda atoms: restraint.addSphere(
                atoms, [0.0, 0.0, 0.0, 0.0], 2.0, 1.0, INSIDE, ATOMWISE
            ),
        ),
        (
            "addSphere rejects a nonfinite center",
            ADD_SPHERE_CONTEXT,
            "apo_geometric_restraint_force_add_sphere",
            "Origin component must be finite; index 2",
            lambda atoms: restraint.addSphere(
                atoms, [0.0, 0.0, -math.inf], 2.0, 1.0, INSIDE, ATOMWISE
            ),
        ),
        (
            "addSphere rejects a nonfinite force constant",
            ADD_SPHERE_CONTEXT,
            "apo_geometric_restraint_force_add_sphere",
            "Force constant must be finite",
            lambda atoms: restraint.addSphere(
                atoms, ORIGIN, math.nan, 1.0, INSIDE, ATOMWISE
            ),
        ),
        (
            "addSphere rejects a nonfinite offset",
            ADD_SPHERE_CONTEXT,
            "apo_geometric_restraint_force_add_sphere",
            "Offset must be finite",
            lambda atoms: restraint.addSphere(
                atoms, ORIGIN, 2.0, -math.inf, INSIDE, ATOMWISE
            ),
        ),
    ]

    conversion_cases: list[tuple[str, type[Exception], Callable[[], None]]] = [
        (
            "addPlane rejects a non-iterable origin",
            TypeError,
            lambda: restraint.addPlane(
                pair,
                cast(Sequence[float], 1.0),
                X_DIRECTION,
                2.0,
                1.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "addCylinder rejects a None direction",
            TypeError,
            lambda: restraint.addCylinder(
                first,
                ORIGIN,
                cast(Sequence[float], None),
                2.0,
                1.0,
                OUTSIDE,
                ATOMWISE,
            ),
        ),
        (
            "addSphere rejects an unparsable center element",
            ValueError,
            lambda: restraint.addSphere(
                pair,
                cast(Sequence[float], [0.0, "center", 0.0]),
                2.0,
                1.0,
                INSIDE,
                ATOMWISE,
            ),
        ),
        (
            "addPlane rejects a non-numeric force constant",
            TypeError,
            lambda: restraint.addPlane(
                first,
                ORIGIN,
                X_DIRECTION,
                cast(float, object()),
                1.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "addCylinder rejects an unparsable offset",
            ValueError,
            lambda: restraint.addCylinder(
                pair, ORIGIN, Z_DIRECTION, 2.0, cast(float, "far"), OUTSIDE, ATOMWISE
            ),
        ),
        (
            "addSphere rejects an overflowing force constant",
            OverflowError,
            lambda: restraint.addSphere(
                first, ORIGIN, cast(float, 10**400), 1.0, INSIDE, ATOMWISE
            ),
        ),
        (
            "addPlane rejects a None offset",
            TypeError,
            lambda: restraint.addPlane(
                pair, ORIGIN, X_DIRECTION, 2.0, cast(float, None), SYMMETRIC, ATOMWISE
            ),
        ),
    ]

    atom_routes: list[tuple[AtomInput, str]] = [
        (pair, "_from_selection"),
        (first, "_from_atom_reference"),
    ]

    try:
        for atoms, suffix in atom_routes:
            for label, context, function_prefix, diagnostic, add in invalid_cases:
                error = apo_test.expect_invalid_argument(
                    f"{label} ({suffix[1:]})",
                    lambda add=add, atoms=atoms: add(atoms),
                    diagnostic,
                    expected_context=context,
                )
                apo_test.assert_equal(
                    f"{label} ({suffix[1:]}) names the C entry point",
                    error.message.count(f"{function_prefix}{suffix}:"),
                    1,
                )

        for label, exception_type, action in conversion_cases:
            apo_test.expect_exception(label, exception_type, action)

        assert_no_rcm_term("rejected geometry arguments install no RCM term", restraint)

        restraint.addSphere(first, (0, 0, 0), 2, 1, SYMMETRIC, ATOMWISE)
        restraint.addPlane(first, ORIGIN, (0, 0, 1), 2.0, 1.0, INSIDE, ATOMWISE)
        restraint.addCylinder(pair, ORIGIN, (0, 0, 1), 2.0, 1.0, OUTSIDE, RCM)
        restraint.addPlane(
            pair,
            cast(Sequence[float], (value for value in ORIGIN)),
            cast(Sequence[float], (value for value in X_DIRECTION)),
            cast(float, Fraction(1, 2)),
            cast(float, Fraction(-1, 4)),
            OUTSIDE,
            RCM,
        )

        restraint.reset()
        restraint.setMasses(masses_with(0, 0.0))

        apo_test.expect_invalid_argument(
            "RCM mass validation applies even with a zero force constant",
            lambda: restraint.addSphere(first, ORIGIN, 0.0, 0.0, SYMMETRIC, RCM),
            f"{RCM_TOTAL_DIAGNOSTIC}; observed 0.000000",
            expected_context=ADD_SPHERE_CONTEXT,
        )
    finally:
        first.close()
        pair.close()
        restraint.close()

    return


def check_terms_charmm_accepts(psf: apo.CharmmPsf) -> None:
    print("Checking terms that CHARMM accepts...")

    restraint = apo.GeometricRestraintForce(psf)
    first_selection = select(psf, "bynu 1")
    first = apo.AtomReference(psf, 0)

    charmm_accepted_terms: list[tuple[str, Callable[[AtomInput], None]]] = [
        (
            "Zero harmonic constant",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, X_DIRECTION, 0.0, 2.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "Negative harmonic constant below the plane",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, X_DIRECTION, -2.0, 2.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "Plane with a nonunit oblique normal",
            lambda atoms: restraint.addPlane(
                atoms, [0.5, -1.0, 1.5], [2.0, 4.0, 6.0], 2.0, 2.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "Plane with its normal reversed",
            lambda atoms: restraint.addPlane(
                atoms,
                [0.5, -1.0, 1.5],
                [-1.0, -2.0, -3.0],
                2.0,
                2.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "Cylinder with a nonunit oblique axis",
            lambda atoms: restraint.addCylinder(
                atoms, [1.0, -2.0, 3.0], [2.0, 4.0, 6.0], 2.0, 2.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "Cylinder with its axis reversed",
            lambda atoms: restraint.addCylinder(
                atoms,
                [1.0, -2.0, 3.0],
                [-1.0, -2.0, -3.0],
                2.0,
                2.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "Cylinder axis with a negative offset",
            lambda atoms: restraint.addCylinder(
                atoms, ORIGIN, Z_DIRECTION, 2.0, -2.0, INSIDE, ATOMWISE
            ),
        ),
        (
            "Sphere with a displaced center",
            lambda atoms: restraint.addSphere(
                atoms, [1.25, -2.5, 3.75], 2.0, 2.0, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "Sphere center with a negative offset",
            lambda atoms: restraint.addSphere(
                atoms, ORIGIN, 2.0, -2.0, INSIDE, ATOMWISE
            ),
        ),
        (
            "Tiny plane direction accepted by CHARMM",
            lambda atoms: restraint.addPlane(
                atoms,
                ORIGIN,
                [1.57129096519201e-162, 0.0, 0.0],
                2.0,
                2.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "Tiny cylinder direction accepted by CHARMM",
            lambda atoms: restraint.addCylinder(
                atoms,
                ORIGIN,
                [1.178759765625006e-162, 1.571679687500008e-162, 0.0],
                2.0,
                2.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "Energy underflow suppresses the gradient",
            lambda atoms: restraint.addPlane(
                atoms,
                [-1.0e-300, 0.0, 0.0],
                X_DIRECTION,
                1.0e300,
                0.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "Energy underflow with a negative constant",
            lambda atoms: restraint.addPlane(
                atoms,
                [-1.0e-300, 0.0, 0.0],
                X_DIRECTION,
                -1.0e300,
                0.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "Very large offset balanced by a small constant",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, X_DIRECTION, 1.0e-300, 1.0e150, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "Very large negative offset balanced by a small constant",
            lambda atoms: restraint.addPlane(
                atoms, ORIGIN, X_DIRECTION, 1.0e-300, -1.0e150, SYMMETRIC, ATOMWISE
            ),
        ),
        (
            "Very large reference balanced by a small constant",
            lambda atoms: restraint.addPlane(
                atoms,
                [1.0e150, 0.0, 0.0],
                X_DIRECTION,
                1.0e-300,
                0.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
        (
            "Very large negative reference balanced by a small constant",
            lambda atoms: restraint.addPlane(
                atoms,
                [-1.0e150, 0.0, 0.0],
                X_DIRECTION,
                1.0e-300,
                0.0,
                SYMMETRIC,
                ATOMWISE,
            ),
        ),
    ]

    atom_inputs: list[AtomInput] = [first_selection, first]

    try:
        for label, add in charmm_accepted_terms:
            for atoms in atom_inputs:
                add(atoms)

            print(f"Accepted term: {label}")
    finally:
        first.close()
        first_selection.close()
        restraint.close()

    return


def check_subscription_tags(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> None:
    print("Checking default, explicit, and invalid ForceManager tags...")

    force_manager = make_force_manager(psf, parameters)
    restraint = apo.GeometricRestraintForce(psf)
    second = apo.GeometricRestraintForce(psf)
    recording_library = RecordingLibrary(apo_lib.lib())

    native_tags: list[tuple[str, str | None, bytes]] = [
        ("the default tag", None, b"mmfp"),
        ("an explicit tag", "geometric-restraint", b"geometric-restraint"),
        ("a non-ASCII tag", "g\u00e9om\u00e9trie", "g\u00e9om\u00e9trie".encode()),
        ("a tag with an embedded NUL", "mm\x00fp", b"mm"),
    ]

    try:
        with patch(
            "apocharmm.geometric_restraint_force.lib", return_value=recording_library
        ):
            for label, force_tag, native_tag in native_tags:
                force_manager.subscribe(restraint, force_tag)
                force_manager.unsubscribe(restraint)

                apo_test.assert_equal(
                    f"native code receives {label}",
                    recording_library.force_tags[-1],
                    native_tag,
                )

            apo_test.expect_invalid_argument(
                "native code rejects a tag that starts with NUL as empty",
                lambda: force_manager.subscribe(restraint, "\x00mmfp"),
                "Force tag must not be empty",
                expected_context=SUBSCRIBE_CONTEXT,
            )
            apo_test.assert_equal(
                "native code receives an empty tag after a leading NUL",
                recording_library.force_tags[-1],
                b"",
            )

        apo_test.expect_invalid_argument(
            "ForceManager.subscribe rejects an empty tag",
            lambda: force_manager.subscribe(restraint, ""),
            "Force tag must not be empty",
            expected_context=SUBSCRIBE_CONTEXT,
        )

        rejected_tags: list[tuple[str, object]] = [("an int", 1), ("bytes", b"mmfp")]

        for label, force_tag in rejected_tags:
            error: TypeError = apo_test.expect_exception(
                f"ForceManager.subscribe rejects {label} tag",
                TypeError,
                lambda force_tag=force_tag: force_manager.subscribe(
                    restraint, cast(str, force_tag)
                ),
            )
            apo_test.assert_equal(
                f"ForceManager.subscribe rejects {label} tag message",
                str(error),
                "force_tag must be a str",
            )

            error = apo_test.expect_exception(
                f"the subscription callback rejects {label} tag",
                TypeError,
                lambda force_tag=force_tag: restraint._subscribe_to_force_manager(
                    force_manager, cast(str, force_tag)
                ),
            )
            apo_test.assert_equal(
                f"the subscription callback rejects {label} tag message",
                str(error),
                "force_tag must be a str",
            )

        error = apo_test.expect_exception(
            "the subscription callback rejects a target that is not a ForceManager",
            TypeError,
            lambda: restraint._subscribe_to_force_manager(
                cast(apo.ForceManager, object())
            ),
        )
        apo_test.assert_equal(
            "the subscription callback rejects a non-ForceManager message",
            str(error),
            "force_manager must be a ForceManager",
        )

        apo_test.expect_exception(
            "ForceManager.subscribe rejects a tag that UTF-8 cannot encode",
            UnicodeEncodeError,
            lambda: force_manager.subscribe(restraint, "\udcff"),
        )

        force_manager.subscribe(restraint, "shared-tag")
        force_manager.subscribe(second, "shared-tag")
        force_manager.unsubscribe(second)
        force_manager.unsubscribe(restraint)

        print("Accepted two restraints that share one explicit tag")
    finally:
        second.close()
        restraint.close()
        force_manager.close()

    return


def check_subscription_lifecycle(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters, larger_psf: apo.CharmmPsf
) -> None:
    print(
        "Checking subscription before and after initialization, duplicates, "
        "unsubscription, and resubscription..."
    )

    case: CharmmCase = CHARMM_CASES[0]
    phase: CharmmPhase = case.phases[0]
    coordinates: list[list[float]] = evaluated_coordinates(case)

    force_manager = make_force_manager(psf, parameters)
    restraint = apo.GeometricRestraintForce(psf)
    context: apo.CharmmContext | None = None

    try:
        force_manager.subscribe(restraint)

        apo_test.assert_equal(
            "subscribing before CharmmContext construction does not initialize",
            force_manager.isInitialized(),
            False,
        )

        context = make_context(force_manager)
        baseline: ManagerOutput = evaluate(force_manager, context, coordinates)

        add_charmm_terms(restraint, psf, phase.terms, False)
        check_manager_output(
            "Terms added after subscription before initialization",
            phase,
            baseline,
            evaluate(force_manager, context, coordinates),
        )

        apo_test.expect_invalid_argument(
            "ForceManager.subscribe rejects a duplicate subscription",
            lambda: force_manager.subscribe(restraint),
            "Force is already subscribed to this ForceManager",
            expected_context=SUBSCRIBE_CONTEXT,
        )
        apo_test.expect_invalid_argument(
            "ForceManager.subscribe rejects a duplicate under another tag",
            lambda: force_manager.subscribe(restraint, "second-tag"),
            "Force is already subscribed to this ForceManager",
            expected_context=SUBSCRIBE_CONTEXT,
        )
        check_manager_output(
            "Rejected duplicates add no second contribution",
            phase,
            baseline,
            evaluate(force_manager, context, coordinates),
        )

        force_manager.unsubscribe(restraint)

        check_no_contribution(
            "Unsubscribed restraint",
            baseline,
            evaluate(force_manager, context, coordinates),
        )
        apo_test.expect_invalid_argument(
            "ForceManager.unsubscribe rejects a restraint that is not subscribed",
            lambda: force_manager.unsubscribe(restraint),
            "Force is not subscribed to this ForceManager",
            expected_context=UNSUBSCRIBE_CONTEXT,
        )

        force_manager.subscribe(restraint, "resubscribed")

        check_manager_output(
            "Resubscription keeps the installed terms",
            phase,
            baseline,
            evaluate(force_manager, context, coordinates),
        )

        force_manager.unsubscribe(restraint)

        late = apo.GeometricRestraintForce(psf)

        try:
            add_charmm_terms(late, psf, phase.terms, True)
            force_manager.subscribe(late)

            check_manager_output(
                "Restraint subscribed after initialization",
                phase,
                baseline,
                evaluate(force_manager, context, coordinates),
            )

            force_manager.unsubscribe(late)
        finally:
            late.close()

        mismatched = apo.GeometricRestraintForce(larger_psf)

        try:
            apo_test.expect_invalid_argument(
                "an initialized manager rejects a restraint for another atom count",
                lambda: force_manager.subscribe(mismatched),
                f"Initialization atom count mismatch; expected {LARGER_NUM_ATOMS}, "
                f"observed {NUM_ATOMS}",
                expected_context=SUBSCRIBE_CONTEXT,
            )
            apo_test.expect_invalid_argument(
                "the rejected restraint is not subscribed",
                lambda: force_manager.unsubscribe(mismatched),
                "Force is not subscribed to this ForceManager",
                expected_context=UNSUBSCRIBE_CONTEXT,
            )
        finally:
            mismatched.close()

        closed = apo.GeometricRestraintForce(psf)
        closed.close()

        expect_closed_handle(
            "ForceManager.subscribe rejects a restraint closed before subscription",
            lambda: force_manager.subscribe(closed),
        )
        check_no_contribution(
            "Rejected subscriptions",
            baseline,
            evaluate(force_manager, context, coordinates),
        )
    finally:
        if context is not None:
            context.close()

        restraint.close()
        force_manager.close()

    return


def check_manager_retention(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> None:
    print("Checking wrapper retention and closure while subscribed...")

    case: CharmmCase = CHARMM_CASES[0]
    phase: CharmmPhase = case.phases[0]
    coordinates: list[list[float]] = evaluated_coordinates(case)

    force_manager = make_force_manager(psf, parameters)

    try:
        context = make_context(force_manager)

        try:
            baseline: ManagerOutput = evaluate(force_manager, context, coordinates)

            retained = apo.GeometricRestraintForce(psf)
            retained_weak: weakref.ref[apo.GeometricRestraintForce] = weakref.ref(
                retained
            )

            add_charmm_terms(retained, psf, phase.terms, False)
            force_manager.subscribe(retained)
            del retained
            gc.collect()

            apo_test.assert_equal(
                "ForceManager retains a subscribed wrapper",
                retained_weak() is not None,
                True,
            )
            check_manager_output(
                "Retained wrapper",
                phase,
                baseline,
                evaluate(force_manager, context, coordinates),
            )

            force_manager.unsubscribe(
                cast(apo.GeometricRestraintForce, retained_weak())
            )
            gc.collect()

            apo_test.assert_equal(
                "unsubscription releases the retained wrapper",
                retained_weak() is None,
                True,
            )

            rejected = apo.GeometricRestraintForce(psf)
            rejected_weak: weakref.ref[apo.GeometricRestraintForce] = weakref.ref(
                rejected
            )

            force_manager.subscribe(rejected)
            apo_test.expect_invalid_argument(
                "a duplicate subscription is rejected",
                lambda: force_manager.subscribe(rejected),
                "Force is already subscribed to this ForceManager",
                expected_context=SUBSCRIBE_CONTEXT,
            )
            force_manager.unsubscribe(rejected)
            apo_test.expect_invalid_argument(
                "an empty tag is rejected",
                lambda: force_manager.subscribe(rejected, ""),
                "Force tag must not be empty",
                expected_context=SUBSCRIBE_CONTEXT,
            )
            rejected.close()
            expect_closed_handle(
                "a closed restraint is rejected",
                lambda: force_manager.subscribe(rejected),
            )
            del rejected
            gc.collect()

            apo_test.assert_equal(
                "failed subscriptions retain no wrapper",
                rejected_weak() is None,
                True,
            )

            closed = apo.GeometricRestraintForce(psf)
            closed_weak: weakref.ref[apo.GeometricRestraintForce] = weakref.ref(closed)

            force_manager.subscribe(closed)
            add_charmm_terms(closed, psf, phase.terms, True)
            closed.close()

            check_manager_output(
                "Native subscription after closing its wrapper",
                phase,
                baseline,
                evaluate(force_manager, context, coordinates),
            )
            expect_closed_handle(
                "a closed wrapper can no longer identify its subscription",
                lambda: force_manager.unsubscribe(closed),
            )

            del closed
            gc.collect()

            apo_test.assert_equal(
                "ForceManager keeps the closed wrapper it could not unsubscribe",
                closed_weak() is not None,
                True,
            )
        finally:
            context.close()
    finally:
        force_manager.close()

    gc.collect()

    apo_test.assert_equal(
        "ForceManager.close() releases the retained wrapper",
        closed_weak() is None,
        True,
    )

    return


def check_force_manager_lifetime(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> None:
    print("Checking a restraint that outlives its ForceManager wrapper...")

    case: CharmmCase = CHARMM_CASES[1]
    phase: CharmmPhase = case.phases[0]
    coordinates: list[list[float]] = evaluated_coordinates(case)

    restraint = apo.GeometricRestraintForce(psf)

    try:
        first = make_force_manager(psf, parameters)
        first.subscribe(restraint)
        first.close()

        expect_closed_handle(
            "a closed ForceManager cannot unsubscribe the restraint",
            lambda: first.unsubscribe(restraint),
        )

        restraint.setMasses(case.masses)
        add_charmm_terms(restraint, psf, phase.terms, False)

        second = make_force_manager(psf, parameters)

        try:
            context = make_context(second)

            try:
                baseline: ManagerOutput = evaluate(second, context, coordinates)
                second.subscribe(restraint)

                check_manager_output(
                    "Restraint subscribed to a second ForceManager",
                    phase,
                    baseline,
                    evaluate(second, context, coordinates),
                )

                second.unsubscribe(restraint)
            finally:
                context.close()
        finally:
            second.close()
    finally:
        restraint.close()

    return


def check_context_manager_closure(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> None:
    print("Checking context-manager closure of subscribed restraints...")

    case: CharmmCase = CHARMM_CASES[2]
    phase: CharmmPhase = case.phases[0]
    coordinates: list[list[float]] = evaluated_coordinates(case)

    force_manager = make_force_manager(psf, parameters)

    try:
        context = make_context(force_manager)

        try:
            baseline: ManagerOutput = evaluate(force_manager, context, coordinates)

            with apo.GeometricRestraintForce(psf) as managed:
                force_manager.subscribe(managed)
                add_charmm_terms(managed, psf, phase.terms, False)

                check_manager_output(
                    "Restraint subscribed inside a with block",
                    phase,
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )

                force_manager.unsubscribe(managed)

            expect_closed_handle(
                "with-block exit closes the unsubscribed restraint", managed.reset
            )
            check_no_contribution(
                "Restraint unsubscribed inside a with block",
                baseline,
                evaluate(force_manager, context, coordinates),
            )

            with apo.GeometricRestraintForce(psf) as left_subscribed:
                force_manager.subscribe(left_subscribed)
                add_charmm_terms(left_subscribed, psf, phase.terms, True)

            expect_closed_handle(
                "with-block exit closes the subscribed restraint",
                left_subscribed.reset,
            )
            check_manager_output(
                "Native subscription after with-block exit",
                phase,
                baseline,
                evaluate(force_manager, context, coordinates),
            )
            expect_closed_handle(
                "a restraint closed by with-block exit cannot be unsubscribed",
                lambda: force_manager.unsubscribe(left_subscribed),
            )
        finally:
            context.close()
    finally:
        force_manager.close()

    return


def check_periodic_boundary_conditions(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> None:
    print("Checking P1 enforcement while subscribed and after unsubscription...")

    case: CharmmCase = CHARMM_CASES[3]
    phase: CharmmPhase = case.phases[0]
    coordinates: list[list[float]] = evaluated_coordinates(case)
    p21_diagnostic: str = (
        "Subscribed force does not support periodic boundary condition value 2"
    )

    restraint = apo.GeometricRestraintForce(psf)

    try:
        force_manager = make_force_manager(psf, parameters)

        try:
            force_manager.subscribe(restraint)
            context = make_context(force_manager)

            try:
                baseline: ManagerOutput = evaluate(force_manager, context, coordinates)
                add_charmm_terms(restraint, psf, phase.terms, False)

                rejected_conditions: list[tuple[str, int, str]] = [
                    ("P21", apo.PeriodicBoundaryCondition.P21, p21_diagnostic),
                    ("the integer value of P21", 2, p21_diagnostic),
                    (
                        "the UNSET sentinel",
                        apo.PeriodicBoundaryCondition.UNSET,
                        "Periodic boundary condition must be PBC::P1 or PBC::P21; "
                        "observed 0",
                    ),
                ]

                for label, pbc, diagnostic in rejected_conditions:
                    apo_test.expect_invalid_argument(
                        f"a subscribed restraint rejects {label}",
                        lambda pbc=pbc: force_manager.setPeriodicBoundaryCondition(pbc),
                        diagnostic,
                        expected_context=SET_PBC_CONTEXT,
                    )

                invalid_values: list[object] = [99, -1, "P21"]

                for value in invalid_values:
                    error: ValueError = apo_test.expect_exception(
                        f"the Python enum layer rejects {value!r}",
                        ValueError,
                        lambda value=value: force_manager.setPeriodicBoundaryCondition(
                            cast(int, value)
                        ),
                    )
                    apo_test.assert_equal(
                        f"the Python enum layer rejects {value!r} message",
                        str(error),
                        f"invalid pbc: {value!r}",
                    )

                apo_test.expect_invalid_argument(
                    "a subscribed restraint rejects P21 through CharmmContext",
                    lambda: context.setPeriodicBoundaryCondition(
                        apo.PeriodicBoundaryCondition.P21
                    ),
                    p21_diagnostic,
                    expected_context="CharmmContext.setPeriodicBoundaryCondition(pbc)",
                )

                apo_test.assert_equal(
                    "rejected changes leave the manager in P1",
                    force_manager.getPeriodicBoundaryCondition(),
                    apo.PeriodicBoundaryCondition.P1,
                )
                apo_test.assert_equal(
                    "rejected changes leave the context in P1",
                    context.getPeriodicBoundaryCondition(),
                    apo.PeriodicBoundaryCondition.P1,
                )
                check_manager_output(
                    "Subscription after rejected PBC changes",
                    phase,
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )

                force_manager.unsubscribe(restraint)
            finally:
                context.close()
        finally:
            force_manager.close()

        unsubscribed_manager = make_force_manager(psf, parameters)

        try:
            unsubscribed_manager.subscribe(restraint)
            unsubscribed_manager.unsubscribe(restraint)
            unsubscribed_manager.setPeriodicBoundaryCondition(
                apo.PeriodicBoundaryCondition.P21
            )

            apo_test.assert_equal(
                "an unsubscribed restraint no longer constrains the manager",
                unsubscribed_manager.getPeriodicBoundaryCondition(),
                apo.PeriodicBoundaryCondition.P21,
            )
            apo_test.expect_invalid_argument(
                "ForceManager.subscribe requires P1",
                lambda: unsubscribed_manager.subscribe(restraint),
                p21_diagnostic,
                expected_context=SUBSCRIBE_CONTEXT,
            )

            unsubscribed_manager.setPeriodicBoundaryCondition(
                apo.PeriodicBoundaryCondition.P1
            )
            context = make_context(unsubscribed_manager)

            try:
                baseline = evaluate(unsubscribed_manager, context, coordinates)
                unsubscribed_manager.subscribe(restraint)

                check_manager_output(
                    "Subscription after restoring P1",
                    phase,
                    baseline,
                    evaluate(unsubscribed_manager, context, coordinates),
                )

                unsubscribed_manager.unsubscribe(restraint)
            finally:
                context.close()
        finally:
            unsubscribed_manager.close()
    finally:
        restraint.close()

    return


def check_input_lifetimes_through_manager(
    psf_path: Path, psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> None:
    print("Checking PSF, selection, and reference lifetimes through ForceManager...")

    force_manager = make_force_manager(psf, parameters)

    try:
        context = make_context(force_manager)

        try:
            for case, use_atom_references in (
                (CHARMM_CASES[7], False),
                (CHARMM_CASES[2], True),
            ):
                phase: CharmmPhase = case.phases[0]
                coordinates: list[list[float]] = evaluated_coordinates(case)

                source_psf = apo.CharmmPsf(str(psf_path))
                source_psf_weak: weakref.ref[apo.CharmmPsf] = weakref.ref(source_psf)
                restraint = apo.GeometricRestraintForce(source_psf)

                try:
                    source_psf.close()
                    del source_psf
                    gc.collect()

                    apo_test.assert_equal(
                        f"{case.name}: the restraint retains no PSF wrapper",
                        source_psf_weak() is None,
                        True,
                    )

                    atom_psf = apo.CharmmPsf(str(psf_path))
                    add_charmm_terms(
                        restraint, atom_psf, phase.terms, use_atom_references
                    )
                    atom_psf.close()
                    restraint.setMasses(case.masses)

                    baseline: ManagerOutput = evaluate(
                        force_manager, context, coordinates
                    )
                    force_manager.subscribe(restraint)

                    try:
                        check_manager_output(
                            f"{case.name} after closing every input wrapper",
                            phase,
                            baseline,
                            evaluate(force_manager, context, coordinates),
                        )
                    finally:
                        force_manager.unsubscribe(restraint)
                finally:
                    restraint.close()
        finally:
            context.close()
    finally:
        force_manager.close()

    return


def check_mutation_while_subscribed(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters
) -> None:
    print("Checking mass replacement, reset, and reuse while subscribed...")

    force_manager = make_force_manager(psf, parameters)

    try:
        context = make_context(force_manager)

        try:
            restraint = apo.GeometricRestraintForce(psf)

            try:
                force_manager.subscribe(restraint)

                first: CharmmCase = RCM_MASS_CASES[0]
                coordinates: list[list[float]] = evaluated_coordinates(first)
                baseline: ManagerOutput = evaluate(force_manager, context, coordinates)

                restraint.setMasses(first.masses)
                add_charmm_terms(restraint, psf, first.phases[0].terms, False)

                for case in RCM_MASS_CASES:
                    restraint.setMasses(case.masses)

                    check_manager_output(
                        f"{case.name} after replacing masses while subscribed",
                        case.phases[0],
                        baseline,
                        evaluate(force_manager, context, coordinates),
                    )

                force_manager.unsubscribe(restraint)
            finally:
                restraint.close()

            restraint = apo.GeometricRestraintForce(psf)

            try:
                force_manager.subscribe(restraint)

                case = RESET_REUSE_CASE
                before_reset, after_reset = case.phases
                coordinates = evaluated_coordinates(case)
                baseline = evaluate(force_manager, context, coordinates)

                restraint.setMasses(case.masses)
                add_charmm_terms(restraint, psf, before_reset.terms, True)

                check_manager_output(
                    f"{case.name}, {before_reset.name}",
                    before_reset,
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )

                restraint.reset()

                check_no_contribution(
                    f"{case.name}, after reset",
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )

                restraint.reset()

                check_no_contribution(
                    f"{case.name}, after a repeated reset",
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )

                add_charmm_terms(restraint, psf, after_reset.terms, False)

                check_manager_output(
                    f"{case.name}, {after_reset.name}",
                    after_reset,
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )

                force_manager.unsubscribe(restraint)
            finally:
                restraint.close()
        finally:
            context.close()
    finally:
        force_manager.close()

    return


def check_charmm_numerics(psf: apo.CharmmPsf, parameters: apo.CharmmParameters) -> None:
    print("Checking CHARMM energies and forces through ForceManager...")

    force_manager = make_force_manager(psf, parameters)

    try:
        context = make_context(force_manager)

        try:
            for case in CHARMM_CASES:
                run_charmm_case(psf, force_manager, context, case, False)

                selections: list[apo.AtomSelection] = [
                    select(psf, term.atoms)
                    for phase in case.phases
                    for term in phase.terms
                ]

                try:
                    single_atom_terms: bool = all(
                        selection.getNumSelected() == 1 for selection in selections
                    )
                finally:
                    for selection in selections:
                        selection.close()

                if single_atom_terms:
                    run_charmm_case(psf, force_manager, context, case, True)
        finally:
            context.close()
    finally:
        force_manager.close()

    return


def check_many_terms(
    psf: apo.CharmmPsf, parameters: apo.CharmmParameters, larger_psf: apo.CharmmPsf
) -> None:
    print("Checking more than 100 terms and more than 100 selected atoms...")

    case: CharmmCase = CHARMM_CASES[0]
    phase: CharmmPhase = case.phases[0]
    coordinates: list[list[float]] = evaluated_coordinates(case)
    count: int = DUPLICATE_TERM_COUNT
    duplicates = CharmmPhase(
        f"{count} copies of one term",
        phase.terms,
        count * phase.energy,
        {
            atom: (count * force[0], count * force[1], count * force[2])
            for atom, force in phase.forces.items()
        },
        count * phase.energy_tolerance,
        count * phase.force_tolerance,
    )

    force_manager = make_force_manager(psf, parameters)

    try:
        context = make_context(force_manager)

        try:
            restraint = apo.GeometricRestraintForce(psf)

            try:
                force_manager.subscribe(restraint)
                baseline: ManagerOutput = evaluate(force_manager, context, coordinates)

                for copy in range(count):
                    add_charmm_terms(restraint, psf, phase.terms, copy % 2 == 1)

                check_manager_output(
                    f"{case.name}, {duplicates.name}",
                    duplicates,
                    baseline,
                    evaluate(force_manager, context, coordinates),
                )

                force_manager.unsubscribe(restraint)
            finally:
                restraint.close()
        finally:
            context.close()
    finally:
        force_manager.close()

    larger_restraint = apo.GeometricRestraintForce(larger_psf)
    larger_all = select(larger_psf, "all")

    try:
        apo_test.assert_equal(
            "larger selection size", larger_all.getNumSelected(), LARGER_NUM_ATOMS
        )

        for selection_mode in apo.GeometricRestraintSelectionMode:
            for addition in make_term_additions(
                larger_restraint, SYMMETRIC, selection_mode
            ):
                addition.add(larger_all)

        larger_restraint.setMasses([2.0 * mass for mass in larger_psf.getMasses()])
        larger_restraint.reset()
    finally:
        larger_all.close()
        larger_restraint.close()

    return


def main(argc: int, argv: list[str]) -> int:
    repo_root: Path = Path(argv[1]) if argc > 1 else Path(".")
    output_dir: Path = repo_root / "test/pytest"
    output_dir.mkdir(parents=True, exist_ok=True)

    psf_path: Path = output_dir / "tmp_python_api_geometric_restraint_force.psf"
    parameter_path: Path = output_dir / "tmp_python_api_geometric_restraint_force.prm"
    apo_test.remove_if_exists(psf_path)
    apo_test.remove_if_exists(parameter_path)

    larger_psf_path: str = apo_test.require_file(apo_test.get_data_dir() / "1lvz.psf")

    try:
        apo_test.write_text_file(psf_path, TEST_PSF_TEXT)
        apo_test.write_text_file(parameter_path, TEST_PARAMETER_TEXT)

        check_exports_and_enums()

        psf = apo.CharmmPsf(str(psf_path))
        second_psf = apo.CharmmPsf(str(psf_path))
        larger_psf = apo.CharmmPsf(larger_psf_path)
        parameters = apo.CharmmParameters(str(parameter_path))

        try:
            apo_test.assert_equal("test PSF atom count", psf.getNumAtoms(), NUM_ATOMS)
            apo_test.assert_equal("test PSF masses", psf.getMasses(), STANDARD_MASSES)
            apo_test.assert_equal(
                "larger PSF atom count", larger_psf.getNumAtoms(), LARGER_NUM_ATOMS
            )

            check_construction(psf_path, psf)
            check_psf_lifetime(psf_path, larger_psf)
            check_closure(psf)
            check_finalizer_ownership(psf)
            check_enum_validation(psf)
            check_atom_inputs(psf, second_psf)
            check_rejected_atom_forms(psf)
            check_closed_atom_inputs(psf)
            check_empty_selection(psf, larger_psf)
            check_force_local_validation(psf, larger_psf)
            check_input_lifetime(psf_path, psf)
            check_mass_conversion(psf)
            check_mass_buffer_copying(psf)
            check_mass_length_and_finiteness(psf)
            check_masses_charmm_accepts(psf)
            check_failed_mass_replacement(psf)
            check_reset(psf)
            check_reset_preserves_masses(psf)
            check_geometry_arguments(psf)
            check_terms_charmm_accepts(psf)
            check_subscription_tags(psf, parameters)
            check_subscription_lifecycle(psf, parameters, larger_psf)
            check_manager_retention(psf, parameters)
            check_force_manager_lifetime(psf, parameters)
            check_context_manager_closure(psf, parameters)
            check_periodic_boundary_conditions(psf, parameters)
            check_input_lifetimes_through_manager(psf_path, psf, parameters)
            check_mutation_while_subscribed(psf, parameters)
            check_charmm_numerics(psf, parameters)
            check_many_terms(psf, parameters, larger_psf)
        finally:
            parameters.close()
            larger_psf.close()
            second_psf.close()
            psf.close()
    finally:
        print("Cleaning up GeometricRestraintForce Python API test files...")
        apo_test.remove_if_exists(parameter_path)
        apo_test.remove_if_exists(psf_path)

    print(
        "\033[32m"
        + "PASS: GeometricRestraintForce Python API tests completed."
        + "\033[0m"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(len(sys.argv), sys.argv))
