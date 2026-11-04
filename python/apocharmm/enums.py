# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

from enum import IntEnum


class CrystalType(IntEnum):
    """
    @brief Selects the crystal type owned by a CharmmContext.

    `CUBIC` uses one isotropic X/Y/Z degree of freedom. `TETRAGONAL`
    uses one coupled X/Y degree of freedom and one Z degree of freedom.
    `ORTHORHOMBIC` uses independent X, Y, and Z degrees of freedom.
    `UNSET` represents an unusable crystal configuration, not an aperiodic
    system, and is rejected by `CharmmContext.setCrystalType()`. The type also
    labels restart cells and is not inferred from instantaneous box
    dimensions. Compatible changes on the context are allowed between
    completed propagation calls; attached piston integrators must then be
    reinitialized before their state is used.

    @anchor python_crystal_type
    @see charmm_context
    @see cuda_integrators
    """

    ## @brief Represents an unconfigured, non-runnable crystal.
    UNSET = 0
    ## @brief Selects one isotropic X/Y/Z piston degree of freedom.
    CUBIC = 1
    ## @brief Selects coupled X/Y and independent Z piston degrees of freedom.
    TETRAGONAL = 2
    ## @brief Selects independent X, Y, and Z piston degrees of freedom.
    ORTHORHOMBIC = 3


class DistanceRestraintCondition(IntEnum):
    """
    @anchor python_distance_restraint_condition
    @brief Selects the one-sided activation mode for a distance-restraint term.

    `NONE` applies no one-sided activation gate. `POSITIVE` is active exactly
    when the term deviation satisfies `D >= 0`. `NEGATIVE` is active exactly
    when the term deviation satisfies `D <= 0`.
    """

    ## @brief Applies no one-sided activation gate.
    NONE = 0
    ## @brief Selects exact positive-side activation at `D >= 0`
    POSITIVE = 1
    ## @brief Selects exact negative-side activation at `D <= 0`
    NEGATIVE = -1


class PeriodicBoundaryCondition(IntEnum):
    """
    @anchor python_periodic_boundary_condition
    @brief Identifies a periodic-boundary configuration value.

    `UNSET` is an invalid, non-runnable sentinel. It does not select an
    aperiodic system and is rejected by the public periodic-boundary setters.
    `P1` selects conventional three-dimensional translational periodicity. `P21`
    selects the P2_1 screw-symmetry condition.

    Integer values `0`, `1`, and `2` can be converted to this enumeration.
    """

    ## @brief Selects an unconfigured, non-runnable state.
    UNSET = 0
    ## @brief Selects conventional translational periodicity.
    P1 = 1
    ## @brief Selects P2_1 screw-symmetry periodicity.
    P21 = 2


class VdwType(IntEnum):
    """
    @anchor python_vdw_type
    @brief Identifies a native van der Waals model code.

    `CharmmContext.setVdwType()` accepts native model values `VSH` through
    `DBEXP`, whose integer values are `1` through `6`. `NONE` is representable
    by this Python enumeration but is rejected by the current native
    CharmmContext setter.
    """

    ## @brief Represents no selected van der Waals model.
    NONE = 0
    ## @brief Selects native model code `VDW_VSH`.
    VSH = 1
    ## @brief Selects native model code `VDW_VSW`.
    VSW = 2
    ## @brief Selects native model code `VDW_VFSW`.
    VFSW = 3
    ## @brief Selects native model code `VDW_VGSH`.
    VGSH = 4
    ## @brief Selects native model code `VDW_CUT`.
    CUT = 5
    ## @brief Selects native model code `VDW_DBEXP`.
    DBEXP = 6
