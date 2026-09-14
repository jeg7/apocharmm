# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import apocharmm as apo
import math
import sys


def main(argc, argv):
    # Input variables
    box_dims = [50.0, 50.0, 50.0]
    random_seed = 314159
    temperature = 300.0
    num_steps = 10000
    time_step = 0.002

    # Load CHARMM force field
    prm = apo.CharmmParameters("toppar/par_all36m_prot.prm")

    # Load PSF and coordinates
    psf = apo.CharmmPsf("test/data/1lvz.psf")
    crd = apo.CharmmCrd("test/data/1lvz.cor")

    # Setup the CharmmContext
    ctx = apo.CharmmContext(psf, prm)
    ctx.setBoxDimensions(box_dims)
    ctx.setCoordinates(crd)
    ctx.setRandomSeed(random_seed)
    ctx.assignVelocitiesAtTemperature(temperature)
    ctx.useHolonomicConstraints(False)

    # Setup the integrator
    integrator = apo.CudaLangevinThermostatIntegrator(time_step)
    integrator.setReferenceTemperature(temperature)
    integrator.setThermostatFriction(1.0)
    integrator.setThermostatRngSeed(random_seed)
    integrator.setCharmmContext(ctx)

    # Select atoms
    selector = apo.AtomSelector(psf)
    n_selection = selector.select("atom A000 1 N")
    ht1_selection = selector.select("atom A000 1 HT1")
    ht2_selection = selector.select("atom A000 1 HT2")

    n_indices = n_selection.getAtomIndices()
    ht1_indices = ht1_selection.getAtomIndices()
    ht2_indices = ht2_selection.getAtomIndices()

    if len(n_indices) != 1 or len(ht1_indices) != 1 or len(ht2_indices) != 1:
        raise RuntimeError(
            "Expected selections 'atom A000 1 N', 'atom A000 1 HT1', and 'atom A000 1 HT2' to each match exactly one atom"
        )

    n_index = n_indices[0]
    ht1_index = ht1_indices[0]
    ht2_index = ht2_indices[0]

    # Calculate reference values from the starting coordinates
    coordinates = crd.getCoordinates()
    r_n_ht1 = math.dist(coordinates[n_index], coordinates[ht1_index])
    r_n_ht2 = math.dist(coordinates[n_index], coordinates[ht2_index])
    r_ht1_ht2 = math.dist(coordinates[ht1_index], coordinates[ht2_index])

    # Setup distance restraints
    resd = apo.DistanceRestraintForce(psf.getNumAtoms())

    # Conventional harmonic-distance term on r(N, HT1)
    resd.addRestraint(
        [[n_index, ht1_index]],
        [1.0],
        1.0,
        r_n_ht1 - 0.05,
        1,
        2,
        apo.DistanceRestraintCondition.NONE,
    )

    # Reaction-coordinate term on r(N, HT1) - r(HT1, HT2)
    resd.addRestraint(
        [[n_index, ht1_index], [ht1_index, ht2_index]],
        [1.0, -1.0],
        0.25,
        r_n_ht1 - r_ht1_ht2 - 0.05,
        1,
        2,
        apo.DistanceRestraintCondition.NONE,
    )

    # Positive-only high-power term on r(N, HT2)^6
    resd.addRestraint(
        [[n_index, ht2_index]],
        [1.0],
        0.001,
        r_n_ht2**6 - 0.5,
        6,
        4,
        apo.DistanceRestraintCondition.POSITIVE,
    )

    resd.setScale(0.5)

    # Subscribe distance restraints
    fm = ctx.getForceManager()
    fm.subscribe(resd)

    # Run simulation
    integrator.propagate(num_steps)

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
