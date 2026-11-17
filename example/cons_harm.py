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
    num_steps = 1000
    time_step = 0.001

    # Load CHARMM force field
    prm = apo.CharmmParameters("toppar/toppar_water_ions.str")

    # Load PSF and coordinates
    psf = apo.CharmmPsf("test/data/nacl_pair.psf")
    crd = apo.CharmmCrd("test/data/nacl_pair.cor")

    # Setup CharmmContext
    ctx = apo.CharmmContext(psf, prm)
    ctx.setBoxDimensions(box_dims)
    ctx.setCoordinates(crd)
    ctx.setRandomSeed(random_seed)
    ctx.assignVelocitiesAtTemperature(temperature)

    # Setup integrator
    integrator = apo.CudaLangevinThermostatIntegrator(time_step)
    integrator.setThermostatFriction(1.0)
    integrator.setThermostatRngSeed(random_seed)
    integrator.setCharmmContext(ctx)

    # # DCD subscriber to visualize the effects of restraints
    # dcd = apo.DcdSubscriber("tmpHarmRestraint.dcd", 1)
    # integrator.subscribe(dcd)

    # Generate selection for restraints
    selector = apo.AtomSelector(psf)
    selection = selector.select("all")

    # Setup harmonic restraint
    harm = apo.HarmonicRestraintForce(psf)
    harm.setSelection(selection)
    harm.setReferenceCoordinates(crd.getCoordinates())
    # harm.setMasses(psf.getMasses())

    # Subscribe harmonic restraint
    fm = ctx.getForceManager()
    fm.subscribe(harm)

    # Compute initial distance between Na-Cl before any dynamics
    xyzq = ctx.getCoordinatesCharges()
    dx0 = xyzq[1][0] - xyzq[0][0]
    dy0 = xyzq[1][1] - xyzq[0][1]
    dz0 = xyzq[1][2] - xyzq[0][2]
    r0 = math.sqrt(dx0**2 + dy0**2 + dz0**2)
    print(f"r0 = {r0}")

    # Run dynamics with progressively stronger harmonic restraints.
    # (NOT ACTUALLY GOOD PRACTICE, THIS IS ONLY FOR DEMO/TESTING)
    force_constant = 1e-8
    for i in range(14):
        harm.setForceConstant(force_constant)
        integrator.propagate(num_steps)
        force_constant *= 10.0

    xyzq = ctx.getCoordinatesCharges()
    dx = xyzq[1][0] - xyzq[0][0]
    dy = xyzq[1][1] - xyzq[0][1]
    dz = xyzq[1][2] - xyzq[0][2]
    r = math.sqrt(dx**2 + dy**2 + dz**2)
    print(f"r = {r}")

    assert abs(r0 - r) < 0.01

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
