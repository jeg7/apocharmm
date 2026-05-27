# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import apocharmm as apo
import sys


def run0(
    box_dims,
    random_seed,
    temperature,
    use_holonomic_constraints,
    time_step,
    num_steps,
):
    # Load CHARMM force field
    prm = apo.CharmmParameters(["test/data/toppar_water_ions.str"])

    # Load PSF and coordinates
    psf = apo.CharmmPsf("test/data/waterbox.psf")
    crd = apo.CharmmCrd("test/data/waterbox.crd")

    # Setup the ForceManager
    fm = apo.ForceManager(psf, prm)
    fm.setBoxDimensions(box_dims)

    # Setup the CharmmContext
    ctx = apo.CharmmContext(fm)
    ctx.setCoordinates(crd)
    ctx.setRandomSeedForVelocities(random_seed)
    ctx.assignVelocitiesAtTemperature(temperature)
    ctx.useHolonomicConstraints(use_holonomic_constraints)

    # Setup the integrator
    integrator = apo.CudaNoseHooverIntegrator(time_step)
    integrator.setCharmmContext(ctx)
    integrator.setReferenceTemperature(temperature)

    # Create a DCD and restart file subscriber
    dcd = apo.DcdSubscriber("tmp_nvt_nose_hoover_0.dcd")
    rst = apo.RestartSubscriber("tmp_nvt_nose_hoover_0.rst", num_steps)

    # Subscribe the DCD and restart subscribers to the integrator
    integrator.subscribe(dcd)
    integrator.subscribe(rst)

    # Run simulation
    integrator.propagate(num_steps)

    print("T_ref  = {}".format(integrator.getReferenceTemperature()))
    print("T_avg  = {}".format(integrator.getAverageTemperature()))
    print("T_inst = {}".format(integrator.getInstantaneousTemperature()))

    return


def run1(
    box_dims,
    random_seed,
    temperature,
    use_holonomic_constraints,
    time_step,
    num_steps,
):
    # Load CHARMM force field
    prm = apo.CharmmParameters(["test/data/toppar_water_ions.str"])

    # Load PSF and coordinates
    psf = apo.CharmmPsf("test/data/waterbox.psf")
    crd = apo.CharmmCrd("test/data/waterbox.crd")

    # Setup the ForceManager
    fm = apo.ForceManager(psf, prm)
    fm.setBoxDimensions(box_dims)

    # Setup the CharmmContext
    ctx = apo.CharmmContext(fm)
    ctx.setCoordinates(crd)
    ctx.useHolonomicConstraints(use_holonomic_constraints)

    # Setup the integrator
    integrator = apo.CudaNoseHooverIntegrator(time_step)
    integrator.setCharmmContext(ctx)
    integrator.initializeFromRestartFile("tmp_nvt_nose_hoover_0.rst")

    # Create a DCD and restart file subscriber
    dcd = apo.DcdSubscriber("tmp_nvt_nose_hoover_1.dcd")
    rst = apo.RestartSubscriber("tmp_nvt_nose_hoover_1.rst", num_steps)

    # Subscribe the DCD and restart subscribers to the integrator
    integrator.subscribe(dcd)
    integrator.subscribe(rst)

    # Run simulation
    integrator.propagate(num_steps)

    print("T_ref  = {}".format(integrator.getReferenceTemperature()))
    print("T_avg  = {}".format(integrator.getAverageTemperature()))
    print("T_inst = {}".format(integrator.getInstantaneousTemperature()))

    return


def main(argc, argv):
    # Input variables
    box_dims = [50.0, 50.0, 50.0]
    random_seed = 314159
    temperature = 300.0
    use_holonomic_constraints = True
    num_steps = 10000
    time_step = 0.002

    # Set up simulation, run, and generate a restart file
    run0(
        box_dims,
        random_seed,
        temperature,
        use_holonomic_constraints,
        time_step,
        num_steps,
    )

    # Set up simulation from restart file and continute running
    run1(
        box_dims,
        random_seed,
        temperature,
        use_holonomic_constraints,
        time_step,
        num_steps,
    )

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
