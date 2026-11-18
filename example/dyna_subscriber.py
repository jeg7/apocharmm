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


def main(argc: int, argv: list[str]) -> int:
    # Input variables
    box_dims: list[float] = [50.0, 50.0, 50.0]
    random_seed: int = 314159
    temperature: float = 300.0
    time_step: float = 0.002
    npt_steps: int = 200
    npt_report_frequency: int = 50
    nvt_steps: int = 50
    restraint_force_constant: float = 10.0
    npt_path: str = "tmp_dyna_subscriber_npt.out"
    nvt_path: str = "tmp_dyna_subscriber_nvt.out"

    # Load CHARMM force field
    prm = apo.CharmmParameters("toppar/toppar_water_ions.str")

    # Load PSF and coordinates
    psf = apo.CharmmPsf("test/data/waterbox.psf")
    crd = apo.CharmmCrd("test/data/waterbox.crd")

    # Setup the CharmmContext. The Langevin piston needs the crystal type, which
    # must be set before the integrator is attached.
    ctx = apo.CharmmContext(psf, prm)
    ctx.setCrystalType(apo.CrystalType.CUBIC)
    ctx.setBoxDimensions(box_dims)
    ctx.setCoordinates(crd)
    ctx.setRandomSeed(random_seed)
    ctx.useHolonomicConstraints(True)
    ctx.assignVelocitiesAtTemperature(temperature)

    # NPT dynamics with the Langevin piston and Nose-Hoover temperature control.
    # The reference pressure keeps its default of 1 atm. The Nose-Hoover and
    # piston masses are the common estimates, 0.2 and 0.02 times the total mass.
    # The piston mass and friction follow the attachment, because the number of
    # piston degrees of freedom comes from the context's crystal type.
    total_mass: float = psf.getTotalMass()
    npt_integrator = apo.CudaLangevinPistonIntegrator(time_step)
    npt_integrator.setReferenceTemperature(temperature)
    npt_integrator.useNoseHooverThermostat(True)
    npt_integrator.setNoseHooverPistonMass(0.2 * total_mass)
    npt_integrator.setLangevinPistonFrictionSeed(random_seed)
    npt_integrator.setCharmmContext(ctx)
    npt_integrator.setLangevinPistonMass([0.02 * total_mass])
    npt_integrator.setLangevinPistonFriction(20.0)

    # The DynaSubscriber writes CHARMM's DYNA energy blocks: the header and a
    # step-0 block when the run starts, then a block every npt_report_frequency
    # steps. With the piston, each block also has the pressure and the crystal
    # parameters. The piston computes its first pressure in its first step, so
    # the step-0 block shows zero pressure, and the header, which labels the
    # lines of the step-0 block, has no label for the XTLE> lines that follow.
    with apo.DynaSubscriber(npt_path, npt_report_frequency) as dyna:
        npt_integrator.subscribe(dyna)
        npt_integrator.propagate(npt_steps)
        npt_integrator.unsubscribe(dyna)

    print(f"==== NPT: {npt_path} ====")
    with open(npt_path) as dyna_file:
        print(dyna_file.read(), end="")

    # Restrain the oxygen of the water nearest the box center to its position
    # in the input coordinates. The NPT run moved it, so the restraint energy is
    # nonzero from the step-0 block on, and every block has a CONSTR> line.
    selector = apo.AtomSelector(psf)
    oxygen = selector.select("atom WZ1 2338 OH2")

    harm = apo.HarmonicRestraintForce(psf)
    harm.setSelection(oxygen)
    harm.setReferenceCoordinates(crd.getCoordinates())
    harm.setForceConstant(restraint_force_constant)

    fm = ctx.getForceManager()
    fm.subscribe(harm)

    # NVT dynamics with the Langevin thermostat, continuing from the NPT state
    nvt_integrator = apo.CudaLangevinThermostatIntegrator(time_step)
    nvt_integrator.setReferenceTemperature(temperature)
    nvt_integrator.setThermostatFriction(1.0)
    nvt_integrator.setThermostatRngSeed(random_seed)
    nvt_integrator.setCharmmContext(ctx)

    # Without a report frequency, the subscriber writes a block every 10 steps,
    # CHARMM's default NPRINT.
    with apo.DynaSubscriber(nvt_path) as dyna:
        nvt_integrator.subscribe(dyna)
        nvt_integrator.propagate(nvt_steps)
        nvt_integrator.unsubscribe(dyna)

    print(f"==== NVT: {nvt_path} ====")
    with open(nvt_path) as dyna_file:
        print(dyna_file.read(), end="")

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
