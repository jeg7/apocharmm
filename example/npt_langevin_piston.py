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


def main(argc, argv):
    # Input variables
    box_dims = [50.0, 50.0, 50.0]
    random_seed = 314259
    temperature = 300.0
    use_holonomic_constraints = True
    use_nose_hoover_thermostat = True
    num_steps = 10000
    time_step = 0.002

    # Load CHARMM force field
    prm = apo.CharmmParameters("toppar/toppar_water_ions.str")

    # Load PSF and coordinates
    psf = apo.CharmmPsf("test/data/waterbox.psf")
    crd = apo.CharmmCrd("test/data/waterbox.crd")

    # Setup the ForceManager
    fm = apo.ForceManager(psf, prm)
    fm.setBoxDimensions(box_dims)

    # Setup the CharmmContext
    ctx = apo.CharmmContext(fm)
    ctx.setCrystalType(apo.CrystalType.CUBIC)
    ctx.setCoordinates(crd)
    ctx.setRandomSeed(random_seed)
    ctx.useHolonomicConstraints(use_holonomic_constraints)

    # NPT Heating & Equilibration
    integrator = apo.CudaLangevinPistonIntegrator(time_step)
    integrator.useNoseHooverThermostat(use_nose_hoover_thermostat)
    integrator.setLangevinPistonFrictionSeed(random_seed)
    integrator.setCharmmContext(ctx)
    integrator.setLangevinPistonFriction(20.0)

    num_heat_steps = 50000
    ihtfrq = 100
    teminc = 2.0
    firstt = 30.0
    finalt = temperature
    step = 0
    temp = firstt
    while temp <= finalt:
        round(temp, 10)  # round() prevents precision errors
        ctx.assignVelocitiesAtTemperature(temp)
        integrator.propagate(ihtfrq)
        step += ihtfrq
        temp += teminc

    if step < num_heat_steps:
        integrator.propagate(num_heat_steps - step)

    box = ctx.getBoxDimensions()
    avg_temp = integrator.getAverageTemperature()
    avg_pres_tensor = integrator.getAveragePressureTensor()
    avg_pres_scalar = integrator.getAveragePressureScalar()

    print("==== HEATING ====")
    print(f"        Box Dims: {box[0]:.12f} x {box[1]:.12f} x {box[2]:.12f}")
    print(f"     Temperature: {avg_temp:.12f} K")
    print(f"        Pressure: {avg_pres_scalar:.12f}")
    print(
        f" Pressure Tensor: {avg_pres_tensor[0][0]:20.12f} {avg_pres_tensor[0][1]:20.12f} {avg_pres_tensor[0][2]:20.12f}"
    )
    print(
        f"                  {avg_pres_tensor[1][0]:20.12f} {avg_pres_tensor[1][1]:20.12f} {avg_pres_tensor[1][2]:20.12f}"
    )
    print(
        f"                  {avg_pres_tensor[2][0]:20.12f} {avg_pres_tensor[2][1]:20.12f} {avg_pres_tensor[2][2]:20.12f}"
    )

    integrator.resetAverages()
    integrator.setLangevinPistonFriction(0.0)
    integrator.propagate(100000)

    box = ctx.getBoxDimensions()
    avg_temp = integrator.getAverageTemperature()
    avg_pres_tensor = integrator.getAveragePressureTensor()
    avg_pres_scalar = integrator.getAveragePressureScalar()

    print("==== EQUILIBRATION ====")
    print(f"        Box Dims: {box[0]:.12f} x {box[1]:.12f} x {box[2]:.12f}")
    print(f"     Temperature: {avg_temp:.12f} K")
    print(f"        Pressure: {avg_pres_scalar:.12f}")
    print(
        f" Pressure Tensor: {avg_pres_tensor[0][0]:20.12f} {avg_pres_tensor[0][1]:20.12f} {avg_pres_tensor[0][2]:20.12f}"
    )
    print(
        f"                  {avg_pres_tensor[1][0]:20.12f} {avg_pres_tensor[1][1]:20.12f} {avg_pres_tensor[1][2]:20.12f}"
    )
    print(
        f"                  {avg_pres_tensor[2][0]:20.12f} {avg_pres_tensor[2][1]:20.12f} {avg_pres_tensor[2][2]:20.12f}"
    )

    # Setup new integrator for running at higher pressure
    integrator = apo.CudaLangevinPistonIntegrator(0.002)
    integrator.useNoseHooverThermostat(use_nose_hoover_thermostat)
    integrator.setLangevinPistonFrictionSeed(random_seed)
    integrator.setReferencePressure([[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]])
    integrator.setCharmmContext(ctx)

    for ITER in range(21):
        integrator.propagate(num_steps)

        box = ctx.getBoxDimensions()
        avg_temp = integrator.getAverageTemperature()
        avg_pres_tensor = integrator.getAveragePressureTensor()
        avg_pres_scalar = integrator.getAveragePressureScalar()

        print(f"==== ITER {ITER} ====")
        print(f"        Box Dims: {box[0]:.12f} x {box[1]:.12f} x {box[2]:.12f}")
        print(f"     Temperature: {avg_temp:.12f} K")
        print(f"        Pressure: {avg_pres_scalar:.12f}")
        print(
            f" Pressure Tensor: {avg_pres_tensor[0][0]:20.12f} {avg_pres_tensor[0][1]:20.12f} {avg_pres_tensor[0][2]:20.12f}"
        )
        print(
            f"                  {avg_pres_tensor[1][0]:20.12f} {avg_pres_tensor[1][1]:20.12f} {avg_pres_tensor[1][2]:20.12f}"
        )
        print(
            f"                  {avg_pres_tensor[2][0]:20.12f} {avg_pres_tensor[2][1]:20.12f} {avg_pres_tensor[2][2]:20.12f}"
        )

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
