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

    # Load CHARMM force field
    prm = apo.CharmmParameters("toppar/toppar_water_ions.str")

    # Load PSF and coordinates. The sodium starts at the origin and the chloride
    # at (2.82, 2.82, 2.82) angstroms.
    psf = apo.CharmmPsf("test/data/nacl_pair.psf")
    crd = apo.CharmmCrd("test/data/nacl_pair.cor")

    # Setup the CharmmContext. Geometric restraints require P1, which is also
    # the default. Setting it here makes the requirement explicit.
    ctx = apo.CharmmContext(psf, prm)
    ctx.setPeriodicBoundaryCondition(apo.PeriodicBoundaryCondition.P1)
    ctx.setBoxDimensions(box_dims)
    ctx.setCoordinates(crd)

    # Evaluate the unrestraint system for comparison
    ctx.calculatePotentialEnergy()
    fm = ctx.getForceManager()
    unrestrained_energy = fm.getPotentialEnergy()
    unrestrained_forces = fm.getForces()

    # Select atoms with CHARMM-style expressions rather than atom indices.
    # select() returns an AtomSelection that can hold many atoms, and
    # selectAtom() returns an AtomReference to exactly one.
    selector = apo.AtomSelector(psf)
    ions = selector.select("all")
    sodium = selector.selectAtom("atom NACL 1 SOD")
    chloride = selector.selectAtom("atom NACL 2 CLA")

    # One GeometricRestraintForce holds every term. An active term adds the
    # harmonic energy 0.5 * K * (q - offset)^2, where q is the distance from the
    # plane (signed for SYMMETRIC), the cylinder axis, or the sphere center.
    # SYMMETRIC acts on both sides of the offset, INSIDE only where q > offset,
    # and OUTSIDE only where q < offset. Lengths are in angstroms and force
    # constants in kcal/mol/A^2.
    geo = apo.GeometricRestraintForce(psf)

    # Plane z = 0 with its normal along z. RCM evaluates the term once at the
    # mass-weighted center of the selection (PSF masses) and shared the force by
    # mass fraction, so this term holds the center of mass of the pair on the
    # plane.
    geo.addPlane(
        ions,
        origin=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        force_constant=5.0,
        offset=0.0,
        activation=apo.GeometricRestraintActivation.SYMMETRIC,
        selection_mode=apo.GeometricRestraintSelectionMode.RCM,
    )

    # Infinite cylinder around the z axis. ATOMWISE evaluates each selected atom
    # on its own, so each ion moves freely within 3 angstroms of the axis and is
    # pulled back toward it beyond that radius.
    geo.addCylinder(
        ions,
        origin=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        force_constant=10.0,
        offset=3.0,
        activation=apo.GeometricRestraintActivation.INSIDE,
        selection_mode=apo.GeometricRestraintSelectionMode.ATOMWISE,
    )

    # Sphere around the starting position of the sodium. The chloride is pushed
    # outward whenever it is closer than 5.5 angstroms to the center.
    geo.addSphere(
        chloride,
        center=[0.0, 0.0, 0.0],
        force_constant=10.0,
        offset=5.5,
        activation=apo.GeometricRestraintActivation.OUTSIDE,
        selection_mode=apo.GeometricRestraintSelectionMode.ATOMWISE,
    )

    # Subscribe the restraint once. The ForceManager evaluates every term.
    fm.subscribe(geo)

    # Evaluate the restrained system
    ctx.calculatePotentialEnergy()
    restrained_energy = fm.getPotentialEnergy()
    restrained_forces = fm.getForces()

    # The ForceManager reports aggregate values, so the restraint contribution
    # is the change from the unrestrained evaluation.
    restraint_energy = restrained_energy - unrestrained_energy
    print(f"Unrestrained potential energy = {unrestrained_energy:.6f} kcal/mol")
    print(f"Restrained potential energy   = {restrained_energy:.6f} kcal/mol")
    print(f"Geometric restraint energy    = {restraint_energy:.6f} kcal/mol")

    # Terms add on shared atoms. The sodium lies inside the cylinder, so it
    # feels only its mass share of the plane term. The chloride feels all three
    # terms. getAtomIndex() gives each ion's row in the force lists.
    for name, atom in [("SOD", sodium), ("CLA", chloride)]:
        i = atom.getAtomIndex()
        fx = restrained_forces[i][0] - unrestrained_forces[i][0]
        fy = restrained_forces[i][1] - unrestrained_forces[i][1]
        fz = restrained_forces[i][2] - unrestrained_forces[i][2]
        print(f"{name} restraint force = {fx:.6f} {fy:.6f} {fz:.6f} kcal/mol/A")

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
