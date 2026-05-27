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
    # Make CHARMM PSF object
    psf = apo.CharmmPsf("test/data/1lvz.psf")

    natom = psf.getNumAtoms()
    nbond = psf.getNumBonds()
    nangl = psf.getNumAngles()
    ndihe = psf.getNumDihedrals()
    nimpr = psf.getNumImpropers()
    ncrtm = psf.getNumCrossTerms()

    segis = psf.getSegmentIdentifiers()
    resis = psf.getResidueIdentifiers()
    resns = psf.getResidueNames()
    names = psf.getAtomNames()
    types = psf.getAtomTypes()
    chrgs = psf.getCharges()
    amass = psf.getMasses()

    net_charge = psf.getNetCharge()
    total_mass = psf.getTotalMass()
    file_name = psf.getFileName()

    for i in range(natom):
        print(
            "{:8}: {:8} {:10} {:8} {:8} {:8} {:8} {:8}".format(
                i, segis[i], resis[i], resns[i], names[i], types[i], chrgs[i], amass[i]
            )
        )
    print("Net charge: {:8}".format(net_charge))
    print("Total mass: {:8}".format(total_mass))
    print("File name: '{}'".format(file_name))

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
