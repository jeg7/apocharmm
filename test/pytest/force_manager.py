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
    # Make Force Manager object
    # prm = apo.CharmmParameters("test/data/toppar_water_ions.str")
    prm = apo.CharmmParameters(
        ["test/data/par_all36m_prot.prm", "test/data/toppar_water_ions.str"]
    )
    psf = apo.CharmmPsf("test/data/1lvz.psf")

    fm = apo.ForceManager(psf, prm)

    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
