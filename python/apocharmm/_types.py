# BEGINLICENSE
# This file is part of apoCHARMM, which is distributed under the BSD 3-clause
# license, as described in the LICENSE file in the top level directory of this
# project.
#
# Author: James E. Gonzales II
#
# ENDLICENSE

import os

FilePath = str | bytes | os.PathLike[str] | os.PathLike[bytes]
FilePaths = FilePath | list[FilePath] | tuple[FilePath, ...]

BoxDimensions = list[float] | tuple[float, float, float]
