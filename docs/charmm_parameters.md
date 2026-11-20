# CharmmParameters {#charmm_parameters}

## Purpose

`CharmmParameters` is apoCHARMM's host-side representation of the bonded and
Lennard-Jones records read from CHARMM `.prm` and `.str` files. It parses force
field records, stores lookup tables keyed by atom type, and combines those
tables with a [CharmmPSF](@ref CharmmPSF) when [ForceManager](@ref ForceManager)
initializes its bonded and direct-space CUDA force objects.

Use this subsystem when constructing a simulation context or force manager from
CHARMM force-field files. It stores force-field parameters only. Topology, atom
charges, masses, and connectivity remain in [CharmmPSF](@ref CharmmPSF);
coordinates remain in the context and coordinate subsystems.

## Quick Start

The Python API accepts one path or an ordered `list` or `tuple` of paths:

```python
import apocharmm as apo

parameters = apo.CharmmParameters("toppar/toppar_water_ions.str")
psf = apo.CharmmPsf("test/data/waterbox.psf")
context = apo.CharmmContext(psf, parameters)
context.setBoxDimensions([50.0, 50.0, 50.0])
```

After construction, `parameters` owns a parsed native parameter set. The context
and its native ForceManager retain the native parameter data needed for later
initialization. Coordinates must still be supplied before an energy or dynamics
operation that requires them.

Direct C++ use follows the same ownership relationship:

```cpp
#include "CharmmParameters.h"
#include "CharmmPSF.h"
#include "ForceManager.h"

#include <memory>

const auto parameters = std::make_shared<CharmmParameters>(
    "toppar/toppar_water_ions.str");
const auto psf =
    std::make_shared<CharmmPSF>("test/data/waterbox.psf");
ForceManager forceManager(psf, parameters);
```

This constructs an uninitialized `ForceManager`; box and PME configuration are
separate ForceManager responsibilities.

## Construction and Required State

The default C++ constructor creates an empty parameter set. The file-reading C++
constructors, both C ABI creation functions, and the Python constructor parse
files immediately on the calling host thread. A multi-file construction reads
files in the supplied order.

Paths must be non-empty. File-name extensions are not validated. A path whose
uppercase spelling contains `TOPPAR` is
treated as a stream-style topology/parameter file: parsing first searches for a
line containing both `READ` and `PARA`, then starts processing the following
parameter block. Other paths are processed from the beginning of the file.

During parsing, `!` comments are removed, tabs are converted to spaces, leading
and trailing whitespace is removed, and record text is converted to uppercase.
Lines beginning with `*` are skipped. The following sections are recognized:

| Section                        | Current behavior                                                                |
| ------------------------------ | ------------------------------------------------------------------------------- |
| `BONDS`                        | Parses four-token harmonic bond records.                                        |
| `ANGLES`                       | Parses five-token angle records or seven-token angle plus Urey-Bradley records. |
| `DIHEDRALS`                    | Parses seven-token proper-dihedral Fourier records.                             |
| Headings beginning with `IMPR` | Parses seven-token harmonic improper records.                                   |
| `CMAP`                         | Parses nine-token headers, each followed by a 24 x 24 grid of energies.         |
| `NONBONDED`                    | Parses four-token regular or seven-token regular plus 1-4 records.              |
| `NBFIX`                        | Parses four-token regular or six-token regular plus 1-4 pair overrides.         |
| `ATOMS`, `HBOND`               | Recognized but ignored.                                                         |

Supported data-record layouts are:

* `BONDS`: `[type1, type2, kb, b0]`.
* `ANGLES`: `[type1, type2, type3, kTheta, theta0_degrees]`, optionally
  followed by `[kub, s0]`.
* `DIHEDRALS`: `[type1, type2, type3, type4, kChi, n, delta_degrees]`.
* `IMPR*`: `[type1, type2, type3, type4, kPsi, ignored_multiplicity,
  psi0_degrees]`.
* `CMAP`: `[type1, type2, type3, type4, type5, type6, type7, type8, 24]`,
  followed by 576 energies in kcal mol^-1 on as many lines as needed. Phi, the
  first torsion, is the outer loop and psi the inner loop, both from -180
  degrees in 15-degree steps.
* `NONBONDED`: `[type, ignored, epsilon, Rmin/2]`, optionally followed by
  `[ignored_1_4, epsilon14, Rmin/2_14]`.
* `NBFIX`: `[type1, type2, emin, Rmin]`, optionally followed by
  `[emin14, Rmin14]`.

The ignored numeric fields are still parsed and must be valid finite numbers.

The public C++ method
[CharmmParameters::readCharmmParameterFile](@ref CharmmParameters::readCharmmParameterFile)
merges an additional file into an existing object. It is not exposed by the
current C ABI or Python wrapper. It also does not append the path to
[CharmmParameters::getPrmFilePaths](@ref CharmmParameters::getPrmFilePaths);
that getter reports only paths supplied to a successful file-reading
constructor.

Duplicate handling is section-specific:

* BOND, ANGLE, Urey-Bradley, IMPROPER, and NBFIX maps keep the first value for a
  canonical key.
* DIHEDRALS records append every Fourier term under the key in file-read order.
* CMAP records replace an earlier map for the same key, as with CHARMM's
  flexible parameter reader. CMAP keys keep the listed atom-type order, so,
  unlike a proper dihedral, a reversed torsion does not match.
* Regular NONBONDED values replace earlier values for the atom type. Explicit
  1-4 values are replaced only by a later record that also contains 1-4 fields.

No physical-range validation is performed for parsed force constants,
distances, phases, or multiplicities. Numeric fields must nevertheless be
finite, fully parseable, and representable by the destination native type.
Construction also does not require that any supported parameter record be
loaded; a readable non-TOPPAR file containing no recognized records succeeds
with an empty parameter set.

## Ownership and Lifetime

A C++ `CharmmParameters` object owns all strings, maps, and vectors that it
parses. Its map getters return borrowed const references that alias this
storage.
Those references must not outlive the object or an assignment or move that
replaces its state. Incremental reads are visible through existing map
references; appending proper-dihedral terms can invalidate references and
iterators into the affected term vector.

The output of
[CharmmParameters::getBondedParamsAndLists](@ref CharmmParameters::getBondedParamsAndLists)
and
[CharmmParameters::getVdwParamsAndTypes](@ref CharmmParameters::getVdwParamsAndTypes)
is an owned host-side copy. The input `std::shared_ptr<CharmmPSF>` is borrowed
for the call and is neither mutated nor retained.

Each public `apo_charmm_parameters` C handle owns a private
`std::shared_ptr<CharmmParameters>`. Destroy it exactly once with
[apo_charmm_parameters_destroy](@ref apo_charmm_parameters_destroy). The destroy
function accepts `NULL`. A C ForceManager created from the handle copies native
shared ownership, so the source parameter handle may be destroyed without
invalidating that manager.

The Python [python_charmm_parameters](@ref python_charmm_parameters) wrapper
owns one C handle. `close()` and context-manager exit release it idempotently.
Original Python path objects and temporary encoded path buffers are not
retained. A Python ForceManager wrapper also retains the parameter wrapper,
while the native manager separately retains the native shared object.

None of these layers provides internal synchronization for concurrent host
mutation, use, or destruction. Coordinate concurrent access externally.

## Data, Shapes, and AKMA Units

The parsed value types use apoCHARMM's AKMA conventions:

| Value                             | Meaning and unit                                                       |
| --------------------------------- | ---------------------------------------------------------------------- |
| `BondValues::kb`                  | Harmonic bond or Urey-Bradley force constant, kcal mol^-1 angstrom^-2. |
| `BondValues::b0`                  | Equilibrium bond or Urey-Bradley distance, angstroms.                  |
| `AngleValues::kTheta`             | Harmonic angle force constant, kcal mol^-1 radian^-2.                  |
| `AngleValues::theta0`             | Equilibrium angle, radians; input degrees are converted while parsing. |
| `DihedralValues::kChi`            | Proper-dihedral Fourier amplitude, kcal mol^-1.                        |
| `DihedralValues::n`               | Dimensionless integer multiplicity.                                    |
| `DihedralValues::delta`           | Proper-dihedral phase, degrees.                                        |
| `ImDihedralValues::kPsi`          | Harmonic improper force constant, kcal mol^-1 radian^-2.               |
| `ImDihedralValues::psi0`          | Improper equilibrium phase as parsed, degrees.                         |
| `CmapValues::values`              | CMAP grid energies, kcal mol^-1.                                       |
| `CmapValues::coeff`               | Bicubic CMAP coefficients, 16 per grid cell, kcal mol^-1.              |
| `VdwParameters::epsilon`          | NONBONDED well-depth value as written, kcal mol^-1.                    |
| `VdwParameters::rmin_2`           | Half of the minimum-energy pair distance, angstroms.                   |
| `NBFixParameters::emin`, `emin14` | Non-negative pair well-depth magnitudes, kcal mol^-1.                  |
| `NBFixParameters::rmin`, `rmin14` | Pair minimum-energy distances, angstroms.                              |

Default-initializing the public `NBFixParameters` aggregate leaves its four
`double` members indeterminate. Value-initialize it or initialize all six
members before reading it.

### Bonded packing

[BondedParamsAndLists](@ref BondedParamsAndLists) contains four host vectors.
`paramsSize` and `listsSize` each have six entries in this fixed order:

1. bond;
2. Urey-Bradley;
3. angle;
4. proper dihedral;
5. improper dihedral;
6. CMAP.

`paramsVal` concatenates coefficient rows in the same category order:

| Category          | Row layout                                 |
| ----------------- | ------------------------------------------ |
| Bond              | `[b0, kb]`                                 |
| Urey-Bradley      | `[s0, kub]`                                |
| Angle             | `[theta0_radians, kTheta]`                 |
| Proper dihedral   | `[signed_n, kChi, sin(delta), cos(delta)]` |
| Improper dihedral | `[psi0_degrees, kPsi, 0, 1]`               |
| CMAP              | One bicubic coefficient, `[c]`             |

For a proper-dihedral key with multiple Fourier terms, every nonterminal row has
its multiplicity negated to signal continuation to the CUDA consumer. The final
row retains a positive multiplicity. This encoding assumes positive input
multiplicities.

Each CMAP map present in the PSF contributes 9216 consecutive one-coefficient
rows in the order of [CmapValues::coeff](@ref CmapValues::coeff).

`listVal` concatenates zero-based PSF atom-index rows. A `type` field is an
index into the corresponding category's coefficient rows; a CMAP `type`
indexes maps instead, each spanning 9216 rows:

| Category                     | Row layout                                                       |
| ---------------------------- | ---------------------------------------------------------------- |
| Bond and Urey-Bradley        | `[i, j, type, 13]`                                               |
| Angle                        | `[i, j, k, type, 13, 13]`                                        |
| Proper and improper dihedral | `[i, j, k, l, type, 13, 13, 13]`                                 |
| CMAP                         | `[i1, j1, k1, l1, i2, j2, k2, l2, type, 13, 13, 13, 13, 13, 13]` |

The shift value `13` is the zero periodic image under apoCHARMM's 27-cell shift
encoding. CharmmParameters creates these rows in host memory. ForceManager later
passes them to `CudaBondedForce::setup_list` and `setup_coef`, which allocate
and
copy the device data.

Bond and angle keys are canonicalized by sorting their interchangeable outer
atom types. Proper-dihedral lookup first attempts an exact canonical key, then
`X-middle-middle-X`. Improper lookup first attempts an exact key, then
`outer-X-X-outer`. Missing Urey-Bradley records are allowed; entries with
`abs(kub) <= 0.01` are omitted from packed output. Missing required bond, angle,
proper-dihedral, or improper parameters fail the packing operation.

### Lennard-Jones packing

For a PSF with `N` distinct atom types, the types are assigned indices in
ascending lexicographic order. Both `vdwParams` and `vdw14Params` contain
`N * (N + 1)` floats. For `0 <= j <= i < N`, pair `(i, j)` begins at offset

```text
2 * (i * (i + 1) / 2 + j)
```

and stores `[C6, C12]`, where

```text
C6  = 2 * epsilon_pair * Rmin_pair^6
C12 = epsilon_pair * Rmin_pair^12
```

`C6` has units kcal mol^-1 angstrom^6 and `C12` has units
kcal mol^-1 angstrom^12. Without NBFIX, the implementation computes
`epsilon_pair = sqrt(epsilon_i * epsilon_j)` and sets `Rmin_pair` to the sum of
the two per-type `Rmin/2` values. The 1-4 table substitutes explicit per-type
1-4 values where available.
`vdwTypes` and `vdw14Types` each contain one zero-based type index per PSF atom;
the current implementation produces identical arrays.

These arrays are host-owned copies. ForceManager later transfers them into the
direct-space CUDA force object. CharmmParameters itself performs no CUDA
allocation, transfer, stream operation, or synchronization.

## Errors

The C++ layer throws `ApoCharmmError` with
`ApoCharmmErrorCode::InvalidArgument` for an empty path, an empty constructor
file list, or a null PSF supplied to a packing method. It throws
`ApoCharmmErrorCode::Runtime` for file-open or read failures, a missing TOPPAR
parameter block, malformed or non-finite numeric data, invalid record shapes,
missing required bonded parameters, or missing regular NONBONDED parameters for
a PSF atom type. Host allocation can also propagate `std::bad_alloc` from the
native C++ API. Incremental file loading is not transactional: records merged
before a later failure remain observable.

The C ABI maps invalid inputs to `APO_STATUS_INVALID_ARGUMENT` and parsing,
file-I/O, allocation, or other uncategorized native failures to
`APO_STATUS_RUNTIME_ERROR`. Each status-returning call clears the calling
thread's previous diagnostic. On failure, copy the borrowed text returned by
[apo_last_error](@ref apo_last_error) before another guarded C ABI call on the
same thread. A successful destroy preserves the previous diagnostic.

The Python wrapper raises `TypeError` when `os.fsencode` cannot convert a path.
It raises `RuntimeError` when `APOCHARMM_LIBRARY_PATH` is unset or empty, when
native construction reports success but returns a NULL handle, or when
closed-object use accesses the released handle. Loading the configured shared
library can raise `OSError`. Native invalid-argument and runtime statuses become
`ApoCharmmError`.

## Important Behavior and Limitations

* `ATOMS` and `HBOND` sections are ignored.
* CMAP grids must have 24 points per torsion, the size of the CHARMM protein
  maps, while CHARMM accepts any size. CMAP torsions have no wildcard matching
  and must be listed in the parameter file's order.
* The parser normalizes record text to uppercase but preserves constructor path
  strings exactly as supplied. File-name extensions are not checked, and a
  readable file with no recognized records can produce an empty parameter set.
* Python paths containing embedded null bytes are not rejected; the current
  C-string boundary exposes only the prefix before the first null byte.
* Incremental reads mutate existing maps directly and can leave a partial merge
  after failure.
* Duplicate precedence differs by section; do not assume every later file
  overrides every earlier record.
* Proper-dihedral multiplicities are not range-validated even though packed
  continuation requires a positive terminal multiplicity.
* The current improper packing writes `[psi0, kPsi, 0, 1]`, while the CUDA
  improper consumer interprets its first component as a periodicity/mode field.
  Nonzero `psi0` therefore does not select the consumer's harmonic branch. This
  is tracked as an implementation defect rather than a stable API guarantee.
* The current 1-4 Lennard-Jones packing uses regular NBFIX `emin` and `rmin`
  values even when explicit `emin14` and `rmin14` values were parsed. This is
  tracked as an implementation defect rather than a stable API guarantee.
* `getVdwParamsAndTypes` accepts a non-const lvalue reference to a shared
  pointer even though the implementation does not modify that shared pointer or
  its PSF. This unnecessarily restricts native call sites.
* CharmmParameters is host-only. CUDA allocation and transfer occur later in
  ForceManager initialization.

## Related Subsystems

* [CharmmPSF](@ref charmm_psf) supplies atom types and bonded topology used
  during packing.
* [ForceManager](@ref force_manager) consumes the packed host data and
  initializes CUDA force objects.
* [CharmmContext](@ref charmm_context) retains the ForceManager used by a
  simulation state.
* [ApoCharmmError](@ref apocharmm_error) defines native error categories and
  C/Python error translation.

## Developer Architecture

### Layers and ownership graph

The public C++ layer consists of [CharmmParameters](@ref CharmmParameters), its
key and value types, and the two packed host-data structs. The implementation in
`src/CharmmParameters.cu` performs only host-side parsing and packing despite
the `.cu` extension.

The public C ABI exposes only construction and destruction. Its private
`apo_charmm_parameters` definition owns a
`std::shared_ptr<CharmmParameters>`. `apo_force_manager_create` copies that
shared pointer into the private ForceManager handle before constructing the
native manager. The Python class owns the C handle through `_ApoObject` and
performs no independent parameter parsing.

```text
Python CharmmParameters
        |
        | owns
        v
apo_charmm_parameters C handle
        |
        | shared ownership
        v
native CharmmParameters <----- shared ownership ----- C ForceManager handle
```

### State and parsing invariants

All parsed atom-type tokens are uppercase. Bond keys order their two atom types;
angle keys order their outer atom types; torsion keys choose one canonical
forward/reverse ordering. The key constructors themselves do not enforce these
invariants, so code that inserts directly into maps must canonicalize first.
`BondKey`, `AngleKey`, and `DihedralKey` use descending lexicographic map
ordering; NBFIX tuple keys and string-keyed maps use the standard ascending
ordering.

The parser dispatches by an internal section state. Each recognized record
helper validates its exact token count and delegates numeric conversion to the
shared string utilities, which reject incomplete, out-of-range, and non-finite
values with `ApoCharmmErrorCode::Runtime`. Parsed maps are updated record by
record; there is no staging object or rollback boundary.

### Host/device data flow

`getBondedParamsAndLists` and `getVdwParamsAndTypes` are the native-to-GPU data
boundary. They read a borrowed PSF, deduplicate parameter rows, assign local
type
indices, and return owned host vectors. During `ForceManager::initialize`, the
bonded vectors are passed to `CudaBondedForce::setup_list` and `setup_coef`, and
the Lennard-Jones vectors are passed to the direct-space force setters. Those
consumers perform host-to-device allocation and copies. No stream is accepted or
synchronized by CharmmParameters itself.

### Error boundaries

Native parsing and matching failures use `ApoCharmmError`. The C ABI's guarded
boundary clears the thread-local diagnostic, maps native categories to
`apo_status`, catches legacy standard exceptions, and prevents exceptions from
crossing C. The Python prototype configuration translates non-OK statuses into
`ApoCharmmError` while preserving Python-side path-conversion errors as
`TypeError`. Shared-library discovery and loading remain ordinary Python
`RuntimeError` and `OSError` paths before a native status-returning call occurs.

### Extension points and performance constraints

CMAP grids of other sizes are the clearest extension point. CHARMM sizes each
map independently, while the parser and the CUDA kernel assume 24 points per
torsion; other sizes need per-map grid metadata in the packed rows and in the
kernel.

Packing is intentionally performed once during ForceManager initialization.
The implementation uses ordered maps and repeated linear searches through
vectors of already encountered keys. This keeps the packing format simple but
can become expensive for force fields or topologies with many distinct
parameter keys. Preserve the category order, row widths, type-index locality,
and proper-dihedral continuation encoding when reorganizing this code.

Relevant regression coverage is in
`test/unittests/unittest-charmmParameters.cpp`,
`test/unittests/unittest-capiCharmmParameters.cpp`, and
`test/pytest/python_api_charmm_parameters.py`. ForceManager initialization and
CudaBondedForce tests exercise the downstream packed-data consumers. The CMAP
case in `test/unittests/unittest-forceManager.cpp` checks the 1UBQ CMAP energy
and forces against OpenMM reference values, the bonded gradient against a
finite difference, and that the CMAP virial does not change under a
periodic-image shift.

## API Reference

C++ types:

* [CharmmParameters](@ref CharmmParameters)
* [BondKey](@ref BondKey) and [BondValues](@ref BondValues)
* [AngleKey](@ref AngleKey) and [AngleValues](@ref AngleValues)
* [DihedralKey](@ref DihedralKey), [DihedralValues](@ref DihedralValues), and
  [ImDihedralValues](@ref ImDihedralValues)
* [CmapKey](@ref CmapKey) and [CmapValues](@ref CmapValues)
* [VdwParameters](@ref VdwParameters) and
  [NBFixParameters](@ref NBFixParameters)
* [BondedParamsAndLists](@ref BondedParamsAndLists) and
  [VdwParamsAndTypes](@ref VdwParamsAndTypes)

C ABI:

* [apo_charmm_parameters](@ref apo_charmm_parameters)
* [apo_charmm_parameters_create](@ref apo_charmm_parameters_create)
* [apo_charmm_parameters_create_from_files](@ref apo_charmm_parameters_create_from_files)
* [apo_charmm_parameters_destroy](@ref apo_charmm_parameters_destroy)

Python:

* [python_charmm_parameters](@ref python_charmm_parameters)
