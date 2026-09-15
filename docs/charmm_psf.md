# CharmmPSF {#charmm_psf}

## Purpose

`CharmmPSF` is apoCHARMM's representation of the molecular topology and
per-atom metadata read from a CHARMM Protein Structure File. It owns atom
identifiers, names, types, charges, masses, bonded topology, connectivity,
exclusion data, residue intervals, recognized water tuples, and interpreted
bonded-component intervals.

Use this subsystem when constructing an apoCHARMM force manager or simulation
context from a PSF file. `CharmmPSF` supplies topology and atom metadata. Force
constants and Lennard-Jones parameters remain in
[CharmmParameters](@ref CharmmParameters), while coordinates, velocities, box
state, and force-evaluation output remain in [CharmmContext](@ref CharmmContext)
and its collaborators.

The native C++ API exposes the complete stored representation, including
low-level mutable access. The public C ABI and Python wrapper expose
construction, lifetime, topology counts, per-atom metadata, aggregate charge and
mass, and the stored path. They do not currently expose native bond, angle,
dihedral, connectivity, water, residue-interval, or group arrays.

## Quick Start

The highest-level supported workflow is the Python wrapper:

```python
from pathlib import Path

import apocharmm as apo

with apo.CharmmPsf(Path("test/data/nacl_pair.psf")) as psf:
    assert psf.getNumAtoms() == 2
    assert psf.getSegmentIdentifiers() == ["NACL", "NACL"]
    assert psf.getCharges() == [1.0, -1.0]
    assert psf.getNetCharge() == 0.0
```

Construction parses the file before returning. The context manager owns the
native handle for the body of the `with` statement and releases it on exit. The
returned lists are independent Python copies.

Direct native use follows the same parse-before-return model:

```cpp
#include "CharmmPSF.h"

int main() {
  CharmmPSF psf("test/data/nacl_pair.psf");

  if (psf.getNumAtoms() != 2)
    return 1;

  return psf.getNetCharge() == 0.0 ? 0 : 1;
}
```

Link the example against the native apoCHARMM library and the CUDA runtime. The
constructor can perform CUDA allocation while creating derived
[CudaContainer](@ref CudaContainer) data even though the primary PSF records are
host-resident.

## Construction and Required State

The default native constructor creates an incomplete object. All six stored
counts are `-1`, all vectors and CudaContainers are empty, and the file name is
empty. `setNumAtoms()` can initialize only the atom count and the seven per-atom
host vectors. It does not create a complete topology.

The file constructor requires a non-empty path and reads the complete file into
host memory. It searches for the following sections in this order:

1. TITLE;
2. ATOM;
3. BOND;
4. ANGLE;
5. DIHEDRAL;
6. IMPROPER;
7. DONOR;
8. ACCEPTOR;
9. CROSS-TERM.

Donor and acceptor record contents are skipped. Sections between ACCEPTOR and
CROSS-TERM are ignored. The parser does not validate the file-name extension or
require a particular PSF header spelling before the first supported section.

Section counts are parsed as unsigned values. ATOM, BOND, ANGLE, DIHEDRAL,
IMPROPER, and CROSS-TERM counts must fit in `int`. Empty topology sections are
accepted. Every ATOM record must contain at least eight whitespace-separated
tokens. The parser stores atom metadata according to record order and does not
use the first atom-serial token to reorder records.

Charges and masses must be fully parseable finite floating-point values.
Residue identifiers must fit in `int`. No physical range is enforced for
charge, mass, or residue identifiers.

Bond, angle, proper-dihedral, improper-dihedral, and cross-term atom numbers
must be in the one-based file range `[1, num_atoms]`. Bond, angle, proper, and
improper records are converted to zero-based indices. The current cross-term
implementation does not perform that final conversion.

After primary parsing, construction performs this derived-state sequence:

1. recognize consecutive `OT`, `HT`, `HT` water triples;
2. construct bonded connected-component intervals;
3. build 1-2, 1-3, and 1-4 connectivity and CHARMM exclusion arrays.

Residue intervals are built while ATOM records are parsed. Operations that
consume a PSF through [CharmmParameters](@ref CharmmParameters) or
[ForceManager](@ref ForceManager) expect the parsed counts, vectors, atom-index
conventions, and derived state to remain consistent.

## Ownership and Lifetime

A native `CharmmPSF` exclusively owns its scalar state, host vectors, strings,
sets, and three CudaContainers. File paths and parsed strings are copied.
Constructors and setters do not retain references to caller-owned strings or
vectors.

Const C++ accessors return borrowed references to member storage. The member
reference must not outlive the PSF object. Iterators, element references, and
raw pointers obtained from a returned vector or CudaContainer follow that
container's invalidation rules. Resizing, assignment, clearing, destruction,
or device reallocation can invalidate previously borrowed addresses.

Copy construction creates independent host storage and independently copies the
host and device mirrors of the residue, water, and group CudaContainers.
Pre-existing divergence between a source container's mirrors is preserved rather
than reconciled. Copy assignment is explicitly defaulted, performs sequential
memberwise assignment, and is not transactional.

Move construction and move assignment transfer all host containers, all three
CudaContainer mirrors, and the stored file path without copying topology or
device data. The moved-from source is restored to the exact default-constructed
state: all six counts are `-1`, all host containers and CudaContainers are
empty, and the stored path is empty. Move assignment is `noexcept`; cleanup
failure while releasing the destination's former device allocations is discarded
through the normal non-throwing destruction path.

The implicit destructor is non-throwing. Nested device allocations are released
through [CudaContainer](@ref CudaContainer) and
[DeviceVector](@ref DeviceVector) non-throwing destruction. Cleanup failures
during destruction are not propagated.

Each public `apo_charmm_psf` handle owns a private
`std::shared_ptr<CharmmPSF>`. Release the handle exactly once with
[apo_charmm_psf_destroy](@ref apo_charmm_psf_destroy); passing `NULL` to that
destroy function is accepted. A C ForceManager or CharmmContext created from the
PSF copies native shared ownership, so destroying the source PSF handle does not
invalidate those objects.

The Python [python_charmm_psf](@ref python_charmm_psf) wrapper owns one C
handle. `close()` and context-manager exit release it idempotently. Calling a
method after closure raises `RuntimeError`. Python ForceManager and
CharmmContext wrappers retain source-wrapper references where required, while
the corresponding native objects separately retain the shared native PSF.

No layer provides internal synchronization for concurrent mutation or
destruction. Externally serialize overlapping access whenever one operation can
mutate the object, a returned native container, or associated CUDA storage.

## Data, Shapes, and AKMA Units

The normal per-atom vectors each contain `getNumAtoms()` elements in PSF
ATOM-record order:

| Data | Representation | Physical meaning or unit |
| --- | --- | --- |
| Segment identifiers | `std::vector<std::string>` | Dimensionless text |
| Residue identifiers | `std::vector<int>` | Dimensionless integer labels |
| Residue names | `std::vector<std::string>` | Dimensionless text |
| Atom names | `std::vector<std::string>` | Dimensionless text |
| Atom types | `std::vector<std::string>` | Dimensionless CHARMM type labels |
| Charges | `std::vector<double>` | Elementary-charge units |
| Masses | `std::vector<double>` | Atomic mass units |

PSF parsing requires finite charge and mass values but performs no positivity or
other physical-range validation.

[Bond](@ref Bond) stores two zero-based atom indices. [Angle](@ref Angle) stores
three zero-based indices with the central atom in `jatom`.
[Dihedral](@ref Dihedral) stores four zero-based indices and represents both
proper and improper records. The current [CrossTerm](@ref CrossTerm)
representation contains two four-atom tuples but retains one-based file atom
numbers.

The 1-2, 1-3, and 1-4 connectivity members are host vectors of sets. The outer
index is a zero-based source atom. Each set contains zero-based connected atom
indices. `getInclusionExclusionLists()` returns an owned flattened copy:

- `sizes[0]` is the number of 1-4 inclusion pairs;
- `sizes[1]` is the number of 1-2/1-3 exclusion pairs;
- the first `2 * sizes[0]` flattened elements are zero-based inclusion pairs;
- the remainder are zero-based exclusion pairs;
- each pair is emitted once with its lower atom index first.

`getIblo14()` and `getInb14()` use the separate CHARMM exclusion
representation. `iblo14[i]` is the cumulative number of flattened entries
through atom `i`. `inb14` stores one-based atom numbers for connected atoms
whose zero-based index is greater than the source atom's index.

The three derived CudaContainers are one-dimensional arrays:

| Container | Element layout |
| --- | --- |
| Water molecules | `int4(oxygen, hydrogen1, hydrogen2, 0)` |
| Residues | `int2(first_atom, last_atom)` |
| Groups | `int2(first_atom, last_atom)` |

All listed atom indices in those CudaContainers are zero-based and interval
endpoints are inclusive. Water recognition currently requires consecutive atom
types `OT`, `HT`, `HT`. Residue intervals currently split when the numeric
residue identifier changes. Group intervals currently assume every bonded
component occupies contiguous atom indices.

The host and device arrays in each [CudaContainer](@ref CudaContainer) are
mirrors by convention, not continuously coherent storage. The PSF file
constructor uses CudaContainer modifiers to populate both mirrors. Later
host-only mutation is not visible on the device until `transferToDevice()`
succeeds. Device-only mutation is not visible on the host until
`transferToHost()` succeeds. These transfers preserve values without unit
conversion.

## Errors

The native file constructor throws [ApoCharmmError](@ref ApoCharmmError) with
`ApoCharmmErrorCode::InvalidArgument` for an empty path. File I/O failures,
missing sections, malformed records, non-finite numeric values, unsupported
counts, and out-of-range topology atom numbers use
`ApoCharmmErrorCode::Runtime`. CUDA failures while creating or copying residue,
water, or group CudaContainers use `ApoCharmmErrorCode::Cuda`.

`setNumAtoms()` reports `ApoCharmmErrorCode::InvalidArgument` for a negative
count. `setAtomCharges()` reports `ApoCharmmErrorCode::NotInitialized` when the
atom count is negative and `ApoCharmmErrorCode::InvalidArgument` for a length
mismatch. Aggregate charge, mass, and inclusion/exclusion methods report
`ApoCharmmErrorCode::NotInitialized` for a negative atom count and
`ApoCharmmErrorCode::Runtime` for inconsistent vector lengths.

Native host allocation can propagate `std::bad_alloc`, and oversized strings or
vectors can propagate `std::length_error`. Operations that modify several
containers are not transactional. An exception can leave already-applied
changes observable when the object itself survives the failed operation.

The C ABI maps invalid inputs to `APO_STATUS_INVALID_ARGUMENT`, parser and
uncategorized runtime failures to `APO_STATUS_RUNTIME_ERROR`, CUDA failures to
`APO_STATUS_CUDA_ERROR`, and uninitialized aggregate state to
`APO_STATUS_NOT_INITIALIZED`. Every status-returning function clears the calling
thread's previous diagnostic. Failure leaves borrowed thread-local text
available through [apo_last_error](@ref apo_last_error). Copy that text before
another guarded call on the same thread. Normal destruction preserves the
previous diagnostic.

C ABI count and scalar output slots are set to zero before handle validation.
After those output pointers are validated, a later failure leaves the zero
visible. Array and text-buffer functions validate the handle and required
capacity before writing; their documented failures leave existing buffer
contents unchanged and therefore potentially stale.

The Python constructor can raise `TypeError` during `os.fsencode`, `OSError`
while loading the configured shared library, `RuntimeError` for missing library
configuration or an impossible successful NULL-handle result, and
`ApoCharmmError` for native status failures. Closed-object use raises
`RuntimeError`. Text decoding can raise `UnicodeDecodeError`.

## Important Behavior and Limitations

### Native mutable access {#charmm_psf_mutation}

Every non-const native accessor is an unchecked escape hatch. The object does
not maintain dirty flags and does not automatically:

- update a stored count after vector resizing;
- validate newly supplied atom indices;
- rebuild residue intervals after metadata changes;
- rebuild water tuples after atom-type changes;
- rebuild connectivity, groups, or exclusions after bond changes;
- rebuild parameter or force data retained by a ForceManager;
- transfer a modified CudaContainer mirror;
- reparse a file after changing the stored file-name string.

After low-level mutation, the caller must restore all affected invariants before
passing the PSF to another subsystem. There is no public one-step native
reinitialization method.

The current parser has several visible constraints:

- it requires the supported sections in a fixed order;
- it ignores donor and acceptor contents;
- it ignores sections between ACCEPTOR and CROSS-TERM;
- it does not validate or reorder by the ATOM record's serial-number token;
- it treats a change in numeric residue identifier as the only residue boundary;
- it assumes bonded connected components occupy contiguous atom intervals;
- it recognizes only consecutive `OT`, `HT`, `HT` water records;
- it retains cross-term atom numbers as one-based values.

The cross-term index convention differs from all other parsed topology records
and is an implementation defect, not a recommended consumer contract. The
current CharmmParameters CMAP packing path is disabled, so the defect is not
covered by an active CMAP consumer.

The C ABI exposes segment identifiers, residue names, atom names, and atom types
as fixed eight-byte fields. Longer native strings are silently truncated. The
fields are space-padded and are not null-terminated. The Python wrapper decodes
those fields as UTF-8 and removes leading and trailing whitespace.

The C ABI file-name getter has no companion required-length query. It writes all
elements in the supplied output buffer, padding beyond the stored path with
spaces. The Python wrapper consequently uses a fixed 1024-byte buffer. Paths
requiring more than 1024 encoded bytes fail, non-UTF-8 path bytes cannot be
decoded by the current wrapper, and trailing spaces in a path are removed.

Python path values are converted with `os.fsencode`. Embedded null bytes are not
rejected before construction; the C-string boundary exposes only the prefix
before the first null byte.

CharmmPSF does not store a CUDA stream or device identifier. Device operations
use the CUDA runtime state current on the calling thread. CudaContainer
modifiers can allocate, copy, launch work on the default stream, or release
device storage. The PSF constructor does not provide an independent
CharmmPSF-level synchronization contract.

## Related Subsystems

- [CharmmParameters](@ref charmm_parameters) consumes PSF atom types and
  topology while packing bonded and Lennard-Jones data.
- [ForceManager](@ref force_manager) retains shared PSF ownership and
  initializes native force implementations.
- [CharmmContext](@ref charmm_context) retains a PSF and imports its charges and
  masses into simulation state.
- [CudaContainer](@ref cuda_container) documents the host/device mirrors used
  for residues, waters, and groups.
- [ApoCharmmError](@ref apocharmm_error) documents native exceptions and
  C/Python error translation.

## Developer Architecture

The public native declarations are in `include/CharmmPSF.h`. Parsing,
derivation, aggregate calculations, and accessor definitions are in
`src/CharmmPSF.cu`, which is registered in `src/CMakeLists.txt`. Primary PSF
parsing is host-side despite the `.cu` extension. CUDA work enters through the
three CudaContainer members.

The native ownership graph has one `CharmmPSF` as the exclusive owner of all
parsed and derived storage. ForceManager and CharmmContext normally hold the
same PSF through `std::shared_ptr<CharmmPSF>`. They do not copy topology during
ordinary association.

The public C ABI type is declared in
`capi/include/apocharmm_c/CharmmPsf.h`. Its private definition in
`capi/private_include/apocharmm_c/detail/CharmmPsfHandle.h` owns one shared
native pointer. `capi/src/apocharmm_c/CharmmPsf.cpp` validates all handles and
buffers, translates native exceptions through the shared guard, and copies
selected values into caller-owned outputs.

The Python class in `python/apocharmm/charmm_psf.py` owns the C handle through
`_ApoObject`. It configures ctypes prototypes once per process and converts
every returned C value into a new Python value. The module is re-exported as
`apocharmm.CharmmPsf` by `python/apocharmm/__init__.py`.

The principal native lifecycle is:

1. initialize sentinel state;
2. read and parse the complete file;
3. build residue intervals during ATOM parsing;
4. parse primary bonded topology;
5. recognize water tuples;
6. derive bonded-component intervals;
7. derive connectivity and exclusion arrays;
8. pass shared ownership to parameter, force, or context collaborators.

Normal parsed-state invariants include nonnegative counts, per-atom vectors
matching the atom count, topology-vector lengths matching their counts, valid
atom indices, outer connectivity vectors matching the atom count, and coherent
derived CudaContainer mirrors. The public mutable accessors bypass every one of
these invariants.

Water, residue, and group CudaContainer modifiers are the native-to-CUDA data
boundary. Repeated `push_back()` can grow host and device allocations
independently and launch a one-thread device write. `shrink_to_fit()` can
reallocate device storage. Construction therefore has allocation and transfer
costs proportional to the derived-record counts, with repeated growth overhead
when many records are appended.

The C ABI guard is the language boundary. It clears thread-local diagnostics,
maps `ApoCharmmErrorCode` values to exact `apo_status` values, maps legacy
`std::invalid_argument` to `APO_STATUS_INVALID_ARGUMENT`, maps other exceptions
to `APO_STATUS_RUNTIME_ERROR`, and prevents exceptions from crossing into C.
The Python status wrapper raises `ApoCharmmError` for nonzero native statuses.

Extension points visible in the current implementation include adding parser
support for skipped PSF sections, adding safe topology-copy APIs to the C ABI,
exposing structured topology in Python, replacing fixed-width text outputs with
length-aware APIs, making derived-state rebuilding explicit, supporting
additional water naming conventions, and enabling CMAP consumption after the
cross-term indexing contract is corrected.

Performance-sensitive paths are complete-file loading, repeated host string
allocation, set-based connectivity construction, repeated CudaContainer
appends, device reallocation during shrinking, deep copying of all three
CudaContainers, and reconstruction of flattened inclusion/exclusion lists on
every call.

Focused native coverage is in
`test/unittests/unittest-charmmPSF.cpp`. Direct C ABI coverage is in
`test/unittests/unittest-capiCharmmPsf.cpp`. Python coverage is in
`test/pytest/python_api_charmm_psf.py`. Stable repository data includes
`test/data/nacl_pair.psf` and the larger PSF fixtures used by those suites.

Known architectural constraints directly visible in the implementation include
unrestricted mutable access, one-based cross terms, residue grouping that
ignores segment changes, connected-component interval assumptions, fixed-width C
ABI text fields, and the Python file-name buffer limit.

## API Reference

- [CharmmPSF](@ref CharmmPSF) is the complete native C++ class reference.
- [Bond](@ref Bond), [Angle](@ref Angle), [Dihedral](@ref Dihedral), and
  [CrossTerm](@ref CrossTerm) describe native topology records.
- [InclusionExclusion](@ref InclusionExclusion) describes the owned flattened
  pair-list result.
- [apo_charmm_psf](@ref apo_charmm_psf) is the public opaque C ABI handle.
- [apo_charmm_psf_create](@ref apo_charmm_psf_create) and
  [apo_charmm_psf_destroy](@ref apo_charmm_psf_destroy) manage C handle
  lifetime.
- `apo_charmm_psf_get_*` functions expose C ABI counts, metadata, aggregates,
  and the stored path.
- [python_charmm_psf](@ref python_charmm_psf) is the Python wrapper reference.
