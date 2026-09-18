# Atom Selection and Atom References {#atom_selection}

## Purpose

The atom-selection subsystem evaluates CHARMM-style text expressions against one
`CharmmPSF` and exposes two deliberately different result types:

- `AtomSelection` is a set of zero or more zero-based atom indices. It stores
  one bit per atom and is appropriate for masks, membership queries, and
  operations that consume arbitrary atom subsets.
- `AtomReference` is one topology-aware atom identity. It stores one validated
  zero-based index plus shared ownership of the native PSF that defines that
  index. Its storage is constant in the PSF atom count.

`AtomSelector::select()` and its C and Python wrappers produce set-valued
`AtomSelection` results. `AtomSelector::selectAtom()` and its wrappers require
an expression to select exactly one atom and produce an `AtomReference`.

Neither result contains coordinates, forces, energies, charges, masses, or
other physical state. Selection and reference operations are host-resident and
do not allocate CUDA storage, transfer device data, use a CUDA stream, launch a
kernel, or synchronize a device.

## Quick Start

The Python API exposes both result types from the top-level package:

```python
import apocharmm as apo

psf = apo.CharmmPsf("test/data/nacl_pair.psf")
selector = apo.AtomSelector(psf)

all_atoms = selector.select("all")
chloride = selector.selectAtom("bynu 2")
constructed = apo.AtomReference(psf, 1)

assert all_atoms.getNumSelected() == psf.getNumAtoms()
assert chloride.getAtomIndex() == 1
assert chloride == constructed
```

`select()` accepts zero, one, or many matches. `selectAtom()` reports an
invalid-argument error unless the result contains exactly one atom. Direct
`AtomReference` construction uses a zero-based Python index; `BYNU` expression
values remain CHARMM-style one-based atom numbers.

`AtomReference` is the endpoint type used by
[DistanceRestraintForce](@ref distance_restraint_force) in C++ and Python.
Python callers normally obtain exact-one endpoints with `selectAtom()` and pass
the references directly to `DistanceRestraintForce.addRestraint()`. The Python
wrapper passes borrowed reference handles through the typed C ABI; native code
extracts and retains only the indices. The force deliberately ignores topology
provenance.

Direct C++ use follows the same distinction:

```cpp
#include "AtomReference.h"
#include "AtomSelector.h"
#include "CharmmPSF.h"

#include <cassert>
#include <memory>

int main() {
  auto psf = std::make_shared<CharmmPSF>("test/data/nacl_pair.psf");
  AtomSelector selector(psf);

  AtomSelection allAtoms = selector.select("all");
  AtomReference chloride = selector.selectAtom("bynu 2");
  AtomReference constructed(psf, 1);

  assert(allAtoms.getNumSelected() == psf->getNumAtoms());
  assert(chloride == constructed);
  return 0;
}
```

The C ABI uses opaque owned handles and preserves the topology-aware type:

```c
apo_charmm_psf *psf = NULL;
apo_atom_selector *selector = NULL;
apo_atom_reference *reference = NULL;
int atom_index = -1;

if (apo_charmm_psf_create(&psf, "test/data/nacl_pair.psf") != APO_STATUS_OK)
  return 1;
if (apo_atom_selector_create(&selector, psf) != APO_STATUS_OK)
  return 1;
if (apo_atom_selector_select_atom(&reference, selector, "bynu 2") !=
    APO_STATUS_OK)
  return 1;
if (apo_atom_reference_get_atom_index(&atom_index, reference) != APO_STATUS_OK)
  return 1;

apo_atom_reference_destroy(reference);
apo_atom_selector_destroy(selector);
apo_charmm_psf_destroy(psf);
```

Every successful C factory result is newly owned and must be destroyed exactly
once with the corresponding destroy function.

## Construction and Required State

A native `AtomSelector` requires a non-null `std::shared_ptr<const CharmmPSF>`
whose atom count is initialized and non-negative. The C ABI constructor requires
a live `apo_charmm_psf` handle. The Python constructor requires a live
`CharmmPsf`. File-based PSF construction normally establishes the metadata and
derived tables used by selection.

`AtomSelection` requires a non-negative atom count in C++. C and Python callers
normally receive an immutable selection from a selector rather than constructing
one directly. A selection's atom count is fixed unless C++ code explicitly
calls `AtomSelection::setNumAtoms()`, which discards every previous selection
bit and initializes the complete new range to `NONE` or `ALL`.

`AtomReference` can be constructed directly in all three public language
layers:

- C++: `AtomReference(std::shared_ptr<const CharmmPSF>, int)`;
- C ABI: `apo_atom_reference_create()`;
- Python: `AtomReference(CharmmPsf, int)`.

The PSF must be non-null and initialized, and the zero-based index must be in
`[0, atom_count)`. Python additionally requires an actual `CharmmPsf`, requires
an `int`, rejects `bool`, and rejects integers outside the signed C `int` range
before calling the C ABI. Native initialization and atom-range validation remain
native responsibilities.

Each selection expression must be complete. Parentheses must balance, binary
operators require selections on both sides, prefix operators require a following
selection, and field tokens require their documented values. The public
`SelectionParser` entry point is intended for the complete token vector produced
by `SelectionTokenizer::tokenize()`, including its terminal token.

## Set-Valued and Exact-One Results

`AtomSelector::select()` evaluates an expression and returns its complete
`AtomSelection`, including an empty result. The result owns independent bit
storage and does not remember which PSF produced it.

`AtomSelector::selectAtom()` first calls the same native set-valued selection
path. It holds the resulting `AtomSelection` only as a temporary, checks
`getNumSelected()`, extracts the sole index, constructs an `AtomReference` from
the selector's retained PSF, and destroys the temporary selection on return.
The exact cardinality requirement is therefore implemented once in native code
and is shared by C++, C, and Python.

Zero matches and multiple matches both raise or return an invalid-argument
failure with a diagnostic of the form:

```text
Atom selection must match exactly one atom; observed N
```

There is intentionally no conversion from an arbitrary `AtomSelection` to an
`AtomReference`. `AtomSelection` retains only an atom count and bitset, not the
native topology provenance required for pointer-identity semantics. An equal
atom count or a single set bit cannot establish which `CharmmPSF` defines that
index.

## Topology Identity and Equality

An `AtomReference` identifies an atom by the pair:

```text
(native CharmmPSF object identity, zero-based atom index)
```

Topology identity is pointer identity of the retained native `CharmmPSF`
object. It is not path equality, parsed-content equality, atom-count equality,
or value equality of PSF fields. Parsing the same file twice creates two native
PSF objects and therefore two different topologies.

`hasSameTopology()` compares only native PSF object identity. Equality compares
both topology identity and the stored index. No ordering relation is defined.
The Python class defines equality through the C ABI, returns `NotImplemented`
for unrelated Python types, and remains unhashable.

## Ownership and Lifetime

In C++, `AtomSelector` shares ownership of its const PSF and does not clone the
topology. Each returned `AtomSelection` owns independent bit storage and retains
neither the selector nor the PSF. Each returned or directly constructed
`AtomReference` retains its own copied `std::shared_ptr<const CharmmPSF>`.
Closing or destroying the selector or the public PSF owner therefore does not
invalidate an existing reference.

The C ABI returns newly owned `apo_atom_selector`, `apo_atom_selection`, and
`apo_atom_reference` handles. Destroy them with
`apo_atom_selector_destroy()`, `apo_atom_selection_destroy()`, and
`apo_atom_reference_destroy()`, respectively. Constructor and accessor inputs
are borrowed for one call. A selector and every atom reference copy native PSF
shared ownership, so a public `apo_charmm_psf` handle may be destroyed after
those collaborators are created. An `apo_atom_selection` is independent of both
source handles but retains no PSF provenance.

The private C atom-reference handle contains one nullable
`std::unique_ptr<AtomReference>` named `object`. This fits the common
`require_handle_object()` validation path while giving each public handle sole
ownership of one native value. The native value itself owns shared const PSF
state.

Python wrappers own their C handles and release them through `close()`,
`destroy()`, context-manager exit, or finalization. Closure is idempotent.
`AtomSelector` retains the source `CharmmPsf` Python wrapper in addition to the
native selector's shared ownership. `AtomReference` deliberately does not retain
the Python `CharmmPsf` wrapper because its native object already keeps the PSF
alive. `AtomSelection` and `AtomReference` remain valid after their selector or
source PSF wrapper is closed. Methods on a closed wrapper raise `RuntimeError`
through the inherited handle property.

No layer provides internal synchronization. Do not overlap destruction,
closure, assignment, or mutable PSF access with a query on the same object.
Concurrent read-only use requires all referenced handles to remain live and the
shared PSF to remain immutable.

## Data, Shapes, and AKMA Units

Atom counts, selected counts, atom numbers, atom indices, token positions, and
selection bits are dimensionless. No AKMA physical unit applies.

`AtomSelection` stores one bit per atom in 64-bit host words. Atom `i` maps to
word `i / 64` and bit offset `i % 64`; unused high bits in the final word are
zero. Its storage is linear in the PSF atom count.

`AtomReference` stores one shared pointer and one `int`. Its storage is constant
in the PSF atom count. Public reference indices are zero-based.

Public selection-result indices are also zero-based. `BYNU` expression values
are the exception: they use CHARMM-style one-based atom numbers and are
converted to zero-based result indices.

The supported primary forms are case-insensitive:

| Form | PSF field or meaning |
| --- | --- |
| `ALL` | Every atom. |
| `NONE` | No atoms. |
| `TYPE value` | PSF atom name. |
| `CHEM value` | PSF atom type. |
| `SEGI value` | Segment identifier. |
| `RESI value` | Residue identifier formatted as a decimal integer. |
| `RESN value` | Residue name. |
| `BYNU value` | One-based atom number. |
| `ATOM segi resi type` | Intersection of the three supplied fields. |

The tokenizer accepts the longer conventional spellings represented by their
first four characters, such as `CHEMICAL`, `SEGID`, `RESID`, `RESNAME`, and
`BYNUM`. Prefer the canonical spellings above because the current first-four-
character classification also accepts some unintended suffixes.

A single field value supports case-insensitive wildcard matching:

| Wildcard | Meaning |
| --- | --- |
| `*` | Zero or more arbitrary bytes. |
| `#` | Zero or more decimal digits. |
| `%` | Exactly one arbitrary byte. |
| `+` | Exactly one decimal digit. |

`first:last` creates an inclusive range. Reversed endpoints are normalized.
`BYNU` requires integer endpoints, clamps them to the valid one-based atom
range, and converts the result to zero-based indices. Other ranges use numeric
comparison when both endpoints and the PSF field value are complete integers;
otherwise they use case-insensitive lexicographic comparison. Wildcards are not
expanded inside range endpoints.

The binary operators are `.AND.` and `.OR.`. `.AND.` has higher precedence;
operators of equal precedence associate left to right. Parentheses override
precedence. Prefix operators are:

- `.NOT.`, which complements across the complete PSF atom range;
- `.BYRES.`, which replaces each selected atom with its complete residue;
- `.BYGROUP.`, which replaces each selected atom with its complete PSF group;
- `.BONDED.`, which replaces selected atoms with the union of their direct 1-2
  neighbors. Original atoms remain selected only if the connectivity table lists
  them as neighbors.

C++ accepts a `std::string_view`, including non-null-terminated storage, and
reports zero-based byte positions. The C ABI accepts a null-terminated byte
string. Python encodes `str` as UTF-8, so native error positions are UTF-8 byte
offsets rather than Python character indices.

## Cross-Language APIs

### C++

`AtomSelector::select(std::string_view)` returns `AtomSelection`.
`AtomSelector::selectAtom(std::string_view)` returns `AtomReference` and leaves
all parser and cardinality validation in native code. `AtomReference` exposes
`getAtomIndex()`, `getPsf()`, `hasSameTopology()`, `operator==`, and
`operator!=`.

### C ABI

`apo_atom_selector_select()` returns a newly owned `apo_atom_selection`.
`apo_atom_selector_select_atom()` returns a newly owned
`apo_atom_reference`; it never collapses the result to a raw integer.
`apo_atom_reference_create()` performs direct construction from a borrowed PSF
handle and zero-based index. Reference queries are:

- `apo_atom_reference_get_atom_index()`;
- `apo_atom_reference_has_same_topology()`;
- `apo_atom_reference_equals()`.

Every output handle is cleared to `NULL` before other argument validation and
remains `NULL` on verified failure. Scalar query outputs are initialized before
handle validation: the index becomes `-1` and Boolean results become `false`.
Valid atom indices are nonnegative, so `-1` is an invalid sentinel. Callers must
still check the returned status before interpreting any scalar output. Destroy
functions accept `NULL` and prevent C++ exceptions from crossing the C boundary.

### Python

`AtomSelector.select()` returns `AtomSelection`.
`AtomSelector.selectAtom()` returns `AtomReference` by adopting the newly owned
C handle through the private `AtomReference._from_handle()` factory. The factory
does not repeat PSF/index construction. Direct construction is
`AtomReference(psf, atom_index)`. Public methods are `getAtomIndex()` and
`hasSameTopology()`; `==` and `!=` use native equality through the C ABI.

Both selector methods require `str`, encode it as UTF-8, and delegate grammar
handling to native code. Neither method duplicates parser or cardinality logic.

## Errors and Diagnostics

Native C++ construction and evaluation throw `ApoCharmmError` with
`ApoCharmmErrorCode::InvalidArgument` for null inputs, out-of-range reference
indices, lexical or syntax errors, invalid ranges, incompatible selection
sizes, out-of-range query indices, out-of-range neighbor indices, and exact-one
cardinality failures. `ApoCharmmErrorCode::NotInitialized` reports a PSF with a
negative atom count. `ApoCharmmErrorCode::Runtime` reports malformed residue,
group, connectivity, token-stack, operator-stack, or field-token state. Host
allocation can also raise `std::bad_alloc` or `std::length_error`.

C ABI functions return `APO_STATUS_OK` on success. Invalid pointers, handles,
buffers, indices, empty C strings, invalid expressions, and exact-one
cardinality failures return `APO_STATUS_INVALID_ARGUMENT`. Reference or selector
construction from an uninitialized PSF returns `APO_STATUS_NOT_INITIALIZED`.
Invalid residue, group, or bonded-connectivity state, native allocation failure,
internal parser failure, or another translated exception returns
`APO_STATUS_RUNTIME_ERROR`. Per-atom metadata-length consistency remains a
caller-maintained native precondition rather than a validated C status path.

Status-returning functions clear the current thread's previous diagnostic at
entry. Success leaves it empty. On failure, call `apo_last_error()` immediately
on the same thread; its borrowed pointer is invalidated by the next
diagnostic-changing C ABI call on that thread. Destroy functions use the
no-throw destroy guard and preserve an existing diagnostic on normal return.

Python rejects incorrect wrapper or scalar types with `TypeError`. An atom index
outside the signed C `int` range raises `ValueError`. UTF-8 encoding can raise
`UnicodeEncodeError`. Closed objects and impossible success-with-NULL factory
results raise `RuntimeError`. Nonzero native statuses become `ApoCharmmError`,
which retains the status, Python operation context, and copied native
diagnostic.

## Important Behavior and Limitations

Every selector call scans PSF fields to construct primitive selections and
rebuilds atom-to-residue and atom-to-group lookup arrays. Residue and group
ranges are therefore validated even when the expression does not use
`.BYRES.` or `.BYGROUP.`. No parsed-expression or result cache is retained.
`selectAtom()` has the same selection cost and additionally constructs a
short-lived `AtomSelection` before creating its constant-space result.

Selection evaluation reads the current host-side PSF state. Mutating a shared
PSF through unchecked native access can change later selection results. An
existing `AtomReference` continues to name the same native PSF object and index,
but mutation can change what topology metadata that index describes. The
selector is not a snapshot.

The parser validates residue and group ranges and the bonded-connectivity vector
length, but callers must still preserve per-atom metadata lengths and valid
connectivity indices.

The public C and Python `AtomSelection` interfaces are query-only. C++
additionally supports individual bit mutation, clearing, filling, resizing,
intersection, and union. Logical operations require equal atom counts. No layer
supports ordering or hashing of `AtomReference` values.

The Python selector wrappers currently permit embedded null characters in a
`str` before passing it through `ctypes.c_char_p`. The C ABI parses only the
prefix before the first null byte. Avoid embedded null characters. Direct C++
tokenization instead sees an embedded null as a control byte and rejects it.

Token positions use `int` internally without an explicit source-length check.
Expressions larger than `INT_MAX` bytes cannot retain reliable diagnostic
positions. The selection language is byte-oriented and should be written with
ASCII keywords, operators, and field patterns.

## Related Subsystems

- [CharmmPSF](@ref charmm_psf) supplies atom metadata, residue and group
  intervals, and bonded connectivity, and defines AtomReference topology
  identity.
- `HarmonicRestraintForce` consumes an [AtomSelection](@ref AtomSelection) to
  choose restrained atoms.
- `HarmonicCenterOfMassRestraintForce` consumes a nonempty
  [AtomSelection](@ref AtomSelection) to define its restrained center.
- [ApoCharmmError](@ref apocharmm_error) defines native and C ABI failure
  reporting.

## Developer Architecture

The native subsystem has five cooperating layers:

1. `SelectionTokenizer` scans source bytes and emits owned `SelectionToken`
   values plus one terminal token.
2. `SelectionParser` builds residue and group lookup arrays, evaluates the token
   stream with operator and selection stacks, and reads host PSF fields.
3. `AtomSelection` stores a set-valued result as compact host bit words and
   implements set algebra and membership queries.
4. `AtomReference` stores one validated index and shared const ownership of its
   native PSF. Its identity and equality operations use native PSF pointer
   identity.
5. `AtomSelector` owns the PSF relationship, composes tokenization with parsing
   for `select()`, and implements `selectAtom()` by validating a temporary
   selection and constructing an AtomReference with the retained PSF.

The C ABI wraps selectors and selections in private structures containing
`std::shared_ptr`. It wraps each atom reference in a private structure
containing one nullable `std::unique_ptr<AtomReference>` named `object`. The
common member name allows all handles to use `require_handle_object()`. The C
boundary validates pointers and buffers, initializes outputs, translates
`ApoCharmmErrorCode` values to exact `apo_status` values, and translates other
exceptions to `APO_STATUS_RUNTIME_ERROR`.

The Python layer contains no grammar or cardinality implementation. It
configures `ctypes` prototypes, performs Python type and integer-width checks,
adopts newly owned C handles, and converts nonzero statuses to `ApoCharmmError`.

The central invariants are:

- a selection atom count is non-negative;
- its word-vector length is `ceil(atom_count / 64)`;
- out-of-range bits in the final word are zero;
- binary selection operands have equal atom counts;
- a reference retains a non-null initialized PSF and an index in its atom range;
- topology identity is equality of native PSF object addresses;
- a parser token vector has a reachable terminal token;
- PSF per-atom arrays agree with the atom count;
- residue and group records are valid inclusive zero-based ranges;
- bonded-connectivity storage has one valid neighbor set per atom.

Primitive field selection is linear in the PSF atom count. Wildcard matching is
quadratic in the value and pattern lengths in the worst case, with two
pattern-length work arrays. Residue and group lookup construction is linear in
the total number of atoms covered by their intervals. Bitwise intersection and
union process 64 atoms per word. AtomReference queries are constant time.

To add a field, update `SelectionTokenType`, bare-token classification,
`SelectionParser::isFieldToken()`, `SelectionParser::getFieldName()`, and
`SelectionParser::getFieldValue()`, then add native, C ABI, and Python tests. To
add an operator, update tokenization, prefix or binary classification,
precedence where applicable, and `SelectionParser::applyTopOperator()`. Grammar
logic and exact-one cardinality must remain native; the C ABI and Python
wrappers should continue to forward expression bytes rather than duplicate them.

Relevant tests are:

- `test/unittests/unittest-atomReference.cpp` for the native value type;
- `test/unittests/unittest-atomSelection.cpp` for selection parsing and native
  exact-one behavior;
- `test/unittests/unittest-capiAtomReference.cpp` for the C reference and
  selector factory interfaces;
- `test/unittests/unittest-capiAtomSelection.cpp` for set-valued C results;
- `test/pytest/python_api_atom_reference.py` for the Python reference and
  exact-one factory;
- `test/pytest/python_api_atom_selection.py` for set-valued Python selection.

The restraint example `example/cons_harm.py` demonstrates using a selection as a
collaborating object.

Visible technical debt includes copy-like const-rvalue overloads in
`AtomSelection`, nontransactional copy assignment, incomplete validation of
mutably corrupted PSF per-atom arrays, `int` token positions, and embedded-null
truncation at the Python-to-C boundary.

## API Reference

C++:

- [AtomReference](@ref AtomReference)
- [AtomSelection](@ref AtomSelection) and
  [AtomSelection::InitialValue](@ref AtomSelection::InitialValue)
- [AtomSelector](@ref AtomSelector),
  [AtomSelector::select](@ref AtomSelector::select), and
  [AtomSelector::selectAtom](@ref AtomSelector::selectAtom)
- [SelectionTokenType](@ref SelectionTokenType) and
  [SelectionToken](@ref SelectionToken)
- [SelectionTokenizer](@ref SelectionTokenizer)
- [SelectionParser](@ref SelectionParser)

C ABI:

- [apo_atom_reference](@ref apo_atom_reference),
  [apo_atom_reference_create](@ref apo_atom_reference_create), and
  [apo_atom_reference_destroy](@ref apo_atom_reference_destroy)
- [apo_atom_reference_get_atom_index](@ref apo_atom_reference_get_atom_index)
- [apo_atom_reference_has_same_topology](@ref apo_atom_reference_has_same_topology)
- [apo_atom_reference_equals](@ref apo_atom_reference_equals)
- [apo_atom_selector](@ref apo_atom_selector) and
  [apo_atom_selector_create](@ref apo_atom_selector_create)
- [apo_atom_selector_select](@ref apo_atom_selector_select),
  [apo_atom_selector_select_atom](@ref apo_atom_selector_select_atom), and
  [apo_atom_selector_destroy](@ref apo_atom_selector_destroy)
- [apo_atom_selection](@ref apo_atom_selection) and
  [apo_atom_selection_destroy](@ref apo_atom_selection_destroy)
- [apo_atom_selection_get_num_atoms](@ref apo_atom_selection_get_num_atoms)
- [apo_atom_selection_get_num_selected](@ref apo_atom_selection_get_num_selected)
- [apo_atom_selection_get_atom_indices](@ref apo_atom_selection_get_atom_indices)
- [apo_atom_selection_contains](@ref apo_atom_selection_contains)

Python:

- [python_atom_reference](@ref python_atom_reference)
- [python_atom_selector](@ref python_atom_selector)
- [python_atom_selection](@ref python_atom_selection)
