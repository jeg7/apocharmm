# Atom Selection {#atom_selection}

## Purpose

The atom-selection subsystem converts a CHARMM-style text expression into a
compact set of zero-based atom indices associated with one `CharmmPSF`.
`AtomSelector` evaluates expressions, while `AtomSelection` stores and queries
the result. Use this subsystem when an apoCHARMM operation needs a reproducible
subset of atoms derived from PSF metadata, residue or group membership, or
direct bonded connectivity.

Selection results contain no coordinates, charges, masses, or physical
quantities. They are host-resident, dimensionless index sets. Creating or
querying a selection does not launch a CUDA kernel, transfer data, use a CUDA
stream, or synchronize a device.

## Quick Start

The Python API is the highest-level public interface:

```python
import apocharmm as apo

psf = apo.CharmmPsf("test/data/nacl_pair.psf")
selector = apo.AtomSelector(psf)
all_atoms = selector.select("all")

assert all_atoms.getNumSelected() == psf.getNumAtoms()
print(all_atoms.getAtomIndices())
```

`getAtomIndices()` prints a new Python list containing every zero-based atom
index in ascending order.

Direct C++ use follows the same lifecycle:

```cpp
#include "AtomSelector.h"
#include "CharmmPSF.h"

#include <cassert>
#include <memory>

int main() {
  auto psf = std::make_shared<CharmmPSF>("test/data/nacl_pair.psf");
  AtomSelector selector(psf);
  AtomSelection allAtoms = selector.select("all");

  assert(allAtoms.getNumSelected() == psf->getNumAtoms());
  return 0;
}
```

## Construction and Required State

A native `AtomSelector` requires a non-null `std::shared_ptr<const CharmmPSF>`
whose atom count is non-negative. The C ABI constructor requires a live
`apo_charmm_psf` handle. The Python constructor requires a live `CharmmPsf`.
File-based PSF construction normally establishes the metadata and derived tables
used by selection.

`AtomSelection` requires a non-negative atom count in C++. C and Python callers
normally receive an immutable selection from a selector rather than constructing
one directly. A selection's atom count is fixed unless C++ code explicitly
calls `AtomSelection::setNumAtoms()`, which discards every previous selection
bit and initializes the complete new range to `NONE` or `ALL`.

Each selection expression must be complete. Parentheses must balance, binary
operators require selections on both sides, prefix operators require a following
selection, and field tokens require their documented values. The public
`SelectionParser` entry point is intended for the complete token vector produced
by `SelectionTokenizer::tokenize()`, including its terminal token.

## Ownership and Lifetime

In C++, `AtomSelector` shares ownership of its const PSF. It does not clone the
topology. Each returned `AtomSelection` owns independent bit storage and retains
neither the selector nor the PSF. Copy construction and copy assignment create
independent storage. Move construction and move assignment transfer the owned
word storage without allocation and leave the source as a valid zero-atom
selection.

The C ABI returns newly owned `apo_atom_selector` and `apo_atom_selection`
handles. Destroy them with `apo_atom_selector_destroy()` and
`apo_atom_selection_destroy()`, respectively. Constructor input handles and
accessor input handles are borrowed for one call. A selector retains shared
native ownership of its PSF, so the public PSF handle may be destroyed after the
selector is created. A selection returned by `apo_atom_selector_select()` is
independent of both source handles.

The Python wrappers own their C handles and release them through `close()`,
`destroy()`, context-manager exit, or finalization. `AtomSelector` also retains
the source `CharmmPsf` Python object. An `AtomSelection` remains valid after its
selector or PSF wrapper is closed. Methods on a closed wrapper raise
`RuntimeError`.

No layer provides internal synchronization. Do not overlap destruction or
closure with another operation on the same object. Concurrent read-only use
requires the shared PSF and selection to remain immutable.

## Data, Shapes, and AKMA Units

Atom counts, selected counts, atom numbers, atom indices, token positions, and
selection bits are dimensionless. No AKMA physical unit applies.

`AtomSelection` stores one bit per atom in 64-bit host words. Atom `i` maps to
word `i / 64` and bit offset `i % 64`; unused high bits in the final word are
zero. Public indices are zero-based. `BYNU` expression values are the exception:
they use CHARMM-style one-based atom numbers and are converted to zero-based
result indices.

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

## Errors

Native C++ construction and evaluation throw `ApoCharmmError` with
`ApoCharmmErrorCode::InvalidArgument` for null inputs, negative atom counts,
lexical or syntax errors, invalid ranges, incompatible selection sizes,
out-of-range query indices, and out-of-range neighbor indices read from PSF
bonded connectivity. `ApoCharmmErrorCode::NotInitialized` reports a PSF with a
negative atom count. `ApoCharmmErrorCode::Runtime` reports malformed residue,
group, connectivity, token-stack, operator-stack, or field-token state. Host
allocation can also raise `std::bad_alloc` or `std::length_error`.

C ABI functions return `APO_STATUS_OK` on success. Invalid pointers, handles,
buffers, indices, empty C strings, and invalid expressions return
`APO_STATUS_INVALID_ARGUMENT`. Selector construction from an uninitialized PSF
returns `APO_STATUS_NOT_INITIALIZED`. Invalid residue, group, or
bonded-connectivity state, native allocation failure, internal parser failure,
or another translated exception returns `APO_STATUS_RUNTIME_ERROR`. Per-atom
metadata-length consistency remains a caller-maintained native precondition
rather than a validated C status path. Output handles are set to `NULL` before
validation. Scalar outputs are initialized to zero or `false` where documented.
Call `apo_last_error()` immediately after a failure on the same thread; the
borrowed diagnostic is invalidated by the next status-returning C ABI call on
that thread.

Python rejects incorrect wrapper or scalar types with `TypeError`. An atom index
outside the signed C `int` range raises `ValueError`. UTF-8 encoding can raise
`UnicodeEncodeError`. Closed objects raise `RuntimeError`. Nonzero native
statuses become `ApoCharmmError`, which retains the status, Python operation
context, and copied native diagnostic.

## Important Behavior and Limitations

Every selector call scans PSF fields to construct primitive selections and
rebuilds atom-to-residue and atom-to-group lookup arrays. Residue and group
ranges are therefore validated even when the expression does not use
`.BYRES.` or `.BYGROUP.`. No parsed-expression or result cache is retained.

Selection evaluation reads the current host-side PSF state. Mutating a shared
PSF through unchecked native access can change results. The parser validates
residue and group ranges and the bonded-connectivity vector length, but callers
must still preserve per-atom metadata lengths and valid connectivity indices.
The selector is not a snapshot.

The public C and Python selection interfaces are query-only. C++ additionally
supports individual bit mutation, clearing, filling, resizing, intersection,
and union. Logical operations require equal atom counts.

The Python wrapper currently permits embedded null characters in a `str` before
passing it through `ctypes.c_char_p`. The C ABI parses only the prefix before the
first null byte. Avoid embedded null characters. Direct C++ tokenization instead
sees an embedded null as a control byte and rejects it.

Token positions use `int` internally without an explicit source-length check.
Expressions larger than `INT_MAX` bytes cannot retain reliable diagnostic
positions. The selection language is byte-oriented and should be written with
ASCII keywords, operators, and field patterns.

## Related Subsystems

- [CharmmPSF](@ref charmm_psf) supplies atom metadata, residue and group
  intervals, and bonded connectivity.
- `HarmonicRestraintForce` consumes an [AtomSelection](@ref AtomSelection) to
  choose restrained atoms.
- `HarmonicCenterOfMassRestraintForce` consumes a nonempty
  [AtomSelection](@ref AtomSelection) to define its restrained center.
- [ApoCharmmError](@ref apocharmm_error) defines native and C ABI failure
  reporting.

## Developer Architecture

The subsystem has four native layers:

1. `SelectionTokenizer` scans source bytes and emits owned `SelectionToken`
   values plus one terminal token.
2. `SelectionParser` builds residue and group lookup arrays, evaluates the token
   stream with operator and selection stacks, and reads host PSF fields.
3. `AtomSelection` stores the result as compact host bit words and implements
   set algebra and membership queries.
4. `AtomSelector` owns the PSF relationship and composes tokenization with
   parsing for the public C++ entry point.

The C ABI wraps native selectors and selections in private structures containing
`std::shared_ptr`. The C boundary validates pointers and buffers, initializes
outputs, translates `ApoCharmmErrorCode` values to exact `apo_status` values,
and translates other exceptions to `APO_STATUS_RUNTIME_ERROR`. The Python layer
contains no grammar implementation; it configures `ctypes` prototypes, performs
Python type and integer-width checks, and converts nonzero statuses to
`ApoCharmmError`.

The central invariants are:

- a selection atom count is non-negative;
- its word-vector length is `ceil(atom_count / 64)`;
- out-of-range bits in the final word are zero;
- binary selection operands have equal atom counts;
- a parser token vector has a reachable terminal token;
- PSF per-atom arrays agree with the atom count;
- residue and group records are valid inclusive zero-based ranges;
- bonded-connectivity storage has one valid neighbor set per atom.

Primitive field selection is linear in the PSF atom count. Wildcard matching is
quadratic in the value and pattern lengths in the worst case, with two
pattern-length work arrays. Residue and group lookup construction is linear in
the total number of atoms covered by their intervals. Bitwise intersection and
union process 64 atoms per word.

To add a field, update `SelectionTokenType`, bare-token classification,
`SelectionParser::isFieldToken()`, `SelectionParser::getFieldName()`, and
`SelectionParser::getFieldValue()`, then add native, C ABI, and Python tests. To
add an operator, update tokenization, prefix or binary classification,
precedence where applicable, and `SelectionParser::applyTopOperator()`. Grammar
logic must remain native; the C ABI and Python wrappers should continue to
forward expression bytes rather than duplicate parsing.

Relevant tests are `test/unittests/unittest-atomSelection.cpp`,
`test/unittests/unittest-capiAtomSelection.cpp`, and
`test/pytest/python_api_atom_selection.py`. The restraint example
`example/cons_harm.py` demonstrates using a selection as a collaborating object.

Visible technical debt includes nontransactional copy assignment, incomplete
validation of mutably corrupted PSF per-atom arrays, `int` token positions, and
embedded-null truncation at the Python-to-C boundary.

## API Reference

C++:

- [AtomSelection](@ref AtomSelection) and
  [AtomSelection::InitialValue](@ref AtomSelection::InitialValue)
- [AtomSelector](@ref AtomSelector)
- [SelectionTokenType](@ref SelectionTokenType) and
  [SelectionToken](@ref SelectionToken)
- [SelectionTokenizer](@ref SelectionTokenizer)
- [SelectionParser](@ref SelectionParser)

C ABI:

- [apo_atom_selector](@ref apo_atom_selector) and
  [apo_atom_selector_create](@ref apo_atom_selector_create)
- [apo_atom_selector_select](@ref apo_atom_selector_select) and
  [apo_atom_selector_destroy](@ref apo_atom_selector_destroy)
- [apo_atom_selection](@ref apo_atom_selection) and
  [apo_atom_selection_destroy](@ref apo_atom_selection_destroy)
- [apo_atom_selection_get_num_atoms](@ref apo_atom_selection_get_num_atoms)
- [apo_atom_selection_get_num_selected](@ref apo_atom_selection_get_num_selected)
- [apo_atom_selection_get_atom_indices](@ref apo_atom_selection_get_atom_indices)
- [apo_atom_selection_contains](@ref apo_atom_selection_contains)

Python:

- [python_atom_selector](@ref python_atom_selector)
- [python_atom_selection](@ref python_atom_selection)
