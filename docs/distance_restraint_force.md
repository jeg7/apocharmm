# DistanceRestraintForce {#distance_restraint_force}

## Purpose

`DistanceRestraintForce` implements the CHARMM `RESDistance`/`RESD` restraint
model. It is not limited to a harmonic restraint between one pair of atoms.

One `DistanceRestraintForce` object may contain multiple restraint terms. Each
term may contain one or more weighted atom pairs. Terms accumulate additively in
energy, potential-energy gradient, and internal virial.

A single-pair term can represent an ordinary distance restraint. A multi-pair
term can represent a reaction coordinate such as a difference of distances.

## Mathematical Model

For one term, let `r_j` be the raw primary-coordinate distance of pair `j`, and
let `c_j` be that pair's coefficient. The term deviation and active energy are:

```text
D = sum_j(c_j * r_j^IVAL) - RVAL

U = SCALE * (KVAL / EVAL) * D^EVAL
```

When a one-sided condition makes the term inactive, its energy, gradient, and
virial contribution are zero.

For an object containing multiple terms:

```text
U_RESD = sum_t(U_t)
```

A conventional harmonic distance restraint is the special case:

```text
one pair
coefficient = 1
IVAL = 1
EVAL = 2
RVAL = target_distance
condition = NONE
```

which gives:

```text
U = SCALE * (KVAL / 2) * (r - RVAL)^2
```

The factor `KVAL / EVAL` is part of the RESD definition; do not replace it with
an unnormalized power expression.

## Parameters and Conditions

### Pair coefficients

Each coefficient multiplies one powered pair distance before the pair
contributions are summed.

Nonzero positive, negative, and fractional coefficients are supported. Negative
coefficients permit reaction coordinates such as:

```text
D = r01 - r12 - RVAL
```

Repeating the same pair within a term is supported, and its coefficients
accumulate algebraically.

Every coefficient must be nonzero. The current apoCHARMM API also requires
coefficients to be finite.

### KVAL

`KVAL` is the term force constant. It appears in both the energy and its
derivative through the normalized factor `KVAL / EVAL`.

Positive and negative nonzero values are supported. Zero is rejected. The
current apoCHARMM API also requires `KVAL` to be finite.

### RVAL

`RVAL` is the reference value subtracted from the coefficient-weighted reaction
coordinate.

Negative, zero, positive, fractional, and near-boundary floating-point values
all behave as they do in CHARMM. The apoCHARMM API also requires `RVAL` to be
finite.

CHARMM reads an omitted `RVAL` as zero. apoCHARMM has no omitted form: every
term-creation API takes `RVAL` explicitly.

### IVAL

`IVAL` is the integer exponent applied independently to every pair distance in
one term:

```text
r_j^IVAL
```

The supported typed values are:

```text
-2, -1, 0, 1, 2, 6, 7
```

CHARMM stores `IVAL=1` when the keyword is omitted. The apoCHARMM C++, C, and
Python term-creation APIs require an explicit distance exponent and do not
expose an omitted `IVAL` argument.

CHARMM accepts any integer `IVAL`. apoCHARMM accepts only the values listed
above, whose CHARMM behavior has been verified, and rejects other values instead
of extrapolating.

### EVAL

`EVAL` is the integer exponent applied to `D`. It also supplies the denominator
in `KVAL / EVAL`.

The supported typed values are:

```text
1, 2, 3, 4
```

Zero and negative values are rejected, as they are in CHARMM. CHARMM uses
`EVAL=2` when the keyword is omitted; every apoCHARMM term-creation API requires
an explicit value instead.

For negative `D`, ordinary signed integer powers are used. Odd and even `EVAL`
values therefore retain their ordinary signed-power behavior.

### SCALE

`SCALE` is one global multiplicative factor applied to every stored term's
energy, gradient, and internal virial.

Positive, zero, and negative finite values are supported. As in CHARMM, the
initial value is:

```text
SCALE = 1
```

Appending another term preserves the current scale. `reset()` restores the scale
to one.

A scale of zero produces zero contribution and may return before pair geometry
is evaluated.

The typed API stores `setScale()` as object state. It does not reproduce a quirk
of CHARMM's parser, which resets the scale to one whenever it reads a
`RESDistance` command while no terms exist, so a standalone scale command issued
before the first definition is discarded by that definition.

### NONE

`DistanceRestraintCondition.NONE` applies no one-sided activation gate. The term
is always active.

### POSITIVE

`DistanceRestraintCondition.POSITIVE` is active exactly when:

```text
D >= 0
```

It is inactive when `D < 0`.

### NEGATIVE

`DistanceRestraintCondition.NEGATIVE` is active exactly when:

```text
D <= 0
```

It is inactive when `D > 0`.

### reset

`reset()` removes every stored term and restores `SCALE` to one. A new term may
be added after reset; terms that existed before reset remain absent.

`reset()` changes restraint configuration. It does not clear force, energy, or
virial output left by an earlier calculation.

## Construction and Defaults

A newly constructed apoCHARMM object has:

- no restraint terms
- global `SCALE=1`
- a fixed positive atom count supplied to the constructor
- the default force-manager tag `"resd"` in the Python API

CHARMM's `RESDistance` parser fills in omitted keywords with `IVAL=1`, `EVAL=2`,
and `RVAL=0`. An omitted `KVAL` is read as zero, and CHARMM ignores a definition
whose force constant is zero; every pair needs an explicit nonzero coefficient.
apoCHARMM does not reproduce these parser defaults: its term-creation APIs
require explicit coefficients, `KVAL`, `RVAL`, `IVAL`, and `EVAL`.

The C++ `addRestraint()` condition argument defaults to
`DistanceRestraintCondition::NONE`. The C ABI and Python API require an explicit
condition value.

## Atom References, Pairs, and Indexing

`DistanceRestraintForce` remains atom-count based:

```cpp
DistanceRestraintForce(numAtoms)
```

It does not retain or derive its atom count from a `CharmmPSF`.

All public C++, C, and Python term-creation interfaces require topology-aware
`AtomReference` endpoints. There is no public distance-restraint interface that
accepts raw integer endpoint pairs.

The native C++ method accepts:

```cpp
std::vector<std::array<AtomReference, 2>>
```

The C ABI accepts parallel arrays of borrowed `apo_atom_reference` handles. The
Python method accepts sequences containing two live `AtomReference` wrappers per
pair and passes their borrowed handles through that typed C ABI. Python does not
pre-extract raw endpoint indices for restraint registration.

For every endpoint, the native force extracts `getAtomIndex()` and stores only
that zero-based integer in its existing flattened index storage. It retains no
`AtomReference`, public C handle, Python wrapper, or source `CharmmPSF` after
`addRestraint()` returns.

Topology provenance is deliberately ignored. The force does not call
`hasSameTopology()` and does not require the endpoints of a pair to originate
from the same native PSF object.

Consequently:

* topology A index zero paired with topology B index one is accepted when both
  indices are valid for the force;
* topology A index zero paired with topology B index zero is rejected because
  both endpoints map to force-array element zero;
* a reference valid for its source topology is rejected when its extracted index
  is outside the force-local `[0, numAtoms)` range.

Pair definitions and coefficients remain separate parallel collections and must
have matching lengths. Pair order and coefficient correspondence are preserved.
Each term must contain at least one pair. Repeated pairs remain supported.

Both extracted indices in one pair must be distinct. apoCHARMM rejects identical
indices deliberately: in CHARMM, a pair that names the same atom twice gives a
finite energy but nonfinite gradients and virials, not a usable restraint.

## Periodic Coordinates and Images

RESD uses the raw primary-coordinate displacement and distance. It does not
replace a pair displacement with a minimum-image displacement.

This matches CHARMM for:

- nonperiodic calculations
- orthorhombic P1 boxes with 90-degree angles
- BYATOM image activation
- BYRESIDUE image activation

In those cases, image activation does not alter the pair distance used by RESD.
The internal virial uses the same raw displacement, and no image-shift
correction is added to make it minimum-image invariant. The external RESD virial
is zero.

CHARMM's behavior has not been verified for other periodic configurations, so
CHARMM-compatible RESD behavior is not supported for:

- P21
- triclinic cells
- monoclinic cells
- nonorthorhombic cells
- non-P1 cells
- other image modes or periodic configurations

The object stores force-manager box dimensions only for lifecycle compatibility;
they never cause a pair displacement to be replaced by its minimum image.

## Exact Zero-Distance Behavior

apoCHARMM rejects every evaluated pair whose exact distance is zero. Detection
occurs before negative-power evaluation or gradient formation.

This is deliberately stricter than CHARMM, whose zero-distance result depends on
`IVAL`:

| IVAL | CHARMM result at zero distance |
|---:|---|
| `-2` | Infinite deviation and energy; NaN Cartesian values and virial |
| `-1` | Infinite deviation and energy; NaN Cartesian values and virial |
| `0` | Finite energy; NaN Cartesian values and virial |
| `1` | Finite energy; NaN Cartesian values and virial |
| `2` | Finite energy; exact zero Cartesian values and virial |
| `6` | Finite energy; exact zero Cartesian values and virial |
| `7` | Finite energy; exact zero Cartesian values and virial |

Rather than reproduce these exponent-dependent branches, several of which
produce infinities or NaNs, apoCHARMM rejects every evaluated zero-distance
pair.

When `SCALE=0`, evaluation may return zero before reading pair geometry, so an
otherwise singular pair need not be evaluated.

## Exact One-Sided Boundaries

Neither selector has a tolerance: like CHARMM, apoCHARMM compares `D` with zero
exactly.

`POSITIVE` includes equality at `D=0`. `NEGATIVE` also includes equality at
`D=0`.

With `EVAL=1`, the energy at `D=0` is zero, but a coordinate-dependent active
term can have a nonzero gradient and internal virial at that boundary. The first
derivative is therefore discontinuous when crossing between the inactive and
active sides.

Deviations at the limit of double precision are still classified by their exact
sign, for example:

```text
D = +2.220446049250313e-16
D = -4.440892098500626e-16
```

Do not add an epsilon or tolerance to either selector.

## clear Versus reset

The native C++ `clear()` operation clears accumulated force, energy, and virial
output. It does not remove restraint terms and does not restore the scale.

`reset()` removes all restraint terms and restores the scale to one. It does not
substitute for output clearing.

The C ABI and Python wrapper expose `reset()` for configuration replacement.
Normal C and Python calculations rely on `ForceManager` and `CharmmContext` to
drive output clearing and force evaluation.

## Numerical Scope and Limitations

The typed API accepts only the exponents whose CHARMM behavior has been
verified:

```text
IVAL in {-2, -1, 0, 1, 2, 6, 7}
EVAL in {1, 2, 3, 4}
```

CHARMM accepts any integer `IVAL` and any positive `EVAL`, so these sets are a
deliberate subset of CHARMM's range.

CHARMM's parser truncates a non-integer exponent such as `1.5` to `1` and prints
only a warning. The typed apoCHARMM API takes integer exponents directly and
rejects unsupported values instead of converting them.

At finite nonzero distance, `IVAL=0` gives a coordinate-independent `r^0`
contribution and zero pair gradient.

CHARMM gives finite results for `IVAL=6` and `IVAL=7` at a pair distance of two,
but overflow and underflow limits at other distances are not specified.

This page does not specify:

- numerical overflow and underflow thresholds
- magnitude limits for coefficients, `KVAL`, `RVAL`, and `SCALE`
- CHARMM's handling of nonfinite input
- CHARMM's maximum term and pair counts (apoCHARMM's own limit is the `int`
  index range)

The apoCHARMM API requires finite coefficients, `KVAL`, `RVAL`, and `SCALE`.
These checks are apoCHARMM safety policies, not statements about CHARMM's own
limits.

## Python Use and Lifetime

Construct the force with the same atom count as its `ForceManager`:

```python
restraint = apo.DistanceRestraintForce(psf.getNumAtoms())
```

Create exact-one atom references through `AtomSelector.selectAtom()`:

```python
selector = apo.AtomSelector(psf)
first = selector.selectAtom("atom SEG1 1 CA")
second = selector.selectAtom("atom SEG1 1 CB")
```

Pass the references, not their integer indices, to `addRestraint()`:

```python
restraint.addRestraint(
    [[first, second]],
    [1.0],
    2.0,
    3.0,
    1,
    2,
    apo.DistanceRestraintCondition.NONE,
)
```

The Python wrapper places the two endpoint-handle sequences and coefficients in
temporary arrays. The C ABI borrows the handles and copies temporary native
`AtomReference` values; the native force then extracts and stores only their
zero-based indices. After successful return, no Python wrapper, public C handle,
native `AtomReference`, or source topology is retained. Closing or destroying
the references does not invalidate the stored restraint.

Set the global scale with `setScale()` and subscribe through the manager:

```python
force_manager.subscribe(restraint)
```

Omitting the tag uses the default tag:

```text
resd
```

The manager retains the restraint after successful subscription. Unsubscribe
before closing the restraint:

```python
force_manager.unsubscribe(restraint)
restraint.close()
```

Closing a still-subscribed Python wrapper destroys its identifying C handle but
does not by itself remove the manager's native subscription.

The wrapper and native object provide no internal host-thread synchronization.
Serialize configuration, subscription changes, calculation, reset,
unsubscription, and closure.

A complete runnable example is provided in:

```text
example/distance_restraint.py
```

## Cross-Language Interfaces

The native C++ interface is declared in:

```text
include/DistanceRestraintForce.h
```

The C ABI is declared in:

```text
capi/include/apocharmm_c/DistanceRestraintForce.h
```

The AtomReference-only term-creation entry point is:

```text
apo_distance_restraint_force_add_restraint
```

The Python wrapper and condition enumeration are implemented in:

```text
python/apocharmm/distance_restraint_force.py
python/apocharmm/enums.py
```

## Related Subsystems

- [ForceManager](@ref force_manager) owns subscribed force resources and
  aggregates their outputs.
- [CharmmContext](@ref charmm_context) owns molecular state and drives manager
  initialization and calculations.
- [CUDA Integrators](@ref cuda_integrators) propagate a context after the
  restraint is subscribed.
- [ApoCharmmError](@ref apocharmm_error) describes cross-language validation and
  error propagation.
- [HarmonicRestraintForce](@ref harmonic_restraint_force) restrains absolute
  atom positions rather than a RESD distance reaction coordinate.
- [HarmonicCenterOfMassRestraintForce](@ref harmonic_center_of_mass_restraint_force)
  restrains a selected group center.
