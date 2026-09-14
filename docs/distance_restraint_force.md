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

The characterized behavior includes negative, zero, positive, fractional, and
near-boundary floating-point values. The current apoCHARMM API requires `RVAL`
to be finite.

No omitted-value default for `RVAL` has been established.

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

In the characterized CHARMM syntax, omitting `IVAL` stores `IVAL=1`. The
apoCHARMM C++, C, and Python term-creation APIs require an explicit distance
exponent and do not expose an omitted `IVAL` argument.

The complete CHARMM integer range is unresolved. Values outside the listed
characterized set are rejected by the current typed API rather than inferred.

### EVAL

`EVAL` is the integer exponent applied to `D`. It also supplies the denominator
in `KVAL / EVAL`.

The supported typed values are:

```text
1, 2, 3, 4
```

Zero and negative values are rejected. No omitted-value default for `EVAL` has
been established, so every apoCHARMM term-creation API requires it explicitly.

For negative `D`, ordinary signed integer powers are used. Odd and even `EVAL`
values therefore retain their ordinary signed-power behavior.

### SCALE

`SCALE` is one global multiplicative factor applied to every stored term's
energy, gradient, and internal virial.

Positive, zero, and negative finite values are supported. Its confirmed initial
value is:

```text
SCALE = 1
```

Appending another term preserves the current scale. `reset()` restores the scale
to one.

A scale of zero produces zero contribution and may return before pair geometry
is evaluated.

The typed API stores `setScale()` as object state. It does not reproduce the
separate CHARMM text-parser command form in which a standaline scale command
issued before a subsequent definition was discarded by that definition.

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

## Construction and Confirmed Defaults

A newly constructed apoCHARMM object has:

- no restraint terms
- global `SCALE=1`
- a fixed positive atom count supplied to the constructor
- the default force-manager tag `"resd"` in the Python API

The confirmed CHARMM omission default for `IVAL` is one in the tested definition
syntax.

The following omission defaults remain unresolved and must not be inferred:

- `EVAL`
- `KVAL`
- `RVAL`
- pair coefficients

The C++ `addRestraint()` condition argument defaults to
`DistanceRestraintCondition::NONE`. The C ABI and Python API require an explicit
condition value.

## Atom Pairs and Indexing

Atom indices are zero-based in the C++ and Python APIs and in the arrays passed
through the C ABI.

Each term must contain at least one pair. The number of pair definitions must
match the number of coefficients.

Both indices in a pair must be in range and must be distinct. Rejecting
identical indices is an intentional apoCHARMM safety policy. The characterized
CHARMM identical-index case produced finite energy but nonfinite gradients and
virials rather than a usable restraint.

## Periodic Coordinates and Images

RESD uses the raw primary-coordinate displacement and distance. It does not
replace a pair displacement with a minimum-image displacement.

This behavior is established for:

- nonperiodic calculations
- orthorhombic P1 boxes with 90-degree angles
- characterized BYATOM image activation
- characterized BYRESIDUE image activation

In those cases, image activation does not alter the pair distance used by RESD.
The internal virial uses the same raw displacement, and no image-shift
correction is added to make it minimum-image invariant. The characterized
external RESD virial is zero.

Compatibility is not claimed for an uncharacterized periodic configuration. The
following are unsupported for CHARMM-compatible RESD behavior:

- P21
- triclinic cells
- monoclinic cells
- nonorthorhombic cells
- non-P1 cells
- other uncharacterized image modes or periodic configurations

The current object stores force-manager box dimensions for lifecycle
compatibility, but those dimensions do not cause minimum-image replacement of
the characterized raw pair displacement.

## Exact Zero-Distance Behavior

The current apoCHARMM implementation rejects every evaluated pair whose exact
distance is zero. Detection occurs before negative-power evaluation or gradient
formation.

This is a deliberate safety narrowing of the characterized CHARMM behavior. The
supplied CHARMM testing showed:

| IVAL | Characterized exact-zero result |
|---:|---|
| `-2` | Infinite deviation and energy; NaN Cartesian values and virial |
| `-1` | Infinite deviation and energy; NaN Cartesian values and virial |
| `0` | Finite energy; NaN Cartesian values and virial |
| `1` | Finite energy; NaN Cartesian values and virial |
| `2` | Finite energy; exact zero Cartesian values and virial |
| `6` | Finite energy; exact zero Cartesian values and virial |
| `7` | Finite energy; exact zero Cartesian values and virial |

apoCHARMM does not reproduce these exponent-dependent exact-zero branches. It
rejects all evaluated exact-zero pairs instead.

When `SCALE=0`, evaluation may return zero before reading pair geometry, so an
otherwise singular pair need not be evaluated.

Exact-zero behavior for uncharacterized `IVAL` values remains unresolved.

## Exact One-Sided Boundaries

There is no nonzero selector tolerance in the characterized behavior.

`POSITIVE` includes equality at `D=0`. `NEGATIVE` also includes equality at
`D=0`.

With `EVAL=1`, the energy at `D=0` is zero, but a coordinate-dependent active
term can have a nonzero gradient and internal virial at that boundary. The first
derivative is therefore discontinuous when crossing between the inactive and
active sides.

The characterized adjacent binary64 deviations were distinguished by their exact
signs:

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

The public typed API deliberately restricts exponents to the characterized sets:

```text
IVAL in {-2, -1, 0, 1, 2, 6, 7}
EVAL in {1, 2, 3, 4}
```

This restriction must not be described as the complete accepted CHARMM range.

The CHARMM text parser converted the tested token `1.5` to stored integer `1`.
The typed apoCHARMM API does not silently reproduce that parser conversion;
exponents are integer API values, and unsupported values are rejected.

At finite nonzero distance, `IVAL=0` gives a coordinate-independent `r^0`
contribution and zero pair gradient.

`IVAL=6` and `IVAL=7` remained finite at distance two in the characterized
cases. That observation does not establish general overflow or underflow limits.

The following remain unresolved:

- accepted `IVAL` values outside the characterized set
- accepted positive `EVAL` values above four
- numerical overflow and underflow thresholds
- magnitude limits for coefficients, `KVAL`, `RVAL`, and `SCALE`
- external CHARMM behavior for nonfinite values
- maximum term and pair counts
- exact-zero behavior for uncharacterized `IVAL` values

The current apoCHARMM API requires finite coefficients, `KVAL`, `RVAL`, and
`SCALE`. These finite-value checks are API safety policies, not claims about
unrestricted CHARMM behavior.

## Python Use and Lifetime

Construct the force with the same atom count as its `ForceManager`:

```python
restraint = apo.DistanceRestraintForce(psf.getNumAtoms())
```

Add each term with `addRestraint()`, set the global scale with `setScale()`, and
subscribe through the manager:

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

The stable C ABI is declared in:

```text
capi/include/apocharmm_c/DistanceRestraintForce.h
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
- [ApoCharmError](@ref apocharmm_error) describes cross-language validation and
  error propagation.
- [HarmonicRestraintFroce](@ref harmonic_restraint_force) restrains absolute
  atom positions rather than a RESD distance reaction coordinate.
- [HarmonicCenterOfMassRestraintForce](@ref harmonic_center_of_mass_restraint_force)
  restrains a selected group center.
