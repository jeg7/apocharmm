# GeometricRestraintForce {#geometric_restraint_force}

## Purpose

`GeometricRestraintForce` provides native C++ harmonic restraints relative to
planes, infinite cylinders, and spheres. One object can combine geometries,
activation modes, and independent-atom or center-of-mass selections. Overlapping
selections and repeated terms contribute additively.

For position `r`, reference `c`, and normalized direction `n`, define:

```text
u = r - c
s = dot(n, u)
p = u - dot(n, u) * n
```

The geometry determines the reaction coordinate `q` and its gradient:

| Geometry and mode | Reaction coordinate | Coordinate gradient |
|---|---|---|
| Plane, SYMMETRIC | `q = s` | `n` |
| Plane, INSIDE or OUTSIDE | `q = abs(s)` | `sign(s) * n`, with `+n` at zero |
| Cylinder | `q = norm(p)` | `p / q` |
| Sphere | `q = norm(u)` | `u / q` |

For an active evaluation with force constant `K` and offset `D`:

```text
Delta = q - D
U_mmfp = (K / 2) * Delta^2
gradient = K * Delta * grad(q)
physical_force = -gradient
```

`INSIDE` is active when `q > D`, `OUTSIDE` when `q < D`, and `SYMMETRIC`
has no one-sided gate. Equality contributes zero. Positive, zero, and negative
finite force constants and offsets are accepted. A one-sided plane uses its
absolute slab distance, rather than a signed half-space coordinate.

Cylinder and sphere evaluations are suppressed completely when their computed
squared radial distance is at or below `1e-10` square angstroms. This check
precedes activation and includes the exact cylinder axis or sphere center.
For other active evaluations, the arithmetic computes
`(0.5 * K) * (Delta * Delta)`. A computed zero energy, including underflow to
zero, suppresses the gradient even when energy output is disabled.

### Atomwise and RCM application

`SelectionMode::ATOMWISE` evaluates every selected atom independently and
ignores masses. `SelectionMode::RCM` evaluates once at the signed mass-weighted
center, then distributes the center gradient:

```text
M = sum over selected atoms i of m_i
R = (sum over selected atoms i of m_i * r_i) / M
gradient_i = (m_i / M) * gradient_R
```

One force-wide mass vector serves every RCM term. Finite positive, zero, and
negative individual masses are accepted. Every nonempty RCM selection requires
a finite nonzero signed total and finite mass fractions. Negative totals and
tiny nonzero totals are accepted within that arithmetic envelope. A zero-mass
member receives no gradient; a negative member receives a signed share.

### Coordinates and virial

The supported manager configuration is orthorhombic `PBC::P1`. Evaluations use
the supplied raw Cartesian coordinates: selected atoms are not unwrapped, and
atom-to-reference or center-to-reference displacements are not minimum-imaged.
Changing only the stored box lengths leaves the geometric contribution
unchanged. Other force-field contributions can still change with the box.

The complete external virial tensor is:

```text
W_ab = sum over atoms i of r_i,a * gradient_i,b
```

It uses raw absolute positions and the double-precision gradient accumulated at
each atom before conversion to fixed point. RCM terms use each atom's
distributed gradient in this sum. Components are stored in
`XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ` order. The tensor need not be symmetric or
translation invariant. This is the external coordinate-gradient tensor; it does
not define CHARMM's internal pressure accumulator.

## Quick Start

This native C++ example uses repository fixtures and assumes execution from the
repository root. It restrains the mass-weighted X coordinate of a sodium/chloride
pair toward one angstrom. The constructor obtains the masses from the PSF.

```cpp
#include "AtomSelection.h"
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "ForceManager.h"
#include "GeometricRestraintForce.h"
#include "PBC.h"

#include <memory>

int main() {
  using GeoForce = GeometricRestraintForce<long long int, double>;

  auto parameters =
      std::make_shared<CharmmParameters>("toppar/toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>("test/data/nacl_pair.psf");
  auto coordinates = std::make_shared<CharmmCrd>("test/data/nacl_pair.cor");

  auto forceManager = std::make_shared<ForceManager>(psf, parameters);
  forceManager->setPeriodicBoundaryCondition(PBC::P1);
  forceManager->setBoxDimensions({50.0, 50.0, 50.0});

  auto context = std::make_shared<CharmmContext>(forceManager);
  context->setCoordinates(coordinates);
  context->useHolonomicConstraints(false);

  AtomSelection selection(psf->getNumAtoms(), AtomSelection::InitialValue::ALL);
  auto restraint = std::make_shared<GeoForce>(psf);
  restraint->addPlane(selection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                      2.0, 1.0, GeoForce::Activation::SYMMETRIC,
                      GeoForce::SelectionMode::RCM);

  forceManager->subscribe(restraint, "mmfp", restraint->getStream(),
                          restraint->getForce(), restraint->getEnergyVirial());
  context->calculatePotentialEnergy(false, true);
  forceManager->unsubscribe(restraint);

  return 0;
}
```

The context initializes the fully configured manager. Subscription then
initializes the restraint with the manager's atom count and box. Normal manager
evaluation clears the requested outputs and includes the restraint's energy,
gradient, and tensor. The printed energy table includes the ordinary force field
as well as MMFP; it is not an isolated restraint-energy report.

## Construction and Required State

Construction requires a non-null PSF with a positive atom count and a complete
finite mass vector. The atom count is fixed for the object's lifetime. The
constructor copies masses, creates the `mmfp` energy term and private CUDA
resources, and clears force, energy, and virial output. Initially there are no
terms, and stored box lengths are zero.

Each geometry accepts an `AtomSelection` with the same represented atom count
or an `AtomReference` with an in-range zero-based index. Only extracted indices
are retained. A valid empty selection is a no-op after parameter validation.
Term defaults are offset zero, `Activation::INSIDE`, and
`SelectionMode::ATOMWISE`; specify other behavior explicitly.

A force may be subscribed before or after manager initialization. The manager
supplies a matching atom count and three finite positive box lengths through
`initialize()`. That hook synchronizes dirty term data; it does not replace the
output owners or clear outputs. A direct calculation uses raw coordinates and
does not require stored box lengths for its geometry.

The manager accepts this force only under `PBC::P1`. Its
`ForceManager::setPeriodicBoundaryCondition()` validates compatibility before
changing its state, so a rejected PBC mutation preserves the manager's previous
configuration. `PBC::UNSET` is an invalid sentinel, not an aperiodic mode.

## Ownership and Lifetime

The force owns copied term parameters, selected indices, active masses, working
storage, and a private CUDA stream. It retains neither the source PSF nor the
selection or atom-reference objects. Changing the source PSF does not update the
force's mass snapshot; use `setMasses()` to replace it.

ForceManager retains shared ownership of a subscribed force and its supplied
resources. Pass the stream, force array, and energy-virial object belonging to
that same force. Unsubscription releases the manager's owners without resetting
the force, which can subsequently be reused or resubscribed.

`getForce()` and `getEnergyVirial()` return shared owners that can preserve
output allocations after force destruction. `getStream()` shares only the host
holder of the CUDA handle: the force destructor destroys the stream and sets the
holder to null. Retaining that holder does not extend the stream's lifetime.
Copy and move construction and assignment are deleted.

## Data, Shapes, and AKMA Units

For a force with `N = getNumAtoms()` atoms, the interface uses these
AKMA-compatible quantities:

| Quantity | Native representation | Meaning and units |
|---|---|---|
| Selection | `AtomSelection` representing `N` atoms, or `AtomReference` | Zero-based atom membership |
| Origin or center | `std::array<double, 3>` | Finite `[x, y, z]`, angstroms |
| Plane normal or cylinder axis | `std::array<double, 3>` | Finite dimensionless direction, normalized on addition |
| Force constant | `double` | Signed `K`, kilocalories per mole per square angstrom |
| Offset | `double` | Signed `D`, angstroms |
| Active masses | `std::vector<double>` of length `N` | Finite signed masses, normally atomic mass units |
| Box | `std::vector<double>` of length 3 | Positive finite `[x, y, z]` lengths, angstroms |
| Coordinates | Borrowed device `float4[N]` | Angstroms in `x`, `y`, `z`; charge `w` ignored |
| Force output | `Force<long long int>` | Fixed-point potential gradient, scale `2^40` |
| Energy | `CudaEnergyVirial` term `mmfp` | Kilocalories per mole |
| Virial | Nine components in `Virial_t::virmat` | Kilocalories per mole |

Force components use structure-of-arrays storage at offsets zero, `stride()`,
and `2 * stride()`. Divide each stored integer by `2^40` to recover the gradient
in kilocalories per mole per angstrom; negate it to obtain physical force.
Getters perform no implicit device-to-host transfer or synchronization.

## Errors

Native failures use [ApoCharmmError](@ref ApoCharmmError). `InvalidArgument`
covers invalid shapes, indices, enums, finite-value requirements, box lengths,
unsupported PBC, and invalid nonempty RCM mass totals or fractions.
Construction also rejects atom counts beyond the existing force-storage
representation limit. An uninitialized PSF is `NotInitialized`; an inconsistent
PSF mass count is `Runtime`.

Evaluation raises `Runtime` for nonfinite arithmetic, unsupported internal term
state, an unrepresentable converted gradient, or fixed-point addition overflow.
CUDA allocation, transfer, launch, and synchronization failures use `Cuda`.
Host allocation or container limits can raise `std::bad_alloc` or
`std::length_error`. Detailed method-specific conditions are in the native API
reference.

Term addition and mass replacement validate proposals before committing their
logical state. Numeric rejection within `calcForce()` preserves that call's
pre-existing public output. This does not undo clearing performed earlier by
the caller or manager. CUDA failures do not provide an output rollback guarantee.

## Important Behavior and Limitations

### Mutation, reset, and clear

Additions, mass replacement, and reset remain available after subscription or
initialization. They do not require resubscription or replace output owners.

| Operation | Result |
|---|---|
| `addPlane`, `addCylinder`, `addSphere` | Append copied terms; update device mirrors before use |
| `setMasses` | Validate every installed RCM selection and replace the force-wide mass vector |
| `reset` | Remove all terms; preserve masses, box, resources, and current output |
| `clear()` | Clear force, energy, and tensor output; retain configuration |
| `clear(calcEnergy, calcVirial)` | Clear forces and only the requested optional outputs |
| `calcForce` | Accumulate gradients and requested energy/tensor; preserve disabled outputs |

Preserved energy or tensor values can be stale for new coordinates. After reset,
new terms may be added immediately. A calculation with no terms contributes
nothing and does not clear, validate the coordinate pointer, or synchronize.

### Numerical and CUDA limits

Geometry, RCM arithmetic, and ordered accumulation use double precision. `CT`
selects only the final Cartesian-gradient conversion; both supported
specializations consume `float4` coordinates. Choosing `CT = double` cannot
recover coordinate precision lost at input, especially with nearly cancelling
signed RCM masses.

Before fixed-point conversion, gradient components must lie in
`[-8388608, 8388608)`. Conversion through `CT` and addition to existing output
must also remain representable. Finite input values alone do not guarantee
finite intermediate products, centers, energies, or tensors. Directions require
a positive finite `std::hypot` norm; this typed envelope does not reproduce
every CHARMM text-parser underflow or overflow path.

Serialize host configuration, calculation, clearing, subscription operations,
and destruction. Mutate between manager evaluations after outstanding users of
the outputs finish. Keep borrowed device coordinates and the CUDA context alive
for the queued work. `calcForce()` synchronizes to validate staged results, then
enqueues a final commit that may still be running on return. Direct consumers
must order a stream synchronization or equivalent dependency before using that
output. CUDA graph capture is unsupported. Release owned resources before
resetting their CUDA device.

## Related Subsystems

- [AtomSelection](@ref atom_selection) describes typed atom membership.
- [AtomReference](@ref AtomReference) provides a single force-local atom index.
- [CharmmPSF](@ref charmm_psf) supplies the initial atom count and masses.
- [ForceManager](@ref force_manager) owns subscriptions and aggregates outputs.
- [CharmmContext](@ref charmm_context) supplies coordinates and drives
  evaluation.
- [HarmonicRestraintForce](@ref harmonic_restraint_force) restrains individual
  Cartesian references.
- [HarmonicCenterOfMassRestraintForce](@ref harmonic_center_of_mass_restraint_force)
  provides a periodically imaged group-center restraint.
- [ApoCharmmError](@ref apocharmm_error) describes native error categories.

## Developer Architecture

### Public interface and storage

The native contract is in `include/GeometricRestraintForce.h`; implementation
and the `long long int`/`float` and `long long int`/`double` instantiations are
in `src/GeometricRestraintForce.cu`. Terms use flattened membership ranges and
parallel geometry, activation, selection-mode, origin, direction, offset, and
coefficient arrays. An inverse membership list preserves term insertion order
for each atom. RCM reduction workspace grows with the selected memberships.
Block/chunk sizes do not impose a fixed term or selection capacity.

### Evaluation and aggregation

Dirty host term mirrors are synchronized before use. RCM work first reduces
mass-weighted coordinates. Geometry evaluation then writes private membership
gradients and term energies. Output staging sums gradients per atom, checks
conversion and addition, and constructs the tensor from the new double-gradient
sum. A checked host synchronization precedes commitment to public outputs.

`contributesVirial` and `computesVirial` are both true. ForceView dispatches
selective clearing, P1 compatibility, initialization, box propagation, and
calculation. ForceManager aggregates the supplied tensor once and skips its
ordinary coordinate-gradient reconstruction for this force.

### Tests and extension constraints

`test/unittests/unittest-geometricRestraintForce.cpp` covers native geometry,
activation, signed masses, output controls, mutation, manager ownership and
lifecycle, accumulation, dynamic capacity, and stable derivative checks.
`test/unittests/unittest-forceManager.cpp` covers the shared manager machinery.
The public potential enum contains only `HARMONIC`. Changes to representations,
imaging, ordering, or tensor handling require corresponding numerical and
lifecycle review; they are not additional modes implied by the current API.

## API Reference

- Native C++: [GeometricRestraintForce](@ref GeometricRestraintForce).
- Typed inputs: [AtomSelection](@ref AtomSelection), [AtomReference](@ref AtomReference), and [CharmmPSF](@ref CharmmPSF).
- Integration: [ForceManager](@ref ForceManager), [ForceView](@ref ForceView), and [PBC](@ref PBC).
