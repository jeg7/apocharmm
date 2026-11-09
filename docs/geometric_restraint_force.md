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

## C ABI

`capi/include/apocharmm_c/GeometricRestraintForce.h` exposes construction, term
addition, force-wide mass replacement, reset, and force-manager subscription
for the `GeometricRestraintForce<long long int, float>` specialization. It does
not expose direct calculation or output access; a subscribed manager evaluates
the restraint.

### Ownership, borrowing, and lifetime

[apo_geometric_restraint_force_create](@ref apo_geometric_restraint_force_create)
borrows an `apo_charmm_psf` handle only while the native constructor copies the
atom count and masses. The returned handle shares ownership of the native
restraint and retains no PSF owner, so the PSF handle may be destroyed, and its
native PSF mutated, without affecting the restraint. Creation stores a non-NULL
handle only on success; with a non-NULL `out`, every failure leaves
`*out == NULL`. Construction and mass replacement use synchronous device
transfers, so they can wait for other work on the device.
[apo_geometric_restraint_force_destroy](@ref apo_geometric_restraint_force_destroy)
accepts `NULL` and releases the handle's native owner. Without a subscribing
manager, this also releases the native restraint, its CUDA stream, and its
outputs.

Each geometry has one entry point for a borrowed `apo_atom_selection` and one
for a borrowed `apo_atom_reference`. No entry point accepts raw atom indices.

| Geometry | AtomSelection input | AtomReference input |
|---|---|---|
| Plane | [apo_geometric_restraint_force_add_plane_from_selection](@ref apo_geometric_restraint_force_add_plane_from_selection) | [apo_geometric_restraint_force_add_plane_from_atom_reference](@ref apo_geometric_restraint_force_add_plane_from_atom_reference) |
| Cylinder | [apo_geometric_restraint_force_add_cylinder_from_selection](@ref apo_geometric_restraint_force_add_cylinder_from_selection) | [apo_geometric_restraint_force_add_cylinder_from_atom_reference](@ref apo_geometric_restraint_force_add_cylinder_from_atom_reference) |
| Sphere | [apo_geometric_restraint_force_add_sphere_from_selection](@ref apo_geometric_restraint_force_add_sphere_from_selection) | [apo_geometric_restraint_force_add_sphere_from_atom_reference](@ref apo_geometric_restraint_force_add_sphere_from_atom_reference) |

Native code copies the selected or referenced indices and every parameter, so
atom handles and caller arrays may be released as soon as a call returns. An
atom reference contributes only its index, which must lie in the restraint's
atom range; its topology is neither compared nor retained.

Origins, centers, and directions are passed as a `const double *` with a length
that must equal three, and masses as a `const double *` with a length that must
equal `N`. Force constants and offsets are `double`. Activation and selection
mode use `apo_geometric_restraint_activation` and
`apo_geometric_restraint_selection_mode`, whose explicit values equal the native
enumerators; undeclared values are rejected before native conversion. C has no
default arguments: pass offset zero,
`APO_GEOMETRIC_RESTRAINT_ACTIVATION_INSIDE`, and
`APO_GEOMETRIC_RESTRAINT_SELECTION_MODE_ATOMWISE` for the native defaults.

[apo_geometric_restraint_force_set_masses](@ref apo_geometric_restraint_force_set_masses)
copies the caller array into the native transactional replacement; a rejected
replacement leaves the previous masses in effect.
[apo_geometric_restraint_force_reset](@ref apo_geometric_restraint_force_reset)
removes every term while preserving the active masses and existing output.

### Force-manager subscription

[apo_force_manager_subscribe_geometric_restraint_force](@ref apo_force_manager_subscribe_geometric_restraint_force)
copies a nonempty tag and makes the manager retain shared owners of the native
restraint, its stream holder, its force array, and its energy-virial object.
Neither C handle is retained. Subscription is accepted before or after manager
initialization; an initialized manager immediately initializes the restraint
with its atom count and box. The manager must use `APO_PBC_P1`; while the
restraint is subscribed, changing the manager to `APO_PBC_P21` is rejected and
leaves the manager unchanged. The same restraint cannot be subscribed twice to
one manager, although tags need not be unique.
[apo_force_manager_unsubscribe_geometric_restraint_force](@ref apo_force_manager_unsubscribe_geometric_restraint_force)
removes the subscription by restraint identity without resetting the restraint,
which can then be subscribed again.

Term addition, mass replacement, and reset remain available while subscribed
and take effect at the next manager evaluation. Destroying the restraint handle
does not unsubscribe it: the manager continues to evaluate the retained native
restraint, and the C ABI can no longer remove it, until the manager is
destroyed. Unsubscribe first when later explicit removal is needed.

### Statuses and thread safety

Every status-returning function clears the calling thread's diagnostic at entry
and leaves it empty on success. `APO_STATUS_INVALID_ARGUMENT` covers invalid
handles, pointers, lengths, and enum values and every native argument
rejection. `APO_STATUS_NOT_INITIALIZED` reports a PSF with a negative atom
count. `APO_STATUS_CUDA_ERROR` reports CUDA failures during construction, mass
replacement, or immediate initialization on subscription; term addition, reset,
and unsubscription perform no CUDA work.
`APO_STATUS_RUNTIME_ERROR` covers an inconsistent PSF mass count, host
allocation failures, and other C++ exceptions. Read failure text with
[apo_last_error](@ref apo_last_error) on the same thread.

The handle and native object provide no internal synchronization. Serialize
every call that uses one restraint handle, including subscription,
unsubscription, destruction, and manager evaluations of the restraint, and do
not destroy or mutate a borrowed PSF, selection, or reference during a call.

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

The C ABI is declared in `capi/include/apocharmm_c/GeometricRestraintForce.h`,
and its activation and selection-mode enumerations are declared in
`capi/include/apocharmm_c/Enums.h`. The private shared-owner handle is defined
in `capi/private_include/apocharmm_c/detail/GeometricRestraintForceHandle.h`,
and the enumeration conversions are in
`capi/private_include/apocharmm_c/detail/EnumConversion.h`, and guarded entry
points are implemented in `capi/src/apocharmm_c/GeometricRestraintForce.cpp`.

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
`test/unittests/unittest-capiGeometricRestraintForce.cpp` covers C ABI
construction, ownership, borrowing, typed atom inputs, enumeration and array
validation, status mapping, mass replacement, reset, manager subscription, tags,
P1 enforcement, ownership and lifecycle, and a CHARMM c52a1 oracle subset
evaluated through `ForceManager` in a zero-interaction twelve-atom system.
The public potential enum contains only `HARMONIC`. Changes to representations,
imaging, ordering, or tensor handling require corresponding numerical and
lifecycle review; they are not additional modes implied by the current API.

## API Reference

- Native C++: [GeometricRestraintForce](@ref GeometricRestraintForce).
- Typed inputs: [AtomSelection](@ref AtomSelection), [AtomReference](@ref AtomReference), and [CharmmPSF](@ref CharmmPSF).
- Integration: [ForceManager](@ref ForceManager), [ForceView](@ref ForceView), and [PBC](@ref PBC).
- C ABI: [apo_geometric_restraint_force](@ref apo_geometric_restraint_force), [apo_geometric_restraint_activation](@ref apo_geometric_restraint_activation), [apo_geometric_restraint_selection_mode](@ref apo_geometric_restraint_selection_mode), [apo_geometric_restraint_force_create](@ref apo_geometric_restraint_force_create), [apo_geometric_restraint_force_destroy](@ref apo_geometric_restraint_force_destroy), [apo_geometric_restraint_force_set_masses](@ref apo_geometric_restraint_force_set_masses), [apo_geometric_restraint_force_reset](@ref apo_geometric_restraint_force_reset), [apo_force_manager_subscribe_geometric_restraint_force](@ref apo_force_manager_subscribe_geometric_restraint_force), and [apo_force_manager_unsubscribe_geometric_restraint_force](@ref apo_force_manager_unsubscribe_geometric_restraint_force); the six term-addition functions are linked in the C ABI section.
