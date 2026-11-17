# HarmonicCenterOfMassRestraintForce {#harmonic_center_of_mass_restraint_force}

## Purpose

`HarmonicCenterOfMassRestraintForce` reproduces CHARMM `CONS HMCM`
(`ecmcns` in `source/energy/ecnstr.F90`): a harmonic restraint on the
geometric or mass-weighted center of one nonempty atom selection. It is
appropriate when a group center should remain at a reference position or on a
shell at a reference distance from that position. Use
[HarmonicRestraintForce](@ref harmonic_restraint_force) instead when individual
atoms should remain near independent absolute Cartesian references.

Let `S` be the selected atoms. The weight of atom `i` is one when mass weighting
is disabled and its configured mass `m_i` when mass weighting is enabled. As in
CHARMM, the center is built from the raw coordinates, without unwrapping:

```text
W = sum over i in S of w_i
C = (1 / W) * sum over i in S of w_i * r_i
```

The three-element reference mask controls which Cartesian components
participate. Active components of the raw displacement `C - R` from the
reference position `R` are kept, and inactive components are set to zero. Let
`d` be that masked displacement, `s = |d|`, and `r0` the reference distance.
The implemented energy and the gradient of atom `i` are:

```text
U_hmcm = k * (s - r0)^2
gradient_i = w_i * (d * 2 * k / W * (s - r0) / s)
```

Both follow CHARMM's operation order, and there is no factor of one half. With
`r0 = 0`, the energy is `k * s^2` and the gradient is `w_i * 2 * k * d / W`. At
`s = 0`, the gradient is zero. CHARMM also gives zero there when `r0 = 0`, but
divides by zero otherwise.

As in CHARMM, the force constant, the reference distance, and the masses may
have either sign, provided the selected total weight is finite and nonzero.

apoCHARMM stores this potential gradient in the force array and subtracts it in
the integration path. The physical restoring force therefore has the opposite
sign. The restraint energy is accumulated under the tag `hmcm`. CHARMM evaluates
`CONS HMCM` after it forms its internal virial, so the restraint contributes no
virial.

## Quick Start

The Python API is the highest-level supported interface. This example uses
repository test data and assumes it is run from the repository root.

```python
import apocharmm as apo

parameters = apo.CharmmParameters("toppar/toppar_water_ions.str")
psf = apo.CharmmPsf("test/data/nacl_pair.psf")
coordinates = apo.CharmmCrd("test/data/nacl_pair.cor")

context = apo.CharmmContext(psf, parameters)
context.setBoxDimensions((50.0, 50.0, 50.0))
context.setCoordinates(coordinates)

selector = apo.AtomSelector(psf)
selection = selector.select("all")

restraint = apo.HarmonicCenterOfMassRestraintForce(psf)
restraint.setSelection(selection)
restraint.setReferencePosition((1.41, 1.41, 1.41))
restraint.setMassWeighting(True)
restraint.setForceConstant(1.0)

force_manager = context.getForceManager()
force_manager.subscribe(restraint)

context.calculatePotentialEnergy()

force_manager.unsubscribe(restraint)
restraint.close()
```

The restraint copies the PSF masses at construction, and
`setMassWeighting(True)` weights the center with them, like CHARMM's `WEIG`
or `MASS` keyword, so this example restrains the selected mass-weighted center
toward the reference position; `setMasses()` instead replaces the stored
masses and enables weighting in one call. The total potential energy can also
contain ordinary force-field terms.

Direct C++ use supplies the restraint's stream and output resources during
subscription:

```cpp
#include "AtomSelection.h"
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "HarmonicCenterOfMassRestraintForce.h"

#include <memory>
#include <string>
#include <vector>

using HmcmForce =
    HarmonicCenterOfMassRestraintForce<long long int, float>;

const std::string dataPath = "test/data/";

auto parameters =
    std::make_shared<CharmmParameters>("toppar/toppar_water_ions.str");
auto psf = std::make_shared<CharmmPSF>(dataPath + "nacl_pair.psf");
auto coordinates =
    std::make_shared<CharmmCrd>(dataPath + "nacl_pair.cor");

auto context = std::make_shared<CharmmContext>(psf, parameters);
context->setBoxDimensions({50.0, 50.0, 50.0});
context->setCoordinates(coordinates);

AtomSelection selection(psf->getNumAtoms(),
                        AtomSelection::InitialValue::ALL);

auto restraint = std::make_shared<HmcmForce>(psf);
restraint->setSelection(selection);
restraint->setReferencePosition({1.41, 1.41, 1.41});
restraint->setMassWeighting(true);
restraint->setForceConstant(1.0);

auto forceManager = context->getForceManager();
forceManager->subscribe(restraint, "hmcm", restraint->getStream(),
                        restraint->getForce(),
                        restraint->getEnergyVirial());

context->calculatePotentialEnergy();

forceManager->unsubscribe(restraint);
```

## Construction and Required State

Construction requires a non-null PSF with an initialized, strictly positive atom
count and exactly one finite mass per atom. The constructor copies the count,
which is fixed for the object's lifetime, and the masses; it retains no PSF.
The constructor establishes these defaults:

- every atom is selected;
- selected weights are uniform;
- the stored masses are the copied PSF masses, which mass weighting uses, like
  CHARMM's `WEIG` or `MASS` keyword;
- mass weighting is disabled;
- the force constant is zero;
- the reference position is `(0, 0, 0)`;
- all three reference-mask components are active;
- the reference distance is zero;
- box dimensions are unset at `(0, 0, 0)`;
- the output energy term is named `hmcm`;
- a private CUDA stream, fixed-point force array, and energy-virial object are
  allocated.

Force evaluation does not require box dimensions, because CHARMM's `CONS HMCM`
never reads the unit cell. `ForceManager` still supplies its atom count and box
through `initialize()`, which validates and stores them. An already initialized
manager invokes that hook during subscription; otherwise manager initialization
invokes it later.

Normal manager-driven order is:

1. construct the restraint from the PSF of the manager's system;
2. set the selection, reference position, optional mask and distance, weighting,
   and force constant;
3. set the context or manager box dimensions;
4. subscribe the restraint;
5. let `CharmmContext` or an integrator drive force calculation;
6. unsubscribe before closing or destroying the public handle.

Direct native calculation additionally requires a valid device coordinate
pointer, `clear()`, `calcForce()`, and synchronization of `getStream()` before
host code consumes results.

Changing selection or weighting rebuilds the selected host arrays; the next
`initialize()` or `calcForce()` transfers them and resizes reduction storage.
Do not overlap such changes with queued calculation or with another host
thread. Setters do not synchronize the private stream.

## Ownership and Lifetime

The native object owns or retains:

- the copied `AtomSelection`;
- a host copy of all masses;
- host/device mirrors of selected indices, selected weights, reduction
  partials, and restraint state;
- host scalar force, reference, mask, distance, weighting, and box state;
- shared ownership of one `Force<AT>`;
- shared ownership of one `CudaEnergyVirial` containing `hmcm`;
- shared ownership of a host scalar containing the private CUDA stream handle.

`getForce()`, `getEnergyVirial()`, and `getStream()` return copied shared
owners. The force and energy objects can outlive the restraint, but no restraint
remains to update them. A stream holder can also outlive the restraint, but the
restraint destructor destroys the underlying CUDA stream and writes a null
value into the shared holder.

The public C handle owns a shared pointer to the native
`HarmonicCenterOfMassRestraintForce<long long int, float>`. Successful manager
subscription copies the native owner. Destroying the C handle therefore does
not remove an existing subscription.

The Python wrapper owns its C handle. `ForceManager.subscribe()` retains the
Python wrapper only after native subscription succeeds, and the native manager
retains the underlying restraint independently. Unsubscribe before `close()`.
Closing a subscribed wrapper destroys the identifying C handle but leaves the
native subscription active, while the closed wrapper can no longer perform
normal unsubscription.

Selections, positions, masks, masses, and tags are copied. No C++, C, or Python
caller buffer is retained.

Do not copy or copy-assign the native class. Its implicit copy operations share
stream and output owners without establishing a safe independent CUDA-resource
lifecycle.

## Data, Shapes, and AKMA Units

For a restraint containing `N` atoms:

| Quantity | Native C++ | C ABI | Python | Meaning and units |
|---|---|---|---|---|
| Selection | `AtomSelection` with `N` atoms | `apo_atom_selection` with `N` atoms | `AtomSelection` | Nonempty zero-based atom membership; dimensionless |
| Force constant | `double` | `double` | `float()`-convertible object | Finite `k` of either sign, kcal mol^-1 angstrom^-2 |
| Reference position | `std::vector<double>` of length 3 | `double[3]` | Iterable of 3 `float()`-convertible values | `[x, y, z]`, angstroms |
| Reference mask | `std::vector<int>` of length 3 | `int[3]` | `None` or iterable of 3 `int()`-convertible values | `[x, y, z]`, each 0 or 1; dimensionless |
| Reference distance | `double` | `double` | `float()`-convertible object | Finite `r0` of either sign, angstroms |
| Masses | `std::vector<double>` of length `N` | `double[N]` | Iterable of `N` `float()`-convertible values | Finite values of either sign with a finite nonzero selected total, normally atomic mass units |
| Box | `std::vector<double>` of length 3 | Supplied through `ForceManager` | Supplied through `CharmmContext` or `ForceManager` | Orthorhombic `[x, y, z]` lengths, angstroms; stored but not used in evaluation |
| Current coordinates | Device `float4[N]` | Not exposed directly | Not exposed directly | `[x, y, z, charge]`; Cartesian values in angstroms, charge ignored |
| Force output | `Force<AT>` | Managed through `ForceManager` | Managed through `ForceManager` | Structure-of-arrays potential gradient, kcal mol^-1 angstrom^-1 |
| Energy output | `CudaEnergyVirial` term `hmcm` | Managed through `ForceManager` | Managed through `ForceManager` | Kilocalories per mole |

The force constant is not mass-normalized: masses define only the normalized
center. Multiplying all selected masses by the same nonzero factor leaves the
center, energy, and forces unchanged. Individual zero or negative masses are
accepted, but the selected total must be finite and nonzero, and every weight
fraction finite, whenever mass weighting is enabled.

For the public fixed-point specialization, raw `long long int` force-array
values use scale `2^40`. Multiplying by `1 / 2^40` recovers the stored potential
gradient. The active force vectors use structure-of-arrays layout with an
implementation-selected stride:

```text
x: force[0 * stride + atom]
y: force[1 * stride + atom]
z: force[2 * stride + atom]
```

No coordinate is imaged: the center and the center-to-reference displacement
use the raw coordinates the restraint receives.

## Errors

### Native C++

Verified argument failures raise
[ApoCharmmError](@ref ApoCharmmError) with
`ApoCharmmErrorCode::InvalidArgument` for:

- construction from a null PSF, a PSF with zero atoms, or a PSF with a NaN or
  infinite mass;
- selection or initialization atom-count mismatch;
- an empty selection;
- a zero or non-finite selected total weight, or a non-finite weight fraction,
  while mass weighting is active;
- a non-finite force constant;
- a reference-position vector whose length is not three;
- a non-finite reference-position component;
- a reference-mask vector whose length is not three;
- a mask value other than zero or one;
- a mask with no active component;
- a non-finite reference distance;
- a mass vector with the wrong length;
- a non-finite mass;
- a box vector whose length is not three;
- a non-positive or non-finite box length.

Construction from a PSF whose atom count is not initialized raises
`ApoCharmmErrorCode::NotInitialized`, and construction from a PSF whose mass
count differs from its atom count raises `ApoCharmmErrorCode::Runtime`.

CUDA allocation, transfer, clearing, synchronization, stream creation, and
immediate kernel-launch failures raise `ApoCharmmError` with
`ApoCharmmErrorCode::Cuda`.

Host allocation and implementation-limit failures can raise standard
`std::bad_alloc` or `std::length_error`. Selection and weighting updates are not
transactional; their symbol contracts describe possible partial state after
failure.

`calcForce()` does not validate the coordinate pointer, device residence,
length, or concurrent access. Violating those preconditions is outside its
checked contract.

### C ABI

Every status-returning C function clears the calling thread's previous
diagnostic at entry. Success returns `APO_STATUS_OK` and leaves the diagnostic
empty. Failure leaves nonempty text available through
[apo_last_error](@ref apo_last_error) on the same thread.

`APO_STATUS_INVALID_ARGUMENT` covers invalid handles, null required pointers,
shape or length mismatches, invalid numeric values, invalid masks, empty
selection or tag, a zero or non-finite selected total weight or a non-finite
weight fraction, duplicate subscription, and unsubscription of a restraint that
is not subscribed.

`APO_STATUS_NOT_INITIALIZED` reports a PSF with a negative atom count.

`APO_STATUS_CUDA_ERROR` is returned by construction, selection updates, mass
updates, or weighting updates when their categorized native CUDA operations
fail.

`APO_STATUS_RUNTIME_ERROR` covers an inconsistent PSF mass count, host
allocation, diagnostic construction, and uncategorized standard or nonstandard
C++ exceptions.

Creation sets a valid output slot to `NULL` before native construction and
leaves it `NULL` on failure. Setter and manager functions have no output
parameters. Destruction accepts `NULL`, prevents exceptions from crossing the
ABI boundary, and normally preserves a stale thread-local diagnostic.

### Python

Python conversion occurs before the C call:

- incompatible objects, non-iterable inputs, or failed element conversion can
  raise `TypeError`;
- failed `float()` or `int()` conversion can raise `ValueError`;
- numeric conversion can raise `OverflowError`;
- temporary list or `ctypes` allocation can raise `MemoryError`;
- using a closed wrapper, or constructing from a closed `CharmmPsf`, raises
  `RuntimeError`;
- force-tag encoding can raise `UnicodeEncodeError`;
- every nonzero native status raises `ApoCharmmError` with status, context, and
  a copied native diagnostic.

`setMassWeighting()` performs truth-value conversion through `ctypes.c_bool`
rather than enforcing an exact Python `bool`. Reference-mask elements are
converted with `int()` and then narrowed to C `int` before native zero-or-one
validation.

## Important Behavior and Limitations

`calcForce()` and `clear()` enqueue work on the restraint's private CUDA stream.
Neither method synchronizes before returning. `ForceManager` orders and
synchronizes subscribed streams during aggregate force and energy processing.
Direct native callers must synchronize explicitly.

Selection, mass, and weighting changes rebuild the selected host arrays and
mark them dirty. `initialize()` and `calcForce()` transfer them with
`CudaContainer` transfers and clears that call `cudaDeviceSynchronize()`, so
several changes cost one transfer before the next calculation; those
transfers can wait for unrelated work already submitted to the current CUDA
device. Force constants, reference values, masks, reference distance, and box
dimensions are host scalar state and do not trigger CUDA synchronization.

Force and energy updates are additive. `clear()` is required before fresh
output. Within `calcForce()` itself, `calcEnergy == false` leaves the existing
`hmcm` energy unchanged, so direct callers can observe a stale disabled energy.
`calcVirial` is ignored: the restraint declares `contributesVirial == false`,
so `ForceManager` passes `false` and leaves it out of the aggregate virial,
like CHARMM's internal virial. The current `ForceManager` calculation path
calls `clear()` for every subscribed force before evaluation, so manager-driven
outputs normally begin at zero even when energy calculation is disabled.

A zero force constant is a complete no-op. The call does not read coordinates
or modify output.

The center uses the raw coordinates the restraint receives. `CharmmContext`
recenters every bond-connected group whose center leaves the half box, so a
selection spanning several groups, or a reference position outside the
centered cell, can see its raw center move by a box length, as CHARMM's image
centering can. Keep such selections within one group or choose the reference
accordingly. Reference positions are not wrapped or rescaled when box
dimensions change.

At exactly zero displacement the gradient is zero. CHARMM divides by zero there
when the reference distance is nonzero.

For `CT = float`, the center sums and the total weight are formed in single
precision, so masses that nearly cancel can lose their total to rounding.

Selection or weighting changes can reallocate selected-index, selected-weight,
and partial-sum device storage. Any internal pointer borrowed from those
containers is invalid after reallocation. Public APIs do not expose those
pointers directly.

The linked library provides explicit native instantiations only for:

```cpp
HarmonicCenterOfMassRestraintForce<long long int, float>
HarmonicCenterOfMassRestraintForce<long long int, double>
```

`CT` is the representation of the force arithmetic in the three kernels,
including the center reductions and the energy value, which is converted to
double before it is added to the energy storage. The public C ABI and Python
API use the first specialization, so they evaluate the force and the energy
value in single precision.

No layer provides internal host-thread synchronization.

## Related Subsystems

- [AtomSelection](@ref atom_selection) defines the selected atom set.
- [ForceManager](@ref force_manager) owns subscribed calculation resources and
  aggregates force and energy outputs.
- [CharmmContext](@ref charmm_context) owns molecular state and drives manager
  initialization and force calculation.
- [CudaContainer](@ref cuda_container) implements the parameter mirrors and
  synchronized transfer behavior used by selection and mass updates.
- [HarmonicRestraintForce](@ref harmonic_restraint_force) restrains individual
  atoms to absolute Cartesian reference positions.
- [ApoCharmmError](@ref apocharmm_error) documents native, C ABI, and Python
  error propagation.
- `Force` defines fixed-point structure-of-arrays force storage.
- `CudaEnergyVirial` stores the `hmcm` energy.

## Developer Architecture

### Public and private layers

The public native template is declared in
`include/HarmonicCenterOfMassRestraintForce.h` and implemented in
`src/HarmonicCenterOfMassRestraintForce.cu`.

The stable C ABI is declared in
`capi/include/apocharmm_c/HarmonicCenterOfMassRestraintForce.h`. Its private
shared-owner handle is defined in
`capi/private_include/apocharmm_c/detail/HarmonicCenterOfMassRestraintForceHandle.h`,
and guarded entry points are implemented in
`capi/src/apocharmm_c/HarmonicCenterOfMassRestraintForce.cpp`.

The Python wrapper is implemented in
`python/apocharmm/harmonic_center_of_mass_restraint_force.py`. It configures
`ctypes` prototypes, owns one C handle, converts Python inputs, and delegates
nonzero status handling to the shared Python error layer.

### Ownership graph

After successful Python subscription, the relevant ownership graph is:

```text
Python HarmonicCenterOfMassRestraintForce --owns--> C restraint handle
C restraint handle --shared--> native restraint
Python ForceManager --retains--> Python restraint wrapper
native ForceManager --shared--> native restraint
native ForceManager --shared--> restraint stream holder
native ForceManager --shared--> restraint Force
native ForceManager --shared--> restraint CudaEnergyVirial
native restraint --shared--> stream holder, Force, CudaEnergyVirial
```

Unsubscription removes the manager-held native owners and then the Python
manager-held wrapper reference. Destruction of the restraint C handle removes
only that handle's native owner.

### State and invariants

The immutable atom count determines the complete mass array and output force
size. Normal calculation state satisfies:

```text
m_Masses.size() == m_NumAtoms
m_Selection.getNumAtoms() == m_NumAtoms
m_NumSelected == m_Selection.getNumSelected()
m_AtomIndices.size() == m_NumSelected
m_AtomWeights.size() == m_NumSelected
m_PartialSums.size() == ceil(m_NumSelected / 256)
m_RestraintState.size() == 1
sum(m_AtomWeights) is finite and nonzero
reference masks are 0 or 1 and at least one is active
```

`m_AtomIndices` stores selected indices in ascending order.
`m_AtomWeights` stores either one or the corresponding value from `m_Masses`.
Host scalar force, reference, mask, and distance values are passed by value to
kernels at launch time.

A failed selection or weighting update can temporarily violate the selected
count and selected-array invariants because mutation is sequential rather than
transactional.

### Host and device data flow

Selection and mass configuration begins in host memory. `updateSelectedAtoms()`
materializes the selected vectors in the host mirrors and sets
`m_DeviceDataDirty`. `syncDeviceData()`, called by `initialize()` and
`calcForce()`, transfers them to device memory, resizes one partial record per
256 selected atoms, clears the partial and state containers, and clears the
flag. These `CudaContainer` operations synchronize the current CUDA device.

Current coordinates remain device-resident and are borrowed from the manager.
The restraint does not copy coordinates to host. Force and energy
contributions remain device-resident until manager aggregation or an explicit
native copy.

### Kernel pipeline

One private stream orders three kernels:

1. `PartialSumsKernel` sums `w_i * r_i` and `w_i` over the raw selected
   coordinates and writes one partial record per block.
2. `StateKernel` reduces the partials, forms the center, applies the Cartesian
   mask to the raw center-to-reference displacement, adds the energy when
   requested, and writes the per-axis gradient factors
   `d * 2 * k / W * (s - r0) / s` in one state record.
3. `ApplyForcesKernel` multiplies those factors by each selected weight and
   atomically adds the result to the force array.

Every kernel operation uses `CT`, except that the energy value is converted to
double when it is added to the energy storage. The state record has the
matching four-component type:

```text
state[0] = {factorX, factorY, factorZ, 0}
```

Immediate launch checks do not synchronize. Later asynchronous failures surface
at a synchronization or subsequent CUDA API boundary.

### Error boundaries

Native configuration performs categorized validation before or during
mutation. `calcForce()` checks only immediate CUDA launches; coordinate-pointer
correctness is a caller precondition.

C ABI entry points validate opaque handles and required pointers before
dereferencing them. The common guard maps `ApoCharmmErrorCode` to `apo_status`,
maps legacy `std::invalid_argument` to `APO_STATUS_INVALID_ARGUMENT`, maps other
exceptions to `APO_STATUS_RUNTIME_ERROR`, and prevents exceptions from crossing
the ABI boundary.

Python wrappers perform Python conversion before the C call. Configured status
callbacks convert every nonzero C result into `ApoCharmmError`.

### Extension points and constraints

`ForceManager::subscribe()` accepts this restraint through the generic
force-view interface together with its stream, force storage, energy storage,
and static `contributesVirial` value. A compatible new force type must provide
the same lifecycle and resource surface.

Adding multiple independently restrained groups requires separate selection,
weight, state, energy, and output bookkeeping; the current object represents
one group and one `hmcm` energy term.

Adding another `AT` or `CT` combination requires a valid device accumulation
path and an explicit template instantiation in
`src/HarmonicCenterOfMassRestraintForce.cu`.

### Performance-sensitive behavior

Center reduction uses 256 threads per block. Force writes are atomic. Energy
uses one atomic addition from the state kernel. Selected-array changes are
transferred, with storage resizing and device-wide synchronization, by the
first calculation after the change; a calculation without changes only checks
the dirty flag.

### Tests and current technical debt

Relevant validation and behavior coverage is located in:

- `test/unittests/unittest-harmonicCenterOfMassRestraintForce.cpp`;
- `test/unittests/unittest-capiHarmonicCenterOfMassRestraintForce.cpp`;
- `test/pytest/python_api_harmonic_center_of_mass_restraint_force.py`.

The native tests cover defaults, validation, weighting, masks, radial distance,
raw coordinates, evaluation without a box, signed parameters, the zero-distance
gradient, the arithmetic type, force and energy values, clearing, the unchanged
manager virial, and manager subscription. Direct C ABI tests cover handles,
pointers, arrays, statuses, diagnostics, subscription, and diagnostic
clearing. Python tests cover conversion, validation, subscription,
propagation, error translation, a manager evaluation with a raw center and
signed parameters, and closed-object behavior.

Current implementation constraints visible to maintainers include unsafe
compiler-generated native copy operations, sequential nontransactional
selection and weighting updates, permissive Python truth-value conversion for
mass weighting, and integer conversion before reference-mask validation.

## API Reference

### Native C++

- [HarmonicCenterOfMassRestraintForce](@ref HarmonicCenterOfMassRestraintForce)
- [AtomSelection](@ref AtomSelection)
- [ForceManager](@ref ForceManager)

### C ABI

- [apo_harmonic_center_of_mass_restraint_force](@ref apo_harmonic_center_of_mass_restraint_force)
- [apo_harmonic_center_of_mass_restraint_force_create](@ref apo_harmonic_center_of_mass_restraint_force_create)
- [apo_harmonic_center_of_mass_restraint_force_destroy](@ref apo_harmonic_center_of_mass_restraint_force_destroy)
- [apo_harmonic_center_of_mass_restraint_force_set_selection](@ref apo_harmonic_center_of_mass_restraint_force_set_selection)
- [apo_harmonic_center_of_mass_restraint_force_set_force_constant](@ref apo_harmonic_center_of_mass_restraint_force_set_force_constant)
- [apo_harmonic_center_of_mass_restraint_force_set_reference_position](@ref apo_harmonic_center_of_mass_restraint_force_set_reference_position)
- [apo_harmonic_center_of_mass_restraint_force_set_reference_distance](@ref apo_harmonic_center_of_mass_restraint_force_set_reference_distance)
- [apo_harmonic_center_of_mass_restraint_force_set_masses](@ref apo_harmonic_center_of_mass_restraint_force_set_masses)
- [apo_harmonic_center_of_mass_restraint_force_set_mass_weighting](@ref apo_harmonic_center_of_mass_restraint_force_set_mass_weighting)
- [apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force](@ref apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force)
- [apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force](@ref apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force)

### Python

- [apocharmm.HarmonicCenterOfMassRestraintForce](@ref python_harmonic_center_of_mass_restraint_force)
- [apocharmm.ForceManager](@ref python_force_manager)
- [apocharmm.AtomSelection](@ref python_atom_selection)
