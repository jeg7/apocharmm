# CharmmContext {#charmm_context}

## Purpose

`CharmmContext` is apoCHARMM's owner and coordinator for mutable molecular
state. It stores coordinates, charges, velocities, inverse masses, box and
periodic-boundary state, random-velocity state, degree-of-freedom bookkeeping,
and force-evaluation output. It delegates bonded, direct-space, reciprocal, and
subscribed-force calculations to an attached [ForceManager](@ref ForceManager).

Use a context whenever force calculation or integration requires one coherent
set of molecular state. The context is also the point at which a
[CharmmPSF](@ref CharmmPSF), [CharmmParameters](@ref CharmmParameters), and
[ForceManager](@ref ForceManager) are reconciled and, once all required state is
available, initialized.

The subsystem is exposed through native C++, the public C ABI, and Python.

## Quick Start

The highest-level supported path is the Python API:

```python
from pathlib import Path

import apocharmm as apo

data = Path("test/data")

parameters = apo.CharmmParameters("toppar/toppar_water_ions.str")
psf = apo.CharmmPsf(data / "waterbox.psf")
coordinates = apo.CharmmCrd(data / "waterbox.crd")

context = apo.CharmmContext(psf, parameters)
context.setBoxDimensions((50.0, 50.0, 50.0))
context.setCoordinates(coordinates)
context.calculatePotentialEnergy(print_energy=True)
```

Run the example from the repository root. The final call computes forces,
potential energy, and virial and writes a CHARMM-style energy table to standard
output.

The equivalent direct native workflow is:

```cpp
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"

#include <memory>

int main() {
  auto parameters = 
      std::make_shared<CharmmParameters>("toppar/toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>("test/data/waterbox.psf");
  auto coordinates =
      std::make_shared<CharmmCrd>("test/data/waterbox.crd");

  auto context = std::make_shared<CharmmContext>(psf, parameters);
  context->setBoxDimensions({50.0, 50.0, 50.0});
  context->setCoordinates(coordinates);
  context->calculatePotentialEnergy(false, true);

  return 0;
}
```

Constructing the native context with `std::make_shared` permits it to establish
the force manager's weak backlink.

## Construction and Required State

The two principal construction paths are:

1. A PSF and parameter set. The context creates its own `ForceManager`, imports
   PSF charges and masses, and waits for valid box dimensions before
   force-manager initialization.
2. An existing `ForceManager`. The context imports any PSF, parameters, box,
   and periodic-boundary state already present. A complete uninitialized
   manager is initialized during construction.

Native C++ also exposes a default constructor for incremental setup. The C ABI
and Python wrapper do not expose that path.

The force manager is considered complete when all of the following are present:

- a `ForceManager`;
- a `CharmmPSF`;
- `CharmmParameters`;
- exactly three positive box dimensions.

The current completeness predicate does not require coordinates or an explicit
periodic-boundary setter. A normal force manager supplies `PBC::P1` by default.

`PBC::UNSET`, `APO_PBC_UNSET`, and `PeriodicBoundaryCondition.UNSET` represent
the same invalid, unconfigured sentinel. They do not select an aperiodic system.
The native, C ABI, and Python periodic-boundary setters reject this value;
runnable configurations use `P1` or `P21`.

For custom nonbonded configuration, call `setKappa()`, `setCutoff()`,
`setCtonnb()`, `setCtofnb()`, `setFFTGrid()`, `setPmeSplineOrder()`, and
`setVdwType()` before the operation that supplies the final initialization
prerequisite. In the PSF/parameter construction path, setting valid box
dimensions normally triggers initialization.

Coordinates can be supplied before or after the box. If coordinates are already
present when initialization occurs, setup finishes by image-centering the
coordinates and building the neighbor list. A later successful coordinate
update rebuilds the neighbor list when the manager was already initialized.

Velocity-dependent temperature calculation additionally requires a positive
degree-of-freedom count. That count is established by
`useHolonomicConstraints()` and is recomputed automatically after successful
force-manager initialization.

## Ownership and Lifetime

Native `CharmmContext` stores shared owners of its PSF, parameter set, and force
manager. The force manager stores only a weak reference back to the context, so
the normal ownership graph has no cycle:

```text
CharmmContext
  |-- shared_ptr --> CharmmPSF
  |-- shared_ptr --> CharmmParameters
  |-- shared_ptr --> ForceManager
                         |
                         `-- weak_ptr --> CharmmContext
```

Native collaborator getters return copied `std::shared_ptr` instances. The
returned owner can keep that collaborator alive after the context is destroyed.

The C ABI `apo_charmm_context` handle owns a shared native context and retains
shared native collaborator references. Constructor and setter input handles are
borrowed; the C handle itself is not retained. After a successful operation, the
caller may destroy the source collaborator handle without invalidating the
context.

`apo_charmm_context_get_force_manager()` allocates a newly owned C handle that
shares the existing native manager. Release it with
`apo_force_manager_destroy()`. Release a context handle with
`apo_charmm_context_destroy()`. Context destruction accepts `NULL` and is
non-throwing.

The Python wrapper owns its context C handle and retains the collaborator
wrappers supplied during construction or setter calls. `close()` is idempotent,
and the wrapper supports `with` statements. Array inputs are copied into
temporary C buffers and are never retained. Array outputs are new Python
containers.

Native container and topology getters that return references are borrowed
aliases. Their lifetime ends when the owning context or collaborator is
destroyed. Reallocation or structural mutation can invalidate previously
borrowed elements, iterators, and device pointers.

## Data, Shapes, and AKMA Units

The principal per-atom layouts are:

| State | Native record | Python/C shape | Component order | Units |
|---|---|---|---|---|
| Coordinates and charges | `float4` and `double4` | `(N, 4)` | `x, y, z, charge` | angstrom, elementary charge |
| Coordinates | `double3` input | `(N, 3)` | `x, y, z` | angstrom |
| Velocities and inverse masses | `double4` | `(N, 4)` | `vx, vy, vz, 1/mass` | angstrom per AKMA time, reciprocal atomic mass unit |
| Velocities | `double3` input | `(N, 3)` | `vx, vy, vz` | angstrom per AKMA time |
| Masses | scalar input | `(N,)` | atom order | atomic mass unit |
| Box dimensions | scalar vector | `(3,)` | X, Y, Z | angstrom |
| FFT grid | integer vector | `(3,)` | X, Y, Z | dimensionless grid points |

Coordinate-only setters preserve charge. Charge-only setters preserve
coordinates. Velocity-only setters preserve inverse mass. Mass setters store
the reciprocal and preserve velocity.

The context keeps coordinate/charge data in both single and double precision.
Normal setters update both host mirrors and transfer both arrays to device.
Velocity and inverse-mass setters update their host mirror and transfer it to
device.

Temperature uses kelvin. Kappa uses inverse angstroms. Cutoff, `ctonnb`, and
`ctofnb` use angstroms. Volume uses cubic angstroms. Kinetic and potential
energy use kilocalories per mole. Force uses kilocalories per mole per angstrom.
The exact public interpretation and component order of the nine-element pressure
and virial-related containers are not established by the current subsystem
implementation.

The C ABI function named `apo_charmm_context_get_velocity_mass()` and Python
method named `getVelocityMass()` return inverse mass in the fourth component,
not mass.

## Errors

### Native C++

Verified native failures use [ApoCharmmError](@ref ApoCharmmError) and the
following codes:

- `ApoCharmmErrorCode::InvalidArgument` for null collaborators, invalid counts
  and shapes, invalid scalar ranges, atom-count mismatches, and unsupported
  configuration values.
- `ApoCharmmErrorCode::NotInitialized` when required atom, PSF, force-manager,
  box, velocity, or degree-of-freedom state is absent.
- `ApoCharmmErrorCode::NotImplemented` for pressure computation, force reset
  mode, and composite-manager energy-table printing.
- `ApoCharmmErrorCode::Runtime` for CHARMM velocity-file open, read, and parse
  failures and other explicitly categorized runtime conditions.
- `ApoCharmmErrorCode::Cuda` for checked CUDA allocation, transfer, launch,
  stream, and synchronization failures.

`calculatePotentialEnergy(..., true)` can also propagate `std::out_of_range` if
a named energy component required by the formatter is absent.

Mutating operations are not transactional. An exception from a later CUDA or
initialization step does not guarantee rollback of earlier host-side changes.

### C ABI

Every status-returning entry point catches native exceptions. Native error codes
map to the corresponding `apo_status`; `std::invalid_argument` maps to
`APO_STATUS_INVALID_ARGUMENT`; other exceptions map to
`APO_STATUS_RUNTIME_ERROR`.

Each status call clears the current thread's previous diagnostic at entry. On
failure, `apo_last_error()` returns the current thread's diagnostic until the
next C ABI call on that thread or thread exit. Successful destruction is an
exception: the non-throwing destroy guard does not clear a stale diagnostic.

Owned-handle output slots are set to `NULL` before later validation. Scalar and
buffer outputs are written only after their direct validation and required
native transfer succeeds.

### Python

Python performs wrapper-level type, shape, enum, path, and uint64 checks before
calling the C ABI. Those checks raise `TypeError` or `ValueError`. Access to a
closed wrapper raises `RuntimeError`. Every nonzero native status is raised as
`ApoCharmmError` with the operation label and native diagnostic preserved.

## Important Behavior and Limitations

`setCoordinatesCharges()` and `setCoordinates()` transfer coordinate mirrors to
device. If the manager was already initialized, they then image groups and
rebuild the neighbor list. Charge-only changes do not rebuild it.

`imageCentering()` mutates device storage only. It can change double-precision
coordinates, single-precision coordinates, velocity components, and force
components. It finishes with a device-wide synchronization, but it does not
update host mirrors. Transfer affected containers from device before reading
their host arrays.

The image-centering kernel treats each PSF group as one inclusive contiguous
atom-index range and images the group based on the arithmetic center of its
coordinates. `PBC::P21` X-boundary crossings additionally reflect Y and Z
coordinates, velocities, and forces.

`assignVelocitiesAtTemperature()` constructs a new generator from the stored
seed on every call. Repeating the call with the same seed, temperature, masses,
and atom order produces the same sampled sequence. It does not remove
center-of-mass motion or rescale the result.

`calculatePotentialEnergy()` also calculates forces and virial. In the Python
wrapper, reset mode is always disabled. Native and C ABI callers can pass reset
explicitly, but the current force manager reports
`ApoCharmmErrorCode::NotImplemented` or `APO_STATUS_NOT_IMPLEMENTED` when it is
enabled.

Configuration setters for kappa, cutoff, `ctonnb`, `ctofnb`, FFT grid, PME
spline order, and van der Waals model update stored force-manager fields. The
current implementation does not rebuild already initialized direct- or
reciprocal-space force objects. Configure these values before initialization.

The mutable C++ container and vector accessors intentionally bypass context
invariants. They do not synchronize mirrors, update state flags, propagate box
changes, or rebuild neighbor lists.

Copy construction does not copy the force-manager association and does not copy
the kinetic-virial tensor. Copy assignment is compiler-generated and therefore
has different memberwise behavior.

Pressure computation is not implemented. The pressure container is exposed but
has no supported producer in `CharmmContext`.

The subsystem has no internal locking. External synchronization is required for
concurrent use, including concurrent access through returned collaborators or
mutable storage aliases.

## Related Subsystems

- [ForceManager](@ref force_manager) performs bonded, direct-space, reciprocal,
  and subscribed force calculations.
- [CharmmPSF](@ref charmm_psf) supplies topology, charges, masses, groups,
  bonds, water records, and exclusions.
- [CharmmParameters](@ref charmm_parameters) supplies force-field parameters.
- [CudaContainer](@ref cuda_container) explains the context's paired host/device
  arrays, mirror coherence, transfers, synchronization, and reference
  invalidation.
- [ApoCharmmError](@ref apocharmm_error) describes native error codes and
  diagnostics.
- [PBC](@ref PBC) and
  [python_periodic_boundary_condition](@ref python_periodic_boundary_condition)
  describe periodic
  boundary identifiers.
- [python_vdw_type](@ref python_vdw_type) describes Python van der Waals model
  values.

## Developer Architecture

The public native class is declared in `include/CharmmContext.h` and implemented
in `src/CharmmContext.cu`. The implementation owns context-local CUDA
containers, performs velocity generation and kinetic-energy reduction, and
coordinates force-manager initialization and neighbor-list state.

The public C ABI is declared in
`capi/include/apocharmm_c/CharmmContext.h`. Its private ownership payload is
defined in `capi/private_include/apocharmm_c/detail/CharmmContextHandle.h`, and
entry points are implemented in `capi/src/apocharmm_c/CharmmContext.cpp`. Every
status entry point uses the shared C ABI guard and validation helpers.
Constructor and force-manager output functions clear owned-handle output slots
before later validation.

The Python wrapper is `python/apocharmm/charmm_context.py`. It initializes
`ctypes` prototypes lazily, normalizes enums and paths, converts nested rows to
flat buffers, and retains Python collaborators. It deliberately exposes only a
subset of the native context: pressure, imaging, neighbor-list reset, kinetic
energy, direct force access, virial access, and mutable native containers are
not currently public Python methods.

The setup lifecycle is:

```text
attach/import collaborators
        |
        v
reconcile context and ForceManager state
        |
        v
complete state? -- no --> remain uninitialized
        |
       yes
        |
        v
ForceManager::initialize()
        |
        v
reimport canonical manager state
        |
        v
recompute degrees of freedom
        |
        v
coordinates already present? -- yes --> image and rebuild neighbor list
```

The force-manager backlink is established with
`CharmmContext::weak_from_this()`. A stack-allocated or uniquely owned native
context therefore cannot establish the backlink, although it can still retain
and call its force manager.

Coordinate setters update host mirrors and use `CudaContainer` transfers, which
perform synchronous runtime copies. Image centering runs on the default stream
and calls `cudaDeviceSynchronize()`. Force calculation itself uses the
force-manager's bonded, reciprocal, direct, and aggregation streams.

Extension work should preserve these boundaries:

- Native public contracts belong in `include/CharmmContext.h`.
- Algorithms and synchronization constraints belong in
  `src/CharmmContext.cu`.
- New C ABI functions must validate borrowed handles and outputs, use the
  standard guard, define ownership, and preserve output state on failure.
- New Python methods must configure one status prototype, preserve required
  collaborator lifetimes, expose Python-native shapes, and distinguish
  wrapper-side `TypeError` or `ValueError` from native `ApoCharmmError`.
- Any operation that changes coordinate geometry must define whether it
  transfers mirrors, images coordinates, and invalidates or rebuilds the
  neighbor list.
- Any operation that changes force configuration after initialization must
  either rebuild the affected force objects or reject the transition.

Relevant tests are:

- `test/unittests/unittest-charmmContext.cpp`;
- `test/unittests/unittest-capiCharmmContext.cpp`;
- `test/pytest/python_api_charmm_context.py`;
- the setup and force-manager tests exercised indirectly by those files.

`example/energy.py` is the current user-facing energy-evaluation example.

Visible architectural constraints include nontransactional state transitions,
different copy-construction and copy-assignment behavior, mutable invariant
escape hatches, post-initialization configuration that does not rebuild active
force objects, incomplete pressure support, and public `ctonnb`/`ctofnb` names
whose current backend forwarding order requires clarification.

## API Reference

- Native C++: [CharmmContext](@ref CharmmContext) and [PBC](@ref PBC).
- Public C ABI: [apo_charmm_context](@ref apo_charmm_context),
  `apo_charmm_context_create()`,
  `apo_charmm_context_create_from_psf_parameters()`, and the
  `apo_charmm_context_*` functions in
  `capi/include/apocharmm_c/CharmmContext.h`.
- Python: [python_charmm_context](@ref python_charmm_context),
  [python_periodic_boundary_condition](@ref python_periodic_boundary_condition),
  and [python_vdw_type](@ref python_vdw_type).
