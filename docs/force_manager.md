# ForceManager {#force_manager}

## Purpose

`ForceManager` owns and coordinates apoCHARMM's built-in bonded,
direct-space, and reciprocal-space force implementations. It combines a
[CharmmPSF](@ref CharmmPSF), [CharmmParameters](@ref CharmmParameters), periodic
box, long-range electrostatic configuration, nonbonded cutoff and switching
configuration, CUDA streams, force arrays, energy and virial storage, and
optional subscribed forces.

Users normally configure a manager and then attach it to a
[CharmmContext](@ref CharmmContext). The context reconciles molecular state and
initializes the manager once the PSF, parameters, and valid box dimensions are
available. Direct C++ callers may invoke
[ForceManager::initialize](@ref ForceManager::initialize) explicitly.

The native C++ interface exposes force evaluation and device storage. The
public C ABI and Python wrapper currently expose construction, configuration,
state inspection, and force subscription support used by higher-level
objects. Python force evaluation is performed through `CharmmContext`.

## Quick Start

The highest-level explicit ForceManager workflow uses Python:

```python
from pathlib import Path

import apocharmm as apo

parameters = apo.CharmmParameters("toppar/toppar_water_ions.str")
psf = apo.CharmmPsf("test/data/waterbox.psf")
coordinates = apo.CharmmCrd("test/data/waterbox.crd")

force_manager = apo.ForceManager(psf, parameters)
force_manager.setBoxDimensions((50.0, 50.0, 50.0))

context = apo.CharmmContext(force_manager)
context.setCoordinates(coordinates)
context.calculatePotentialEnergy(print_energy=True)

assert force_manager.isInitialized()
```

Run this example from the repository root. Constructing the context with the
complete manager initializes the native force backends. The final call computes
force, potential-energy, and virial state and prints a CHARMM-style energy
table.

The equivalent direct native setup is:

```cpp
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "ForceManager.h"

#include <memory>

int main() {
  auto parameters =
      std::make_shared<CharmmParameters>("toppar/toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>("test/data/waterbox.psf");
  auto coordinates =
      std::make_shared<CharmmCrd>("test/data/waterbox.crd");

  auto forceManager = std::make_shared<ForceManager>(psf, parameters);
  forceManager->setBoxDimensions({50.0, 50.0, 50.0});

  auto context = std::make_shared<CharmmContext>(forceManager);
  context->setCoordinates(coordinates);
  context->calculatePotentialEnergy(false, true);

  return 0;
}
```

Constructing the native context with `std::make_shared` permits the context to
establish the manager's weak backlink.

## Construction and Required State

A default-constructed manager has no PSF or parameter set and cannot be
initialized. Construction from a PSF and parameters retains both objects but
also leaves the manager uninitialized.

The base defaults are:

| Setting | Default | Unit or meaning |
|---|---:|---|
| Box | sentinel values near `-9999.9999` | not configured |
| Kappa | `0.34` | inverse angstroms |
| Pair-list cutoff | `14.0` | angstroms |
| `ctonnb` | `12.0` | current outer/off switching distance, angstroms |
| `ctofnb` | `10.0` | current inner/on switching distance, angstroms |
| FFT grid | `(-1, -1, -1)` | automatic selection |
| PME spline order | `4` | dimensionless |
| PBC | [PBC::P1](@ref PBC::P1) | translational periodicity |
| VDW model | `VDW_VFSW` | native model code `3` |
| Energy printing | disabled | standard output side effect |

`UNSET` is reserved as an invalud periodic-boundary sentinel. It does not
represent an aperiodic force calculation. `ForceManager` accepts only `P1` and
`P21`, and initialization verifies that the stored value is one of those
runnable modes.

Before initialization, provide:

1. a non-null PSF;
2. a non-null parameter set;
3. exactly three finite positive box dimensions; and
4. a positive pair-list cutoff no greater than half the X box length.

When any FFT dimension remains non-positive, initialization selects an even
dimension based on the corresponding box length, with a minimum of two.

Initialization constructs bonded, reciprocal, and direct streams and backends,
initializes already subscribed forces, allocates aggregate force and energy
storage, performs a device-wide synchronization, and builds SHAKE records. The
initialized flag becomes true only after every phase succeeds.

Use configuration setters before the first initialization. Replacing the PSF,
parameters, or PBC clears the initialized flag but does not deallocate existing
CUDA resources. Other scalar setters generally update only stored
configuration and do not rebuild an active backend.

## Ownership and Lifetime

The native manager retains shared ownership of its PSF and parameter set.
[ForceManager::getPsf](@ref ForceManager::getPsf) and
[ForceManager::getPrm](@ref ForceManager::getPrm) return copied shared owners
that may outlive the manager.

The manager stores its [CharmmContext](@ref CharmmContext) association as a
`std::weak_ptr`. The manager therefore does not keep the context alive. Calling
[ForceManager::getContext](@ref ForceManager::getContext) acquires a new shared
owner only while the context still exists.

The manager owns its built-in CUDA backends, aggregate work containers, cached
clear graph, and underlying CUDA streams. Stream getters return shared ownership
of the host allocation containing a `cudaStream_t`, but they do not transfer
ownership of the CUDA stream itself. The manager destroys the CUDA stream
during cleanup even when an external shared pointer still exists.

Force-storage getters return copied shared owners of `Force` objects. An
external owner can preserve that storage after manager destruction. It does not
preserve backend updates, and an old force object is not updated if a later
initialization replaces the manager's member.

A successful C ABI constructor creates an owned
[apo_force_manager](@ref apo_force_manager). The handle independently retains
native shared ownership of the PSF and parameter set, so the source C handles
may be destroyed. Release the manager handle with
[apo_force_manager_destroy](@ref apo_force_manager_destroy).

The Python `ForceManager` owns its C handle and retains the source
`CharmmPsf` and `CharmmParameters` wrappers. It also retains every successfully
subscribed Python force object until unsubscription or `close()`.

Native subscription retains six corresponding pieces of state: the concrete
force, type-erased view, tag, stream, fixed-point force storage, and
energy-virial storage. Successful operations require these parallel vectors to
remain the same length. Unsubscription releases only the manager's owners; an
object remains alive while another shared or Python reference exists.

## Data, Shapes, and AKMA Units

apoCHARMM uses AKMA-compatible molecular mechanics units for this subsystem.

| Quantity | Representation | Unit |
|---|---|---|
| Box | three values, `[x, y, z]` | angstroms |
| Kappa | scalar | inverse angstroms |
| Cutoff and switching distances | scalar | angstroms |
| FFT dimensions | three integers | dimensionless grid points |
| PME spline order | integer | dimensionless |
| Coordinate-charge input | `numAtoms` device `float4` records, `[x, y, z, q]` | angstroms and elementary charge |
| Component forces | device `Force<long long int>` | scaled force representation |
| Aggregate forces | device `Force<double>` | kilocalories per mole per angstrom |
| Potential energy | one `double` | kilocalories per mole |
| Virial | nine `double` values | kilocalories per mole |
| Atom masses | PSF values | atomic mass units |
| SHAKE inverse masses | `float4.x` and `float4.w` | inverse atomic mass units |
| SHAKE bond-length square | `float4.z` | square angstroms |

`Force<T>` stores components in structure-of-arrays order. For a reported
stride `s`, X occupies `[0, numAtoms)`, Y begins at `s`, and Z begins at
`2 * s`. The stride may be greater than the atom count because allocation is
aligned. Fixed-point component forces must be multiplied by
`INV_FORCE_SCALE` to recover their floating-point value.

The SHAKE atom container has one `int4` per selected heavy-atom-centered group:
`(heavy, hydrogen1, hydrogen2, hydrogen3)`. Unused hydrogen slots are `-1`.
The corresponding `float4` is an internal solver record containing an inverse
heavy mass, an average-mass field, a squared equilibrium bond length, and an
inverse hydrogen mass. This layout is exposed by the native accessors but is
not a stable interchange format.

The aggregate potential-energy container has one element. It is updated only
when force calculation requests energy. The aggregate virial container has
nine elements; the current repository does not establish a public
component-to-matrix order or sign convention.

## Errors

### Native C++

Validated native failures use [ApoCharmmError](@ref ApoCharmmError).

- `InvalidArgument` covers null collaborators, invalid box length or values,
  invalid scalar ranges, invalid FFT dimensions, invalid VDW codes, invalid
  subscription resources, duplicates, unknown subscriptions, and a cutoff
  larger than half the X box length.
- `NotInitialized` covers operations requiring a PSF or successful manager
  initialization and initialization attempted without required collaborators
  or box state.
- `Runtime` covers a non-positive PSF atom count, unsupported child-manager
  operations in the base class, and propagated topology or parameter failures.
- `Cuda` covers stream creation, CUDA allocation, backend setup, transfer,
  graph, kernel, and synchronization failures.
- `NotImplemented` is raised when the force-calculation `reset` argument is
  true.

File parsing through `CharmmPSF` and `CharmmParameters` can raise categorized
runtime or invalid-argument errors. Host allocation can also raise standard
allocation exceptions.

Initialization, box propagation, and subscription-vector growth are not fully
transactional. Consult each symbol contract before assuming that failure
preserves all prior state.

### C ABI

Every status-returning function is guarded. Success returns
`APO_STATUS_OK` and clears the calling thread's previous diagnostic. Failure
returns the status corresponding to the native category and leaves nonempty
thread-local text available through [apo_last_error](@ref apo_last_error).

Invalid handles, pointers, buffer sizes, enum values, and scalar values return
`APO_STATUS_INVALID_ARGUMENT`. Missing native PSF state in
[apo_force_manager_get_num_atoms](@ref apo_force_manager_get_num_atoms) returns
`APO_STATUS_NOT_INITIALIZED`. Categorized CUDA failures return
`APO_STATUS_CUDA_ERROR`. Native allocation and uncategorized standard
exceptions return `APO_STATUS_RUNTIME_ERROR`.

Creation sets a valid output slot to `NULL` before collaborator validation.
The atom-count getter sets its output to zero before handle validation. Other
getters leave output storage unchanged on failure. Box and FFT getters write
only their first three output elements and leave trailing capacity unchanged.

[apo_force_manager_destroy](@ref apo_force_manager_destroy) accepts `NULL`, does
not throw across the ABI, and normally preserves a stale diagnostic.

### Python

Python argument conversion can raise `TypeError`, `ValueError`, or
`OverflowError` before the native call. Using a closed wrapper raises
`RuntimeError`. A nonzero native status raises `ApoCharmmError` with the
numeric status, status name, Python operation context, native diagnostic, and
rendered message retained by the exception.

## Important Behavior and Limitations

Force evaluation uses separate bonded, reciprocal, direct, subscribed, and
aggregate streams. The second force phase enqueues component work without
waiting. The third phase synchronizes each component stream before aggregate
force addition and synchronizes additional work required for requested energy
and virial output. Aggregate force storage is complete when the full
`calcForce()` call returns.

The first force-clear phase captures a CUDA graph that references the three
built-in fixed-point force arrays. The graph is reused on later evaluations.

`reset=true` is recognized but not implemented. Rebuild the neighbor list
through the dedicated native method where appropriate.

When `calcEnergy` is false, the prior aggregate energy remains observable and
is stale. When `calcVirial` is false, previous virial state can remain
observable. `getEnergyComponents()` reads current host mirrors and performs no
device transfer or synchronization.

`getPotentialEnergies()` is a legacy host-side sum. It currently omits CMAP and
subscribed-force energies and is not equivalent to the aggregate energy
produced by an energy-requesting force calculation.

For [PBC::P21](@ref PBC::P21), `getVirial()` divides the reciprocal virial
contribution by two before summation.

Mutable native accessors expose box, SHAKE, energy-virial, virial, and child
storage directly. Mutation bypasses validation, propagation, and lifecycle
logic.

`ctonnb` and `ctofnb` currently follow backend behavior rather than conventional
CHARMM spelling: `ctonnb` is forwarded as the outer/off distance and `ctofnb`
as the inner/on distance. Their relative ordering is not validated.

The base manager is not composite. Its child-manager and child-energy
operations reject use, although mutable child storage remains exposed for
legacy compatibility and derived implementations.

The subsystem is not thread-safe. Concurrent host configuration, force
evaluation, subscription, mutable access, or destruction requires external
synchronization.

## Related Subsystems

- [CharmmContext](@ref charmm_context) owns mutable molecular state and drives
  ForceManager initialization and calculation.
- [CharmmPSF](@ref charmm_psf) supplies atom, topology, mass, charge, exclusion,
  and constraint information.
- [CharmmParameters](@ref charmm_parameters) supplies bonded and nonbonded
  parameter records.
- `Force` stores component and aggregate device force arrays.
- `CudaEnergyVirial` stores named energy components and virial data.
- [CudaContainer](@ref cuda_container) describes host/device mirrored container
  behavior.
- `ForceManagerComposite` is the direct composite-manager extension.
- [HarmonicRestraintForce](@ref harmonic_restraint_force) and
  [HarmonicCenterOfMassRestraintForce](@ref harmonic_center_of_mass_restraint_force)
  are subscription clients.
- [ApoCharmmError](@ref apocharmm_error) describes native, C ABI, and Python
  error propagation.

## Developer Architecture

### Public and private layers

The public native contract is declared in `include/ForceManager.h` and
implemented in `src/ForceManager.cu`. [ForceView](@ref ForceView) is the
non-owning type-erasure adapter for subscribed force implementations.

The stable C ABI is declared in
`capi/include/apocharmm_c/ForceManager.h`. Its private owning handle is defined
in `capi/private_include/apocharmm_c/detail/ForceManagerHandle.h`, and guarded
operations are implemented in `capi/src/apocharmm_c/ForceManager.cpp`.

The Python wrapper in `python/apocharmm/force_manager.py` configures C function
prototypes, owns one C handle, retains Python collaborators, converts Python
values into fixed-width C types, and delegates native status handling to the
shared Python error infrastructure.

### Lifecycle and ownership graph

A successful native initialization has the following ownership graph:

```text
CharmmContext --shared--> ForceManager
CharmmContext <--weak---- ForceManager
ForceManager --shared--> CharmmPSF
ForceManager --shared--> CharmmParameters
ForceManager --unique--> bonded/reciprocal/direct backends
ForceManager --shared--> built-in stream holders and force arrays
ForceManager --shared--> each subscribed force and its resources
```

The C handle adds retained native owners for the PSF, parameters, and manager.
The Python wrapper adds strong Python references for the source collaborator
wrappers and subscribed force wrappers.

### State and lifecycle

The base constructor establishes configuration defaults and allocates
nine-element virial work containers. Collaborator setters establish input
state. Initialization allocates backend resources and changes the initialized
flag only at completion. Force calculation is split into clear, enqueue, and
aggregate phases. Destruction explicitly destroys CUDA streams and graph
objects through non-throwing cleanup before ordinary member destruction
releases remaining owners.

Changing the PSF, parameters, or PBC clears the initialized flag but does not
call the cleanup routine. Reinitialization therefore requires architectural
work before it can be considered a supported state transition.

### Internal invariants

The manager relies on these invariants:

- PSF and parameter objects describe the same atom and force-field model.
- `m_BoxDimensions` has three values and agrees with `m_BoxX`, `m_BoxY`, and
  `m_BoxZ`.
- Built-in streams, backends, and force arrays are all present when
  `m_IsInitialized` is true.
- All subscription vectors have the same length.
- Every subscription index describes one force, its stream, force storage,
  energy-virial storage, and tag.
- A cached clear graph refers to the currently active built-in force arrays and
  aggregate stream.
- SHAKE atom and parameter arrays have equal length.
- Component force arrays share a compatible atom count and force stride for
  aggregation and virial calculation.

Mutable public accessors and partial allocation failures can violate several of
these invariants.

### Host/device data flow

Coordinate-charge input is device-resident. Built-in and subscribed forces
write fixed-point device force arrays on their own streams. The aggregate phase
waits for each stream and converts/adds each source into a double-precision
device array on the manager stream.

Energy components are accumulated on the device. An energy-requesting
calculation copies each component energy-virial object to host storage, performs
a device synchronization, launches total-energy kernels on the manager stream,
and synchronizes that stream. Host aggregate energy is transferred only when
explicitly requested by surrounding code.

Virial work uses both host and device storage. `getVirial()` obtains component
arrays, transfers them to host, performs the aggregate sum on the host, and
transfers the result back to the device.

Box setters copy host values and delegate updates to constructed force objects.
The manager adds no explicit global synchronization around that propagation.

### Error boundaries

Native validation uses categorized `ApoCharmmError`. CUDA checks use the CUDA
category. The C ABI guard translates categories to exact `apo_status` values,
catches legacy standard exceptions, and stores a thread-local diagnostic.
Python status functions convert every nonzero status into one
`ApoCharmmError`.

Destruction is isolated through non-throwing native and C ABI cleanup paths.
CUDA destruction errors are not printed or rethrown.

### Extension points

Derived managers may override initialization, configuration propagation, force
calculation, potential-energy output, child-manager access, and composite
identity. The base `addForceManager()` and
`computeAllChildrenPotentialEnergy()` deliberately reject use.

A subscribed force need not derive from a common base class. It must satisfy
the compile-time interface consumed by [ForceView](@ref ForceView) and supply
matching stream, fixed-point force, and energy-virial shared owners. New force
types must also provide a stable `contributesVirial` constant.

The Python extension protocol consists of
`_subscribe_to_force_manager()` and `_unsubscribe_from_force_manager()`.
Native wrappers should route both callbacks through status-checked C ABI
functions and retain any collaborators required by the native force.

### Performance-sensitive behavior

Separate CUDA streams allow built-in force components to execute
independently. The cached clear graph reduces repeated force-clearing launch
overhead. Aggregate-force construction currently performs host-side stream
synchronization before each addition. Energy calculation uses a device-wide
synchronization before total-energy kernels. Virial calculation converts
fixed-point force arrays to double and introduces additional stream
synchronizations.

The `getPotentialEnergies()` compatibility path performs device-to-host copies
and a host-side sum. Prefer aggregate energy produced by the normal force
calculation path when possible.

### Relevant tests

- `test/unittests/unittest-forceManager.cpp` covers native defaults,
  validation, copying, initialization, force evaluation, subscriptions, box
  propagation, and error categories.
- `test/unittests/unittest-capiForceManager.cpp` covers C ABI ownership,
  validation, output state, status mapping, and thread-local diagnostic
  behavior.
- `test/pytest/python_api_force_manager.py` covers Python defaults,
  setter/getter round trips, enum conversion, close behavior, subscription
  retention, and Python/native exception boundaries.
- `test/pytest/python_api_charmm_context.py` exercises ForceManager through the
  higher-level Python context.
- `example/energy.py` demonstrates the context-centered energy workflow.

### Visible architectural constraints

The current implementation contains duplicated float and double box state,
mutable accessors that bypass invariants, a non-transactional initialization
path, non-transactional parallel subscription growth, a cached graph tied to
specific force buffers, a legacy incomplete host energy sum, and unresolved
public virial ordering and force-sign documentation. These constraints should
be preserved or explicitly resolved when reorganizing the subsystem.

## API Reference

- Native C++: [ForceManager](@ref ForceManager) and [ForceView](@ref ForceView).
- Public C ABI: [apo_force_manager](@ref apo_force_manager),
  [apo_force_manager_create](@ref apo_force_manager_create),
  [apo_force_manager_destroy](@ref apo_force_manager_destroy),
  [apo_force_manager_set_box_dimensions](@ref apo_force_manager_set_box_dimensions),
  and the remaining `apo_force_manager_*` configuration and query functions.
- Python: [python_force_manager](@ref python_force_manager).
- Related enums: [PBC](@ref PBC), [apo_pbc](@ref apo_pbc),
  [python_periodic_boundary_condition](@ref python_periodic_boundary_condition),
  and [python_vdw_type](@ref python_vdw_type).
