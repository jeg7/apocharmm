# CUDA Integrators {#cuda_integrators}

## Purpose

The CUDA-integrator subsystem advances a [CharmmContext](@ref CharmmContext)
through molecular-dynamics steps on a CUDA device.
[CudaIntegrator](@ref CudaIntegrator) provides the common context, time-step,
restart, subscriber, step-counter, neighbor-list, center-of-mass, and
CUDA-stream lifecycle. It is a polymorphic base that delegates initialization
and each dynamics step to a concrete integrator.

The currently documented concrete choices are:

- [CudaNoseHooverIntegrator](@ref CudaNoseHooverIntegrator) for deterministic
  constant-temperature propagation with a Nose-Hoover extended variable.
- [CudaLangevinThermostatIntegrator](@ref CudaLangevinThermostatIntegrator) for
  stochastic constant-temperature propagation with one Philox random-number
  state per atom.
- [CudaLangevinPistonIntegrator](@ref CudaLangevinPistonIntegrator) for
  pressure-controlled propagation with cubic, tetragonal, or orthorhombic box
  degrees of freedom and optional Nose-Hoover temperature control.

The native base class can be constructed directly, but its default
initialization, restart, descriptor, and single-step hooks report
`ApoCharmmErrorCode::NotImplemented`. Normal user code constructs one of the
concrete classes.

## Quick Start

The following Python example propagates a small system with the stochastic
Langevin thermostat:

```{.py}
import apocharmm as apo

parameters = apo.CharmmParameters("toppar/toppar_water_ions.str")
psf = apo.CharmmPsf("test/data/nacl_pair.psf")
coordinates = apo.CharmmCrd("test/data/nacl_pair.cor")

context = apo.CharmmContext(psf, parameters)
context.setBoxDimensions([50.0, 50.0, 50.0])
context.setCoordinates(coordinates)
context.useHolonomicConstraints(False)
context.setRandomSeed(314159)
context.assignVelocitiesAtTemperature(300.0)

integrator = apo.CudaLangevinThermostatIntegrator(0.001)
integrator.setReferenceTemperature(300.0)
integrator.setThermostatFriction(1.0)
integrator.setThermostatRngSeed(987654321)
integrator.setCharmmContext(context)
integrator.propagate(100)

print(integrator.getAverageTemperature())
```

The time step is `0.001` ps, the target temperature is 300 K, and the
thermostat friction is 1 ps^-1. After successful propagation, the final line
prints the finite running-average temperature maintained by the integrator.

The corresponding native workflow is:

```cpp
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaLangevinThermostatIntegrator.h"

#include <filesystem>
#include <memory>

int main() {
  const std::filesystem::path prmPath = "toppar/toppar_water_ions.str";
  const std::filesystem::path psfPath = "test/data/nacl_pair.psf";
  const std::filesystem::path crdPath = "test/data/nacl_pair.cor";

  auto parameters = std::make_shared<CharmmParameters>(prmPath);
  auto psf = std::make_shared<CharmmPSF>(psfPath);
  auto coordinates = std::make_shared<CharmmCrd>(crdPath);

  auto context = std::make_shared<CharmmContext>(psf, parameters);
  context->setBoxDimensions({50.0, 50.0, 50.0});
  context->setCoordinates(coordinates);
  context->useHolonomicConstraints(false);
  context->setRandomSeed(314159);
  context->assignVelocitiesAtTemperature(300.0);

  auto integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(0.001);
  integrator->setReferenceTemperature(300.0);
  integrator->setThermostatFriction(1.0);
  integrator->setThermostatRngSeed(987654321ULL);
  integrator->setCharmmContext(context);
  integrator->propagate(100);

  return 0;
}
```

Repository-relative `test/data` paths are suitable for repository tests and
examples. Applications should supply their own topology, parameter, and
coordinate paths.

## Construction and Required State

Public time-step arguments use picoseconds and must be finite and greater than
zero. The base implementation stores the value internally in AKMA time by
dividing by `0.0488882129`; [getTimeStep()](@ref CudaIntegrator::getTimeStep)
converts it back to picoseconds.

Configure the time step and parameters that do not require attached state before
attaching the context. Set crystal-dependent piston masses and friction after
attachment. [setCharmmContext()](@ref CudaIntegrator::setCharmmContext) may be
called only once. It retains the context, resizes the base per-atom work arrays,
constructs any configured holonomic-constraint helper, and immediately invokes
the concrete initialization hook. A failure clears the retained context and
constraint helper, but previously resized work storage can remain observable.

A context used for propagation must have a non-null, initialized
[ForceManager](@ref ForceManager). The context must also contain the
coordinates, velocity/inverse-mass state, atom count, degree-of-freedom count,
box state, and other force-manager inputs required by the selected force
configuration. [propagate()](@ref CudaIntegrator::propagate) rejects a
non-positive number of steps and checks context and force-manager readiness
before changing the trajectory.

For [CudaLangevinPistonIntegrator](@ref CudaLangevinPistonIntegrator), call
[CharmmContext::setCrystalType()](@ref CharmmContext::setCrystalType) before
attaching that context. The context owns the crystal selection; the piston
integrator has no crystal setter or independent crystal value. The supported
layouts are:

- `CRYSTAL::CUBIC`: one isotropic box degree of freedom.
- `CRYSTAL::TETRAGONAL`: one shared X/Y degree of freedom and one Z degree of
  freedom.
- `CRYSTAL::ORTHORHOMBIC`: independent X, Y, and Z degrees of freedom.

Context attachment allocates the crystal-dependent piston arrays and computes
default piston masses. Set custom Langevin-piston masses and friction after
attachment. A custom mass vector must contain exactly one, two, or three values
according to the context's layout. Reinitializing the same crystal preserves
these configured masses and friction. Both initialization and each piston step
validate that the context's finite positive box matches its declared symmetry;
equal edge lengths do not relabel an explicitly orthorhombic context.

Change a context's crystal type only between completed propagation calls.
For example, after cubic equilibration, select `CRYSTAL::ORTHORHOMBIC` on the
same context, call `integrator.initialize()`, then supply any custom three-axis
piston masses before starting the next phase. Reinitialization is available as
native `CudaIntegrator::initialize()`, C `apo_cuda_integrator_initialize()`, and
Python `CudaIntegrator.initialize()`. Box dimensions must satisfy the
new type; equal lengths are valid orthorhombic dimensions. Calls involving one
context/integrator must be serialized, including subscriber callbacks.

A type change makes the attached piston's derived layout stale. Piston state
getters/setters, inherited displacement-history access, restart loading, and
propagation reject the stale layout with `NotInitialized` before using it.
`getCrystalType()` continues to report the context's current type. Call
`initialize()` to rebuild the layout; do not reuse previously borrowed native
container aliases across the transition. An initialization failure also blocks
state access until initialization succeeds.

After a type change, call `initialize()` on the attached piston integrator,
then set custom masses for the new one-, two-, or three-axis layout if the
defaults are not wanted; friction only needs setting again when the next phase
needs a new coefficient. Particle velocities and the completed-step counter
survive the transition, but this begins a newly initialized phase, not an exact
continuation of the previous piston history, and restart import/export must
wait until reinitialization has completed. The exact list of state that a
layout change resets and retains is documented on
[initializeImpl()](@ref CudaLangevinPistonIntegrator::initializeImpl).

A successful restart import also clears the run origin of every attached
subscriber, so the next `propagate()` begins a new CHARMM-style run at the
imported step.

Attaching a context initializes the concrete integrator automatically.
[initialize()](@ref CudaIntegrator::initialize) repeats concrete
initialization and can overwrite derived displacement, random-number, kinetic,
or extended-variable state. Use
[initializeFromRestartFile()](@ref CudaIntegrator::initializeFromRestartFile)
after attaching the context when restoring a saved trajectory.

[setTimeStep()](@ref CudaIntegrator::setTimeStep) changes only the base
time-step scalar. It does not rebuild an existing holonomic-constraint helper or
recalculate every derived coefficient and displacement history. Treat the time
step as immutable after context attachment.

## Ownership and Lifetime

A native integrator retains its [CharmmContext](@ref CharmmContext) with a
`std::shared_ptr`. It also retains every subscribed
[Subscriber](@ref Subscriber) with a `std::shared_ptr`. Subscription installs
context and integrator backlinks in the subscriber, so the native graph can
contain a strong-reference cycle. Unsubscribe objects explicitly before
releasing the last external integrator and subscriber owners.

Each native integrator owns two CUDA stream handles through shared pointers.
The concrete stochastic integrators additionally own raw CUDA allocations for
Philox random-number states. Their destructors are non-throwing and suppress
CUDA cleanup errors.

The C ABI concrete handle owns a shared native concrete integrator and embeds
an `apo_cuda_integrator` base view. Conversion functions ending in
`_as_cuda_integrator` return a borrowed pointer to that embedded view. The
borrowed view has no destroy function and becomes invalid when its concrete
handle is destroyed.

`apo_cuda_integrator_set_charmm_context()` borrows the C context handle for one
call but retains the shared native context on success. The source context handle
may then be destroyed without invalidating the integrator. Subscription
similarly retains the native subscriber, but the concrete C subscriber handle
must remain alive when its embedded base view will later be passed to
unsubscription.

Each Python concrete wrapper owns its concrete C handle and stores a borrowed
base-integrator view. The base wrapper retains the supplied Python context and
successfully subscribed Python subscriber wrappers. `close()` destroys the
concrete C handle, invalidates both handle properties, and clears those Python
references. It does not implicitly unsubscribe native subscribers; unsubscribe
before closure.

Native, C ABI, and Python objects provide no locking. Do not overlap mutation,
initialization, propagation, restart loading, subscription, unsubscription, or
destruction involving the same integrator or context from multiple host
threads.

## Data, Shapes, and AKMA Units

The public quantities whose units are established by the current
implementation are:

| Quantity | Public representation | Unit |
|---|---|---|
| Time step | Scalar | picoseconds |
| Internal time step | Native protected scalar | AKMA time |
| Reference and measured temperature | Scalar | kelvin |
| Thermostat and piston friction | Scalar | inverse picoseconds |
| Kinetic energy | Two-element native container | kilocalories per mole |
| Pressure tensor | Nine row-major values | atmospheres |
| Pressure scalar | One value | atmospheres |
| Box and piston position state | Scalar arrays | angstroms |
| Piston on-step velocity | Scalar array | angstroms per AKMA time |
| Piston delta pressure | Scalar array | atmospheres per angstrom |
| Instantaneous surface tension | Scalar | dyne per centimeter |
| Seeds, sequence positions, frequencies, and iteration counts | Integer | dimensionless |

The exact dimensional convention for the public Nose-Hoover coupling mass,
Nose-Hoover extended-variable velocity/increment values, and
Langevin-piston mass is not stated by the current repository. Their setters and
getters therefore document only validation, shape, copying, and the native
numeric convention.

The base displacement arrays are [CudaContainer](@ref CudaContainer) objects
with one `double4` record per atom. The `x`, `y`, and `z` components are
angstrom-valued displacement or reference-coordinate components. The
integrators do not assign public meaning to the `w` component. Native accessors
return borrowed container aliases and perform no implicit host/device transfer.

Nose-Hoover and Langevin-piston kinetic-energy and running-temperature arrays
have two elements:

- Element 0 is the three-point “new” or “Jung” estimator.
- Element 1 is the on-step “old” estimator.

The native instantaneous-temperature methods select element 1 when
`useOldTemperature(true)` is active and element 0 otherwise. The current
Nose-Hoover and Langevin-piston C ABI running-average getters select the
opposite indices: they return element 0 when the old-temperature flag is true
and element 1 when it is false. Their Python wrappers inherit that current C
ABI behavior.

The Langevin thermostat always reports element 0 through its C ABI and Python
running-average and instantaneous-temperature methods.

Piston pressure tensors are flat, row-major 3-by-3 arrays in this order:

```text
xx xy xz
yx yy yz
zx zy zz
```

The Python piston wrapper accepts an outer iterable of inner iterables and
flattens all values in iteration order. Native validation requires exactly nine
total values but the current wrapper does not independently require three rows
of three values. Tensor getters return newly allocated three-by-three Python
lists.

The Langevin thermostat maintains one Philox state per atom. The Langevin
piston maintains one state per active crystal degree of freedom. Native
serialized RNG-state strings are opaque implementation representations; callers
may save and restore them through the native API but should not parse or modify
their contents.

## Errors

### Native C++

Verified validation and lifecycle failures use
[ApoCharmmError](@ref ApoCharmmError):

- `ApoCharmmErrorCode::InvalidArgument` reports invalid scalar ranges, null or
  duplicate attachments, invalid array lengths, unsupported crystal values,
  incompatible restart state, and invalid propagation counts.
- `ApoCharmmErrorCode::NotInitialized` reports missing context, force-manager,
  crystal, RNG, or other required state.
- `ApoCharmmErrorCode::NotImplemented` reports direct use of unimplemented base
  hooks and unsupported subscriber operations.
- `ApoCharmmErrorCode::Runtime` reports restart-file open, format, parse, and
  state-shape failures, NaN detection, and other explicitly categorized runtime
  failures.
- `ApoCharmmErrorCode::Cuda` reports checked CUDA allocation, copy, stream,
  synchronization, and kernel-launch failures.

Host allocation and string construction can additionally propagate standard
exceptions such as `std::bad_alloc`, `std::length_error`, or
`std::system_error`. Subscriber implementations retain a small number of
legacy `std::invalid_argument` and `std::runtime_error` paths.

Initialization, restart loading, propagation, and multi-subscriber operations
are not transactional. State changes, completed steps, updated counters, and
written output from before a failure remain observable.

### C ABI

Every status-returning integrator function clears the calling thread's previous
diagnostic at entry. Native `ApoCharmmErrorCode` values map to their
corresponding `apo_status`; `std::invalid_argument` maps to
`APO_STATUS_INVALID_ARGUMENT`; other C++ exceptions map to
`APO_STATUS_RUNTIME_ERROR`.

After a nonzero status, call [apo_last_error()](@ref apo_last_error) immediately
on the same thread. The returned diagnostic pointer is borrowed and remains
valid only until the next diagnostic-changing C ABI call on that thread or
thread exit.

Owned-handle output slots are set to `NULL` before later construction or
validation. Scalar and fixed-buffer getters write their output only after
handle, pointer, shape, and transfer operations succeed unless the individual
function documents a different count-output rule. Destroy functions accept
`NULL`, do not throw, and preserve an existing diagnostic when destruction
succeeds.

### Python

Python wrapper checks raise `TypeError` for rejected object types or native
scalar conversions and `ValueError` for explicit integer-range checks. Access
through a closed wrapper raises `RuntimeError`. File-path encoding can raise
`TypeError`, `ValueError`, or `OSError` according to `os.fsencode()`.

Every nonzero C ABI status is raised as [apocharmm.ApoCharmmError](
@ref python_apocharmm_error_class) with the native status and diagnostic.
Python array inputs are copied into temporary C buffers. Tensor, mass, and
scalar outputs are returned as new Python values and do not alias native
storage.

## Important Behavior and Limitations

[propagate()](@ref CudaIntegrator::propagate) resets the context neighbor list
before the loop, sets the per-call neighbor-list counter to zero, stores the
requested length, fixes the run origin of any subscriber that does not have one
yet, and announces the starting step and requested length to each subscriber.
The cumulative step count advances only after a concrete step returns
successfully. A failing step can still have partly modified the state.

[getCurrentPropagatedStep()](@ref CudaIntegrator::getCurrentPropagatedStep)
derives its compatibility value from the cumulative count modulo `INT_MAX + 1`;
it does not maintain a separate clock. Internal center-of-mass and debug
schedules use the full count and do not restart at that wrap boundary. Calling
[propagateOneStep()](@ref CudaIntegrator::propagateOneStep) directly remains
uncounted and does not announce a run, invoke scheduled subscribers, reset the
neighbor list, or perform the periodic NaN check.

Subscriber reporting intervals are copied into a parallel scheduling array at
subscription time. Changing a subscriber's own reporting frequency afterward
does not change the cached schedule. A subscriber is updated when the number of
steps completed since its run origin is divisible by its interval, which is
CHARMM's `MOD(ISTEP,NSAVC)` test; successive `propagate()` calls continue the
same run rather than restarting the count at one.

The default nonbonded-list update interval is 20 steps and the default
center-of-mass-removal interval is 1000 steps. Because the cumulative counter
starts at zero, each concrete integrator performs its center-of-mass operation
during the first propagated step. These operations can synchronize the
integrator stream and transfer velocity or displacement storage between device
and host.

Concrete initialization and each complete concrete step synchronize the private
integrator stream at their documented completion boundary. Intermediate work is
issued asynchronously on that stream. Context force calculations and explicit
[CudaContainer](@ref CudaContainer) transfers can introduce additional stream
or device-wide synchronization. The separately allocated integrator memcpy
stream is not used by the three inspected concrete implementations.

The Langevin thermostat advances its recorded RNG sequence position by four per
propagated step because each per-atom Philox call consumes a `curand_normal4`
group. The Langevin piston advances its position once per predictor-corrector
iteration, so one propagated step advances it by the configured maximum
predictor-corrector iteration count.

Changing a Langevin thermostat seed or sequence position reinitializes its
per-atom RNG states when a context is attached. Changing a piston seed or
sequence position reinitializes its RNG states only after context attachment
has established the piston degree-of-freedom count.

Mutable native container accessors bypass shape, coherence, and initialization
invariants. Resizing or replacing kinetic-energy, temperature, pressure,
extended-variable, or displacement arrays can cause later kernels or C ABI
getters to read invalid layouts. They are contributor-level escape hatches, not
ordinary configuration APIs.

The current implementation has several material limitations:

- Zero is accepted as a Nose-Hoover coupling mass even though active propagation
  divides by that value.
- The piston constant-surface-tension flag has no public target-surface-tension
  setter, and the target container is not initialized after allocation.
- Piston half-step velocity storage is allocated and exposed but is not updated
  by the current propagation path.
- The two native `getAverageOldTemperature()` piston declarations have no
  definitions.
- Implicit native copies can share stream handles or duplicate raw RNG pointers.
  Treat all four native integrator classes as non-copyable until copy and move
  semantics are made explicit.

## Related Subsystems

- [CharmmContext](@ref charmm_context) owns the molecular state and coordinates
  force evaluation.
- [ForceManager](@ref force_manager) provides the force backend whose
  initialized state is required for propagation.
- [Subscriber](@ref subscriber) documents output scheduling, cached reporting
  intervals, backlinks, and lifecycle behavior.
- [CudaContainer](@ref cuda_container) documents the host/device mirror
  containers returned by native integrator state accessors.
- [ApoCharmmError](@ref apocharmm_error) documents native error categories,
  source locations, C ABI status mapping, and Python exception behavior.

## Developer Architecture

The public native declarations are in:

```text
include/CudaIntegrator.h
include/CudaNoseHooverIntegrator.h
include/CudaLangevinThermostatIntegrator.h
include/CudaLangevinPistonIntegrator.h
```

Their implementations and CUDA kernels are in the correspondingly named files
under `src/`. The public C ABI is declared under
`capi/include/apocharmm_c/`, implemented under `capi/src/apocharmm_c/`, and
backed by private handle structs under
`capi/private_include/apocharmm_c/detail/`. Python wrappers under
`python/apocharmm/` use `ctypes` and those C ABI entry points.

The native lifecycle is:

1. Construct the context and concrete integrator.
2. Configure the time step and parameters that do not require piston storage.
3. For a piston, select the crystal type on the context.
4. Call `setCharmmContext()`; the base and concrete initialization allocate the
   required state.
5. For a piston, set any custom piston masses and friction.
6. Propagate or restore a restart after validating its context compatibility.

A concrete extension derives from [CudaIntegrator](@ref CudaIntegrator) and
implements `initializeImpl()`, `initializeFromRestartFileImpl()`, and
`propagateOneStepImpl()`. The base final methods enforce the common lifecycle
and readiness boundary. An extension must preserve the invariant that
`m_CoordsRef`, `m_CoordsDelta`, and `m_CoordsDeltaPrevious` contain one
`double4` record per context atom after successful attachment.

The base owns the scheduling arrays. Entry `i` of `m_Subscribers` must always
correspond to entry `i` of `m_ReportFreqList`. The mutable native accessors to
those arrays are retained for compatibility but can violate this invariant.

The deterministic Nose-Hoover and piston paths derive two kinetic estimators
from previous-half-step displacement, on-step velocity, and next-half-step
displacement. A predictor-corrector loop updates the Nose-Hoover extended
variable. The piston path additionally computes kinetic and virial pressure,
updates crystal degrees of freedom, scales coordinates and box dimensions, and
repeats the piston/pressure correction for the configured iteration count.

The stochastic thermostat updates positions with Philox normal variates,
applies holonomic constraints when enabled, derives velocities from corrected
displacements, and computes both kinetic estimators. The stochastic piston uses
one Philox state per crystal degree of freedom and applies random forcing inside
the predictor-corrector loop.

Host/device flow is explicit. Kernels consume context and integrator device
arrays. C ABI scalar and buffer getters generally call `transferToHost()` before
reading a native `CudaContainer`. Restart readers parse host strings and arrays,
then transfer restored state to the device. No automatic dirty-state tracking
exists between mutable native aliases and later consumers.

The native `ApoCharmmError` boundary is inside the C++ implementation. The C ABI
guard translates all escaping C++ failures into `apo_status` and thread-local
diagnostics. The Python shared-library wrapper installs a status error checker,
so nonzero statuses become `ApoCharmmError` before wrapper methods continue to
their post-call ownership updates.

Performance-sensitive operations include context attachment, holonomic setup,
RNG allocation and initialization, force calculation, neighbor-list rebuilding,
full per-atom device copies, pressure reductions, predictor-corrector loops,
subscriber transfers, and stream synchronization. Repeated parameter changes
that trigger reinitialization should not be placed in a propagation loop.

The focused native regression files are:

```text
test/unittests/unittest-cudaIntegrator.cpp
test/unittests/unittest-cudaNoseHooverIntegrator.cpp
test/unittests/unittest-cudaLangevinThermostatIntegrator.cpp
test/unittests/unittest-cudaLangevinPistonIntegrator.cpp
```

Direct C ABI coverage exists in:

```text
test/unittests/unittest-capiCudaIntegrator.cpp
test/unittests/unittest-capiCudaLangevinThermostatIntegrator.cpp
test/unittests/unittest-capiCudaLangevinPistonIntegrator.cpp
```

There is currently no direct C ABI Nose-Hoover integrator test. Python API
coverage is in the four `python_api_cuda_*integrator.py` files under
`test/pytest/`. End-to-end examples are:

```text
example/nvt_nose_hoover.py
example/nvt_langevin_thermostat.py
example/npt_langevin_piston.py
```

## API Reference

### Native C++

- [CudaIntegrator](@ref CudaIntegrator)
- [CudaNoseHooverIntegrator](@ref CudaNoseHooverIntegrator)
- [CudaLangevinThermostatIntegrator](@ref CudaLangevinThermostatIntegrator)
- [CudaLangevinPistonIntegrator](@ref CudaLangevinPistonIntegrator)
- [CRYSTAL](@ref CRYSTAL)

### C ABI

- [apo_cuda_integrator](@ref c_apo_cuda_integrator)
- [apo_cuda_nose_hoover_integrator](@ref c_apo_cuda_nose_hoover_integrator)
- [apo_cuda_langevin_thermostat_integrator](@ref c_apo_cuda_langevin_thermostat_integrator)
- [apo_cuda_langevin_piston_integrator](@ref c_apo_cuda_langevin_piston_integrator)
- [apo_crystal_type](@ref apo_crystal_type)

### Python

- [apocharmm.CudaIntegrator](@ref python_cuda_integrator)
- [apocharmm.CudaNoseHooverIntegrator](@ref python_cuda_nose_hoover_integrator)
- [apocharmm.CudaLangevinThermostatIntegrator](@ref python_cuda_langevin_thermostat_integrator)
- [apocharmm.CudaLangevinPistonIntegrator](@ref python_cuda_langevin_piston_integrator)
- [apocharmm.CrystalType](@ref python_crystal_type)
