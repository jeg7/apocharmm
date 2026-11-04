# Subscriber {#subscriber}

## Purpose

The Subscriber subsystem schedules side effects during dynamics propagation.
A concrete [Subscriber](@ref Subscriber) writes selected simulation state when
an attached `CudaIntegrator` reaches the subscriber's reporting interval. The
interval is a positive, dimensionless number of propagated steps.

Use [DcdSubscriber](@ref DcdSubscriber) for coordinate trajectories and
[RestartSubscriber](@ref RestartSubscriber) for restart state. Those two
reporters are available through C++, the C ABI, and Python.

Subscribers are synchronous observers, not background tasks. An update runs on
the propagation caller's thread after the corresponding dynamics step and may
perform force evaluation, CUDA transfers, device-wide synchronization, and file
I/O before propagation continues.

## Quick Start

The Python example below writes two DCD frames while propagating four steps. It
uses stable repository test data and unsubscribes before the context manager
closes the DCD handle.

```python
import apocharmm as apo

prm = apo.CharmmParameters("toppar/toppar_water_ions.str")
psf = apo.CharmmPsf("test/data/nacl_pair.psf")
crd = apo.CharmmCrd("test/data/nacl_pair.cor")

ctx = apo.CharmmContext(psf, prm)
ctx.setBoxDimensions([50.0, 50.0, 50.0])
ctx.setCoordinates(crd)
ctx.useHolonomicConstraints(False)
ctx.assignVelocitiesAtTemperature(300.0)

integrator = apo.CudaNoseHooverIntegrator(0.001)
integrator.setReferenceTemperature(300.0)
integrator.setNoseHooverPistonMass(300.0)
integrator.useOldTemperature(False)
integrator.setCharmmContext(ctx)

with apo.DcdSubscriber("trajectory.dcd", 2) as dcd:
    integrator.subscribe(dcd)
    integrator.propagate(4)
    integrator.unsubscribe(dcd)
```

The equivalent direct C++ workflow uses shared ownership because native
subscription obtains a shared self-reference from the integrator.

```cpp
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaNoseHooverIntegrator.h"
#include "DcdSubscriber.h"

#include <filesystem>
#include <memory>
#include <vector>

int main() {
  const std::filesystem::path prmPath = "toppar/toppar_water_ions.str";
  const std::filesystem::path psfPath = "test/data/nacl_pair.psf";
  const std::filesystem::path crdPath = "test/data/nacl_pair.cor";

  const auto prm = std::make_shared<CharmmParameters>(prmPath);
  const auto psf = std::make_shared<CharmmPSF>(psfPath);
  const auto crd = std::make_shared<CharmmCrd>(crdPath);

  const auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(std::vector<double>{50.0, 50.0, 50.0});
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);
  ctx->assignVelocitiesAtTemperature(300.0);

  const auto integrator =
      std::make_shared<CudaNoseHooverIntegrator>(0.001);
  integrator->setReferenceTemperature(300.0);
  integrator->setNoseHooverPistonMass(300.0);
  integrator->useOldTemperature(false);
  integrator->setCharmmContext(ctx);

  const auto dcd = std::make_shared<DcdSubscriber>("trajectory.dcd", 2);
  integrator->subscribe(dcd);
  integrator->propagate(4);
  integrator->unsubscribe(dcd);

  return 0;
}
```

## Construction and Required State

A concrete file-writing subscriber validates its positive report frequency,
copies its `std::filesystem::path`, checks the nonempty parent component with
`std::filesystem::exists()`, and creates or truncates the output during
construction. [DcdSubscriber](@ref DcdSubscriber) opens binary output, while
[RestartSubscriber](@ref RestartSubscriber) uses text output.

Establish state in this order:

1. Construct and initialize the [CharmmContext](@ref CharmmContext). For restart
   import or export, explicitly select its crystal type with
   [setCrystalType()](@ref CharmmContext::setCrystalType). Choose the same type
   in the receiving simulation; equal box lengths do not change the selection.
2. Construct the concrete integrator and attach the context.
3. Construct the subscriber with its final path and report frequency.
4. Subscribe it to the integrator.
5. Propagate dynamics.
6. Unsubscribe before closing or destroying either wrapper.

Native `CudaIntegrator::subscribe()` requires the integrator itself to be owned
by `std::shared_ptr`. It calls `Subscriber::setCharmmContext()` with the
integrator's current context, then `Subscriber::setIntegrator()`, then appends
the subscriber and a copy of its current frequency to parallel arrays. Set the
integrator context first. Subscribing while that context is null can succeed,
and a later context setter does not update the already attached subscriber.

The scheduling interval is a snapshot. Changing a subscriber's native
frequency after subscription changes the property read by some file metadata,
but it does not change the integrator's cached callback interval. Scheduling
follows CHARMM's `MOD(ISTEP,NSAVC)` rule with `ISTEP` counted from the run
origin: the completed-step count at the first `propagate()` call after
subscription. Successive `propagate(num_steps)` calls continue that run, so a
subscriber with frequency 100 is called at steps 100, 200, and so on relative
to its origin no matter how the steps are split across calls. Subscribing
again, reopening a DCD file, or importing a restart with
`initializeFromRestartFile()` starts a new run at the next propagation. A DCD
file that already holds frames rejects such a new run; open it under a new path
first, as CHARMM opens a new trajectory unit for a restarted run.

[DcdSubscriber](@ref DcdSubscriber) requires a live binary stream, a context
with a positive atom count, exactly three positive box lengths, and a valid
single-precision coordinate/charge device container.
[RestartSubscriber](@ref RestartSubscriber) additionally requires the subscriber
and integrator to retain the same context and supports only
`CudaNoseHooverIntegrator`, `CudaLangevinPistonIntegrator`, and
`CudaLangevinThermostatIntegrator`. Restart import and export require a context
configured as `CRYSTAL::CUBIC`, `CRYSTAL::TETRAGONAL`, or
`CRYSTAL::ORTHORHOMBIC`, with box dimensions consistent with that selection.
The type comes exclusively from the context. DCD output requires valid box
dimensions but does not require a configured crystal label.

## Ownership and Lifetime

At the native layer, an integrator retains every subscribed object through
`std::shared_ptr<Subscriber>`. The subscriber retains the context and integrator
through shared pointers. This produces the following strong ownership graph:

```text
CudaIntegrator -> Subscriber -> CudaIntegrator
                       |
                       +------> CharmmContext
```

`CudaIntegrator::unsubscribe()` erases the integrator-to-subscriber edge, but it
does not clear the subscriber's context or integrator pointers. Unsubscribe
before releasing final owners. The retained backlinks also mean that the same
native subscriber cannot currently be resubscribed through the normal
attachment path.

The public C ABI owns concrete handles such as
[apo_dcd_subscriber](@ref apo_dcd_subscriber) and
[apo_restart_subscriber](@ref apo_restart_subscriber). Each concrete handle
embeds an [apo_subscriber](@ref apo_subscriber) base view. Conversion returns a
borrowed pointer into that concrete allocation; it is not a separately owned
handle and has no destroy function. Destroying the concrete C handle invalidates
the base pointer even when a native integrator continues to retain the
underlying C++ object. Keep the concrete handle alive until after C ABI
unsubscription.

Python `DcdSubscriber` and `RestartSubscriber` wrappers own their concrete C
handles and store the borrowed base pointer. A Python `CudaIntegrator` appends a
successfully subscribed wrapper to its `_subscribers` list, preserving the C
handle and base-view lifetime. `unsubscribe()` removes that retained Python
reference after native removal. Closing a subscriber clears its borrowed base
view, so close only after unsubscription.

The subsystem performs no internal host-thread locking. Serialize propagation,
updates, frequency changes, attachment, unsubscription, stream operations, and
destruction involving shared objects.

## Data, Shapes, and AKMA Units

The reporting interval and frame counters are dimensionless step counts.
Integrator constructor time steps and public `getTimeStep()` values use
picoseconds. The DCD header converts that value to AKMA time units.

[DcdSubscriber](@ref DcdSubscriber) writes:

- one six-`double` unit-cell record per frame, with X, Y, and Z box lengths in
  elements 0, 2, and 5;
- three Fortran-style coordinate records containing all X values, all Y values,
  and all Z values;
- one IEEE-like native `float` per coordinate component in atom order;
- coordinates and box lengths in angstroms;
- no charge, velocity, mass, force, or energy values.

DCD record markers and integer fields use signed 32-bit integers; coordinates
use IEEE binary32 and cell entries use IEEE binary64. Output uses native byte
order. The first saved step is the completed integration step, frame count is
the number successfully written, and planned duration is the latest announced
endpoint minus the run origin, including unsaved trailing steps, exactly as a
CHARMM run of that length records `NSTEP`. The native integrator announces each
request through `onPropagationStart(startStep, numSteps)`; DCD owns the output
plan and can update its duration even when a request produces no new frame.

[RestartSubscriber](@ref RestartSubscriber) writes a version-52 CHARMM formatted
restart core and a trailing apoCHARMM RNG extension. The header line carries
the marker `APO` in columns 32-34, past every field CHARMM's `READYN` reads, so
CHARMM ignores the marker and never reaches the extension, while apoCHARMM
readers require the extension whenever the marker is present. Quantities
include:

| Section or value | Shape and order | Unit |
| --- | --- | --- |
| Box lengths | Three diagonal lengths in X, Y, Z order | Angstrom |
| `XOLD, YOLD, ZOLD` | One double-precision XYZ row per atom | Angstrom |
| `VX, VY, VZ` | One double-precision velocity row per atom | Angstrom per AKMA time unit |
| `X, Y, Z` | One previous leapfrog-displacement XYZ row per atom | Angstrom |
| Average temperature | One scalar | Kelvin |
| Atom, step, and degree-of-freedom fields | Scalar integer fields | Dimensionless |
| Standard scalar seed | D22.15 real field | Dimensionless |
| Native RNG seed and state | Exact integer/state text in the extension | Dimensionless |

Restart Nose-Hoover and Langevin-piston velocity/force state retains the
integrators' native CHARMM-compatible AKMA units. Numeric fields use D22.15
precision. Unavailable diagnostic histories are reset explicitly, including
the thermal-piston reference-energy origin. Matching atom order, potential,
constraints, timestep, and thermostat/piston settings remain the caller's
responsibility. Native stochastic imports restore the RNG extension; CHARMM
imports start a fresh stream from the caller's configured apoCHARMM seed. A
CHARMM restart written with `JHSTRT=0` (for example at the end of an
equilibration run whose last step rescaled velocities) is resumed as CHARMM
resumes it, through the two-step start that regenerates the previous
displacement from velocities and forces (`IGVOPT=2`). CHARMM's `NDEGF` is not
required to match the context, because CHARMM itself recomputes it after
reading a restart; files written by apoCHARMM must still match. The shared
reader `apo::ReadRestartFile` returns a `RestartSnapshot` that retains every
record of the supported layout except CHARMM's own random-number fields,
including the statistics apoCHARMM does not use, so a file can be inspected
or carried forward without reparsing.

## Errors

C++ constructors and base mutators use [ApoCharmmError](@ref ApoCharmmError).
Empty paths, nonexistent checked parent paths, and nonpositive frequencies use
`ApoCharmmErrorCode::InvalidArgument`. Missing required update state uses
`ApoCharmmErrorCode::NotInitialized`. Unsupported restart-integrator types use
`ApoCharmmErrorCode::NotImplemented`. File open/write and state-consistency
failures use `ApoCharmmErrorCode::Runtime`. CUDA transfers, kernels, and
synchronization failures use `ApoCharmmErrorCode::Cuda`.

C ABI status functions clear the calling thread's previous diagnostic at entry.
Native error categories map to the matching `APO_STATUS_*` value. Unexpected
standard or nonstandard exceptions map to `APO_STATUS_RUNTIME_ERROR`. On
failure, copy [apo_last_error](@ref apo_last_error) immediately on the same
thread; its pointer is borrowed and changes after the next diagnostic-changing
call on that thread. Concrete destroy functions are void, accept `NULL`, and do
not normally clear a previous diagnostic.

Python configures each status-returning function with a shared error callback.
Every nonzero status raises `ApoCharmmError` containing the numeric status,
symbolic status name, operation context, and copied native diagnostic. Python
also raises `TypeError` for wrapper/type or `ctypes` conversion failures,
`ValueError` for values outside a signed 32-bit C `int`, and `RuntimeError` for
closed wrappers or impossible successful-null-handle results.

A propagation failure is not transactional at any interface. The integrator
increments its completed-step count after each concrete step returns and before
reporting. The latest request length is independent metadata. A start hook can
update planned output metadata before a later hook or step fails; completed
dynamics, changed host/device state, and written bytes remain observable.

## Important Behavior and Limitations

- Subscriber callbacks run synchronously and can substantially increase the
  wall time of the propagation step that triggers them.
- DCD coordinate output and restart output perform device-to-host transfers that
  call `cudaDeviceSynchronize()` on the current device. They do not use a
  subscriber-specific CUDA stream.
- `Subscriber::setFilePath()` changes only the stored logical path. It does not
  reopen the stream. Mutable `getFilePath()` access bypasses all validation.
- DCD writes are not atomic. For scheduled output, `NSTEP` is the announced
  endpoint minus the run origin fixed at the first propagation after
  subscription, reopening, or restart import. Without a propagation
  announcement, manual snapshots record the observed span from the first
  snapshot.
- Restart formatting and device transfers finish before update truncation,
  but construction truncates immediately and output is not atomic. Unavailable
  diagnostic histories reset; previous-energy initialization fields use the
  CHARMM sentinel. `NSAVC` comes from attached DCD subscribers, or is zero when
  none is attached. Different simultaneous DCD frequencies are rejected;
  `NSAVV` remains zero because velocity trajectories are not written.

## Related Subsystems

- [CharmmContext](@ref charmm_context) supplies coordinates, velocities, box
  state, energies, and the force manager queried by subscribers.
- [ForceManager](@ref force_manager) supplies potential energies.
- [CudaContainer](@ref cuda_container) defines the explicit device-to-host
  transfers and synchronization used by output writers.
- [ApoCharmmError](@ref apocharmm_error) defines native errors, C ABI
  diagnostics, and Python error translation.
- [Coordinates](@ref coordinates) describes the coordinate inputs used to
  initialize the context before propagation.

## Developer Architecture

The native public layer begins at [Subscriber](@ref Subscriber). Its extension
point is the pure virtual `update()` method. A concrete reporter normally
delegates path, stream, frequency, context, and integrator storage to the base
and adds only the format-specific state required to write one update. Override
`openFile()` only when the format requires non-default stream flags or format
state reset, as [DcdSubscriber](@ref DcdSubscriber) does for binary output.

The scheduling collaborator is `CudaIntegrator`. It stores parallel
`m_Subscribers` and `m_ReportFreqList` arrays. Entry `i` in one must always
correspond to entry `i` in the other. Subscription appends both; unsubscription
erases both; `reportIfNeeded()` indexes both and invokes `update()`. Mutable
accessors expose these arrays, so callers can violate the invariant. A future
reorganization should replace the parallel representation with one attachment
record and define explicit backlink clearing and resubscription semantics.

The native implementation layer under `src/` performs DCD and restart format
generation and CUDA transfers. DCD writes header, unit-cell, and coordinate
records directly to `std::fstream`. Restart assembles one monolithic
CHARMM-style file and reads integrator-specific containers through dynamic
casts.

The C ABI exposes only the base operations, DCD writer, restart writer, and the
integrator attachment/propagation operations. Private handle structs store a
concrete `std::shared_ptr` and an embedded base struct that shares the same
native owner. C functions validate handles, translate exceptions with
`apocharmm_c::guard`, and never expose the private C++ layout as a stable public
contract.

The Python layer mirrors that mapping. Concrete wrappers own their C handles,
base methods operate on borrowed embedded views, and the integrator retains
Python wrappers to preserve those views. Status translation is centralized in
`configure_status_function()` rather than duplicated by subscriber wrappers.

Host/device data flow is explicit. DCD copies `CudaContainer<float4>` device
coordinates to host before splitting components. Restart copies each required
context or integrator container to host before text formatting. These transfers
are blocking at the device level.

The error boundary is layered. Public C++ contracts name native categories.
`apocharmm_c::guard` maps those categories and captures diagnostics. Python
`errcheck` immediately copies the thread-local C diagnostic into an owned
exception string. Destroy functions use a separate non-throwing boundary because
they cannot return status.

Relevant tests are registered in `test/unittests/CMakeLists.txt`. Native
coverage is concentrated in `unittest-subscriber.cpp`,
`unittest-dcdSubscriber.cpp`, `unittest-restartSubscriber.cpp`, and
`unittest-cudaIntegrator.cpp`. Direct C ABI coverage uses the corresponding
`unittest-capi*.cpp` files. Python coverage uses the subscriber, DCD-subscriber,
restart-subscriber, and CUDA-integrator pytest modules.

## API Reference

C++:

- [Subscriber](@ref Subscriber)
- [DcdSubscriber](@ref DcdSubscriber)
- [RestartSubscriber](@ref RestartSubscriber)

C ABI:

- [apo_subscriber](@ref apo_subscriber)
- [apo_subscriber_set_report_frequency](@ref apo_subscriber_set_report_frequency)
- [apo_subscriber_get_report_frequency](@ref apo_subscriber_get_report_frequency)
- [apo_dcd_subscriber](@ref apo_dcd_subscriber)
- [apo_dcd_subscriber_create](@ref apo_dcd_subscriber_create)
- [apo_dcd_subscriber_create_with_report_frequency](@ref apo_dcd_subscriber_create_with_report_frequency)
- [apo_dcd_subscriber_destroy](@ref apo_dcd_subscriber_destroy)
- [apo_dcd_subscriber_as_subscriber](@ref apo_dcd_subscriber_as_subscriber)
- [apo_restart_subscriber](@ref apo_restart_subscriber)
- [apo_restart_subscriber_create](@ref apo_restart_subscriber_create)
- [apo_restart_subscriber_create_with_report_frequency](@ref apo_restart_subscriber_create_with_report_frequency)
- [apo_restart_subscriber_destroy](@ref apo_restart_subscriber_destroy)
- [apo_restart_subscriber_as_subscriber](@ref apo_restart_subscriber_as_subscriber)

Python:

- [python_subscriber](@ref python_subscriber)
- [python_dcd_subscriber](@ref python_dcd_subscriber)
- [python_restart_subscriber](@ref python_restart_subscriber)
