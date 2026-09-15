# CudaContainer {#cuda_container}

## Purpose

`CudaContainer<T>` is apoCHARMM's native owning container for a contiguous,
one-dimensional host array paired with a contiguous CUDA-device array. Use it
when native C++ or CUDA code needs the same logical data set on both sides of
the host/device boundary and explicit synchronization is acceptable. Use
[DeviceVector](@ref DeviceVector) when only device ownership and vector-like
capacity management are needed.

The two arrays are mirrors by convention, not continuously coherent storage.
Host element access changes only the host array. Device kernels change only the
device array. Constructors, assignments, `set()` operations, and explicit
transfer operations establish coherence at the points documented by the API.

`CudaContainer` is currently a native C++ subsystem. The repository does not
expose it through the public C ABI or the Python package.

## Quick Start

The following complete native example initializes both mirrors, changes the host
copy, transfers that change to the device, overwrites the host value, and then
restores the device value to the host:

```cpp
#include "CudaContainer.h"

#include <vector>

int main() {
  CudaContainer<double> values(std::vector<double>{1.0, 2.0, 3.0});

  values.at(1) = 4.0;
  values.transferToDevice();

  values.at(1) = 0.0;
  values.transferToHost();

  return values.at(1) == 4.0 ? 0 : 1;
}
```

Link the example against apoCHARMM and the CUDA runtime. It returns zero when
the round trip succeeds. `at()` accesses host memory only; neither host write in
the example is visible on the device until `transferToDevice()` is called.

## Construction and Required State

The default constructor creates two empty mirrors. The size constructor gives
both mirrors the requested active length, value-initializes the host elements,
and leaves the device elements uninitialized. Call `set()` or perform an
explicit transfer before consuming those device values.

Construction from `std::vector<T>` deep-copies the host values and transfers a
nonempty active range to a new device allocation. Construction from
[DeviceVector](@ref DeviceVector) deep-copies the active device range and
transfers a nonempty range to a new host vector. Nonempty transfers finish with
`cudaDeviceSynchronize()`; empty sources issue no transfer or synchronization.
Copy construction from another `CudaContainer<T>` instead copies the two source
mirrors independently and does not reconcile existing divergence between them.

Rvalue construction from `std::vector<T>` transfers its host allocation and
creates the device mirror. Rvalue construction from `DeviceVector<T>` transfers
its device allocation and creates the host mirror. These converting operations
can still throw while allocating or populating the missing mirror.

Move construction from another `CudaContainer<T>` transfers both mirrors without
allocation, copying, transfer, or synchronization. Existing divergence between
the source mirrors is preserved, and the moved-from container becomes empty.

Normal successful modifiers maintain equal active lengths. `resize()` changes
both lengths but does not copy values between the mirrors: new host elements are
value-initialized and new device elements contain unspecified bytes. Mutable
references returned by `getHostArray()` and `getDeviceArray()` can change one
length independently. Before any transfer or device print, restore the invariant

```cpp
container.getHostArray().size() == container.getDeviceArray().size()
```

and ensure that both nonempty mirrors own storage for that active range.

CUDA operations use the CUDA runtime state current on the calling thread. The
container stores neither a device identifier nor a stream. Establish the
intended CUDA device before construction and before later operations that touch
the device allocation.

## Ownership and Lifetime

A `CudaContainer<T>` exclusively owns its `std::vector<T>` host mirror and its
[DeviceVector](@ref DeviceVector) device mirror. Copying creates independent
storage. Rvalue conversion from a host or device vector transfers the compatible
source allocation. Moving another `CudaContainer<T>` transfers both mirrors and
preserves any existing divergence. No accessor or setter transfers ownership.

`at()` and `operator[]` return borrowed references to host elements.
`getHostArray()` and `getDeviceArray()` return borrowed references to the member
containers themselves. Those member-container references remain valid until the
owning `CudaContainer` is destroyed. Pointers, iterators, and element references
obtained from them follow the invalidation rules of `std::vector` and
[DeviceVector](@ref DeviceVector). In particular, growth, assignment, clearing,
destruction, and capacity-changing operations can invalidate previously borrowed
addresses.

The destructor is `noexcept`. Host storage is released normally. The nested
`DeviceVector` attempts `cudaFree`, discards the CUDA status, clears its
metadata, and does not publish a cleanup diagnostic. Use `clear()` when an
explicit device cleanup failure must be reported before destruction.

## Data, Shapes, and AKMA Units

Each mirror is one contiguous array of active elements in index order
`[0, size())`. The host mirror uses `std::vector<T>` storage and the device
mirror uses [DeviceVector](@ref DeviceVector) storage. `CudaContainer` adds no
multidimensional shape, stride, transposition, or component reordering. CUDA
vector types such as `float4` retain the binary layout and `x`, `y`, `z`, `w`
component order defined by the CUDA type.

`size()` and element indices are dimensionless counts. The container does not
assign physical meaning or AKMA units to `T`. The owning subsystem determines
whether an element represents a coordinate, velocity, force, parameter, index,
or another quantity and therefore determines its units. Transfers preserve the
stored binary representation without unit conversion.

## Errors

Bounds-checked host access reports an invalid index by throwing
[ApoCharmmError](@ref ApoCharmmError) with
`ApoCharmmErrorCode::InvalidArgument`. CUDA allocation, copy, cleanup, immediate
launch-check, and synchronization failures are converted to
`ApoCharmmErrorCode::Cuda` by the native CUDA checker. Construction of native or
CUDA diagnostics can additionally propagate `std::bad_alloc` or
`std::length_error`. Host-vector allocation and length failures propagate those
same standard exceptions directly. See [ApoCharmmError](@ref apocharmm_error)
for native error formatting and captured source-location behavior.

Operations that update both mirrors are sequential rather than transactional.
For example, host-vector assignment changes the host mirror before device
resizing and copying, while device-vector assignment changes the device mirror
before host resizing and copying. A later failure can therefore leave different
lengths or values observable in the two mirrors. Re-establish a known state with
`clear()`, a successful `set()` call, or another fully successful assignment
before relying on mirror coherence.

Rvalue conversion assignment constructs a complete temporary replacement before
changing the destination. If replacement construction fails, the destination is
unchanged. A failure after the source allocation has been transferred can still
leave the source empty. Same-type move construction and assignment do not
allocate or perform a CUDA transfer. Same-type move assignment releases the
destination's former device allocation through non-throwing cleanup, so a
cleanup failure is discarded.

`push_back()` checks the CUDA status returned by `cudaGetLastError()`
immediately after its one-thread append kernel is launched. The checker does not
clear a pre-existing CUDA last-error value first, so a stale status can be
attributed to the append. The call does not wait for kernel completion; an
asynchronous execution failure can surface at a later CUDA call.

There is no CudaContainer-specific C ABI status, `apo_last_error()` contract, or
Python exception behavior because no C ABI handle or Python wrapper exists for
this subsystem.

## Important Behavior and Limitations

`at()` and `operator[]` operate only on the host mirror. The checked overload
throws for an out-of-range host index; the unchecked overload has undefined
behavior for an out-of-range index. Neither operation performs an implicit
transfer. Device code obtains a borrowed device owner through `getDeviceArray()`
and a device pointer through its `data()` method.

`transferToDevice()` copies the complete host active range to the device.
`transferToHost()` copies the complete device active range to the host. Their
`transferFromHost()` and `transferFromDevice()` names are compatibility aliases.
The implementations do not resize or validate either mirror. For a nonempty
container they call `cudaMemcpy` without a stream argument and then
`cudaDeviceSynchronize()`, which waits for all previously requested work on the
current CUDA device rather than only work associated with this container.

`push_back()` can independently reallocate the host and device owners. It then
launches the device write on the default stream and returns after only an
immediate launch check. Prefer `resize()`, initialize the host range, and
perform one `transferToDevice()` when the final length is known. `resize()`
itself does not establish value coherence for newly grown elements.

`shrink_to_fit()` applies the host request first and then asks the device owner
to reduce capacity to its active size. It preserves each mirror's active prefix
independently and does not synchronize values. Host capacity reduction remains
non-binding. A device reallocation invalidates prior device pointers.

`printDeviceArray()` launches one default-stream thread per host-side active
element, prints the corresponding device value with CUDA device `printf`, and
then synchronizes the current CUDA device. It requires a nonempty container with
matching mirror lengths and no more than the maximum `unsigned int` element
count used by its kernel indexing. Values are printed as one scalar or in CUDA
component order, and line order across threads is not guaranteed. The method
does not transfer pending host changes before printing.

The implementation provides library instantiations only for `int`, `int2`,
`int3`, `int4`, `unsigned int`, `float`, `float2`, `float3`, `float4`,
`long long int`, `longlong2`, `longlong3`, `longlong4`,
`unsigned long long int`, `std::size_t`, `double`, `double2`, `double3`, and
`double4`. It has no iterator interface, allocator customization, stream
selection, device tracking, automatic dirty-state tracking, or internal locking.
Callers must serialize overlapping access whenever any operation can mutate a
mirror or its contents.

## Related Subsystems

- [DeviceVector](@ref device_vector) documents the owned CUDA allocation used
  for the device mirror.
- [ApoCharmmError](@ref apocharmm_error) documents native exceptions and CUDA
  error conversion.

## Developer Architecture

The public template declarations and supported explicit instantiations are in
`include/CudaContainer.h`. Implementations are in `src/CudaContainer.cu`, which
is compiled into the native apoCHARMM library by `src/CMakeLists.txt`. The
subsystem has no public C ABI header, private C ABI handle, C ABI
implementation, Python wrapper, direct C ABI test, Python API test, or
standalone repository example.

The ownership graph has one `CudaContainer<T>` as the exclusive parent of one
host `std::vector<T>` and one device `DeviceVector<T>`. The normal structural
invariant is equal active lengths. Value coherence is a state established only
by a successful coherent constructor, coherent assignment, `set()`, or explicit
transfer. Mutable mirror access is the intentional escape hatch around both the
length and coherence invariants.

Host-to-device constructors, host assignments, host setters, scalar setters, and
`transferToDevice()` end at the host-to-device transfer boundary. Device-to-host
constructors, device assignments, device setters, and `transferToHost()` end at
the reverse boundary. For nonempty mirrors, the explicit transfer routines
perform `cudaMemcpy` and then `cudaDeviceSynchronize()`.
CudaContainer-to-CudaContainer copy paths copy each owner separately and do not
issue an explicit container-level synchronization. Same-type move paths exchange
both owners without a CUDA operation. Host- and device-rvalue conversion paths
exchange the compatible owner and explicitly populate the missing mirror.

The native CUDA checker is the error boundary. CUDA runtime statuses and
immediate kernel-launch statuses become `ApoCharmmErrorCode::Cuda`. The
`APOCHARMM_REQUIRE` check in `at()` establishes the only
`ApoCharmmErrorCode::InvalidArgument` path. Destruction delegates to the
non-throwing `DeviceVector` cleanup path so no exception escapes the destructor.
There is no lower-language error boundary because the subsystem is not mapped to
the C ABI or Python.

Adding a supported element representation requires adding the explicit
`CudaContainer<T>` instantiation and adding a compatible `printKernel` overload
for its scalar or component layout. The type must remain valid for byte-for-byte
CUDA copies and by-value kernel argument passing. Changing the instantiation
model or moving template definitions into the header is a separate architectural
change that must be evaluated across all native callers.

Performance-sensitive paths are device allocation, reallocation and preserved-
prefix copies in [DeviceVector](@ref DeviceVector), full-range host/device
transfers, the container-wide `cudaDeviceSynchronize()` calls, and repeated
`push_back()` growth. The class does not batch transfers, track dirty ranges, or
accept an application stream.

The focused native regression suite is
`test/unittests/unittest-cudaContainer.cpp`. It covers construction, deep
copies, host access, checked-index errors, assignment, self-assignment, clear,
resize, capacity reduction, append, setters, transfer directions, and transfer
aliases. `test/unittests/unittest-nothrowDestruction.cpp` statically verifies
the no-throw destructor contract. The current suite does not exercise
`printDeviceArray()` or deliberate mirror-length divergence.

Architectural constraints visible in the current implementation include mutable
accessors that can bypass the equal-length invariant, sequential two-mirror
updates with a weak failure guarantee, device-wide synchronization in explicit
transfers, and a print path whose launch size is derived from the host mirror.

## API Reference

- [CudaContainer](@ref CudaContainer) is the native C++ template and complete
  symbol reference.
- [DeviceVector](@ref DeviceVector) is the owned CUDA-device mirror
  implementation.
- [ApoCharmmError](@ref ApoCharmmError) and
  [ApoCharmmErrorCode](@ref ApoCharmmErrorCode) describe native failures.
