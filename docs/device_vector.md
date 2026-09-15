# DeviceVector {#device_vector}

## Purpose

`DeviceVector<T>` is apoCHARMM's native owning container for a contiguous,
one-dimensional sequence in CUDA device memory. Use it when native C++ or CUDA
code needs allocation ownership and vector-like size/capacity management but
does not need a host mirror. Use [CudaContainer](@ref CudaContainer) instead
when the same data must be maintained on both the host and device.

`DeviceVector` is currently a native C++ subsystem. The repository does not
expose it through the public C ABI or the Python package.

## Quick Start

The following complete native example copies a host vector to device memory and
then copies the active range back to the host:

```cpp
#include "DeviceVector.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <vector>

int main() {
  const std::vector<float> source{1.0f, 2.0f, 3.0f};
  DeviceVector<float> device(source);

  std::vector<float> roundTrip(device.size());
  cudaCheck(cudaMemcpy(roundTrip.data(), device.data(),
                       roundTrip.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaDeviceSynchronize());

  return roundTrip == source ? 0 : 1;
}
```

Link the example against apoCHARMM and the CUDA runtime. It returns zero after a
successful round trip. The pointer returned by `device.data()` is a CUDA device
pointer and is supplied to `cudaMemcpy`; host code does not dereference it.

## Construction and Required State

The default constructor creates an empty vector without storage. A size
constructor allocates exactly the requested number of active slots, but it does
not initialize them. Construction from `std::vector<T>` performs a
host-to-device copy of the active host elements. Construction from another
`DeviceVector<T>` performs a deep device-to-device copy and sets capacity to the
source size rather than the source capacity.

A temporary `std::vector<T>` is accepted by the same const-reference overload as
a host lvalue because its host allocation cannot be transferred into CUDA device
memory. The active elements are copied host-to-device.

Move construction from another `DeviceVector<T>` transfers its device pointer,
size, and capacity without allocation, copying, synchronization, or another CUDA
runtime call. The moved-from vector becomes empty with zero capacity and a null
device pointer.

CUDA allocation and copy calls use the CUDA runtime state current on the calling
thread. `DeviceVector` does not retain a device identifier or stream. Code must
therefore establish the intended CUDA device before construction and before
later operations that touch the allocation.

For every requested element count, the corresponding `count * sizeof(T)` byte
count must be representable as `std::size_t`. Newly allocated or newly exposed
element slots contain unspecified bytes until caller code writes them.

## Ownership and Lifetime

Normal construction gives the vector exclusive ownership of its CUDA
allocation. `data()` returns a borrowed device pointer; it never transfers
ownership. The pointer remains usable only while its allocation remains owned
and unchanged.

`clear()`, destruction, assignment that changes capacity, `shrink_to_fit()` that
changes capacity, `resize()` beyond capacity, and `push_back()` that grows the
allocation invalidate pointers into the old allocation. `swap()` does not
change either device address, but it exchanges which object owns each address.

The destructor is `noexcept`. It attempts `cudaFree`, discards the CUDA return
status, and clears the object's metadata. Explicit `clear()` reports a CUDA
failure instead, although the current cleanup helper has already cleared the
stored pointer when that failure is reported.

Move assignment transfers allocation ownership and metadata, then releases the
destination's previous allocation through the non-throwing destruction path.
Consequently, move assignment is `noexcept`; a failure while releasing the old
CUDA allocation is discarded and that allocation may remain reserved. Self-move
assignment is a no-op.

`assignData()` is an unchecked legacy ownership escape hatch. It replaces only
the stored pointer; it does not free the old allocation or change size or
capacity. Because later cleanup passes the replacement pointer to `cudaFree`, a
non-null replacement must be treated as transferred to the vector and must
cover at least `capacity()` elements. Prefer normal construction, assignment,
`resize()`, or `swap()` whenever possible.

## Data, Shapes, and AKMA Units

Storage is one contiguous array of `capacity()` elements in device memory. The
active range is the prefix `[0, size())`; the spare range
`[size(), capacity())` has no defined contents. There is no additional shape,
stride, padding, or host mirror imposed by `DeviceVector`. CUDA vector element
types such as `float4` retain the layout defined by their CUDA type.

`size()` and `capacity()` are dimensionless element counts. `DeviceVector`
does not assign physical meaning or AKMA units to an element. Units, component
meaning, and any higher-dimensional interpretation are properties of the code
that owns the `DeviceVector<T>`.

## Errors

Native allocation, copy, deallocation, and immediate kernel-launch checks report
failure by throwing [ApoCharmmError](@ref ApoCharmmError) with
`ApoCharmmErrorCode::Cuda`. Diagnostic construction can additionally propagate
`std::bad_alloc` or `std::length_error`. See
[ApoCharmmError](@ref apocharmm_error) for the native error format and
source-location behavior.

A failed provisional allocation or device-to-device prefix copy during
reallocation leaves the original vector unchanged. A failure while freeing the
old allocation occurs after the cleanup helper has cleared the pointer and
metadata, so the vector reports an empty state and the old allocation may remain
reserved. Assignment performs its final host-to-device or device-to-device copy
after size and capacity are updated; failure of that final copy leaves those
new metadata values observable and the active contents unspecified.

`push_back()` checks only the CUDA status returned by `cudaGetLastError()`
immediately after launch. The helper does not clear a pre-existing CUDA
last-error value before the launch, so a stale status can be attributed to
`push_back()` after its kernel has been enqueued. The method does not wait for
the kernel to finish, so an asynchronous execution failure may surface at a
later CUDA call. The destructor does not throw or publish a cleanup diagnostic.

There is no DeviceVector-specific C ABI status mapping, `apo_last_error()`
contract, or Python exception behavior because no C ABI or Python wrapper exists
for this subsystem.

## Important Behavior and Limitations

`resize()` changes the logical size. It preserves an existing allocation when
capacity is sufficient, including `resize(0)` on a vector with nonzero capacity.
Growing beyond capacity allocates exactly the requested capacity and preserves
the old active prefix. Shrinking does not destroy element objects or release
storage. `shrink_to_fit()` is the operation that reduces capacity to size.

`push_back()` grows a full allocation to
`capacity() + capacity() / 2 + 1`. It preserves the active prefix and launches a
one-thread kernel on the default stream to store the new value. Pre-sizing with
`resize()` is preferable when the final length is known because repeated growth
allocates replacement buffers and copies existing device data.

The implementation uses `cudaMemcpy` without an explicit stream for host/device
and device/device transfers and does not issue a separate
`cudaDeviceSynchronize()`. Only `push_back()` launches a kernel, and that launch
uses the default stream. Callers that combine these operations with work on
other streams must establish the required CUDA ordering themselves.

The class does not construct or destroy individual `T` objects and currently
provides library specializations only for the scalar and CUDA vector types
listed in [DeviceVector](@ref DeviceVector). It has no iterator interface,
bounds-checked element access, allocator customization, stream selection, device
tracking, or internal locking. Callers must serialize access whenever any caller
may mutate an object.

## Related Subsystems

- [CudaContainer](@ref cuda_container) maintains a host `std::vector<T>` and a
  device `DeviceVector<T>` with explicit transfer operations.
- [ApoCharmmError](@ref apocharmm_error) describes native and CUDA error
  reporting.

## Developer Architecture

The public declarations and supported specializations are in
`include/DeviceVector.h`. Implementations are in `src/DeviceVector.cu`, which is
compiled into the native apoCHARMM library by `src/CMakeLists.txt`. There is no
C ABI handle, C ABI implementation, Python wrapper, direct C ABI test, Python
test, or standalone example for this class in the current repository.

The ownership state is the tuple `m_Data`, `m_Size`, and `m_Capacity`. Normal
paths maintain `m_Size <= m_Capacity` and keep `m_Data == nullptr` exactly when
capacity is zero. `assignData()` can bypass both the allocation ownership and
pointer/capacity parts of that invariant, so new code should not use it as a
routine initialization path.

Reallocation is staged through a provisional device buffer. The implementation
allocates the new buffer, copies `min(old_size, new_capacity)` active elements,
releases the old buffer, and only then commits the new pointer and metadata. The
catch path releases the provisional buffer with a non-throwing helper. Because
the old-buffer helper clears the pointer before checking `cudaFree`, a failure
at that step cannot restore the old ownership state.

Host constructors and host assignments copy active `std::vector` elements with
`cudaMemcpyHostToDevice`. Copy construction and copy assignment use
`cudaMemcpyDevicetoDevice`. Move construction and move assignment transfer the
device pointer and metadata without copying device data. `push_back()` is the
only element-writing primitive; it invokes the internal `SetBackKernel` with one
block and one thread. None of these paths stores a stream for later use.

The native CUDA checker is the subsystem's error boundary: CUDA return codes and
immediate launch errors become `ApoCharmmErrorCode::Cuda`. Destruction uses the
separate non-throwing cleanup helper so an exception cannot escape a destructor.
There is no lower-language boundary because `DeviceVector` is not exposed by the
C ABI.

Adding a supported element type requires updating the explicit specialization
set and ensuring that every operation remains valid for raw byte copies and
by-value CUDA kernel argument passing. Moving template definitions into a header
or changing the instantiation model is a separate architectural change and must
be evaluated across all native call sites.

The focused native regression suite is
`test/unittests/unittest-deviceVector.cpp`. It covers construction, deep copies,
capacity changes, prefix preservation, append behavior, immediate launch-error
reporting, assignments, and swaps. `test/unittests/unittest-cudaContainer.cpp`
exercises the principal owning collaborator. Current technical debt visible at
this layer includes unchecked `assignData()` ownership replacement, unchecked
size arithmetic, and loss of the old allocation pointer when `cudaFree` reports
failure.

## API Reference

- [DeviceVector](@ref DeviceVector) is the native C++ template and complete
  symbol reference.
- [CudaContainer](@ref CudaContainer) is the host/device paired-container
  collaborator.
- [ApoCharmmError](@ref ApoCharmmError) and
  [ApoCharmmErrorCode](@ref ApoCharmmErrorCode) describe native failures.
