// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include <cstddef>
#include <vector>
#include <vector_types.h>

/**
 * @brief Stores an owned, contiguous sequence in CUDA device memory.
 *
 * `DeviceVector` owns one CUDA allocation containing `capacity()` adjacent
 * element slots and identifies the active prefix with `size()`. It has no host
 * mirror, iterator interface, stream member, or recorded CUDA device.
 * Allocation and copy operations therefore use the CUDA runtime state current
 * on the calling thread. Use @ref CudaContainer when coordinated host and
 * device representations are required.
 *
 * Elements are handlded as raw storage. The container does not construct,
 * destroy, or value-initialize individual elements. The specializations
 * provided by the current apoCHARMM library are `int`, `int2`, `int3`, `int4`,
 * `unsigned int`, `float`, `float2`, `float3`, `float4`, `long long int`,
 * `longlong2`, `longlong3`, `longlong4`, `unsigned long long int`,
 * `std::size_t`, `double`, `double2`, `double3`, and `double4`.
 *
 * Copy construction and copy assignment allocate independent device storage and
 * copy the active elements. Move construction and move assignment transfer the
 * device pointer and its size and capacity metadata without a CUDA operation. A
 * moved-from vector is empty, has zero capacity, and stores a null pointer.
 *
 * @tparam T Element representation stored in device memory. A specialization
 * must be safe to copy byte-for-byte and pass by value to a CUDA kernel.
 *
 * @warning The class does not protect its metadata with locks. Callers must
 * serialize access to one object whenever any caller may mutate it.
 * @warning Pointers returned by `data()` are device pointers. Host code must
 * not dereference them directly.
 * @see device_vector
 */
template <typename T> class DeviceVector {
public: // Member functions
  /**
   * @brief Constructs an empty vector without owned device storage.
   *
   * @post `empty()` is `true`, `size()` and `capacity()` are zero, and `data()`
   * is `nullptr`.
   * @note This constructor performs no CUDA operation.
   */
  DeviceVector(void);

  /**
   * @brief Constructs a vector with uninitialized device storage.
   *
   * @param[in] count Number of active element slots to allocate.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if the CUDA
   * allocation fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @pre `count * sizeof(T)` is representable as `std::size_t`.
   * @post On success, `size()` and `capacity()` equal `count`. A zero count
   * produces `data() == nullptr`; otherwise `data()` identifies an owned device
   * allocation.
   * @warning The active element slots are not initialized.
   */
  DeviceVector(const std::size_t count);

  /**
   * @brief Constructs a device vector by copying a host vector.
   *
   * @param[in] other Borrowed, read-only host vector. Its active elements are
   * copied during the call, and no reference to it is retained.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device allocation
   * or the host-to-device copy fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @post On success, `size()` and `capacity()` equal `other.size()`, and the
   * active device elements are an independent copy of `other`.
   * @note The copy uses `cudaMemcpy` without an explicit stream and performs no
   * separate device synchronization.
   */
  DeviceVector(const std::vector<T> &other);

  /**
   * @brief Constructs an independent copy of another device vector.
   *
   * @param[in] other Borrowed, read-only source vector. Its active device
   * elements are copied during the call, and no pointer into it is retained.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device allocation
   * or the device-to-device copy fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @post On success, `size()` and `capacity()` equal `other.size()`, and the
   * active elements reside in an allocation independent of `other`.
   * @note Spare capacity in `other` is not copied.
   * @note The copy uses `cudaMemcpy` without an explicit stream and performs no
   * separate device synchronization.
   */
  DeviceVector(const DeviceVector<T> &other);

  /**
   * @brief Constructs a device vector by transferring another allocation.
   *
   * The device pointer, active size, and capacity are transferred without
   * allocation, copying, synchronization, or another CUDA runtime call.
   *
   * @param[in,out] other Vector whose allocation and metadata are transferred.
   *
   * @post `data()`, `size()`, and `capacity()` equal the corresponding pre-move
   * values from `other`.
   * @post `other.data() == nullptr`, `other.size() == 0`, and
   * `other.capacity() == 0`.
   * @note Previously borrowed pointers into `other` retain their pointer values
   * but are now owned by this object.
   */
  DeviceVector(DeviceVector<T> &&other) noexcept;

  /**
   * @brief Releases owned device storage without propagating CUDA failures.
   *
   * The destructor passes the stored pointer to the non-throwing CUDA cleanup
   * helper, clears the metadata, and ignores the return status from `cudaFree`.
   *
   * @warning If CUDA cannot release the allocation, the storage may remain
   * reserved and no diagnostic is reported.
   */
  ~DeviceVector(void) noexcept;

  /**
   * @brief Replaces the vector with a copy of a host vector.
   *
   * The destination capacity is changed to `other.capacity()`, then
   * `other.size()` active elements are copied from host to device.
   *
   * @param[in] other Borrowed, read-only host vector. No reference to it is
   * retained after the call.
   * @return A borrowed reference aliasing this destination object. It remains
   * valid for the lifetime of the destination.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if allocation,
   * device-to-device prefix preservation, deallocation, or the final
   * host-to-device copy fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @post On success, `size()` equals `other.size()`, `capacity()` equals
   * `other.capacity()`, and the active device elements copy `other`.
   * @note Slots in `[size(), capacity())` have unspecified contents.
   * @note Any capacity change invalidates previously returned device pointers.
   * @note CUDA copies use `cudaMemcpy` without an explicit stream and perform
   * no separate device synchronization.
   * @warning If provisional allocation or prefix preservation fails, the
   * destination remains unchanged. If capacity adjustment fails while
   * releasing the old allocation, the destination is reset to an empty state
   * and the old allocation may remain reserved. If the final host-to-device
   * copy fails, the new size and capacity remain observable and active contents
   * are unspecified.
   */
  DeviceVector<T> &operator=(const std::vector<T> &other);

  /**
   * @brief Replaces the vector with a deep copy of another device vector.
   *
   * The destination capacity is changed to `other.capacity()`, then
   * `other.size()` active elements are copied device-to-device.
   *
   * @param[in] other Borrowed, read-only source vector. No pointer into it is
   * retained after the call.
   * @return A borrowed reference aliasing this destination object. It remains
   * valid for the lifetime of the destination.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if allocation,
   * device-to-device prefix preservation, deallocation, or the final
   * device-to-device copy fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @post On success, `size()` equals `other.size()`, `capacity()` equals
   * `other.capacity()`, and the active elements are an independent copy.
   * @note Source slots in `[other.size(), other.capacity())` are not copied.
   * @note Any capacity change invalidates previously returned device pointers.
   * @note The implementation does not special-case self-assignment and still
   * issues the device-to-device copy.
   * @note CUDA copies use `cudaMemcpy` without an explicit stream and perform
   * no separate device synchronization.
   * @warning If provisional allocation or prefix preservation fails, the
   * destination remains unchanged. If capacity adjustment fails while
   * releasing the old allocation, the destination is reset to an empty state
   * and the old allocation may remain reserved. If the final device-to-device
   * copy fails, the new size and capacity remain observable and active contents
   * are unspecified.
   */
  DeviceVector<T> &operator=(const DeviceVector<T> &other);

  /**
   * @brief Replaces this vector by transferring another vector's allocation.
   *
   * The source device pointer, active size, and capacity are transfered without
   * allocation, copying, or synchronization. The destination's former
   * allocation is released through the same non-throwing cleanup path used by
   * destruction. Self-move assignment is a no-op.
   *
   * @param[in,out] other Vector whose allocation and metadata are transferred.
   * @return A borrowed mutable reference to this vector.
   *
   * @post For distinct objects, `data()`, `size()`, and `capacity()` equal the
   * corresponding pre-move values from `other`.
   * @post For distinct objects, `other.data() == nullptr`, `other.size() == 0`,
   * and `other.capacity() == 0`.
   * @post Self-move assignment leaves the vector unchanged.
   * @warning A CUDA failure while releasing the destination's previous
   * allocation is discarded, consistent with destructor cleanup. That
   * allocation may remain reserved.
   */
  DeviceVector<T> &operator=(DeviceVector<T> &&other) noexcept;

public: // Element access
  /**
   * @brief Returns a read-only pointer to the owned device allocation.
   *
   * @return A borrowed CUDA device pointer to the first allocated element slot,
   * or `nullptr` when no allocation is owned. The pointee storage remains valid
   * until this object is destroyed, cleared, assigned replacement storage, or
   * reallocated. `swap()` preserves the pointer value but transfers ownership
   * to the other vector.
   * @warning Host code must not dereference the returned pointer directly.
   */
  const T *data(void) const;

  /**
   * @brief Returns a mutable pointer to the owned device allocation.
   *
   * @return A borrowed CUDA device pointer to the first allocated element slot,
   * or `nullptr` when no allocation is owned. The pointee storage remains valid
   * until this object is destroyed, cleared, assigned replacement storage, or
   * reallocated. `swap()` preserves the pointer value but transfers ownership
   * to the other vector.
   * @note Mutations through the pointer do not change `size()` or `capacity()`.
   * @warning Host code must not dereference the returned pointer directly.
   */
  T *data(void);

  /**
   * @brief Replaces the stored device pointer without changing metadata.
   *
   * This legacy escape hatch performs no copy, allocation, deallocation, or
   * validation. Subsequent `clear()` or destruction passes the replacement
   * pointer to `cudaFree`, so a non-null pointer is effectively transferred to
   * this object.
   *
   * @param[in] data Nullable CUDA allocation pointer to store. When non-null,
   * it must be releasable with `cudaFree` and identify storage for at least
   * `capacity()` elements.
   *
   * @post `data()` equals `data`; `size()` and `capacity()` are unchanged.
   * @warning Any previously owned allocation is not released and becomes
   * unreachable through this object.
   * @warning Passing `nullptr` while `capacity()` is nonzero, passing borrowed
   * storage, or passing a buffer shorter than `capacity()` breaks the ownership
   * and storage invariants required by later operations.
   */
  void assignData(T *data);

public: // Capacity
  /**
   * @brief Returns whether the active element range is empty.
   *
   * @return `true` when `size()` is zero; otherwise `false`. An empty vector
   * may still own storage when `capacity()` is nonzero.
   */
  bool empty(void) const;

  /**
   * @brief Returns the number of active element slots.
   *
   * @return The length of the active prefix in elements.
   */
  std::size_t size(void) const;

  /**
   * @brief Returns the number of allocated element slots.
   *
   * @return The device-allocation capacity in elements, not bytes.
   */
  std::size_t capacity(void) const;

  /**
   * @brief Reduces device capacity to the current size.
   *
   * The active prefix is preserved. The call is a no-op when `size()` already
   * equals `capacity()`; otherwise it replaces or releases the allocation.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if allocation,
   * device-to-device prefix copying, or deallocation fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @post On success, `capacity()` equals `size()`. If the size is zero,
   * `data()` is `nullptr`.
   * @note A capacity change invalidates previously returned device pointers.
   * @note CUDA copies use `cudaMemcpy` without an explicit stream and perform
   * no separate device synchronization.
   * @warning If provisional allocation or prefix copying fails, the original
   * vector remains unchanged. If releasing the original allocation fails, the
   * vector is reset to an empty state and the allocation may remain reserved.
   */
  void shrink_to_fit(void);

public: // Modifiers
  /**
   * @brief Releases all device storage and resets the vector.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if `cudaFree` fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @post On success, `empty()` is `true`, `size()` and `capacity()` are zero,
   * and `data()` is `nullptr`.
   * @note All previously returned device pointers are invalidated.
   * @warning On a deallocation failure, the object still reports the empty
   * state and loses the allocation pointer; the device storage may remain
   * reserved.
   */
  void clear(void);

  /**
   * @brief Appends one value to the active device sequence.
   *
   * When the allocation is full, capacity grows to
   * `capacity() + capacity() / 2 + 1` and the existing active prefix is copied
   * to the replacement allocation. A single CUDA thread then writes the new
   * value on the default stream.
   *
   * @param[in] value Host value copied into the new device element. No
   * reference to `value` is retained after kernel launch.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if allocation,
   * device-to-device prefix copying, deallocation, or `cudaGetLastError()`
   * returns a non-success status immediately after launch.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @pre When growth is required, the growth expression and resulting byte
   * count are representable as `std::size_t`.
   * @post After a successful immediate launch check, `size()` is increased by
   * one and the write has been enqueued.
   * @note Growth invalidates previously returned device pointers. Appending
   * within spare capacity preserves the allocation address.
   * @warning The call does not synchronize the default stream. Kernel execution
   * errors may be reported by a later CUDA operation rather than this call.
   * @warning The launch checker does not clear a pre-existing CUDA last-error
   * value before launch. A stale error can therefore make this call throw after
   * the kernel has been enqueued.
   * @warning If the immediate launch-status check fails, `size()` is not
   * increased. Any completed growth remains in effect, and storage outside the
   * active prefix may have been modified by an enqueued kernel.
   */
  void push_back(const T &value);

  /**
   * @brief Changes the number of active element slots.
   *
   * Growing beyond capacity reallocates exactly `count` slots and preserves the
   * old active prefix. Resizing within capacity changes only the logical size.
   * Newly exposed elements are not initialized, and shrinking does not release
   * storage.
   *
   * @param[in] count Requested active element count.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if required
   * allocation, device-to-device prefix copying, or deallocation fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @pre `count * sizeof(T)` is representable as `std::size_t` when allocation
   * or reallocation is required.
   * @post On success, `size()` equals `count`. Capacity is unchanged when it
   * was already sufficient; otherwise it equals `count`.
   * @note Reallocation invalidates previously returned device pointers.
   * `resize(0)` preserves an existing nonzero-capacity allocation.
   * @note CUDA copies use `cudaMemcpy` without an explicit stream and perform
   * no separate device synchronization.
   * @warning If provisional allocation or prefix copying fails, the original
   * vector remains unchanged. If releasing the original allocation fails, the
   * vector is reset to an empty state and the allocation may remain reserved.
   */
  void resize(const std::size_t count);

  /**
   * @brief Exchanges allocation ownership and metadata with another vector.
   *
   * @param[in,out] other Vector whose size, capacity, and device pointer are
   * exchanged with this object. No element data is copied.
   *
   * @post Each vector owns the allocation and metadata previously owned by the
   * other. Device pointer values remain unchanged, but their owning objects are
   * exchanged.
   * @note The operation performs no CUDA call, supports self-swap, and does not
   * synchronize any stream.
   */
  void swap(DeviceVector<T> &other) noexcept;

private:
  /**
   * @brief Allocates uninitialized device storage without changing size.
   *
   * @param[in] count Number of element slots to allocate.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if `cudaMalloc`
   * fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @pre No allocation is currently owned, and `count * sizeof(T)` is
   * representable as `std::size_t`.
   * @post On success, `m_Capacity` equals `count`; `m_Size` is unchanged.
   */
  void allocate(const std::size_t count);

  /**
   * @brief Reallocates device storage while preserving the active prefix.
   *
   * @param[in] count Requested allocation capacity in elements.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if allocation,
   * device-to-device prefix copying, or deallocation fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @pre `count * sizeof(T)` is representable as `std::size_t` when `count` is
   * nonzero.
   * @post On success, `m_Capacity` equals `count` and `m_Size` equals the
   * lesser of its previous value and `count`.
   */
  void reallocate(const std::size_t count);

  /**
   * @brief Releases the stored device pointer and clears ownership metadata.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if `cudaFree` fails.
   * @throws std::bad_alloc If reporting a CUDA failure cannot allocate
   * diagnostic storage.
   * @throws std::length_error If a CUDA failure diagnostic exceeds an
   * implementation-defined string limit.
   *
   * @post On success, `m_Size` and `m_Capacity` are zero and `m_Data` is
   * `nullptr`.
   * @warning The metadata and pointer are cleared before a failed `cudaFree` is
   * reported, so the original allocation cannot be recovered through this
   * object.
   */
  void deallocate(void);

private:
  /** @brief Stores the number of active slots in the allocation prefix. */
  std::size_t m_Size;

  /** @brief Stores the allocation length in elements. */
  std::size_t m_Capacity;

  /**
   * @brief Owns the CUDA allocation used by the vector.
   *
   * Normal construction and modifier paths keep this pointer null exactly when
   * `m_Capacity` is zero. `assignData()` can bypass that invariant.
   */
  T *m_Data;
};

template class DeviceVector<int>;
template class DeviceVector<int2>;
template class DeviceVector<int3>;
template class DeviceVector<int4>;
template class DeviceVector<unsigned int>;
template class DeviceVector<float>;
template class DeviceVector<float2>;
template class DeviceVector<float3>;
template class DeviceVector<float4>;
template class DeviceVector<long long int>;
template class DeviceVector<longlong2>;
template class DeviceVector<longlong3>;
template class DeviceVector<longlong4>;
template class DeviceVector<unsigned long long int>;
template class DeviceVector<std::size_t>;
template class DeviceVector<double>;
template class DeviceVector<double2>;
template class DeviceVector<double3>;
template class DeviceVector<double4>;
