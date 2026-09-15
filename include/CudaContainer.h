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

#include "DeviceVector.h"

#include <cstddef>
#include <vector>
#include <vector_types.h>

/**
 * @brief Owns explicity synchronized host and CUDA-device array mirrors.
 *
 * `CudaContainer<T>` stores one contiguous `std::vector<T>` in host memory
 * and one contiguous @ref DeviceVector in CUDA device memory. Construction
 * from one source establishes equal active lengths. Copying another
 * container preserves any source divergence, and mutable mirror access can
 * introduce new divergence. Element values are synchronized only by
 * constructors, assignments, `set()`, and the explicit transfer methods
 * documented below. Host element access never performs an implicit device
 * transfer.
 *
 * Both mirrors are owned exclusively by the container. Accessors return
 * borrowed references; they do not transfer ownership. Mutable mirror access
 * is an escape hatch that can make the active lengths or values diverge.
 * Transfer and print operations rely on the caller preserving equal active
 * lengths and valid storage after using that escape hatch.
 *
 * CUDA operations use the CUDA runtime state current on the calling thread.
 * The class does not retain a CUDA device identifier or stream and provides
 * no internal host-thread synchronization.
 *
 * Copying a container duplicates its host and device mirrors independently and
 * preserves any existing divergence. Moving a container transfers both mirrors
 * without allocation, copying, transfer, or synchronization and leaves the
 * source empty. Rvalue construction from a host or device vector transfers the
 * compatible source allocation and creates the missing mirror.
 *
 * @tparam T Element representation copied byte-for-byte between host and
 * device memory. The library currently instantiates `int`, `int2`, `int3`,
 * `int4`, `unsigned int`, `float`, `float2`, `float3`, `float4`,
 * `long long int`, `longlong2`, `longlong3`, `longlong4`,
 * `unsigned long long int`, `std::size_t`, `double`, `double2`, `double3`,
 * and `double4`.
 *
 * @note The container assigns no physical meaning or AKMA unit to `T`. Units
 * and component interpretation belong to the owning subsystem.
 * @warning Operations that update both mirrors are sequential rather than
 * transactional. A failure after one mirror changes can leave the two
 * mirrors divergent.
 * @warning The class is not thread-safe. Callers must serialize overlapping
 * access whenever any operation can mutate either mirror or its contents.
 * @see cuda_container
 * @see DeviceVector
 */
template <typename T> class CudaContainer {
public: // Member functions
  /**
   * @brief Constructs an empty host/device container.
   *
   * Neither mirror owns an allocation and both active lengths are zero.
   *
   * @post `size() == 0`, `getHostArray().empty()` is `true`, and
   * `getDeviceArray().empty()` is `true`.
   */
  CudaContainer(void);

  /**
   * @brief Constructs host and device mirrors with the requested length.
   *
   * The host elements are value-initialized by `std::vector<T>`. The device
   * allocation contains unspecified bytes; this constructor does not transfer
   * the host values to the device or synchronize the CUDA device.
   *
   * @param[in] count Dimensionless number of elements in each mirror. The byte
   * count `count * sizeof(T)` must be representable as `std::size_t`.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if CUDA rejects the
   * device allocation.
   * @throws std::bad_alloc If host allocation or error-diagnostic construction
   * fails.
   * @throws std::length_error If `count` or an error diagnostic exceeds an
   * implementation limit.
   *
   * @post On success, both active lengths equal `count` and the device
   * capacity equals `count`.
   * @warning For nonzero `count`, the mirrors are not value-coherent until the
   * caller initializes them with `set()` or performs an explicit transfer.
   */
  CudaContainer(const std::size_t count);

  /**
   * @brief Constructs coherent mirrors by copying a host vector.
   *
   * The input is borrowed only for the duration of construction. The host
   * elements are deep-copied and an independent device allocation is created.
   * For a nonempty source, the complete active range is copied from host to
   * device and followed by `cudaDeviceSynchronize()`.
   *
   * @param[in] other Host vector whose `other.size()` contiguous elements are
   * copied. Element units and component meaning are preserved unchanged. An
   * empty source performs no CUDA transfer or synchronization.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, host-to-device copying, or device synchronization fails.
   * @throws std::bad_alloc If host allocation or error-diagnostic construction
   * fails.
   * @throws std::length_error If the source length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both mirrors have `other.size()` elements and contain
   * independent copies of `other`.
   */
  CudaContainer(const std::vector<T> &other);

  /**
   * @brief Constructs coherent mirrors by transferring a host vector.
   *
   * Device storage is allocated before the host allocation is transferred. The
   * source host allocation is then exchanged into this container and copied to
   * the device, followed by `cudaDeviceSynchronize()` for a nonempty range.
   *
   * @param[in,out] other Host vector whose allocation is transferred.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, host-to-device copying, or synchronization fails.
   * @throws std::bad_alloc If allocation or diagnostic construction fails.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation limit.
   *
   * @post On success, both mirrors contain the original values from `other`.
   * @post On success, `other.empty()` is `true`.
   * @warning A transfer or synchronization failure after the host allocation
   * has been exchanged leaves `other` empty; failed construction releases the
   * transferred host and device storage.
   */
  CudaContainer(std::vector<T> &&other);

  /**
   * @brief Constructs coherent mirrors by copying a device vector.
   *
   * The source is borrowed only for the duration of construction. Its active
   * device elements are deep-copied into an independent device allocation and
   * an equally sized host vector is created. For a nonempty source, the device
   * range is copied to the host and followed by `cudaDeviceSynchronize()`.
   *
   * @param[in] other Device vector whose active range `[0, other.size())` is
   * copied. The source allocation remains owned by `other`. An empty source
   * performs no device-to-host transfer or synchronization.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, device-to-device copying, device-to-host copying, or
   * synchronization fails.
   * @throws std::bad_alloc If host allocation or error-diagnostic construction
   * fails.
   * @throws std::length_error If the source length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both mirrors have `other.size()` elements and contain
   * independent copies of the source device values.
   */
  CudaContainer(const DeviceVector<T> &other);

  /**
   * @brief Constructs coherent mirrors by transferring a device vector.
   *
   * Host storage is allocated first. The source device allocation is then
   * exchanged into this container and copied to the host, followed by
   * `cudaDeviceSynchronize()` for a nonempty range.
   *
   * @param[in,out] other Device vector whose allocation and metadata are
   * transferred.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device-to-host
   * copying or synchronization fails.
   * @throws std::bad_alloc If host allocation or diagnostic construction fails.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation limit.
   *
   * @post On success, both mirrors contain the original active values from
   * `other`.
   * @post On success, `other.data() == nullptr`, `other.size() == 0`, and
   * `other.capacity() == 0`.
   * @warning A transfer or synchronization failure after the device allocation
   * has been exchanged leaves `other` empty; failed construction releases the
   * transferred allocation.
   */
  CudaContainer(DeviceVector<T> &&other);

  /**
   * @brief Constructs independent copies of another container's two mirrors.
   *
   * The host mirror and device mirror are copied separately. No transfer is
   * performed between them, so any value or length divergence already present
   * in `other` is preserved in the new object.
   *
   * @param[in] other Container borrowed for the duration of construction.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation or device-to-device copying fails.
   * @throws std::bad_alloc If host allocation or error-diagnostic construction
   * fails.
   * @throws std::length_error If a mirror length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, neither mirror aliases storage owned by `other`.
   * @note The device copy uses `cudaMemcpy` without an explicit stream and
   * does not issue an additional `cudaDeviceSynchronize()`.
   */
  CudaContainer(const CudaContainer<T> &other);

  /**
   * @brief Constructs a container by transferring both source mirrors.
   *
   * Existing host/device length or value divergence is preserved exactly. No
   * allocation, CUDA copy, transfer, or synchronization is performed.
   *
   * @param[in,out] other Container whose two mirrors are transferred.
   *
   * @post This object owns the host and device storage previously owned by
   * `other`.
   * @post `other.size() == 0`, `other.getHostArray().empty()` is `true`,
   * `other.getDeviceArray().size() == 0`, and
   * `other.getDeviceArray().data() == nullptr`.
   */
  CudaContainer(CudaContainer<T> &&other) noexcept;

  /**
   * @brief Destroys both owned mirrors without propagating cleanup failures.
   *
   * Host storage is released normally. The device owner attempts `cudaFree`,
   * discards its return status, and clears its metadata.
   *
   * @post All references, host pointers, and device pointers borrowed from
   * this object are invalid.
   */
  ~CudaContainer(void) noexcept = default;

  /**
   * @brief Replaces both mirrors with a coherent copy of a host vector.
   *
   * The host mirror is assigned first and the device mirror is resized. For a
   * nonempty source, the complete host range is copied to the device and
   * followed by `cudaDeviceSynchronize()`.
   *
   * @param[in] other Host vector borrowed for the duration of the assignment.
   * No reference to it is retained. An empty source performs no CUDA transfer
   * or synchronization.
   * @return A borrowed reference to this container.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device resizing,
   * host-to-device copying, or device synchronization fails.
   * @throws std::bad_alloc If host allocation or error-diagnostic construction
   * fails.
   * @throws std::length_error If the source length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both mirrors contain independent copies of `other`.
   * @warning On failure after host assignment, the host mirror can contain the
   * new values while the device mirror has an old or partially updated state.
   */
  CudaContainer<T> &operator=(const std::vector<T> &other);

  /**
   * @brief Replaces both mirrors from a transferred host vector.
   *
   * A complete replacement container is constructed first and then moved into
   * this object. The destination is unchanged if replacement construction
   * fails.
   *
   * @param[in,out] other Host vector whose allocation is transferred.
   * @return A borrowed mutable reference to this container.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if allocation,
   * host-to-device copying, or synchronization fails.
   * @throws std::bad_alloc If allocation or diagnostic construction fails.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation limit.
   *
   * @post On success, both mirrors contain the original values from `other`
   * and `other.empty()` is `true`.
   * @warning If replacement construction fails after consuming `other`, the
   * destination remains unchanged but `other` remains empty.
   */
  CudaContainer<T> &operator=(std::vector<T> &&other);

  /**
   * @brief Replaces both mirrors with a coherent copy of a device vector.
   *
   * The device mirror is assigned first and the host mirror is resized. For a
   * nonempty source, the complete device range is copied to the host and
   * followed by `cudaDeviceSynchronize()`.
   *
   * @param[in] other Device vector borrowed for the duration of the
   * assignment. Its allocation remains owned by `other`. An empty source
   * performs no device-to-host transfer or synchronization.
   * @return A borrowed reference to this container.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, device-to-device copying, device-to-host copying, or
   * synchronization fails.
   * @throws std::bad_alloc If host allocation or error-diagnostic construction
   * fails.
   * @throws std::length_error If the source length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both mirrors contain independent copies of the active
   * source range.
   * @warning On failure after device assignment, the device mirror can contain
   * the new values while the host mirror has an old or partially updated
   * state.
   */
  CudaContainer<T> &operator=(const DeviceVector<T> &other);

  /**
   * @brief Replaces both mirrors from a transferred device vector.
   *
   * A complete replacement container is constructed first and then moved into
   * this object. The destination is unchanged if replacement construction
   * fails.
   *
   * @param[in,out] other Device vector whose allocation and metadata are
   * transferred.
   * @return A borrowed mutable reference to this container.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device-to-host
   * copying or synchronization fails.
   * @throws std::bad_alloc If host allocation or diagnostic construction fails.
   * @throws std::length_error If storage or a diagnostic exceeds an
   * implementation limit.
   *
   * @post On success, both mirrors contain the original active values from
   * `other`.
   * @post On success, `other.data() == nullptr`, `other.size() == 0`, and
   * `other.capacity() == 0`.
   * @warning If replacement construction fails after consuming `other`, the
   * destination remains unchanged but `other` remains empty.
   */
  CudaContainer<T> &operator=(DeviceVector<T> &&other);

  /**
   * @brief Replaces each mirror with the corresponding mirror from another
   * container.
   *
   * The host mirror is assigned first and the device mirror second. No
   * transfer is performed between them; source divergence is copied as-is.
   *
   * @param[in] other Container borrowed for the duration of assignment.
   * @return A borrowed reference to this container.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, device-to-device copying, or device cleanup fails.
   * @throws std::bad_alloc If host allocation or error-diagnostic construction
   * fails.
   * @throws std::length_error If a mirror length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both destination mirrors own storage independent of
   * `other` and reproduce the corresponding source mirror.
   * @warning On failure after host assignment, the destination mirrors can
   * represent different source states.
   */
  CudaContainer<T> &operator=(const CudaContainer<T> &other);

  /**
   * @brief Replaces this container by transferring both source mirrors.
   *
   * Existing source divergence is preserved exactly. The destination's former
   * mirrors are released through non-throwing destruction. Self-move assignment
   * is a no-op.
   *
   * @param[in,out] other Container whose two mirrors are transferred.
   * @return A borrowed mutable reference to this container.
   *
   * @post For distinct objects, this container owns the host and device storage
   * previously owned by `other`.
   * @post For distinct objects, `other.size() == 0`,
   * `other.getHostArray().empty()` is `true`,
   * `other.getDeviceArray().size() == 0`, and
   * `other.getDeviceArray().data() == nullptr`.
   * @post Self-move assignment leaves the container unchanged.
   * @warning CUDA cleanup failure for the destination's former device
   * allocation is discarded, consistent with destruction.
   */
  CudaContainer<T> &operator=(CudaContainer<T> &&other) noexcept;

public: // Element access
  /**
   * @brief Returns a checked const reference to one host element.
   *
   * This method reads only the host mirror and performs no CUDA transfer.
   *
   * @param[in] pos Zero-based, dimensionless host element index.
   * @return A borrowed const reference to host element `pos`. The reference is
   * invalidated by owner destruction or any host-vector operation that
   * invalidates references.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `pos >= size()`.
   * @throws std::bad_alloc If construction of the invalid-index diagnostic
   * fails.
   * @throws std::length_error If the invalid-index diagnostic exceeds an
   * implementation limit.
   */
  const T &at(const std::size_t pos) const;

  /**
   * @brief Returns a checked mutable reference to one host element.
   *
   * Mutating the returned reference changes only the host mirror. Call
   * `transferToDevice()` before device code consumes the new value.
   *
   * @param[in] pos Zero-based, dimensionless host element index.
   * @return A borrowed mutable reference to host element `pos`. The reference
   * is invalidated by owner destruction or any host-vector operation that
   * invalidates references.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::InvalidArgument` if
   * `pos >= size()`.
   * @throws std::bad_alloc If construction of the invalid-index diagnostic
   * fails.
   * @throws std::length_error If the invalid-index diagnostic exceeds an
   * implementation limit.
   *
   * @post Mutation through the returned reference does not update the device
   * mirror automatically.
   */
  T &at(const std::size_t pos);

  /**
   * @brief Returns an unchecked const reference to one host element.
   *
   * This method reads only the host mirror and performs no CUDA transfer.
   *
   * @param[in] pos Zero-based, dimensionless host element index.
   * @return A borrowed const reference to host element `pos`, subject to
   * normal `std::vector` reference-invalidation rules.
   *
   * @pre `pos < size()`.
   * @warning Passing an out-of-range index has undefined behavior.
   */
  const T &operator[](const std::size_t pos) const;

  /**
   * @brief Returns an unchecked mutable reference to one host element.
   *
   * Mutating the returned reference changes only the host mirror.
   *
   * @param[in] pos Zero-based, dimensionless host element index.
   * @return A borrowed mutable reference to host element `pos`, subject to
   * normal `std::vector` reference-invalidation rules.
   *
   * @pre `pos < size()`.
   * @post Mutation through the returned reference does not update the device
   * mirror automatically.
   * @warning Passing an out-of-range index has undefined behavior.
   */
  T &operator[](const std::size_t pos);

  /**
   * @brief Returns a borrowed const reference to the host mirror.
   *
   * The returned `std::vector<T>` remains owned by this container. Its values
   * can be stale relative to the device mirror until `transferToHost()` runs.
   *
   * @return A borrowed const reference valid until this container is
   * destroyed. References or pointers to its elements follow normal
   * `std::vector` invalidation rules.
   */
  const std::vector<T> &getHostArray(void) const;

  /**
   * @brief Returns a borrowed mutable reference to the host mirror.
   *
   * Direct element mutation bypasses device synchronization. Directly changing
   * the vector's size can break the equal-length invariant required by
   * transfer and print operations.
   *
   * @return A borrowed mutable reference valid until this container is
   * destroyed. No ownership is transferred.
   *
   * @warning Preserve `getHostArray().size() == getDeviceArray().size()`
   * unless all later operations account explicitly for divergent lengths.
   */
  std::vector<T> &getHostArray(void);

  /**
   * @brief Returns a borrowed const reference to the device mirror.
   *
   * The returned @ref DeviceVector remains owned by this container. Its values
   * can be stale relative to the host mirror until `transferToDevice()` runs.
   *
   * @return A borrowed const reference valid until this container is
   * destroyed. A device pointer obtained from it is invalidated according to
   * @ref DeviceVector reallocation and cleanup rules.
   */
  const DeviceVector<T> &getDeviceArray(void) const;

  /**
   * @brief Returns a borrowed mutable reference to the device mirror.
   *
   * Direct device mutation bypasses host synchronization. Resizing, clearing,
   * or replacing the nested device allocation can break the equal-length and
   * ownership invariants expected by this container.
   *
   * @return A borrowed mutable reference valid until this container is
   * destroyed. No ownership is transferred.
   *
   * @warning Preserve `getHostArray().size() == getDeviceArray().size()` and
   * the normal
   * @ref DeviceVector ownership invariant before calling transfers or
   * `printDeviceArray()`.
   */
  DeviceVector<T> &getDeviceArray(void);

public: // Capacity
  /**
   * @brief Returns the active length of the host mirror.
   *
   * @return Dimensionless number of elements in `getHostArray()`.
   *
   * @note This method does not inspect the device mirror. The returned value
   * can differ from `getDeviceArray().size()` after mutable mirror access or a
   * partially completed operation.
   */
  std::size_t size(void) const;

  /**
   * @brief Requests capacity reduction for both mirrors without transferring
   * values.
   *
   * The host request is applied first. The device owner then reallocates to
   * exactly its active size and preserves its active device prefix. Existing
   * host/device value divergence is not reconciled.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, device-to-device copying, or device cleanup fails.
   * @throws std::bad_alloc If host shrinking or error-diagnostic construction
   * fails.
   * @throws std::length_error If an allocation request or error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both active lengths are unchanged; device capacity
   * equals device size. Host capacity reduction remains a non-binding request.
   * @warning Host element references can be invalidated before a later device
   * failure is reported. A device reallocation invalidates prior device
   * pointers.
   */
  void shrink_to_fit(void);

public: // Modifiers
  /**
   * @brief Clears both mirrors and releases the device allocation.
   *
   * The host mirror is cleared first. The device owner then resets its active
   * length and capacity and attempts to release its CUDA allocation.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if `cudaFree`
   * reports failure.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @post On success, both mirrors are empty and the device data pointer is
   * null.
   * @warning If device cleanup fails, the host mirror and device metadata are
   * already empty and the old device allocation may remain reserved.
   */
  void clear(void);

  /**
   * @brief Appends one value to both mirrors.
   *
   * The host append occurs first. The device owner may allocate a larger
   * buffer, preserve the old device prefix, and then enqueue a one-thread
   * write kernel on the default stream. The method checks only the immediate
   * launch status and does not wait for the append kernel to finish.
   *
   * @param[in] value Element value copied into the new host slot and passed by
   * value to the device append kernel.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, prefix copying, cleanup, or the immediate kernel-launch check
   * fails.
   * @throws std::bad_alloc If host growth or error-diagnostic construction
   * fails.
   * @throws std::length_error If growth or an error diagnostic exceeds an
   * implementation limit.
   *
   * @post After a successful immediate launch check, both active lengths are
   * increased by one and the device write is enqueued.
   * @warning A failure after the host append can leave the mirrors with
   * different lengths or values.
   * @warning Existing host references and device pointers can be invalidated
   * by growth. Prefer `resize()` followed by initialization when the final
   * length is known.
   */
  void push_back(const T &value);

  /**
   * @brief Resizes both active ranges without synchronizing their values.
   *
   * The host vector is resized first and the device vector second. Growing the
   * host value-initializes new elements. Newly exposed or allocated device
   * slots contain unspecified bytes until written. Shrinking preserves the
   * retained prefix independently in each mirror.
   *
   * @param[in] count New dimensionless active element count for both mirrors.
   * The byte count `count * sizeof(T)` must be representable as `std::size_t`.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, prefix copying, or cleanup fails.
   * @throws std::bad_alloc If host growth or error-diagnostic construction
   * fails.
   * @throws std::length_error If `count` or an error diagnostic exceeds an
   * implementation limit.
   *
   * @post On success, both active lengths equal `count`.
   * @warning Growing does not make the new host and device elements coherent.
   * A failure after host resizing can leave the mirror lengths divergent.
   */
  void resize(const std::size_t count);

  /**
   * @brief Replaces both mirrors with a coherent host-vector copy.
   *
   * The host values are copied and the device active range is resized to
   * match. For a nonempty source, the full host range is copied to device
   * memory and followed by `cudaDeviceSynchronize()`.
   *
   * @param[in] values Host vector borrowed for the duration of the operation.
   * No reference to it is retained. An empty source performs no CUDA transfer
   * or synchronization.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device resizing,
   * host-to-device copying, or device synchronization fails.
   * @throws std::bad_alloc If host assignment or error-diagnostic construction
   * fails.
   * @throws std::length_error If the input length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both mirrors contain independent copies of `values`.
   * @warning A failure after host assignment can leave the mirrors divergent.
   */
  void set(const std::vector<T> &values);

  /**
   * @brief Replaces both mirrors with a coherent device-vector copy.
   *
   * The active device range is deep-copied first and the host vector is
   * resized. For a nonempty source, the full device range is copied to host
   * memory and followed by `cudaDeviceSynchronize()`.
   *
   * @param[in] values Device vector borrowed for the duration of the
   * operation. Its allocation remains owned by `values`. An empty source
   * performs no device-to-host transfer or synchronization.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if device
   * allocation, device-to-device copying, device-to-host copying, or
   * synchronization fails.
   * @throws std::bad_alloc If host resizing or error-diagnostic construction
   * fails.
   * @throws std::length_error If the input length or an error diagnostic
   * exceeds an implementation limit.
   *
   * @post On success, both mirrors contain independent copies of the active
   * device source range.
   * @warning A failure after device assignment can leave the mirrors
   * divergent.
   */
  void set(const DeviceVector<T> &values);

  /**
   * @brief Sets every current element to one value in both mirrors.
   *
   * The host mirror is filled without changing its active length. The complete
   * host range is then copied to the device and followed by
   * `cudaDeviceSynchronize()`. An empty host mirror makes the transfer a
   * no-op.
   *
   * @param[in] value Element value copied into every active slot. Its physical
   * units, if any, are defined by the owning subsystem.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if host-to-device
   * copying or device synchronization fails.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @pre The active mirror lengths are equal and both nonempty mirrors own
   * valid storage for that active range.
   * @post On success, both mirrors retain their current lengths and every
   * active element equals `value`.
   * @warning A transfer failure occurs after the host mirror has been filled
   * and can leave the device mirror unchanged or partially updated.
   */
  void set(const T value);

  /**
   * @brief Sets every current element to one value in both mirrors.
   *
   * This compatibility alias delegates directly to `set(const T)`.
   *
   * @param[in] value Element value copied into every active slot. Its physical
   * units, if any, are defined by the owning subsystem.
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if host-to-device
   * copying or device synchronization fails.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @pre The active mirror lengths are equal and both nonempty mirrors own
   * valid storage for that active range.
   * @post On success, both mirrors retain their current lengths and every
   * active element equals `value`.
   */
  void setToValue(const T value);

  /**
   * @brief Transfers the complete host mirror to device memory.
   *
   * For a nonempty host mirror, the method copies `size()` contiguous elements
   * with `cudaMemcpyHostToDevice` and then calls `cudaDeviceSynchronize()`. It
   * does not allocate, resize, or select a stream. An empty host mirror
   * returns without issuing a CUDA call.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if copying or
   * device synchronization fails.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @pre The active mirror lengths are equal and every nonempty mirror owns
   * valid storage for at least `size()` elements.
   * @post On success, the active device values equal the active host values.
   * @warning This call synchronizes all previously requested work on the
   * current CUDA device, not only work associated with this container.
   */
  void transferToDevice(void);

  /**
   * @brief Transfers the complete device mirror to host memory.
   *
   * For a nonempty host mirror, the method copies `getDeviceArray().size()`
   * contiguous elements with `cudaMemcpyDeviceToHost` and then calls
   * `cudaDeviceSynchronize()`. It does not allocate, resize, or select a
   * stream. An empty host mirror returns without issuing a CUDA call.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if copying or
   * device synchronization fails.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @pre The active mirror lengths are equal and every nonempty mirror owns
   * valid storage for that active range.
   * @post On success, the active host values equal the active device values.
   * @warning This call synchronizes all previously requested work on the
   * current CUDA device, not only work associated with this container.
   */
  void transferToHost(void);

  /**
   * @brief Transfers the complete device mirror to host memory.
   *
   * This compatibility alias delegates directly to `transferToHost()` and has
   * the same equal-length precondition, device-wide synchronization, and error
   * behavior.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if copying or
   * device synchronization fails.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @pre The active mirror lengths are equal and both nonempty mirrors own
   * valid storage for that active range.
   * @post On success, the active host values equal the active device values.
   */
  void transferFromDevice(void);

  /**
   * @brief Transfers the complete host mirror to device memory.
   *
   * This compatibility alias delegates directly to `transferToDevice()` and
   * has the same equal-length precondition, device-wide synchronization, and
   * error behavior.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if copying or
   * device synchronization fails.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @pre The active mirror lengths are equal and both nonempty mirrors own
   * valid storage for that active range.
   * @post On success, the active device values equal the active host values.
   */
  void transferFromHost(void);

public:
  /**
   * @brief Prints the active device mirror with CUDA device `printf`.
   *
   * One default-stream kernel thread is assigned per element. Each emitted
   * line contains the zero-based index followed by the scalar value or the
   * CUDA vector components in `x`, `y`, `z`, `w` order. The method does not
   * transfer host values to the device. It checks the immediate launch status
   * and then calls `cudaDeviceSynchronize()` so device output is flushed
   * before return.
   *
   * @throws ApoCharmmError With `ApoCharmmErrorCode::Cuda` if the immediate
   * kernel-launch check or device synchronization fails.
   * @throws std::bad_alloc If CUDA error-diagnostic construction fails.
   * @throws std::length_error If the CUDA error diagnostic exceeds an
   * implementation limit.
   *
   * @pre `size() > 0`, `size()` is no greater than the maximum value of
   * `unsigned int`, the active mirror lengths are equal, and the device mirror
   * owns valid storage for the active range.
   * @post On success, the container values and allocations are unchanged.
   * @note Output line order is not guaranteed across CUDA threads.
   * @warning The printed values can be stale relative to the host mirror
   * unless the caller first calls `transferToDevice()`.
   */
  void printDeviceArray(void) const;

  // TODO : enable a range-based iterator
private:
  /** @brief Owns the contiguous host mirror. */
  std::vector<T> m_HostArray;

  /** @brief Owns the contiguous CUDA-device mirror. */
  DeviceVector<T> m_DeviceArray;
};

template class CudaContainer<int>;
template class CudaContainer<int2>;
template class CudaContainer<int3>;
template class CudaContainer<int4>;
template class CudaContainer<unsigned int>;
template class CudaContainer<float>;
template class CudaContainer<float2>;
template class CudaContainer<float3>;
template class CudaContainer<float4>;
template class CudaContainer<long long int>;
template class CudaContainer<longlong2>;
template class CudaContainer<longlong3>;
template class CudaContainer<longlong4>;
template class CudaContainer<unsigned long long int>;
template class CudaContainer<std::size_t>;
template class CudaContainer<double>;
template class CudaContainer<double2>;
template class CudaContainer<double3>;
template class CudaContainer<double4>;
