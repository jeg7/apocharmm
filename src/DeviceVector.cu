// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "DeviceVector.h"

#include "cuda_utils.h"

#include <utility>

template <typename T>
DeviceVector<T>::DeviceVector(void)
    : m_Size(0), m_Capacity(0), m_Data(nullptr) {}

template <typename T>
DeviceVector<T>::DeviceVector(const std::size_t count) : DeviceVector() {
  this->allocate(count);
  m_Size = count;
}

template <typename T>
DeviceVector<T>::DeviceVector(const std::vector<T> &other)
    : DeviceVector(other.size()) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
DeviceVector<T>::DeviceVector(const DeviceVector<T> &other)
    : DeviceVector(other.size()) {
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
DeviceVector<T>::DeviceVector(DeviceVector<T> &&other) noexcept
    : DeviceVector() {
  this->swap(other);
}

template <typename T> DeviceVector<T>::~DeviceVector(void) noexcept {
  ::deallocate_noexcept(&m_Data);
  m_Size = 0;
  m_Capacity = 0;
}

template <typename T>
DeviceVector<T> &DeviceVector<T>::operator=(const std::vector<T> &other) {
  this->reallocate(other.capacity());
  m_Size = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyHostToDevice));
  return *this;
}

template <typename T>
DeviceVector<T> &DeviceVector<T>::operator=(const DeviceVector<T> &other) {
  this->reallocate(other.capacity());
  m_Size = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(m_Data),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  return *this;
}

template <typename T>
DeviceVector<T> &DeviceVector<T>::operator=(DeviceVector<T> &&other) noexcept {
  if (this != &other) {
    DeviceVector<T> replacement(std::move(other));
    this->swap(replacement);
  }

  return *this;
}

template <typename T> const T *DeviceVector<T>::data(void) const {
  return m_Data;
}

template <typename T> T *DeviceVector<T>::data(void) { return m_Data; }

template <typename T> void DeviceVector<T>::assignData(T *data) {
  m_Data = data;
  return;
}

template <typename T> bool DeviceVector<T>::empty(void) const {
  return (m_Size == 0);
}

template <typename T> std::size_t DeviceVector<T>::size(void) const {
  return m_Size;
}

template <typename T> std::size_t DeviceVector<T>::capacity(void) const {
  return m_Capacity;
}

template <typename T> void DeviceVector<T>::shrink_to_fit(void) {
  this->reallocate(m_Size);
  return;
}

template <typename T> void DeviceVector<T>::clear(void) {
  this->deallocate();
  return;
}

/**
 * @brief Stores an appended value in the last active device slot.
 *
 * The current caller launches one block with one thread on the default stream.
 * The coordinate guard makes only global thread zero perform the write.
 *
 * @tparam T Element representation stored by value.
 * @param[in,out] data Device buffer containing at least `size` element slots.
 * The pointer is borrowed for the duration of the kernel.
 * @param[in] size One-based active length identifying the destination as
 * `data[size - 1]`.
 * @param[in] value Value to store in the destination slot.
 *
 * @pre `data` is non-null and `size` is greater than zero.
 */
template <typename T>
__global__ static void SetBackKernel(T *data, const std::size_t size,
                                     const T value) {
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
      (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
    data[size - 1] = value;
  return;
}

template <typename T> void DeviceVector<T>::push_back(const T &value) {
  if (m_Size >= m_Capacity) // Increase size of memory block by 50%
    this->reallocate(m_Capacity + (m_Capacity / 2) + 1);

  cudaCheckLaunch(SetBackKernel<<<1, 1>>>(m_Data, m_Size + 1, value));

  m_Size++;

  return;
}

template <typename T> void DeviceVector<T>::resize(const std::size_t count) {
  if (m_Capacity == 0)
    this->allocate(count);
  else if (m_Capacity < count)
    this->reallocate(count);
  m_Size = count;
  return;
}

template <typename T>
void DeviceVector<T>::swap(DeviceVector<T> &other) noexcept {
  std::swap(m_Size, other.m_Size);
  std::swap(m_Capacity, other.m_Capacity);
  std::swap(m_Data, other.m_Data);
  return;
}

template <typename T> void DeviceVector<T>::allocate(const std::size_t count) {
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&m_Data), count * sizeof(T)));
  m_Capacity = count;
  return;
}

template <typename T>
void DeviceVector<T>::reallocate(const std::size_t count) {
  if (count == m_Capacity) // No need for a new memory block
    return;

  if (count == 0) {
    this->deallocate();
    return;
  }

  const std::size_t copySize = (count < m_Size) ? count : m_Size;
  T *data = nullptr;

  // Stage the replacement through a provisional allocation so allocation and
  // prefix-copy failures preserve the old buffer. The old allocation is freed
  // before the replacement metadata is committed; a cudaFree failure therefore
  // leaves this vector reset while the catch path discards the provisional
  // buffer with the non-throwing cleanup helper.
  try {
    // Allocate new memory block
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));

    // Copy relevant data to new memory block
    if (copySize > 0) {
      cudaCheck(cudaMemcpy(static_cast<void *>(data),
                           static_cast<const void *>(m_Data),
                           copySize * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    // Free old memory block
    this->deallocate();
  } catch (...) {
    ::deallocate_noexcept(&data);
    throw;
  }

  // Assign new memory block
  m_Size = copySize;
  m_Capacity = count;
  m_Data = data;

  return;
}

template <typename T> void DeviceVector<T>::deallocate(void) {
  m_Size = 0;
  m_Capacity = 0;
  ::deallocate(&m_Data);
  return;
}
