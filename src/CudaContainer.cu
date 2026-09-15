// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Andrew Simmonett, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaContainer.h"

#include "ApoCharmmError.h"
#include "cuda_utils.h"

#include <cstdio>
#include <string>
#include <utility>

template <typename T>
CudaContainer<T>::CudaContainer(void) : m_HostArray(), m_DeviceArray() {}

template <typename T>
CudaContainer<T>::CudaContainer(const std::size_t count)
    : m_HostArray(count), m_DeviceArray(count) {}

template <typename T>
CudaContainer<T>::CudaContainer(const std::vector<T> &other)
    : m_HostArray(other), m_DeviceArray(other.size()) {
  this->transferToDevice();
}

template <typename T>
CudaContainer<T>::CudaContainer(std::vector<T> &&other)
    : m_HostArray(), m_DeviceArray(other.size()) {
  m_HostArray.swap(other);
  this->transferToDevice();
}

template <typename T>
CudaContainer<T>::CudaContainer(const DeviceVector<T> &other)
    : m_HostArray(other.size()), m_DeviceArray(other) {
  this->transferToHost();
}

template <typename T>
CudaContainer<T>::CudaContainer(DeviceVector<T> &&other)
    : m_HostArray(other.size()), m_DeviceArray() {
  m_DeviceArray.swap(other);
  this->transferToHost();
}

template <typename T>
CudaContainer<T>::CudaContainer(const CudaContainer<T> &other)
    : m_HostArray(other.getHostArray()), m_DeviceArray(other.getDeviceArray()) {
}

template <typename T>
CudaContainer<T>::CudaContainer(CudaContainer<T> &&other) noexcept
    : m_HostArray(), m_DeviceArray() {
  m_HostArray.swap(other.m_HostArray);
  m_DeviceArray.swap(other.m_DeviceArray);
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(const std::vector<T> &other) {
  m_HostArray = other;
  m_DeviceArray.resize(other.size());
  this->transferToDevice();
  return *this;
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(std::vector<T> &&other) {
  CudaContainer<T> replacement(std::move(other));
  *this = std::move(replacement);
  return *this;
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(const DeviceVector<T> &other) {
  m_DeviceArray = other;
  m_HostArray.resize(other.size());
  this->transferToHost();
  return *this;
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(DeviceVector<T> &&other) {
  CudaContainer<T> replacement(std::move(other));
  *this = std::move(replacement);
  return *this;
}

template <typename T>
CudaContainer<T> &CudaContainer<T>::operator=(const CudaContainer<T> &other) {
  m_HostArray = other.getHostArray();
  m_DeviceArray = other.getDeviceArray();
  return *this;
}

template <typename T>
CudaContainer<T> &
CudaContainer<T>::operator=(CudaContainer<T> &&other) noexcept {
  if (this != &other) {
    CudaContainer<T> replacement(std::move(other));
    m_HostArray.swap(replacement.m_HostArray);
    m_DeviceArray.swap(replacement.m_DeviceArray);
  }

  return *this;
}

template <typename T>
const T &CudaContainer<T>::at(const std::size_t pos) const {
  APOCHARMM_REQUIRE(pos < m_HostArray.size(),
                    ApoCharmmErrorCode::InvalidArgument,
                    "CudaContainer index is out of range; expected [0, " +
                        std::to_string(m_HostArray.size()) + "), observed " +
                        std::to_string(pos));

  return m_HostArray[pos];
}

template <typename T> T &CudaContainer<T>::at(const std::size_t pos) {
  APOCHARMM_REQUIRE(pos < m_HostArray.size(),
                    ApoCharmmErrorCode::InvalidArgument,
                    "CudaContainer index is out of range; expected [0, " +
                        std::to_string(m_HostArray.size()) + "), observed " +
                        std::to_string(pos));

  return m_HostArray[pos];
}

template <typename T>
const T &CudaContainer<T>::operator[](const std::size_t pos) const {
  return m_HostArray[pos];
}

template <typename T> T &CudaContainer<T>::operator[](const std::size_t pos) {
  return m_HostArray[pos];
}

template <typename T>
const std::vector<T> &CudaContainer<T>::getHostArray(void) const {
  return m_HostArray;
}

template <typename T> std::vector<T> &CudaContainer<T>::getHostArray(void) {
  return m_HostArray;
}

template <typename T>
const DeviceVector<T> &CudaContainer<T>::getDeviceArray(void) const {
  return m_DeviceArray;
}

template <typename T> DeviceVector<T> &CudaContainer<T>::getDeviceArray(void) {
  return m_DeviceArray;
}

template <typename T> std::size_t CudaContainer<T>::size(void) const {
  return m_HostArray.size();
}

template <typename T> void CudaContainer<T>::shrink_to_fit(void) {
  m_HostArray.shrink_to_fit();
  m_DeviceArray.shrink_to_fit();
  return;
}

template <typename T> void CudaContainer<T>::clear(void) {
  m_HostArray.clear();
  m_DeviceArray.clear();
  return;
}

template <typename T> void CudaContainer<T>::push_back(const T &value) {
  // JEG260420: I imagine this is super slow, since there is the potential to
  // have to reallocate memory both on the host and device. This should really
  // only be used during some kind of set up ONLY IF there's no easy way to know
  // the size of memory needed before hand.
  m_HostArray.push_back(value);
  m_DeviceArray.push_back(value);
  return;
}

template <typename T> void CudaContainer<T>::resize(const std::size_t count) {
  m_HostArray.resize(count);
  m_DeviceArray.resize(count);
  return;
}

template <typename T> void CudaContainer<T>::set(const std::vector<T> &values) {
  m_HostArray = values;
  m_DeviceArray.resize(values.size());
  this->transferToDevice();
  return;
}

template <typename T>
void CudaContainer<T>::set(const DeviceVector<T> &values) {
  m_DeviceArray = values;
  m_HostArray.resize(values.size());
  this->transferToHost();
  return;
}

template <typename T> void CudaContainer<T>::set(const T value) {
  m_HostArray.assign(m_HostArray.size(), value);
  this->transferToDevice();
  return;
}

template <typename T> void CudaContainer<T>::setToValue(const T value) {
  this->set(value);
  return;
}

// JEG260813: Transfers intentionally do not resize either mirror. When entered
// with equal active lengths, normal successful modifiers preserve that
// relationship. Mutable mirror access is an escape hatch whose caller must
// restore the invariant before copying. Both directions use cudaMemcpy without
// a stream argument, then synchronize the current CUDA device.
template <typename T> void CudaContainer<T>::transferToDevice(void) {
  if (m_HostArray.empty())
    return;
  cudaCheck(cudaMemcpy(static_cast<void *>(m_DeviceArray.data()),
                       static_cast<const void *>(m_HostArray.data()),
                       m_HostArray.size() * sizeof(T), cudaMemcpyHostToDevice));
  cudaCheck(cudaDeviceSynchronize());
  return;
}

template <typename T> void CudaContainer<T>::transferToHost(void) {
  if (m_HostArray.empty())
    return;
  cudaCheck(cudaMemcpy(static_cast<void *>(m_HostArray.data()),
                       static_cast<const void *>(m_DeviceArray.data()),
                       m_DeviceArray.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaDeviceSynchronize());
  return;
}

template <typename T> void CudaContainer<T>::transferFromDevice(void) {
  this->transferToHost();
  return;
}

template <typename T> void CudaContainer<T>::transferFromHost(void) {
  this->transferToDevice();
  return;
}

// JEG260813: The print-kernel overload set is restricted to the element types
// explicitly instantiated in CudaContainer.h. Each thread prints at most one
// element, and CUDA device printf does not guarantee line ordering across
// threads.
__global__ void printKernel(const int *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %d\n", index, data[index]);
  return;
}

__global__ void printKernel(const int2 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %d, %d\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const int3 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %d, %d, %d\n", index, data[index].x, data[index].y,
           data[index].z);
  }
  return;
}

__global__ void printKernel(const int4 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %d, %d, %d, %d\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  }
  return;
}

__global__ void printKernel(const unsigned int *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %u\n", index, data[index]);
  return;
}

__global__ void printKernel(const float *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e\n", index, data[index]);
  return;
}

__global__ void printKernel(const float2 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const float3 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z);
  }
  return;
}

__global__ void printKernel(const float4 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %e, %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  }
  return;
}

__global__ void printKernel(const long long int *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %lld\n", index, data[index]);
  return;
}

__global__ void printKernel(const longlong2 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %lld, %lld\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const longlong3 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %lld, %lld, %lld\n", index, data[index].x, data[index].y,
           data[index].z);
  }
  return;
}

__global__ void printKernel(const longlong4 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %lld, %lld, %lld, %lld\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  }
  return;
}

__global__ void printKernel(const unsigned long long int *data,
                            const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %llu\n", index, data[index]);
  return;
}

__global__ void printKernel(const std::size_t *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %llu\n", index, data[index]);
  return;
}

__global__ void printKernel(const double *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e\n", index, data[index]);
  return;
}

__global__ void printKernel(const double2 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    printf("%u: %e, %e\n", index, data[index].x, data[index].y);
  return;
}

__global__ void printKernel(const double3 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z);
  }
  return;
}

__global__ void printKernel(const double4 *data, const std::size_t size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("%u: %e, %e, %e, %e\n", index, data[index].x, data[index].y,
           data[index].z, data[index].w);
  }
  return;
}

// JEG260813: Launch geometry and the device read range both use the host-side
// logical length. This relies on equal mirror sizes, a nonzero length, and an
// element count representable by unsigned int kernel indexing.
template <typename T> void CudaContainer<T>::printDeviceArray(void) const {
  constexpr unsigned int blockDim = 256;
  const unsigned int gridDim = (this->size() + blockDim - 1) / blockDim;

  cudaCheckLaunch(
      printKernel<<<gridDim, blockDim>>>(m_DeviceArray.data(), this->size()));
  cudaCheck(cudaDeviceSynchronize());

  return;
}
