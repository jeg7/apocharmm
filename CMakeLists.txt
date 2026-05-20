cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(apoCHARMM
  VERSION 1.0.0
  LANGUAGES C CXX CUDA
)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Specify the C++ standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${apoCHARMM_SOURCE_DIR}/bin
  )

  # Compiler flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wno-deprecated-gpu-targets")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g3 -G -O0 -gencode=arch=compute_70,code=sm_70")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90")
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

add_subdirectory(${apoCHARMM_SOURCE_DIR}/src)
add_subdirectory(${apoCHARMM_SOURCE_DIR}/test)

# Turn on building the Python API
option(APOCHARMM_ENABLE_C_INTERFACE "Build apoCHARMM C ABI interface" ON)
