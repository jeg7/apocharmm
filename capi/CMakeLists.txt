add_library(apocharmm_c SHARED
  # C ABI utility
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/Error.cpp

  # Manager objects
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/CharmmContext.cpp
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/ForceManager.cpp

  # Primary objects
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/CharmmCrd.cpp
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/CharmmParameters.cpp
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/CharmmPsf.cpp

  # Integrators
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/CudaIntegrator.cpp
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/CudaNoseHooverIntegrator.cpp

  # Subscribers
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/Subscriber.cpp
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/DcdSubscriber.cpp
  ${apoCHARMM_SOURCE_DIR}/capi/src/apocharmm_c/RestartSubscriber.cpp
)

target_include_directories(apocharmm_c
  PUBLIC
  ${apoCHARMM_SOURCE_DIR}/capi/include
  PRIVATE
  ${apoCHARMM_SOURCE_DIR}/capi/private_include
)

target_link_libraries(apocharmm_c PRIVATE
  apoCHARMMlib
  CUDA::cudart
)

target_compile_definitions(apocharmm_c PRIVATE
  APOCHARMM_C_BUILD
)

set_target_properties(apocharmm_c PROPERTIES
  OUTPUT_NAME apocharmm_c
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY SHARED
)
