# Add any new tests to this list!
set(SOURCES_UNITTESTS_TESTS
  # Base tests
  unittest-apoCharmmError.cpp
  unittest-strUtils.cpp
  unittest-deviceVector.cpp
  unittest-cudaContainer.cpp
  unittest-nothrowDestruction.cpp

  # Manager tests
  unittest-charmmContext.cpp
  unittest-forceManager.cpp

  # Object tests
  unittest-charmmParameters.cpp
  unittest-charmmPSF.cpp
  unittest-coordinates.cpp
  unittest-charmmCrd.cpp
  unittest-atomSelection.cpp

  unittest-cmap.cpp

  # Subscriber tests
  unittest-subscriber.cpp
  unittest-dcdSubscriber.cpp
  unittest-restartSubscriber.cpp

  # Integrator tests
  unittest-cudaIntegrator.cpp
  unittest-cudaNoseHooverIntegrator.cpp
  unittest-cudaLangevinThermostatIntegrator.cpp
  unittest-cudaLangevinPistonIntegrator.cpp

  # Restraint tests
  unittest-harmonicRestraintForce.cpp
  unittest-harmonicCenterOfMassRestraintForce.cpp
)

set(SOURCES_CAPI_TESTS
  # Base tests
  unittest-capiError.cpp

  # Manager tests
  unittest-capiCharmmContext.cpp
  unittest-capiForceManager.cpp

  # Object tests
  unittest-capiCharmmParameters.cpp
  unittest-capiCharmmPsf.cpp
  unittest-capiCharmmCrd.cpp
  unittest-capiAtomSelection.cpp

  # Subscriber tests
  unittest-capiSubscriber.cpp
  unittest-capiDcdSubscriber.cpp
  unittest-capiRestartSubscriber.cpp

  # Integrator tests
  unittest-capiCudaIntegrator.cpp
  unittest-capiCudaLangevinThermostatIntegrator.cpp
  unittest-capiCudaLangevinPistonIntegrator.cpp

  # Restraint tests
  unittest-capiHarmonicRestraintForce.cpp
  unittest-capiHarmonicCenterOfMassRestraintForce.cpp
)

###################################################
#### STUFF BELOW SHOULDN'T NEED TO BE MODIFIED ####
###################################################
########################################################################
set(SOURCES_UNITTESTS_MAIN
  unittest-main.cpp
)

set(SOURCES_UNITTESTS_HEADERS
  apo_test_helpers.h
  catch.hpp
)

# A function to check that the given hard-coded list contains all headers and
# sources in the given folder.
function(CheckFileList LIST_VAR FOLDER)
  file(GLOB GLOBBED_LIST CONFIGURE_DEPENDS
    "${FOLDER}/*.cpp"
    "${FOLDER}/*.cu"
    "${FOLDER}/*.h"
    "${FOLDER}/*.hpp"
    "${FOLDER}/*.hcu"
    "${FOLDER}/*.cuh"
  )

  list(REMOVE_ITEM GLOBBED_LIST ${${LIST_VAR}})

  if(GLOBBED_LIST)
    list(SORT GLOBBED_LIST)

    set(MESSAGE "The following files are in ${FOLDER} but are not listed in ${LIST_VAR}:\n")
    foreach(file_name ${GLOBBED_LIST})
      file(RELATIVE_PATH relative_file_name "${FOLDER}" "${file_name}")
      string(APPEND MESSAGE "  ${relative_file_name}\n")
    endforeach()

    string(APPEND MESSAGE "\nAdd them to ${LIST_VAR} in ${CMAKE_CURRENT_LIST_FILE}, or remove them if they are obsolete.")
    message(FATAL_ERROR "${MESSAGE}")
  endif()
endfunction()

set(UNITTEST_DIR ${PROJECT_SOURCE_DIR}/test/unittests)

set(HEADER_DIR ${UNITTEST_DIR})

set(APOCHARMM_DIR_DEFINITIONS
  APOCHARMM_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data"
  APOCHARMM_TOPPAR_DIR="${PROJECT_SOURCE_DIR}/toppar"
)

# Fail if any unit-test source or header is not listed.
set(SOURCES_ALL
  ${SOURCES_UNITTESTS_MAIN}
  ${SOURCES_UNITTESTS_TESTS}
  ${SOURCES_CAPI_TESTS}
  ${SOURCES_UNITTESTS_HEADERS}
)

foreach(name ${SOURCES_ALL})
  list(APPEND SOURCES_ALL_PATH ${UNITTEST_DIR}/${name})
endforeach()

CheckFileList(SOURCES_ALL_PATH ${UNITTEST_DIR})

set(TARGETS_UNITTESTS_TESTS)

set(TARGETS_CAPI_TESTS)

# Define shared Catch2 main object.
add_library(CatchMain OBJECT
  ${UNITTEST_DIR}/${SOURCES_UNITTESTS_MAIN}
  ${HEADER_DIR}/catch.hpp
)

foreach(source_file ${SOURCES_UNITTESTS_TESTS})
  get_filename_component(name "${source_file}" NAME_WE)
  list(APPEND TARGETS_UNITTESTS_TESTS ${name})

  add_executable(${name}
    ${UNITTEST_DIR}/${source_file}
    $<TARGET_OBJECTS:CatchMain>
    ${HEADER_DIR}/catch.hpp
  )

  target_compile_definitions(${name} PRIVATE
    ${APOCHARMM_DIR_DEFINITIONS}
  )

  # JEG260810: So we can use generic help function for C API testing
  target_include_directories(${name} PRIVATE
      ${apoCHARMM_SOURCE_DIR}/capi/private_include
      ${apoCHARMM_SOURCE_DIR}/capi/include
    )

  target_link_libraries(${name} PRIVATE
    ${EXTERNAL_LIBRARIES}
    apoCHARMMlib
    apocharmm_c
    CUDA::cudart
  )

  set_target_properties(${name} PROPERTIES
    CUDA_RUNTIME_LIBRARY SHARED
    LINKER_LANGUAGE CUDA
  )

  add_test(NAME ${name}
    COMMAND ${name}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/unittests
  )

  set_tests_properties(${name} PROPERTIES
    LABELS "cpp_api"
  )
endforeach()

if(TARGET apocharmm_c)
  target_compile_definitions(apocharmm_c PRIVATE
    APOCHARMM_C_TESTING
  )

  foreach(source_file ${SOURCES_CAPI_TESTS})
    get_filename_component(name "${source_file}" NAME_WE)
    list(APPEND TARGETS_CAPI_TESTS ${name})

    add_executable(${name}
      ${UNITTEST_DIR}/${source_file}
      $<TARGET_OBJECTS:CatchMain>
      ${HEADER_DIR}/catch.hpp
    )

    target_compile_definitions(${name} PRIVATE
      APOCHARMM_C_TESTING
      ${APOCHARMM_DIR_DEFINITIONS}
    )

    target_include_directories(${name} PRIVATE
      ${apoCHARMM_SOURCE_DIR}/capi/private_include
      ${apoCHARMM_SOURCE_DIR}/capi/include
    )

    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-pthread>
      $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-pthread>
      $<$<COMPILE_LANG_AND_ID:CXX,AppleClang>:-pthread>
    )

    target_link_options(${name} PRIVATE
      $<$<CXX_COMPILER_ID:GNU>:-pthread>
      $<$<CXX_COMPILER_ID:Clang>:-pthread>
      $<$<CXX_COMPILER_ID:AppleClang>:-pthread>
    )

    target_link_libraries(${name} PRIVATE
      apoCHARMMlib
      apocharmm_c
    )

    set_target_properties(${name} PROPERTIES
      LINKER_LANGUAGE CXX
    )

    add_test(NAME ${name}
      COMMAND ${name}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/unittests
    )

    set_tests_properties(${name} PROPERTIES
      LABELS "c_api"
    )
  endforeach()
endif()

set(TARGETS_ALL
  ${TARGETS_UNITTESTS_TESTS}
  ${TARGETS_CAPI_TESTS}
  CatchMain
)

foreach(name ${TARGETS_ALL})
  target_include_directories(${name} PRIVATE ${HEADER_DIR})

  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wmissing-noreturn>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-int-in-bool-context>

    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wweak-vtables>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wexit-time-destructors>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wglobal-constructors>

    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/w44265>
  )
endforeach()
