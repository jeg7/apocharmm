# Add any new tests to this list!
set(PYTHON_API_TESTS
  # Utility tests
  python_api_error.py
  python_api_base.py

  # Manager tests
  python_api_charmm_context.py
  python_api_force_manager.py

  # Object tests
  python_api_charmm_parameters.py
  python_api_charmm_psf.py
  python_api_charmm_crd.py
  python_api_atom_reference.py
  python_api_atom_selection.py

  # Subscriber tests
  python_api_subscriber.py
  python_api_dcd_subscriber.py
  python_api_restart_subscriber.py

  # Integrator tests
  python_api_cuda_integrator.py
  python_api_cuda_nose_hoover_integrator.py
  python_api_cuda_langevin_thermostat_integrator.py
  python_api_cuda_langevin_piston_integrator.py

  # Restraint tests
  python_api_harmonic_restraint_force.py
  python_api_harmonic_center_of_mass_restraint_force.py
  python_api_distance_restraint_force.py
)

# Python files that are imported by tests, but should not be registered as tests.
set(PYTHON_API_SUPPORT_FILES
  apo_test_helpers.py
)

###################################################
#### STUFF BELOW SHOULDN'T NEED TO BE MODIFIED ####
###################################################
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PYTHON_API_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# A function to check that the hard-coded lists contain all Python files in the
# given folder.
function(CheckPythonFileList LIST_VAR FOLDER)
  file(GLOB GLOBBED_LIST CONFIGURE_DEPENDS
    "${FOLDER}/*.py"
  )

  list(REMOVE_ITEM GLOBBED_LIST ${${LIST_VAR}})

  if(GLOBBED_LIST)
    list(SORT GLOBBED_LIST)

    set(MESSAGE "The following Python files are in ${FOLDER} but are not listed in ${LIST_VAR}:\n")
    foreach(file_name ${GLOBBED_LIST})
      file(RELATIVE_PATH relative_file_name "${FOLDER}" "${file_name}")
      string(APPEND MESSAGE "  ${relative_file_name}\n")
    endforeach()

    string(APPEND MESSAGE "\nAdd test files to PYTHON_API_TESTS, add helper files to PYTHON_API_SUPPORT_FILES, or remove obsolete files.")
    message(FATAL_ERROR "${MESSAGE}")
  endif()
endfunction()

# Fail if any Python test/support file is not listed.
set(PYTHON_API_ALL_FILES
  ${PYTHON_API_TESTS}
  ${PYTHON_API_SUPPORT_FILES}
)

foreach(name ${PYTHON_API_ALL_FILES})
  list(APPEND PYTHON_API_ALL_PATHS ${PYTHON_API_TEST_DIR}/${name})
endforeach()

CheckPythonFileList(PYTHON_API_ALL_PATHS ${PYTHON_API_TEST_DIR})

foreach(test_file ${PYTHON_API_TESTS})
  get_filename_component(test_name "${test_file}" NAME_WE)

  add_test(NAME ${test_name}
    COMMAND ${Python3_EXECUTABLE} ${PYTHON_API_TEST_DIR}/${test_file}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )

  set_tests_properties(${test_name} PROPERTIES
    LABELS "python_api"
    ENVIRONMENT "APOCHARMM_REPO_ROOT=${PROJECT_SOURCE_DIR};APOCHARMM_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/test/data;APOCHARMM_TOPPAR_DIR=${PROJECT_SOURCE_DIR}/toppar;APOCHARMM_LIBRARY_PATH=$<TARGET_FILE:apocharmm_c>;PYTHONPATH=${PROJECT_SOURCE_DIR}/python:$ENV{PYTHONPATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:apocharmm_c>:$ENV{LD_LIBRARY_PATH}"
  )
endforeach()

add_custom_target(check-python-api
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L python_api DEPENDS apocharmm_c
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
