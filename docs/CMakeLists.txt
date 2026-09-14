cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Allow the documentation to be configured without enabling the C++/CUDA
# project. Keep this version synchronized with the top-level project()
# declaration.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(apoCHARMM
    VERSION 1.0.0
    LANGUAGES NONE
  )

  get_filename_component(
    apoCHARMM_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ABSOLUTE
  )
endif()

find_package(Doxygen REQUIRED COMPONENTS doxygen)

set(APOCHARMM_DOCUMENTATION_PAGES
  "${CMAKE_CURRENT_SOURCE_DIR}/doxygen_mainpage.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/python_api.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/apocharmm_error.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/atom_selection.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/charmm_context.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/charmm_parameters.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/charmm_psf.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/coordinates.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda_container.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda_integrators.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/device_vector.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/distance_restraint_force.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/force_manager.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/harmonic_center_of_mass_restraint_force.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/harmonic_restraint_force.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/subscriber.md"
)

set(APOCHARMM_DOCUMENTATION_CPP_API
  "${apoCHARMM_SOURCE_DIR}/include/ApoCharmmError.h"
  "${apoCHARMM_SOURCE_DIR}/include/AtomSelection.h"
  "${apoCHARMM_SOURCE_DIR}/include/AtomSelector.h"
  "${apoCHARMM_SOURCE_DIR}/include/CharmmContext.h"
  "${apoCHARMM_SOURCE_DIR}/include/CharmmCrd.h"
  "${apoCHARMM_SOURCE_DIR}/include/CharmmPSF.h"
  "${apoCHARMM_SOURCE_DIR}/include/CharmmParameters.h"
  "${apoCHARMM_SOURCE_DIR}/include/Coordinates.h"
  "${apoCHARMM_SOURCE_DIR}/include/CudaContainer.h"
  "${apoCHARMM_SOURCE_DIR}/include/CudaIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/include/CudaLangevinPistonIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/include/CudaLangevinThermostatIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/include/CudaNoseHooverIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/include/DeviceVector.h"
  "${apoCHARMM_SOURCE_DIR}/include/DistanceRestraintForce.h"
  "${apoCHARMM_SOURCE_DIR}/include/ForceManager.h"
  "${apoCHARMM_SOURCE_DIR}/include/HarmonicCenterOfMassRestraintForce.h"
  "${apoCHARMM_SOURCE_DIR}/include/HarmonicRestraintForce.h"
  "${apoCHARMM_SOURCE_DIR}/include/PBC.h"
  "${apoCHARMM_SOURCE_DIR}/include/SelectionParser.h"
  "${apoCHARMM_SOURCE_DIR}/include/SelectionToken.h"
  "${apoCHARMM_SOURCE_DIR}/include/SelectionTokenizer.h"
  "${apoCHARMM_SOURCE_DIR}/include/Subscriber.h"
  "${apoCHARMM_SOURCE_DIR}/include/DcdSubscriber.h"
  "${apoCHARMM_SOURCE_DIR}/include/RestartSubscriber.h"
)

set(APOCHARMM_DOCUMENTATION_C_API
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/AtomSelection.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/AtomSelector.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CharmmContext.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CharmmCrd.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CharmmParameters.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CharmmPsf.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CudaIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CudaLangevinPistonIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CudaLangevinThermostatIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/CudaNoseHooverIntegrator.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/DcdSubscriber.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/DistanceRestraintForce.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/Enums.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/Error.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/ForceManager.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/HarmonicCenterOfMassRestraintForce.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/HarmonicRestraintForce.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/RestartSubscriber.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/Status.h"
  "${apoCHARMM_SOURCE_DIR}/capi/include/apocharmm_c/Subscriber.h"
)

set(APOCHARMM_DOCUMENTATION_PYTHON_API
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/atom_selection.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/atom_selector.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/charmm_context.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/charmm_crd.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/charmm_parameters.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/charmm_psf.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/cuda_integrator.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/cuda_langevin_piston_integrator.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/cuda_langevin_thermostat_integrator.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/cuda_nose_hoover_integrator.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/dcd_subscriber.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/distance_restraint_force.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/enums.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/error.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/force_manager.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/harmonic_center_of_mass_restraint_force.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/harmonic_restraint_force.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/restart_subscriber.py"
  "${apoCHARMM_SOURCE_DIR}/python/apocharmm/subscriber.py"
)

set(APOCHARMM_DOCUMENTATION_INPUTS
  ${APOCHARMM_DOCUMENTATION_PAGES}
  ${APOCHARMM_DOCUMENTATION_CPP_API}
  ${APOCHARMM_DOCUMENTATION_C_API}
  ${APOCHARMM_DOCUMENTATION_PYTHON_API}
)

set(DOXYGEN_PROJECT_NAME "${PROJECT_NAME}")
set(DOXYGEN_PROJECT_NUMBER "${PROJECT_VERSION}")
set(DOXYGEN_PROJECT_BRIEF
  "High-performance molecular dynamics simulations on GPUs"
)
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
  "${CMAKE_CURRENT_SOURCE_DIR}/doxygen_mainpage.md"
)

set(DOXYGEN_INPUT_ENCODING "UTF-8")
set(DOXYGEN_RECURSIVE NO)
set(DOXYGEN_MARKDOWN_SUPPORT YES)
set(DOXYGEN_AUTOLINK_SUPPORT YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

# The Python wrappers use Doxygen commands inside their docstrings.
set(DOXYGEN_PYTHON_DOCSTRING NO)

set(DOXYGEN_INTERNAL_DOCS NO)
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_EXTRACT_STATIC NO)
set(DOXYGEN_EXTRACT_LOCAL_CLASSES NO)
set(DOXYGEN_EXTRACT_LOCAL_METHODS NO)
set(DOXYGEN_EXTRACT_ANON_NSPACES NO)
set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
set(DOXYGEN_HIDE_UNDOC_CLASSES YES)
set(DOXYGEN_HIDE_UNDOC_NAMESPACES YES)
set(DOXYGEN_HIDE_UNDOC_RELATIONS YES)

set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
set(DOXYGEN_SEARCH_INCLUDES NO)
set(DOXYGEN_PREDEFINED
  "APOCHARMM_C_API="
  "__host__="
  "__device__="
  "__global__="
  "__forceinline__="
  "__shared__="
  "__constant__="
  "__managed__="
)
set(DOXYGEN_SKIP_FUNCTION_MACROS NO)

set(DOXYGEN_FULL_PATH_NAMES YES)
set(DOXYGEN_STRIP_FROM_PATH "${apoCHARMM_SOURCE_DIR}")
set(DOXYGEN_STRIP_FROM_INC_PATH
  "${apoCHARMM_SOURCE_DIR}/include"
  "${apoCHARMM_SOURCE_DIR}/capi/include"
)

# Do not publish generated special-purpose lists.
set(DOXYGEN_GENERATE_TODOLIST NO)
set(DOXYGEN_GENERATE_TESTLIST NO)
set(DOXYGEN_GENERATE_BUGLIST NO)
set(DOXYGEN_GENERATE_DEPRECATEDLIST NO)

# Do not publish source, file, or namespace browsing pages.
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_INLINE_SOURCES NO)
set(DOXYGEN_VERBATIM_HEADERS NO)
set(DOXYGEN_REFERENCES_LINK_SOURCE NO)
set(DOXYGEN_SHOW_INCLUDE_FILES NO)
set(DOXYGEN_SHOW_USED_FILES NO)
# Global C++, C, and Python module symbols require their containing file or
# package pages as documentation targets. These support pages are not placed in
# the primary navigation because the HTML index and tree view remain disabled.
set(DOXYGEN_SHOW_FILES YES)
set(DOXYGEN_SHOW_NAMESPACES YES)
set(DOXYGEN_ALPHABETICAL_INDEX NO)

# Use the subsystem hierarchy as a persistent left-hand navigation tree.
set(DOXYGEN_DISABLE_INDEX YES)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_SEARCHENGINE YES)
set(DOXYGEN_SERVER_BASED_SEARCH NO)
set(DOXYGEN_EXTERNAL_SEARCH NO)

# Keep normal documentation builds quiet while preserving strict diagnostics.
set(DOXYGEN_QUIET YES)
set(DOXYGEN_WARNINGS YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)
set(DOXYGEN_WARN_NO_PARAMDOC NO)
set(DOXYGEN_WARN_LOGFILE
  "${CMAKE_CURRENT_BINARY_DIR}/doxygen-warnings.log"
)
set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)

set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_HTML_OUTPUT "html")
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_GENERATE_XML NO)
set(DOXYGEN_GENERATE_MAN NO)
set(DOXYGEN_GENERATE_RTF NO)
set(DOXYGEN_GENERATE_DOCBOOK NO)

# Keep documentation generation explicit. Do not add ALL.
doxygen_add_docs(
  docs
  ${APOCHARMM_DOCUMENTATION_INPUTS}
  USE_STAMP_FILE
  WORKING_DIRECTORY "${apoCHARMM_SOURCE_DIR}"
  COMMENT "Generating apoCHARMM subsystem documentation with Doxygen"
)
