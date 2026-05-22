set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__APO_DEBUG__")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -D__APO_DEBUG__")

add_library(apoCHARMMlib
  ${apoCHARMM_SOURCE_DIR}/src/AtomSelection.cpp
  ${apoCHARMM_SOURCE_DIR}/src/AtomSelector.cpp
  ${apoCHARMM_SOURCE_DIR}/src/SelectionParser.cpp
  ${apoCHARMM_SOURCE_DIR}/src/SelectionTokenizer.cpp
  ${apoCHARMM_SOURCE_DIR}/src/BEDSSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/CharmmContext.cu
  ${apoCHARMM_SOURCE_DIR}/src/Checkpoint.cu
  ${apoCHARMM_SOURCE_DIR}/src/CheckpointSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/Coordinates.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaContainer.cu
  # ${apoCHARMM_SOURCE_DIR}/src/CudaBAOABIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaBondedForce.cu
  ${apoCHARMM_SOURCE_DIR}/src/DynaSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/EnergyVirial.cpp
  ${apoCHARMM_SOURCE_DIR}/src/CudaEnergyVirial.cu
  ${apoCHARMM_SOURCE_DIR}/src/str_utils.cpp
  ${apoCHARMM_SOURCE_DIR}/src/cuda_utils.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaEMap.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaPMEDirectForce.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaDirectForceKernels.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaPMEReciprocalForce.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaPMEReciprocalGeneralForce.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaHolonomicConstraint.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaNeighborListBuild.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaNeighborListSort.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaNeighborList.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaTopExcl.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaPMERecip.cu
  ${apoCHARMM_SOURCE_DIR}/src/DcdSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/EDSForceManager.cu
  ${apoCHARMM_SOURCE_DIR}/src/EDSSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/Matrix3d.cu
  ${apoCHARMM_SOURCE_DIR}/src/reduce.cu
  ${apoCHARMM_SOURCE_DIR}/src/FEPSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/Force.cu
  ${apoCHARMM_SOURCE_DIR}/src/ForceManager.cu
  ${apoCHARMM_SOURCE_DIR}/src/ForceManagerComposite.cu
  ${apoCHARMM_SOURCE_DIR}/src/ForceManagerGenerator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CharmmCrd.cu
  ${apoCHARMM_SOURCE_DIR}/src/CharmmParameters.cu
  ${apoCHARMM_SOURCE_DIR}/src/CharmmPSF.cu
  ${apoCHARMM_SOURCE_DIR}/src/CharmmResidueTopology.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaLangevinThermostatIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaLeapFrogIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaVerletIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaVMMSVelocityVerletIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaVelocityVerletIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaNoseHooverThermostatIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaLangevinPistonIntegrator.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaP21NeighborListBuild.cu
  # ${apoCHARMM_SOURCE_DIR}/src/CudaExternalForce.cu
  ${apoCHARMM_SOURCE_DIR}/src/HarmonicRestraintForce.cu
  ${apoCHARMM_SOURCE_DIR}/src/CompositeSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/DualTopologySubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/BEDSForceManager.cu
  ${apoCHARMM_SOURCE_DIR}/src/MBARSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/FEPEIForceManager.cu
  ${apoCHARMM_SOURCE_DIR}/src/MBARForceManager.cu
  ${apoCHARMM_SOURCE_DIR}/src/CudaMinimizer.cu
  # ${apoCHARMM_SOURCE_DIR}/src/NetCDFSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/PDB.cu
  ${apoCHARMM_SOURCE_DIR}/src/PertForceManager.cu
  # ${apoCHARMM_SOURCE_DIR}/src/PrintEnergiesGraph.cu
  ${apoCHARMM_SOURCE_DIR}/src/RestartSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/ReplicaExchange.cu
  # ${apoCHARMM_SOURCE_DIR}/src/CudaIntegratorGraph.cu
  ${apoCHARMM_SOURCE_DIR}/src/NeighborListSort.cu
  ${apoCHARMM_SOURCE_DIR}/src/NonEquilibriumForceManager.cu
  # ${apoCHARMM_SOURCE_DIR}/src/SimpleLeapfrogGraph.cu
  ${apoCHARMM_SOURCE_DIR}/src/StateSubscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/Subscriber.cu
  ${apoCHARMM_SOURCE_DIR}/src/DeviceVector.cu
  # ${apoCHARMM_SOURCE_DIR}/src/TestForce.cu
  # ${apoCHARMM_SOURCE_DIR}/src/TestSystem.cu
  ${apoCHARMM_SOURCE_DIR}/src/XYZSubscriber.cu
  # ${apoCHARMM_SOURCE_DIR}/src/Utilities.cu
  ${apoCHARMM_SOURCE_DIR}/src/Logger.cu
  ${apoCHARMM_SOURCE_DIR}/src/tinyxml2.cpp
)

target_compile_definitions(apoCHARMMlib PUBLIC USE_TEXTURE_OBJECTS)

set_target_properties(apoCHARMMlib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
# set_target_properties(apoCHARMMlib PROPERTIES
#   CUDA_ARCHITECTURES "80"
# )
set_target_properties(apoCHARMMlib PROPERTIES
  CUDA_ARCHITECTURES "60;70;80;90"
)
# set_target_properties(apoCHARMMlib PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
# )
# set_target_properties(apoCHARMMlib PROPERTIES
#   COMPILE_FLAGS "-rdc=true"
# )
# set_target_properties(apoCHARMMlib PROPERTIES
#   LINK_FLAGS "-lcudadevrt"
# )
target_include_directories(apoCHARMMlib PUBLIC ${apoCHARMM_SOURCE_DIR}/include)

target_link_libraries(apoCHARMMlib
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::cufft
)
