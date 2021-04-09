cmake_minimum_required(VERSION 3.8  FATAL_ERROR)  

cmake_policy(SET CMP0054 NEW)

project(Morphogenesis LANGUAGES C CXX CUDA)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "~/apps/morphogenesis/0.1.1" CACHE PATH "..." FORCE)
endif()
message ( "CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")


set(CMAKE_C_STANDARD 17)
#set(CMAKE_CPP_STANDARD 11) # set(CMAKE_CPP_STANDARD 17)  NB need (<=11 for VTP)  &&  (>+11 for C++ threads)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)


find_package(CUDA REQUIRED)

find_package(VTK COMPONENTS 
    CommonCore
    IOGeometry
    IOLegacy
    IOXML
    InteractionStyle
)

if (NOT VTK_FOUND)
  message("VTK not found: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")

include_directories(${CUDA_INCLUDE_DIRS} $(VTK_INCLUDE_DIRS) )#"/home/hoc041/Programming/Cuda/fluids_v4/"

add_library(fluid_systemPTX OBJECT  fluid_system_cuda.cu  fluid_system_cuda.cuh)
set_property(TARGET fluid_systemPTX PROPERTY CUDA_PTX_COMPILATION ON)
set(output_file ${CMAKE_CURRENT_BINARY_DIR}/embedded_objs.h)

get_filename_component(cuda_compiler_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
find_program(bin_to_c  NAMES bin2c  PATHS ${cuda_compiler_bin}  )
if(NOT bin_to_c)
  message(FATAL_ERROR
    "bin2c not found:\n"
    "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
    "  cuda_compiler_bin='${cuda_compiler_bin}'\n"
    )
endif()

add_custom_command(
  OUTPUT "${output_file}"
  COMMAND ${CMAKE_COMMAND}
    "-DBIN_TO_C_COMMAND=${bin_to_c}"
    "-DOBJECTS=$<TARGET_OBJECTS:fluid_systemPTX>"
    "-DOUTPUT=${output_file}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/bin2c_wrapper.cmake
  VERBATIM
  DEPENDS $<TARGET_OBJECTS:fluid_systemPTX>
  COMMENT "Converting Object files to a C header"
  )

add_executable(load_sim
        load_sim.cpp
        #allocator.cu
        #allocator.cuh
        fluid.cpp
        fluid.h
        matrix.cpp 
        matrix.h
        matrix_inline.h
        fluid_system.cpp  
        file_IO.cpp
        host_CUDA.cpp
        fluid_system.h
        vector.cpp
        vector.h
        vector_inline.h
        ${output_file}
)

add_executable(thread_demo
        thread_demo.cpp
        thread_manager.cpp
        thread_manager.h
        ${output_file}
)

add_executable(check_demo
        check_demo.cpp
        #allocator.cu
        #allocator.cuh
        fluid.cpp
        fluid.h
        matrix.cpp 
        matrix.h
        matrix_inline.h
        fluid_system.cpp    
        file_IO.cpp
        host_CUDA.cpp
        fluid_system.h
        vector.cpp
        vector.h
        vector_inline.h
        ${output_file}
)

add_executable(make_demo
        make_demo.cpp
        #allocator.cu
        #allocator.cuh
        fluid.cpp
        fluid.h
        matrix.cpp 
        matrix.h
        matrix_inline.h
        fluid_system.cpp    
        file_IO.cpp
        host_CUDA.cpp
        fluid_system.h
        vector.cpp
        vector.h
        vector_inline.h
        ${output_file}
)

add_executable(make_demo2
        make_demo2.cpp
        #allocator.cu
        #allocator.cuh
        fluid.cpp
        fluid.h
        matrix.cpp 
        matrix.h
        matrix_inline.h
        fluid_system.cpp    
        file_IO.cpp
        host_CUDA.cpp
        fluid_system.h
        vector.cpp
        vector.h
        vector_inline.h
        ${output_file}
)

add_executable(SpecfileBatchGenerator
        SpecfileBatchGenerator.cpp
        #allocator.cu
        #allocator.cuh
        fluid.cpp
        fluid.h
        matrix.cpp 
        matrix.h
        matrix_inline.h
        fluid_system.cpp    
        file_IO.cpp
        host_CUDA.cpp
        fluid_system.h
        vector.cpp
        vector.h
        vector_inline.h
        ${output_file}
)

#file:///home/nick/Programming/Cuda/Morphogenesis/src/SpecfileBatchGenerator.cpp
        
add_dependencies(load_sim fluid_systemPTX)
add_dependencies(check_demo fluid_systemPTX)
add_dependencies(make_demo fluid_systemPTX)
add_dependencies(make_demo2 fluid_systemPTX)
add_dependencies(SpecfileBatchGenerator fluid_systemPTX)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(load_sim PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(check_demo PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(make_demo PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(make_demo2 PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(thread_demo PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(SpecfileBatchGenerator PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )

target_compile_features(load_sim PUBLIC cxx_std_11)
target_compile_features(check_demo PUBLIC cxx_std_11)
target_compile_features(make_demo PUBLIC cxx_std_11)
target_compile_features(make_demo2 PUBLIC cxx_std_11)
target_compile_features(thread_demo PUBLIC cxx_std_11)
target_compile_features(SpecfileBatchGenerator PUBLIC cxx_std_11)

target_link_libraries(load_sim PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${VTK_LIBRARIES} VTK::IOExport )
target_link_libraries(check_demo PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${VTK_LIBRARIES} VTK::IOExport )
target_link_libraries(make_demo PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${VTK_LIBRARIES} VTK::IOExport )
target_link_libraries(make_demo2 PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${VTK_LIBRARIES} VTK::IOExport )
target_link_libraries(SpecfileBatchGenerator PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${VTK_LIBRARIES} VTK::IOExport )
target_link_libraries(thread_demo PRIVATE -lpthread )

#nb edit code so that installed binary looks for ptx in correct location             
install(TARGETS fluid_systemPTX make_demo make_demo2 check_demo load_sim thread_demo SpecfileBatchGenerator
   EXPORT fluid_systemPTX
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   OBJECTS DESTINATION ptx
   )
install(EXPORT fluid_systemPTX DESTINATION lib/fluid_systemPTX)

#set(ENV{MORPHOGENESIS_HOME} ${CMAKE_INSTALL_PREFIX})
#set(ENV_CMD "export MORPHOGENESIS_HOME=\"${CMAKE_INSTALL_PREFIX}\"")
#execute_process(COMMAND export MORPHOGENESIS_HOME=/home/nick/apps/morphogenesis/0.1.1) 
#${CMAKE_INSTALL_PREFIX}
#execute_process(COMMAND echo $MORPHOGENESIS_HOME)
#message("Environment variable MORPHOGENESIS_HOME: $ENV{MORPHOGENESIS_HOME}")
