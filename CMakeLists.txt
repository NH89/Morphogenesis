cmake_minimum_required(VERSION 3.8  FATAL_ERROR)  
project(fluids_v4 LANGUAGES CXX CUDA)

find_package(CUDA 9.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} "/home/hoc041/Programming/Cuda/fluids_v4/")
###

add_library(fluid_systemPTX OBJECT  fluid_system_cuda.cu  fluid_system_cuda.cuh)
#target_compile_features(fluid_systemPTX PUBLIC cxx_std_11)
#set_target_properties( fluid_system  PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
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

#######
add_executable(fluids_v4 
        main.cpp 
        allocator.cu
        allocator.cuh
        fluid.cpp
        fluid.h
        matrix.cpp 
        matrix.h
        matrix_inline.h
        fluid_system.cpp  
        fluid_system.h
        vector.cpp
        vector.h
        vector_inline.h
        ${output_file}
        )
add_dependencies(fluids_v4 fluid_systemPTX)
target_include_directories(fluids_v4 PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )

target_compile_features(fluids_v4 PUBLIC cxx_std_11)  
#set_property(TARGET fluids_v4 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(fluids_v4 PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda)
#target_link_libraries(fluids_v4 PRIVATE fluid_systemPTX)

###
#nb edit code so that installed binary looks for ptx in correct location             
install(TARGETS fluid_systemPTX fluids_v4
   EXPORT fluid_systemPTX
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   OBJECTS DESTINATION ptx
   )
install(EXPORT fluid_systemPTX DESTINATION lib/fluid_systemPTX)

