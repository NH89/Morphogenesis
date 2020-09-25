cmake_minimum_required(VERSION 3.8  FATAL_ERROR)  
project(Morphogenesis LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CPP_STANDARD 17)

find_package(CUDA 9.1 REQUIRED)
#find_package(HDF5 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} )#"/home/hoc041/Programming/Cuda/fluids_v4/"

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "./install" CACHE PATH "..." FORCE)
endif()
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
add_executable(load_sim
        load_sim.cpp
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
        thread_manager.cpp
        thread_manager.h
        ${output_file}
)

add_executable(thread_demo
        thread_demo.cpp
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
        thread_manager.cpp
        thread_manager.h
        ${output_file}
)

add_executable(check_demo
        check_demo.cpp
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
        thread_manager.cpp
        thread_manager.h
        ${output_file}
)

add_executable(make_demo
        make_demo.cpp
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
        thread_manager.cpp
        thread_manager.h
        ${output_file}
)

add_executable(morphogenesis 
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
        thread_manager.cpp
        thread_manager.h
        ${output_file}
        )
    
        
add_dependencies(load_sim fluid_systemPTX)
add_dependencies(check_demo fluid_systemPTX)
add_dependencies(make_demo fluid_systemPTX)
add_dependencies(morphogenesis fluid_systemPTX)
add_dependencies(thread_demo fluid_systemPTX)

target_include_directories(load_sim PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(check_demo PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(make_demo PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(morphogenesis PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories(thread_demo PRIVATE  ${CMAKE_CURRENT_BINARY_DIR} )

target_compile_features(load_sim PUBLIC cxx_std_11)
target_compile_features(check_demo PUBLIC cxx_std_11)
target_compile_features(make_demo PUBLIC cxx_std_11)
target_compile_features(morphogenesis PUBLIC cxx_std_11)
target_compile_features(thread_demo PUBLIC cxx_std_11)
#set_property(TARGET fluids_v4 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(load_sim PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${HDF5_LIBRARIES}  ) # ${HDF5_CXX_LIBRARIES}
target_link_libraries(check_demo PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${HDF5_LIBRARIES}  ) # ${HDF5_CXX_LIBRARIES}
target_link_libraries(make_demo PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${HDF5_LIBRARIES}  ) # ${HDF5_CXX_LIBRARIES}
target_link_libraries(morphogenesis PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${HDF5_LIBRARIES}  ) # ${HDF5_CXX_LIBRARIES}
target_link_libraries(thread_demo PRIVATE "cudart" ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}  -lcuda ${HDF5_LIBRARIES}  ) # ${HDF5_CXX_LIBRARIES}
#target_link_libraries(fluids_v4 PRIVATE fluid_systemPTX)

###
#nb edit code so that installed binary looks for ptx in correct location             
install(TARGETS fluid_systemPTX morphogenesis make_demo check_demo load_sim thread_demo
   EXPORT fluid_systemPTX
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   OBJECTS DESTINATION ptx
   )
install(EXPORT fluid_systemPTX DESTINATION lib/fluid_systemPTX)


#file(MAKE_DIRECTORY check demo out)	# subdirectories for data



#How to set default CMAKE_INSTALL_PREFIX ?
