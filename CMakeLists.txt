# cmake_minimum_required(VERSION 2.8)
# project(fluids_v4)
# 
# find_package(CUDA REQUIRED)
# 
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_60,code=sm_60)# nb Quadro k4000 is a 'Fermi' card and has compute capability 2.0
# set(CUDA_SEPARABLE_COMPILATION ON)
# set(BUILD_SHARED_LIBS OFF)
# 
# include_directories(.)
# 
# 
# _add_package_CUDA()
# 
# _add_package_CUDPP()
# 
# cuda_add_executable(fluids_v4     
#     main.cpp   
#     allocator.cpp
#     fluid.cpp   
#     matrix.cpp  
#     fluid_system.cpp     
#     vector.cpp
# )
# 
# cuda_add_library(fluid_system_host   fluid_system_cuda.cu)
#     
# target_link_libraries(fluids_v4 cuda)
# target_link_libraries(fluids_v4 fluid_system_host)

#############################################################################
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

add_library(fluid_system STATIC
  fluid_system_cuda.cu
  fluid_system_cuda.cuh
  fluid.cpp
  fluid.h
  )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(fluid_system PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( fluid_system
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

if(BUILD_TESTING)

  add_executable(fluids_v4 
        main.cpp 
        allocator.cpp
        allocator.h
        fluid.cpp
        fluid.h
        matrix.cpp 
        matrix.h
        matrix_inline.h
        fluid_system.cpp  
        fluid_system.h
        vector.cpp
        vector.h
        vector_inline.h)

  set_target_properties(fluids_v4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(fluids_v4 PRIVATE fluid_system)

#   add_test(NAME particles_10k COMMAND fluids_v4 10000 )
#   add_test(NAME particles_256k COMMAND fluids_v4 256000 )

  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET fluids_v4 PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()

