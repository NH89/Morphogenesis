cmake_minimum_required(VERSION 2.8)
project(fluids_v4)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)# nb Quadro k4000 is a 'Fermi' card and has compute capability 2.0
set(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS OFF)

include_directories(.)

cuda_add_executable(fluids_v4     
    main.cpp   
    fluid.cpp   
    matrix.cpp  
    fluid_system.cpp     
    vector.cpp
)

cuda_add_library(fluid_system_host   fluid_system_cuda.cu)
    
target_link_libraries(fluids_v4 cuda)
target_link_libraries(fluids_v4 fluid_system_host)
