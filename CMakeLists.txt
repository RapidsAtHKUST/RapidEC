cmake_minimum_required(VERSION 3.10)
project(crypto-accelerator LANGUAGES CXX CUDA)

find_package(CUDA)
find_library(gmp gmp)
find_package(OpenMP)

set(CMAKE_CXX_FLAGS "-O3")
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "All")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")

add_subdirectory(src)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

set_target_properties(RapidSV PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main main.cpp)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main PRIVATE RapidSV ${CUDA_LIBRARIES} gmp OpenMP::OpenMP_CXX)
