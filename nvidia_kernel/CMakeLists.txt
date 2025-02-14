cmake_minimum_required(VERSION 3.21)
SET(CMAKE_INSTALL_PREFIX "$ENV{MLDIR}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(Kernels VERSION 1.0 LANGUAGES CXX CUDA)
#set_source_files_properties(kernel.cu PROPERTIES LANGUAGE ${GPU_RUNTIME})
include_directories("../inc")
add_library(CUDAKernel SHARED kernel.cu)

target_include_directories(CUDAKernel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(CUDAKernel PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

install(TARGETS CUDAKernel)
install(FILES CUDAKernel.h DESTINATION inc)

