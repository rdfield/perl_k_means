cmake_minimum_required(VERSION 3.21)
SET(CMAKE_INSTALL_PREFIX "$ENV{MLDIR}")

project(Kernels VERSION 1.0)
set(GPU_RUNTIME "HIP")
enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
set(ROCM_ROOT "/opt/rocm")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
set_source_files_properties(kernel.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})
include_directories("../inc")
add_library(ROCMKernel SHARED kernel.hip)

target_include_directories(ROCMKernel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ROCMKernel)
install(FILES ROCMKernel.h DESTINATION inc)
