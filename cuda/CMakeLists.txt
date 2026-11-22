cmake_minimum_required(VERSION 3.24)

# before project(), or MSVC's makefile generators have already picked Debug
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

project(gpu_sort LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# builds for the GPU in this machine unless told otherwise, e.g. -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(gpu_sort src/main.cu)
target_include_directories(gpu_sort PRIVATE src)
target_compile_options(gpu_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# the CUDA 13 headers refuse MSVC's traditional preprocessor
if(MSVC)
  target_compile_options(gpu_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
