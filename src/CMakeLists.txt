# Define the project name and version
project(MiniDNN LANGUAGES CXX CUDA)

# Specify the minimum required version of CMake
cmake_minimum_required(VERSION 3.20)

# Enable C++14 and CUDA architectures
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# Define source directories
set(LAYER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/layer)
set(LOSS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/loss)
set(OPTIMIZER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/optimizer)
set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/layer/kernels)

# Collect source files
file(GLOB LIB_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${LAYER_DIR}/*.cpp
    ${LOSS_DIR}/*.cpp
    ${OPTIMIZER_DIR}/*.cpp
)

file(GLOB CUDA_SRCS
    ${KERNELS_DIR}/*.cu
)

# Add the main library
add_library(MiniDNNLib ${LIB_SRCS})

# Add the CUDA library
add_library(MiniDNNCudaLib ${CUDA_SRCS})

# Set properties for the CUDA library
set_target_properties(MiniDNNCudaLib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link the CUDA library with the main library
target_link_libraries(MiniDNNLib PRIVATE MiniDNNCudaLib)

# Include directories for the libraries
target_include_directories(MiniDNNLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
