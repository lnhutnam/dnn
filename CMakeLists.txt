cmake_minimum_required(VERSION 3.20)
project(myproject LANGUAGES CXX CUDA)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80) # Adjust based on your hardware

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/third_party/eigen")

# Add the src subdirectory
add_subdirectory(src)

# Add the executable target
add_executable(demo demo.cc)

# Link CUDA and other required libraries to the executable
target_link_libraries(demo PRIVATE MiniDNNLib CUDA::cudart)
