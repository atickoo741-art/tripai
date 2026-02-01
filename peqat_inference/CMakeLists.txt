cmake_minimum_required(VERSION 3.18)
project(peqat_inference LANGUAGES CUDA CXX)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# CUDA architecture for DGX Spark (Blackwell)
set(CMAKE_CUDA_ARCHITECTURES 90)

# Optional: Add other architectures for compatibility
# set(CMAKE_CUDA_ARCHITECTURES "86;89;90")  # Ampere, Ada, Blackwell

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Source files
set(CUDA_SOURCES
    peqat_kernels.cu
)

set(CPP_SOURCES
    peqat_ops.cpp
)

# Build shared library
add_library(peqat_kernels SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Link CUDA runtime
target_link_libraries(peqat_kernels
    CUDA::cudart
)

# Include directories
target_include_directories(peqat_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set CUDA properties
set_target_properties(peqat_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Build test executable
add_executable(test_peqat test_peqat.cpp)

target_link_libraries(test_peqat
    peqat_kernels
    CUDA::cudart
)

target_include_directories(test_peqat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
