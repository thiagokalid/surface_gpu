cmake_minimum_required(VERSION 3.10)
project(CudaCppProject LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)  # Replace with your target architecture(s)

# Compiler flags for C++ and CUDA
set(CXX_FLAGS -O3 -Wall -fPIC)  # Corrected to pass each flag separately

set(CUDA_FLAGS 
    -O3
    -Xcompiler -Wall
    -Xcompiler -fPIC
    -Xcompiler -std=c++14
)

# Directories
set(SRC_DIR src)
set(CPP_DIR ${SRC_DIR}/cpp)
set(INCLUDE_DIR ${CPP_DIR}/include)
set(BUILD_DIR build)

# Find all the source files
file(GLOB CPP_SOURCES "${CPP_DIR}/*.cpp")
file(GLOB CUDA_SOURCES "${CPP_DIR}/*.cu")

# Debugging: Check if CUDA sources were found
message(STATUS "CUDA sources: ${CUDA_SOURCES}")

# Debugging: Check if CPP sources were found
message(STATUS "CPP sources: ${CPP_SOURCES}")


# Create object files for C++ sources
foreach(CPP_FILE ${CPP_SOURCES})
    get_filename_component(CPP_FILE_NAME ${CPP_FILE} NAME_WE)
    set(CPP_OBJ_FILE "${BUILD_DIR}/${CPP_FILE_NAME}.o")
    list(APPEND CPP_OBJ_FILES ${CPP_OBJ_FILE})
endforeach()

# Create shared object files (.so) for CUDA sources
foreach(CUDA_FILE ${CUDA_SOURCES})
    get_filename_component(CUDA_FILE_NAME ${CUDA_FILE} NAME_WE)
    set(CUDA_OBJ_FILE "${BUILD_DIR}/${CUDA_FILE_NAME}.so")
    list(APPEND CUDA_OBJ_FILES ${CUDA_OBJ_FILE})
endforeach()

# Debugging: Check if CUDA sources were found
message(STATUS "CUDA objects: ${CUDA_OBJ_FILES}")

# Debugging: Check if CPP sources were found
message(STATUS "CPP objects: ${CPP_OBJ_FILES}")

# Include directories
include_directories(${INCLUDE_DIR})

# Add the C++ source files (as object files)
add_library(cpp_objects OBJECT ${CPP_SOURCES})
target_compile_options(cpp_objects PRIVATE ${CXX_FLAGS})

# Add the CUDA source files (as shared objects)
add_library(cuda_objects SHARED ${CUDA_SOURCES})
target_compile_options(cuda_objects PRIVATE ${CUDA_FLAGS})  # Now passing flags correctly

# Combine both C++ and CUDA object files into a custom target
add_custom_target(build_all
    DEPENDS ${CPP_OBJ_FILES} ${CUDA_OBJ_FILES}
)

# Clean target (rename it to avoid conflict with CMake's internal name)
add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BUILD_DIR}
)

# Declare the default target (all object files)
add_dependencies(build_all cpp_objects cuda_objects)

# Declare that the 'clean_build' target should not generate a file
set_property(TARGET clean_build PROPERTY EXTERNAL_PROJECT 1)

# Enable verbose output for the build process
set(CMAKE_VERBOSE_MAKEFILE ON)

