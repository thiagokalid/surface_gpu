cmake_minimum_required(VERSION 3.10)
project(CudaCppProject LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)  # Replace with your target architecture(s)

# Compiler flags for C++ and CUDA
set(CXX_FLAGS 
    -O3
)  # Simplify C++ flags (remove -fPIC, -std=c++14)

set(CUDA_FLAGS
    -shared
    -Xcompiler
    -fPIC
    -O3
    -std=c++14
)

# Directories
set(SRC_DIR src)
set(CPP_DIR ${SRC_DIR}/cpp)
set(INCLUDE_DIR ${CPP_DIR}/include)
set(BUILD_DIR bin)

# Create build file
# Ensure the build directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${BUILD_DIR})

# Find all the source files
file(GLOB CPP_SOURCES "${CPP_DIR}/*.cpp")
file(GLOB CUDA_SOURCES "${CPP_DIR}/*.cu")

## Debugging: Check if CPP sources were found
message(STATUS "CUDA sources: ${CUDA_SOURCES}")

# Debugging: Check if CPP sources were found
message(STATUS "CPP sources: ${CPP_SOURCES}")

# Include directories
include_directories(${INCLUDE_DIR})

# Create object files for C++ sources
foreach(CPP_FILE ${CPP_SOURCES})
    get_filename_component(CPP_FILE_NAME ${CPP_FILE} NAME_WE)
    set(CPP_OBJ_FILE "${BUILD_DIR}/${CPP_FILE_NAME}.o")
    
    # Add custom command to compile the object files
    add_custom_command(
        OUTPUT ${CPP_OBJ_FILE}
	COMMAND ${CMAKE_CUDA_COMPILER} ${CXX_FLAGS} -I${INCLUDE_DIR} -c ${CPP_FILE} -o ${CPP_OBJ_FILE}
        DEPENDS ${CPP_FILE}
        COMMENT "Building object file ${CPP_OBJ_FILE}"
    )

    list(APPEND CPP_OBJ_FILES ${CPP_OBJ_FILE})
endforeach()

# Create shared object files for CUDA sources (unchanged)
foreach(CUDA_FILE ${CUDA_SOURCES})
    get_filename_component(CUDA_FILE_NAME ${CUDA_FILE} NAME_WE)
    set(CUDA_OBJ_FILE "${BUILD_DIR}/${CUDA_FILE_NAME}.so")

    add_custom_command(
        OUTPUT ${CUDA_OBJ_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} ${CUDA_FLAGS}  -I${INCLUDE_DIR} -o ${CUDA_OBJ_FILE} ${CUDA_FILE}
        DEPENDS ${CUDA_FILE}
        COMMENT "Building shared object file ${CUDA_OBJ_FILE}"
    )

    list(APPEND CUDA_OBJ_FILES ${CUDA_OBJ_FILE})
endforeach()

# Combine both C++ and CUDA object files into a custom target
add_custom_target(build_all
    DEPENDS ${CPP_OBJ_FILES} ${CUDA_OBJ_FILES}
)

# Clean target (rename it to avoid conflict with CMake's internal name)
add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMENT "Cleaning all build-related files except CMakeLists.txt"
)


# Declare that the 'clean_build' target should not generate a file
set_property(TARGET clean_build PROPERTY EXTERNAL_PROJECT 1)

# Enable verbose output for the build process
set(CMAKE_VERBOSE_MAKEFILE ON)

