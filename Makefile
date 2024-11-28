# Makefile to build the C++/CUDA project with Python bindings using PyBind11

# Compiler and linker settings
CXX = g++
CUDA = nvcc

# Directories
SRC_DIR = src
BUILD_DIR = build
CPP_DIR = $(SRC_DIR)/cpp
INCLUDE_DIR = $(CPP_DIR)/include

# Files (using wildcards to include all cpp/cuda files)
CPP_SOURCES = $(wildcard $(CPP_DIR)/*.cpp)
CUDA_SOURCES = $(wildcard $(CPP_DIR)/*.cu)

# Object files (with path to build directory)
CPP_OBJ_FILES = $(patsubst $(SRC_DIR)/cpp/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SOURCES))
CUDA_OBJ_FILES = $(patsubst $(SRC_DIR)/cpp/%.cu, $(BUILD_DIR)/%.so, $(CUDA_SOURCES))

# Combine C++ and CUDA object files
OBJ_FILES = $(CPP_OBJ_FILES) $(CUDA_OBJ_FILES)

# Compiler flags for g++ (host compiler)
CXX_FLAGS = -O3 -Wall -fPIC -std=c++14 -I$(INCLUDE_DIR)
# Compiler flags for nvcc (CUDA compiler)
CUDA_FLAGS = -O3 -Xcompiler -Wall -Xcompiler -fPIC -Xcompiler -std=c++14 -I$(INCLUDE_DIR)

# Default target (no linking into a shared library)
all: $(CPP_OBJ_FILES) $(CUDA_OBJ_FILES)

# Compilation rules
$(BUILD_DIR)/%.o: $(SRC_DIR)/cpp/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.so: $(SRC_DIR)/cpp/%.cu
	$(CUDA) $(CUDA_FLAGS) $< -o $@

# Clean up
clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.so

# Declare phony targets
.PHONY: all clean

