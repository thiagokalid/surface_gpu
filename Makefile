# Makefile to build the C++/CUDA project with Python bindings using PyBind11

# Compiler and linker settings
CXX = g++
CUDA = nvcc
PYTHON = python3
PYBIND11_INCLUDE = /path/to/pybind11/include  # Adjust path to PyBind11 include folder

# Directories
SRC_DIR = src
BUILD_DIR = build
CPP_DIR = $(SRC_DIR)/cpp
CUDA_DIR = $(SRC_DIR)/cpp
BINDINGS_DIR = $(SRC_DIR)/pybind
INCLUDE_DIR = $(CPP_DIR)/include
PYTHON_MODULES_DIR = $(SRC_DIR)/python_modules

# Files
CPP_SOURCES = $(CPP_DIR)/my_cpp_function.cpp
CUDA_SOURCES = $(CUDA_DIR)/my_cuda_function.cu
BINDINGS_SOURCES = $(BINDINGS_DIR)/bindings.cpp
PYTHON_MODULE = my_module

# Output files
LIBRARY_NAME = lib$(PYTHON_MODULE).so  # Shared library for Python bindings
OBJ_FILES = $(CPP_SOURCES:.cpp=.o) $(CUDA_SOURCES:.cu=.o)

# Compiler flags
CXX_FLAGS = -O3 -Wall -fPIC -std=c++14
CUDA_FLAGS = -O3 --compiler-bindir=$(CXX)
LDFLAGS = -shared -fPIC

# Default target
all: $(BUILD_DIR)/$(LIBRARY_NAME)

# Rule to compile C++ sources into object files
$(CPP_DIR)/%.o: $(CPP_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -I$(PYBIND11_INCLUDE) -c $< -o $@

# Rule to compile CUDA sources into object files
$(CUDA_DIR)/%.o: $(CUDA_DIR)/%.cu
	$(CUDA) $(CUDA_FLAGS) -I$(INCLUDE_DIR) -I$(PYBIND11_INCLUDE) -c $< -o $@

# Rule to compile Python bindings
$(BUILD_DIR)/$(LIBRARY_NAME): $(OBJ_FILES) $(BINDINGS_SOURCES)
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES) $(BINDINGS_SOURCES) -I$(INCLUDE_DIR) -I$(PYBIND11_INCLUDE) -L$(BUILD_DIR) -o $(BUILD_DIR)/$(LIBRARY_NAME) $(LDFLAGS)

# Clean rule to remove object files and shared library
clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/$(LIBRARY_NAME)

# Install the Python module
install:
	python3 setup.py install

# Running tests (if you have any test files)
test: all
	pytest tests/  # Adjust based on your testing setup
