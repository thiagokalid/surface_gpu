import os
import ctypes
import numpy as np
from ctypes import c_float, c_int

# Path to the shared library (.so file)
lib_path = '../build/vector_add.so'

# Load the shared library
try:
    clib = ctypes.CDLL(os.path.abspath(lib_path))
    print("Library loaded successfully!")
except Exception as e:
    print(f"Error loading library: {e}")

# Define argument and return types of the CUDA function
# Here we define that the function 'call_cuda_function' takes three pointers to float arrays and an integer
clib.call_cuda_function.argtypes = [ctypes.POINTER(c_float), ctypes.POINTER(c_float), ctypes.POINTER(c_float), c_int]
clib.call_cuda_function.restype = None

# Define the size of the array
N = 1024  # Length of the arrays

# Initialize input arrays
a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

# Convert the arrays to ctypes pointers
a_ptr = a.ctypes.data_as(ctypes.POINTER(c_float))
b_ptr = b.ctypes.data_as(ctypes.POINTER(c_float))
c_ptr = c.ctypes.data_as(ctypes.POINTER(c_float))

# Call the CUDA function
clib.call_cuda_function(a_ptr, b_ptr, c_ptr, N)

# Print the result (output array c)
print("Output array:", c)

