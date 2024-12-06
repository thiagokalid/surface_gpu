import os
import ctypes
import numpy as np
from ctypes import c_float, c_int

# Path to the shared library (.so file)
lib_path = '../build/parrilla_generalized.so'

# Load the shared library
try:
    clib = ctypes.CDLL(os.path.abspath(lib_path))
    print("Library loaded successfully!")
except Exception as e:
    print(f"Error loading library: {e}")

# Define argument and return types of the CUDA function (float result, float xA, float zA, float xF, float zF, float* xS, float* zS, int N)
# Here we define that the function 'call_cuda_function' takes three pointers to float arrays and an integer
clib.parrilla_generalized.argtypes = [
    c_float, c_float,  # xA, zA
    c_float, c_float,  # xF, zF
    ctypes.POINTER(c_float), ctypes.POINTER(c_float), # xS, zS (vectors)
    c_int  # length of xS and zS
]
clib.parrilla_generalized.restype = c_float

# Initialize input arrays
xA, zA = 0., 0.
xF, zF = 0., 10.

# Linear function: f(x) = 5
deltax = 1e-1
xS = np.arange(-5, 5 + deltax, deltax)
zS = xS + 5

N = len(zS)

#
xS_ptr = xS.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
zS_ptr = zS.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Call the CUDA function
result = clib.parrilla_generalized(xA, zA, xF, zF, xS_ptr, zS_ptr, N)

# Print the result (output array c)
print("Output array:", result)

