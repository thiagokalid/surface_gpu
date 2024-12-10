import os
import ctypes
import numpy as np
from ctypes import c_float, c_int

LIBPATH = '../build/'

shared_libraries = [
    "parrilla_generalized.so"
]

for lib in shared_libraries:
    try:
        clib = ctypes.CDLL(os.path.abspath(LIBPATH + lib))
        print("Library loaded successfully!")
    except Exception as e:
        raise ImportError(f"Error loading library: {e}")


def parrilla_2007(xA: float, zA: float, xF: float, zF: float, xS: np.ndarray, zS: np.ndarray, c1: float, c2: float, maxiter: int=100, epsilon: int=2):
    # Here we define that the function 'call_cuda_function' takes three pointers to float arrays and an integer
    clib.parrilla_generalized.argtypes = [
        c_float, c_float,  # xA, zA
        c_float, c_float,  # xF, zF
        ctypes.POINTER(c_float), ctypes.POINTER(c_float),  # xS, zS (vectors)
        c_float, c_float,  # c1, c2
        c_int,  # length of xS and zS
        c_int, c_int  # maxIter and epsilon
    ]
    clib.parrilla_generalized.restype = c_int

    xS_ptr = xS.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    zS_ptr = zS.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    N = len(xS)

    # Call the CUDA function
    k = clib.parrilla_generalized(xA, zA, xF, zF, xS_ptr, zS_ptr, c1, c2, N, maxiter, epsilon)

    return int(k)