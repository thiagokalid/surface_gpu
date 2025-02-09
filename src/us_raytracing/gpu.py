import os
import ctypes
import numpy as np
from ctypes import c_float, c_int

__all__ = ["parrilla_2007"]

LIBPATH = '../bin/'

shared_libraries = [
    "parrilla_2007.so"
]

lib = {}
for library in shared_libraries:
    try:
        lib[library] = ctypes.CDLL(os.path.abspath(LIBPATH + library))
        print("Library loaded successfully!")
    except Exception as e:
        raise ImportError(f"Error loading library: {e}")


def parrilla_2007(xA: np.ndarray, zA: np.ndarray, xF: np.ndarray, zF: np.ndarray, xS: np.ndarray, zS: np.ndarray, c1: float, c2: float, maxiter: int=100, tolerance: float=1e-4):
    # Here we define that the function 'call_cuda_function' takes three pointers to float arrays and an integer
    clib = lib["parrilla_2007.so"]
    clib.parrilla_2007.argtypes = [
        ctypes.POINTER(c_float), ctypes.POINTER(c_float),  # xA, zA (vectors)
        ctypes.POINTER(c_float), ctypes.POINTER(c_float),  # xF, zF (vectors)
        ctypes.POINTER(c_float), ctypes.POINTER(c_float),  # xS, zS (vectors)
        c_float, c_float,  # c1, c2
        c_int, c_int, c_int,  # length of xA, xF, xS
        c_int, c_float  # maxIter and epsilon
    ]
    clib.parrilla_2007.restype = ctypes.POINTER(c_int)

    xS_ptr = xS.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    zS_ptr = zS.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    xA_ptr = xA.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    zA_ptr = zA.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    xF_ptr = xF.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    zF_ptr = zF.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    N = len(xS)
    Na = len(xA)
    Nf = len(xF)

    # Call the CUDA function
    k_vector = clib.parrilla_2007(xA_ptr, zA_ptr, xF_ptr, zF_ptr, xS_ptr, zS_ptr, c1, c2, Na, Nf, N, maxiter, tolerance)
    k = np.ctypeslib.as_array(k_vector, shape=(Na, Nf))
    return k
