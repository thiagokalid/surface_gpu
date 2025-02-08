#ifndef PARRILLA_2007_H
#define PARRILLA_2007_H

#include <cuda_runtime.h>

// Function prototypes for device and host functions

// Device function to compute the linear interpolation at point x
__device__ double _lin_interp(float x0, float xf, float y0, float yf, float x);

// Device function to compute the derivative at point x[k] + xstep
__device__ double _compute_Mk(int k, float* xS, float* zS);

// Device function to calculate the value of tof
__device__ double tof(int k, float* xS, float* zS, float x1, float z1, float c, float Mk);

// Device function to compute the step
__device__ double step(float k_float, float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2);

// Global kernel function to execute on the GPU
__global__ void _parrilla_2007(int* k_result, float* xA, float* zA, float* xF, float* zF, float* xS, float* zS, float c1, float c2, int N);

// Host function to call the kernel
extern "C" int* parrilla_2007(float* xA, float* zA, float* xF, float* zF, float* xS, float* zS, float c1, float c2, int Na, int Nf, int N, int maxIter, float tolerance);

#endif // PARRILLA_2007_H
