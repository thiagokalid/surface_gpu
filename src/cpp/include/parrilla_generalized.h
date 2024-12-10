#ifndef PARRILLA_GENERALIZED_H
#define PARRILLA_GENERALIZED_H

#include <cuda_runtime.h>

// Function prototypes for device and host functions

// Device function to calculate the value of tof
__device__ float tof(int k, float* xS, float* zS, float x1, float z1, float c);

// Device function to compute the step
__device__ int step(int k0, float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2);

// Global kernel function to execute on the GPU
__global__ void _parrilla_generalized(int* k_result, float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2, int N);

// Host function to call the kernel
extern "C" int parrilla_generalized(float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2, int N);

#endif // PARRILLA_GENERALIZED_H
