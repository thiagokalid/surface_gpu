#ifndef PARRILLA_GENERALIZED_H
#define PARRILLA_GENERALIZED_H

#include <cuda_runtime.h>

// Function prototype for the CUDA kernel
__global__ void _parrilla_generalized(float xA, float zA, float xF, float zF, float* xS, float* zS, int N);

// Function prototype for the host-side wrapper function
extern "C" float parrilla_generalized(float xA, float zA, float xF, float zF, float* xS, float* zS, int N);

#endif // PARRILLA_GENERALIZED_H
