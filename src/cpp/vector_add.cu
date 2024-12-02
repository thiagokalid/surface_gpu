// File: cuda_functions.cu

#include <cuda_runtime.h>

extern "C" {
    __global__ void add_arrays(float* a, float* b, float* c, int N) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < N) {
            c[index] = a[index] + b[index];
        }
    }

    void call_cuda_function(float* a, float* b, float* c, int N) {
        float *d_a, *d_b, *d_c;

        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_c, N * sizeof(float));

        cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        add_arrays<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

        cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
}

