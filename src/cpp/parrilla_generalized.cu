#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

extern "C" {

    // Device function to calculate the value of tof
    __device__ float tof(int k, float* xS, float* zS, float x1, float z1, float c) {
        // Calculate Mk
        float Mk = (zS[k + 1] - zS[k]) / (xS[k + 1] - xS[k]);

        // Return the result
        return (1 / c) * ((xS[k] - x1) + Mk * (zS[k] - z1)) / sqrt(pow(xS[k] - x1, 2) + pow(zS[k] - z1, 2));
    }

    // Device function to compute the step
    __device__ int step(int k0, float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2) {
        // Compute Vk0 and Vk using the tof function
        float Vk0 = tof(k0, xS, zS, xA, zA, c1) + tof(k0, xS, zS, xF, zF, c2);
        float Vk = tof(k0 + 1, xS, zS, xA, zA, c1) + tof(k0 + 1, xS, zS, xF, zF, c2);

        // The step difference is simply the difference between Vk and Vk0
        return round(Vk0 / (Vk - Vk0));
    }

    // Global kernel function
    __global__ void _parrilla_generalized(int* k_result, float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2, int N, int maxIter, int epsilon) {
        // Compute position that this thread is responsible for
        const uint col = blockIdx.x * blockDim.x + threadIdx.x;
        const uint row = blockIdx.y * blockDim.y + threadIdx.y;
 
        // Solver parameters:
        bool converged = false;
        int k0 = 0;
        int k = 1000;

        //printf("xA, zA = (%f, %f)\n", xA, zA);
        //printf("xF, zF = (%f, %f)\n", xF, zF);
        //printf("xS, zS = (%f, %f)\n", xS[25], zS[25]);

        // Newton-Raphson method:
        for (int i = 0; i < maxIter; i++) {
            // Newton-step:
            // int istep = step(k0, xA, zA, xF, zF, xS, zS, c1, c2)
            int istep = step(k0, xA, zA, xF, zF, xS, zS, c1, c2);

            k = k0 - istep;

            // Check if k is within indexable bounds:
            if (k < 0) {
                k = 0;
            } else if (k >= N) {
                k = N - 3;
            }

            // Debug print
            // printf("Thread %d; iteration %d; k0 = %d;k= %d; step = %d;\n", threadIdx.x, i, k0, k, istep);


            // Check stopping criteria
            if (abs(k - k0) <= epsilon) {
                converged = true;
                break;
            } else{
                k0 = k;
            }

        }
        *k_result = k;
    }

    // Host function to call the kernel
    int parrilla_generalized(float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2, int N, int maxIter, int epsilon) {
        float *d_xS, *d_zS;
        int *d_k, *k;

        // Allocate memory on the host (CPU)
        k = (int*)malloc(sizeof(int));

        // Surface: Allocate memory on the device
        cudaMalloc(&d_xS, N * sizeof(float));
        cudaMalloc(&d_zS, N * sizeof(float));
        cudaMemcpy(d_xS, xS, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zS, zS, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_k, sizeof(int));

        // Run the kernel:
        // Create as many blocks as necessary to map all of C
        dim3 gridDim(40, 60, 1);  // 32 threads per block (adjust as necessary)
        dim3 blockDim(64, 1, 1);             // Example block size of 32 threads per block

        // Call the kernel
        _parrilla_generalized<<<gridDim, blockDim>>>(d_k, xA, zA, xF, zF, d_xS, d_zS, c1, c2, N, maxIter, epsilon);

        // Check for any kernel errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }

        // Wait for the kernel to finish
        cudaDeviceSynchronize();

        // Copy the results back to host
        //cudaMemcpy(xS, d_xS, N * sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(zS, d_zS, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(k, d_k, sizeof(int), cudaMemcpyDeviceToHost);


        // Free up memory allocated on GPU:
        cudaFree(d_xS);
        cudaFree(d_zS);
        cudaFree(d_k);

        return *k;
    }
}
