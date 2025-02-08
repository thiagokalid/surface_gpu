#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define BLOCK_SIZE 32

extern "C" {
    // Device function to compute the linear interpolation at point x
    __device__ double _lin_interp(float x0, float xf, float y0, float yf, float x) {
        if (x > xf){
            return yf;
        } else if (x < x0){
            return y0;
        } else {
            return  (yf - y0)/(x0 - xf) * x;
        }
    }

    // Device function to compute the derivative at point x[k] + xstep
    __device__ double _compute_Mk(int k, float* xS, float* zS) {
        double delta = 0.1;
        double xk_delta = xS[k] + delta;
        double zk_delta = _lin_interp(xS[k], xS[k+1], zS[k], zS[k+1], xk_delta);
        return (zk_delta - zS[k]) / (xk_delta - xS[k]);
    }

    // Device function to calculate the value of tof
    __device__ double tof(int k, float* xS, float* zS, float x1, float z1, float c, float Mk) {
        // Return the result
        return (1 / c) * ((xS[k] - x1) + Mk * (zS[k] - z1)) / sqrt(pow(xS[k] - x1, 2) + pow(zS[k] - z1, 2));
    }

    // Device function to compute the step
    __device__ double step(float k_float, float xA, float zA, float xF, float zF, float* xS, float* zS, float c1, float c2) {
        // Compute Vk0 and Vk using the tof function
        int k0 = round(k_float);
        double Mk = _compute_Mk(k0, xS, zS);
        double Vk0 = tof(k0, xS, zS, xA, zA, c1, Mk) + tof(k0, xS, zS, xF, zF, c2, Mk);
        double Vk = tof(k0 + 1, xS, zS, xA, zA, c1, Mk) + tof(k0 + 1, xS, zS, xF, zF, c2, Mk);

        // The step difference is simply the difference between Vk and Vk0
        return Vk0 / (Vk - Vk0);
    }

    // Global kernel function
    __global__ void _parrilla_2007(int* d_k, float* d_xA, float* d_zA, float* d_xF, float* d_zF, float* xS, float* zS, float c1, float c2, int Na, int Nf, int N, int maxIter, float tolerance) {
        // Compute position that this thread is responsible for
        int c = blockIdx.y * blockDim.y + threadIdx.y;
        int r = blockIdx.x * blockDim.x + threadIdx.x;

        // Solver parameters:
        bool converged = false;
        float k0 = 0;
        float k;


        // Newton-Raphson method:
        if( r < Na && c < Nf){
            double xA = d_xA[r];
            double zA = d_zA[r];
            double xF = d_xF[c];
            double zF = d_zF[c];

            for (int i = 0; i < maxIter; i++) {
                // Newton-step:
                double istep = step(k0, xA, zA, xF, zF, xS, zS, c1, c2);

                k = k0 - istep;

                // Check if k is within indexable bounds:
                if (k < 0) {
                    k = 0;
                } else if (k >= N-2) {
                    k = N - 3;
                }

                // Check stopping criteria
                if (abs(k - k0) <= tolerance) {
                    converged = true;
                    break;
                }
                k0 = k;
            }
            int idx = c + r * Nf;
            d_k[idx] = round(k);
        }
    }

    // Host function to call the kernel
    int* parrilla_2007(float* xA, float* zA, float* xF, float* zF, float* xS, float* zS, float c1, float c2, int Na, int Nf, int N, int maxIter, float tolerance) {
        float *d_xS, *d_zS, *d_xF, *d_zF, *d_xA, *d_zA;

        int *d_k;
        int *k = new int[Na * Nf];

        // Surface: Allocate memory on the device
        cudaMalloc(&d_xS, N * sizeof(float));
        cudaMalloc(&d_zS, N * sizeof(float));
        cudaMemcpy(d_xS, xS, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zS, zS, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_xA, Na * sizeof(float));
        cudaMalloc(&d_zA, Na * sizeof(float));
        cudaMemcpy(d_xA, xA, Na * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zA, zA, Na * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_xF, Nf * sizeof(float));
        cudaMalloc(&d_zF, Nf * sizeof(float));
        cudaMemcpy(d_xF, xF, Nf * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zF, zF, Nf * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_k, Na * Nf * sizeof(int));

        // Run the kernel:
        // Create as many blocks as necessary to map all of C
        int X = ceilf(Na/(float)BLOCK_SIZE);
        int Y = ceilf(Nf/(float)BLOCK_SIZE);
        // printf("X = %d \n", X);
        // printf("Y = %d \n", Y);

        dim3 gridDim(X, Y, 1);
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

        // Call the kernel
        _parrilla_2007<<<gridDim, blockDim>>>(d_k, d_xA, d_zA, d_xF, d_zF, d_xS, d_zS, c1, c2, Na, Nf, N, maxIter, tolerance);

        // Check for any kernel errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }

        // Wait for the kernel to finish
        cudaDeviceSynchronize();

        // Copy the results back to host
        cudaMemcpy(k, d_k, Na * Nf * sizeof(int), cudaMemcpyDeviceToHost);


        // Free up memory allocated on GPU:
        cudaFree(d_xS);
        cudaFree(d_zS);
        cudaFree(d_zA);
        cudaFree(d_zA);
        cudaFree(d_zF);
        cudaFree(d_zF);
        cudaFree(d_k);

        return k;
    }
}
