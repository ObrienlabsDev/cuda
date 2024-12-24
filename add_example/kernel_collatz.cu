
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

/**
* Michael O'Brien 20241223
* michael at obrienlabs.dev
* Collatz sequence running on NVidia GPUs like the RTX-3500 ada,A4000,A4500,4090 ada and A6000
*/


/* CUDA Kernel runs on GPU device streaming core */
__global__ void addArrays(unsigned long long* a, unsigned long long* c, int N)
{
    // Calculate this thread's index
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Check boundary (in case N is not a multiple of blockDim.x)
    int path = 0;
    unsigned long long max = a[i];
    unsigned long long current = a[i];

    if (i < N)
    {
        do {
          path += 1;
            if (current % 2 == 0) {
                current = current >> 1;
            }
            else {
                current = 1 + current * 3;
                if (current > max) {
                    max = current;
                }
            }
        } while (current > 1);
    }
    c[i] = max;
}

/* Host progrem */
int main()
{
    const int N = 8;

    // Host arrays
    unsigned long long h_a[N];
    for (int q = 0; q < N; q++) {
        h_a[q] = 27;
    }

    unsigned long long h_result[N] = { 0 };

    // Device pointers
    unsigned long long* d_a = nullptr;
    unsigned long long* d_c = nullptr;

    time_t timeStart, timeEnd;
    double timeElapsed;

    time(&timeStart);

    // Allocate memory on the GPU
    size_t size = N * sizeof(unsigned long long);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_c, size);

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // 256 threads per block:
    int threadsPerBlock = 256;
    // Number of blocks = ceiling(N / threadsPerBlock)
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    // kernelName<<<numBlocks, threadsPerBlock>>>(parameters...);
    addArrays << <blocks, threadsPerBlock >> > (d_a, d_c, N);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result from device back to host
    cudaMemcpy(h_result, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "collatz:\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << i << ": " << h_a[i] << " = " << h_result[i] << "\n";
    }

    time(&timeEnd);
    timeElapsed = difftime(timeEnd, timeStart);

    //std::cout << "2 + 7 = " << c << std::endl;
    printf("duration: %.f\n", timeElapsed);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}

