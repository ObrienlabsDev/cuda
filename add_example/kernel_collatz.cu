
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

__global__ void addArrays(const int* a, int* c, int N)
{
    // Calculate this thread's index
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Check boundary (in case N is not a multiple of blockDim.x)
    int path = 0;
    int max = a[i];
    int current = a[i];

    if (i < N)
    {
        do {
        //for (int q = 0; q < 109; q++) {
          path += 1;
            //c[i] = a[i] + b[i];
            if (current % 2 == 0) {
                current = current >> 1;
            }
            else {
                current = 1 + current * 3;
                if (current > max) {
                    max = current;
                }
            }
        //}
        } while (current > 1);
    }

    c[i] = max;
}

int main()
{
    const int N = 5;

    // Host arrays
    //const int h_a[N] = { 27, 27, 27, 27, 27};
    int h_a[N];
    for (int q = 0; q < N; q++) {
        h_a[q] = 27;
    }
    int h_c[N] = { 0 };  // will hold the result

    // Device pointers
    int* d_a = nullptr;
    int* d_c = nullptr;

    time_t timeStart, timeEnd;
    double timeElapsed;

    time(&timeStart);

    // Allocate memory on the GPU
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_c, size);

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Decide on grid/block size.
    // For N=5, we can just use 1 block with 5 threads.
    // But let's future-proof slightly by choosing e.g. 256 threads per block:
    int threadsPerBlock = 256;
    // Number of blocks = ceiling(N / threadsPerBlock)
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    // kernelName<<<numBlocks, threadsPerBlock>>>(parameters...);
    addArrays << <blocks, threadsPerBlock >> > (d_a, d_c, N);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result from device back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "collatz:\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_a[i] << " = " << h_c[i] << "\n";
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

