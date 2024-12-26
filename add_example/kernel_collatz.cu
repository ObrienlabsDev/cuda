
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

/**
* Michael O'Brien 20241223
* michael at obrienlabs.dev
* Collatz sequence running on NVidia GPUs like the RTX-3500 ada,A4000,A4500,4090 ada and A6000
* http://www.ericr.nl/wondrous/pathrecs.html
* https://github.com/obrienlabs/benchmark/blob/master/ObjectiveC/128bit/main.m
* https://github.com/obrienlabs/benchmark/blob/master/collatz_vs10/collatz_vs10/collatz_vs10.cpp
* https://github.com/ObrienlabsDev/cuda/blob/main/add_example/kernel_collatz.cu
* https://github.com/ObrienlabsDev/collatz/blob/main/src/main/java/dev/obrienlabs/collatz/service/CollatzUnitOfWork.java
* 
*/


/* CUDA Kernel runs on GPU device streaming core */
__global__ void addArrays(unsigned long long* a, unsigned long long* c, int threads, unsigned long long iterations)
{
    // Calculate this thread's index
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Check boundary (in case N is not a multiple of blockDim.x)
    int path = 0;
    unsigned long long max = a[i];
    unsigned long long current = a[i];

    if (i < threads)
    {
        // takes 130 sec on a mobile RTX-3500 ada 
        for (unsigned long q = 0; q < iterations; q++) {
            path = 0;
            max = a[i];
            current = a[i];

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
    }
    c[i] = max;
}

/* Host progrem */
int main(int argc, char* argv[])
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        fprintf(stderr, "two GPU's required: found: % d found.\n", deviceCount);
        return 1;
    }
    const int dev0 = 0;
    const int dev1 = 1;

    int cores = (argc > 1) ? atoi(argv[1]) : 5120; // get command
    // exited with code -1073741571 any higher
    const int threads = 32768 - 1536;// 7168 * 4;
    // GPU0: Iterations: 8388608 Threads : 31232 ThreadsPerBlock : 64 Blocks : 488
    int iterationPower = 23;
    unsigned long long iterations = 1 << iterationPower;
    const int threadsPerBlock = 128;

    // debug is 32x slower than release
    // iterpower,threadsPerBlock,cores,seconds
    // RTX-3500 Ada
    // 256 threads per block is double the SM core count of 128 cores per SM:
    // 22, 256, 4096 = 130s
    // 22, 128, 4096 = 124
    // 22, 256, 5120 = 132
    // 22, 128, 5120 = 125
    // 22, 64. 5120  = 125

    // 4090
    // 22,64,5120, 94, 25 TDP
    // 22,128,5120, 94
    // 22,256,5120, 99
    // 22,128,16384, 99, 35 TDP
    // 22,128,16384, 94, 35 TDP exe
    // 23,128, 7168x8,128,229
    // 
    // RTX-a4500 Ampere
    // 22,64,5120, 140 exe 53 TDP


    // Host arrays
    unsigned long long h_a0[threads];
    unsigned long long h_a1[threads];

    for (int q = 0; q < threads; q++) {
        h_a0[q] = 8528817511;
        h_a1[q] = 8528817511;
    }

    unsigned long long h_result0[threads] = { 0 };
    unsigned long long h_result1[threads] = { 0 };

    // Device pointers
    unsigned long long* d_a0 = nullptr;
    unsigned long long* d_c0 = nullptr;
    unsigned long long* d_a1 = nullptr;
    unsigned long long* d_c1 = nullptr;


    time_t timeStart, timeEnd;
    double timeElapsed;

    time(&timeStart);

    //int N_per_gpu = N / 2;
    // Allocate memory on the GPU
    size_t size = threads * sizeof(unsigned long long);
    printf("array allocation bytes per GPU: %d * %d is %d\n", sizeof(unsigned long long), threads, size);
    cudaSetDevice(dev0);
    cudaMalloc((void**)&d_a0, size);
    cudaMalloc((void**)&d_c0, size);
    // Copy input data from host to device
    cudaMemcpy(d_a0, h_a0, size, cudaMemcpyHostToDevice);
    cudaSetDevice(dev1);
    cudaMalloc((void**)&d_a1, size);
    cudaMalloc((void**)&d_c1, size);
    // Copy input data from host to device
    cudaMemcpy(d_a1, h_a1, size, cudaMemcpyHostToDevice);

    // Number of blocks = ceiling(N / threadsPerBlock)
    int blocks = (threads + threadsPerBlock - 1) / threadsPerBlock;
    // maximums for 4090 single 2*28672 or split - 4.7A
    // GPU0: Iterations: 8388608 Threads: 28672 ThreadsPerBlock: 128 Blocks: 224
    // GPU0: Iterations: 8388608 Threads: 28672 ThreadsPerBlock: 128 Blocks: 224
    // 32k - 1.5k
    // GPU0: Iterations: 8388608 Threads: 31232 ThreadsPerBlock: 64 Blocks: 488
    printf("GPU0: Iterations: %lld Threads: %d ThreadsPerBlock: %d Blocks: %d\n", iterations, threads, threadsPerBlock, blocks);
    printf("GPU1: Iterations: %lld Threads: %d ThreadsPerBlock: %d Blocks: %d\n", iterations, threads, threadsPerBlock, blocks);

    // Launch kernel
    cudaSetDevice(dev1);
    // kernelName<<<numBlocks, threadsPerBlock>>>(parameters...);
    addArrays << <blocks, threadsPerBlock >> > (d_a1, d_c1, threads, iterations);

    cudaSetDevice(dev0);
    // kernelName<<<numBlocks, threadsPerBlock>>>(parameters...);
    addArrays << <blocks, threadsPerBlock >> > (d_a0, d_c0, threads, iterations);
    
    // Wait for GPU to finish before accessing on host
    cudaSetDevice(dev0);
    cudaDeviceSynchronize();
    cudaSetDevice(dev1);
    cudaDeviceSynchronize();

    // Copy result from device back to host
    cudaMemcpy(h_result0, d_c0, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result1, d_c1, size, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "collatz:\n";
    int i = 0;
    //for (int i = 0; i < threads; i++)
    //{
        std::cout << "GPU0: " << i << ": " << h_a0[i] << " = " << h_result0[i] << "\n";
        std::cout << "GPU1: " << i << ": " << h_a1[i] << " = " << h_result1[i] << "\n";
    //}

    time(&timeEnd);
    timeElapsed = difftime(timeEnd, timeStart);

    //std::cout << "2 + 7 = " << c << std::endl;
    printf("duration: %.f\n", timeElapsed);

    // Free GPU memory
    cudaSetDevice(dev0);
    cudaFree(d_a0);
    cudaFree(d_c0);
    cudaSetDevice(dev1);
    cudaFree(d_a1);
    cudaFree(d_c1);

    free(h_a0);
    free(h_a1);


    return 0;
}

