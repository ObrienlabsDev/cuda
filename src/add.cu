#include <iostream>
#include <cuda_runtime.h>
// 20241223
// A simple CUDA kernel that adds two integers and stores the result in c
__global__ void add(int a, int b, int* c)
{
    *c = a + b;
}

int main()
{
    int c;        // Host variable to store the result
    int* d_c;     // Pointer to device memory

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_c, sizeof(int));

    // Launch the 'add' kernel on 1 block with 1 thread
    add<<<1, 1>>>(2, 7, d_c);

    // Copy the result from GPU to CPU
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "2 + 7 = " << c << std::endl;

    // Free the GPU memory
    cudaFree(d_c);

    return 0;
}
