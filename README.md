# CUDA examples
## CUDA add
see Visual Studio 2022 code around CUDA 12.6 - https://github.com/ObrienlabsDev/cuda/tree/main/add_example

Dockerized
```
cd cuda/add_example
docker build -t cuda-add .
docker run --rm --gpus all cuda-add
```

## Collatz
```
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
    // 22,64,5120, 94
    // 22,128,5120, 94
```
