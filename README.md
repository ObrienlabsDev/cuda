# CUDA examples
## CUDA add
see Visual Studio 2022 code around CUDA 12.6 - https://github.com/ObrienlabsDev/cuda/tree/main/add_example

Dockerized
```
cd cuda/add_example
docker build -t cuda-add .
docker run --rm --gpus all cuda-add
```
