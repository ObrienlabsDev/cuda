# Dockerfile
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

# docker build -t cuda-add .
# docker run --rm --gpus all cuda-add


# Install any dependencies if needed (often CMake, build-essential, etc.)
# For example:
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy your CUDA source file into the image
COPY kernel_collatz.cu /app

# Compile the program with NVCC
RUN nvcc -o kernel kernel_collatz.cu

# By default, run the resulting binary
CMD ["./kernel"]
